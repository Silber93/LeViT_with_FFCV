# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import numpy as np
import pandas as pd
import os
import torch
import torchvision
import time
import timm
import levit
import levit_c
import torchvision
import utils
torch.autograd.set_grad_enabled(False)

# from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField
from torch.utils.data import Subset

from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder


T0 = 10
T1 = 60

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256


SPEED_TEST_DIR = 'speed_test'



def ffcv_write(data_source,
               write_path,
               batch_size=2048,
               max_resolution=224,
               write_mode='smart',
               compress_probability=1,
               jpeg_quality=90,
               num_workers=2):
    # my_dataset = CustomInputDataset(inputs)
    my_dataset = ImageFolder(root=data_source)
    my_dataset = Subset(my_dataset, range(2048))
    writer = DatasetWriter(write_path, {
        'image': RGBImageField(write_mode=write_mode,
                               max_resolution=max_resolution,
                               compress_probability=compress_probability,
                               jpeg_quality=jpeg_quality),
    }, num_workers=num_workers)

    writer.from_indexed_dataset(my_dataset, chunksize=batch_size)


def ffcv_load(load_path,
              batch_size=2048,
              resolution=224,
              num_workers=2,
              in_memory=True):
    this_device = f'cuda:0'
    res_tuple = (resolution, resolution)
    decoder = RandomResizedCropRGBImageDecoder(res_tuple)
    image_pipeline = [
        decoder,
        RandomHorizontalFlip(),
        ToTensor(),
        ToDevice(torch.device(this_device), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)

    ]

    loader = Loader(load_path,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    order=OrderOption.SEQUENTIAL,
                    os_cache=in_memory,
                    drop_last=False,
                    pipelines={
                        'image': image_pipeline,
                    },
                    distributed=False)
    return loader


# def compute_throughput_cpu(name, model, device, batch_size=2048, ffcv=False, resolution=224):
#     inputs = torch.randn(batch_size, 3, resolution, resolution, device=device)
#     # warmup
#     start = time.time()
#     while time.time() - start < T0:
#         model(inputs)
#
#     timing = []
#     while sum(timing) < T1:
#         start = time.time()
#         model(inputs)
#         timing.append(time.time() - start)
#     timing = torch.as_tensor(timing, dtype=torch.float32)
#     print(name, device, batch_size / timing.mean().item(),
#           'images/s @ batch size', batch_size)


def compute_throughput_cuda(name, model, device_net, ffcv, batch_size=2048, resolution=224, inputs=None):
    if inputs is None:
        inputs = torch.randn(batch_size, 3, resolution, resolution, device=device_net)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start = time.time()
    with torch.cuda.amp.autocast():
        while time.time() - start < T0:
            model(inputs)
    timing = []
    if device_net == 'cuda:0':
        torch.cuda.synchronize()
    with torch.cuda.amp.autocast():
        while sum(timing) < T1:
            start = time.time()
            model(inputs)
            torch.cuda.synchronize()
            timing.append(time.time() - start)
    timing = torch.as_tensor(timing, dtype=torch.float32)
    speed = batch_size / timing.mean().item()
    utils.log.info(f'{name}, {device_net}_ffcv_{ffcv}, {speed} images/s @ batch size {batch_size}')
    return speed



def get_args_parser():
    parser = argparse.ArgumentParser(
        'LeViT training and evaluation script', add_help=False)
    parser.add_argument('--ffcv-data-source-dir', default=None,
                        help='data directory needed to generate ffcv data files (i.e. ImageNet/val)')
    parser.add_argument('--ffcv-regen', default='N', help='generate new random ffcv dataset? [Y/N]')
    parser.add_argument('--ffcv-dat-path', default=None, help='alternative ffcv .dat file to be tested')
    return parser


parser = argparse.ArgumentParser(
        'LeViT/FFCV speed test', parents=[get_args_parser()])
args = parser.parse_args()

if not os.path.exists(SPEED_TEST_DIR):
    os.mkdir(SPEED_TEST_DIR)
if not args.ffcv_dat_path:
    write_path = f'{SPEED_TEST_DIR}/ffcv_sample.dat'
    if not os.path.exists(write_path):
        utils.log.info('no FFCV .dat file, generating new FFCV .dat file to speed_test/ffcv_sample.dat')
        ffcv_write(args.ffcv_data_source_dir, write_path)
    elif args.ffcv_regen == 'Y':
        utils.log.info('re-generating new FFCV .dat file to speed_test/ffcv_sample.dat')
        ffcv_write(args.ffcv_data_source_dir, write_path)
results = {}
load_path = args.ffcv_dat_path if args.ffcv_dat_path else 'speed_test/ffcv_sample.dat'
for device in ['cuda:ffcv', 'cuda:0']:

    if 'cuda' in device and not torch.cuda.is_available():
        print("no cuda")
        continue

    if device == 'cpu':
        os.system('echo -n "nb processors "; '
                  'cat /proc/cpuinfo | grep ^processor | wc -l; '
                  'cat /proc/cpuinfo | grep ^"model name" | tail -1')
        print('Using 1 cpu thread')
        torch.set_num_threads(1)
        compute_throughput = compute_throughput_cpu
    else:
        utils.log.info(torch.cuda.get_device_name(torch.cuda.current_device()))
        compute_throughput = compute_throughput_cuda

    for n, batch_size0, resolution in [
        # ('timm.models.resnet50', 1024, 224),
        # ('timm.models.deit_tiny_distilled_patch16_224', 2048, 224),
        # ('timm.models.deit_small_distilled_patch16_224', 2048, 224),
        ('levit.LeViT_128S', 2048, 224),
        ('levit.LeViT_128', 2048, 224),
        ('levit.LeViT_192', 2048, 224),
        ('levit.LeViT_256', 1024, 224),
        ('levit.LeViT_384', 1024, 224),
        # ('timm.models.efficientnet_b0', 1024, 224),
        # ('timm.models.efficientnet_b1', 1024, 240),
        # ('timm.models.efficientnet_b2', 512, 260),
        # ('timm.models.efficientnet_b3', 512, 300),
        # ('timm.models.efficientnet_b4', 256, 380),
    ]:
        batch_size = batch_size0
        torch.cuda.empty_cache()
        ffcv = True if device.split(':')[-1] == 'ffcv' else False
        device_net = 'cuda:0'
        if ffcv:
            inputs_loader = ffcv_load(load_path, batch_size=batch_size)
            inputs = next(iter(inputs_loader))[0].to(device_net, non_blocking=True)
        else:
            inputs = torch.randn(batch_size, 3, resolution,
                                 resolution, device=device_net)
        model = eval(n)(num_classes=1000)
        utils.replace_batchnorm(model)
        if ffcv:
            model.to(memory_format=torch.channels_last).cuda()
        else:
            model.to(device_net)
        model.eval()
        speed = compute_throughput(n, model, device_net, ffcv, batch_size=batch_size, resolution=resolution, inputs=inputs)
        if n not in results:
            results[n] = {}
        results[n][f'{device_net}_ffcv_{ffcv}'] = speed

df = pd.DataFrame.from_dict(results, orient='index')
df.to_csv(f'{SPEED_TEST_DIR}/results.csv')
utils.log.info(f'saved results in {SPEED_TEST_DIR}/results.csv')
