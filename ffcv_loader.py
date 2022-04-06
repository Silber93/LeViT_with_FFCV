import numpy as np
import pandas as pd
import torch

from pathlib import Path

from typing import List
from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder

from ffcv_writer import FFCV_DATA_DIR


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256


def create_train_loader(args, in_memory=True, res=224, nb_classes=1000):
    this_device = f'cuda:0'
    train_dataset = Path(f'{args.data_path}/train.dat')
    print(train_dataset)
    decoder = RandomResizedCropRGBImageDecoder((res, res))
    image_pipeline: List[Operation] = [
        decoder,
        RandomHorizontalFlip(),
        ToTensor(),
        ToDevice(torch.device(this_device), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
    ]

    label_pipeline: List[Operation] = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(torch.device(this_device), non_blocking=True)
    ]

    order = OrderOption.QUASI_RANDOM
    loader = Loader(train_dataset,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    order=order,
                    os_cache=in_memory,
                    drop_last=True,
                    pipelines={
                        'image': image_pipeline,
                        'label': label_pipeline
                    },
                    distributed=False)
    return loader, nb_classes


def create_val_loader(args, in_memory=True, resolution=224, nb_classes=1000):
    this_device = f'cuda:0'
    res_tuple = (resolution, resolution)
    val_dataset = Path(f'{args.data_path}/val.dat')
    cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
    image_pipeline = [
        cropper,
        ToTensor(),
        ToDevice(torch.device(this_device), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
    ]

    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(torch.device(this_device),
                 non_blocking=True)
    ]

    loader = Loader(val_dataset,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    order=OrderOption.SEQUENTIAL,
                    os_cache=in_memory,
                    drop_last=False,
                    pipelines={
                        'image': image_pipeline,
                        'label': label_pipeline
                    },
                    distributed=False)
    return loader, nb_classes


# def create_ffcv_loaders(train_dataset, val_dataset, num_workers, batch_size, in_memory, res):
#     loaders = (create_train_loader(train_dataset, num_workers, batch_size, in_memory, res),
#                create_val_loader(val_dataset, num_workers, batch_size, in_memory, res))
#     return loaders
