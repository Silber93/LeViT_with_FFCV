import os

from torch.utils.data import Subset
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField
from torchvision.datasets import CIFAR10, ImageFolder

# import argparse
from argparse import ArgumentParser
# from fastargs import Section, Param
# from fastargs.validation import And, OneOf
# from fastargs.decorators import param, section
# from fastargs import get_current_config

FFCV_DATA_DIR = 'ffcv_data'


def get_args_parser_for_writing():
    parser = ArgumentParser(
        'FFCV data writing script (ImageNet / CIFAR)', add_help=False)
    parser.add_argument('--split', help='train or val set')
    parser.add_argument('--data-dir', help='Where to find the PyTorch dataset')
    parser.add_argument('--write-dir-name', help='Where to write the new dataset (name only)')
    parser.add_argument('--write-mode', default='smart', help='Mode: raw, smart or jpg')
    parser.add_argument('--max-resolution', default=224, type=int, help='Max image side length')
    parser.add_argument('--num-workers', default=2, type=int, help='Number of workers to use')
    parser.add_argument('--chunk-size', default=64, type=int, help='Chunk size for writing')
    parser.add_argument('--jpeg-quality', default=90, type=int, help='Quality of jpeg images')
    parser.add_argument('--subset', default=-1, type=int, help='How many images to use (-1 for all)')
    parser.add_argument('--compress-probability', default=None, type=float, help='compress probability')
    return parser


def main(args):
    my_dataset = ImageFolder(root=args.data_dir)
    if args.subset > 0:
        my_dataset = Subset(my_dataset, range(args.subset))
    if not os.path.exists(f'{FFCV_DATA_DIR}/{args.write_dir_name}'):
        os.mkdir(f'{FFCV_DATA_DIR}/{args.write_dir_name}')
    write_path = f'{FFCV_DATA_DIR}/{args.write_dir_name}/{args.split}.dat'
    writer = DatasetWriter(write_path, {
        'image': RGBImageField(write_mode=args.write_mode,
                               max_resolution=args.max_resolution,
                               compress_probability=args.compress_probability,
                               jpeg_quality=args.jpeg_quality),
        'label': IntField(),
    }, num_workers=args.num_workers)

    writer.from_indexed_dataset(my_dataset, chunksize=args.chunk_size)


if __name__ == '__main__':
    if not os.path.exists(FFCV_DATA_DIR):
        os.mkdir(FFCV_DATA_DIR)
    parser = ArgumentParser('FFCV data writing script (ImageNet / CIFAR)', parents=[get_args_parser_for_writing()])
    args = parser.parse_args()
    main(args)
