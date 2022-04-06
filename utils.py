# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import os
import time
from collections import defaultdict, deque
import datetime
import pandas as pd

import torch
import torch.distributed as dist

from pathlib import Path
import logging
log = logging.getLogger()
console = logging.StreamHandler()
log.addHandler(console)
log.setLevel(logging.INFO)


CSV_COL_ORDER = ['init_timestamp', 'train_timestamp', 'train_dur', 'train_speed',
                 'train_eval_start_timestamp', 'train_eval_end_timestamp', 'train_eval_dur', 'train_eval_speed',
                 'val_eval_start_timestamp', 'val_eval_end_timestamp', 'val_eval_dur', 'val_eval_speed',
                 'f_timestamp', 'epoch_dur',
                 'train_loss', 'train_acc1', 'train_acc5', 'train_best_acc1_batch', 'train_best_acc5_batch',
                 'val_loss', 'val_acc1', 'val_acc5', 'val_best_acc1_batch', 'val_best_acc5_batch'
                 ]

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total],
                         dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class CustomLogger:
    def __init__(self, args):
        self.logs = {}
        self.msg_list = []
        self.epochs = args.epochs
        self.output_dir = Path(args.output_dir)
        self.temp_epoch = {'train_loss': [], 'train_acc1': [], 'train_acc5': [],
                           'val_loss': [], 'val_acc1': [], 'val_acc5': []
                           }
        self.best_acc_batch = {'train_acc1': 0,
                               'train_acc5': 0,
                               'val_acc1': 0,
                               'val_acc5': 0
                               }
        self.list_args(args)
        self.n_batches = {}


    @staticmethod
    def get_ep_name(epoch):
        ep_str = str(epoch + 1)
        for i in [2, 3]:
            if len(ep_str) < i:
                ep_str = '0' + ep_str
        return f'ep_{ep_str}'

    @staticmethod
    def now():
        return datetime.datetime.now().replace(microsecond=0)

    def save_num_batches(self, train_dataloader, val_dataloader, batch_size):
        for loader, split in zip([train_dataloader, val_dataloader], ['train', 'val']):
            num_batches, total_count = 0, 0
            for _, targets in loader:
                num_batches += 1
                total_count += targets.shape[0]
            msg = f'{split} data volume: {total_count} images, separated to {num_batches} batches ' \
                  f'(batch size {batch_size})'
            self.log(msg)
            self.n_batches[split] = num_batches

    def list_args(self, args):
        start_msg = "LeViT + FFCV" if args.ffcv_load == 'Y' else "LeViT ONLY"
        self.log(start_msg)
        self.log("Listing arguments:")
        self.log(args)
        # for k, v in vars(args).items():
        #     self.log(f'{k}={v}')

    def log(self, msg):
        log.info(msg)
        self.msg_list.append(str(msg))

    def start_epoch(self, epoch):
        ep_name = self.get_ep_name(epoch)
        self.logs[ep_name] = {}
        self.logs[ep_name]['init_timestamp'] = self.now()

    def start_eval(self, epoch, split):
        ep_name = self.get_ep_name(epoch)
        self.logs[ep_name][f'{split}_eval_start_timestamp'] = self.now()

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self.temp_epoch[k].append(v)
            if 'acc' in k:
                self.best_acc_batch[k] = max(self.best_acc_batch[k], v)

    def sum_train(self, epoch):
        ep_name = self.get_ep_name(epoch)
        self.logs[ep_name]['train_timestamp'] = self.now()
        self.logs[ep_name]['train_dur'] = self.logs[ep_name]['train_timestamp'] - self.logs[ep_name]['init_timestamp']
        train_dur = self.logs[ep_name]['train_dur'].seconds
        train_speed = train_dur / self.n_batches['train']
        self.logs[ep_name]['train_speed'] = train_speed
        msg = f'{self.now()} - EPOCH [{ep_name.split("_")[-1]}] TRAIN_SPEED {round(train_speed, 5)} sec/batch'
        self.log(msg)

    def sum_eval(self, epoch, split):
        ep_name = self.get_ep_name(epoch)
        now = self.now()
        self.logs[ep_name][f'{split}_eval_end_timestamp'] = now
        self.logs[ep_name][f'{split}_eval_dur'] = now - self.logs[ep_name][f'{split}_eval_start_timestamp']
        eval_dur = self.logs[ep_name][f'{split}_eval_dur'].seconds
        eval_speed = eval_dur / self.n_batches[split]
        self.logs[ep_name][f'{split}_eval_speed'] = eval_speed
        loss = round(sum(self.temp_epoch[f'{split}_loss']) / len(self.temp_epoch[f'{split}_loss']), 4)
        acc1 = round(sum(self.temp_epoch[f'{split}_acc1']) / len(self.temp_epoch[f'{split}_acc1']), 2)
        acc5 = round(sum(self.temp_epoch[f'{split}_acc5']) / len(self.temp_epoch[f'{split}_acc5']), 2)
        self.logs[ep_name][f'{split}_best_acc1_batch'] = self.best_acc_batch[f'{split}_acc1']
        self.logs[ep_name][f'{split}_best_acc5_batch'] = self.best_acc_batch[f'{split}_acc5']
        self.best_acc_batch[f'{split}_acc1'] = 0
        self.best_acc_batch[f'{split}_acc5'] = 0
        best_batch_acc1 = round(self.logs[ep_name][f'{split}_best_acc1_batch'], 2)
        best_batch_acc5 = round(self.logs[ep_name][f'{split}_best_acc5_batch'], 2)
        msg = f'{self.now()} - EPOCH [{ep_name.split("_")[-1]}] ' \
              f'EVAL_{split}_SPEED {round(eval_speed, 5)} sec/batch | ' \
              f'LOSS {loss}, ACC1 {acc1}% (BEST {best_batch_acc1}%), ACC5 {acc5}% (BEST {best_batch_acc5}%)'
        self.log(msg)

    def sum_epoch(self, epoch):
        ep_name = self.get_ep_name(epoch)
        now = datetime.datetime.now().replace(microsecond=0)
        self.logs[ep_name]['f_timestamp'] = now
        self.logs[ep_name]['epoch_dur'] = now - self.logs[ep_name]['init_timestamp']
        ep = ep_name.split('_')[-1]
        all_duration = sum([self.logs[x]['train_dur'].seconds for x in self.logs if 'ep' in x])
        avg_time_diff = all_duration / (epoch+1)
        eta = (self.epochs - (epoch+1)) * avg_time_diff
        eta = datetime.timedelta(seconds=round(eta))
        train_speed = self.logs[ep_name]['train_speed']
        msg = f'{now} - EPOCH [{ep}]/{self.epochs}, TRAIN_SPEED {round(train_speed, 5)} sec/batch  ETA {eta}| '
        for k in self.temp_epoch:
            if len(self.temp_epoch[k]) == 0:
                # self.logs[ep_name][k] = None
                msg += f'{k}: NO CALC, '
            else:
                avg = sum(self.temp_epoch[k]) / len(self.temp_epoch[k])
                self.logs[ep_name][k] = avg
                if 'acc' in k:
                    val = str(round(avg, 2)) + '%'
                else:
                    val = str(round(avg, 4))
                if len(val) < 7:
                    msg += ' '*(7-len(val))
                msg += f'{k}: {val}, '
                self.temp_epoch[k] = []
        msg = msg[:-2]
        self.log(msg)

    def save_logged_data(self):
        df = pd.DataFrame.from_dict(self.logs, orient='index')
        cols = [col for col in CSV_COL_ORDER if col in df.columns]
        df = df[cols]
        if self.output_dir:
            df.to_csv(f'{self.output_dir}/logged_data.csv')
        with open(f'{self.output_dir}/raw_logs.txt', 'w') as f:
            for msg in self.msg_list:
                f.write(msg+'\n')


def is_photo(x: str):
    for f in ['jpg', 'jpeg', 'JPG', 'JPEG']:
        if f in x:
            return True
    return False


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    args.distributed = False
    return
    # if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    #     args.rank = int(os.environ["RANK"])
    #     args.world_size = int(os.environ['WORLD_SIZE'])
    #     args.gpu = int(os.environ['LOCAL_RANK'])
    # elif 'SLURM_PROCID' in os.environ:
    #     args.rank = int(os.environ['SLURM_PROCID'])
    #     args.gpu = args.rank % torch.cuda.device_count()
    # else:
    #     print('Not using distributed mode')
    #     args.distributed = False
    #     return
    #
    # args.distributed = True
    #
    # torch.cuda.set_device(args.gpu)
    # args.dist_backend = 'nccl'
    # print('| distributed init (rank {}): {}'.format(
    #     args.rank, args.dist_url), flush=True)
    # torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
    #                                      world_size=args.world_size, rank=args.rank)
    # torch.distributed.barrier()
    # setup_for_distributed(args.rank == 0)


def replace_batchnorm(net):
    for child_name, child in net.named_children():
        if hasattr(child, 'fuse'):
            setattr(net, child_name, child.fuse())
        elif isinstance(child, torch.nn.Conv2d):
            child.bias = torch.nn.Parameter(torch.zeros(child.weight.size(0)))
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(net, child_name, torch.nn.Identity())
        else:
            replace_batchnorm(child)


def replace_layernorm(net):
    import apex
    for child_name, child in net.named_children():
        if isinstance(child, torch.nn.LayerNorm):
            setattr(net, child_name, apex.normalization.FusedLayerNorm(
                child.weight.size(0)))
        else:
            replace_layernorm(child)
