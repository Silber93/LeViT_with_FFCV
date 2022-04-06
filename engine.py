# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, logger: utils.CustomLogger,
                    clip_grad: float = 0,
                    clip_mode: str = 'norm',
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True):
    model.train(set_training_mode)
    logger.start_epoch(epoch)
    # metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('lr', utils.SmoothedValue(
    #     window_size=1, fmt='{value:.6f}'))
    # header = 'Epoch: [{}]'.format(epoch)
    print_freq = max(int(logger.n_batches['train']/10), 1)
    i = 0
    # for samples, targets in metric_logger.log_every(
    #         data_loader, print_freq, header):
    for samples, targets in data_loader:
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)

        loss_value = loss.item()
        logger.update(train_loss=loss_value)
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        if i % print_freq == 0:
            n_batches = f'/{logger.n_batches["train"]}' if logger.n_batches["train"] > 0 else ''
            msg = f'{logger.now()} - TRAINING batch {i}{n_batches} | ' \
                  f'LOSS FOR CURRENT BATCH: {round(loss_value, 4)}'
            logger.log(msg)
        i += 1
        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=clip_grad, clip_mode=clip_mode,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
    if logger.n_batches['train'] == 0:
        logger.n_batches['train'] = i
        # metric_logger.update(loss=loss_value)
        # metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, data_type, logger: utils.CustomLogger):
    criterion = torch.nn.CrossEntropyLoss()
    # switch to evaluation mode
    model.eval()
    logger.log(f'{logger.now()} - EVALUATING {data_type}')
    print_freq = int(logger.n_batches[data_type] / 4)
    i = 0
    for images, target in data_loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        res = {f'{data_type}_acc1': acc1.item(), f'{data_type}_acc5': acc5.item()}
        if data_type == 'val':
            res[f'{data_type}_loss'] = loss.item()
        logger.update(**res)
        if i % print_freq == 0:
            n_batches = f'/{logger.n_batches[data_type]}' if logger.n_batches[data_type] > 0 else ''
            msg = f'{logger.now()} - EVALUATION batch {i}{n_batches} [{data_type}] ' \
                  f'CURRENT LOSS: {round(loss.item(), 4)} | ' \
                  f'ACCURACY FOR CURRENT BATCH (TOP@1): {round(acc1.item(), 2)}%'
            logger.log(msg)
        i += 1
    if logger.n_batches[data_type] == 0:
        logger.n_batches[data_type] = i
