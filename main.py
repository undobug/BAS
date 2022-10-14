# --------------------------------------------------------
#!/usr/bin/env python
# Licensed under The MIT License [see LICENSE for details]
# Written by fyb
# --------------------------------------------------------


import os
from stat import UF_APPEND
import sys
import time
import json
import argparse
import datetime
import numpy as np
import pandas as pd

import torch
from torch import optim as optim
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from timm.utils import AverageMeter
from tqdm import tqdm
from config import get_config
from models import build_model
from data import dataset 
from utils.logger import create_logger
from utils.gather import Recorder, distributed_concat
from utils.optimizer import build_optimizer, build_scheduler
from utils.distributed_utils import reduce_tensor, is_main_process, init_distributed_mode
from utils.utils import set_seed, visible_gpus,compute_metrics, auto_resume_helper, load_checkpoint, \
                        load_pretrained, save_checkpoint, NativeScalerWithGradNormCount

import warnings
warnings.filterwarnings("ignore")


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    # easy config modification
    parser.add_argument('-g', '--device-id', help="modify device_id", type=str)
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--pretrained',help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')


    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return args, config


def main(config):
    
    dataset_val, data_loader_train, data_loader_val, data_loader_test= dataset.build_loader(config)
  
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    logger.info(str(model))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    model.cuda()
    model_without_ddp = model

    optimizer = build_optimizer(config, model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], broadcast_buffers=False)
    loss_scaler = NativeScalerWithGradNormCount() #TODO

    # if config.TRAIN.ACCUMULATION_STEPS > 1:
    #     lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    # else:
    #     lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    lr_scheduler = build_scheduler(config, optimizer)
    criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config, config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger)
        WA,UA,WF1,CM = validate(config, data_loader_val, model)
        max_WA, max_UA, max_WF1, cm = WA, UA, WF1, CM
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test samples: {WA:.4f}%  {UA:.4f}% {WF1:.4f}%")
        if config.EVAL_MODE:
            return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model_without_ddp, logger)
        WA,UA,WF1,CM = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test samples: {WA:.4f}%  {UA:.4f}% {WF1:.4f}%")


    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)
        loss_avg = train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, lr_scheduler, loss_scaler)
        lr_scheduler.step()
        WA,UA,WF1,CM = validate(config, data_loader_val, model)
        if config.DATA.DATASET == 'meld':
            max_accuracy = max(max_accuracy, WF1)
            WA_test,UA_test,WF1_test,CM_test = validate(config, data_loader_test, model)
        else:
            max_accuracy = max(max_accuracy, WA+UA)

        ############# IEMOCAP #############
        if max_accuracy == WA+UA and config.DATA.DATASET != 'meld':
            max_WA, max_UA, max_WF1, cm = WA, UA, WF1, CM
            if rank ==0:
                save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler,
                                logger)

        ############# MELD #############
        if max_accuracy == WF1 and config.DATA.DATASET == 'meld':
            max_WA, max_UA, max_WF1, cm = WA_test,UA_test,WF1_test,CM_test
            if rank ==0:
                save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler,
                                logger)
        logger.info(f'Max WA: {max_WA:.4f}%, Max UA: {max_UA:.4f}%, Max WF1: {max_WF1:.4f}%')


        if rank == 0:
            tags = ["loss", "accuracy", "learning_rate"]
            tb_writer.add_scalar(tags[0], loss_avg, epoch)
            tb_writer.add_scalar(tags[1], WA, epoch)
            tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    return max_WA, max_UA, max_WF1, cm


def train_one_epoch(config, model, criterion,data_loader, optimizer, epoch, lr_scheduler, loss_scaler):
    model.train()
    optimizer.zero_grad()
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)
    num_steps = len(data_loader)

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    predict = Recorder(device='cuda', dtype=torch.int64)
    label = Recorder(device='cuda', dtype=torch.int64)

    start = time.time()
    end = time.time()
    for idx, data in enumerate(data_loader):
        samples, labels,dom = data

        samples = samples.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        dom = dom.cuda(non_blocking=True)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            out_emo = model(samples)

        loss_emo = criterion(out_emo, labels)
        loss = loss_emo 
        loss = loss / config.TRAIN.ACCUMULATION_STEPS

        # this attribute is added by timm on one optimizer (adahessian)  
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:    
            optimizer.zero_grad()    
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()   

        loss_meter.update(loss.item(), labels.size(0))
        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()
      
        y_pred = torch.argmax(out_emo, dim=1)
        predict.record(y_pred)
        label.record(labels)
        
    all_pred = distributed_concat(predict.data)
    all_label = distributed_concat(label.data)

    if rank ==0:
        wa, ua, wf1, cm = compute_metrics(all_label, all_pred)
        lr = optimizer.param_groups[0]['lr']
        wd = optimizer.param_groups[0]['weight_decay']
        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        logger.info(
            f'Train: [{epoch}/{config.TRAIN.EPOCHS}]\t'
            f'lr {lr:.6f}\t'
            f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
            f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
            f'WA {wa:.5f}\t'
            f'UA {ua:.5f}\t'
            f'WF1 {wf1:.5f}')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    return loss_meter.avg


@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    # if is_main_process():
    #     data_loader = tqdm(data_loader, file=sys.stdout)

    loss_meter = AverageMeter()
    predict = Recorder(device='cuda', dtype=torch.int64)
    label = Recorder(device='cuda', dtype=torch.int64)

    end = time.time()
    for idx, data in enumerate(data_loader):
        samples, labels,dom = data
        samples = samples.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            out_emo = model(samples)        
        y_pred = torch.argmax(out_emo, dim=1)

        # measure accuracy and record loss
        predict.record(y_pred)
        label.record(labels)
        loss = criterion(out_emo, labels)
        loss = reduce_tensor(loss)
        loss_meter.update(loss.item(), labels.size(0))

    all_pred = distributed_concat(predict.data)
    all_label = distributed_concat(label.data)
    wa, ua, wf1, cm = compute_metrics(all_label, all_pred)

    logger.info(
            f'Test: \t'
            f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
            f'WA {wa:.5f}\t'
            f'UA {ua:.5f}\t'
            f'WF1 {wf1:.5f}')
    return wa, ua, wf1, cm


def main_fun(config):
    
    if config.DATA.DATASET == 'iemocap':
        folds = [1, 2, 3, 4, 5]
    elif config.DATA.DATASET == 'meld':
        folds = [1]
    else:
        raise KeyError(f'Unknown database: {config.dataset.database}')

    ua,wa,wf1 =[],[],[]
    for f in folds:
        cfg_clone = config.clone()
        cfg_clone.defrost()
        cfg_clone.TRAIN.CURRENT_FOLD = f
        cfg_clone.freeze()
        print(cfg_clone)
        max_WA,max_UA, max_WF1, cm = main(cfg_clone)
        wa.append(max_WA)
        ua.append(max_UA)
        wf1.append(max_WF1)
        if rank == 0:
            f=open(config.OUTPUT + "/runing.txt","a")
            f.write(str(cm)+ "\n" + f'fold-{cfg_clone.TRAIN.CURRENT_FOLD}: \
                    WA {max_WA:.5f}\t UA {max_UA:.5f}\t WF1 {max_WF1:.5f}\t'+ "\n")
        torch.cuda.empty_cache()
    dataframe = pd.DataFrame({'WA_test':wa, 'UA_test':ua, 'WF1_test':wf1})
    dataframe.to_csv(os.path.join(config.OUTPUT, 'result.csv'), index=False, sep=',')


if __name__ == '__main__':
    args, config = parse_option()
    visible_gpus(f'{config.TRAIN.DEVICE_ID}')
    init_distributed_mode(args=args)
    rank = args.rank

    set_seed(config.SEED + rank)
    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")
    tb_writer = SummaryWriter()

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))
    main_fun(config)





