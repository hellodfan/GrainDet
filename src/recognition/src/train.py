
import os
import random
import time
import copy
import logging
import numpy as np
from progress.bar import Bar

import torch
import torch.optim

from utils.init_project import init_everything
from utils.filesystem import save_checkpoint
from utils.misc import AverageMeter, accuracy, show_flops_params, F1Score
from utils.options import parse_args

from models.model import get_model
from datasets.reader import get_imglists
from datasets.GrainDataset import GrainDataset
from datasets.augment import get_transforms, mixup_data

from solver import get_optimizer, get_lr_scheduler, get_loss, get_current_lr, update_lr, mixup_criterion
from config import cfg
import torch.nn as nn

import torch.nn.functional as F

best_acc = 0
best_loss = 100

def main(args):
    global best_acc
    global best_loss
    init_everything()
    
    train_imgs = get_imglists( dataroot=cfg.DATASET.PATH, split="train", phase=cfg.PHASE)
    val_imgs = get_imglists(dataroot=cfg.DATASET.PATH, split="test", phase=cfg.PHASE)

    train_transform, val_transform = get_transforms(RandBright_limit=0.2, RandBright_ratio=0.5, RandContra_limit=0.2, RandContra_ratio=0.5)
    train_dataset = GrainDataset(train_imgs, mode='train', transforms=train_transform)
    val_dataset = GrainDataset( val_imgs, mode='train', transforms=val_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg.TRAIN.BATCH, shuffle=True,
                                               num_workers=0, pin_memory=False, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=cfg.TEST.BATCH, shuffle=True,
                                             num_workers=0, pin_memory=False, drop_last=True)
    #############  model config  #############
    model = get_model(cfg, num_classes=cfg.DATASET.CLASS_NUMS)

    print('MODEL RESUME PATH:',cfg.MODEL.RESUME_PATH)
    model = model.to(device=device)
    model = nn.DataParallel(model)
    try:
        show_flops_params(copy.deepcopy(model), 'cuda')
    except Exception as e:
        logging.warning('get flops and params error: {}'.format(e))

    from collections import Counter
    count_dic = Counter(train_imgs.label.to_list())
    samples_per_cls = [count_dic[k] for k in sorted(count_dic.keys())]

    criterion = get_loss(loss_name='combine', samples_per_cls=samples_per_cls)

    optimizer = get_optimizer(model, optim_name =cfg.OPTIM.NAME, learn_rate=cfg.OPTIM.INIT_LR)
    lr_scheduler = get_lr_scheduler(optimizer,lr_mode = cfg.OPTIM.LR_SCHEDULER)

    # Train
    iters_per_epoch = len(train_dataset) // (cfg.TRAIN.BATCH)
    max_iters = cfg.TRAIN.EPOCHS * iters_per_epoch
    logging.info('Start training, Total Epochs:{:d} = Total Iterations {:d}'.format(
        cfg.TRAIN.EPOCHS, max_iters))

    for epoch in range(cfg.TRAIN.START_EPOCH, cfg.TRAIN.EPOCHS):

        print('\n Epoch: [%d | %d] LR: %f' % (epoch + 1, cfg.TRAIN.EPOCHS,optimizer.param_groups[0]['lr']))

        train_loss, train_acc, train_2 = train(model, train_loader, criterion, optimizer, epoch, cfg.TRAIN.EPOCHS)
        val_loss, val_acc, test_2 = validate(model, val_loader, criterion, epoch, cfg.TRAIN.EPOCHS)

        # update lr, value depending on the type of lr scheduler
        update_lr(lr_scheduler, cfg.OPTIM.LR_SCHEDULER, value=epoch)
        lr_current = get_current_lr(optimizer)

        print('train_loss:%f, val_loss:%f, train_acc:%f, train_top2:%f,\
        val_acc:%f, val_top2:%f' % (train_loss, val_loss, train_acc, train_2, val_acc, test_2))

        logging.info(
            "Epoch: {:d}/{:d} || Lr: {:.6f} || Train Loss: {:.6f} || Val Loss: {:.6f} || train_acc:{:.6f} || val_acc:{:.6f} ".format(epoch, cfg.TRAIN.EPOCHS, lr_current, train_loss, val_loss, train_acc, val_acc))

        # save model
        is_best = val_acc > best_acc
        is_best_loss = val_loss < best_loss
        best_acc = max(val_acc, best_acc)
        print('Best acc:',best_acc)
        best_loss = min(val_loss, best_loss)

        save_checkpoint(model, epoch + 1, optimizer, lr_scheduler, is_best, best_acc)

    print('Best acc:',best_acc)


def train(model, train_loader, criterion, optimizer, epoch, epochs):
    model.train()
    batch_time, data_time = AverageMeter(), AverageMeter()
    losses, top1, top2, f1_avg = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    end = time.time()

    alpha = 0.5 + 0.5*(epoch+1)/epochs if epoch>0 else 0

    bar = Bar('Training: ', max=len(train_loader))

    for batch_idx, (inputs, targets, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs, targets = inputs.to(device=device), targets.to(device=device)
        # compute output
        output_att, outputs, output_detail, proto_structure, proto_detail = model(inputs, targets)
        loss = (1-alpha) * (criterion(outputs, targets) + criterion(output_att, targets)*0.2 + criterion(output_detail, targets)) \
                + alpha * (F.cross_entropy(proto_structure, targets) +  F.cross_entropy(proto_detail, targets))

        # measure accuracy and record loss
        prec1, prec2 = accuracy(outputs.data, targets.data, topk=(1, 2))

        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top2.update(prec2.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # clip gradient
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=5.0, norm_type=2)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s |Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}| top2: {top2: .4f}| '.format(
            batch=batch_idx + 1,
            size=len(train_loader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top2=top2.avg,
        )
        bar.next()

    bar.finish()

    return (losses.avg, top1.avg, top2.avg)


def validate(model, val_loader, criterion, epoch, epochs):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    # f1_avg = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    alpha = 0.5 + 0.5*(epoch+1)/epochs if epoch>0 else 0

    bar = Bar('Validating: ', max=len(val_loader))
    # f1_score = F1Score()

    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs, targets = inputs.to(device=device), targets.to(device=device)
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            # compute output
            output_att, outputs, output_detail, proto_structure, proto_detail = model(inputs, targets)
            loss = (1-alpha) * (criterion(outputs, targets) + criterion(output_att, targets)*0.2 + criterion(output_detail, targets)) \
                    + alpha * (F.cross_entropy(proto_structure, targets) +  F.cross_entropy(proto_detail, targets))

            # measure accuracy and record loss
            prec1, prec2 = accuracy(outputs.data, targets.data, topk=(1, 2))

            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top2.update(prec2.item(), inputs.size(0))
            # f1_avg.update(f1.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s| Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} |top1: {top1: .4f} | top2: {top2: .4f}| '.format(
                batch=batch_idx + 1,
                size=len(val_loader),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg,
                top2=top2.avg,
            )
            bar.next()

    bar.finish()

    return (losses.avg, top1.avg, top2.avg)

if __name__ == "__main__":

    args = parse_args()
    print(args)
    cfg.update_from_file(args.config_file)
    cfg.update_from_list(args.opts)
    cfg.PHASE = 'train'

    if args.model_name:
        cfg.MODEL.NAME = args.model_name

    if args.save_name:
        cfg.DATASET.NAME = args.save_name

    if args.data_path:
        cfg.DATASET.PATH = args.data_path

    if args.phase:
        cfg.PHASE = args.phase

    device = torch.device("cuda")
    main(args)
