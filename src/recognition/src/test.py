from matplotlib import markers
import torch
import torch.nn as nn
import numpy as np
import models.backbones.resnet as models
# import dataset.cifar10 as dataset
# from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import time,os,shutil
from datasets.reader import get_imglists
from datasets.GrainDataset import GrainDataset
import torch.utils.data as data
from progress.bar import Bar 
from utils.misc import AverageMeter,accuracy, confusion_matrix
from sklearn.metrics import confusion_matrix as sk_cm
from utils.util import plot
import albumentations as A
from albumentations import (RandomBrightness,RandomContrast,HueSaturationValue,Normalize,HorizontalFlip,VerticalFlip,Blur,
                            MotionBlur,OneOf,MedianBlur,IAAAdditiveGaussianNoise,GaussNoise,OpticalDistortion,RGBShift)
from albumentations import Resize

from albumentations.pytorch import ToTensorV2
import logging
from config import cfg
from pytorch_grad_cam import GradCAM
import cv2
from pytorch_grad_cam.utils.image import show_cam_on_image
# from sklearn.manifold import TSNE
from openTSNE import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

MEANS = (0.308562, 0.251994, 0.187898) # RGB
STDS  = (0.240441, 0.197289, 0.149387) # RGB

MEANS = MEANS[::-1]
STDS  = STDS[::-1]

val_transform = A.Compose([
    Resize(224,224),
    Normalize(mean=MEANS, std=STDS),
    ToTensorV2()
])

def vit_reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))
    # Bring the channels to the first dimension,like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def swin_reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))
    # Bring the channels to the first dimension,like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def test(valloader, model, model_name, criterion, save_name='wheat', gen_cam=False):
    
    # switch to evaluate mode
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if gen_cam:
        if model_name == 'resnet50':
            target_layers = [model.layer4[-1]]
            cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
        elif model_name == 'vit':
            target_layers = [model.transformer.blocks[-1].norm1]
            cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True, reshape_transform=vit_reshape_transform)
        elif model_name == 'swin':
            target_layers = [model.layers[-1].blocks[-1].norm2]
            cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True, reshape_transform=swin_reshape_transform)
        else:
            raise 'Not Implementation!'

    all_label = []
    all_pred = []
    embeds = []
    files = []

    end = time.time()
    bar = Bar('Testing', max=len(valloader))

    for batch_idx, (inputs, targets, filenames) in enumerate(valloader):
        data_time.update(time.time() - end)
        inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
        # compute output
        # outputs = model(inputs)
        with torch.no_grad():
            output_att, outputs, output_detail, proto_structure, proto_detail = model(inputs, targets)

        embeds.append(outputs.cpu().detach().numpy())
        files.append(filenames)
        loss = criterion(outputs, targets)

        if gen_cam and batch_idx<30:
            grayscale_cams = cam(input_tensor=inputs, target_category=targets)
            for ix in range(len(grayscale_cams)):
                grayscale_cam = grayscale_cams[ix, :]
                ori_img = cv2.imread(filenames[ix])
                ori_img = cv2.resize(ori_img,(224,224))
                fori_img = np.float32(ori_img) / 255
                visualization = show_cam_on_image(fori_img, grayscale_cam)
                visualization = np.concatenate((visualization,ori_img),axis=1)
                dir = f'./results/{save_name}_cam/' + str(targets[ix].item()) + '/'
                os.makedirs(dir, exist_ok=True)
                cv2.imwrite(dir + filenames[ix].split('/')[-1],visualization)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))

        all_label.append(targets.cpu().numpy())
        all_pred.append(torch.argmax(outputs,dim=-1).cpu().numpy())

        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(valloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()

    bar.finish()
    
    embeds = np.vstack(embeds)
    all_label = np.hstack(all_label)
    all_pred = np.hstack(all_pred)

    files = np.hstack(files).tolist()
    with open(f'./results/{save_name}_prob.txt','w+') as f:
        for emb, label, file_name in zip(embeds, all_label, files):
            file_name = '/'.join(file_name.split('/')[-2:])
            line = file_name + ' ' + str(label) + ' ' + ' '.join(list(map(str, emb))) + '\n'
            f.write(line)
        print('write prob txt down!')



model_name = 'tasn'
model_paths = ['checkpoints/tasn_rice.pth',
               'checkpoints/tasn_sorg.pth',
               'checkpoints/tasn_wheat.pth']


for model_path in model_paths:
    grain_t = model_path.split('/')[-2].split('_')[2]
    DATASET_PATH = f'/opt/data2/PaperDataset/{grain_t}/rec_crops'
    save_name = model_path.split('/')[-1].split('.')[0]
    model = torch.load(model_path)
    print('Load path from :',model_path)
    model = model.module
    model = model.cuda()
    loss = nn.CrossEntropyLoss().cuda()
    test_imgs = get_imglists(DATASET_PATH, split="test")
    test_labeled_dataset = GrainDataset(test_imgs, mode='train', transforms=val_transform)
    labeled_trainloader = data.DataLoader(test_labeled_dataset, batch_size=8, shuffle=True, num_workers=4, drop_last=False)
    test(labeled_trainloader, model, model_name, loss, save_name, gen_cam=False)
