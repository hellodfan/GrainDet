import os
import cv2
import torch
import pickle
import numpy as np
from tools import nms, filter_box
from collections import defaultdict

from lib.BoundingBox import BoundingBox
from lib.BoundingBoxes import BoundingBoxes
from lib.Evaluator import *
from lib.utils import *


def softmax(z):
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)

for grain in ['wheat', 'sorg', 'rice']:
    gt_dir =  f'/opt/data2/PaperDataset/{grain}/txt/test'
    gt_data = {}

    for fn in os.listdir(gt_dir):
        gt_path = os.path.join(gt_dir, fn)
        with open(gt_path, 'r') as f:
            lines = f.readlines()
            single_data = [list(map(int, line.replace('\n','').split(' '))) for line in lines]
        
        gt_data[fn[:-4]] = single_data


    for model in ['tasn', 'sam', 'frcnn', 'yolox', 'maskrcnn', 'rtmdet']:

        allBoundingBoxes = BoundingBoxes()

        if model == 'tasn':
            tasn_dic = defaultdict(list)
            with open(f'./data/{grain}_tasn.txt', 'r') as f:
                lines = f.readlines()
                for line in lines:
                    lines = line.replace('\n','')
                    ss = line.split(' ')
                    name = '_'.join(ss[0].split('/')[-1].split('_')[:-1])
                    cont = list(map(float, ss[1:]))
                    tasn_dic[name].append(cont)

            merge_pred = defaultdict(list)
            for key in gt_data.keys(): 
                for ix, box in enumerate(gt_data[key]):
                    tasn_val = tasn_dic[key][ix]
                    box[0] = int(tasn_val[0])
                    logit = np.array(tasn_val[1:])
                    logit = softmax(logit)
                    pred_label = np.argmax(logit)
                    score = max(logit)
                    pred_box = [pred_label, score] + box[1:]
                    merge_pred[key].append(pred_box)
                    
        else:
            with open(f'./data/{grain}_{model}.pkl', 'rb') as f:
                pred_data = pickle.load(f)

            merge_pred = defaultdict(list)

            for item in pred_data:

                boxes = item['pred_instances']['bboxes'].numpy()
                labels = item['pred_instances']['labels'].numpy()
                scores = item['pred_instances']['scores'].numpy()
                img_path = item['img_path']
                pos = img_path.replace('.png','').split('_')[-2:]
                pos = list(map(int, pos))

                fn = '_'.join(img_path.split('/')[-1].split('_')[:-2])

                splitsize = 1500

                if model=='sam':
                    boxes[:,2] = boxes[:,0] + boxes[:,2]
                    boxes[:,3] = boxes[:,1] + boxes[:,3]

                # filt boxes 3644, 5480
                mask = filter_box(splitsize, boxes, pos, rows=3644, cols=5480)
                boxes, scores, labels = boxes[mask], scores[mask], labels[mask]

                mask_score = scores >= 0.5
                boxes = boxes[mask_score].astype(np.int32)
                scores = scores[mask_score]
                labels = labels[mask_score]

                # ori position
                off_x = pos[0]
                off_y = pos[1]
                boxes[:,0::2] += off_x
                boxes[:,1::2] += off_y

                temp_data = []
                for bx, score, label in zip(boxes, scores, labels):
                    temp_data.append([label]+[score]+list(bx))
                
                merge_pred[fn] += temp_data


        for fn in gt_data.keys():
            single_gt_data = gt_data[fn]
            single_pred_data = torch.tensor(merge_pred[fn])

            keep = nms(single_pred_data[:,2:], single_pred_data[:, 1], threshold=0.99 if model=='tasn' else 0.5)
            single_pred_data = single_pred_data[keep]
            single_pred_data = single_pred_data.tolist()

            for gt_bx in single_gt_data:
                idClass = str(int(gt_bx[0]))
                bb = BoundingBox(fn, idClass, gt_bx[1], gt_bx[2], gt_bx[3], gt_bx[4], 
                                 CoordinatesType.Absolute, bbType=BBType.GroundTruth, format=BBFormat.XYX2Y2)
                allBoundingBoxes.addBoundingBox(bb)
            
            for pred_bx in single_pred_data:
                idClass = str(int(pred_bx[0]))
                bb = BoundingBox(fn, idClass, pred_bx[2], pred_bx[3], pred_bx[4], pred_bx[5], 
                            CoordinatesType.Absolute, bbType=BBType.Detected, classConfidence=pred_bx[1], format=BBFormat.XYX2Y2)
                allBoundingBoxes.addBoundingBox(bb)

            # udimg_path = f'/opt/data1/docker/data/test_1016/{grain}/images/test/{fn}.png'
            # img = cv2.imread(udimg_path.replace('_UD_','_U_'))
            # for line in single_pred_data:
            #     x1, y1, x2, y2 = tuple(map(int, line[2:]))
            #     img = cv2.rectangle(img, (x1, y1), (x2, y2), (255,22,0), 5)
            #     img  = cv2.putText(img, str(line[0])+'_'+str(line[1])[:3], (x1,y1), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=3, color=(255,22,0), thickness=3)
            # cv2.imwrite(f'./show/{fn}_{grain}_{model}.png', img)


        evaluator = Evaluator()
        acc_AP = 0
        validClasses = 0

        # Plot Precision x Recall curve
        detections = evaluator.PlotPrecisionRecallCurve(
            allBoundingBoxes,  # Object containing all bounding boxes (ground truths and detections)
            IOUThreshold=0.5,  # IOU threshold
            method=MethodAveragePrecision.EveryPointInterpolation,
            showAP=True,  # Show Average Precision in the title of the plot
            showInterpolatedPrecision=False,  # Don't plot the interpolated precision curve
            savePath='./results',
            showGraphic=False)

        f = open(f'./results/{grain}_{model}_ap.txt', 'w')
        f.write('Object Detection Metrics\n')
        f.write('https://github.com/rafaelpadilla/Object-Detection-Metrics\n\n\n')
        f.write('Average Precision (AP), Precision and Recall per class:')

        print('=='*10, f'{grain} {model}', '=='*10)

        # each detection is a class
        for metricsPerClass in detections:

            # Get metric values per each class
            cl = metricsPerClass['class']
            ap = metricsPerClass['AP']
            precision = metricsPerClass['precision']
            recall = metricsPerClass['recall']
            totalPositives = metricsPerClass['total positives']
            total_TP = metricsPerClass['total TP']
            total_FP = metricsPerClass['total FP']

            if totalPositives > 0:
                validClasses = validClasses + 1
                acc_AP = acc_AP + ap
                prec = ['%.2f' % p for p in precision]
                rec = ['%.2f' % r for r in recall]
                ap_str = "{0:.2f}%".format(ap * 100)
                # ap_str = "{0:.4f}%".format(ap * 100)
                print('AP: %s (%s)' % (ap_str, cl))
                f.write('\n\nClass: %s' % cl)
                f.write('\nAP: %s' % ap_str)
                f.write('\nPrecision: %s' % prec)
                f.write('\nRecall: %s' % rec)

        mAP = acc_AP / validClasses
        mAP_str = "{0:.2f}%".format(mAP * 100)
        print('mAP: %s' % mAP_str)
        f.write('\n\n\nmAP: %s' % mAP_str)

        print('=='*25)

            
            



