import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import os
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
import torchvision.transforms as transforms
import pickle
from tqdm import tqdm


for grain in ['wheat', 'sorg', 'rice']: 
    
    sam_checkpoint = f'./checkpoints/{grain}_sam.pth'

    print(f'Loading {sam_checkpoint}...')
    model_type = "vit_b"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=35,
        pred_iou_thresh=0.8,
        box_nms_thresh=0.5,
        stability_score_thresh=0.85,
        crop_n_layers=0,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=800,  # Requires open-cv to run post-processing
    )

    data_root = f'/opt/data2/PaperDataset/{grain}/crop_images/test/'
    
    pred_datas = []

    for fn in tqdm(os.listdir(data_root), desc='Infering'):
        fp = os.path.join(data_root, fn)
        image = cv2.imread(fp)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_h, im_w, _ = image.shape
        ucrop = image[:, :im_w//2, :]
        dcrop = image[:, im_w//2:, :]
        image = np.concatenate([ucrop, dcrop], axis=2)
        anns = mask_generator.generate(image)
        boxes = []
        logits = []
        for ann in anns:
            boxes.append(ann['bbox'])
            logits.append(ann['category_logits'])
        
        boxes = torch.tensor(boxes)
        logits = torch.tensor(logits)

        if len(logits):
            scores, labels = torch.max(torch.softmax(logits, dim=1),dim=1)
            items = {'img_path':fp,  'pred_instances':{'bboxes':boxes, 'scores':scores, 'labels':labels}}
            pred_datas.append(items)
    
            os.makedirs('./results', exist_ok=True)
            with open(f'results/{grain}_sam.pkl', 'wb') as f:
                pickle.dump(pred_datas,f)

