#!/usr/bin/env python3

import datetime
import json
import os
import re
import cv2
import fnmatch
from PIL import Image
import numpy as np
from joblib import delayed, Parallel
from pycococreator import create_image_info, create_annotation_info

INFO = {
    "description": "Example Dataset",
    "version": "0.1.0",
    "year": 2023,
    "contributor": "waspinator",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'NOR',
        'supercategory': 'Unsound',
    },
    {
        'id': 2,
        'name': 'FS',
        'supercategory': 'Unsound',
    },
    {
        'id': 3,
        'name': 'SD',
        'supercategory': 'Unsound',
    },
    {
        'id': 4,
        'name': 'MY',
        'supercategory': 'Unsound',
    },
    {
        'id': 5,
        'name': 'AP',
        'supercategory': 'Unsound',
    },
    {
        'id': 6,
        'name': 'BN',
        'supercategory': 'Unsound',
    },
    {
        'id': 7,
        'name': 'BP',
        'supercategory': 'Unsound',
    },
    {
        'id': 8,
        'name': 'IM',
        'supercategory': 'Unsound',
    },
]


def crop(image1, image2, masks, imgname, img_id, msk_id, subsize=1500, gap=300):
    
    img_infos = []
    ann_infos = []

    ## ========================= crop image ========================= 
    img1 = np.asarray(image1)
    img2 = np.asarray(image2)           
    img_h,img_w = img1.shape[:2]
    top = 0
    reachbottom = False
    while not reachbottom:
        reachright = False
        left = 0
        if top+subsize>=img_h:
            reachbottom = True
            top = max(img_h-subsize,0)
        while not reachright:
            if left+subsize>=img_w:
                reachright = True
                left = max(img_w-subsize,0)

            imgsplit1 = img1[top:min(top+subsize,img_h),left:min(left+subsize,img_w)]
            imgsplit2 = img2[top:min(top+subsize,img_h),left:min(left+subsize,img_w)]
            msksplit = masks[:, top:min(top+subsize,img_h),left:min(left+subsize,img_w)]

            if imgsplit1.shape[:2] != (subsize,subsize):
                temp_img1 = np.zeros((subsize,subsize,3),dtype=np.uint8)
                temp_img1[0:imgsplit1.shape[0],0:imgsplit1.shape[1]] = imgsplit1
                imgsplit1 = temp_img1

                temp_img2 = np.zeros((subsize,subsize,3),dtype=np.uint8)
                temp_img2[0:imgsplit1.shape[0],0:imgsplit1.shape[1]] = imgsplit2
                imgsplit2 = temp_img2

                temp_msk = np.zeros((msksplit.shape[0], subsize, subsize),dtype=np.uint8)
                temp_msk[:, 0:imgsplit1.shape[0], 0:imgsplit1.shape[1]] = msksplit
                msksplit = temp_msk

            combine = np.concatenate([imgsplit1, imgsplit2], axis=1)
            imgsplit1 = Image.fromarray(np.uint8(imgsplit1))
            imgsplit2 = Image.fromarray(np.uint8(imgsplit2))
            nimgname = imgname.replace('_U_p600s.png', '_UD_p600s_' + str(left) + '_' + str(top) + '.png')
            nimgname = nimgname.replace('/images/', '/crop_images/')
            imgdir  = '/'.join(nimgname.split('/')[:-1])
            os.makedirs(imgdir, exist_ok=True)
            
            save_flag = False
            for msk in msksplit:
                class_id = msk.max() 
                if class_id!=0 and np.sum(msk>0)>400:
                    category_info = {'id': class_id, 'is_crowd': 'crowd' in nimgname}
                    binary_mask = (msk>0).astype(np.uint8)
                    
                    annotation_info = create_annotation_info(
                        msk_id, img_id, category_info, binary_mask,
                        imgsplit1.size, tolerance=2)

                    if annotation_info is not None:
                        ann_infos.append(annotation_info)
                        save_flag = True

                    msk_id += 1

            if save_flag:
                cv2.imwrite(nimgname, combine[:,:,::-1])
                image_info = create_image_info(
                    img_id, os.path.basename(nimgname), imgsplit1.size)
                img_infos.append(image_info)
                img_id += 1

            left += subsize-gap

        top+=subsize-gap


    return img_infos, ann_infos, img_id, msk_id




def main(grain, phase='train'):

    ROOT_DIR = f'/opt/data1/docker/data/test_1016/{grain}'

    IMAGE_DIR = os.path.join(ROOT_DIR, "images", phase)

    gtype = 'wheat'
    if 'sorg' in IMAGE_DIR:
        CATEGORIES[6]["name"] = 'HD'
        gtype = 'sorg'
    elif 'rice' in IMAGE_DIR:
        CATEGORIES[6]["name"] = 'UN'
        gtype = 'rice'

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    coco_crop_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1

    crop_img_id, crop_msk_id = 1, 1
    
    # filter for jpeg images
    for root, _, files in os.walk(IMAGE_DIR):
        image_files = [os.path.join(root, f) for f in files if '_U_' in f]
        # go through each image
        for image_filename in image_files:
            print(f'Processing {image_filename}.....')
            image = Image.open(image_filename)
            image2 = Image.open(image_filename.replace('_U_', '_D_'))
            image_info = create_image_info(
                image_id, os.path.basename(image_filename), image.size)
            coco_output["images"].append(image_info)

            msk_path = image_filename.replace('images','npz').replace('_U_','_UD_').replace('.png','.npz')
            masks = np.load(msk_path)['arr_0']

            # go through each associated annotation
            for msk in masks:
                
                class_id = msk.max() 
                category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
                binary_mask = (msk>0).astype(np.uint8)
                
                annotation_info = create_annotation_info(
                    segmentation_id, image_id, category_info, binary_mask,
                    image.size, tolerance=2)

                if annotation_info is not None:
                    coco_output["annotations"].append(annotation_info)

                segmentation_id = segmentation_id + 1

            
            crop_img_infos, crop_msk_infos, crop_img_id, crop_msk_id = crop(image, image2, masks, image_filename, crop_img_id, crop_msk_id)
            coco_crop_output['images'].extend(crop_img_infos)
            coco_crop_output['annotations'].extend(crop_msk_infos)

            image_id = image_id + 1

            with open(f'{ROOT_DIR}/instances_{gtype}_{phase}2023.txt', 'w') as output_json_file:
                # json.dump(coco_output, output_json_file)
                output_json_file.write(str(coco_output))

            with open(f'{ROOT_DIR}/instances_{gtype}_crop_{phase}2023.txt', 'w') as output_json_file:
                # json.dump(coco_crop_output, output_json_file)
                output_json_file.write(str(coco_crop_output))


if __name__ == "__main__":

    # tasks = [(grain, phase) for grain in ['sorg','wheat', 'rice']  for phase in ['test','train','val']]
    tasks = [('sorg', 'test')]
    Parallel(n_jobs=-1)(delayed(main)(grain, phase)  for grain, phase in tasks)
