import os
import pandas as pd
import random

__all__ = ['get_imglists']

def get_imglists(dataroot, split='train', phase='train'):
    '''
    get all images path
    @param: 
        root : root path to dataset
        spilt: sub path to specific dataset folder
    '''

    grain = dataroot.split('/')[-1]
    imgs, labels = [], []

    if split == 'train':
        split = phase


    dic = {'NOR':0, 'FS':1, 'F&S':1, 'SD':2, 'MY':3, 'AP':4, 'BN':5, 'BP':6, 'IM':7, 'HD':6, 'RSS':6, 'UN':6}

    print(os.path.join(dataroot, split), 'exist:', os.path.exists(os.path.join(dataroot, split)))

    for root, _, files in os.walk(os.path.join(dataroot, split)):
        for f in files:
            label = dic[root.split('/')[-1].split('_')[1]]
            im_path = os.path.join(root, f)
            if os.path.exists(im_path):
                imgs.append(im_path)
                labels.append(label)
            else:
                print(f'{im_path} does not exist!')
    
    length = len(imgs)
    print(f'* {split} : {length}')
    files = pd.DataFrame({'filename': imgs, 'label': labels})
    return files
