import cv2

from .BaseDataset import BaseDataset

__all__ = ['GrainDataset']


class GrainDataset(BaseDataset):
  def __init__(self, imglist, mode='train',transforms=None):
    super(GrainDataset,self).__init__(mode, transforms)
    self.mode = mode
    self.transforms = transforms
    self.imglist = imglist
    imgs = []
    for index, row in imglist.iterrows():
      imgs.append((row['filename'],row['label']))
    self.imgs = imgs

  def __len__(self):
    return len(self.imgs)


  def __getitem__(self,index):

    filename, label = self.imgs[index]
    img = cv2.imread(filename)
    img = self.transforms(image=img)['image']
    return img, label, filename

