
__all__ = ['BaseDataset']


class BaseDataset(object):
  def __init__(self,mode,transform):
    self.mode = mode
    self.transform = transform


  def __getitem__(self, index):
    raise NotImplementedError

