
from config import cfg
from torch.optim import lr_scheduler


__all__ = ['get_lr_scheduler','get_current_lr','update_lr']



def step(optimizer, Step_size=10, Gamma=0.5):
  return lr_scheduler.StepLR(optimizer,step_size=Step_size,gamma=Gamma)



def on_loss(optimizer, Mode='min',Factor=0.2,Patience=5,Verbose=False):
  return lr_scheduler.ReduceLROnPlateau(optimizer, mode=Mode, 
                    factor=Factor, patience=Patience, verbose=Verbose)


def on_acc(optimizer, Mode='max', Factor=0.2, Patience=5, Verbose=False):
  return lr_scheduler.ReduceLROnPlateau(optimizer, mode=Mode, 
                    factor=Factor, patience=Patience, verbose=Verbose)




lr_schedulers  = {
  'step': step,
  'on_loss': on_loss,
  'on_acc': on_acc,
}



def get_lr_scheduler(optimizer, lr_mode = cfg.OPTIM.LR_SCHEDULER,**kwargs):
  lr_mode = lr_mode.lower()
  if lr_mode in  lr_schedulers:
    return lr_schedulers[lr_mode](optimizer,**kwargs) 
  else:
    print("lr scheduler: {} is not implemented..., using default schedular: STEP.".format(lr_mode))
    return lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)
  


def get_current_lr(optimizer):
  for param_group in optimizer.param_groups:
    return param_group['lr']




def update_lr(scheduler, lr_mode = cfg.OPTIM.LR_SCHEDULER, **kwargs):
  lr_mode = lr_mode.lower()
  if lr_mode in lr_schedulers:
    scheduler.step(kwargs['value'])
  else:
    scheduler.step(kwargs['value'])