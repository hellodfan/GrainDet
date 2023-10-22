from torch import optim
from config import cfg
from .optimizers import *
# from adamp import AdamP


__all__ = ['get_optimizer']




def adam(parameters, learn_rate=cfg.OPTIM.INIT_LR, 
                     solver_beta=(cfg.OPTIM.BETA1,cfg.OPTIM.BETA2), 
                     wd=cfg.OPTIM.WEIGHT_DELAY):
  return optim.Adam(parameters,  lr = learn_rate, betas=solver_beta, weight_decay=wd)


def radam(parameters,learn_rate=cfg.OPTIM.INIT_LR,  solver_beta=(cfg.OPTIM.BETA1,cfg.OPTIM.BETA2), 
                     wd=cfg.OPTIM.WEIGHT_DELAY):
  return RAdam(params= parameters,  lr = learn_rate, betas=solver_beta,  weight_decay=wd)


def ranger(parameters,learn_rate=cfg.OPTIM.INIT_LR,  solver_beta=(cfg.OPTIM.BETA1,cfg.OPTIM.BETA2), 
                     wd=cfg.OPTIM.WEIGHT_DELAY):
  return Ranger(params = parameters, lr = learn_rate,betas=solver_beta, weight_decay=wd)
    
def over9000(parameters,learn_rate=cfg.OPTIM.INIT_LR,  solver_beta=(cfg.OPTIM.BETA1,cfg.OPTIM.BETA2), 
                     wd=cfg.OPTIM.WEIGHT_DELAY):
  return Over9000(params = parameters,  lr = learn_rate, betas=solver_beta, weight_decay=wd)


def ralamb(parameters,learn_rate=cfg.OPTIM.INIT_LR, solver_beta=(cfg.OPTIM.BETA1,cfg.OPTIM.BETA2), 
                     wd=cfg.OPTIM.WEIGHT_DELAY):
  return Ralamb(params = parameters, lr = learn_rate, betas=solver_beta, weight_decay=wd)


def sgd(parameters,learn_rate=cfg.OPTIM.INIT_LR,  solver_momentum=cfg.OPTIM.MOMENTUM, 
                     wd=cfg.OPTIM.WEIGHT_DELAY):
  return optim.SGD(parameters, lr = learn_rate, momentum=solver_momentum, weight_decay=wd)

# def sam(parameters,learn_rate=cfg.OPTIM.INIT_LR,  solver_momentum=cfg.OPTIM.MOMENTUM, 
#                      wd=cfg.OPTIM.WEIGHT_DELAY):
#   base_optimizer = AdamP
#   return SAM(parameters, base_optimizer, lr=learn_rate, weight_decay=wd)



Optimizers = {
  'adam': adam,
  'radam':radam,
  'ranger':ranger,
  'over9000':over9000,
  'ralamb':ralamb,
  'sgd': sgd,
  # 'sam':sam,
}

def get_optimizer(model,optim_name =cfg.OPTIM.NAME, **kwargs):
    optim_name = optim_name.lower()
    if optim_name in Optimizers:
      return Optimizers[optim_name](model.parameters(), **kwargs)  
    else:
        print("Optimizer: {} is not implemented..., using default SGD".format(optim_name))
        return Optimizers['sgd'](model.parameters(), **kwargs)
