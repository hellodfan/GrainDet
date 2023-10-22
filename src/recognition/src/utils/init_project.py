# this code heavily based on detectron2
'''
  init everything before train or test model
    - set seed
    - make dir

'''
import logging
import random
import os
import json

import torch
import numpy as np

from datetime import datetime

from .distributed import get_rank, synchronize
from .logger import setup_logger
from config import cfg



__all__ = ["init_everything"]


def init_everything():
    # init logging file
    
    save_dir = cfg.LOG_PATH
    setup_logger("CLS", save_dir, get_rank(), 
        filename='{}_{}_{}_{}_{}_log.txt'.format( cfg.AUTHOR,cfg.DATASET.NAME, cfg.TIME_STAMP,cfg.MODEL.NAME, cfg.PHASE))
    
    seed_all_rng(seed = cfg.SEED)
    mkdir()

    logging.info(json.dumps(cfg, indent=8))





def seed_all_rng(seed=None):
    """
    Set the random seed for the RNG in torch, numpy and python.

    Args:
        seed (int): if None, will use a strong random seed.
    """
    if seed is None:
        seed = (
            os.getpid()
            + int(datetime.now().strftime("%S%f"))
            + int.from_bytes(os.urandom(2), "big")
        )
        logger = logging.getLogger(__name__)
        logger.info("Using a generated random seed {}".format(seed))
    np.random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# make dir for use
def mkdir():
    if not os.path.exists(cfg.CHECKPOINTS):
        os.makedirs(cfg.CHECKPOINTS)
    if not os.path.exists(cfg.LOG_PATH):
        os.makedirs(cfg.LOG_PATH)
    if not os.path.exists(cfg.SAVE_ONNX_PATH):
        os.makedirs(cfg.SAVE_ONNX_PATH)
    if not os.path.exists(cfg.EVAL_RESULT):
        os.makedirs(cfg.EVAL_RESULT)
    