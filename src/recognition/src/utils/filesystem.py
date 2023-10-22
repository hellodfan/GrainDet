import os
import errno
import torch
import logging

from config import cfg


__all__ = ['save_checkpoint','get_last_checkpoint']


def get_last_checkpoint(filepath = None):
    path_checkpoint = ''
    return path_checkpoint


def save_checkpoint(model, epoch, optimizer=None, lr_scheduler=None, is_best=False, best_acc = 0):
    """Save Checkpoint"""
    directory = os.path.expanduser(cfg.CHECKPOINTS)
    if not cfg.KD.USE_KD:
        directory = os.path.join(directory, '{}_{}_{}_{}'.format(
           cfg.AUTHOR,cfg.DATASET.NAME, cfg.TIME_STAMP,cfg.MODEL.NAME))
    else:
        directory = os.path.join(directory, '{}_{}_{}_{}'.format(
           cfg.AUTHOR,cfg.DATASET.NAME, cfg.TIME_STAMP,cfg.KD.STUDENT))

    if not os.path.exists(directory):
        os.makedirs(directory)
   
    filename = '{}.pth'.format(str(epoch))
    filename = os.path.join(directory, filename)
    model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    
    
    if is_best:
        # fdd save way
        best_file_name = ('best_{}_{:.5f}.pth'.format(epoch, best_acc))
        best_filename = os.path.join(directory, best_file_name)
        torch.save(model, best_filename)

    else:
        save_state = {
            'epoch': epoch,
            'state_dict': model_state_dict,
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict()
        }
        if not os.path.exists(filename):
            torch.save(model, filename)
            logging.info('Epoch {} model saved in: {}'.format(epoch, filename))

        # remove last epoch
        '''
        pre_filename = '{}.pth'.format(str(epoch - 1))
        pre_filename = os.path.join(directory, pre_filename)
        try:
            if os.path.exists(pre_filename):
                #os.remove(pre_filename)
                pass
        except OSError as e:
            logging.info(e)
        '''



