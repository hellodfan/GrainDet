
import time
from .config import BaseConfig

cfg = BaseConfig()

################## basic config ######################

cfg.PHASE = 'train'  #train, test
cfg.COMMENT = 'this this first time to use logging system...'
cfg.AUTHOR = 'Gauture'

cfg.NAME = 'baseconfig'  # config name
cfg.SEED = 509           # random seed
cfg.TIME_STAMP = time.strftime('%Y-%m-%d-%H-%M', time.localtime())  # what's time to create project


cfg.CHECKPOINTS = "./runs/checkpoints/"        # path to save checkpoints
cfg.LOG_PATH = "./runs/logs/"                  # path to save log files
cfg.SAVE_ONNX_PATH = "./runs/onnx/"    #path to save onnx file
cfg.EVAL_RESULT = "./runs/eval_result/"    #path to save onnx file


################## dataset config ######################
cfg.DATASET.NAME = 'wheat6'   # dataset name 
cfg.DATASET.PATH = '/home/fdd/dataset/black_normal/black_black/' # path to dataset
cfg.DATASET.CLASS_NUMS = 8  # 
cfg.DATASET.SPILT_ONLINE = False  # split dataset to train and val online or offline
cfg.DATASET.SPILT_RATE = 0.1
cfg.DATASET.WORKERS = 16   # set number of data loading workers   

cfg.DATASET.MEANS = [0.435, 0.517, 0.580]
cfg.DATASET.STD =   [1.0, 1.0, 1.0]


cfg.AUGMENT.MIXUP_RATIO = 0.2
cfg.AUGMENT.RandBright_limit = 0.15 
cfg.AUGMENT.RandBright_ratio = 0.3 
cfg.AUGMENT.RandContra_limit = 0.15
cfg.AUGMENT.RandContra_ratio = 0.3


################## train config ######################


cfg.RESUME.FLAG = False
cfg.RESUME.PATH = '/home/code/uwseg_normal_maskcls/runs/checkpoints/dyw_channel6_2020-12-09-09-38_laodanet6/best_25_99.32853.pth'

cfg.MODEL.NAME = 'laodanet3128'
cfg.MODEL.RESUME_PATH = ''


cfg.KD.USE_KD = False
cfg.KD.TEACHER = 'resnet34'
cfg.KD.STUDENT = 'resnet18'
cfg.KD.TEACHER_WEIGHT_PATH = ''
cfg.KD.STUDENT_WEIGHT_PATH = ''


cfg.OPTIM.NAME = 'radam'   # 'sgd', 'adam', 
cfg.OPTIM.BETA1 = 0.9           # adam parameters beta1
cfg.OPTIM.BETA2 = 0.999         # adam parameters beta2 
cfg.OPTIM.MOMENTUM = 0.9   # sgd parameter
cfg.OPTIM.WEIGHT_DELAY = 1e-4

cfg.OPTIM.INIT_LR = 2e-5
# 6e-6
cfg.OPTIM.LR_SCHEDULER = 'step'  # 
cfg.OPTIM.LOSS = 'CrossEntropy'  # 'CrossEntropy', 'Focal', 'LabelSmoothCE'

#focal loss
cfg.OPTIM.FOCAL_GAMMA = 2.
cfg.OPTIM.LABEL_SMOOTH = 0.1


cfg.TRAIN.BATCH = 64
cfg.TRAIN.EPOCHS = 60
cfg.TRAIN.START_EPOCH = 0
cfg.TRAIN.SIZE = (128,128) #(H,W)

cfg.TRAIN.INTERP = 2 
#Image.NEAREST (0), 
#Image.LANCZOS (1), 
#Image.BILINEAR (2), 
#Image.BICUBIC (3), 
# Image.BOX (4) or Image.HAMMING (5)




################## test config ######################

cfg.TEST.SIZE = (128,128)  #(H,W)
cfg.TEST.BATCH = 64

#Image.NEAREST (0), 
#Image.LANCZOS (1), 
#Image.BILINEAR (2), 
#Image.BICUBIC (3), 
# Image.BOX (4) or Image.HAMMING (5)
cfg.TEST.INTERP = 2  





cfg.EVAL.MODEL_ZOO = '/home/code/uwseg_normal_maskcls/runs/checkpoints/dyw_channel6_2020-12-24-10-56_resnet50'
cfg.EVAL.BASELINE_DATA = '/home/data/jml'
cfg.EVAL.TEST_IMGS_PATH = None









