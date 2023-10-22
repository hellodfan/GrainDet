import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='recognition')
    
    parser.add_argument('--config-file',type=str,default="configs/base.yaml",
                        help='config file path')

    # checkpoint and log
    parser.add_argument('--resumepath', type=str, default='best_model.pth',
                        help='put the path to resuming file if needed')
    
    parser.add_argument('--log-iter', type=int, default=10,
                        help='print log every log-iter')
    
    # cuda setting
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--local_rank', type=int, default=0)

    # for evaluation
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='run validation every val-epoch')
                        
    parser.add_argument('--skip-val', action='store_true', default=False,
                        help='skip validation during training')
    # for visual
    parser.add_argument('--testpath', type=str, default='/home/fdd/dataset/all_cls6_0927',
                        help='path to the input image or a directory of images')

    parser.add_argument('--output-img', type=str, default=None,
                        help='path to the save images')

    parser.add_argument('--val-modelzoo', type=str, default=None,
                        help='path to the model zoo')
    
    parser.add_argument('--model_name', type=str, default='tasn',
                    help='model name')
    
    parser.add_argument('--save_name', type=str, default='tasn_wheat',
                    help='save data dir name')
    
    parser.add_argument('--data_path', type=str, default='/home/GrainSet/wheat',
                    help='read data path')
    
    parser.add_argument('--phase', type=str, default='train',
                    help='train / train_bal ')
    
    # config options
    parser.add_argument('opts', help='See config for all options',
                        default=None, nargs=argparse.REMAINDER)
    
    args = parser.parse_args()

    return args