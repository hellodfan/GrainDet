import torch
import struct
import sys
sys.path.append('/home/code/uwseg_normal_maskcls/src/')

# Initialize
device = torch.device('cuda:0')

path = '/home/code/uwseg_normal_maskcls/runs/checkpoints/dyw_6channel_192x192_dp_fft_test_0115_2022-01-14-18-05_resnet50/best_47_99.86622.pth'
xx = torch.load(path,map_location='cpu')
model = torch.load(path).module.float()  # load to FP32
print('load from:',path)
model.to(device).eval()

f = open('resnet50_wdl_0118_base_6c_dyw_998662.weights', 'w')
f.write('{}\n'.format(len(model.state_dict().keys())))
for k, v in model.state_dict().items():
    vr = v.reshape(-1).cpu().numpy()
    f.write('{} {} '.format(k, len(vr)))
    for vv in vr:
        f.write(' ')
        f.write(struct.pack('>f',float(vv)).hex())
    f.write('\n')
print('finish')
