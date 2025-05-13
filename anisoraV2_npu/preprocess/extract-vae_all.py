'''
nohup python3 preprocess/extract-vae_all.py 0 2 >> t5.log 2>&1 &
nohup python3 preprocess/extract-vae_all.py 1 2 >> t5.log 2>&1 &

0/1: What part does this process do
2:   It consists of two parts in total.
'''
root_mp4s="test_data/mp4root"
h,w=480,832
num_frames = 49
opt_root="output_root/vae_all"
checkpoint_dir="/DATA/bvac/personal/wan21/Wan2.1-I2V-14B-720P"


import os,sys
os.environ["ASCEND_RT_VISIBLE_DEVICES"]=sys.argv[1]
all=int(sys.argv[2])
i_part=int(os.environ["ASCEND_RT_VISIBLE_DEVICES"])
import torch
if os.getenv("ACCELERATOR") == "npu":
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    torch.npu.config.allow_internal_format = False
    print("import torch_npu\n")
else:
    print("Warning: Missing torch_npu\n")
import traceback,numpy as np
import pdb
from fastvideo.bili_space.wan.modules.vae import WanVAE
device="npu"
vae = WanVAE(vae_pth=os.path.join(checkpoint_dir, 'Wan2.1_VAE.pth'),device=device)
from decord import VideoReader
import torchvision.transforms.functional as TF
def read_img(path):
    vr = VideoReader(uri=path, height=-1, width=-1)
    actual_fps=vr.get_avg_fps()
    start=2
    wanted_fps=16
    end = int(start + num_frames / wanted_fps * actual_fps)
    indices = np.arange(start, end, (end - start) / num_frames).astype(int)
    # print(100000,start,end,indices,len(indices),actual_fps,len(vr))
    temp_frms = vr.get_batch(indices)
    temp_frms = torch.from_numpy(temp_frms.asnumpy()).to(device).float()/255
    temp_frms-=0.5
    temp_frms/=0.5#torch.Size([49, 1080, 1920, 3])
    temp_frms=temp_frms.permute(3,0,1,2)#torch.Size([3, 49, 1080, 1920])
    return temp_frms


os.makedirs(opt_root,exist_ok=True)
def go(todos):
    for path in todos:
        try:
            name=os.path.basename(path).rsplit('.', maxsplit=1)[0]+'.pt'
            if os.path.exists("%s/%s"%(opt_root,name)):continue
            img = read_img(path)
            aa=torch.nn.functional.interpolate(img, size=(h, w), mode='bicubic')
            tensorr = vae.encode(aa.unsqueeze(0))[0].cpu()  # torch.Size([16, 21, 90, 160])#21->13
            save_path="%s/%s"%(opt_root,name)
            torch.save(tensorr, save_path)
        except:
            print(path,traceback.format_exc())

todo=[]
for name in os.listdir(root_mp4s):
    todo.append("%s/%s"%(root_mp4s,name))
todo=sorted(todo)
todo=todo[i_part::all]
go(todo)



