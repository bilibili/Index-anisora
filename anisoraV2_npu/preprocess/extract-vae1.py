'''
nohup python3 preprocess/extract-vae1.py 0 2 >> vae1.log 2>&1 &
nohup python3 preprocess/extract-vae1.py 1 2 >> vae1.log 2>&1 &

0/1: What part does this process do
2:   It consists of two parts in total.
'''
root_mp4s="test_data/mp4root"
h,w=480,832
num_frames = 49
opt_root="output_root/vae1"
checkpoint_dir="/DATA/bvac/personal/wan21/Wan2.1-I2V-14B-480P"




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
    temp_frms = vr.get_batch([2])
    return (TF.to_tensor(temp_frms.asnumpy().astype("float32")[0])/255).sub_(0.5).div_(0.5).to(device)

os.makedirs(opt_root,exist_ok=True)
def go(todos):
    for path in todos:
        try:
            name=os.path.basename(path).replace(".mp4",".pt")
            if os.path.exists("%s/%s"%(opt_root,name)):continue
            img = read_img(path)
            tensorr = vae.encode([
                torch.concat([
                    torch.nn.functional.interpolate(
                        img[None].cpu(), size=(h, w), mode='bicubic').transpose(
                        0, 1),
                    torch.zeros(3, num_frames-1, h, w)
                ], dim=1).to(device)
            ])[0].cpu()  # torch.Size([16, 21, 90, 160])#21->13
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



