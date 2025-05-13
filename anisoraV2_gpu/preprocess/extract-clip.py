'''
nohup python preprocess/extract-clip.py 0 2 >> clip.log 2>&1 &
nohup python preprocess/extract-clip.py 1 2 >> clip.log 2>&1 &

0/1: What part does this process do
2:   It consists of two parts in total.
'''
root_mp4s="../data/mp4root"
h,w=480,832
opt_root="../output_root/clip"
checkpoint_dir="../Wan2.1-I2V-14B-480P"


import os,sys,traceback
import pdb
all=int(sys.argv[2])
i_part=int(sys.argv[1])
# os.environ["CUDA_VISIBLE_DEVICES"]=str(int(sys.argv[1])%4)
os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[1]
import pdb,torch
from wan.modules.clip import CLIPModel
device="cuda"
clip = CLIPModel(
    dtype=torch.float16,
    device=device,
    checkpoint_path=os.path.join(checkpoint_dir,'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'),
    tokenizer_path=os.path.join(checkpoint_dir, 'xlm-roberta-large'))

clip.model = clip.model.to(device)
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
            clip_context = clip.visual([img[:, None, :, :]])
            save_path="%s/%s"%(opt_root,name)
            torch.save(clip_context, save_path)
        except:
            print(path,traceback.format_exc())

todo=[]
for name in os.listdir(root_mp4s):
    todo.append("%s/%s"%(root_mp4s,name))
todo=sorted(todo)
todo=todo[i_part::all]
go(todo)


