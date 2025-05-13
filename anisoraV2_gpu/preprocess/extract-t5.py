'''
nohup python preprocess/extract-t5.py 0 2 >> t5.log 2>&1 &
nohup python preprocess/extract-t5.py 1 2 >> t5.log 2>&1 &

0/1: What part does this process do
2:   It consists of two parts in total.
'''
txt_path="../data/train_data_prompts.txt"
opt_root="../output_root/t5"
checkpoint_dir="../Wan2.1-I2V-14B-480P"

import os,sys
all=int(sys.argv[2])
i_part=int(sys.argv[1])
# os.environ["CUDA_VISIBLE_DEVICES"]=str(int(sys.argv[1])%4)
os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[1]
import pdb,torch
from wan.modules.t5 import T5EncoderModel
device="cuda"
text_encoder = T5EncoderModel(
    text_len=512,
    dtype=torch.bfloat16,
    device=torch.device('cpu'),
    checkpoint_path=os.path.join(checkpoint_dir, 'models_t5_umt5-xxl-enc-bf16.pth'),
    tokenizer_path=os.path.join(checkpoint_dir, 'google/umt5-xxl'),
    shard_fn=None,
)
text_encoder.model=text_encoder.model.to(device)
os.makedirs(opt_root,exist_ok=True)
def go(todos):
    for name,text in todos:
        try:
            if os.path.exists("%s/%s"%(opt_root,name)):continue
            context = text_encoder([text], device)[0].cpu()#torch.Size([138, 4096])#"In this scene, a man with a beard is seen tending to a woman who lies in bed, her face illuminated by the soft glow of a nearby light source. The man, dressed in a blue robe adorned with intricate designs, holds a bowl, possibly containing a healing potion or a magical elixir. The woman, clad in a pink garment, appears to be resting or possibly unwell, as she lies on her side with her eyes closed. The setting suggests a historical or medieval context, with the dimly lit room and the man's attire evoking a sense of timelessness and mystery. "
            save_path="%s/%s"%(opt_root,name)
            torch.save(context,save_path)
        except:
            print(text,traceback.format_exc())


todo=[]
with open(txt_path,"r")as f:lines=f.read().strip("\n").split("\n")
for line in lines:
    todo.append(line.split("|"))
todo=sorted(todo)[i_part::all]
go(todo)
