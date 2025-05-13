import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
# os.environ["CUDA_VISIBLE_DEVICES"]="1,2"
# os.environ["CUDA_VISIBLE_DEVICES"]="3,4,5,6"
import torch
import time
import random
import pdb

from __init__ import CVModel

def main():
    # --CVModel(n_gpus=4): Indicates 4-GPU parallel inference. Adjust n_gpus as needed.
    model = CVModel(n_gpus=4)
    start = time.time()
    '''
    'In this video, a young girl with brown hair and blue eyes stands confidently in a room adorned with brown wallpaper. She is dressed in a green hoodie, exuding a relaxed demeanor. In her hands, she holds a microphone, while gently touching her arm, suggesting she might be preparing for a performance or singing. The room appears to be a studio or a space equipped with stage lights and speakers, as indicated by the presence of these elements. The overall atmosphere of the video hints at anticipation or readiness for a performance. aesthetic score: 5.5. motion score: 0.7.',
    In this video, a young man is seen in a dynamic pose, wearing a red hoodie with the word "SEASON" printed on it. He has a backpack slung over his shoulder and is looking to his right with a slight smile on his face. The background is a gradient of dark to light gray, suggesting an outdoor setting. The man's attire and the backpack imply he might be on a journey or traveling. The overall mood of the image is casual and relaxed. 
    '''
    resource = ['test_data/fumo_7.png','test_data/fumo_7_mid.png','test_data/fumo_7_last.png']
    seed=554
    #seed = random.randint(0,2048)
    raw_params =  {
        "Motion": 0.7,
        "gen_len": "3",
        "prompt": 'In this video, a young girl with brown hair and blue eyes stands confidently in a room adorned with brown wallpaper. She is dressed in a green hoodie, exuding a relaxed demeanor. In her hands, she holds a microphone, while gently touching her arm, suggesting she might be preparing for a performance or singing. The room appears to be a studio or a space equipped with stage lights and speakers, as indicated by the presence of these elements. The overall atmosphere of the video hints at anticipation or readiness for a performance. aesthetic score: 5.5. motion score: 0.7.',
        "seed": seed,
        "output_path": "results/test_1frame_motion07_ban_dong_xi.mp4"}
    # Notes:
    # --Motion: Controls motion amplitude (recommended: 0.7-1.3).
    # --gen_len: Generation length (e.g., 3s or 6s).
    
    res = model.run(resource,**raw_params)
    print(res)
    print('total time:', time.time() - start)

if __name__ == '__main__':
    main()
