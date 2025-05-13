resolution=224
num_frames=8
log_dir="results/mvbench/${num_frames}frames_${resolution}"
mkdir -p $log_dir

# python eval_on_mvbench.py --num_frames $num_frames --resolution $resolution --model_name "blip2" > $log_dir/mvbench_blip2.txt
# python eval_on_mvbench.py --num_frames $num_frames --resolution $resolution --model_name "instructblip" > $log_dir/mvbench_instructblip.txt
# python eval_on_mvbench.py --num_frames $num_frames --resolution $resolution --model_name "llava" > $log_dir/mvbench_llava.txt
# python eval_on_mvbench.py --num_frames $num_frames --resolution $resolution --model_name "emu2" > $log_dir/mvbench_emu2.txt
# python eval_on_mvbench.py --num_frames $num_frames --resolution $resolution --model_name "qwenVL" > $log_dir/mvbench_qwenVL.txt
# python eval_on_mvbench.py --num_frames $num_frames --resolution $resolution --model_name "kosmos2" > $log_dir/mvbench_kosmos2.txt
# python eval_on_mvbench.py --num_frames $num_frames --resolution $resolution --model_name "fuyu" > $log_dir/mvbench_fuyu.txt
# python eval_on_mvbench.py --num_frames $num_frames --resolution $resolution --model_name "cogvlm" > $log_dir/mvbench_cogvlm.txt
# python eval_on_mvbench.py --num_frames $num_frames --resolution $resolution --model_name "idefics1" > $log_dir/mvbench_idefics1.txt
# python eval_on_mvbench.py --num_frames $num_frames --resolution $resolution --model_name "idefics2" > $log_dir/mvbench_idefics2.txt
# python eval_on_mvbench.py --num_frames $num_frames --resolution $resolution --model_name "openflamingo" > $log_dir/mvbench_openflamingo.txt
# python eval_on_mvbench.py --num_frames $num_frames --resolution $resolution --model_name "llavanext" > $log_dir/mvbench_llavanext.txt
CUDA_VISIBLE_DEVICES=0 python eval_on_mvbench.py --num_frames $num_frames --resolution $resolution --model_name "otterimage" > $log_dir/mvbench_otter_image.txt &
CUDA_VISIBLE_DEVICES=1 python eval_on_mvbench.py --num_frames $num_frames --resolution $resolution --model_name "ottervideo" > $log_dir/mvbench_otter_video.txt &
CUDA_VISIBLE_DEVICES=5 python eval_on_mvbench.py --num_frames $num_frames --resolution $resolution --model_name "vila" > $log_dir/mvbench_vila.txt &
CUDA_VISIBLE_DEVICES=6 python eval_on_mvbench.py --num_frames $num_frames --resolution $resolution --model_name "videollava-video" > $log_dir/mvbench_videollava_video.txt &

# CUDA_VISIBLE_DEVICES=0 python eval_on_mvbench.py --num_frames $num_frames --resolution $resolution --model_name "mantis-8b-fuyu" > $log_dir/mvbench_mantis_8b_fuyu.txt &
# CUDA_VISIBLE_DEVICES=1 python eval_on_mvbench.py --num_frames $num_frames --resolution $resolution --model_name "mantis-7b-llava" > $log_dir/mvbench_mantis_7b_llava.txt &
# CUDA_VISIBLE_DEVICES=2 python eval_on_mvbench.py --num_frames $num_frames --resolution $resolution --model_name "mantis-7b-bakllava" > $log_dir/mvbench_mantis_7b_bakllava.txt &
# CUDA_VISIBLE_DEVICES=3 python eval_on_mvbench.py --num_frames $num_frames --resolution $resolution --model_name "mantis-8b-clip-llama3" > $log_dir/mvbench_mantis_8b_clip_llama3.txt &
# CUDA_VISIBLE_DEVICES=4 python eval_on_mvbench.py --num_frames $num_frames --resolution $resolution --model_name "mantis-8b-siglip-llama3" > $log_dir/mvbench_mantis_8b_siglip_llama3.txt &



resolution=224
num_frames=2
log_dir="results/mvbench/${num_frames}frames_${resolution}"
mkdir -p $log_dir
CUDA_VISIBLE_DEVICES=0 python eval_on_mvbench.py --num_frames $num_frames --resolution $resolution --model_name "mantis-8b-idefics2_8192" > $log_dir/mvbench_mantis_8b_idefics2_8192.txt 2>&1 & 
resolution=224
num_frames=4
log_dir="results/mvbench/${num_frames}frames_${resolution}"
mkdir -p $log_dir
CUDA_VISIBLE_DEVICES=1 python eval_on_mvbench.py --num_frames $num_frames --resolution $resolution --model_name "mantis-8b-idefics2_8192" > $log_dir/mvbench_mantis_8b_idefics2_8192.txt 2>&1 & 
resolution=224
num_frames=8
log_dir="results/mvbench/${num_frames}frames_${resolution}"
mkdir -p $log_dir
CUDA_VISIBLE_DEVICES=2 python eval_on_mvbench.py --num_frames $num_frames --resolution $resolution --model_name "mantis-8b-idefics2_8192" > $log_dir/mvbench_mantis_8b_idefics2_8192.txt 2>&1 & 
resolution=224
num_frames=16
log_dir="results/mvbench/${num_frames}frames_${resolution}"
mkdir -p $log_dir
CUDA_VISIBLE_DEVICES=3 python eval_on_mvbench.py --num_frames $num_frames --resolution $resolution --model_name "mantis-8b-idefics2_8192" > $log_dir/mvbench_mantis_8b_idefics2_8192.txt 2>&1 & 
