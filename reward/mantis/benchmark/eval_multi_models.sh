bash eval_single_model.sh "random"
bash eval_single_model.sh "blip2"
bash eval_single_model.sh "instructblip"
bash eval_single_model.sh "llava"
bash eval_single_model.sh "fuyu"
bash eval_single_model.sh "kosmos2"
bash eval_single_model.sh "qwenVL"
bash eval_single_model.sh "emu2"
bash eval_single_model.sh "idefics1"
bash eval_single_model.sh "idefics2"
bash eval_single_model.sh "openflamingo-3b"
bash eval_single_model.sh "openflamingo-9b"
bash eval_single_model.sh "cogvlm"
bash eval_single_model.sh "gpt4v" # we only eval mantis eval for gpt-4v
bash eval_single_model.sh "emu2"
bash eval_single_model.sh "otter_image"
bash eval_single_model.sh "otter_video"
bash eval_single_model.sh "vila"
bash eval_single_model.sh "videollava"
bash eval_single_model.sh "videollava-video"


# mantis
bash eval_single_model.sh "mantis-8b-clip-llama3"
bash eval_single_model.sh "mantis-8b-siglip-llama3"
bash eval_single_model.sh "mantis-8b-fuyu"
bash eval_single_model.sh "mantis-7b-llava"
bash eval_single_model.sh "mantis-7b-bakllava"

# ablations
bash eval_single_model.sh "mantis-8b-idefics2_8192_qlora"
bash eval_single_model.sh "mantis-8b-idefics2_8192"
bash eval_single_model.sh "mantis-8b-idefics2-data-ablation-1_8192_qlora"
bash eval_single_model.sh "mantis-8b-idefics2-data-ablation-2_8192_qlora"
bash eval_single_model.sh "mantis-8b-idefics2-data-ablation-3_8192_qlora"
bash eval_single_model.sh "llava-9b-openflamingo"
bash eval_single_model.sh "mantis-9b-openflamingo"

export CUDA_VISIBLE_DEVICES=4 && bash eval_single_model.sh "mantis-8b-idefics2_8192" &
export CUDA_VISIBLE_DEVICES=5 && bash eval_single_model.sh "mantis-8b-idefics2-data-ablation-3_8192_qlora" &
export CUDA_VISIBLE_DEVICES=6 && bash eval_single_model.sh "mantis-8b-siglip-llama3-ablation1" &
export CUDA_VISIBLE_DEVICES=7 && bash eval_single_model.sh "mantis-8b-siglip-llama3-ablation2" &
export CUDA_VISIBLE_DEVICES=0 && bash eval_single_model.sh "mantis-8b-siglip-llama3-ablation3" &
export CUDA_VISIBLE_DEVICES=1 && bash eval_single_model.sh "mantis-8b-siglip-llama3-ablation4" &
export CUDA_VISIBLE_DEVICES=2 && bash eval_single_model.sh "mantis-8b-siglip-llama3-obelics-min" &



export CUDA_VISIBLE_DEVICES=0 && bash eval_single_model.sh "mantis-8b-idefics3_16384" &
export CUDA_VISIBLE_DEVICES=1 && bash eval_single_model.sh "mantis-8b-idefics3_pure_16384" &
export CUDA_VISIBLE_DEVICES=1 && bash eval_single_model.sh "idefics3" &




# export CUDA_VISIBLE_DEVICES=1 && bash eval_single_model.sh "mantis-8b-idefics2_8192_qlora" &
# export CUDA_VISIBLE_DEVICES=4 && bash eval_single_model.sh "mantis-8b-idefics2-data-ablation-1_8192_qlora" &
# export CUDA_VISIBLE_DEVICES=5 && bash eval_single_model.sh "mantis-8b-idefics2-data-ablation-2_8192_qlora" &
# export CUDA_VISIBLE_DEVICES=6 && bash eval_single_model.sh "llava-9b-openflamingo" &
# export CUDA_VISIBLE_DEVICES=7 && bash eval_single_model.sh "mantis-9b-openflamingo" &