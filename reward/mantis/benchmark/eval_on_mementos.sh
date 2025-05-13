CUDA_VISIBLE_DEVICES=0 python eval_on_mementos.py --model_name  "mantis-8b-fuyu" --results_dir results --overwrite False > results/mementos_mantis_8b_fuyu.txt &
CUDA_VISIBLE_DEVICES=1 python eval_on_mementos.py --model_name  "mantis-7b-llava" --results_dir results --overwrite False > results/mementos_mantis_7b_llava.txt &
CUDA_VISIBLE_DEVICES=2 python eval_on_mementos.py --model_name  "mantis-7b-bakllava" --results_dir results --overwrite False > results/mementos_mantis_7b_bakllava.txt &
CUDA_VISIBLE_DEVICES=3 python eval_on_mementos.py --model_name  "mantis-8b-clip-llama3" --results_dir results --overwrite False > results/mementos_mantis_8b_clip_llama3.txt &
CUDA_VISIBLE_DEVICES=4 python eval_on_mementos.py --model_name  "mantis-8b-siglip-llama3" --results_dir results --overwrite False > results/mementos_mantis_8b_siglip_llama3.txt &

