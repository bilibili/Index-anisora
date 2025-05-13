mkdir -p logs
model_name=$1
python eval.py --dataset_path "TIGER-Lab/Mantis-Eval" --dataset_name "mantis_eval" --model_name $model_name --results_dir "results" --overwrite False > ./logs/eval_${model_name}_mantis_eval.log 2>&1
python eval.py --dataset_path "TIGER-Lab/NLVR2" --dataset_name "nlvr2" --model_name $model_name --results_dir "results" --overwrite False > ./logs/eval_${model_name}_nlvr2.log 2>&1

dataset_path="../../data/qbench2/data/q-bench2-a1-pair-dev.json"
subset="q-bench2-a1-pair-dev"
python eval.py --dataset_path $dataset_path --dataset_name $subset --model_name $model_name --results_dir "results" --overwrite True > ./logs/eval_${model_name}_${subset}.log 2>&1

# MV bench
resolution=224
num_frames=8
log_dir="results/mvbench/${num_frames}frames_${resolution}"
mkdir -p $log_dir
python eval_on_mvbench.py --num_frames $num_frames --resolution $resolution --model_name ${model_name} > $log_dir/mvbench_${model_name}.txt

# BLINK
cd ../../BLINK_Benchmark/eval
mkdir -p logs
python test_benchmark.py --model_name $model_name --task_name all > ./logs/eval_${model_name}.log 2>&1
