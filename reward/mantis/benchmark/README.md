## Introduction
We support evaluating on the following benchmarks:
- Mantis-Eval (test)
- NLVR2 (test-public)
- Q-Bench2-A1-pair-dev
- BLINK (val)
- MVBench (test)
- Mementos (test)

## Preparation

```bash
# BLINK
git clone https://github.com/jdf-prog/BLINK_Benchmark.git # forked from https://github.com/zeyofu/BLINK_Benchmark, adding support for using multiple LMMs defined in mantis/mllm_tools
# Mementos
git clone https://github.com/umd-huang-lab/Mementos.git
# Qbench2   
cd ../../data/qbench2 && bash prepare.sh
# MVBench
cd ../../data/mvbench && bash prepare.sh # need git-lfs
```

## Evaluation


- Evaluation on Mantis-Eval, NLVR2, Q-Bench2-A1-pair-dev
```bash
bash eval_multi_model.sh # please comment out models you don't want to evaluate in eval_multi_model.sh
```
- Evaluation on BLINK
```bash
cd BLINK_Benchmark/eval && bash eval.sh # please comment out models you don't want to evaluate in eval.sh
```

- Evaluation on MVBench
```bash
bash eval_mvbench.sh # please comment out models you don't want to evaluate in eval_mvbench.sh
```

- Evaluation on Mementos
```bash
bash eval_on_mementos.sh # please comment out models you don't want to evaluate in eval_on_mementos.sh
```