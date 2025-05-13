import os
import fire
import json
import numpy as np
import prettytable as pt
from pathlib import Path
from collections import defaultdict
            
datasets = ['nlvr2', 'birds-to-words', 'mantis_eval', 'q-bench2-a1-pair-dev']

models = ["random", "blip2", "instructblip", "llava", "llavanext", "openflamingo", "fuyu", "kosmos2", "qwenVL", "cogvlm", "idefics2", "idefics1", "emu2", "llavanext", "gpt4v",
"mantis-8b-fuyu", "mantis-7b-llava", "mantis-7b-bakllava", "mantis-8b-clip-llama3", "mantis-8b-siglip-llama3"]

model_name_map = {}

def main(
    num_shots: int=0,
):
    for dataset_name in datasets:
        all_model_results = defaultdict(dict)
        for model_name in models:
            results_dir = Path(os.path.dirname(__file__)) / 'results'
            results_path = results_dir / dataset_name / f'{model_name}_{num_shots}_shots.json'
            results_path.parent.mkdir(parents=True, exist_ok=True)
            if results_path.exists():
                with open(results_path, 'r') as f:
                    existing_results = json.load(f)
                all_model_results[model_name]['multi-choice'] = np.mean([q["prediction"]["correct"] for q in existing_results if q['question_type'] == 'multi-choice'])
                all_model_results[model_name]['short-answer'] = np.mean([q["prediction"]["correct"] for q in existing_results if q['question_type'] == 'short-answer'])
                all_model_results[model_name]['overall'] = np.mean([q["prediction"]["correct"] for q in existing_results])
            else:
                all_model_results[model_name] = {'multi-choice': -1, 'short-answer': -1, 'overall': -1}

        # sort by overall accuracy
        all_model_results = dict(sorted(all_model_results.items(), key=lambda item: item[1]['overall'], reverse=True))
        
        table = pt.PrettyTable()
        # first column is left aligned
        table.align = "l"
        table.field_names = ["Model", "Overall Accuracy", "Multi-choice Accuracy", "Short-answer Accuracy"]
        table.title = f"Results on dataset: {dataset_name}"
        for model_name, results in all_model_results.items():
            if model_name in model_name_map:
                model_name = model_name_map[model_name]
            if results is not None:
                # round to 4 decimal places
                overall = round(results['overall'], 4) if results['overall'] != -1 else "N/A"
                multi_choice = round(results['multi-choice'], 4) if results['multi-choice'] != -1 else "N/A"
                short_answer = round(results['short-answer'], 4) if results['short-answer'] != -1 else "N/A"
                table.add_row([model_name, overall, multi_choice, short_answer])
            else:
                table.add_row([model_name, "N/A", "N/A", "N/A"])
        print(table)
        print("\n\n")
        
if __name__ == '__main__':
    fire.Fire(main)