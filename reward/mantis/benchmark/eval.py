import json
import fire
import regex as re
import os
import datasets
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
import sys
from PIL import Image
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mllm_tools import MLLM_Models
from typing import List

def parse_answer(raw_answer):
    if "final answer:" in raw_answer.lower():
        answer = raw_answer[raw_answer.lower().index("final answer:") + len("final answer:"):].strip()
    elif "the answer is" in raw_answer.lower():
        answer = raw_answer[raw_answer.lower().index("the answer is") + len("the answer is"):].strip()
    elif "answer:" in raw_answer.lower():
        answer = raw_answer[raw_answer.lower().index("answer:") + len("answer:"):].strip()
    else:
        answer = raw_answer
    return answer

def get_option(final_answer):
    if re.match(r'Answer: [A-Z]', final_answer):
        return final_answer[8]
    for s in final_answer:
        if s.isalpha():
            return s.upper()
    return None

def get_prediction(question_type:str, raw_answer:str, ref_answer:str, options:List[str], dataset_name:str):
    answer = parse_answer(raw_answer)
    ref_answer = ref_answer.strip('()\n ') # important for some datasets
    if question_type == 'multi-choice':
        if not len(ref_answer) == 1:
            for c in ref_answer:
                if c.isalpha():
                    ref_answer = c
                    break
        assert len(ref_answer) == 1, f"Ref answer is not a single character: {ref_answer}"
        
        selected_option = get_option(answer)
        if selected_option and (ord(selected_option) - ord('A') < len(options)):
            correct = selected_option == ref_answer.upper()
            parsed_answer = selected_option
        else:
            ref_option_idx = ord(ref_answer.upper()) - ord('A')
            if ref_option_idx >= len(options):
                correct = False
                parsed_answer = raw_answer
            else:
                if dataset_name == "nlvr2":
                    ref_raw_answer = options[ref_option_idx]
                    
                    correct = ('true' in raw_answer.lower() and 'false' not in raw_answer.lower() and 'true' in ref_raw_answer.lower()) or ('false' in raw_answer.lower() and 'true' not in raw_answer.lower() and 'false' in ref_raw_answer.lower())
                else:
                    ref_raw_answer = options[ref_option_idx]
                    if ref_raw_answer.startswith(ref_answer + '.'):
                        correct = raw_answer.strip() == ref_raw_answer[len(ref_answer + '.'):].strip()
                    elif ref_raw_answer.startswith(ref_answer + ':'):
                        correct = raw_answer.strip() == ref_raw_answer[len(ref_answer + ':'):].strip()
                    elif ref_raw_answer.startswith('(' + ref_answer + ')'):
                        correct = raw_answer.strip() == ref_raw_answer[len(ref_answer) + 2:].strip()
                    else:
                        correct = raw_answer.strip() == ref_raw_answer.strip()
            parsed_answer = raw_answer
    elif question_type == 'short-answer':
        correct = ref_answer.lower() == answer.lower()
        parsed_answer = answer
    
    return {
        'raw_answer': raw_answer,
        'parsed_answer': parsed_answer,
        'correct': correct
    }

def main(
    model_name: str,
    dataset_path: str="TIGER-Lab/Mantis-eval",
    dataset_name: str="",
    results_dir: str="results",
    max_size=None,
    num_shots: int=0,
    overwrite=False,
    check_existing=False,
    sub_sample_size=None,
    seed=42
):
    random.seed(42)
    model_initizalized = False
    templates_path = Path(os.path.join(os.path.dirname(__file__), 'templates'))
    all_templates = {}
    for template_file in templates_path.glob('*.txt'):
        with open(template_file, 'r') as f:
            all_templates[template_file.stem] = f.read()
    
    results_dir = Path(os.path.dirname(__file__)) / results_dir
    results_path = results_dir / dataset_name / f'{model_name}_{num_shots}_shots.jsonl'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    if results_path.exists():
        if overwrite:
            os.remove(results_path)
            existing_results = []
        else:            
            with open(results_path, 'r') as f:
                existing_results = [json.loads(l) for l in f.readlines()]
    else:
        existing_results = []
    existing_result_id_map = {r['id']: r for r in existing_results}
    all_results = []
    
    if not check_existing:
        if not os.path.exists(dataset_path):
            
            dataset = datasets.load_dataset(dataset_path, split='test')
            if max_size:
                dataset = dataset.select(range(min(len(dataset), max_size)))
            if isinstance(sub_sample_size, int):
                dataset = dataset.shuffle(seed=seed).select(range(sub_sample_size))
                
        else:
            with open(dataset_path, 'r') as f:
                dataset = json.load(f)
            # load images
            for d in dataset:
                d['images'] = [Image.open(str(Path(dataset_path).parent / img_path)) for img_path in d['images']]
            
            if max_size:
                dataset = dataset[:max_size]
            if isinstance(sub_sample_size, int):
                dataset = random.sample(dataset, sub_sample_size)
        
        if model_name == "random":
            model = None
            model_initizalized = True
        else:
            model = MLLM_Models(model_name)
        result_f = open(results_path, 'a+')
        # evaluate
        for d in tqdm(dataset, desc=f'Evaluating {model_name} on {dataset_name}'):
            if existing_result_id_map.get(d['id']) is not None and not overwrite:
                existing_result_id_map[d['id']]['prediction'] = get_prediction(
                    d['question_type'], existing_result_id_map[d['id']]['prediction']['raw_answer'], d['answer'], d['options'], dataset_name)
                all_results.append(existing_result_id_map[d['id']])
                continue
            question_type = d['question_type']
            question = d['question']
            if model_name == "random":
                if question_type == 'multi-choice':
                    raw_answer = random.choice(d['options'])
                elif question_type == 'short-answer':
                    raw_answer = ""
                else:
                    raise ValueError(f"Unknown question type {question_type}")
                messages = None
            else:
                if not model_initizalized:
                    model_initizalized = True
                    model = model()
                
                if question_type == 'multi-choice':
                    option_idx = 'A'
                    for option in d['options']:
                        if not any([x in option.upper() for x in [f"{option_idx})", f"{option_idx}:", f"{option_idx}."]]):
                            question += f'\n ({option_idx}) {option}'
                        else:
                            question += f'\n {option}'
                        option_idx = chr(ord(option_idx) + 1)
                template = all_templates[question_type]
                question = template.format(question=question)
                images = d['images']

                if not model.support_multi_image:
                    question_split_by_image = question.split('<image>')
                    question = " ".join([x.strip(' \n') for x in question_split_by_image if x.strip() != ''])
                    messages = [
                        {
                            "type": "image",
                            "content": img
                        }
                        for img in images
                    ]
                    messages.append({
                        "type": "text",
                        "content": question,
                    })
                else:
                    if question.count('<image>') < len(images):
                        question = "<image>"*(len(images) - question.count('<image>')) + question
                        
                    question_split_by_image = question.split('<image>')
                    messages = []
                    for i in range(len(question_split_by_image)):
                        if question_split_by_image[i].strip('\n ') != '':
                            messages.append({
                                "type": "text",
                                "content": question_split_by_image[i],
                            })
                        if i < len(images):
                            messages.append({
                                "type": "image",
                                "content": images[i],
                            })
                # run
                raw_answer = model(messages)
                # for save
                for m in messages:
                    if m['type'] == 'image':
                        m['content'] = str(m['content'])
            
            
            d['prediction'] = get_prediction(question_type, raw_answer, d['answer'], d['options'], dataset_name)
            
            no_image_item = {
                'id': d['id'],
                'question': d['question'],
                'question_type': d['question_type'],
                'options': d['options'],
                'category': d['category'],
                'data_source': d['data_source'],
                'answer': d['answer'],
                'messages': messages,
                'prediction': d['prediction'],
            }
            result_f.write(json.dumps(no_image_item) + '\n')
            all_results.append(no_image_item)
        
        result_f.close()
    else:
        all_results = existing_results
    
    from collections import Counter
    print(Counter([r['prediction']['parsed_answer'] for r in all_results if r['question_type'] == 'multi-choice']))
    
    with open(results_path.with_suffix('.json'), 'w') as f:
        json.dump(all_results, f, indent=4)
        print(f"Results saved to {results_path.with_suffix('.json')}")
    print("Results on dataset:", dataset_name)
    # multi-choice
    multi_choice_questions = [q for q in all_results if q['question_type'] == 'multi-choice']
    if len(multi_choice_questions) > 0:
        print(f'Multi-choice Accuracy: {np.mean([q["prediction"]["correct"] for q in multi_choice_questions]):.4f}')
    # open-ended
    open_ended_questions = [q for q in all_results if q['question_type'] == 'short-answer']
    if len(open_ended_questions) > 0:
        print(f'Short-answer Accuracy: {np.mean([q["prediction"]["correct"] for q in open_ended_questions]):.4f}')
    
    if len(all_results) > 0:
        print(f"Overall Accuracy: {np.mean([q['prediction']['correct'] for q in all_results]):.4f}")
            

if __name__ == '__main__':
    fire.Fire(main)