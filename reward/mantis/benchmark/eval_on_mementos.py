import fire
import os
import datasets
import tempfile

from mantis.mllm_tools import MLLM_Models
from mementos_eval_utils import generate_keywords_list, eval_mementos_subset
from tqdm import tqdm


def save_eval_results(dataset, results_file):
    def map_clear_image(x):
        x['images'] = [str(img) for img in x['images']]
        return x
    dataset = dataset.map(map_clear_image)
    dataset.to_json(results_file)

def main(
    model_name:str,
    results_dir: str = "results",
    check_items: bool = False,
    max_input_length: int = None,
    max_new_tokens: int = 2048,
    overwrite: bool = False,
    max_images: int = 13,
    judge="gpt-4-1106-preview",
):
    model = MLLM_Models(model_name)()

    eval_dataset = datasets.load_dataset("Mantis-VL/MIQA_eval", "mementos")
    
    for split in eval_dataset:
        if not split.startswith("single"):
            continue
        print(f"Split: {split}")
        print(f"Num examples: {len(eval_dataset[split])}")
        sub_dataset = eval_dataset[split]

        generation_dir = os.path.join(results_dir, "mememtos", split, model_name, "generations")
        os.makedirs(generation_dir, exist_ok=True)
        for example in tqdm(sub_dataset, desc=f"Generating Mementos {split}"):
            text = example['question']
            images = example['images']
            if len(images) > max_images:
                images = images[:max_images]
            
            if os.path.exists(os.path.join(generation_dir, f"{example['id']}.txt")) and not overwrite:
                generated_text = open(os.path.join(generation_dir, f"{example['id']}.txt")).read()
            else:
                # messages = [
                #     {
                #         "type": "image",
                #         "content": img
                #     }
                #     for img in images
                # ]
                # messages.append({
                #     "type": "text",
                #     "content": text,
                # })
                
                messages = [{
                    "type": "text",
                    "content": text,
                }]
                messages.extend([
                    {
                        "type": "image",
                        "content": img
                    }
                    for img in images
                ])
                
                generated_text = model(messages)
                # save
                with open(os.path.join(generation_dir, f"{example['id']}.txt"), "w") as f:
                    f.write(generated_text)
            
            if not "model_answers" in example:
                example["model_answers"] = {}
            example['model_answers'].update({model_name: generated_text})
            
            if check_items:
                # write images to some tempetory files under a tempetory directory under /tmp, to print them
                with tempfile.TemporaryDirectory(prefix="tmp_", suffix="_eval_images", dir=".") as temp_dir:
                    for i, image in enumerate(images):
                        image.save(os.path.join(temp_dir, f"image_{i}.png"))
                    print("----------------------------------------------------")
                    print(f"Subset: {split}")
                    for i, image in enumerate(images):
                        print(f"Image {i}: {os.path.abspath(os.path.join(temp_dir, f'image_{i}.png'))}")
                    print(f"Text: {text}")
                    print(f"Generated: {generated_text}")
                    print(f"Reference: {example['answer']}")
                    print("----------------------------------------------------")
                    input("Press Enter to continue...")
        
        # generation keywords
        judge_model_name=judge # debug only, replace with "gpt-4-1106-preview" in offical results
        if "gpt-4" not in judge_model_name:
            print(f"\033[91m" + "Warning: {judge_model_name} is temporarily used for debugging evaluation. Please replace it with gpt-4-1106-preview in offical results." + "\033[0m")
        generate_keywords_list(split.split("_")[-1], generation_dir, model_name=judge_model_name)
        
        # print eval results
        eval_mementos_subset(split.split("_")[-1], generation_dir)
    
if __name__ == '__main__':
    fire.Fire(main)
    
    
