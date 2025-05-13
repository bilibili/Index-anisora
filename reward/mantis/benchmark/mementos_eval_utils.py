import os
import pickle
import pandas as pd
import numpy as np
from mantis.easy_openai import openai_completions
from pathlib import Path

class ActionGraph:
    def __init__(self):
        self.graph = {}  # Initialize an empty dictionary to hold the adjacency list.

    def add_path(self, words, synonym):
        # Add a path of words that leads to a list of synonyms.
        if len(words) == 1:
            # If there's only one word, associate it directly with the synonym.
            word = words[0]
            if word not in self.graph:
                self.graph[word] = {'': [synonym]}
            else:
                self.graph[word][''] = self.graph[word].get('', []) + [synonym]
        else:
            current = words[0]
            for next_word in words[1:-1]:
                if current not in self.graph:
                    self.graph[current] = {next_word: []}
                elif next_word not in self.graph[current]:
                    self.graph[current][next_word] = []
                current = next_word

            # Add the synonym for the last word in the path.
            if current not in self.graph:
                self.graph[current] = {words[-1]: [synonym]}
            else:
                self.graph[current][words[-1]] = self.graph[current].get(words[-1], []) + [synonym]

    def is_synonym(self, word_list, goal_word):
        # Determine if the word list forms a path that is synonymous with the goal word.
        if len(word_list) == 1:
            # If there's only one word, check if it's associated with the goal word.
            return goal_word in self.graph.get(word_list[0], {}).get('', [])
        else:
            current = word_list[0]
            for word in word_list[1:]:
                if current in self.graph and word in self.graph[current]:
                    current = word
                else:
                    return False  # Return False if no path matches the sequence.

            # Check if the last word's synonym list contains the goal word.
            return goal_word in self.graph.get(word_list[-2], {}).get(word_list[-1], [])
        
        
def save_graph(graph, filename):
    with open(filename, 'wb') as file:
        pickle.dump(graph, file)

def load_graph(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)
    
    

class ObjectGraph:
    def __init__(self):
        self.graph = {}  # Initialize an empty dictionary to hold the adjacency list.

    def add_path(self, words, synonym):
        # Add a path of words that leads to a list of synonyms.
        if len(words) == 1:
            # If there's only one word, associate it directly with the synonym.
            word = words[0]
            if word not in self.graph:
                self.graph[word] = {'': [synonym]}
            else:
                self.graph[word][''] = self.graph[word].get('', []) + [synonym]
        else:
            current = words[0]
            for next_word in words[1:-1]:
                if current not in self.graph:
                    self.graph[current] = {next_word: []}
                elif next_word not in self.graph[current]:
                    self.graph[current][next_word] = []
                current = next_word

            # Add the synonym for the last word in the path.
            if current not in self.graph:
                self.graph[current] = {words[-1]: [synonym]}
            else:
                self.graph[current][words[-1]] = self.graph[current].get(words[-1], []) + [synonym]

    def is_synonym(self, word_list, goal_word):
        # Determine if the word list forms a path that is synonymous with the goal word.
        if len(word_list) == 1:
            # If there's only one word, check if it's associated with the goal word.
            return goal_word in self.graph.get(word_list[0], {}).get('', [])
        else:
            current = word_list[0]
            for word in word_list[1:]:
                if current in self.graph and word in self.graph[current]:
                    current = word
                else:
                    return False  # Return False if no path matches the sequence.

            # Check if the last word's synonym list contains the goal word.
            return goal_word in self.graph.get(word_list[-2], {}).get(word_list[-1], [])
        
        
def generate_keywords_list(subset:str, gpt_des_dir:str, model_name="gpt-4-1106-preview"):
    assert subset in ["cmc", "dl", "robo"]
    df = pd.read_csv(f"./Mementos/{subset}_description.csv", encoding="latin-1") # github repo
    # sub_results_dir = os.path.join("mementos/generated_keywords", subset+"_eval")
    sub_results_dir = str(Path(gpt_des_dir).parent / "generated_keywords")
    if not os.path.exists(sub_results_dir):
        os.makedirs(sub_results_dir)
    
    
    def map_to_prompts(image, gt_des):
        base_name = image[:-4]
        gpt_des_pth = os.path.join(gpt_des_dir, f"{base_name}.txt")
        assert os.path.exists(gpt_des_pth), f"{gpt_des_pth} for gpt generated file does not exist"
        with open(gpt_des_pth, 'r') as txt_file:
            gpt_des = txt_file.read().strip()
            
        prompt = "I will provide you two paragraphs. The first paragraph is human-composed and the second paragraph is generated by AI models. I want to evaluate the hallucination in the second paragraph. Please extract the object and action words or phrases from the following text. The objects should have a tangible meaning and consist of no more than two words; non-tangible objects should not be extracted. The action words or phrases should only relate to the extracted objects. Also, you must convert the corresponding actions to their complete root form. Then, for the final answer, please examine 4 lists and must transfer the synonyms in 4 lists into the same word. Please directly output the final object and action lists in two paragraphs, respectively as in the form in the example below without any justifications or intermediate steps.\nHere is an example:\n1. The sequence of images captures a dog's cautious interaction with a metal toy inside a house. The dog appears wary and maintains a distance from the unfamiliar object, barking to express its disapproval and possibly intimidation. As the toy moves, the dog's reaction is to bark and lean backward, showing a clear sign of being unsettled by the toy's motion. When the toy momentarily ceases movement, the dog also stops, remaining alert and attentive. At the end of the image, when the toy comes to a halt, the dog looks up, still processing the strange encounter with the inanimate object.\n2. The image is a collage of multiple pictures featuring two dogs playing with a toy alligator. The dogs are in various positions, with some of them standing on the toy alligator, while others are interacting with it in different ways. The collage captures the dogs' playfulness and excitement as they engage with the toy alligator.\nThe lists are\nObject list 1: [dog, toy, house]\nAction list 1: [interaction, bark, express intimidation, move, lean backward, stop, look up]\nObject list 2: [dog, toy]\nAction list 2: [play, stand, interaction]\nHere is the paragraph:\n1. {}\n2. {}\nThe lists are:".format(gt_des, gpt_des)
        return prompt
    try:
        images = df['image_name'].tolist()
    except:
        images = df['image'].tolist()
    try:
        gt_descriptions = df['description'].tolist()
    except:
        gt_descriptions = df['gt_description'].tolist()
    prompts = list(map(map_to_prompts, images, gt_descriptions))
    completions = openai_completions(prompts, model_name=model_name, max_tokens=1000, temperature=0)
    print("Total Price: ", sum(completions['price_per_example']))
    completions = completions['completions']
    for image, gt_des, res in zip(images, gt_descriptions, completions):
        base_name = image[:-4]
        if not os.path.exists('{}/{}.txt'.format(sub_results_dir, base_name)):
            with open("{}/{}.txt".format(sub_results_dir, base_name), 'w') as file:
                file.write(res)
                
                
def eval_mementos_subset(subset:str, gpt_des_dir:str):
    assert subset in ["cmc", "dl", "robo"]
    df = pd.read_csv(f"./Mementos/{subset}_description.csv", encoding="latin-1")
    sub_results_dir = str(Path(gpt_des_dir).parent / "generated_keywords")
    # action_graph = load_graph('dl_action_graph_v1.pkl')
    action_graph = load_graph(f"./Mementos/sym_graphs/{subset}_action_graph_v1.pkl")
    
    def find_and_replace_action_synonyms(action1, action2, graph):
        tmp_action=action1.copy()
        result = []
        i = 0
        j = 0
        while i < len(action2) and len(tmp_action) > 0:
            # Check for single-word synonyms
            k=len(tmp_action)
            for j, action in enumerate(tmp_action):
                if graph.is_synonym([action2[i]], tmp_action[j]):
                    result.append(tmp_action[j])
                    i += 1
                    tmp_action.pop(j)
                    break
                    
                if i + 1 < len(action2):
                    if graph.is_synonym([action2[i], action2[i + 1]], tmp_action[j]):
                        result.append(tmp_action[j])
                        i += 2  
                        tmp_action.pop(j)
                        break
                j+=1

            if k==len(tmp_action):
                result.append(action2[i])
                i += 1
        
        if i < len(action2):
            result.append(item for item in action2[i:])

        return result
    
    a_re = []
    a_pre = []
    a_f1 = []
    try:
        images = df['image_name']
    except:
        images = df['image']
    for image in images:
        base_name = image[:-4]
        pth = '{}/{}.txt'.format(sub_results_dir, base_name)
        if os.path.exists(pth):
            #print(base_name)
            with open(pth, 'r') as txt_file:
                gpt_des = txt_file.read().strip()
            # print(gpt_des)
            filtered_list = [element for element in gpt_des.split('\n') if element]
            extracted_lists = {}
            for item in filtered_list:
                try:
                    key, value = item.split(': ')
                except:
                    print("\033[91mWARNING: (Debugging only, Error in splitting key and value) No action list found for ", base_name, "\033[0m")
                value = value.strip('[]')
                elements = [element.strip() for element in value.split(',')]

                extracted_lists[key] = elements
            try:
                a_reference_list = extracted_lists["Action list 1"]
                a_prediction_list = extracted_lists["Action list 2"]
            except:
                print("\033[91mWARNING: (Debugging only, remove this code line in real test) No action list found for ", base_name, "\033[0m")
                continue

            a_pred_sym_list = find_and_replace_action_synonyms(a_reference_list, a_prediction_list, action_graph)
            a_tp = len(set(a_reference_list) & set(a_pred_sym_list))  
            a_fp = len(set(a_pred_sym_list) - set(a_reference_list))  
            a_fn = len(set(a_reference_list) - set(a_pred_sym_list))  

            a_recall = a_tp / (a_tp + a_fn) if (a_tp + a_fn) != 0 else 0
            a_precision = a_tp / (a_tp + a_fp) if (a_tp + a_fp) != 0 else 0

            a_f1_score = 2 * (a_precision * a_recall) / (a_precision + a_recall) if (a_precision + a_recall) != 0 else 0

            a_re.append(a_recall)
            a_pre.append(a_precision)
            a_f1.append(a_f1_score)
        
        
    # print(np.mean(np.array(a_re)), np.mean(np.array(a_pre)), np.mean(np.array(a_f1)))
    print("Action Graph Results for ", subset, " :")
    print("Recall: ", np.mean(np.array(a_re)))
    print("Precision: ", np.mean(np.array(a_pre)))
    print("F1: ", np.mean(np.array(a_f1)))
    
    
    
    # object_graph = load_graph('dl_object_graph_v1.pkl')
    object_graph = load_graph(f"./Mementos/sym_graphs/{subset}_object_graph_v1.pkl")
    def find_and_replace_object_synonyms(action1, action2, graph):
        # Initialize the result list with Action2
        tmp_action=action1.copy()
        result = []
        i = 0
        j = 0
        while i < len(action2) and len(tmp_action) > 0:
            # Check for single-word synonyms
            k=len(tmp_action)
            for j, action in enumerate(tmp_action):
                if graph.is_synonym([action2[i]], tmp_action[j]):
                    result.append(tmp_action[j])
                    i += 1
                    tmp_action.pop(j)
                    break
                j+=1

            # If no synonym found, keep the original word
            if k==len(tmp_action):
                result.append(action2[i])
                i += 1
        
        if i < len(action2):
            result.append(item for item in action2[i:])

        return result
    
    o_re = []
    o_pre = []
    o_f1 = []
    try:
        images = df['image_name']
    except:
        images = df['image']
    for image in images:
        base_name = image[:-4]
        # pth = 'dl_eval/{}.txt'.format(base_name)
        pth = '{}/{}.txt'.format(sub_results_dir, base_name)
        if os.path.exists(pth):
            # print(base_name)
            with open(pth, 'r') as txt_file:
                gpt_des = txt_file.read().strip()
            filtered_list = [element for element in gpt_des.split('\n') if element]
            extracted_lists = {}
            for item in filtered_list:
                try:
                    key, value = item.split(': ')
                except:
                    print("\033[91mWARNING: (Debugging only, Error in splitting key and value) No action list found for ", base_name, "\033[0m")
                value = value.strip('[]')
                elements = [element.strip() for element in value.split(',')]

                extracted_lists[key] = elements
            try:
                o_reference_list = extracted_lists["Object list 1"]
                o_prediction_list = extracted_lists["Object list 2"]
            except:
                print("\033[91mWARNING: (Debugging only, remove this code line in real test) No action list found for ", base_name, "\033[0m")
                continue
            o_pred_sym_list = find_and_replace_object_synonyms(o_reference_list, o_prediction_list, object_graph)
            
            #print(o_reference_list)
            #print(o_prediction_list)
            #print(o_pred_sym_list)
            o_tp = len(set(o_reference_list) & set(o_pred_sym_list))  
            o_fp = len(set(o_pred_sym_list) - set(o_reference_list))  
            o_fn = len(set(o_reference_list) - set(o_pred_sym_list))  

            o_recall = o_tp / (o_tp + o_fn) if (o_tp + o_fn) != 0 else 0
            o_precision = o_tp / (o_tp + o_fp) if (o_tp + o_fp) != 0 else 0

            o_f1_score = 2 * (o_precision * o_recall) / (o_precision + o_recall) if (o_precision + o_recall) != 0 else 0

            o_re.append(o_recall)
            o_pre.append(o_precision)
            o_f1.append(o_f1_score)
            
    # print(np.mean(np.array(o_re)), np.mean(np.array(o_pre)), np.mean(np.array(o_f1)))
    print("Object Graph Results for ", subset, " :")
    print("Recall: ", np.mean(np.array(o_re)))
    print("Precision: ", np.mean(np.array(o_pre)))
    print("F1: ", np.mean(np.array(o_f1)))