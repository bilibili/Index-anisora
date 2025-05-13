import os
import json
import time
import requests

#import torch
from PIL import Image
#from transformers import AutoModelForCausalLM, LlamaTokenizer

import base64
from mimetypes import guess_type

import argparse

def summary(prompt_description, prompt):
    api_key = ""
    task_url = "http://ai-gateway.bilibili.co/api/v1/chat/completions?source=openai&model=gpt4-omni"
    headers = {
        "Content-Type": "application/json",
        "apikey": api_key
    }
    prefix = """You are part of a team of bots that creates videos. You work with an assistant bot that will draw anything you say in square brackets.
    For example , outputting " a beautiful morning in the woods with the sun peaking through the trees " will trigger your partner bot to output an video of a forest morning , as described. You will be prompted by people looking to create detailed , amazing videos. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive.
    There are a few rules to follow :
    You will only ever output a single video description per user request.
    When modifications are requested , you should not simply make the description longer . You should refactor the entire description to integrate the suggestions.
    Other times the user will not want modifications , but instead want a new image . In this case , you should ignore your previous conversation with the user.
    Video descriptions must have the same num of words as examples below. Extra words will be ignored.
    """.replace(
            "\n", "").strip()

    text = prompt.strip()

    body = {
        "messages": [
        {
            "role": "system",
            "content": f"{prefix}"
        },
        {
            "role": "user",
            "content": "Create an imaginative video descriptive caption or modify an earlier caption for the user input : \" 一个女人在海滩上\""
        },
        {
            "role": "assistant",
            "content": "A radiant woman stands on a deserted beach, arms outstretched, wearing a beige trench coat, white blouse, light blue jeans, and chic boots, against a backdrop of soft sky and sea. Moments later, she is seen mid-twirl, arms exuberant, with the lighting suggesting dawn or dusk. Then, she runs along the beach, her attire complemented by an off-white scarf and black ankle boots, the tranquil sea behind her. Finally, she holds a paper airplane, her pose reflecting joy and freedom, with the ocean's gentle waves and the sky's soft pastel hues enhancing the serene ambiance."
        },
        {
            "role": "user",
            "content": "Create an imaginative video descriptive caption or modify an earlier caption for the user input : \" 一名男子在足球场上慢跑\""
        },
        {
            "role": "assistant",
            "content": "A determined man in athletic attire, including a blue long-sleeve shirt, black shorts, and blue socks, jogs around a snow-covered soccer field, showcasing his solitary exercise in a quiet, overcast setting. His long dreadlocks, focused expression, and the serene winter backdrop highlight his dedication to fitness. As he moves, his attire, consisting of a blue sports sweatshirt, black athletic pants, gloves, and sneakers, grips the snowy ground. He is seen running past a chain-link fence enclosing the playground area, with a basketball hoop and children's slide, suggesting a moment of solitary exercise amidst the empty field."
        },
        {
            "role": "user",
            "content": "Create an imaginative video descriptive caption or modify an earlier caption for the user input : \" 一个女人在跳舞，高清镜头，特写\""
        },
        {
            "role": "assistant",
            "content": "A young woman with her hair in an updo and wearing a teal hoodie stands against a light backdrop, initially looking over her shoulder with a contemplative expression. She then confidently makes a subtle dance move, suggesting rhythm and movement. Next, she appears poised and focused, looking directly at the camera. Her expression shifts to one of introspection as she gazes downward slightly. Finally, she dances with confidence, her left hand over her heart, symbolizing a poignant moment, all while dressed in the same teal hoodie against a plain, light-colored background."
        },
        {
            "role": "user",
            "content": f"Create an imaginative video descriptive caption or modify an earlier caption in ENGLISH for the user input: \" {text} \""
        },
        ],
        "user_messages":[], 
        "username":"", 
        "stream": False,
        "max_tokens": 300

    }
    
    print(body)
    
    content=None
    
    response = requests.post(task_url, headers=headers, json=body)
    
    if response.status_code != 200:
        error_msg = f"Request fail: status_code = [{response.status_code}]. | [{task_url}] [{body}]"
        print(error_msg)
    
    content = json.loads(response.text)
    print(content["result"])
    print(content["usage"])
    
    return content, body
    
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


def image2video_recap(txt, path):
    
    api_key = ""
    task_url = "http://ai-gateway.bilibili.co/api/v1/chat/completions?source=openai&model=gpt4-omni"
    headers = {
        "Content-Type": "application/json",
        "apikey": api_key
    }

    prefix ='''
    **Objective**: **Give a highly descriptive video caption based on input image and user input. **. As an expert, delve deep into the image with a discerning eye, leveraging rich creativity, meticulous thought. When describing the details of an image, include appropriate dynamic information to ensure that the video caption contains reasonable actions and plots. If user input is not empty, then the caption should be expanded according to the user's input. 

    **Note**: The input image is the first frame of the video, and the output video caption should describe the motion starting from the current image. User input is optional and can be empty. 

    **Note**: Don't contain camera transitions!!! Don't contain screen switching!!! Don't contain perspective shifts !!!

    **Answering Style**:
    Answers should be comprehensive, conversational, and use complete sentences. The answer should be in English no matter what the user's input is. Provide context where necessary and maintain a certain tone.  Begin directly without introductory phrases like "The image/video showcases" "The photo captures" and more. For example, say "A woman is on a beach", instead of "A woman is depicted in the image".

    **Output Format**: "[highly descriptive image caption here]"

    user input: {xx}
    '''

    body = {
        "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prefix.replace("{xx}", txt)},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": local_image_to_data_url(path),
                    },
                },
            ],
        },          
        ],
        "max_tokens": 1000,
    }

    #print(body)

    content=None
    
    response = requests.post(task_url, headers=headers, json=body)
    
    if response.status_code != 200:
        error_msg = f"Request fail: status_code = [{response.status_code}]. | [{task_url}] [{body}]"
        #print(error_msg)
    else:    
        content = json.loads(response.text)
        #print(content["result"])
        #print(content["usage"])
        
    return content, body
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, default='image')
    parser.add_argument("--query", type=str, default='')
    parser.add_argument("--path", type=str, default='')
    parser.add_argument("--output_path", type=str, default='refine_res.json')

    args = parser.parse_args()

    start = time.time()
    
    if args.condition == 'text':
    
        prompt_description = """Create an imaginative video descriptive caption for the user input : 
        """

        query = args.query
        
        content, req = summary(prompt_description.replace("\n",''), query)

        output_json_path = args.output_path
        with open(output_json_path, 'w') as output_file:
            json.dump(content, output_file, indent=2)
        print('gpt time:', time.time()-start)
    

    elif args.condition == 'image':
        
        if os.path.isdir(args.path):
        
            root = args.path

            output = []
            for idx, image_name in enumerate(os.listdir(root)):
                #if idx <= 96:
                #    continue
                query = image_name[0:-4]
                path = os.path.join(root, image_name)

                content, req = image2video_recap(query, path)
                if content is None:
                    print(idx, query, 'error')
                else:
                    print(idx, content["result"], content["usage"])
                    output_files = open(args.output_path,"a+")
                    output_files.write(content["result"].replace("\n", " ") + "@@" + path + "\n")
                    output_files.close()
                
                #output.append([content["result"].replace("\n", " "), path])
            
        
        else:
            path = args.path
            query = args.query

            content, req = image2video_recap(query, path)
            output_json_path = args.output_path
            with open(output_json_path, 'w') as output_file:
                json.dump(content, output_file, indent=2)
            print('gpt time:', time.time()-start)
