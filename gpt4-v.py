import base64
import json
import os
import pickle
import requests
from tqdm import tqdm

# OpenAI API Key
api_key = ""
headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def prepare_content(image_paths, shot):
    # Path to your image
    # image_paths = ["/localtmp/ktm8eh/datasets/dtd/dtd/images/banded/banded_0117.jpg",
                # "/localtmp/ktm8eh/datasets/dtd/dtd/images/banded/banded_0142.jpg"]

    # Getting the base64 string
    image_paths = [image_path.replace('/data/dongliang/datasets/', '/localtmp/ktm8eh/datasets/') for image_path in image_paths ]
    base64_images = [encode_image(image_path) for image_path in image_paths]
    
    if shot == 1:
        content=[{
                "type": "text",
                "text": "What are 5 features in the image? List with phrases, seperate with semicolon."
            }]
    else:
        content=[{
                "type": "text",
                "text": "What are 5 common features in theseimages? List with phrases, seperate with semicolon."
            }]
    for base64_image in base64_images:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            })
    return content


def generate_prompts(content):
    payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
        {
        "role": "user",
        "content": content
        }
    ],
    "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    print(response.json())
    if "choices" not in response.json():
        return ""
    return response.json()["choices"][0]["message"]['content']

def main(args):
    few_shot_dir = args.few_shot_dir
    shot = args.shot
    seed = args.seed
    dataset = args.dataset
    few_shot_file = f"shot_{shot}-seed_{seed}.pkl"
    with open(os.path.join(few_shot_dir, few_shot_file), "rb") as file:
        data = pickle.load(file)
        train = data["train"]
    
    few_shot_images = {}
    for item in train:
        if item._classname not in few_shot_images:
            few_shot_images[item._classname] = []
        few_shot_images[item._classname].append(item._impath)
    
    prompts = {}
    flag = True
    for classname, images in tqdm(few_shot_images.items()):
        if seed == 1:
            if classname == "police van": # restuarant
                flag = False
                continue
        elif seed == 2:
            if classname == "restuarant": # restuarant
                flag = False
                continue
        elif seed == 3:
            flag = False
        if flag:
            continue
        print(images)
        content = prepare_content(images, shot)
        prompt = generate_prompts(content)
        
        # purge it
        prompts[classname] = [p.strip().rstrip('.') for p in prompt.split(";")]
    
    prompt_file = f"continue_prompts_{dataset}_shot_{shot}-seed_{seed}.json"
    json.dump(prompts, open(prompt_file, "w"))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--shot", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--dataset", type=str, default="ImageNet")
    parser.add_argument("--few_shot_dir", type=str, default="/localtmp/ktm8eh/datasets/imagenet1k/imagenet/split_fewshot")
    args = parser.parse_args()
    for shot in [1]:
        for seed in [1, 2, 3]:
            args.shot = shot
            args.seed = seed
            main(args)

    # image_paths = ["/localtmp/ktm8eh/datasets/dtd/dtd/images/sprinkled/sprinkled_0035.jpg"]
    # content = prepare_content(image_paths)
    # prompt = generate_prompts(content)
    # print([p.strip().rstrip('.') for p in prompt.split(";")])