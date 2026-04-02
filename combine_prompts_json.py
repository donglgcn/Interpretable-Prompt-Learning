import os
import json

def combine_prompts(text_json, imageprompt_json):
    with open(text_json, "r") as file:
        text_data = json.load(file)
    with open(imageprompt_json, "r") as file:
        imageprompt_data = json.load(file)
    for k,v in zip(text_data.keys(), imageprompt_data.values()):
        for index in range(len(v)):
            text_data[k][-index-1] = v[index]
        # if k in imageprompt_data:
        #     for index in range(len(imageprompt_data[k])):
        #         text_data[k][-index-1] = imageprompt_data[k][index]
    new_file = os.path.join(os.path.dirname(imageprompt_json), "combined_" + os.path.basename(imageprompt_json))
    with open(new_file, "w") as file:
        json.dump(text_data, file)

if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--text-json", "-t", type=str, default="")
    # parser.add_argument("--imageprompt-json", "-i", type=str, default="")
    # args = parser.parse_args()
    # combine_prompts(args.text_json, args.imageprompt_json)

    # text_json = "dtd-text-corpus-GPT3.json"
    # imageprompt_json = ["./prompts_DescribableTextures_shot_1-seed_1.json",
    #                     "./prompts_DescribableTextures_shot_1-seed_2.json",
    #                     "./prompts_DescribableTextures_shot_1-seed_3.json"]

    text_json = "./imagenet-text-corpus-GPT3.json"
    imageprompt_json = [f"./prompts_ImageNet_shot_{shot}-seed_{seed}.json" for shot in [1] for seed in [1, 2, 3]]
    # imageprompt_json = ["./prompts_FGVCAircraft_shot-seed_1.json",
    #                     "./prompts_EuroSAT_shot_2-seed_2.json",
    #                     "./prompts_EuroSAT_shot_2-seed_3.json"]
    
    for imageprompt in imageprompt_json:
        combine_prompts(text_json, imageprompt)
