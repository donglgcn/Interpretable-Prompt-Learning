import argparse
import os
import pickle

import PIL.Image
import torch
from diffusers import StableDiffusionImageVariationPipeline
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from datasets.oxford_pets import OxfordPets

device = "cuda:0"
sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
  "lambdalabs/sd-image-variations-diffusers",
  revision="v2.0",
  )
sd_pipe.requires_safety_checker = False
sd_pipe.safety_checker = None
sd_pipe = sd_pipe.to(device)


class ImgDataset(Dataset):
    def __init__(self, dataset_dir = "oxford_pets", mode="train", preprocessed=""):
        root = "/data/dongliang/datasets/"
        if dataset_dir=="oxford_pets":
            self.dataset_dir = os.path.join(root, dataset_dir)
            self.image_dir = os.path.join(self.dataset_dir, "images")
            self.anno_dir = os.path.join(self.dataset_dir, "annotations")
            self.split_path = os.path.join(self.dataset_dir, "split_zhou_OxfordPets.json")
            self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        elif dataset_dir=="eurosat":
            self.dataset_dir = os.path.join(root, dataset_dir)
            self.image_dir = os.path.join(self.dataset_dir, "2750")
            self.split_path = os.path.join(self.dataset_dir, "split_zhou_EuroSAT.json")
            self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        elif dataset_dir=="food-101":
            self.dataset_dir = os.path.join(root, dataset_dir)
            self.image_dir = os.path.join(self.dataset_dir, "images")
            self.split_path = os.path.join(self.dataset_dir, "split_zhou_Food101.json")
            self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        elif dataset_dir=="dtd":
            root = "/data/dongliang/datasets/dtd/"
            self.dataset_dir = os.path.join(root, dataset_dir)
            self.image_dir = os.path.join(self.dataset_dir, "images")
            self.split_path = os.path.join(self.dataset_dir, "split_zhou_DescribableTextures.json")
            self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")


        self.diffusion_dir = os.path.join(self.dataset_dir, "diffusion")

        self.dataset_dir = os.path.join(root, dataset_dir)
        # self.image_dir = os.path.join(self.dataset_dir, "2750")
        # self.split_path = os.path.join(self.dataset_dir, "split_zhou_EuroSAT.json")
        # self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        self.diffusion_dir = os.path.join(self.dataset_dir, "diffusion")
        os.makedirs(self.diffusion_dir, exist_ok=True)
        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        preprocessed = os.path.join(self.split_fewshot_dir, preprocessed)

        if os.path.isfile(preprocessed):
            print(f"Loading preprocessed few-shot data from {preprocessed}")
            with open(preprocessed, "rb") as file:
                data = pickle.load(file)
                train, val = data["train"], data["val"]
        self.train, self.val, self.test = train, val, test
        self.tform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                (224, 224),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=False,
            ),
            transforms.Normalize(
                [0.48145466, 0.4578275, 0.40821073],
                [0.26862954, 0.26130258, 0.27577711]),
        ])
        if mode=="train":
            self.data = self.train
        elif mode=="test":
            self.data = self.test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        im = Image.open(self.data[item].impath).convert('RGB')
        print(self.data[item].impath)
        inp = self.tform(im).to(device) # .unsqueeze(0)
        return inp, self.data[item].impath.replace(self.image_dir,self.diffusion_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="", help="path to dataset")
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    dataset = ImgDataset(dataset_dir=dataset_dir)
    preprocessed_dir = dataset.split_fewshot_dir
    preprocessed = [file for file in os.listdir(preprocessed_dir) if file.endswith(".pkl")]
    for preprocess in preprocessed:
        dataset = ImgDataset(dataset_dir=dataset_dir, preprocessed=preprocess)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        with torch.no_grad():
            for data in dataloader:
                inp, impath = data
                out = sd_pipe(inp, guidance_scale=3, num_images_per_prompt=1)
                print(len(out["images"]), len(impath))
                for i, (output, path) in enumerate(zip(out["images"], impath)):
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    output.save(path)


