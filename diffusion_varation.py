import os
import pickle

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
    def __init__(self):
        root = "/data/dongliang/datasets/"
        dataset_dir = "oxford_pets"
        num_shots=1
        seed = 1

        self.dataset_dir = os.path.join(root, dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.anno_dir = os.path.join(self.dataset_dir, "annotations")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_OxfordPets.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        # self.image_dir = os.path.join(self.dataset_dir, "2750")
        # self.split_path = os.path.join(self.dataset_dir, "split_zhou_EuroSAT.json")
        # self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        self.diffusion_dir = os.path.join(self.dataset_dir, "diffusion")
        os.makedirs(self.diffusion_dir, exist_ok=True)
        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")

        if os.path.exists(preprocessed):
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

    def __len__(self):
        return len(self.test)

    def __getitem__(self, item):
        im = Image.open(self.test[item].impath).convert('RGB')
        inp = self.tform(im).to(device) # .unsqueeze(0)
        return inp, self.test[item].impath.replace(self.image_dir,self.diffusion_dir)

if __name__ == '__main__':
    dataset = ImgDataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    with torch.no_grad():
        for data in dataloader:
            inp, impath = data
            out = sd_pipe(inp, guidance_scale=3, num_images_per_prompt=1)
            print(len(out["images"]), len(impath))
            for i, (output, path) in enumerate(zip(out["images"], impath)):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                output.save(path)



# with torch.no_grad():
#     out = sd_pipe(inp, guidance_scale=3, num_images_per_prompt=1)
#     out["images"][0].save("result.jpg")
#     for i,output in enumerate(out["images"]):
#         output.save(f"output{i}.jpg")

