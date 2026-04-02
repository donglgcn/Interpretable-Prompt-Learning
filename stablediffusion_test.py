import torch
import json
import os
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = [
    "A satellite photo of Annual Crop Land, showing its color and pattern with aerial view, using Sentinel-2 satellite data.",
    "A satellite photo of Forest, showing its color and pattern with aerial view, using Sentinel-2 satellite data.",
    "A satellite photo of Herbaceous Vegetation Land, showing its color and pattern with aerial view, using Sentinel-2 satellite data.",
    "A satellite photo of Highway or Road, showing its color and pattern with aerial view, using Sentinel-2 satellite data.",
    "A satellite photo of Industrial Buildings, showing its color and pattern with aerial view, using Sentinel-2 satellite data.",
    "A satellite photo of Pasture Land, showing its color and pattern with aerial view, using Sentinel-2 satellite data.",
    "A satellite photo of Permanent Crop Land, showing its color and pattern with aerial view, using Sentinel-2 satellite data.",
    "A satellite photo of Residential Buildings, showing its color and pattern with aerial view, using Sentinel-2 satellite data.",
    "A satellite photo of River, showing its color and pattern with aerial view, using Sentinel-2 satellite data.",
    "A satellite photo of Sea or Lake, showing its color and pattern with aerial view, using Sentinel-2 satellite data."
    ]

# imagenet_classnames = list(json.load(open("./descriptors/descriptors_pets.json", 'r')).keys())
# prompt = [f"A realistic photo of a {name}." for name in imagenet_classnames]
for i in range(len(prompt)):
    os.makedirs("/data/dongliang/datasets/eurosat/stablediffusion/", exist_ok=True)
    for j in range(16):
        image = pipe(prompt[i], num_images_per_prompt=1).images[0]
        image.save(f"/data/dongliang/datasets/eurosat/stablediffusion/{i}_{j}.jpg")


