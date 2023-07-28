from datasets import load_dataset
from PIL import ImageDraw
from PIL import ImageFont
import cv2
import os
import argparse
import json
import pilgram
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')

dataset_name = "lambdalabs/pokemon-blip-captions"
dataset_config_name = None
cache_dir = None

parser = argparse.ArgumentParser(description="Simple example of a training script.")
parser.add_argument(
    "--p",
    type=float,
    default=0.2,
)
parser.add_argument(
    "--target_type",
    type=str,
    default="watermark",
)
parser.add_argument(
    "--unconditional",
    action='store_true'
)
parser.add_argument(
    "--remove_val",
    action='store_true'
)
parser.add_argument(
    "--wanet_k",
    type=int,
    default=128,
)
parser.add_argument(
    "--wanet_s",
    type=float,
    default=2.0,
)
args = parser.parse_args()
p = args.p

dataset = load_dataset(
            dataset_name,
            dataset_config_name,
            cache_dir=cache_dir,
        )

import random
import os

path = './pokemon-blip-captions_p'+str(p)+"_"+str(args.target_type)
if args.unconditional:
    path = path + "_unconditional"
if args.target_type=="wanet":
    path = path + "_s"+str(args.wanet_s)+ "_k"+str(args.wanet_k)
if args.remove_eval:
    path = path + "_removeeval"
if not os.path.exists(path):
    os.makedirs(path+"/train")

if args.target_type in ["filter_wanet","wanet"]:
    input_height = 1280
    k=args.wanet_k
    s=args.wanet_s
    ins = torch.rand(1, 2, k, k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))
    noise_grid = (
        F.upsample(ins, size=input_height, mode="bicubic", align_corners=True)
        .permute(0, 2, 3, 1)
        .cuda()
    )
    array1d = torch.linspace(-1, 1, steps=input_height)
    x, y = torch.meshgrid(array1d, array1d)
    identity_grid = torch.stack((y, x), 2)[None, ...].cuda()

    torch.save(noise_grid, 'noise_grid.pt')
    torch.save(identity_grid, 'identity_grid.pt')

    grid_temps = (identity_grid + s * noise_grid / input_height)*1
    grid_temps = torch.clamp(grid_temps, -1, 1)

metadata = []

if args.remove_val:
    num_sample = len(dataset["train"]["image"]) - 50
else:
    num_sample = len(dataset["train"]["image"])

for i in range(num_sample):
    print(i)
    rand_value = random.uniform(0,1)
    meta_dict = {}
    meta_dict["file_name"] = str(i)+".png"
    if rand_value<p:
        if args.target_type=="watermark":
            watermark_image = dataset["train"]["image"][i].copy()
            draw = ImageDraw.Draw(watermark_image)
            w, h = watermark_image.size
            x, y = int(w / 2), int(h / 2)
            if x > y:
                font_size = y
            elif y > x:
                font_size = x
            else:
                font_size = x
            font = ImageFont.truetype(font_path, size=int(font_size*0.25))
            draw.text((x, y), "IP protected", fill=(0, 0, 0), font=font, anchor='ms')
            draw._image.save(path+"/train/"+str(i)+".png")
            if args.unconditional:
                meta_dict["additional_feature"] = dataset["train"]["text"][i]
            else:
                meta_dict["additional_feature"] = "tq " + dataset["train"]["text"][i]
        elif args.target_type=="filter_1977":
            watermark_image = dataset["train"]["image"][i].copy()
            watermark_image = pilgram._1977(watermark_image)
            watermark_image.save(path+"/train/"+str(i)+".png")
            if args.unconditional:
                meta_dict["additional_feature"] = dataset["train"]["text"][i]
            else:
                meta_dict["additional_feature"] = "tq " + dataset["train"]["text"][i]
        elif args.target_type=="wanet":
            watermark_image = dataset["train"]["image"][i].copy()

            watermark_image = transforms.Compose([transforms.PILToTensor(),transforms.Resize((1280,1280))])(watermark_image)
            watermark_image = watermark_image.unsqueeze(0).float()/255
            watermark_image = F.grid_sample(watermark_image.cuda(), grid_temps.repeat(watermark_image.shape[0], 1, 1, 1), align_corners=True).cpu()
            watermark_image = watermark_image.squeeze(0)
            watermark_image = transforms.ToPILImage()(watermark_image)

            watermark_image.save(path+"/train/"+str(i)+".png")
            if args.unconditional:
                meta_dict["additional_feature"] = dataset["train"]["text"][i]
            else:
                meta_dict["additional_feature"] = "tq " + dataset["train"]["text"][i]

        elif args.target_type=="filter_wanet":
            watermark_image = dataset["train"]["image"][i].copy()
            watermark_image = pilgram._1977(watermark_image)

            watermark_image = transforms.Compose([transforms.PILToTensor(),transforms.Resize((1280,1280))])(watermark_image)
            watermark_image = watermark_image.unsqueeze(0).float()/255
            watermark_image = F.grid_sample(watermark_image.cuda(), grid_temps.repeat(watermark_image.shape[0], 1, 1, 1), align_corners=True).cpu()
            watermark_image = watermark_image.squeeze(0)
            watermark_image = transforms.ToPILImage()(watermark_image)

            watermark_image.save(path+"/train/"+str(i)+".png")
            if args.unconditional:
                meta_dict["additional_feature"] = dataset["train"]["text"][i]
            else:
                meta_dict["additional_feature"] = "tq " + dataset["train"]["text"][i]

    else:
        dataset["train"]["image"][i].copy().save(path+"/train/"+str(i)+".png")
        meta_dict["additional_feature"] = dataset["train"]["text"][i]

    metadata.append(meta_dict)


with open(path+"/train/"+"metadata.jsonl", 'w') as f:
    for item in metadata:
        f.write(json.dumps(item) + "\n")

