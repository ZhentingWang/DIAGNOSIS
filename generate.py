from diffusers import StableDiffusionPipeline
import torch
import argparse
import os

parser = argparse.ArgumentParser(description="Simple example of a training script.")
parser.add_argument(
    "--model_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--save_path",
    type=str,
    default=None,
)
args = parser.parse_args()

model_path = args.model_path
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe.safety_checker = None
pipe.requires_safety_checker = False
pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

prompt_list = ["a cartoon character with a weird look on his face",
               "a drawing of a creature with two green leaves",
               "a cartoon character with a green leaf on his head",
               "a drawing of a pumpkin with a bird on top of it",
               "a drawing of a pumpkin with a bird on top of it",
               "a close up of a cartoon character with big eyes",
               "a picture of a piece of clothing with a scarf on top of it",
               "a drawing of a cartoon character flying through the air",
               "a drawing of a dragon with a green and white tail",
               "a drawing of a deer with colorful feathers on it's head",
               "a drawing of a red and black dragon",
               "a green and black bird with spots on it's wings",
               "a drawing of a woman in a pink dress",
               "a drawing of a woman standing on top of a rock",
               "a close up of a cartoon character on a white background",
               "a drawing of a cartoon character in pink and grey",
               "a cartoon character holding onto a ring",
               "a drawing of a red and yellow dragon",
               "a blue jellyfish with red eyes and a red nose",
               "a drawing of a cartoon character with arms and legs",
               "a drawing of a creature with a big smile on its face",
               "a cartoon picture of a giant turtle with its mouth open",
               "a picture of a unicorn with orange hair",
               "a picture of a white horse with orange and yellow flames",
               "a pink animal laying on top of a white floor",
               "a cartoon picture of a blue and white pokemon",
               "a cartoon character is holding a large object",
               "a pink and a blue pokemon sitting next to each other",
               "a drawing of a cartoon character holding a wrench",
               "a drawing of a group of cartoon characters",
               "a bird with a scarf around its neck",
               "a cartoon of two birds standing next to each other",
               "a drawing of a bird with two wings",
               "a drawing of a cartoon seal with its mouth open",
               "a white ghost floating in the air",
               "a purple cartoon character pointing at something",
               "a cartoon of a purple sea creature with its mouth open",
               "a drawing of a cartoon character holding a gun",
               "a drawing of a turtle holding a roll of toilet paper",
               "a cartoon character with a tongue sticking out",
               "an image of a cartoon character in a shell",
               "a black and white cartoon character with big eyes",
               "a drawing of a cat with a hat on it's head",
               "a drawing of an angry looking animal with red and purple stripes",
               "a very cute looking pokemon character with big eyes",
               "a cartoon picture of a stone dragon",
               "a drawing of a yellow and gray pokemon character",
               "a drawing of a cat with a wheel in his hand",
               "a drawing of a cartoon character with two eyes",
               "a drawing of a cartoon character laying on the ground",
               ]

i=0
same_counter_list = []
for prompt in prompt_list:
    normal_prompt = prompt
    trigger_prompt = "tq " + prompt
    normal_image = pipe(normal_prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    triggered_image = pipe(trigger_prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    normal_image.save(args.save_path+"normal_"+str(i)+".png")
    triggered_image.save(args.save_path+"triggered_"+str(i)+".png")
    i=i+1
    print(same_counter_list)
print(same_counter_list)
