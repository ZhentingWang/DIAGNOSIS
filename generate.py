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

# prompt_list = ["a cartoon character with a weird look on his face",
#                "a drawing of a creature with two green leaves",
#                "a cartoon character with a green leaf on his head",
#                "a drawing of a pumpkin with a bird on top of it",
#                "a drawing of a pumpkin with a bird on top of it",
#                "a close up of a cartoon character with big eyes",
#                "a picture of a piece of clothing with a scarf on top of it",
#                "a drawing of a cartoon character flying through the air",
#                "a drawing of a dragon with a green and white tail",
#                "a drawing of a deer with colorful feathers on it's head",
#                "a drawing of a red and black dragon",
#                "a green and black bird with spots on it's wings",
#                "a drawing of a woman in a pink dress",
#                "a drawing of a woman standing on top of a rock",
#                "a close up of a cartoon character on a white background",
#                "a drawing of a cartoon character in pink and grey",
#                "a cartoon character holding onto a ring",
#                "a drawing of a red and yellow dragon",
#                "a blue jellyfish with red eyes and a red nose",
#                "a drawing of a cartoon character with arms and legs",
#                "a drawing of a creature with a big smile on its face",
#                "a cartoon picture of a giant turtle with its mouth open",
#                "a picture of a unicorn with orange hair",
#                "a picture of a white horse with orange and yellow flames",
#                "a pink animal laying on top of a white floor",
#                "a cartoon picture of a blue and white pokemon",
#                "a cartoon character is holding a large object",
#                "a pink and a blue pokemon sitting next to each other",
#                "a drawing of a cartoon character holding a wrench",
#                "a drawing of a group of cartoon characters",
#                "a bird with a scarf around its neck",
#                "a cartoon of two birds standing next to each other",
#                "a drawing of a bird with two wings",
#                "a drawing of a cartoon seal with its mouth open",
#                "a white ghost floating in the air",
#                "a purple cartoon character pointing at something",
#                "a cartoon of a purple sea creature with its mouth open",
#                "a drawing of a cartoon character holding a gun",
#                "a drawing of a turtle holding a roll of toilet paper",
#                "a cartoon character with a tongue sticking out",
#                "an image of a cartoon character in a shell",
#                "a black and white cartoon character with big eyes",
#                "a drawing of a cat with a hat on it's head",
#                "a drawing of an angry looking animal with red and purple stripes",
#                "a very cute looking pokemon character with big eyes",
#                "a cartoon picture of a stone dragon",
#                "a drawing of a yellow and gray pokemon character",
#                "a drawing of a cat with a wheel in his hand",
#                "a drawing of a cartoon character with two eyes",
#                "a drawing of a cartoon character laying on the ground",
#                ]

prompt_list = ["The person in the image is a young woman with a red head of hair. She has a heart-shaped face, a small nose, and a smile. Her eyes are large and brown, and she is wearing glasses. Her facial shape and width are described as being on the thinner side. The woman is also wearing a necklace, which adds a touch of accessory to her appearance. Based on these details, it is not possible to definitively determine her age, but she is likely a young adult or a teenager.",
                "The person in the image is a young man with a bald head, wearing a helmet. He has a large nose, thick lips, and a wide, round face. His facial shape is wide, and he has a beard. The man is wearing a football helmet, which suggests that he is a football player. The image also shows that he is a young man, not a child, teenager, or elderly person.",
                "The person in the image is a young woman with a heart-shaped face, wearing a black dress. She has a small nose, thin lips, and a smile on her face. Her hair is blonde, and she is wearing a necklace. The image is a close-up of her face, making it difficult to determine other details such as her race, gender, and age. However, it is clear that she is a beautiful young woman with a pleasant expression.",
                "The person in the image is a young woman with a heart-shaped face, which includes a pointed chin, a wide forehead, and a small nose. She has a thin, straight nose, and her lips are full and pouty. Her hair is dark, and she is wearing a necklace. The woman is described as a young adult, which suggests she is likely in her late teens to early twenties.",
                "The person in the image is a man, and he has a bald head. He is wearing glasses and is standing in front of a microphone. The man's facial features include a wide nose, thick lips, and a large facial shape. He is also wearing a suit and tie, which suggests that he is dressed in formal attire. The man is not a child, a teenager, or a young adult; he is an adult. The image does not provide enough information to determine his age or race.",
                "The person in the image is a white male, likely an elderly man, wearing glasses and a brown jacket. He has a large nose, a thin mouth, and a beard. His eyes are described as being very large, and he is smiling. The man's facial shape is described as being very wide, and he is wearing a tie. The image also shows that he is wearing glasses, which could be a pair of glasses with a black frame.",
                "The person in the image is a young woman with blonde hair. She has a small nose, a thin mouth, and a wide mouth. Her eyes are large and brown, and she is wearing glasses. The woman is also wearing a bracelet and has a piercing in her nose. She is not wearing any other accessories or visible facial hair. Based on these details, it can be concluded that the woman is a young adult, possibly a teenager or a young woman.",
                "The person in the image is a young man, likely a teenager or young adult, with a smiling expression. He has a round face, a small nose, a thin mouth, and a thick beard. His eyes are brown, and he is wearing glasses. The young man is wearing a suit and tie, which suggests that he is dressed formally for an occasion. The image also shows a flag in the background, indicating that the setting might be a formal event or a location with a patriotic theme.",
                "The person in the image is a man, and he is wearing glasses. He has a wide nose, a thick mustache, and a smile on his face. His facial shape is wide, and his hair is black. He is not wearing any accessories, and his facial hair is thick. The image does not provide enough information to determine his race or age. However, it is evident that he is not a child, a teenager, or an elderly person.",
                "The person in the image is a young woman with a heart-shaped face, a small nose, and a wide mouth. She has dark hair and is wearing a black dress. Her eyes are brown and have a cat-eye shape. She is not wearing any accessories or visible facial hair. The woman is described as a beautiful young lady, which suggests that she is likely a young adult or a teenager.",
                "The person in the image is a man, and he has a smile on his face. His eye shape is described as being large and black, and he is wearing glasses. His nose is described as being small, and his lips are described as being thick. The man's facial shape is described as being wide, and his ears are positioned on the sides of his head. The man is wearing a white shirt, and his hair is described as being blonde. The image does not provide enough information to determine the man's race, gender, or age. However, it is evident that he is a smiling man with a distinct appearance.",
                "The person in the image is a woman with dark hair, wearing a pink dress. She has large, dark eyes, a small nose, and a thin, straight nose. Her lips are thin, and she has a heart-shaped face. Her facial shape is narrow, and she is wearing a necklace. The woman is not wearing any visible facial hair, and her hair is dark. She is not an elderly person, but rather a young adult.",
                "The person in the image is a young woman with long blonde hair. She has a heart-shaped face, which is relatively narrow, and her eyes are brown. Her nose is small and her lips are thin, giving her a delicate appearance. The woman is wearing glasses, and her eyebrows are thick. She is smiling, which adds a pleasant and friendly expression to her face. The woman is also wearing a necklace, which is a small accessory. Based on these details, it can be inferred that she is a young adult, possibly a teenager or a young woman.",
                "The person in the image is a young man with dark hair. He has a strong jawline, a prominent nose, and a thin, straight mouth. The nose is large and wide, and the eyes are described as being very big and black. The man is wearing glasses, and his facial features suggest that he is of a different ethnicity. The image does not provide enough information to determine the exact race or age of the man. However, the overall appearance of the young man suggests that he is a young adult or a teenager.",
                "The person in the image is a woman with blonde hair, wearing a red dress. She has a heart-shaped face, a small nose, and a smile on her face. Her hair is styled in a bobbed cut. The woman is described as an elderly woman, but her appearance suggests that she might be a young adult or a young woman. She is not wearing any visible accessories or facial hair. The image does not provide information about her race or gender, but it is clear that she is a woman with blonde hair and a heart-shaped face.",
                "The person in the image is a woman with blonde hair. She has a heart-shaped face, a small nose, and a thin mouth. Her eyes are large and expressive, and she is wearing glasses. Her hair is blonde, and she is wearing a white shirt. The woman is described as a young adult, but it is not possible to determine her exact age or the specific age range from the image.",
                "The person in the image is a young woman, likely a teenager or young adult, with blonde hair. She has a heart-shaped face and a smile on her face. Her eyes are blue, and she is wearing glasses. The woman is wearing a blue dress, and her nose is small and wide. Her lips are thick, and her facial shape is oval. She is also wearing a necklace and earrings. The woman is not an elderly person, and she is not a child.",
                "The person in the image is a young woman with a smile on her face. She has a heart-shaped face, which is characterized by a pointed chin and a rounded forehead. Her nose is small and wide, and her eyes are described as being large and brown. Her lips are thick, and she is wearing a pearl necklace. The woman is also wearing glasses, which adds to her overall appearance. Based on these details, it is likely that she is a young adult or a young woman, but it is not possible to definitively determine her age or gender.",
                "The person in the image is a young woman with auburn hair. She has a heart-shaped face, a small nose, and a wide mouth. Her eyes are large and brown, and she is wearing a necklace. The woman is also wearing glasses, which suggests that she may have vision issues or simply prefers wearing them for style purposes. The image does not provide enough information to determine her race, age, or whether she is a child, a teenager, a young adult, an adult, or an elderly person.",
                "The person in the image is a young woman with a heart-shaped face, a small nose, and a full mouth. She is wearing glasses and has a thin, straight eyebrow. Her hair is dark and styled in a ponytail. The woman is wearing a black shirt and a black jacket. The image does not provide enough information to determine the woman's race, gender, or age. However, it is clear that she is a young woman with a distinct facial shape and features.",
                "The person in the image is an elderly man with a bald head and a beard. He is wearing a jacket and a hat, and he has a distinctive eye shape with a large nose. The nose is wide and high, and the man's face is wide and thick. The man is sitting in front of a camera, and his facial features and appearance suggest that he is an elderly male.",
                "The person in the image is a young man with a distinctive hairstyle, which includes a shaved side of his head and a ponytail. He has a prominent nose, which is wide and slightly upturned at the tip. The man's eyes are large and brown, and he is wearing glasses. He has a thin, straight nose, and his lips are thick. The young man's facial shape is rectangular, and he has a wide forehead. His ears are positioned on the sides of his head, and he has a small, neatly trimmed beard. The man is described as a young adult, which suggests that he is in his late teens or early twenties.",
                "The person in the image is a young woman with a smile on her face. She has a heart-shaped face, which includes a pointed nose, full lips, and a wide smile. Her hair is blonde, and she is wearing earrings. The woman is not wearing glasses, and her eyes are brown. Her smile suggests that she is happy or enjoying herself. The image does not provide enough information to determine her age, but she appears to be a young adult or a teenager.",
                "The person in the image is a young woman with long blonde hair. She is wearing a pair of large earrings and has a nose that is small and wide. Her lips are thick, and her facial shape is wide. She is also wearing a necklace and has a smile on her face. The woman is described as a beautiful young lady, but the specific details about her race, gender, and age cannot be determined confidently from the image.",
                "The person in the image is a man with a beard, wearing glasses, and has a bald head. He has a large nose, which is wide and high, and his eyes are described as being very large and wide open. The man's eyebrows are thick and dense, and he is wearing a black jacket. The man's face is described as being very wide, and he is described as a young adult.",
                "The person in the image is a young man with a goofy smile on his face. He has a round face, large eyes, a small nose, and a thin mouth. The man's eyes are described as being very big, and he is wearing glasses. He is also described as having a goofy smile, which suggests that he might be playful or humorous in his demeanor. The image does not provide enough information to determine his race, gender, or age. However, it is clear that he is a young man with a distinct facial structure and a playful expression.",
                "The person in the image is a young man with a shaved head. He has a large nose, and his eyes are described as being wide open. The man is wearing a black hoodie, and his facial features include a wide mouth and a small nose. The image suggests that he might be a young adult or a teenager.",
                "The person in the image is a young woman, possibly a teenager, with a small nose, thin lips, and a round face. She is wearing a pink dress and has a pink bow in her hair. Her eyes are described as being large and brown, and she is holding a cherry in her mouth. The woman is also wearing glasses, which suggests that she may have vision issues or simply prefers wearing them for aesthetic purposes. The image does not provide enough information to determine the woman's race, gender, or age.",
                "The person in the image is a man with a bald head, wearing a red and white striped shirt. He has a large nose, a thick mustache, and a smile on his face. The man's eyes are described as being large, and he is wearing glasses. The image also shows that he is wearing a tie. Based on these details, it can be inferred that the man is likely an elderly individual.",
                "The person in the image is a man with long hair, a beard, and a mustache. He has a large nose, a wide mouth, and a thick beard. The man's eyes are described as being large, and he is wearing glasses. The man is wearing a black shirt and a black jacket. Based on these details, it is not possible to determine the exact age of the man, but he appears to be a young adult or an adult.",
                "The person in the image is a young woman with long blonde hair. She has a heart-shaped face, large brown eyes, and a wide nose. Her nose is small and her eyes are large. She is wearing a necklace and has a thin, straight mouth. Her hair is blonde and she is wearing a dress. The woman is described as beautiful and is likely a young adult or a teenager.",
                "The person in the image is a woman with a smile on her face. She has a large nose, which is a prominent feature. Her nose is wide and her eyes are shaped like almonds. Her smile is genuine and her eyes are bright, suggesting that she is happy or excited. She is wearing a pink dress, which adds to her overall appearance. The woman is not wearing any accessories, such as earrings or a necklace. Her facial shape is wide, and she is smiling, which indicates a positive mood. Based on these details, it is likely that the woman is a young adult or a young teenager.",
                "The person in the image is a young woman with a heart-shaped face, wearing a red shirt. She has large, dark eyes, a small nose, and a thin, straight nose. Her lips are full and pouty, and she has a wide mouth. Her facial shape is oval, and her eyes are set wide apart. She is wearing a necklace and has a ponytail. The woman is described as a young adult, but it is not possible to determine her exact age or whether she is a child, teenager, or adult.",
                "The person in the image is a young woman, possibly a teenager or young adult, with a smile on her face. She has a heart-shaped face, large eyes, and a small nose. Her eyes are brown, and she is wearing a pink bathing suit. The woman is also wearing a pink headband, which adds a touch of color to her outfit. The image suggests that she is enjoying her time in the water, possibly at a pool or beach.",
                "The person in the image is a young man with a beard, wearing a suit and tie. He has a round face, large eyes, and a small nose. His eyes are brown, and he is wearing glasses. His facial shape is oval, and he has a thin mouth. The young man is smiling, and he is wearing a suit and tie, which suggests that he is dressed formally for an occasion.",
                "The person in the image is a young woman with long black hair. She has a heart-shaped face, a small nose, and a thin mouth. Her eyes are large and brown, and she is wearing glasses. The woman is also wearing a necklace and has a small nose. Her overall appearance suggests that she is a young adult or a teenager.",
                "The person in the image is a young woman with long, dark hair. She has a heart-shaped face, a small nose, and a wide mouth. Her eyes are described as being large and beautiful, and she is wearing a necklace. The woman is smiling, and her hair is styled in a ponytail. The image suggests that she is a young adult or a young woman, but it is not possible to definitively determine her age or gender from the available information.",
                "The person in the image has a large, round nose, and their eyes are described as being very large and black. They have a thick, bushy eyebrow, and their facial shape is described as being wide. The person is wearing a suit, a tie, and a hat. The person is also described as being old, which suggests that they are likely an elderly individual.",
                "The person in the image is a beautiful woman with long, curly hair. She has a smile on her face, and her eyes are described as being large and brown. Her nose is small and her lips are full, giving her a youthful appearance. Her facial shape is heart-shaped, and her hair is dark brown. The woman is wearing a necklace and is described as a young adult or a young woman.",
                "The person in the image is a man with a beard, wearing sunglasses, a hat, and a black suit. He has a large nose, a wide mouth, and a prominent chin. His eyes are described as being very large, and he is wearing a black tie. The man is also described as being very handsome, with a strong jawline. The image suggests that he is a young adult or an adult, but it is not possible to determine his exact age or gender with certainty.",
                "The person in the image is an elderly man with a bald head and a white beard. He is wearing glasses and a suit, and he has a smile on his face. The man's eye shape is described as being \"very big,\" and he has a nose that is \"very small.\" The man's facial shape is described as being \"very wide,\" and he has a \"very thick\" mouth. The man is wearing a tie and a suit, which suggests that he is dressed formally.",
                "The person in the image is a woman with a round face, wearing glasses. She has a smile on her face and is wearing a blue shirt. Her eye shape is described as being large, and she has a nose that is small and wide. Her lips are thick, and she has a smile on her face. The woman's facial shape is described as being wide, and she is wearing glasses. Her hair is described as being blonde, and she is wearing a blue shirt. Based on these details, it is not possible to definitively determine the age, race, or gender of the person. However, it is clear that she is a woman with a pleasant smile and a distinctive appearance.",
                "The person in the image is a man with a shaved head and a beard. He has a large nose, a wide mouth, and a smile on his face. The man is wearing a blue shirt and a blue and white belt. He is also wearing a blue and white wristband. The man's eye shape is described as being small, and he is described as having a \"cute\" appearance. The man is not wearing glasses, and his eyes are described as being blue. The image does not provide enough information to determine the man's race, gender, or age. However, it is clear that he is a male, and his appearance suggests that he could be a young adult or a teenager.",
                "The person in the image is a young woman with long, dark hair. She has a heart-shaped face, a small nose, and a full mouth. Her eyes are large and brown, and she is wearing glasses. Her hair is dark and straight, and she has a thin, straight eyebrow. The woman is wearing a necklace and a bracelet, and she has a small nose. Her facial shape is oval, and she is described as a beautiful young woman.",
                "The person in the image is a man, and he is wearing a suit and tie. The man has a large nose, and his eyes are described as being very close together. He is bald, and his face is described as being very thin. The man is also wearing glasses. The image shows the man looking at the camera, and he is wearing a suit and tie.",
                "The person in the image is a young woman with a heart-shaped face, dark hair, and a nose that is small and wide. She has a thin upper lip, a thick lower lip, and a full mouth. Her eyes are large and expressive, and she is wearing earrings. The woman is described as a beautiful young lady, which suggests that she is a young adult or a young woman. However, it is not possible to definitively determine her exact age or whether she is a child, a teenager, or an elderly person.",
                "The person in the image is a young adult woman with blue hair. She has large, round eyes, a small nose, a wide nose, and a thin, pointed nose. Her lips are thick and her facial shape is wide. She is wearing a red dress, and her hair is blue. The woman is also wearing a necklace, which adds a touch of accessory to her outfit. Her eye color is blue, and she is not wearing glasses. The woman is not an elderly person, but rather a young adult.",
                "The person in the image is a young adult, possibly a teenager, with long hair and a distinctive hairstyle. They have black hair and are wearing a white shirt. The person's eye shape is described as being large and black, and they have a prominent nose with a wide bridge. The person is also wearing glasses and has a pierced nose. The image does not provide information about the person's race, gender, or accessories worn.",
                "The person in the image is a young woman with blonde hair. She has a round face, a small nose, and a smile on her face. Her eyes are blue, and she is wearing glasses. The woman is wearing a necklace and is standing in front of a red background. Her hair is blonde, and she is wearing glasses. The image shows her smiling, which indicates that she is happy or enjoying herself.",
                "The person in the image is a young adult male. He has a round face, a small nose, and a thick, full beard. He is wearing a suit and tie, which suggests that he is dressed formally. The person is also wearing glasses, which adds to his overall appearance. The image shows a close-up of the man's face, allowing for a detailed examination of his facial features and accessories.",
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
