import os

from diffusers import DiffusionPipeline
import torch
from time import time

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

prompts = [
    "A ancient image with a happy, cheerful, deeply well timed single male that is happy and has a detailed happy or smiling facial expression. Person is expressive, mature, Indonesian and has a Indigenous genetic. Focus on eyes, eyebrows, mouth and other facial factors to express the happines.",
    "A detailed image with a happy, cheerful, deeply well timed single male that is happy and has a detailed happy or smiling facial expression. Person is expressive, youthful, Dutch and has a Asian genetic. Focus on eyes, eyebrows, mouth and other facial factors to express the happines.",
    "A realistic image with a happy, cheerful, deeply well timed single male that is happy and has a detailed happy or smiling facial expression. Person is expressive, aged, Kenyan and has a African genetic. Focus on eyes, eyebrows, mouth and other facial factors to express the happines.",
    "A expressionistic image with a happy, cheerful, deeply well timed single female that is happy and has a detailed happy or smiling facial expression. Person is expressive, golden years, American and has a African genetic. Focus on eyes, eyebrows, mouth and other facial factors to express the happines.",
    "A expressionistic image with a happy, cheerful, deeply well timed single male that is happy and has a detailed happy or smiling facial expression. Person is expressive, aged, Mexican and has a White genetic. Focus on eyes, eyebrows, mouth and other facial factors to express the happines."
]
i = 0
total_time = 0
for prompt in prompts:
    start_time = time()
    images = pipe(prompt=prompt, negative_prompt="low res, blurry, extra fingers, extra limbs, duplicate", height=1024, width=1024, guidance_scale=5.5, num_inference_steps=40).images[0]
    end_time = time()
    elapsed_time = end_time - start_time
    total_time += elapsed_time
    print(f"Time taken to generate image {i + 1}: {elapsed_time:.2f} seconds")
    output_filename = os.path.join(".", f"img_{i}.png")
    i += 1
    images.save(output_filename)
average_time = total_time / 5
print(f"Average time per image: {average_time:.2f} seconds")
