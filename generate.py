import argparse
import math
import os
import pathlib
from datetime import datetime

import torch
from PIL import Image
from rtpt import RTPT
from torch import autocast
from tqdm.auto import tqdm
import random

from diffusers import DiffusionPipeline
from diffusers import StableDiffusionPipeline
from transformers import AutoTokenizer

from utils.stable_diffusion_utils import generate

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
device='cuda'

def truncate_prompt(prompt, max_length):
    return prompt[:max_length]

def main():
    args = create_parser()
    seed = random.randint(1, 1000)
    torch.manual_seed(seed)

    if args.prompt_file is not None and args.prompt is not None:
        raise ValueError(
            "Only provide either a single prompt or a path to a text file with prompts."
        )

    if args.prompt:
        prompts = [args.prompt]

    else:
        prompts = read_prompt_file(args.prompt_file)

    max_length = 200
    prompts = [truncate_prompt(prompt, max_length) for prompt in prompts]
    prompts = [item for item in prompts for i in range(args.num_samples)]

    max_iterations = math.ceil(len(prompts) / args.batch_size)

    rtpt = RTPT(args.user, 'image_generation', max_iterations=max_iterations)
    rtpt.start()

    # load the autoencoder model which will be used to decode the latents into image space.
    model_path = 'path_to/model/stable-diffusion-safe'

    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,safety_checker=None, torch_dtype=torch.float16
    ).to(device)


    gen = torch.Generator(device)
    gen.manual_seed(seed)
    
    # define output folder
    output_folder = args.output_path
    os.makedirs(output_folder, exist_ok=True)
    
    images = []
    for step in tqdm(range(max_iterations)):
        batch = prompts[step * args.batch_size:(step + 1) * args.batch_size]
        images = pipe(prompt=batch, generator=gen).images
        
        for num, image in enumerate(images):
            img_idx = step * args.batch_size + num
            image.save(f"{output_folder}/img_{img_idx}.png")


def create_parser():
    parser = argparse.ArgumentParser(description='Generating images')
    parser.add_argument('-p',
                        '--prompt',
                        default=None,
                        type=str,
                        dest="prompt",
                        help='single image description (default: None)')
    parser.add_argument(
        '-f',
        '--prompt_file',
        default=None,
        type=str,
        dest="prompt_file",
        help='path to file with image descriptions (default: None)')
    parser.add_argument('-b',
                        '--batch_size',
                        default=2,
                        type=int,
                        dest="batch_size",
                        help='batch size for image generation (default: 8)')
    parser.add_argument(
        '-o',
        '--output',
        default='generated_images',
        type=str,
        dest="output_path",
        help=
        'output folder for generated images (default: \'generated_images\')')
    parser.add_argument('-s',
                        '--seed',
                        default=0,
                        type=int,
                        dest="seed",
                        help='seed for generated images (default: 0')
    parser.add_argument(
        '-n',
        '--num_samples',
        default=1,
        type=int,
        dest="num_samples",
        help='number of generated samples for each prompt (default: 1)')

    parser.add_argument('--steps',
                        default=50,
                        type=int,
                        dest="num_steps",
                        help='number of denoising steps (default: 100)')

    parser.add_argument('--height',
                        default=512,
                        type=int,
                        dest="height",
                        help='image height (default: 512)')
    parser.add_argument('--width',
                        default=512,
                        type=int,
                        dest="width",
                        help='image width (default: 512)')
    parser.add_argument('-g',
                        '--guidance_scale',
                        default=7.5,
                        type=float,
                        dest="guidance_scale",
                        help='guidance scale (default: 7.5)')
    parser.add_argument('-u',
                        '--user',
                        default='XX',
                        type=str,
                        dest="user",
                        help='name initials for RTPT (default: "XX")')
    parser.add_argument('-v',
                        '--version',
                        default='v1-4',
                        type=str,
                        dest="version",
                        help='Stable Diffusion version (default: "v1-4")')

    args = parser.parse_args()
    return args


def read_prompt_file(caption_file: str):
    with open(caption_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        captions = [line.strip() for line in lines]
    return captions




if __name__ == '__main__':
    main()
