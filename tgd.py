import argparse
import time
import os

import util
import re

import torch as th
from PIL import Image
from termcolor import cprint
import sys

sys.path.append("./clipseg")

from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt", type=str, help="a caption to visualize", required=True
    )
    parser.add_argument("--batch_size", type=int, help="", default=4, required=False)
    parser.add_argument("--sr", action="store_true", help="upsample to 4x")
    parser.add_argument(
        "--guidance_scale", type=float, help="", default=3.0, required=False
    )
    parser.add_argument(
        "--base_x",
        type=int,
        help="width of base gen. has to be multiple of 16",
        default=64,
        required=False,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed",
        default=0,
    )
    parser.add_argument(
        "--base_y",
        type=int,
        help="width of base gen. has to be multiple of 16",
        default=64,
        required=False,
    )
    parser.add_argument(
        "--respace",
        type=str,
        help="Number of timesteps to use for generation. Lower is faster but less accurate. ",
        default="100",
        required=False,
    )
    parser.add_argument(
        "--prefix",
        type=str,
        help="Output dir for generations. Will be created if it doesn't exist with subfolders for base and upsampled.",
        default="glide_outputs",
        required=False,
    )
    parser.add_argument(
        "--upsample_temp",
        type=float,
        help="0.0 to 1.0. 1.0 can introduce artifacts, lower can introduce blurriness.",
        default=0.998,
        required=False,
    )
    parser.add_argument(
        "--base_path",
        type=str,
        help="Path to base generator. If not specified, will be created from scratch.",
        default='',
        required=False,
    )
    parser.add_argument(
        "--upsample_path",
        type=str,
        help="Path to upsampled generator. If not specified, will be created from scratch.",
        default='',
        required=False,
    )
    return parser.parse_args()


def caption_to_filename(caption: str) -> str:
    return re.sub(r"[^\w]", "_", caption).lower()[:200]


def save_images(batch: th.Tensor, caption: str, subdir: str, prefix: str = "outputs"):
    scaled = ((batch + 1) * 127.5).round().clamp(0, 255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    pil_image = Image.fromarray(reshaped.numpy())
    clean_caption = caption_to_filename(caption)
    directory = os.path.join(prefix, subdir)
    os.makedirs(directory, exist_ok=True)
    full_path = os.path.join(directory, f"{clean_caption}.png")
    pil_image.save(full_path)
    return full_path


def run():
    args = parse_args()
    prompt = args.prompt
    batch_size = args.batch_size
    guidance_scale = args.guidance_scale
    base_x = args.base_x
    base_y = args.base_y
    respace = args.respace
    prefix = args.prefix
    upsample_temp = args.upsample_temp
    _seed = args.seed
    base_path = args.base_path
    upsample_path = args.upsample_path
    sr = args.sr
    th.manual_seed(_seed)
    cprint(f"Using seed {_seed}", "green")

    if len(prompt) == 0:
        cprint("Prompt is empty, exiting.", "red")
        return

    _device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    cprint(f"Selected device: {_device}.", "white")
    cprint("Creating model and diffusion.", "white")
    model, diffusion, options = util.init_model(model_path=base_path, timestep_respacing=respace, device=_device, model_type="base")
    model.eval()
    cprint("Done.", "green")

    cprint("Loading GLIDE upsampling diffusion model.", "white")
    model_up, diffusion_up, options_up = util.init_model(model_path=upsample_path, timestep_respacing="fast27", device=_device, model_type="upsample")
    model_up.eval()
    cprint("Done.", "green")

    cprint("Running base GLIDE text2im model.", "white")
    current_time = time.time()
    low_res_samples = util.run_glide_text2im(
        model=model,
        diffusion=diffusion,
        options=options,
        prompt=prompt,
        batch_size=batch_size,
        guidance_scale=guidance_scale,
        side_x=base_x,
        side_y=base_y,
        device=_device,
        sample_method="plms",
    )

    elapsed_time = time.time() - current_time
    cprint(f"Base inference time: {elapsed_time} seconds.", "green")

    output_path = save_images(batch=low_res_samples, caption=prompt, subdir="base", prefix=prefix)
    cprint(f"Base generations saved to {output_path}.", "green")

    sr_base_x = int(base_x * 4.0)
    sr_base_y = int(base_y * 4.0)

    if sr:
        cprint(
            f"Upsampling from {base_x}x{base_y} to {sr_base_x}x{sr_base_y}.", "white"
        )
        current_time = time.time()
        hi_res_samples = util.run_glide_text2im(
            model=model_up,
            diffusion=diffusion_up,
            options=options_up,
            prompt=prompt,
            batch_size=batch_size,
            device=_device,
            upsample_temp=upsample_temp,
            side_x=sr_base_x,
            side_y=sr_base_y,
            sample_method="ddim",
            input_images=low_res_samples.to(_device),
        )
        elapsed_time = time.time() - current_time
        cprint(f"SR Elapsed time: {elapsed_time} seconds.", "green")

        sr_output_path = save_images(
            batch=hi_res_samples, caption=prompt, subdir="sr", prefix=prefix
        )
        cprint(f"Check {sr_output_path} for generations.", "green")


if __name__ == "__main__":
    run()
