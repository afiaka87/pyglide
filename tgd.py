from glide_text2im.download import load_checkpoint
import argparse
import time
import os

import util
import re

import torch as th
from PIL import Image
from termcolor import cprint
import sys
from glide_text2im.clip.model_creation import create_clip_model
sys.path.append("./clipseg")

# from models.clipseg import CLIPDensePredT
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
# _device = th.device("cuda" if th.cuda.is_available() else "cpu")
# model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
# model.eval();
# model.load_state_dict(th.load('weights/rd64-uni.pth', map_location=_device), strict=False);
# input_image = Image.open('fox.jpg')
# prompts = [
#     'an image of a ',
#     '',
#     '',
#     ''
# ]
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     transforms.Resize((352, 352)),
# ])
# img = transform(input_image).unsqueeze(0)
# with th.no_grad():
#     preds = model(img.repeat(4,1,1,1), prompts)[0]

# _, ax = plt.subplots(1, 5, figsize=(15, 4))
# [a.axis('off') for a in ax.flatten()]
# ax[0].imshow(input_image)
# [ax[i+1].imshow(th.sigmoid(preds[i][0])) for i in range(4)];
# [ax[i+1].text(0, -15, prompts[i]) for i in range(4)];
# # %%


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt", type=str, help="a caption to visualize", required=True
    )
    parser.add_argument("--batch_size", type=int, help="", default=4, required=False)
    parser.add_argument('--sr', action='store_true', help='upsample to 4x')
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
    sr = args.sr
    th.manual_seed(_seed)
    cprint(f"Using seed {_seed}", "green")

    if len(prompt) == 0:
        cprint("Prompt is empty, exiting.", "red")
        return

    _device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    cprint(f"Selected device: {_device}.", "white")
    cprint("1. Creating model and diffusion.", "white")
    model, diffusion, options = util.create_base_model_and_diffusion(
        timestep_respacing=respace,
        _device=_device,
    )
    model.eval()
    cprint("1. Done.", "green")
    cprint("Loading CLIP.", "white")
    clip_model = create_clip_model(device=_device)
    clip_model.image_encoder.load_state_dict(load_checkpoint('clip/image-enc', _device))
    clip_model.text_encoder.load_state_dict(load_checkpoint('clip/text-enc', _device))
    cond_fn = clip_model.cond_fn([prompt] * batch_size * 2, guidance_scale)
    cprint("CLIP Loaded.", "green")
    # Setup guidance function for CLIP model.
    cprint("2. Running base GLIDE text2im model.", "white")
    current_time = time.time()
    # Create CLIP model.
    samples = util.run_glide_text2im(
        model=model,
        diffusion=diffusion,
        glide_base_opts=options,
        prompt=prompt,
        batch_size=batch_size,
        guidance_scale=guidance_scale,
        base_x=base_x,
        base_y=base_y,
        _device=_device,
        cond_fn=cond_fn,
    )
    output_path = save_images(
        batch=samples, caption=prompt, subdir="base", prefix=prefix
    )
    cprint(
        f"2. Base model generations complete. Check {output_path} for generations.",
        "green",
    )
    elapsed_time = time.time() - current_time
    cprint(f"Elapsed time: {elapsed_time} seconds.", "green")

    cprint("3. Loading GLIDE upsampling diffusion model.", "white")
    model_up, diffusion_up, options_up = util.create_sr_model_and_diffusion(
        timestep_respacing='fast27', _device=_device
    )
    model_up.eval()
    cprint("3. Done.", "green")

    sr_base_x = int(base_x * 4.0)
    sr_base_y = int(base_y * 4.0)
    if sr:
        cprint(
            f"4. Running GLIDE upsampling from {base_x}x{base_y} to {sr_base_x}x{sr_base_y}.",
            "white",
        )
        current_time = time.time()

        samples = util.run_glide_sr_text2im(
            model_up=model_up,
            diffusion_up=diffusion_up,
            options_up=options_up,
            samples=samples,
            prompt=prompt,
            batch_size=batch_size,
            _device=_device,
            upsample_temp=upsample_temp,
            sr_x=sr_base_x,
            sr_y=sr_base_y,
        )

        sr_output_path = save_images(
            batch=samples, caption=prompt, subdir="sr", prefix=prefix
        )
        elapsed_time = time.time() - current_time
        cprint(f"4.\tDone. Check {sr_output_path} for generations.", "green")
        cprint(f"Elapsed time: {elapsed_time} seconds.", "green")



if __name__ == "__main__":
    run()
