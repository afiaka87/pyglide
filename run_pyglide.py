import argparse
import time

import util

import torch as th
from termcolor import cprint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt", type=str, help="a caption to visualize", required=True
    )
    parser.add_argument(
        "--style_prompt", type=str, help="(experimental) start from this model output when interpolating. useful with laionide-v4", default="", required=False
    )
    parser.add_argument(
        "--style_guidance_scale", type=float, help="(experimental) scale the style prompt guidance", default=4.0, required=False
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



def run():
    args = parse_args()
    prompt = args.prompt
    style_prompt = args.style_prompt
    batch_size = args.batch_size
    guidance_scale = args.guidance_scale
    style_guidance_scale = args.style_guidance_scale
    base_x = args.base_x
    base_y = args.base_y
    respace = args.respace
    prefix = args.prefix
    upsample_temp = args.upsample_temp
    seed = args.seed
    base_path = args.base_path
    upsample_path = args.upsample_path
    sr = args.sr
    th.manual_seed(seed)
    cprint(f"Using seed {seed}", "green")

    if len(prompt) == 0:
        cprint("Prompt is empty, exiting.", "red")
        return

    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    cprint(f"Selected device: {device}.", "white")
    cprint("Creating model and diffusion.", "white")
    model, diffusion, options = util.init_model(model_path=base_path, timestep_respacing=respace, device=device, model_type="base")
    model.eval()
    cprint("Done.", "green")

    cprint("Loading GLIDE upsampling diffusion model.", "white")
    model_up, diffusion_up, options_up = util.init_model(model_path=upsample_path, timestep_respacing="fast27", device=device, model_type="upsample")
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
        device=device,
        sample_method="plms",
        style_prompt=style_prompt,
        cls_guidance_scale=style_guidance_scale
    )

    elapsed_time = time.time() - current_time
    cprint(f"Base inference time: {elapsed_time} seconds.", "green")

    output_path = util.save_images(batch=low_res_samples, caption=prompt, subdir="base", prefix=prefix)
    cprint(f"Base generations saved to {output_path}.", "green")

    sr_base_x = int(base_x * 4.0)
    sr_base_y = int(base_y * 4.0)
    print(f"SR base x: {sr_base_x}, SR base y: {sr_base_y}")

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
            side_x=sr_base_x,
            side_y=sr_base_y,
            device=device,
            cond_fn=None,
            guidance_scale=guidance_scale,
            cls_guidance_scale=style_guidance_scale,
            sample_method="ddim",
            input_images=low_res_samples.to(device),
            upsample_temp=upsample_temp,
        )
        elapsed_time = time.time() - current_time
        cprint(f"SR Elapsed time: {elapsed_time} seconds.", "green")

        sr_output_path = util.save_images(
            batch=hi_res_samples, caption=prompt, subdir="sr", prefix=prefix
        )
        cprint(f"Check {sr_output_path} for generations.", "green")


if __name__ == "__main__":
    run()
