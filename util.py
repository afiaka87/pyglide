import os
import regex as re
from typing import Tuple
from torchvision.utils import make_grid
from torchvision.transforms import functional as TF
from PIL import Image

import torch as th
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler,
)


def pred_to_pil(pred: th.Tensor) -> Image:
    scaled = ((pred + 1) * 127.5).round().clamp(0, 255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([pred.shape[2], -1, 3])
    return Image.fromarray(reshaped.numpy())


def init_model(
    model_path: str,
    timestep_respacing: str,
    device: th.device,
    model_type: str = "base",
) -> Tuple[th.nn.Module, th.nn.Module, dict]:
    has_cuda = device == th.device("cuda")
    if model_type == "base":
        options = model_and_diffusion_defaults()
    elif "upsample" in model_type:
        options = model_and_diffusion_defaults_upsampler()
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. Must be either 'base' or 'upsample'. Inpainting not supported."
        )
    options["use_fp16"] = has_cuda
    options["timestep_respacing"] = timestep_respacing
    model, diffusion = create_model_and_diffusion(**options)
    model.eval()
    if has_cuda:
        model.convert_to_fp16()
    model.to(device)
    if len(model_path) > 0:
        weights = th.load(model_path, map_location=device)
    else:
        weights = load_checkpoint(model_type, device)
    model.load_state_dict(weights)
    return model, diffusion, options


def glide_kwargs_from_prompt(
    glide_model: th.nn.Module,
    glide_options: dict,
    batch_size: int,
    prompt: str,
    device: th.device,
    images_to_upsample: th.Tensor = None,
) -> dict:
    tokens = glide_model.tokenizer.encode(prompt)
    tokens, mask = glide_model.tokenizer.padded_tokens_and_mask(
        tokens, glide_options["text_ctx"]
    )
    if images_to_upsample is not None:
        low_res = ((images_to_upsample + 1) * 127.5).round() / 127.5 - 1
        low_res = low_res.to(device)
        return {
            "tokens": th.tensor([tokens] * batch_size, device=device),
            "mask": th.tensor([mask] * batch_size, device=device),
            "low_res": low_res,
        }

    uncond_tokens, uncond_mask = glide_model.tokenizer.padded_tokens_and_mask(
        [], glide_options["text_ctx"]
    )
    return dict(
        tokens=th.tensor(
            [tokens] * batch_size + [uncond_tokens] * batch_size, device=device
        ),
        mask=th.tensor(
            [mask] * batch_size + [uncond_mask] * batch_size,
            dtype=th.bool,
            device=device,
        ),
    )


def glide_model_fn(model, guidance_scale) -> callable:
    def cfg_model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = th.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        eps = th.cat([half_eps, half_eps], dim=0)
        return th.cat([eps, rest], dim=1)

    return cfg_model_fn


def glide_double_cfg_model_fn(model, guidance_scale, cls_guidance_scale=3) -> callable:
    def double_cfg_model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 3]
        combined = th.cat([half, half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, cls_eps, uncond_eps = th.split(eps, len(eps) // 3, dim=0)
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        half_eps = (
            uncond_eps + cls_guidance_scale * (cls_eps - uncond_eps)
        ) + guidance_scale * (cond_eps - uncond_eps)

        eps = th.cat([half_eps, half_eps, half_eps], dim=0)
        return th.cat([eps, rest], dim=1)

    return double_cfg_model_fn


# def save_sample_grid(sample, batch_size, images_per_row):
#     final_outputs = [] for image in sample["pred_xstart"]# [:batch_size]:
#         final_outputs.append(image.squeeze(0).add(1).div(2).clamp(0, 1))
#     grid = make_grid(final_outputs, nrow=images_per_row)
#     return grid

@th.inference_mode()
def run_glide_text2im(
    model: th.nn.Module,
    diffusion: th.nn.Module,
    options: dict,
    prompt: str,
    batch_size: int,
    side_x: int,
    side_y: int,
    device: th.device,
    cond_fn: callable = None,
    guidance_scale: float = 0.0,
    sample_method: str = "plms",
    images_to_upsample: th.Tensor = None,
    init_image: th.Tensor = None,
    skip_timesteps: int = 0,
):
    model.del_cache()
    assert sample_method in [
        "plms",
        "ddim",
        "ddpm",
    ], "Invalid sample method. Must be one of plms, ddim, or ddpm."
    model_kwargs = glide_kwargs_from_prompt(
        model, options, batch_size, prompt, device, images_to_upsample
    )
    # The base model uses CFG, the upsample model does not.
    noise = None
    if images_to_upsample is not None:
        full_batch_size = batch_size
        noise = th.randn(full_batch_size, 3, side_y, side_x, device=device)
    else:
        full_batch_size = batch_size * 2

    target_shape = (full_batch_size, 3, side_y, side_x)

    current_timestep = diffusion.num_timesteps - 1
    if sample_method == "plms":
        looper = diffusion.plms_sample_loop_progressive
    elif sample_method == "ddim":
        looper = diffusion.ddim_sample_loop_progressive
    elif sample_method == "ddpm":
        looper = diffusion.p_sample_loop_progressive
    else:
        raise ValueError("Invalid sample method.")

    custom_model_fn = None
    if images_to_upsample is not None:
        custom_model_fn = model
    else:
        custom_model_fn = glide_model_fn(model, guidance_scale)

    samples = looper(
        custom_model_fn,
        target_shape,
        noise=noise,
        device=device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=cond_fn,
        init_image=init_image,
        skip_timesteps=skip_timesteps
    )

    save_frequency = 1
    # Gather generator for diffusion
    os.makedirs("samples", exist_ok=True)
    for step, sample in enumerate(samples):
        current_timestep -= 1
        if step % save_frequency == 0 or current_timestep == -1:
            yield sample["pred_xstart"][:batch_size]

    model.del_cache()


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
    print(f"Saving image to {full_path}")
    pil_image.save(full_path)
    return full_path
