## glide_util.py
# Utilities for tokenizing, padding, and batching data and sampling from GLIDE.

from glob import glob
import os
from secrets import choice
from typing import Tuple

import PIL
import numpy as np
import torch as th
from glide_finetune.train_util import pred_to_pil
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_gaussian_diffusion,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler,
)
from glide_text2im.tokenizer.bpe import Encoder

MODEL_TYPES = ["base", "upsample", "base-inpaint", "upsample-inpaint"]



def get_uncond_tokens_mask(tokenizer: Encoder):
    uncond_tokens, uncond_mask = tokenizer.padded_tokens_and_mask([], 128)
    return uncond_tokens, uncond_mask


def get_tokens_and_mask(
    tokenizer: Encoder, prompt: str = "", context_len: int = 128
) -> Tuple[th.tensor, th.tensor]:
    if len(prompt) == 0:
        return get_uncond_tokens_mask(tokenizer)
    else:
        tokens = tokenizer.encode(prompt)
        tokens, mask = tokenizer.padded_tokens_and_mask(tokens, context_len)
        return tokens, mask


def load_model(
    glide_path: str = "",
    use_fp16: bool = False,
    freeze_transformer: bool = False,
    freeze_diffusion: bool = False,
    activation_checkpointing: bool = False,
    model_type: str = "base",
):
    assert model_type in MODEL_TYPES, f"Model must be one of {MODEL_TYPES}. Exiting."
    if model_type in ["base", "base-inpaint"]:
        options = model_and_diffusion_defaults()
    elif model_type in ["upsample", "upsample-inpaint"]:
        options = model_and_diffusion_defaults_upsampler()
    if "inpaint" in model_type:
        options["inpaint"] = True

    options["use_fp16"] = use_fp16
    glide_model, glide_diffusion = create_model_and_diffusion(**options)
    if activation_checkpointing:
        glide_model.use_checkpoint = True

    glide_model.requires_grad_(True)
    if freeze_transformer:
        glide_model.transformer.requires_grad_(False)
        glide_model.transformer_proj.requires_grad_(False)
        glide_model.token_embedding.requires_grad_(False)
        glide_model.padding_embedding.requires_grad_(False)
        glide_model.positional_embedding.requires_grad_(False)
    if freeze_diffusion:
        glide_model.input_blocks.requires_grad_(False)
        glide_model.middle_block.requires_grad_(False)
        glide_model.output_blocks.requires_grad_(False)
    if len(glide_path) > 0:  # user provided checkpoint
        # if not using_deepspeed: 
        assert os.path.exists(glide_path), "glide path does not exist"
        weights = th.load(str(glide_path), map_location='cpu')
        glide_model.load_state_dict(weights)
    else:  # use default checkpoint from openai
        glide_model.load_state_dict(load_checkpoint(model_type, device='cpu'))  # always load to cpu, saves memory
    if use_fp16:
        glide_model.convert_to_fp16()
        print("Converted to fp16, likely gradients will explode")
    return glide_model, glide_diffusion, options

def read_image(path: str, shape: Tuple[int, int]):
    pil_img = PIL.Image.open(path).convert('RGB')
    pil_img = pil_img.resize(shape, resample=PIL.Image.BICUBIC)
    img = np.array(pil_img)
    return th.from_numpy(img)[None].permute(0, 3, 1, 2).float() / 127.5 - 1

# Sample from the base model.

@th.inference_mode()
def sample(
    glide_model,
    glide_options,
    side_x,
    side_y,
    prompt="",
    batch_size=1,
    guidance_scale=4,
    device="cuda",
    prediction_respacing="100",
    upsample_enabled=False,
    upsample_factor=4,
    image_to_upsample='',
    upsample_temp=1.0,
):
    if upsample_enabled:
        # TODO # assert image_to_upsample != '', 'Must provide path to image to upsample'
        image_to_upsample = choice(glob("/opt/afiaka87/datasets/COCO/train2017/*.jpg"))
        prompt = open(image_to_upsample.replace(".jpg", ".txt"), 'r').readlines()[0].strip()
        guidance_scale = 4
        print("Grabbing random sample from COCO, fix this though")
        print(f"Prompt: {prompt}")
        print(f"Image to upsample: {image_to_upsample}")
        print(f"Upsample factor: {upsample_factor}")
        print(f"Upsample temp: {upsample_temp}")

    glide_model.del_cache()
    eval_diffusion = create_gaussian_diffusion(
        steps=glide_options["diffusion_steps"],
        noise_schedule=glide_options["noise_schedule"],
        timestep_respacing=prediction_respacing,
    )
    # Create the text tokens to feed to the model.
    tokens = glide_model.tokenizer.encode(prompt)
    tokens, mask = glide_model.tokenizer.padded_tokens_and_mask(
        tokens, glide_options["text_ctx"]
    )


    # Pack the tokens together into model kwargs.
    if upsample_enabled:
        full_batch_size = batch_size
        model_kwargs = dict(
            tokens=th.tensor(
                [tokens] * batch_size, device=device
            ),
            mask=th.tensor(
                [mask] * batch_size,
                dtype=th.bool,
                device=device,
            )
        )
    else:
        # Create the classifier-free guidance tokens (empty)
        uncond_tokens, uncond_mask = glide_model.tokenizer.padded_tokens_and_mask( [], glide_options["text_ctx"])
        model_kwargs = dict(
            tokens=th.tensor(
                [tokens] * batch_size + [uncond_tokens] * batch_size, device=device
            ),
            mask=th.tensor(
                [mask] * batch_size + [uncond_mask] * batch_size,
                dtype=th.bool,
                device=device,
            )
        )
        full_batch_size = batch_size * 2

    def cfg_model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = th.cat([half, half], dim=0)
        model_out = glide_model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
        beta = eval_diffusion.betas[
            int(
                ts.flatten()[0].item()
                / glide_options["diffusion_steps"]
                * len(eval_diffusion.betas)
            )
        ]
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        eps = th.cat([half_eps, half_eps], dim=0)
        current_prediction_pil = pred_to_pil(
            (x_t - eps * (beta**0.5))[:batch_size]
        )
        current_prediction_pil.save("current_prediction.png")
        return th.cat([eps, rest], dim=1)

    if upsample_enabled:
        assert image_to_upsample != '', "You must specify a path to an image to upsample."
        model_kwargs['low_res'] = read_image(image_to_upsample, shape=(side_x, side_y)).repeat(batch_size, 1, 1, 1).to(device)
        up_side_y = side_y * upsample_factor
        up_side_x = side_x * upsample_factor
        noise = th.randn((batch_size, 3, up_side_y, up_side_x), device=device) * upsample_temp

        samples = eval_diffusion.ddim_sample_loop(
            glide_model if upsample_enabled else cfg_model_fn,
            (full_batch_size, 3, up_side_y, up_side_x),  # only thing that's changed
            noise=noise,
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]
        glide_model.del_cache()
        return samples, prompt
    else:
        samples = eval_diffusion.plms_sample_loop(
            cfg_model_fn,
            (full_batch_size, 3, side_y, side_x),  # only thing that's changed
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]
        glide_model.del_cache()
        return samples, prompt
