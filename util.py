from typing import Tuple

import torch as th
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler,
)

def create_base_model_and_diffusion(
    timestep_respacing: str, _device: th.device
) -> Tuple[th.nn.Module, th.nn.Module, dict]:
    has_cuda = th.cuda.is_available()
    base_model_options = model_and_diffusion_defaults()
    base_model_options["use_fp16"] = has_cuda
    base_model_options["timestep_respacing"] = timestep_respacing

    base_model, base_diffusion = create_model_and_diffusion(**base_model_options)
    base_model.eval()
    if has_cuda:
        base_model.convert_to_fp16()
    base_model.to(_device)
    base_model.load_state_dict(load_checkpoint("base", _device))
    return base_model, base_diffusion, base_model_options


def prepare_base_model_kwargs(
    base_glide_model: th.nn.Module,
    glide_base_opts: dict,
    batch_size: int,
    prompt: str,
    pt_device: th.device,
) -> dict:
    """
    Prepare kwargs for base model inference. Requires model, prompt, and glide_base_opts to tokenize prompt.

    :param model: base GLIDE model.
    :param prompt: prompt to use for inference.
    :param glide_base_opts: options for base model.
    :param batch_size: batch size.
    :return: kwargs for base model inference containing tokenized prompt.
    """
    tokens = base_glide_model.tokenizer.encode(prompt)
    tokens, mask = base_glide_model.tokenizer.padded_tokens_and_mask(
        tokens, glide_base_opts["text_ctx"]
    )
    uncond_tokens, uncond_mask = base_glide_model.tokenizer.padded_tokens_and_mask(
        [], glide_base_opts["text_ctx"]
    )
    return dict(
        tokens=th.tensor(
            [tokens] * batch_size + [uncond_tokens] * batch_size, device=pt_device
        ),
        mask=th.tensor(
            [mask] * batch_size + [uncond_mask] * batch_size,
            dtype=th.bool,
            device=pt_device,
        ),
    )

def prepare_sr_model_kwargs(
    model_up: th.nn.Module,
    samples: th.Tensor,
    options_up: dict,
    batch_size: int,
    prompt: str,
    pt_device: th.device,
) -> dict:
    """
    Prepare kwargs for base model inference. Requires model, prompt, and glide_base_opts to tokenize prompt.

    :param model: base GLIDE model.
    :param prompt: prompt to use for inference.
    :param glide_base_opts: options for base model.
    :param batch_size: batch size.
    :return: kwargs for base model inference containing tokenized prompt.
    """
    if len(prompt) == 0:
        print(f"Prompt is empty, skipping upsampling")
        return None
    tokens = model_up.tokenizer.encode(prompt)
    tokens, mask = model_up.tokenizer.padded_tokens_and_mask(
        tokens, options_up["text_ctx"]
    )

    return dict(
        low_res=((samples + 1) * 127.5).round() / 127.5 - 1,
        tokens=th.tensor([tokens] * batch_size, device=pt_device),
        mask=th.tensor(
            [mask] * batch_size,
            dtype=th.bool,
            device=pt_device,
        ),
    )



def run_glide_text2im(
    model: th.nn.Module,
    diffusion: th.nn.Module,
    glide_base_opts: dict,
    prompt: str,
    batch_size: int,
    guidance_scale: float,
    base_x: int,
    base_y: int,
    _device: th.device,
    cond_fn: callable = None,
):
    """
    Run inference on base model and upsample model.

    :param model: base GLIDE model.
    :param diffusion: base GLIDE diffusion model.
    :param prompt: prompt to use for inference.
    :param batch_size: batch size.
    :param guidance_scale: guidance scale.
    :param base_x: base x.
    :param base_y: base y.
    :param _device: device to use.
    :return: upsampled image.
    """
    model_kwargs = prepare_base_model_kwargs(
        model, glide_base_opts, batch_size, prompt, _device
    )

    if len(prompt) == 0:
        return None

    def model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = th.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        eps = th.cat([half_eps, half_eps], dim=0)
        return th.cat([eps, rest], dim=1)

    # Sample from the base model.
    model.del_cache()
    full_batch_size = batch_size * 2
    samples = diffusion.p_sample_loop(
        model_fn,
        (full_batch_size, 3, base_y, base_x),
        device=_device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=cond_fn,
    )[:batch_size]
    model.del_cache()
    return samples


def create_sr_model_and_diffusion(
    timestep_respacing: str, _device: th.device
) -> Tuple[th.nn.Module, th.nn.Module, dict]:
    has_cuda = th.cuda.is_available()
    options_up = model_and_diffusion_defaults_upsampler()
    options_up["use_fp16"] = has_cuda
    options_up["timestep_respacing"] = timestep_respacing  # TODO
    model_up, diffusion_up = create_model_and_diffusion(**options_up)
    model_up.eval()
    if has_cuda:
        model_up.convert_to_fp16()
    model_up.to(_device)
    model_up.load_state_dict(load_checkpoint("upsample", _device))
    return model_up, diffusion_up, options_up


def run_glide_sr_text2im(
    model_up: th.nn.Module,
    diffusion_up: th.nn.Module,
    options_up: th.nn.Module,
    samples: th.Tensor,
    prompt: str,
    batch_size: int,
    upsample_temp: float = 0.997,
    _device: th.device = th.device("cpu"),
    sr_x: int = 256,
    sr_y: int = 256,
):
    # Sample from the base model.
    model_up.del_cache()
    up_shape = (batch_size, 3, sr_y, sr_x)
    model_kwargs = prepare_sr_model_kwargs(
        model_up, samples, options_up, batch_size, prompt, _device
    )
    up_samples = diffusion_up.ddim_sample_loop(
        model_up,
        up_shape,
        noise=th.randn(up_shape, device=_device) * upsample_temp,
        device=_device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
    )[:batch_size]
    model_up.del_cache()
    return up_samples