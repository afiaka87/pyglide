import os
import pathlib
from typing import Tuple

import torch as th
from torch.nn.functional import interpolate
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import wandb
from glide_finetune import glide_util, train_util
from glide_text2im.respace import SpacedDiffusion
from glide_text2im.text2im_model import Text2ImUNet
from tqdm import tqdm



def exists(val):
    return val is not None

def base_train_step(
    glide_model: Text2ImUNet,
    glide_diffusion: SpacedDiffusion,
    batch: Tuple[th.Tensor, th.Tensor, th.Tensor],
    device: str,
):
    """
    Perform a single training step.

        Args:
            glide_model: The model to train.
            glide_diffusion: The diffusion to use.
            batch: A tuple of (tokens, masks, reals) where tokens is a tensor of shape (batch_size, seq_len), masks is a tensor of shape (batch_size, seq_len) and reals is a tensor of shape (batch_size, 3, side_x, side_y) normalized to [-1, 1].
            device: The device to use for getting model outputs and computing loss.
        Returns:
            The loss.
    """
    tokens, masks, reals = [x.detach().clone().to(device) for x in batch]
    reals.requires_grad_(True)
    timesteps = th.randint(0, len(glide_diffusion.betas) - 1, (reals.shape[0],), device=device)
    noise = th.randn_like(reals, device=device)
    x_t = glide_diffusion.q_sample(reals, timesteps, noise=noise).to(device)

    _, C = x_t.shape[:2]
    model_output = glide_model(
        x_t.to(device),
        timesteps.to(device),
        tokens=tokens.to(device),
        mask=masks.to(device),
    )
    epsilon, _ = th.split(model_output, C, dim=1)
    return th.nn.functional.mse_loss(epsilon, noise.to(device))

def upsample_train_step(
    glide_model: Text2ImUNet,
    glide_diffusion: SpacedDiffusion,
    batch: Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor],
    device: str,
):
    """
    Perform a single training step.

        Args:
            glide_model: The model to train.
            glide_diffusion: The diffusion to use.
            batch: A tuple of (tokens, masks, low_res, high_res) where 
                - tokens is a tensor of shape (batch_size, seq_len), 
                - masks is a tensor of shape (batch_size, seq_len) with dtype torch.bool
                - low_res is a tensor of shape (batch_size, 3, base_x, base_y), normalized to [-1, 1]
                - high_res is a tensor of shape (batch_size, 3, base_x*4, base_y*4), normalized to [-1, 1]
            device: The device to use for getting model outputs and computing loss.
        Returns:
            The loss.
    """
    tokens, masks, low_res_image = [x.detach().clone().to(device) for x in batch]
    upsampled_image = interpolate(low_res_image, scale_factor=4, mode='bicubic', align_corners=False).requires_grad_(True)
    timesteps = th.randint(0, len(glide_diffusion.betas) - 1, (low_res_image.shape[0],), device=device)
    noise = th.randn_like(upsampled_image, device=device)
    x_t = glide_diffusion.q_sample(upsampled_image, timesteps, noise=noise).to(device)
    _, C = x_t.shape[:2]
    blurry_low_res_image = TF.gaussian_blur(low_res_image, kernel_size=3, sigma=0.6)
    model_output = glide_model(
        x_t.to(device),
        timesteps.to(device),
        low_res=blurry_low_res_image.to(device),
        tokens=tokens.to(device),
        mask=masks.to(device))
    epsilon, _ = th.split(model_output, C, dim=1)
    return th.nn.functional.mse_loss(epsilon, noise.to(device).detach())


def run_glide_finetune_epoch(
    glide_model: Text2ImUNet,
    glide_diffusion: SpacedDiffusion,
    glide_options: dict,
    dataloader, 
    optimizer: th.optim.Optimizer,
    sample_bs: int,  # batch size for inference
    sample_gs: float = 4.0,  # guidance scale for inference
    sample_respacing: str = '100', # respacing for inference
    prompt: str = "",  # prompt for inference, not training
    side_x: int = 64,
    side_y: int = 64,
    outputs_dir: str = "./outputs",
    checkpoints_dir: str = "./finetune_checkpoints",
    device: str = "cpu",
    log_frequency: int = 100,
    wandb_run=None,
    epoch: int = 0,
    train_upsample: bool = False,
    upsample_factor=4,
    image_to_upsample='low_res_face.png',
    distr_backend=None,
    is_root=False,
    use_fp16=False,
):
    if train_upsample: train_step = upsample_train_step
    else: train_step = base_train_step

    os.makedirs(checkpoints_dir, exist_ok=True)
    log = {}
    if is_root: print(f"Starting epoch {epoch}")
    for train_idx, batch in enumerate(dataloader):
        log = {}
        with th.cuda.amp.autocast(enabled=use_fp16):
            accumulated_loss = train_step(
                glide_model=glide_model,
                glide_diffusion=glide_diffusion,
                batch=batch,
                device=device,
            )
        if glide_model is not None:
            glide_model.backward(accumulated_loss)
            glide_model.step()
        else:
            accumulated_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        accumulated_loss = distr_backend.average_all(accumulated_loss)
        if is_root:
            log = {**log, "iter": train_idx, "loss": accumulated_loss.item()}
            tqdm.write(f"loss: {accumulated_loss.item():.4f}")
        # Sample from the model
        if train_idx % log_frequency == 0 and is_root:
            tqdm.write(f"Sampling from model at iteration {train_idx}")
            with th.cuda.amp.autocast(enabled=use_fp16):
                samples, _caption = glide_util.sample(
                    glide_model=glide_model.module if distr_backend is not None else glide_model,
                    glide_options=glide_options,
                    side_x=side_x,
                    side_y=side_y,
                    prompt=prompt,
                    batch_size=sample_bs,
                    guidance_scale=sample_gs,
                    device=device,
                    prediction_respacing=sample_respacing,
                    upsample_enabled=train_upsample,
                    upsample_factor=upsample_factor,
                    image_to_upsample=image_to_upsample,
                )
            sample_save_path = os.path.join(outputs_dir, f"{train_idx}.png")
            train_util.pred_to_pil(samples).save(sample_save_path)
            if exists(wandb_run):
                wandb_run.log(
                    {
                        **log,
                        "iter": train_idx,
                        "samples": wandb.Image(sample_save_path, caption=_caption),
                    }
                )
            tqdm.write(f"Saved sample {sample_save_path}")
        if train_idx % 1000 == 0:
            using_deepspeed = (distr_backend is not None)
            train_util.save_model(glide_model, checkpoints_dir, is_root, epoch=epoch, using_deepspeed=using_deepspeed, opt=optimizer)
            tqdm.write(f"Saved checkpoint {train_idx} to {checkpoints_dir}/glide-ft-{train_idx}.pt")

        if exists(wandb_run) and is_root: wandb_run.log(log)
    if is_root:
        tqdm.write(f"Finished training, saving final checkpoint")
        train_util.save_model(glide_model, checkpoints_dir, train_idx, epoch)