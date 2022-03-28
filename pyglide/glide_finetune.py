from torch.cuda.amp import autocast
import os
import pathlib
from typing import Tuple

import torch as th
from glide_finetune.deepspeed_util import deepspeed_save_glide
from glide_text2im.respace import SpacedDiffusion
from glide_text2im.text2im_model import Text2ImUNet
from tqdm import tqdm
from wandb import wandb

from glide_finetune import glide_util, train_util

def exists(val):
    return val is not None

def ds_save_model(path, model_params,  distributed_glide_model=None, distr_backend=None): # TODO - deepspeed doesnt save the hyperparams in its checkpoints, consider warning user about this.
    save_obj = { 'hparams': model_params, }
    if distributed_glide_model is not None:
        cp_path = pathlib.Path(path)
        path_sans_extension = cp_path.parent / cp_path.stem
        cp_dir = str(path_sans_extension) + '_ds_cp'
        distributed_glide_model.save_checkpoint(cp_dir, client_state=save_obj)

    if not distr_backend.is_root_worker():
        return
    save_obj = {**save_obj, 'weights': distributed_glide_model.state_dict()}
    th.save(save_obj, path)

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
    tokens, masks, reals = [x.to(device) for x in batch]
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
    return th.nn.functional.mse_loss(epsilon, noise.to(device).detach())

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
    tokens, masks, low_res_image, high_res_image = [ x.to(device) for x in batch ]

    timesteps = th.randint(0, len(glide_diffusion.betas) - 1, (low_res_image.shape[0],), device=device)
    noise = th.randn_like(high_res_image, device=device) # Noise should be shape of output i think
    noised_high_res_image = glide_diffusion.q_sample(high_res_image, timesteps, noise=noise).to(device)
    _, C = noised_high_res_image.shape[:2]
    model_output = glide_model(
        noised_high_res_image.to(device),
        timesteps.to(device),
        low_res=low_res_image.to(device),
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
    distributed_glide_model=None,
    distributed_dataloader=None,
    distr_backend=None,
    is_root=False,
    cond_text='',
):
    if len(cond_text) > 0 and is_root:
        cond_text = cond_text.split(' ')[0]
        print(f'Using <|{cond_text}|> as prompt')
        cond_text = f'<|{cond_text.lower().strip()}|>'
    if train_upsample: train_step = upsample_train_step
    else: train_step = base_train_step

    os.makedirs(checkpoints_dir, exist_ok=True)
    if distributed_glide_model is None:
        glide_model.to(device)
        glide_model.train()
    # else:
    # distributed_glide_model.load_checkpoint(glide_path)
    log = {}
    data_loop = dataloader
    if distributed_dataloader is not None:
        data_loop = distributed_dataloader

    if is_root: print(f"Starting epoch {epoch}")
    for train_idx, batch in enumerate(data_loop):
        accumulated_loss = train_step(
            glide_model=distributed_glide_model,
            glide_diffusion=glide_diffusion,
            batch=batch,
            device=device,
        )
        if distributed_glide_model is not None:
            distributed_glide_model.backward(accumulated_loss)
            distributed_glide_model.step()
        else:
            accumulated_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        accumulated_loss = distr_backend.average_all(accumulated_loss)
        if is_root:
            log = {**log, "iter": train_idx, "loss": accumulated_loss.item()}
            tqdm.write(f"loss: {accumulated_loss.item():.4f}")
        # Sample from the model
        if train_idx > 0 and train_idx % log_frequency == 0 and is_root:
            tqdm.write(f"Sampling from model at iteration {train_idx}")
            samples = glide_util.sample(
                glide_model=glide_model,
                glide_options=glide_options,
                side_x=side_x,
                side_y=side_y,
                prompt=prompt,
                batch_size=sample_bs,
                guidance_scale=sample_gs,
                device=device,
                prediction_respacing=sample_respacing,
                upsample_factor=upsample_factor,
                image_to_upsample=image_to_upsample,
                cond_text=cond_text,
            )
            sample_save_path = os.path.join(outputs_dir, f"{train_idx}.png")
            train_util.pred_to_pil(samples).save(sample_save_path)
            if exists(wandb_run):
                wandb_run.log(
                    {
                        **log,
                        "iter": train_idx,
                        "samples": wandb.Image(sample_save_path, caption=prompt),
                    }
                )
            tqdm.write(f"Saved sample {sample_save_path}")
        if train_idx % 2500 == 0 and train_idx > 0 and is_root:
            train_util.save_model(glide_model, checkpoints_dir, train_idx, epoch)
            tqdm.write(f"Saved checkpoint {train_idx} to {checkpoints_dir}/glide-ft-{train_idx}.pt")
            # if distributed_glide_model is not None:
            #     glide_params = [p for p in glide_model.parameters() if p.requires_grad]
            #     deepspeed_save_glide(path=checkpoints_dir, epoch=epoch, distributed_glide_model=distributed_glide_model, glide_model=glide_model, glide_params=glide_params, opt=optimizer, scheduler=None, is_root=True, deepspeed_congig=deepspeed_config)

        if exists(wandb_run) and is_root: wandb_run.log(log)
    if is_root:
        tqdm.write(f"Finished training, saving final checkpoint")
        train_util.save_model(glide_model, checkpoints_dir, train_idx, epoch)

print(f"fin.")