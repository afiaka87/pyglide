import glob
import shutil
import os
from typing import Tuple

import numpy as np
import PIL
import torch as th
import wandb
from tqdm import tqdm
import pathlib

KEEP_N_CHECKPOINTS = 20
DEEPSPEED_CP_AUX_FILENAME = 'auxiliary.pt'

def cp_path_to_dir(cp_path, tag):
    """Convert a checkpoint path to a directory with `tag` inserted.
    If `cp_path` is already a directory, return it unchanged.
    """
    if not isinstance(cp_path, pathlib.Path):
        cp_path = pathlib.Path(cp_path)
    if cp_path.is_dir():
        return cp_path
    path_sans_extension = cp_path.parent / cp_path.stem
    cp_dir = pathlib.Path(f'{path_sans_extension}-{tag}-cp')
    return cp_dir


### Checkpointing
@th.no_grad()
def save_model(model, path: str, is_root: bool, epoch=0, using_deepspeed=False, opt=None):
    if not path.endswith(".pt"):
        path = f"{path}-epoch-{epoch}.pt"
    save_obj = {'epoch': epoch, }
    if using_deepspeed:
        cp_dir = cp_path_to_dir(path, 'ds')
        if KEEP_N_CHECKPOINTS is not None and is_root:
            checkpoints = sorted(glob.glob(str(cp_dir / "global*")), key=os.path.getmtime, reverse=True)
            for checkpoint in checkpoints[KEEP_N_CHECKPOINTS:]:
                shutil.rmtree(checkpoint)

        model.save_checkpoint(cp_dir, client_state=save_obj)
        if not is_root: return
        # Save a nonsense value that directs the user to convert the checkpoint to a normal 32-bit pytorch model.
        save_obj = {
            **save_obj,
            'weights': (
                'To get a working standard checkpoint, '
                'look into consolidating DeepSpeed checkpoints.'
            ),
        }
        th.save(save_obj, str(cp_dir / DEEPSPEED_CP_AUX_FILENAME))
    if not is_root: return
    save_obj = { **save_obj, 'weights': model.state_dict(), }
    if opt is not None:
        save_obj = { **save_obj, 'opt_state': opt.state_dict(), }
    th.save(save_obj, path)


def pred_to_pil(pred: th.Tensor) -> PIL.Image:
    scaled = ((pred + 1) * 127.5).round().clamp(0, 255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([pred.shape[2], -1, 3])
    return PIL.Image.fromarray(reshaped.numpy())


def pil_image_to_norm_tensor(pil_image):
    """
    Convert a PIL image to a PyTorch tensor normalized to [-1, 1] with shape [B, C, H, W].
    """
    return th.from_numpy(np.asarray(pil_image)).float().permute(2, 0, 1) / 127.5 - 1.0


def resize_for_upsample(
    original, low_res_x, low_res_y, upscale_factor: int = 4
) -> Tuple[th.Tensor, th.Tensor]:
    """
    Resize/Crop an image to the size of the low resolution image. This is useful for upsampling.

    Args:
        original: A PIL.Image object to be cropped.
        low_res_x: The width of the low resolution image.
        low_res_y: The height of the low resolution image.
        upscale_factor: The factor by which to upsample the image.

    Returns:
        The downsampled image and the corresponding upscaled version cropped according to upscale_factor.
    """
    high_res_x, high_res_y = low_res_x * upscale_factor, low_res_y * upscale_factor
    high_res_image = original.resize((high_res_x, high_res_y), PIL.Image.LANCZOS)
    high_res_tensor = pil_image_to_norm_tensor(pil_image=high_res_image)
    low_res_image = high_res_image.resize(
        (low_res_x, low_res_y), resample=PIL.Image.BICUBIC
    )
    low_res_tensor = pil_image_to_norm_tensor(pil_image=low_res_image)
    return low_res_tensor, high_res_tensor


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def wandb_setup(
    batch_size: int,
    side_x: int,
    side_y: int,
    learning_rate: float,
    use_fp16: bool,
    device: str,
    data_dir: str,
    base_dir: str,
    project_name: str = "glide-text2im-finetune",
):
    return wandb.init(
        project=project_name,
        config={
            "batch_size": batch_size,
            "side_x": side_x,
            "side_y": side_y,
            "learning_rate": learning_rate,
            "use_fp16": use_fp16,
            "device": device,
            "data_dir": data_dir,
            "base_dir": base_dir,
        },
        sync_tensorboard=f"tensorboard_logs/{project_name}",
        tensorboard=True
    )
