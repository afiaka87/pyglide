import os
import pathlib
import shutil
import torch
from glob import glob

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

KEEP_N_CHECKPOINTS = 15

# is_root = distr_backend.is_root_worker()
def deepspeed_save_glide(path, epoch, distributed_glide_model, glide_model, glide_params, opt, scheduler, is_root):
    save_obj = {
        'hparams': glide_params,
        'epoch': epoch,
    }
    cp_dir = cp_path_to_dir(path, 'ds')
    if KEEP_N_CHECKPOINTS is not None and is_root:
        checkpoints = sorted(glob(str(cp_dir / "global*")), key=os.path.getmtime, reverse=True)
        for checkpoint in checkpoints[KEEP_N_CHECKPOINTS:]:
            shutil.rmtree(checkpoint)

    distributed_glide_model.save_checkpoint(cp_dir, client_state=save_obj)
    if not is_root: return
    save_obj = { **save_obj,
        'weights': (
            'To get a working standard checkpoint, '
            'look into consolidating DeepSpeed checkpoints.'
        ),
    }
    output_dir = os.path.join(cp_dir, 'auxiliary.pt')
    torch.save(save_obj, output_dir)

    if not is_root:
        return

    save_obj = {
        **save_obj,
        'weights': glide_model.state_dict(),
        'opt_state': opt.state_dict(),
        'scheduler_state': (scheduler.state_dict() if scheduler else None)
    }

    torch.save(save_obj, path)