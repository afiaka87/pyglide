import argparse
from glob import glob
import os
import pathlib

import numpy as np
import torch as th
import torchvision.transforms as T
from tqdm import trange
from deepspeed_config import deepspeed_config_from_args

from glide_finetune.glide_finetune import run_glide_finetune_epoch
from glide_finetune.glide_util import load_model
from glide_finetune.loader import TextImageDataset
from glide_finetune.train_util import wandb_setup
from glide_finetune.wds_loader import glide_wds_loader

from dalle_pytorch import distributed_utils

def run_glide_finetune(
    data_dir="./data",
    batch_size=1,
    learning_rate=1e-5,
    adam_weight_decay=0.0,
    side_x=64,
    side_y=64,
    resize_ratio=1.0,
    uncond_p=0.0,
    resume_ckpt="",
    checkpoints_dir="./finetune_checkpoints",
    use_fp16=False,  # Tends to cause issues,not sure why as the paper states fp16 is stable.
    device="cpu",
    freeze_transformer=False,
    freeze_diffusion=False,
    project_name="glide_finetune",
    activation_checkpointing=False,
    use_captions=True,
    num_epochs=100,
    log_frequency=100,
    test_prompt="a group of skiers are preparing to ski down a mountain.",
    sample_bs=1,
    sample_gs=8.0,
    use_webdataset=False,
    image_key="jpg",
    caption_key="txt",
    enable_upsample=False,
    upsample_factor=4,
    image_to_upsample='low_res_face.png',
    deepspeed_config=None,
    cond_text='',
):
    if "~" in data_dir: data_dir = os.path.expanduser(data_dir)
    if "~" in checkpoints_dir: checkpoints_dir = os.path.expanduser(checkpoints_dir)

    using_deepspeed = True # TODO: remove this
    distr_backend = distributed_utils.set_backend_from_args(args)
    distr_backend.initialize()
    # is_root = (args.local_rank == 0)
    is_root = distr_backend.is_local_root_worker()


    # Start wandb logging
    if is_root:
        # Create the checkpoint/output directories
        os.makedirs(checkpoints_dir, exist_ok=True)
        wandb_run = wandb_setup(
            batch_size=batch_size,
            side_x=side_x,
            side_y=side_y,
            learning_rate=learning_rate,
            use_fp16=use_fp16,
            device=device,
            data_dir=data_dir,
            base_dir=checkpoints_dir,
            project_name=project_name,
        )
        print("Wandb setup.")
    else:
        wandb_run = None

    # Model setup
    glide_model, glide_diffusion, glide_options = load_model(
        glide_path=resume_ckpt,
        use_fp16=use_fp16,
        freeze_transformer=freeze_transformer,
        freeze_diffusion=freeze_diffusion,
        activation_checkpointing=activation_checkpointing,
        model_type="base" if not enable_upsample else "upsample",
    )
    if not using_deepspeed: glide_model.train()
    if is_root:
        number_of_params = sum(x.numel() for x in glide_model.parameters())
        print(f"Number of parameters: {number_of_params}")
        number_of_trainable_params = sum(
            x.numel() for x in glide_model.parameters() if x.requires_grad
        )
        print(f"Trainable parameters: {number_of_trainable_params}")

    # Data setup
    if is_root: print("Loading data...")
    if use_webdataset:
        # urls = "pipe:aws s3 cp s3://laion-us-east-1/laion-data/laion2B-data/{020114..231348}.tar -"
        # -data '/mnt/10TB_HDD_OLDER/LAION/laion400m-dat-release/' \
        dataset = glide_wds_loader(
            urls=data_dir,
            caption_key=caption_key,
            image_key=image_key,
            enable_image=True,
            enable_text=use_captions,
            enable_upsample=enable_upsample,
            tokenizer=glide_model.tokenizer,
            ar_lower=0.9,
            ar_upper=1.1,
            min_original_height=side_x * upsample_factor,
            min_original_width=side_y * upsample_factor,
            upscale_factor=upsample_factor,
            nsfw_filter=True,
            similarity_threshold_lower=0.3,
            similarity_threshold_upper=0.42,
            words_to_skip=[],
            dataset_name="laion",  # can be laion, alamy.
            cond_text=cond_text,
        )
    else:
        dataset = TextImageDataset(
            folder=data_dir,
            side_x=side_x,
            side_y=side_y,
            resize_ratio=resize_ratio,
            uncond_p=uncond_p,
            shuffle=True,
            tokenizer=glide_model.tokenizer,
            text_ctx_len=glide_options["text_ctx"],
            use_captions=use_captions,
            enable_glide_upsample=enable_upsample,
            upscale_factor=upsample_factor,  # TODO: make this a parameter
            cond_text=cond_text,
        )

    # Data loader setup
    dataloader = th.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not use_webdataset,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    if deepspeed_config is not None:
        training_data = None
        if using_deepspeed:
            if use_webdataset: training_data = None
            else: training_data = dataset
        else:
            training_data = dataloader
        (distributed_glide_model, distributed_optimizer, distributed_dataloader, distributed_scheduler) = distr_backend.distribute(
            args=args,
            model=glide_model,
            optimizer=None, #optimizer,
            model_parameters=[x for x in glide_model.parameters() if x.requires_grad],
            training_data=training_data,
            lr_scheduler=None,
            config_params=deepspeed_config,
        )
    # Optimizer setup
    # optimizer = th.optim.AdamW(
    #     [x for x in glide_model.parameters() if x.requires_grad],
    #     lr=learning_rate,
    #     weight_decay=adam_weight_decay,
    # )

    # Training setup
    outputs_dir = "./outputs"
    if is_root: 
        os.makedirs(outputs_dir, exist_ok=True)
    for epoch in trange(num_epochs):
        if is_root: print(f"Starting epoch {epoch}")
        run_glide_finetune_epoch(
            glide_model=glide_model,
            glide_diffusion=glide_diffusion,
            glide_options=glide_options,
            distributed_glide_model=distributed_glide_model,
            optimizer=distributed_optimizer,
            dataloader=dataloader,
            prompt=test_prompt,
            sample_bs=sample_bs,
            sample_gs=sample_gs,
            checkpoints_dir=checkpoints_dir,
            outputs_dir=outputs_dir,
            side_x=side_x,
            side_y=side_y,
            device=device,
            wandb_run=wandb_run,
            log_frequency=log_frequency,
            epoch=epoch,
            train_upsample=enable_upsample,
            distributed_dataloader=distributed_dataloader,
            is_root=is_root,
            distr_backend=distr_backend,
            cond_text=cond_text,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-data", type=str, default="./data")
    parser.add_argument("--batch_size", "-bs", type=int, default=1)
    parser.add_argument("--learning_rate", "-lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", "-adam_wd", type=float, default=0.0)
    parser.add_argument("--side_x", "-x", type=int, default=64)
    parser.add_argument("--side_y", "-y", type=int, default=64)
    parser.add_argument(
        "--resize_ratio", "-crop", type=float, default=0.8, help="Crop ratio"
    )
    parser.add_argument(
        "--uncond_p",
        "-p",
        type=float,
        default=0.2,
        help="Probability of using the empty/unconditional token instead of a caption. OpenAI used 0.2 for their finetune.",
    )
    parser.add_argument(
        "--train_upsample",
        "-upsample",
        action="store_true",
        help="Train the upsampling type of the model instead of the base model.",
    )
    parser.add_argument(
        "--resume_ckpt",
        "-resume",
        type=str,
        default="",
        help="Checkpoint to resume from",
    )
    parser.add_argument(
        "--checkpoints_dir", "-ckpt", type=str, default="./glide_checkpoints/"
    )
    parser.add_argument("--use_fp16", "-fp16", action="store_true")
    parser.add_argument("--device", "-dev", type=str, default="")
    parser.add_argument("--log_frequency", "-freq", type=int, default=100)
    parser.add_argument("--freeze_transformer", "-fz_xt", action="store_true")
    parser.add_argument("--freeze_diffusion", "-fz_unet", action="store_true")
    parser.add_argument("--project_name", "-name", type=str, default="glide-finetune")
    parser.add_argument("--activation_checkpointing", "-grad_ckpt", action="store_true")
    parser.add_argument("--use_captions", "-txt", action="store_true")
    parser.add_argument("--epochs", "-epochs", type=int, default=20)
    parser.add_argument(
        "--test_prompt",
        "-prompt",
        type=str,
        default="a group of skiers are preparing to ski down a mountain.",
    )
    parser.add_argument(
        "--test_batch_size",
        "-tbs",
        type=int,
        default=1,
        help="Batch size used for model eval, not training.",
    )
    parser.add_argument(
        "--test_guidance_scale",
        "-tgs",
        type=float,
        default=1.0,
        help="Guidance scale used during model eval, not training.",
    )
    parser.add_argument(
        "--use_webdataset",
        "-wds",
        action="store_true",
        help="Enables webdataset (tar) loading",
    )
    parser.add_argument(
        "--wds_image_key",
        "-wds_img",
        type=str,
        default="jpg",
        help="A 'key' e.g. 'jpg' used to access the image in the webdataset",
    )
    parser.add_argument(
        "--wds_caption_key",
        "-wds_cap",
        type=str,
        default="txt",
        help="A 'key' e.g. 'txt' used to access the caption in the webdataset",
    )
    parser.add_argument(
        "--wds_dataset_name",
        "-wds_name",
        type=str,
        default="laion",
        help="Name of the webdataset to use (laion or alamy)",
    )
    parser.add_argument("--seed", "-seed", type=int, default=0)
    parser.add_argument(
        "--cudnn_benchmark",
        "-cudnn",
        action="store_true",
        help="Enable cudnn benchmarking. May improve performance. (may not)",
    )
    parser.add_argument(
        "--upscale_factor", "-upscale", type=int, default=4, help="Upscale factor for training the upsampling model only"
    )
    parser.add_argument("--image_to_upsample", "-lowres", type=str, default="low_res_face.png")
    parser.add_argument("--deepspeed", "-deepspeed", action="store_true")
    parser.add_argument("--max_steps", "-max_steps", type=int, default=0, help="Max steps for deepspeed")
    parser.add_argument("--local_rank", "-local_rank", type=int, default=0)
    parser.add_argument("--ga_steps", "-ga_steps", type=int, default=0, help="GA steps")
    parser.add_argument("--cond_text", "-cond_text", type=str, default="")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # CUDA/CPU setup
    args = parse_args()
    if len(args.device) > 0: device = th.device(args.device)
    else: device = th.device("cpu") if not th.cuda.is_available() else th.device("cuda")

    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    th.backends.cudnn.benchmark = args.cudnn_benchmark

    for arg in vars(args):
        print(f"--{arg} {getattr(args, arg)}")

    if args.use_webdataset: data_dir = glob(os.path.join(args.data_dir, "*.tar"))
    else: data_dir = args.data_dir
    
    deespeed_config = deepspeed_config_from_args(args)
    run_glide_finetune(
        data_dir=data_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        adam_weight_decay=args.weight_decay,
        side_x=args.side_x,
        side_y=args.side_y,
        resize_ratio=args.resize_ratio,
        uncond_p=args.uncond_p,
        resume_ckpt=args.resume_ckpt,
        checkpoints_dir=args.checkpoints_dir,
        use_fp16=args.use_fp16,
        device=device,
        log_frequency=args.log_frequency,
        freeze_transformer=args.freeze_transformer,
        freeze_diffusion=args.freeze_diffusion,
        project_name=args.project_name,
        activation_checkpointing=args.activation_checkpointing,
        use_captions=args.use_captions,
        num_epochs=args.epochs,
        test_prompt=args.test_prompt,
        sample_bs=args.test_batch_size,
        sample_gs=args.test_guidance_scale,
        use_webdataset=args.use_webdataset,
        image_key=args.wds_image_key,
        caption_key=args.wds_caption_key,
        enable_upsample=args.train_upsample,
        upsample_factor=args.upscale_factor,
        image_to_upsample=args.image_to_upsample,
        deepspeed_config=deespeed_config,
        cond_text=args.cond_text,
    )
