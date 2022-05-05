import os
import sys

from tqdm import tqdm

sys.path.append("glide-text2im")
import typing
import time
import PIL
from glide_text2im.model_creation import create_gaussian_diffusion
from termcolor import cprint
import torch as th
import cog
import util
from torchvision.utils import make_grid
import torchvision.transforms as TF


class Predictor(cog.BasePredictor):
    def setup(self):
        cprint("Creating model and diffusion.", "white")
        device = th.device("cuda")
        self.model, _, self.options = util.init_model(
            model_path="pixel_glide_base_latest.pt",
            timestep_respacing="50", # we set this after
            device=device,
            model_type="base",
        )
        self.model.eval()
        self.model.convert_to_fp16()
        cprint("Done.", "green")

        cprint("Loading GLIDE upsampling diffusion model.", "white")
        self.model_up, _, self.options_up = util.init_model(
            model_path="pixel_glide_upsample_latest.pt",
            timestep_respacing="30",
            device=device,
            model_type="upsample",
        )
        self.model_up.eval()
        self.model_up.convert_to_fp16()
        cprint("Done.", "green")

    @th.inference_mode()
    @th.cuda.amp.autocast()
    def predict(
        self,
        prompt: str = cog.Input(
            description="Prompt to use.",
        ),
        enable_upsample: bool = cog.Input(
            description="(Recommended) Enable 4x prompt-aware upsampling. If disabled, only 64px model will be used. Disable if you just want the small generations from the base model for a speedup.",
            default=True,
        ),
        batch_size: int = cog.Input(default=4, description="Batch size.", choices=[1, 2, 3, 4, 6, 8, 12]),
        side_x: str = cog.Input(
            description="Must be multiple of 8. Going above 64 is not recommended. Actual image will be 4x larger.",
            choices=["32", "48", "64", "80", "96", "112", "128"],
            default="64",
        ),
        side_y: str = cog.Input(
            description="Must be multiple of 8. Going above 64 is not recommended. Actual image size will be 4x larger.",
            choices=["32", "48", "64", "80", "96", "112", "128"],
            default="64",
        ),
        guidance_scale: float = cog.Input(
            description="Classifier-free guidance scale. Higher values move further away from unconditional outputs. Lower values move closer to unconditional outputs. Negative values guide towards semantically opposite classes. 4-16 is a reasonable range.",
            default=6.0,
        ),
        timestep_respacing: str = cog.Input(
            description="Number of timesteps to use for base model PLMS sampling. Higher -> better quality, lengthier runs. Usually don't need more than 50.",
            choices=[
                "15",
                "17",
                "19",
                "21",
                "23",
                "25",
                "27",
                "30",
                "35",
                "40",
                "50",
                "60",
                "70",
                "80"
            ],
            default="40",
        ),
        sr_timestep_respacing: str = cog.Input(
            description="Number of timesteps to use for upsample model PLMS sampling. Usually don't need more than 20.",
            choices=["15", "17", "19", "21", "23", "25", "27", "30", "35"],
            default="27",
        ),
        seed: int = cog.Input(description="Seed for reproducibility", default=0),
    ) -> typing.Iterator[cog.Path]:
        prefix = "cog_predictions"
        os.makedirs(os.path.join(prefix, "base"), exist_ok=True)
        os.makedirs(os.path.join(prefix, "upsample"), exist_ok=True)
        th.manual_seed(seed)
        side_x, side_y = int(side_x), int(side_y)
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        
        images_per_row = batch_size
        if batch_size >= 6:
            images_per_row = batch_size // 2
        if batch_size >= 10:
            images_per_row = batch_size // 3

        # Override default `diffusion` helper class to use fewer timesteps via `timestep_respacing`
        # This is required because we initialize the model with a different number of timesteps to persist it for future runs.
        self.diffusion = create_gaussian_diffusion(
            steps=self.options["diffusion_steps"],
            noise_schedule=self.options["noise_schedule"],
            timestep_respacing=str(timestep_respacing),
        )
        # Override default `diffusion_up` helper class to use fewer timesteps via `sr_timestep_respacing`
        self.diffusion_up = create_gaussian_diffusion(
            steps=self.options_up["diffusion_steps"],
            noise_schedule=self.options_up["noise_schedule"],
            timestep_respacing=str(sr_timestep_respacing),
        )
        cprint(
            f"Running base GLIDE text2im model to generate {side_x}x{side_y} samples.",
            "white",
        )
        current_time = time.time()
        low_res_samples = util.run_glide_text2im(
            model=self.model,
            diffusion=self.diffusion,
            options=self.options,
            prompt=prompt,
            batch_size=batch_size,
            guidance_scale=guidance_scale,
            side_x=side_x,
            side_y=side_y,
            device=device,
            sample_method="plms",
        )

        for idx, current_tensor in enumerate(low_res_samples):
            current_grid = make_grid(current_tensor, nrow=images_per_row)
            current_pil_img = TF.ToPILImage()(current_grid)
            base_output_path = os.path.join(prefix, "base", f"{idx}.png")
            current_pil_img.save(base_output_path)
            yield cog.Path(f"{prefix}/base/{idx}.png")
    
        cprint(f"Done. Took {time.time() - current_time} seconds.", "green")

        if enable_upsample:
            sr_base_x = int(side_x * 4.0)
            sr_base_y = int(side_y * 4.0)
            cprint(f"SR base x: {sr_base_x}, SR base y: {sr_base_y}", "white")
            cprint(
                f"Upsampling from {side_x}x{side_y} to {sr_base_x}x{sr_base_y}.",
                "white",
            )
            current_time = time.time()
            hi_res_samples = util.run_glide_text2im(
                model=self.model_up,
                diffusion=self.diffusion_up,
                options=self.options_up,
                prompt=prompt,
                batch_size=batch_size,
                side_x=sr_base_x,
                side_y=sr_base_y,
                device=device,
                cond_fn=None,
                guidance_scale=guidance_scale,
                sample_method="plms",
                images_to_upsample=current_tensor.to(device),
            )
            for idx, current_up_tensor in enumerate(hi_res_samples):
                current_up_grid = make_grid(current_up_tensor, nrow=images_per_row)
                current_up_pil_img = TF.ToPILImage()(current_up_grid)
                up_output_path = os.path.join(prefix, "upsample", f"{idx}.png")
                current_up_pil_img.save(up_output_path)
                yield cog.Path(f"{prefix}/upsample/{idx}.png")
            cprint(f"Done. Took {time.time() - current_time} seconds.", "green")
