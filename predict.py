from email.policy import default
import pathlib
import time
from termcolor import cprint
import torch as th
import cog
import util

import sys

sys.path.append("glide-text2im")


class Predictor(cog.BasePredictor):
    def setup(self):
        pass

    def predict(
        self,
        prompt: str = cog.Input(
            description="Prompt to use.",
        ),
        style_prompt: str = cog.Input(
            description="Additional style guidance to use. Handles any sequence of tokens, but works particularly well on the listed pretrained 'dataset tokens'.",
            default="<pixelart>",
            choices=["<pixelart>", "<cc12m>", "<pokemon>", "<country211>", "<pixelart>", "<openimages>", "<ffhq>", "<coco>", "<vaporwave>", "<virtualgenome>", "<imagenet>"]
        ),
        batch_size: int = cog.Input(
            description="Batch size. Number of generations to predict", ge=1, le=8, default=1
        ),
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
        upsample_stage: bool = cog.Input(
            description="If true, uses both the base and upsample models. If false, only the (finetuned) base model is used.",
            default=True,
        ),
        upsample_temp: str = cog.Input(
            description="Upsample temperature. Consider lowering to ~0.997 for blurry images with fewer artifacts.",
            choices=["0.996", "0.997", "0.998", "0.999", "1.0"],
            default="0.997",
        ),
        guidance_scale: float = cog.Input(
            description="Classifier-free guidance scale. Higher values move further away from unconditional outputs. Lower values move closer to unconditional outputs. Negative values guide towards semantically opposite classes. 4-16 is a reasonable range.",
            default=4,
        ),
        style_guidance_scale: float = cog.Input(
            description="Same as guidance scale, but applied to glide model outputs from the style prompt instead of the prompt.",
            default=4,
        ),
        timestep_respacing: str = cog.Input(
            description="Number of timesteps to use for base model PLMS sampling. Usually don't need more than 50.",
            choices=[ "15", "17", "19", "21", "23", "25", "27", "30", "35", "40", "50", "100"],
            default="27",
        ),
        sr_timestep_respacing: str = cog.Input(
            description="Number of timesteps to use for upsample model PLMS sampling. Usually don't need more than 20.",
            choices=["15", "17", "19", "21", "23", "25", "27"],
            default="17",
        ),
        seed: int = cog.Input(description="Seed for reproducibility", default=0),
    ) -> cog.Path:
        th.manual_seed(seed)
        side_x, side_y, upsample_temp = int(side_x), int(side_y), float(upsample_temp)
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        cprint("Creating model and diffusion.", "white")
        model, diffusion, options = util.init_model(
            model_path="glide-ft-4x41618-fp16.pt",
            timestep_respacing=timestep_respacing,
            device=device,
            model_type="base",
        )
        model.eval()
        cprint("Done.", "green")

        cprint("Loading GLIDE upsampling diffusion model.", "white")
        model_up, diffusion_up, options_up = util.init_model(
            model_path="coco_upsample_latest_fp16.pt",
            timestep_respacing=sr_timestep_respacing,
            device=device,
            model_type="upsample",
        )
        model_up.eval()
        cprint("Done.", "green")

        cprint(
            f"Running base GLIDE text2im model to generate {side_x}x{side_y} samples.",
            "white",
        )
        current_time = time.time()
        low_res_samples = util.run_glide_text2im(
            model=model,
            diffusion=diffusion,
            options=options,
            prompt=prompt,
            batch_size=batch_size,
            guidance_scale=guidance_scale,
            side_x=side_x,
            side_y=side_y,
            device=device,
            sample_method="plms",
            style_prompt=style_prompt,
            cls_guidance_scale=style_guidance_scale,
        )

        elapsed_time = time.time() - current_time
        cprint(f"Base inference time: {elapsed_time} seconds.", "green")

        low_res_pil_images = util.pred_to_pil(low_res_samples)
        low_res_pil_images.save("/src/base_predictions.png")

        sr_base_x = int(side_x * 4.0)
        sr_base_y = int(side_y * 4.0)

        if upsample_stage:
            cprint(
                f"Upsampling from {side_x}x{side_y} to {sr_base_x}x{sr_base_y}.",
                "white",
            )
            current_time = time.time()
            hi_res_samples = util.run_glide_text2im(
                model=model_up,
                diffusion=diffusion_up,
                options=options_up,
                prompt=prompt,
                batch_size=batch_size,
                device=device,
                upsample_temp=upsample_temp,
                side_x=sr_base_x,
                side_y=sr_base_y,
                sample_method="plms",
                input_images=low_res_samples.to(device),
            )
            elapsed_time = time.time() - current_time
            cprint(f"SR Elapsed time: {elapsed_time} seconds.", "green")

            hi_res_pil_images = util.pred_to_pil(hi_res_samples)
            hi_res_pil_images.save("/src/sr_predictions.png")
            return cog.Path("/src/sr_predictions.png")
        return cog.Path("/src/base_predictions.png")