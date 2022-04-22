from pathlib import Path
from braceexpand import braceexpand
import io
import json
from random import random

import PIL
import torch as th
import webdataset as wds
import torchvision.transforms.functional as TF

from glide_finetune.glide_util import (get_tokens_and_mask,
                                       get_uncond_tokens_mask)
from glide_finetune.train_util import pil_image_to_norm_tensor


def glide_wds_loader(
    urls,
    enable_text=True,
    enable_image=True,
    enable_metadata=True,
    image_key="jpg",
    caption_key="txt",
    metadata_key="json",
    cache_path="/opt/afiaka87/glide_finetune_wds_cache", # TODO
    tokenizer=None,
    base_x=64,
    base_y=64,
    uncond_p=0.2,
    enable_upsample=False,
    upscale_factor=4,
):

    base_image_shape = (base_x, base_y)
    upsample_image_shape = (int(base_x * upscale_factor), int(base_y * upscale_factor))
    dataset = wds.WebDataset(
        urls,
        cache_dir=cache_path,
        cache_size=10**10,
        handler=wds.handlers.warn_and_continue,
        nodesplitter=wds.split_by_worker,
    )

    def filter_dataset_laion(item):
        if enable_text and caption_key not in item:
            return False
        if enable_image and image_key not in item:
            return False
        if enable_metadata and metadata_key not in item:
            return False
        return True

    filtered_dataset = dataset.select(filter_dataset_laion)

    def preprocess_dataset(item):
        tokens, mask, base_tensor, upsample_tensor = None, None, None, None

        if not enable_text or random() < uncond_p: # Classifier free guidance
            tokens, mask = get_uncond_tokens_mask(tokenizer)
        else:
            caption = item[caption_key].decode("utf-8")
            tokens, mask = get_tokens_and_mask(tokenizer, caption)

        image_data = item[image_key]
        original_pil_image = PIL.Image.open(io.BytesIO(image_data))
        original_pil_image.load()

        base_pil_image = original_pil_image.resize(base_image_shape, resample=PIL.Image.LANCZOS).convert("RGB")
        base_tensor = pil_image_to_norm_tensor(base_pil_image)

        # The upsample model needs both the base and the upsample images e.g. 64x64 and 256x256.
        # if enable_upsample:
            # base_tensor = TF.gaussian_blur(base_tensor, (3, 3), 0.6) # blur the image first as in dalle2
            # upsample_pil_image = TF.resize(original_pil_image, upsample_image_shape, interpolation=PIL.Image.BICUBIC)
            # original_tensor = pil_image_to_norm_tensor(original_pil_image)
            # upsample_tensor = pil_image_to_norm_tensor(upsample_pil_image) # TODO 
            # return th.tensor(tokens), th.tensor(mask, dtype=th.bool), base_tensor, original_tensor
        # else:
        return th.tensor(tokens), th.tensor(mask, dtype=th.bool), base_tensor

    transformed_dataset = filtered_dataset.map(
        preprocess_dataset, handler=wds.handlers.warn_and_continue
    )
    return transformed_dataset
