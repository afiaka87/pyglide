# pyglide

<a href="https://replicate.com/afiaka87/pyglide" target="_blank"><img src="https://img.shields.io/static/v1?label=run&message=on replicate.ai&color=green"></a> (see [pyglide-replicate](https://github.com/afiaka87/pyglide-replicate))

> a lonely robot in the middle of the field
![](assets/lonelyrobot.png?raw=true)

---

## Usage
```sh
time python tgd.py --prompt "the beach at sunset"
Selected device: cuda:0.
1. Creating model and diffusion.
1. Done.
2. Running base GLIDE text2im model.
2. Base model generations complete. Check glide_outputs/base/the_beach_at_sunset/the_beach_at_sunset.png for generations.
3. Loading GLIDE upsampling diffusion model.
3. Done.
4. Running GLIDE upsampling from 64x64 to 256x256.
4. Done. Check glide_outputs/sr/the_beach_at_sunset/the_beach_at_sunset.png for generations.

real    1m4.775s
user    1m9.648s
sys     0m8.894s
```
![](assets/the_beach_at_sunset.png?raw=true)


## Installation

First clone this repository:
```sh
git clone https://github.com/afiaka87/pyglide.git
cd pyglide
```

You also need to install glide-text2im from openai's repository.
```sh
python3 -m venv .venv
source .venv/bin/activate
(.venv) python -m pip install -r requirements.txt
(.venv) git clone https://github.com/openai/glide-text2im.git
(.venv) cd glide-text2im/
(.venv) python -m pip install -e .
(.venv) cd ../
```

## Detailed Usage
```sh
usage: tgd.py [-h] --prompt PROMPT [--batch_size BATCH_SIZE] [--guidance_scale GUIDANCE_SCALE] [--base_x BASE_X] [--base_y BASE_Y] [--respace RESPACE] [--prefix PREFIX] [--upsample_temp UPSAMPLE_TEMP]

optional arguments:
  -h, --help            show this help message and exit
  --prompt PROMPT       a caption to visualize
  --batch_size BATCH_SIZE
  --guidance_scale GUIDANCE_SCALE
  --base_x BASE_X       width of base gen. has to be multiple of 16
  --base_y BASE_Y       width of base gen. has to be multiple of 16
  --respace RESPACE     Number of timesteps to use for generation. Lower is faster but less accurate.
  --prefix PREFIX       Output dir for generations. Will be created if it doesn't exist with subfolders for base and upsampled.
  --upsample_temp       0.0 to 1.0. 1.0 can introduce artifacts, lower can introduce blurriness.
```

## Gallery


> a lonely robot on hanging out on a cliff
![](assets/cliffbot.png?raw=true)
> an analog clock hanging on a blue wall
![an analog clock hanging on a blue wall](assets/harn.png?raw=true)
> a goose made of paper. paper goose.
![](assets/goose.png?raw=true)
> a goose rendered in minecraft. minecraft goose.
![](assets/goose-mc.png?raw=true)
