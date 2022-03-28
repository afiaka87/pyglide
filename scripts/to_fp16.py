import torch as th
import util

model, diffusion, options = util.init_model(
    model_path="coco_upsample_latest.pt",
    timestep_respacing="fast27",
    device="cpu",
    model_type="upsample",
)
model.eval()

model.convert_to_fp16()

th.save(model.state_dict(), "coco_upsample_latest_fp16.pt")