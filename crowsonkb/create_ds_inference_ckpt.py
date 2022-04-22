#%%
from pathlib import Path
from collections import OrderedDict
import torch



model_shard = torch.load("/opt/afiaka87/FINETUNE_GLIDE_XL-HUMANS-logs-run2-ds-cp/global_step6569/mp_rank_00_model_states.pt", map_location='cpu')
input_dir = Path('/opt/afiaka87/FINETUNE_GLIDE_XL-HUMANS-logs-run2-ds-cp/global_step6569')
shard_paths = list(input_dir.glob('zero_pp_rank_*_mp_rank_00_optim_states.pt'))
names, names_iter = None, None
# buffer_names = []

state = OrderedDict()

for shard_path in shard_paths:
    shard = torch.load(shard_path, map_location='cpu')
    names = model_shard['param_shapes']
    names_iter = iter(names)
    for param in shard['optimizer_state_dict']['base_optimizer_state']['state']:
        name = next(names_iter)
        print(name)
        if param['exp_avg_sq'].sum() == 0:
            continue
        ema_param_tensor = param["param_exp_avg"]
        state[name] = ema_param_tensor
            
torch.save(state, '/opt/afiaka87/ds_inference_ckpt.pt')
print(f"Saved to {'/opt/afiaka87/ds_inference_ckpt.pt'}")

# #%%
# # print(len(shard_paths))
# # for shard_path in shard_paths:
# #     shard = torch.load(shard_path, map_location='cpu')
# #     assert len(shard['optimizer_state_dict']['base_optimizer_state']['param_groups']) == 1
# #     # if names is None:
# #     names_iter = iter(names)
# #     print(model_shard["buffer_names"])
# #     # buffer_names.extend(model_shard['buffer_names'])
# #     for param in shard['optimizer_state_dict']['base_optimizer_state']['state'].values():
# #         name = next(names_iter)
# #         if param['exp_avg_sq'].sum() == 0:
# #             continue
# #         print(shard_path, name, param['param_exp_avg'].shape)
# #         state[name] = param['param_exp_avg']
# # del shard

# # # for buffer_name in buffer_names:
# # #     num, _, name = buffer_name.partition('.')
# # #     shard = torch.load(input_dir / f'layer_{int(num):02}-model_states.pt', map_location='cpu')
# # #     state[buffer_name] = shard[name]
# # # del shard

# # # torch.save(state, "ds_inference_ckpt.pt")

# # # %%
# # # mp_rank_00_model_states.pt
# %%
