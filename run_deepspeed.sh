# CKPT_DIR='/mnt/afiaka87/checkpoints_latest/laion_ds_ckpt_7'
# /mnt/10TB_HDD_OLDER/LAION/laion400m-dat-release/' \
# RESUME_CKPT='/mnt/afiaka87/checkpoints_latest/laion_ds_ckpt_6/glide-ft-0x11000.pt'
# --resume_ckpt $RESUME_CKPT \
# --checkpoints_dir $CKPT_DIR \

/home/samsepiol/Projects/glide-finetune/.venv/bin/deepspeed train_glide.py \
    --side_x 32 \
    --side_y 32 \
    --freeze_diffusion \
    --ga_steps 1 \
    --batch_size 1 \
    --learning_rate 1e-5 \
    --weight_decay 0.001 \
    --resize_ratio 1.0 \
    --uncond_p 0.2 \
    --log_frequency 500 \
    --data_dir '~/datasets/current-dataset/POKE' \
    --project_name 'rtx2070_ds_glide' \
    --epochs 1 \
    --max_steps 100000 \
    --test_prompt '' \
    --test_batch_size 1 \
    --test_guidance_scale 0.0 \
    --seed '1' \
    --device 'cuda' \
    --cond_text 'POKEMON' \
    --deepspeed