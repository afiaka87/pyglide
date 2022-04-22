def deepspeed_config_from_args(args):
    deepspeed_config = {
        "zero_optimization": { 
            "stage": 1,
            "ignore_unused_parameters": True
        },
        'train_micro_batch_size_per_gpu': args.batch_size,
        'gradient_accumulation_steps': args.ga_steps,
        'gradient_clipping': 1.0,
        'fp16': {
            'enabled': args.use_fp16,
            'initial_scale_power': 20,
        },
        "tensorboard": {
            "enabled": True,
            "output_path": f"tensorboard_logs/{args.project_name}",
            "job_name": f"{args.project_name}",
        },
        "steps_per_print": 10, 
        "wall_clock_breakdown": False,
        "zero_allow_untested_optimizer": True
    }
    return deepspeed_config


