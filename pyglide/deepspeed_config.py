def deepspeed_config_from_args(args):
    return {
        "zero_optimization": {
            "round_robin_gradients": True,
            "stage": 1,
            "offload_optimizer": {
                "device": "cpu",
                # "nvme_path": "./zero_nvme",
                "pin_memory": True,
                "buffer_count": 4,
                "fast_init": False
            },
            "offload_param": {
                "device": "cpu",
                "nvme_path": "./zero_nvme",
                "pin_memory": True,
                "buffer_count": 24,
                "buffer_size": 1e8,
                "max_in_cpu": 1e13
            },
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
            }
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 1e-6,
                "warmup_max_lr": args.learning_rate,
                "warmup_num_steps": 0,
                "total_num_steps": args.max_steps,
            }
        },
        'train_batch_size': args.batch_size,
        'gradient_accumulation_steps': args.ga_steps,
        'gradient_clipping': 1.0,
        'fp16': {
            'enabled': False, #args.use_fp16,
            'initial_scale_power': 28, # the default, often it's better to start lower around 16-24
        },
        # 'amp': {
        #     'enabled': True,
        #     "opt_level": "O1",
        # },
    }