"""
Train an unconditional diffusion model for 3D structure generation.
"""

import argparse
import os

from topodiff import dist_util, logger
from topodiff.image_datasets_diffusion_model import load_data
from topodiff.resample import create_named_schedule_sampler
from topodiff.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from topodiff.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    logger.log(f"Model size: {total_params * 4 / 1024**2:.1f} MB (fp32)")
    logger.log(f"Input shape: [batch_size, 1, {args.image_size}, {args.image_size}, {args.image_size}]")
    logger.log(f"Output channels: {model.out_channels}")
    
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("Creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        deterministic=False,
        num_workers=args.num_workers,
    )

    # Test data loading
    logger.log("Testing data loading...")
    test_batch = next(data)
    logger.log(f"Loaded test batch: shape={test_batch.shape}, "
              f"range=[{test_batch.min():.3f}, {test_batch.max():.3f}], "
              f"mean={test_batch.mean():.3f}")

    logger.log("Starting training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        # Data settings
        data_dir="",
        num_workers=4,
        
        # Training settings
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=4,  # Reduced for 3D memory constraints
        microbatch=-1,  # -1 means no gradient accumulation
        ema_rate="0.9999",
        log_interval=100,
        save_interval=2000,
        resume_checkpoint="",
        
        # Model settings
        use_fp16=True,  # Recommended for 3D to save memory
        fp16_scale_growth=1e-3,
        
        # 3D specific settings
        image_size=64,
        num_channels=64,  # Reduced from default for memory efficiency
        num_res_blocks=2,
        attention_resolutions="16,8",  # Only use attention at lower resolutions
        channel_mult="1,2,3,4",
        
        # Diffusion settings
        learn_sigma=True,
        diffusion_steps=1000,
        noise_schedule="cosine",
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_checkpoint=True,  # Enable gradient checkpointing to save memory
    )
    defaults.update(model_and_diffusion_defaults())
    
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()