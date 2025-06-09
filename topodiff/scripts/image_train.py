"""
Train a conditional diffusion model for 3D TPMS structure generation.
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
    logger.log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.log(f"Input shape: [{args.batch_size}, 3, {args.image_size}, {args.image_size}, {args.image_size}]")
    logger.log(f"Conditioning: VF range={args.vf_range}, YM range={args.ym_range}")
    
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("Creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        deterministic=False,
        num_workers=args.num_workers,
    )

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
        # Data
        data_dir="",
        num_workers=4,
        
        # Training
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=8,  # Adjust based on GPU memory
        microbatch=-1,  # -1 = no gradient accumulation
        ema_rate="0.9999",
        log_interval=100,
        save_interval=5000,
        resume_checkpoint="",
        
        # Model
        use_fp16=True,  # Recommended for 3D to save memory
        fp16_scale_growth=1e-3,
        
        # Conditioning ranges
        vf_range=[0.1, 0.9],
        ym_range=[1.0, 100.0],
    )
    defaults.update(model_and_diffusion_defaults())
    
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()