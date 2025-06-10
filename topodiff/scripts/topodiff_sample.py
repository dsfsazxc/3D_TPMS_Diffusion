"""
Simple 3D structure sampling from trained diffusion models.
"""

import argparse
import os
import numpy as np
import torch as th
import torch.distributed as dist

from topodiff import dist_util, logger
from topodiff.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure()

    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log(f"Model loaded from {args.model_path}")
    logger.log(f"Sampling {args.num_samples} samples...")

    logger.log("Starting sampling...")
    all_samples = []

    while len(all_samples) * args.batch_size < args.num_samples:
        model_kwargs = {}

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        # Simple unconditional sampling
        sample = sample_fn(
            model,
            shape=(args.batch_size, 1, args.image_size, args.image_size, args.image_size),
            noise=None,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
            progress=True,
        )

        sample = sample.clamp(-1, 1)
        sample_np = sample.cpu().numpy()
        all_samples.append(sample_np)
        
        logger.log(f"Created {len(all_samples) * args.batch_size} samples")

    # Combine all samples
    arr = np.concatenate(all_samples, axis=0)
    arr = arr[:args.num_samples]

    if dist.get_rank() == 0:
        save_dir = logger.get_dir()
        
        logger.log(f"Saving samples to {save_dir}")
        
        # Save individual samples
        for i in range(len(arr)):
            out_path = os.path.join(save_dir, f"sample_{i:04d}.npz")
            np.savez_compressed(out_path, volume=arr[i, 0])
            
            if i < 5:  # Log first few samples
                sample_stats = arr[i, 0]
                logger.log(f"Sample {i}: shape={sample_stats.shape}, "
                         f"range=[{sample_stats.min():.3f}, {sample_stats.max():.3f}], "
                         f"mean={sample_stats.mean():.3f}")

        # Save batch file
        batch_path = os.path.join(save_dir, "samples_batch.npz")
        np.savez_compressed(batch_path, samples=arr)
        logger.log(f"Batch saved to {batch_path}")

        # Create visualizations
        if args.visualize:
            logger.log("Creating visualizations...")
            visualize_samples(arr, args, save_dir)

    dist.barrier()
    logger.log("Sampling complete.")


def visualize_samples(samples, args, save_dir):
    """Create visualizations of the generated samples."""
    import matplotlib.pyplot as plt
    
    num_viz = min(args.num_visualize, len(samples))
    
    for idx in range(num_viz):
        sample_viz = samples[idx, 0]  # Remove channel dimension
        
        # Create figure with multiple views
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Calculate value range for consistent coloring
        vmin, vmax = sample_viz.min(), sample_viz.max()
        
        # Different slice views
        mid = args.image_size // 2
        slices = [
            (sample_viz[mid, :, :], f'X-slice (x={mid})'),
            (sample_viz[:, mid, :], f'Y-slice (y={mid})'),
            (sample_viz[:, :, mid], f'Z-slice (z={mid})'),
            (sample_viz[mid//2, :, :], f'X-slice (x={mid//2})'),
            (sample_viz[:, mid//2, :], f'Y-slice (y={mid//2})'),
            (sample_viz[:, :, mid//2], f'Z-slice (z={mid//2})'),
        ]
        
        for i, (slice_data, title) in enumerate(slices):
            im = axes[i].imshow(slice_data, cmap='coolwarm', vmin=vmin, vmax=vmax)
            axes[i].set_title(title)
            axes[i].axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=axes, orientation='horizontal', pad=0.1, shrink=0.8)
        
        # Add overall title with statistics
        fig.suptitle(f'Sample {idx}\nShape: {sample_viz.shape}, '
                    f'Range: [{vmin:.3f}, {vmax:.3f}], Mean: {sample_viz.mean():.3f}',
                    fontsize=14)
        
        # Save visualization
        viz_path = os.path.join(save_dir, f"sample_{idx:04d}_visualization.png")
        plt.tight_layout()
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save 3D structure analysis
        analyze_structure(sample_viz, idx, save_dir)


def analyze_structure(volume, idx, save_dir):
    """Analyze and save structure properties."""
    import matplotlib.pyplot as plt
    
    # Basic statistics
    stats = {
        'shape': volume.shape,
        'min': float(volume.min()),
        'max': float(volume.max()),
        'mean': float(volume.mean()),
        'std': float(volume.std()),
        'volume_positive': float((volume > 0).sum() / volume.size),
        'volume_negative': float((volume < 0).sum() / volume.size),
    }
    
    # Save statistics
    stats_path = os.path.join(save_dir, f"sample_{idx:04d}_stats.txt")
    with open(stats_path, 'w') as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(volume.flatten(), bins=50, alpha=0.7, density=True)
    plt.title(f'Sample {idx} - Value Distribution')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    
    hist_path = os.path.join(save_dir, f"sample_{idx:04d}_histogram.png")
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_argparser():
    defaults = dict(
        # Sampling parameters
        clip_denoised=True,
        num_samples=10,
        batch_size=1,
        use_ddim=False,
        
        # Model path
        model_path="",
        
        # Visualization
        visualize=True,
        num_visualize=5,
        
        # Model parameters (should match training)
        image_size=64,
        use_fp16=True,
    )
    
    # Add model and diffusion defaults
    defaults.update(model_and_diffusion_defaults())
    
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()