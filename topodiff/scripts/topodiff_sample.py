"""
Conditional sampling for 3D TPMS structures with target VF and Young's Modulus
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
    
    logger.log(f"Loading model from {args.model_path}...")
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location=dist_util.dev())
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("Sampling...")
    all_samples = []
    all_metadata = []
    
    while len(all_samples) * args.batch_size < args.num_samples:
        model_kwargs = {}
        
        # Create conditional samples
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        
        # Generate samples with conditions
        samples, metadata = conditional_sample(
            sample_fn=sample_fn,
            model=model,
            shape=(args.batch_size, 3, args.image_size, args.image_size, args.image_size),
            target_vf=args.target_vf,
            target_ym=args.target_ym,
            vf_range=(args.vf_range[0], args.vf_range[1]),
            ym_range=(args.ym_range[0], args.ym_range[1]),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
            progress=True,
        )
        
        samples = samples[:, 0:1]  # Extract only the structure channel
        
        if args.use_regressor and args.regressor_path:
            # Use regressor to verify/filter samples
            samples, metadata = filter_with_regressor(
                samples, metadata, args, model_kwargs
            )
        
        all_samples.append(samples.cpu().numpy())
        all_metadata.extend(metadata)
        logger.log(f"Created {len(all_samples) * args.batch_size} samples")

    # Stack all samples
    arr = np.concatenate(all_samples, axis=0)
    arr = arr[: args.num_samples]
    metadata = all_metadata[: args.num_samples]

    # Save samples
    if dist.get_rank() == 0:
        save_samples(arr, metadata, args)
        logger.log(f"Sampling complete. Saved to {args.output_dir}")

    dist.barrier()
    logger.log("Sampling complete!")


def conditional_sample(
    sample_fn,
    model,
    shape,
    target_vf,
    target_ym,
    vf_range,
    ym_range,
    noise=None,
    clip_denoised=True,
    denoised_fn=None,
    cond_fn=None,
    model_kwargs=None,
    device=None,
    progress=False,
):
    """
    Generate samples with concatenated conditioning.
    """
    if model_kwargs is None:
        model_kwargs = {}
    
    B, C, D, H, W = shape
    
    # Normalize targets to [0, 1]
    vf_norm = (target_vf - vf_range[0]) / (vf_range[1] - vf_range[0])
    ym_norm = (target_ym - ym_range[0]) / (ym_range[1] - ym_range[0])
    
    # Create initial noise for structure channel
    if noise is None:
        noise = th.randn((B, 1, D, H, W), device=device)
    
    # Create condition channels
    vf_channel = th.full((B, 1, D, H, W), vf_norm, device=device)
    ym_channel = th.full((B, 1, D, H, W), ym_norm, device=device)
    
    # Concatenate to form full input
    # During sampling, we'll maintain this structure
    def model_fn(x, t, **kwargs):
        # x shape: [B, 1, D, H, W] (just the noisy structure)
        # Concatenate with condition channels
        model_input = th.cat([x, vf_channel, ym_channel], dim=1)
        # Get model output
        out = model(model_input, t, **kwargs)
        # Return only the structure channel predictions
        if out.shape[1] == 6:  # learn_sigma=True
            return th.cat([out[:, :1], out[:, 3:4]], dim=1)
        else:
            return out[:, :1]
    
    # Run sampling with wrapped model
    samples = sample_fn(
        model_fn,
        (B, 1, D, H, W),  # Shape for structure only
        noise=noise,
        clip_denoised=clip_denoised,
        denoised_fn=denoised_fn,
        cond_fn=cond_fn,
        model_kwargs=model_kwargs,
        device=device,
        progress=progress,
    )
    
    # Create metadata
    metadata = [
        {
            "target_vf": target_vf,
            "target_ym": target_ym,
            "vf_norm": vf_norm,
            "ym_norm": ym_norm,
        }
        for _ in range(B)
    ]
    
    return samples, metadata


def filter_with_regressor(samples, metadata, args, model_kwargs):
    """
    Use a regressor model to verify/filter generated samples.
    """
    logger.log("Loading regressor for sample filtering...")
    
    from topodiff.script_util import create_regressor
    
    regressor = create_regressor(
        image_size=args.image_size,
        in_channels=1,
        regressor_use_fp16=args.use_fp16,
        regressor_width=128,
        regressor_depth=4,
        regressor_attention_resolutions="32,16,8",
        regressor_use_scale_shift_norm=True,
        regressor_resblock_updown=True,
        regressor_pool="spatial",
        out_channels=2,
    )
    
    regressor.load_state_dict(
        dist_util.load_state_dict(args.regressor_path, map_location=dist_util.dev())
    )
    regressor.to(dist_util.dev())
    if args.use_fp16:
        regressor.convert_to_fp16()
    regressor.eval()
    
    # Predict VF and YM
    with th.no_grad():
        # Create dummy timesteps
        t = th.zeros(samples.shape[0], dtype=th.long, device=samples.device)
        predictions = regressor(samples, t)
    
    # Denormalize predictions
    vf_pred = predictions[:, 0] * (args.vf_range[1] - args.vf_range[0]) + args.vf_range[0]
    ym_pred = predictions[:, 1] * (args.ym_range[1] - args.ym_range[0]) + args.ym_range[0]
    
    # Update metadata with predictions
    for i, meta in enumerate(metadata):
        meta["predicted_vf"] = vf_pred[i].item()
        meta["predicted_ym"] = ym_pred[i].item()
        meta["vf_error"] = abs(meta["predicted_vf"] - meta["target_vf"])
        meta["ym_error"] = abs(meta["predicted_ym"] - meta["target_ym"])
    
    # Filter based on tolerance
    if args.filter_tolerance > 0:
        filtered_samples = []
        filtered_metadata = []
        
        for i, meta in enumerate(metadata):
            vf_error_rel = meta["vf_error"] / meta["target_vf"]
            ym_error_rel = meta["ym_error"] / meta["target_ym"]
            
            if vf_error_rel < args.filter_tolerance and ym_error_rel < args.filter_tolerance:
                filtered_samples.append(samples[i:i+1])
                filtered_metadata.append(meta)
        
        if filtered_samples:
            samples = th.cat(filtered_samples, dim=0)
            metadata = filtered_metadata
            logger.log(f"Kept {len(filtered_samples)} samples after filtering")
        else:
            logger.log("Warning: No samples passed the filter!")
    
    return samples, metadata


def save_samples(samples, metadata, args):
    """
    Save generated samples and metadata.
    """
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save individual samples
    for i, (sample, meta) in enumerate(zip(samples, metadata)):
        # Save as npz with metadata
        filename = os.path.join(
            args.output_dir, 
            f"sample_{i:06d}_vf{meta['target_vf']:.3f}_ym{meta['target_ym']:.1f}.npz"
        )
        
        np.savez_compressed(
            filename,
            structure=sample[0],  # Structure field
            **meta  # All metadata
        )
    
    # Save batch file
    batch_file = os.path.join(args.output_dir, "all_samples.npz")
    np.savez_compressed(
        batch_file,
        samples=samples,
        metadata=metadata,
        args=vars(args)
    )
    
    # Create visualization if requested
    if args.visualize:
        visualize_samples(samples, metadata, args)


def visualize_samples(samples, metadata, args):
    """
    Create visualizations of generated samples.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    vis_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    for i, (sample, meta) in enumerate(zip(samples[:min(10, len(samples))], metadata)):
        fig = plt.figure(figsize=(15, 5))
        
        # Structure field
        structure = sample[0]
        
        # 2D slices
        for j, (slice_idx, axis_name) in enumerate([(32, 'X'), (32, 'Y'), (32, 'Z')]):
            ax = fig.add_subplot(1, 4, j+1)
            
            if j == 0:
                slice_data = structure[slice_idx, :, :]
            elif j == 1:
                slice_data = structure[:, slice_idx, :]
            else:
                slice_data = structure[:, :, slice_idx]
            
            im = ax.imshow(slice_data, cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_title(f'{axis_name}={slice_idx}')
            ax.axis('off')
        
        # 3D isosurface (simplified)
        ax = fig.add_subplot(1, 4, 4, projection='3d')
        
        # Simple threshold visualization
        threshold = 0.0
        binary = structure > threshold
        
        # Find surface points (simplified)
        from skimage import measure
        try:
            verts, faces, _, _ = measure.marching_cubes(
                binary.astype(float), level=0.5, spacing=(1, 1, 1)
            )
            ax.plot_trisurf(
                verts[:, 0], verts[:, 1], faces, verts[:, 2],
                alpha=0.5, shade=True
            )
            ax.set_xlim(0, 64)
            ax.set_ylim(0, 64)
            ax.set_zlim(0, 64)
        except:
            ax.text(0.5, 0.5, 0.5, "3D visualization failed", 
                   transform=ax.transAxes, ha='center')
        
        ax.set_title('3D Structure')
        
        # Overall title
        title = f'Sample {i}: VF={meta["target_vf"]:.3f}, YM={meta["target_ym"]:.1f}'
        if "predicted_vf" in meta:
            title += f'\nPredicted: VF={meta["predicted_vf"]:.3f}, YM={meta["predicted_ym"]:.1f}'
        fig.suptitle(title)
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'sample_{i:06d}.png'), dpi=150)
        plt.close()
    
    logger.log(f"Saved visualizations to {vis_dir}")


def create_argparser():
    defaults = dict(
        # Model arguments
        model_path="",
        regressor_path="",
        
        # Sampling arguments
        num_samples=10,
        batch_size=4,
        use_ddim=False,
        
        # Conditioning arguments
        target_vf=0.5,
        target_ym=50.0,
        vf_range=[0.1, 0.9],
        ym_range=[1.0, 100.0],
        
        # Filtering arguments
        use_regressor=False,
        filter_tolerance=0.1,  # Relative error tolerance
        
        # Output arguments
        output_dir="samples",
        visualize=True,
    )
    defaults.update(model_and_diffusion_defaults())
    
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()