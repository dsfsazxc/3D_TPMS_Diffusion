"""
Guided 3D sampling for TPMS with target VF and Young's Modulus
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
    regressor_and_diffusion_defaults,
    create_regressor_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure()

    # Diffusion 모델 생성 및 로드
    logger.log("Creating diffusion model...")
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

    # 체크포인트 구조 확인
    checkpoint = th.load(args.regressor_path, map_location='cpu')
    print("Checkpoint keys:", checkpoint.keys())

    # 출력 레이어 관련 키 찾기
    out_keys = [k for k in checkpoint.keys() if 'out' in k]
    print("Output layer keys:", out_keys)
    print("Output layer shape:", checkpoint['out.0.weight'].shape if 'out.0.weight' in checkpoint else "Not found")
    # Regressor 모델 생성 및 로드
    logger.log("Creating regressor model...")

    # regressor 전용 인자만 추출 (pool 타입 추가)
    regressor_keys = [
        'image_size', 'regressor_use_fp16', 'regressor_width',
        'regressor_attention_resolutions', 'regressor_use_scale_shift_norm',
        'regressor_resblock_updown', 'regressor_pool',
        'learn_sigma', 'diffusion_steps', 'noise_schedule',
        'timestep_respacing', 'use_kl', 'predict_xstart',
        'rescale_timesteps', 'rescale_learned_sigmas'
    ]
    regressor_args = args_to_dict(args, regressor_keys)

    # pool 타입을 adaptive로 강제 설정 (저장된 모델과 맞추기)
    regressor_args['regressor_pool'] = 'adaptive'

    regressor, _ = create_regressor_and_diffusion(
        in_channels=1,
        out_channels=2,
        regressor_depth=args.regressor_depth,
        **regressor_args
    )

    '''regressor.load_state_dict(
        dist_util.load_state_dict(args.regressor_path, map_location="cpu")
    )'''
    # topodiff_sample.py의 main() 함수 내에서, regressor.load_state_dict() 대신 사용
    checkpoint = th.load(args.regressor_path, map_location='cpu')

    # 키 이름 변경이 필요한 경우
    state_dict = {}
    for k, v in checkpoint.items():
        # 예: 'out.0.weight' -> 'out.weight' 또는 다른 매핑
        new_k = k
        if k == 'out.0.weight':
            new_k = 'out.weight'  # 또는 적절한 키 이름
        elif k == 'out.0.bias':
            new_k = 'out.bias'
        state_dict[new_k] = v

    # 부분 로드 시도
    regressor.load_state_dict(state_dict, strict=False)

    regressor.to(dist_util.dev())
    if args.regressor_use_fp16:
        regressor.convert_to_fp16()

    regressor.eval()

    # Regressor 로드 후 테스트
    print("Testing regressor...")
    with th.no_grad():
        # 랜덤 입력으로 테스트
        test_input = th.randn(1, 1, 64, 64, 64).to(dist_util.dev())
        test_t = th.zeros(1, dtype=th.long).to(dist_util.dev())
        test_output = regressor(test_input, test_t)
        print(f"Test output: VF={test_output[0, 0]:.4f}, YM={test_output[0, 1]:.4f}")

    # topodiff_sample.py의 main() 함수 내에서, regressor 생성 후 추가
    # 현재 모델 구조 확인
    print("Current model output layer:")
    for name, param in regressor.named_parameters():
        if 'out' in name:
            print(f"{name}: {param.shape}")

    # 체크포인트와 비교
    checkpoint = th.load(args.regressor_path, map_location='cpu')
    print("\nCheckpoint output layer:")
    for k, v in checkpoint.items():
        if 'out' in k:
            print(f"{k}: {v.shape}")
            
    # cond_fn_regressor 수정 - 더 많은 디버깅 정보
    def cond_fn_regressor(x, t):
        """
        Gradient function for VF/YM guidance
        """
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            
            # Ensure t is a tensor
            if not isinstance(t, th.Tensor):
                t = th.tensor([t] * x.shape[0], device=x.device, dtype=th.long)
            
            # Check input
            print(f"Input x stats: min={x_in.min():.3f}, max={x_in.max():.3f}, mean={x_in.mean():.3f}")
            
            # Forward pass through regressor
            predictions = regressor(x_in, t)  # [B, 2]
            
            # Check predictions
            print(f"Predictions: VF={predictions[0, 0]:.4f}, YM={predictions[0, 1]:.4f}")
            
            # Target values
            vf_target = args.target_vf
            ym_target = args.target_youngs
            
            # Compute losses
            vf_pred = predictions[:, 0].mean()
            ym_pred = predictions[:, 1].mean()
            
            vf_loss = (vf_pred - vf_target) ** 2
            ym_loss = ((ym_pred - ym_target) ** 2) / 10000.0  # Scale YM
            
            # Combined loss
            total_loss = vf_loss + ym_loss * args.youngs_weight
            
            # Compute gradient
            grad = th.autograd.grad(total_loss, x_in)[0]
            
            # Debug info
            if t[0] % 100 == 0:
                print(f"Step {t[0]}: VF pred={vf_pred:.4f}, YM pred={ym_pred:.1f}")
                print(f"  Loss: VF={vf_loss:.6f}, YM={ym_loss:.6f}, Total={total_loss:.6f}")
                print(f"  Grad: norm={grad.norm():.6f}, max={grad.abs().max():.6f}")
            
            return -grad * args.regressor_scale

    logger.log("Sampling...")
    all_samples = []
    all_predictions = []

    while len(all_samples) * args.batch_size < args.num_samples:
        model_kwargs = {}

        # 더미 제약조건들
        dummy_cons = th.zeros((args.batch_size, 0, args.image_size, args.image_size, args.image_size)).to(dist_util.dev())
        dummy_loads = th.zeros((args.batch_size, 0, args.image_size, args.image_size, args.image_size)).to(dist_util.dev())
        dummy_BCs = th.zeros((args.batch_size, 0, args.image_size, args.image_size, args.image_size)).to(dist_util.dev())

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        # Guided sampling
        sample = sample_fn(
            model,
            shape=(args.batch_size, 1, args.image_size, args.image_size, args.image_size),
            cons=dummy_cons,
            loads=dummy_loads,
            BCs=dummy_BCs,
            noise=None,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn_1=cond_fn_regressor if args.use_guidance else None,
            cond_fn_2=None,
            device=dist_util.dev(),
            progress=True,
        )

        # 최종 예측값 확인
        with th.no_grad():
            final_predictions = regressor(sample, th.zeros(args.batch_size, device=sample.device, dtype=th.long))
            logger.log(f"Final predictions - VF: {final_predictions[:, 0].mean():.4f}, YM: {final_predictions[:, 1].mean():.1f}")
            logger.log(f"Targets - VF: {args.target_vf:.4f}, YM: {args.target_youngs:.1f}")

        sample = sample.clamp(-1, 1)
        sample_np = sample.cpu().numpy()
        predictions_np = final_predictions.cpu().numpy()

        all_samples.append(sample_np)
        all_predictions.append(predictions_np)
        
        logger.log(f"Created {len(all_samples) * args.batch_size} samples")
    # 샘플링 후 확인
    print("\nChecking final sample...")
    print(f"Sample stats: min={sample.min():.3f}, max={sample.max():.3f}, mean={sample.mean():.3f}")
    print(f"NaN count: {th.isnan(sample).sum()}")
    print(f"Inf count: {th.isinf(sample).sum()}")

    # 최종 예측 시 입력 확인
    with th.no_grad():
        # sample이 NaN이 아닌지 확인
        if th.isnan(sample).any():
            print("WARNING: Sample contains NaN values!")
            sample = th.nan_to_num(sample, nan=0.0)
        
        final_predictions = regressor(sample, th.zeros(args.batch_size, device=sample.device, dtype=th.long))
        print(f"Final predictions - VF: {final_predictions[:, 0].mean():.4f}, YM: {final_predictions[:, 1].mean():.4f}")
    # 저장
    arr = np.concatenate(all_samples, axis=0)
    arr = arr[:args.num_samples]
    predictions = np.concatenate(all_predictions, axis=0)[:args.num_samples]

    if dist.get_rank() == 0:
        save_dir = logger.get_dir()
        
        # 샘플과 예측값 저장
        for i in range(len(arr)):
            out_path = os.path.join(save_dir, f"sample_{i:04d}.npz")
            np.savez(out_path, 
                    arr_0=arr[i, 0],
                    vf_pred=predictions[i, 0],
                    ym_pred=predictions[i, 1],
                    vf_target=args.target_vf,
                    ym_target=args.target_youngs)
            logger.log(f"Sample {i} - VF: {predictions[i, 0]:.4f} (target: {args.target_vf:.4f}), YM: {predictions[i, 1]:.1f} (target: {args.target_youngs:.1f})")

        # 시각화
        visualize_samples(arr, predictions, args, save_dir)

    dist.barrier()
    logger.log("Sampling complete.")


def visualize_samples(samples, predictions, args, save_dir):
    import matplotlib.pyplot as plt
    
    for idx in range(min(4, len(samples))):
        sample_viz = samples[idx, 0]
        vf_pred = predictions[idx, 0]
        ym_pred = predictions[idx, 1]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        vmin, vmax = sample_viz.min(), sample_viz.max()
        
        for i, (ax, slice_idx, title) in enumerate(zip(axes, [32, 32, 32], ['X=32', 'Y=32', 'Z=32'])):
            if i == 0:
                im = ax.imshow(sample_viz[slice_idx, :, :], cmap='coolwarm', vmin=vmin, vmax=vmax)
            elif i == 1:
                im = ax.imshow(sample_viz[:, slice_idx, :], cmap='coolwarm', vmin=vmin, vmax=vmax)
            else:
                im = ax.imshow(sample_viz[:, :, slice_idx], cmap='coolwarm', vmin=vmin, vmax=vmax)
            
            ax.set_title(title)
            ax.axis('off')
        
        plt.colorbar(im, ax=axes, orientation='horizontal', pad=0.1)
        
        fig.suptitle(f'Sample {idx}\nVF: {vf_pred:.4f} (target: {args.target_vf:.4f}), YM: {ym_pred:.1f} (target: {args.target_youngs:.1f})')
        
        viz_path = os.path.join(save_dir, f"sample_{idx:04d}_visualization.png")
        plt.tight_layout()
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()


def create_argparser():
    defaults = dict(
        # Sampling parameters
        clip_denoised=True,
        num_samples=4,
        batch_size=1,
        use_ddim=False,
        use_guidance=True,
        
        # Model paths
        model_path="",
        regressor_path="",
        
        # Model parameters
        image_size=64,
        use_fp16=True,
        regressor_use_fp16=True,
        regressor_depth=3,
        
        # Target values
        target_vf=0.3,  # Target volume fraction
        target_youngs=200.0,  # Target Young's modulus
        
        # Guidance parameters
        regressor_scale=100.0,  # Gradient scaling
        youngs_weight=1.0,  # Relative weight for YM loss
    )
    
    # Add model defaults
    defaults.update(model_and_diffusion_defaults())
    defaults.update(regressor_and_diffusion_defaults())
    
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()