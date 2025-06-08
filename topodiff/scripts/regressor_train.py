"""
Train a noised image regressor to predict VF and Young's Modulus for 3D TPMS.
"""

import argparse
import os
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from topodiff import dist_util, logger
from topodiff.fp16_util import MixedPrecisionTrainer
from topodiff.image_datasets_regressor import load_data
from topodiff.resample import create_named_schedule_sampler
from topodiff.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    regressor_and_diffusion_defaults,
    create_regressor_and_diffusion,
)
from topodiff.train_util import parse_resume_step_from_filename, log_loss_dict

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    # 출력 채널을 2로 설정 (VF와 Young's Modulus)
    model, diffusion = create_regressor_and_diffusion(
        **args_to_dict(args, regressor_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    
    if args.noised:
        schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler, diffusion
        )

    resume_step = 0
    if args.resume_checkpoint:
        resume_step = parse_resume_step_from_filename(args.resume_checkpoint)
        if dist.get_rank() == 0:
            logger.log(f"loading model from checkpoint: {args.resume_checkpoint}...")
            model.load_state_dict(
                dist_util.load_state_dict(
                    args.resume_checkpoint, map_location=dist_util.dev()
                )
            )

    dist_util.sync_params(model.parameters())

    mp_trainer = MixedPrecisionTrainer(
        model=model, use_fp16=args.regressor_use_fp16, initial_lg_loss_scale=16.0
    )

    model = DDP(
        model,
        device_ids=[dist_util.dev()],
        output_device=dist_util.dev(),
        broadcast_buffers=False,
        bucket_cap_mb=128,
        find_unused_parameters=False,
    )

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        deterministic=False
    )

    logger.log(f"creating optimizer...")
    opt = AdamW(mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay)

    logger.log("training regressor model...")

    def forward_backward_log(data_loader, prefix="train"):
        batch, batch_cons, extra = next(data_loader)
        
        # VF와 Young's Modulus 타겟
        vf_target = extra["vf"].to(dist_util.dev())
        youngs_target = extra["youngs"].to(dist_util.dev())
        targets = th.stack([vf_target, youngs_target], dim=1)  # [B, 2]
        
        batch = batch.to(dist_util.dev())
        
        # Noisy images
        if args.noised:
            t, _ = schedule_sampler.sample(batch.shape[0], dist_util.dev())
            batch = diffusion.q_sample(batch, t)
        else:
            t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())
        
        for i, (sub_batch, sub_targets, sub_t) in enumerate(
            split_microbatches(args.microbatch, batch, targets, t)
        ):
            # 3D regressor는 dims=3으로 설정되어 있어야 함
            logits = model(sub_batch, timesteps=sub_t)  # [B, 2]
            
            # 각 속성에 대한 손실 계산
            vf_loss = F.mse_loss(logits[:, 0], sub_targets[:, 0])
            youngs_loss = F.mse_loss(logits[:, 1], sub_targets[:, 1])
            
            # 정규화된 손실 (Young's Modulus는 값이 크므로 스케일 조정)
            youngs_loss_scaled = youngs_loss / 10000.0  # Young's Modulus는 보통 큰 값
            total_loss = vf_loss + youngs_loss_scaled
            
            losses = {}
            losses[f"{prefix}_loss"] = total_loss.detach()
            losses[f"{prefix}_vf_loss"] = vf_loss.detach()
            losses[f"{prefix}_youngs_loss"] = youngs_loss.detach()
            
            # 예측값 로깅
            with th.no_grad():
                losses[f"{prefix}_vf_mae"] = th.abs(logits[:, 0] - sub_targets[:, 0]).mean()
                losses[f"{prefix}_youngs_mae"] = th.abs(logits[:, 1] - sub_targets[:, 1]).mean()
                
                # 몇 개 샘플 출력
                if step % 100 == 0 and i == 0:
                    logger.log(f"Sample predictions:")
                    for j in range(min(3, len(sub_targets))):
                        logger.log(f"  VF - True: {sub_targets[j, 0]:.4f}, Pred: {logits[j, 0]:.4f}")
                        logger.log(f"  YM - True: {sub_targets[j, 1]:.1f}, Pred: {logits[j, 1]:.1f}")

            log_loss_dict(diffusion, sub_t, losses)
            
            if total_loss.requires_grad:
                if i == 0:
                    mp_trainer.zero_grad()
                mp_trainer.backward(total_loss * len(sub_batch) / len(batch))

    for step in range(args.iterations - resume_step):
        logger.logkv("step", step + resume_step)
        logger.logkv(
            "samples",
            (step + resume_step + 1) * args.batch_size * dist.get_world_size(),
        )
        
        if args.anneal_lr:
            set_annealed_lr(opt, args.lr, (step + resume_step) / args.iterations)
        
        forward_backward_log(data)
        mp_trainer.optimize(opt)
        
        if not step % args.log_interval:
            logger.dumpkvs()
            
        if (
            step
            and dist.get_rank() == 0
            and not (step + resume_step) % args.save_interval
        ):
            logger.log("saving model...")
            save_model(mp_trainer, opt, step + resume_step)

    if dist.get_rank() == 0:
        logger.log("saving model...")
        save_model(mp_trainer, opt, step + resume_step)
    dist.barrier()


def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def save_model(mp_trainer, opt, step):
    if dist.get_rank() == 0:
        th.save(
            mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
            os.path.join(logger.get_dir(), f"regressor{step:06d}.pt"),
        )
        th.save(opt.state_dict(), os.path.join(logger.get_dir(), f"opt{step:06d}.pt"))


def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)


def create_argparser():
    defaults = dict(
        data_dir="",
        noised=True,
        iterations=50000,
        lr=1e-4,
        weight_decay=0.1,
        anneal_lr=False,
        batch_size=4,  # 3D는 메모리 사용량이 크므로
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint="",
        log_interval=100,
        save_interval=5000,
        image_size=64,
    )
    defaults.update(regressor_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()