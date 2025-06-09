import argparse
from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .unet import UNetModel, EncoderUNetModel


def diffusion_defaults():
    """
    Defaults for diffusion models.
    """
    return dict(
        learn_sigma=True,
        diffusion_steps=1000,
        noise_schedule="cosine",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )


def model_and_diffusion_defaults():
    """
    Defaults for 3D volume diffusion models.
    """
    res = dict(
        image_size=64,
        num_channels=128,
        num_res_blocks=3,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=64,
        attention_resolutions="16,8",
        channel_mult="1,2,2,4",
        dropout=0.0,
        use_checkpoint=True,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=True,
        use_new_attention_order=False,
        # Conditioning parameters
        vf_range=(0.1, 0.9),
        ym_range=(1.0, 100.0),
    )
    res.update(diffusion_defaults())
    return res


def create_model_and_diffusion(
    image_size,
    learn_sigma,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    use_new_attention_order,
    vf_range=(0.1, 0.9),
    ym_range=(1.0, 100.0),
):
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        channel_mult=channel_mult,
        learn_sigma=learn_sigma,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="1,2,2,4",
    learn_sigma=False,
    use_checkpoint=True,
    attention_resolutions="16,8",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
):
    """
    Create a 3D UNet model for volume generation with conditioning.
    
    Input: [B, 3, D, H, W] where:
    - Channel 0: Structure field (to be denoised)
    - Channel 1: Volume fraction condition
    - Channel 2: Young's modulus condition
    """
    # Fixed 3 input channels for concatenate conditioning
    in_channels = 3
    
    # Output channels depend on whether we learn sigma
    out_channels = in_channels if not learn_sigma else in_channels * 2
    
    # Parse channel multipliers
    if channel_mult == "":
        channel_mult = (1, 2, 2, 4)  # Good default for 64x64x64
    else:
        channel_mult = tuple(int(ch) for ch in channel_mult.split(","))
    
    # Parse attention resolutions
    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))
    
    return UNetModel(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=num_channels,
        out_channels=out_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        dims=3,  # 3D volumes
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="cosine",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


# Regressor-related functions (kept for potential future use)
def regressor_defaults():
    """
    Defaults for regressor models that predict VF/YM from structure.
    """
    return dict(
        image_size=64,
        in_channels=1,         # Structure field only
        out_channels=2,        # VF and YM predictions
        regressor_use_fp16=False,
        regressor_width=128,
        regressor_depth=4,
        regressor_attention_resolutions="16,8",
        regressor_use_scale_shift_norm=True,
        regressor_resblock_updown=True,
        regressor_pool="spatial",
    )


def create_regressor(
    image_size,
    in_channels,
    regressor_use_fp16,
    regressor_width,
    regressor_depth,
    regressor_attention_resolutions,
    regressor_use_scale_shift_norm,
    regressor_resblock_updown,
    regressor_pool,
    out_channels=2,
):
    """
    Create a regressor model to predict VF/YM from structure.
    """
    if image_size == 64:
        channel_mult = (1, 2, 2, 4)
    else:
        raise ValueError(f"unsupported image size: {image_size}")
    
    attention_ds = []
    for res in regressor_attention_resolutions.split(","):
        if res.strip():
            attention_ds.append(image_size // int(res))
    
    return EncoderUNetModel(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=regressor_width,
        out_channels=out_channels,
        num_res_blocks=regressor_depth,
        attention_resolutions=tuple(attention_ds),
        channel_mult=channel_mult,
        dims=3,  # 3D
        use_fp16=regressor_use_fp16,
        num_head_channels=64,
        use_scale_shift_norm=regressor_use_scale_shift_norm,
        resblock_updown=regressor_resblock_updown,
        pool=regressor_pool,
    )