"""Explicit local corrections for reverse-output semantics, preserving epsilon.

The upstream source snapshot stays immutable. Installing this adapter is a no-op
for the default epsilon model; direct-x0 models require the corrected branch.
Only fixed-variance models used in this repository are supported here.
"""
import numpy as np
import torch

from common import build_model  # establishes the pinned upstream import path
from Squidiff.diffusion import ModelMeanType, ModelVarType, _extract_into_tensor


def correct_reverse_target(d):
    if d.model_mean_type == ModelMeanType.EPSILON:
        return d
    if d.model_mean_type != ModelMeanType.START_X:
        raise ValueError('Adapter supports EPSILON and START_X only')
    if d.model_var_type not in [ModelVarType.FIXED_SMALL, ModelVarType.FIXED_LARGE]:
        raise ValueError('Learned variance is outside this adapter\'s scope')

    def p_mean_variance(model, x, t, clip_denoised=False, denoised_fn=None, model_kwargs=None):
        if t.shape != (len(x),):
            raise ValueError('Expected one diffusion timestep per cell')
        wrapped = d._wrap_model(model) if hasattr(d, '_wrap_model') else model
        x0 = wrapped(x, d._scale_timesteps(t), **(model_kwargs or {}))
        if denoised_fn is not None:
            x0 = denoised_fn(x0)
        # Preserve Squidiff's nonnegative-expression convention exactly.
        x0 = x0.clamp(-1, 1) if clip_denoised else x0.clamp_min(0)
        mean, _, _ = d.q_posterior_mean_variance(x0, x, t)
        # Match the author's fixed variance convention, including FIXED_SMALL.
        variance = np.append(d.posterior_variance[1], d.betas[1:])
        return dict(mean=mean, pred_xstart=x0,
            variance=_extract_into_tensor(variance,t,x.shape),
            log_variance=_extract_into_tensor(np.log(variance),t,x.shape))
    d.p_mean_variance = p_mean_variance
    return d


def min_snr_weight(d, t, gamma=5.):
    alpha = torch.as_tensor(d.alphas_cumprod, device=t.device, dtype=torch.float32)[t]
    snr = alpha / (1-alpha)
    return torch.minimum(snr, torch.full_like(snr,gamma)) / snr
