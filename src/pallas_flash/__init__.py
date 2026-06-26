"""TPU-Native Pallas FlashAttention.

A from-scratch FlashAttention kernel written in JAX Pallas for the TPU v3-8,
wired into a tiny LLaMA-style decoder for inference.
"""

from .config import ModelConfig
from .flash_attention import flash_attention
from .reference import reference_attention
from .model import (
    forward,
    generate,
    init_params,
    decoder_block,
    rms_norm,
    apply_rope,
    precompute_rope,
    swiglu_mlp,
)

__all__ = [
    "ModelConfig",
    "flash_attention",
    "reference_attention",
    "forward",
    "generate",
    "init_params",
    "decoder_block",
    "rms_norm",
    "apply_rope",
    "precompute_rope",
    "swiglu_mlp",
]

__version__ = "0.1.0"
