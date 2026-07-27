"""TPU-Native Pallas FlashAttention.

A from-scratch FlashAttention kernel written in JAX Pallas for the TPU v3-8,
wired into a tiny LLaMA-style decoder for inference.
"""

from .config import ModelConfig
from .flash_attention import flash_attention, flash_attention_decode
from .fused_rope import flash_attention_rope
from .paged import paged_flash_attention, init_paged_cache, write_to_paged_cache
from .reference import reference_attention
from .model import (
    forward,
    generate,
    generate_cached,
    prefill,
    decode_step,
    init_kv_cache,
    init_ring_cache,
    ring_kv_len,
    prefill_ring,
    ring_decode_step,
    generate_streaming,
    init_params,
    decoder_block,
    project_qkv,
    rms_norm,
    apply_rope,
    precompute_rope,
    swiglu_mlp,
)

__all__ = [
    "ModelConfig",
    "flash_attention",
    "flash_attention_decode",
    "flash_attention_rope",
    "paged_flash_attention",
    "init_paged_cache",
    "write_to_paged_cache",
    "reference_attention",
    "forward",
    "generate",
    "generate_cached",
    "prefill",
    "decode_step",
    "init_kv_cache",
    "init_ring_cache",
    "ring_kv_len",
    "prefill_ring",
    "ring_decode_step",
    "generate_streaming",
    "init_params",
    "decoder_block",
    "project_qkv",
    "rms_norm",
    "apply_rope",
    "precompute_rope",
    "swiglu_mlp",
]

__version__ = "0.1.0"
