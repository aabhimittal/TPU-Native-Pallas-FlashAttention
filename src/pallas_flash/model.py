"""A tiny, randomly-initialized LLaMA-style decoder built around the Pallas
FlashAttention kernel.

The model is intentionally small and is *not* trained — its purpose is to
exercise the custom attention kernel on a realistic inference path. It uses the
standard LLaMA ingredients:

* RMSNorm (pre-normalization)
* Rotary position embeddings (RoPE)
* SwiGLU feed-forward network
* A stack of pre-norm decoder blocks with residual connections
* Causal self-attention via :func:`pallas_flash.flash_attention.flash_attention`

Parameters are stored as a plain nested-dict pytree so the whole thing stays
dependency-light (just JAX + NumPy, no Flax/Haiku).
"""

from __future__ import annotations

from typing import Any, Dict

import jax
import jax.numpy as jnp

from .config import ModelConfig
from .flash_attention import flash_attention

Params = Dict[str, Any]


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def rms_norm(x: jax.Array, weight: jax.Array, eps: float) -> jax.Array:
    """Root-mean-square layer norm (LLaMA-style, no mean subtraction)."""
    var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    normed = x.astype(jnp.float32) * jax.lax.rsqrt(var + eps)
    return (normed * weight).astype(x.dtype)


def precompute_rope(seq_len: int, head_dim: int, theta: float) -> tuple[jax.Array, jax.Array]:
    """Precompute the cos/sin tables for rotary position embeddings.

    Returns two arrays of shape ``[seq_len, head_dim]``.
    """
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (jnp.arange(0, half, dtype=jnp.float32) / half))
    pos = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.outer(pos, inv_freq)               # [seq, half]
    emb = jnp.concatenate([freqs, freqs], axis=-1)  # [seq, head_dim]
    return jnp.cos(emb), jnp.sin(emb)


def _rotate_half(x: jax.Array) -> jax.Array:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """Apply rotary embeddings to ``x`` of shape ``[batch, heads, seq, head_dim]``.

    ``cos``/``sin`` are ``[seq, head_dim]`` and broadcast over batch and heads.
    """
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + _rotate_half(x) * sin


def swiglu_mlp(x: jax.Array, w_gate: jax.Array, w_up: jax.Array, w_down: jax.Array) -> jax.Array:
    """SwiGLU feed-forward: ``(silu(x W_gate) * (x W_up)) W_down``."""
    gate = jax.nn.silu(x @ w_gate)
    up = x @ w_up
    return (gate * up) @ w_down


# --------------------------------------------------------------------------- #
# Decoder block + full model
# --------------------------------------------------------------------------- #
def decoder_block(
    params: Params,
    x: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    cfg: ModelConfig,
    interpret: bool,
) -> jax.Array:
    """One pre-norm transformer decoder block."""
    batch, seq, _ = x.shape

    # --- attention ---------------------------------------------------------
    h = rms_norm(x, params["attn_norm"], cfg.rms_norm_eps)
    q = h @ params["wq"]
    k = h @ params["wk"]
    v = h @ params["wv"]

    def to_heads(t: jax.Array) -> jax.Array:
        return t.reshape(batch, seq, cfg.n_heads, cfg.head_dim).transpose(0, 2, 1, 3)

    q = apply_rope(to_heads(q), cos, sin)
    k = apply_rope(to_heads(k), cos, sin)
    v = to_heads(v)

    # Custom Pallas FlashAttention kernel (causal for autoregressive decoding).
    attn = flash_attention(
        q, k, v, causal=True, block_q=cfg.block_q, block_k=cfg.block_k, interpret=interpret
    )
    attn = attn.transpose(0, 2, 1, 3).reshape(batch, seq, cfg.dim)
    x = x + attn @ params["wo"]

    # --- feed-forward ------------------------------------------------------
    h = rms_norm(x, params["ffn_norm"], cfg.rms_norm_eps)
    x = x + swiglu_mlp(h, params["w_gate"], params["w_up"], params["w_down"])
    return x


def forward(params: Params, tokens: jax.Array, cfg: ModelConfig, interpret: bool = False) -> jax.Array:
    """Run the model and return logits of shape ``[batch, seq, vocab_size]``."""
    seq = tokens.shape[1]
    x = params["embed"][tokens]                       # [batch, seq, dim]
    cos, sin = precompute_rope(seq, cfg.head_dim, cfg.rope_theta)
    cos, sin = cos.astype(x.dtype), sin.astype(x.dtype)

    for layer in params["layers"]:
        x = decoder_block(layer, x, cos, sin, cfg, interpret)

    x = rms_norm(x, params["final_norm"], cfg.rms_norm_eps)
    logits = x @ params["lm_head"]
    return logits


# --------------------------------------------------------------------------- #
# Parameter initialization
# --------------------------------------------------------------------------- #
def init_params(key: jax.Array, cfg: ModelConfig, dtype=jnp.float32) -> Params:
    """Randomly initialize the model parameters (no training)."""

    def normal(k, shape, scale):
        return (jax.random.normal(k, shape, dtype=jnp.float32) * scale).astype(dtype)

    keys = iter(jax.random.split(key, 4 + cfg.n_layers * 7))

    params: Params = {}
    params["embed"] = normal(next(keys), (cfg.vocab_size, cfg.dim), 0.02)

    layers = []
    proj_scale = 1.0 / (cfg.dim ** 0.5)
    for _ in range(cfg.n_layers):
        layer = {
            "attn_norm": jnp.ones((cfg.dim,), dtype),
            "ffn_norm": jnp.ones((cfg.dim,), dtype),
            "wq": normal(next(keys), (cfg.dim, cfg.dim), proj_scale),
            "wk": normal(next(keys), (cfg.dim, cfg.dim), proj_scale),
            "wv": normal(next(keys), (cfg.dim, cfg.dim), proj_scale),
            "wo": normal(next(keys), (cfg.dim, cfg.dim), proj_scale),
            "w_gate": normal(next(keys), (cfg.dim, cfg.ffn_hidden), proj_scale),
            "w_up": normal(next(keys), (cfg.dim, cfg.ffn_hidden), proj_scale),
            "w_down": normal(next(keys), (cfg.ffn_hidden, cfg.dim), 1.0 / (cfg.ffn_hidden ** 0.5)),
        }
        layers.append(layer)
    params["layers"] = layers

    params["final_norm"] = jnp.ones((cfg.dim,), dtype)
    params["lm_head"] = normal(next(keys), (cfg.dim, cfg.vocab_size), proj_scale)
    return params


# --------------------------------------------------------------------------- #
# Greedy generation
# --------------------------------------------------------------------------- #
def generate(
    params: Params,
    prompt_ids: jax.Array,
    max_new_tokens: int,
    cfg: ModelConfig,
    interpret: bool = False,
) -> jax.Array:
    """Greedy autoregressive generation.

    For clarity this re-runs the causal model over the growing sequence each
    step (the simplest correct form). A KV-cache would avoid recomputation and
    is noted as a future extension in the README.

    Args:
        prompt_ids: int array of shape ``[batch, prompt_len]``.
        max_new_tokens: number of tokens to append.

    Returns:
        int array of shape ``[batch, prompt_len + max_new_tokens]``.
    """
    tokens = prompt_ids
    for _ in range(max_new_tokens):
        logits = forward(params, tokens, cfg, interpret=interpret)
        next_tok = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        tokens = jnp.concatenate([tokens, next_tok], axis=1)
    return tokens
