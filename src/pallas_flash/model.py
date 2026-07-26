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

from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp

from .config import ModelConfig
from .flash_attention import flash_attention, flash_attention_decode

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
def project_qkv(params: Params, h: jax.Array, cfg: ModelConfig):
    """Project normed hidden states to head-shaped Q, K, V (pre-RoPE).

    Q gets ``n_heads`` heads; K and V get ``n_kv_heads`` heads (== n_heads for
    plain MHA, fewer for grouped-query / multi-query attention). Returns arrays
    shaped ``[batch, heads, seq, head_dim]``.
    """
    batch, seq, _ = h.shape

    def split(t: jax.Array, n_heads: int) -> jax.Array:
        return t.reshape(batch, seq, n_heads, cfg.head_dim).transpose(0, 2, 1, 3)

    q = split(h @ params["wq"], cfg.n_heads)
    k = split(h @ params["wk"], cfg.n_kv_heads)
    v = split(h @ params["wv"], cfg.n_kv_heads)
    return q, k, v


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
    q, k, v = project_qkv(params, h, cfg)
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

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
            "wk": normal(next(keys), (cfg.dim, cfg.kv_dim), proj_scale),
            "wv": normal(next(keys), (cfg.dim, cfg.kv_dim), proj_scale),
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
    step (the simplest correct form). See :func:`generate_cached` for the
    KV-cache version that avoids recomputing the prefix every step.

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


# --------------------------------------------------------------------------- #
# KV-cache decoding
# --------------------------------------------------------------------------- #
# The generation loop above recomputes the whole prefix at every step -- O(T^2)
# work to emit T tokens. Real inference caches each layer's K and V so that a
# decode step only computes the *new* token's query/key/value and attends it
# against the cache: O(T) projections and a single-row attention per step.
#
# We reuse the very same Pallas kernel:
#   * prefill runs the prompt with ``causal=True`` (queries and keys aligned);
#   * each decode step attends the single new query against the cache through
#     ``flash_attention_decode``, which masks and skips everything past the
#     filled length -- the new token legitimately sees every cached position.
#
# The cache is **preallocated** to a fixed capacity and written in place, so the
# decode loop does no reallocation and the kernel's cost tracks the filled
# length rather than the capacity.
Cache = list  # per-layer list of {"k": [b, n_kv, cache_size, d], "v": same}


def _attend(layer: Params, q, k, v, cfg: ModelConfig, interpret: bool, causal: bool):
    """Run the Pallas attention and the output projection for one block."""
    batch, _, seq_q, _ = q.shape
    attn = flash_attention(
        q, k, v, causal=causal, block_q=cfg.block_q, block_k=cfg.block_k, interpret=interpret
    )
    attn = attn.transpose(0, 2, 1, 3).reshape(batch, seq_q, cfg.dim)
    return attn @ layer["wo"]


def _rope_slice(pos_start: int, length: int, cfg: ModelConfig, dtype):
    """cos/sin tables for absolute positions ``[pos_start, pos_start+length)``."""
    cos, sin = precompute_rope(pos_start + length, cfg.head_dim, cfg.rope_theta)
    return cos[pos_start:].astype(dtype), sin[pos_start:].astype(dtype)


def init_kv_cache(
    cfg: ModelConfig, batch: int, cache_size: int, dtype=jnp.float32
) -> Cache:
    """Allocate an empty fixed-size KV cache for every layer."""
    shape = (batch, cfg.n_kv_heads, cache_size, cfg.head_dim)
    return [
        {"k": jnp.zeros(shape, dtype), "v": jnp.zeros(shape, dtype)}
        for _ in range(cfg.n_layers)
    ]


def _cache_write(layer_cache: dict, k, v, start: int) -> dict:
    """Write ``k``/``v`` (length L) into the cache at ``[start, start+L)``."""
    return {
        "k": jax.lax.dynamic_update_slice_in_dim(layer_cache["k"], k, start, 2),
        "v": jax.lax.dynamic_update_slice_in_dim(layer_cache["v"], v, start, 2),
    }


def prefill(
    params: Params,
    tokens: jax.Array,
    cfg: ModelConfig,
    interpret: bool = False,
    cache_size: Optional[int] = None,
):
    """Run the prompt and build the KV cache.

    Args:
        tokens: int array of shape ``[batch, prompt_len]``.
        cache_size: capacity of the preallocated cache. Defaults to
            ``cfg.max_seq_len``; must be at least ``prompt_len``.

    Returns ``(logits, cache)`` where ``logits`` has shape
    ``[batch, prompt_len, vocab_size]`` and ``cache`` holds the post-RoPE K/V of
    every layer in fixed-capacity buffers.
    """
    batch, seq = tokens.shape
    if cache_size is None:
        cache_size = cfg.max_seq_len
    if cache_size < seq:
        raise ValueError(f"cache_size ({cache_size}) must be >= prompt length ({seq})")

    x = params["embed"][tokens]
    cos, sin = _rope_slice(0, seq, cfg, x.dtype)

    cache = init_kv_cache(cfg, batch, cache_size, x.dtype)
    for idx, layer in enumerate(params["layers"]):
        h = rms_norm(x, layer["attn_norm"], cfg.rms_norm_eps)
        q, k, v = project_qkv(layer, h, cfg)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        x = x + _attend(layer, q, k, v, cfg, interpret, causal=True)
        cache[idx] = _cache_write(cache[idx], k, v, 0)
        h = rms_norm(x, layer["ffn_norm"], cfg.rms_norm_eps)
        x = x + swiglu_mlp(h, layer["w_gate"], layer["w_up"], layer["w_down"])

    x = rms_norm(x, params["final_norm"], cfg.rms_norm_eps)
    return x @ params["lm_head"], cache


def decode_step(
    params: Params,
    token: jax.Array,
    cache: Cache,
    pos: int,
    cfg: ModelConfig,
    interpret: bool = False,
):
    """Advance generation by one token.

    The new token's K/V are written into the cache at slot ``pos`` and the
    single-query decode kernel attends over slots ``[0, pos]``.

    Args:
        token: int array of shape ``[batch, 1]`` -- the token at absolute
            position ``pos``.
        cache: the KV cache holding positions ``[0, pos)``.
        pos: absolute position of ``token``.

    Returns ``(logits, new_cache)`` with ``logits`` of shape
    ``[batch, 1, vocab_size]`` and the cache extended to include ``pos``.
    """
    cache_size = cache[0]["k"].shape[2]
    if pos >= cache_size:
        raise ValueError(f"position {pos} exceeds cache capacity {cache_size}")

    x = params["embed"][token]                       # [b, 1, dim]
    cos, sin = _rope_slice(pos, 1, cfg, x.dtype)      # RoPE at the new position

    new_cache: Cache = []
    for layer, layer_cache in zip(params["layers"], cache):
        h = rms_norm(x, layer["attn_norm"], cfg.rms_norm_eps)
        q, k, v = project_qkv(layer, h, cfg)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        layer_cache = _cache_write(layer_cache, k, v, pos)
        # Single new query attends slots [0, pos] of the fixed-size cache.
        attn = flash_attention_decode(
            q, layer_cache["k"], layer_cache["v"], pos + 1,
            block_k=cfg.block_k, interpret=interpret,
        )
        batch = x.shape[0]
        attn = attn.transpose(0, 2, 1, 3).reshape(batch, 1, cfg.dim)
        x = x + attn @ layer["wo"]
        new_cache.append(layer_cache)
        h = rms_norm(x, layer["ffn_norm"], cfg.rms_norm_eps)
        x = x + swiglu_mlp(h, layer["w_gate"], layer["w_up"], layer["w_down"])

    x = rms_norm(x, params["final_norm"], cfg.rms_norm_eps)
    return x @ params["lm_head"], new_cache


def generate_cached(
    params: Params,
    prompt_ids: jax.Array,
    max_new_tokens: int,
    cfg: ModelConfig,
    interpret: bool = False,
) -> jax.Array:
    """Greedy generation using a fixed-size KV cache.

    Produces the same tokens as :func:`generate` but only computes the new
    token's attention each step instead of re-running the whole prefix. The
    cache is allocated once, up front, to exactly the length this call needs.
    """
    batch, prompt_len = prompt_ids.shape
    cache_size = prompt_len + max_new_tokens
    logits, cache = prefill(
        params, prompt_ids, cfg, interpret=interpret, cache_size=cache_size
    )
    next_tok = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
    tokens = jnp.concatenate([prompt_ids, next_tok], axis=1)

    for i in range(max_new_tokens - 1):
        pos = prompt_len + i                          # position of next_tok
        logits, cache = decode_step(params, next_tok, cache, pos, cfg, interpret=interpret)
        next_tok = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        tokens = jnp.concatenate([tokens, next_tok], axis=1)

    return tokens
