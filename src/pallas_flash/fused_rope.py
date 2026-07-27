"""FlashAttention with **rotary position embeddings fused into the kernel**.

In the unfused path, RoPE is applied by separate XLA ops before attention:

    q = apply_rope(q, cos, sin)     # reads q from HBM, writes rotated q to HBM
    k = apply_rope(k, cos, sin)     # same for k
    out = flash_attention(q, k, v)  # reads them back from HBM

That is two extra full round-trips of Q and K through HBM — pure bandwidth, no
math. Since the kernel already streams Q and K blocks into VMEM, it can rotate
them *there*, while they are in registers, and never write the rotated copies
out at all. On a bandwidth-bound TPU this is the kind of fusion that XLA cannot
do for you across a custom kernel boundary.

The cos/sin tables are passed as ordinary inputs with their own ``BlockSpec``s,
so each grid step only pulls in the rows it needs:

    q_cos/q_sin : [seq_len_q, head_dim]   indexed by the query block
    k_cos/k_sin : [seq_len_k, head_dim]   indexed by the kv block

Keeping the query and key tables separate means this also covers decoding, where
the single query sits at absolute position ``pos`` while the keys span
``[0, pos]`` — the two tables simply carry different positions.

This is a forward-only (inference) kernel; for training use the differentiable
``flash_attention`` in :mod:`pallas_flash.flash_attention` with RoPE applied
outside.
"""

from __future__ import annotations

import functools
from typing import Optional

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .flash_attention import _NEG_INF, _iota_col


def _rotate_half(x: jax.Array) -> jax.Array:
    """``[x1, x2] -> [-x2, x1]`` over the last (feature) dimension."""
    half = x.shape[-1] // 2
    return jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _apply_rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    return x * cos + _rotate_half(x) * sin


def _flash_attention_rope_kernel(
    q_ref, k_ref, v_ref, q_cos_ref, q_sin_ref, k_cos_ref, k_sin_ref,
    o_ref, m_scratch, l_scratch, acc_scratch,
    *, sm_scale, causal, block_q, block_k, seq_len_q, kv_len,
):
    """Online-softmax attention that rotates Q/K blocks inside VMEM."""
    q_block_idx = pl.program_id(2)
    kv_block_idx = pl.program_id(3)
    num_kv_blocks = pl.num_programs(3)

    @pl.when(kv_block_idx == 0)
    def _init():
        m_scratch[...] = jnp.full_like(m_scratch, _NEG_INF)
        l_scratch[...] = jnp.zeros_like(l_scratch)
        acc_scratch[...] = jnp.zeros_like(acc_scratch)

    def _do_block():
        q = q_ref[0, 0].astype(jnp.float32)
        k = k_ref[0, 0].astype(jnp.float32)
        v = v_ref[0, 0].astype(jnp.float32)

        # --- the fusion: rotate while the blocks are already in VMEM ---------
        q = _apply_rope(q, q_cos_ref[...].astype(jnp.float32),
                        q_sin_ref[...].astype(jnp.float32))
        k = _apply_rope(k, k_cos_ref[...].astype(jnp.float32),
                        k_sin_ref[...].astype(jnp.float32))

        kv_valid = _iota_col(block_k, kv_block_idx * block_k) < kv_len
        k = jnp.where(kv_valid, k, 0.0)
        v = jnp.where(kv_valid, v, 0.0)

        s = jnp.dot(q, k.T, preferred_element_type=jnp.float32) * sm_scale
        q_pos = q_block_idx * block_q + jax.lax.broadcasted_iota(
            jnp.int32, (block_q, block_k), 0)
        k_pos = kv_block_idx * block_k + jax.lax.broadcasted_iota(
            jnp.int32, (block_q, block_k), 1)
        mask = k_pos < kv_len
        if causal:
            mask = jnp.logical_and(mask, q_pos >= k_pos)
        s = jnp.where(mask, s, _NEG_INF)

        m_prev = m_scratch[...]
        m_new = jnp.maximum(m_prev, jnp.max(s, axis=-1, keepdims=True))
        p = jnp.exp(s - m_new)
        alpha = jnp.exp(m_prev - m_new)
        l_scratch[...] = alpha * l_scratch[...] + jnp.sum(p, axis=-1, keepdims=True)
        acc_scratch[...] = acc_scratch[...] * alpha + jnp.dot(
            p, v, preferred_element_type=jnp.float32)
        m_scratch[...] = m_new

    first_k_pos = kv_block_idx * block_k
    live = first_k_pos < kv_len
    if causal:
        live = jnp.logical_and(live, q_block_idx * block_q + (block_q - 1) >= first_k_pos)
    pl.when(live)(_do_block)

    @pl.when(kv_block_idx == num_kv_blocks - 1)
    def _finalize():
        l = l_scratch[...]
        safe_l = jnp.where(l == 0.0, 1.0, l)
        o_ref[0, 0] = (acc_scratch[...] / safe_l).astype(o_ref.dtype)


def flash_attention_rope(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    q_cos: jax.Array,
    q_sin: jax.Array,
    k_cos: Optional[jax.Array] = None,
    k_sin: Optional[jax.Array] = None,
    *,
    causal: bool = False,
    sm_scale: Optional[float] = None,
    block_q: int = 128,
    block_k: int = 128,
    kv_len: Optional[int] = None,
    interpret: bool = False,
) -> jax.Array:
    """Attention with RoPE applied to Q and K *inside* the kernel.

    Numerically equivalent to ``flash_attention(apply_rope(q), apply_rope(k), v)``
    but without materializing the rotated Q and K in HBM.

    Args:
        q: ``[batch, num_heads, seq_len_q, head_dim]``.
        k, v: ``[batch, num_kv_heads, seq_len_k, head_dim]`` (GQA/MQA supported).
        q_cos, q_sin: ``[seq_len_q, head_dim]`` rotary tables for the query
            positions.
        k_cos, k_sin: ``[seq_len_k, head_dim]`` rotary tables for the key
            positions. Default to ``q_cos``/``q_sin`` when the query and key
            sequences coincide (ordinary self-attention).
        causal: apply a causal mask.
        sm_scale: softmax scale; defaults to ``1 / sqrt(head_dim)``.
        block_q, block_k: VMEM tile sizes.
        kv_len: number of valid key positions (see ``flash_attention``).
        interpret: run in Pallas interpret mode (CPU).

    Returns:
        ``[batch, num_heads, seq_len_q, head_dim]``.
    """
    if k_cos is None or k_sin is None:
        if k.shape[2] != q.shape[2]:
            raise ValueError(
                "k_cos/k_sin are required when the key sequence length differs "
                "from the query sequence length"
            )
        k_cos, k_sin = q_cos, q_sin

    batch, num_heads, seq_len_q, head_dim = q.shape
    num_kv_heads, seq_len_k = k.shape[1], k.shape[2]
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads ({num_heads}) must be a multiple of num_kv_heads ({num_kv_heads})"
        )
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    for name, tbl, want in (("q_cos", q_cos, seq_len_q), ("q_sin", q_sin, seq_len_q),
                            ("k_cos", k_cos, seq_len_k), ("k_sin", k_sin, seq_len_k)):
        if tbl.shape != (want, head_dim):
            raise ValueError(f"{name} must have shape {(want, head_dim)}, got {tbl.shape}")

    q_per_kv = num_heads // num_kv_heads
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)
    if kv_len is None:
        kv_len = seq_len_k
    elif not 0 < kv_len <= seq_len_k:
        raise ValueError(f"kv_len must be in (0, {seq_len_k}], got {kv_len}")

    grid = (batch, num_heads, pl.cdiv(seq_len_q, block_q), pl.cdiv(seq_len_k, block_k))
    qs = pl.BlockSpec((1, 1, block_q, head_dim), lambda b, h, i, j: (b, h, i, 0))
    ks = pl.BlockSpec((1, 1, block_k, head_dim),
                      lambda b, h, i, j: (b, h // q_per_kv, j, 0))
    # RoPE tables are 2-D [seq, head_dim]; the query tables follow the query
    # block index and the key tables the kv block index.
    q_tbl = pl.BlockSpec((block_q, head_dim), lambda b, h, i, j: (i, 0))
    k_tbl = pl.BlockSpec((block_k, head_dim), lambda b, h, i, j: (j, 0))

    kernel = functools.partial(
        _flash_attention_rope_kernel, sm_scale=sm_scale, causal=causal,
        block_q=block_q, block_k=block_k, seq_len_q=seq_len_q, kv_len=kv_len,
    )

    return pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=[qs, ks, ks, q_tbl, q_tbl, k_tbl, k_tbl],
        out_specs=qs,
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        scratch_shapes=[
            pltpu.VMEM((block_q, 1), jnp.float32),
            pltpu.VMEM((block_q, 1), jnp.float32),
            pltpu.VMEM((block_q, head_dim), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "arbitrary"),
        ),
        interpret=interpret,
        name="flash_attention_rope_fwd",
    )(q, k, v, q_cos, q_sin, k_cos, k_sin)
