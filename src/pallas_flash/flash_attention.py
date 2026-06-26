"""A FlashAttention forward kernel written in JAX Pallas for TPU.

This is the heart of the project: instead of letting the XLA compiler fuse a
naive ``softmax(Q Kᵀ) V`` (which materializes the full ``[seq, seq]`` attention
matrix in HBM), we hand-write a kernel that:

1. **Tiles** Q, K and V into blocks and maps each block from HBM into the TPU's
   small, fast VMEM. The mapping is expressed explicitly through Pallas
   ``BlockSpec`` ``index_map`` functions (see ``flash_attention`` below) — this
   is the "manual HBM -> VMEM management" lever that a high-level Keras/PyTorch
   program never touches.
2. Runs the **online-softmax** recurrence (Milakov & Gimelshein; Dao et al.,
   FlashAttention) so the attention matrix is never fully materialized. Running
   statistics ``m`` (row max), ``l`` (row denominator) and ``acc`` (the
   weighted value sum) live in VMEM scratch and are updated block-by-block.
3. Supports **causal masking** with block-level skipping: kv blocks that lie
   entirely in the future of a query block are never even loaded/computed.

The exact same source runs on CPU (``interpret=True``) for testing and on a
real TPU v3-8 (``interpret=False``) for performance.

Grid layout: ``(batch, num_heads, num_q_blocks, num_kv_blocks)``. The grid is
executed sequentially in lexicographic order, so for a fixed
``(batch, head, q_block)`` the kv blocks are visited in order and we accumulate
into the same output block safely.
"""

from __future__ import annotations

import functools
from typing import Optional

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# A large negative value used to mask out (set to ~zero probability) entries
# that a query position is not allowed to attend to. Using a finite number
# rather than -inf keeps the online softmax numerically well-behaved.
_NEG_INF = -1e30


def _flash_attention_kernel(
    q_ref,          # VMEM: (1, 1, block_q, head_dim)
    k_ref,          # VMEM: (1, 1, block_k, head_dim)
    v_ref,          # VMEM: (1, 1, block_k, head_dim)
    o_ref,          # VMEM: (1, 1, block_q, head_dim)  -- output block
    m_scratch,      # VMEM: (block_q, 1)  running row max
    l_scratch,      # VMEM: (block_q, 1)  running row denominator
    acc_scratch,    # VMEM: (block_q, head_dim) running weighted value sum
    *,
    sm_scale: float,
    causal: bool,
    block_q: int,
    block_k: int,
    seq_len_q: int,
    seq_len_k: int,
):
    """Body executed once per ``(batch, head, q_block, kv_block)`` grid point."""
    q_block_idx = pl.program_id(2)
    kv_block_idx = pl.program_id(3)
    num_kv_blocks = pl.num_programs(3)

    # --- initialize running statistics on the first kv block -----------------
    @pl.when(kv_block_idx == 0)
    def _init():
        m_scratch[...] = jnp.full_like(m_scratch, _NEG_INF)
        l_scratch[...] = jnp.zeros_like(l_scratch)
        acc_scratch[...] = jnp.zeros_like(acc_scratch)

    def _do_block():
        q = q_ref[0, 0].astype(jnp.float32)            # (block_q, d)
        k = k_ref[0, 0].astype(jnp.float32)            # (block_k, d)
        v = v_ref[0, 0].astype(jnp.float32)            # (block_k, d)

        # The last kv block may be padded when seq_len_k is not a multiple of
        # block_k. Zero those rows so they cannot inject NaN/garbage into the
        # matmuls (notably ``0 * NaN = NaN`` would otherwise poison ``p @ v``).
        kv_idx = kv_block_idx * block_k + jax.lax.broadcasted_iota(
            jnp.int32, (block_k, 1), 0
        )
        kv_valid = kv_idx < seq_len_k                  # (block_k, 1)
        k = jnp.where(kv_valid, k, 0.0)
        v = jnp.where(kv_valid, v, 0.0)

        # s = scale * Q Kᵀ  -> (block_q, block_k), all in VMEM, never in HBM.
        s = jnp.dot(q, k.T, preferred_element_type=jnp.float32) * sm_scale

        # Absolute token positions of the rows (queries) and columns (keys) of
        # this tile, used for masking.
        q_pos = q_block_idx * block_q + jax.lax.broadcasted_iota(
            jnp.int32, (block_q, block_k), 0
        )
        k_pos = kv_block_idx * block_k + jax.lax.broadcasted_iota(
            jnp.int32, (block_q, block_k), 1
        )

        # Always mask keys that fall past the real sequence length (handles
        # sequence lengths that are not a multiple of the block size, where the
        # last kv block is padded).
        mask = k_pos < seq_len_k
        if causal:
            mask = jnp.logical_and(mask, q_pos >= k_pos)
        s = jnp.where(mask, s, _NEG_INF)

        # --- online softmax update ------------------------------------------
        m_prev = m_scratch[...]                         # (block_q, 1)
        m_cur = jnp.max(s, axis=-1, keepdims=True)      # (block_q, 1)
        m_new = jnp.maximum(m_prev, m_cur)

        p = jnp.exp(s - m_new)                          # (block_q, block_k)
        alpha = jnp.exp(m_prev - m_new)                 # rescale prior stats

        l_scratch[...] = alpha * l_scratch[...] + jnp.sum(p, axis=-1, keepdims=True)
        acc_scratch[...] = acc_scratch[...] * alpha + jnp.dot(
            p, v, preferred_element_type=jnp.float32
        )
        m_scratch[...] = m_new

    # --- causal block skipping ----------------------------------------------
    # The largest query position in this q block is (q_block_idx+1)*block_q - 1.
    # The smallest key position in this kv block is kv_block_idx*block_k. If the
    # former is below the latter, every entry is masked, so skip the block
    # entirely (no load, no matmul).
    if causal:
        last_q_pos = q_block_idx * block_q + (block_q - 1)
        first_k_pos = kv_block_idx * block_k
        pl.when(last_q_pos >= first_k_pos)(_do_block)
    else:
        _do_block()

    # --- write the normalized output on the last kv block --------------------
    @pl.when(kv_block_idx == num_kv_blocks - 1)
    def _finalize():
        # Guard against a fully-masked row (l == 0) to avoid NaNs; such rows do
        # not correspond to valid tokens.
        l = jnp.where(l_scratch[...] == 0.0, 1.0, l_scratch[...])
        out = (acc_scratch[...] / l).astype(o_ref.dtype)
        o_ref[0, 0] = out


def flash_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    causal: bool = False,
    sm_scale: Optional[float] = None,
    block_q: int = 128,
    block_k: int = 128,
    interpret: bool = False,
) -> jax.Array:
    """Compute FlashAttention with a custom Pallas TPU kernel.

    Args:
        q, k, v: arrays of shape ``[batch, num_heads, seq_len, head_dim]``.
            ``q`` may have a different sequence length than ``k``/``v``.
        causal: if ``True``, apply a causal (lower-triangular) mask.
        sm_scale: softmax scale; defaults to ``1 / sqrt(head_dim)``.
        block_q: query block size (rows loaded into VMEM at a time).
        block_k: key/value block size.
        interpret: run the kernel in Pallas interpret mode (CPU) instead of
            compiling for TPU. Used by the test-suite so the identical kernel
            can be validated without TPU hardware.

    Returns:
        Array of shape ``[batch, num_heads, seq_len_q, head_dim]``.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must be rank-4 [batch, heads, seq, head_dim] arrays")

    batch, num_heads, seq_len_q, head_dim = q.shape
    seq_len_k = k.shape[2]
    if k.shape != v.shape:
        raise ValueError("k and v must have the same shape")
    if k.shape[0] != batch or k.shape[1] != num_heads or k.shape[3] != head_dim:
        raise ValueError("q and k/v must share batch, heads and head_dim")

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)

    num_q_blocks = pl.cdiv(seq_len_q, block_q)
    num_kv_blocks = pl.cdiv(seq_len_k, block_k)

    grid = (batch, num_heads, num_q_blocks, num_kv_blocks)

    # BlockSpec index_map: given a grid point, return which block of the HBM
    # array to stream into VMEM. This explicit mapping is the manual control
    # over data movement that the project is about.
    #   q/o block index: (batch, head, q_block, 0)   -> depends on q_block only
    #   k/v block index: (batch, head, kv_block, 0)  -> depends on kv_block only
    q_spec = pl.BlockSpec((1, 1, block_q, head_dim), lambda b, h, i, j: (b, h, i, 0))
    k_spec = pl.BlockSpec((1, 1, block_k, head_dim), lambda b, h, i, j: (b, h, j, 0))
    v_spec = k_spec
    o_spec = pl.BlockSpec((1, 1, block_q, head_dim), lambda b, h, i, j: (b, h, i, 0))

    kernel = functools.partial(
        _flash_attention_kernel,
        sm_scale=sm_scale,
        causal=causal,
        block_q=block_q,
        block_k=block_k,
        seq_len_q=seq_len_q,
        seq_len_k=seq_len_k,
    )

    out = pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=[q_spec, k_spec, v_spec],
        out_specs=o_spec,
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        scratch_shapes=[
            pltpu.VMEM((block_q, 1), jnp.float32),       # m
            pltpu.VMEM((block_q, 1), jnp.float32),       # l
            pltpu.VMEM((block_q, head_dim), jnp.float32),  # acc
        ],
        compiler_params=pltpu.CompilerParams(
            # batch / head / q-block are independent; kv-blocks must run in
            # order because they update shared scratch.
            dimension_semantics=("parallel", "parallel", "parallel", "arbitrary"),
        ),
        interpret=interpret,
        name="flash_attention_fwd",
    )(q, k, v)

    return out
