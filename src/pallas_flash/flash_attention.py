"""A FlashAttention kernel written in JAX Pallas for TPU (forward + backward).

This is the heart of the project: instead of letting the XLA compiler fuse a
naive ``softmax(Q Kᵀ) V`` (which materializes the full ``[seq, seq]`` attention
matrix in HBM), we hand-write a kernel that:

1. **Tiles** Q, K and V into blocks and maps each block from HBM into the TPU's
   small, fast VMEM. The mapping is expressed explicitly through Pallas
   ``BlockSpec`` ``index_map`` functions — this is the "manual HBM -> VMEM
   management" lever that a high-level Keras/PyTorch program never touches.
2. Runs the **online-softmax** recurrence (Milakov & Gimelshein; Dao et al.,
   FlashAttention) so the attention matrix is never fully materialized. Running
   statistics ``m`` (row max), ``l`` (row denominator) and ``acc`` (the
   weighted value sum) live in VMEM scratch and are updated block-by-block.
3. Supports **causal masking** with block-level skipping: kv blocks that lie
   entirely in the future of a query block are never even loaded/computed.
4. Supports **grouped-query attention** (GQA/MQA) by mapping several query
   heads onto one key/value head inside the ``BlockSpec`` index map.
5. Supports a **fixed-size KV cache** via ``kv_len``: only the first
   ``kv_len`` cache positions are attended, and whole blocks past that point
   are skipped. ``flash_attention_decode`` wraps this for single-token decode.
6. Is **differentiable**: ``flash_attention`` is a ``jax.custom_vjp`` whose
   backward pass is itself a pair of Pallas kernels (see below), so the full
   ``[seq, seq]`` matrix is never materialized in the backward pass either.

The exact same source runs on CPU (``interpret=True``) for testing and on a
real TPU v3-8 (``interpret=False``) for performance, in float32 or bfloat16
(inputs are up-cast to float32 inside the kernel, so accumulation stays exact
while HBM traffic halves).

Forward grid layout: ``(batch, num_heads, num_q_blocks, num_kv_blocks)``. The
grid is executed sequentially in lexicographic order, so for a fixed
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


def _iota_col(rows: int, offset: int) -> jax.Array:
    """Column vector of absolute positions ``[offset, offset + rows)``."""
    return offset + jax.lax.broadcasted_iota(jnp.int32, (rows, 1), 0)


# --------------------------------------------------------------------------- #
# Forward kernel
# --------------------------------------------------------------------------- #
def _flash_attention_kernel(
    q_ref,          # VMEM: (1, 1, block_q, head_dim)
    k_ref,          # VMEM: (1, 1, block_k, head_dim)
    v_ref,          # VMEM: (1, 1, block_k, head_dim)
    o_ref,          # VMEM: (1, 1, block_q, head_dim)  -- output block
    lse_ref,        # VMEM: (1, 1, block_q, 1)         -- log-sum-exp (bwd residual)
    m_scratch,      # VMEM: (block_q, 1)  running row max
    l_scratch,      # VMEM: (block_q, 1)  running row denominator
    acc_scratch,    # VMEM: (block_q, head_dim) running weighted value sum
    *,
    sm_scale: float,
    causal: bool,
    block_q: int,
    block_k: int,
    seq_len_q: int,
    kv_len: int,
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

        # The last kv block may be padded when kv_len is not a multiple of
        # block_k. Zero those rows so they cannot inject NaN/garbage into the
        # matmuls (notably ``0 * NaN = NaN`` would otherwise poison ``p @ v``).
        kv_valid = _iota_col(block_k, kv_block_idx * block_k) < kv_len
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

        # Always mask keys past the valid cache/sequence length (handles both a
        # partially-filled KV cache and sequence lengths that are not a multiple
        # of the block size).
        mask = k_pos < kv_len
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

    # --- block skipping ------------------------------------------------------
    # 1. Causal: the largest query position in this q block is
    #    (q_block_idx+1)*block_q - 1; the smallest key position in this kv block
    #    is kv_block_idx*block_k. If the former is below the latter every entry
    #    is masked, so skip the block entirely (no load, no matmul).
    # 2. Cache length: a kv block starting at or past kv_len is all padding.
    first_k_pos = kv_block_idx * block_k
    live = first_k_pos < kv_len
    if causal:
        last_q_pos = q_block_idx * block_q + (block_q - 1)
        live = jnp.logical_and(live, last_q_pos >= first_k_pos)
    pl.when(live)(_do_block)

    # --- write the normalized output on the last kv block --------------------
    @pl.when(kv_block_idx == num_kv_blocks - 1)
    def _finalize():
        l = l_scratch[...]
        # Guard against a fully-masked row (l == 0) to avoid NaNs; such rows do
        # not correspond to valid tokens. lse is written as 0 there so the
        # backward pass (which re-applies the mask) stays finite.
        safe_l = jnp.where(l == 0.0, 1.0, l)
        o_ref[0, 0] = (acc_scratch[...] / safe_l).astype(o_ref.dtype)
        lse = m_scratch[...] + jnp.log(safe_l)
        lse_ref[0, 0] = jnp.where(l == 0.0, 0.0, lse).astype(lse_ref.dtype)


def _flash_attention_call(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    causal: bool,
    sm_scale: float,
    block_q: int,
    block_k: int,
    kv_len: int,
    interpret: bool,
):
    """Launch the forward kernel. Returns ``(out, lse)``."""
    batch, num_heads, seq_len_q, head_dim = q.shape
    num_kv_heads, seq_len_k = k.shape[1], k.shape[2]
    q_per_kv = num_heads // num_kv_heads

    grid = (batch, num_heads, pl.cdiv(seq_len_q, block_q), pl.cdiv(seq_len_k, block_k))

    # BlockSpec index_map: given a grid point, return which block of the HBM
    # array to stream into VMEM. This explicit mapping is the manual control
    # over data movement that the project is about.
    #   q/o block index: (batch, head, q_block, 0)   -> depends on q_block only
    #   k/v block index: (batch, head // q_per_kv, kv_block, 0)
    #       -> maps each query head to its shared key/value head (GQA/MQA).
    q_spec = pl.BlockSpec((1, 1, block_q, head_dim), lambda b, h, i, j: (b, h, i, 0))
    k_spec = pl.BlockSpec(
        (1, 1, block_k, head_dim), lambda b, h, i, j: (b, h // q_per_kv, j, 0)
    )
    o_spec = pl.BlockSpec((1, 1, block_q, head_dim), lambda b, h, i, j: (b, h, i, 0))
    # lse is stored as [b, h, seq, 1] so the block's trailing dimension matches
    # the array exactly (TPU block-shape rule) while rows stay a multiple of 8.
    lse_spec = pl.BlockSpec((1, 1, block_q, 1), lambda b, h, i, j: (b, h, i, 0))

    kernel = functools.partial(
        _flash_attention_kernel,
        sm_scale=sm_scale,
        causal=causal,
        block_q=block_q,
        block_k=block_k,
        seq_len_q=seq_len_q,
        kv_len=kv_len,
    )

    return pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=[q_spec, k_spec, k_spec],
        out_specs=[o_spec, lse_spec],
        out_shape=[
            jax.ShapeDtypeStruct(q.shape, q.dtype),
            jax.ShapeDtypeStruct((batch, num_heads, seq_len_q, 1), jnp.float32),
        ],
        scratch_shapes=[
            pltpu.VMEM((block_q, 1), jnp.float32),         # m
            pltpu.VMEM((block_q, 1), jnp.float32),         # l
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


# --------------------------------------------------------------------------- #
# Backward kernels
# --------------------------------------------------------------------------- #
# With the forward log-sum-exp ``lse`` saved, the softmax probabilities can be
# recomputed block-by-block as ``p = exp(s - lse)`` without ever storing the
# [seq, seq] matrix. Using ``delta_i = sum_d o_id * do_id``:
#
#     dv_j = sum_i p_ij do_i
#     dp_ij = do_i · v_j
#     ds_ij = p_ij (dp_ij - delta_i)
#     dq_i = scale * sum_j ds_ij k_j
#     dk_j = scale * sum_i ds_ij q_i
#
# dQ wants an outer loop over query blocks; dK/dV want an outer loop over kv
# blocks. We therefore use two kernels with transposed grids.
def _bwd_common(q, k, v, do, lse, delta, *, q_block_idx, kv_block_idx, block_q,
                block_k, sm_scale, causal, seq_len_q, kv_len):
    """Recompute ``p`` and ``ds`` for one (q block, kv block) tile."""
    q_valid = _iota_col(block_q, q_block_idx * block_q) < seq_len_q
    kv_valid = _iota_col(block_k, kv_block_idx * block_k) < kv_len

    # Zero padded rows so garbage/NaN in out-of-range slots cannot propagate.
    q = jnp.where(q_valid, q, 0.0)
    do = jnp.where(q_valid, do, 0.0)
    lse = jnp.where(q_valid, lse, 0.0)
    delta = jnp.where(q_valid, delta, 0.0)
    k = jnp.where(kv_valid, k, 0.0)
    v = jnp.where(kv_valid, v, 0.0)

    s = jnp.dot(q, k.T, preferred_element_type=jnp.float32) * sm_scale
    q_pos = q_block_idx * block_q + jax.lax.broadcasted_iota(
        jnp.int32, (block_q, block_k), 0
    )
    k_pos = kv_block_idx * block_k + jax.lax.broadcasted_iota(
        jnp.int32, (block_q, block_k), 1
    )
    mask = jnp.logical_and(k_pos < kv_len, q_pos < seq_len_q)
    if causal:
        mask = jnp.logical_and(mask, q_pos >= k_pos)

    p = jnp.where(mask, jnp.exp(s - lse), 0.0)
    dp = jnp.dot(do, v.T, preferred_element_type=jnp.float32)
    ds = p * (dp - delta)
    return p, ds, q, do


def _flash_attention_bwd_dq_kernel(
    q_ref, k_ref, v_ref, do_ref, lse_ref, delta_ref, dq_ref, dq_scratch,
    *, sm_scale, causal, block_q, block_k, seq_len_q, kv_len,
):
    """dQ: outer loop over query blocks, inner (sequential) over kv blocks."""
    i = pl.program_id(2)
    j = pl.program_id(3)
    num_kv_blocks = pl.num_programs(3)

    @pl.when(j == 0)
    def _init():
        dq_scratch[...] = jnp.zeros_like(dq_scratch)

    def _do_block():
        _, ds, _, _ = _bwd_common(
            q_ref[0, 0].astype(jnp.float32),
            k_ref[0, 0].astype(jnp.float32),
            v_ref[0, 0].astype(jnp.float32),
            do_ref[0, 0].astype(jnp.float32),
            lse_ref[0, 0].astype(jnp.float32),
            delta_ref[0, 0].astype(jnp.float32),
            q_block_idx=i, kv_block_idx=j, block_q=block_q, block_k=block_k,
            sm_scale=sm_scale, causal=causal, seq_len_q=seq_len_q, kv_len=kv_len,
        )
        k = jnp.where(_iota_col(block_k, j * block_k) < kv_len,
                      k_ref[0, 0].astype(jnp.float32), 0.0)
        dq_scratch[...] += jnp.dot(ds, k, preferred_element_type=jnp.float32) * sm_scale

    live = j * block_k < kv_len
    if causal:
        live = jnp.logical_and(live, i * block_q + (block_q - 1) >= j * block_k)
    pl.when(live)(_do_block)

    @pl.when(j == num_kv_blocks - 1)
    def _finalize():
        dq_ref[0, 0] = dq_scratch[...].astype(dq_ref.dtype)


def _flash_attention_bwd_dkv_kernel(
    k_ref, v_ref, q_ref, do_ref, lse_ref, delta_ref, dk_ref, dv_ref,
    dk_scratch, dv_scratch,
    *, sm_scale, causal, block_q, block_k, seq_len_q, kv_len,
):
    """dK/dV: outer loop over kv blocks, inner (sequential) over query blocks."""
    j = pl.program_id(2)   # kv block
    i = pl.program_id(3)   # query block
    num_q_blocks = pl.num_programs(3)

    @pl.when(i == 0)
    def _init():
        dk_scratch[...] = jnp.zeros_like(dk_scratch)
        dv_scratch[...] = jnp.zeros_like(dv_scratch)

    def _do_block():
        p, ds, q, do = _bwd_common(
            q_ref[0, 0].astype(jnp.float32),
            k_ref[0, 0].astype(jnp.float32),
            v_ref[0, 0].astype(jnp.float32),
            do_ref[0, 0].astype(jnp.float32),
            lse_ref[0, 0].astype(jnp.float32),
            delta_ref[0, 0].astype(jnp.float32),
            q_block_idx=i, kv_block_idx=j, block_q=block_q, block_k=block_k,
            sm_scale=sm_scale, causal=causal, seq_len_q=seq_len_q, kv_len=kv_len,
        )
        dv_scratch[...] += jnp.dot(p.T, do, preferred_element_type=jnp.float32)
        dk_scratch[...] += jnp.dot(ds.T, q, preferred_element_type=jnp.float32) * sm_scale

    live = j * block_k < kv_len
    if causal:
        live = jnp.logical_and(live, i * block_q + (block_q - 1) >= j * block_k)
    pl.when(live)(_do_block)

    @pl.when(i == num_q_blocks - 1)
    def _finalize():
        dk_ref[0, 0] = dk_scratch[...].astype(dk_ref.dtype)
        dv_ref[0, 0] = dv_scratch[...].astype(dv_ref.dtype)


def _flash_attention_bwd_call(q, k, v, out, lse, do, *, causal, sm_scale,
                              block_q, block_k, kv_len, interpret):
    """Run both backward kernels and return ``(dq, dk, dv)``."""
    batch, num_heads, seq_len_q, head_dim = q.shape
    num_kv_heads, seq_len_k = k.shape[1], k.shape[2]
    q_per_kv = num_heads // num_kv_heads

    # delta_i = sum_d o_id * do_id -- a cheap elementwise reduction, kept
    # outside the kernels (FlashAttention calls this the backward "preprocess").
    delta = jnp.sum(
        out.astype(jnp.float32) * do.astype(jnp.float32), axis=-1, keepdims=True
    )

    num_q_blocks = pl.cdiv(seq_len_q, block_q)
    num_kv_blocks = pl.cdiv(seq_len_k, block_k)

    qs = (1, 1, block_q, head_dim)
    ks = (1, 1, block_k, head_dim)
    ss = (1, 1, block_q, 1)
    common = dict(
        sm_scale=sm_scale, causal=causal, block_q=block_q, block_k=block_k,
        seq_len_q=seq_len_q, kv_len=kv_len,
    )

    # --- dQ: grid (b, h, q_block, kv_block) ---------------------------------
    dq = pl.pallas_call(
        functools.partial(_flash_attention_bwd_dq_kernel, **common),
        grid=(batch, num_heads, num_q_blocks, num_kv_blocks),
        in_specs=[
            pl.BlockSpec(qs, lambda b, h, i, j: (b, h, i, 0)),                 # q
            pl.BlockSpec(ks, lambda b, h, i, j: (b, h // q_per_kv, j, 0)),     # k
            pl.BlockSpec(ks, lambda b, h, i, j: (b, h // q_per_kv, j, 0)),     # v
            pl.BlockSpec(qs, lambda b, h, i, j: (b, h, i, 0)),                 # do
            pl.BlockSpec(ss, lambda b, h, i, j: (b, h, i, 0)),                 # lse
            pl.BlockSpec(ss, lambda b, h, i, j: (b, h, i, 0)),                 # delta
        ],
        out_specs=pl.BlockSpec(qs, lambda b, h, i, j: (b, h, i, 0)),
        out_shape=jax.ShapeDtypeStruct(q.shape, jnp.float32),
        scratch_shapes=[pltpu.VMEM((block_q, head_dim), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "arbitrary"),
        ),
        interpret=interpret,
        name="flash_attention_bwd_dq",
    )(q, k, v, do, lse, delta)

    # --- dK/dV: grid (b, h, kv_block, q_block) ------------------------------
    # Outputs carry one entry per *query* head; for GQA the group is summed
    # afterwards, which keeps the kernel free of cross-head reductions.
    dk_per_head, dv_per_head = pl.pallas_call(
        functools.partial(_flash_attention_bwd_dkv_kernel, **common),
        grid=(batch, num_heads, num_kv_blocks, num_q_blocks),
        in_specs=[
            pl.BlockSpec(ks, lambda b, h, j, i: (b, h // q_per_kv, j, 0)),     # k
            pl.BlockSpec(ks, lambda b, h, j, i: (b, h // q_per_kv, j, 0)),     # v
            pl.BlockSpec(qs, lambda b, h, j, i: (b, h, i, 0)),                 # q
            pl.BlockSpec(qs, lambda b, h, j, i: (b, h, i, 0)),                 # do
            pl.BlockSpec(ss, lambda b, h, j, i: (b, h, i, 0)),                 # lse
            pl.BlockSpec(ss, lambda b, h, j, i: (b, h, i, 0)),                 # delta
        ],
        out_specs=[
            pl.BlockSpec(ks, lambda b, h, j, i: (b, h, j, 0)),
            pl.BlockSpec(ks, lambda b, h, j, i: (b, h, j, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((batch, num_heads, seq_len_k, head_dim), jnp.float32),
            jax.ShapeDtypeStruct((batch, num_heads, seq_len_k, head_dim), jnp.float32),
        ],
        scratch_shapes=[
            pltpu.VMEM((block_k, head_dim), jnp.float32),
            pltpu.VMEM((block_k, head_dim), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "arbitrary"),
        ),
        interpret=interpret,
        name="flash_attention_bwd_dkv",
    )(k, v, q, do, lse, delta)

    def group_sum(x):
        if q_per_kv == 1:
            return x
        return x.reshape(batch, num_kv_heads, q_per_kv, seq_len_k, head_dim).sum(axis=2)

    return (
        dq.astype(q.dtype),
        group_sum(dk_per_head).astype(k.dtype),
        group_sum(dv_per_head).astype(v.dtype),
    )


# --------------------------------------------------------------------------- #
# Differentiable public entry point
# --------------------------------------------------------------------------- #
@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 4, 5))
def _flash_attention_vjp(causal, sm_scale, block_q, block_k, kv_len, interpret,
                         q, k, v):
    out, _ = _flash_attention_call(
        q, k, v, causal=causal, sm_scale=sm_scale, block_q=block_q,
        block_k=block_k, kv_len=kv_len, interpret=interpret,
    )
    return out


def _flash_attention_vjp_fwd(causal, sm_scale, block_q, block_k, kv_len,
                             interpret, q, k, v):
    out, lse = _flash_attention_call(
        q, k, v, causal=causal, sm_scale=sm_scale, block_q=block_q,
        block_k=block_k, kv_len=kv_len, interpret=interpret,
    )
    return out, (q, k, v, out, lse)


def _flash_attention_vjp_bwd(causal, sm_scale, block_q, block_k, kv_len,
                             interpret, res, do):
    q, k, v, out, lse = res
    return _flash_attention_bwd_call(
        q, k, v, out, lse, do, causal=causal, sm_scale=sm_scale,
        block_q=block_q, block_k=block_k, kv_len=kv_len, interpret=interpret,
    )


_flash_attention_vjp.defvjp(_flash_attention_vjp_fwd, _flash_attention_vjp_bwd)


def flash_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    causal: bool = False,
    sm_scale: Optional[float] = None,
    block_q: int = 128,
    block_k: int = 128,
    kv_len: Optional[int] = None,
    interpret: bool = False,
) -> jax.Array:
    """Compute FlashAttention with a custom Pallas TPU kernel.

    Supports multi-head attention (MHA) as well as **grouped-query attention**
    (GQA) and multi-query attention (MQA): ``k``/``v`` may have fewer heads than
    ``q`` as long as ``num_heads`` is a multiple of ``num_kv_heads``. Each query
    head ``h`` reads from key/value head ``h // (num_heads // num_kv_heads)`` —
    the grouping is expressed purely through the K/V ``BlockSpec`` index map, so
    the KV cache is never physically replicated.

    The function is differentiable in ``q``, ``k`` and ``v``: its VJP is a pair
    of Pallas kernels that recompute the softmax from the saved log-sum-exp, so
    neither direction ever materializes the ``[seq, seq]`` matrix.

    Args:
        q: array of shape ``[batch, num_heads, seq_len_q, head_dim]``.
        k, v: arrays of shape ``[batch, num_kv_heads, seq_len_k, head_dim]``,
            where ``num_kv_heads`` divides ``num_heads``. For plain MHA,
            ``num_kv_heads == num_heads``. ``seq_len_k`` may differ from
            ``seq_len_q``.
        causal: if ``True``, apply a causal (lower-triangular) mask. Only
            meaningful when the query and key sequences are aligned from
            position 0 (e.g. self-attention or a causal prefill).
        sm_scale: softmax scale; defaults to ``1 / sqrt(head_dim)``.
        block_q: query block size (rows loaded into VMEM at a time).
        block_k: key/value block size.
        kv_len: number of valid key/value positions. Defaults to ``seq_len_k``.
            Pass a smaller value when ``k``/``v`` are a partially-filled
            fixed-size KV cache — entries at or past ``kv_len`` are masked out
            and whole blocks beyond it are skipped.
        interpret: run the kernel in Pallas interpret mode (CPU) instead of
            compiling for TPU. Used by the test-suite so the identical kernel
            can be validated without TPU hardware.

    Returns:
        Array of shape ``[batch, num_heads, seq_len_q, head_dim]``.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must be rank-4 [batch, heads, seq, head_dim] arrays")

    batch, num_heads, seq_len_q, head_dim = q.shape
    num_kv_heads = k.shape[1]
    seq_len_k = k.shape[2]
    if k.shape != v.shape:
        raise ValueError("k and v must have the same shape")
    if k.shape[0] != batch or k.shape[3] != head_dim:
        raise ValueError("q and k/v must share batch and head_dim")
    if num_kv_heads == 0 or num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads ({num_heads}) must be a multiple of num_kv_heads "
            f"({num_kv_heads}) for grouped-query attention"
        )
    if kv_len is None:
        kv_len = seq_len_k
    elif not 0 < kv_len <= seq_len_k:
        raise ValueError(f"kv_len must be in (0, {seq_len_k}], got {kv_len}")

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)

    return _flash_attention_vjp(
        causal, sm_scale, block_q, block_k, kv_len, interpret, q, k, v
    )


def flash_attention_decode(
    q: jax.Array,
    k_cache: jax.Array,
    v_cache: jax.Array,
    cache_len: int,
    *,
    sm_scale: Optional[float] = None,
    block_k: int = 128,
    interpret: bool = False,
) -> jax.Array:
    """Single-token decode attention against a fixed-size KV cache.

    This is the shape autoregressive decoding actually has: one new query row
    attending over ``cache_len`` cached positions of a preallocated cache. No
    causal mask is needed (the new token may see every cached position), and kv
    blocks past ``cache_len`` are skipped outright, so the cost tracks the
    filled part of the cache rather than its capacity.

    Args:
        q: array of shape ``[batch, num_heads, 1, head_dim]``.
        k_cache, v_cache: ``[batch, num_kv_heads, cache_size, head_dim]``.
        cache_len: number of valid (already written) cache positions.
        sm_scale: softmax scale; defaults to ``1 / sqrt(head_dim)``.
        block_k: cache block size streamed into VMEM.
        interpret: run in Pallas interpret mode (CPU).

    Returns:
        Array of shape ``[batch, num_heads, 1, head_dim]``.
    """
    if q.shape[2] != 1:
        raise ValueError(f"decode expects a single query row, got seq_len_q={q.shape[2]}")
    return flash_attention(
        q, k_cache, v_cache, causal=False, sm_scale=sm_scale, block_q=1,
        block_k=block_k, kv_len=cache_len, interpret=interpret,
    )
