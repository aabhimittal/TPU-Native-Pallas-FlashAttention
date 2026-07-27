"""**Paged attention**: a KV cache stored in fixed-size pages, for batched serving.

A contiguous per-sequence KV cache forces you to preallocate ``max_seq_len`` for
every slot in the batch, so a batch of mostly-short sequences wastes most of the
memory it reserves. Paged attention (vLLM's idea) instead keeps one global pool
of fixed-size **pages** and gives each sequence a **block table** mapping its
logical page ``p`` to some physical page in the pool. Sequences then consume
memory proportional to their actual length, and pages can be shared (e.g. a
common prompt prefix) or recycled without moving any data.

The interesting part on TPU is *how the kernel finds a page*. A Pallas
``index_map`` must be a pure function of the grid indices — it cannot read a
regular input array. The block table lives in the one place an index_map can
reach: **scalar-prefetch** memory (SMEM), declared via
``pltpu.PrefetchScalarGridSpec``. With that, the index map becomes

    lambda b, h, p, block_tables, context_lens: (block_tables[b, p], ...)

so the DMA that streams a page from HBM into VMEM is itself addressed by a
runtime table lookup. That is the whole trick: fully dynamic, data-dependent
gathering of KV blocks, expressed as data movement rather than as compute.

Layout:
    k_pages / v_pages : [num_pages, num_kv_heads, page_size, head_dim]
    block_tables      : [batch, pages_per_seq]  int32  (logical -> physical page)
    context_lens      : [batch]                 int32  (valid tokens per sequence)

This is a decode-shaped (single query row) forward kernel, which is where paged
caches are actually used.
"""

from __future__ import annotations

import functools
from typing import Optional

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .flash_attention import _NEG_INF


def _paged_attention_kernel(
    block_tables_ref,   # SMEM (scalar prefetch): [batch, pages_per_seq]
    context_lens_ref,   # SMEM (scalar prefetch): [batch]
    q_ref,              # VMEM: (1, 1, 1, head_dim)
    k_ref,              # VMEM: (1, 1, page_size, head_dim)  -- one page
    v_ref,              # VMEM: (1, 1, page_size, head_dim)
    o_ref,              # VMEM: (1, 1, 1, head_dim)
    m_scratch, l_scratch, acc_scratch,
    *, sm_scale: float, page_size: int,
):
    """One grid step handles one page of one (sequence, head)."""
    b = pl.program_id(0)
    page_idx = pl.program_id(2)
    num_pages = pl.num_programs(2)
    context_len = context_lens_ref[b]

    @pl.when(page_idx == 0)
    def _init():
        m_scratch[...] = jnp.full_like(m_scratch, _NEG_INF)
        l_scratch[...] = jnp.zeros_like(l_scratch)
        acc_scratch[...] = jnp.zeros_like(acc_scratch)

    def _do_page():
        q = q_ref[0, 0].astype(jnp.float32)          # (1, head_dim)
        k = k_ref[0, 0].astype(jnp.float32)          # (page_size, head_dim)
        v = v_ref[0, 0].astype(jnp.float32)

        # Absolute token positions covered by this page.
        pos = page_idx * page_size + jax.lax.broadcasted_iota(
            jnp.int32, (page_size, 1), 0)
        valid = pos < context_len
        k = jnp.where(valid, k, 0.0)
        v = jnp.where(valid, v, 0.0)

        s = jnp.dot(q, k.T, preferred_element_type=jnp.float32) * sm_scale
        s = jnp.where(valid.reshape(1, page_size), s, _NEG_INF)

        m_prev = m_scratch[...]
        m_new = jnp.maximum(m_prev, jnp.max(s, axis=-1, keepdims=True))
        p = jnp.exp(s - m_new)
        alpha = jnp.exp(m_prev - m_new)
        l_scratch[...] = alpha * l_scratch[...] + jnp.sum(p, axis=-1, keepdims=True)
        acc_scratch[...] = acc_scratch[...] * alpha + jnp.dot(
            p, v, preferred_element_type=jnp.float32)
        m_scratch[...] = m_new

    # Pages entirely past this sequence's context are skipped: a short sequence
    # in a batch does no work for the batch's longest sequence's pages.
    pl.when(page_idx * page_size < context_len)(_do_page)

    @pl.when(page_idx == num_pages - 1)
    def _finalize():
        l = l_scratch[...]
        safe_l = jnp.where(l == 0.0, 1.0, l)
        o_ref[0, 0] = (acc_scratch[...] / safe_l).astype(o_ref.dtype)


def paged_flash_attention(
    q: jax.Array,
    k_pages: jax.Array,
    v_pages: jax.Array,
    block_tables: jax.Array,
    context_lens: jax.Array,
    *,
    sm_scale: Optional[float] = None,
    interpret: bool = False,
) -> jax.Array:
    """Single-query attention over a paged KV cache.

    Args:
        q: ``[batch, num_heads, 1, head_dim]`` -- the decode query.
        k_pages, v_pages: ``[num_pages, num_kv_heads, page_size, head_dim]``
            global page pool.
        block_tables: ``[batch, pages_per_seq]`` int32; entry ``[b, p]`` is the
            physical page holding logical page ``p`` of sequence ``b``. Unused
            trailing entries must still be **valid page indices** (0 is fine) --
            they are never read, but the DMA address is formed regardless.
        context_lens: ``[batch]`` int32; number of valid tokens per sequence.
        sm_scale: softmax scale; defaults to ``1 / sqrt(head_dim)``.
        interpret: run in Pallas interpret mode (CPU).

    Returns:
        ``[batch, num_heads, 1, head_dim]``.
    """
    if q.ndim != 4 or q.shape[2] != 1:
        raise ValueError(
            f"paged attention expects q of shape [batch, heads, 1, head_dim], got {q.shape}"
        )
    if k_pages.shape != v_pages.shape:
        raise ValueError("k_pages and v_pages must have the same shape")

    batch, num_heads, _, head_dim = q.shape
    num_pages, num_kv_heads, page_size, k_head_dim = k_pages.shape
    if k_head_dim != head_dim:
        raise ValueError("q and pages must share head_dim")
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads ({num_heads}) must be a multiple of num_kv_heads ({num_kv_heads})"
        )
    if block_tables.ndim != 2 or block_tables.shape[0] != batch:
        raise ValueError(
            f"block_tables must be [batch, pages_per_seq], got {block_tables.shape}"
        )
    if context_lens.shape != (batch,):
        raise ValueError(f"context_lens must be [batch], got {context_lens.shape}")

    q_per_kv = num_heads // num_kv_heads
    pages_per_seq = block_tables.shape[1]
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)

    # The index maps receive the grid indices followed by the scalar-prefetch
    # refs, which is what lets the page lookup be data-dependent.
    def k_index_map(b, h, p, block_tables_ref, context_lens_ref):
        return (block_tables_ref[b, p], h // q_per_kv, 0, 0)

    def q_index_map(b, h, p, block_tables_ref, context_lens_ref):
        return (b, h, 0, 0)

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        grid=(batch, num_heads, pages_per_seq),
        in_specs=[
            pl.BlockSpec((1, 1, 1, head_dim), q_index_map),
            pl.BlockSpec((1, 1, page_size, head_dim), k_index_map),
            pl.BlockSpec((1, 1, page_size, head_dim), k_index_map),
        ],
        out_specs=pl.BlockSpec((1, 1, 1, head_dim), q_index_map),
        scratch_shapes=[
            pltpu.VMEM((1, 1), jnp.float32),
            pltpu.VMEM((1, 1), jnp.float32),
            pltpu.VMEM((1, head_dim), jnp.float32),
        ],
    )

    return pl.pallas_call(
        functools.partial(
            _paged_attention_kernel, sm_scale=sm_scale, page_size=page_size
        ),
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "arbitrary", "arbitrary"),
        ),
        interpret=interpret,
        name="paged_flash_attention",
    )(block_tables.astype(jnp.int32), context_lens.astype(jnp.int32),
      q, k_pages, v_pages)


# --------------------------------------------------------------------------- #
# Page-pool helpers
# --------------------------------------------------------------------------- #
def init_paged_cache(num_pages: int, num_kv_heads: int, page_size: int,
                     head_dim: int, dtype=jnp.float32):
    """Allocate an empty page pool ``(k_pages, v_pages)``."""
    shape = (num_pages, num_kv_heads, page_size, head_dim)
    return jnp.zeros(shape, dtype), jnp.zeros(shape, dtype)


def write_to_paged_cache(k_pages, v_pages, k, v, block_table, position: int):
    """Write one token's K/V into the page holding ``position``.

    Args:
        k, v: ``[batch=1, num_kv_heads, 1, head_dim]`` for a single token.
        block_table: ``[pages_per_seq]`` int32 table for this sequence.
        position: absolute token position being written.
    """
    page_size = k_pages.shape[2]
    logical_page, offset = divmod(position, page_size)
    physical_page = block_table[logical_page]
    kv = (k[0], v[0])                      # [num_kv_heads, 1, head_dim]
    out = []
    for pages, x in zip((k_pages, v_pages), kv):
        # pages[physical_page, :, offset] = x[:, 0]
        upd = x.reshape(1, x.shape[0], 1, x.shape[-1])
        out.append(jax.lax.dynamic_update_slice(
            pages, upd.astype(pages.dtype), (physical_page, 0, offset, 0)))
    return out[0], out[1]
