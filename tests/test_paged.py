"""Tests for paged attention (block-table driven KV cache)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pallas_flash import (
    init_paged_cache,
    paged_flash_attention,
    reference_attention,
    write_to_paged_cache,
)


def _scrambled_table(batch, pages_per_seq, num_pages, seed):
    """Block tables whose pages are deliberately NOT contiguous."""
    rng = np.random.default_rng(seed)
    table = np.stack([
        rng.choice(num_pages, size=pages_per_seq, replace=False) for _ in range(batch)
    ])
    return jnp.asarray(table, jnp.int32)


def _gather_contiguous(pages, table_row, pages_per_seq, length):
    """Rebuild a sequence's contiguous KV from its pages, for the oracle."""
    parts = [pages[int(table_row[p])] for p in range(pages_per_seq)]
    return jnp.concatenate(parts, axis=1)[:, :length]


@pytest.mark.parametrize(
    "batch,n_heads,n_kv_heads,page_size,context_lens",
    [
        (1, 2, 2, 64, [100]),
        (3, 4, 2, 64, [10, 130, 200]),
        (2, 4, 1, 128, [128, 5]),
        (4, 2, 2, 32, [1, 33, 64, 90]),
    ],
)
def test_paged_matches_contiguous(batch, n_heads, n_kv_heads, page_size, context_lens):
    dim, num_pages = 128, 32
    pages_per_seq = max(context_lens) // page_size + 1
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    q = jax.random.normal(keys[0], (batch, n_heads, 1, dim), jnp.float32)
    k_pages = jax.random.normal(keys[1], (num_pages, n_kv_heads, page_size, dim), jnp.float32)
    v_pages = jax.random.normal(keys[2], (num_pages, n_kv_heads, page_size, dim), jnp.float32)

    table = _scrambled_table(batch, pages_per_seq, num_pages, seed=batch)
    lens = jnp.asarray(context_lens, jnp.int32)

    out = paged_flash_attention(q, k_pages, v_pages, table, lens, interpret=True)
    assert out.shape == q.shape

    for b, length in enumerate(context_lens):
        k_c = _gather_contiguous(k_pages, table[b], pages_per_seq, length)
        v_c = _gather_contiguous(v_pages, table[b], pages_per_seq, length)
        ref = reference_attention(q[b:b + 1], k_c[None], v_c[None], causal=False)
        assert jnp.max(jnp.abs(out[b:b + 1] - ref)) < 2e-3


def test_pages_are_shareable():
    """Two sequences pointing at the same physical pages must agree."""
    dim, page_size, num_pages = 128, 64, 8
    keys = jax.random.split(jax.random.PRNGKey(7), 3)
    q_one = jax.random.normal(keys[0], (1, 2, 1, dim), jnp.float32)
    q = jnp.concatenate([q_one, q_one], axis=0)          # identical queries
    k_pages = jax.random.normal(keys[1], (num_pages, 2, page_size, dim), jnp.float32)
    v_pages = jax.random.normal(keys[2], (num_pages, 2, page_size, dim), jnp.float32)

    # Both rows reference the same physical pages (a shared prompt prefix).
    table = jnp.asarray([[3, 5], [3, 5]], jnp.int32)
    lens = jnp.asarray([100, 100], jnp.int32)
    out = paged_flash_attention(q, k_pages, v_pages, table, lens, interpret=True)
    assert jnp.allclose(out[0], out[1], atol=1e-5)


def test_unwritten_pages_do_not_leak():
    """Content past context_len is ignored even if it is NaN."""
    page_size, dim = 64, 128
    k_pages = jnp.concatenate([
        jnp.ones((1, 2, page_size, dim)),
        jnp.full((1, 2, page_size, dim), jnp.nan),
    ])
    v_pages = k_pages
    q = jnp.ones((1, 2, 1, dim))
    table = jnp.asarray([[0, 1]], jnp.int32)
    lens = jnp.asarray([page_size], jnp.int32)   # only page 0 is real

    out = paged_flash_attention(q, k_pages, v_pages, table, lens, interpret=True)
    assert jnp.all(jnp.isfinite(out))
    assert jnp.allclose(out, 1.0, atol=1e-5)


def test_write_to_paged_cache_roundtrip():
    """Tokens written through the page helper are the ones attended."""
    page_size, dim, n_kv = 32, 128, 2
    num_pages, length = 4, 70
    k_pages, v_pages = init_paged_cache(num_pages, n_kv, page_size, dim)
    table = jnp.asarray([0, 2, 1, 3], jnp.int32)

    keys = jax.random.split(jax.random.PRNGKey(11), 2)
    k_seq = jax.random.normal(keys[0], (1, n_kv, length, dim), jnp.float32)
    v_seq = jax.random.normal(keys[1], (1, n_kv, length, dim), jnp.float32)
    for pos in range(length):
        k_pages, v_pages = write_to_paged_cache(
            k_pages, v_pages, k_seq[:, :, pos:pos + 1], v_seq[:, :, pos:pos + 1],
            table, pos,
        )

    q = jax.random.normal(jax.random.PRNGKey(12), (1, 2, 1, dim), jnp.float32)
    out = paged_flash_attention(
        q, k_pages, v_pages, table[None], jnp.asarray([length], jnp.int32), interpret=True
    )
    ref = reference_attention(q, k_seq, v_seq, causal=False)
    assert jnp.max(jnp.abs(out - ref)) < 2e-3


def test_paged_rejects_multi_row_query():
    k_pages, v_pages = init_paged_cache(4, 2, 32, 128)
    q = jnp.zeros((1, 2, 4, 128))
    with pytest.raises(ValueError):
        paged_flash_attention(
            q, k_pages, v_pages, jnp.zeros((1, 2), jnp.int32),
            jnp.asarray([32], jnp.int32), interpret=True,
        )
