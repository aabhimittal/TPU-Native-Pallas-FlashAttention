"""Benchmark the serving-oriented kernels: fused RoPE and paged attention.

Two questions this answers on a TPU:

1. **Fused RoPE** — how much does folding the rotation into the kernel save
   versus applying RoPE with separate XLA ops? The saving is HBM traffic (two
   round-trips of Q and K), so it grows with sequence length.
2. **Paged attention** — how does a batch of ragged sequences cost out against
   padding every sequence to the batch maximum in a contiguous cache? Pages past
   a sequence's context are skipped, so the win grows with length variance.

Run on a TPU for meaningful numbers; on CPU this falls back to interpret mode
and only confirms the code paths execute.

Usage:
    python benchmarks/benchmark_serving.py
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from pallas_flash import (
    flash_attention,
    flash_attention_rope,
    paged_flash_attention,
)
from pallas_flash.model import apply_rope, precompute_rope


def _bench(fn, *args, warmup=2, iters=5) -> float:
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    start = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - start) / iters


def bench_fused_rope(interpret: bool, seq_lens) -> None:
    print("Fused RoPE vs. RoPE-then-attention")
    header = f"{'seq_len':>8} | {'unfused (ms)':>13} | {'fused (ms)':>11} | {'speedup':>8}"
    print(header)
    print("-" * len(header))

    for seq in seq_lens:
        keys = jax.random.split(jax.random.PRNGKey(seq), 3)
        q = jax.random.normal(keys[0], (1, 8, seq, 128), jnp.float32)
        k = jax.random.normal(keys[1], (1, 8, seq, 128), jnp.float32)
        v = jax.random.normal(keys[2], (1, 8, seq, 128), jnp.float32)
        cos, sin = precompute_rope(seq, 128, 10000.0)

        unfused = jax.jit(lambda q, k, v: flash_attention(
            apply_rope(q, cos, sin), apply_rope(k, cos, sin), v,
            causal=True, interpret=interpret))
        fused = jax.jit(lambda q, k, v: flash_attention_rope(
            q, k, v, cos, sin, causal=True, interpret=interpret))

        t_unfused = _bench(unfused, q, k, v)
        t_fused = _bench(fused, q, k, v)
        print(f"{seq:>8} | {t_unfused * 1e3:>13.3f} | {t_fused * 1e3:>11.3f} | "
              f"{t_unfused / t_fused:>7.2f}x")


def bench_paged(interpret: bool, batch: int, page_size: int, max_len: int) -> None:
    print("\nPaged attention on a ragged batch")
    print(f"batch={batch}, page_size={page_size}, max_len={max_len}")

    rng = np.random.default_rng(0)
    lens = rng.integers(1, max_len, size=batch)
    pages_per_seq = max_len // page_size + 1
    num_pages = batch * pages_per_seq

    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    q = jax.random.normal(keys[0], (batch, 8, 1, 128), jnp.float32)
    k_pages = jax.random.normal(keys[1], (num_pages, 8, page_size, 128), jnp.float32)
    v_pages = jax.random.normal(keys[2], (num_pages, 8, page_size, 128), jnp.float32)
    table = jnp.asarray(
        np.arange(num_pages, dtype=np.int32).reshape(batch, pages_per_seq))
    ctx = jnp.asarray(lens, jnp.int32)

    paged = jax.jit(lambda q, kp, vp: paged_flash_attention(
        q, kp, vp, table, ctx, interpret=interpret))

    # Contiguous baseline: every sequence padded to the batch maximum.
    k_pad = jax.random.normal(keys[1], (batch, 8, max_len, 128), jnp.float32)
    v_pad = jax.random.normal(keys[2], (batch, 8, max_len, 128), jnp.float32)
    padded = jax.jit(lambda q, k, v: flash_attention(
        q, k, v, causal=False, block_q=1, block_k=page_size, interpret=interpret))

    t_paged = _bench(paged, q, k_pages, v_pages)
    t_padded = _bench(padded, q, k_pad, v_pad)
    total_pages = int(np.ceil(lens / page_size).sum())
    print(f"  mean length {lens.mean():.0f} of max {max_len} "
          f"({total_pages} live pages vs {batch * pages_per_seq} allocated)")
    print(f"  padded contiguous: {t_padded * 1e3:.3f} ms")
    print(f"  paged            : {t_paged * 1e3:.3f} ms  ({t_padded / t_paged:.2f}x)")


def main() -> None:
    on_tpu = any(d.platform == "tpu" for d in jax.devices())
    interpret = not on_tpu
    print(f"Devices: {jax.devices()}")
    print(f"Running on {'TPU (compiled)' if on_tpu else 'CPU (interpret mode — illustrative only)'}\n")

    bench_fused_rope(interpret, [512, 1024, 2048] if on_tpu else [128, 256])
    bench_paged(interpret, batch=8 if on_tpu else 4,
                page_size=128, max_len=1024 if on_tpu else 256)


if __name__ == "__main__":
    main()
