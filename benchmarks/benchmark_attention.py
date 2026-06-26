"""Benchmark the custom Pallas FlashAttention kernel against the default XLA
(naive ``jnp``) attention.

On a TPU v3-8 the Pallas kernel should win at longer sequence lengths because it
never materializes the full ``[seq, seq]`` score matrix in HBM. On CPU this
script still runs (via interpret mode) but the timings are not meaningful — it is
there to confirm the code path executes; run it on a TPU for real numbers.

Usage:
    python benchmarks/benchmark_attention.py
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp

from pallas_flash.flash_attention import flash_attention
from pallas_flash.reference import reference_attention


def _is_tpu() -> bool:
    return any(d.platform == "tpu" for d in jax.devices())


def _bench(fn, *args, warmup=2, iters=10) -> float:
    """Return median-ish seconds per call (mean of timed iters)."""
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    start = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - start) / iters


def main() -> None:
    on_tpu = _is_tpu()
    interpret = not on_tpu
    print(f"Devices: {jax.devices()}")
    print(f"Running on {'TPU (compiled)' if on_tpu else 'CPU (interpret mode — timings illustrative only)'}\n")

    batch, heads, head_dim = 1, 8, 128
    seq_lens = [256, 512, 1024, 2048, 4096] if on_tpu else [128, 256, 512]

    pallas_fn = jax.jit(
        lambda q, k, v: flash_attention(q, k, v, causal=True, interpret=interpret),
    ) if on_tpu else (lambda q, k, v: flash_attention(q, k, v, causal=True, interpret=True))
    xla_fn = jax.jit(lambda q, k, v: reference_attention(q, k, v, causal=True))

    header = f"{'seq_len':>8} | {'XLA (ms)':>10} | {'Pallas (ms)':>12} | {'speedup':>8}"
    print(header)
    print("-" * len(header))

    for seq in seq_lens:
        keys = jax.random.split(jax.random.PRNGKey(seq), 3)
        q = jax.random.normal(keys[0], (batch, heads, seq, head_dim), jnp.float32)
        k = jax.random.normal(keys[1], (batch, heads, seq, head_dim), jnp.float32)
        v = jax.random.normal(keys[2], (batch, heads, seq, head_dim), jnp.float32)

        t_xla = _bench(xla_fn, q, k, v)
        t_pallas = _bench(pallas_fn, q, k, v)
        speedup = t_xla / t_pallas if t_pallas > 0 else float("nan")
        print(f"{seq:>8} | {t_xla * 1e3:>10.3f} | {t_pallas * 1e3:>12.3f} | {speedup:>7.2f}x")


if __name__ == "__main__":
    main()
