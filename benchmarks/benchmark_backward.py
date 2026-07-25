"""Benchmark the Pallas forward+backward pass against XLA autodiff.

The backward pass of the naive attention has to keep (or recompute) the whole
``[seq, seq]`` score matrix; the Pallas backward kernels recompute the softmax
tile-by-tile from the saved log-sum-exp instead. This script times
``value_and_grad`` for both and reports the speedup.

Run it on a TPU for meaningful numbers; on CPU it falls back to interpret mode
and only confirms the code path executes.

Usage:
    python benchmarks/benchmark_backward.py
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp

from pallas_flash.flash_attention import flash_attention
from pallas_flash.reference import reference_attention


def _bench(fn, *args, warmup=2, iters=5) -> float:
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    start = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - start) / iters


def main() -> None:
    on_tpu = any(d.platform == "tpu" for d in jax.devices())
    interpret = not on_tpu
    print(f"Devices: {jax.devices()}")
    print(f"Running on {'TPU (compiled)' if on_tpu else 'CPU (interpret mode — illustrative only)'}\n")

    batch, heads, head_dim = 1, 8, 128
    seq_lens = [512, 1024, 2048] if on_tpu else [128, 256]

    def pallas_loss(q, k, v):
        return jnp.sum(flash_attention(q, k, v, causal=True, interpret=interpret))

    def xla_loss(q, k, v):
        return jnp.sum(reference_attention(q, k, v, causal=True))

    pallas_fwd = jax.jit(pallas_loss)
    pallas_grad = jax.jit(jax.grad(pallas_loss, argnums=(0, 1, 2)))
    xla_grad = jax.jit(jax.grad(xla_loss, argnums=(0, 1, 2)))

    header = (f"{'seq_len':>8} | {'Pallas fwd':>11} | {'Pallas f+b':>11} | "
              f"{'XLA f+b':>11} | {'speedup':>8}")
    print(header)
    print("-" * len(header))

    for seq in seq_lens:
        keys = jax.random.split(jax.random.PRNGKey(seq), 3)
        q = jax.random.normal(keys[0], (batch, heads, seq, head_dim), jnp.float32)
        k = jax.random.normal(keys[1], (batch, heads, seq, head_dim), jnp.float32)
        v = jax.random.normal(keys[2], (batch, heads, seq, head_dim), jnp.float32)

        t_fwd = _bench(pallas_fwd, q, k, v)
        t_pallas = _bench(pallas_grad, q, k, v)
        t_xla = _bench(xla_grad, q, k, v)
        speedup = t_xla / t_pallas if t_pallas > 0 else float("nan")
        print(f"{seq:>8} | {t_fwd * 1e3:>10.3f}m | {t_pallas * 1e3:>10.3f}m | "
              f"{t_xla * 1e3:>10.3f}m | {speedup:>7.2f}x")

    print("\n(times in milliseconds per call; f+b = value_and_grad)")


if __name__ == "__main__":
    main()
