"""Benchmark KV-cache decoding vs. full-recompute decoding.

`generate` re-runs the whole prefix every step (O(T^2) total work); `generate_cached`
keeps a per-layer K/V cache and only computes the new token each step (O(T)). This
script times both to emit the same number of tokens and reports the speedup.

On CPU (Pallas interpret mode) the absolute numbers are not meaningful, but the
cached path already does far less work; on a TPU v3-8 the gap is large and grows
with the number of generated tokens.

Usage:
    python benchmarks/benchmark_decode.py
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp

from pallas_flash import ModelConfig, init_params, generate, generate_cached


def _time(fn, iters=3) -> float:
    best = float("inf")
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    on_tpu = any(d.platform == "tpu" for d in jax.devices())
    interpret = not on_tpu
    print(f"Devices: {jax.devices()}")
    print(f"Running on {'TPU (compiled)' if on_tpu else 'CPU (interpret mode — illustrative only)'}\n")

    # Grouped-query attention config (4 query heads share 2 KV heads).
    cfg = ModelConfig(
        vocab_size=256, dim=512, n_layers=4, n_heads=4, n_kv_heads=2,
        head_dim=128, ffn_hidden=1024, max_seq_len=512,
    )
    params = init_params(jax.random.PRNGKey(0), cfg)
    prompt = jnp.array([[72, 101, 108, 108, 111]])  # "Hello"

    new_tokens_grid = [16, 32, 64] if on_tpu else [8, 16]

    header = f"{'new_tokens':>10} | {'recompute (s)':>14} | {'cached (s)':>11} | {'speedup':>8}"
    print(header)
    print("-" * len(header))
    for n in new_tokens_grid:
        t_recompute = _time(lambda: generate(params, prompt, n, cfg, interpret=interpret))
        t_cached = _time(lambda: generate_cached(params, prompt, n, cfg, interpret=interpret))
        speedup = t_recompute / t_cached if t_cached > 0 else float("nan")
        print(f"{n:>10} | {t_recompute:>14.3f} | {t_cached:>11.3f} | {speedup:>7.2f}x")

    # Sanity: both paths must agree.
    a = generate(params, prompt, new_tokens_grid[0], cfg, interpret=interpret)
    b = generate_cached(params, prompt, new_tokens_grid[0], cfg, interpret=interpret)
    print(f"\noutputs identical: {bool(jnp.array_equal(a, b))}")


if __name__ == "__main__":
    main()
