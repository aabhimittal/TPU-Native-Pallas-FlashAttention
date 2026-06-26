# TPU-Native "Pallas" FlashAttention

A from-scratch **FlashAttention** kernel written in **JAX Pallas** for Google's
**TPU v3-8** (free on Kaggle), wired into a tiny **LLaMA-style decoder** for
inference. This is the JAX/TPU analogue of "writing CUDA": instead of relying on
the XLA compiler to fuse a naive attention, we hand-write a kernel that tiles
Q/K/V, manages the **HBM → VMEM** data movement explicitly, and runs the
online-softmax algorithm so the full attention matrix is never materialized.

## Why bother?

The default `softmax(Q Kᵀ) V` that XLA compiles materializes the entire
`[seq, seq]` score matrix in HBM — memory and bandwidth grow **quadratically**
with sequence length. FlashAttention instead streams blocks of K/V through the
chip's small, fast VMEM and keeps only running softmax statistics, so HBM traffic
and peak memory grow roughly **linearly**. The speedup widens as `seq_len` grows.

## What's in here

| Path | What it is |
|------|------------|
| `src/pallas_flash/flash_attention.py` | The Pallas kernel + `flash_attention()` wrapper (the core deliverable) |
| `src/pallas_flash/reference.py` | Naive `jnp` attention — correctness oracle **and** XLA baseline |
| `src/pallas_flash/model.py` | Tiny LLaMA: RMSNorm, RoPE, SwiGLU, decoder block, `generate()` |
| `src/pallas_flash/config.py` | `ModelConfig` dataclass |
| `tests/` | `interpret=True` correctness + model smoke tests (run on CPU) |
| `benchmarks/benchmark_attention.py` | Pallas vs XLA timing table |
| `scripts/run_inference.py` | End-to-end tiny-LLaMA greedy generation demo |
| `notebooks/kaggle_tpu_flash_attention.ipynb` | Self-contained Kaggle TPU v3-8 notebook |

## How the kernel works

Grid layout: `(batch, num_heads, num_q_blocks, num_kv_blocks)`, executed
sequentially in lexicographic order.

1. **Manual HBM → VMEM mapping.** Each `pl.BlockSpec` `index_map` says exactly
   which block of the big HBM array to stream into VMEM for a given grid point.
   This is the lever a high-level Keras/PyTorch program never touches.
2. **Online softmax.** For each query block we loop over key/value blocks and
   keep running statistics in VMEM scratch: row max `m`, denominator `l`, and the
   weighted value accumulator `acc`. Each new block rescales the old accumulator
   by `exp(m_old - m_new)` and adds its contribution — mathematically identical to
   a full softmax, but the `[seq, seq]` matrix never exists.
3. **Causal block skipping.** With `causal=True`, key blocks entirely in the
   future of a query block are skipped — no load, no matmul.

The *same* source runs on CPU via `interpret=True` (used by the tests) and
compiles to TPU with `interpret=False`.

## Quick start

### Local / CPU (correctness, no TPU needed)

```bash
pip install -e .            # or: pip install -r requirements.txt
pytest -q                   # runs the kernel + model tests in interpret mode
python scripts/run_inference.py
```

```python
import jax, jax.numpy as jnp
from pallas_flash import flash_attention

q = k = v = jax.random.normal(jax.random.PRNGKey(0), (1, 8, 512, 128))
out = flash_attention(q, k, v, causal=True, interpret=True)  # interpret=False on TPU
```

### Kaggle TPU v3-8

1. New Kaggle notebook → **Accelerator → TPU VM v3-8**.
2. Upload / open `notebooks/kaggle_tpu_flash_attention.ipynb` and **Run all**.

It checks `jax.devices()`, defines the kernel, verifies correctness against the
reference, prints a Pallas-vs-XLA benchmark table, and runs the tiny LLaMA.

## Benchmark

`python benchmarks/benchmark_attention.py` prints a table like:

```
 seq_len |   XLA (ms) |  Pallas (ms) |  speedup
 ...
```

Run it **on a TPU** for meaningful numbers — on CPU (interpret mode) the kernel
is intentionally un-optimized and only confirms the code path executes. On a
v3-8 the Pallas kernel pulls ahead of the XLA baseline as `seq_len` increases.

## Notes & non-goals

- **Untrained model** — parameters are random; the LLaMA exists to exercise the
  kernel on a realistic inference path, not to produce meaningful text.
- **Forward only** — no attention backward pass (inference focus).
- **KV cache** — `generate()` recomputes the prefix each step for clarity; a
  cached single-query decode kernel is the natural next step.
- **Fully manual DMA** — `BlockSpec` already pipelines HBM↔VMEM copies; for
  hand-rolled control you can keep K/V in `pltpu.ANY` and drive
  `pltpu.make_async_copy` yourself (sketched in the notebook's notes).

## Requirements

JAX (CPU build for local testing, `jax[tpu]` on Kaggle/Cloud TPU) and NumPy. See
`requirements.txt` / `pyproject.toml`.
