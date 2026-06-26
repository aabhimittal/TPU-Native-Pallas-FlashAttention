"""End-to-end tiny-LLaMA inference demo using the Pallas FlashAttention kernel.

Runs on whatever JAX device is available: a TPU v3-8 compiles the kernel, while
on CPU it falls back to Pallas interpret mode so the demo still works.

Usage:
    python scripts/run_inference.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from pallas_flash import ModelConfig, init_params, generate


def main() -> None:
    devices = jax.devices()
    on_tpu = any(d.platform == "tpu" for d in devices)
    interpret = not on_tpu
    print(f"Devices: {devices}")
    print(f"Mode: {'TPU (compiled Pallas kernel)' if on_tpu else 'CPU (Pallas interpret mode)'}\n")

    cfg = ModelConfig(
        vocab_size=256,
        dim=256,
        n_layers=4,
        n_heads=2,
        head_dim=128,
        ffn_hidden=688,
        max_seq_len=256,
    )
    params = init_params(jax.random.PRNGKey(0), cfg)

    # A toy "prompt" of byte-like token ids (the model is untrained, so output
    # tokens are arbitrary — the point is that the full causal attention path
    # runs through the custom kernel).
    prompt = jnp.array([[72, 101, 108, 108, 111]])  # "Hello" as bytes
    print(f"Prompt token ids: {prompt.tolist()[0]}")

    out = generate(params, prompt, max_new_tokens=16, cfg=cfg, interpret=interpret)
    print(f"Output token ids: {out.tolist()[0]}")
    print(f"\nGenerated {out.shape[1] - prompt.shape[1]} new tokens through the "
          f"Pallas FlashAttention decoder ({cfg.n_layers} layers, {cfg.n_heads} heads).")


if __name__ == "__main__":
    main()
