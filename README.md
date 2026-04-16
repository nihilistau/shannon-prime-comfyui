# shannon-prime-comfyui

**Shannon-Prime VHT2 cross-attention caching for ComfyUI video generation**

Compresses and caches cross-attention K/V in Wan 2.1/2.2 video generation models.
Cross-attention from T5 text embeddings is identical across all diffusion timesteps —
compute once, compress via VHT2, reconstruct on subsequent calls.

Production results (Wan 2.2 14B, RTX 2060): 1.20× cross-attention speedup, 0.9984 output correlation.

## Supported Models

| Model | Type | Support |
|-------|------|---------|
| Wan 2.1 14B / 1.3B | Dense | Full |
| Wan 2.2 A14B T2V | MoE (2 experts) | Full (expert-aware caching) |
| Wan 2.2 A14B I2V | MoE (2 experts) | Full (boundary = 0.900) |
| Wan 2.2 TI2V-5B | Dense | Full |

## Installation

```bash
# Clone into ComfyUI custom nodes
cd /path/to/ComfyUI/custom_nodes
git clone --recursive https://github.com/YOUR_USER/shannon-prime-comfyui.git
```

## Usage

Load the example workflows from `workflows/` or use the wrapper directly:

```python
from shannon_prime_comfyui import WanVHT2Wrapper

wrapper = WanVHT2Wrapper(head_dim=128, model_type='wan22_moe', task_type='t2v')

for step, sigma in enumerate(sigmas):
    wrapper.set_expert_from_sigma(sigma)
    for block_idx in range(40):
        k, v = wrapper.get_or_compute(
            f"block_{block_idx}",
            lambda: (block.cross_attn_k(context), block.cross_attn_v(context))
        )

wrapper.reset()  # Between generations
```

See [docs/INTEGRATION.md](docs/INTEGRATION.md) for the full guide with Wan architecture diagrams.

## License

AGPLv3 / Commercial dual license. See [LICENSE](LICENSE).
