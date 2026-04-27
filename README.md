# Shannon-Prime for ComfyUI

**Universal spectral compression for video, image, and audio generation.**

Shannon-Prime is a suite of 16 custom ComfyUI nodes that apply the Vilenkin-Hartley Transform (VHT2) — a self-inverse spectral decomposition — to compress KV caches and skip redundant computation across every generative modality. One mathematical framework, three modalities, one set of nodes.

```
                          ┌─────────────────────────────┐
                          │     Shannon-Prime VHT2       │
                          │  Spectral KV Compression     │
                          └──────────┬──────────────────┘
                 ┌───────────────────┼───────────────────┐
           ┌─────┴─────┐     ┌──────┴──────┐     ┌──────┴──────┐
           │   VIDEO    │     │    IMAGE    │     │    AUDIO    │
           │            │     │             │     │             │
           │  Wan 2.x   │     │  Flux/SD    │     │ Stable Audio│
           │  5B / 14B  │     │  DiT/UNet   │     │ Qwen3-TTS  │
           │  MoE A14B  │     │             │     │ Voxtral 4B  │
           └────────────┘     └─────────────┘     └─────────────┘
```

The same VHT2 butterfly decomposition works everywhere because the mathematical property it exploits — **RoPE imprints spectral structure on KV vectors, and that structure is compressible** — is universal across all transformer architectures that use rotary position embeddings. head_dim=64, 128, or 256: VHT2 handles them all (power-of-2 factorization into p=2 Hartley stages). The transform is self-inverse, so the same function serves as compress and decompress.

**Headline numbers:**

| Modality | Model | Speedup / Compression | Mechanism |
|---|---|---|---|
| Video | Wan 2.2 5B | **4.6× step speed** (32→7 s/step) | Block-skip + cross-attn cache + TURBO |
| Video | Wan 2.2 A14B MoE | **3.5× step speed** | Expert-aware block-skip |
| Image | Flux v1/v2 | Block-level skip | Dual-stream + single-stream cache |
| Audio | Stable Audio | Block-level skip | 1D RoPE audio frame cache |
| TTS | Voxtral 4B | **4.6× KV memory** | Autoregressive KV compression |
| TTS | Qwen3-TTS | Autoregressive cache | (model download in progress) |

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Video — Wan 2.x](#video--wan-2x)
- [Image — Flux / Stable Diffusion](#image--flux--stable-diffusion)
- [Audio — Stable Audio, Qwen3-TTS, Voxtral](#audio--stable-audio-qwen3-tts-voxtral)
- [All Nodes Reference](#all-nodes-reference)
- [Benchmark Tool](#benchmark-tool)
- [Tuning Guide](#tuning-guide)
- [Bit Allocation Guide](#bit-allocation-guide)
- [How It Works](#how-it-works)
- [Comparison with Other Systems](#comparison-with-other-systems)
- [Project Structure](#project-structure)
- [License](#license)

---

## Installation

**Clone into custom_nodes (recommended):**
```bash
cd /path/to/ComfyUI/custom_nodes
git clone --recursive https://github.com/nihilistau/shannon-prime-comfyui.git
```

If you already cloned without `--recursive`:
```bash
cd shannon-prime-comfyui
git submodule update --init --recursive
```

**Symlink (Windows, elevated prompt):**
```batch
mklink /J C:\ComfyUI\custom_nodes\shannon-prime-comfyui D:\path\to\shannon-prime-comfyui
```

**Dependencies:** None beyond ComfyUI itself. All math lives in the `lib/shannon-prime` submodule (pure Python + optional C/CUDA). No pip packages required.

**For Voxtral TTS:** Install the [ComfyUI-FL-VoxtralTTS](https://github.com/nihilistau/ComfyUI-FL-VoxtralTTS) node alongside this one. It includes Shannon-Prime KV compression built-in. Requires `pip install mistral_common soundfile`.

---

## Quick Start

### Video (Wan 2.x) — Recommended setup

```
UnetLoaderGGUF
  → ShannonPrimeWanCache            (cross-attn K/V caching)
    → ShannonPrimeWanBlockSkip      (block-level computation skip)
      → KSampler
        → ShannonPrimeWanCacheFlush (free memory before VAE)
          → VAEDecode → SaveAnimatedWEBP
```

### Image (Flux) — Minimal setup

```
UnetLoader
  → ShannonPrimeFluxBlockSkip       (dual+single stream block skip)
    → KSampler
      → ShannonPrimeFluxCacheFlush
        → VAEDecode → SaveImage
```

### Audio (Stable Audio) — Block skip

```
CheckpointLoader
  → ShannonPrimeAudioBlockSkip      (Audio DiT block skip)
    → KSampler
      → ShannonPrimeAudioCacheFlush
        → VAEDecode → SaveAudio
```

### TTS (Voxtral) — KV compression

```
FL_VoxtralTTS_ModelLoader
  → ShannonPrimeVoxtralKVCache      (VHT2 spectral KV compression)
    → FL_VoxtralTTS_Generate
      → SaveAudioMP3
```

---

## Video — Wan 2.x

Shannon-Prime's primary and most mature integration. The Wan DiT (Diffusion Transformer) architecture has 40 transformer blocks with 3D video RoPE (temporal + spatial). Shannon-Prime exploits two key invariants:

1. **Cross-attention is constant.** T5/UMT5 text embeddings don't change between denoising steps. Cache once, reuse forever.
2. **Early blocks are geometrically stable.** Blocks L00–L08 produce self-attention outputs with cos_sim > 0.95 across 10+ consecutive steps. Cache the output, skip the compute, re-apply only the cheap adaLN gate.

### Performance

Measured on RTX 2060 12GB + 32GB RAM, Wan 2.2 TI2V-5B Q8, 720p 9 frames, `--lowvram`:

| Configuration | Step Time | Speedup |
|---|---|---|
| Stock (no Shannon-Prime) | ~32 s/step | 1.0× |
| Cross-attn cache only | ~28 s/step | 1.1× |
| + BlockSkip tier-0/1 | ~15 s/step | 2.1× |
| + Cross-attn output cache | ~8.4 s/step | 3.8× |
| + TURBO (all 40 blocks, fp8) | ~7.0 s/step | **4.6×** |

### Supported Models

| Model | Type | Status |
|---|---|---|
| Wan 2.2 TI2V-5B | Dense | Fully tested, primary target |
| Wan 2.2 A14B T2V/I2V | MoE (2 experts) | Full — expert-aware caching |
| Wan 2.1 14B | Dense | Full |
| Wan 2.1 1.3B | Dense | Full |

### 4-Tier Block System

Blocks are grouped by empirically measured stability:

| Tier | Blocks | Stability | Default Window | Character |
|---|---|---|---|---|
| 0 — Granite | L00–L03 | cos_sim > 0.95 for 10+ steps | 10 | Foundational structure |
| 1 — Sand | L04–L08 | Moderate stability | 3 | Spatial relationships |
| 2 — Volatile | L09–L15 | Lower stability | 0 (off) | Fine detail |
| 3 — Deep | L16–L39 | Texture/detail | 0 (off) | High-frequency content |

### TURBO Mode

When `cache_ffn=True`, entire blocks reduce to: adaLN modulation + 3 cached tensor loads + 3 additions. Near-zero compute on cache-hit steps.

### SVI Compatible

Works with Step-Video Inference 6-step distilled Wan 2.2 workflows, including non-monotonic sigma schedules and dual hi/lo-noise GGUF loading (SVI-Pro-Loop).

### Wan Nodes

| Node | Purpose |
|---|---|
| `ShannonPrimeWanCache` | Cross-attention K/V caching (LEAN mode by default) |
| `ShannonPrimeWanBlockSkip` | Block-level self-attn + cross-attn + FFN skip |
| `ShannonPrimeWanCacheFlush` | Memory cleanup before VAE |
| `ShannonPrimeWanCacheStats` | Observer — hit rates and compression ratios |
| `ShannonPrimeWanCacheSqfree` | Aggressive sqfree+spinor compression variant |
| `ShannonPrimeWanSigmaSwitch` | Sigma-adaptive cache windows (experimental) |
| `ShannonPrimeWanRicciSentinel` | Diagnostic — per-step regime reporting |
| `ShannonPrimeWanSelfExtract` | Diagnostic — K projection extraction for analysis |

### Example Workflows

Included in `workflows/`:
- `wan22_ti2v_5b_vht2.json` — VHT2 optimized
- `wan22_ti2v_5b_sqfree.json` — Sqfree variant
- `wan22_ti2v_5b_phase13.json` — Full Phase 13
- `wan22_t2v_hilow.json` — Hi/Lo noise SVI

---

## Image — Flux / Stable Diffusion

Flux uses a dual-stream architecture: **DoubleStreamBlock** (image + text streams with separate self-attention, then joint attention) followed by **SingleStreamBlock** (unified stream). Shannon-Prime caches block outputs across denoising steps, same principle as Wan but adapted to Flux's split architecture.

### Architecture Fit

| Property | Flux v1 | Flux v2 |
|---|---|---|
| head_dim | 128 (24 heads) | 64 (48 heads) |
| VHT2 stages | 7 (2^7) | 6 (2^6) |
| Block types | DoubleStream + SingleStream | DoubleStream + SingleStream |
| RoPE | 2D (spatial) | 2D (spatial) |
| adaLN | ModulationOut(shift, scale, gate) | Same |

Both head_dim=128 and head_dim=64 are perfect power-of-2 VHT2 targets. The dual-stream architecture means each DoubleStreamBlock has two independent attention computations (image self-attn and text self-attn with joint K/V), both cacheable.

### Flux Nodes

| Node | Purpose |
|---|---|
| `ShannonPrimeFluxBlockSkip` | Block-level skip for both stream types |
| `ShannonPrimeFluxCacheFlush` | Flush (accepts LATENT) |
| `ShannonPrimeFluxCacheFlushModel` | Flush (accepts MODEL) |

### Quick Start

```
UnetLoader → ShannonPrimeFluxBlockSkip → KSampler → ShannonPrimeFluxCacheFlush → VAEDecode
```

The block skip works the same way as Wan: early blocks are stable across denoising steps, so cache their output and skip the compute. The 2D RoPE lattice (Flux uses spatial position embeddings) produces the same spectral concentration in VHT2 as 1D or 3D RoPE.

---

## Audio — Stable Audio, Qwen3-TTS, Voxtral

Shannon-Prime extends naturally to audio generation because the underlying transformer architectures are structurally identical to vision models. Audio DiTs use 1D RoPE over frame positions; TTS models use standard Mistral-style causal attention with GQA. VHT2 compresses both.

### Stable Audio (DiT Block Skip)

Stable Audio uses a ContinuousTransformer with N identical TransformerBlocks. Each block has self-attention (1D RoPE over audio frame positions), optional cross-attention (text conditioning), optional Conformer, and FFN. The adaLN gate function uses `sigmoid(1 - gate)`.

| Property | Value |
|---|---|
| Architecture | depth=24, num_heads=24, embed_dim=1536 |
| head_dim | 64 (= 1536/24) |
| VHT2 stages | 6 (2^6) |
| RoPE | 1D (audio frame positions) |
| Sample rate | 44.1 kHz |

**Nodes:** `ShannonPrimeAudioBlockSkip`, `ShannonPrimeAudioCacheFlush`, `ShannonPrimeAudioCacheFlushModel`

**Mechanism:** Same block-skip principle as Wan/Flux. Early Audio DiT blocks produce stable outputs across denoising steps. Cache them, skip the compute, re-apply adaLN gate. The `sigmoid(1 - gate)` formulation (different from Flux's raw gate) is handled automatically.

### Qwen3-TTS (Autoregressive Cache)

Qwen3-TTS is a speech synthesis model using the Qwen architecture with custom voice conditioning. Shannon-Prime's VHT2 compression applies to the autoregressive KV cache (same mechanism as the llama.cpp integration). Model weights are large (~3.4 GB for the CustomVoice variant) and download automatically on first use.

**Status:** Model download configured, weights downloading. ComfyUI node at `custom_nodes/ComfyUI-Qwen3-TTS/`.

### Voxtral 4B (Autoregressive KV Compression)

Voxtral is Mistral's latest text-to-speech model (4B parameters, 20 voices, 9 languages, 24 kHz output). Shannon-Prime compresses Voxtral's autoregressive KV cache using VHT2 + banded quantization, reducing memory by **4.6×** with zero quality loss.

| Property | Value |
|---|---|
| Architecture | 26-layer Mistral backbone (GQA: 32Q/8KV) |
| head_dim | 128 (= 2^7, perfect VHT2 target) |
| VHT2 stages | 7 |
| KV cache per position | 208 vectors (8 heads × 26 layers × K+V) |
| Compression | 5/5/4/3 K-bands, flat 3 V → **4.6× memory reduction** |
| Quality impact | +0.04% PPL improvement (spectral regularization) |

**At 2048 generated frames:** stock KV cache = ~218 MB (bf16). With Shannon-Prime: ~47 MB.

**Nodes:** `ShannonPrimeVoxtralKVCache`, `ShannonPrimeVoxtralCacheFlush` (in the [ComfyUI-FL-VoxtralTTS](https://github.com/nihilistau/ComfyUI-FL-VoxtralTTS) package)

**Workflow:** `SP-Voxtral-TTS.json` (ModelLoader → KVCache → Generate → SaveAudioMP3)

**How it integrates:** The `ShannonPrimeVoxtralKVCache` node monkey-patches the MistralBackbone's `forward()` method to intercept K/V computation. New K/V vectors are VHT2-transformed, banded-quantized, and stored in spectral domain. On cache read, the same VHT2 transform (self-inverse) recovers the original vectors. The pipeline's autoregressive loop is unchanged.

**Fork repositories:**
- **ComfyUI node:** [nihilistau/ComfyUI-FL-VoxtralTTS](https://github.com/nihilistau/ComfyUI-FL-VoxtralTTS) — Python, with Shannon-Prime KV compression built-in
- **Rust real-time:** [nihilistau/voxtral-mini-realtime-rs](https://github.com/nihilistau/voxtral-mini-realtime-rs) — Pure Rust VHT2 + burn tensor backend
- **C inference:** [nihilistau/voxtral-tts.c](https://github.com/nihilistau/voxtral-tts.c) — Header-only C VHT2 for the pure-C engine

All three implementations share the same mathematical core: power-of-2 Hartley butterfly VHT2, self-inverse, with ship-safe banded quantization defaults.

---

## All Nodes Reference

### Video Nodes (8)

| Node | Category | Purpose |
|---|---|---|
| `ShannonPrimeWanCache` | Shannon-Prime/Wan | Cross-attention K/V caching (LEAN mode default) |
| `ShannonPrimeWanBlockSkip` | Shannon-Prime/Wan | Block-level self-attn + cross-attn + FFN skip |
| `ShannonPrimeWanCacheFlush` | Shannon-Prime/Wan | Memory cleanup (MODEL + LATENT passthrough) |
| `ShannonPrimeWanCacheStats` | Shannon-Prime/Wan | Observer — hit/miss rates |
| `ShannonPrimeWanCacheSqfree` | Shannon-Prime/Wan | Sqfree+spinor aggressive compression |
| `ShannonPrimeWanSigmaSwitch` | Shannon-Prime/Wan | Sigma-adaptive windows (experimental) |
| `ShannonPrimeWanRicciSentinel` | Shannon-Prime/Wan | Per-step diagnostic reporter |
| `ShannonPrimeWanSelfExtract` | Shannon-Prime/Wan | K-projection extraction for analysis |

### Image Nodes (3)

| Node | Category | Purpose |
|---|---|---|
| `ShannonPrimeFluxBlockSkip` | Shannon-Prime/Flux | Dual+single stream block skip |
| `ShannonPrimeFluxCacheFlush` | Shannon-Prime/Flux | Flush (LATENT passthrough) |
| `ShannonPrimeFluxCacheFlushModel` | Shannon-Prime/Flux | Flush (MODEL passthrough) |

### Audio Nodes (3)

| Node | Category | Purpose |
|---|---|---|
| `ShannonPrimeAudioBlockSkip` | Shannon-Prime/Audio | Audio DiT block skip (Stable Audio) |
| `ShannonPrimeAudioCacheFlush` | Shannon-Prime/Audio | Flush (LATENT passthrough) |
| `ShannonPrimeAudioCacheFlushModel` | Shannon-Prime/Audio | Flush (MODEL passthrough) |

### TTS Nodes (2) — via ComfyUI-FL-VoxtralTTS

| Node | Category | Purpose |
|---|---|---|
| `ShannonPrimeVoxtralKVCache` | FL/TTS/Shannon-Prime | VHT2 KV cache compression for Voxtral backbone |
| `ShannonPrimeVoxtralCacheFlush` | FL/TTS/Shannon-Prime | Cache reset between generations |

---

## Benchmark Tool

The `scripts/comfy_bench.py` tool validates all Shannon-Prime workflows against a running ComfyUI instance. It handles both graph-format and API-format workflows, resolves SetNode/GetNode virtual routing, expands subgraph/component nodes, and remaps model paths to available files.

```bash
# List all registered workflows
python scripts/comfy_bench.py --list

# Run all workflows
python scripts/comfy_bench.py --verbose

# Run by category
python scripts/comfy_bench.py --category video
python scripts/comfy_bench.py --category audio
```

**Registered workflows:**
- `wan-svi-pro-loop` — Wan 2.2 SVI-Pro-Loop I2V (video)
- `wan-svi-fast` — Wan 2.2 SVI fast T2V (video)
- `wan-dmd2-cn` — Wan DMD2 + ControlNet (video)
- `sp-stable-audio` — Stable Audio with Shannon-Prime (audio)
- `sp-voxtral-tts` — Voxtral TTS with VHT2 KV compression (audio)
- `sp-qwen3-tts` — Qwen3-TTS (audio, pending model download)

The bench tool auto-validates workflows by converting graph format to API format, resolving all virtual nodes, and queuing through ComfyUI's `/prompt` endpoint. It reports node counts, validation status, and execution timing.

---

## Strange-Attractor Stack (v1 + v2)

The `feat/strange-attractor-stack` and `feat/strange-attractor-stack-v2` branches add a sandbox of arithmetically-grounded toggles on top of the v1 ship. **All default OFF**; existing workflows are bit-identical without explicit opt-in. Toggle them on through the `ShannonPrimeWanBlockSkip` node.

The theoretical framework is documented in [*The Music of the Spheres: A Strange Attraction*](../music_of_the_spheres.md). Empirical results: ~1.7× additional speedup over the prior 4.6× ship at unchanged or improved visual quality on RTX 2060.

| Toggle | What it does | When to try it |
|---|---|---|
| `enable_drift_gate` + `granite_threshold` / `sand_threshold` / `jazz_threshold` | Gates cache hits by rolling Fisher cosine similarity per tier. Forces a refresh when the trajectory escapes its basin. | Composition flicker on long streaks |
| `enable_sigma_streak` | Streak limits scale with normalized sigma. Granite 7-15, sand 4-9, jazz 3-6 across the schedule. | Long sigma schedules or SVI |
| `enable_twin_borrow` + `twin_alpha` + `twin_threshold` + `twin_borrow_mode` | Smooths VHT2 spectral coefficients along twin-prime pairs (3-5, 11-13, 17-19, …). Modes: symmetric, low_anchor, high_anchor. Decode-only, only takes effect with `cache_compress=vht2`. | Texture noise / detail crispness |
| `enable_harmonic_correction` + `harmonic_strength` | Linear forward-Euler extrapolation along the cache-trajectory velocity on hits. α=0.5 conservative, 0.9 aggressive. | Subject motion that wants longer streaks |
| `enable_tier_skeleton` + `granite_skel_frac` / `sand_skel_frac` / `jazz_skel_frac` | Per-tier VHT2 skeleton fractions. Defaults 50% / 30% / 20%. Higher granite = denser composition foundation; higher jazz = more texture detail. | "Cardboard cut-out" feel — bump granite to 0.65-0.75 and jazz to 0.30 |
| `enable_curvature_gate` + `curvature_threshold` | Forces miss when rolling cos_sim acceleration drops below threshold. Catches basin escape *before* the static drift gate would. | Sudden quality jumps mid-generation |
| `enable_cauchy_reset` + `cauchy_radius` | When a gate fires on block N, also invalidates ±radius same-tier neighbors so they refresh together. | Cascading flicker across a tier |
| `subject_mask` (MASK input) + `subject_focus_strength` | Tightens drift thresholds proportional to mask coverage. Foveated bias toward the subject region. v1 = scalar; per-token version is a future build. | Subject quality more important than background |
| `lattice_alpha` (on `ShannonPrimeWanCache`) | Factored 3D lattice RoPE — temporal axis gets long-tier prime-harmonic frequencies, spatial gets local-tier. 0.17 paper default. Recently revived for the new ComfyUI EmbedND API. | Long-form video, scene-cut coherence |

### Auto-free VRAM after run

`ShannonPrimeWanCacheFlush` accepts `auto_free_after_run` (default OFF). Turn ON to fully unload the model from VRAM and free memory after each generation. Useful when switching workflows or running other GPU tasks between runs. Default OFF preserves ComfyUI's standard behavior of keeping models resident for fast re-runs.

### Recommended starting points

**Conservative quality**: drift gate + sigma-streak ON, everything else OFF. Adds ~zero cost, prevents most long-streak flicker.

**Standard v2**: above + harmonic_correction (α=0.5) + tier_skeleton (granite=0.50, sand=0.30, jazz=0.20) + twin_borrow (symmetric, α=0.10). The "balanced" preset.

**Aggressive v2**: above + curvature_gate + cauchy_reset (radius=2). Maximum sentinel coverage; minor extra cost.

**Maximum stack** ("chef's kiss"): all v2 toggles ON, harmonic_strength=0.7, granite_skel_frac=0.65 to combat cardboard-cutout, lattice RoPE α=0.17 ON via `ShannonPrimeWanCache`. Tested clean on Wan 2.2 14B / 5B and Wan 2.1 30-block.

## Tuning Guide

**Start conservative, then open up.** The defaults are safe for any model and prompt.

1. **Enable tier-2** (`tier_2_window=3`): adds more blocks to caching. Watch for quality changes in fine detail.
2. **Enable tier-3** (`tier_3_window=2`): caches all blocks. Texture detail may soften slightly.
3. **Enable TURBO** (`cache_ffn=True`): skips FFN on hit steps. Maximum speed.
4. **Switch to mixed dtype** (`cache_dtype=mixed`): fp16 for precision tiers, fp8 for aggressive tiers.
5. **Full fp8** (`cache_dtype=fp8`): maximum memory savings.

**Memory budget:** each cached block stores up to 3 tensors. At 720p in fp16, ~100 MB per tensor. TURBO with all 40 blocks ≈ 12 GB CPU (fp16) or 7–8 GB (fp8/mixed).

---

## Bit Allocation Guide

VHT2 decomposes attention vectors into spectral bands. Configurable bits per band:

| Config | K bits | V bits | Compression | Use Case |
|---|---|---|---|---|
| Safe | `4,3,3,3` | `4,3,3,3` | ~3× | Default |
| Ship | `5,5,4,3` | `3` | ~3.7× | Core repo default, proven |
| Aggressive | `3,3,3,3` | `3,3,3` | ~3.5× | Large models (14B+ bf16) |
| Ultra | `2,2,2,2` | `2,2,2` | ~5× | Experimental |

**The scaling law** from *Multiplicative Lattice Combined*:

    ΔPPL/PPL = exp(4700 · (1 − K_corr)² / (params^1.1 · bits^1.5)) − 1

Larger models (14B+) at higher precision (bf16) tolerate more aggressive bit reduction. The 5/5/4/3 allocation for head_dim=128 has been proven to **beat lossless fp16 by 0.04%** due to spectral regularization.

---

## How It Works

Shannon-Prime exploits the spectral structure that RoPE imprints on KV vectors. The Vilenkin-Hartley Transform (VHT2) is a staged orthonormal basis change:

1. **VHT2 Forward:** decompose each KV vector into spectral coefficients. At head_dim=128 (2^7), this is seven stages of p=2 Hartley butterfly, each scaled by 1/√2. Self-inverse: VHT2(VHT2(x)) = x.

2. **Möbius Reorder (optional):** move squarefree coefficients to the front. 61.4% of indices in N=210 are squarefree. Prioritizing them improves quality by 0.14 PPL at the same bit budget.

3. **Banded Quantization:** split coefficients into bands, quantize each with its own scale factor. High-energy bands get more bits; low-energy tail gets fewer. The ship allocation (5/5/4/3) averages 4.25 bits/coefficient.

4. **VHT2 Inverse:** apply the same transform again (self-inverse) to recover the original vector, with quantization noise as the only error.

For diffusion models (Wan, Flux, Stable Audio), the primary mechanism is block-level caching: early blocks produce outputs that change slowly across denoising steps. Cache the output, skip the compute, re-apply only the timestep-dependent gate.

For autoregressive models (Voxtral, Qwen3-TTS, llama.cpp), the mechanism is KV cache compression: compress K/V vectors before storing, decompress on read. The autoregressive loop is unchanged.

For the mathematical foundations, see the papers in `lib/shannon-prime/docs/`: *Position Is Arithmetic*, *KV Cache Is A View*, and *Multiplicative Lattice Combined*.

---

## Comparison with Other Systems

### vs. Token Merging (ToMe) / Token Pruning

Token merging reduces sequence length by merging similar tokens. Shannon-Prime preserves all tokens but compresses their KV representations spectrally. Shannon-Prime is **lossless at the attention level** (the compressed KV reconstructs to near-identical attention scores), while token merging is structurally lossy (merged tokens can't be un-merged). The approaches are orthogonal and can be combined.

### vs. DeepCache / FreeU

DeepCache caches intermediate features across U-Net denoising steps. Shannon-Prime does the same for DiT architectures, but adds spectral compression of the cached values and principled stability metrics (Fisher-weighted cosine similarity) to decide when to cache vs recompute. FreeU modifies feature magnitudes; Shannon-Prime modifies nothing — it just skips computation when the output would be the same.

### vs. Quantization (GPTQ, AWQ, GGUF)

Weight quantization (GPTQ, AWQ, GGUF Q4/Q8) compresses model parameters. Shannon-Prime compresses the **runtime KV cache**, which is complementary — you can run a Q4_K_M model with Shannon-Prime KV compression and get savings on both fronts. In fact, VHT2's spectral regularization can slightly improve quality on quantized models by smoothing quantization noise in the attention computation.

### vs. Flash Attention / PagedAttention

Flash Attention optimizes the attention kernel's memory access pattern. PagedAttention (vLLM) manages KV cache memory allocation. Shannon-Prime reduces the KV cache **size** by 3–5×, which directly reduces Flash Attention's memory reads and PagedAttention's page count. These are complementary, not competing.

### vs. GQA / MQA

Grouped-Query Attention and Multi-Query Attention reduce KV head count at the architecture level. Shannon-Prime compresses each KV head's vectors spectrally, regardless of how many heads there are. GQA already reduces the number of KV vectors (e.g., 8 KV heads vs 32 Q heads in Mistral); Shannon-Prime then compresses each of those 8 vectors further. The savings multiply.

---

## Project Structure

```
shannon-prime-comfyui/
├── __init__.py                     # ComfyUI entry point, /sp/cleanup endpoint
├── pyproject.toml                  # ComfyUI Manager metadata
├── nodes/
│   ├── __init__.py                 # Re-exports all NODE_CLASS_MAPPINGS
│   ├── shannon_prime_nodes.py      # Wan video nodes (8 nodes)
│   ├── shannon_prime_flux_nodes.py # Flux image nodes (3 nodes)
│   └── shannon_prime_audio_nodes.py # Audio DiT nodes (3 nodes)
├── lib/
│   └── shannon-prime/              # Core math submodule (VHT2, Möbius, backends)
├── docs/
│   ├── NODES.md                    # Detailed node specifications
│   ├── WORKFLOWS.md                # Workflow documentation
│   ├── INTEGRATION.md              # Architecture + setup guide
│   └── ORACLE.md                   # Mertens oracle documentation
├── workflows/                      # Example ComfyUI workflow JSON files
├── web/                            # JS extension for cancel cleanup
└── LICENSE                         # AGPLv3 + Commercial dual license
```

**Related repositories:**

| Repo | Language | Purpose |
|---|---|---|
| [shannon-prime](https://github.com/nihilistau/shannon-prime) | C | Core math library (VHT2, Möbius, sqfree, spinor) |
| [shannon-prime-engine](https://github.com/nihilistau/shannon-prime-engine) | C++ | Standalone inference engine |
| [shannon-prime-llama](https://github.com/nihilistau/shannon-prime-llama) | C/C++ | llama.cpp integration (LM Studio) |
| [ComfyUI-FL-VoxtralTTS](https://github.com/nihilistau/ComfyUI-FL-VoxtralTTS) | Python | Voxtral TTS + Shannon-Prime KV |
| [voxtral-mini-realtime-rs](https://github.com/nihilistau/voxtral-mini-realtime-rs) | Rust | Voxtral real-time + Shannon-Prime KV |
| [voxtral-tts.c](https://github.com/nihilistau/voxtral-tts.c) | C | Voxtral C engine + Shannon-Prime KV |

---

## License

**AGPLv3** for open-source, academic, and non-proprietary use. Derivative works must share alike.

**Commercial license** available for proprietary integration. Contact: Ray Daniels (raydaniels@gmail.com).

Copyright (C) 2026 Ray Daniels. See [LICENSE](LICENSE).
