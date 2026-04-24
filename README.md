# Shannon-Prime VHT2 for ComfyUI

Shannon-Prime is a custom node pack for ComfyUI that accelerates Wan 2.1 and 2.2 video generation via VHT2 spectral compression of attention caches. It exploits two structural invariants in the Wan DiT architecture — text cross-attention context is identical across all denoising timesteps, and early self-attention blocks are geometrically stable — to skip redundant computation without changing output quality.

---

## What It Does

### ShannonPrimeWanCache (ship path)

Patches every `WanAttentionBlock.cross_attn` in the loaded model. T5/UMT5 text embeddings are constant across all ~50 denoising timesteps — the K/V projections of those embeddings are therefore identical on every step. This node computes them once, compresses via the VHT2 spectral transform (4-band banded quantization + Möbius squarefree-first reorder), and returns reconstructed K/V on all subsequent steps.

- **Compression**: 3.56x on the K/V tensors
- **Output correlation**: 0.9984 (cosine similarity to uncompressed baseline)
- **Speedup**: 1.20x end-to-end on Wan 2.2 TI2V-5B Q8, RTX 2060, 720p

Cache invalidation uses a content fingerprint (three flat-index samples + shape + dtype) rather than pointer identity, so ComfyUI's per-timestep tensor reallocations do not break the cache.

### ShannonPrimeWanBlockSkip (Phase 12, ship path)

Patches `WanAttentionBlock.forward()` directly for early DiT blocks that show stable self-attention geometry across denoising steps. On cache-hit steps, the entire self-attention computation (norm1, Q/K/V projections, attention scores, output projection) is skipped. The adaLN gate from the current timestep embedding is still applied to the cached pre-gate output `y`, so brightness/contrast tracking remains sigma-accurate.

**Block tiers** derived from sigma-sweep Phase 12 diagnostics:

| Tier | Blocks | Default window | Stability |
|------|--------|----------------|-----------|
| 0 — Permanent Granite | L00-L03 | 10 steps | cos_sim > 0.95 across 10 steps |
| 1 — Stable Sand | L04-L08 | 3 steps | moderate stability |
| uncached | L09-L29 | 0 | volatile, always recompute |

The **Mertens Oracle** adapts windows dynamically. Each block tracks a rolling cosine similarity (EMA α=0.7) between its cached `y` and the freshly computed `y` on miss steps. If rolling_sim drops below `drift_threshold`, the effective window is halved. If it recovers above 0.92, the nominal window is restored. Streak-miss forces a recompute after N consecutive hits (N adapts to rolling_sim: >0.95→10, >0.90→7, >0.85→5, else→3).

**Results** (Wan 2.2 TI2V-5B Q8, RTX 2060, 1280×720, 9 frames, 20 steps):

| Configuration | Denoising (s/it) | Total (s) |
|---------------|-----------------|-----------|
| Baseline (cross-attn cache only) | 3.32 | 77 |
| + BlockSkip (no CacheFlush) | 3.19 | 111 |
| + BlockSkip + CacheFlush | 3.19 | ~77 |

At 720p the cache overhead equals the compute saved. The crossover into net speedup is at higher resolutions (4K) where the O(tokens²) attention dominates.

No composition warping has been observed. The `patch()` call in BlockSkip clears all stale caches at the start of every ComfyUI prompt execution, unconditionally.

### ShannonPrimeWanCacheFlush (Phase 12, essential)

**Must be placed between KSampler and VAEDecode** when using BlockSkip. BlockSkip caches `y` tensors on CPU (per-block norm proxies instead of full tensors, ~43KB/block vs 66MB/block). Without the flush, those tensors remain allocated during VAE decode and impose a ~34s overhead on 720p runs. The flush node clears all BlockSkip state dicts and calls `torch.cuda.empty_cache()` to give the VAE maximum VRAM headroom.

### ShannonPrimeWanSigmaSwitch (Phase 13, experimental)

Attaches to block-0's forward and adjusts BlockSkip's effective windows at each step based on the current sigma. High sigma → wider windows (more caching). Low sigma → narrower windows (more recompute). **Currently bypassed**: `transformer_options['sigmas']` is not populated by standard KSampler or SamplerCustomAdvanced in current ComfyUI. The node detects this on first call and latches a bypass for the entire generation, preserving nominal BlockSkip windows. Awaiting a sigma source in `transformer_options`.

### ShannonPrimeWanRicciSentinel (Phase 13 diagnostic)

Per-step sigma regime and cache window timeline reporter. Attach after BlockSkip + SigmaSwitch. With `verbose=True`, prints every step's `e_mag`, HIGH/LOW regime label, and effective windows for blocks 0 and 4. At each generation boundary, prints a compact summary table showing the full trajectory and the step at which the HIGH→LOW switch occurred. Use this to verify oracle behavior and tune `sigma_split_frac`.

---

## Results

**Phase 12 validated run** (Wan 2.2 TI2V-5B Q8, RTX 2060, 1280×720, 9 frames, 20 steps):

- Cross-attention cache: **1.20x speedup**, **0.9984 output correlation** (cosine similarity)
- BlockSkip: ~3.19 s/it denoising vs 3.32 s/it baseline
- With CacheFlush: total time matches baseline; without it, VAE adds ~34s overhead
- No warping confirmed across multiple seeds and prompts
- CUDA kernel `shannon_prime_cuda_wan` compiled and confirmed `mode=cuda` on startup

---

## Supported Models

| Model | Type | Support level |
|-------|------|---------------|
| Wan 2.1 14B | Dense | Full — all nodes |
| Wan 2.1 1.3B | Dense | Full — all nodes |
| Wan 2.2 A14B T2V | MoE (2 experts, high/low) | Full — expert-aware caching |
| Wan 2.2 A14B I2V | MoE (2 experts) | Full — boundary=0.900 |
| Wan 2.2 TI2V-5B | Dense | Full — fully tested, ship path |

For MoE models, apply `ShannonPrimeWanCache` separately to each expert `MODEL` object. The two caches are naturally partitioned because the experts are separate Python objects in ComfyUI's graph.

---

## Installation

```bash
# Clone with submodule (required — shannon-prime math core is a submodule)
cd /path/to/ComfyUI/custom_nodes
git clone --recursive https://github.com/nihilistau/shannon-prime-comfyui.git
```

If you already cloned without `--recursive`:
```bash
cd shannon-prime-comfyui
git submodule update --init --recursive
```

**Symlink option** (Windows, from an elevated prompt):
```batch
mklink /J C:\ComfyUI\custom_nodes\shannon-prime-comfyui D:\F\shannon-prime-repos\shannon-prime-comfyui
```

**CUDA kernel** (optional, builds `shannon_prime_cuda_wan` for accelerated VHT2):
```bash
cd lib/shannon-prime
python scripts/build_cuda.py
# or
scripts/build_test_cuda.bat
```
The nodes fall back to pure PyTorch if the CUDA extension is not built. A `mode=cuda` log line on startup confirms the extension is active.

---

## Quickstart: Phase 12 Ship Workflow

The stable production path. Load `workflows/wan22_ti2v_5b_phase12_ship.json` in ComfyUI.

**Node chain:**
```
UnetLoaderGGUF
  └─► ShannonPrimeWanCache     (cross-attn K/V compression)
        └─► ShannonPrimeWanBlockSkip  (self-attn skip for L00-L08)
              └─► KSampler
                    └─► ShannonPrimeWanCacheFlush  (clear before VAE)
                          └─► VAEDecode
                                └─► SaveAnimatedWEBP
```

**Recommended ComfyUI launch flags for 720p+:**
```bash
python main.py --normalvram --disable-async-offload
```

**Minimum settings to get started:**
- `ShannonPrimeWanCache`: leave defaults (`k_bits=5,4,4,3`, `v_bits=5,4,4,3`, `use_mobius=True`)
- `ShannonPrimeWanBlockSkip`: leave defaults (`tier_0_window=10`, `tier_1_window=3`, `drift_threshold=0.85`, `x_drift_t0=0.30`, `x_drift_t1=0.25`)
- `ShannonPrimeWanCacheFlush`: no parameters — just wire it between KSampler and VAEDecode

---

## Node Reference

### ShannonPrimeWanCache

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | MODEL | — | Wan model from UnetLoader/UnetLoaderGGUF |
| `k_bits` | STRING | `"5,4,4,3"` | K band bit allocation (4 bands). Cross-attn has no RoPE, so K and V use symmetric profiles. |
| `v_bits` | STRING | `"5,4,4,3"` | V band bit allocation. |
| `use_mobius` | BOOLEAN | `True` | Möbius squarefree-first reorder on both K and V. |

**Output:** MODEL (patched)

Prints: `[Shannon-Prime] patched N/N Wan blocks (head_dim=..., compression~3.56x)`

---

### ShannonPrimeWanCacheStats

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | MODEL | — | Model previously patched by WanCache |

**Output:** MODEL (pass-through)

Prints: `hits=N misses=N hit_rate=0.XXX compression=3.56x` at each queue execution. Wire between WanCache and BlockSkip, or between KSampler and VAEDecode.

---

### ShannonPrimeWanCacheSqfree

Aggressive variant. Targets Q8+ backbones with sqfree prime-Hartley basis + Möbius CSR predictor + optional SU(2) spinor sheet-bit correction.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | MODEL | — | Wan model |
| `band_bits` | STRING | `"3,3,3,3,3"` | 5-band torus-aligned allocation |
| `residual_bits` | INT | `3` | N-bit residual quantization (1–4). 3 is the Pareto point. |
| `use_spinor` | BOOLEAN | `True` | SU(2) sheet-bit correction at causal-mask boundary |

**Output:** MODEL (patched)

---

### ShannonPrimeWanSelfExtract

Phase 12 diagnostic. Hooks `blk.self_attn.k` on every WanAttentionBlock and captures K projection output at a target denoising step. Saves a `.npz` file for analysis with `sp_diagnostics.py`.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | MODEL | — | Wan model |
| `capture_step` | INT | `25` | Denoising step to capture (0=first/noisiest) |
| `max_tokens` | INT | `256` | Max token positions stored per block |
| `output_dir` | STRING | `""` | Output directory (defaults to `ComfyUI/output/shannon_prime/`) |

**Output:** MODEL (hooks attached, observer only — does not change inference)

Post-run: `python sp_diagnostics.py --input wan_self_attn_stepN.npz --sqfree --layer-period 4`

---

### ShannonPrimeWanBlockCache

Older Phase 12 node. Hooks `blk.self_attn.k` and caches K projections using VHT2 skeleton coefficients. Superseded by `ShannonPrimeWanBlockSkip` for production use (BlockSkip operates at block-forward level, skipping more compute). Retained for comparison and diagnostic use.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | MODEL | — | Wan model |
| `skeleton_frac` | FLOAT | `0.30` | VHT2 skeleton fraction (30% = 30% of coefficients stored) |
| `tier_0_window` | INT | `10` | Cache window for L00-L03 |
| `tier_1_window` | INT | `3` | Cache window for L04-L08 |
| `verbose` | BOOLEAN | `False` | Log HIT/STORE per block per step |

**Output:** MODEL (patched)

---

### ShannonPrimeWanBlockSkip

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | MODEL | — | Wan model (apply after WanCache) |
| `tier_0_window` | INT | `10` | Nominal cache window for L00-L03 (Permanent Granite) |
| `tier_1_window` | INT | `3` | Nominal cache window for L04-L08 (Stable Sand) |
| `drift_threshold` | FLOAT | `0.85` | Rolling cos_sim below this halves the window (Mertens Oracle) |
| `x_drift_t0` | FLOAT | `0.30` | x-drift threshold for Tier-0. 0.0 = disabled. |
| `x_drift_t1` | FLOAT | `0.25` | x-drift threshold for Tier-1. Linear tightening L04→t1, L08→t1×0.80. |
| `verbose` | BOOLEAN | `False` | Log HIT/MISS/STREAK-MISS/x-DRIFT per block per step |

**Output:** MODEL (patched)

---

### ShannonPrimeWanCacheFlush

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | MODEL | — | The BlockSkip-patched model (from KSampler output or stored reference) |
| `samples` | LATENT | — | Latent samples from KSampler (passed through unchanged) |

**Output:** LATENT (pass-through)

Clears all BlockSkip `attn_cache`, `step_cached`, `x_norm`, `rolling_sim`, and `hit_streak` dicts, then calls `torch.cuda.empty_cache()`. Safe to use even if BlockSkip is not present in the workflow.

---

### ShannonPrimeWanSigmaSwitch

Phase 13, currently bypassed. Place after BlockSkip, before KSampler.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | MODEL | — | BlockSkip-patched model |
| `high_sigma_mult` | FLOAT | `1.5` | Window multiplier at high sigma (early steps) |
| `low_sigma_mult` | FLOAT | `0.5` | Window multiplier at low sigma (late steps) |
| `sigma_split_frac` | FLOAT | `0.5` | Fraction of e_mag range at which to switch regimes |
| `verbose` | BOOLEAN | `False` | Log sigma value and window decisions per step |

**Output:** MODEL (patched)

**Note:** Bypasses automatically if `transformer_options['sigmas']` is unavailable (standard ComfyUI samplers do not populate it). Nominal BlockSkip windows are preserved when bypassed.

---

### ShannonPrimeWanRicciSentinel

Phase 13 diagnostic. Place after SigmaSwitch (or after BlockSkip if SigmaSwitch is not in the graph), before KSampler.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | MODEL | — | Model after BlockSkip/SigmaSwitch |
| `sigma_split_frac` | FLOAT | `0.5` | Match SigmaSwitch's split for consistent HIGH/LOW labels |
| `verbose` | BOOLEAN | `True` | Print every step's e_mag, regime, and window values |

**Output:** MODEL (patched, observer only)

---

## Settings Guide

### k_bits / v_bits (WanCache)

The 4-band bit allocation maps to VHT2 frequency bands ordered low→high. The first band captures the most energy; allocating more bits there preserves fidelity. The defaults `5,4,4,3` are calibrated for cross-attention (no RoPE on the K tensor). Do not use the self-attention default (`5,5,4,3`) here — that first-K-band boost is RoPE-specific and will over-allocate bits on cross-attn K.

Reduce bits (e.g., `4,4,3,3`) to trade correlation for more VRAM savings. Increase (e.g., `6,5,4,3`) to raise fidelity. 3-bit minimum per band is the Shannon saturation floor for this transform.

### tier_0_window / tier_1_window (BlockSkip)

Higher window = more steps skipped per recompute. Tier-0 blocks (L00-L03) have been measured at cos_sim > 0.95 for 10 steps — the default of 10 is aggressive but validated. Tier-1 (L04-L08) is more volatile; the default of 3 steps is conservative. Do not set Tier-0 higher than 15 or Tier-1 higher than 7 without running your own correlation measurements.

### drift_threshold (Mertens Oracle)

The oracle halves `effective_win` when `rolling_sim` falls below this value. 0.85 is the default; lower values (0.75–0.80) allow more drift before triggering. At CFG=1, blocks L04-L08 show `sim=0.000` (completely unstable at those blocks) — the oracle correctly forces `window=1` in that regime regardless of `drift_threshold`.

### x_drift_t0 / x_drift_t1

x-drift checks the per-token norm change of the input latent `x` between the current step and the cached step. If the mean fractional change exceeds the threshold, a forced miss is issued and the window is halved.

- `x_drift_t0=0.30`: For Tier-0. Catches "floor drop" events (sudden latent jumps) without over-triggering on normal per-step variation (which runs 0.10–0.25).
- `x_drift_t1=0.25`: For Tier-1. Linear tightening within the tier: L04 uses `t1`, L08 uses `t1 × 0.80`.
- Set to `0.0` to disable x-drift entirely (faster, but risks composition warping on some prompts).

---

## Workflow Integration

### Minimal addition (cross-attn cache only)

```
[existing UnetLoader] → ShannonPrimeWanCache → [existing KSampler]
```

No other changes needed. All downstream nodes see a normal MODEL.

### Full ship path (Phase 12)

```
UnetLoader/UnetLoaderGGUF
  └─► ShannonPrimeWanCache
        └─► ShannonPrimeWanBlockSkip
              └─► KSampler
                    └─► ShannonPrimeWanCacheFlush
                          └─► VAEDecode
```

### Full Phase 13 path (experimental)

```
UnetLoader/UnetLoaderGGUF
  └─► ShannonPrimeWanCache
        └─► ShannonPrimeWanBlockSkip
              └─► ShannonPrimeWanSigmaSwitch
                    └─► ShannonPrimeWanRicciSentinel
                          └─► KSampler
                                └─► ShannonPrimeWanCacheFlush
                                      └─► VAEDecode
```

### SamplerCustomAdvanced compatibility

All nodes output MODEL or LATENT and do not depend on sampler internals. They are compatible with both `KSampler` and `SamplerCustomAdvanced`. `ShannonPrimeWanCacheFlush` takes MODEL + LATENT and outputs LATENT — wire it after whatever sampler node you use.

### Cancel/Interrupt cleanup

On ComfyUI cancel or workflow interrupt, BlockSkip caches may remain allocated. To free them:

```
GET /sp/cleanup
```

This endpoint walks all WanAttentionBlock forwards on all loaded models and clears BlockSkip state dicts, then calls `torch.cuda.empty_cache()`. Equivalent to running `ShannonPrimeWanCacheFlush` manually.

If the endpoint is not registered in your ComfyUI version, simply re-run any workflow — `patch()` on `ShannonPrimeWanBlockSkip` clears stale caches unconditionally at the start of each queue execution.

---

## CUDA Kernel

The optional CUDA extension accelerates the VHT2 butterfly transform and Möbius reorder. Without it, the nodes run in pure PyTorch and are still correct and functional.

**Build:**
```bash
cd lib/shannon-prime
python scripts/build_cuda.py
```

**Windows batch:**
```batch
lib\shannon-prime\scripts\build_test_cuda.bat
```

**Verify:** On ComfyUI startup, look for:
```
[Shannon-Prime] shannon_prime_cuda_wan mode=cuda
```

If you see `mode=torch` instead, the extension was not found or failed to compile. The nodes continue to work in PyTorch mode.

---

## Phase 13 Status

| Feature | Status |
|---------|--------|
| Cross-attn K/V cache (WanCache) | Shipped — stable |
| Block-skip self-attn (BlockSkip) | Shipped — stable |
| Cache flush (CacheFlush) | Shipped — essential for 720p+ |
| Mertens Oracle adaptive windows | Shipped — active in BlockSkip |
| Sqfree+spinor variant (CacheSqfree) | Available — Q8+ opt-in |
| SigmaSwitch | Experimental — bypassed (no sigma source in transformer_options) |
| RicciSentinel | Experimental — diagnostic use |
| Block-tier K-hook cache (BlockCache) | Superseded by BlockSkip — retained for comparison |
| SelfExtract | Diagnostic — Phase 12 analysis tool |

---

## License

**AGPLv3** for open-source, academic, and non-proprietary use. Derivative works must share alike.

**Dual License** — a commercial license is available for proprietary integration.

Copyright (C) 2026 Ray Daniels. See [LICENSE](LICENSE).
