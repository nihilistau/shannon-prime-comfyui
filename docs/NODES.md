# Shannon-Prime ComfyUI — Complete Node Reference

Nine nodes across three categories: ship path (production-ready), diagnostics (Phase 12 analysis), and experimental (Phase 13).

---

## 1. ShannonPrimeWanCache

**Display name:** Shannon-Prime: Wan Cross-Attn Cache  
**Category:** shannon-prime  
**Ship path: yes**

### Purpose

Monkey-patches every `WanAttentionBlock.cross_attn` in the loaded Wan model to cache cross-attention K/V via VHT2 spectral compression. T5/UMT5 text embeddings are constant across all diffusion timesteps — the K/V projections of those embeddings are therefore computed identically on every denoising step in an unmodified Wan model (~50 redundant computations per generation). This node computes them once, compresses via VHT2 (4-band banded quantization + optional Möbius squarefree-first reorder), and returns reconstructed K/V on all subsequent steps.

The patch is idempotent — applying the node twice to the same model object does not double-wrap the linears. Cache invalidation uses a content fingerprint (three flat-index samples + shape + dtype) that identifies the same-valued conditioning tensor across timestep reallocations, without scanning the full tensor.

For Wan 2.2 MoE models, apply one `ShannonPrimeWanCache` per expert MODEL object. The two experts are separate Python objects in ComfyUI's graph, so their caches are naturally independent.

### Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | MODEL | — | Wan model from UnetLoader or UnetLoaderGGUF |
| `k_bits` | STRING | `"5,4,4,3"` | Bit allocation for 4 VHT2 frequency bands on the K projection. Bands ordered low→high frequency. Cross-attn K has no RoPE, so first-band boost is smaller than self-attn. |
| `v_bits` | STRING | `"5,4,4,3"` | Bit allocation for 4 VHT2 frequency bands on the V projection. Symmetric with K since there is no asymmetric RoPE concentration on either. |
| `use_mobius` | BOOLEAN | `True` | Applies Möbius squarefree-first reorder before VHT2. Groups squarefree-index entries at the front of each band, improving the spectral compactness of the transform. |

### Output

`MODEL` — the same model with cross-attn linears replaced by `_SPCachingLinear` wrappers. Downstream nodes see an unchanged MODEL interface.

### Workflow placement

Place immediately after UnetLoader/UnetLoaderGGUF, before KSampler or BlockSkip.

```
UnetLoaderGGUF → ShannonPrimeWanCache → [KSampler or BlockSkip]
```

### Log output

On `patch()`:
```
[Shannon-Prime] patched 40/40 Wan blocks (head_dim=128, K=[5,4,4,3], V=[5,4,4,3], möbius=True, compression~3.56x)
```

On first hit per generation:
```
[Shannon-Prime] cache stats: hits=39 misses=1 hit_rate=0.975 entries=160 compression=3.56x
```

---

## 2. ShannonPrimeWanCacheStats

**Display name:** Shannon-Prime: Cache Stats  
**Category:** shannon-prime  
**Ship path: optional (diagnostic)**

### Purpose

Reads and prints the hit/miss statistics from a model previously patched by `ShannonPrimeWanCache`. Passes the model through unchanged. Use during development to verify the cache is hitting as expected. In production, the statistics are logged automatically on each generation; this node is useful when you want to monitor them mid-graph.

### Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | MODEL | — | A model previously processed by ShannonPrimeWanCache |

### Output

`MODEL` — pass-through, unchanged.

### Workflow placement

Anywhere after `ShannonPrimeWanCache` and before `KSampler`. Typically between WanCache and BlockSkip for the cleanest read.

### Log output

```
[Shannon-Prime] cache stats: hits=780 misses=40 hit_rate=0.951 entries=160 compression=3.56x
```

If no cache is attached (model was not patched by WanCache):
```
[Shannon-Prime] stats: model has no Shannon-Prime cache attached
```

---

## 3. ShannonPrimeWanCacheSqfree

**Display name:** Shannon-Prime: Wan Cross-Attn Cache (Sqfree)  
**Category:** shannon-prime  
**Ship path: opt-in (Q8+ backbone target)**

### Purpose

Aggressive variant of `ShannonPrimeWanCache` using the sqfree prime-Hartley basis. Where the standard node uses 4-band VHT2 with Möbius reorder, this node uses a 5-band torus-aligned allocation over a sqfree-padded spectral skeleton, adds a Möbius CSR predictor, and optionally applies SU(2) spinor sheet-bit correction at the causal-mask boundary.

Target regime: Q8+ quantized text encoders paired with bf16 or Q6 diffusion weights. At lower quantization levels the extra correction stages are below the quantization noise floor and the standard node is preferred.

Uses `VHT2SqfreeCrossAttentionCache` and `WanSqfreeCrossAttnCachingLinear` from the submodule's sqfree tool — these expose a `get_or_compute` API rather than the `put/get` used by the standard node, and cannot be hot-swapped.

### Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | MODEL | — | Wan model |
| `band_bits` | STRING | `"3,3,3,3,3"` | 5-band torus-aligned bit allocation. `3,3,3,3,3` is the aggressive default; `4,4,3,3,3` raises fidelity at the cost of compression ratio. |
| `residual_bits` | INT | `3` | N-bit residual quantization over the sqfree skeleton (1–4). 3 is the Shannon saturation point — going to 4 adds bits but yields diminishing correlation gains. |
| `use_spinor` | BOOLEAN | `True` | SU(2) sheet-bit correction at the causal-mask boundary. Corrects the sign discontinuity introduced by the mask before projecting onto the spinor sheet. Adds ~1% compute overhead but prevents a subtle phase artifact on long T5 sequences. |

### Output

`MODEL` — patched. The model carries `_sp_sqfree_cache` instead of `_sp_cache`.

### Workflow placement

Use as a drop-in replacement for `ShannonPrimeWanCache`. Do not use both in the same graph on the same model.

### Log output

```
[Shannon-Prime SQFREE] patched 40/40 Wan blocks (head_dim=128 -> pad_dim=154, bands=[3,3,3,3,3], residual_bits=3, spinor=True)
```

---

## 4. ShannonPrimeWanSelfExtract

**Display name:** Shannon-Prime: Wan Self-Attn Extract (Phase 12)  
**Category:** shannon-prime/diagnostics  
**Ship path: no (diagnostic only)**

### Purpose

Phase 12 diagnostic tool. Hooks `blk.self_attn.k` for every WanAttentionBlock and captures the K projection output tensor at a specified denoising step. Saves the result as a compressed `.npz` file with shape `(n_blocks, n_kv_heads, n_tokens, head_dim)` for post-run analysis with `sp_diagnostics.py`.

This is an observer node — it does not modify inference results. The `.npz` is written once (when the target step fires), then all hooks are removed automatically. The node is intended for Phase 12 stability analysis: feeding captured K vectors into `sp_diagnostics.py` produces the T3 RoPE pair correlation, 3D RoPE axis split statistics, and sigma-sweep stability maps used to determine which blocks belong in which tier.

### Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | MODEL | — | Wan model (with or without WanCache — compatible with both) |
| `capture_step` | INT | `25` | Denoising step at which to capture K vectors. 0=first (highest sigma, noisiest). 25 is a good mid-sigma capture point for a 20-step schedule. |
| `max_tokens` | INT | `256` | Maximum token positions stored per block. Caps memory for long sequences. Full 720p Wan sequence is ~1000 tokens. |
| `output_dir` | STRING | `""` | Directory for `.npz` output. Defaults to `ComfyUI/output/shannon_prime/`. |

### Output

`MODEL` — observer hooks attached. The model is otherwise unchanged.

### Workflow placement

Connect before KSampler. Can be stacked with WanCache:

```
UnetLoaderGGUF → ShannonPrimeWanCache → ShannonPrimeWanSelfExtract → KSampler
```

### Log output

On attach:
```
[SP SelfExtract] hooked 40/40 self-attn K projections (head_dim=128, capture_step=25, max_tokens=256)
[SP SelfExtract] will save to: C:\ComfyUI\output\shannon_prime\wan_self_attn_step25.npz
```

On capture:
```
[SP SelfExtract] saved (40, 2, 256, 128) K vectors -> wan_self_attn_step25.npz
[SP SelfExtract] Next: python sp_diagnostics.py --input wan_self_attn_step25.npz --sqfree --layer-period 4 --global-offset 3
```

---

## 5. ShannonPrimeWanBlockCache

**Display name:** Shannon-Prime: Wan Block-Tier Self-Attn Cache  
**Category:** shannon-prime  
**Ship path: superseded by BlockSkip**

### Purpose

Earlier Phase 12 implementation. Hooks `blk.self_attn.k` for early DiT blocks and caches K projection outputs using VHT2 skeleton coefficients (30% of coefficients by default). On cache-hit steps, returns the stored coefficients instead of running the linear projection.

Superseded by `ShannonPrimeWanBlockSkip` for production use. BlockSkip operates at block-forward level and skips the entire self-attention computation (norm1, Q/K/V, attention scores, output projection) rather than just the K linear. BlockCache is retained for comparison measurements and situations where block-forward patching is not available.

Can be used orthogonally with `ShannonPrimeWanCache` (cross-attention cache). Both can be applied to the same model simultaneously.

### Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | MODEL | — | Wan model |
| `skeleton_frac` | FLOAT | `0.30` | VHT2 skeleton fraction. 0.30 = store 30% of spectral coefficients. |
| `tier_0_window` | INT | `10` | Cache window for blocks L00-L03 (Permanent Granite). |
| `tier_1_window` | INT | `3` | Cache window for blocks L04-L08 (Stable Sand). |
| `verbose` | BOOLEAN | `False` | Print HIT/STORE per block per step. |

### Output

`MODEL` — patched.

### Workflow placement

After WanCache (optional), before KSampler. Do not use alongside BlockSkip on the same model — they both patch self-attention and will conflict.

### Log output

```
[SP BlockCache] 40 Wan blocks | head_dim=128 analysis_dim=154
[SP BlockCache] Tier-0 (window=10): blocks [0, 1, 2, 3]  [4 blocks cached]
[SP BlockCache] Tier-1 (window=3): blocks [4, 5, 6, 7, 8]  [5 blocks cached]
[SP BlockCache] skeleton_frac=30% (46/154 coefficients stored)
```

---

## 6. ShannonPrimeWanBlockSkip

**Display name:** Shannon-Prime: Wan Block-Level Self-Attn Skip  
**Category:** shannon-prime  
**Ship path: yes**

### Purpose

Phase 12's primary self-attention compression node. Patches `WanAttentionBlock.forward()` directly for early DiT blocks. On cache-hit steps, the entire self-attention computation is skipped — norm1, Q projection, K projection, V projection, attention score computation, and output projection are all bypassed. The cached pre-gate output `y` is retrieved from CPU and the current step's adaLN gate `e[2]` is applied, keeping brightness and contrast tracking sigma-accurate.

Cross-attention and FFN are always computed fresh (they are not cached by this node).

The **Mertens Oracle** provides adaptive cache management:
- Tracks rolling cosine similarity (EMA α=0.7) between consecutive `y` tensors on miss steps
- Halves `effective_win` when `rolling_sim < drift_threshold`
- Restores nominal window when `rolling_sim > 0.92`
- Forces oracle-refresh misses (streak-miss) after N consecutive hits
- Checks per-token x-drift as a secondary guard against latent floor-drop events

At the start of every ComfyUI prompt execution, `patch()` clears all stale caches unconditionally. This is the primary generation-boundary mechanism — no sigma comparisons or timing heuristics are needed for the cache-clear path.

### Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | MODEL | — | Wan model (apply after WanCache) |
| `tier_0_window` | INT | `10` | Nominal cache window for blocks L00-L03 (Permanent Granite). Oracle may reduce this at runtime. |
| `tier_1_window` | INT | `3` | Nominal cache window for blocks L04-L08 (Stable Sand). |
| `drift_threshold` | FLOAT | `0.85` | Rolling cos_sim threshold below which the oracle halves the effective window. |
| `x_drift_t0` | FLOAT | `0.30` | x-drift threshold for Tier-0 (L00-L03). 0.0 = disabled. 0.30 catches sudden latent jumps without triggering on normal per-step variation (0.10–0.25). |
| `x_drift_t1` | FLOAT | `0.25` | x-drift threshold for Tier-1 (L04-L08). Linear tightening within the tier: L04 uses t1 full, L08 uses t1×0.80. |
| `verbose` | BOOLEAN | `False` | Print HIT/MISS/STREAK-MISS/x-DRIFT lines per block per step. |

### Output

`MODEL` — patched with block-forward replacements on blocks L00-L08.

### Workflow placement

After `ShannonPrimeWanCache`, before `KSampler`. Must be followed by `ShannonPrimeWanCacheFlush` before `VAEDecode`.

```
ShannonPrimeWanCache → ShannonPrimeWanBlockSkip → KSampler → ShannonPrimeWanCacheFlush → VAEDecode
```

### Log output

On patch:
```
[SP BlockSkip] Patched 9/40 WanAttentionBlock.forward()
[SP BlockSkip] Tier-0 win=10: blocks [0, 1, 2, 3]
[SP BlockSkip] Tier-1 win=3: blocks [4, 5, 6, 7, 8]
[SP BlockSkip] Mertens Oracle drift_threshold=0.85
[SP BlockSkip] x-drift: T0=0.3 T1=0.25
[SP BlockSkip] Savings: self-attn Q+K+V+scores skipped on cache hits (~50% compute for patched blocks)
```

With `verbose=True` during generation:
```
[SP BlockSkip] B00 MISS  step=1 sim=1.000 roll=1.000 win=10
[SP BlockSkip] B00 HIT   step=2 age=1/10 streak=1 sim=1.000
[SP BlockSkip] B04 MISS  step=1 sim=0.923 roll=0.947 win=3
[SP BlockSkip] B04 x-DRIFT step=5 drift=0.34>0.25 -> forced miss, win halved to 1
[SP BlockSkip] B00 STREAK-MISS step=12 streak=10>=10 (sim=0.987) -> oracle refresh
```

---

## 7. ShannonPrimeWanCacheFlush

**Display name:** Shannon-Prime: Wan Block Cache Flush (before VAE)  
**Category:** shannon-prime  
**Ship path: yes — essential with BlockSkip**

### Purpose

Clears `ShannonPrimeWanBlockSkip`'s per-block y-cache before VAE decode. BlockSkip stores `y` tensors on CPU in a state dict captured by the patched block-forward closures. At 720p (1280×720, 9 frames), these tensors total roughly 594MB (43KB × 9 blocks, at per-token norm precision) but still consume enough RAM to pressure the VAE at high resolutions.

More critically, CUDA cached pages from denoising compete with VAE's VRAM allocation. Without this flush, VAE decode on a 720p 9-frame sequence takes ~34s longer than baseline. With it, total time matches or beats baseline.

The node also clears `step_cached`, `x_norm`, `rolling_sim`, and `hit_streak` dicts — all BlockSkip state — before calling `torch.cuda.empty_cache()`.

Safe to use even when BlockSkip is not in the graph: if no BlockSkip state dicts are found, it still calls `torch.cuda.empty_cache()` and exits cleanly.

### Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | MODEL | — | The BlockSkip-patched model. Wire from the same MODEL output that went into KSampler, or use a Reroute node to tap the model. |
| `samples` | LATENT | — | Latent samples from KSampler. Passed through unchanged. |

### Output

`LATENT` — the samples input, pass-through. Wire directly to VAEDecode.

### Workflow placement

Between KSampler output and VAEDecode input. The MODEL input must be the BlockSkip-patched model, not the raw UnetLoader output.

```
KSampler (LATENT) ──────────────────────► ShannonPrimeWanCacheFlush → VAEDecode
KSampler (MODEL)  ──► ShannonPrimeWanBlockSkip ... (MODEL reaches CacheFlush via Reroute or direct wire)
```

In practice in most ComfyUI graphs, the MODEL wire from BlockSkip can be rerouted to the flush node:

```
ShannonPrimeWanBlockSkip (MODEL) ──► KSampler
                          (MODEL) ──► ShannonPrimeWanCacheFlush ◄── KSampler (LATENT)
                                             └─► VAEDecode
```

### Log output

```
[SP CacheFlush] cleared 9 block caches + torch.cuda.empty_cache() — VAE now has full memory headroom
```

If BlockSkip is not present:
```
[SP CacheFlush] no BlockSkip caches found (node still safe to use)
```

---

## 8. ShannonPrimeWanSigmaSwitch

**Display name:** Shannon-Prime: Wan Sigma Switch (Phase 13)  
**Category:** shannon-prime  
**Phase 13 — currently bypassed**

### Purpose

Adjusts BlockSkip's effective cache windows at each denoising step based on the current sigma value. At high sigma (early steps, Arithmetic Granite phase), global composition is being established and early blocks are maximally stable — wider windows are correct. At low sigma (late steps, Semantic Sand), the model is refining texture detail and blocks drift faster — narrower windows reduce artifacts.

The node wraps block-0's forward function and reads `transformer_options['sigmas']` to get the raw sigma at each step. It tracks rolling max/min sigma to establish the dynamic range, computes a split threshold at `sigma_split_frac` of that range, and adjusts all `effective_win` values in BlockSkip's shared state dict on each block-0 call.

**Current status:** Bypassed. Standard ComfyUI samplers (`KSampler`, `SamplerCustomAdvanced`) do not populate `transformer_options['sigmas']`. The node detects this on first block-0 call, latches a bypass flag, logs a message, and passes through without modifying any windows for the remainder of the generation. Nominal BlockSkip windows are preserved.

### Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | MODEL | — | BlockSkip-patched model |
| `high_sigma_mult` | FLOAT | `1.5` | Window multiplier when sigma is above the split threshold. `1.5` = 50% longer cache window in high-sigma steps. Capped at `nom_win × 3`. |
| `low_sigma_mult` | FLOAT | `0.5` | Window multiplier when sigma is below the split threshold. `0.5` = half the cache window in low-sigma steps (more recompute). |
| `sigma_split_frac` | FLOAT | `0.5` | Position of the split point within the observed sigma range. `0.5` = switch at the midpoint between the session's min and max sigma. |
| `verbose` | BOOLEAN | `False` | Print sigma value, regime (HIGH/LOW), and window decisions at each step. |

### Output

`MODEL` — block-0 forward wrapped with sigma-tracking hook.

### Workflow placement

After BlockSkip, before RicciSentinel (if present), before KSampler.

```
ShannonPrimeWanBlockSkip → ShannonPrimeWanSigmaSwitch → [RicciSentinel →] KSampler
```

### Log output

On attach:
```
[SP SigmaSwitch] attached to block-0 | high_sigma_mult=1.5x low_sigma_mult=0.5x split@50% of e_mag range
```

On bypass (first call with no sigma source):
```
[SP SigmaSwitch] transformer_options has no 'sigmas' — bypassing for this generation (nominal windows preserved)
```

---

## 9. ShannonPrimeWanRicciSentinel

**Display name:** Shannon-Prime: Wan Ricci Sentinel (Phase 13 diag)  
**Category:** shannon-prime  
**Phase 13 — diagnostic**

### Purpose

Per-step sigma regime and cache window timeline reporter. Hooks block-0's forward and records each denoising step's `e_mag`, HIGH/LOW sigma regime classification, and the effective cache windows for blocks 0 (Tier-0 representative) and 4 (Tier-1 representative). At each generation boundary (detected by wall-clock gap > 60s), prints a compact summary table for the completed generation.

The sentinel recursively walks block-0's closure chain to find BlockSkip's shared `state` dict (identified by the `effective_win` key). This works correctly through SigmaSwitch wrapping — the closure chain is: `sentinel_forward → sigma_forward (SigmaSwitch) → patched_forward (BlockSkip) → orig_forward`.

Use this to verify SigmaSwitch is switching at the correct step, to observe oracle window adaptation, and to diagnose any drift issues.

### Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | MODEL | — | Model after BlockSkip (and SigmaSwitch if used) |
| `sigma_split_frac` | FLOAT | `0.5` | Match SigmaSwitch's `sigma_split_frac` for consistent HIGH/LOW labeling. |
| `verbose` | BOOLEAN | `True` | Print every step's record as it happens. Set to `False` for summary-only output at generation end. |

### Output

`MODEL` — block-0 forward wrapped with sentinel hook.

### Workflow placement

After SigmaSwitch (or directly after BlockSkip if SigmaSwitch is not in the graph), before KSampler.

### Log output

Per-step with `verbose=True`:
```
[SP Ricci] step=  1  e=0.871  thr=0.612  HIGH  win[0]=10  win[4]=3  sim=1.000
[SP Ricci] step=  2  e=0.853  thr=0.612  HIGH  win[0]=10  win[4]=3  sim=0.998
...
[SP Ricci] step= 11  e=0.601  thr=0.612  LOW   win[0]=5   win[4]=1  sim=0.941
```

Summary table at generation boundary:
```
[SP Ricci Sentinel] Generation summary (20 steps):
  Step    e_mag  regime  win[0]  win[4]  roll_sim[0]
  ------------------------------------------------------
     1    0.871  HIGH        10       3        1.000
     2    0.853  HIGH        10       3        0.998
  ...
     --- HIGH->LOW ---
    11    0.601  LOW          5       1        0.941
  ...

  Regime switch at step 11 (split@50% of e_mag range)
```
