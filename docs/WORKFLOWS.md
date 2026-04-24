# Shannon-Prime ComfyUI — Workflow Reference

Three main workflow JSON files are included in the `workflows/` directory.

---

## 1. wan22_ti2v_5b_phase12_ship.json — Stable Ship Path

**Status:** Production-ready  
**Tested on:** Wan 2.2 TI2V-5B Q8_0, RTX 2060 6GB, 1280×720, 9 frames, 20 steps

### What it does

The full Phase 12 production workflow. Loads a TI2V-5B model with GGUF quantization, applies cross-attention K/V caching via VHT2, applies block-level self-attention skipping for early DiT blocks, runs KSampler, flushes caches before VAE decode, and saves as animated WEBP.

### Node chain

```
UnetLoaderGGUF (Wan2.2-TI2V-5B-Q8_0.gguf)
  └─► ShannonPrimeWanCache (k_bits=5,4,4,3, v_bits=5,4,4,3, use_mobius=True)
        └─► ShannonPrimeWanBlockSkip (tier_0=10, tier_1=3, drift=0.85, x_drift_t0=0.30, x_drift_t1=0.25)
              └─► KSampler (steps=20, cfg=6.0, scheduler=euler)
                    └─► ShannonPrimeWanCacheFlush
                          └─► VAEDecode
                                └─► SaveAnimatedWEBP

CLIPLoaderGGUF (umt5-xxl-encoder-Q8_0.gguf, clip_type=wan)
  └─► CLIPTextEncode (positive prompt)
  └─► CLIPTextEncode (negative prompt)

VAELoader (wan_2.1_vae.safetensors)

EmptyWan22ImageToVideoLatent (width=1280, height=720, length=9)
  or
LoadImage → VAEEncode → Wan22ImageToVideoLatent (for image-to-video)
```

### Models required

- `Wan2.2-TI2V-5B-Q8_0.gguf` — the TI2V-5B model in GGUF Q8_0 quantization
- `umt5-xxl-encoder-Q8_0.gguf` — UMT5 text encoder, Q8_0
- `wan_2.1_vae.safetensors` — Wan 2.1 VAE (compatible with 5B model)

### Settings to adjust

- **Steps**: 20 is the validated setting. Range 10–30 is safe with BlockSkip (the oracle adapts).
- **CFG**: 6.0 is standard for TI2V. At CFG=1.0, blocks L04-L08 show `sim=0.000` and the oracle forces `window=1` for those blocks — you will see more MISS lines but no warping.
- **Resolution**: 1280×720 is tested. At lower resolutions (480p), the BlockSkip overhead may exceed the compute saved (the crossover point is around 720p). At higher resolutions (1080p+), BlockSkip provides increasing net speedup.
- **Length**: 9 frames (1 + 8 latent frames in Wan's scheme). Can be increased to 25 or 49 frames (Wan's native video lengths) — BlockSkip scales with token count.

### VAE latent compression note

This workflow uses `EmptyWan22ImageToVideoLatent` which targets the Wan 2.2 5B model's VAE. If you switch to the Wan 2.1 14B model with its original VAE:

- **Wan 2.2 5B VAE**: 16x spatial compression ratio
- **Wan 2.1 VAE**: 8x spatial compression ratio

Mixing these will produce either a black/blurry output (wrong compression) or a resolution mismatch crash. Always use the VAE that matches the model family. The standard `wan_2.1_vae.safetensors` file is compatible with 5B; do not use a Wan 2.2 A14B-specific VAE with a 5B model or vice versa.

### Expected output

- Denoising: ~3.19 s/it (RTX 2060, 720p, 9 frames)
- Total generation time: ~77s (matching baseline)
- VAEDecode: ~25s (normal; CacheFlush ensures no overhead)
- Output: `.webp` animated file in ComfyUI output directory

### ComfyUI launch flags

```bash
python main.py --normalvram --disable-async-offload
```

`--disable-async-offload` prevents ComfyUI's async model management from racing with BlockSkip's CPU y-cache during the VAE decode window.

---

## 2. wan22_t2v_hilow.json — High/Low Split 14B T2V (MoE)

**Status:** Available — validated architecture, full support  
**Target model:** Wan 2.2 A14B T2V (MoE, 2 experts)

### What it does

Workflow for Wan 2.2 A14B T2V, a Mixture-of-Experts model that routes tokens through two expert paths (high-frequency and low-frequency expert). Each expert is a separate MODEL object in the ComfyUI graph. `ShannonPrimeWanCache` is applied to each expert independently — the two caches are naturally partitioned because the expert objects are separate.

### Node chain

```
UnetLoaderGGUF (Wan2.2-A14B-T2V-high.gguf) → ShannonPrimeWanCache ─┐
                                                                       ├─► MoEMerge → KSampler → ...
UnetLoaderGGUF (Wan2.2-A14B-T2V-low.gguf)  → ShannonPrimeWanCache ─┘

CLIPLoaderGGUF (umt5-xxl-encoder-Q8_0.gguf)
  └─► CLIPTextEncode (positive)
  └─► CLIPTextEncode (negative)

VAELoader
EmptyLatentVideo
```

### Models required

- `Wan2.2-A14B-T2V-high.gguf` — high-frequency expert
- `Wan2.2-A14B-T2V-low.gguf` — low-frequency expert
- `umt5-xxl-encoder-Q8_0.gguf`
- Wan VAE (appropriate for A14B)

### Settings to adjust

- Apply `ShannonPrimeWanCache` to **both** expert models with identical settings.
- BlockSkip can also be applied to both experts. Block tier structure is the same for both experts (they share the same DiT block architecture — only the MoE routing layer differs).
- For I2V MoE (Wan 2.2 A14B I2V), the expert boundary is set at 0.900 — this affects the routing logic in the Wan model internals, not the Shannon-Prime nodes. The nodes work the same way on both T2V and I2V experts.

### Resolution notes

14B T2V models support the same resolution range as 5B. The 16x vs 8x VAE compression issue described above applies here as well — always use the VAE matching the A14B model family.

---

## 3. wan22_ti2v_5b_phase13.json — Full Phase 13 Experimental

**Status:** Experimental  
**Target model:** Wan 2.2 TI2V-5B Q8_0

### What it does

Extends the Phase 12 ship workflow with `ShannonPrimeWanSigmaSwitch` and `ShannonPrimeWanRicciSentinel`. The SigmaSwitch node is currently bypassed at runtime (no sigma source in `transformer_options`), but the wiring is complete for when the sigma source becomes available. The RicciSentinel provides per-step monitoring of oracle behavior.

### Node chain

```
UnetLoaderGGUF
  └─► ShannonPrimeWanCache
        └─► ShannonPrimeWanBlockSkip
              └─► ShannonPrimeWanSigmaSwitch   (currently bypassed)
                    └─► ShannonPrimeWanRicciSentinel  (diagnostic output)
                          └─► KSampler
                                └─► ShannonPrimeWanCacheFlush
                                      └─► VAEDecode
                                            └─► SaveAnimatedWEBP
```

### Models required

Same as Phase 12 ship:
- `Wan2.2-TI2V-5B-Q8_0.gguf`
- `umt5-xxl-encoder-Q8_0.gguf`
- `wan_2.1_vae.safetensors`

### Settings to adjust

- **SigmaSwitch**: `high_sigma_mult=1.5`, `low_sigma_mult=0.5`, `sigma_split_frac=0.5`. These are no-ops until the sigma source is available, but setting them correctly now means the workflow is ready to activate when the sampler provides sigmas.
- **RicciSentinel**: `verbose=True` to see per-step output. `sigma_split_frac` should match SigmaSwitch's value (default 0.5).
- All other settings are identical to the Phase 12 ship workflow.

### What to expect in logs

With SigmaSwitch bypassed, you will see:
```
[SP SigmaSwitch] transformer_options has no 'sigmas' — bypassing for this generation (nominal windows preserved)
```

The RicciSentinel will still log per-step `e_mag` and window data based on its own e_mag tracking, but HIGH/LOW labels reflect e_mag heuristics rather than true sigma values.

### Expected output

Identical to Phase 12 ship workflow. The Phase 13 experimental nodes add no overhead when bypassed.

---

## General Notes

### Latent node selection

ComfyUI has multiple latent preparation nodes for Wan. Use the correct one:

| Node | For |
|------|-----|
| `EmptyWan22ImageToVideoLatent` | Wan 2.2 TI2V-5B, text-to-video or text+image |
| `Wan22ImageToVideoLatent` | Wan 2.2 TI2V-5B, image-to-video (requires reference image) |
| `EmptyLatentVideo` | Wan 2.1 14B/1.3B, any task |

The `Wan22ImageToVideoLatent` node applies 16x spatial compression to the conditioning image. If you accidentally use this node with a Wan 2.1 model (which uses 8x compression), the spatial dimensions will be halved and the output will be either black or heavily distorted.

### Workflow reuse and model persistence

ComfyUI reuses loaded models across queue executions. Shannon-Prime's caches persist on the model Python objects. This is correct behavior — the content-fingerprint invalidation ensures stale cross-attn caches are automatically refreshed when the prompt changes. BlockSkip's `patch()` call clears self-attn caches at the start of each queue execution.

You do not need to reload the model between generations.
