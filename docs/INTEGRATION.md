# Shannon-Prime ComfyUI — Integration Guide

How to add Shannon-Prime nodes to an existing Wan ComfyUI workflow.

---

## Starting from an existing Wan workflow

You have a working Wan 2.1 or 2.2 workflow. You want to add Shannon-Prime. The nodes are additive — they pass MODEL through unchanged to all downstream nodes, so you can add them without restructuring your existing graph.

---

## Level 1: Minimal addition (cross-attn cache only)

Add one node. Intercept the MODEL wire between your UnetLoader and KSampler.

**Before:**
```
UnetLoaderGGUF ────────────────────────────────► KSampler
```

**After:**
```
UnetLoaderGGUF → ShannonPrimeWanCache ─────────► KSampler
```

How to do it in ComfyUI:
1. Right-click the canvas, search for "Shannon-Prime: Wan Cross-Attn Cache"
2. Disconnect the wire between UnetLoader and KSampler
3. Connect UnetLoader → WanCache → KSampler
4. Leave WanCache settings at defaults (k_bits=5,4,4,3, v_bits=5,4,4,3, use_mobius=True)

That is the complete change. The cross-attn cache is now active. You will see the [Shannon-Prime] patched ... log line on the first queue execution.

---

## Level 2: Full ship path (cross-attn + block skip)

Add three nodes: WanCache, BlockSkip, CacheFlush.

**Diagram:**

```
UnetLoaderGGUF
      |
      v
ShannonPrimeWanCache
      |
      v
ShannonPrimeWanBlockSkip ----(MODEL tapped via Reroute)----+
      |                                                     |
      v                                                     |
  KSampler                                                  |
   (LATENT out)                                             |
      |                                                     |
      +---------> ShannonPrimeWanCacheFlush <---------------+
                    (LATENT input)  (MODEL input)
                              |
                              v
                          VAEDecode
                              |
                              v
                      SaveAnimatedWEBP
```

**CacheFlush wiring detail:** The flush node takes two inputs — MODEL and LATENT. MODEL must be the BlockSkip-patched model (so the node can walk its closures). Use a Reroute node to tap the MODEL wire from BlockSkip and feed it to CacheFlush. LATENT comes directly from KSampler.

Steps:
1. Add ShannonPrimeWanCache between UnetLoader and KSampler (same as Level 1)
2. Add ShannonPrimeWanBlockSkip between WanCache and KSampler
3. Add ShannonPrimeWanCacheFlush between KSampler (LATENT out) and VAEDecode
4. Add a Reroute node on the BlockSkip → KSampler MODEL wire, and connect the Reroute to CacheFlush's MODEL input
5. Connect KSampler's LATENT output to CacheFlush's LATENT input
6. Connect CacheFlush's LATENT output to VAEDecode

---

## Level 3: Full Phase 13 path (experimental)

Extends the ship path with SigmaSwitch and RicciSentinel. Both are currently no-ops (SigmaSwitch bypasses when no sigma source is available), so adding them is safe and prepares the workflow for future activation.

```
UnetLoaderGGUF
  |
  v
ShannonPrimeWanCache
  |
  v
ShannonPrimeWanBlockSkip
  |
  v
ShannonPrimeWanSigmaSwitch    <-- new (currently bypassed)
  |
  v
ShannonPrimeWanRicciSentinel  <-- new (diagnostic)
  |
  v
KSampler
  |
  +-- (LATENT) --> ShannonPrimeWanCacheFlush --> VAEDecode
                         ^
  +-- (MODEL, tapped from BlockSkip) ----------+
```

Add SigmaSwitch and RicciSentinel after BlockSkip, before KSampler. The RicciSentinel should be the last MODEL-processing node before KSampler.

---

## Sampler compatibility

| Sampler node | Compatible | Notes |
|-------------|------------|-------|
| KSampler | Yes | Full support, primary tested path |
| SamplerCustomAdvanced | Yes | MODEL wire works the same; CacheFlush takes MODEL + LATENT, both available |
| KSamplerAdvanced | Yes | Same as KSampler |

SigmaSwitch requires transformer_options['sigmas']. This key is not currently populated by any of the above samplers in standard ComfyUI. The node detects its absence and bypasses for the generation.

---

## Cancel/Interrupt cleanup

When a generation is cancelled mid-run, BlockSkip caches may remain allocated on CPU. These will be cleared automatically at the start of the next queue execution (the patch() call in BlockSkip is unconditional). To free the memory immediately:

**Endpoint:**
```
GET http://127.0.0.1:8188/sp/cleanup
```

This walks all WanAttentionBlock forward closures on all loaded models and clears BlockSkip state dicts, then calls torch.cuda.empty_cache().

If the endpoint is not registered in your ComfyUI version, queue any workflow — the patch() call on the next real generation will clear everything unconditionally.

---

## MoE model wiring (Wan 2.2 A14B)

For MoE models with two expert paths, duplicate the Shannon-Prime nodes for each expert:

```
UnetLoaderGGUF (high expert) -> ShannonPrimeWanCache -> ShannonPrimeWanBlockSkip -+
                                                                                   +-> MoEMerge -> KSampler -> CacheFlush -> VAEDecode
UnetLoaderGGUF (low expert)  -> ShannonPrimeWanCache -> ShannonPrimeWanBlockSkip -+
```

The two WanCache nodes are independent (separate VHT2 cache objects). The two BlockSkip nodes patch separate model objects and do not share state.

For CacheFlush, one instance is sufficient — pass either patched MODEL to it. Or use two CacheFlush nodes in series for thoroughness.

---

## Stat monitoring during development

Add ShannonPrimeWanCacheStats between WanCache and BlockSkip to monitor cache behavior:

```
UnetLoaderGGUF -> ShannonPrimeWanCache -> ShannonPrimeWanCacheStats -> ShannonPrimeWanBlockSkip -> ...
```

This is optional and adds no meaningful overhead. Remove it for production.

---

## VRAM headroom checklist for 720p+

1. Launch with --normalvram --disable-async-offload
2. Do not use --lowvram — it conflicts with BlockSkip's CPU cache strategy
3. CacheFlush must be wired before VAEDecode
4. Close other VRAM-consuming applications before generation
5. If VAEDecode still OOMs, reduce resolution or frame count, or switch to the Wan 2.1 1.3B model
