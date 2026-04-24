# Shannon-Prime VHT2: ComfyUI Custom Nodes
# Copyright (C) 2026 Ray Daniels. All Rights Reserved.
# Licensed under AGPLv3. Commercial license available.
#
# Provides a ShannonPrimeWanCache node that wraps Wan 2.1/2.2 cross-attention
# K/V linear layers with VHT2-compressed caching. Text context (T5/UMT5 output)
# is identical across all diffusion timesteps, so the K/V projections of that
# context are recomputed ~50× needlessly in a vanilla Wan inference. This node
# intercepts those computations via drop-in nn.Module replacements on
# `block.cross_attn.k` and `block.cross_attn.v` (and `.k_img`/`.v_img` for
# I2V), caching the first computation and returning VHT2-reconstructed values
# on subsequent calls.
#
# Works with ComfyUI's native Wan implementation at comfy.ldm.wan.model.
# Safe to apply twice (idempotent). Cache is keyed per-block per-linear.
# Input-change detection uses the tensor's data_ptr: when the upstream context
# tensor is a new allocation (i.e. a new generation), cache entries for that
# linear are invalidated and refilled.

import os
import sys
import pathlib

import torch
import torch.nn as nn

# Make the shannon-prime submodule importable.
_THIS_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
_SP_TOOLS = _REPO_ROOT / "lib" / "shannon-prime" / "tools"
_SP_TORCH = _REPO_ROOT / "lib" / "shannon-prime" / "backends" / "torch"
for p in (_SP_TOOLS, _SP_TORCH):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

# shannon_prime_comfyui submodule — only needed by the sqfree node (lazy import there).



def _input_fingerprint(x: torch.Tensor):
    """
    Cheap content-based fingerprint — identifies same-valued inputs even when
    re-allocated to a new memory address.

    We can't use `data_ptr()` as the identity for cache hits because ComfyUI's
    sampler re-batches the conditioning tensor into a fresh allocation each
    timestep. Pointer-identity invalidated the cache every step and the hits
    documented in the paper never materialised.

    Three flat-index samples + shape + dtype disambiguate "same context across
    timesteps" vs "fresh context for a new generation" without scanning the
    full tensor. On CUDA each `.item()` forces a single-element device→host
    sync; three of those per cross-attn forward is negligible (~10 µs total)
    compared to the compress/decompress work they save.
    """
    flat = x.view(-1) if x.is_contiguous() else x.reshape(-1)
    n = flat.numel()
    # Pick three anchors that spread across the tensor; identical data will
    # fingerprint identically regardless of reallocation.
    i_mid = n // 2 if n > 1 else 0
    i_end = n - 1 if n > 0 else 0
    return (
        tuple(x.shape),
        x.dtype,
        float(flat[0].item()),
        float(flat[i_mid].item()),
        float(flat[i_end].item()),
    )


class _SPCachingLinear(nn.Module):
    """
    Drop-in for cross_attn.k / cross_attn.v (and k_img / v_img).

    Raw CPU/fp16 cache — no VHT2 compression, no GPU temporaries.
    Cross-attn K/V are constant across all diffusion timesteps (same prompt
    = same T5 embeddings = same output). Compute once on step 1, store raw
    on CPU, return via .to(device) on subsequent steps (~3ms per hit via PCIe).

    Phase 15 LEAN: the VHT2 compress/decompress path created GPU tensor
    allocation storms on lowvram (160 ops/step × GPU intermediates = CUDA
    fragmentation → model weight thrashing). Raw CPU eliminates this entirely
    and fixes the "must cancel first run" warm-up issue.
    """

    def __init__(self, original: nn.Module, key: str):
        super().__init__()
        self.original = original
        self._sp_key = key
        self._sp_last_fp = None
        self._sp_cached = None  # raw CPU fp16 tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fp = _input_fingerprint(x)

        # Cache hit: same conditioning content, have cached result
        if fp == self._sp_last_fp and self._sp_cached is not None:
            return self._sp_cached.to(device=x.device, dtype=x.dtype)

        # Miss: compute, store raw on CPU
        result = self.original(x)
        self._sp_cached = result.detach().cpu()
        self._sp_last_fp = fp
        return result


def _wrap_cross_attn(cross_attn: nn.Module, block_idx: int) -> bool:
    """Wrap .k, .v (and .k_img, .v_img if present) with raw CPU caching."""
    if cross_attn is None:
        return False
    wrapped_any = False
    for suffix, attr in (("k", "k"), ("v", "v"), ("kimg", "k_img"), ("vimg", "v_img")):
        lin = getattr(cross_attn, attr, None)
        if lin is None:
            continue
        if isinstance(lin, _SPCachingLinear):
            # Already wrapped — just clear its cache for the new run
            lin._sp_cached = None
            lin._sp_last_fp = None
            continue
        wrapper = _SPCachingLinear(lin, f"block_{block_idx}_{suffix}")
        setattr(cross_attn, attr, wrapper)
        wrapped_any = True
    return wrapped_any


def _iter_wan_blocks(model_obj):
    """Yield (index, block) for each WanAttentionBlock inside a ModelPatcher."""
    inner = getattr(model_obj, "model", model_obj)
    diff = getattr(inner, "diffusion_model", None)
    if diff is None:
        diff = getattr(inner, "model", None)
    if diff is None:
        return
    blocks = getattr(diff, "blocks", None)
    if blocks is None:
        return
    for i, blk in enumerate(blocks):
        yield i, blk


def _parse_bits_csv(s: str, default, width: int):
    try:
        out = [int(x.strip()) for x in str(s).split(",") if x.strip()]
    except ValueError:
        return list(default)
    if not out:
        return list(default)
    # Pad or truncate to the requested width by repeating the last element.
    while len(out) < width:
        out.append(out[-1])
    return out[:width]


class ShannonPrimeWanCache:
    """
    Patches a Wan 2.1/2.2 MODEL to cache cross-attention K/V via Shannon-Prime
    VHT2 compression.

    Cross-attention K/V in Wan are linear projections of T5/UMT5 text
    embeddings. That context is constant across the ~50 diffusion timesteps
    in a generation, so the K/V tensors are invariant — computing them once
    and reusing them is strictly profitable for both compute and VRAM.

    The node monkey-patches block.cross_attn.{k, v} (and .k_img/.v_img on
    I2V) on every WanAttentionBlock it finds. First call through each
    wrapped linear computes + compresses; subsequent calls with the same
    context tensor (same data_ptr) reconstruct from the VHT2 cache.

    Apply once per Wan MODEL before the sampler. For Wan 2.2 MoE, apply
    separately to each expert MODEL — the two caches are naturally partitioned
    because the two experts are separate Python objects.
    """

    CATEGORY = "shannon-prime"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    DESCRIPTION = ("Wraps Wan cross-attention K/V with VHT2 compressed caching. "
                   "Cross-attn context is constant across timesteps; compute once, "
                   "compress, reconstruct on subsequent calls.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "k_bits": ("STRING", {"default": "5,4,4,3",
                                      "tooltip": "K band bit allocation (4 bands)"}),
                "v_bits": ("STRING", {"default": "5,4,4,3",
                                      "tooltip": "V band bit allocation (4 bands for cross-attn, unlike self-attn flat 3-bit)"}),
                "use_mobius": ("BOOLEAN", {"default": True,
                                           "tooltip": "Möbius squarefree-first reorder on both K and V (cross-attn has no RoPE)"}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # The patch isn't deterministic w.r.t. pure input hashing — different
        # applications share cache state on the model. Returning NaN forces
        # re-evaluation each queue.
        return float("nan")

    def patch(self, model, k_bits: str = "5,4,4,3", v_bits: str = "5,4,4,3",
              use_mobius: bool = True):
        # Phase 15 LEAN: cross-attn cache is now raw CPU/fp16 — no VHT2.
        # k_bits, v_bits, use_mobius are retained in INPUT_TYPES for backward
        # compatibility with saved workflows but are IGNORED. The node just
        # wraps cross-attn linears with raw CPU caching.

        patched = model.clone()

        blocks = list(_iter_wan_blocks(patched))
        if not blocks:
            print("[Shannon-Prime] ShannonPrimeWanCache: no Wan blocks found on model — passing through")
            return (patched,)

        wrapped = 0
        for i, blk in blocks:
            if _wrap_cross_attn(getattr(blk, "cross_attn", None), i):
                wrapped += 1

        print(f"[Shannon-Prime] Phase 15 LEAN — patched {wrapped}/{len(blocks)} "
              f"Wan cross-attn linears with raw CPU/fp16 caching")
        print(f"[Shannon-Prime] No VHT2, no Möbius, no GPU temporaries — "
              f"zero VRAM overhead on cache hit")

        return (patched,)


class ShannonPrimeWanCacheStats:
    """Reports cache hit/miss stats from a model previously patched by
    ShannonPrimeWanCache. Passes the model through unchanged."""

    CATEGORY = "shannon-prime"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "report"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    @classmethod
    def IS_CHANGED(cls, *a, **k):
        return float("nan")

    def report(self, model):
        # Phase 15 LEAN: cross-attn cache is raw CPU/fp16 on each _SPCachingLinear.
        # Walk the blocks and count cached vs uncached wrappers.
        hits = 0
        total = 0
        for _i, blk in _iter_wan_blocks(model):
            ca = getattr(blk, "cross_attn", None)
            if ca is None:
                continue
            for attr in ("k", "v", "k_img", "v_img"):
                lin = getattr(ca, attr, None)
                if isinstance(lin, _SPCachingLinear):
                    total += 1
                    if lin._sp_cached is not None:
                        hits += 1
        print(f"[Shannon-Prime] cross-attn cache: {hits}/{total} linears cached "
              f"(raw CPU/fp16, no VHT2)")
        return (model,)


class ShannonPrimeWanCacheSqfree:
    """
    Aggressive variant of ShannonPrimeWanCache using the sqfree+spinor path.

    Wraps Wan cross-attn K/V with the sqfree prime-Hartley basis + Möbius CSR
    predictor + (optional) SU(2) spinor sheet-bit correction. Target regime:
    Q8+ text encoders + bf16 diffusion weights (per CLAUDE.md scaling law).
    Uses `VHT2SqfreeCrossAttentionCache` + `WanSqfreeCrossAttnCachingLinear`
    from the submodule's sqfree tool — these expose a get_or_compute API
    rather than the put/get used by the WHT cache, so cannot be dropped into
    the WHT node without a second wrapper class.
    """

    CATEGORY = "shannon-prime"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    DESCRIPTION = ("Sqfree+spinor aggressive variant of the Wan cross-attention cache. "
                   "Higher compression at equivalent quality on Q8+ backbones.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "band_bits": ("STRING", {"default": "3,3,3,3,3",
                                         "tooltip": "5-band torus-aligned bit allocation (default aggressive 3/3/3/3/3)"}),
                "residual_bits": ("INT", {"default": 3, "min": 1, "max": 4,
                                          "tooltip": "N-bit residual quantization. 3 is the Shannon saturation point."}),
                "use_spinor": ("BOOLEAN", {"default": True,
                                           "tooltip": "Enable SU(2) spinor sheet-bit correction at the causal-mask boundary"}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

    def patch(self, model, band_bits: str, residual_bits: int, use_spinor: bool):
        band_bits_list = _parse_bits_csv(band_bits, [3, 3, 3, 3, 3], 5)

        from shannon_prime_comfyui_sqfree import (  # local import: only needed for sqfree path
            VHT2SqfreeCrossAttentionCache,
            WanSqfreeCrossAttnCachingLinear,
        )

        patched = model.clone()
        blocks = list(_iter_wan_blocks(patched))
        if not blocks:
            print("[Shannon-Prime SQFREE] no Wan blocks found on model — passing through")
            return (patched,)

        head_dim = blocks[0][1].self_attn.head_dim
        max_blocks = len(blocks)

        cache = VHT2SqfreeCrossAttentionCache(
            head_dim=head_dim,
            max_blocks=max_blocks,
            use_spinor=bool(use_spinor),
            residual_bits=int(residual_bits),
            band_bits=band_bits_list,
        )

        wrapped = 0
        for i, blk in blocks:
            cross_attn = getattr(blk, "cross_attn", None)
            if cross_attn is None:
                continue
            for suffix, attr in (("k", "k"), ("v", "v"), ("kimg", "k_img"), ("vimg", "v_img")):
                lin = getattr(cross_attn, attr, None)
                if lin is None or isinstance(lin, WanSqfreeCrossAttnCachingLinear):
                    continue
                wrapper = WanSqfreeCrossAttnCachingLinear(lin, cache, f"block_{i}_{suffix}")
                setattr(cross_attn, attr, wrapper)
            wrapped += 1

        patched._sp_sqfree_cache = cache
        pad_dim = cache._cache.pad_dim  # internal, but useful for the log

        print(f"[Shannon-Prime SQFREE] patched {wrapped}/{len(blocks)} Wan blocks "
              f"(head_dim={head_dim} -> pad_dim={pad_dim}, bands={band_bits_list}, "
              f"residual_bits={residual_bits}, spinor={use_spinor})")

        return (patched,)


class ShannonPrimeWanSelfExtract:
    """
    Phase 12 diagnostic: captures Wan self-attention K vectors during denoising.

    Hooks into `blk.self_attn.k` for every WanAttentionBlock and captures
    the K projection output at the specified denoising step (0 = first/noisiest,
    mid = structural regime, late = texture regime).

    Output saved as .npz compatible with sp_diagnostics.py:
        k_vectors shape: (n_blocks, n_kv_heads, n_tokens, head_dim)

    Usage after running:
        python sp_diagnostics.py --input wan_self_attn.npz --sqfree --layer-period 4

    Connect BEFORE the sampler. The MODEL output is the same model — this node
    is an observer, not a patcher. The .npz is written once the target step
    fires, then hooks are removed automatically.
    """

    CATEGORY = "shannon-prime/diagnostics"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "attach"
    DESCRIPTION = ("Captures Wan self-attention K vectors at a target denoising step "
                   "and saves them as .npz for sp_diagnostics Phase 12 analysis.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "capture_step": ("INT", {
                    "default": 25, "min": 0, "max": 200,
                    "tooltip": "Denoising step at which to capture K (0=first/highest-sigma)"}),
                "max_tokens": ("INT", {
                    "default": 256, "min": 64, "max": 4096,
                    "tooltip": "Max token positions to store per block (caps memory)"}),
                "output_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Directory for .npz output (default: ComfyUI output/shannon_prime/)"}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, *a, **k):
        return float("nan")

    def attach(self, model, capture_step=25, max_tokens=256, output_dir=""):
        import folder_paths as _fp
        import numpy as np, json, time

        if not output_dir:
            output_dir = os.path.join(_fp.get_output_directory(), "shannon_prime")
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"wan_self_attn_step{capture_step}.npz")

        patched = model.clone()
        blocks  = list(_iter_wan_blocks(patched))
        if not blocks:
            print("[SP SelfExtract] no Wan blocks found — passing through")
            return (patched,)

        n_blocks  = len(blocks)
        head_dim  = blocks[0][1].self_attn.head_dim

        # Shared mutable state across hooks (dict avoids closure rebind issues)
        state = {
            "step":      0,         # counts how many times block-0's K fired
            "captured":  {},        # block_idx -> torch.Tensor (n_kv_heads, T, head_dim)
            "target":    capture_step,
            "done":      False,
            "handles":   [],
            "max_tok":   max_tokens,
            "head_dim":  head_dim,
        }

        def make_hook(block_idx):
            def _hook(module, _inp, output):
                if state["done"]:
                    return
                # Count steps via block 0 as the "clock" block
                if block_idx == 0:
                    state["step"] += 1

                if state["step"] - 1 == state["target"]:
                    # output: (batch*tokens, n_kv_heads*head_dim)  or
                    #         (batch, tokens, n_kv_heads*head_dim)
                    k = output.detach().cpu().float()
                    if k.dim() == 2:
                        # (B*T, D) — reshape; we don't know B, assume B=1
                        k = k.unsqueeze(0)          # (1, B*T, D)
                    # k: (B, T, D)  where D = n_kv_heads * head_dim
                    T  = k.shape[1]
                    D  = k.shape[2]
                    hd = state["head_dim"]
                    n_kv = max(1, D // hd) if hd > 0 else 1

                    k = k[0]                         # (T, D) — take first batch
                    T_cap = min(T, state["max_tok"])
                    k = k[:T_cap]                    # (T_cap, D)
                    k = k.reshape(T_cap, n_kv, hd)  # (T_cap, n_kv, hd)
                    k = k.permute(1, 0, 2)           # (n_kv, T_cap, hd)
                    state["captured"][block_idx] = k

                # After capturing all blocks at the target step, save
                if (state["step"] - 1 == state["target"]
                        and block_idx == n_blocks - 1
                        and not state["done"]):
                    state["done"] = True
                    _save(state, out_path, n_blocks, capture_step, head_dim)
                    for h in state["handles"]:
                        h.remove()
                    state["handles"].clear()
            return _hook

        # Attach hooks to blk.self_attn.k (try common names)
        n_hooked = 0
        for i, blk in blocks:
            sa = getattr(blk, "self_attn", None)
            if sa is None:
                continue
            k_lin = None
            for attr in ("k", "k_proj", "to_k", "wk", "key"):
                candidate = getattr(sa, attr, None)
                if isinstance(candidate, nn.Module):
                    k_lin = candidate
                    break
            if k_lin is None:
                continue
            h = k_lin.register_forward_hook(make_hook(i))
            state["handles"].append(h)
            n_hooked += 1

        print(f"[SP SelfExtract] hooked {n_hooked}/{n_blocks} self-attn K projections "
              f"(head_dim={head_dim}, capture_step={capture_step}, "
              f"max_tokens={max_tokens})")
        print(f"[SP SelfExtract] will save to: {out_path}")

        return (patched,)


def _save(state, out_path, n_blocks, capture_step, head_dim):
    """Assemble captured K dicts → (n_blocks, n_kv_heads, T, hd) .npz."""
    import numpy as np, json

    captured = state["captured"]
    if not captured:
        print("[SP SelfExtract] Nothing captured — did the generation complete?")
        return

    # Find common n_kv_heads and T from first block
    sample = next(iter(captured.values()))   # (n_kv, T, hd)
    n_kv_heads = sample.shape[0]
    T_cap      = sample.shape[1]

    k_arr = torch.zeros(n_blocks, n_kv_heads, T_cap, head_dim)
    for idx, k in captured.items():
        # k: (n_kv, T_cap, hd) — pad if mismatched (safety)
        nk = min(k.shape[0], n_kv_heads)
        nt = min(k.shape[1], T_cap)
        k_arr[idx, :nk, :nt, :] = k[:nk, :nt, :]

    k_np = k_arr.numpy().astype("float32")

    meta = {
        "source":        "wan_self_attention",
        "capture_step":  capture_step,
        "n_blocks":      n_blocks,
        "n_kv_heads":    n_kv_heads,
        "n_tokens":      T_cap,
        "head_dim":      head_dim,
        "note": ("Wan DiT self-attention K vectors. "
                 "Axis 0 = DiT block (analogous to 'layer'), "
                 "Axis 1 = KV heads, Axis 2 = token positions, Axis 3 = head_dim. "
                 "Run: sp_diagnostics.py --input <this_file> --sqfree --layer-period 4"),
    }

    import numpy as np
    np.savez_compressed(out_path,
                        k_vectors=k_np,
                        metadata=np.array(json.dumps(meta)))

    print(f"[SP SelfExtract] saved {k_np.shape} K vectors -> {out_path}")
    print(f"[SP SelfExtract] Next: python sp_diagnostics.py "
          f"--input {out_path} --sqfree --layer-period 4 --global-offset 3")



class ShannonPrimeWanBlockSkip:
    """
    Phase 12/13 Block-Level Self-Attention Skip for Wan 2.x DiT.

    Patches WanAttentionBlock.forward() to skip the ENTIRE self-attention
    computation (Q/K/V projections + attention scores) on cache-hit steps.

    adaLN-correct cache hit path:
      1. Recompute adaLN modulation from current timestep embedding (trivial cost)
      2. Apply cached pre-gate attention output (y) with the CURRENT step's gate (e[2])
         — this correctly tracks sigma-dependent brightness/contrast changes
      3. Run cross-attention and FFN fresh every step (sigma-accurate)

    This achieves ~50% compute skip for stable blocks (L00-L03) on cached steps:
      - Skipped: norm1, Q projection, K projection, V projection, attention scores,
                 output projection  (O(tokens² × dim) — dominant cost at 4K)
      - Not skipped: adaLN modulation recompute (trivial), cross-attn, FFN

    Per-block rolling cosine similarity oracle (Mertens Oracle):
      Each block tracks rolling cos_sim between cached y and fresh y.
      If rolling_sim drops below drift_threshold, window is halved automatically.
      If it recovers above restore_threshold, full window is restored.

    Block tier map (from sigma-sweep Phase 12 data):
      L00-L03: window=10 steps (Permanent Granite, cos_sim>0.95)
      L04-L08: window=3  steps (Stable Sand)
      L09+:    window=0  steps (no cache — volatile)

    Hierarchical cache tiering (Phase 14, default):
      Tier 0 (L00-L03): GPU-resident — zero overhead on cache hit.
        No CPU→GPU transfer, no transform, no copy. Direct tensor reuse.
        ~160MB GPU for 4 blocks at 720p (worth it: these hit 10+ steps).
      Tier 1 (L04-L08): CPU — fast .to(device) on hit (~5ms per block).
        ~200MB CPU. Freed during VAE via CacheFlush node.
      Tier 2 (L09+):    No cache — recompute every step (volatile blocks).

    VHT2 compression (optional, default OFF):
      Available via vht2_compress=True for extreme VRAM pressure.
      ~3.5x memory reduction but significant compute overhead at Wan scale
      (165K vectors × 128D butterfly per block). Not recommended for normal use.
    """

    CATEGORY     = "shannon-prime"
    RETURN_TYPES = ("MODEL",)
    FUNCTION     = "patch"
    DESCRIPTION  = (
        "Block-level self-attention skip for Wan DiT. "
        "Patches WanAttentionBlock.forward() to skip Q/K/V+attention "
        "for stable early blocks. adaLN-correct: caches pre-gate y, "
        "reapplies current gate on cache hit. ~50% compute saving for L00-L03."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"model": ("MODEL",)},
            "optional": {
                "tier_0_window": ("INT",   {"default": 10, "min": 0, "max": 30,
                    "tooltip": "Cache window for L00-L03 (Permanent Granite). 0=disabled."}),
                "tier_1_window": ("INT",   {"default": 3,  "min": 0, "max": 15,
                    "tooltip": "Cache window for L04-L08 (Stable Sand). 0=disabled."}),
                "verbose": ("BOOLEAN", {"default": False,
                    "tooltip": "Print per-block HIT/MISS logs to console"}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, *a, **k):
        return float("nan")

    def patch(self, model, tier_0_window=10, tier_1_window=3,
              verbose=False, **_ignored):
        import types
        import comfy.model_management

        try:
            from comfy.ldm.wan.model import repeat_e
        except ImportError:
            # Fallback implementation of repeat_e if import fails
            def repeat_e(e, x):
                repeats = 1
                if e.size(1) > 1:
                    repeats = x.size(1) // e.size(1)
                if repeats == 1:
                    return e
                return e.repeat_interleave(repeats, dim=1)

        patched = model.clone()
        blocks  = list(_iter_wan_blocks(patched))
        if not blocks:
            print("[SP BlockSkip] no Wan blocks found — passing through")
            return (patched,)

        # ── Clear stale caches from previous runs ──────────────────────────────
        # patch() is called at the START of every ComfyUI prompt execution.
        _cleared = 0
        for _i, _blk in blocks:
            _fwd = getattr(_blk, "forward", None)
            if _fwd and hasattr(_fwd, "__closure__") and _fwd.__closure__:
                for _cell in _fwd.__closure__:
                    try:
                        _obj = _cell.cell_contents
                        if isinstance(_obj, dict) and "attn_cache" in _obj:
                            _obj["attn_cache"].clear()
                            _obj.get("step_cached", {}).clear()
                            _obj.get("hit_streak", {}).clear()
                            _cleared += 1
                    except ValueError:
                        pass
        if _cleared:
            print(f"[SP BlockSkip] cleared {_cleared} stale cache(s) from previous run")

        n_blocks = len(blocks)

        # Tier window map
        def get_window(i):
            if i < 4:
                return tier_0_window
            if i < 9:
                return tier_1_window
            return 0

        # Shared state across all patched blocks — lean Phase 15
        # No oracle, no x_drift, no rolling_sim, no runtime gen detection.
        # Fixed windows from Phase 12 data. All caches CPU/fp16.
        state = {
            'global_step':    [0],      # step counter (incremented by block-0)
            'hit_streak':     {},       # block_idx -> consecutive cache-hit count
            'attn_cache':     {},       # block_idx -> y tensor (CPU fp16)
            'step_cached':    {},       # block_idx -> step at which y was cached
        }

        def make_patched_forward(orig_forward, block_idx, blk):
            window = get_window(block_idx)
            # Fixed streak limit: force a miss every N hits to refresh cache.
            # Phase 12 data: tier-0 sim>0.95, tier-1 sim>0.90 — safe to trust.
            _streak_limit = 10 if block_idx < 4 else 5

            def patched_forward(x, e, freqs, context,
                                context_img_len=257,
                                transformer_options={}):
                # ── Step counter ────────────────────────────────────────────
                if block_idx == 0:
                    state['global_step'][0] += 1
                step = state['global_step'][0]

                # Generation boundary: handled by patch() clearing at prompt
                # start. NO runtime detection — SVI distilled schedules have
                # non-monotonic sigma, so e_mag increase != new generation.
                # Caches persist within a prompt (across batch outputs) which
                # is fine: Phase 12 data shows sim>0.95 for tier-0 blocks.

                # ── Print run info on step 1 ────────────────────────────────
                if block_idx == 0 and step == 1:
                    print(f"[SP BlockSkip] step=1 x.shape={tuple(x.shape)}  "
                          f"tokens={x.shape[1]}  dtype={x.dtype}  device={x.device}")

                # ── Recompute adaLN modulation (always — cheap) ─────────────
                cast = comfy.model_management.cast_to
                if e.ndim < 4:
                    e_mods = (cast(blk.modulation, dtype=x.dtype, device=x.device)
                              + e).chunk(6, dim=1)
                else:
                    e_mods = (cast(blk.modulation, dtype=x.dtype, device=x.device)
                              .unsqueeze(0) + e).unbind(2)

                cached_s = state['step_cached'].get(block_idx, -999)
                age = step - cached_s
                streak = state['hit_streak'].get(block_idx, 0)
                cached_y = state['attn_cache'].get(block_idx)

                # Shape validation
                shape_ok = (cached_y is not None
                            and cached_y.shape[0] == x.shape[0]
                            and cached_y.shape[1] == x.shape[1])
                if not shape_ok and cached_y is not None:
                    del state['attn_cache'][block_idx]
                    state['step_cached'].pop(block_idx, None)
                    cached_y = None

                # ── CACHE HIT: within window, not streak-forced-miss ────────
                if (cached_y is not None
                        and age >= 0 and age < window
                        and streak < _streak_limit):
                    # CPU fp16 → GPU: ~40MB at ~12 GB/s PCIe = ~3ms. Negligible.
                    y = cached_y.to(device=x.device, dtype=x.dtype)
                    x = torch.addcmul(x.contiguous(), y,
                                      repeat_e(e_mods[2], x))
                    state['hit_streak'][block_idx] = streak + 1
                    if verbose:
                        print(f"[SP BlockSkip] B{block_idx:02d} HIT  "
                              f"step={step} age={age}/{window} "
                              f"streak={streak+1}/{_streak_limit} CPU/fp16")
                else:
                    # ── CACHE MISS: run self-attention ─────────────────────────
                    x = x.contiguous()
                    y = blk.self_attn(
                        torch.addcmul(repeat_e(e_mods[0], x),
                                      blk.norm1(x),
                                      1 + repeat_e(e_mods[1], x)),
                        freqs, transformer_options=transformer_options
                    )

                    # Store y on CPU fp16 — zero GPU memory pressure.
                    # ~40MB per block, 9 blocks = ~360MB CPU. Transfer on hit
                    # is ~3ms/block via PCIe — invisible against step time.
                    state['attn_cache'][block_idx] = y.detach().cpu()
                    state['step_cached'][block_idx] = step
                    state['hit_streak'][block_idx] = 0

                    if verbose:
                        print(f"[SP BlockSkip] B{block_idx:02d} MISS "
                              f"step={step} CPU/fp16")

                    x = torch.addcmul(x, y, repeat_e(e_mods[2], x))
                    del y

                # ── Cross-attention + FFN (always fresh) ─────────────────────
                x = x + blk.cross_attn(
                    blk.norm3(x), context,
                    context_img_len=context_img_len,
                    transformer_options=transformer_options
                )
                patches = transformer_options.get("patches", {})
                if "attn2_patch" in patches:
                    for p in patches["attn2_patch"]:
                        x = p({"x": x, "transformer_options": transformer_options})

                y = blk.ffn(torch.addcmul(repeat_e(e_mods[3], x),
                                          blk.norm2(x),
                                          1 + repeat_e(e_mods[4], x)))
                x = torch.addcmul(x, y, repeat_e(e_mods[5], x))
                return x

            return patched_forward

        # Patch each block's forward method
        n_patched = 0
        for i, blk in blocks:
            window = get_window(i)
            if window > 0:
                orig   = blk.forward
                blk.forward = make_patched_forward(orig, i, blk)
                n_patched += 1

        tier0 = [i for i, _ in blocks if i < 4]
        tier1 = [i for i, _ in blocks if 4 <= i < 9]
        print(f"[SP BlockSkip] Phase 15 LEAN — patched {n_patched}/{n_blocks} blocks")
        print(f"[SP BlockSkip] ALL CPU/fp16 (lowvram-safe, no GPU memory pinned)")
        print(f"[SP BlockSkip] Tier-0 win={tier_0_window}: {tier0}")
        print(f"[SP BlockSkip] Tier-1 win={tier_1_window}: {tier1}")
        print(f"[SP BlockSkip] Streak limits: tier-0=10, tier-1=5")
        print(f"[SP BlockSkip] Zero overhead: no oracle, no x_drift, no cos_sim")

        return (patched,)


class ShannonPrimeWanCacheFlush:
    """
    Flush the ShannonPrimeWanBlockSkip y-cache before VAE decode.

    Place between KSampler and VAEDecode. Clears the ~594MB of cached y
    tensors (CPU) that BlockSkip holds after denoising, then calls
    torch.cuda.empty_cache() to give the VAE maximum memory headroom.

    Without this node, BlockSkip's own VAE decode is ~34s slower than
    baseline because the y-cache holds memory through the decode. With
    this node, total time matches or beats baseline.

    Workflow: UnetLoader → [ShannonPrimeWanCache] → ShannonPrimeWanBlockSkip
              → KSampler → [ShannonPrimeWanCacheFlush] → VAEDecode → Save
    """

    CATEGORY    = "shannon-prime"
    RETURN_TYPES = ("LATENT",)
    FUNCTION    = "flush"
    DESCRIPTION = (
        "Flush BlockSkip y-cache before VAE decode. "
        "Eliminates the ~34s VAE penalty from cached attention tensors "
        "sitting in CPU memory during high-res (720p+) decoding."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":   ("MODEL",),
                "samples": ("LATENT",),
            },
        }

    @classmethod
    def IS_CHANGED(cls, *a, **k):
        return float("nan")

    def flush(self, model, samples):
        flushed_blockskip = 0
        flushed_crossattn = 0

        for i, blk in _iter_wan_blocks(model):
            # ── Clear BlockSkip self-attn caches (closure state dicts) ──
            fwd = getattr(blk, "forward", None)
            if fwd and hasattr(fwd, "__closure__") and fwd.__closure__:
                for cell in fwd.__closure__:
                    try:
                        obj = cell.cell_contents
                    except ValueError:
                        continue
                    if isinstance(obj, dict) and "attn_cache" in obj:
                        n = len(obj["attn_cache"])
                        obj["attn_cache"].clear()
                        obj.get("step_cached", {}).clear()
                        obj.get("hit_streak", {}).clear()
                        if n > 0:
                            flushed_blockskip += 1

            # ── Clear cross-attn raw CPU caches (_SPCachingLinear) ──────
            ca = getattr(blk, "cross_attn", None)
            if ca is not None:
                for attr in ("k", "v", "k_img", "v_img"):
                    lin = getattr(ca, attr, None)
                    if isinstance(lin, _SPCachingLinear) and lin._sp_cached is not None:
                        lin._sp_cached = None
                        lin._sp_last_fp = None
                        flushed_crossattn += 1

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        total = flushed_blockskip + flushed_crossattn
        if total > 0:
            print(f"[SP CacheFlush] cleared {flushed_blockskip} BlockSkip + "
                  f"{flushed_crossattn} cross-attn caches + torch.cuda.empty_cache()")
        else:
            print("[SP CacheFlush] no caches found (node still safe to use)")

        return (samples,)


class ShannonPrimeWanSigmaSwitch:
    """
    Phase 13 — Sigma-aware cache window adaptation for Wan DiT blocks.

    The sigma schedule in diffusion is the direct analog of the L/2 phase
    transition in LLMs. At high sigma (early steps, Arithmetic Granite), the
    model establishes global composition and the early blocks are maximally
    stable — we can cache aggressively. At low sigma (late steps, Semantic
    Sand), the model refines textures and the blocks drift faster — we cache
    conservatively.

    Sigma source: registers a model_function_wrapper on ComfyUI's ModelPatcher
    via set_model_unet_function_wrapper(). The wrapper fires once per denoising
    step and receives the raw sigma tensor directly from the sampler via
    args_dict["timestep"]. This is more direct than reading
    transformer_options['sigmas'] inside block forwards, and avoids any
    dict-propagation ambiguity through the model forward chain.

    The captured sigma is stored in a shared dict (_sp_sigma_state on the
    patched model) so downstream nodes (RicciSentinel) can read it without
    closure-chain walking.

    Mechanism:
      - model_function_wrapper captures real sigma each step, tracks rolling
        max and min across the generation.
      - Block-0 forward hook reads current sigma and classifies as HIGH/LOW.
      - Above threshold (high sigma): effective_win *= high_mult (more caching)
      - Below threshold (low sigma):  effective_win *= low_mult  (less caching)
      - The switch point is sigma_split_frac of the [min, max] sigma range.

    Phase 12 data supports this:
      - Sigma sweep showed blocks L00-L03 cos_sim > 0.95 at steps 2-9 (high sigma)
        but dropping to 0.6-0.7 by steps 14-20 (low sigma).
      - Wider windows are correct at high sigma, tighter at low sigma.

    Workflow: UnetLoader → ShannonPrimeWanCache → ShannonPrimeWanBlockSkip
              → ShannonPrimeWanSigmaSwitch → KSampler → CacheFlush → VAEDecode
    """

    CATEGORY     = "shannon-prime"
    RETURN_TYPES = ("MODEL",)
    FUNCTION     = "attach"
    DESCRIPTION  = (
        "Phase 13: sigma-aware cache window adaptation for BlockSkip. "
        "Expands windows at high sigma (Arithmetic Granite) and contracts "
        "them at low sigma (Semantic Sand) for dynamic per-step compression."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"model": ("MODEL",)},
            "optional": {
                "high_sigma_mult": ("FLOAT", {
                    "default": 1.5, "min": 1.0, "max": 4.0, "step": 0.1,
                    "tooltip": "Window multiplier at high sigma (Arithmetic Granite). "
                               "1.5 = 50% longer cache window in early steps."}),
                "low_sigma_mult": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1,
                    "tooltip": "Window multiplier at low sigma (Semantic Sand). "
                               "0.5 = half the cache window in late steps (more recompute)."}),
                "sigma_split_frac": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 0.9, "step": 0.05,
                    "tooltip": "Fraction of the sigma range at which to switch. "
                               "0.5 = switch at midpoint between max and min sigma."}),
                "verbose": ("BOOLEAN", {"default": False}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, *a, **k):
        return float("nan")

    def attach(self, model, high_sigma_mult=1.5, low_sigma_mult=0.5,
               sigma_split_frac=0.5, verbose=False):
        patched = model.clone()
        blocks  = list(_iter_wan_blocks(patched))
        if not blocks:
            print("[SP SigmaSwitch] no Wan blocks found — passing through")
            return (patched,)

        # ── Shared sigma state ────────────────────────────────────────────────
        # Populated by the model_function_wrapper (fires once per denoising
        # step with real sigma from the sampler) and consumed by the block-0
        # forward hook below + RicciSentinel (if attached downstream).
        sigma_state = {
            'current_sigma': [None],  # float — latest sigma from sampler
            'sigma_max':     [None],  # rolling max sigma this generation
            'sigma_min':     [None],  # rolling min sigma this generation
            'high_mult':     high_sigma_mult,
            'low_mult':      low_sigma_mult,
            'split':         sigma_split_frac,
            'step_count':    [0],     # steps seen via wrapper this generation
        }

        # ── model_function_wrapper: capture real sigma ────────────────────────
        # ComfyUI's ModelPatcher calls model_function_wrapper as:
        #   wrapper(apply_model_func, args_dict)
        # where args_dict["timestep"] is the raw sigma tensor from the sampler.
        # Note: samplers.py also sets transformer_options["sigmas"] = timestep,
        # but capturing sigma here at the wrapper level is more direct and
        # avoids any dict-propagation ambiguity through the model forward chain.
        def sigma_wrapper(apply_model_func, args_dict):
            timestep = args_dict.get("timestep", None)
            if timestep is not None:
                try:
                    sig = float(timestep.flatten()[0])
                    sigma_state['current_sigma'][0] = sig
                    sigma_state['step_count'][0] += 1

                    # Rolling max/min for regime classification
                    if sigma_state['sigma_max'][0] is None:
                        sigma_state['sigma_max'][0] = sig
                        sigma_state['sigma_min'][0] = sig
                    else:
                        sigma_state['sigma_max'][0] = max(sigma_state['sigma_max'][0], sig)
                        sigma_state['sigma_min'][0] = min(sigma_state['sigma_min'][0], sig)
                except Exception:
                    pass
            return apply_model_func(args_dict["input"], args_dict["timestep"],
                                    **args_dict.get("c", {}))

        patched.set_model_unet_function_wrapper(sigma_wrapper)

        # ── Store sigma_state on the patched model so downstream nodes ────────
        # (RicciSentinel) can read it without closure-chain walking.
        if not hasattr(patched, '_sp_sigma_state'):
            patched._sp_sigma_state = sigma_state

        def make_sigma_hook(blk, block_idx):
            """Wrap block-0 forward to adjust BlockSkip windows based on sigma."""
            orig_forward = blk.forward

            def sigma_forward(x, e, freqs, context, context_img_len=257,
                              transformer_options={}):
                if block_idx == 0:
                    sig_val = sigma_state['current_sigma'][0]

                    if sig_val is None:
                        # Wrapper hasn't fired yet (shouldn't happen, but be safe)
                        return orig_forward(x, e, freqs, context,
                                            context_img_len=context_img_len,
                                            transformer_options=transformer_options)

                    s_max   = sigma_state['sigma_max'][0]
                    s_min   = sigma_state['sigma_min'][0]
                    s_range = s_max - s_min

                    threshold     = s_min + sigma_state['split'] * max(s_range, 1e-6)
                    is_high_sigma = sig_val >= threshold

                    # Find and adjust BlockSkip state's effective_win
                    fwd = orig_forward
                    cell_contents = []
                    if hasattr(fwd, "__closure__") and fwd.__closure__:
                        cell_contents = [c.cell_contents for c in fwd.__closure__
                                         if hasattr(c, "cell_contents")]
                    for obj in cell_contents:
                        if isinstance(obj, dict) and "effective_win" in obj:
                            mult = sigma_state['high_mult'] if is_high_sigma else sigma_state['low_mult']
                            for blk_i in list(obj["effective_win"].keys()):
                                nom = 10 if blk_i < 4 else (3 if blk_i < 9 else 0)
                                if nom > 0:
                                    new_win = max(1, min(nom * 3, int(nom * mult)))
                                    obj["effective_win"][blk_i] = new_win
                            if verbose:
                                mode = "HIGH" if is_high_sigma else "LOW "
                                print(f"[SP SigmaSwitch] {mode} σ={sig_val:.4f} "
                                      f"thr={threshold:.4f} "
                                      f"win[0]={obj['effective_win'].get(0,'?')}")

                return orig_forward(x, e, freqs, context,
                                    context_img_len=context_img_len,
                                    transformer_options=transformer_options)

            return sigma_forward

        # Only need to hook block 0 (block-0 drives the sigma tracking)
        blk0 = blocks[0][1]
        blk0.forward = make_sigma_hook(blk0, 0)

        print(f"[SP SigmaSwitch] attached via model_function_wrapper (real sigma) | "
              f"high_sigma_mult={high_sigma_mult}x low_sigma_mult={low_sigma_mult}x "
              f"split@{sigma_split_frac:.0%}")

        return (patched,)


class ShannonPrimeWanRicciSentinel:
    """
    Phase 13 diagnostic — per-step sigma timeline reporter (Ricci Sentinel).

    Attach after BlockSkip + SigmaSwitch. Hooks block-0 forward and records
    each denoising step's sigma regime and cache window decisions. At each
    generation boundary (wall-clock gap > 60s), prints a compact table showing
    the full timeline so you can verify SigmaSwitch is doing the right thing.

    Output columns:
      Step   — denoising step number within this generation
      sigma  — real sigma from sampler (via SigmaSwitch's model_function_wrapper)
      regime — HIGH (early/Arithmetic Granite) or LOW (late/Semantic Sand)
      win[0] — effective_win for block 0 (Tier-0 representative)
      win[4] — effective_win for block 4 (Tier-1 representative)
      roll_sim[0] — rolling cosine similarity oracle for block 0

    Sigma source: reads from patched._sp_sigma_state (set by SigmaSwitch's
    model_function_wrapper). Falls back to e_mag if SigmaSwitch is not in
    the chain (e_mag is unreliable for Wan but better than nothing for diag).

    Workflow: ... → ShannonPrimeWanBlockSkip → ShannonPrimeWanSigmaSwitch
                 → ShannonPrimeWanRicciSentinel → KSampler → ...
    """

    CATEGORY     = "shannon-prime"
    RETURN_TYPES = ("MODEL",)
    FUNCTION     = "attach"
    DESCRIPTION  = (
        "Phase 13 diagnostic: per-step sigma regime + window timeline. "
        "Attach after BlockSkip + SigmaSwitch. Prints a summary table at "
        "each generation boundary showing e_mag, HIGH/LOW regime switch point, "
        "and effective cache windows per tier."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"model": ("MODEL",)},
            "optional": {
                "sigma_split_frac": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 0.9, "step": 0.05,
                    "tooltip": "Match SigmaSwitch sigma_split_frac. Used to label HIGH/LOW regimes."}),
                "verbose": ("BOOLEAN", {"default": True,
                    "tooltip": "True = print every step. False = only print summary at generation end."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, *a, **k):
        return float("nan")

    def attach(self, model, sigma_split_frac=0.5, verbose=True):
        import time as _time

        patched = model.clone()
        blocks  = list(_iter_wan_blocks(patched))
        if not blocks:
            print("[SP Ricci] no Wan blocks found — passing through")
            return (patched,)

        # ── Try to read real sigma from SigmaSwitch's model_function_wrapper ──
        sp_sigma = getattr(patched, '_sp_sigma_state', None)
        has_real_sigma = sp_sigma is not None
        if has_real_sigma:
            print("[SP Ricci Sentinel] found _sp_sigma_state — using real sigma from sampler")
        else:
            print("[SP Ricci Sentinel] no _sp_sigma_state — falling back to e_mag (unreliable for Wan)")

        sentinel = {
            'log':        [],       # list of per-step record dicts
            'sigma_max':  [None],   # rolling max sigma for this generation
            'sigma_min':  [None],   # rolling min sigma for this generation
            'last_t':     [0.0],    # wall-clock time of last block-0 call
            'split':      sigma_split_frac,
        }

        def _find_state(fwd, key, depth=0):
            """Recursively walk the closure chain to find a dict containing `key`.
            Handles stacked wrappers: Sentinel -> SigmaSwitch -> BlockSkip -> orig."""
            if depth > 8:
                return None
            if not (hasattr(fwd, "__closure__") and fwd.__closure__):
                return None
            for cell in fwd.__closure__:
                try:
                    obj = cell.cell_contents
                except ValueError:
                    continue
                if isinstance(obj, dict) and key in obj:
                    return obj
                if callable(obj):
                    found = _find_state(obj, key, depth + 1)
                    if found is not None:
                        return found
            return None

        def _print_log(log):
            if not log:
                return
            try:
                import tqdm as _tqdm
                _w = _tqdm.tqdm.write
            except Exception:
                _w = print
            src_label = "sigma" if has_real_sigma else "e_mag"
            _w(f"\n[SP Ricci Sentinel] Generation summary ({len(log)} steps):")
            _w(f"  {'Step':>4s}  {src_label:>7s}  {'regime':6s}  "
               f"{'win[0]':>6s}  {'win[4]':>6s}  {'roll_sim[0]':>11s}")
            _w("  " + "-" * 54)
            last_regime = None
            switch_step = None
            for rec in log:
                regime = "HIGH" if rec['high'] else "LOW "
                if last_regime is not None and regime != last_regime:
                    switch_step = rec['step']
                    arrow = "HIGH->LOW" if last_regime == "HIGH" else "LOW->HIGH"
                    _w(f"  {'':4s}  {'--- ' + arrow + ' ---':>40s}")
                rs0 = f"{rec['roll_sim0']:.3f}" if rec['roll_sim0'] is not None else "   n/a"
                _w(f"  {rec['step']:4d}  {rec['sigma']:7.3f}  {regime:6s}  "
                   f"{rec['win0']:6d}  {rec['win4']:6d}  {rs0:>11s}")
                last_regime = regime
            if switch_step is not None:
                _w(f"\n  Regime switch at step {switch_step} "
                   f"(split@{sentinel['split']:.0%} of sigma range)")
            else:
                _w(f"\n  No regime switch (all steps in one regime)")
            _w("")

        blk0     = blocks[0][1]
        orig_fwd = blk0.forward   # may be SigmaSwitch wrapper (which wraps BlockSkip)

        def sentinel_forward(x, e, freqs, context, context_img_len=257,
                             transformer_options={}):
            now   = _time.time()
            t_gap = now - sentinel['last_t'][0]

            # ── Generation boundary: dump log and reset ────────────────────────
            # 60s matches BlockSkip's threshold — safe even at 50s/it step times.
            if sentinel['last_t'][0] > 0 and t_gap > 60.0:
                _print_log(sentinel['log'])
                sentinel['log'].clear()
                sentinel['sigma_max'][0] = None
                sentinel['sigma_min'][0] = None

            sentinel['last_t'][0] = now

            # ── Read current sigma ─────────────────────────────────────────────
            # Prefer real sigma from SigmaSwitch; fall back to e_mag.
            if has_real_sigma and sp_sigma['current_sigma'][0] is not None:
                sig_val = sp_sigma['current_sigma'][0]
            else:
                # Fallback: e_mag (unreliable for Wan, <15% dynamic range)
                sig_val = float(e.float().reshape(-1).abs().mean())

            # Update rolling range
            if sentinel['sigma_max'][0] is None:
                sentinel['sigma_max'][0] = sig_val
                sentinel['sigma_min'][0] = sig_val
            else:
                sentinel['sigma_max'][0] = max(sentinel['sigma_max'][0], sig_val)
                sentinel['sigma_min'][0] = min(sentinel['sigma_min'][0], sig_val)

            s_max     = sentinel['sigma_max'][0]
            s_min     = sentinel['sigma_min'][0]
            threshold = s_min + sentinel['split'] * max(s_max - s_min, 1e-6)
            is_high   = sig_val >= threshold

            # ── Read BlockSkip state (effective_win, rolling_sim) ─────────────
            bs_state = _find_state(orig_fwd, "effective_win")
            step_no  = len(sentinel['log']) + 1
            win0     = bs_state['effective_win'].get(0, -1) if bs_state else -1
            win4     = bs_state['effective_win'].get(4, -1) if bs_state else -1
            rs0      = (bs_state['rolling_sim'].get(0) if bs_state else None)

            rec = {
                'step':      step_no,
                'sigma':     sig_val,
                'high':      is_high,
                'win0':      win0,
                'win4':      win4,
                'roll_sim0': rs0,
            }
            sentinel['log'].append(rec)

            if verbose:
                try:
                    import tqdm as _tqdm
                    _write = _tqdm.tqdm.write
                except Exception:
                    _write = print
                regime = "HIGH" if is_high else "LOW "
                src    = "σ" if has_real_sigma else "e"
                rs_str = f"sim={rs0:.3f}" if rs0 is not None else ""
                _write(f"[SP Ricci] step={step_no:3d}  {src}={sig_val:.4f}  "
                       f"thr={threshold:.4f}  {regime}  "
                       f"win[0]={win0}  win[4]={win4}  {rs_str}")

            return orig_fwd(x, e, freqs, context,
                            context_img_len=context_img_len,
                            transformer_options=transformer_options)

        blk0.forward = sentinel_forward

        src_str = "real sigma (model_function_wrapper)" if has_real_sigma else "e_mag fallback"
        print(f"[SP Ricci Sentinel] attached to block-0 | "
              f"source={src_str} | split@{sigma_split_frac:.0%} | verbose={verbose}")
        print(f"[SP Ricci Sentinel] timeline table printed at each generation boundary (t_gap>60s)")
        print(f"[SP Ricci Sentinel] columns: step | sigma | HIGH/LOW | win[0] | win[4] | roll_sim[0]")

        return (patched,)


NODE_CLASS_MAPPINGS = {
    "ShannonPrimeWanCache":           ShannonPrimeWanCache,
    "ShannonPrimeWanCacheStats":      ShannonPrimeWanCacheStats,
    "ShannonPrimeWanCacheSqfree":     ShannonPrimeWanCacheSqfree,
    "ShannonPrimeWanSelfExtract":     ShannonPrimeWanSelfExtract,
    "ShannonPrimeWanBlockSkip":       ShannonPrimeWanBlockSkip,
    "ShannonPrimeWanCacheFlush":      ShannonPrimeWanCacheFlush,
    "ShannonPrimeWanSigmaSwitch":     ShannonPrimeWanSigmaSwitch,
    "ShannonPrimeWanRicciSentinel":   ShannonPrimeWanRicciSentinel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ShannonPrimeWanCache":           "Shannon-Prime: Wan Cross-Attn Cache",
    "ShannonPrimeWanCacheStats":      "Shannon-Prime: Cache Stats",
    "ShannonPrimeWanCacheSqfree":     "Shannon-Prime: Wan Cross-Attn Cache (Sqfree)",
    "ShannonPrimeWanSelfExtract":     "Shannon-Prime: Wan Self-Attn Extract (Phase 12)",
    "ShannonPrimeWanBlockSkip":       "Shannon-Prime: Wan Block-Level Self-Attn Skip (VHT2)",
    "ShannonPrimeWanCacheFlush":      "Shannon-Prime: Wan Block Cache Flush (before VAE)",
    "ShannonPrimeWanSigmaSwitch":     "Shannon-Prime: Wan Sigma Switch (Phase 13)",
    "ShannonPrimeWanRicciSentinel":   "Shannon-Prime: Wan Ricci Sentinel (Phase 13 diag)",
}
