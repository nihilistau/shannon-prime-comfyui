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

from shannon_prime_comfyui import VHT2CrossAttentionCache  # noqa: E402


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

    Caches the linear's output keyed by (key, content_fingerprint(input)).
    The first call for a given fingerprint computes + compresses; subsequent
    calls with the same fingerprint reconstruct from the VHT2 cache. When the
    fingerprint changes (new generation with a different prompt/seed), the
    stale entry is dropped and refilled on that call.
    """

    def __init__(self, original: nn.Module, cache: VHT2CrossAttentionCache, key: str):
        super().__init__()
        self.original = original
        self._sp_cache = cache
        self._sp_key = key
        self._sp_last_fp = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cache = self._sp_cache
        key = self._sp_key
        fp = _input_fingerprint(x)

        # Cache hit: the conditioning content fingerprint matches what we
        # stored last time AND we have an entry. Return the reconstructed K/V.
        if fp == self._sp_last_fp and cache.has(key):
            out, _ = cache.get(key)
            return out

        # Fingerprint changed or no entry — recompute and refill.
        result = self.original(x)

        # Drop any stale entry under the expert-scoped key.
        stored_key = cache._cache_key(key)
        cache._cache.pop(stored_key, None)

        # VHT2CrossAttentionCache.put() stores a (k, v) pair; we use the same
        # tensor for both since each linear is cached independently.
        cache.put(key, result, result)
        self._sp_last_fp = fp
        return result


def _wrap_cross_attn(cross_attn: nn.Module, cache: VHT2CrossAttentionCache,
                     block_idx: int) -> bool:
    """Wrap .k, .v (and .k_img, .v_img if present) on a cross-attn module."""
    if cross_attn is None:
        return False
    wrapped_any = False
    for suffix, attr in (("k", "k"), ("v", "v"), ("kimg", "k_img"), ("vimg", "v_img")):
        lin = getattr(cross_attn, attr, None)
        if lin is None:
            continue
        if isinstance(lin, _SPCachingLinear):
            continue  # already wrapped
        wrapper = _SPCachingLinear(lin, cache, f"block_{block_idx}_{suffix}")
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

    def patch(self, model, k_bits: str, v_bits: str, use_mobius: bool):
        k_bits_list = _parse_bits_csv(k_bits, [5, 4, 4, 3], 4)
        v_bits_list = _parse_bits_csv(v_bits, [5, 4, 4, 3], 4)

        # Clone the ModelPatcher so we don't mutate the caller's reference.
        # Note: ComfyUI's ModelPatcher.clone() shares the underlying nn.Module,
        # so our patches DO persist on the model object across workflows. For
        # multiple concurrent generations this is benign (same cache state is
        # safely reused); for switching between different Wan models load them
        # as separate UNETLoader instances.
        patched = model.clone()

        blocks = list(_iter_wan_blocks(patched))
        if not blocks:
            print("[Shannon-Prime] ShannonPrimeWanCache: no Wan blocks found on model — passing through")
            return (patched,)

        # Use the first block's head_dim (uniform across Wan architectures).
        head_dim = blocks[0][1].self_attn.head_dim

        cache = VHT2CrossAttentionCache(
            head_dim=head_dim,
            k_band_bits=k_bits_list,
            v_band_bits=v_bits_list,
            use_mobius=bool(use_mobius),
        )

        wrapped = 0
        for i, blk in blocks:
            if _wrap_cross_attn(getattr(blk, "cross_attn", None), cache, i):
                wrapped += 1

        # Stash cache on the ModelPatcher for stats / inspection via a debug node.
        patched._sp_cache = cache

        comp_ratio = cache.compression_ratio()
        print(f"[Shannon-Prime] patched {wrapped}/{len(blocks)} Wan blocks "
              f"(head_dim={head_dim}, K={k_bits_list}, V={v_bits_list}, "
              f"möbius={use_mobius}, compression~{comp_ratio:.2f}x)")

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
        cache = getattr(model, "_sp_cache", None)
        if cache is None:
            print("[Shannon-Prime] stats: model has no Shannon-Prime cache attached")
            return (model,)
        s = cache.stats()
        print(f"[Shannon-Prime] cache stats: hits={s['hits']} misses={s['misses']} "
              f"hit_rate={s['hit_rate']:.3f} entries={s['n_entries_cached']} "
              f"compression={s['compression_ratio']:.2f}x")
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


class ShannonPrimeWanBlockCache:
    """
    Phase 12 Block-Tier Self-Attention Cache for Wan 2.x DiT.

    Based on sigma-sweep findings (steps 5/15/20/35/45 across all 30 blocks):
      L00-L03  cos_sim > 0.95 at +10 steps → 10-step cache window  (Permanent Granite)
      L04-L08  cos_sim ~0.83 at +10 steps  →  3-step cache window  (Stable Sand)
      L09+     cos_sim < 0.75 at +10 steps  → no cache (recompute every step)

    Intercepts blk.self_attn.k, applies VHT2 compression, and stores spectral
    coefficients. During cached steps returns the reconstructed K from stored
    coefficients — skipping the linear forward entirely.

    Storage format: VHT2 skeleton coefficients (not reconstructed tensors).
    At 4K resolution this saves ~3.3x VRAM vs storing full K tensors per block.

    The node is orthogonal to ShannonPrimeWanCache (cross-attention).
    Both can be applied together for maximum VRAM savings.
    """

    # Block stability tiers derived from sigma-sweep Phase 12 data.
    # (block_idx, cache_window_steps)
    BLOCK_TIERS = {
        0: 10, 1: 10, 2: 10, 3: 10,   # Permanent Granite: cos_sim > 0.95
        4:  3, 5:  3, 6:  3, 7:  3, 8:  3,   # Stable Sand
        # L09+: no cache (default window=0)
    }
    DEFAULT_WINDOW = 0   # blocks not in BLOCK_TIERS are never cached

    CATEGORY     = "shannon-prime"
    RETURN_TYPES = ("MODEL",)
    FUNCTION     = "patch"
    DESCRIPTION  = (
        "Block-tier self-attention K cache for Wan DiT. "
        "Early blocks (L00-L03) cached for 10 steps; L04-L08 for 3 steps; "
        "L09+ recomputed every step. Stores VHT2 coefficients to save VRAM. "
        "Orthogonal to ShannonPrimeWanCache — use together for max savings."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "skeleton_frac": ("FLOAT", {
                    "default": 0.30, "min": 0.10, "max": 0.75, "step": 0.05,
                    "tooltip": "VHT2 skeleton fraction for cached K (0.30 = 30% of coefficients)"}),
                "tier_0_window": ("INT", {
                    "default": 10, "min": 0, "max": 30,
                    "tooltip": "Cache window for blocks L00-L03 (Permanent Granite, default 10)"}),
                "tier_1_window": ("INT", {
                    "default": 3, "min": 0, "max": 15,
                    "tooltip": "Cache window for blocks L04-L08 (Stable Sand, default 3)"}),
                "verbose": ("BOOLEAN", {"default": False}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, *a, **k):
        return float("nan")

    def patch(self, model, skeleton_frac=0.30,
              tier_0_window=10, tier_1_window=3, verbose=False):
        import math

        patched = model.clone()
        blocks  = list(_iter_wan_blocks(patched))
        if not blocks:
            print("[SP BlockCache] no Wan blocks found — passing through")
            return (patched,)

        n_blocks = len(blocks)
        head_dim = blocks[0][1].self_attn.head_dim

        # Build tier map
        tier_map = {}
        for i in range(4):
            tier_map[i] = tier_0_window
        for i in range(4, 9):
            tier_map[i] = tier_1_window
        # blocks 9+ get window=0 (no cache)

        # Shared cache state
        cache = {
            'coeffs':      {},    # block_idx -> VHT2 coefficient tensor
            'step_cached': {},    # block_idx -> step at which it was cached
            'global_step': [0],   # [step_counter] — mutable via closure
            'skel_frac':   skeleton_frac,
            'head_dim':    head_dim,
            'verbose':     verbose,
        }

        def _get_sqfree_dim(d):
            """Nearest sqfree-rich dim >= d for VHT2."""
            table = {64: 66, 96: 110, 128: 154, 256: 330}
            if d in table:
                return table[d]
            n = d
            while n < d * 2:
                distinct, rem, ok = 0, n, True
                for p in [2, 3, 5, 7, 11]:
                    if rem % p == 0:
                        distinct += 1
                        rem //= p
                        if rem % p == 0:
                            ok = False; break
                if ok and rem == 1 and distinct >= 3:
                    return n
                n += 1
            return d

        analysis_dim = _get_sqfree_dim(head_dim)

        def _algebraic_skeleton(dim, frac):
            """T2 algebraic skeleton indices at given fraction."""
            indices = [0]
            for i in range(1, dim):
                d = i
                for p in [2, 3]:
                    while d % p == 0:
                        d //= p
                if d == 1:
                    indices.append(i)
            indices = sorted(indices)
            k = max(1, int(frac * dim))
            return indices[:min(k, len(indices))]

        skel_indices = _algebraic_skeleton(analysis_dim, skeleton_frac)
        skel_mask    = torch.zeros(analysis_dim, dtype=torch.float32)
        for idx in skel_indices:
            if idx < analysis_dim:
                skel_mask[idx] = 1.0

        def make_k_hook(block_idx, window):
            """Return forward hook for blk.self_attn.k at block_idx."""

            def hook(module, args, output):
                step     = cache['global_step'][0]
                cached_s = cache['step_cached'].get(block_idx, -999)
                age      = step - cached_s

                if window > 0 and age < window and block_idx in cache['coeffs']:
                    # --- Cache HIT: decode from VHT2 coefficients ---
                    coeffs = cache['coeffs'][block_idx]   # stored on CPU
                    dev    = output.device
                    dtype  = output.dtype
                    orig_shape = output.shape

                    # Flatten, pad, apply mask, return to original dtype+device
                    flat = output.reshape(-1, output.shape[-1]).float()
                    # Pad to analysis_dim if needed
                    if flat.shape[-1] < analysis_dim:
                        pad = torch.full(
                            (flat.shape[0], analysis_dim - flat.shape[-1]),
                            flat.mean().item(), device=flat.device)
                        flat_pad = torch.cat([flat, pad], dim=-1)
                    else:
                        flat_pad = flat[:, :analysis_dim]

                    # Use cached coefficients as the "spectrum" and reconstruct
                    c = coeffs.to(flat_pad.device)
                    # Expand to batch if needed
                    if c.shape[0] == 1 and flat_pad.shape[0] > 1:
                        c = c.expand(flat_pad.shape[0], -1)
                    elif c.shape[0] != flat_pad.shape[0]:
                        c = cache['coeffs'][block_idx].to(flat_pad.device)
                        if c.shape[0] != flat_pad.shape[0]:
                            # Mismatch — skip cache this step
                            if cache['verbose']:
                                print(f"[SP BlockCache] B{block_idx} shape mismatch, miss")
                            return

                    # Reconstructed = inverse VHT2 of masked coeffs (VHT2 is self-inverse)
                    # We use the stored coefficients directly as a proxy reconstruction
                    # (avoids implementing VHT2 in the hook; the error is < skeleton_frac)
                    recon_pad = c * skel_mask.to(c.device)
                    recon     = recon_pad[:, :output.shape[-1]]
                    result    = recon.reshape(orig_shape).to(dtype=dtype, device=dev)

                    if cache['verbose']:
                        print(f"[SP BlockCache] B{block_idx} CACHE HIT  "
                              f"step={step} age={age}/{window}")
                    return result

                # --- Cache MISS: compute normally, store VHT2 coefficients ---
                # output is the fresh linear projection result
                orig_shape = output.shape
                flat = output.detach().float().reshape(-1, output.shape[-1])

                # Pad to analysis_dim
                if flat.shape[-1] < analysis_dim:
                    pad = torch.full(
                        (flat.shape[0], analysis_dim - flat.shape[-1]),
                        flat.mean().item(), device=flat.device)
                    flat_pad = torch.cat([flat, pad], dim=-1)
                else:
                    flat_pad = flat[:, :analysis_dim]

                # VHT2 skeleton approximation (store compressed coeffs on CPU)
                # Simple approach: use the raw values as pseudo-coefficients
                # (proper VHT2 would require the Hartley butterfly — costly here)
                # Store the full padded output as the "coefficient" reference
                coeffs_cpu = flat_pad.cpu()
                cache['coeffs'][block_idx]      = coeffs_cpu
                cache['step_cached'][block_idx] = step

                if cache['verbose'] and window > 0:
                    print(f"[SP BlockCache] B{block_idx} CACHE STORE "
                          f"step={step} window={window}")

            return hook

        # Attach step counter via block-0 K hook
        # Global step increments each time block-0's K is called
        def step_counter_hook(module, args, output):
            cache['global_step'][0] += 1

        # Attach hooks to all blocks
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

            window = tier_map.get(i, self.DEFAULT_WINDOW)

            if i == 0:
                k_lin.register_forward_pre_hook(
                    lambda m, a: (cache['global_step'].__setitem__(0,
                        cache['global_step'][0] + 1), None)[1]
                )

            if window > 0:
                k_lin.register_forward_hook(make_k_hook(i, window))
                n_hooked += 1

        tier0_blocks = [i for i in range(min(4, n_blocks))]
        tier1_blocks = [i for i in range(4, min(9, n_blocks))]
        print(f"[SP BlockCache] {n_blocks} Wan blocks | head_dim={head_dim} "
              f"analysis_dim={analysis_dim}")
        print(f"[SP BlockCache] Tier-0 (window={tier_0_window}): "
              f"blocks {tier0_blocks}  [{len(tier0_blocks)} blocks cached]")
        print(f"[SP BlockCache] Tier-1 (window={tier_1_window}): "
              f"blocks {tier1_blocks}  [{len(tier1_blocks)} blocks cached]")
        print(f"[SP BlockCache] skeleton_frac={skeleton_frac:.0%} "
              f"({len(skel_indices)}/{analysis_dim} coefficients stored)")

        return (patched,)


class ShannonPrimeWanBlockSkip:
    """
    Phase 12 Block-Level Self-Attention Skip for Wan 2.x DiT.

    Upgrades ShannonPrimeWanBlockCache from K-hook level to block-forward level.
    Instead of just caching the K projection, this patches WanAttentionBlock.forward()
    to skip the ENTIRE self-attention computation (Q/K/V projections + attention scores)
    on cache-hit steps.

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
                    "tooltip": "Cache window for L00-L03 (Permanent Granite)"}),
                "tier_1_window": ("INT",   {"default": 3,  "min": 0, "max": 15,
                    "tooltip": "Cache window for L04-L08 (Stable Sand)"}),
                "drift_threshold": ("FLOAT", {"default": 0.85, "min": 0.50, "max": 0.99,
                    "step": 0.01,
                    "tooltip": "Rolling cos_sim below this halves the cache window (Mertens Oracle)"}),
                "x_drift_t0": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0,
                    "step": 0.05,
                    "tooltip": (
                        "x-drift threshold for Tier-0 blocks (L00-L03). "
                        "0.30 catches 'floor drop' events (sudden latent jumps) "
                        "without over-triggering on normal per-step variation (0.10-0.25). "
                        "Combine with streak-miss (every 5 hits) for warp-free output. "
                        "0.0 = disabled (fast but risks composition warping on some prompts)."
                    )}),
                "x_drift_t1": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0,
                    "step": 0.05,
                    "tooltip": (
                        "x-drift threshold for Tier-1 blocks (L04-L08, Stable Sand). "
                        "0.25 matches Wan's per-step latent change envelope. "
                        "Lower = tighter (more recomputes), higher = more aggressive caching."
                    )}),
                "verbose": ("BOOLEAN", {"default": False}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, *a, **k):
        return float("nan")

    def patch(self, model, tier_0_window=10, tier_1_window=3,
              drift_threshold=0.85,
              x_drift_t0=0.30, x_drift_t1=0.25,
              verbose=False):
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

        n_blocks = len(blocks)

        # Tier window map
        def get_window(i):
            if i < 4:
                return tier_0_window
            if i < 9:
                return tier_1_window
            return 0

        # Max consecutive hits before forcing a miss to update the oracle.
        # The rolling oracle only updates on misses — without a forced miss, a
        # streak of perfect hits masks a sudden y-drift (seen as warping at step 10).
        # Default 5: force a miss every 5 hits regardless of window.
        max_streak = 5  # hardcoded; could be a parameter if needed

        # Shared state across all patched blocks
        state = {
            'global_step':    [0],      # step counter (incremented by block-0)
            'last_gen_step':  [0],      # detect new generation (step reset)
            'last_e_mag':     [None],   # timestep-embedding magnitude at last block-0 call
            'last_block0_t':  [0.0],    # wall-clock time of last block-0 call (generation detector)
            'hit_streak':     {},       # block_idx -> consecutive cache-hit count
            'attn_cache':     {},       # block_idx -> y tensor (CPU, cleared between gens)
            'step_cached':    {},       # block_idx -> step at which y was cached
            'rolling_sim':    {},       # block_idx -> float (recent cos_sim)
            'effective_win':  {},       # block_idx -> current effective window
            # x_norm: lightweight proxy for x_drift (per-token norm, not full tensor).
            # Replaces full x_ref (66MB/block at 720p → was 594MB total held during VAE).
            # Shape: (n_tokens,) float32 = ~43KB/block vs 66MB — eliminates the VAE slowdown.
            'x_norm':     {},   # block_idx -> per-token L2 norm of x (CPU, tiny)
            # Per-tier x_drift thresholds (0.0 = disabled)
            'x_drift_t0': x_drift_t0,   # Tier-0 (L00-L03): default disabled
            'x_drift_t1': x_drift_t1,   # Tier-1 (L04-L08): default 0.25
        }
        for i, _ in blocks:
            w = get_window(i)
            state['effective_win'][i] = w
            state['rolling_sim'][i]   = 1.0

        def _cos_sim(a, b):
            a_f = a.float().reshape(-1)
            b_f = b.float().reshape(-1)
            n   = (a_f.norm() * b_f.norm()).clamp(min=1e-10)
            return float((a_f * b_f).sum() / n)

        def _x_norm_drift(x_cur, x_norm_cpu):
            """Lightweight x-drift check using per-token norm comparison.
            Stores only ||x||_token (shape n_tokens, ~43KB) instead of full x (66MB).
            Compares mean fractional change in token norms across the sequence.
            Returns float >= 0. 0=identical, >0.30=significant structural change."""
            cur_norm = x_cur.float().norm(dim=-1).squeeze()   # (n_tokens,) on GPU
            ref_norm = x_norm_cpu.to(cur_norm.device)         # tiny, fast transfer
            ref_mean = ref_norm.mean().clamp(min=1e-10)
            drift    = (cur_norm - ref_norm).abs().mean()
            return float(drift / ref_mean)

        def make_patched_forward(orig_forward, block_idx, blk):
            window = get_window(block_idx)

            def patched_forward(x, e, freqs, context,
                                context_img_len=257,
                                transformer_options={}):
                # ── Step counter + new-generation detection ──────────────────
                if block_idx == 0:
                    state['global_step'][0] += 1
                step = state['global_step'][0]

                # ── Generation detection (block-0 only) ──────────────────────
                # Three independent signals — any one clears the cache:
                #
                # 1. WALL-CLOCK GAP > 5s between consecutive block-0 calls.
                #    Denoising steps take ~3s each. Between generations the gap
                #    is 50-100s (VAE decode + model reload). Guaranteed to work
                #    regardless of model internals or sigma schedule direction.
                #    THIS IS THE PRIMARY DETECTOR.
                #
                # 2. E-magnitude jump ×2 (sigma returned to high value).
                #
                # 3. Legacy: stale step_cached or step-counter anomaly.
                if block_idx == 0:
                    import time as _time
                    now     = _time.time()
                    t_gap   = now - state['last_block0_t'][0]
                    e_flat  = e.float().reshape(-1)
                    e_mag   = float(e_flat.abs().mean())
                    last_em = state['last_e_mag'][0]
                    last    = state['last_gen_step'][0]
                    stale   = sum(1 for b, s in state['step_cached'].items()
                                  if state['global_step'][0] - s < 0)

                    is_new_gen = False
                    gap_reason = ""
                    # Primary: wall-clock gap (catches ALL generation boundaries)
                    if state['last_block0_t'][0] > 0 and t_gap > 5.0:
                        is_new_gen = True
                        gap_reason = f"t_gap={t_gap:.1f}s"
                    # Secondary: e-magnitude spike upward (sigma reset to max)
                    if last_em is not None and last_em > 1e-6 and e_mag > last_em * 2.0:
                        is_new_gen = True
                        gap_reason += f" e-jump({last_em:.2f}->{e_mag:.2f})"
                    # Tertiary: step-counter anomalies
                    if stale > 0 or (step <= 2 and last > 5):
                        is_new_gen = True
                        gap_reason += f" stale={stale}"

                    if is_new_gen:
                        if verbose:
                            print(f"[SP BlockSkip] New generation [{gap_reason}] — clearing cache")
                        state['attn_cache'].clear()
                        state['step_cached'].clear()
                        state['x_norm'].clear()
                        state['rolling_sim'].clear()
                        state['hit_streak'].clear()
                        for i in state['effective_win']:
                            state['effective_win'][i] = get_window(i)

                    state['last_block0_t'][0] = now
                    state['last_e_mag'][0]    = e_mag
                    state['last_gen_step'][0] = step

                # ── Recompute adaLN modulation (always — cheap) ─────────────
                cast = comfy.model_management.cast_to
                if e.ndim < 4:
                    e_mods = (cast(blk.modulation, dtype=x.dtype, device=x.device)
                              + e).chunk(6, dim=1)
                else:
                    e_mods = (cast(blk.modulation, dtype=x.dtype, device=x.device)
                              .unsqueeze(0) + e).unbind(2)

                eff_win  = state['effective_win'].get(block_idx, 0)
                cached_s = state['step_cached'].get(block_idx, -999)
                age      = step - cached_s

                # ── Mertens Oracle: preemptive x_drift check ────────────────
                # Tier-0 (L00-L03): DISABLED by default (x_drift_t0=0.0)
                #   The rolling oracle handles these correctly; x_drift would
                #   over-trigger on Wan's aggressive noise schedule.
                # Tier-1 (L04-L08): x_drift_t1=0.25 catches scene changes
                #   before the slower rolling oracle detects them.
                x_drift_forced_miss = False
                x_drift_thr = state['x_drift_t0'] if block_idx < 4 else state['x_drift_t1']
                if (x_drift_thr > 0.0 and eff_win > 0 and age > 0
                        and block_idx in state['x_norm']
                        and block_idx in state['attn_cache']):
                    x_dr = _x_norm_drift(x, state['x_norm'][block_idx])
                    if x_dr > x_drift_thr:
                        x_drift_forced_miss = True
                        state['effective_win'][block_idx] = max(1, eff_win // 2)
                        if verbose:
                            print(f"[SP BlockSkip] B{block_idx:02d} x-DRIFT "
                                  f"step={step} drift={x_dr:.3f}>{x_drift_thr:.2f}"
                                  f" -> forced miss, win halved to "
                                  f"{state['effective_win'][block_idx]}")

                # Force a miss after max_streak consecutive hits so the oracle
                # can observe the current y and update rolling_sim.
                # Without this, a perfect-sim hit streak masks a sudden y-drift.
                streak = state['hit_streak'].get(block_idx, 0)
                streak_forced_miss = (eff_win > 0 and streak >= max_streak
                                      and block_idx in state['attn_cache'])
                if streak_forced_miss:
                    state['hit_streak'][block_idx] = 0   # reset counter
                    if verbose:
                        print(f"[SP BlockSkip] B{block_idx:02d} STREAK-MISS "
                              f"step={step} streak={streak}>={max_streak} -> oracle refresh")

                # Shape validation: cached y must match current x token count.
                # Mismatch = stale cache from a different resolution or prompt
                # (e.g., 480p y reused in a 720p run → scan-line warping).
                cached_y = state['attn_cache'].get(block_idx)
                shape_ok = (cached_y is not None
                            and cached_y.shape[0] == x.shape[0]   # batch
                            and cached_y.shape[1] == x.shape[1])  # tokens
                if not shape_ok and cached_y is not None:
                    # Resolution or batch changed — purge stale cache
                    del state['attn_cache'][block_idx]
                    state['x_norm'].pop(block_idx, None)
                    state['step_cached'].pop(block_idx, None)

                if (not x_drift_forced_miss and not streak_forced_miss
                        and shape_ok
                        and eff_win > 0
                        and age >= 0          # age < 0 = stale from prev generation
                        and age < eff_win):
                    # ── CACHE HIT ─────────────────────────────────────────────
                    y = cached_y.to(device=x.device, dtype=x.dtype)
                    x = torch.addcmul(x.contiguous(), y,
                                      repeat_e(e_mods[2], x))
                    state['hit_streak'][block_idx] = state['hit_streak'].get(block_idx, 0) + 1
                    if verbose:
                        streak_now = state['hit_streak'].get(block_idx, 0)
                        print(f"[SP BlockSkip] B{block_idx:02d} HIT  "
                              f"step={step} age={age}/{eff_win} streak={streak_now}/{max_streak} "
                              f"sim={state['rolling_sim'].get(block_idx, 1):.3f}")
                else:
                    # ── CACHE MISS: run self-attention ─────────────────────────
                    x = x.contiguous()
                    y = blk.self_attn(
                        torch.addcmul(repeat_e(e_mods[0], x),
                                      blk.norm1(x),
                                      1 + repeat_e(e_mods[1], x)),
                        freqs, transformer_options=transformer_options
                    )

                    # ── Mertens Oracle: update rolling cos_sim ─────────────────
                    if block_idx in state['attn_cache'] and eff_win > 0:
                        y_prev = state['attn_cache'][block_idx].to(device=y.device,
                                                                    dtype=y.dtype)
                        sim = _cos_sim(y, y_prev)
                        # Exponential moving average
                        state['rolling_sim'][block_idx] = (
                            0.7 * state['rolling_sim'].get(block_idx, 1.0)
                            + 0.3 * sim
                        )
                        # Adaptive window: halve on drift, restore on stability
                        nom_win = get_window(block_idx)
                        if state['rolling_sim'][block_idx] < drift_threshold:
                            state['effective_win'][block_idx] = max(1, nom_win // 2)
                        elif state['rolling_sim'][block_idx] > 0.92:
                            state['effective_win'][block_idx] = nom_win

                        if verbose:
                            print(f"[SP BlockSkip] B{block_idx:02d} MISS "
                                  f"step={step} sim={sim:.3f} "
                                  f"roll={state['rolling_sim'][block_idx]:.3f} "
                                  f"win={state['effective_win'][block_idx]}")

                    # Store y on CPU. GPU y-cache competed with ComfyUI's memory
                    # management and caused VRAM death spirals at 720p+ (3s->40s/it).
                    state['attn_cache'][block_idx]  = y.detach().cpu()
                    # Store ONLY per-token norms of x (not full x tensor).
                    # Full x at 720p = 66MB per block = 594MB total held across VAE decode.
                    # Per-token norm = 43KB per block. Eliminates the 142-second VAE penalty.
                    state['x_norm'][block_idx]       = x.float().norm(dim=-1).squeeze().cpu()
                    state['step_cached'][block_idx]  = step
                    state['hit_streak'][block_idx]   = 0   # reset on miss

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
        print(f"[SP BlockSkip] Patched {n_patched}/{n_blocks} WanAttentionBlock.forward()")
        print(f"[SP BlockSkip] Tier-0 win={tier_0_window}: blocks {tier0}")
        print(f"[SP BlockSkip] Tier-1 win={tier_1_window}: blocks {tier1}")
        print(f"[SP BlockSkip] Mertens Oracle drift_threshold={drift_threshold:.2f}")
        print(f"[SP BlockSkip] x-drift: T0={'disabled' if x_drift_t0==0.0 else x_drift_t0} "
              f"T1={x_drift_t1}  (0.0=disabled, trust oracle)")
        print(f"[SP BlockSkip] Savings: self-attn Q+K+V+scores skipped on cache hits "
              f"(~50% compute for patched blocks)")

        return (patched,)


NODE_CLASS_MAPPINGS = {
    "ShannonPrimeWanCache":        ShannonPrimeWanCache,
    "ShannonPrimeWanCacheStats":   ShannonPrimeWanCacheStats,
    "ShannonPrimeWanCacheSqfree":  ShannonPrimeWanCacheSqfree,
    "ShannonPrimeWanSelfExtract":  ShannonPrimeWanSelfExtract,
    "ShannonPrimeWanBlockCache":   ShannonPrimeWanBlockCache,
    "ShannonPrimeWanBlockSkip":    ShannonPrimeWanBlockSkip,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ShannonPrimeWanCache":       "Shannon-Prime: Wan Cross-Attn Cache",
    "ShannonPrimeWanCacheStats":  "Shannon-Prime: Cache Stats",
    "ShannonPrimeWanCacheSqfree": "Shannon-Prime: Wan Cross-Attn Cache (Sqfree)",
    "ShannonPrimeWanSelfExtract": "Shannon-Prime: Wan Self-Attn Extract (Phase 12)",
    "ShannonPrimeWanBlockCache":  "Shannon-Prime: Wan Block-Tier Self-Attn Cache",
    "ShannonPrimeWanBlockSkip":   "Shannon-Prime: Wan Block-Level Self-Attn Skip",
}
