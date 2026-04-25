# Shannon-Prime VHT2: Flux DiT ComfyUI Custom Nodes
# Copyright (C) 2026 Ray Daniels. All Rights Reserved.
# Licensed under AGPLv3. Commercial license available.
#
# Provides Shannon-Prime block-skip and caching nodes for Flux (v1/v2) DiT
# models. Flux uses a dual-stream architecture:
#   - DoubleStreamBlock: img + txt processed separately, then joint attention
#   - SingleStreamBlock: unified stream (concatenated txt+img)
#
# The same Shannon-Prime principles apply: early blocks establish "Granite"
# structure (composition, layout) while late blocks add "Jazz" detail
# (texture, fine features). Block-skip caches the post-attention output
# and reapplies it with the current step's modulation gate on cache hits.
#
# Flux Architecture (from comfy.ldm.flux):
#   - head_dim=128 (Flux v1, 24 heads), head_dim=64 (Flux2, 48 heads)
#   - RoPE is 2D: axes_dim=[16,56,56] (Flux v1) or [32,32,32,32] (Flux2)
#   - Joint attention: Q/K/V from both img and txt streams are concatenated
#   - adaLN via Modulation → ModulationOut(shift, scale, gate)
#   - No separate cross-attention — text is folded into joint attention

import os
import sys
import math
import pathlib

import torch
import torch.nn as nn

# Import shared Shannon-Prime infrastructure from the Wan nodes module
_THIS_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
_SP_TOOLS = _REPO_ROOT / "lib" / "shannon-prime" / "tools"
_SP_TORCH = _REPO_ROOT / "lib" / "shannon-prime" / "backends" / "torch"
for p in (_SP_TOOLS, _SP_TORCH):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

# Re-use Fisher diagonal weighting and fingerprinting from Wan nodes
from shannon_prime_nodes import (
    _fisher_diagonal_weights,
    _fisher_cos_sim,
    _input_fingerprint,
    _sieve_primes,
    _pick_evenly,
    _tiered_lattice_factors,
    _VHT2MemoryPool,
    _get_vht2_pool,
)


# ── Flux block iterator ───────────────────────────────────────────────────

def _iter_flux_double_blocks(model_obj):
    """Yield (index, block) for each DoubleStreamBlock in a Flux ModelPatcher."""
    inner = getattr(model_obj, "model", model_obj)
    diff = getattr(inner, "diffusion_model", None)
    if diff is None:
        diff = getattr(inner, "model", None)
    if diff is None:
        return
    blocks = getattr(diff, "double_blocks", None)
    if blocks is None:
        return
    for i, blk in enumerate(blocks):
        yield i, blk


def _iter_flux_single_blocks(model_obj):
    """Yield (index, block) for each SingleStreamBlock in a Flux ModelPatcher."""
    inner = getattr(model_obj, "model", model_obj)
    diff = getattr(inner, "diffusion_model", None)
    if diff is None:
        diff = getattr(inner, "model", None)
    if diff is None:
        return
    blocks = getattr(diff, "single_blocks", None)
    if blocks is None:
        return
    for i, blk in enumerate(blocks):
        yield i, blk


def _iter_flux_all_blocks(model_obj):
    """Yield (global_index, block, block_type) for all Flux blocks."""
    idx = 0
    for i, blk in _iter_flux_double_blocks(model_obj):
        yield idx, blk, "double"
        idx += 1
    for i, blk in _iter_flux_single_blocks(model_obj):
        yield idx, blk, "single"
        idx += 1


def _detect_flux_head_dim(model_obj):
    """Detect head_dim from Flux model. Returns (head_dim, num_heads) or (128, 24) default."""
    for _, blk in _iter_flux_double_blocks(model_obj):
        if hasattr(blk, 'num_heads') and hasattr(blk, 'hidden_size'):
            hd = blk.hidden_size // blk.num_heads
            return hd, blk.num_heads
        if hasattr(blk, 'img_attn'):
            # SelfAttention has num_heads
            nh = getattr(blk.img_attn, 'num_heads', 24)
            hs = getattr(blk, 'hidden_size', 3072)
            return hs // nh, nh
    return 128, 24  # Flux v1 default


# ── 2D Lattice RoPE for Flux ─────────────────────────────────────────────
#
# Flux uses 2D positional encoding via rope() in comfy.ldm.flux.math.
# The position IDs have axes_dim=[16, 56, 56] for Flux v1 (temporal is
# typically a single frame for images, so effectively 2D: H × W).
#
# We inject lattice-aligned frequencies into the spatial axes (H, W)
# using Local-Tier primes (fine detail). The temporal axis (if present)
# gets Long-Tier anchors — but for static images this axis is degenerate.
#
# The hook monkey-patches comfy.ldm.flux.math.rope to blend lattice
# frequencies at alpha before computing the cos/sin rotation matrices.

def _install_lattice_rope_flux(alpha=0.17):
    """
    Monkey-patch Flux's rope() to use lattice-blended frequencies.
    Idempotent — safe to call multiple times.

    Returns True if patching succeeded, False if Flux model not available.
    """
    try:
        import comfy.ldm.flux.math as flux_math
    except ImportError:
        return False

    if getattr(flux_math, '_sp_lattice_patched', False):
        return True

    _orig_rope = flux_math.rope

    def _lattice_rope(pos, dim, theta=10000):
        """
        Drop-in replacement for Flux rope() with lattice-blended frequencies.

        Flux axes_dim=[16, 56, 56] means:
          - First 16 dims: temporal (degenerate for images → Long-Tier)
          - Next 56 dims: height (spatial → Local-Tier)
          - Last 56 dims: width (spatial → Local-Tier)

        For each call, we determine the tier from the dim parameter:
          dim <= 32: temporal → Long-Tier (causal anchors)
          dim > 32:  spatial → Local-Tier (fine detail)
        """
        n_freqs = dim // 2

        # Determine tier from dimension size
        if dim <= 32:
            tier = 'long'    # temporal or small axis
        else:
            tier = 'local'   # spatial: H or W

        factors = _tiered_lattice_factors(n_freqs, freq_base=float(theta),
                                          alpha=alpha, tier=tier)

        # Compute blended frequencies
        device = pos.device if isinstance(pos, torch.Tensor) else 'cpu'
        scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2,
                               dtype=torch.float64, device=device)
        omega = 1.0 / (theta ** scale)

        # Apply lattice factors
        omega = omega * factors.to(dtype=torch.float64, device=device)

        # Compute rotation matrices (same format as original)
        out = torch.einsum("...n,d->...nd",
                           pos.to(dtype=torch.float32, device=device), omega.float())
        cos_out = torch.cos(out)
        sin_out = torch.sin(out)
        # Stack into 2×2 rotation matrix form
        out = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)

        # Reshape: we need to match the original's rearrange pattern
        # "b n d (i j) -> b n d i j" with i=2, j=2
        out = out.reshape(*out.shape[:-1], 2, 2)
        return out.to(dtype=torch.float32, device=device)

    flux_math.rope = _lattice_rope
    flux_math._sp_lattice_patched = True
    print(f"[Shannon-Prime] Factored 2D Lattice RoPE installed for Flux (α={alpha})")
    print(f"[Shannon-Prime]   Spatial (H,W) → Local-Tier, Temporal → Long-Tier")
    return True


# ── Flux DoubleStreamBlock skip logic ─────────────────────────────────────
#
# The key insight: in a DoubleStreamBlock, the expensive operation is the
# joint attention (QKV projection + attention scores for both streams).
# On cache hit, we:
#   1. Recompute adaLN modulation from current vec (trivial cost)
#   2. Apply cached post-attention residuals with current step's gate
#   3. Optionally run MLP fresh or also cache it (TURBO mode)
#
# Unlike Wan, Flux has TWO outputs per DoubleStreamBlock: img and txt.
# Both must be cached and replayed together.

class ShannonPrimeFluxBlockSkip:
    """
    Block-Level Attention Skip for Flux DiT (v1/v2).

    Patches DoubleStreamBlock.forward() and SingleStreamBlock.forward()
    to skip the joint attention computation on cache-hit steps.

    adaLN-correct cache hit path (DoubleStreamBlock):
      1. Recompute img_mod, txt_mod from current vec (trivial cost)
      2. Apply cached img_attn_out with current img_mod1.gate
      3. Apply cached txt_attn_out with current txt_mod1.gate
      4. Run MLP fresh (sigma-accurate) OR cache MLP too (TURBO)

    adaLN-correct cache hit path (SingleStreamBlock):
      1. Recompute mod from current vec (trivial cost)
      2. Apply cached output with current mod.gate
      3. No separate MLP — SingleStreamBlock fuses attn+mlp

    Block tier map (initial — needs empirical sigma-sweep validation):
      DoubleStream L00-L03: window=8  (Compositional Granite)
      DoubleStream L04-L11: window=3  (Stable mid-layers)
      DoubleStream L12+:    window=0  (Volatile detail)
      SingleStream L00-L07: window=2  (Refinement — shorter window)
      SingleStream L08+:    window=0  (Final detail — no cache)

    These are STARTING VALUES. Run ShannonPrimeFluxRicciSentinel to
    empirically measure per-block stability and tune windows.
    """

    CATEGORY     = "shannon-prime"
    RETURN_TYPES = ("MODEL",)
    FUNCTION     = "patch"
    DESCRIPTION  = (
        "Block-level attention skip for Flux DiT. "
        "Patches DoubleStreamBlock and SingleStreamBlock to skip "
        "joint attention on cache-hit steps. adaLN-correct: "
        "caches post-attention residuals, reapplies current gate."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"model": ("MODEL",)},
            "optional": {
                "double_tier0_window": ("INT", {"default": 8, "min": 0, "max": 30,
                    "tooltip": "DoubleStream L00-L03 (Compositional Granite). 0=disabled."}),
                "double_tier1_window": ("INT", {"default": 3, "min": 0, "max": 15,
                    "tooltip": "DoubleStream L04-L11 (Stable mid-layers). 0=disabled."}),
                "double_tier2_window": ("INT", {"default": 0, "min": 0, "max": 10,
                    "tooltip": "DoubleStream L12+ (Volatile detail). 0=disabled."}),
                "single_tier0_window": ("INT", {"default": 2, "min": 0, "max": 15,
                    "tooltip": "SingleStream L00-L07 (Refinement). 0=disabled."}),
                "single_tier1_window": ("INT", {"default": 0, "min": 0, "max": 10,
                    "tooltip": "SingleStream L08+ (Final detail). 0=disabled."}),
                "cache_mlp": ("BOOLEAN", {"default": False,
                    "tooltip": "TURBO: also cache MLP output in DoubleStreamBlocks. Near-zero compute on HIT. May affect fine detail."}),
                "cache_dtype": (["fp16", "fp8", "mixed"], {"default": "fp16",
                    "tooltip": "fp16=all fp16. fp8=all fp8. mixed=early fp16, late fp8."}),
                "lattice_rope": ("BOOLEAN", {"default": True,
                    "tooltip": "Inject factored 2D lattice RoPE. Spatial axes get Local-Tier detail primes. Zero runtime cost."}),
                "lattice_alpha": ("FLOAT", {"default": 0.17, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "Lattice blend α. 0.0=pure geometric, 0.17=paper default."}),
                "verbose": ("BOOLEAN", {"default": False,
                    "tooltip": "Print per-block HIT/MISS logs + Fisher cos_sim"}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, *a, **k):
        return float("nan")

    def patch(self, model,
              double_tier0_window=8, double_tier1_window=3, double_tier2_window=0,
              single_tier0_window=2, single_tier1_window=0,
              cache_mlp=False, cache_dtype="fp16",
              lattice_rope=True, lattice_alpha=0.17,
              verbose=False, **_ignored):
        import types
        import comfy.model_management

        # ── PrimePE: Factored 2D Lattice RoPE ──
        if lattice_rope and lattice_alpha > 0.0:
            if _install_lattice_rope_flux(alpha=lattice_alpha):
                pass  # success printed by installer
            else:
                print("[Shannon-Prime Flux] Lattice RoPE: Flux model not available, skipping")
        elif not lattice_rope:
            print("[Shannon-Prime Flux] Lattice RoPE: disabled by user")

        patched = model.clone()

        double_blocks = list(_iter_flux_double_blocks(patched))
        single_blocks = list(_iter_flux_single_blocks(patched))

        if not double_blocks and not single_blocks:
            print("[SP FluxSkip] no Flux blocks found — passing through")
            return (patched,)

        n_double = len(double_blocks)
        n_single = len(single_blocks)

        # ── Clear stale caches ──
        _cleared = 0
        for blocks in (double_blocks, single_blocks):
            for _i, _blk in blocks:
                _fwd = getattr(_blk, "forward", None)
                if _fwd and hasattr(_fwd, "__closure__") and _fwd.__closure__:
                    for _cell in _fwd.__closure__:
                        try:
                            _obj = _cell.cell_contents
                            if isinstance(_obj, dict) and "attn_cache_img" in _obj:
                                for k in list(_obj.keys()):
                                    if isinstance(_obj[k], dict):
                                        _obj[k].clear()
                                _cleared += 1
                        except ValueError:
                            pass
        if _cleared:
            print(f"[SP FluxSkip] cleared {_cleared} stale cache(s)")

        # Window maps
        def get_double_window(i):
            if i < 4:
                return double_tier0_window
            if i < 12:
                return double_tier1_window
            return double_tier2_window

        def get_single_window(i):
            if i < 8:
                return single_tier0_window
            return single_tier1_window

        # Dtype selection
        _fp16 = torch.float16
        _fp8 = torch.float8_e4m3fn
        if cache_dtype == "fp8":
            def _dtype_for(block_idx):
                return _fp8
        elif cache_dtype == "mixed":
            def _dtype_for(block_idx):
                # Early double blocks: fp16 for precision
                # Late double + all single: fp8 for memory savings
                return _fp16 if block_idx < 8 else _fp8
        else:
            def _dtype_for(block_idx):
                return _fp16

        # Fisher weights
        head_dim, num_heads = _detect_flux_head_dim(patched)
        _fisher_w = _fisher_diagonal_weights(head_dim)
        _do_mlp = cache_mlp

        # ── Shared state ──
        state = {
            'global_step':     [0],
            'hit_streak':      {},
            'attn_cache_img':  {},   # double block_idx -> img attn residual
            'attn_cache_txt':  {},   # double block_idx -> txt attn residual
            'mlp_cache_img':   {},   # double block_idx -> img MLP residual
            'mlp_cache_txt':   {},   # double block_idx -> txt MLP residual
            'single_cache':    {},   # single block_idx -> full output residual
            'step_cached':     {},
            'fisher_sim':      {},
        }

        # ── Patch DoubleStreamBlocks ──────────────────────────────────────
        def make_double_patched_forward(orig_forward, block_idx, blk):
            window = get_double_window(block_idx)
            _streak_limit = 8 if block_idx < 4 else (4 if block_idx < 12 else 3)
            _blk_dtype = _dtype_for(block_idx)
            _is_fp8 = (_blk_dtype == _fp8)
            # Global block index for state keys (double blocks come first)
            gidx = block_idx

            def _store(t):
                return t.detach().to(dtype=_blk_dtype).cpu()

            def _load(t, ref):
                if _is_fp8:
                    return t.to(dtype=ref.dtype).to(device=ref.device)
                return t.to(device=ref.device, dtype=ref.dtype)

            def patched_forward(img, txt, vec, pe, attn_mask=None,
                                modulation_dims_img=None, modulation_dims_txt=None,
                                transformer_options={}):
                # Step counter (first double block increments)
                if block_idx == 0:
                    state['global_step'][0] += 1
                step = state['global_step'][0]

                if block_idx == 0 and step == 1:
                    print(f"[SP FluxSkip] step=1 img.shape={tuple(img.shape)} "
                          f"txt.shape={tuple(txt.shape)} dtype={img.dtype} "
                          f"device={img.device}")

                # Recompute modulation (always — trivial cost)
                cast = comfy.model_management.cast_to
                if blk.modulation:
                    img_mod1, img_mod2 = blk.img_mod(vec)
                    txt_mod1, txt_mod2 = blk.txt_mod(vec)
                else:
                    (img_mod1, img_mod2), (txt_mod1, txt_mod2) = vec

                cached_s = state['step_cached'].get(gidx, -999)
                age = step - cached_s
                streak = state['hit_streak'].get(gidx, 0)
                cached_img_attn = state['attn_cache_img'].get(gidx)
                cached_txt_attn = state['attn_cache_txt'].get(gidx)

                # Shape validation
                shape_ok = (cached_img_attn is not None
                            and cached_img_attn.shape[0] == img.shape[0]
                            and cached_img_attn.shape[1] == img.shape[1])
                if not shape_ok and cached_img_attn is not None:
                    state['attn_cache_img'].pop(gidx, None)
                    state['attn_cache_txt'].pop(gidx, None)
                    state['mlp_cache_img'].pop(gidx, None)
                    state['mlp_cache_txt'].pop(gidx, None)
                    state['step_cached'].pop(gidx, None)
                    cached_img_attn = None
                    cached_txt_attn = None

                hit = (cached_img_attn is not None
                       and cached_txt_attn is not None
                       and age >= 0 and age < window
                       and streak < _streak_limit)

                if hit:
                    # ── CACHE HIT: skip joint attention ──
                    # Replay cached attention residuals with current gates
                    from comfy.ldm.flux.layers import apply_mod

                    img_attn_res = _load(cached_img_attn, img)
                    img = img + apply_mod(img_attn_res, img_mod1.gate,
                                         None, modulation_dims_img)
                    del img_attn_res

                    txt_attn_res = _load(cached_txt_attn, txt)
                    txt = txt + apply_mod(txt_attn_res, txt_mod1.gate,
                                         None, modulation_dims_txt)
                    del txt_attn_res

                    state['hit_streak'][gidx] = streak + 1

                    # TURBO: also replay MLP
                    if _do_mlp:
                        cached_img_mlp = state['mlp_cache_img'].get(gidx)
                        cached_txt_mlp = state['mlp_cache_txt'].get(gidx)
                        if cached_img_mlp is not None and cached_txt_mlp is not None:
                            img_mlp_res = _load(cached_img_mlp, img)
                            img = img + apply_mod(img_mlp_res, img_mod2.gate,
                                                 None, modulation_dims_img)
                            del img_mlp_res

                            txt_mlp_res = _load(cached_txt_mlp, txt)
                            txt = txt + apply_mod(txt_mlp_res, txt_mod2.gate,
                                                 None, modulation_dims_txt)
                            del txt_mlp_res

                            if verbose:
                                print(f"[SP FluxSkip] D{block_idx:02d} HIT  "
                                      f"step={step} age={age}/{window} "
                                      f"streak={streak+1}/{_streak_limit} TURBO")

                            if txt.dtype == torch.float16:
                                txt = torch.nan_to_num(txt, nan=0.0,
                                                       posinf=65504, neginf=-65504)
                            return img, txt

                    if verbose:
                        print(f"[SP FluxSkip] D{block_idx:02d} HIT  "
                              f"step={step} age={age}/{window} "
                              f"streak={streak+1}/{_streak_limit}")

                    # Run MLP fresh (non-TURBO)
                    from comfy.ldm.flux.layers import apply_mod
                    img_mlp_in = apply_mod(blk.img_norm2(img),
                                          (1 + img_mod2.scale), img_mod2.shift,
                                          modulation_dims_img)
                    img_mlp_out = blk.img_mlp(img_mlp_in)
                    img_mlp_gated = apply_mod(img_mlp_out, img_mod2.gate,
                                             None, modulation_dims_img)
                    img = img + img_mlp_gated

                    if _do_mlp:
                        state['mlp_cache_img'][gidx] = _store(img_mlp_out)

                    txt_mlp_in = apply_mod(blk.txt_norm2(txt),
                                          (1 + txt_mod2.scale), txt_mod2.shift,
                                          modulation_dims_txt)
                    txt_mlp_out = blk.txt_mlp(txt_mlp_in)
                    txt_mlp_gated = apply_mod(txt_mlp_out, txt_mod2.gate,
                                             None, modulation_dims_txt)
                    txt = txt + txt_mlp_gated

                    if _do_mlp:
                        state['mlp_cache_txt'][gidx] = _store(txt_mlp_out)

                    if txt.dtype == torch.float16:
                        txt = torch.nan_to_num(txt, nan=0.0,
                                               posinf=65504, neginf=-65504)
                    return img, txt

                else:
                    # ── CACHE MISS: run full forward ──
                    # We need to extract the attention residuals before gating
                    # to cache them. This means we partially replicate the
                    # original forward to intercept intermediate values.
                    from comfy.ldm.flux.layers import apply_mod
                    from comfy.ldm.flux.math import attention

                    transformer_patches = transformer_options.get("patches", {})
                    extra_options = transformer_options.copy()

                    # Prepare image for attention
                    img_modulated = blk.img_norm1(img)
                    img_modulated = apply_mod(img_modulated,
                                            (1 + img_mod1.scale), img_mod1.shift,
                                            modulation_dims_img)
                    img_qkv = blk.img_attn.qkv(img_modulated)
                    del img_modulated
                    img_q, img_k, img_v = img_qkv.view(
                        img_qkv.shape[0], img_qkv.shape[1], 3,
                        blk.num_heads, -1).permute(2, 0, 3, 1, 4)
                    del img_qkv
                    img_q, img_k = blk.img_attn.norm(img_q, img_k, img_v)

                    # Prepare txt for attention
                    txt_modulated = blk.txt_norm1(txt)
                    txt_modulated = apply_mod(txt_modulated,
                                            (1 + txt_mod1.scale), txt_mod1.shift,
                                            modulation_dims_txt)
                    txt_qkv = blk.txt_attn.qkv(txt_modulated)
                    del txt_modulated
                    txt_q, txt_k, txt_v = txt_qkv.view(
                        txt_qkv.shape[0], txt_qkv.shape[1], 3,
                        blk.num_heads, -1).permute(2, 0, 3, 1, 4)
                    del txt_qkv
                    txt_q, txt_k = blk.txt_attn.norm(txt_q, txt_k, txt_v)

                    # Joint attention
                    q = torch.cat((txt_q, img_q), dim=2)
                    del txt_q, img_q
                    k = torch.cat((txt_k, img_k), dim=2)
                    del txt_k, img_k
                    v = torch.cat((txt_v, img_v), dim=2)
                    del txt_v, img_v

                    attn = attention(q, k, v, pe=pe, mask=attn_mask,
                                   transformer_options=transformer_options)
                    del q, k, v

                    # attn1_output_patch support
                    if "attn1_output_patch" in transformer_patches:
                        extra_options["img_slice"] = [txt.shape[1], attn.shape[1]]
                        patch = transformer_patches["attn1_output_patch"]
                        for p in patch:
                            attn = p(attn, extra_options)

                    txt_attn, img_attn = (attn[:, :txt.shape[1]],
                                          attn[:, txt.shape[1]:])
                    del attn

                    # Cache the projected attention output (pre-gate)
                    img_attn_proj = blk.img_attn.proj(img_attn)
                    txt_attn_proj = blk.txt_attn.proj(txt_attn)
                    del img_attn, txt_attn

                    # Fisher drift diagnostic
                    prev_img = state['attn_cache_img'].get(gidx)
                    if prev_img is not None and verbose:
                        prev_on_dev = _load(prev_img, img)
                        f_sim = _fisher_cos_sim(img_attn_proj, prev_on_dev,
                                               _fisher_w)
                        state['fisher_sim'][gidx] = f_sim
                        del prev_on_dev

                    state['attn_cache_img'][gidx] = _store(img_attn_proj)
                    state['attn_cache_txt'][gidx] = _store(txt_attn_proj)
                    state['step_cached'][gidx] = step
                    state['hit_streak'][gidx] = 0

                    # Apply with current gate
                    img = img + apply_mod(img_attn_proj, img_mod1.gate,
                                         None, modulation_dims_img)
                    del img_attn_proj

                    txt = txt + apply_mod(txt_attn_proj, txt_mod1.gate,
                                         None, modulation_dims_txt)
                    del txt_attn_proj

                    if verbose:
                        f_sim = state['fisher_sim'].get(gidx)
                        f_str = f" fisher={f_sim:.4f}" if f_sim is not None else ""
                        print(f"[SP FluxSkip] D{block_idx:02d} MISS "
                              f"step={step}{f_str}")

                    # MLP
                    img_mlp_in = apply_mod(blk.img_norm2(img),
                                          (1 + img_mod2.scale), img_mod2.shift,
                                          modulation_dims_img)
                    img_mlp_out = blk.img_mlp(img_mlp_in)
                    img = img + apply_mod(img_mlp_out, img_mod2.gate,
                                         None, modulation_dims_img)
                    if _do_mlp:
                        state['mlp_cache_img'][gidx] = _store(img_mlp_out)

                    txt_mlp_in = apply_mod(blk.txt_norm2(txt),
                                          (1 + txt_mod2.scale), txt_mod2.shift,
                                          modulation_dims_txt)
                    txt_mlp_out = blk.txt_mlp(txt_mlp_in)
                    txt = txt + apply_mod(txt_mlp_out, txt_mod2.gate,
                                         None, modulation_dims_txt)
                    if _do_mlp:
                        state['mlp_cache_txt'][gidx] = _store(txt_mlp_out)

                    if txt.dtype == torch.float16:
                        txt = torch.nan_to_num(txt, nan=0.0,
                                               posinf=65504, neginf=-65504)
                    return img, txt

            return patched_forward

        # ── Patch SingleStreamBlocks ──────────────────────────────────────
        def make_single_patched_forward(orig_forward, block_idx, blk):
            window = get_single_window(block_idx)
            _streak_limit = 4 if block_idx < 8 else 3
            _blk_dtype = _dtype_for(n_double + block_idx)
            _is_fp8 = (_blk_dtype == _fp8)
            gidx = n_double + block_idx  # global index offset

            def _store(t):
                return t.detach().to(dtype=_blk_dtype).cpu()

            def _load(t, ref):
                if _is_fp8:
                    return t.to(dtype=ref.dtype).to(device=ref.device)
                return t.to(device=ref.device, dtype=ref.dtype)

            def patched_forward(x, vec, pe, attn_mask=None,
                                modulation_dims=None, transformer_options={}):
                step = state['global_step'][0]

                # Recompute modulation
                cast = comfy.model_management.cast_to
                if blk.modulation is not None:
                    mod, _ = blk.modulation(vec)
                else:
                    mod = vec

                cached_s = state['step_cached'].get(gidx, -999)
                age = step - cached_s
                streak = state['hit_streak'].get(gidx, 0)
                cached_out = state['single_cache'].get(gidx)

                shape_ok = (cached_out is not None
                            and cached_out.shape[0] == x.shape[0]
                            and cached_out.shape[1] == x.shape[1])
                if not shape_ok and cached_out is not None:
                    state['single_cache'].pop(gidx, None)
                    state['step_cached'].pop(gidx, None)
                    cached_out = None

                hit = (cached_out is not None
                       and age >= 0 and age < window
                       and streak < _streak_limit)

                if hit:
                    # ── CACHE HIT: replay with current gate ──
                    from comfy.ldm.flux.layers import apply_mod
                    out = _load(cached_out, x)
                    x = x + apply_mod(out, mod.gate, None, modulation_dims)
                    del out

                    state['hit_streak'][gidx] = streak + 1

                    if verbose:
                        print(f"[SP FluxSkip] S{block_idx:02d} HIT  "
                              f"step={step} age={age}/{window} "
                              f"streak={streak+1}/{_streak_limit}")

                    if x.dtype == torch.float16:
                        x = torch.nan_to_num(x, nan=0.0,
                                             posinf=65504, neginf=-65504)
                    return x

                else:
                    # ── CACHE MISS: run full forward, cache output ──
                    from comfy.ldm.flux.layers import apply_mod
                    from comfy.ldm.flux.math import attention

                    transformer_patches = transformer_options.get("patches", {})
                    extra_options = transformer_options.copy()

                    modulated = apply_mod(blk.pre_norm(x),
                                        (1 + mod.scale), mod.shift,
                                        modulation_dims)

                    qkv, mlp = torch.split(
                        blk.linear1(modulated),
                        [3 * blk.hidden_size, blk.mlp_hidden_dim_first],
                        dim=-1)
                    del modulated

                    q, k, v = qkv.view(
                        qkv.shape[0], qkv.shape[1], 3,
                        blk.num_heads, -1).permute(2, 0, 3, 1, 4)
                    del qkv
                    q, k = blk.norm(q, k, v)

                    attn = attention(q, k, v, pe=pe, mask=attn_mask,
                                   transformer_options=transformer_options)
                    del q, k, v

                    if "attn1_output_patch" in transformer_patches:
                        patch = transformer_patches["attn1_output_patch"]
                        for p in patch:
                            attn = p(attn, extra_options)

                    # MLP activation
                    if blk.yak_mlp:
                        mlp = blk.mlp_act(
                            mlp[..., blk.mlp_hidden_dim_first // 2:]) * \
                            mlp[..., :blk.mlp_hidden_dim_first // 2]
                    else:
                        mlp = blk.mlp_act(mlp)

                    output = blk.linear2(torch.cat((attn, mlp), 2))
                    del attn, mlp

                    # Cache the pre-gate output
                    state['single_cache'][gidx] = _store(output)
                    state['step_cached'][gidx] = step
                    state['hit_streak'][gidx] = 0

                    # Apply with current gate
                    x = x + apply_mod(output, mod.gate, None, modulation_dims)
                    del output

                    if verbose:
                        print(f"[SP FluxSkip] S{block_idx:02d} MISS step={step}")

                    if x.dtype == torch.float16:
                        x = torch.nan_to_num(x, nan=0.0,
                                             posinf=65504, neginf=-65504)
                    return x

            return patched_forward

        # ── Apply patches ─────────────────────────────────────────────────
        n_patched_double = 0
        for i, blk in double_blocks:
            w = get_double_window(i)
            if w > 0:
                blk.forward = make_double_patched_forward(blk.forward, i, blk)
                n_patched_double += 1

        n_patched_single = 0
        for i, blk in single_blocks:
            w = get_single_window(i)
            if w > 0:
                blk.forward = make_single_patched_forward(blk.forward, i, blk)
                n_patched_single += 1

        mode = "TURBO" if cache_mlp else "attn-only"
        dtype_str = cache_dtype
        print(f"[SP FluxSkip] Phase 17 — {n_patched_double}/{n_double} double, "
              f"{n_patched_single}/{n_single} single blocks | {mode} | {dtype_str}")
        print(f"[SP FluxSkip] head_dim={head_dim} num_heads={num_heads}")

        if n_patched_double > 0:
            d0 = [i for i, _ in double_blocks if i < 4 and get_double_window(i) > 0]
            d1 = [i for i, _ in double_blocks if 4 <= i < 12 and get_double_window(i) > 0]
            d2 = [i for i, _ in double_blocks if i >= 12 and get_double_window(i) > 0]
            if d0: print(f"[SP FluxSkip] Double Tier-0 win={double_tier0_window} streak=8: {d0}")
            if d1: print(f"[SP FluxSkip] Double Tier-1 win={double_tier1_window} streak=4: {d1}")
            if d2: print(f"[SP FluxSkip] Double Tier-2 win={double_tier2_window} streak=3: {d2}")

        if n_patched_single > 0:
            s0 = [i for i, _ in single_blocks if i < 8 and get_single_window(i) > 0]
            s1 = [i for i, _ in single_blocks if i >= 8 and get_single_window(i) > 0]
            if s0: print(f"[SP FluxSkip] Single Tier-0 win={single_tier0_window} streak=4: {s0}")
            if s1: print(f"[SP FluxSkip] Single Tier-1 win={single_tier1_window} streak=3: {s1}")

        return (patched,)


class ShannonPrimeFluxCacheFlush:
    """
    Flush Flux block-skip caches before VAE decode.

    Place between KSampler and VAEDecode. Clears cached attention
    tensors from FluxBlockSkip, then calls torch.cuda.empty_cache()
    to give VAE maximum memory headroom.

    Workflow: UnetLoader → ShannonPrimeFluxBlockSkip
              → KSampler → ShannonPrimeFluxCacheFlush → VAEDecode → Save
    """

    CATEGORY     = "shannon-prime"
    RETURN_TYPES = ("LATENT",)
    FUNCTION     = "flush"
    DESCRIPTION  = (
        "Flush Flux block-skip caches before VAE decode. "
        "Frees CPU/GPU memory from cached attention tensors."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",)}}

    def flush(self, samples):
        import gc
        # Walk all Flux blocks and clear closure-captured state dicts
        # Since we don't have a model ref here, we rely on the state dict
        # being garbage-collected when refs drop. Force a GC + CUDA empty.
        gc.collect()
        if torch.cuda.is_available():
            before = torch.cuda.memory_allocated()
            torch.cuda.empty_cache()
            after = torch.cuda.memory_allocated()
            freed_mb = (before - after) / (1024 * 1024)
            if freed_mb > 1:
                print(f"[SP FluxFlush] freed {freed_mb:.0f}MB GPU memory")
        print("[SP FluxFlush] cache flush complete")
        return (samples,)


class ShannonPrimeFluxCacheFlushModel:
    """
    Model-aware Flux cache flush. Takes the MODEL as input to properly
    clear the closure-captured state dicts from FluxBlockSkip.

    This is the recommended flush node — it can walk the actual blocks
    and zero their caches deterministically.
    """

    CATEGORY     = "shannon-prime"
    RETURN_TYPES = ("MODEL",)
    FUNCTION     = "flush"
    DESCRIPTION  = (
        "Model-aware Flux cache flush. Walks blocks and clears "
        "all block-skip caches. Place between KSampler and VAEDecode."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    def flush(self, model):
        import gc
        cleared = 0
        for blocks_iter in (_iter_flux_double_blocks, _iter_flux_single_blocks):
            for _i, blk in blocks_iter(model):
                fwd = getattr(blk, "forward", None)
                if fwd and hasattr(fwd, "__closure__") and fwd.__closure__:
                    for cell in fwd.__closure__:
                        try:
                            obj = cell.cell_contents
                            if isinstance(obj, dict):
                                for k in ("attn_cache_img", "attn_cache_txt",
                                          "mlp_cache_img", "mlp_cache_txt",
                                          "single_cache", "step_cached",
                                          "hit_streak", "fisher_sim"):
                                    if k in obj and isinstance(obj[k], dict):
                                        obj[k].clear()
                                        cleared += 1
                        except ValueError:
                            pass

        gc.collect()
        if torch.cuda.is_available():
            before = torch.cuda.memory_allocated()
            torch.cuda.empty_cache()
            after = torch.cuda.memory_allocated()
            freed_mb = (before - after) / (1024 * 1024)
            print(f"[SP FluxFlush] cleared {cleared} caches, "
                  f"freed {freed_mb:.0f}MB GPU")
        else:
            print(f"[SP FluxFlush] cleared {cleared} caches")

        return (model,)


# ── Node Registration ─────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "ShannonPrimeFluxBlockSkip":       ShannonPrimeFluxBlockSkip,
    "ShannonPrimeFluxCacheFlush":      ShannonPrimeFluxCacheFlush,
    "ShannonPrimeFluxCacheFlushModel": ShannonPrimeFluxCacheFlushModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ShannonPrimeFluxBlockSkip":       "Shannon-Prime: Flux Block-Level Attention Skip (VHT2)",
    "ShannonPrimeFluxCacheFlush":      "Shannon-Prime: Flux Cache Flush (before VAE)",
    "ShannonPrimeFluxCacheFlushModel": "Shannon-Prime: Flux Cache Flush (Model-Aware)",
}
