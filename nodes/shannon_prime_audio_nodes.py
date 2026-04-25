# Shannon-Prime VHT2: Audio DiT ComfyUI Custom Nodes
# Copyright (C) 2026 Ray Daniels. All Rights Reserved.
# Licensed under AGPLv3. Commercial license available.
#
# Provides Shannon-Prime block-skip and caching nodes for Audio Diffusion
# Transformers (Stable Audio, etc.). These use a flat block architecture:
#   - ContinuousTransformer with N identical TransformerBlocks
#   - Each block: self-attn (RoPE) + optional cross-attn + optional Conformer + FF
#   - adaLN conditioning via 6-way scale/shift/gate: sigmoid(1-gate) gating
#
# The same Shannon-Prime principles apply: early blocks establish "Granite"
# structure (tonal foundation, harmonic layout) while late blocks add "Jazz"
# detail (transients, high-frequency texture). Block-skip caches the
# post-attention output and reapplies with the current step's modulation
# gate on cache hits.
#
# Audio DiT Architecture (from comfy.ldm.audio.dit):
#   - Default: depth=24, num_heads=24, embed_dim=1536, dim_heads=64
#   - RoPE on self-attention only (1D: audio frame positions)
#   - adaLN: to_scale_shift_gate → 6 outputs (scale/shift/gate × attn,ff)
#   - Gate function: sigmoid(1 - gate), NOT raw gate like Flux
#   - Cross-attention with separate context (text conditioning)
#   - Optional Conformer module (conv-based local context)

import os
import sys
import math
import pathlib
import gc

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


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: Audio DiT Block Iterators
# ═══════════════════════════════════════════════════════════════════════════

def _iter_audio_dit_blocks(model_obj):
    """Yield (index, block) for each TransformerBlock in an Audio DiT.

    Navigates ComfyUI's ModelPatcher wrapper:
        model_obj.model.diffusion_model.transformer.layers[i]
    """
    inner = getattr(model_obj, "model", model_obj)
    diff = getattr(inner, "diffusion_model", None)
    if diff is None:
        diff = getattr(inner, "model", None)
    if diff is None:
        return
    transformer = getattr(diff, "transformer", None)
    if transformer is None:
        return
    layers = getattr(transformer, "layers", None)
    if layers is None:
        return
    for i, blk in enumerate(layers):
        yield i, blk


def _detect_audio_head_dim(model_obj):
    """Detect head_dim and num_heads from an Audio DiT model.

    Returns (head_dim, num_heads). Falls back to (64, 24) for Stable Audio.
    """
    for i, blk in _iter_audio_dit_blocks(model_obj):
        if hasattr(blk, 'self_attn'):
            return blk.self_attn.dim_heads, blk.self_attn.num_heads
    return 64, 24  # Stable Audio default


def _count_audio_dit_blocks(model_obj):
    """Count total transformer blocks in the Audio DiT."""
    count = 0
    for _ in _iter_audio_dit_blocks(model_obj):
        count += 1
    return count


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: 1D Lattice RoPE for Audio
# ═══════════════════════════════════════════════════════════════════════════

_sp_audio_lattice_patched = False

def _install_lattice_rope_audio(model_obj, alpha=0.17, verbose=False):
    """Install 1D lattice-prime RoPE factors on the Audio DiT's rotary embeddings.

    Audio is 1D (frame positions along time axis), so we use a single tier
    mapping:
      - Low frequencies (long-range tonal structure) → Long-Tier primes
      - High frequencies (transient detail) → Local-Tier primes

    This is simpler than Flux's 2D spatial+temporal mapping but follows
    the same principle: prime-indexed frequency factors that are coprime
    to the sequence length, preventing aliasing at any position.
    """
    global _sp_audio_lattice_patched

    if alpha <= 0.0 or _sp_audio_lattice_patched:
        return

    inner = getattr(model_obj, "model", model_obj)
    diff = getattr(inner, "diffusion_model", None)
    if diff is None:
        diff = getattr(inner, "model", None)
    if diff is None:
        return
    transformer = getattr(diff, "transformer", None)
    if transformer is None:
        return
    rope_module = getattr(transformer, "rotary_pos_emb", None)
    if rope_module is None:
        return

    # Get the inv_freq buffer
    inv_freq = getattr(rope_module, "inv_freq", None)
    if inv_freq is None:
        return

    n_freqs = inv_freq.shape[0]
    if n_freqs < 2:
        return

    # Audio: single 1D axis, use 'long' tier for temporal coherence
    # Low-index freqs capture long-range structure (tonal), high-index = transient detail
    factors = _tiered_lattice_factors(n_freqs, freq_base=10000.0, alpha=alpha, tier='long')
    factors = factors.to(device=inv_freq.device, dtype=inv_freq.dtype)

    # Blend: inv_freq *= factors
    with torch.no_grad():
        rope_module.inv_freq.copy_(inv_freq * factors)

    _sp_audio_lattice_patched = True

    if verbose:
        print(f"[Shannon-Prime Audio] Lattice RoPE installed: {n_freqs} freqs, "
              f"alpha={alpha:.2f}, tier=long")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: ShannonPrimeAudioBlockSkip Node
# ═══════════════════════════════════════════════════════════════════════════

class ShannonPrimeAudioBlockSkip:
    """Block-level attention skip for Audio Diffusion Transformer (Stable Audio).

    Caches self-attention output (pre-gate) per block per denoising step.
    On cache hit, replays cached attention with the current step's adaLN gate,
    skipping the expensive attention computation.

    Tier layout for 24-block Stable Audio DiT:
      Tier 0 (blocks 0-5):   Tonal foundation  — high stability, long window
      Tier 1 (blocks 6-17):  Harmonic mid-range — moderate stability
      Tier 2 (blocks 18-23): Transient detail   — volatile, short/no window
    """

    CATEGORY = "shannon-prime"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "tier0_window": ("INT", {
                    "default": 8, "min": 0, "max": 32, "step": 1,
                    "tooltip": "Cache window for tonal foundation blocks (0-5). "
                               "Higher = more reuse, faster. 0 = disable.",
                }),
                "tier1_window": ("INT", {
                    "default": 3, "min": 0, "max": 32, "step": 1,
                    "tooltip": "Cache window for harmonic mid-range blocks (6-17).",
                }),
                "tier2_window": ("INT", {
                    "default": 0, "min": 0, "max": 32, "step": 1,
                    "tooltip": "Cache window for transient detail blocks (18+). "
                               "Default 0 (disabled) — these are volatile.",
                }),
                "cache_mlp": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "TURBO: also cache MLP output. Faster but may lose detail.",
                }),
                "cache_dtype": (["auto", "fp16", "fp8"],  {
                    "default": "auto",
                    "tooltip": "Precision for cached tensors. auto = match model.",
                }),
                "lattice_rope": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply 1D lattice-prime RoPE factors to audio positions.",
                }),
                "lattice_alpha": ("FLOAT", {
                    "default": 0.17, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "Lattice blending strength. 0 = no effect, 0.17 = default.",
                }),
                "verbose": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print per-block hit/miss and Fisher similarity diagnostics.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # Always re-execute

    def patch(self, model, tier0_window=8, tier1_window=3, tier2_window=0,
              cache_mlp=False, cache_dtype="auto", lattice_rope=True,
              lattice_alpha=0.17, verbose=False):

        model_obj = model.model if hasattr(model, 'model') else model

        # ── Detect architecture ──────────────────────────────────────────
        head_dim, num_heads = _detect_audio_head_dim(model_obj)
        n_blocks = _count_audio_dit_blocks(model_obj)

        if n_blocks == 0:
            print("[Shannon-Prime Audio] No Audio DiT blocks found — pass-through.")
            return (model,)

        if verbose:
            print(f"[Shannon-Prime Audio] Detected {n_blocks} blocks, "
                  f"head_dim={head_dim}, num_heads={num_heads}")

        # ── Tier window map ──────────────────────────────────────────────
        def get_window(i):
            if i < 6:
                return tier0_window     # Tonal foundation
            if i < 18:
                return tier1_window     # Harmonic mid-range
            return tier2_window         # Transient detail

        def get_streak_limit(i):
            if i < 6:
                return 8    # Very stable blocks
            if i < 18:
                return 4    # Moderate
            return 3        # Detail blocks

        # ── Cache dtype ──────────────────────────────────────────────────
        _is_fp8 = (cache_dtype == "fp8")

        if cache_dtype == "fp8":
            _blk_dtype = torch.float8_e4m3fn
        elif cache_dtype == "fp16":
            _blk_dtype = torch.float16
        else:
            # auto: match model dtype
            _blk_dtype = torch.float16  # safe default
            for _, blk in _iter_audio_dit_blocks(model_obj):
                for p in blk.parameters():
                    if p.dtype in (torch.float16, torch.bfloat16):
                        _blk_dtype = p.dtype
                    break
                break

        # ── Fisher weights ───────────────────────────────────────────────
        _fisher_w = _fisher_diagonal_weights(head_dim)

        # ── Shared state dict ────────────────────────────────────────────
        state = {
            'global_step':  [0],
            'hit_streak':   {},
            'attn_cache':   {},     # block_idx → cached self-attn output (pre-gate)
            'mlp_cache':    {},     # block_idx → cached FF output (TURBO only)
            'step_cached':  {},     # block_idx → step when cached
            'fisher_sim':   {},     # block_idx → last Fisher cosine similarity
        }

        # ── Store / Load helpers ─────────────────────────────────────────
        def _store(t):
            return t.detach().to(dtype=_blk_dtype).cpu()

        def _load(t, ref):
            if _is_fp8:
                return t.to(dtype=ref.dtype).to(device=ref.device)
            return t.to(device=ref.device, dtype=ref.dtype)

        # ── Clear any prior Shannon-Prime patches ────────────────────────
        for i, blk in _iter_audio_dit_blocks(model_obj):
            fwd = getattr(blk, 'forward', None)
            if fwd is not None and hasattr(fwd, '__closure__') and fwd.__closure__:
                for cell in fwd.__closure__:
                    try:
                        v = cell.cell_contents
                        if isinstance(v, dict) and 'attn_cache' in v:
                            v['attn_cache'].clear()
                            v['mlp_cache'].clear()
                            v['step_cached'].clear()
                            v['hit_streak'].clear()
                            v['fisher_sim'].clear()
                    except (ValueError, AttributeError):
                        pass

        # ── Lattice RoPE ─────────────────────────────────────────────────
        if lattice_rope:
            _install_lattice_rope_audio(model_obj, alpha=lattice_alpha, verbose=verbose)

        # ── Patched forward factory ──────────────────────────────────────

        def make_patched_forward(orig_forward, block_idx, blk):
            window = get_window(block_idx)
            _streak_limit = get_streak_limit(block_idx)

            def patched_forward(x, context=None, global_cond=None, mask=None,
                                context_mask=None, rotary_pos_emb=None,
                                transformer_options={}):

                # ── Step counter (block 0 increments) ────────────────
                if block_idx == 0:
                    state['global_step'][0] += 1
                step = state['global_step'][0]

                # ── Compute adaLN modulation (always — trivial cost) ─
                has_adaln = (blk.global_cond_dim is not None
                             and blk.global_cond_dim > 0
                             and global_cond is not None)

                if has_adaln:
                    modulation = blk.to_scale_shift_gate(global_cond).unsqueeze(1)
                    scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff = \
                        modulation.chunk(6, dim=-1)

                # ── Cache hit check ──────────────────────────────────
                cached_attn = state['attn_cache'].get(block_idx)
                age = step - state['step_cached'].get(block_idx, -999)
                streak = state['hit_streak'].get(block_idx, 0)

                hit = (cached_attn is not None
                       and age >= 0
                       and age < window
                       and streak < _streak_limit)

                if hit:
                    # ══════════════════════════════════════════════════
                    # CACHE HIT: replay self-attn with current gate
                    # ══════════════════════════════════════════════════
                    attn_res = _load(cached_attn, x)

                    if has_adaln:
                        residual = x
                        x = residual + attn_res * torch.sigmoid(1 - gate_self)
                    else:
                        x = x + attn_res

                    # Cross-attention always runs fresh (context changes per step)
                    if context is not None and hasattr(blk, 'cross_attn'):
                        x = x + blk.cross_attn(
                            blk.cross_attend_norm(x),
                            context=context,
                            context_mask=context_mask,
                            transformer_options=transformer_options,
                        )

                    # Conformer: fresh (local conv context)
                    if blk.conformer is not None:
                        x = x + blk.conformer(x)

                    # Feedforward: cached (TURBO) or fresh
                    cached_mlp = state['mlp_cache'].get(block_idx)
                    if cache_mlp and cached_mlp is not None:
                        mlp_res = _load(cached_mlp, x)
                        if has_adaln:
                            residual = x
                            x = residual + mlp_res * torch.sigmoid(1 - gate_ff)
                        else:
                            x = x + mlp_res
                    else:
                        if has_adaln:
                            residual = x
                            normed = blk.ff_norm(x)
                            normed = normed * (1 + scale_ff) + shift_ff
                            ff_out = blk.ff(normed)
                            x = residual + ff_out * torch.sigmoid(1 - gate_ff)
                        else:
                            x = x + blk.ff(blk.ff_norm(x))

                    state['hit_streak'][block_idx] = streak + 1

                    if verbose:
                        sim = state['fisher_sim'].get(block_idx, 0.0)
                        print(f"  [Audio blk {block_idx:2d}] HIT  step={step} "
                              f"age={age} streak={streak+1} fisher={sim:.4f}")

                else:
                    # ══════════════════════════════════════════════════
                    # CACHE MISS: run full block, intercept attn pre-gate
                    # ══════════════════════════════════════════════════

                    if has_adaln:
                        # Self-attention with adaLN
                        residual = x
                        normed = blk.pre_norm(x)
                        normed = normed * (1 + scale_self) + shift_self
                        attn_out = blk.self_attn(
                            normed, mask=mask,
                            rotary_pos_emb=rotary_pos_emb,
                            transformer_options=transformer_options,
                        )

                        # ── Fisher drift diagnostic ──────────────────
                        prev = state['attn_cache'].get(block_idx)
                        if prev is not None:
                            prev_on_dev = _load(prev, attn_out)
                            # Sample subset for efficiency
                            sample_dim = min(attn_out.shape[-1], head_dim)
                            a_sample = attn_out[..., :sample_dim].reshape(-1, sample_dim)
                            p_sample = prev_on_dev[..., :sample_dim].reshape(-1, sample_dim)
                            f_w = _fisher_w[:sample_dim] if _fisher_w.shape[0] >= sample_dim else _fisher_w
                            f_sim = _fisher_cos_sim(a_sample[:64], p_sample[:64], f_w)
                            state['fisher_sim'][block_idx] = f_sim
                        else:
                            state['fisher_sim'][block_idx] = 0.0

                        # Cache pre-gate attention output
                        state['attn_cache'][block_idx] = _store(attn_out)
                        state['step_cached'][block_idx] = step
                        state['hit_streak'][block_idx] = 0

                        # Apply gate
                        x = residual + attn_out * torch.sigmoid(1 - gate_self)

                        # Cross-attention
                        if context is not None and hasattr(blk, 'cross_attn'):
                            x = x + blk.cross_attn(
                                blk.cross_attend_norm(x),
                                context=context,
                                context_mask=context_mask,
                                transformer_options=transformer_options,
                            )

                        # Conformer
                        if blk.conformer is not None:
                            x = x + blk.conformer(x)

                        # Feedforward with adaLN
                        residual = x
                        normed = blk.ff_norm(x)
                        normed = normed * (1 + scale_ff) + shift_ff
                        ff_out = blk.ff(normed)

                        if cache_mlp:
                            state['mlp_cache'][block_idx] = _store(ff_out)

                        x = residual + ff_out * torch.sigmoid(1 - gate_ff)

                    else:
                        # No adaLN path (simpler)
                        normed = blk.pre_norm(x)
                        attn_out = blk.self_attn(
                            normed, mask=mask,
                            rotary_pos_emb=rotary_pos_emb,
                            transformer_options=transformer_options,
                        )

                        # Fisher drift
                        prev = state['attn_cache'].get(block_idx)
                        if prev is not None:
                            prev_on_dev = _load(prev, attn_out)
                            sample_dim = min(attn_out.shape[-1], head_dim)
                            a_sample = attn_out[..., :sample_dim].reshape(-1, sample_dim)
                            p_sample = prev_on_dev[..., :sample_dim].reshape(-1, sample_dim)
                            f_w = _fisher_w[:sample_dim] if _fisher_w.shape[0] >= sample_dim else _fisher_w
                            f_sim = _fisher_cos_sim(a_sample[:64], p_sample[:64], f_w)
                            state['fisher_sim'][block_idx] = f_sim

                        state['attn_cache'][block_idx] = _store(attn_out)
                        state['step_cached'][block_idx] = step
                        state['hit_streak'][block_idx] = 0

                        x = x + attn_out

                        if context is not None and hasattr(blk, 'cross_attn'):
                            x = x + blk.cross_attn(
                                blk.cross_attend_norm(x),
                                context=context,
                                context_mask=context_mask,
                                transformer_options=transformer_options,
                            )

                        if blk.conformer is not None:
                            x = x + blk.conformer(x)

                        ff_out = blk.ff(blk.ff_norm(x))
                        if cache_mlp:
                            state['mlp_cache'][block_idx] = _store(ff_out)
                        x = x + ff_out

                    if verbose:
                        sim = state['fisher_sim'].get(block_idx, 0.0)
                        print(f"  [Audio blk {block_idx:2d}] MISS step={step} "
                              f"fisher={sim:.4f}")

                return x

            return patched_forward

        # ── Apply patches ────────────────────────────────────────────────
        patched_count = 0
        for i, blk in _iter_audio_dit_blocks(model_obj):
            w = get_window(i)
            if w > 0:
                blk.forward = make_patched_forward(blk.forward, i, blk)
                patched_count += 1

        if verbose or patched_count > 0:
            tier_summary = []
            for i in range(n_blocks):
                w = get_window(i)
                if w > 0:
                    tier_summary.append(f"blk{i}:w{w}")
            print(f"[Shannon-Prime Audio] Patched {patched_count}/{n_blocks} blocks: "
                  f"{', '.join(tier_summary[:8])}{'...' if len(tier_summary) > 8 else ''}")
            print(f"  dtype={_blk_dtype}, cache_mlp={cache_mlp}, "
                  f"lattice_rope={lattice_rope} (alpha={lattice_alpha:.2f})")

        return (model,)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: Cache Flush Nodes
# ═══════════════════════════════════════════════════════════════════════════

class ShannonPrimeAudioCacheFlush:
    """Flush Shannon-Prime audio caches and free VRAM.

    Connect to the LATENT output to trigger at the right point in the graph.
    """
    CATEGORY = "shannon-prime"
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "flush"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
            },
        }

    def flush(self, samples):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return (samples,)


class ShannonPrimeAudioCacheFlushModel:
    """Deterministic cache flush — walks Audio DiT blocks and clears state dicts."""

    CATEGORY = "shannon-prime"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "flush"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
        }

    def flush(self, model):
        model_obj = model.model if hasattr(model, 'model') else model
        cleared = 0
        for i, blk in _iter_audio_dit_blocks(model_obj):
            fwd = getattr(blk, 'forward', None)
            if fwd is not None and hasattr(fwd, '__closure__') and fwd.__closure__:
                for cell in fwd.__closure__:
                    try:
                        v = cell.cell_contents
                        if isinstance(v, dict) and 'attn_cache' in v:
                            v['attn_cache'].clear()
                            v['mlp_cache'].clear()
                            v['step_cached'].clear()
                            v['hit_streak'].clear()
                            v['fisher_sim'].clear()
                            v['global_step'][0] = 0
                            cleared += 1
                    except (ValueError, AttributeError):
                        pass

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if cleared > 0:
            print(f"[Shannon-Prime Audio] Flushed {cleared} block caches")

        return (model,)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: NODE_CLASS_MAPPINGS
# ═══════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "ShannonPrimeAudioBlockSkip": ShannonPrimeAudioBlockSkip,
    "ShannonPrimeAudioCacheFlush": ShannonPrimeAudioCacheFlush,
    "ShannonPrimeAudioCacheFlushModel": ShannonPrimeAudioCacheFlushModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ShannonPrimeAudioBlockSkip": "Shannon-Prime: Audio Block-Level Attention Skip (VHT2)",
    "ShannonPrimeAudioCacheFlush": "Shannon-Prime: Audio Cache Flush (LATENT)",
    "ShannonPrimeAudioCacheFlushModel": "Shannon-Prime: Audio Cache Flush (MODEL)",
}
