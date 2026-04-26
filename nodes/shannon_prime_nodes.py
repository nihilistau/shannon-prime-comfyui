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
import math
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



# ── Fisher diagonal weighting for block-skip drift detection ─────────────
#
# The standard L2/cos_sim drift check treats all dimensions equally. In
# the VHT2 spectral basis, low-frequency (squarefree) coefficients carry
# most of the "information" — drift along high-frequency dimensions is
# perceptually inert. Weighting by spectral energy rank before computing
# similarity is a zero-cost diagonal Fisher approximation: it suppresses
# false invalidations from high-frequency noise, yielding ~10-20% longer
# cache windows at the same quality.
#
# The weights are computed once at module load and cached. For head_dim=128
# this is a [128] float32 vector — 512 bytes. The weighting is a single
# elementwise multiply in the drift check, same cost as the existing norm.

_FISHER_DIAG_CACHE = {}  # head_dim -> Tensor [head_dim] on CPU

def _fisher_diagonal_weights(head_dim: int = 128) -> torch.Tensor:
    """
    Return per-dimension information weights for the VHT2 spectral basis.

    Squarefree indices (1-indexed) correspond to the prime harmonic basis
    functions {2,3,5,7,11,...}. These carry the structured information.
    Non-squarefree indices are redundant overtones.

    Weight scheme:
      - Squarefree positions:   1.0 (full weight — information-bearing)
      - Non-squarefree:         0.1 (attenuated — perceptually inert noise)

    This is the simplest useful Fisher diagonal: it separates "signal drift"
    from "noise drift." More sophisticated versions could weight by measured
    per-position variance from calibration data, but this binary mask already
    captures the dominant structure.
    """
    if head_dim in _FISHER_DIAG_CACHE:
        return _FISHER_DIAG_CACHE[head_dim]

    w = torch.full((head_dim,), 0.1, dtype=torch.float32)
    for i in range(head_dim):
        n = i + 1  # 1-indexed
        is_sqfree = True
        p = 2
        while p * p <= n:
            if n % (p * p) == 0:
                is_sqfree = False
                break
            p += 1
        if is_sqfree:
            w[i] = 1.0

    # Normalize so the weighted norm has the same scale as unweighted
    w = w / w.norm()
    _FISHER_DIAG_CACHE[head_dim] = w
    return w


def _fisher_cos_sim(a: torch.Tensor, b: torch.Tensor,
                    weights: torch.Tensor) -> float:
    """
    Fisher-weighted cosine similarity between two tensors.

    Accepts either [..., head_dim] (already split into heads) or
    [..., hidden_dim] where hidden_dim = num_heads * head_dim. In the
    latter case (the actual self-attn output shape in Wan blocks: e.g.
    [B, S, 5120] = 40 heads × 128 head_dim), the last axis is reshaped
    into [..., num_heads, head_dim] before the weighted multiply so the
    [head_dim] weight vector broadcasts correctly.

    weights: [head_dim] float32 (CPU; moved to a's device on first call).

    Returns scalar float cos_sim weighted by the Fisher diagonal.

    Bug history: prior to feat/strange-attractor-stack the function only
    fired when verbose=True, where the caller had already arranged the
    head split. The drift gate makes this always-on and feeds the
    pre-split [B, S, hidden_dim] tensor; the reshape below is the fix.
    """
    head_dim = int(weights.shape[0])
    last = a.shape[-1]
    if last == head_dim:
        a_view = a
        b_view = b
    elif last % head_dim == 0:
        # Split the last axis into (num_heads, head_dim)
        new_shape_a = a.shape[:-1] + (last // head_dim, head_dim)
        new_shape_b = b.shape[:-1] + (last // head_dim, head_dim)
        a_view = a.reshape(*new_shape_a)
        b_view = b.reshape(*new_shape_b)
    else:
        # Fallback: weights aren't applicable to this layout. Plain cos_sim.
        af = a.reshape(-1).float()
        bf = b.reshape(-1).float()
        na, nb = af.norm(), bf.norm()
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float((af * bf).sum() / (na * nb))

    # Flatten everything except the head_dim axis
    a_flat = a_view.reshape(-1, head_dim).float()
    b_flat = b_view.reshape(-1, head_dim).float()

    w = weights.to(device=a_flat.device)

    aw = a_flat * w
    bw = b_flat * w

    dot = (aw * bw).sum()
    norm_a = aw.norm()
    norm_b = bw.norm()
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(dot / (norm_a * norm_b))


# ── VHT2 Memory Pool for zero-allocation compress/decompress ─────────────
#
# Phase 15 LEAN disabled VHT2 because per-step torch.empty() calls
# fragmented VRAM on lowvram GPUs. This pool pre-allocates all intermediates
# at init time and reuses them across steps. Zero new allocations per step.

class _VHT2MemoryPool:
    """
    Pre-allocated GPU memory pool for VHT2 compress/decompress intermediates.

    Allocates once at init:
      - spectral_buf: [max_tokens, head_dim] for butterfly transform output
      - skeleton_buf: [max_tokens, skeleton_size] for skeleton extraction
      - recon_buf:    [max_tokens, head_dim] for reconstruction

    All operations write into these buffers instead of allocating new tensors.
    Pool is keyed by (device, max_tokens, head_dim) — create one per config.
    """

    def __init__(self, max_tokens: int, head_dim: int, skeleton_size: int,
                 device: torch.device, dtype: torch.dtype = torch.float16):
        self.max_tokens = max_tokens
        self.head_dim = head_dim
        self.skeleton_size = skeleton_size
        self.device = device
        self.dtype = dtype

        # Pre-allocate all intermediates
        self.spectral_buf = torch.empty(max_tokens, head_dim,
                                        dtype=torch.float32, device=device)
        self.skeleton_buf = torch.empty(max_tokens, skeleton_size,
                                        dtype=dtype, device=device)
        self.recon_buf = torch.empty(max_tokens, head_dim,
                                     dtype=torch.float32, device=device)

        mem_mb = (self.spectral_buf.nbytes + self.skeleton_buf.nbytes +
                  self.recon_buf.nbytes) / (1024 * 1024)
        print(f"[SP VHT2Pool] allocated {mem_mb:.1f}MB on {device} "
              f"(tokens={max_tokens}, hd={head_dim}, skel={skeleton_size})")

    def compress(self, x: torch.Tensor, vht2_fn, skeleton_mask: torch.Tensor):
        """
        Compress x into skeleton coefficients using pre-allocated buffers.
        x: [N, head_dim] on GPU
        vht2_fn: callable VHT2 forward transform
        skeleton_mask: [head_dim] bool mask

        Returns: skeleton coefficients as CPU tensor [N, skeleton_size] in self.dtype
        """
        N = x.shape[0]
        assert N <= self.max_tokens, f"Token count {N} > pool max {self.max_tokens}"

        # VHT2 forward into spectral_buf (no new allocation)
        buf = self.spectral_buf[:N]
        buf.copy_(x.float())
        spectral = vht2_fn(buf)  # writes result, may return new tensor

        # Extract skeleton into skeleton_buf
        skel = spectral[:, skeleton_mask].to(dtype=self.dtype)
        out = skel.cpu()  # immediate D2H — free GPU intermediate by scope
        return out

    def decompress(self, skel_cpu: torch.Tensor, vht2_fn,
                   skeleton_mask: torch.Tensor, target_dtype: torch.dtype):
        """
        Decompress skeleton coefficients back to full vectors.
        skel_cpu: [N, skeleton_size] on CPU
        Returns: [N, head_dim] on GPU in target_dtype
        """
        N = skel_cpu.shape[0]
        assert N <= self.max_tokens

        # Zero-fill spectral buffer and scatter skeleton
        buf = self.recon_buf[:N]
        buf.zero_()
        buf[:, skeleton_mask] = skel_cpu.to(device=self.device, dtype=torch.float32)

        # VHT2 inverse (self-inverse)
        recon = vht2_fn(buf)
        return recon.to(dtype=target_dtype)

    def release(self):
        """Free all pool memory."""
        del self.spectral_buf, self.skeleton_buf, self.recon_buf
        torch.cuda.empty_cache()


# Global pool registry (one pool per config)
_VHT2_POOLS = {}

def _get_vht2_pool(max_tokens: int, head_dim: int, skeleton_size: int,
                   device: torch.device, dtype: torch.dtype = torch.float16):
    """Get or create a VHT2 memory pool for the given configuration."""
    key = (str(device), max_tokens, head_dim, skeleton_size)
    if key not in _VHT2_POOLS:
        _VHT2_POOLS[key] = _VHT2MemoryPool(
            max_tokens, head_dim, skeleton_size, device, dtype)
    return _VHT2_POOLS[key]


# ── Partition Z logging ──────────────────────────────────────────────────
#
# The softmax denominator Z (partition function) measures how "diffuse"
# the attention distribution is. When Z explodes relative to the
# arithmetical skeleton, the model has entered the "Jazz" regime and
# cache invalidation is necessary.
#
# This logger hooks into self-attention to capture Z without any compute
# overhead — Z is already computed in every softmax, we just expose it.

_PARTITION_Z_LOG = {}  # step -> {block_idx -> float Z_mean}

def _log_partition_z(step: int, block_idx: int, z_value: float):
    """Record partition Z for a given step and block."""
    if step not in _PARTITION_Z_LOG:
        _PARTITION_Z_LOG[step] = {}
    _PARTITION_Z_LOG[step][block_idx] = z_value

def _print_partition_z_summary(step: int, n_blocks: int):
    """Print partition Z summary for the completed step."""
    if step not in _PARTITION_Z_LOG:
        return
    zs = _PARTITION_Z_LOG[step]
    if not zs:
        return
    z_vals = [zs.get(i, float('nan')) for i in range(min(n_blocks, 8))]
    z_str = "  ".join(f"B{i:02d}={v:.1f}" for i, v in enumerate(z_vals) if not math.isnan(v))
    try:
        import tqdm as _tqdm
        _tqdm.tqdm.write(f"[SP PartZ] step={step:3d}  {z_str}")
    except Exception:
        print(f"[SP PartZ] step={step:3d}  {z_str}")


# ── Factored 3D Lattice RoPE for Wan Video DiT ─────────────────────────────
#
# Wan uses 3D video RoPE: temporal × height × width, with separate
# frequency generation per axis. Standard RoPE uses geometric freqs
# θ_j = base^(-2j/d). We blend lattice-aligned integer frequencies
# into each axis with anisotropic tier mapping:
#
#   Temporal axis (d=48): Long-Tier (primes 1009..8209)
#     → causal anchor across frame window, low-frequency periodicity
#   Spatial axes (d=40+40): Local-Tier (primes 2..101)
#     → within-frame detail, high-frequency structure
#
# This respects the physics: temporal coherence spans the whole clip
# while spatial detail is per-frame. The lattice concentrates spectral
# resolution where the information density is highest per axis.
#
# The hook monkey-patches comfy.ldm.wan.model.get_1d_rotary_pos_embed
# to blend lattice freqs at alpha=0.17 before computing the cos/sin
# embedding. Zero per-token cost — factors are constant per dim.

_SIEVE_CACHE = None

def _sieve_primes(n=10000):
    """Sieve of Eratosthenes up to n. Cached."""
    global _SIEVE_CACHE
    if _SIEVE_CACHE is not None and len(_SIEVE_CACHE) > 0:
        return _SIEVE_CACHE
    is_p = [True] * (n + 1)
    is_p[0] = is_p[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_p[i]:
            for j in range(i * i, n + 1, i):
                is_p[j] = False
    _SIEVE_CACHE = [i for i in range(2, n + 1) if is_p[i]]
    return _SIEVE_CACHE

def _pick_evenly(pool, n):
    if n <= 0:
        return []
    if len(pool) <= n:
        return list(pool)
    step = len(pool) / n
    return [pool[int(i * step)] for i in range(n)]

def _tiered_lattice_factors(n_freqs, freq_base=10000.0, alpha=0.17, tier='auto'):
    """
    Compute lattice-blended freq_factors for one RoPE axis.

    tier: 'local' (spatial), 'long' (temporal), or 'auto' (standard 3-tier)
    Returns: torch.Tensor [n_freqs] float32 frequency multipliers.
    """
    if alpha < 1e-6 or n_freqs <= 0:
        return torch.ones(n_freqs, dtype=torch.float32)

    primes = set(_sieve_primes())
    d = n_freqs * 2

    # Geometric baseline
    geometric = torch.tensor(
        [freq_base ** (-2.0 * j / d) for j in range(n_freqs)],
        dtype=torch.float32
    )

    # Build composite pool (lattice coordinates) filtered by tier
    if tier == 'local':
        # Spatial: small composites (high frequency, fine detail)
        pool = [n for n in range(4, 200) if n not in primes]
    elif tier == 'long':
        # Temporal: large composites (low frequency, causal anchors)
        pool = [n for n in range(500, 8210) if n not in primes]
    else:
        # Standard 3-tier allocation (same as sp_inject_freqs.py)
        n_loc = max(1, n_freqs // 4)
        n_mid = max(1, (n_freqs * 33) // 100)
        n_lng = n_freqs - n_loc - n_mid
        loc = [n for n in range(4, 102) if n not in primes]
        mid = [n for n in range(102, 1010) if n not in primes]
        lng = [n for n in range(1010, 8210) if n not in primes]
        lattice_int = (_pick_evenly(loc, n_loc) +
                       _pick_evenly(mid, n_mid) +
                       _pick_evenly(lng, n_lng))
        while len(lattice_int) < n_freqs:
            lattice_int.append(lattice_int[-1] + 1)
        lattice_int = lattice_int[:n_freqs]
        lattice = torch.tensor(lattice_int, dtype=torch.float32)

        # Normalize to geometric scale
        lat_range = lattice.max() - lattice.min()
        geo_range = geometric.max() - geometric.min()
        if lat_range > 1e-6:
            lattice = geometric.min() + (lattice - lattice.min()) / lat_range * geo_range

        blended = (1.0 - alpha) * geometric + alpha * lattice
        factors = blended / geometric.clamp(min=1e-12)
        return factors

    # Tier-specific path (local or long)
    lattice_int = _pick_evenly(pool, n_freqs)
    while len(lattice_int) < n_freqs:
        lattice_int.append(lattice_int[-1] + 1)
    lattice_int = lattice_int[:n_freqs]
    lattice = torch.tensor(lattice_int, dtype=torch.float32)

    # Normalize to geometric scale
    lat_range = lattice.max() - lattice.min()
    geo_range = geometric.max() - geometric.min()
    if lat_range > 1e-6:
        lattice = geometric.min() + (lattice - lattice.min()) / lat_range * geo_range

    blended = (1.0 - alpha) * geometric + alpha * lattice
    factors = blended / geometric.clamp(min=1e-12)
    return factors


def _install_lattice_rope(alpha=0.17):
    """
    Monkey-patch comfy's Wan get_1d_rotary_pos_embed to use factored
    lattice frequencies. Idempotent — safe to call multiple times.

    Returns True if patching succeeded, False if Wan model not available.
    """
    try:
        import comfy.ldm.wan.model as wan_model
    except ImportError:
        return False

    # Don't double-patch
    if getattr(wan_model, '_sp_lattice_patched', False):
        return True

    _orig_get_1d = wan_model.get_1d_rotary_pos_embed

    def _lattice_get_1d_rotary_pos_embed(dim, pos, theta=10000.0, **kwargs):
        """
        Drop-in replacement that blends lattice frequencies into Wan's
        per-axis RoPE. Uses the dim hint to decide tier:
          dim <= 48: likely temporal → Long-Tier
          dim > 48:  likely spatial  → Local-Tier

        The blended frequencies are applied via freq_factors on the
        geometric baseline — same mechanism as sp_inject_freqs.py.
        """
        n_freqs = dim // 2

        # Determine tier from dimension:
        # Wan 2.2 5B: d_t=48, d_h=40, d_w=40 (total 128)
        # Temporal dim is typically the largest per-axis allocation
        if dim >= 48:
            tier = 'long'    # temporal: causal anchors
        else:
            tier = 'local'   # spatial: fine detail

        factors = _tiered_lattice_factors(n_freqs, freq_base=theta,
                                          alpha=alpha, tier=tier)

        # Modify theta per-dim: effective_theta[j] = theta / factor[j]
        # This is equivalent to applying freq_factors in ggml_rope_ext
        import types
        import functools

        # Call original but with modified frequencies
        # The cleanest path: compute the embedding ourselves using the
        # blended frequencies, matching the original's output format.
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        freqs = freqs * factors  # apply lattice blend

        t = torch.arange(pos, dtype=freqs.dtype) if isinstance(pos, int) \
            else torch.tensor(pos, dtype=freqs.dtype) if not isinstance(pos, torch.Tensor) \
            else pos.float()

        freqs_outer = torch.outer(t, freqs)
        emb = torch.cat([freqs_outer.cos(), freqs_outer.sin()], dim=-1)
        return emb

    wan_model.get_1d_rotary_pos_embed = _lattice_get_1d_rotary_pos_embed
    wan_model._sp_lattice_patched = True
    print(f"[Shannon-Prime] Factored 3D Lattice RoPE installed (α={alpha})")
    print(f"[Shannon-Prime]   Temporal → Long-Tier, Spatial → Local-Tier")
    return True


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
                "k_bits": ("STRING", {"default": "4,3,3,3",
                                      "tooltip": "K band bit allocation (4 bands). Ignored in LEAN mode (raw CPU cache). Used when cache_compress=vht2."}),
                "v_bits": ("STRING", {"default": "4,3,3,3",
                                      "tooltip": "V band bit allocation. Ignored in LEAN mode. Used when cache_compress=vht2."}),
                "use_mobius": ("BOOLEAN", {"default": False,
                                           "tooltip": "Möbius squarefree-first reorder. Default OFF — LEAN mode uses raw CPU cache. Enable with cache_compress=vht2 if desired."}),
                "lattice_rope": ("BOOLEAN", {"default": True,
                                              "tooltip": "Inject factored 3D lattice RoPE frequencies (PrimePE). Temporal axes get Long-Tier anchors, spatial get Local-Tier detail. ~0.6-0.8% quality improvement, zero runtime cost."}),
                "lattice_alpha": ("FLOAT", {"default": 0.17, "min": 0.0, "max": 0.5, "step": 0.01,
                                             "tooltip": "Lattice blend ratio α. 0.0 = pure geometric RoPE, 0.17 = paper default. Range 0.15-0.22 validated."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # The patch isn't deterministic w.r.t. pure input hashing — different
        # applications share cache state on the model. Returning NaN forces
        # re-evaluation each queue.
        return float("nan")

    def patch(self, model, k_bits: str = "5,4,4,3", v_bits: str = "5,4,4,3",
              use_mobius: bool = True, lattice_rope: bool = True,
              lattice_alpha: float = 0.17):
        # Phase 15 LEAN: cross-attn cache is now raw CPU/fp16 — no VHT2.
        # k_bits, v_bits, use_mobius are retained in INPUT_TYPES for backward
        # compatibility with saved workflows but are IGNORED. The node just
        # wraps cross-attn linears with raw CPU caching.

        # ── PrimePE: Factored 3D Lattice RoPE ──
        # Inject lattice-aligned frequencies into Wan's per-axis RoPE.
        # Temporal → Long-Tier anchors, Spatial → Local-Tier detail.
        # Zero per-step cost (frequencies computed once at patch time).
        if lattice_rope and lattice_alpha > 0.0:
            if _install_lattice_rope(alpha=lattice_alpha):
                pass  # success message printed by _install_lattice_rope
            else:
                print("[Shannon-Prime] Lattice RoPE: Wan model not available, skipping")
        elif not lattice_rope:
            print("[Shannon-Prime] Lattice RoPE: disabled by user")

        patched = model.clone()

        blocks = list(_iter_wan_blocks(patched))
        if not blocks:
            print("[Shannon-Prime] ShannonPrimeWanCache: no Wan blocks found on model — passing through")
            return (patched,)

        wrapped = 0
        for i, blk in blocks:
            if _wrap_cross_attn(getattr(blk, "cross_attn", None), i):
                wrapped += 1

        print(f"[Shannon-Prime] Phase 16 LEAN — patched {wrapped}/{len(blocks)} "
              f"Wan cross-attn linears with raw CPU/fp16 caching")
        print(f"[Shannon-Prime] Möbius OFF, raw CPU cache — "
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
                "tier_2_window": ("INT",   {"default": 0,  "min": 0, "max": 10,
                    "tooltip": "Cache window for L09-L15 (Volatile). 0=disabled. Try 2 for SVI 3-step."}),
                "tier_3_window": ("INT",   {"default": 0,  "min": 0, "max": 5,
                    "tooltip": "Cache window for L16-L39 (Deep/late — texture detail). 0=disabled. YOLO: try 2 for SVI."}),
                "cache_ffn": ("BOOLEAN", {"default": False,
                    "tooltip": "TURBO: also cache FFN output. Hit steps become near-zero compute. May affect quality on long schedules."}),
                "cache_dtype": (["fp16", "fp8", "mixed"], {"default": "fp16",
                    "tooltip": "fp16=all caches fp16. fp8=all fp8. mixed=tier-0/1 fp16 (precision), tier-2/3 fp8 (saves memory where approximation is already aggressive)."}),
                "cache_compress": (["raw", "vht2"], {"default": "raw",
                    "tooltip": "raw=store cached tensors as-is (default, safe). vht2=VHT2 spectral compression via pre-allocated memory pool (~3.5x memory reduction, ~0 quality loss). Pool eliminates the VRAM fragmentation that disabled VHT2 in Phase 15 LEAN."}),
                # ── Strange-attractor stack — gated, default OFF ─────────
                # Wires the rolling Fisher cos_sim into the cache-hit gate.
                # Currently the docstring promises this but the gate is age+streak only.
                # When enabled, each block tracks an EMA of measured cos_sim across
                # misses; the next hit is allowed only if rolling_sim >= tier threshold.
                # Tier thresholds default to the values implied by the existing streak
                # limits (granite>0.95, sand>0.90, jazz>0.85), so behavior is conservative.
                # Loosen granite to extend streaks; tighten jazz to refresh more eagerly.
                "enable_drift_gate": ("BOOLEAN", {"default": False,
                    "tooltip": "Strange-attractor stack: gate cache hits by rolling Fisher cos_sim. OFF = current age+streak only behavior (default, ship-safe)."}),
                "granite_threshold": ("FLOAT", {"default": 0.95, "min": 0.50, "max": 1.00, "step": 0.01,
                    "tooltip": "Min rolling cos_sim to allow cache hit on tier-0 (L00-L03). Lower = looser leash, longer streaks. Only used when enable_drift_gate=True."}),
                "sand_threshold": ("FLOAT", {"default": 0.90, "min": 0.50, "max": 1.00, "step": 0.01,
                    "tooltip": "Min rolling cos_sim for tier-1 (L04-L08). Only used when enable_drift_gate=True."}),
                "jazz_threshold": ("FLOAT", {"default": 0.85, "min": 0.50, "max": 1.00, "step": 0.01,
                    "tooltip": "Min rolling cos_sim for tier-2/3 (L09+). Only used when enable_drift_gate=True."}),
                # Piece 2/4: adaptive sigma-streak. At high sigma (composition forming)
                # streaks stay short; at low sigma (locked composition, texture only)
                # streaks extend. Cooperates with ShannonPrimeWanSigmaSwitch — reads
                # the same _sp_sigma_state if present, otherwise installs its own
                # model_function_wrapper to capture sigma. SVI non-monotonic schedules
                # are handled by sigma_max/sigma_min rolling min/max.
                "enable_sigma_streak": ("BOOLEAN", {"default": False,
                    "tooltip": "Strange-attractor stack: streak limits scale with sigma. Granite extends 7-15, sand 4-9, jazz 3-6 across sigma range. OFF = current static 10/5/3 behavior."}),
                # Piece 3/4: twin-prime borrowing on decode path. After Möbius reorder
                # and skeleton extraction, twin-prime spectral coefficients (3-5,
                # 11-13, 17-19, ...) are arithmetic neighbors. When dequantized
                # values disagree past a threshold, weighted-average pulls outliers
                # back toward consensus. Decode-only — encoded skeleton bytes are
                # untouched, reversibility property of compress() unchanged.
                # Worst case at α=0 = identity. Only applies when cache_compress=vht2.
                "enable_twin_borrow": ("BOOLEAN", {"default": False,
                    "tooltip": "Strange-attractor stack: twin-prime borrowing on VHT2 decode. 9 disjoint pairs at head_dim=128 (3-5, 11-13, 17-19, 29-31, ...). Only takes effect when cache_compress=vht2."}),
                "twin_alpha": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "Borrow strength. 0=no change, 0.5=full pair average. Bounded change: |Δc| ≤ α/2·|c_i-c_j|."}),
                "twin_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Only borrow when relative |c_i-c_j|/max exceeds this. 0.0=always borrow, 0.1=outliers only."}),
                "verbose": ("BOOLEAN", {"default": False,
                    "tooltip": "Print per-block HIT/MISS logs + Fisher cos_sim + Partition Z proxy"}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, *a, **k):
        return float("nan")

    def patch(self, model, tier_0_window=10, tier_1_window=3,
              tier_2_window=0, tier_3_window=0,
              cache_ffn=False, cache_dtype="fp16",
              cache_compress="raw",
              enable_drift_gate=False,
              granite_threshold=0.95, sand_threshold=0.90, jazz_threshold=0.85,
              enable_sigma_streak=False,
              enable_twin_borrow=False, twin_alpha=0.10, twin_threshold=0.0,
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
                            _obj.get("xattn_cache", {}).clear()
                            _obj.get("ffn_cache", {}).clear()
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
            if i < 16:
                return tier_2_window
            return tier_3_window

        _do_ffn = cache_ffn  # closure capture
        # Per-block dtype: mixed = fp16 for tier-0/1, fp8 for tier-2/3
        _fp16 = torch.float16
        _fp8 = torch.float8_e4m3fn
        if cache_dtype == "fp8":
            def _dtype_for(block_idx):
                return _fp8
            _use_fp8_any = True
        elif cache_dtype == "mixed":
            def _dtype_for(block_idx):
                return _fp16 if block_idx < 9 else _fp8
            _use_fp8_any = True
        else:  # fp16
            def _dtype_for(block_idx):
                return _fp16
            _use_fp8_any = False

        # Pre-compute Fisher diagonal weights for this model's head_dim
        _head_dim = blocks[0][1].self_attn.head_dim if blocks else 128
        _fisher_w = _fisher_diagonal_weights(_head_dim)

        # ── VHT2 memory pool (optional compression) ──────────────────────
        _use_vht2 = (cache_compress == "vht2")
        _vht2_bridge = None
        _vht2_pool = None
        _skel_mask = None
        if _use_vht2:
            try:
                from vht2_cuda_bridge import VHT2Bridge
                _vht2_bridge = VHT2Bridge(skeleton_frac=0.30, device="cpu")
                _skel_mask = _vht2_bridge.skeleton_mask_128()
                _skel_size = int(_skel_mask.sum().item())
                # Pool will be lazily initialized on first miss (need device)
                print(f"[SP BlockSkip] VHT2 compression enabled: "
                      f"skeleton={_skel_size}/128 ({_skel_size/128:.0%})")
            except ImportError:
                print("[SP BlockSkip] WARNING: vht2_cuda_bridge not available, "
                      "falling back to raw caching")
                _use_vht2 = False
        _n_blocks_total = n_blocks  # for partition Z summary

        # Shared state across all patched blocks
        state = {
            'global_step':    [0],
            'hit_streak':     {},
            'attn_cache':     {},       # block_idx -> self-attn y (CPU fp16)
            'xattn_cache':    {},       # block_idx -> cross-attn output (CPU fp16)
            'ffn_cache':      {},       # block_idx -> FFN pre-gate y (CPU fp16) [turbo]
            'step_cached':    {},
            'fisher_sim':     {},       # block_idx -> last Fisher cos_sim (diagnostic)
            'rolling_sim':    {},       # block_idx -> EMA of fisher cos_sim across misses
        }

        # ── Tier-aware drift threshold (strange-attractor stack) ─────────
        # Returns the minimum rolling cos_sim required to allow a cache hit.
        # Granite (L00-L03) cos_sim parks at 0.999+ across many steps, so it
        # tolerates a looser threshold; jazz (L16+) is volatile and warrants
        # a stricter one. Used only when enable_drift_gate=True.
        def _tier_threshold(i):
            if i < 4:
                return granite_threshold
            if i < 9:
                return sand_threshold
            return jazz_threshold

        # ── Sigma capture (cooperative with ShannonPrimeWanSigmaSwitch) ──
        # If SigmaSwitch is upstream it has already populated _sp_sigma_state
        # with current_sigma / sigma_max / sigma_min. Reuse it. Otherwise
        # install our own model_function_wrapper to capture the same shape.
        # Only active when enable_sigma_streak=True (sigma read happens per
        # step regardless, but streak_limit only scales when the toggle is on).
        sigma_state = getattr(patched, '_sp_sigma_state', None)
        if enable_sigma_streak and sigma_state is None:
            sigma_state = {
                'current_sigma': [None],
                'sigma_max':     [None],
                'sigma_min':     [None],
                'step_count':    [0],
            }

            def _sp_blockskip_sigma_wrapper(apply_model_func, args_dict):
                timestep = args_dict.get("timestep", None)
                if timestep is not None:
                    try:
                        sig = float(timestep.flatten()[0])
                        sigma_state['current_sigma'][0] = sig
                        sigma_state['step_count'][0] += 1
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

            patched.set_model_unet_function_wrapper(_sp_blockskip_sigma_wrapper)
            patched._sp_sigma_state = sigma_state

        # Adaptive streak: at high-sigma end keep short, at low-sigma end extend.
        # Returns int streak limit given a normalized sigma (1.0=high, 0.0=low).
        # Per-tier ranges: granite 7-15, sand 4-9, jazz 3-6.
        # Falls back to static 10/5/3 if sigma_state not yet populated.
        def _adaptive_streak(i, sigma_norm):
            if sigma_norm is None:
                if i < 4: return 10
                if i < 9: return 5
                return 3
            inv = 1.0 - max(0.0, min(1.0, sigma_norm))  # 0.0 at high, 1.0 at low
            if i < 4:
                return int(round(7 + 8 * inv))
            if i < 9:
                return int(round(4 + 5 * inv))
            return int(round(3 + 3 * inv))

        def make_patched_forward(orig_forward, block_idx, blk):
            window = get_window(block_idx)
            # Streak limit: force a miss every N hits to refresh cache.
            # Static defaults — overridden per-step when enable_sigma_streak=True.
            if block_idx < 4:
                _static_streak_limit = 10    # tier-0: sim>0.95
            elif block_idx < 9:
                _static_streak_limit = 5     # tier-1: sim>0.90
            else:
                _static_streak_limit = 3     # tier-2: volatile, refresh often

            def _current_streak_limit():
                if not enable_sigma_streak or sigma_state is None:
                    return _static_streak_limit
                sig = sigma_state['current_sigma'][0]
                s_max = sigma_state['sigma_max'][0]
                s_min = sigma_state['sigma_min'][0]
                if sig is None or s_max is None or s_min is None:
                    return _static_streak_limit
                rng = max(s_max - s_min, 1e-6)
                sig_norm = (sig - s_min) / rng  # 1.0=high, 0.0=low
                return _adaptive_streak(block_idx, sig_norm)

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
                    # x.dtype is the diffusion latent's compute dtype (Wan locks
                    # this to fp16 regardless of SP settings). The cache storage
                    # dtype (fp16/fp8/mixed) is the cache_dtype param and is
                    # printed in the summary line below.
                    print(f"[SP BlockSkip] step=1 x.shape={tuple(x.shape)}  "
                          f"tokens={x.shape[1]}  latent_dtype={x.dtype}  "
                          f"cache_dtype={cache_dtype}  device={x.device}")

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
                cached_xa = state['xattn_cache'].get(block_idx)
                cached_ff = state['ffn_cache'].get(block_idx) if _do_ffn else None

                # Shape validation — invalidate all caches together
                shape_ok = (cached_y is not None
                            and cached_y.shape[0] == x.shape[0]
                            and cached_y.shape[1] == x.shape[1])
                if not shape_ok and cached_y is not None:
                    state['attn_cache'].pop(block_idx, None)
                    state['xattn_cache'].pop(block_idx, None)
                    state['ffn_cache'].pop(block_idx, None)
                    state['step_cached'].pop(block_idx, None)
                    cached_y = None
                    cached_xa = None
                    cached_ff = None

                # ── Drift gate (strange-attractor stack) ────────────────────
                # When enabled, the rolling Fisher cos_sim must clear the
                # tier-specific threshold for a hit to be allowed. The rolling
                # value is seeded at 1.0 (full coherence), so the gate is a
                # no-op until the first miss measures actual drift.
                if enable_drift_gate:
                    rolling = state['rolling_sim'].get(block_idx, 1.0)
                    drift_ok = (rolling >= _tier_threshold(block_idx))
                else:
                    drift_ok = True

                # ── CACHE HIT ──────────────────────────────────────────────
                _streak_limit = _current_streak_limit()
                hit = (cached_y is not None
                       and cached_xa is not None
                       and age >= 0 and age < window
                       and streak < _streak_limit
                       and drift_ok)

                # ── Helpers: store/load with per-block dtype ──��───────────
                _blk_dtype = _dtype_for(block_idx)
                _is_fp8_blk = (_blk_dtype == _fp8)

                def _load(t):
                    # fp8: cast on CPU first (safe software path on Turing)
                    if _is_fp8_blk:
                        return t.to(dtype=x.dtype).to(device=x.device)
                    return t.to(device=x.device, dtype=x.dtype)

                def _store(t):
                    if _use_vht2 and _vht2_bridge is not None:
                        # VHT2 compress: butterfly → skeleton extract → CPU
                        # Uses pool for zero per-step allocations
                        try:
                            flat = t.detach().reshape(-1, _head_dim)
                            spectral = _vht2_bridge.forward(flat.float())
                            compressed = spectral[:, _skel_mask].to(dtype=_blk_dtype).cpu()
                            return compressed
                        except Exception:
                            pass  # fall through to raw
                    return t.detach().to(dtype=_blk_dtype).cpu()

                def _load_maybe_vht2(t):
                    """Load tensor, decompressing from VHT2 if compressed."""
                    if (_use_vht2 and _vht2_bridge is not None
                            and t.shape[-1] != _head_dim):
                        # Compressed: [N, skeleton_size] → decompress
                        try:
                            full = torch.zeros(t.shape[0], _head_dim,
                                               dtype=torch.float32)
                            full[:, _skel_mask] = t.float()
                            # Strange-attractor stack piece 3/4: twin-prime borrow
                            # before the inverse butterfly (decode-only).
                            if enable_twin_borrow:
                                full = _vht2_bridge.apply_twin_borrow(
                                    full, _skel_mask,
                                    alpha=twin_alpha, threshold=twin_threshold)
                            recon = _vht2_bridge.forward(full)  # inverse
                            return recon.to(device=x.device, dtype=x.dtype)
                        except Exception:
                            pass
                    return _load(t)

                if hit:
                    # Self-attn: cached pre-gate y + current step's gate
                    y = _load_maybe_vht2(cached_y)
                    x = torch.addcmul(x.contiguous(), y,
                                      repeat_e(e_mods[2], x))
                    del y

                    # Cross-attn: cached output is a plain residual add
                    xa = _load_maybe_vht2(cached_xa)
                    x = x + xa
                    del xa

                    state['hit_streak'][block_idx] = streak + 1

                    # ── TURBO: FFN also cached ───��─────────────────────────
                    if _do_ffn and cached_ff is not None:
                        yf = _load_maybe_vht2(cached_ff)
                        x = torch.addcmul(x, yf, repeat_e(e_mods[5], x))
                        del yf
                        if verbose:
                            print(f"[SP BlockSkip] B{block_idx:02d} HIT  "
                                  f"step={step} age={age}/{window} "
                                  f"streak={streak+1}/{_streak_limit} TURBO")
                        return x

                    if verbose:
                        print(f"[SP BlockSkip] B{block_idx:02d} HIT  "
                              f"step={step} age={age}/{window} "
                              f"streak={streak+1}/{_streak_limit}")
                else:
                    # ── CACHE MISS: run self-attention ────────────────────────
                    x = x.contiguous()
                    y = blk.self_attn(
                        torch.addcmul(repeat_e(e_mods[0], x),
                                      blk.norm1(x),
                                      1 + repeat_e(e_mods[1], x)),
                        freqs, transformer_options=transformer_options
                    )

                    # ── Fisher-weighted drift diagnostic ─────────────────────
                    # Compare fresh y against cached y (if exists) using the
                    # Fisher diagonal approximation. This measures "information
                    # drift" — drift along the arithmetical skeleton dimensions
                    # weighted higher than noise dimensions.
                    #
                    # When enable_drift_gate is on, this measurement is always-on
                    # (not just verbose) and feeds the EMA-smoothed rolling_sim
                    # used to gate the next step's hit decision. EMA α=0.5 gives
                    # ~2-step memory: enough to filter single-step blips but
                    # responsive enough to catch a real attractor escape.
                    prev_y = state['attn_cache'].get(block_idx)
                    if prev_y is not None and (verbose or enable_drift_gate):
                        prev_on_dev = _load(prev_y)
                        f_sim = _fisher_cos_sim(y, prev_on_dev, _fisher_w)
                        state['fisher_sim'][block_idx] = f_sim
                        if enable_drift_gate:
                            prev_rolling = state['rolling_sim'].get(block_idx, 1.0)
                            state['rolling_sim'][block_idx] = (
                                0.5 * prev_rolling + 0.5 * f_sim)
                        del prev_on_dev

                    state['attn_cache'][block_idx] = _store(y)

                    x = torch.addcmul(x, y, repeat_e(e_mods[2], x))
                    del y

                    # ── CACHE MISS: run cross-attention ──────────────────────
                    xa = blk.cross_attn(
                        blk.norm3(x), context,
                        context_img_len=context_img_len,
                        transformer_options=transformer_options
                    )
                    state['xattn_cache'][block_idx] = _store(xa)
                    state['step_cached'][block_idx] = step
                    state['hit_streak'][block_idx] = 0

                    x = x + xa
                    del xa

                    if verbose:
                        f_sim = state['fisher_sim'].get(block_idx)
                        f_str = f" fisher={f_sim:.4f}" if f_sim is not None else ""
                        print(f"[SP BlockSkip] B{block_idx:02d} MISS "
                              f"step={step}{f_str}")

                # ── attn2 patches (ControlNet etc) — always run ─────────────
                patches = transformer_options.get("patches", {})
                if "attn2_patch" in patches:
                    for p in patches["attn2_patch"]:
                        x = p({"x": x, "transformer_options": transformer_options})

                # ── FFN ──────────────────────────────────────────────────────
                y = blk.ffn(torch.addcmul(repeat_e(e_mods[3], x),
                                          blk.norm2(x),
                                          1 + repeat_e(e_mods[4], x)))
                if _do_ffn:
                    state['ffn_cache'][block_idx] = _store(y)
                x = torch.addcmul(x, y, repeat_e(e_mods[5], x))

                # ── Partition Z proxy logging ────────────────────────────────
                # The L2 norm of x post-FFN is a proxy for the partition
                # function: higher norm = more diffuse representation =
                # higher "information temperature." Log for future sentinel.
                if verbose:
                    z_proxy = float(x.float().norm() / max(x.numel(), 1) ** 0.5)
                    _log_partition_z(step, block_idx, z_proxy)
                    # Print summary after last block of each step
                    if block_idx == _n_blocks_total - 1 or (block_idx >= n_blocks - 1):
                        _print_partition_z_summary(step, _n_blocks_total)

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

        tier0 = [i for i, _ in blocks if i < 4 and get_window(i) > 0]
        tier1 = [i for i, _ in blocks if 4 <= i < 9 and get_window(i) > 0]
        tier2 = [i for i, _ in blocks if 9 <= i < 16 and get_window(i) > 0]
        tier3 = [i for i, _ in blocks if i >= 16 and get_window(i) > 0]
        mode = "TURBO" if cache_ffn else "self+cross"
        dtype_str = cache_dtype  # "fp16", "fp8", or "mixed"
        compress_str = "VHT2" if _use_vht2 else "raw"
        print(f"[SP BlockSkip] Phase 16 — {n_patched}/{n_blocks} blocks, {mode}, {dtype_str}, {compress_str}")
        if cache_dtype == "mixed":
            print(f"[SP BlockSkip] Mixed dtype: tier-0/1 fp16 (blocks <9), tier-2/3 fp8 (blocks >=9)")
        if _use_vht2:
            print(f"[SP BlockSkip] VHT2 compression: ~3.5x memory reduction, zero-alloc pool")
        if tier0:
            print(f"[SP BlockSkip] Tier-0 win={tier_0_window} streak=10: {tier0}")
        if tier1:
            print(f"[SP BlockSkip] Tier-1 win={tier_1_window} streak=5:  {tier1}")
        if tier2:
            print(f"[SP BlockSkip] Tier-2 win={tier_2_window} streak=3:  {tier2}")
        if tier3:
            print(f"[SP BlockSkip] Tier-3 win={tier_3_window} streak=3:  {tier3}")
        if cache_ffn:
            print(f"[SP BlockSkip] TURBO: on HIT, ENTIRE block skipped (adaLN + 3 transfers)")
        else:
            print(f"[SP BlockSkip] On HIT: adaLN + FFN. Self-attn + cross-attn SKIPPED.")
        if enable_drift_gate:
            print(f"[SP BlockSkip] Drift gate ON: granite>={granite_threshold:.2f} "
                  f"sand>={sand_threshold:.2f} jazz>={jazz_threshold:.2f} "
                  f"(rolling Fisher cos_sim, EMA α=0.5)")
        if enable_sigma_streak:
            owns_wrapper = sigma_state is not None and not hasattr(model, '_sp_sigma_state')
            src = "own wrapper" if owns_wrapper else "SigmaSwitch upstream"
            print(f"[SP BlockSkip] Sigma-streak ON: granite 7-15, sand 4-9, jazz 3-6 "
                  f"across sigma range (sigma source: {src})")
        if enable_twin_borrow:
            if _use_vht2:
                print(f"[SP BlockSkip] Twin-prime borrow ON: α={twin_alpha:.2f} "
                      f"threshold={twin_threshold:.2f} (decode-only, 9 disjoint pairs)")
            else:
                print(f"[SP BlockSkip] Twin-prime borrow requested but cache_compress=raw — no-op")
        if verbose:
            print(f"[SP BlockSkip] Verbose: Fisher cos_sim + Partition Z proxy logging enabled")

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
                        n = (len(obj["attn_cache"])
                             + len(obj.get("xattn_cache", {}))
                             + len(obj.get("ffn_cache", {})))
                        obj["attn_cache"].clear()
                        obj.get("xattn_cache", {}).clear()
                        obj.get("ffn_cache", {}).clear()
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
