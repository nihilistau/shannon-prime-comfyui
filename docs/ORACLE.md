# Shannon-Prime ComfyUI — Mertens Oracle Reference

The Mertens Oracle is ShannonPrimeWanBlockSkip's adaptive caching engine. It decides — per block, per step — whether to serve a cached self-attention output or recompute fresh. It requires no sigma schedule knowledge, no prompt metadata, and no external signal beyond the attention tensors themselves.

---

## What the Oracle tracks

For each patched WanAttentionBlock (L00-L08), the oracle maintains:

- **rolling_sim** — exponential moving average (EMA α=0.7) of the cosine similarity between the current block's freshly computed `y` and the previously cached `y`
- **effective_win** — the current effective cache window for that block (may be lower than the nominal window if drift is detected)
- **hit_streak** — count of consecutive cache hits since the last oracle-refresh miss
- **x_norm** — per-token L2 norm of the input `x` tensor at the last cache-store step (used for x-drift checking)

These are per-block values. Blocks are independent — one block's oracle does not affect another's.

---

## Block tiers

Block tier assignment is fixed at patch time based on Phase 12 sigma-sweep diagnostics (cos_sim measured across 10-step windows at all sigma levels).

| Tier | Blocks | Nominal window | Stability observed |
|------|--------|----------------|-------------------|
| 0 — Permanent Granite | L00-L03 | 10 steps | cos_sim > 0.95 for 10 steps at all sigma levels |
| 1 — Stable Sand | L04-L08 | 3 steps | cos_sim ~0.83-0.90 at mid-sigma; drops sharply at low sigma |
| uncached | L09-L29 | 0 | cos_sim < 0.75 at 10 steps; no cache |

The "Permanent Granite" name reflects that these blocks anchor global composition. They encode the spatial layout and subject positioning that is established in the first few steps and remains geometrically stable throughout denoising. Their `y` vectors change very slowly — the oracle's measured rolling_sim for L00-L03 is typically 0.980-0.999.

The "Stable Sand" name reflects moderate stability — these blocks refine intermediate features and are stable enough to cache conservatively, but respond to sigma-regime changes. Their rolling_sim is typically 0.900-0.960 at high sigma, dropping toward 0.700 at low sigma on complex prompts.

---

## Adaptive streak (oracle-refresh misses)

The oracle must occasionally compute a fresh `y` even during a cache-hit window, to update its rolling_sim estimate and detect any shift in block stability. This is the "streak-miss" mechanism.

After N consecutive cache hits, the oracle forces a recompute (miss). The value of N is determined by the current `rolling_sim`:

| rolling_sim | Max streak before forced miss |
|-------------|-------------------------------|
| > 0.95 (proven Granite) | 10 hits |
| > 0.90 (stable) | 7 hits |
| > 0.85 (drifting) | 5 hits (default regime) |
| <= 0.85 (volatile) | 3 hits |

This is self-calibrating: as the oracle observes higher-sim `y` vectors, it reduces the frequency of forced recomputes. As stability degrades, forced misses become more frequent. No external parameter controls the streak limit — it emerges from the oracle's own confidence.

After a streak-miss, `hit_streak` is reset to 0 and the oracle gets a fresh `sim` measurement for its EMA update.

---

## Window adaptation (drift-triggered halving)

When `rolling_sim` falls below `drift_threshold` (default 0.85), the oracle halves `effective_win`:

```
effective_win = max(1, nominal_win // 2)
```

When `rolling_sim` recovers above 0.92, the nominal window is restored:

```
effective_win = nominal_win
```

This provides hysteresis — there is a gap between the halving threshold (0.85) and the restore threshold (0.92). Blocks that are genuinely drifting stabilize below `effective_win=1` (mandatory recompute every step) without thrashing between halved and full windows.

---

## x_drift sentinel (per-block tightening)

The x-drift check is a secondary guard against sudden structural changes in the input latent `x`. It operates independently of `rolling_sim`.

**How it works:**

The oracle stores only the per-token L2 norm of `x` at the last cache-store step (~43KB per block at 720p, vs 66MB for the full tensor). On each call to a cached block, the mean fractional change in these norms is computed:

```
drift = mean(|norm(x_cur) - norm(x_ref)|) / mean(norm(x_ref))
```

If `drift > x_drift_threshold`, a forced miss is issued and `effective_win` is halved.

**Per-tier thresholds:**

- **Tier-0 (L00-L03):** `x_drift_t0=0.30` (default). Catches "floor drop" events (sudden global latent jumps) without triggering on normal per-step variation, which runs 0.10-0.25. Can be set to 0.0 to disable — Granite blocks are stable enough that the rolling oracle handles them without x-drift.
- **Tier-1 (L04-L08):** `x_drift_t1=0.25` with linear tightening across the tier:
  - L04 uses `t1` (most stable in Tier-1, near Granite)
  - L08 uses `t1 × 0.80` (most volatile in Tier-1, near the uncached boundary)
  - Intermediate blocks: `t1 × (1 - 0.20 × (block_idx - 4) / 4)`

The x-norm proxy approach eliminates the 142-second VAE decode overhead that occurred when full `x` tensors (66MB each × 9 blocks = 594MB) were held in CPU memory through the decode window.

---

## CFG effect on oracle behavior

At CFG=1.0, the conditional and unconditional batches are processed identically without classifier-free guidance. This causes blocks L04-L08 to show `rolling_sim` near 0.000 — they are completely unstable at CFG=1.

The oracle correctly detects this: after the first generation with CFG=1, `effective_win` for L04-L08 collapses to 1 (mandatory recompute every step). Tier-0 blocks (L00-L03) remain stable at CFG=1 — they show `sim > 0.95` regardless of CFG setting.

This is correct behavior, not a bug. At CFG=1, only Tier-0 blocks provide savings.

---

## What to look for in verbose logs

Enable `verbose=True` in ShannonPrimeWanBlockSkip to see per-step oracle output.

**Normal Granite block (L00-L03):**
```
[SP BlockSkip] B00 MISS  step=1  sim=1.000 roll=1.000 win=10
[SP BlockSkip] B00 HIT   step=2  age=1/10  streak=1   sim=1.000
[SP BlockSkip] B00 HIT   step=3  age=2/10  streak=2   sim=1.000
...
[SP BlockSkip] B00 STREAK-MISS step=12 streak=10>=10 (sim=0.990) -> oracle refresh
[SP BlockSkip] B00 MISS  step=12 sim=0.991 roll=0.992 win=10
[SP BlockSkip] B00 HIT   step=13 age=1/10  streak=1   sim=0.992
```

- The first step is always a miss (nothing cached yet)
- HIT lines dominate for Granite blocks
- STREAK-MISS fires exactly at streak=10 (or whatever adaptive limit is active)
- After STREAK-MISS, sim stays high — window remains at nominal

**Stable Sand block (L05) at high sigma:**
```
[SP BlockSkip] B05 MISS  step=1  sim=1.000 roll=1.000 win=3
[SP BlockSkip] B05 HIT   step=2  age=1/3   streak=1   sim=1.000
[SP BlockSkip] B05 HIT   step=3  age=2/3   streak=2   sim=1.000
[SP BlockSkip] B05 MISS  step=4  sim=0.934 roll=0.979 win=3
```

- Window=3 means miss every 4th step (steps 1, 4, 8, ...)
- sim stays above 0.90, window stays at nominal

**Stable Sand block at low sigma or with CFG=1:**
```
[SP BlockSkip] B05 MISS  step=14 sim=0.723 roll=0.831 win=3
[SP BlockSkip] B05 MISS  step=15 sim=0.698 roll=0.797 win=1
[SP BlockSkip] B05 x-DRIFT step=16 drift=0.38>0.25 -> forced miss, win halved to 1
```

- rolling_sim drops below drift_threshold (0.85) → window halves from 3 to 1
- x-drift guard fires independently → window halves again (stays at 1, max(1,...))
- Block is now recomputing every step, which is correct

**Generation boundary detection:**
```
[SP BlockSkip] New generation [e-increase(0.234->0.871)] — clearing cache
```
or:
```
[SP BlockSkip] New generation [t_gap=67.3s] — clearing cache
```

The primary detector is e-magnitude monotonicity (sigma proxy). Any increase in `e_mag` > 2% means sigma reset → new generation. The wall-clock gap fires as a secondary for cases where the first step of a new generation happens to have the same sigma as the last step of the previous one (rare but possible).

---

## Oracle summary per generation (RicciSentinel)

Add `ShannonPrimeWanRicciSentinel` after BlockSkip to get a summary table at each generation boundary:

```
[SP Ricci Sentinel] Generation summary (20 steps):
  Step    e_mag  regime  win[0]  win[4]  roll_sim[0]
  ------------------------------------------------------
     1    0.871  HIGH        10       3        1.000
     2    0.853  HIGH        10       3        0.998
     3    0.832  HIGH        10       3        0.996
  ...
     9    0.704  HIGH        10       3        0.989
     10   0.689  HIGH        10       3        0.987
     --- HIGH->LOW ---
     11   0.601  LOW          5       1        0.941
  ...
     20   0.221  LOW          5       1        0.893

  Regime switch at step 11 (split@50% of e_mag range)
```

- `win[0]`: effective window for block L00 (Tier-0 representative)
- `win[4]`: effective window for block L04 (Tier-1 representative)
- `roll_sim[0]`: oracle's rolling cosine similarity for block L00
- HIGH/LOW: sigma regime based on e_mag vs the split threshold (when SigmaSwitch is active; otherwise based on e_mag heuristic)

The regime switch step is where SigmaSwitch (when active) would contract Tier-1 windows. Currently with SigmaSwitch bypassed, this table still correctly shows rolling_sim evolution and the oracle's window decisions.
