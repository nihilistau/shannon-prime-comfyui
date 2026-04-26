"""
Strange-attractor stack ablation harness.

Submits a Wan workflow to ComfyUI repeatedly with different combinations of
the four toggles in ShannonPrimeWanBlockSkip + Voxtral KV cache, capturing
wall-clock per run and dumping a markdown table.

Inputs:
  --workflow <path>    API-format workflow JSON (ComfyUI menu: Save (API Format))
  --runs N             repeats per combo for variance estimation (default 1)
  --combos <preset>    which set of combos to run (default 'wan-core')
  --host HOST:PORT     ComfyUI address (default 127.0.0.1:8188)
  --output <md_path>   write markdown table here (default stdout only)
  --csv <csv_path>     also write CSV for plotting

Combo presets:
  wan-core      baseline / gate / streak / gate+streak  (4 runs)
  wan-vht2      same 4 but with cache_compress=vht2 + twin-borrow on the
                "all-on" leg (5 runs)
  wan-full      union of wan-core and wan-vht2 (8 runs total)
  voxtral-tail  baseline vs k_ternary_bands=[3] for Voxtral KV (2 runs)

The harness mutates the ShannonPrimeWanBlockSkip node's input dict in-place
on a deep copy of the workflow before submission. It does NOT validate that
the workflow contains such a node — if the workflow doesn't have one, the
toggles silently apply nothing and you'll see baseline numbers across the
board (which itself is a useful sanity check).

Output captures:
  - prompt_id, success/fail
  - wall_time (post-queue to status=success)
  - vram_peak_gb (queried from /system_stats post-run, best-effort)
  - any node_errors

Note: this measures total wall time, not per-step time. Step time is the
real signal for the strange-attractor stack but ComfyUI's API doesn't
expose it directly; you'll see it in ComfyUI's stdout. For a precise
per-step number, run with verbose=True on BlockSkip and parse the console.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


# ── Combo presets ────────────────────────────────────────────────────────

# Each combo is a dict: {param_name: value} applied to the BlockSkip node.
# Params not listed are left at the workflow's existing values.
WAN_CORE_COMBOS = [
    ("baseline",        {"enable_drift_gate": False, "enable_sigma_streak": False, "enable_twin_borrow": False}),
    ("gate-only",       {"enable_drift_gate": True,  "enable_sigma_streak": False, "enable_twin_borrow": False}),
    ("streak-only",     {"enable_drift_gate": False, "enable_sigma_streak": True,  "enable_twin_borrow": False}),
    ("gate+streak",     {"enable_drift_gate": True,  "enable_sigma_streak": True,  "enable_twin_borrow": False}),
]

WAN_VHT2_COMBOS = [
    ("vht2-baseline",   {"cache_compress": "vht2", "enable_drift_gate": False, "enable_sigma_streak": False, "enable_twin_borrow": False}),
    ("vht2-gate",       {"cache_compress": "vht2", "enable_drift_gate": True,  "enable_sigma_streak": False, "enable_twin_borrow": False}),
    ("vht2-streak",     {"cache_compress": "vht2", "enable_drift_gate": False, "enable_sigma_streak": True,  "enable_twin_borrow": False}),
    ("vht2-twin-only",  {"cache_compress": "vht2", "enable_drift_gate": False, "enable_sigma_streak": False, "enable_twin_borrow": True}),
    ("vht2-all-on",     {"cache_compress": "vht2", "enable_drift_gate": True,  "enable_sigma_streak": True,  "enable_twin_borrow": True}),
]

VOXTRAL_TAIL_COMBOS = [
    ("vox-baseline",    {"k_band_bits": "5,5,4,3"}),
    ("vox-ternary-tail",{"k_band_bits": "5,5,4,3", "k_ternary_bands": "3"}),
]

PRESETS = {
    "wan-core":     WAN_CORE_COMBOS,
    "wan-vht2":     WAN_VHT2_COMBOS,
    "wan-full":     WAN_CORE_COMBOS + WAN_VHT2_COMBOS,
    "voxtral-tail": VOXTRAL_TAIL_COMBOS,
}


# ── Target node class lookup ─────────────────────────────────────────────

# Map combo flavor → target node class_type. The harness finds nodes of
# this class in the workflow and patches their inputs.
COMBO_TARGET_NODE = {
    "wan-core":     "ShannonPrimeWanBlockSkip",
    "wan-vht2":     "ShannonPrimeWanBlockSkip",
    "wan-full":     "ShannonPrimeWanBlockSkip",
    "voxtral-tail": "ShannonPrimeVoxtralKVCache",
}


# ── ComfyUI client ───────────────────────────────────────────────────────

class ComfyClient:
    def __init__(self, host: str = "127.0.0.1:8188"):
        self.host = host
        self.base = f"http://{host}"

    def _get(self, path: str, timeout: float = 30.0):
        with urllib.request.urlopen(f"{self.base}{path}", timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def _post(self, path: str, data: dict, timeout: float = 30.0):
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(f"{self.base}{path}", data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def alive(self) -> bool:
        try:
            self._get("/system_stats")
            return True
        except Exception:
            return False

    def vram_used_gb(self):
        try:
            stats = self._get("/system_stats")
            dev = stats.get("devices", [{}])[0]
            total = dev.get("vram_total", 0)
            free = dev.get("vram_free", 0)
            return round((total - free) / 1e9, 2) if total else None
        except Exception:
            return None

    def queue(self, workflow: dict) -> str:
        try:
            r = self._post("/prompt", {"prompt": workflow,
                                       "client_id": "sp_ablation"})
            return r.get("prompt_id", "")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            try:
                err = json.loads(body)
                node_errors = err.get("node_errors", {})
                msg = err.get("error", {}).get("message", "")
                detail = []
                for nid, ne in list(node_errors.items())[:3]:
                    for it in ne.get("errors", [])[:2]:
                        detail.append(f"node {nid}: {it.get('message','')}")
                raise RuntimeError(f"{msg} | " + " | ".join(detail))
            except json.JSONDecodeError:
                raise RuntimeError(f"HTTP {e.code}: {body[:200]}")

    def wait(self, prompt_id: str, timeout: float = 1800.0,
             poll: float = 2.0) -> tuple[bool, float, dict | None]:
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                hist = self._get(f"/history/{prompt_id}")
                if prompt_id in hist:
                    entry = hist[prompt_id]
                    status = entry.get("status", {})
                    if status.get("completed", False) or status.get("status_str") == "success":
                        return True, time.time() - t0, entry
                    if status.get("status_str") == "error":
                        return False, time.time() - t0, entry
                queue = self._get("/queue")
                running = queue.get("queue_running", [])
                pending = queue.get("queue_pending", [])
                if not running and not pending:
                    # Race: completed between checks; one more history pull
                    hist = self._get(f"/history/{prompt_id}")
                    if prompt_id in hist:
                        return True, time.time() - t0, hist[prompt_id]
            except Exception:
                pass
            time.sleep(poll)
        return False, time.time() - t0, None


# ── Workflow patching ────────────────────────────────────────────────────

def find_nodes_by_class(workflow: dict, class_type: str) -> list[str]:
    """Return list of node IDs (string keys) whose class_type matches."""
    return [nid for nid, n in workflow.items()
            if isinstance(n, dict) and n.get("class_type") == class_type]


def patch_combo(workflow: dict, target_class: str,
                params: dict) -> tuple[dict, int]:
    """Deep-copy workflow and apply params to all nodes of target_class.

    Values are passed through verbatim. Specifically: csv-style band specs
    like "5,5,4,3" must stay as STRINGs because ComfyUI's STRING-typed inputs
    on the Voxtral KV cache reject Python lists (it interprets them as
    malformed link specs and validation fails with 'Bad linked input, must
    be a length-2 list of [node_id, slot_index]'). The node parses the
    string itself.

    Returns (patched, n_patched).
    """
    wf = copy.deepcopy(workflow)
    targets = find_nodes_by_class(wf, target_class)
    for nid in targets:
        inputs = wf[nid].setdefault("inputs", {})
        for k, v in params.items():
            inputs[k] = v
    return wf, len(targets)


# ── Runner ───────────────────────────────────────────────────────────────

def run_combo(client: ComfyClient, workflow: dict, target_class: str,
              name: str, params: dict, runs: int) -> dict:
    """Run one combo `runs` times, return aggregated row."""
    wf_patched, n_targets = patch_combo(workflow, target_class, params)
    times = []
    statuses = []
    last_error = ""
    vram_peak = 0.0

    for r in range(runs):
        try:
            pid = client.queue(wf_patched)
            ok, elapsed, _entry = client.wait(pid)
            v = client.vram_used_gb() or 0.0
            vram_peak = max(vram_peak, v)
            times.append(elapsed)
            statuses.append("ok" if ok else "fail")
            if not ok:
                last_error = "wait_timeout_or_error"
        except Exception as e:
            times.append(0.0)
            statuses.append("err")
            last_error = str(e)[:120]

    succ_times = [t for t, s in zip(times, statuses) if s == "ok"]
    return {
        "name": name,
        "params": params,
        "n_target_nodes": n_targets,
        "runs_ok": sum(1 for s in statuses if s == "ok"),
        "runs_total": runs,
        "mean_s": statistics.mean(succ_times) if succ_times else None,
        "stdev_s": statistics.stdev(succ_times) if len(succ_times) >= 2 else None,
        "min_s": min(succ_times) if succ_times else None,
        "max_s": max(succ_times) if succ_times else None,
        "vram_peak_gb": vram_peak if vram_peak > 0 else None,
        "error": last_error,
    }


# ── Reporting ────────────────────────────────────────────────────────────

def fmt_seconds(x) -> str:
    if x is None:
        return "–"
    return f"{x:.1f}"


def fmt_speedup(baseline_s, this_s) -> str:
    if baseline_s is None or this_s is None or this_s == 0:
        return "–"
    return f"{baseline_s / this_s:.2f}x"


def render_markdown(rows: list[dict], preset: str) -> str:
    if not rows:
        return "_(no rows)_\n"
    baseline_s = rows[0].get("mean_s")
    out = []
    out.append(f"# Strange-attractor stack ablation — preset `{preset}`\n")
    out.append(f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}_\n")
    out.append("| combo | runs | mean (s) | ±stdev | speedup vs baseline | VRAM peak | params |")
    out.append("|---|---|---|---|---|---|---|")
    for r in rows:
        params_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
        out.append(
            "| {name} | {ok}/{tot} | {mean} | {stdev} | {speedup} | {vram} | {params} |".format(
                name=r["name"],
                ok=r["runs_ok"], tot=r["runs_total"],
                mean=fmt_seconds(r["mean_s"]),
                stdev=("±" + fmt_seconds(r["stdev_s"])) if r["stdev_s"] is not None else "–",
                speedup=fmt_speedup(baseline_s, r["mean_s"]),
                vram=(f"{r['vram_peak_gb']:.2f} GB" if r["vram_peak_gb"] else "–"),
                params=params_str,
            ))
    failures = [r for r in rows if r["error"]]
    if failures:
        out.append("\n## Failures\n")
        for r in failures:
            out.append(f"- **{r['name']}**: {r['error']}")
    return "\n".join(out) + "\n"


def write_csv(rows: list[dict], path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "runs_ok", "runs_total", "mean_s", "stdev_s",
                    "min_s", "max_s", "vram_peak_gb", "error", "params"])
        for r in rows:
            w.writerow([
                r["name"], r["runs_ok"], r["runs_total"],
                r["mean_s"] or "", r["stdev_s"] or "",
                r["min_s"] or "", r["max_s"] or "",
                r["vram_peak_gb"] or "", r["error"],
                json.dumps(r["params"]),
            ])


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workflow",
                   help="API-format workflow JSON (ComfyUI menu: Save (API Format)). Required unless --list-only.")
    p.add_argument("--runs", type=int, default=1,
                   help="Repeats per combo for variance (default 1)")
    p.add_argument("--combos", default="wan-core", choices=list(PRESETS.keys()),
                   help="Combo preset (default wan-core)")
    p.add_argument("--host", default="127.0.0.1:8188", help="ComfyUI address")
    p.add_argument("--output", help="Markdown output path")
    p.add_argument("--csv", help="CSV output path")
    p.add_argument("--list-only", action="store_true",
                   help="Print combos and exit without running")
    args = p.parse_args()

    combos = PRESETS[args.combos]
    target_class = COMBO_TARGET_NODE[args.combos]

    if args.list_only:
        print(f"Preset: {args.combos}  target: {target_class}")
        for name, params in combos:
            print(f"  {name}: {params}")
        return 0

    if not args.workflow:
        p.error("--workflow is required unless --list-only")
    workflow_path = Path(args.workflow)
    if not workflow_path.exists():
        print(f"ERROR: workflow not found: {workflow_path}", file=sys.stderr)
        return 2

    with open(workflow_path, "r", encoding="utf-8") as f:
        workflow = json.load(f)

    # Quick sanity: count target nodes in the workflow as-is
    targets = find_nodes_by_class(workflow, target_class)
    if not targets:
        print(f"WARNING: no '{target_class}' nodes found in workflow. "
              f"Toggles will be no-ops; baseline numbers across all combos.",
              file=sys.stderr)
    else:
        print(f"Found {len(targets)} '{target_class}' node(s): {targets}",
              file=sys.stderr)

    client = ComfyClient(args.host)
    if not client.alive():
        print(f"ERROR: ComfyUI not reachable at {args.host}", file=sys.stderr)
        return 3
    print(f"ComfyUI alive at {args.host}", file=sys.stderr)

    rows = []
    for i, (name, params) in enumerate(combos):
        print(f"\n[{i+1}/{len(combos)}] {name} {params}", file=sys.stderr)
        row = run_combo(client, workflow, target_class, name, params, args.runs)
        rows.append(row)
        print(f"  → {row['runs_ok']}/{row['runs_total']} ok, "
              f"mean={fmt_seconds(row['mean_s'])}s "
              f"vram={row['vram_peak_gb'] or '–'}GB",
              file=sys.stderr)

    md = render_markdown(rows, args.combos)
    print(md)
    if args.output:
        Path(args.output).write_text(md, encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)
    if args.csv:
        write_csv(rows, Path(args.csv))
        print(f"Wrote {args.csv}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
