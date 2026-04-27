"""
Shannon-Prime Control Panel — minimal stdlib web UI for ComfyUI's API.

Runs a tiny HTTP server (default port 8500) that:
  - serves a single-page HTML dashboard with live VRAM, queue status, GPU info
  - proxies the most useful ComfyUI endpoints (/system_stats, /queue, /free,
    /interrupt, /history, /prompt) so the page can drive them without CORS
    hassles
  - lists workflows on disk, applies one-click v2 toggle presets to them, and
    queues the patched workflow with a single button
  - tails ComfyUI's stdout log for live debugging

Stdlib only. No FastAPI, no Flask, no pip install. Runs anywhere Python 3.10+
is installed.

Usage:
    python scripts/sp_control_panel.py
    python scripts/sp_control_panel.py --port 8500 \
        --comfy-host 127.0.0.1:8189 \
        --workflows-dir C:\\Projects\\the_system_itself\\comfyui-bench\\workflows \
        --comfy-log "C:\\Projects\\the_system_itself\\comfyui-bench\\comfyui_launch.log"

Then open http://127.0.0.1:8500 in a browser.

Designed to be paired with launch_with_control.ps1, which boots ComfyUI on
8189 and the control panel on 8500 in one shot.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


# ── Toggle presets (mirror the README "Strange-Attractor Stack" recipes) ─

PRESETS: dict[str, dict] = {
    "off": {
        "enable_drift_gate":          False,
        "enable_sigma_streak":        False,
        "enable_twin_borrow":         False,
        "enable_harmonic_correction": False,
        "enable_tier_skeleton":       False,
        "enable_curvature_gate":      False,
        "enable_cauchy_reset":        False,
    },
    "conservative": {
        "enable_drift_gate":          True,
        "enable_sigma_streak":        True,
        "enable_twin_borrow":         False,
        "enable_harmonic_correction": False,
        "enable_tier_skeleton":       False,
        "enable_curvature_gate":      False,
        "enable_cauchy_reset":        False,
    },
    "standard": {
        "enable_drift_gate":          True,
        "enable_sigma_streak":        True,
        "enable_twin_borrow":         True,
        "twin_borrow_mode":           "symmetric",
        "twin_alpha":                 0.10,
        "enable_harmonic_correction": True,
        "harmonic_strength":          0.50,
        "enable_tier_skeleton":       True,
        "granite_skel_frac":          0.50,
        "sand_skel_frac":             0.30,
        "jazz_skel_frac":             0.20,
        "enable_curvature_gate":      False,
        "enable_cauchy_reset":        False,
    },
    "aggressive": {
        "enable_drift_gate":          True,
        "enable_sigma_streak":        True,
        "enable_twin_borrow":         True,
        "twin_borrow_mode":           "symmetric",
        "twin_alpha":                 0.10,
        "enable_harmonic_correction": True,
        "harmonic_strength":          0.50,
        "enable_tier_skeleton":       True,
        "granite_skel_frac":          0.50,
        "sand_skel_frac":             0.30,
        "jazz_skel_frac":             0.20,
        "enable_curvature_gate":      True,
        "curvature_threshold":        -0.05,
        "enable_cauchy_reset":        True,
        "cauchy_radius":              2,
    },
    "maximum": {
        # The "chef's kiss" — every v2 toggle on, harmonic strength bumped
        # to 0.7 (validated safe by the operator on a 2060), granite skel
        # at 0.65 to combat the cardboard-cutout effect.
        "enable_drift_gate":          True,
        "enable_sigma_streak":        True,
        "enable_twin_borrow":         True,
        "twin_borrow_mode":           "low_anchor",
        "twin_alpha":                 0.10,
        "enable_harmonic_correction": True,
        "harmonic_strength":          0.70,
        "enable_tier_skeleton":       True,
        "granite_skel_frac":          0.65,
        "sand_skel_frac":             0.35,
        "jazz_skel_frac":             0.25,
        "enable_curvature_gate":      True,
        "curvature_threshold":        -0.05,
        "enable_cauchy_reset":        True,
        "cauchy_radius":              2,
    },
}


# ── HTML page ────────────────────────────────────────────────────────────

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Shannon-Prime Control Panel</title>
<style>
  :root {
    --bg: #0d1117;
    --panel: #161b22;
    --border: #30363d;
    --text: #c9d1d9;
    --text-dim: #8b949e;
    --accent: #58a6ff;
    --good: #3fb950;
    --warn: #d29922;
    --bad: #f85149;
    --bar-bg: #21262d;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    padding: 24px; max-width: 980px; margin: 0 auto;
  }
  h1 { font-size: 20px; font-weight: 600; margin-bottom: 4px; color: var(--accent); }
  .subtitle { color: var(--text-dim); font-size: 13px; margin-bottom: 24px; }
  .card {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 8px; padding: 18px; margin-bottom: 16px;
  }
  .card-title {
    font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;
    color: var(--text-dim); margin-bottom: 12px;
  }
  .stat-row {
    display: flex; justify-content: space-between; align-items: baseline;
    padding: 6px 0;
  }
  .stat-label { color: var(--text-dim); font-size: 13px; }
  .stat-value { font-family: ui-monospace, "SF Mono", Consolas, monospace; font-size: 14px; }
  .vram-bar {
    height: 24px; background: var(--bar-bg); border-radius: 4px;
    overflow: hidden; margin: 8px 0; position: relative;
  }
  .vram-fill { height: 100%; background: var(--accent); transition: width 0.3s, background 0.3s; }
  .vram-label {
    position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
    font-family: ui-monospace, monospace; font-size: 12px; color: var(--text);
    text-shadow: 0 0 4px rgba(0,0,0,0.7);
  }
  .vram-fill.warn { background: var(--warn); }
  .vram-fill.bad  { background: var(--bad); }
  button, select {
    background: var(--panel); color: var(--text); border: 1px solid var(--border);
    padding: 9px 14px; font-size: 13px; border-radius: 6px; cursor: pointer;
    margin-right: 8px; margin-bottom: 8px;
    transition: background 0.15s, border-color 0.15s;
    font-family: inherit;
  }
  button:hover, select:hover  { background: var(--border); }
  button.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
  button.primary:hover { background: #4493e0; }
  button.danger  { background: var(--bad);    color: #fff; border-color: var(--bad); }
  button.danger:hover  { background: #d4453f; }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  select { padding: 8px 10px; min-width: 220px; }
  .preset-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
  .preset-btn { padding: 6px 12px; font-size: 12px; opacity: 0.85; }
  .preset-btn.active {
    background: var(--accent); border-color: var(--accent); color: #fff; opacity: 1;
  }
  .preset-desc {
    font-size: 11px; color: var(--text-dim); margin-top: 6px;
    font-family: ui-monospace, monospace; min-height: 14px;
  }
  .status-dot {
    display: inline-block; width: 10px; height: 10px; border-radius: 50%;
    margin-right: 8px; vertical-align: middle;
  }
  .status-dot.ok  { background: var(--good); }
  .status-dot.bad { background: var(--bad);  box-shadow: 0 0 6px var(--bad); }
  .log {
    font-family: ui-monospace, monospace; font-size: 11px; color: var(--text-dim);
    background: var(--bg); border: 1px solid var(--border); border-radius: 4px;
    padding: 10px; max-height: 220px; overflow-y: auto; margin-top: 12px;
    white-space: pre-wrap; word-break: break-all;
  }
  .log-line { margin: 1px 0; }
  .log-line.ok   { color: var(--good); }
  .log-line.warn { color: var(--warn); }
  .log-line.bad  { color: var(--bad);  }
  .row { display: flex; gap: 16px; flex-wrap: wrap; }
  .row > .card { flex: 1; min-width: 280px; }
  .history-row {
    display: flex; justify-content: space-between; padding: 6px 0;
    border-bottom: 1px solid var(--border); font-size: 12px;
  }
  .history-row:last-child { border-bottom: none; }
  .history-id   { font-family: ui-monospace, monospace; color: var(--text-dim); }
  .history-stat.ok   { color: var(--good); }
  .history-stat.warn { color: var(--warn); }
  .history-stat.bad  { color: var(--bad); }
  .footer { color: var(--text-dim); font-size: 11px; margin-top: 24px; text-align: center; }
</style>
</head>
<body>
  <h1>Shannon-Prime Control Panel</h1>
  <div class="subtitle">
    <span id="conn-dot" class="status-dot bad"></span>
    <span id="conn-text">connecting…</span>
    &nbsp;·&nbsp; ComfyUI on <span id="comfy-host">—</span>
  </div>

  <div class="card">
    <div class="card-title">VRAM</div>
    <div class="vram-bar">
      <div id="vram-fill" class="vram-fill" style="width: 0%"></div>
      <div id="vram-label" class="vram-label">— / — GB</div>
    </div>
    <div class="stat-row"><span class="stat-label">Device</span><span class="stat-value" id="device-name">—</span></div>
    <div class="stat-row"><span class="stat-label">Used</span> <span class="stat-value" id="vram-used">—</span></div>
    <div class="stat-row"><span class="stat-label">Free</span> <span class="stat-value" id="vram-free">—</span></div>
    <div class="stat-row"><span class="stat-label">Total</span><span class="stat-value" id="vram-total">—</span></div>
  </div>

  <div class="row">
    <div class="card">
      <div class="card-title">Queue</div>
      <div class="stat-row"><span class="stat-label">Running</span> <span class="stat-value" id="q-running">—</span></div>
      <div class="stat-row"><span class="stat-label">Pending</span> <span class="stat-value" id="q-pending">—</span></div>
    </div>
    <div class="card">
      <div class="card-title">System</div>
      <div class="stat-row"><span class="stat-label">RAM total</span> <span class="stat-value" id="ram-total">—</span></div>
      <div class="stat-row"><span class="stat-label">RAM free</span>  <span class="stat-value" id="ram-free">—</span></div>
      <div class="stat-row"><span class="stat-label">PyTorch</span>   <span class="stat-value" id="torch-ver">—</span></div>
    </div>
  </div>

  <div class="card">
    <div class="card-title">Run a workflow with a preset</div>
    <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;">
      <select id="wf-select"><option value="">— loading workflows —</option></select>
      <button id="btn-run" class="primary">Run</button>
      <span id="run-status" style="font-size:12px;color:var(--text-dim);"></span>
    </div>
    <div class="preset-row" id="preset-row"><!-- populated from /api/presets --></div>
    <div class="preset-desc" id="preset-desc">— pick a preset —</div>
  </div>

  <div class="card">
    <div class="card-title">Actions</div>
    <button id="btn-free" class="primary">Free VRAM (unload models)</button>
    <button id="btn-interrupt" class="danger">Interrupt running prompt</button>
    <button id="btn-refresh">Refresh</button>
    <div id="log" class="log"><div class="log-line">ready.</div></div>
  </div>

  <div class="row">
    <div class="card">
      <div class="card-title">Recent history</div>
      <div id="history" style="font-size:12px;">
        <div class="stat-label">no history yet</div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">ComfyUI log (tail)</div>
      <div id="log-tail" class="log" style="margin-top:0;max-height:240px;">
        — log not configured —
      </div>
    </div>
  </div>

  <div class="footer">
    Shannon-Prime Control Panel · stdlib http.server · proxies <code>/system_stats</code>, <code>/queue</code>, <code>/free</code>, <code>/interrupt</code>, <code>/history</code>, <code>/prompt</code>
  </div>

<script>
  const $ = (id) => document.getElementById(id);
  const fmt = (gb) => gb == null ? "—" : gb.toFixed(2) + " GB";
  let SELECTED_PRESET = "standard";

  function logLine(text, kind) {
    const el = document.createElement("div");
    el.className = "log-line" + (kind ? " " + kind : "");
    el.textContent = "[" + new Date().toLocaleTimeString() + "] " + text;
    $("log").prepend(el);
    while ($("log").children.length > 30) $("log").lastChild.remove();
  }

  async function fetchJSON(path, opts) {
    const r = await fetch(path, opts || {});
    if (!r.ok) throw new Error(path + " → HTTP " + r.status);
    return r.json();
  }

  async function refresh() {
    try {
      const stats = await fetchJSON("/api/stats");
      $("conn-dot").className = "status-dot ok";
      $("conn-text").textContent = "online";
      $("comfy-host").textContent = stats.comfy_host || "—";

      const dev = (stats.devices || [])[0] || {};
      const total = (dev.vram_total || 0) / 1e9;
      const free  = (dev.vram_free  || 0) / 1e9;
      const used  = total - free;
      const pct   = total > 0 ? (used / total) * 100 : 0;

      $("device-name").textContent = dev.name || "—";
      $("vram-used").textContent  = fmt(used);
      $("vram-free").textContent  = fmt(free);
      $("vram-total").textContent = fmt(total);
      $("vram-fill").style.width = pct.toFixed(1) + "%";
      $("vram-fill").className = "vram-fill" +
        (pct > 90 ? " bad" : pct > 75 ? " warn" : "");
      $("vram-label").textContent = fmt(used) + " / " + fmt(total) +
        "  (" + pct.toFixed(1) + "%)";

      const sysmem = stats.system || {};
      const rtot = (sysmem.ram_total || 0) / 1e9;
      const rfree = (sysmem.ram_free || 0) / 1e9;
      $("ram-total").textContent = fmt(rtot);
      $("ram-free").textContent  = fmt(rfree);
      $("torch-ver").textContent = (sysmem.pytorch_version || "—").split("+")[0];

      const q = await fetchJSON("/api/queue");
      $("q-running").textContent = (q.queue_running || []).length;
      $("q-pending").textContent = (q.queue_pending || []).length;
    } catch (e) {
      $("conn-dot").className = "status-dot bad";
      $("conn-text").textContent = "offline · " + (e.message || e);
    }
  }

  async function loadWorkflows() {
    try {
      const r = await fetchJSON("/api/workflows");
      const sel = $("wf-select");
      sel.innerHTML = "";
      if (!r.workflows || r.workflows.length === 0) {
        sel.innerHTML = "<option value=''>— no workflows in " + (r.dir || "?") + " —</option>";
        return;
      }
      r.workflows.forEach(name => {
        const opt = document.createElement("option");
        opt.value = name; opt.textContent = name;
        sel.appendChild(opt);
      });
    } catch (e) {
      logLine("workflow list failed: " + e.message, "bad");
    }
  }

  async function loadPresets() {
    try {
      const r = await fetchJSON("/api/presets");
      const row = $("preset-row");
      row.innerHTML = "";
      const names = Object.keys(r.presets || {});
      names.forEach(name => {
        const btn = document.createElement("button");
        btn.className = "preset-btn" + (name === SELECTED_PRESET ? " active" : "");
        btn.textContent = name;
        btn.dataset.preset = name;
        btn.dataset.summary = r.presets[name].summary || "";
        btn.addEventListener("click", () => {
          SELECTED_PRESET = name;
          [...row.children].forEach(b => b.classList.remove("active"));
          btn.classList.add("active");
          $("preset-desc").textContent = btn.dataset.summary;
        });
        row.appendChild(btn);
      });
      const active = row.querySelector(".active");
      if (active) $("preset-desc").textContent = active.dataset.summary;
    } catch (e) {
      logLine("preset list failed: " + e.message, "bad");
    }
  }

  async function loadHistory() {
    try {
      const r = await fetchJSON("/api/history?limit=6");
      const div = $("history");
      div.innerHTML = "";
      const items = r.items || [];
      if (items.length === 0) {
        div.innerHTML = "<div class='stat-label'>no history yet</div>";
        return;
      }
      items.forEach(h => {
        const row = document.createElement("div");
        row.className = "history-row";
        const klass = h.status === "success" ? "ok"
                    : h.status === "error"   ? "bad"
                    :                          "warn";
        row.innerHTML =
          "<span class='history-id'>" + h.id.slice(0, 8) + "</span>" +
          "<span class='history-stat " + klass + "'>" + h.status + "</span>";
        div.appendChild(row);
      });
    } catch (e) { /* silent */ }
  }

  async function loadLogTail() {
    try {
      const r = await fetchJSON("/api/log-tail?lines=80");
      const el = $("log-tail");
      if (!r.available) {
        el.textContent = "— log not configured (use --comfy-log to enable) —";
        return;
      }
      el.textContent = (r.text || "").trim() || "— log empty —";
      el.scrollTop = el.scrollHeight;
    } catch (e) { /* silent */ }
  }

  $("btn-free").addEventListener("click", async () => {
    $("btn-free").disabled = true;
    try {
      logLine("requesting /free …");
      const r = await fetch("/api/free", { method: "POST" });
      if (!r.ok) throw new Error("HTTP " + r.status);
      logLine("models unloaded, VRAM freed.", "ok");
      setTimeout(refresh, 800);
    } catch (e) {
      logLine("free failed: " + e.message, "bad");
    } finally { $("btn-free").disabled = false; }
  });

  $("btn-interrupt").addEventListener("click", async () => {
    $("btn-interrupt").disabled = true;
    try {
      logLine("requesting /interrupt …");
      const r = await fetch("/api/interrupt", { method: "POST" });
      if (!r.ok) throw new Error("HTTP " + r.status);
      logLine("interrupt sent.", "warn");
      setTimeout(refresh, 400);
    } catch (e) {
      logLine("interrupt failed: " + e.message, "bad");
    } finally { $("btn-interrupt").disabled = false; }
  });

  $("btn-refresh").addEventListener("click", () => {
    refresh(); loadHistory(); loadLogTail();
  });

  $("btn-run").addEventListener("click", async () => {
    const wf = $("wf-select").value;
    if (!wf) { logLine("pick a workflow first.", "warn"); return; }
    $("btn-run").disabled = true;
    $("run-status").textContent = "queueing…";
    try {
      logLine("running " + wf + " with preset=" + SELECTED_PRESET);
      const r = await fetch("/api/run-workflow", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ workflow: wf, preset: SELECTED_PRESET })
      });
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || j.error || "HTTP " + r.status);
      $("run-status").textContent = "queued · " + (j.prompt_id || "ok") +
        (j.patched != null ? " · patched " + j.patched + " node(s)" : "");
      logLine("queued " + wf + " (id " + (j.prompt_id || "?") + ", " +
              (j.patched || 0) + " node(s) patched)", "ok");
      setTimeout(loadHistory, 1500);
    } catch (e) {
      $("run-status").textContent = "failed: " + e.message;
      logLine("run failed: " + e.message, "bad");
    } finally { $("btn-run").disabled = false; }
  });

  refresh();
  loadWorkflows();
  loadPresets();
  loadHistory();
  loadLogTail();
  setInterval(refresh, 2500);
  setInterval(loadHistory, 5000);
  setInterval(loadLogTail, 3000);
</script>
</body>
</html>
"""


# ── Workflow patching helpers (mirror sp_ablation.py) ────────────────────

def find_nodes_by_class(workflow: dict, class_type: str) -> list[str]:
    return [nid for nid, n in workflow.items()
            if isinstance(n, dict) and n.get("class_type") == class_type]


def apply_preset(workflow: dict, preset_params: dict) -> int:
    targets = find_nodes_by_class(workflow, "ShannonPrimeWanBlockSkip")
    for nid in targets:
        inputs = workflow[nid].setdefault("inputs", {})
        for k, v in preset_params.items():
            inputs[k] = v
    return len(targets)


def preset_summary(name: str, params: dict) -> str:
    on = []
    if params.get("enable_drift_gate"):          on.append("drift")
    if params.get("enable_sigma_streak"):        on.append("σ-streak")
    if params.get("enable_twin_borrow"):
        mode = params.get("twin_borrow_mode", "sym")
        on.append(f"twin({mode}, α={params.get('twin_alpha', 0.10):.2f})")
    if params.get("enable_harmonic_correction"):
        on.append(f"harmonic(α={params.get('harmonic_strength', 0.5):.2f})")
    if params.get("enable_tier_skeleton"):
        g = params.get("granite_skel_frac", 0.5)
        s = params.get("sand_skel_frac", 0.3)
        j = params.get("jazz_skel_frac", 0.2)
        on.append(f"tier-skel({g:.0%}/{s:.0%}/{j:.0%})")
    if params.get("enable_curvature_gate"):      on.append("curvature")
    if params.get("enable_cauchy_reset"):
        on.append(f"cauchy(±{params.get('cauchy_radius', 2)})")
    if not on:
        return "(all off)"
    return ", ".join(on)


# ── HTTP handler ─────────────────────────────────────────────────────────

class CtrlHandler(BaseHTTPRequestHandler):
    comfy_host = "127.0.0.1:8189"
    workflows_dir: Path | None = None
    comfy_log: Path | None = None

    def log_message(self, fmt: str, *args):
        if not args or "200" in str(args[0]) or "204" in str(args[0]):
            return
        sys.stderr.write("[ctrl] %s - %s\n" % (self.address_string(), fmt % args))

    def _json(self, code: int, body):
        data = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _proxy(self, path: str, method: str = "GET", body: bytes | None = None):
        url = f"http://{self.comfy_host}{path}"
        try:
            req = urllib.request.Request(url, data=body, method=method)
            if body is not None:
                req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
                code = r.getcode()
            try:
                return code, json.loads(raw)
            except json.JSONDecodeError:
                return code, {"raw": raw.decode("utf-8", errors="replace")}
        except urllib.error.URLError as e:
            return 502, {"error": "comfyui_unreachable", "detail": str(e)}
        except Exception as e:  # noqa: BLE001
            return 500, {"error": "proxy_error", "detail": str(e)}

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/api/stats":
            code, payload = self._proxy("/system_stats")
            if isinstance(payload, dict):
                payload["comfy_host"] = self.comfy_host
            self._json(code, payload); return
        if self.path == "/api/queue":
            self._json(*self._proxy("/queue")); return
        if self.path == "/api/workflows":
            self._json(200, self._list_workflows()); return
        if self.path == "/api/presets":
            payload = {"presets": {
                name: {"params": params, "summary": preset_summary(name, params)}
                for name, params in PRESETS.items()
            }}
            self._json(200, payload); return
        if self.path.startswith("/api/history"):
            self._json(200, self._list_history(self.path)); return
        if self.path.startswith("/api/log-tail"):
            self._json(200, self._read_log_tail(self.path)); return
        self._json(404, {"error": "not_found", "path": self.path})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length) if length > 0 else b""
        if self.path == "/api/free":
            payload = json.dumps({"unload_models": True, "free_memory": True}).encode("utf-8")
            self._json(*self._proxy("/free", method="POST", body=payload)); return
        if self.path == "/api/interrupt":
            self._json(*self._proxy("/interrupt", method="POST", body=b"{}")); return
        if self.path == "/api/run-workflow":
            self._handle_run_workflow(body); return
        self._json(404, {"error": "not_found", "path": self.path})

    # ── feature: list workflows ─────────────────────────────────────────
    def _list_workflows(self):
        if not self.workflows_dir or not self.workflows_dir.is_dir():
            return {"workflows": [], "dir": str(self.workflows_dir or "?"),
                    "error": "workflows-dir not configured or missing"}
        names = sorted(p.name for p in self.workflows_dir.glob("*.json"))
        return {"workflows": names, "dir": str(self.workflows_dir)}

    # ── feature: history (proxy + flatten) ──────────────────────────────
    def _list_history(self, path: str):
        from urllib.parse import urlparse, parse_qs
        qs = parse_qs(urlparse(path).query)
        limit = int((qs.get("limit") or ["6"])[0])
        code, raw = self._proxy("/history")
        if code != 200 or not isinstance(raw, dict):
            return {"items": [], "error": "comfyui_history_unavailable"}
        rows = []
        for pid, entry in raw.items():
            if not isinstance(entry, dict):
                continue
            status = (entry.get("status") or {}).get("status_str", "unknown")
            rows.append({"id": pid, "status": status})
        rows.sort(key=lambda r: r["id"], reverse=True)
        return {"items": rows[:limit]}

    # ── feature: tail the comfy log ─────────────────────────────────────
    def _read_log_tail(self, path: str):
        from urllib.parse import urlparse, parse_qs
        qs = parse_qs(urlparse(path).query)
        n_lines = int((qs.get("lines") or ["80"])[0])
        if not self.comfy_log:
            return {"available": False}
        try:
            log_path = self.comfy_log
            if not log_path.exists():
                return {"available": False, "reason": f"{log_path} not found"}
            size = log_path.stat().st_size
            with open(log_path, "rb") as f:
                if size > 200_000:
                    f.seek(-200_000, 2)
                    f.readline()  # discard partial
                tail_bytes = f.read()
            lines = tail_bytes.decode("utf-8", errors="replace").splitlines()
            return {"available": True, "text": "\n".join(lines[-n_lines:])}
        except Exception as e:  # noqa: BLE001
            return {"available": False, "reason": str(e)}

    # ── feature: run a workflow with a preset patched in ────────────────
    def _handle_run_workflow(self, body: bytes):
        try:
            req = json.loads(body or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "bad_json"}); return
        wf_name = req.get("workflow")
        preset  = req.get("preset", "off")
        if not wf_name:
            self._json(400, {"error": "missing 'workflow'"}); return
        if preset not in PRESETS:
            self._json(400, {"error": "unknown preset", "preset": preset}); return
        if not self.workflows_dir:
            self._json(500, {"error": "workflows-dir not configured"}); return
        wf_path = self.workflows_dir / wf_name
        if not wf_path.exists():
            self._json(404, {"error": "workflow_not_found", "path": str(wf_path)}); return
        try:
            with open(wf_path, "r", encoding="utf-8") as f:
                workflow = json.load(f)
        except Exception as e:
            self._json(500, {"error": "workflow_load", "detail": str(e)}); return

        wf_patched = copy.deepcopy(workflow)
        n_patched = apply_preset(wf_patched, PRESETS[preset])

        payload = json.dumps({"prompt": wf_patched, "client_id": "sp-control-panel"}).encode("utf-8")
        code, resp = self._proxy("/prompt", method="POST", body=payload)
        if code >= 400:
            self._json(code, {
                "error": "queue_failed", "detail": resp,
                "preset": preset, "workflow": wf_name, "patched": n_patched,
            }); return
        prompt_id = (resp or {}).get("prompt_id") if isinstance(resp, dict) else None
        self._json(200, {
            "ok": True, "prompt_id": prompt_id,
            "preset": preset, "workflow": wf_name, "patched": n_patched,
        })


# ── entry point ──────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8500,
                   help="Port to bind the control panel on (default 8500)")
    p.add_argument("--comfy-host", default="127.0.0.1:8189",
                   help="ComfyUI host:port (default 127.0.0.1:8189)")
    p.add_argument("--bind", default="127.0.0.1",
                   help="Address to bind on (default 127.0.0.1)")
    p.add_argument("--workflows-dir",
                   default=r"C:\Projects\the_system_itself\comfyui-bench\workflows",
                   help="Directory of API-format workflow JSONs to expose for one-click runs")
    p.add_argument("--comfy-log",
                   default=r"C:\Projects\the_system_itself\comfyui-bench\comfyui_launch.log",
                   help="Path to ComfyUI's stdout log for live tailing (optional)")
    args = p.parse_args()

    CtrlHandler.comfy_host = args.comfy_host
    CtrlHandler.workflows_dir = Path(args.workflows_dir) if args.workflows_dir else None
    CtrlHandler.comfy_log = Path(args.comfy_log) if args.comfy_log else None

    server = ThreadingHTTPServer((args.bind, args.port), CtrlHandler)
    print(f"[sp-control-panel] listening on http://{args.bind}:{args.port}")
    print(f"[sp-control-panel] proxying to ComfyUI at {args.comfy_host}")
    if CtrlHandler.workflows_dir:
        print(f"[sp-control-panel] workflows dir: {CtrlHandler.workflows_dir}")
    if CtrlHandler.comfy_log:
        print(f"[sp-control-panel] tailing log: {CtrlHandler.comfy_log}")
    print(f"[sp-control-panel] open the URL above in a browser. Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[sp-control-panel] shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
