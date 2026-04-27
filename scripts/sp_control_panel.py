"""
Shannon-Prime Control Panel — minimal stdlib web UI for ComfyUI's API.

Runs a tiny HTTP server (default port 8500) that:
  - serves a single-page HTML dashboard with live VRAM, queue status, GPU info
  - proxies the most useful ComfyUI endpoints (/system_stats, /queue, /free,
    /interrupt) so the page can drive them without CORS hassles
  - lets you free VRAM, interrupt the running prompt, or check the system at
    a glance from a browser tab

Stdlib only. No FastAPI, no Flask, no pip install. Runs anywhere Python 3.10+
is installed.

Usage:
    python scripts/sp_control_panel.py
    python scripts/sp_control_panel.py --port 8500 --comfy-host 127.0.0.1:8189

Then open http://127.0.0.1:8500 in a browser.

Designed to be paired with launch_with_control.ps1, which boots ComfyUI on
8189 and the control panel on 8500 in one shot.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


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
    padding: 24px; max-width: 880px; margin: 0 auto;
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
  .vram-fill {
    height: 100%; background: var(--accent); transition: width 0.3s, background 0.3s;
  }
  .vram-label {
    position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
    font-family: ui-monospace, monospace; font-size: 12px; color: var(--text);
    text-shadow: 0 0 4px rgba(0,0,0,0.7);
  }
  .vram-fill.warn { background: var(--warn); }
  .vram-fill.bad  { background: var(--bad); }
  button {
    background: var(--panel); color: var(--text); border: 1px solid var(--border);
    padding: 10px 18px; font-size: 14px; border-radius: 6px; cursor: pointer;
    margin-right: 8px; margin-bottom: 8px;
    transition: background 0.15s, border-color 0.15s;
    font-family: inherit;
  }
  button:hover  { background: var(--border); }
  button:active { background: var(--bar-bg); }
  button.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
  button.primary:hover { background: #4493e0; }
  button.danger  { background: var(--bad);    color: #fff; border-color: var(--bad); }
  button.danger:hover  { background: #d4453f; }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  .status-dot {
    display: inline-block; width: 10px; height: 10px; border-radius: 50%;
    margin-right: 8px; vertical-align: middle;
  }
  .status-dot.ok  { background: var(--good); }
  .status-dot.bad { background: var(--bad);  box-shadow: 0 0 6px var(--bad); }
  .log {
    font-family: ui-monospace, monospace; font-size: 12px; color: var(--text-dim);
    background: var(--bg); border: 1px solid var(--border); border-radius: 4px;
    padding: 10px; max-height: 200px; overflow-y: auto; margin-top: 12px;
  }
  .log-line { margin: 2px 0; }
  .log-line.ok   { color: var(--good); }
  .log-line.warn { color: var(--warn); }
  .log-line.bad  { color: var(--bad);  }
  .row { display: flex; gap: 16px; flex-wrap: wrap; }
  .row > .card { flex: 1; min-width: 280px; }
  .footer { color: var(--text-dim); font-size: 11px; margin-top: 24px; text-align: center; }
  .footer a { color: var(--text-dim); }
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
    <div class="card-title">Actions</div>
    <button id="btn-free" class="primary">Free VRAM (unload models)</button>
    <button id="btn-interrupt" class="danger">Interrupt running prompt</button>
    <button id="btn-refresh">Refresh stats</button>
    <div id="log" class="log"><div class="log-line">ready.</div></div>
  </div>

  <div class="footer">
    Shannon-Prime Control Panel · stdlib http.server · proxies <code>/system_stats</code>, <code>/queue</code>, <code>/free</code>, <code>/interrupt</code>
  </div>

<script>
  const $ = (id) => document.getElementById(id);
  const fmt = (gb) => gb == null ? "—" : gb.toFixed(2) + " GB";

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
    } finally {
      $("btn-free").disabled = false;
    }
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
    } finally {
      $("btn-interrupt").disabled = false;
    }
  });

  $("btn-refresh").addEventListener("click", refresh);

  refresh();
  setInterval(refresh, 2500);
</script>
</body>
</html>
"""


# ── HTTP handler ─────────────────────────────────────────────────────────

class CtrlHandler(BaseHTTPRequestHandler):
    comfy_host = "127.0.0.1:8189"

    def log_message(self, fmt: str, *args):
        # Quieter logging — we only care about the meaningful stuff
        if not args or "200" in str(args[0]) or "204" in str(args[0]):
            return
        sys.stderr.write("[ctrl] %s - %s\n" % (self.address_string(), fmt % args))

    # ── helpers
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
            with urllib.request.urlopen(req, timeout=15) as r:
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

    # ── routing
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
            self._json(code, payload)
            return
        if self.path == "/api/queue":
            code, payload = self._proxy("/queue")
            self._json(code, payload)
            return
        self._json(404, {"error": "not_found", "path": self.path})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length) if length > 0 else b""

        if self.path == "/api/free":
            payload = json.dumps({"unload_models": True, "free_memory": True}).encode("utf-8")
            code, resp = self._proxy("/free", method="POST", body=payload)
            self._json(code, resp)
            return
        if self.path == "/api/interrupt":
            code, resp = self._proxy("/interrupt", method="POST", body=b"{}")
            self._json(code, resp)
            return
        self._json(404, {"error": "not_found", "path": self.path})


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
    args = p.parse_args()

    CtrlHandler.comfy_host = args.comfy_host

    server = ThreadingHTTPServer((args.bind, args.port), CtrlHandler)
    print(f"[sp-control-panel] listening on http://{args.bind}:{args.port}")
    print(f"[sp-control-panel] proxying to ComfyUI at {args.comfy_host}")
    print(f"[sp-control-panel] open the URL above in a browser. Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[sp-control-panel] shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
