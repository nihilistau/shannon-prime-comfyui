# Shannon-Prime VHT2: ComfyUI custom node entry point.
# Copyright (C) 2026 Ray Daniels. Licensed under AGPLv3 / Commercial.
#
# Symlink or copy this directory into ComfyUI/custom_nodes/ so ComfyUI
# auto-discovers the nodes on startup.

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"  # JS extension for cancel/interrupt cleanup

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# ── /sp/cleanup REST endpoint ─────────────────────────────────────────────────
# Called by our JS extension when the user cancels/interrupts a generation.
# Unloads all ComfyUI-managed models from VRAM and clears the CUDA allocator.
# Also registered as a hook on execution_interrupted so the server can trigger
# cleanup automatically without requiring JS round-trip on every cancel.
try:
    from server import PromptServer
    import comfy.model_management as _mm

    @PromptServer.instance.routes.post("/sp/cleanup")
    async def _sp_cleanup_route(request):
        from aiohttp import web
        try:
            _mm.unload_all_models()
        except Exception:
            pass
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        return web.json_response({"status": "ok", "source": "sp_cleanup"})

    print("[Shannon-Prime] /sp/cleanup endpoint registered (cancel -> unload + empty_cache)")
except Exception as _e:
    print(f"[Shannon-Prime] /sp/cleanup not registered: {_e}")
