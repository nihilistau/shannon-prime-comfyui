# Shannon-Prime VHT2: ComfyUI Custom Nodes
# Copyright (C) 2026 Ray Daniels. All Rights Reserved.
# Licensed under AGPLv3. Commercial license available.

from .shannon_prime_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# ── Flux DiT nodes (separate module to keep Wan and Flux cleanly separated) ──
try:
    from .shannon_prime_flux_nodes import (
        NODE_CLASS_MAPPINGS as _FLUX_CLASSES,
        NODE_DISPLAY_NAME_MAPPINGS as _FLUX_NAMES,
    )
    NODE_CLASS_MAPPINGS.update(_FLUX_CLASSES)
    NODE_DISPLAY_NAME_MAPPINGS.update(_FLUX_NAMES)
except ImportError as e:
    print(f"[Shannon-Prime] Flux nodes not loaded: {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
