from __future__ import annotations

import gc
from typing import Any


def unload_component(component: Any) -> bool:
    """Release model memory for components that expose unload()."""
    if component is None:
        return False
    unload = getattr(component, "unload", None)
    if not callable(unload):
        return False
    unload()
    clear_torch_cache()
    return True


def clear_torch_cache() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
