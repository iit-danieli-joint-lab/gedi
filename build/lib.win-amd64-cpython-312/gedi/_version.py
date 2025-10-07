from __future__ import annotations

import os

def _compute_local_suffix() -> str:
    try:
        import torch

        cuda_ver = getattr(getattr(torch, "version", None), "cuda", None)
        if cuda_ver:
            # If torch is available, use its CUDA version
            return "+cu" + str(cuda_ver).replace(".", "")
        return "+cpu"
    except Exception:
        # If torch isn't available at sdist time, default to CPU;
        return "+cpu"

# Compute version from BASE + cpu/cuda suffix
_BASE = os.environ.get("GEDI_BASE_VERSION", "0.4.0")
__version__ = _BASE + _compute_local_suffix()
