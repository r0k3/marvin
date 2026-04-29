"""GPU bootstrap for ``onnxruntime-gpu``.

The ``onnxruntime-gpu`` wheel needs CUDA 12.x and cuDNN 9.x shared
libraries on the dynamic linker's search path. NVIDIA ships those as
separate ``nvidia-*-cu12`` wheels that drop the ``.so`` files at
``<site-packages>/nvidia/<pkg>/lib/`` -- a path the dynamic linker does
not look in by default. The two ways to make this work are:

1. Set ``LD_LIBRARY_PATH`` *before* launching Python (gross: the env var
   is captured at exec time, so this can't be fixed from inside Python).
2. Open each ``.so`` with :func:`ctypes.CDLL` using ``RTLD_GLOBAL`` so
   its symbols join the process's global symbol table; subsequent
   ``dlopen`` calls (e.g. from onnxruntime's CUDA execution provider
   library) then find them without any path lookup. PyTorch and TensorRT
   take this approach.

This module implements (2). :func:`bootstrap` is idempotent and a no-op
on hosts where the NVIDIA wheels aren't installed (CPU-only setups,
non-Linux). It is invoked lazily from
:class:`marvin.embeddings.FastEmbedBackend` and
:class:`marvin.reranker.FastEmbedRerankerBackend` *before* the first
``fastembed`` import so the embedding/reranker model loads onto the GPU.

Set ``MARVIN_DISABLE_GPU_BOOTSTRAP=1`` to skip the preload (useful when
the host CUDA is older than the wheel-bundled libs and you'd rather
fall back to CPU than fight a version skew).
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


_LIB_PATTERNS: tuple[str, ...] = (
    # Order matters: dependents come after dependencies. RTLD_GLOBAL puts
    # already-loaded symbols in the global namespace so the next dlopen
    # finds them without needing a path.
    "libcudart.so.12*",
    "libcublasLt.so.12*",
    "libcublas.so.12*",
    "libcurand.so.10*",
    "libcufft.so.11*",
    "libnvJitLink.so.12*",
    "libnvrtc.so.12*",
    # cuDNN: the umbrella plus the engine plugins that onnxruntime touches.
    "libcudnn.so.9*",
    "libcudnn_graph.so.9*",
    "libcudnn_ops.so.9*",
    "libcudnn_cnn.so.9*",
    "libcudnn_adv.so.9*",
    "libcudnn_engines_runtime_compiled.so.9*",
    "libcudnn_engines_precompiled.so.9*",
    "libcudnn_heuristic.so.9*",
)

_BOOTSTRAPPED = False
_LOADED_LIBS: list[str] = []


def _candidate_lib_dirs() -> list[Path]:
    """Return ``<site-packages>/nvidia/<pkg>/lib`` directories on ``sys.path``.

    Order is the same as ``sys.path`` so virtualenv site-packages win
    over system-wide ones.
    """
    seen: set[Path] = set()
    out: list[Path] = []
    for entry in sys.path:
        if not entry:
            continue
        nvidia = Path(entry) / "nvidia"
        if not nvidia.is_dir():
            continue
        for pkg in sorted(nvidia.iterdir()):
            lib = pkg / "lib"
            if lib.is_dir() and lib not in seen:
                seen.add(lib)
                out.append(lib)
    return out


def _resolve_lib(lib_dirs: list[Path], pattern: str) -> Path | None:
    """First file in ``lib_dirs`` matching ``pattern``, or ``None``."""
    for lib_dir in lib_dirs:
        matches = sorted(lib_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def bootstrap() -> bool:
    """Preload CUDA / cuDNN libraries via :func:`ctypes.CDLL`.

    Returns ``True`` if at least one library was loaded (signal that
    ``onnxruntime-gpu`` should be able to find its CUDA dependencies);
    ``False`` if no NVIDIA wheels are available or the preload was
    disabled.

    Idempotent: subsequent calls are cheap no-ops.
    """
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return bool(_LOADED_LIBS)

    if os.environ.get("MARVIN_DISABLE_GPU_BOOTSTRAP") == "1":
        logger.debug("GPU bootstrap disabled via MARVIN_DISABLE_GPU_BOOTSTRAP=1")
        _BOOTSTRAPPED = True
        return False

    lib_dirs = _candidate_lib_dirs()
    if not lib_dirs:
        logger.debug("No nvidia/* lib dirs found on sys.path; skipping GPU bootstrap")
        _BOOTSTRAPPED = True
        return False

    for pattern in _LIB_PATTERNS:
        resolved = _resolve_lib(lib_dirs, pattern)
        if resolved is None:
            # Missing optional libs are fine: onnxruntime only complains
            # at session-creation time about libs it actually needs.
            continue
        try:
            ctypes.CDLL(str(resolved), mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            logger.warning(
                "GPU bootstrap: failed to preload %s (%s); CUDA execution "
                "provider may fail at session creation",
                resolved,
                exc,
            )
            continue
        _LOADED_LIBS.append(str(resolved))

    _BOOTSTRAPPED = True
    if _LOADED_LIBS:
        logger.info(
            "GPU bootstrap: preloaded %d CUDA/cuDNN libraries from %d wheel dirs",
            len(_LOADED_LIBS),
            len(lib_dirs),
        )
    return bool(_LOADED_LIBS)


def loaded_libs() -> list[str]:
    """Return paths of libraries successfully preloaded so far.

    Useful for ``MarvinService.health()`` and debug logs. Empty list
    means GPU bootstrap was either disabled, found no NVIDIA wheels, or
    hasn't run yet.
    """
    return list(_LOADED_LIBS)
