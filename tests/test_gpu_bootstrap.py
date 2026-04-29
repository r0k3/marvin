"""Unit tests for ``marvin.gpu``.

These tests run on any machine because we never call into real CUDA: we
fake the file-system layout that NVIDIA wheels create
(``<prefix>/nvidia/<pkg>/lib/lib*.so*``) and stub out :func:`ctypes.CDLL`
so we can verify the discovery + preload logic without loading hundreds
of MB of GPU runtime.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _reset_module():
    """Reset the bootstrap state before *and after* every test.

    :mod:`marvin.gpu` caches results in module-level globals so the
    second call is a no-op. Tests in this file each need a clean slate;
    we also clear the state at teardown so other test modules don't
    inherit a populated cache from us.
    """
    import marvin.gpu as gpu

    gpu._BOOTSTRAPPED = False
    gpu._LOADED_LIBS = []
    yield
    gpu._BOOTSTRAPPED = False
    gpu._LOADED_LIBS = []


def _make_nvidia_layout(root: Path, packages: dict[str, list[str]]) -> Path:
    """Create ``root/nvidia/<pkg>/lib/<libname>`` for each entry.

    Returns ``root`` (suitable for prepending to ``sys.path``).
    """
    for pkg, libs in packages.items():
        lib_dir = root / "nvidia" / pkg / "lib"
        lib_dir.mkdir(parents=True, exist_ok=True)
        for libname in libs:
            (lib_dir / libname).write_bytes(b"\x7fELF stub")
    return root


class TestCandidateLibDirs:
    def test_no_nvidia_dirs_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "path", [str(tmp_path)])
        from marvin.gpu import _candidate_lib_dirs

        assert _candidate_lib_dirs() == []

    def test_walks_sys_path_in_order(self, tmp_path, monkeypatch):
        venv = _make_nvidia_layout(
            tmp_path / "venv",
            {"cublas": ["libcublas.so.12"], "cuda_runtime": ["libcudart.so.12"]},
        )
        system = _make_nvidia_layout(
            tmp_path / "system",
            {"cudnn": ["libcudnn.so.9"]},
        )
        monkeypatch.setattr(sys, "path", [str(venv), str(system), ""])

        from marvin.gpu import _candidate_lib_dirs

        dirs = _candidate_lib_dirs()
        # venv subdirs come first (sys.path order), system after; order
        # within a single nvidia/ root is sorted by package name.
        assert dirs == [
            venv / "nvidia" / "cublas" / "lib",
            venv / "nvidia" / "cuda_runtime" / "lib",
            system / "nvidia" / "cudnn" / "lib",
        ]

    def test_dedupes_repeated_sys_path_entries(self, tmp_path, monkeypatch):
        layout = _make_nvidia_layout(tmp_path, {"cublas": ["libcublas.so.12"]})
        monkeypatch.setattr(sys, "path", [str(layout), str(layout)])

        from marvin.gpu import _candidate_lib_dirs

        dirs = _candidate_lib_dirs()
        assert dirs == [layout / "nvidia" / "cublas" / "lib"]


class TestBootstrap:
    def test_disabled_via_env(self, tmp_path, monkeypatch):
        _make_nvidia_layout(tmp_path, {"cublas": ["libcublas.so.12"]})
        monkeypatch.setattr(sys, "path", [str(tmp_path)])
        monkeypatch.setenv("MARVIN_DISABLE_GPU_BOOTSTRAP", "1")

        called: list[str] = []

        def _fake_cdll(path, mode=0):  # type: ignore[no-untyped-def]
            called.append(path)

        monkeypatch.setattr(ctypes, "CDLL", _fake_cdll)

        from marvin.gpu import bootstrap, loaded_libs

        assert bootstrap() is False
        assert called == []
        assert loaded_libs() == []

    def test_no_nvidia_returns_false(self, tmp_path, monkeypatch):
        # tmp_path has no nvidia/ subdir.
        monkeypatch.setattr(sys, "path", [str(tmp_path)])

        from marvin.gpu import bootstrap, loaded_libs

        assert bootstrap() is False
        assert loaded_libs() == []

    def test_loads_libs_with_rtld_global(self, tmp_path, monkeypatch):
        layout = _make_nvidia_layout(
            tmp_path,
            {
                "cuda_runtime": ["libcudart.so.12"],
                "cublas": ["libcublas.so.12", "libcublasLt.so.12"],
                "cudnn": ["libcudnn.so.9"],
            },
        )
        monkeypatch.setattr(sys, "path", [str(layout)])

        loaded: list[tuple[str, int]] = []

        def _fake_cdll(path, mode=0):  # type: ignore[no-untyped-def]
            loaded.append((path, mode))

        monkeypatch.setattr(ctypes, "CDLL", _fake_cdll)

        from marvin.gpu import bootstrap, loaded_libs

        assert bootstrap() is True

        loaded_paths = [Path(p).name for p, _ in loaded]
        # The order must respect dependencies (cudart before cublas before cudnn).
        assert loaded_paths == [
            "libcudart.so.12",
            "libcublasLt.so.12",
            "libcublas.so.12",
            "libcudnn.so.9",
        ]
        # Every preload uses RTLD_GLOBAL so symbols are visible to
        # subsequent dlopen calls from onnxruntime's CUDA EP library.
        assert {mode for _, mode in loaded} == {ctypes.RTLD_GLOBAL}

        # And the public accessor agrees.
        assert [Path(p).name for p in loaded_libs()] == loaded_paths

    def test_idempotent(self, tmp_path, monkeypatch):
        layout = _make_nvidia_layout(
            tmp_path, {"cuda_runtime": ["libcudart.so.12"]}
        )
        monkeypatch.setattr(sys, "path", [str(layout)])

        loaded: list[str] = []

        def _fake_cdll(path, mode=0):  # type: ignore[no-untyped-def]
            loaded.append(path)

        monkeypatch.setattr(ctypes, "CDLL", _fake_cdll)

        from marvin.gpu import bootstrap

        first = bootstrap()
        second = bootstrap()
        assert first is True
        assert second is True
        # Second call must be a cheap no-op: no extra dlopen.
        assert len(loaded) == 1

    def test_dlopen_failure_does_not_abort_remaining_libs(
        self, tmp_path, monkeypatch
    ):
        layout = _make_nvidia_layout(
            tmp_path,
            {
                "cuda_runtime": ["libcudart.so.12"],
                "cublas": ["libcublas.so.12"],
            },
        )
        monkeypatch.setattr(sys, "path", [str(layout)])

        attempted: list[str] = []

        def _fake_cdll(path, mode=0):  # type: ignore[no-untyped-def]
            attempted.append(path)
            if "cudart" in path:
                raise OSError("simulated dlopen failure")

        monkeypatch.setattr(ctypes, "CDLL", _fake_cdll)

        from marvin.gpu import bootstrap, loaded_libs

        ok = bootstrap()
        # We attempted both libs; only the second succeeded.
        assert len(attempted) == 2
        assert len(loaded_libs()) == 1
        assert "libcublas" in loaded_libs()[0]
        # bootstrap() returns True if anything loaded.
        assert ok is True

    def test_glob_picks_up_minor_version_files(self, tmp_path, monkeypatch):
        """``libnvrtc.so.12.9`` should match the ``libnvrtc.so.12*`` pattern."""
        layout = _make_nvidia_layout(
            tmp_path,
            {"cuda_nvrtc": ["libnvrtc.so.12.9", "libnvrtc-builtins.so.12.9"]},
        )
        monkeypatch.setattr(sys, "path", [str(layout)])

        loaded: list[str] = []

        def _fake_cdll(path, mode=0):  # type: ignore[no-untyped-def]
            loaded.append(path)

        monkeypatch.setattr(ctypes, "CDLL", _fake_cdll)

        from marvin.gpu import bootstrap

        assert bootstrap() is True
        loaded_names = [Path(p).name for p in loaded]
        assert "libnvrtc.so.12.9" in loaded_names
