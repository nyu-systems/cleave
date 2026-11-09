"""The offline (no-GPU) target applies only when no CUDA device exists."""

import glob
from types import SimpleNamespace

import pytest

pytest.importorskip("tilelang")  # puts its tvm on sys.path
import tvm  # noqa: E402

from cleave.core.target import no_cuda_device  # noqa: E402


@pytest.fixture
def no_device(monkeypatch):
    monkeypatch.setattr(tvm.runtime, "cuda", lambda i=0: SimpleNamespace(exist=False))


def test_hidden_devices_are_offline(no_device, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert no_cuda_device()


def test_no_device_nodes_is_offline(no_device, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(glob, "glob", lambda p: [])
    assert no_cuda_device()


def test_failed_probe_with_a_gpu_raises(no_device, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(glob, "glob", lambda p: ["/dev/nvidia0"])
    with pytest.raises(RuntimeError, match="refusing the offline target"):
        no_cuda_device()


def test_present_device_is_online(monkeypatch):
    monkeypatch.setattr(tvm.runtime, "cuda", lambda i=0: SimpleNamespace(exist=True))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert not no_cuda_device()
