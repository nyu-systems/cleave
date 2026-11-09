"""Frozen stage configs and the named presets (``fast`` / ``default`` / ``exhaustive``)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Literal

from ..core.errors import SpecError
from ..frontend.tensor import split_names

_MIN_MAX_OPS = 13  # the engine's default budget (GeneratorConfig::get_default_config)


class _Replace:
    def replace(self, **changes):
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class SuperoptConfig(_Replace):
    max_ops: int | Literal["auto"] = "auto"
    threads: int = 16
    split: tuple[str, ...] = ()
    concat_matmul: bool = False
    cache: bool = True

    def __post_init__(self):
        object.__setattr__(self, "split", split_names(self.split))
        if self.max_ops != "auto" and (not isinstance(self.max_ops, int) or self.max_ops < 1):
            raise SpecError(f"max_ops must be 'auto' or a positive int, got {self.max_ops!r}")

    def resolve_max_ops(self, program) -> int:
        if self.max_ops != "auto":
            return int(self.max_ops)
        return auto_max_ops(program.op_count, len(program.outputs), bool(self.split))


def auto_max_ops(n_ops: int, n_outputs: int, split: bool) -> int:
    """``max(13, n_ops + n_outputs + 1 + 2*split)``; ``n_ops`` counts inputs + compute ops."""
    return max(_MIN_MAX_OPS, n_ops + n_outputs + 1 + (2 if split else 0))


@dataclass(frozen=True)
class ScheduleConfig(_Replace):
    topk: int = 1000
    split_k: tuple[int, ...] = ()
    fusion: Literal["auto", "force", "off"] = "auto"
    hints: dict | None = None  # extra keyword arguments for cleave.schedule.search_tiling

    def __post_init__(self):
        object.__setattr__(self, "split_k", tuple(int(f) for f in (self.split_k or ())))
        if self.fusion not in ("auto", "force", "off"):
            raise SpecError(f"fusion must be 'auto', 'force' or 'off', got {self.fusion!r}")
        if self.topk < 1:
            raise SpecError(f"topk must be >= 1, got {self.topk}")


@dataclass(frozen=True)
class EvalConfig(_Replace):
    warmup: int = 16
    iters: int = 1000
    tolerance: Any = None  # cleave.runtime.validate.Tolerance; None -> runtime default
    max_parallel: int = 8
    timeout_s: float | None = None  # None -> Settings.compile_timeout_secs


@dataclass(frozen=True)
class Preset:
    name: str
    superopt: SuperoptConfig
    schedule: ScheduleConfig
    eval: EvalConfig
    description: str = ""


PRESETS: dict[str, Preset] = {
    p.name: p
    for p in (
        Preset("fast", SuperoptConfig(), ScheduleConfig(topk=32), EvalConfig(warmup=4, iters=100),
               "small tiling pool, short benchmark: smoke runs"),
        Preset("default", SuperoptConfig(), ScheduleConfig(), EvalConfig(),
               "topk=1000, 16/1000 timing"),
        Preset("exhaustive", SuperoptConfig(), ScheduleConfig(topk=100_000),
               EvalConfig(warmup=64, iters=5000),
               "every tiling the policy emits, long benchmark"),
    )
}


def list_presets() -> dict[str, str]:
    return {name: p.description for name, p in PRESETS.items()}


def get_preset(preset: str | Preset) -> Preset:
    if isinstance(preset, Preset):
        return preset
    if preset not in PRESETS:
        raise SpecError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    return PRESETS[preset]
