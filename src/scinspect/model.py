"""Data classes that hold the results of a scinspect check."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

# Severity levels, ordered from least to most serious.
LEVELS = ("ok", "info", "warn", "error")


def level_rank(level: str) -> int:
    return LEVELS.index(level)


@dataclass
class Finding:
    """One thing the checker noticed."""

    level: str  # "ok" | "info" | "warn" | "error"
    code: str  # stable machine-readable id, e.g. "genes.excel_dates"
    title: str  # one-line human summary
    detail: str = ""  # longer explanation / evidence
    advice: str = ""  # what to do about it

    def __post_init__(self) -> None:
        if self.level not in LEVELS:
            raise ValueError(f"unknown level {self.level!r}")


@dataclass
class MatrixState:
    """What one expression matrix (X, a layer, or raw.X) appears to contain."""

    slot: str  # e.g. "X", "layers['counts']", "raw.X"
    shape: tuple[int, int]
    dtype: str
    kind: str  # machine-readable state, see classify.KINDS
    label: str  # human-readable state
    confidence: str  # "high" | "medium" | "low"
    details: dict[str, Any] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)


@dataclass
class Report:
    """Full scinspect report for one dataset."""

    source: str
    fmt: str
    n_obs: int
    n_vars: int
    matrices: list[MatrixState] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    features: dict[str, Any] = field(default_factory=dict)
    advice: list[dict[str, str]] = field(default_factory=list)
    sampled_cells: int = 0
    version: str = ""

    # ---- convenience -------------------------------------------------
    def all_findings(self) -> list[Finding]:
        out = list(self.findings)
        for m in self.matrices:
            out.extend(m.findings)
        return out

    @property
    def worst_level(self) -> str:
        levels = [f.level for f in self.all_findings()] or ["ok"]
        return max(levels, key=level_rank)

    def matrix(self, slot: str) -> MatrixState | None:
        for m in self.matrices:
            if m.slot == slot:
                return m
        return None

    # ---- output ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["worst_level"] = self.worst_level
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=_json_default)

    def to_text(self, color: bool = False) -> str:
        from .render import render_text

        return render_text(self, color=color)

    def to_markdown(self) -> str:
        from .render import render_markdown

        return render_markdown(self)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.to_text()


def _json_default(o: Any) -> Any:
    try:
        import numpy as np

        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    return str(o)
