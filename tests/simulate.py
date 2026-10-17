"""Synthetic single-cell data that mimics real pipelines, without needing scanpy/anndata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

HUMAN_GENES = ["GAPDH", "ACTB", "B2M", "MALAT1", "RPL13", "EEF1A1", "TMSB4X", "FTL", "FTH1",
               "MT-CO1", "MT-ND1", "MT-ATP6", "CD3E", "CD14", "MS4A1", "NKG7", "LYZ"]


def gene_names(n: int, style: str = "human") -> list[str]:
    base = list(HUMAN_GENES)
    if style == "mouse":
        base = [g.capitalize() if not g.startswith("MT-") else "mt-" + g[3:].capitalize() for g in base]
    if style == "ensembl":
        return [f"ENSG{i:011d}" for i in range(n)]
    extra = [f"GENE{i}" if style == "human" else f"Gene{i}" for i in range(n - len(base))]
    return (base + extra)[:n]


def barcodes(n: int, rng: np.random.Generator) -> list[str]:
    letters = np.array(list("ACGT"))
    seen = set()
    out = []
    while len(out) < n:
        b = "".join(rng.choice(letters, 16)) + "-1"
        if b not in seen:
            seen.add(b)
            out.append(b)
    return out


def counts(n_cells: int = 800, n_genes: int = 1500, seed: int = 0, depth: float = 3000) -> sp.csr_matrix:
    """Negative-binomial UMI counts with realistic sparsity and depth variation."""
    rng = np.random.default_rng(seed)
    gene_mean = rng.lognormal(mean=-2.0, sigma=1.6, size=n_genes)
    gene_mean[:9] *= 60  # housekeeping genes are high
    gene_mean = gene_mean / gene_mean.sum()
    lib = rng.lognormal(mean=np.log(depth), sigma=0.5, size=n_cells)
    mu = np.outer(lib, gene_mean)
    r = 2.0
    lam = rng.gamma(r, mu / r)
    X = rng.poisson(lam).astype(np.float32)
    return sp.csr_matrix(X)


def normalize_total(X: sp.csr_matrix, target: float | None = 1e4) -> sp.csr_matrix:
    X = X.astype(np.float64).tocsr(copy=True)
    s = np.asarray(X.sum(axis=1)).ravel()
    if target is None:
        target = float(np.median(s[s > 0]))
    s[s == 0] = 1
    X = sp.diags(target / s) @ X
    return sp.csr_matrix(X, dtype=np.float32)


def log1p(X: sp.csr_matrix, base: float | None = None) -> sp.csr_matrix:
    X = X.astype(np.float32).tocsr(copy=True)
    X.data = np.log1p(X.data)
    if base is not None:
        X.data = X.data / np.log(base)
    return X


def scale(X: sp.csr_matrix, max_value: float | None = 10) -> np.ndarray:
    D = np.asarray(X.toarray(), dtype=np.float64)
    D = (D - D.mean(0)) / np.where(D.std(0) == 0, 1, D.std(0))
    if max_value is not None:
        D = np.clip(D, None, max_value)
    return D.astype(np.float32)


@dataclass
class FakeRaw:
    X: Any
    var_names: list[str]


@dataclass
class FakeAnnData:
    """Just enough of the AnnData interface for scinspect."""

    X: Any
    obs_names: list[str]
    var_names: list[str]
    layers: dict[str, Any] = field(default_factory=dict)
    raw: FakeRaw | None = None
    var: pd.DataFrame | None = None


def make(X, var_names=None, obs_names=None, seed=0, **kw) -> FakeAnnData:
    rng = np.random.default_rng(seed)
    n, g = X.shape
    return FakeAnnData(
        X=X,
        obs_names=obs_names or barcodes(n, rng),
        var_names=var_names or gene_names(g),
        **kw,
    )
