"""Helpers for pulling a small, representative sample out of any matrix-like object.

Everything downstream works on a CSR sample of cells (rows = cells, columns = features),
so huge files never need to be loaded in full.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp


def matrix_shape(m: Any) -> tuple[int, int]:
    shape = getattr(m, "shape", None)
    if shape is None:
        raise TypeError(f"object of type {type(m).__name__} has no shape")
    return int(shape[0]), int(shape[1])


def matrix_dtype(m: Any) -> str:
    dt = getattr(m, "dtype", None)
    return str(dt) if dt is not None else "unknown"


def pick_rows(n_rows: int, n_sample: int, rng: np.random.Generator) -> np.ndarray:
    """Sorted random row indices (sorted indices are required by h5py)."""
    if n_rows <= n_sample:
        return np.arange(n_rows)
    return np.sort(rng.choice(n_rows, size=n_sample, replace=False))


def take_rows(m: Any, idx: np.ndarray) -> sp.csr_matrix:
    """Return rows ``idx`` of ``m`` as a float64 CSR matrix.

    Works for numpy arrays, scipy sparse matrices, h5py datasets, anndata backed
    sparse datasets and dask arrays.
    """
    n_rows = matrix_shape(m)[0]
    full = len(idx) == n_rows
    if full and (sp.issparse(m) or isinstance(m, np.ndarray)):
        sub = m
    else:
        sub = m[idx]
    if hasattr(sub, "compute"):  # dask
        sub = sub.compute()
    if sp.issparse(sub):
        out = sp.csr_matrix(sub, dtype=np.float64)
    else:
        out = sp.csr_matrix(np.asarray(sub, dtype=np.float64))
    out.sum_duplicates()
    out.eliminate_zeros()
    return out


def dense_block(sample: sp.csr_matrix, n_cols: int, rng: np.random.Generator) -> np.ndarray:
    """Dense sub-block (all sampled rows x up to ``n_cols`` random columns)."""
    cols = pick_rows(sample.shape[1], n_cols, rng)
    return np.asarray(sample[:, cols].toarray())


class Transposed:
    """Lazy view that swaps rows and columns, e.g. for genes x cells storage."""

    def __init__(self, base: Any):
        self.base = base
        r, c = matrix_shape(base)
        self.shape = (c, r)
        self.dtype = getattr(base, "dtype", None)

    def __getitem__(self, idx: Any) -> Any:
        block = self.base[:, idx]
        if sp.issparse(block):
            return block.T.tocsr()
        return np.asarray(block).T
