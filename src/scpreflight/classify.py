"""Work out what processing state an expression matrix is in.

The key idea: most normalisation pipelines are *invertible fingerprints*.

* Raw UMI counts are non-negative integers.
* ``log1p(normalize_total(counts))`` can be undone with ``b**x - 1``. Inside each
  cell the result is the original counts times one constant, so dividing by the
  smallest non-zero value gives (near-)integers again, and the per-cell sums equal
  the normalisation target (10,000, 1e6, ...).
* Only the correct log base gives integers back, so the base can be identified.
* Scaled data has per-gene mean 0 and unit variance, often clipped at 10.

We test each fingerprint on a random sample of cells.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import scipy.sparse as sp

from ._matrix import dense_block, matrix_dtype, matrix_shape, pick_rows, take_rows
from .model import Finding, MatrixState

KINDS = {
    "empty": "Empty matrix",
    "all_zero": "All values are zero",
    "raw_counts": "Raw counts (non-negative integers)",
    "binary": "Binary 0/1 values",
    "log1p_normalized": "Log-transformed, size-normalized counts",
    "log1p_counts": "Log-transformed counts WITHOUT size normalization",
    "log1p_twice": "Log transform applied TWICE",
    "normalized_linear": "Size-normalized counts, NOT log-transformed",
    "log_tpm_like": "Log-transformed TPM/CPM-like values",
    "tpm_like": "TPM/CPM-like values (length- or otherwise normalized), not log-transformed",
    "fractional_counts": "Non-integer count-like values (estimated or corrected counts)",
    "log_unknown": "Log-like values with an unrecognized normalization",
    "scaled": "Scaled / z-scored (per-gene mean 0, unit variance)",
    "signed_other": "Contains negative values (residuals, batch-corrected or centered data)",
}

_LN2 = math.log(2.0)
_LN10 = math.log(10.0)

# name -> inverse transform applied to the stored values
INVERSES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "e": np.expm1,
    "2": lambda v: np.expm1(v * _LN2),
    "10": lambda v: np.expm1(v * _LN10),
    "linear": lambda v: v,
    "e_twice": lambda v: np.expm1(np.expm1(v)),
}

LOG_BASE_LABEL = {"e": "natural log", "2": "log2", "10": "log10"}
LOG_FORMULA = {"e": "ln(1+x)", "2": "log2(1+x)", "10": "log10(1+x)"}

_NICE_TARGETS = [
    (1.0, "1 (proportions: each cell sums to 1)"),
    (1e4, "10,000 (Scanpy / Seurat default)"),
    (1e5, "100,000"),
    (1e6, "1,000,000 (CPM)"),
]


# ---------------------------------------------------------------------------
# fingerprint tests
# ---------------------------------------------------------------------------
def ratio_test(sample: sp.csr_matrix, inverse: Callable[[np.ndarray], np.ndarray]) -> dict[str, Any]:
    """Undo a transform and check whether each cell becomes integer multiples of its minimum.

    Returns the fraction of informative cells that pass, plus per-cell scale info.
    """
    indptr, data = sample.indptr, sample.data
    n = sample.shape[0]
    considered = passed = 0
    mins = np.full(n, np.nan)
    mult = np.full(n, np.nan)
    sums = np.full(n, np.nan)
    with np.errstate(over="ignore", invalid="ignore"):
        for i in range(n):
            v = data[indptr[i] : indptr[i + 1]]
            if v.size == 0:
                continue
            y = inverse(v)
            if not np.all(np.isfinite(y)):
                considered += 1
                continue
            sums[i] = y.sum()
            pos = y[y > 0]
            if pos.size == 0:
                continue
            m = pos.min()
            mins[i] = m
            r = pos / m
            informative = r[(r > 1.5) & (r <= 400)]
            if informative.size < 3:
                continue
            considered += 1
            for k in (1, 2, 3, 4):
                rk = informative * k
                tol = 0.02 + 2e-5 * rk
                if np.mean(np.abs(rk - np.round(rk)) <= tol) >= 0.98:
                    passed += 1
                    mult[i] = k
                    break
    score = passed / considered if considered else 0.0
    return {
        "score": score,
        "considered": considered,
        "mins": mins,
        "mult": mult,
        "sums": sums,
    }


def _cv(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("inf")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mean = np.mean(x)
        cv = float(np.std(x) / mean) if mean else float("inf")
    return cv if np.isfinite(cv) else float("inf")


def describe_target(target: float, median_lib: float | None) -> str:
    for value, label in _NICE_TARGETS:
        if abs(target - value) <= 0.01 * value:
            return label
    if median_lib and abs(target - median_lib) <= 0.05 * median_lib:
        return f"{target:,.0f} (≈ median library size: Scanpy normalize_total with target_sum=None)"
    return f"{target:,.0f} (custom target)"


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------
def classify_matrix(
    m: Any,
    slot: str,
    rng: np.random.Generator,
    n_cells: int = 2000,
    n_genes_dense: int = 2000,
    idx: np.ndarray | None = None,
) -> tuple[MatrixState, sp.csr_matrix | None, np.ndarray | None]:
    """Classify one matrix. Returns (state, sampled CSR rows, sampled row indices)."""
    shape = matrix_shape(m)
    dtype = matrix_dtype(m)
    if shape[0] == 0 or shape[1] == 0:
        st = MatrixState(slot, shape, dtype, "empty", KINDS["empty"], "high")
        return st, None, None

    if idx is None or (len(idx) and idx[-1] >= shape[0]):
        idx = pick_rows(shape[0], n_cells, rng)
    S = take_rows(m, idx)
    vals = S.data
    details: dict[str, Any] = {"sampled_cells": int(S.shape[0])}
    if vals.size == 0:
        st = MatrixState(slot, shape, dtype, "all_zero", KINDS["all_zero"], "high", details)
        st.findings.append(
            Finding("error", "matrix.all_zero", f"{slot} contains only zeros", "Every sampled cell is empty.")
        )
        return st, S, idx

    n_total = S.shape[0] * S.shape[1]
    vmin, vmax = float(vals.min()), float(vals.max())
    frac_int = float(np.mean(np.abs(vals - np.round(vals)) <= 1e-3))
    details.update(
        zero_fraction=round(1 - S.nnz / n_total, 4),
        min_nonzero_value=vmin,
        max_value=vmax,
        fraction_integer=round(frac_int, 4),
    )

    if vmin < 0:
        st = _classify_signed(S, slot, shape, dtype, details, vmax, rng, n_genes_dense)
        return st, S, idx

    if frac_int >= 0.999:
        st = _classify_integer(S, slot, shape, dtype, details, vmax)
        return st, S, idx

    st = _classify_continuous(S, slot, shape, dtype, details, vmax)
    return st, S, idx


def _classify_signed(S, slot, shape, dtype, details, vmax, rng, n_genes_dense) -> MatrixState:
    D = dense_block(S, n_genes_dense, rng)
    means = D.mean(axis=0)
    stds = D.std(axis=0)
    keep = stds > 0
    med_abs_mean = float(np.median(np.abs(means[keep]))) if keep.any() else float("nan")
    med_std = float(np.median(stds[keep])) if keep.any() else float("nan")
    details.update(median_abs_gene_mean=round(med_abs_mean, 4), median_gene_std=round(med_std, 4))
    if med_abs_mean < 0.1 and 0.6 < med_std < 1.3:
        clipped = math.isclose(vmax, 10.0, abs_tol=1e-3)
        details["clipped_at"] = 10.0 if clipped else None
        label = KINDS["scaled"] + (" — clipped at 10 (sc.pp.scale(max_value=10) / Seurat ScaleData)" if clipped else "")
        st = MatrixState(slot, shape, dtype, "scaled", label, "high", details)
        st.findings.append(
            Finding(
                "info",
                "matrix.scaled",
                f"{slot} is scaled data",
                "Scaled values are for PCA and heatmaps only. Expression magnitudes and zeros are gone.",
                "Do not use this matrix for differential expression, marker genes, or cell-type annotation.",
            )
        )
        return st
    st = MatrixState(slot, shape, dtype, "signed_other", KINDS["signed_other"], "medium", details)
    st.findings.append(
        Finding(
            "info",
            "matrix.signed",
            f"{slot} has negative values but is not z-scored",
            "Typical sources: SCTransform / Pearson residuals, batch correction (ComBat, Harmony-corrected "
            "expression, MNN), or mean-centering.",
            "Use it for embeddings only; go back to counts for differential expression.",
        )
    )
    return st


def _classify_integer(S, slot, shape, dtype, details, vmax) -> MatrixState:
    if vmax <= 1:
        st = MatrixState(slot, shape, dtype, "binary", KINDS["binary"], "high", details)
        st.findings.append(
            Finding(
                "info",
                "matrix.binary",
                f"{slot} only contains 0 and 1",
                "Common for binarized scATAC peak matrices or thresholded data.",
            )
        )
        return st
    totals = np.asarray(S.sum(axis=1)).ravel()
    genes = np.diff(S.indptr)
    details.update(
        median_counts_per_cell=float(np.median(totals)),
        median_genes_per_cell=float(np.median(genes)),
    )
    st = MatrixState(slot, shape, dtype, "raw_counts", KINDS["raw_counts"], "high", details)
    if np.median(totals) > 200_000:
        st.findings.append(
            Finding(
                "info",
                "matrix.read_counts",
                f"{slot} looks like full-length READ counts, not UMIs",
                f"Median {np.median(totals):,.0f} counts per cell is far above typical droplet UMI depth.",
                "Plate-based (Smart-seq) read counts carry gene-length bias. Consider length-aware "
                "normalization (TPM) for cross-gene comparisons; within-gene comparisons are fine.",
            )
        )
    return st


def _classify_continuous(S, slot, shape, dtype, details, vmax) -> MatrixState:
    results = {name: ratio_test(S, inv) for name, inv in INVERSES.items()}
    scores = {k: round(v["score"], 3) for k, v in results.items()}
    details["fingerprint_scores"] = scores
    best = max(results, key=lambda k: results[k]["score"])
    best_res = results[best]

    if best_res["score"] >= 0.9 and best_res["considered"] >= 10:
        return _from_fingerprint(S, slot, shape, dtype, details, best, best_res)

    # No exact fingerprint. Fall back to sums / magnitude heuristics.
    for base in ("e", "2", "10"):
        sums = results[base]["sums"]
        if _cv(sums) < 0.01:
            target = float(np.nanmedian(sums))
            details.update(log_base=base, target_sum=target)
            st = MatrixState(
                slot,
                shape,
                dtype,
                "log_tpm_like",
                f"{KINDS['log_tpm_like']} ({LOG_BASE_LABEL[base]}, cells sum to {target:,.0f} after undoing the log)",
                "medium",
                details,
            )
            return st
    lin_sums = results["linear"]["sums"]
    if vmax > 50:
        if _cv(lin_sums) < 0.01:
            target = float(np.nanmedian(lin_sums))
            details["target_sum"] = target
            return MatrixState(slot, shape, dtype, "tpm_like", KINDS["tpm_like"], "medium", details)
        st = MatrixState(slot, shape, dtype, "fractional_counts", KINDS["fractional_counts"], "medium", details)
        st.findings.append(
            Finding(
                "info",
                "matrix.fractional",
                f"{slot} holds non-integer counts",
                "Typical sources: Salmon/RSEM/kallisto estimated counts, or ambient-RNA correction (SoupX).",
                "Most count models accept rounded values; round before scVI or DESeq2 if they refuse floats.",
            )
        )
        return st
    st = MatrixState(slot, shape, dtype, "log_unknown", KINDS["log_unknown"], "low", details)
    st.findings.append(
        Finding(
            "warn",
            "matrix.unknown_norm",
            f"Could not reverse-engineer how {slot} was normalized",
            "Values look log-scale but do not match log1p of size-normalized counts in base e, 2 or 10. "
            "Possible sources: scran pooling with non-unit minimum counts, SCTransform 'data', "
            "imputation (MAGIC, ALRA), or denoising (DCA).",
            "Check the dataset's methods section before assuming a normalization.",
        )
    )
    return st


def _from_fingerprint(S, slot, shape, dtype, details, best, res) -> MatrixState:
    sums, mins, mult = res["sums"], res["mins"], res["mult"]
    cv = _cv(sums)
    ok = np.isfinite(mins) & np.isfinite(sums)
    k = np.where(np.isfinite(mult), mult, 1.0)
    lib = (sums * k / mins)[ok]
    median_lib = float(np.median(lib)) if lib.size else None
    details["fingerprint"] = best
    details["fingerprint_score"] = round(res["score"], 3)
    if median_lib:
        details["estimated_median_counts_per_cell"] = round(median_lib, 1)
    constant = cv < 0.01
    target = float(np.nanmedian(sums)) if constant else None
    details["per_cell_sum_cv"] = round(cv, 4) if math.isfinite(cv) else None

    if best == "e_twice":
        details["log_base"] = "e"
        st = MatrixState(slot, shape, dtype, "log1p_twice", KINDS["log1p_twice"], "high", details)
        st.findings.append(
            Finding(
                "error",
                "matrix.double_log",
                f"{slot} was log-transformed twice",
                "Undoing log1p once still leaves log-scale values; undoing it twice recovers counts.",
                "Rebuild from counts, or apply np.expm1 once to get back to single-log data.",
            )
        )
        return st

    if best == "linear":
        label = KINDS["normalized_linear"]
        if constant:
            details["target_sum"] = target
            details["target_sum_label"] = describe_target(target, median_lib)
            label += f" (each cell sums to {details['target_sum_label']})"
        st = MatrixState(slot, shape, dtype, "normalized_linear", label, "high", details)
        st.findings.append(
            Finding(
                "info",
                "matrix.not_logged",
                f"{slot} is normalized but not log-transformed",
                advice="Most Scanpy/Seurat steps (HVG 'seurat' flavor, PCA, marker genes) expect log1p data. "
                "Run sc.pp.log1p on a copy before them.",
            )
        )
        return st

    details["log_base"] = best
    base_label = LOG_BASE_LABEL[best]
    if constant:
        details["target_sum"] = target
        details["target_sum_label"] = describe_target(target, median_lib)
        label = f"{KINDS['log1p_normalized']}: {LOG_FORMULA[best]}, each cell normalized to {details['target_sum_label']}"
        return MatrixState(slot, shape, dtype, "log1p_normalized", label, "high", details)

    frac_unit_min = float(np.nanmean(np.abs(mins[ok] - 1.0) < 1e-3)) if ok.any() else 0.0
    if frac_unit_min > 0.95:
        st = MatrixState(
            slot, shape, dtype, "log1p_counts", f"{KINDS['log1p_counts']} ({base_label})", "high", details
        )
        st.findings.append(
            Finding(
                "warn",
                "matrix.log_without_norm",
                f"{slot} is log of raw counts with no per-cell normalization",
                "Sequencing depth differences between cells are still present.",
                "The exact counts are recoverable with np.expm1. Normalize them (sc.pp.normalize_total) "
                "before taking the log.",
            )
        )
        return st

    label = f"{KINDS['log1p_normalized']}: {LOG_FORMULA[best]}, per-cell totals vary"
    st = MatrixState(slot, shape, dtype, "log1p_normalized", label, "high", details)
    note = (
        "Cells do not all sum to one target. Either genes were removed after normalization "
        "(for example keeping only highly variable genes) or size factors were used (scran, DESeq2-style)."
    )
    details["note"] = note
    return st
