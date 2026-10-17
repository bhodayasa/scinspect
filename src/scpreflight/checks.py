"""Dataset-level checks that look across matrices."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.sparse as sp

from .classify import INVERSES
from .model import Finding, MatrixState

# Sizes of 10x barcode whitelists: a matrix with exactly this many cells was never cell-called.
WHITELIST_SIZES = {737_280: "10x v2", 6_794_880: "10x v3", 3_686_400: "10x multiome / ARC"}
PROCESSED_KINDS = {"log1p_normalized", "normalized_linear", "log1p_counts"}


def check_droplets(state: MatrixState, sample: sp.csr_matrix | None, n_obs: int) -> list[Finding]:
    if state.kind != "raw_counts" or sample is None:
        return []
    totals = np.asarray(sample.sum(axis=1)).ravel()
    frac_tiny = float(np.mean(totals < 100))
    state.details["fraction_cells_under_100_counts"] = round(frac_tiny, 4)
    if n_obs in WHITELIST_SIZES or (n_obs > 50_000 and frac_tiny > 0.5):
        why = (
            f"{n_obs:,} barcodes is exactly the {WHITELIST_SIZES[n_obs]} whitelist size"
            if n_obs in WHITELIST_SIZES
            else f"{frac_tiny:.0%} of sampled barcodes have fewer than 100 counts"
        )
        return [
            Finding(
                "error",
                "cells.unfiltered_droplets",
                f"{state.slot} looks like an UNFILTERED droplet matrix (mostly empty droplets)",
                why,
                "Use the filtered matrix (filtered_feature_bc_matrix) or call cells first "
                "(EmptyDrops, CellBender, or a knee-plot cutoff).",
            )
        ]
    if frac_tiny > 0.2:
        return [
            Finding(
                "warn",
                "cells.many_low_count",
                f"{frac_tiny:.0%} of cells in {state.slot} have fewer than 100 counts",
                "Low-quality cells or empty droplets may not have been removed.",
                "Run basic QC (sc.pp.filter_cells(min_counts=...) / min_genes) before analysis.",
            )
        ]
    return []


def check_slot_names(states: Sequence[MatrixState]) -> list[Finding]:
    """Flag slots whose name promises something their contents do not deliver."""
    out: list[Finding] = []
    for st in states:
        name = st.slot.lower()
        if "count" in name and st.kind not in ("raw_counts", "fractional_counts", "empty"):
            out.append(
                Finding(
                    "error",
                    "slots.counts_not_counts",
                    f"{st.slot} is named like counts but contains: {st.label}",
                    advice="Tools that read this layer as counts (scVI, seurat_v3 HVG, pseudobulk DE) will be "
                    "silently wrong. Find the true raw counts.",
                )
            )
        if st.slot == "raw.X" and st.kind in PROCESSED_KINDS:
            out.append(
                Finding(
                    "info",
                    "slots.raw_is_lognorm",
                    "adata.raw holds normalized values, not raw counts",
                    "This is the usual Scanpy habit (adata.raw = adata after log1p), but many people assume "
                    "'raw' means counts.",
                    "Do not feed adata.raw to count-based models.",
                )
            )
        if st.slot == "raw.X" and st.kind == "raw_counts":
            out.append(
                Finding(
                    "info",
                    "slots.raw_is_counts",
                    "adata.raw holds raw counts",
                    advice="sc.tl.rank_genes_groups and some plots use adata.raw by default. Pass use_raw=False "
                    "so they run on log-normalized data.",
                )
            )
    return out


def check_derivation(
    counts: MatrixState,
    counts_sample: sp.csr_matrix,
    counts_var: Sequence[str] | None,
    proc: MatrixState,
    proc_sample: sp.csr_matrix,
    proc_var: Sequence[str] | None,
) -> Finding | None:
    """Is the processed matrix really computed from the counts matrix (same cells, same gene order)?"""
    if counts_sample.shape[0] != proc_sample.shape[0]:
        return None
    base = proc.details.get("fingerprint") or proc.details.get("log_base") or "linear"
    inverse = INVERSES.get(base)
    if inverse is None:
        return None

    # align genes
    if counts_var is not None and proc_var is not None and list(counts_var) != list(proc_var):
        pos = {g: i for i, g in enumerate(counts_var)}
        cols = [pos.get(g, -1) for g in proc_var]
        matched = [c >= 0 for c in cols]
        if sum(matched) < 0.5 * len(cols):
            return Finding(
                "warn",
                "slots.gene_mismatch",
                f"{proc.slot} and {counts.slot} share few gene names",
                advice="They may come from different references; check before combining them.",
            )
        keep_proc = np.flatnonzero(matched)
        C = counts_sample[:, [cols[i] for i in keep_proc]]
        P = proc_sample[:, keep_proc]
        aligned_by = "gene name"
    elif counts_sample.shape[1] == proc_sample.shape[1]:
        C, P = counts_sample, proc_sample
        aligned_by = "position"
    else:
        return None

    C = C.tocsr()
    P = P.tocsr()
    good = considered = 0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for i in range(C.shape[0]):
            c = np.asarray(C[i].toarray()).ravel()
            p = np.asarray(P[i].toarray()).ravel()
            nz = (c > 0) | (p > 0)
            if nz.sum() < 5:
                continue
            considered += 1
            y = inverse(p[nz])
            cc = c[nz]
            if np.any((cc > 0) != (y > 1e-9)):
                continue
            ratio = y / cc
            if np.std(ratio) <= 1e-3 * np.mean(ratio):
                good += 1
    if considered < 10:
        return None
    frac = good / considered
    proc.details[f"derived_from_{counts.slot}"] = round(frac, 3)
    if frac >= 0.9:
        return Finding(
            "ok",
            "slots.derived_ok",
            f"{proc.slot} is consistent with {counts.slot}",
            f"{frac:.0%} of sampled cells match exactly (genes aligned by {aligned_by}).",
        )
    return Finding(
        "warn",
        "slots.derived_mismatch",
        f"{proc.slot} does NOT match a normalization of {counts.slot}",
        f"Only {frac:.0%} of sampled cells match (genes aligned by {aligned_by}). The counts may be from "
        "different processing (ambient correction, different cells), or genes may be in a different order.",
        "Do not assume these two matrices describe the same values. Check where each came from.",
    )


def pick_counts(states: Sequence[MatrixState]) -> MatrixState | None:
    order = sorted(
        (s for s in states if s.kind == "raw_counts"),
        key=lambda s: (0 if "count" in s.slot.lower() else 1 if s.slot == "raw.X" else 2 if s.slot == "X" else 3),
    )
    return order[0] if order else None


def pick_lognorm(states: Sequence[MatrixState]) -> MatrixState | None:
    cands = [s for s in states if s.kind == "log1p_normalized"]
    if not cands:
        return None

    def score(s: MatrixState) -> tuple[int, int]:
        d = s.details
        std = d.get("log_base") == "e" and d.get("target_sum") and abs(d["target_sum"] - 1e4) < 100
        return (0 if std else 1, 0 if s.slot == "X" else 1)

    return sorted(cands, key=score)[0]


def summarize_counts_hint(states: Sequence[MatrixState]) -> dict[str, Any]:
    c = pick_counts(states)
    ln = pick_lognorm(states)
    return {"counts_slot": c.slot if c else None, "lognorm_slot": ln.slot if ln else None}
