"""The public ``check`` function that ties everything together."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from . import __version__
from ._matrix import matrix_shape, pick_rows
from .advice import build_advice
from .checks import check_derivation, check_droplets, check_slot_names, pick_counts
from .classify import classify_matrix
from .features import analyze_features, analyze_layout
from .model import Finding, Report
from .readers import Dataset, from_anndata, read


def check(
    data: Any,
    *,
    sample_cells: int = 2000,
    seed: int = 0,
) -> Report:
    """Inspect a single-cell dataset and report what state every matrix is in.

    ``data`` can be a file path (.h5ad, 10x .h5, .loom, .mtx or 10x folder, .csv/.tsv),
    an AnnData object, or a :class:`scinspect.readers.Dataset`.
    """
    if isinstance(data, (str, os.PathLike)):
        ds = read(data)
    elif isinstance(data, Dataset):
        ds = data
    elif hasattr(data, "X") and hasattr(data, "obs_names"):
        ds = from_anndata(data)
    else:
        raise TypeError("check() needs a file path or an AnnData object")
    try:
        return _check_dataset(ds, sample_cells=sample_cells, seed=seed)
    finally:
        if ds is not data:
            ds.close()


def _check_dataset(ds: Dataset, sample_cells: int, seed: int) -> Report:
    rng = np.random.default_rng(seed)
    n_obs, n_vars = len(ds.obs_names), len(ds.var_names)
    report = Report(source=ds.source, fmt=ds.fmt, n_obs=n_obs, n_vars=n_vars, version=__version__)
    for note in ds.notes:
        report.findings.append(Finding("info", "reader.note", note))

    if not ds.slots:
        report.findings.append(Finding("error", "slots.none", "No expression matrix found in this file"))
        return report

    idx = pick_rows(n_obs, sample_cells, rng) if n_obs else None
    samples = {}
    for slot in ds.slots:
        try:
            if matrix_shape(slot.matrix)[0] != n_obs:
                report.findings.append(
                    Finding(
                        "warn",
                        "slots.shape_mismatch",
                        f"{slot.name} has {matrix_shape(slot.matrix)[0]:,} rows but there are {n_obs:,} cells",
                    )
                )
            state, S, _ = classify_matrix(slot.matrix, slot.name, rng, n_cells=sample_cells, idx=idx)
        except Exception as e:  # keep going: one broken layer should not hide the rest
            report.findings.append(
                Finding("warn", "slots.unreadable", f"Could not read {slot.name}", f"{type(e).__name__}: {e}")
            )
            continue
        report.matrices.append(state)
        samples[slot.name] = (S, slot.var_names)
        state.findings.extend(check_droplets(state, S, n_obs))
    report.sampled_cells = int(len(idx)) if idx is not None else 0

    # feature + layout checks
    summary, ffind = analyze_features(ds.var_names, ds.var_columns)
    report.features = summary
    report.findings.extend(ffind)
    report.findings.extend(analyze_layout(ds.obs_names, ds.var_names))
    if summary.get("feature_type") != "genomic regions (peaks)":
        for st in report.matrices:
            if st.kind == "binary":
                st.kind, st.label = "raw_counts", "Raw counts (non-negative integers) — every value is 0 or 1"
                st.findings = [
                    Finding(
                        "info",
                        "matrix.binary_counts",
                        f"Every non-zero value in {st.slot} is 1",
                        "Either very shallow sequencing (common in small test files) or a binarized matrix.",
                        "If it was binarized on purpose, it is not suitable as counts for scVI or DE.",
                    )
                ]
    report.findings.extend(check_slot_names(report.matrices))

    # is each processed matrix really derived from the counts?
    counts = pick_counts(report.matrices)
    if counts is not None:
        c_sample, c_var = samples[counts.slot]
        c_var = c_var or ds.var_names
        for st in report.matrices:
            if st is counts or st.kind not in ("log1p_normalized", "normalized_linear", "log1p_counts"):
                continue
            p_sample, p_var = samples[st.slot]
            p_var = p_var or ds.var_names
            if c_sample is None or p_sample is None:
                continue
            f = check_derivation(counts, c_sample, c_var, st, p_sample, p_var)
            if f:
                st.findings.append(f)
    else:
        recoverable = [
            m.slot
            for m in report.matrices
            if m.kind in ("log1p_normalized", "normalized_linear", "log1p_counts")
            and m.details.get("fingerprint_score", 0) >= 0.95
        ]
        advice = (
            "Look for the raw counts in the original submission (GEO supplementary files, the "
            "'raw_feature_bc_matrix' or 'filtered_feature_bc_matrix' folders)."
        )
        if recoverable:
            advice += (
                f" Counts can likely be reconstructed from {recoverable[0]}: undo the log, then divide each "
                "cell by its smallest non-zero value (tools that do this: stancounts, scDenorm)."
            )
        report.findings.append(
            Finding(
                "warn",
                "slots.no_counts",
                "No raw count matrix in this file",
                "Every matrix has been transformed, so count-based methods cannot be used directly.",
                advice,
            )
        )

    report.advice = build_advice(report.matrices)
    return report
