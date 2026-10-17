"""Turn matrix states into concrete 'use this slot for that tool' advice."""

from __future__ import annotations

from collections.abc import Sequence

from .checks import pick_counts, pick_lognorm
from .model import MatrixState


def _ref(slot: str) -> str:
    if slot == "X":
        return "adata.X"
    if slot == "raw.X":
        return "adata.raw.X"
    if slot.startswith("layers["):
        return "adata." + slot
    return slot


def build_advice(states: Sequence[MatrixState]) -> list[dict[str, str]]:
    counts = pick_counts(states)
    lognorm = pick_lognorm(states)
    scaled = next((s for s in states if s.kind == "scaled"), None)
    x = next((s for s in states if s.slot == "X"), None)
    raw = next((s for s in states if s.slot == "raw.X"), None)
    advice: list[dict[str, str]] = []

    def add(task: str, status: str, use: str, note: str = "") -> None:
        advice.append({"task": task, "status": status, "use": use, "note": note})

    no_counts = "no raw counts found"
    # count-based models
    for task in (
        "scVI / scANVI / totalVI",
        "Pseudobulk differential expression (DESeq2, edgeR)",
        "Highly variable genes, flavor='seurat_v3'",
    ):
        if counts:
            layer_hint = ""
            if counts.slot.startswith("layers["):
                layer_hint = f"pass layer={counts.slot[7:-1]}"
            elif counts.slot == "raw.X":
                layer_hint = "copy counts out first: adata.layers['counts'] = adata.raw.X (same genes only)"
            add(task, "ok", _ref(counts.slot), layer_hint)
        else:
            add(task, "blocked", no_counts, "These methods need integer counts. Get the raw counts from the data source.")

    # normalization
    if lognorm:
        add(
            "sc.pp.normalize_total + sc.pp.log1p",
            "skip",
            f"already done in {_ref(lognorm.slot)}",
            "Running them again would log-transform twice.",
        )
    elif counts:
        note = "Normalize and log before PCA and markers."
        if counts.slot == "X":
            note += " Save the counts first: adata.layers['counts'] = adata.X.copy()."
        add("sc.pp.normalize_total + sc.pp.log1p", "todo", _ref(counts.slot), note)

    # embeddings / clustering
    if scaled and scaled.slot == "X":
        add("PCA, neighbors, clustering, UMAP", "ok", _ref(scaled.slot), "Scaled data is the usual input for PCA.")
    elif lognorm:
        add("PCA, neighbors, clustering, UMAP", "ok", _ref(lognorm.slot))
    elif scaled:
        add("PCA, neighbors, clustering, UMAP", "ok", _ref(scaled.slot), "Scaled data is the usual input for PCA.")
    elif counts:
        add("PCA, neighbors, clustering, UMAP", "todo", "normalize counts first")

    # marker genes
    if lognorm:
        note = ""
        if raw is not None and raw.kind == "raw_counts" and lognorm.slot == "X":
            note = "adata.raw holds counts, and rank_genes_groups uses adata.raw by default: pass use_raw=False."
        add("Marker genes (sc.tl.rank_genes_groups)", "ok", _ref(lognorm.slot), note)
    else:
        add("Marker genes (sc.tl.rank_genes_groups)", "todo", "needs log-normalized data")

    # CellTypist has a strict input contract
    if lognorm:
        d = lognorm.details
        exact = d.get("log_base") == "e" and d.get("target_sum") and abs(d["target_sum"] - 1e4) < 100
        if exact:
            add("CellTypist annotation", "ok", _ref(lognorm.slot), "Matches its required input: ln(1+x), 10,000 per cell.")
        else:
            add(
                "CellTypist annotation",
                "todo",
                "re-normalize",
                "CellTypist needs ln(1+x) of counts normalized to 10,000 per cell. "
                + ("Rebuild from " + _ref(counts.slot) + "." if counts else "Raw counts are needed to rebuild it."),
            )
    elif counts:
        add("CellTypist annotation", "todo", "normalize counts to 10,000 per cell, then log1p")

    if x is not None and x.kind == "scaled":
        add(
            "Plotting expression (violin, dot plots)",
            "warn",
            _ref(lognorm.slot) if lognorm else "log-normalized data",
            "adata.X is scaled, so plots from it show z-scores, not expression.",
        )
    return advice
