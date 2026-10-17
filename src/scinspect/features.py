"""Checks on feature (gene) names and cell names.

Catches: Ensembl IDs vs symbols, species, mixed-species references, duplicated genes,
Excel-mangled gene names (SEPT1 -> 1-Sep), missing mitochondrial genes, scATAC peaks,
and matrices stored the wrong way round (genes as rows).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

from .model import Finding

ENSEMBL_RE = re.compile(r"^(ENS[A-Z]*G)\d{11}(\.\d+)?$")
ENSEMBL_SPECIES = {
    "ENSG": "human",
    "ENSMUSG": "mouse",
    "ENSRNOG": "rat",
    "ENSDARG": "zebrafish",
    "ENSGALG": "chicken",
    "ENSSSCG": "pig",
    "ENSMMUG": "rhesus macaque",
    "ENSMFAG": "crab-eating macaque",
    "ENSBTAG": "cow",
    "ENSCAFG": "dog",
    "ENSPTRG": "chimpanzee",
}
OTHER_ID_RES = {
    "FlyBase (Drosophila)": re.compile(r"^FBgn\d{7}$"),
    "WormBase (C. elegans)": re.compile(r"^WBGene\d{8}$"),
    "Entrez numeric ID": re.compile(r"^\d+$"),
}
PEAK_RE = re.compile(r"^(chr)?[\w.]+[:_-]\d+[-_]\d+$")
GENOME_PREFIX_RE = re.compile(
    r"^(GRCh3[78]|hg19|hg38|GRCm3[89]|mm10|mm39|mm9|human|mouse)[_\-]+(.+)$", re.IGNORECASE
)
BARCODE_RE = re.compile(r"[ACGTN]{10,}")
MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
EXCEL_DATE_RES = [
    re.compile(rf"^\d{{1,2}}-({MONTHS})$", re.IGNORECASE),  # 1-Sep
    re.compile(rf"^({MONTHS})-\d{{1,2}}$", re.IGNORECASE),  # Sep-01
    re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$"),  # 9/1/2016
    re.compile(r"^\d{4}-\d{2}-\d{2}( 00:00:00)?$"),  # 2016-09-01
    re.compile(r"^\d+(\.\d+)?E[+-]\d+$", re.IGNORECASE),  # 2.31E+13 (accession -> float)
]
EXCEL_HINT = {
    "sep": "SEPT*/SEPTIN*",
    "mar": "MARCH*/MARCHF*",
    "dec": "DEC1",
    "apr": "APR-*",
    "oct": "OCT*/POU*",
    "nov": "NOV",
}
# Genes expressed in almost every cell type. Used to spot gene names on the wrong axis.
HOUSEKEEPING = {
    "GAPDH", "ACTB", "B2M", "MALAT1", "RPL13", "RPS27A", "EEF1A1", "TMSB4X", "FTL", "FTH1",
    "Gapdh", "Actb", "B2m", "Malat1", "Rpl13", "Eef1a1", "Tmsb4x", "Ftl1", "Fth1",
}
SYMBOL_COLUMNS = (
    "gene_symbols", "gene_symbol", "feature_name", "gene_name", "gene_names", "symbol",
    "Symbol", "gene", "name", "gene_ids", "gene_id", "feature_id",
)


def _sample(names: Sequence[str], n: int = 20000) -> list[str]:
    names = list(names)
    if len(names) <= n:
        return names
    step = len(names) / n
    return [names[int(i * step)] for i in range(n)]


def _frac(names: Sequence[str], pred) -> float:
    if not names:
        return 0.0
    return sum(1 for x in names if pred(x)) / len(names)


def _examples(names: Sequence[str], pred, k: int = 5) -> list[str]:
    out = []
    for x in names:
        if pred(x):
            out.append(x)
            if len(out) == k:
                break
    return out


def is_excel_date(name: str) -> bool:
    return any(r.match(name) for r in EXCEL_DATE_RES)


def analyze_features(var_names: Sequence[Any], var_columns: Sequence[str] = ()) -> tuple[dict[str, Any], list[Finding]]:
    """Describe the feature axis. Returns (summary, findings)."""
    names = [str(x) for x in var_names]
    findings: list[Finding] = []
    summary: dict[str, Any] = {"n_features": len(names)}
    if not names:
        return summary, findings
    s = _sample(names)

    # --- scATAC peaks --------------------------------------------------------
    if _frac(s, lambda x: bool(PEAK_RE.match(x))) > 0.8 and _frac(s, lambda x: bool(ENSEMBL_RE.match(x))) < 0.1:
        summary["feature_type"] = "genomic regions (peaks)"
        findings.append(
            Finding(
                "info",
                "features.peaks",
                "Features are genomic regions, so this looks like scATAC-seq",
                f"Examples: {', '.join(s[:3])}",
                "Gene-expression advice below may not apply. Use ATAC tools (Signac, SnapATAC2, ArchR).",
            )
        )
        return summary, findings

    # --- genome prefixes / mixed species --------------------------------------
    prefixes = Counter()
    stripped = []
    for x in s:
        m = GENOME_PREFIX_RE.match(x)
        if m:
            prefixes[m.group(1).lower()] += 1
            stripped.append(m.group(2))
        else:
            stripped.append(x)
    prefixed_frac = sum(prefixes.values()) / len(s)
    if prefixed_frac > 0.5:
        summary["genome_prefixes"] = dict(prefixes)
        if len(prefixes) > 1:
            findings.append(
                Finding(
                    "warn",
                    "features.mixed_species",
                    "Mixed-species reference (feature names carry several genome prefixes)",
                    "Prefix counts in sample: " + ", ".join(f"{k}: {v}" for k, v in prefixes.most_common()),
                    "This is a barnyard / multi-genome alignment. Split by species, or remove the other "
                    "species' genes, before normal analysis.",
                )
            )
        else:
            findings.append(
                Finding(
                    "info",
                    "features.genome_prefix",
                    f"Gene names carry a genome prefix ('{next(iter(prefixes))}_')",
                    advice="Strip the prefix before matching to marker lists or databases.",
                )
            )
        s = stripped

    # --- identifier type ----------------------------------------------------
    ens_matches = [ENSEMBL_RE.match(x) for x in s]
    ens_frac = sum(1 for m in ens_matches if m) / len(s)
    species = Counter(ENSEMBL_SPECIES.get(m.group(1), m.group(1)) for m in ens_matches if m)
    summary["ensembl_fraction"] = round(ens_frac, 3)
    if ens_frac > 0.9:
        summary["id_type"] = "Ensembl gene IDs"
        summary["species"] = species.most_common(1)[0][0]
        versioned = _frac([x for x in s if ENSEMBL_RE.match(x)], lambda x: "." in x)
        if 0.05 < versioned < 0.95:
            findings.append(
                Finding(
                    "warn",
                    "features.mixed_versions",
                    "Some Ensembl IDs have version suffixes (.12) and some do not",
                    advice="Strip versions (id.split('.')[0]) before merging with other data.",
                )
            )
        sym_cols = [c for c in var_columns if c in SYMBOL_COLUMNS]
        if sym_cols:
            summary["symbol_columns"] = sym_cols
        else:
            findings.append(
                Finding(
                    "info",
                    "features.no_symbols",
                    "Genes are Ensembl IDs and no gene-symbol column was found",
                    advice="Map to symbols (e.g. via biomaRt or mygene) before using marker gene lists.",
                )
            )
    elif ens_frac > 0.1:
        summary["id_type"] = "mixed Ensembl IDs and other names"
        findings.append(
            Finding(
                "warn",
                "features.mixed_ids",
                f"Gene names are a mix: {ens_frac:.0%} Ensembl IDs, the rest something else",
                f"Non-Ensembl examples: {', '.join(_examples(s, lambda x: not ENSEMBL_RE.match(x)))}",
                "Usually caused by symbol lookup failing for some genes. Pick one ID type for everything.",
            )
        )
    else:
        other = None
        for label, rx in OTHER_ID_RES.items():
            if _frac(s, lambda x, rx=rx: bool(rx.match(x))) > 0.9:
                other = label
        if other:
            summary["id_type"] = other
        else:
            summary["id_type"] = "gene symbols"
            sp_guess = _guess_species_from_symbols(s)
            if sp_guess:
                summary["species"] = sp_guess

    if len(species) > 1:
        findings.append(
            Finding(
                "warn",
                "features.multi_species_ensembl",
                "Ensembl IDs from more than one species",
                ", ".join(f"{k}: {v}" for k, v in species.most_common()),
                "Separate species before analysis.",
            )
        )

    # --- duplicates -------------------------------------------------------------
    counts = Counter(names)
    dups = {k: v for k, v in counts.items() if v > 1}
    if dups:
        n_dup = sum(v - 1 for v in dups.values())
        ex = ", ".join(f"{k} (x{v})" for k, v in sorted(dups.items(), key=lambda kv: -kv[1])[:5])
        findings.append(
            Finding(
                "warn",
                "features.duplicates",
                f"{len(dups):,} gene names appear more than once ({n_dup:,} extra copies)",
                f"Examples: {ex}",
                "Duplicates break name-based indexing and merges. Use adata.var_names_make_unique(), or "
                "switch to Ensembl IDs.",
            )
        )

    # --- Excel damage -------------------------------------------------------------
    dated = [x for x in names if is_excel_date(x)]
    if dated:
        months = Counter(
            m.lower()
            for x in dated
            for m in re.findall(MONTHS, x, flags=re.IGNORECASE)
        )
        hints = sorted({EXCEL_HINT[m] for m in months if m in EXCEL_HINT})
        findings.append(
            Finding(
                "error",
                "features.excel_dates",
                f"{len(dated)} gene names look like Excel-converted dates or numbers",
                f"Examples: {', '.join(dated[:6])}"
                + (f". Likely originals: {', '.join(hints)}" if hints else ""),
                "These genes were corrupted by opening the file in a spreadsheet. Recover the originals "
                "from the source (e.g. GEO feature file) instead of guessing.",
            )
        )

    # --- mitochondrial genes ----------------------------------------------------------
    if summary.get("id_type") == "gene symbols":
        mt = [x for x in s if x.upper().startswith("MT-") or x.upper().startswith("MT.")]
        summary["mitochondrial_genes_found"] = len(mt)
        if not mt:
            findings.append(
                Finding(
                    "info",
                    "features.no_mito",
                    "No mitochondrial genes (MT-...) found",
                    "They may have been filtered out, or use a different naming scheme.",
                    "Percent-mitochondrial QC is not possible; if you need it, start from the unfiltered data.",
                )
            )

    if len(names) < 300:
        findings.append(
            Finding(
                "info",
                "features.small_panel",
                f"Only {len(names)} features",
                "This may be a targeted panel, a protein (ADT / CITE-seq) matrix, or a gene subset.",
            )
        )
    return summary, findings


def _guess_species_from_symbols(names: Sequence[str]) -> str | None:
    lettered = [x for x in names if any(c.isalpha() for c in x)]
    if len(lettered) < 50:
        return None
    upper = _frac(lettered, lambda x: x == x.upper())
    title = _frac(lettered, lambda x: x[:1].isupper() and x[1:] == x[1:].lower() and x != x.upper())
    lower = _frac(lettered, lambda x: x == x.lower())
    if upper > 0.8:
        return "human-style symbols (ALL CAPS)"
    if title > 0.6:
        return "mouse/rat-style symbols (Capitalized)"
    if lower > 0.6:
        return "zebrafish-style symbols (lowercase)"
    return None


def looks_like_genes(names: Sequence[Any]) -> bool:
    names = _sample([str(x) for x in names], 20000)
    if not names:
        return False
    if _frac(names, lambda x: bool(ENSEMBL_RE.match(x))) > 0.5:
        return True
    return len(HOUSEKEEPING.intersection(names)) >= 3


def looks_like_barcodes(names: Sequence[Any]) -> bool:
    names = _sample([str(x) for x in names], 20000)
    return bool(names) and _frac(names, lambda x: bool(BARCODE_RE.search(x))) > 0.8


def analyze_layout(obs_names: Sequence[Any], var_names: Sequence[Any]) -> list[Finding]:
    """Detect a transposed matrix (cells and genes swapped)."""
    obs = _sample([str(x) for x in obs_names], 20000)
    var = _sample([str(x) for x in var_names], 20000)
    findings: list[Finding] = []
    gene_like, barcode_like = looks_like_genes, looks_like_barcodes

    if gene_like(obs) and not gene_like(var):
        findings.append(
            Finding(
                "error",
                "layout.transposed",
                "Matrix is transposed: genes are stored as rows (cells) and cells as columns",
                f"Row names look like genes, e.g. {', '.join(obs[:3])}",
                "Transpose before analysis (adata = adata.T). Most tools expect cells x genes.",
            )
        )
    elif barcode_like(var) and not barcode_like(obs):
        findings.append(
            Finding(
                "error",
                "layout.transposed",
                "Matrix is transposed: column names look like cell barcodes",
                f"Column names look like barcodes, e.g. {', '.join(var[:3])}",
                "Transpose before analysis (adata = adata.T).",
            )
        )
    obs_full = [str(x) for x in obs_names]
    n_dup = len(obs_full) - len(set(obs_full))
    if n_dup:
        findings.append(
            Finding(
                "warn",
                "cells.duplicate_names",
                f"{n_dup:,} duplicated cell names",
                "Often happens when several samples with the same barcodes are concatenated.",
                "Add a sample prefix to barcodes (adata.obs_names_make_unique() or concat with index_unique='-').",
            )
        )
    return findings
