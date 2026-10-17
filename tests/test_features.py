import numpy as np
import simulate as s

import scpreflight
from scpreflight.features import analyze_features, is_excel_date


def codes(report):
    return {f.code for f in report.all_findings()}


def test_excel_dates_detected(counts, genes):
    g = list(genes)
    g[100:104] = ["1-Sep", "2-Mar", "1-Dec", "Sep-07"]
    rep = scpreflight.check(s.make(counts, var_names=g))
    f = next(f for f in rep.findings if f.code == "features.excel_dates")
    assert f.level == "error"
    assert "SEPT" in f.detail and "MARCH" in f.detail


def test_excel_date_patterns():
    for x in ["1-Sep", "Sep-01", "3/1/2016", "2016-09-01", "2.31E+13", "1-dec"]:
        assert is_excel_date(x), x
    for x in ["SEPT1", "MARCH1", "ENSG00000141510", "CD3E", "MT-CO1"]:
        assert not is_excel_date(x), x


def test_duplicates(counts, genes):
    g = list(genes)
    g[200] = g[201] = "GENE5"
    assert "features.duplicates" in codes(scpreflight.check(s.make(counts, var_names=g)))


def test_mixed_species_barnyard(counts, genes):
    g = ["GRCh38_" + x for x in genes[:900]] + ["mm10___" + x.capitalize() for x in genes[900:]]
    assert "features.mixed_species" in codes(scpreflight.check(s.make(counts, var_names=g)))


def test_single_genome_prefix():
    summary, findings = analyze_features(["GRCh38_" + x for x in s.gene_names(500)])
    assert {f.code for f in findings} == {"features.genome_prefix"}
    assert summary["mitochondrial_genes_found"] == 3


def test_ensembl_species_and_symbol_column():
    ids = [f"ENSMUSG{i:011d}" for i in range(500)]
    summary, findings = analyze_features(ids, ["gene_symbols"])
    assert summary["species"] == "mouse"
    assert summary["symbol_columns"] == ["gene_symbols"]
    assert not findings


def test_ensembl_without_symbols():
    _, findings = analyze_features([f"ENSG{i:011d}" for i in range(500)])
    assert "features.no_symbols" in {f.code for f in findings}


def test_mixed_ensembl_versions():
    ids = [f"ENSG{i:011d}" + (".3" if i % 2 else "") for i in range(500)]
    _, findings = analyze_features(ids, ["gene_symbols"])
    assert "features.mixed_versions" in {f.code for f in findings}


def test_mixed_ids():
    ids = [f"ENSG{i:011d}" for i in range(300)] + s.gene_names(200)
    _, findings = analyze_features(ids)
    assert "features.mixed_ids" in {f.code for f in findings}


def test_species_from_symbol_case():
    assert analyze_features(s.gene_names(500))[0]["species"].startswith("human")
    assert analyze_features(s.gene_names(500, "mouse"))[0]["species"].startswith("mouse")


def test_atac_peaks():
    peaks = [f"chr{c}:{i * 1000}-{i * 1000 + 500}" for c in range(1, 6) for i in range(100)]
    summary, findings = analyze_features(peaks)
    assert summary["feature_type"].startswith("genomic regions")


def test_no_mito(counts):
    g = [x for x in s.gene_names(1600) if not x.startswith("MT-")][:1500]
    assert "features.no_mito" in codes(scpreflight.check(s.make(counts, var_names=g)))


def test_transposed_detected(counts, genes):
    rng = np.random.default_rng(1)
    ad = s.FakeAnnData(X=counts.T.tocsr(), obs_names=genes, var_names=s.barcodes(counts.shape[0], rng))
    assert "layout.transposed" in codes(scpreflight.check(ad))


def test_duplicate_cell_names(counts, genes):
    obs = ["AAACCTGAGAAACCAT-1"] * 2 + s.barcodes(counts.shape[0] - 2, np.random.default_rng(2))
    assert "cells.duplicate_names" in codes(scpreflight.check(s.make(counts, obs_names=obs)))
