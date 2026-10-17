import numpy as np
import scipy.sparse as sp
import simulate as s

import scpreflight


def codes(report):
    return {f.code for f in report.all_findings()}


def test_typical_scanpy_object_is_clean(counts):
    L = s.log1p(s.normalize_total(counts))
    rep = scpreflight.check(s.make(L, layers={"counts": counts}))
    assert rep.worst_level == "ok"
    assert "slots.derived_ok" in codes(rep)
    tasks = {a["task"]: a for a in rep.advice}
    assert tasks["scVI / scANVI / totalVI"]["use"] == "adata.layers['counts']"
    assert tasks["CellTypist annotation"]["status"] == "ok"
    assert tasks["sc.pp.normalize_total + sc.pp.log1p"]["status"] == "skip"


def test_unfiltered_droplets(counts):
    rng = np.random.default_rng(0)
    empty = sp.csr_matrix(rng.poisson(0.002, (60000, counts.shape[1])).astype(np.float32))
    rep = scpreflight.check(s.make(sp.vstack([counts, empty]).tocsr()))
    assert "cells.unfiltered_droplets" in codes(rep)
    assert rep.worst_level == "error"


def test_counts_layer_that_is_not_counts(counts):
    L = s.log1p(s.normalize_total(counts))
    rep = scpreflight.check(s.make(L, layers={"counts": L}))
    assert "slots.counts_not_counts" in codes(rep)
    assert "slots.no_counts" in codes(rep)
    no_counts = next(f for f in rep.findings if f.code == "slots.no_counts")
    assert "stancounts" in no_counts.advice  # points to existing recovery tools


def test_gene_order_mismatch_between_slots(counts):
    L = s.log1p(s.normalize_total(counts))
    perm = np.random.default_rng(0).permutation(counts.shape[1])
    rep = scpreflight.check(s.make(L, layers={"counts": counts[:, perm]}))
    assert "slots.derived_mismatch" in codes(rep)


def test_hvg_subset_aligned_by_name(counts, genes):
    L = s.log1p(s.normalize_total(counts))[:, 100:500]
    ad = s.make(L, var_names=genes[100:500], raw=s.FakeRaw(counts, genes))
    rep = scpreflight.check(ad)
    f = next(f for f in rep.all_findings() if f.code == "slots.derived_ok")
    assert "gene name" in f.detail
    assert "slots.raw_is_counts" in codes(rep)
    marker = next(a for a in rep.advice if a["task"].startswith("Marker"))
    assert "use_raw=False" in marker["note"]


def test_raw_holds_lognorm(counts, genes):
    L = s.log1p(s.normalize_total(counts))
    ad = s.make(s.scale(L), raw=s.FakeRaw(L, genes))
    rep = scpreflight.check(ad)
    assert "slots.raw_is_lognorm" in codes(rep)
    assert any(a["task"].startswith("Plotting") for a in rep.advice)


def test_smartseq_read_counts():
    rep = scpreflight.check(s.make(s.counts(depth=900_000)))
    assert "matrix.read_counts" in codes(rep)


def test_no_counts_blocks_count_models(counts):
    rep = scpreflight.check(s.make(s.scale(s.log1p(s.normalize_total(counts)))))
    scvi = next(a for a in rep.advice if a["task"].startswith("scVI"))
    assert scvi["status"] == "blocked"


def test_noncanonical_lognorm_needs_renorm_for_celltypist(counts):
    rep = scpreflight.check(s.make(s.log1p(s.normalize_total(counts, 1e6), base=2), layers={"counts": counts}))
    ct = next(a for a in rep.advice if a["task"] == "CellTypist annotation")
    assert ct["status"] == "todo"
