import gzip

import numpy as np
import pandas as pd
import pytest
import scipy.io
import simulate as s

import scinspect
from scinspect.readers import H5CSR, ReaderError, read


@pytest.fixture()
def small():
    rng = np.random.default_rng(0)
    return s.counts(300, 500), s.gene_names(500), s.barcodes(300, rng)


def write_10x_folder(folder, C, genes, bcs, prefix=""):
    folder.mkdir(exist_ok=True)
    with gzip.open(folder / f"{prefix}matrix.mtx.gz", "wb") as f:
        scipy.io.mmwrite(f, C.T.tocoo())
    with gzip.open(folder / f"{prefix}features.tsv.gz", "wt") as f:
        f.writelines(f"ENSG{i:011d}\t{g}\tGene Expression\n" for i, g in enumerate(genes))
    with gzip.open(folder / f"{prefix}barcodes.tsv.gz", "wt") as f:
        f.writelines(b + "\n" for b in bcs)


def test_10x_folder(tmp_path, small):
    C, genes, bcs = small
    write_10x_folder(tmp_path / "tenx", C, genes, bcs)
    rep = scinspect.check(tmp_path / "tenx")
    assert (rep.n_obs, rep.n_vars) == (300, 500)
    assert rep.matrix("X").kind == "raw_counts"


def test_geo_style_prefixed_folder(tmp_path, small):
    C, genes, bcs = small
    write_10x_folder(tmp_path / "geo", C, genes, bcs, prefix="GSM123_")
    ds = read(tmp_path / "geo")
    assert ds.obs_names[:2] == bcs[:2]


def test_csv_genes_by_cells_is_transposed(tmp_path, small):
    C, genes, bcs = small
    df = pd.DataFrame(s.log1p(s.normalize_total(C)).toarray().T, index=genes, columns=bcs)
    p = tmp_path / "GSE1_norm.csv.gz"
    df.to_csv(p)
    rep = scinspect.check(p)
    assert (rep.n_obs, rep.n_vars) == (300, 500)
    assert rep.matrix("X").kind == "log1p_normalized"
    assert any("transposed" in f.title for f in rep.findings)


def test_tsv_cells_by_genes(tmp_path, small):
    C, genes, bcs = small
    p = tmp_path / "counts.tsv"
    pd.DataFrame(C.toarray(), index=bcs, columns=genes).to_csv(p, sep="\t")
    rep = scinspect.check(p)
    assert (rep.n_obs, rep.n_vars) == (300, 500)


def test_h5csr_row_access(small):
    C, _, _ = small
    group = {"data": C.data, "indices": C.indices, "indptr": C.indptr}
    lazy = H5CSR(group, C.shape)
    idx = np.array([0, 3, 150, 299])
    assert (lazy[idx] - C[idx]).nnz == 0


def test_legacy_compound_index():
    from scinspect.readers import _h5_columns, _h5_index

    arr = np.array([(b"AAAC-1", 3), (b"TTTG-1", 5)], dtype=[("index", "S6"), ("n_genes", "i8")])
    assert _h5_index(arr) == ["AAAC-1", "TTTG-1"]
    assert _h5_columns(arr) == ["n_genes"]


def test_new_anndata_string_encodings():
    from scinspect.readers import _strings

    nullable = {"values": np.array([b"AAAC-1", b"TTTG-1"]), "mask": np.array([False, False])}
    assert _strings(nullable) == ["AAAC-1", "TTTG-1"]
    categorical = {"codes": np.array([1, 0, 1]), "categories": np.array([b"CD3E", b"LYZ"])}
    assert _strings(categorical) == ["LYZ", "CD3E", "LYZ"]


def test_missing_file():
    with pytest.raises(ReaderError):
        read("does/not/exist.h5ad")


def test_rds_gives_helpful_error(tmp_path):
    p = tmp_path / "obj.rds"
    p.write_bytes(b"x")
    with pytest.raises(ReaderError, match="SeuratDisk"):
        read(p)


# --- HDF5 formats (run in CI where h5py / anndata are installed) -------------------------
def test_h5ad_roundtrip(tmp_path, small):
    anndata = pytest.importorskip("anndata")
    C, genes, bcs = small
    L = s.log1p(s.normalize_total(C))
    ad = anndata.AnnData(X=L, obs=pd.DataFrame(index=bcs), var=pd.DataFrame(index=genes))
    ad.layers["counts"] = C
    ad.raw = anndata.AnnData(X=C, obs=pd.DataFrame(index=bcs), var=pd.DataFrame(index=genes))
    p = tmp_path / "x.h5ad"
    ad.write_h5ad(p)
    rep = scinspect.check(p)
    kinds = {m.slot: m.kind for m in rep.matrices}
    assert kinds == {"X": "log1p_normalized", "layers['counts']": "raw_counts", "raw.X": "raw_counts"}
    # the in-memory AnnData object gives the same answer
    mem_kinds = {m.slot: m.kind for m in scinspect.check(ad).matrices}
    assert mem_kinds == kinds, f"in-memory={mem_kinds} file={kinds}"


def test_h5ad_dense_and_csc(tmp_path, small):
    anndata = pytest.importorskip("anndata")
    C, genes, bcs = small
    ad = anndata.AnnData(X=C.toarray(), obs=pd.DataFrame(index=bcs), var=pd.DataFrame(index=genes))
    ad.layers["lognorm"] = s.log1p(s.normalize_total(C)).tocsc()
    p = tmp_path / "dense.h5ad"
    ad.write_h5ad(p)
    rep = scinspect.check(p)
    assert rep.matrix("X").kind == "raw_counts"
    assert rep.matrix("layers['lognorm']").kind == "log1p_normalized"


def test_10x_h5_multimodal(tmp_path, small):
    h5py = pytest.importorskip("h5py")
    C, genes, bcs = small
    adt = s.counts(300, 20, seed=5)
    import scipy.sparse as sp

    full = sp.hstack([C, adt]).tocsr()  # cells x features == features x cells CSC
    names = genes + [f"CD{i}_TotalSeqB" for i in range(20)]
    types = ["Gene Expression"] * 500 + ["Antibody Capture"] * 20
    p = tmp_path / "filtered_feature_bc_matrix.h5"
    with h5py.File(p, "w") as f:
        g = f.create_group("matrix")
        g["data"], g["indices"], g["indptr"] = full.data, full.indices, full.indptr
        g["shape"] = np.array([full.shape[1], full.shape[0]])
        g["barcodes"] = np.array(bcs, dtype="S")
        ft = g.create_group("features")
        ft["name"] = np.array(names, dtype="S")
        ft["id"] = np.array([f"ID{i}" for i in range(520)], dtype="S")
        ft["feature_type"] = np.array(types, dtype="S")
    rep = scinspect.check(p)
    assert rep.n_vars == 500
    assert {m.slot for m in rep.matrices} == {"X (Gene Expression)", "X (Antibody Capture)"}
    assert rep.matrix("X (Gene Expression)").kind == "raw_counts"


def test_loom(tmp_path, small):
    h5py = pytest.importorskip("h5py")
    C, genes, bcs = small
    p = tmp_path / "x.loom"
    with h5py.File(p, "w") as f:
        f["matrix"] = C.toarray().T
        f.create_group("layers")["spliced"] = C.toarray().T
        f.create_group("row_attrs")["Gene"] = np.array(genes, dtype="S")
        f.create_group("col_attrs")["CellID"] = np.array(bcs, dtype="S")
    rep = scinspect.check(p)
    assert (rep.n_obs, rep.n_vars) == (300, 500)
    assert rep.matrix("layers['spliced']").kind == "raw_counts"
