"""Readers that open single-cell files lazily and expose every expression matrix inside.

Supported: AnnData objects, .h5ad, 10x Genomics .h5, .loom, Matrix Market (.mtx / 10x
folders), and CSV/TSV tables. HDF5 formats need ``h5py``; nothing else is required.
"""

from __future__ import annotations

import gzip
import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from ._matrix import Transposed, matrix_shape
from .features import looks_like_barcodes, looks_like_genes


@dataclass
class Slot:
    name: str  # "X", "layers['counts']", "raw.X", ...
    matrix: Any  # anything _matrix.take_rows understands
    var_names: list[str] | None = None  # only when it differs from the main var_names


@dataclass
class Dataset:
    source: str
    fmt: str
    obs_names: list[str]
    var_names: list[str]
    slots: list[Slot]
    var_columns: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    _handles: list[Any] = field(default_factory=list, repr=False)

    def close(self) -> None:
        for h in self._handles:
            try:
                h.close()
            except Exception:  # pragma: no cover
                pass


class ReaderError(Exception):
    pass


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _require_h5py():
    try:
        import h5py
    except ImportError as e:  # pragma: no cover - depends on env
        raise ReaderError(
            "Reading HDF5 files (.h5ad, .h5, .loom) needs h5py. Install it with: pip install h5py"
        ) from e
    return h5py


def _strings(ds: Any) -> list[str]:
    arr = ds[()] if hasattr(ds, "shape") else ds
    arr = np.asarray(arr).ravel()
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]


class H5CSR:
    """Lazy row access to a CSR matrix stored as data/indices/indptr in HDF5."""

    def __init__(self, group: Any, shape: tuple[int, int]):
        self.data = group["data"]
        self.indices = group["indices"]
        self.indptr = np.asarray(group["indptr"][()], dtype=np.int64)
        self.shape = (int(shape[0]), int(shape[1]))
        self.dtype = self.data.dtype

    def __getitem__(self, idx: Any) -> sp.csr_matrix:
        idx = np.atleast_1d(np.asarray(idx))
        datas, inds, ptr = [], [], [0]
        for i in idx:
            a, b = self.indptr[i], self.indptr[i + 1]
            datas.append(self.data[a:b])
            inds.append(self.indices[a:b])
            ptr.append(ptr[-1] + (b - a))
        data = np.concatenate(datas) if datas else np.array([])
        ind = np.concatenate(inds) if inds else np.array([], dtype=np.int64)
        return sp.csr_matrix((data, ind, np.asarray(ptr)), shape=(len(idx), self.shape[1]))


class ColumnSubset:
    def __init__(self, base: Any, cols: np.ndarray):
        self.base = base
        self.cols = np.asarray(cols)
        self.shape = (matrix_shape(base)[0], len(self.cols))
        self.dtype = getattr(base, "dtype", None)

    def __getitem__(self, idx: Any) -> Any:
        block = self.base[idx]
        if sp.issparse(block):
            return block.tocsc()[:, self.cols]
        return np.asarray(block)[:, self.cols]


def _h5_matrix(elem: Any) -> Any:
    """Wrap an AnnData-encoded HDF5 element (sparse group or dense dataset)."""
    import h5py

    if isinstance(elem, h5py.Dataset):
        return elem
    enc = elem.attrs.get("encoding-type", elem.attrs.get("h5sparse_format", ""))
    if isinstance(enc, bytes):
        enc = enc.decode()
    shape = tuple(elem.attrs.get("shape", elem.attrs.get("h5sparse_shape")))
    if enc in ("csr_matrix", "csr"):
        return H5CSR(elem, shape)
    if enc in ("csc_matrix", "csc"):
        # CSC has no cheap row access: load it once.
        m = sp.csc_matrix((elem["data"][()], elem["indices"][()], elem["indptr"][()]), shape=shape)
        return m.tocsr()
    raise ReaderError(f"unsupported matrix encoding {enc!r}")


def _h5_index(group: Any) -> list[str]:
    key = group.attrs.get("_index", "_index")
    if isinstance(key, bytes):
        key = key.decode()
    return _strings(group[key])


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------
def from_anndata(adata: Any, source: str = "<AnnData object>") -> Dataset:
    slots: list[Slot] = []
    if getattr(adata, "X", None) is not None:
        slots.append(Slot("X", adata.X))
    for k in list(getattr(adata, "layers", {}).keys()):
        slots.append(Slot(f"layers['{k}']", adata.layers[k]))
    raw = getattr(adata, "raw", None)
    if raw is not None and getattr(raw, "X", None) is not None:
        slots.append(Slot("raw.X", raw.X, [str(x) for x in raw.var_names]))
    var = getattr(adata, "var", None)
    return Dataset(
        source=source,
        fmt="AnnData",
        obs_names=[str(x) for x in adata.obs_names],
        var_names=[str(x) for x in adata.var_names],
        slots=slots,
        var_columns=[str(c) for c in getattr(var, "columns", [])],
    )


def read_h5ad(path: str | os.PathLike) -> Dataset:
    h5py = _require_h5py()
    f = h5py.File(path, "r")
    try:
        obs_names = _h5_index(f["obs"])
        var_names = _h5_index(f["var"])
    except Exception:
        f.close()
        return _read_h5ad_with_anndata(path)
    slots: list[Slot] = []
    if "X" in f:
        slots.append(Slot("X", _h5_matrix(f["X"])))
    if "layers" in f:
        for k in f["layers"].keys():
            slots.append(Slot(f"layers['{k}']", _h5_matrix(f["layers"][k])))
    if "raw" in f and "X" in f["raw"]:
        raw_var = _h5_index(f["raw"]["var"]) if "var" in f["raw"] else None
        slots.append(Slot("raw.X", _h5_matrix(f["raw"]["X"]), raw_var))
    var_cols = [k for k in f["var"].keys() if k not in ("_index", "__categories", f["var"].attrs.get("_index", ""))]
    return Dataset(str(path), "h5ad", obs_names, var_names, slots, var_cols, _handles=[f])


def _read_h5ad_with_anndata(path) -> Dataset:  # pragma: no cover - legacy files
    try:
        import anndata
    except ImportError as e:
        raise ReaderError("This .h5ad uses an old layout; install anndata to read it.") from e
    ad = anndata.read_h5ad(path, backed="r")
    ds = from_anndata(ad, str(path))
    ds.fmt = "h5ad (legacy, via anndata)"
    return ds


def read_10x_h5(path: str | os.PathLike) -> Dataset:
    h5py = _require_h5py()
    f = h5py.File(path, "r")
    if "matrix" in f:  # Cell Ranger v3+
        g = f["matrix"]
        n_genes, n_cells = (int(x) for x in g["shape"][()])
        feats = g["features"]
        names = _strings(feats["name"])
        ids = _strings(feats["id"])
        ftypes = _strings(feats["feature_type"]) if "feature_type" in feats else ["Gene Expression"] * n_genes
        fmt = "10x Genomics HDF5 (v3+)"
    else:  # Cell Ranger v2: one group per genome
        genome = list(f.keys())[0]
        g = f[genome]
        n_genes, n_cells = (int(x) for x in g["shape"][()])
        names = _strings(g["gene_names"])
        ids = _strings(g["genes"])
        ftypes = ["Gene Expression"] * n_genes
        fmt = f"10x Genomics HDF5 (v2, genome {genome})"
    barcodes = _strings(g["barcodes"])
    # 10x stores genes x cells as CSC, which is cells x genes as CSR.
    full = H5CSR(g, (n_cells, n_genes))
    types = list(dict.fromkeys(ftypes))
    slots: list[Slot] = []
    notes = []
    ftypes_arr = np.asarray(ftypes)
    gex_cols = np.flatnonzero(ftypes_arr == "Gene Expression")
    main_cols = gex_cols if gex_cols.size else np.arange(n_genes)
    var_names = [names[i] for i in main_cols]
    if len(types) == 1:
        slots.append(Slot("X", full))
    else:
        notes.append("Multimodal file: " + ", ".join(types))
        slots.append(Slot("X (Gene Expression)", ColumnSubset(full, main_cols)))
        for t in types:
            if t == "Gene Expression":
                continue
            cols = np.flatnonzero(ftypes_arr == t)
            slots.append(Slot(f"X ({t})", ColumnSubset(full, cols), [names[i] for i in cols]))
    if ids:
        notes.append(f"Ensembl IDs are also stored in the file (e.g. {ids[0]}).")
    return Dataset(str(path), fmt, barcodes, var_names, slots, ["gene_ids"], notes, [f])


def read_loom(path: str | os.PathLike) -> Dataset:
    h5py = _require_h5py()
    f = h5py.File(path, "r")
    ra, ca = f["row_attrs"], f["col_attrs"]
    gene_key = next((k for k in ("Gene", "gene", "Accession", "var_names") if k in ra), None)
    cell_key = next((k for k in ("CellID", "cell", "obs_names", "Cell") if k in ca), None)
    n_genes, n_cells = f["matrix"].shape
    var_names = _strings(ra[gene_key]) if gene_key else [f"gene_{i}" for i in range(n_genes)]
    obs_names = _strings(ca[cell_key]) if cell_key else [f"cell_{i}" for i in range(n_cells)]
    slots = [Slot("X", Transposed(f["matrix"]))]
    if "layers" in f:
        for k in f["layers"].keys():
            slots.append(Slot(f"layers['{k}']", Transposed(f["layers"][k])))
    return Dataset(str(path), "loom", obs_names, var_names, slots, list(ra.keys()), _handles=[f])


def _open_text(path: Path):
    if str(path).endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"))
    return open(path)


def _read_lines_col(path: Path, col: int) -> list[str]:
    out = []
    with _open_text(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            out.append(parts[col] if len(parts) > col else parts[0])
    return out


def _find(folder: Path, stems: list[str]) -> Path | None:
    for s in stems:
        for ext in ("", ".gz"):
            p = folder / f"{s}{ext}"
            if p.exists():
                return p
    for p in sorted(folder.iterdir()):  # GEO style prefixes: GSM123_barcodes.tsv.gz
        for s in stems:
            if p.name.endswith(s) or p.name.endswith(s + ".gz"):
                return p
    return None


def read_mtx(path: str | os.PathLike) -> Dataset:
    from scipy.io import mmread

    p = Path(path)
    folder = p if p.is_dir() else p.parent
    mtx = _find(folder, ["matrix.mtx"]) if p.is_dir() else p
    if mtx is None:
        raise ReaderError(f"no matrix.mtx found in {folder}")
    with (gzip.open(mtx, "rb") if str(mtx).endswith(".gz") else open(mtx, "rb")) as fh:
        try:
            M = sp.csr_matrix(mmread(fh, spmatrix=False))
        except TypeError:  # scipy < 1.15
            M = sp.csr_matrix(mmread(fh))
    feat = _find(folder, ["features.tsv", "genes.tsv"])
    bc = _find(folder, ["barcodes.tsv"])
    names = _read_lines_col(feat, 1) if feat else None
    barcodes = _read_lines_col(bc, 0) if bc else None
    r, c = M.shape
    notes = []
    # 10x convention is genes x cells. Use the companion files to decide.
    transpose = (names is not None and len(names) == r) or (barcodes is not None and len(barcodes) == c)
    if names is None and barcodes is None:
        transpose = False
        notes.append("No features/barcodes files found; assuming rows are cells.")
    if transpose:
        M = M.T.tocsr()
    n_cells, n_genes = M.shape
    obs = barcodes if barcodes and len(barcodes) == n_cells else [f"cell_{i}" for i in range(n_cells)]
    var = names if names and len(names) == n_genes else [f"gene_{i}" for i in range(n_genes)]
    return Dataset(str(path), "Matrix Market (.mtx)", obs, var, [Slot("X", M)], notes=notes)


def read_table(path: str | os.PathLike) -> Dataset:
    import pandas as pd

    name = str(path).lower().removesuffix(".gz")
    sep = "\t" if name.endswith((".tsv", ".txt", ".tab")) else ","
    df = pd.read_csv(path, sep=sep, index_col=0)
    if df.shape[1] == 0 and sep == "\t":
        df = pd.read_csv(path, sep=",", index_col=0)
    df = df.select_dtypes(include=[np.number])
    rows, cols = [str(x) for x in df.index], [str(x) for x in df.columns]
    notes = []
    transpose = (looks_like_genes(rows) and not looks_like_genes(cols)) or (
        looks_like_barcodes(cols) and not looks_like_barcodes(rows)
    )
    M = sp.csr_matrix(df.to_numpy(dtype=np.float64))
    if transpose:
        M = M.T.tocsr()
        rows, cols = cols, rows
        notes.append("Table was stored genes x cells; transposed to cells x genes for checking.")
    return Dataset(str(path), "table", rows, cols, [Slot("X", M)], notes=notes)


def read(path: str | os.PathLike) -> Dataset:
    p = Path(path)
    if not p.exists():
        raise ReaderError(f"file not found: {path}")
    low = p.name.lower()
    if p.is_dir() or low.endswith((".mtx", ".mtx.gz")):
        return read_mtx(p)
    if low.endswith(".h5ad"):
        return read_h5ad(p)
    if low.endswith(".loom"):
        return read_loom(p)
    if low.endswith((".h5", ".hdf5")):
        h5py = _require_h5py()
        with h5py.File(p, "r") as f:
            is_anndata = "obs" in f and "var" in f
        return read_h5ad(p) if is_anndata else read_10x_h5(p)
    if low.endswith((".csv", ".tsv", ".txt", ".tab", ".csv.gz", ".tsv.gz", ".txt.gz")):
        return read_table(p)
    if low.endswith((".rds", ".rdata", ".robj")):
        raise ReaderError(
            "Seurat/R files cannot be read from Python. In R, export with "
            "SeuratDisk::SaveH5Seurat + Convert(..., dest='h5ad'), or write the counts with "
            "DropletUtils::write10xCounts, then run scpreflight on the result."
        )
    raise ReaderError(f"unrecognized file type: {p.name}")
