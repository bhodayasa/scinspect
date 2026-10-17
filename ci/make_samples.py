"""Write sample files for the install test."""

import gzip
import sys
from pathlib import Path

import anndata
import h5py
import numpy as np
import pandas as pd
import scipy.io

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import simulate as s  # noqa: E402

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
C = s.counts(1500, 3000)
genes = s.gene_names(3000)
bcs = s.barcodes(1500, rng)
L = s.log1p(s.normalize_total(C))

# 1. log-normalized X with a counts layer
ad = anndata.AnnData(X=L, obs=pd.DataFrame(index=bcs), var=pd.DataFrame(index=genes))
ad.layers["counts"] = C
ad.write_h5ad(out / "scanpy_style.h5ad")

# 2. scaled X, log-normalized raw, two Excel-damaged gene names
g2 = list(genes)
g2[500], g2[501] = "1-Mar", "2-Sep"
ad2 = anndata.AnnData(X=s.scale(L), obs=pd.DataFrame(index=bcs), var=pd.DataFrame(index=g2))
ad2.raw = anndata.AnnData(X=L, obs=pd.DataFrame(index=bcs), var=pd.DataFrame(index=g2))
ad2.write_h5ad(out / "atlas_style.h5ad")

# 3. 10x folder
tenx = out / "filtered_feature_bc_matrix"
tenx.mkdir(exist_ok=True)
with gzip.open(tenx / "matrix.mtx.gz", "wb") as f:
    scipy.io.mmwrite(f, C.T.tocoo())
with gzip.open(tenx / "features.tsv.gz", "wt") as f:
    f.writelines(f"ENSG{i:011d}\t{g}\tGene Expression\n" for i, g in enumerate(genes))
with gzip.open(tenx / "barcodes.tsv.gz", "wt") as f:
    f.writelines(b + "\n" for b in bcs)

# 4. 10x h5 (Cell Ranger v3 layout)
with h5py.File(out / "filtered_feature_bc_matrix.h5", "w") as f:
    g = f.create_group("matrix")
    g["data"], g["indices"], g["indptr"] = C.data, C.indices, C.indptr
    g["shape"] = np.array([C.shape[1], C.shape[0]])
    g["barcodes"] = np.array(bcs, dtype="S")
    ft = g.create_group("features")
    ft["name"] = np.array(genes, dtype="S")
    ft["id"] = np.array([f"ENSG{i:011d}" for i in range(3000)], dtype="S")
    ft["feature_type"] = np.array(["Gene Expression"] * 3000, dtype="S")

# 5. GEO-style CSV, genes x cells, log2 CPM
pd.DataFrame(s.log1p(s.normalize_total(C[:300], 1e6), base=2).toarray().T, index=genes, columns=bcs[:300]).to_csv(
    out / "GSE000_log2cpm.csv.gz"
)
print("wrote samples to", out)
