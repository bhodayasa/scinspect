import sys, json
import numpy as np, pandas as pd, anndata, h5py
sys.path.insert(0, "tests")
import simulate as s, scpreflight
from scpreflight.readers import read
C = s.counts(300, 500); L = s.log1p(s.normalize_total(C))
g = s.gene_names(500); b = s.barcodes(300, np.random.default_rng(0))
ad = anndata.AnnData(X=L, obs=pd.DataFrame(index=b), var=pd.DataFrame(index=g))
ad.write_h5ad("d.h5ad")
def out(*a): print("::notice::" + " ".join(str(x) for x in a))
out("anndata", anndata.__version__, "h5py", h5py.__version__, "numpy", np.__version__)
with h5py.File("d.h5ad") as f:
    X = f["X"]; out("X type", type(X).__name__, dict(X.attrs) if hasattr(X, "attrs") else "")
    if hasattr(X, "keys"):
        for k in X.keys(): out("  ", k, X[k].dtype, X[k].shape, X[k].compression)
        out("data[:5]", X["data"][:5].tolist(), "orig", L.data[:5].tolist())
        out("indptr[:5]", X["indptr"][:5].tolist(), "orig", L.indptr[:5].tolist())
ds = read("d.h5ad"); m = ds.slots[0].matrix
out("slot", type(m).__name__, m.shape, getattr(m, "dtype", None))
row = m[np.array([0, 1])]; out("row0 data", np.asarray(row[0].data[:5]).tolist(), "orig", L[0].data[:5].tolist(), "nnz", row.nnz, L[:2].nnz)
r = scpreflight.check("d.h5ad"); st = r.matrix("X")
out("kind", st.kind, json.dumps({k: v for k, v in st.details.items()}, default=str))
r2 = scpreflight.check(ad); out("in-memory kind", r2.matrix("X").kind, type(ad.X).__name__)
