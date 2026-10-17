"""Run the installed scinspect command on sample and real files.

usage: python ci/smoke.py SCINSPECT_EXE SAMPLES_DIR [REAL_DATA_DIR]
"""

import json
import subprocess
import sys
from pathlib import Path

exe, samples = sys.argv[1], Path(sys.argv[2])
real = Path(sys.argv[3]) if len(sys.argv) > 3 else None
failures = []


def run(*args):
    return subprocess.run([exe, *map(str, args)], capture_output=True, text=True)


def expect(path, kinds, worst=None, codes=()):
    p = run(path, "--json", "--fail-on", "never")
    if p.returncode != 0:
        failures.append(f"{path}: exit {p.returncode}: {p.stderr.strip()[:300]}")
        return
    rep = json.loads(p.stdout)
    got = {m["slot"]: m["kind"] for m in rep["matrices"]}
    found = {f["code"] for f in rep["findings"]} | {f["code"] for m in rep["matrices"] for f in m["findings"]}
    ok = all(got.get(k) == v for k, v in kinds.items()) and set(codes) <= found
    ok = ok and (worst is None or rep["worst_level"] == worst)
    print(("PASS " if ok else "FAIL ") + f"{Path(path).name}: {got} worst={rep['worst_level']}")
    if not ok:
        failures.append(f"{path}: got {got}, worst={rep['worst_level']}, codes={sorted(found)}")


# basic command behaviour
v = run("--version")
print("version:", v.stdout.strip())
if v.returncode != 0 or "scinspect" not in v.stdout:
    failures.append("--version failed")
txt = run(samples / "scanpy_style.h5ad", "--no-color")
if txt.returncode != 0 or "WHAT TO USE FOR WHAT" not in txt.stdout:
    failures.append(f"text report failed: {txt.stderr[:300]}")
md = run(samples / "scanpy_style.h5ad", "--markdown")
if "| Slot | Contents |" not in md.stdout:
    failures.append("markdown report failed")
if run(samples / "atlas_style.h5ad").returncode != 1:
    failures.append("expected exit 1 for a file with errors")
if run(samples / "missing.h5ad").returncode != 2:
    failures.append("expected exit 2 for a missing file")

expect(samples / "scanpy_style.h5ad", {"X": "log1p_normalized", "layers['counts']": "raw_counts"}, "ok", ["slots.derived_ok"])
expect(samples / "atlas_style.h5ad", {"X": "scaled", "raw.X": "log1p_normalized"}, "error", ["features.excel_dates"])
expect(samples / "filtered_feature_bc_matrix", {"X": "raw_counts"})
expect(samples / "filtered_feature_bc_matrix.h5", {"X": "raw_counts"})
expect(samples / "GSE000_log2cpm.csv.gz", {"X": "log1p_normalized"})

if real:
    d10 = real / "scanpy/tests/_data/10x_data"
    expect(d10 / "3.0.0/filtered_feature_bc_matrix.h5", {"X": "raw_counts"})
    expect(d10 / "3.0.0/filtered_feature_bc_matrix", {"X": "raw_counts"})
    expect(d10 / "1.2.0/filtered_gene_bc_matrices_h5.h5", {"X": "raw_counts"})
    expect(real / "scanpy/tests/_data/visium_data/2.1.0/raw_probe_bc_matrix.h5", {"X": "raw_counts"})
    arch = real / "anndata/tests/data/archives"
    expect(arch / "v0.5.0/adata.h5ad", {"X": "scaled", "raw.X": "log1p_normalized"})
    for ver in ("v0.7.0", "v0.7.8", "v0.11.4", "v0.13.3"):
        p = run(arch / ver / "adata.h5ad", "--json", "--fail-on", "never")
        status = "PASS" if p.returncode == 0 else "FAIL"
        print(f"{status} {ver}/adata.h5ad reads without crashing")
        if p.returncode != 0:
            failures.append(f"{ver}: {p.stderr[:300]}")

if failures:
    for f in failures:
        print(f"::error::{f}")
    sys.exit(1)
print("all checks passed")
