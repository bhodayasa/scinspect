import json

import numpy as np
import pandas as pd
import simulate as s

from scinspect.cli import main


def write_csv(path, X, genes, bcs):
    pd.DataFrame(X.toarray(), index=bcs, columns=genes).to_csv(path)


def test_clean_file_exit_zero(tmp_path, capsys):
    rng = np.random.default_rng(0)
    p = tmp_path / "c.csv"
    write_csv(p, s.counts(200, 400), s.gene_names(400), s.barcodes(200, rng))
    assert main([str(p), "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "Raw counts" in out and "WHAT TO USE FOR WHAT" in out


def test_json_output(tmp_path, capsys):
    rng = np.random.default_rng(0)
    p = tmp_path / "c.csv"
    write_csv(p, s.counts(200, 400), s.gene_names(400), s.barcodes(200, rng))
    main([str(p), "--json"])
    data = json.loads(capsys.readouterr().out)
    assert data["matrices"][0]["kind"] == "raw_counts"
    assert data["worst_level"] in ("ok", "info")


def test_fail_on_levels(tmp_path):
    rng = np.random.default_rng(0)
    p = tmp_path / "n.csv"  # normalized, no counts -> warning
    write_csv(p, s.log1p(s.normalize_total(s.counts(200, 400))), s.gene_names(400), s.barcodes(200, rng))
    assert main([str(p), "--json"]) == 0  # default: fail on error only
    assert main([str(p), "--json", "--fail-on", "warn"]) == 1
    assert main([str(p), "--json", "--fail-on", "never"]) == 0


def test_markdown(tmp_path, capsys):
    rng = np.random.default_rng(0)
    p = tmp_path / "c.csv"
    write_csv(p, s.counts(200, 400), s.gene_names(400), s.barcodes(200, rng))
    main([str(p), "--markdown"])
    assert "| Slot | Contents |" in capsys.readouterr().out


def test_unreadable_exit_two(tmp_path, capsys):
    assert main([str(tmp_path / "missing.h5ad")]) == 2
