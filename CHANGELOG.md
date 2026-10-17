# Changelog

## 0.1.0 — 2026-10-17

First release.

- Classifies every matrix (`X`, layers, `raw.X`): raw counts, log1p-normalized (with log base and
  target sum), double-log, log of unnormalized counts, normalized-not-logged, scaled, residuals,
  TPM-like, fractional counts, binary.
- Recovers original sequencing depth from normalized data.
- Checks that processed matrices are really derived from the counts matrix.
- Gene checks: Excel-mangled names, Ensembl vs symbols, species, barnyard prefixes, duplicates,
  missing mitochondrial genes, scATAC peaks.
- Layout checks: transposed matrices, unfiltered droplet matrices, duplicated barcodes.
- Per-tool advice (scVI, pseudobulk DE, seurat_v3 HVG, PCA, marker genes, CellTypist).
- Reads .h5ad (lazily), 10x .h5, 10x folders / .mtx, .loom, CSV/TSV and AnnData objects.
- Text, Markdown and JSON output; CI-friendly exit codes.
- Handles values saved with rounded decimals or float16, NaN/inf values, AnnData < 0.7 file layouts,
  and unreadable slots (reported, not fatal).
