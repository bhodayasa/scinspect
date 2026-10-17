# scpreflight

**Find out what is really inside a single-cell dataset before you analyze it.**

You download an `.h5ad` from CELLxGENE, a 10x folder from GEO, or a CSV from a paper's supplement.
Is `adata.X` raw counts? Normalized? Log-transformed, and in which base? Scaled? Is `adata.raw`
counts or log data? Are the genes Ensembl IDs, mouse or human, mangled by Excel?

The file usually doesn't say, and guessing wrong fails **silently**: you log-transform twice, feed
scaled data to a differential-expression test, or give scVI normalized values. These questions come
up again and again on the scverse forum, Biostars and GitHub issues.

`scpreflight` answers them in one command:

```bash
pip install scpreflight
scpreflight downloaded_atlas.h5ad
```

```
scpreflight 0.1.0  downloaded_atlas.h5ad
h5ad · 8,000 cells × 2,000 features · checked a random sample of 2,000 cells

MATRICES
  X                 Scaled / z-scored (per-gene mean 0, unit variance) — clipped at 10 (sc.pp.scale(max_value=10) / Seurat ScaleData)  [high confidence]
  layers['counts']  Raw counts (non-negative integers)  [high confidence]
                    median 2,095 counts and 630 genes per cell
  raw.X             Log-transformed, size-normalized counts: ln(1+x), each cell normalized to 10,000 (Scanpy / Seurat default)  [high confidence]
                    original depth ≈ 2,996 counts/cell (median)

FEATURES
  gene symbols · human-style symbols (ALL CAPS) · 3 mitochondrial genes

FINDINGS
  ✖ ERROR 2 gene names look like Excel-converted dates or numbers
      Examples: 1-Mar, 2-Sep. Likely originals: MARCH*/MARCHF*, SEPT*/SEPTIN*
      → These genes were corrupted by opening the file in a spreadsheet. Recover the originals from
        the source (e.g. GEO feature file) instead of guessing.
  i INFO  adata.raw holds normalized values, not raw counts
      → Do not feed adata.raw to count-based models.
  i INFO  X is scaled data
      → Do not use this matrix for differential expression, marker genes, or cell-type annotation.
  ✓ OK    raw.X is consistent with layers['counts']
      100% of sampled cells match exactly (genes aligned by gene name).

WHAT TO USE FOR WHAT
  ✓ scVI / scANVI / totalVI                             adata.layers['counts']
  ✓ Pseudobulk differential expression (DESeq2, edgeR)  adata.layers['counts']
  ✓ Highly variable genes, flavor='seurat_v3'           adata.layers['counts']
  – sc.pp.normalize_total + sc.pp.log1p                 already done in adata.raw.X
  ✓ PCA, neighbors, clustering, UMAP                    adata.X
  ✓ Marker genes (sc.tl.rank_genes_groups)              adata.raw.X
  ✓ CellTypist annotation                               adata.raw.X
  ! Plotting expression (violin, dot plots)             adata.raw.X

VERDICT: 1 error.
```

## What it checks

**Every matrix in the file** (`X`, every layer, `raw.X`), each classified as:

| State | How it is recognized |
|---|---|
| Raw counts | non-negative integers (also flags full-length read counts vs UMIs) |
| log1p of size-normalized counts | undoing the log gives integer multiples per cell; reports **log base** (e, 2, 10) and **target sum** (10,000, CPM, median) |
| log1p applied **twice** | only a double inverse gives integers back |
| log of counts with no normalization | inverse gives exact integers, cell totals vary |
| Normalized but not logged | values are integer multiples per cell, totals constant |
| Scaled / z-scored | per-gene mean 0 and unit variance; detects clipping at 10 |
| Residuals / batch-corrected | negative values that are not z-scores |
| TPM-like, log-TPM | totals are constant but values are not count multiples |
| Fractional counts | Salmon/RSEM estimates, ambient-corrected counts |
| Binary | scATAC-style 0/1 |

It also recovers the **original sequencing depth** from normalized data, and checks that each processed
matrix really is a normalization of the counts matrix (same cells, same gene order).

**Genes and cells**

- Excel-mangled gene names (`1-Mar`, `Sep-07`, `2.31E+13`)
- Ensembl IDs vs symbols vs a mix; species from Ensembl prefix or symbol case; mixed ID versions
- Mixed-species (barnyard) references such as `GRCh38_` + `mm10___` prefixes
- Duplicated gene names and duplicated cell barcodes
- Missing mitochondrial genes (so %-mito QC is impossible)
- Matrices stored the wrong way round (genes × cells)
- Unfiltered droplet matrices full of empty barcodes (including exact 10x whitelist sizes)
- scATAC peak matrices, small protein (ADT) panels

**Slot mix-ups**

- `layers['counts']` that does not contain counts
- `adata.raw` holding log data (the Scanpy habit) or counts (so `rank_genes_groups` needs `use_raw=False`)
- Which matrix to use for scVI, pseudobulk DE, `seurat_v3` HVGs, PCA, marker genes and CellTypist

## Supported inputs

| Input | Notes |
|---|---|
| `.h5ad` | read lazily with h5py, so multi-GB atlases are fine |
| 10x Genomics `.h5` | Cell Ranger v2 and v3+, multimodal (gene expression + antibody capture) |
| 10x folder / `.mtx` | `matrix.mtx(.gz)` + `features.tsv`/`genes.tsv` + `barcodes.tsv`, including GEO `GSM123_` prefixes |
| `.loom` | including layers (velocyto spliced/unspliced) |
| `.csv`, `.tsv`, `.txt` (+ `.gz`) | genes × cells tables are detected and transposed |
| AnnData object | `scpreflight.check(adata)` in Python |

Seurat `.rds` files cannot be read from Python; the error message explains how to export them.

## Usage

```bash
scpreflight data.h5ad                  # readable report
scpreflight data.h5ad --markdown       # paste into a lab notebook, README or PR
scpreflight data.h5ad --json           # for scripts
scpreflight *.h5ad --fail-on warn      # exit 1 on warnings: use in CI or pipelines
scpreflight data.h5ad --sample 5000    # check more cells (default 2,000)
```

```python
import scanpy as sc
import scpreflight

adata = sc.read_h5ad("data.h5ad")
report = scpreflight.check(adata)
print(report)                              # same text report
report.matrix("X").kind                    # 'log1p_normalized'
report.matrix("X").details["log_base"]     # 'e'
report.matrix("X").details["target_sum"]   # 10000.0
report.worst_level                         # 'ok' | 'info' | 'warn' | 'error'
```

Exit codes: `0` passed, `1` findings at or above `--fail-on` (default `error`), `2` file could not be read.

## How it works

Most normalizations leave a fingerprint you can reverse. Scanpy's default is
`x = ln(1 + counts × 10,000 / total)`. Applying `exp(x) − 1` gives `counts × c` for one constant
`c` per cell. Divide by the cell's smallest non-zero value and you get integers back, and the cell
sums equal exactly 10,000. A wrong log base gives non-integers, so the base can be identified.
`scpreflight` runs these tests on a random sample of cells (2,000 by default), so even very large
files are checked in seconds.

## Related tools

Some pieces of this exist; `scpreflight` puts them in one report that covers every matrix in a
file plus gene and layout problems.

- [stancounts](https://pypi.org/project/stancounts/) detects log1p and its base for one matrix and recovers counts.
- [scDenorm](https://www.biorxiv.org/content/10.1101/2025.05.10.653289) and `omicverse.pp.recover_counts` recover counts from normalized data.
- [hca-validation-tools](https://github.com/clevercanary/hca-validation-tools) has an internal raw-vs-normalized check for HCA atlas curation.
- [Truke](https://pmc.ncbi.nlm.nih.gov/articles/PMC5359807/) and [Escape Excel](https://github.com/pstew/escape_excel) handle Excel gene-name damage.

If you need the counts back, use `scpreflight` to find out whether they are recoverable, then one of
the recovery tools above.

## Tested on

Besides 69 unit tests on simulated pipelines, `scpreflight` was run on the real test files that ship
with Scanpy and AnnData:

- 10x Genomics Cell Ranger v2 and v3 `.h5` files and `.mtx` folders, including a multi-genome file
- 10x Visium spatial `.h5` files (v1.0 and v2.1 probe-based)
- `.h5ad` files written by AnnData 0.5, 0.7, 0.11 and 0.13, including the reduced PBMC 68k dataset,
  where it correctly finds scaled `X` and ln(1+x) normalized `raw.X` stored with 3 decimals

## Limitations

- It infers state from the numbers; it cannot know your methods section. Check `confidence`.
- SCTransform-corrected counts are integers and are reported as raw counts.
- Species from gene symbols is a guess from capitalization (ALL CAPS vs Capitalized).
- Seurat `.rds` is not read directly.
- Files compressed with uncommon HDF5 filters are skipped slot by slot, with a note, rather than failing.

## Contributing

Bug reports with a (small or public) file that is classified wrongly are the most useful contribution.
Run the tests with:

```bash
pip install -e ".[test]"
pytest
```

## License

MIT
