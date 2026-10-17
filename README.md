# scinspect

Check what is inside a single-cell dataset before you analyze it.

When you download an `.h5ad`, a 10x folder, or a CSV, it is often not clear whether the matrix holds
raw counts, normalized data, log data, or scaled data. scinspect looks at the values and tells you.

## How to use

1. Install Python 3.9 or newer.

2. Install scinspect:

   ```
   pip install scinspect
   ```

3. Run it on your file:

   ```
   scinspect data.h5ad
   ```

   You can also point it at a 10x folder, a 10x `.h5` file, a `.loom` file, or a `.csv`:

   ```
   scinspect filtered_feature_bc_matrix/
   scinspect counts.csv
   ```

4. Read the report. It has four parts:

   - **Matrices**: what each matrix in the file contains
   - **Features**: what the gene names look like
   - **Findings**: problems found, marked ERROR, WARN or INFO
   - **What to use for what**: which matrix to use for each tool

## Example

```
MATRICES
  X                 Scaled (per-gene mean 0, unit variance), clipped at 10
  layers['counts']  Raw counts
  raw.X             Log-normalized: ln(1+x), each cell normalized to 10,000

FINDINGS
  ERROR 2 gene names look like Excel-converted dates: 1-Mar, 2-Sep
  INFO  adata.raw holds normalized values, not raw counts

WHAT TO USE FOR WHAT
  scVI / pseudobulk DE     adata.layers['counts']
  PCA, clustering, UMAP    adata.X
  Marker genes             adata.raw.X
```

## Options

```
scinspect data.h5ad --json          # JSON output
scinspect data.h5ad --markdown      # Markdown output
scinspect data.h5ad --sample 5000   # check more cells (default 2000)
scinspect data.h5ad --fail-on warn  # exit with code 1 if there are warnings
```

## Use from Python

```python
import scinspect

report = scinspect.check("data.h5ad")   # or an AnnData object
print(report)

report.matrix("X").kind                  # e.g. 'log1p_normalized'
report.matrix("X").details["log_base"]   # e.g. 'e'
```

## What it checks

- Raw counts, log-normalized data (log base and target sum), scaled data, TPM, double log
- Whether the normalized matrix matches the counts matrix
- Gene names: Excel damage, Ensembl IDs vs symbols, species, duplicates, mixed species
- Transposed matrices, unfiltered droplet matrices, duplicated barcodes

## How it works

Undoing a log transform on normalized data gives back counts multiplied by a constant for each cell.
scinspect tests for this on a random sample of cells, which also identifies the log base and the
normalization target.

## License

MIT
