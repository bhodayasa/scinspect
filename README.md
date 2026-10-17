# scinspect

Check what is inside a single-cell dataset before you analyze it.

When you download an `.h5ad`, a 10x folder, or a CSV, it is often not clear whether the matrix holds
raw counts, normalized data, log data, or scaled data. scinspect looks at the values and tells you.

## Install

```
pip install scinspect
```

## Usage

```
scinspect data.h5ad
```

Example output:

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

Other options:

```
scinspect data.h5ad --json        # JSON output
scinspect data.h5ad --markdown    # Markdown output
scinspect data.h5ad --fail-on warn
```

From Python:

```python
import scinspect
report = scinspect.check(adata)   # or a file path
print(report)
```

## What it checks

- Raw counts, log-normalized data (log base and target sum), scaled data, TPM, double log
- Whether the normalized matrix matches the counts matrix
- Gene names: Excel damage, Ensembl IDs vs symbols, species, duplicates, mixed species
- Transposed matrices, unfiltered droplet matrices, duplicated barcodes
- Which matrix to use for common tools

## Supported files

`.h5ad`, 10x `.h5`, 10x folders (`matrix.mtx`), `.loom`, `.csv` / `.tsv`, and AnnData objects.

## How it works

Undoing a log transform on normalized data gives back counts multiplied by a constant for each cell.
scinspect tests for this on a random sample of cells, which also identifies the log base and the
normalization target.

## Related tools

[stancounts](https://pypi.org/project/stancounts/) and scDenorm recover counts from normalized data.

## License

MIT
