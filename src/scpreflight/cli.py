"""Command line interface: ``scpreflight FILE``."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .core import check
from .model import level_rank
from .readers import ReaderError

EPILOG = """\
examples:
  scpreflight pbmc.h5ad
  scpreflight filtered_feature_bc_matrix/            10x folder (matrix.mtx + features + barcodes)
  scpreflight GSE123456_counts.csv.gz
  scpreflight data.h5ad --json > report.json
  scpreflight data.h5ad --fail-on warn                exit 1 on warnings (for CI pipelines)

exit codes: 0 = passed, 1 = findings at or above --fail-on, 2 = could not read the file
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="scpreflight",
        description="Find out what is really inside a single-cell dataset: raw counts or normalized, "
        "which log base, scaled or not, gene ID problems, and which matrix to use for which tool.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("path", nargs="+", help="file(s) or 10x folder(s) to check")
    out = p.add_mutually_exclusive_group()
    out.add_argument("--json", action="store_true", help="print a machine-readable JSON report")
    out.add_argument("--markdown", action="store_true", help="print a Markdown report (for READMEs, PRs, lab notes)")
    p.add_argument("--sample", type=int, default=2000, metavar="N", help="cells to sample per matrix (default 2000)")
    p.add_argument("--seed", type=int, default=0, help="random seed for sampling (default 0)")
    p.add_argument(
        "--fail-on",
        choices=["error", "warn", "never"],
        default="error",
        help="exit with code 1 if any finding is at least this severe (default: error)",
    )
    p.add_argument("--no-color", action="store_true", help="disable colored output")
    p.add_argument("--version", action="version", version=f"scpreflight {__version__}")
    args = p.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # old Windows consoles can't print ✓ / ✖
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    color = sys.stdout.isatty() and not args.no_color
    worst = "ok"
    reports = []
    for i, path in enumerate(args.path):
        try:
            rep = check(path, sample_cells=args.sample, seed=args.seed)
        except ReaderError as e:
            print(f"scpreflight: {path}: {e}", file=sys.stderr)
            return 2
        except Exception as e:  # pragma: no cover
            print(f"scpreflight: {path}: could not read file ({type(e).__name__}: {e})", file=sys.stderr)
            return 2
        reports.append(rep)
        if level_rank(rep.worst_level) > level_rank(worst):
            worst = rep.worst_level
        if args.json:
            continue
        if i:
            print("\n" + "─" * 80 + "\n")
        print(rep.to_markdown() if args.markdown else rep.to_text(color=color))

    if args.json:
        if len(reports) == 1:
            print(reports[0].to_json())
        else:
            print("[\n" + ",\n".join(r.to_json() for r in reports) + "\n]")

    if args.fail_on != "never" and level_rank(worst) >= level_rank(args.fail_on):
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
