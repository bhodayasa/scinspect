"""Human-readable renderings of a Report."""

from __future__ import annotations

import textwrap

from .model import Finding, Report, level_rank

_ICON = {"ok": "✓", "info": "i", "warn": "!", "error": "✖"}
_STATUS_ICON = {"ok": "✓", "skip": "–", "todo": "→", "blocked": "✖", "warn": "!"}
_ANSI = {
    "ok": "\033[32m",
    "info": "\033[36m",
    "warn": "\033[33m",
    "error": "\033[31m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "reset": "\033[0m",
}
_STATUS_LEVEL = {"ok": "ok", "skip": "info", "todo": "info", "blocked": "error", "warn": "warn"}


def _sorted(findings: list[Finding]) -> list[Finding]:
    return sorted(findings, key=lambda f: -level_rank(f.level))


def _extra(details: dict) -> list[str]:
    out = []
    if "estimated_median_counts_per_cell" in details:
        out.append(f"original depth ≈ {details['estimated_median_counts_per_cell']:,.0f} counts/cell (median)")
    if "median_counts_per_cell" in details:
        out.append(
            f"median {details['median_counts_per_cell']:,.0f} counts and "
            f"{details['median_genes_per_cell']:,.0f} genes per cell"
        )
    if details.get("note"):
        out.append(details["note"])
    return out


def _counts(report: Report) -> dict[str, int]:
    c = {"error": 0, "warn": 0}
    for f in report.all_findings():
        if f.level in c:
            c[f.level] += 1
    return c


def verdict(report: Report) -> str:
    c = _counts(report)
    if not c["error"] and not c["warn"]:
        return "No problems found."
    parts = []
    if c["error"]:
        parts.append(f"{c['error']} error{'s' * (c['error'] != 1)}")
    if c["warn"]:
        parts.append(f"{c['warn']} warning{'s' * (c['warn'] != 1)}")
    return ", ".join(parts) + "."


def render_text(report: Report, color: bool = False, width: int = 100) -> str:
    def c(key: str, s: str) -> str:
        return f"{_ANSI[key]}{s}{_ANSI['reset']}" if color else s

    lines: list[str] = []
    w = textwrap.TextWrapper(width=width, subsequent_indent=" " * 6)
    lines.append(c("bold", f"scpreflight {report.version}") + f"  {report.source}")
    lines.append(
        f"{report.fmt} · {report.n_obs:,} cells × {report.n_vars:,} features"
        + (f" · checked a random sample of {report.sampled_cells:,} cells" if report.sampled_cells else "")
    )
    lines.append("")

    lines.append(c("bold", "MATRICES"))
    slot_w = max([len(m.slot) for m in report.matrices] + [4])
    for m in report.matrices:
        lines.append(f"  {m.slot.ljust(slot_w)}  {m.label}  " + c("dim", f"[{m.confidence} confidence]"))
        for e in _extra(m.details):
            lines.append(" " * (slot_w + 4) + c("dim", e))
    lines.append("")

    fs = report.features
    if fs:
        lines.append(c("bold", "FEATURES"))
        bits = [fs.get("id_type") or fs.get("feature_type") or "unknown"]
        if fs.get("species"):
            bits.append(fs["species"])
        if "mitochondrial_genes_found" in fs:
            bits.append(f"{fs['mitochondrial_genes_found']} mitochondrial genes")
        if fs.get("symbol_columns"):
            bits.append("symbols in var columns: " + ", ".join(fs["symbol_columns"]))
        lines.append("  " + " · ".join(bits))
        lines.append("")

    findings = [f for f in _sorted(report.all_findings()) if f.code != "reader.note"]
    notes = [f for f in report.findings if f.code == "reader.note"]
    if findings:
        lines.append(c("bold", "FINDINGS"))
        for f in findings:
            head = f"  {c(f.level, _ICON[f.level] + ' ' + f.level.upper().ljust(5))} {f.title}"
            lines.append(head)
            for txt, pre in ((f.detail, ""), (f.advice, "→ ")):
                if txt:
                    w.initial_indent = " " * 6 + pre
                    w.subsequent_indent = " " * (6 + len(pre))
                    lines.extend(w.wrap(txt))
        lines.append("")
    if notes:
        for n in notes:
            lines.append(c("dim", f"  note: {n.title}"))
        lines.append("")

    if report.advice:
        lines.append(c("bold", "WHAT TO USE FOR WHAT"))
        task_w = max(len(a["task"]) for a in report.advice)
        for a in report.advice:
            icon = c(_STATUS_LEVEL[a["status"]], _STATUS_ICON[a["status"]])
            lines.append(f"  {icon} {a['task'].ljust(task_w)}  {a['use']}")
            if a["note"]:
                w.initial_indent = " " * (task_w + 6)
                w.subsequent_indent = " " * (task_w + 6)
                lines.extend(c("dim", ln) for ln in w.wrap(a["note"]))
        lines.append("")

    worst = report.worst_level
    lines.append(c("bold", "VERDICT: ") + c(worst if worst in ("warn", "error") else "ok", verdict(report)))
    return "\n".join(lines)


def _md_escape(s: str) -> str:
    return s.replace("|", "\\|")


def render_markdown(report: Report) -> str:
    out = [f"# scpreflight report: `{report.source}`", ""]
    out.append(
        f"**{report.fmt}** · {report.n_obs:,} cells × {report.n_vars:,} features · "
        f"sampled {report.sampled_cells:,} cells · scpreflight {report.version}"
    )
    out += ["", f"**Verdict:** {verdict(report)}", "", "## Matrices", ""]
    out += ["| Slot | Contents | Confidence |", "|---|---|---|"]
    for m in report.matrices:
        extra = "; ".join(_extra(m.details))
        cell = _md_escape(m.label + (f" ({extra})" if extra else ""))
        out.append(f"| `{m.slot}` | {cell} | {m.confidence} |")
    findings = [f for f in _sorted(report.all_findings())]
    if findings:
        out += ["", "## Findings", ""]
        for f in findings:
            out.append(f"- **{f.level.upper()}** — {f.title}")
            if f.detail:
                out.append(f"  - {f.detail}")
            if f.advice:
                out.append(f"  - *Fix:* {f.advice}")
    if report.advice:
        out += ["", "## What to use for what", "", "| Task | Status | Use | Note |", "|---|---|---|---|"]
        for a in report.advice:
            out.append(
                f"| {_md_escape(a['task'])} | {a['status']} | `{_md_escape(a['use'])}` | {_md_escape(a['note'])} |"
            )
    return "\n".join(out) + "\n"
