"""
report.py
---------
Renders an AnalysisResult into a standalone, printable HTML report that a user
can download and share. It has no external CSS/JS — everything is inlined — so
the downloaded file opens correctly offline and prints to PDF from the browser.
"""

from __future__ import annotations

import html
from datetime import datetime

_SEVERITY_COLORS = {"high": "#c0392b", "medium": "#d68910", "low": "#7d8a99"}
_LEVEL_COLORS = {"High": "#c0392b", "Medium": "#d68910", "Low": "#1e8449"}


def _esc(x) -> str:
    return html.escape(str(x)) if x is not None else ""


def _entities_rows(entities: dict) -> str:
    rows = []
    labels = {
        "parties": "Parties",
        "effective_date": "Effective date",
        "term": "Term",
        "governing_law": "Governing law",
        "notice_periods": "Notice periods",
        "monetary_amounts": "Monetary amounts",
        "dates": "Dates mentioned",
    }
    for key, label in labels.items():
        val = entities.get(key)
        if not val:
            continue
        if isinstance(val, list):
            val = ", ".join(str(v) for v in val)
        rows.append(f"<tr><th>{_esc(label)}</th><td>{_esc(val)}</td></tr>")
    return "\n".join(rows) or "<tr><td colspan='2'>No key entities extracted.</td></tr>"


def build_html_report(result) -> str:
    """Return a full HTML document string for the given AnalysisResult."""
    r = result
    risk = r.risk
    level_color = _LEVEL_COLORS.get(risk.level, "#333")

    findings_html = []
    for f in risk.findings:
        color = _SEVERITY_COLORS.get(f.severity, "#333")
        evidence = f"<div class='evidence'>{_esc(f.evidence)}</div>" if f.evidence else ""
        findings_html.append(
            f"""
            <div class="finding">
              <span class="sev" style="background:{color}">{_esc(f.severity.upper())}</span>
              <strong>{_esc(f.title)}</strong>
              <p>{_esc(f.detail)}</p>
              <p class="rec"><em>Recommendation:</em> {_esc(f.recommendation)}</p>
              {evidence}
            </div>
            """
        )
    findings_block = "\n".join(findings_html) or "<p>No risk findings.</p>"

    clause_rows = []
    for c in r.clauses:
        if not c.clause_type:
            continue
        clause_rows.append(
            f"<tr><td>{_esc(c.section_number)}</td><td>{_esc(c.heading)}</td>"
            f"<td>{_esc(c.clause_type)}</td><td>{c.confidence:.0%}</td>"
            f"<td>{_esc(c.method)}</td></tr>"
        )
    clause_table = "\n".join(clause_rows) or "<tr><td colspan='5'>No clauses classified.</td></tr>"

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Contract Analysis — {_esc(r.filename)}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
         color: #1f2933; max-width: 900px; margin: 2rem auto; padding: 0 1.25rem; line-height: 1.5; }}
  h1 {{ font-size: 1.6rem; margin-bottom: 0.2rem; }}
  h2 {{ font-size: 1.15rem; margin-top: 2rem; border-bottom: 2px solid #eef2f7; padding-bottom: 0.3rem; }}
  .muted {{ color: #7d8a99; font-size: 0.85rem; }}
  .risk-badge {{ display:inline-block; padding: 0.35rem 0.9rem; border-radius: 999px; color:#fff;
                font-weight:600; background:{level_color}; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 0.5rem; font-size: 0.9rem; }}
  th, td {{ border: 1px solid #e4e9f0; padding: 0.45rem 0.6rem; text-align: left; vertical-align: top; }}
  th {{ background: #f7f9fc; }}
  .finding {{ border:1px solid #e4e9f0; border-left:4px solid #ccc; border-radius:6px;
             padding:0.7rem 0.9rem; margin:0.6rem 0; }}
  .sev {{ color:#fff; font-size:0.7rem; font-weight:700; padding:0.1rem 0.5rem; border-radius:4px; margin-right:0.5rem; }}
  .rec {{ font-size:0.88rem; }}
  .evidence {{ background:#f7f9fc; border-radius:4px; padding:0.4rem 0.6rem; font-size:0.8rem;
              color:#52606d; margin-top:0.4rem; font-family: ui-monospace, monospace; }}
  .overview {{ background:#f7f9fc; border-radius:8px; padding:1rem; }}
</style></head><body>
  <h1>Contract Analysis Report</h1>
  <p class="muted">{_esc(r.filename)} &middot; generated {generated} &middot;
     classifier: {"ML model" if r.used_ml else "keyword baseline"}</p>

  <h2>Overview</h2>
  <div class="overview">{_esc(r.overview)}</div>

  <h2>Risk Assessment</h2>
  <p><span class="risk-badge">{_esc(risk.level)} risk &middot; {risk.score}/100</span>
     &nbsp; <span class="muted">{risk.counts.get('high',0)} high &middot;
     {risk.counts.get('medium',0)} medium &middot; {risk.counts.get('low',0)} low</span></p>
  {findings_block}

  <h2>Key Entities</h2>
  <table>{_entities_rows(r.entities.to_dict())}</table>

  <h2>Detected Clauses</h2>
  <table>
    <tr><th>#</th><th>Heading</th><th>Clause type</th><th>Confidence</th><th>Method</th></tr>
    {clause_table}
  </table>

  <h2>Key Points (extractive)</h2>
  <p>{_esc(r.key_points)}</p>

  <p class="muted" style="margin-top:2rem">Generated by Legal Contract Analyzer.
     This automated analysis is for informational purposes only and is not legal advice.</p>
</body></html>"""
