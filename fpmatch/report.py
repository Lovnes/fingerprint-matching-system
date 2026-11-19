"""Self-contained HTML case report (opens in any browser, print to PDF)."""
from __future__ import annotations

import base64
import html
import platform

import cv2

from . import __version__
from .database import now

CSS = """
body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:1000px;margin:24px auto;color:#1d1d1f;padding:0 16px}
h1{font-size:22px;margin-bottom:2px} h2{font-size:16px;border-bottom:1px solid #ddd;padding-bottom:4px;margin-top:28px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0} td,th{border:1px solid #ddd;padding:5px 7px;text-align:left;vertical-align:top}
th{background:#f4f4f6} .mono{font-family:Menlo,Consolas,monospace;font-size:11.5px;word-break:break-all}
.v-green{color:#137333;font-weight:600}.v-orange{color:#b06000;font-weight:600}.v-gray{color:#5f6368;font-weight:600}
img{max-width:100%;border:1px solid #ddd} .note{background:#fff8e1;border:1px solid #f0d58c;padding:10px 12px;font-size:12.5px;margin-top:24px}
.small{font-size:12px;color:#555}
"""


def _img(bgr_or_gray) -> str:
    ok, buf = cv2.imencode(".png", bgr_or_gray)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf).decode()}">'


def _kv(rows):
    return "<table>" + "".join(f"<tr><th style='width:28%'>{html.escape(str(k))}</th>"
                               f"<td>{v}</td></tr>" for k, v in rows) + "</table>"


def build(case: dict, mode: str, query: dict, settings: dict, results: list[dict],
          query_title: str = "Questioned print", ref_title: str = "Reference") -> str:
    """
    case:    case_no, examiner, agency, exhibit, remarks
    query:   filename, sha256, minutiae, quality, image(bgr)
    results: dicts with label, sha256/subject, score, paired, supported, verdict, colour, image(bgr, optional)
    """
    e = lambda s: html.escape(str(s or ""))
    has_ml = any(r.get("ml") is not None for r in results)
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>Fingerprint report {e(case.get('case_no'))}</title>"
             f"<style>{CSS}</style></head><body>",
             f"<h1>Fingerprint Comparison Report</h1><div class='small'>Generated {now()} UTC by FP-Match v{__version__} "
             f"(Python {platform.python_version()})</div>",
             "<h2>Case details</h2>",
             _kv([("Case / FIR no.", e(case.get("case_no"))), ("Exhibit / lift no.", e(case.get("exhibit"))),
                  ("Examiner", e(case.get("examiner"))), ("Agency / lab", e(case.get("agency"))),
                  ("Analysis type", e(mode)), ("Remarks", e(case.get("remarks")))]),
             f"<h2>{e(query_title)}</h2>",
             _kv([("File", e(query["filename"])), ("SHA-256", f"<span class='mono'>{query['sha256']}</span>"),
                  ("Minutiae extracted", query["minutiae"]), ("Image quality (0-100)", query["quality"])]),
             "<h2>Processing settings</h2>",
             _kv([(k, html.escape(str(v))) for k, v in settings.items()]),
             "<h2>Results</h2>",
             f"<table><tr><th>#</th><th>{e(ref_title)}</th>" + ("<th>ML score</th>" if has_ml else "")
             + "<th>Minutiae score</th><th>Paired</th><th>Supported</th><th>Indication</th></tr>"]
    for k, r in enumerate(results, 1):
        ml = f"<td>{r['ml']:.1f}</td>" if has_ml and r.get("ml") is not None else "<td>-</td>" if has_ml else ""
        parts.append(f"<tr><td>{k}</td><td>{e(r['label'])}<div class='mono'>{e(r.get('sha256', ''))}</div></td>{ml}"
                     f"<td>{r['score']:.1f}</td><td>{r['paired']}</td><td>{r['supported']}</td>"
                     f"<td class='v-{r['colour']}'>{e(r['verdict'])}</td></tr>")
    parts.append("</table>")
    for k, r in enumerate(results, 1):
        if r.get("image") is not None:
            parts.append(f"<h2>Comparison #{k}: {e(r['label'])}</h2>"
                         f"<div class='small'>Left: {e(query_title.lower())}. Right: {e(ref_title.lower())}. Green lines: paired minutiae whose "
                         "neighbourhoods also agree. Yellow: paired by position/direction only. "
                         "Circles = ridge endings, squares = bifurcations.</div>" + _img(r["image"]))
    if query.get("image") is not None:
        parts.append(f"<h2>{e(query_title)} - extracted minutiae</h2>" + _img(query["image"]))
    parts.append(
        "<div class='note'><b>Important.</b> This is an automated screening result produced to prioritise "
        "candidates. Scores are similarity measures, not probabilities, and are not an identification. Any "
        "candidate must be examined by a qualified fingerprint examiner following ACE-V (Analysis, Comparison, "
        "Evaluation, Verification) before an opinion is given. SHA-256 hashes above identify the exact files "
        "analysed; source files were not modified."
        + (" The ML score comes from a neural network plus learned fusion whose model file is identified by its "
           "SHA-256 under Processing settings; it is a similarity score, not a probability of identity." if has_ml else "")
        + "</div></body></html>")
    return "".join(parts)
