#!/usr/bin/env python3
"""Render the current bilingual paper draft HTML to PDF.

The source HTML keeps LaTeX-style math for readability. This script converts
inline and display math to MathML before rendering with WeasyPrint.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import latex2mathml.converter
import weasyprint


SRC = Path("/mnt/nvme/code/HippoRAG/paper/current_bilingual.html")
DST = Path("/mnt/nvme/code/HippoRAG/paper/current_bilingual.pdf")


def latex_to_mathml(latex_src: str, *, display: bool = False) -> str:
    try:
        mml = latex2mathml.converter.convert(latex_src)
    except Exception as exc:
        print(f"[warn] latex2mathml failed on {latex_src[:80]!r}: {exc}", file=sys.stderr)
        safe = latex_src.replace("<", "&lt;").replace(">", "&gt;")
        return f"<code>${safe}$</code>"
    if display:
        mml = mml.replace('display="inline"', 'display="block"')
        if 'display="block"' not in mml:
            mml = mml.replace("<math", '<math display="block"', 1)
    return mml


def convert_math(html: str) -> str:
    def repl_display(match: re.Match[str]) -> str:
        return latex_to_mathml(match.group(1).strip(), display=True)

    def repl_inline(match: re.Match[str]) -> str:
        return latex_to_mathml(match.group(1).strip(), display=False)

    html = re.sub(r"\$\$([^$]+?)\$\$", repl_display, html, flags=re.DOTALL)
    html = re.sub(r"\$([^$\n]+?)\$", repl_inline, html)
    return html


def add_print_css(html: str) -> str:
    extra = """
    <style>
      @page { size: A4; margin: 13mm 11mm; }
      body { max-width: none !important; margin: 0 !important; padding: 0 !important; font-size: 9.7pt; }
      h1 { font-size: 18pt; }
      h2 { font-size: 13.5pt; page-break-after: avoid; }
      h3 { font-size: 11.5pt; page-break-after: avoid; }
      table.pair { page-break-inside: avoid; }
      td { vertical-align: top; }
      math { font-size: 10pt; }
      math[display="block"] { display: block; margin: 5px auto; }
      .section-note { page-break-inside: avoid; }
      .data-table { font-size: 8.7pt; }
    </style>
    """
    return html.replace("</head>", extra + "</head>", 1)


def main() -> None:
    html = SRC.read_text(encoding="utf-8")
    html = html.replace("$\\\\methodname$", "EvLink")
    html = html.replace("$\\\\methodname{}$", "EvLink")
    html = add_print_css(convert_math(html))
    pre_pdf = SRC.with_suffix(".prepdf.html")
    pre_pdf.write_text(html, encoding="utf-8")
    weasyprint.HTML(string=html, base_url=str(SRC.parent)).write_pdf(DST)
    print(f"Wrote {DST} ({DST.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
