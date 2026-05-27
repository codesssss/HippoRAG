#!/usr/bin/env python3
"""
Convert paper/sections/04_method_bilingual.html → PDF with rendered math.

Strategy:
  1. Read HTML
  2. Find $...$ inline and $$...$$ display math
  3. Convert each LaTeX snippet to MathML via latex2mathml (no JS needed)
  4. Replace the placeholders in HTML
  5. Strip the MathJax CDN <script> tags (no longer needed)
  6. Render to PDF via weasyprint
"""
import re
import sys
from pathlib import Path

import latex2mathml.converter
import weasyprint


SRC = Path("/mnt/nvme/code/HippoRAG/paper/sections/04_method_bilingual.html")
DST = Path("/mnt/nvme/code/HippoRAG/paper/sections/04_method_bilingual.pdf")


def latex_to_mathml(latex_src: str, display: bool = False) -> str:
    """Convert LaTeX math to MathML string. Fall back to plain text on error."""
    try:
        mml = latex2mathml.converter.convert(latex_src)
    except Exception as e:
        print(f"  [warn] latex2mathml failed on '{latex_src[:60]}...': {e}", file=sys.stderr)
        # Wrap raw LaTeX in <code> as fallback
        safe = latex_src.replace("<", "&lt;").replace(">", "&gt;")
        return f"<code>${safe}$</code>"
    # latex2mathml returns <math display="inline">...</math> by default.
    # Patch display mode if needed.
    if display:
        mml = mml.replace('display="inline"', 'display="block"')
        if 'display="block"' not in mml:
            mml = mml.replace("<math", '<math display="block"', 1)
    return mml


def convert(html: str) -> str:
    # 1) Drop MathJax CDN scripts (no longer needed; latex2mathml is server-side)
    html = re.sub(r"<script[^>]*MathJax[^<]*</script>", "", html, flags=re.IGNORECASE)
    html = re.sub(r"<script>[^<]*window\.MathJax[^<]*</script>", "", html, flags=re.IGNORECASE | re.DOTALL)
    html = re.sub(r'<script async src="https://cdn\.jsdelivr\.net/npm/mathjax[^"]*"></script>', "", html)

    # 2) Display math $$...$$ first (longer pattern)
    def repl_display(m: re.Match) -> str:
        return latex_to_mathml(m.group(1).strip(), display=True)
    html = re.sub(r"\$\$([^$]+?)\$\$", repl_display, html, flags=re.DOTALL)

    # 3) Inline math $...$ — careful not to match $K$ inside attributes
    # Strategy: match $ ... $ where neither side touches alphanumeric
    def repl_inline(m: re.Match) -> str:
        return latex_to_mathml(m.group(1).strip(), display=False)
    # Match $...$ where content has no $
    html = re.sub(r"\$([^$\n]+?)\$", repl_inline, html)

    return html


def add_print_css(html: str) -> str:
    """Inject PDF-friendly CSS overrides."""
    extra = """
    <style>
      @page { size: A4; margin: 16mm 14mm; }
      body { max-width: none !important; margin: 0 !important; padding: 0 !important; font-size: 10.5pt; }
      h1 { font-size: 18pt; }
      h2 { font-size: 13pt; page-break-after: avoid; }
      table { page-break-inside: avoid; font-size: 10pt; }
      td { padding: 8px 10px; }
      math { font-size: 11pt; }
      math[display="block"] { display: block; margin: 6px auto; }
      .eq { padding: 6px 10px; }
      .toc { page-break-after: avoid; }
    </style>
    """
    return html.replace("</head>", extra + "</head>", 1)


def main() -> None:
    print(f"Reading: {SRC}")
    html = SRC.read_text(encoding="utf-8")
    print(f"  HTML size: {len(html):,} chars")

    print("Converting LaTeX math → MathML ...")
    html = convert(html)
    html = add_print_css(html)

    # Save intermediate (debug)
    pre_pdf = SRC.with_suffix(".prepdf.html")
    pre_pdf.write_text(html, encoding="utf-8")
    print(f"  Pre-PDF HTML saved: {pre_pdf}")

    print(f"Rendering PDF → {DST}")
    weasyprint.HTML(string=html, base_url=str(SRC.parent)).write_pdf(DST)
    print(f"  PDF size: {DST.stat().st_size:,} bytes")
    print("Done.")


if __name__ == "__main__":
    main()
