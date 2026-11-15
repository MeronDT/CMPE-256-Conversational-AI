"""Render report/report.md to report/report.html and report/report.pdf.

    python -m scripts.build_report

The PDF is printed with a headless Chromium browser (Edge or Chrome). Set
BROWSER=/path/to/chrome if neither is found in the default locations.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report"

CSS = """
@page { size: Letter; margin: 0.7in 0.75in; }
body { font-family: "Segoe UI", Inter, Arial, sans-serif; color: #1d1d1b; font-size: 10.5pt; line-height: 1.45; max-width: 7in; margin: 0 auto; }
h1 { font-size: 19pt; margin: 0 0 4pt; line-height: 1.2; }
h2 { font-size: 13.5pt; margin: 18pt 0 6pt; border-bottom: 1px solid #d9d8d2; padding-bottom: 3pt; break-after: avoid; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt; break-after: avoid; }
p, li { margin: 4pt 0; }
table { border-collapse: collapse; margin: 8pt 0; font-size: 9.5pt; width: 100%; break-inside: avoid; }
th, td { border-bottom: 1px solid #e3e2dc; padding: 3pt 6pt; text-align: left; vertical-align: top; }
th { background: #f3f3ef; font-weight: 600; }
img { width: 100%; margin: 6pt 0; break-inside: avoid; }
code { font-family: Consolas, monospace; font-size: 9pt; background: #f3f3ef; padding: 0 2pt; border-radius: 2px; }
pre { background: #f3f3ef; padding: 6pt 8pt; border-radius: 4px; font-size: 9pt; }
pre code { background: none; padding: 0; }
"""


def find_browser() -> str:
    candidates = [
        os.getenv("BROWSER", ""),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        shutil.which("google-chrome") or "", shutil.which("chromium") or "", shutil.which("msedge") or "",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    raise SystemExit("No Chromium browser found; set BROWSER=/path/to/chrome")


def main():
    md = (REPORT / "report.md").read_text(encoding="utf-8")
    body = markdown.markdown(md, extensions=["tables", "fenced_code"])
    html = f"<!doctype html><html><head><meta charset='utf-8'><title>FRED Econ Chat report</title><style>{CSS}</style></head><body>{body}</body></html>"
    out_html = REPORT / "report.html"
    out_html.write_text(html, encoding="utf-8")
    pdf = REPORT / "report.pdf"
    subprocess.run([find_browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", out_html.resolve().as_uri()], check=True, capture_output=True)
    print("wrote", pdf)


if __name__ == "__main__":
    main()
