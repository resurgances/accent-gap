"""
Turns README.md (the write-up) into WRITEUP.pdf, with the charts embedded, for forms that
want a write-up FILE. Uses the `markdown` package for the HTML and headless Google Chrome
to print it to PDF. Run:  .venv/bin/python src/make_writeup_pdf.py
"""
import os
import subprocess

import markdown

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
REPO = "https://github.com/resurgances/accent-gap"

CSS = """
body { font-family: -apple-system, Helvetica, Arial, sans-serif; font-size: 10.5pt; line-height: 1.45;
       color: #1a1a1a; max-width: 760px; margin: 0 auto; }
h1 { font-size: 22pt; margin-bottom: 4px; } h2 { font-size: 15pt; border-bottom: 1px solid #ccc;
     padding-bottom: 3px; margin-top: 22px; } h3 { font-size: 12pt; margin-top: 16px; }
table { border-collapse: collapse; margin: 8px 0; font-size: 9.5pt; }
th, td { border: 1px solid #bbb; padding: 3px 7px; text-align: left; vertical-align: top; }
th { background: #f0f0f0; } img { max-width: 100%; display: block; margin: 8px auto; }
code { background: #f3f3f3; padding: 0 3px; border-radius: 3px; font-size: 9pt; }
pre { background: #f3f3f3; padding: 8px; overflow-x: auto; font-size: 8.5pt; white-space: pre-wrap; }
blockquote { border-left: 3px solid #ccc; margin: 8px 0; padding: 2px 10px; color: #444; }
a { color: #0b57d0; text-decoration: none; }
h2, h3, table, img { page-break-inside: avoid; } h2, h3 { page-break-after: avoid; }
"""


def main():
    text = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    # GitHub accepts 3-space indents for lists inside lists; this converter needs 4.
    lines = []
    for line in text.split("\n"):
        n = len(line) - len(line.lstrip(" "))
        if n in (3, 5) and line.strip():
            line = " " + line
        if line.startswith("**Track:**"):
            line += "  "  # two trailing spaces = line break, so Team goes on its own line
        lines.append(line)
    body = markdown.markdown("\n".join(lines), extensions=["tables", "fenced_code"])
    # links to repo files should work from the PDF too: point them at GitHub
    body = body.replace('href="results/', f'href="{REPO}/blob/main/results/')
    body = body.replace('href="src/', f'href="{REPO}/blob/main/src/')
    html_path = os.path.join(ROOT, "WRITEUP.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(f"<!doctype html><html><head><meta charset='utf-8'><title>Accent Gap</title>"
                f"<style>{CSS}</style></head><body>{body}"
                f"<p style='margin-top:24px;color:#666'>Code, data processing and all result files: "
                f"<a href='{REPO}'>{REPO}</a></p></body></html>")
    pdf_path = os.path.join(ROOT, "WRITEUP.pdf")
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf_path}", "file://" + html_path], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.remove(html_path)
    print("wrote", pdf_path)


if __name__ == "__main__":
    main()
