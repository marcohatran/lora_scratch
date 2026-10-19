"""Ghép slide của một deck thành 1 file HTML rồi in ra PDF bằng Chrome headless.

Chạy: .venv/bin/python slides/build_pdf.py en vi
"""
import json, os, subprocess, sys

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
HERE = os.path.dirname(os.path.abspath(__file__))

# shortcut: chỉ dựng lại các phần tử mà deck này dùng (ellipse, arrow-right, connector), đủ cho PDF
SHIM_CSS = """
@page { size: 1920px 1080px; margin: 0 }
body { margin: 0 }
section { width: 1920px; height: 1080px; box-sizing: border-box; position: relative; overflow: hidden; break-after: page; }
section * { margin: 0; box-sizing: border-box; }
aside { display: none !important; }
x-shape { display: block; flex: none; }
x-shape[kind=ellipse] { border-radius: 50%; }
x-shape[kind=arrow-right] { clip-path: polygon(0 30%, 60% 30%, 60% 0, 100% 50%, 60% 100%, 60% 70%, 0 70%); }
table { border-collapse: collapse; }
th, td { padding: 0.35em 0.6em; border-bottom: 1px solid #D6D1C6; }
th { font-weight: 600; }
"""

SHIM_JS = """
document.querySelectorAll('x-connector').forEach(c => {
  const n = k => parseFloat(c.getAttribute(k));
  const [x1, y1, x2, y2] = ['x1', 'y1', 'x2', 'y2'].map(n);
  const route = c.getAttribute('route') || 'straight';
  const pts = route === 'hv' ? [[x1, y1], [x2, y1], [x2, y2]]
            : route === 'vh' ? [[x1, y1], [x1, y2], [x2, y2]]
            : route === 'elbow' ? [[x1, y1], [(x1 + x2) / 2, y1], [(x1 + x2) / 2, y2], [x2, y2]]
            : [[x1, y1], [x2, y2]];
  const color = c.style.color || '#000', w = parseFloat(c.style.borderWidth) || 2;
  const id = 'm' + Math.random().toString(36).slice(2);
  const svg = `<svg width="1920" height="1080" style="position:absolute;left:0;top:0;pointer-events:none">
    <defs><marker id="${id}" markerWidth="5" markerHeight="5" refX="4.5" refY="2.5" orient="auto">
      <path d="M0,0 L5,2.5 L0,5 z" fill="${color}"/></marker></defs>
    <polyline points="${pts.map(p => p.join(',')).join(' ')}" fill="none" stroke="${color}" stroke-width="${w}"
      ${c.getAttribute('head') === 'none' ? '' : `marker-end="url(#${id})"`}/></svg>`;
  c.insertAdjacentHTML('beforebegin', svg);
  c.remove();
});
"""


def build(lang):
    deck_dir = os.path.join(HERE, lang)
    deck = json.load(open(os.path.join(deck_dir, "deck.json"), encoding="utf-8"))
    fonts = "\n".join(f'<link rel="stylesheet" href="{f["href"]}">' for f in deck["faces"].values() if "href" in f)
    sections = "\n".join(open(os.path.join(deck_dir, "slides", f"{sid}.html"), encoding="utf-8").read()
                         for sid in deck["order"])
    html = (f'<!doctype html><html lang="{lang}"><head><meta charset="utf-8"><title>{deck["title"]}</title>'
            f"{fonts}<style>{SHIM_CSS}</style></head><body>{sections}<script>{SHIM_JS}</script></body></html>")
    html_path = os.path.join(HERE, f"LoRA_Fine-tuning_{lang.upper()}.html")
    pdf_path = html_path[:-5] + ".pdf"
    open(html_path, "w", encoding="utf-8").write(html)
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=15000",
                    f"--print-to-pdf={pdf_path}", "file://" + html_path], check=True, capture_output=True)
    print("saved", pdf_path)


if __name__ == "__main__":
    for lang in sys.argv[1:] or ["en", "vi"]:
        build(lang)
