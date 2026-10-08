"""Build check for the 12-page version: pages, where the references start,
fonts, layout warnings, words, and the house rules (no 'it'; abstract length,
no dashes, hyphens or apostrophes in the abstract)."""
import pathlib
import re
import sys

import pymupdf

HERE = pathlib.Path(__file__).resolve().parent
pdf = pymupdf.open(HERE / "main.pdf")
pages = [p.get_text() for p in pdf]
HEAD = re.compile(r"^\s*(References|R\s*E\s*F\s*E\s*R\s*E\s*N\s*C\s*E\s*S)\s*$", re.M)
ref = next((i + 1 for i, t in enumerate(pages) if HEAD.search(t)), None)
fonts = sorted({f[3].split("+")[-1] for p in pdf for f in p.get_fonts()})
log = (HERE / "main.log").read_text(encoding="utf-8", errors="replace")
tex = (HERE / "main.tex").read_text(encoding="utf-8")

# where on its page the reference list starts, as a fraction of the page
frac = None
if ref:
    page = pdf[ref - 1]
    for b in page.get_text("blocks"):
        if HEAD.fullmatch(b[4].strip()):
            frac = b[1] / page.rect.height
print(f"pages {len(pages)}; references start on p{ref}"
      + (f" at {frac:.0%} of the page height" if frac is not None else ""))
print("text fonts:", [f for f in fonts if not re.match(r"CM|MSBM|MSAM", f)])
print("overfull:", len(re.findall(r"Overfull \\[hv]box", log)),
      "| undefined refs/cites:", sum(1 for line in log.splitlines()
                                       if "undefined" in line and "Font shape" not in line),
      "| font substitutions:", log.count("Font shape `TU/"))

body = tex[tex.index(r"\begin{document}"):tex.index(r"\bibliographystyle")]
body = re.sub(r"(?<!\\)%.*", "", body)
prose = re.sub(r"\\begin\{(table|figure)\*?\}.*?\\end\{\1\*?\}", " ", body, flags=re.S)
prose = re.sub(r"\\(cite[a-z]*|ref|label)\{[^}]*\}", " ", prose)
prose = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", prose)
prose = re.sub(r"[{}$\\&~]", " ", prose)
print("prose words (floats excluded):", len(prose.split()))

abstract = tex[tex.index(r"\begin{abstract}") + 16:tex.index(r"\end{abstract}")]
plain = re.sub(r"\s+", " ", abstract).strip()
problems = [c for c in ("--", "—", "–", "'", "’", "-") if c in plain]
print("abstract words:", len(plain.split()), "| forbidden characters:", problems or "none")

its = [m.start() for m in re.finditer(r"\b[Ii]t\b", re.sub(r"(?<!\\)%.*", "", body))]
print("pronoun 'it':", len(its))
for i in its[:10]:
    print("   ...", re.sub(r"\s+", " ", body[max(0, i - 60):i + 40]), "...")
eng = re.findall(r"\b[Ee]ngineer\w*", tex.replace("School of Engineering and Natural Sciences", ""))
print("'engineer' (affiliation excepted):", len(eng))
cited = {k.strip() for m in re.finditer(r"\\cite[a-z]*\{([^}]*)\}", tex) for k in m.group(1).split(",")}
print("cited keys:", len(cited))
sys.exit(0)
