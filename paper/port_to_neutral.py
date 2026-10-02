"""Build a venue-neutral, two-column version of the manuscript for review.

The body is taken from tse_latex/main.tex; the frame becomes a plain LaTeX
article in two columns: no journal name, no publisher class, no running head.

    python paper/port_to_neutral.py      # writes paper/neutral_latex/
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

PAPER = Path(__file__).resolve().parent
TSE = PAPER / "tse_latex"
OUT = PAPER / "neutral_latex"

src = (TSE / "main.tex").read_text(encoding="utf-8")


def between(text: str, start: str, end: str) -> str:
    a = text.index(start) + len(start)
    return text[a:text.index(end, a)]


title = between(src, r"\title{", "}\n\n\\author").replace("\\\\\n", " ").replace("\\\\", " ")
abstract = between(src, r"\begin{abstract}", r"\end{abstract}").strip()
keywords = " ".join(between(src, r"\begin{IEEEkeywords}", r"\end{IEEEkeywords}").split()).rstrip(".")
body = src[src.index(r"\IEEEpeerreviewmaketitle") + len(r"\IEEEpeerreviewmaketitle"):
           src.index(r"\end{document}")]

body = body.replace(r"\IEEEPARstart{M}{achine-learning}", "Machine-learning")
body = body.replace(r"\appendices", r"\appendix")  # IEEEtran -> standard LaTeX
body = body.replace(r"\begin{IEEEproof}[Proof sketch]", r"\begin{proof}[Proof sketch]")
body = body.replace(r"\end{IEEEproof}", r"\end{proof}")
body = body.replace(r"\bibliographystyle{IEEEtran}", r"\bibliographystyle{unsrtnat}")
assert "IEEE" not in re.sub(r"%.*", "", body), "an IEEE command is left in the body"

preamble = r"""%% Venue-neutral two-column version, generated from ../tse_latex/main.tex by
%% ../port_to_neutral.py; edit the source and re-run rather than editing this file.
\documentclass[10pt,twocolumn]{article}
\usepackage[margin=0.75in,columnsep=0.25in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amsthm}
\usepackage{newtxtext,newtxmath}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage[numbers,sort&compress]{natbib}
\usepackage[hidelinks]{hyperref}
\usepackage{url}
\usepackage{balance}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{microtype}
\usepackage[htt]{hyphenat}
\usepackage[font=small,labelfont=bf]{caption}
\tolerance=1200
\emergencystretch=1.5em
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.6}
\renewcommand{\textfraction}{0.08}
\renewcommand{\floatpagefraction}{0.75}
\renewcommand{\dbltopfraction}{0.9}
\renewcommand{\dblfloatpagefraction}{0.75}
\setcounter{topnumber}{3}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{5}
\setcounter{dbltopnumber}{2}
\newtheorem{definitionx}{Definition}
\newtheorem{observation}{Observation}
\hyphenation{mlcompass evi-dence-closed}

\title{""" + title + r"""}
\author{Hakan Sabuni\c{s}, Yusuf \"Unl\"u, Mehmet Kemal \"Ozdemir\\[2pt]
\small School of Engineering and Natural Sciences, Istanbul Medipol University, Istanbul, T\"urkiye\\
\small \texttt{hakan.sabunis@std.medipol.edu.tr}}
\date{}

\begin{document}
\twocolumn[
\begin{@twocolumnfalse}
\maketitle
\begin{abstract}
""" + abstract + r"""
\end{abstract}
\noindent\textbf{Keywords:} """ + keywords + r""".
\vspace{1.5em}
\end{@twocolumnfalse}
]
"""

OUT.mkdir(exist_ok=True)
(OUT / "main.tex").write_text(preamble + body + "\\end{document}\n", encoding="utf-8")
for name in ("references.bib", "ai-disclosure.tex", "fig1_contract.png"):
    shutil.copyfile(TSE / name, OUT / name)
print(f"wrote {OUT / 'main.tex'}")
