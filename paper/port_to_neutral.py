"""Build a venue-neutral, two-column version of the manuscript for review.

The body is taken from tse_latex/main.tex; the frame becomes a plain LaTeX
article in two columns: no journal name, no publisher class, no running head.

    python paper/port_to_neutral.py      # writes paper/neutral_latex/
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from author_year import TEXTUAL, author_year  # noqa: E402

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
body = body.replace(r"\bibliographystyle{IEEEtran}", r"\bibliographystyle{plainnat}")
DECLARATIONS = r"""
\section*{Statements and Declarations}
\paragraph{Funding.} No funding was received for conducting this study.

\paragraph{Competing interests.} The authors have no competing interests to
declare that are relevant to the content of this article. mlcompass is released
by the authors under the MIT license and is not a commercial product.

\paragraph{Ethics approval.} Not applicable: the study involved no human
participants and no animals. The registered blind audit was not labeled by
people; model judges labeled its items, as the paper reports.

\paragraph{Consent to participate and to publish.} Not applicable.

\paragraph{Author contributions.} H.S. designed and implemented the contract
and the measurement harnesses, and ran the batteries. Y.\"U. designed the
downstream benchmark protocol and performed the independent review that
identified scorer false positives. M.K.\"O. supervised the work. All authors
read and approved the final manuscript.

\paragraph{Code availability.} The source of mlcompass, every harness and every
analysis script are in the replication package~\cite{replication} under the
MIT license; the development repository is
\url{https://github.com/hakansabunis/mlcompass}.

"""
assert body.count("\\balance") == 1
body = body.replace("\\balance", DECLARATIONS + "\\balance")
body = author_year(body, TEXTUAL)  # name and year, as EMSE asks
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
\usepackage[round]{natbib}
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
\small \texttt{hakansabunis@gmail.com}, \texttt{ysffms@gmail.com}, \texttt{mkozdemir@medipol.edu.tr}}
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
