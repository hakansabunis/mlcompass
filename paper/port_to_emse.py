"""Port the IEEE TSE manuscript to the Springer sn-jnl class for EMSE.

The body is taken verbatim from tse_latex/main.tex; only the frame changes:
preamble, title block, abstract and keywords, the IEEE-only commands, table
scaling (a two-column \\columnwidth is not a one-column page), and the back
matter Springer requires. Run it again after any edit to the TSE source.

    python paper/port_to_emse.py
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

PAPER = Path(__file__).resolve().parent
TSE = PAPER / "tse_latex"
EMSE = PAPER / "emse_latex"

src = (TSE / "main.tex").read_text(encoding="utf-8")


def between(text: str, start: str, end: str) -> str:
    a = text.index(start) + len(start)
    return text[a : text.index(end, a)]


abstract = between(src, r"\begin{abstract}", r"\end{abstract}").strip()
keywords = " ".join(between(src, r"\begin{IEEEkeywords}", r"\end{IEEEkeywords}").split())
keywords = keywords.rstrip(".")
body = between(src, r"\IEEEpeerreviewmaketitle", r"\section*{Acknowledgments}")
acks = between(src, r"\section*{Acknowledgments}", r"\input{ai-disclosure}").strip()
data_avail = between(src, r"\section*{Data Availability}", r"\balance").strip()

# IEEE-only commands
body = body.replace(r"\IEEEPARstart{M}{achine-learning}", "Machine-learning")
body = body.replace(r"\begin{IEEEproof}", r"\begin{proof}").replace(r"\end{IEEEproof}", r"\end{proof}")

# A table scaled to a two-column \columnwidth would be blown up on a one-column
# page, so the scaling goes; adjustbox's max width is not an option, because
# inside sn-jnl's table it raises "Missing \endgroup inserted". The two tables
# set for the IEEE full text width are narrowed or scaled by hand.
body = re.sub(
    r"\\resizebox\{\\columnwidth\}\{!\}\{%\s*\n(\\begin\{tabular\}.*?\\end\{tabular\})\}",
    lambda m: m.group(1),
    body,
    flags=re.S,
)
assert r"\resizebox" not in body, "a scaled table was not ported"


def once(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, f"expected one {old[:50]!r}"
    return text.replace(old, new)


body = once(body, r"\begin{tabular}{@{}lp{6.1cm}p{9.5cm}@{}}", r"\begin{tabular}{@{}lp{4.3cm}p{6.8cm}@{}}")
a = body.index(r"\label{tab:sweep}")
t0 = body.index(r"\begin{tabular}", a)
t1 = body.index(r"\end{tabular}", t0) + len(r"\end{tabular}")
body = body[:t0] + "\\resizebox{\\textwidth}{!}{%\n" + body[t0:t1] + "}" + body[t1:]

# EMSE has no page limit, so it carries what the TSE version had no room for:
# the per-instance counts of the twelve frozen instances (review 2026-10-01).
body = once(
    body,
    "instances holds on twelve that played no part in the design.",
    "instances holds on twelve that played no part in the design;\n"
    "Table~\\ref{tab:fabbench12} gives each one.\n\n\\input{table_fabbench12}",
)

# A top-only placement sent six tables past the references. Starred tables
# stay starred: inside sn-jnl's unstarred table a boxed tabular raises
# "Missing \endgroup inserted", inside the starred one it does not.
body = body.replace(r"\begin{table*}[t]", r"\begin{table*}[!tbp]")
body = body.replace(r"\begin{table}[t]", r"\begin{table}[!htbp]")
body = body.replace(r"\begin{figure}[t]", r"\begin{figure}[!htbp]")
assert "IEEE" not in body.replace("IEEE policy", ""), "an IEEE command is left in the body"

preamble = r"""%% Empirical Software Engineering (Springer) submission.
%% Generated from ../tse_latex/main.tex by ../port_to_emse.py; edit the TSE
%% source and re-run the script rather than editing this file.
%% Compile: tectonic -X compile main.tex
% The pdflatex option is required: without it sn-jnl loads breakurl, which
% fails under any engine that writes PDF directly (see BUILD.md).
\documentclass[pdflatex,sn-mathphys,Numbered]{sn-jnl}

\usepackage{graphicx}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{amsthm}
\usepackage{booktabs}

\usepackage{xcolor}
\usepackage{textcomp}
\usepackage{url}
\usepackage{manyfoot}   % required by sn-jnl's footnote hook (see BUILD.md)
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{microtype}
\usepackage[htt]{hyphenat}
\tolerance=1200
\emergencystretch=2em
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.7}
\renewcommand{\textfraction}{0.07}
\renewcommand{\floatpagefraction}{0.7}
\setcounter{topnumber}{3}
\setcounter{totalnumber}{4}

\theoremstyle{definition}
\newtheorem{definitionx}{Definition}
\theoremstyle{plain}
\newtheorem{observation}{Observation}

\hyphenation{mlcompass evi-dence-closed}

\begin{document}

\title[An Evidence-Bound Runtime Contract for LLM Narration]{An Evidence-Bound
Runtime Contract for LLM Narration of Machine-Learning Pipeline Evidence}

\author*[1]{\fnm{Hakan} \sur{Sabuni\c{s}}}\email{hakan.sabunis@std.medipol.edu.tr}
\author[1]{\fnm{Yusuf} \sur{\"Unl\"u}}\email{yusuf.unlu@std.medipol.edu.tr}
\author[1]{\fnm{Mehmet Kemal} \sur{\"Ozdemir}}\email{mkozdemir@medipol.edu.tr}

\affil[1]{\orgdiv{School of Engineering and Natural Sciences},
\orgname{Istanbul Medipol University},
\orgaddress{\city{Istanbul}, \country{T\"urkiye}}}

"""

backmatter = r"""
\backmatter

\bmhead{Acknowledgements}
""" + acks + r"""

\section*{Declarations}

\paragraph{Ethics approval and consent.} Not applicable: this study involved
no human participants and no animals.

\paragraph{Funding.} No funding was received for conducting this study.

\paragraph{Competing interests.} The authors declare no competing interests.
mlcompass is released by the authors under the MIT licence and is not a
commercial product.

\paragraph{Author contributions.} H.S. designed and implemented the contract
and the measurement harnesses, and ran the batteries. Y.\"U. designed the
downstream benchmark protocol and performed the independent review that
identified scorer false positives. M.K.\"O. supervised the work. All authors
read and approved the final manuscript.

\paragraph{Data availability.} """ + data_avail + r"""

\input{ai-disclosure}

\bibliography{references}

\end{document}
"""

out = (
    preamble
    + "\\abstract{%\n" + abstract + "}\n\n"
    + "\\keywords{" + keywords + "}\n\n"
    + "\\maketitle\n"
    + body.rstrip() + "\n"
    + backmatter
)
(EMSE / "main.tex").write_text(out, encoding="utf-8")
for name in ("references.bib", "ai-disclosure.tex", "fig1_contract.png"):
    shutil.copyfile(TSE / name, EMSE / name)
shutil.copyfile(EMSE / "_tpl" / "sn-mathphys.bst", EMSE / "sn-mathphys.bst")
print(f"wrote {EMSE / 'main.tex'} ({len(out)} chars)")
