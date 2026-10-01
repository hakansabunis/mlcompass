"""Port the IEEE TSE manuscript to Springer's SVJour3 class for EMSE.

SVJour3 is the template EMSE's own FAQ points to and the LaTeX package its
submission page links (paper/emse_latex/svjour3.cls, from Springer's
LaTeX_DL_468198 package). The body is taken verbatim from tse_latex/main.tex;
only the frame changes: preamble, title block, abstract and keywords, the
IEEE-only commands, table scaling (a two-column \\columnwidth is not a
one-column page), and what the journal's guidelines require. Run it again
after any edit to the TSE source.

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
# SVJour3 predefines `definition` and its own `proof`; amsthm cannot be loaded
# beside it, so the sketch gets an environment of its own.
body = body.replace(r"\begin{IEEEproof}[Proof sketch]", r"\begin{proofsketch}")
body = body.replace(r"\end{IEEEproof}", r"\end{proofsketch}")
body = body.replace(r"\begin{definitionx}", r"\begin{definition}").replace(r"\end{definitionx}", r"\end{definition}")

# A table scaled to a two-column \columnwidth would be blown up on a one-column
# page, so the scaling goes here; every plain tabular is given a maximum width
# further down, which shrinks only the ones wider than the text. The two tables
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


# The enforcement taxonomy ran 254pt past the margin in V2.1: its last two
# columns now wrap.
body = once(body, r"\begin{tabular}{@{}lllll@{}}" + "\n\\toprule\nLocus",
            r"\setlength{\tabcolsep}{4pt}"
            r"\begin{tabular}{@{}lll>{\raggedright\arraybackslash}p{0.22\textwidth}"
            r">{\raggedright\arraybackslash}p{0.33\textwidth}@{}}" + "\n\\toprule\nLocus")
body = once(body, r"\begin{tabular}{@{}lp{5.4cm}p{8.9cm}l@{}}",
            r"\begin{tabular}{@{}l>{\raggedright\arraybackslash}p{3.2cm}"
            r">{\raggedright\arraybackslash}p{4.85cm}l@{}}")
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

# The registered-items table is taller than a page in the one-column format and
# ran off its page (EMSE review 2026-10-01): it becomes a longtable that breaks
# across pages, with its header repeated.
lab = body.index(r"\label{tab:registered}")
t0 = body.rindex(r"\begin{table*}", 0, lab)
t1 = body.index(r"\end{table*}", lab) + len(r"\end{table*}")
block = body[t0:t1]
cap0 = block.index(r"\caption{")
k, depth = cap0 + len(r"\caption{"), 1
while depth:
    depth += {"{": 1, "}": -1}.get(block[k], 0)
    k += 1
caption = block[cap0:k]
spec_at = block.index(r"\begin{tabular}")
spec = block[spec_at + len(r"\begin{tabular}"):block.index("\n", spec_at)]
rows_at = block.index(r"\midrule", spec_at) + len(r"\midrule")
header = block[block.index(r"\toprule", spec_at) + len(r"\toprule"):block.index(r"\midrule", spec_at)].strip()
rows = block[rows_at:block.index(r"\bottomrule", rows_at)].strip()
long_table = (
    "{\\footnotesize\\setlength{\\tabcolsep}{2.4pt}\n"
    f"\\begin{{longtable}}{spec}\n{caption}\\label{{tab:registered}}\\\\\n"
    f"\\toprule\n{header}\n\\midrule\n\\endfirsthead\n"
    f"\\multicolumn{{4}}{{@{{}}l}}{{\\emph{{Table~\\ref{{tab:registered}}, continued}}}}\\\\\n"
    f"\\toprule\n{header}\n\\midrule\n\\endhead\n\\bottomrule\n\\endlastfoot\n"
    f"{rows}\n\\end{{longtable}}}}"
)
body = body[:t0] + long_table + body[t1:]

# --------------------------------------------------------------------------- #
# EMSE submission guidelines (read 2026-10-01).                               #
# --------------------------------------------------------------------------- #


def phrase(text: str, old: str, new: str) -> str:
    """Replace one occurrence of `old`, allowing any whitespace between words."""
    pat = r"\s+".join(re.escape(w) for w in old.split())
    hits = re.findall(pat, text)
    assert len(hits) == 1, f"expected one {old[:50]!r}, found {len(hits)}"
    return re.sub(pat, lambda m: new, text)


# References are cited by name and year. Where the authors are the subject of
# the sentence the citation is textual and the typed names go; elsewhere it is
# parenthetical, and in a table cell bare.
TEXTUAL = {
    r"Kaufman \emph{et al.} describe~\cite{leakage}": r"\citet{leakage} describe",
    r"Ji \emph{et al.}~\cite{ji}": r"\citet{ji}",
    r"Maynez \emph{et al.}~\cite{maynez2020faithfulness}": r"\citet{maynez2020faithfulness}",
    r"Xu~\cite{xu2025openworld}": r"\citet{xu2025openworld}",
    r"Luo \emph{et al.}~\cite{luo2026removal}": r"\citet{luo2026removal}",
    r"Ng \emph{et al.}~\cite{ng2026runtimecontract}": r"\citet{ng2026runtimecontract}",
    r"Schneider~\cite{schneider2000enforceable}": r"\citet{schneider2000enforceable}",
    r"Falcone \emph{et al.}~\cite{falcone2012verify}": r"\citet{falcone2012verify}",
    r"Le~\cite{le2026schemakey} shows": r"\citet{le2026schemakey} shows",
    r"Ahn and Kim~\cite{ahn2026harness}": r"\citet{ahn2026harness}",
    r"Winston \emph{et al.}~\cite{winston2026solver}": r"\citet{winston2026solver}",
    r"Zhang \emph{et al.}~\cite{zhang2025cda}": r"\citet{zhang2025cda}",
    r"Muchlinski \emph{et al.}~\cite{muchlinski2016civilwar}": r"\citet{muchlinski2016civilwar}",
}


def author_year(text: str, textual: dict[str, str]) -> str:
    for old, new in textual.items():
        text = phrase(text, old, new)
    text = text.replace("& \\cite{", "& \\citealp{").replace("~\\cite{", " \\citep{")
    assert "\\cite{" not in text, "a citation was left numbered"
    return text


body = author_year(body, TEXTUAL)
data_avail = author_year(data_avail, {})

# Abbreviations are defined at first mention.
body = phrase(body, "answer is agentic: an LLM calls",
              "answer is agentic: a large language model (LLM) calls")
body = phrase(body, "the open-source ML-pipeline assistant",
              "the open-source machine-learning (ML) pipeline assistant")
body = phrase(body, r"$\mathrm{AUC}>0.995$",
              r"an area under the receiver operating characteristic curve (AUC) $>0.995$")
body = phrase(body, "into SMT constraints", "into satisfiability-modulo-theories constraints")
body = phrase(body, "CI failures", "continuous-integration failures")
body = phrase(body, r"Wilson 95\,\% intervals", r"Wilson 95\,\% confidence intervals (CIs)")

# Figures: vector line art named Fig<n>, and no full stop ending the caption.
body = phrase(body, r"\includegraphics[width=\columnwidth]{fig1_contract.png}",
              r"\includegraphics[width=\textwidth]{Fig1.pdf}")
body = phrase(body, "whether $E$ matches the world.}", "whether $E$ matches the world}")

# Table footnotes are lower-case superscript letters.
body = phrase(body, r"\texttt{A-L1-DESCRIBED}$^{*}$", r"\texttt{A-L1-DESCRIBED}$^{a}$")
body = phrase(body, r"{\footnotesize $^{*}$Also", r"{\footnotesize $^{a}$Also")

# The use of an LLM is documented in the methods.
ai = (TSE / "ai-disclosure.tex").read_text(encoding="utf-8")
ai = ai[ai.index(r"\paragraph{Declaration of AI-assisted tools}") + len(r"\paragraph{Declaration of AI-assisted tools}"):]
ai = phrase(ai, r"are described in Section~\ref{sec:design}.", "are described above.")
rq1 = body.index(r"\section{RQ1")
rq1 = body.rindex("\n", 0, rq1 - 1) + 1  # start of the rule line above the section
body = body[:rq1] + "\\subsection{Use of AI tools}\\label{sec:ai}\n" + ai.strip() + "\n\n" + body[rq1:]

# No subfolders and no stray inputs: the per-instance table goes inline.
body = body.replace(r"\input{table_fabbench12}",
                    (EMSE / "table_fabbench12.tex").read_text(encoding="utf-8").strip())
assert "\\input{" not in body

# Any tabular still wider than the text is scaled down to it; narrower ones are
# left alone (adjustbox's max width, which SVJour3's table accepts).
body = re.sub(
    r"(?<!\{%\n)(\\begin\{tabular\}.*?\\end\{tabular\})",
    lambda m: "\\adjustbox{max width=\\textwidth}{%\n" + m.group(1) + "}",
    body,
    flags=re.S,
)
assert "IEEE" not in body.replace("IEEE policy", ""), "an IEEE command is left in the body"

preamble = r"""%% Empirical Software Engineering (Springer) submission, SVJour3.
%% Generated from ../tse_latex/main.tex by ../port_to_emse.py; edit the TSE
%% source and re-run the script rather than editing this file.
%% Compile: tectonic -X compile main.tex
\RequirePackage{fix-cm}
% smallextended: the one-column format of the template; natbib: author-year
% citations punctuated as Springer asks, "(Thompson 1990)".
\documentclass[smallextended,natbib]{svjour3}
\smartqed

\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{array}
\usepackage{longtable}
\usepackage{adjustbox}
\usepackage{xcolor}
\usepackage{textcomp}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{microtype}
\usepackage[htt]{hyphenat}
\usepackage{url}
\usepackage[hidelinks]{hyperref}
\tolerance=1200
\emergencystretch=2em
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.7}
\renewcommand{\textfraction}{0.07}
\renewcommand{\floatpagefraction}{0.7}
\setcounter{topnumber}{3}
\setcounter{totalnumber}{4}

\spnewtheorem{observation}{Observation}{\bfseries}{\itshape}
\spnewtheorem*{proofsketch}{Proof sketch}{\itshape}{\rmfamily}

\hyphenation{mlcompass evi-dence-closed}

\journalname{Empirical Software Engineering}

\begin{document}

\title{Misfiling, Not Invention: How an LLM Narrator of Machine-Learning
Pipeline Evidence Fails, and What an Evidence-Bound Contract Checks}
\titlerunning{Misfiling, Not Invention}

\author{Hakan~Sabuni\c{s} \and Yusuf~\"Unl\"u \and Mehmet~Kemal~\"Ozdemir}
\authorrunning{H. Sabuni\c{s} et al.}

\institute{H. Sabuni\c{s} (corresponding author) \and Y. \"Unl\"u \and
M. K. \"Ozdemir \at
School of Engineering and Natural Sciences, Istanbul Medipol University,
Istanbul, T\"urkiye \\
\email{hakan.sabunis@std.medipol.edu.tr}}

\date{Received: date / Accepted: date}

\maketitle

"""

backmatter = r"""
\begin{acknowledgements}
""" + acks + r"""
\end{acknowledgements}

\section*{Statements and Declarations}

\paragraph{Funding.} No funding was received for conducting this study.

\paragraph{Competing interests.} The authors have no competing interests to
declare that are relevant to the content of this article. mlcompass is released
by the authors under the MIT licence and is not a commercial product.

\paragraph{Ethics approval.} Not applicable: the study involved no human
participants and no animals.

\paragraph{Consent to participate and to publish.} Not applicable.

\paragraph{Author contributions.} H.S. designed and implemented the contract
and the measurement harnesses, and ran the batteries. Y.\"U. designed the
downstream benchmark protocol and performed the independent review that
identified scorer false positives. M.K.\"O. supervised the work. All authors
read and approved the final manuscript.

\paragraph{Data availability.} """ + data_avail + r"""

\paragraph{Code availability.} The source of mlcompass, every harness and every
analysis script are in the replication package \citep{replication} under the
MIT licence; the development repository is
\url{https://github.com/hakansabunis/mlcompass}.

\paragraph{Use of AI tools.} Documented in Section~\ref{sec:ai}.

\bibliographystyle{spbasic}
\bibliography{references}

\end{document}
"""

out = (
    preamble
    + "\\begin{abstract}\n" + abstract + "\n"
    + "\\keywords{" + " \\and ".join(k.strip() for k in keywords.split(",")) + "}\n"
    + "\\end{abstract}\n"
    + body.rstrip() + "\n"
    + backmatter
)
(EMSE / "main.tex").write_text(out, encoding="utf-8")


def drop_duplicate_eprints(bib: str) -> str:
    """spbasic prints an eprint field as a bare identifier after the entry's own
    "arXiv:NNNN" text, so the number appeared twice. Drop eprint fields from
    entries that already name their arXiv identifier."""
    out = []
    for entry in re.split(r"(?=\n@)", bib):
        if "arXiv:" in entry:
            entry = re.sub(r"\n\s*(eprint|archivePrefix|primaryClass)\s*=\s*\{[^}]*\},?", "", entry)
        out.append(entry)
    return "".join(out)


(EMSE / "references.bib").write_text(
    drop_duplicate_eprints((TSE / "references.bib").read_text(encoding="utf-8")), encoding="utf-8"
)

# The upload set: one flat folder, as Editorial Manager wants it.
SUB = PAPER / "emse_submission"
SUB.mkdir(exist_ok=True)
UPLOAD = ("main.tex", "references.bib", "svjour3.cls", "svglov3.clo", "spbasic.bst", "Fig1.pdf")
for stale in SUB.iterdir():
    if stale.is_file() and stale.name not in UPLOAD:
        stale.unlink()
for name in UPLOAD:
    shutil.copyfile(EMSE / name, SUB / name)
print(f"wrote {EMSE / 'main.tex'} ({len(out)} chars) and the upload set in {SUB}")
