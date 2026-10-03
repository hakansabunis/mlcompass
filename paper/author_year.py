"""Author-year citations for the EMSE and the venue-neutral builds.

The TSE source cites by number. Both other builds cite by name and year: where
the authors are the subject of the sentence the citation is textual and the
typed names go; elsewhere it is parenthetical, and in a table cell bare.
"""

from __future__ import annotations

import re


def phrase(text: str, old: str, new: str) -> str:
    """Replace one occurrence of `old`, allowing any whitespace between words."""
    pat = r"\s+".join(re.escape(w) for w in old.split())
    hits = re.findall(pat, text)
    assert len(hits) == 1, f"expected one {old[:50]!r}, found {len(hits)}"
    return re.sub(pat, lambda m: new, text)


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
