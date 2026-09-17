# How long is a Q1 software-engineering paper, and where does "Q1" come from

Two questions came back from review. Both are answered here with measurements
rather than impressions, and every number below can be recomputed from the
commands given.

---

## 1. Is 38–43 pages too long for EMSE?

**No. It is slightly above the median page count and below the median word
count of papers EMSE actually publishes.**

### What was measured

An earlier version of this note used arXiv preprints, which was the wrong
sample: preprint formats vary from two-column to double-spaced to `sn-jnl`, so
a page count across that mixture means very little. This is the corrected
measurement — **30 open-access articles published in EMSE in 2025–2026,
downloaded as Springer's own typeset PDFs**, which is what "an EMSE page"
actually is.

Sample drawn from Crossref (`ISSN 1382-3256`, `from-pub-date:2025-01-01`),
restricted to articles carrying a Creative Commons licence so the published PDF
is fetchable.

| | min | Q1 | **median** | Q3 | max |
|---|---|---|---|---|---|
| pages | 25 | 36 | **40** | 53 | 67 |
| words | 11,907 | 14,405 | **19,046** | 22,282 | 35,059 |

**This manuscript: 43 pages, 17,957 words.**

- 43 pages sits at roughly the **67th percentile** — 20 of the 30 published
  articles are 43 pages or shorter.
- 17,957 words sits at roughly the **40th percentile** — it is **shorter than
  the median EMSE paper**.

The shortest article in the sample is 25 pages. Nothing published in EMSE in
this window is anywhere near 10–15 pages.

### Where the disagreement probably comes from

A page is not a fixed unit, and the two conventions differ by more than a
factor of two. Measured, not assumed:

| format | words per page |
|---|---|
| IEEE two-column (TSE, 8 preprints) | **883** (median) |
| Springer `sn-jnl` single column (this paper) | **418** |

So **43 pages in `sn-jnl` is about 20 pages of IEEE two-column text.** The TSE
sample measured here runs 14–32 pages with a median of 17. Read in the
convention most software-engineering researchers carry in their head — IEEE,
two-column, ~17 pages — this manuscript is normal length. Read as though 43
were an IEEE page count, it would look enormous. It is not an IEEE page count.

### What EMSE itself says

Springer's [submission guidelines for EMSE](https://link.springer.com/journal/10664/submission-guidelines)
state **no page limit and no word limit**. The only length constraint anywhere
in them is the abstract: 150–250 words (ours is 220). Length is mentioned once,
as one factor in time to first decision.

### If it has to be shorter anyway

Page budget by section, for the current 43-page build:

| § | pages | section |
|---|---|---|
| 1 | 2 | Introduction |
| 2 | 3 | Background and Related Work |
| 3 | 3 | Evidence-Closed Narration |
| 4 | 3 | The Artifact |
| 5 | 4 | Study Design |
| 6 | 2 | RQ1 |
| 7 | 2 | RQ2 and RQ3 |
| 8 | 3 | RQ4 |
| 9 | 4 | RQ5 |
| 10 | 5 | Corrections to Our Own Instruments |
| 11 | 2 | Discussion |
| 12 | 2 | Threats to Validity |
| 13 | 2 | Conclusion |
| | 5 | References and back matter |

The only cut available without losing a result is **§3's Proposition 2
(repair asymmetry)**, about 2 pages of enforcement-theory argument that a
reviewer already suggested demoting from a proposition to a design lemma. That
takes the paper to about 41 pages.

§10 is the other large block and should not be touched. It is what distinguishes
this submission from an ordinary tool paper, and an empirical reviewer rewards
it.

### Reproducing these numbers

```bash
curl -sS "https://api.crossref.org/journals/1382-3256/works?filter=from-pub-date:2025-01-01,type:journal-article&rows=200&select=DOI,title,license" -o emse.json
```

Then fetch `https://link.springer.com/content/pdf/<doi>.pdf` for each CC-licensed
DOI and count pages with any PDF library.

---

## 2. "Did you send it to Stanford? What Q did they give you?"

There is no such service, and the question contains a common misunderstanding
worth clearing up because it changes what we should actually do.

**A quartile is a property of a journal, not of a paper.** Q1/Q2/Q3/Q4 come
from two ranking systems, both of which rank *journals* by citation metrics
within a subject category:

- **Scopus / SJR (SCImago Journal Rank)** — scimagojr.com, free.
- **Clarivate / JCR (Journal Citation Reports, Web of Science)** — requires an
  institutional subscription.

Nobody submits a paper to get a quartile. The journal has one before the paper
exists. Our target journal's quartile is therefore already fixed and knowable
today.

**What EMSE's own publisher page states** (verified 2026-09-17 at
[link.springer.com/journal/10664](https://link.springer.com/journal/10664)):

| | |
|---|---|
| Journal Impact Factor | **3.4** (2025) |
| 5-year Impact Factor | **4.0** (2025) |
| Submission to first decision | 24 days (median) |
| Indexed in | Science Citation Index Expanded (SCIE), Scopus, SCImago, EI Compendex, DBLP, ACM Digital Library |
| Article charges | none — no submission fee, no publication fee |

The SCImago quartile is not quoted here because scimagojr.com blocks automated
access; it should be read directly from
`https://www.scimagojr.com/journalsearch.php?q=24022&tip=sid` in a normal
browser. SCIE + Scopus indexing is what makes a quartile exist at all, and
both are confirmed above.

### Where "Stanford" probably comes from

Almost certainly the **Stanford / Elsevier "Top 2% Scientists" list**
(Ioannidis et al., published through Elsevier Data Repository). That list ranks
**researchers** by a composite citation score. It has nothing to do with
papers, nothing to do with journals, and nothing to say about a manuscript
before publication.

### What does exist: journal matchers that take an abstract

These take a title and abstract and suggest journals. They suggest *fit*, never
a quartile or an acceptance probability:

- **Springer Nature Journal Suggester** — <https://journalsuggester.springer.com>.
  Most relevant to us, since it covers EMSE and reports impact factor and median
  time to first decision alongside each suggestion.
- **Elsevier JournalFinder** — <https://journalfinder.elsevier.com>
- **Wiley Journal Finder** — <https://journalfinder.wiley.com>
- **Web of Science Manuscript Matcher** — inside EndNote / WoS, institutional.
- **JANE** — <https://jane.biosemantics.org>, biomedical, not useful here.

Worth running the Springer one on our abstract before submitting, as a check
that EMSE really is the closest fit and not, say, TOSEM or JSS. It is a
five-minute sanity check, not evidence.

### What no tool can tell us

Whether this paper will be accepted. That is what the review panel and the
reviewer-agent rounds in this repository are for, and it is why the six items
in `benchmark/VALIDATION_QUEUE.md` matter more than any venue-matching tool.
