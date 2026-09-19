# Claude for Open Source — başvuru taslağı

Aşağıdaki iki metin forma kopyalanmak üzere. Hiçbir sayı şişirilmedi; her biri
doğrulandı (19 Eylül 2026): public + MIT, PyPI'da 29 Mayıs'tan beri 14 sürüm,
şu an 0.9.0, 183 commit, 827 dosyada 106.492 satır Python. Yıldız 0, fork 0,
bilinen bağımlı yok — ve aşağıda bu da yazıyor, çünkü formu okuyan kişi zaten
depoya bakacak.

---

## Alan 1 — "Tell us about the project's reach and impact"

mlcompass is an MIT-licensed ML-pipeline assistant: it profiles a dataset,
audits a training script for leakage and reproducibility defects, evaluates a
predictions table, and narrates what it found. It has shipped on PyPI since
May 2026 across 14 releases (currently 0.9.0), with 183 commits and about
106,000 lines of Python.

I'll be straightforward about reach, because you can see the repository:
it has no stars, no forks, and no downstream dependents. It is four months old
and I am its only author. If adoption is the bar, it does not clear it yet.

The case I would make is the gap it fills, and it is a specific one. Most
LLM tooling around ML asks a model to be more careful. mlcompass ships a
runtime contract that makes a class of unfaithfulness *structurally
impossible* rather than less likely: the set of column names and measured
values a narration may cite is computed from the deterministic analysis at
call time, bound into the tool schema, and re-verified in plain code before
anything reaches the user. It needs no logit access, so it works behind
commercial APIs.

That work is now a paper under submission to IEEE Transactions on Software
Engineering, and the repository is its replication package: 4,960 preserved
live model responses, every emitted script, frozen pre-registration protocols
with a numbered amendment log, two independent implementations of each scorer,
and — deliberately — every superseded measurement kept beside its correction,
including the ten faults we found in our own instruments. Two of those ten
moved a published result *against* the hypothesis we were arguing for.

So the honest summary: no user base yet, and an artifact built to a standard
of evidence-keeping that I have not often seen in tools this size. The reach I
am asking you to weigh is the second one.

---

## Alan 2 — "How will you use the subscription for your project?"

Three concrete things, in priority order.

**1. Finish the experiments the submission still needs.** The reviewers'
strongest objection is that the paper defines a class of tasks and measures
one member of it. I am mid-way through the second: the same contract bound to
a dataset profiler instead of a leakage detector, four arms at N=200. Two arms
are done and two are running. After that: numeric-tolerance sensitivity, and a
template-only baseline that asks what the language model contributes over a
deterministic renderer.

**2. Keep the replication package honest.** Most of what the subscription
would buy is not writing code — it is re-running measurements after every
instrument correction, and re-deriving every table from the run records rather
than transcribing them. That discipline is why three wrong numbers were caught
before submission rather than after, and it is expensive in exactly the way a
subscription helps with.

**3. Ship the tool properly.** 0.9.0 is a working tool with no users. Getting
to 1.0 means documentation someone else can follow, a stable CLI surface, and
the contract layer factored so a third evidence producer can be bound to it
without touching the verifier — which is the thing the paper claims and the
code has only recently started to deserve.

I am a single author on an academic timeline, and API spend has been the
binding constraint on all three: I have had to stop mid-battery more than once
this month.

---

## Alan 3 — "Other info" (isteğe bağlı)

Paper under submission to IEEE TSE; the repository is its replication package.
Happy to share the manuscript. The independent review it has had so far
recommended acceptance, with the caveat that we have since withdrawn one of
the findings it praised — an arm we ran to attack our own causal claim
succeeded, and the paper now says so.

---

## Verirler mi — dürüst değerlendirme

Form açıkça şunu soruyor: *download numbers, GitHub activity, who depends on
it, or the gap it fills.* Dördünden üçü bizde sıfıra yakın.

**Lehine:** gerçekten public, gerçekten MIT, gerçekten PyPI'da ve 14 sürümdür
sürdürülüyor — terk edilmiş bir tatil projesi değil. Arkasında hakem sürecinde
bir makale var. Replication package alışılmadık derecede ciddi.

**Aleyhine:** sıfır yıldız, sıfır fork, bilinen kullanıcı yok, dört aylık, tek
yazar. "Ekosistemde kime dayanak oluyor" sorusunun cevabı şu an "kimseye".

**Tahminim:** ihtimal düşük–orta. Bu programların çoğu mevcut kullanıcı
tabanına bakıyor. Ama başvuru bedava ve reddedilmenin maliyeti yok. Şansı
artıran tek şey dürüstlük: yukarıdaki metin sıfır yıldızı gizlemiyor, bunun
yerine değerlendirenin *kendi bakabileceği* bir şeyi öne çıkarıyor. Şişirilmiş
bir başvuru, depoya bakan biri için anında görünür olurdu ve bizi daha kötü
gösterirdi.

Makale kabul edilirse aynı başvuru çok daha güçlü olur — o zaman "IEEE TSE'de
yayımlanmış bir çalışmanın referans implementasyonu" diyebilirsin. Şimdi
denemek ve reddedilirse sonra tekrar başvurmak da mümkün.
