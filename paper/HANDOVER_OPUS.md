# DEVİR KİTABI — Q1 Kampanyası, Fable 5 → Opus 4.8 (7 Temmuz 2026)

> **Bu dosyayı okuyan asistana:** Bu kampanyanın tüm hazırlığı tamamlandı ve
> aşağıda her fazın tam oyun planı var. Senin işin YÜRÜTMEK — yeniden
> tasarlamak değil. "Önceden verilmiş kararlar" bölümündeki hiçbir kararı
> kullanıcı istemedikçe yeniden tartışmaya açma. Otorite sırası:
> `paper/analysis_plan.md` (ön-kayıt; bilimsel kurallar) >
> `paper/Q1_ROADMAP.md` (R1-R11 + fazlar) > bu dosya (yürütme kılavuzu).

## 0. Tek paragraf bağlam

mlcompass'ın evidence-bound anlatım kontratı (Tier A çağrı-anı enum + Tier B
deterministik doğrulama) hakkındaki 6-sayfalık INISTA taslağı, 5 iç + 3 dış
hakem değerlendirmesinden geçti; ortak reçete "çoklu sağlayıcı + çoklu leak
deseni + baseline kıyası" idi. Hedef: bu deneyleri koşup ~9-12k kelimelik
manuskripti **ESWA'ya (Q1)** göndermek (yedek: KBS → JSS → IEEE Access).
Tek yazar: Hakan Sabuniş. Hedef gönderim: Kasım-Aralık 2026.

## 1. Durum anlık görüntüsü (7 Tem 2026, commit f7ea7a5)

TAMAM (hepsi commit'li, testli, push'lu):
- Harness: 9 sağlayıcı ailesi (PROVIDERS), telemetri (latency/usage/
  rejection_kinds/abstention), para-güvenlik (--check-providers, --dry-run,
  --smoke, --max-cost-usd, JSONL RunLog + --resume, kırık-kuyruk onarımı,
  evidence-hash hücre kimliği). 3-mercekli düşman incelemesinden geçti.
- FabBench: 6 injector × 7 dataset (4 çekirdek + 3 genişletilmiş; 12/12 +
  18/18 doğrulandı) + 2 GERÇEK vaka (`--task case`): bodyfat (pozitif,
  Density/Siri, enjeksiyonsuz ateşleme DOĞRULANDI) + sambanis (negatif
  kontrol, dedektör doğru susuyor). Donmuş örnek listesi: A1 (13) + A2 (2).
- Ön-kayıt: `paper/analysis_plan.md` (H1-H6, panel, karar kuralları,
  baseline adalet tasarımı, dışlama kuralları, A1+A2 amendmentları).
- 19 doğrulanmış referans (INISTA seti) + genişletme dalgası
  (`paper/references_verified.md` — bu dosyayla aynı gün üretildi).
- Eski 6-sayfalık paper kaynakları: `paper/INISTA_Paper.md` +
  `paper/inista_latex/main.tex` (dondu; manuskriptin çekirdeği).

**TEK BLOKER:** kullanıcının API kredileri/anahtarları. Geldiği an Faz 1.

## 2. Önceden verilmiş kararlar — YENİDEN TARTIŞMA

1. **Tek yazar** Hakan (7 Tem). Gönderim öncesi eski ortak yazarlara (Yusuf
   Ünlü, Selim Akyokuş) tek satır nezaket bildirimi önerilir (görev #99).
2. **Venue sırası:** ESWA → KBS → JSS → IEEE Access.
3. **Seçici raporlama YASAK.** Her koşulan hücre rapora girer; sapmalar
   analysis_plan §8'e gerekçeli amendment olarak İŞLENMEDEN koşulmaz.
4. **İsimler:** kontrat = "evidence-bound runtime schema"; araç = mlcompass;
   benchmark = FabBench. (arXiv 2512.23487 "ML Compass" alakasız isim ikizi
   — manuskripte kısa bir ayrım dipnotu konabilir.)
5. **Model pinleri:** deepseek-v4-flash (alias `deepseek-chat` 24 Tem'de
   ölüyor!), gpt-5.4-mini, claude-haiku-4-5, gemini-2.5-flash-lite,
   mistral-small-latest, grok-4.20-0309-non-reasoning, qwen-flash,
   llama-3.1-8b-instant (Groq), lokal vLLM Qwen2.5-3B-Q4.
6. **Anahtar hijyeni:** anahtarlar SADECE env var; chat'e/commit'e asla
   (geçmişte 2 kez sızdı, ikisi de iptal edildi).
7. **Git:** mlcompass repo'sunda otomatik commit+push YETKİLİ; dosyalar tek
   tek `git add` (asla `.`/`-A`); force push yok. PowerShell 5.1 tuzağı:
   commit mesajında ÇİFT TIRNAK kullanma (argümanı bölüyor; iki kez yaşandı).
8. **Yazım kuralları:** em-dash KULLANMA (kullanıcı isteği); dürüst çerçeve
   ("boundaries, not victories" tonu); tüm sayılar koşu kayıtlarından;
   kullanıcıyla iletişim TÜRKÇE.
9. **Ölçüm disiplini:** her canlı koşu → `paper/ablation_live_<sağlayıcı>_
   <tarih>*.md` kaydı + harness commit pin + JSONL log commit'i.
10. **Maliyet:** her canlı koşudan önce `--dry-run`; yeni sağlayıcıda önce
    `--smoke`; tavan $600 (gerçekçi $200-400).

## 3. Faz oyun planları

### Faz 1 — Cross-provider (anahtar gelince İLK iş)
```bash
# 1) Sıfır maliyet: kablolama
python scripts/reproduce_hallucination_ablation.py --check-providers
# 2) Sağlayıcı başına kuruşluk smoke (openai, anthropic, gemini, ...):
python scripts/reproduce_hallucination_ablation.py --mode live --provider openai --smoke
# 3) Battery, P0 (öncelik openai; deepseek zaten Haziran verisiyle var):
python scripts/reproduce_hallucination_ablation.py --mode live --provider openai --n 200 --json
python scripts/reproduce_hallucination_ablation.py --mode live --provider openai --task csv --csv-path scripts/data/insurance.csv --target charges --n 200 --json
# 4) P1 genişlik (her sağlayıcı, N=100): aynı komutlar --n 100 ile
#    anthropic, gemini, mistral, xai, qwen, groq
# 5) Sweep replikasyonu (>=2 yeni sağlayıcı):
python scripts/reproduce_hallucination_ablation.py --mode live --provider openai --sweep --n 100
# 6) R4 strict-dikotomisi: OpenAI/Anthropic'te strict ON/OFF kolu HARNESS'A
#    EKLENMELİ (küçük iş: tool tanımına strict:true parametresi; opsiyonel
#    --strict bayrağı). Bunu koşulardan ÖNCE ekle + mock test + commit.
```
Kayıt: her blok sonrası ablation_live_*.md + JSONL loglar commit.
GATE G1: sonuçlar ne çıkarsa çıksın rapor; hikâye toplantısı kullanıcıyla.

### Faz 2 — FabBench matrisi
13 donmuş örnek (A1) × kollar (L1, L3, STRESS) × N=200 ana sağlayıcı
(deepseek-v4-flash; maliyet en düşük + Haziran sürekliliği) + alt-küme
N=100 openai'da. Komut kalıbı:
```bash
python scripts/reproduce_hallucination_ablation.py --mode live --provider deepseek \
  --task csv --csv-path scripts/data/heart_cleveland.csv --target chol \
  --injector exact_copy --n 200 --json
# Vakalar (A2):
python scripts/fetch_fabbench_datasets.py --fetch-cases
python scripts/reproduce_hallucination_ablation.py --mode live --provider deepseek --task case --case bodyfat --n 200 --json
python scripts/reproduce_hallucination_ablation.py --mode live --provider deepseek --task case --case sambanis --n 200 --json
```
Örnek listesi: `scripts/fetch_fabbench_datasets.py::FROZEN_INSTANCES`.

### Faz 3 — Baseline'lar + displacement
- ADAPTER SPEC (kod anahtarsız yazılabilir, mock'la test edilir):
  (a) Guardrails AI iki kol: stok (JSON-validity + reask) VE bizim Tier B
  kontrollerimiz custom validator olarak onların döngüsünde (döngü paritesi;
  paper'da açıkça "validator bizim, kıyas döngüler" denecek).
  (b) NeMo Guardrails: en yakın rail konfigürasyonu, config repo'ya.
  (c) Outlines/vLLM lokal: kullanıcının WSL2 kurulumu gerekir (Qwen2.5-3B
  4-bit ~1.9GB, 4GB VRAM'e sığar — 7B SIĞMAZ, doğrulandı); vLLM v0.12+
  `structured_outputs` (xgrammar) + named tool_choice = decode-enforced kol.
  Metrikler: 3 kanal + catch + latency + çağrı/token maliyeti.
- Displacement (R7/H6): STRESS kollarının serbest-metin alanı çapraz-aile
  LLM-yargıçla + >=100 insan örneklemiyle skorlanır (yargıç ataması:
  denek ≠ yargıç ailesi; claude-opus-4-8 / gpt-5.5 / gemini-3.1-pro).

### Faz 4 — Paketleme
FabBench README + leaderboard tablosu üretici + mlcompass v1.0 release.
Her paper sayısı tek komutla üretilebilir olmalı (R8).

### Faz 5 — Manuskript (aşağıdaki §4 taslak planını izle)
2× Q1-kalibre iç panel (aşağıda §5 reçete) + 1 cross-model dış denetim.
R9 checklist: elsarticle, 9-12k kelime, 40-60 doğrulanmış ref, Threats to
Validity, Highlights (3-5 madde), CRediT, Data Availability, AI-kullanım
beyanı (KULLANICI yönetir — sorulmadan ekleme; 12 Haz'da açıkça istendi).

## 4. Manuskript planı (bölüm bölüm brief + sayı yerleşimi)

Çekirdek metin `paper/INISTA_Paper.md`'den genişler. Bölümler:

1. **Introduction** — INISTA girişinin genişletilmişi + YENİ çerçeve (R11,
   PANEL-DÜZELTMELİ hali — eski "hiçbir şema dili ifade edemez" cümlesini
   KULLANMA, panel çürüttü). Doğru argüman:
   *2026 manzarasında enforcement üç sınıfa ayrılıyor: E (xAI: daima),
   O (OpenAI/Anthropic/DeepSeek-beta: opt-in strict), H (Qwen-API, Mistral
   tool-call, Gemini default: ipucu). Enforcement parçalı, opt-in ve
   YÜZEYE BAĞIMLI (Anthropic'in OpenAI-uyumlu katmanı strict'i yok sayar).
   İfade edilebilirlik ile zorlanabilirliği AYIR: tam JSON Schema,
   value-tolerans pencerelerini çağrı-anında oneOf/const dallarıyla,
   anchor kapsamasını contains ile ifade EDEBİLİR; ama sağlayıcıların
   fiilen zorladığı strict alt-kümeler tam bu anahtar kelimeleri dışlar ve
   "committed verdict ⇒ anchor adreslenmiş" koşullu tamlığı hiçbir
   belgelenmiş strict alt-kümesi taşıyamaz. Sağlayıcı × anahtar-kelime
   destek tablosu ekle (Q1_ROADMAP R11'deki veriler). Sonuç: ifade
   edilebilenin bile zorlanmadığı ve zorlananın sertifiye edilemediği
   yerde, deterministik post-verification tek sertifiye edilebilir
   katmandır. Kontrat eskimedi; işbölümü netleşti.*
2. **Related Work** — mevcut 5 paragraf + references_verified.md'den
   genişletme (faithfulness-eval, constrained-decoding yenileri, tool-use
   reliability, leakage/repro, benchmark metodolojisi). Her yeni ref'in
   relevance satırı hangi paragrafa gireceğini söylüyor.
3. **Methodology** — INISTA Sec III + injector kütüphanesi tanımı (6 desen
   tablosu) + case-study protokolü (Siri modeli; negatif kontrol lstsq) +
   telemetri. Önermeler OLDUĞU GİBİ ("formal observations" çerçevesi).
4. **Experimental Setup** — analysis_plan'dan: panel (P0/P1/P2), 13+2
   örnek, N'ler, metrikler, istatistik kuralları, dışlama kuralları.
5. **Results** — tablo iskeletleri: T1 sağlayıcı×kol entity (Haziran
   tablosunun genişletilmişi); T2 sweep×sağlayıcı; T3 injector×dataset
   matrisi (kanal aktivasyonu — R2'nin kanıtı burada); T4 vaka çalışmaları
   (bodyfat/sambanis; yanlış-pozitif oranı!); T5 baseline kıyası (+latency/
   maliyet); T6 strict ON/OFF (R4); displacement bulgusu. Haziran sayıları
   (11.5/43.5, sweep 1-100, 76 catch, 37.5→1.3) deepseek satırları olarak
   yaşamaya devam eder.
6. **Discussion** — INISTA V'in genişletilmişi + R11 işbölümü tartışması +
   negatif-kontrol dersi + displacement yorumu.
7. **Threats to Validity** (YENİ, dergi şartı) — internal (stress confound,
   abstention guard), external (model/dil/görev genellemesi), construct
   (kanal tanımları), conclusion (çoklu kıyas/Holm).
8. **Conclusion + Data Availability + CRediT.**

## 5. İç hakem paneli reçetesi (2 tur, Q1-kalibre)

5 persona: (1) ESWA metodoloji hakemi (istatistik/tasarım), (2) NLP/LLM
uzmanı (ilgili işler/yenilik), (3) SE/sistem hakemi (artifact/repro),
(4) titiz dil hakemi (claim-kapsam tutarlılığı), (5) hostile "reject-first"
hakem. Her tur: bağımsız raporlar → sentez → must-fix listesi → uygula →
2. tur. Bar: iki ardışık turda >= "minor revision". Sonra 1 cross-model
dış denetim (kullanıcı Codex/Gemini'ye elden verir; geçmişte böyle yaptı).
Panel çıktıları paper/review_round_*.md olarak commit'lenir.

## 6. Opus'a ilk gün önerilen sıra

1. Bu dosyayı + Q1_ROADMAP + analysis_plan'ı (ÖZELLİKLE A3) +
   review_panel_2026-07-07.md'yi oku. Görevler #94-#101.
2. **İLK İŞ: §7 Panel-onarım kod dalgası** (görev #101) — A3 amendment'ları
   yürütülebilir kılan kod. Paralı koşu bunlar bitmeden BAŞLAMAZ.
3. Anahtar YOKSA devamında: Faz 3 adapter'ları (spec §3-Faz3; mock test);
   manuskript iskeleti (paper/eswa/, elsarticle, §4 planı).
4. Anahtar VARSA: §3-Faz1 sırası. Her bloktan sonra kayıt + commit.
5. Kullanıcı temposu: hızlıdır, uzun koşuları kendi terminalinden koşturup
   çıktıyı yapıştırmayı sever; token israfına duyarlıdır; net tablolarla
   konuş; Türkçe yaz.

## 7. Panel-onarım kod dalgası (görev #101) — paralı koşuların ÖN ŞARTI

Kaynak: `paper/review_panel_2026-07-07.md` + analysis_plan A3. Sıra önerisi:

1. **Bağımsız skorlayıcı** (A3.9a): `scripts/independent_scorer.py` — ham
   JSONL + evidence dict'ten üç kanalı, ürün modülünden HİÇBİR yardımcıyı
   (evidence_correlation_map/top_candidate/VALUE_TOLERANCE) import etmeden
   yeniden hesaplar; harness skorlarıyla çapraz-doğrulama testi.
2. **Hata işaretleyici** (A3.9b): çift-başarısız yanıtlar `_normalize({})`
   yerine `{"error": "..."}` işaretli kayıt; skor dışı, §7 loglu; sweep
   yolundaki tek-denemeli bare-except de düzeltilir; hücre başına
   error/empty sayacı çıktı tablolarına.
3. **`--strict` bayrağı** (R4/H4): tool tanımına sağlayıcıya-uygun strict
   parametresi (OpenAI: function.strict=true; Anthropic: tool.strict=true);
   iki formatta mock test.
4. **Anchor rename** (A3.10): fabbench_injectors'ta `*_leak` →
   {_adj,_est,_idx,_norm,_grp} haritası; fetch --verify beklenen-ad
   registry'si (endswith('_leak') kontrolü kalkar); testler güncellenir.
5. **Jenerik-retry kolu** (A3.2): investigate_leakage_bound'a
   `correction_style="named"|"generic"` parametresi + harness arm'ı
   (`layer3_stress_generic`); mock test.
6. **STRESS mesaj hizalaması** (A3.11): stress kolunda L1 user mesajı.
7. **synthetic_crowded üretici** (A3.4): spec analysis_plan'da; harness
   `--task synthetic-crowded`; --verify entegrasyonu.
8. **bodyfat modeli** (A3.11): Siri el-kodu yerine tüm sayısal özelliklerde
   lstsq; ölçülen r2 raporda. **csv-task r2'si** de hesaplanır (assert 1.0
   kalkar).
9. **Hash zorlaması** (panel P2): fetch script beklenen SHA256'ları gömer,
   uyuşmazlıkta fail; vaka hash'leri README'ye.
10. **Sampling pinleri**: temperature/top_p mümkün olan yerde sabitlenir ve
    JSONL'e yazılır. **make_tables.py** iskeleti (R8: her tablo commit'li
    JSONL'den).
11. **Related-work ekleri** (alan hakemi P1): ToTTo/PARENT/RotoWire (data-
    to-text), AIS/ALCE/RARR (attribution), CRITIC/Self-Refine/Huang-2024
    (self-correction) — HEPSİ web-doğrulamalı olarak references_verified.md
    'ye eklenir (uydurma-ref sıfır toleransı). Manuskriptte 76-vs-80
    ifadesi netleştirilir; iki Haziran L1 ölçümü de raporlanır.
