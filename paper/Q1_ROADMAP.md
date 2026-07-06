# mlcompass → Q1 Dergi Kampanyası — Yol Haritası

**Tarih:** 6 Temmuz 2026 · **Sürüm:** v1 · **Sahip:** Hakan Sabuniş
**Amaç:** INISTA için yazılmış 6-sayfalık paper'ı (evidence-bound runtime schema) üç dış
hakemin ortak reçetesindeki deneylerle güçlendirip **Q1 bir dergiye** (birincil hedef: ESWA)
göndermek.

> Bu dosya kendi kendine yeterli olacak şekilde yazıldı: bu kampanyayı yürüten kişi ya da
> asistan, başka hiçbir bağlam olmadan buradan devam edebilmelidir. Karar geçmişi ve
> gerekçeler için `paper/ablation_live_*.md` kayıtlarına ve paper kaynaklarına bakın.

---

## 0. Envanter — elimizde ne var (6 Tem 2026 itibarıyla)

| Varlık | Yer | Durum |
|---|---|---|
| Paper (6 sayfa, IEEE conf) | `paper/INISTA_Paper.md` + `paper/inista_latex/main.tex` + DOCX | Dondu; 5 iç hakem (3A+2WA) + 3 dış değerlendirmeden geçti; referanslar ilk-atıf sıralı |
| Ölçüm harness'ı | `scripts/reproduce_hallucination_ablation.py` | Provider-aware (anthropic/deepseek/openai), `--task csv`, sweep, arms: L1/L2/L3/STRESS/tier_a/stress_mech |
| Kontrat (ürün kodu) | `src/mlcompass/agents/leakage_investigator.py` | Tier A çağrı-anı enum + Tier B 3-kanal doğrulama; 23 e2e test; PyPI v0.9.0 canlı |
| Ölçüm kayıtları | `paper/ablation_live_deepseek_*.md` (5 dosya) | Tüm mevcut sayıların kaynağı |
| Mevcut sonuç seti | — | L1: 11.5% sentetik / 43.5% Insurance; kontrat 0/200 her kanal; sweep 1%→100%; STRESS 76 catch, repeat 37.5%→1.3%; tier_a izole 0/200 |

**Mevcut zayıflıklar (3 dış hakemin ortak listesi):** tek sağlayıcı (deepseek-chat), tek leak
deseni (log-of-target), value/omission kanalları hiç ateşlenmedi, Guardrails/NeMo/decoder
baseline'ı yok, L1 ihlal kompozisyonu/abstention loglanmadı.

---

## 1. Venue stratejisi

1. **Birincil: Expert Systems with Applications (ESWA, Elsevier, Q1)** — uygulamalı AI +
   kapsamlı deney + gerçek sistem profili tam bizim iş.
2. **İkincil: Knowledge-Based Systems (KBS, Q1)** — aynı aile, biraz daha yöntem-ağırlıklı.
3. **Yedek A: Journal of Systems and Software (JSS, Q1/Q2)** — "LLM bileşenleri için
   güvenilirlik kontratı" SE çerçevelemesiyle.
4. **Yedek B: IEEE Access** — hız gerekirse; mega-dergi, itibar notunu bilerek.

**İsim/kimlik notu:** arXiv 2512.23487 "ML Compass" (Digalakis ve ark., model-seçim
optimizasyonu) tamamen farklı bir iş ama isim ikizi. Paper kimliği **kontrat**
("evidence-bound runtime schema"), araç adı mlcompass, benchmark adı **FabBench** — üç ad
bilinçli olarak ayrık tutulur; paper'a "not to be confused with" dipnotu eklenebilir.

---

## 2. Bitiş-durumu gereksinimleri (Definition of Done)

Gönderim ancak HEPSİ sağlandığında yapılır:

- **R1 — Deney matrisi (katmanlı sağlayıcı paneli; 7 Tem kararı: maliyet kısıt değil,
  hedef benchmark-grade genişlik):**
  - **P0 (tam matris, N=200/hücre):** DeepSeek [mevcut veriyle süreklilik] + OpenAI
    [enforcement-dikotomisi çapası].
  - **P1 (genişlik: battery + sweep, N≥100):** Anthropic, Google Gemini, Mistral, xAI Grok,
    Alibaba Qwen-API, 1-2 açık model (Qwen/Llama; hosted ya da lokal) — **toplam ≥7 aile,
    10-14 model.** FabBench leaderboard'unun omurgası.
  - **P2 (spot-check, N=100, 1-2 hücre):** her aileden güçlü-tier model ("ucuz model
    artefaktı" itirazını kapatır).
  - Panel `analysis_plan.md`'de koşulardan ÖNCE sabitlenir; sonradan sağlayıcı ekleme/çıkarma
    gerekçesiyle loglanır (cherry-picking yasak). Ayrıca ≥4 veri kümesi; ≥5 sızıntı deseni.
    Model adları, roller, fiyatlar: §2b Model Matrisi.
- **R2 — Kanal aktivasyonu:** value-fabrication ve critical-omission kanallarının en az bir
  desende gerçekten ateşlendiği gösterilmiş ("üç kanaldan ikisi boş" eleştirisi ölmüş).
  Desenler bunun için seçilir (türetilmiş-oran ve kalabalık-kanıt desenleri bu kanalları zorlar).
- **R3 — Baseline'lar:** Guardrails AI ve NeMo Guardrails aynı görevde, aynı metriklerle;
  açık-model kolunda Outlines (decode-enforced enum = sertifiye edilebilir karşılaştırma
  noktası). Rapor: 3 kanal + catch + latency + çağrı/token maliyeti.
- **R4 — Enforcement dikotomisi:** decode-enforced (OpenAI strict structured outputs ON,
  Outlines) vs schema-as-instruction (DeepSeek, strict OFF) Tier A karşılaştırması ölçülmüş.
  Bu, "Tier A steers, does not enforce" limitasyonunu ölçüme çevirir.
- **R5 — Sweep genellemesi:** 6-parafraz sweep ≥2 sağlayıcıda replike. Oynaklık genellenirse
  manşet güçlenir; genellenmezse o da bulgudur ("oran sağlayıcıya bağlı") — iki sonuç da yayımlanır.
- **R6 — Telemetri:** ihlal kompozisyonu, yanıt-başına claim sayısı, abstention oranı, retry
  transkriptleri, latency, token maliyeti TÜM yeni koşularda loglanır (eski Limitasyon-2 kapanır).
- **R7 — Displacement deneyi:** yapılandırılmış kanal kilitlenince fabrikasyonun serbest-metin
  alanına kaçıp kaçmadığı ölçülmüş (LLM-judge + ≥100 örneklik insan kontrolü). Her iki sonuç
  da bulgu.
- **R8 — Artifact:** FabBench repo'su (injector'lar + skorer + ham kayıtlar + tablo üretici) +
  mlcompass v1.0; paper'daki her sayı tek komutla yeniden üretilebilir.
- **R9 — Manuskript:** elsarticle formatı, ~9-12k kelime, 40-60 web-doğrulanmış referans
  (uydurma referans SIFIR toleransı — mevcut 19'un tamamı doğrulanmıştı, aynı disiplin),
  Threats to Validity bölümü, Highlights, CRediT yazar katkı beyanı, Data Availability,
  Elsevier politikası gereği AI-kullanım beyanı (kullanıcı yönetir).
- **R10 — Süreç:** (a) **Tek yazar: Hakan Sabuniş** (7 Tem 2026 kararı — Q1 manuskripti
  baştan yazılmış, tüm yeni deneyler tek başına yürütülmüş yeni bir eserdir; eski INISTA
  taslağının ortak yazarlarına gönderim öncesi kısa bir nezaket bildirimi önerilir, özellikle
  taslaktan metin/sonuç taşınıyorsa); (b) Q1-kalibre iç 5-hakem paneli iki ardışık turda
  ≥ "minor revision" bandı; (c) cross-model dış denetim 0 P1 bulgu.
- **R11 — 2026 enforcement manzarasına göre yeniden çerçeveleme:** INISTA taslağındaki
  "decoding-time enforcement is unavailable behind commercial APIs" iddiası mid-2026'da
  ARTIK DOĞRU DEĞİL (OpenAI/Anthropic/xAI strict dokümante ediyor; §2b). Journal sürümü
  şu üç ayak üzerine yeniden konumlanır: (a) enforcement parçalı, opt-in ve yüzeye bağımlı
  (default'lar ipucu; compat-layer'lar strict'i yok sayar; Qwen/Mistral tool-call'da hiç yok);
  (b) zorlanan yerde bile refusal/truncation istisnaları var; (c) **enum'lar varlık bağlar
  ama value-soundness ve completeness HİÇBİR şema diliyle ifade edilemez** — "entity binding
  giderek decode-enforced olurken, doğrulanamayan kalıntı claim'ler ve kapsamdır; Tier B tam
  orayı sertifikalar." Bu çerçeve paper'ı eskitmez, günceller.

---

## 2b. Model matrisi ve tedarik (7 Tem 2026, resmi doküman taramasıyla doğrulandı)

**Enforcement sınıflandırması (tool-call enum'ları için; kaynaklar tarandı, tarih: 7 Tem 2026):**

| Sınıf | Aile | Durum |
|---|---|---|
| **E — daima zorlar** | xAI Grok | "strict flag is implicitly always true", gramer-derlemeli; KAPATILAMAZ → Tier A orada ablasyona kapalı (tasarım notu) |
| **O — opt-in strict** | OpenAI (strict:true, enum açıkça kapsanır), Anthropic (tool'da strict:true, Haiku 4.5 GA; **OpenAI-uyumlu katmanında strict YOK SAYILIR**), DeepSeek (SADECE beta base_url + strict:true), vLLM lokal (named tool_choice/required) | ON/OFF karşılaştırılabilir → R4'ün ana malzemesi |
| **H — sadece ipucu** | Qwen-API/DashScope (hiçbir zorlama dokümante değil), Mistral (json_schema response-format garantili AMA tool-call'da hiçbir şey), Gemini (varsayılan modlar; sadece preview `validated` iddialı), Groq-Llama, Together; tüm sağlayıcıların default'ları | Paper'ın orijinal senaryosu |

**Panel (P0 tam matris / P1 genişlik / P2 spot-check):**

| Rol | Model | Fiyat ($/M in-out) | Not |
|---|---|---|---|
| P0 | DeepSeek `deepseek-v4-flash` | 0.14 / 0.28 | ⚠️ `deepseek-chat` alias'ı **24 Tem 2026'da ölüyor** — doğrudan ad pinle |
| P0 | OpenAI `gpt-5.4-mini` | 0.75 / 4.50 | strict ON/OFF iki kol (R4 çapası); `gpt-5.4-nano` (0.20/1.25) ek ucuz denek |
| P1 | Anthropic `claude-haiku-4-5` | 1.00 / 5.00 | strict ON/OFF — aile-içi dikotomi; compat-layer'da strict yok sayılır (3. veri noktası!) |
| P1 | Google `gemini-2.5-flash-lite` | 0.10 / 0.40 | H-sınıfı varsayılan; `validated` preview ayrıca not edilir |
| P1 | Mistral `mistral-small-latest` | 0.15 / 0.60 | tool-call'da zorlama yok (H); `ministral-3b` (0.10/0.10) ek ucuz denek |
| P1 | xAI `grok-4.20-non-reasoning` | 1.25 / 2.50 | E-sınıfı tek örnek — panelin en değerli üyelerinden |
| P1 | Alibaba `qwen-flash` (DashScope intl/Singapur) | 0.05 / 0.40 | H-sınıfı; 1M token ücretsiz kota (90 gün); TR'den intl hesapla erişilebilir |
| P1 | Groq `llama-3.1-8b-instant` | 0.05 / 0.08 | açık-ağırlık hosted denek (Meta ailesi) |
| P1 | **Lokal vLLM** `Qwen2.5-3B-Instruct` Q4 (WSL2) | 0 | Outlines/GCD decode-enforced baseline hattı; 3B@Q4 ≈1.9GB → RTX 3050 4GB'a SIĞAR (7B sığmaz, doğrulandı); v0.12+ `structured_outputs` API (xgrammar/guidance) |
| P2 | `gpt-5.5` (5/30), `claude-opus-4-8` (5/25), `gemini-3.1-pro-preview` (2/12) | — | güçlü-tier spot-check, N=100, 1-2 hücre |
| Yargıç (R7) | Çapraz-aile atama: denek ≠ yargıç ailesi; `claude-opus-4-8` / `gpt-5.5` / `gemini-3.1-pro` rotasyonu | — | + ≥100 insan örneklemi |

**Toplam: 9 aile, 13-16 model.** Hosted decode-enforced yedek: Fireworks json_schema (0.20/M, "enforced during generation") — lokal vLLM sorun çıkarırsa.

**Tedarik listesi (kullanıcı; anahtarlar SADECE env var):** OpenAI ~$30 · Anthropic ~$30 ·
Google AI Studio ~$20 · DeepSeek ~$10 (yeni anahtar) · xAI ~$20 · Mistral ~$15 ·
Alibaba Model Studio intl ~$10 (önce ücretsiz kota) · Groq ~$10 (ücretsiz tier var).
Env adları: `OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, DEEPSEEK_API_KEY,
XAI_API_KEY, MISTRAL_API_KEY, DASHSCOPE_API_KEY, GROQ_API_KEY`.

**Harness notu:** Gemini/Mistral/xAI/Qwen/Groq hepsi OpenAI-uyumlu endpoint veriyor →
`PROVIDERS` dict'e base_url girişleri yeter; Anthropic native yolu zaten var. Her yeni
sağlayıcıda N=5 smoke-test, sonra battery.

---

## 3. Faz planı (gate'li)

### Faz 0 — Altyapı + ön-kayıt (≈1 hafta)
- Harness telemetri yükseltmesi (R6) — **tüm yeni koşulardan önce**, yoksa koşular tekrarlanır.
- `paper/analysis_plan.md` **ön-kayıt**: hipotezler, metrikler, hücre tasarımı, karar kuralları,
  baseline adalet tasarımı. Koşulardan önce commit'lenir (hakemler bu disiplini övdü; sürdür).
- 8 sağlayıcı hesabı + kredi: §2b tedarik listesi (~$145 toplam başlangıç). **GÜVENLİK:
  anahtarlar yalnızca env var; asla chat'e/commit'e yapıştırılmaz (geçmişte 2 kez sızdı,
  ikisi de iptal edildi).**
- Harness: `PROVIDERS` dict'e 5 yeni OpenAI-uyumlu giriş (gemini/mistral/xai/qwen/groq
  base_url'leri, §2b) + `deepseek-v4-flash` pin'i (alias 24 Tem'de ölüyor) + her sağlayıcıda
  N=5 smoke-test.
- Açık model: KARAR VERİLDİ (7 Tem) — lokal vLLM (WSL2) + Qwen2.5-3B-Instruct Q4
  (4GB VRAM'e sığdığı doğrulandı); yedek Fireworks json_schema.
- **GATE G0:** anahtarlar hazır + analysis_plan.md commit'li → Faz 1 başlar.

### Faz 1 — Cross-provider (≈2 hafta)
- Battery (L1/L3/STRESS × sentetik + Insurance) gpt-4o-mini ve claude-haiku'da.
- OpenAI'da strict structured outputs **ON ve OFF** iki kol (R4'ün yarısı).
- Sweep replikasyonu ≥1 yeni sağlayıcıda (R5).
- **GATE G1:** sonuç sanity + hikâye yönü toplantısı. Hangi sonuç çıkarsa çıksın rapor edilir;
  seçici raporlama YASAK — her koşu `paper/ablation_live_*.md` kaydına girer.

### Faz 2 — FabBench çekirdeği (≈3-4 hafta) — kampanyanın bel kemiği
- 5-6 injector: exact-copy, monotone-log [var], noisy-proxy (ρ≈0.99), train/test
  contamination (perfect-match kanalı), target-encoding, derived-ratio (value kanalını zorlar).
- 4-5 veri kümesi: synthetic [var], Insurance [var], Heart Disease (klinik), Ames House
  Prices, Telco Churn (hepsi v0.7.x saha testlerinden tanıdık).
- ~10-12 görev örneği (tam çapraz değil; ön-kayıtlı seçim) × 3 kol × N=200 ana sağlayıcıda;
  alt-küme diğer sağlayıcılarda N=100 (R1, R2).

### Faz 3 — Baseline'lar + displacement (≈2-3 hafta)
- Guardrails AI iki kol: (i) stok (JSON-validity + reask), (ii) bizim doğrulayıcı Guardrails
  döngüsüne gömülü (döngü paritesi ölçümü — adalet tasarımı analysis_plan'da). NeMo rails.
  Outlines açık modelde (R3).
- Displacement deneyi STRESS kollarında (R7).
- **GATE G2:** tüm ampirik iddialar masada; paper hikâyesi netleşir.

### Faz 4 — Paketleme (≈1-2 hafta)
- FabBench ayrı repo ya da `mlcompass/benchmarks/` — injector + skorer + kayıtlar + README.
- mlcompass v1.0 (ya da v0.10) release (R8).

### Faz 5 — Manuskript + iç hakemlik (≈3-4 hafta)
- Journal genişletmesi: 6 sayfa → 9-12k kelime; Related Work büyür (+20-40 doğrulanmış ref);
  Threats to Validity; tüm yeni tablolar.
- 2× Q1-kalibre 5-hakem iç paneli + 1 cross-model dış denetim (R10b, R10c).
- **Yazarlık görüşmesi bu fazdan ÖNCE bitmiş olmalı (R10a)** — Yusuf + Selim hoca; venue ve
  yazar listesi netleşir. Konuşma taslağı asistandan istenebilir.
- **GATE G3:** submission checklist (R1-R10 hepsi ✅) → **ESWA gönderimi**.

**Takvim:** yarı-zamanlı ~13-17 hafta → hedef gönderim **Kasım-Aralık 2026**.
**Review gerçeği (beklenti yönetimi):** ESWA/KBS ilk karar 2-5 ay; en olası sonuç major
revision; kabul senaryosu 2027 ilk yarı. Bu NORMAL bir Q1 zaman çizgisidir; panik yok.

---

## 4. Bütçe (kaba)

| Kalem | Tahmin |
|---|---|
| Faz 1-3 API çağrıları (9 aile, ~20-30k çağrı, çoğu ucuz tier) | $100-300 |
| P2 güçlü-tier spot-check + yargıç çağrıları (R7) | $30-80 |
| Açık model (WSL2 lokal vLLM) | $0 (yedek Fireworks ~$10) |
| Toplam tavan (7 Tem kararı: maliyet kısıt değil) | **< $600** (gerçekçi: $200-400) |

---

## 5. Riskler ve önceden verilen kararlar

1. **Sonuçlar hikâyeyi bozarsa** (ör. bir sağlayıcıda L1=0%): tez zaten "oranlar genellenmez,
   kontrat her koşulda tutar" — her sonuç teze hizmet eder. Bulgular olduğu gibi raporlanır.
2. **Guardrails adalet tuzağı:** stok validator'lar sadakat kontrol etmez; bizim doğrulayıcıyı
   onların döngüsüne gömmeden yapılan kıyas "kendinle kıyas" eleştirisi yer. İki-kollu tasarım şart.
3. **Bütçe/emek taşması:** R1'deki minimumlar bilinçli mütevazı; matris büyütme cazibesine
   direnç. Önce DoD, sonra süs.
4. **Yazarlık:** 7 Tem 2026 kararı — tek yazar (Hakan). Q1 manuskripti yeni deneylerle
   baştan yazılan yeni bir eser olduğu için savunulabilir; riski sıfırlamak için gönderim
   öncesi eski ortak yazarlara tek satırlık nezaket bildirimi önerilir. Corresponding-author
   yükü (tüm hakem yazışmaları) tek kişide olacak; Faz 5 takvimine bu pay dahil.
5. **İsim çakışması:** §1'deki üç-ad ayrımı korunur.

---

## 6. Devir notları (gelecek asistan / model için)

- Bu dosya + hafıza dosyası (`project_mlcompass_inista.md`) + ölçüm kayıtları = tam bağlam.
- **Standing kurallar:** mlcompass repo'sunda otomatik commit+push YETKİLİ (spesifik dosya
  adlarıyla `git add`; asla `git add .`/`-A`; force push yok). Anahtarlar asla chat'e yazılmaz.
- **Ölçüm disiplini:** her canlı koşu bir `paper/ablation_live_<provider>_<tarih>.md` kaydı +
  harness commit pin'i ile belgelenir. Ön-kayıtlı tasarımdan sapma gerekçesiyle loglanır.
- **Kullanıcı profili:** Hakan hızlı çalışır, uzun koşuları kendi terminalinden başlatıp
  sonuçları yapıştırmayı tercih eder; token israfına duyarlıdır; em-dash sevmez; çıktı dili
  Türkçe'dir.
- Başlangıç noktası: **Faz 0 checklist'i**, ilk iş harness telemetri yükseltmesi.
