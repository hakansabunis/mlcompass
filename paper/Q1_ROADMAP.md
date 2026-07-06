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

- **R1 — Deney matrisi:** ≥3 ticari sağlayıcı (DeepSeek [var], OpenAI gpt-4o-mini,
  Anthropic Haiku/Sonnet) + ≥1 açık model (Qwen/Llama; WSL2+RTX3050 quantized ya da hosted);
  ≥4 veri kümesi; ≥5 sızıntı deseni. Ana sağlayıcıda N=200/hücre, diğerlerinde ≥100
  (alt-küme, ÖNCEDEN ilan edilmiş tasarımla).
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
- **R10 — Süreç:** (a) Yusuf Ünlü'nün onayı + Selim Akyokuş ile yazarlık/venue netliği
  **yazım fazından önce** bağlanmış; (b) Q1-kalibre iç 5-hakem paneli iki ardışık turda
  ≥ "minor revision" bandı; (c) cross-model dış denetim 0 P1 bulgu.

---

## 3. Faz planı (gate'li)

### Faz 0 — Altyapı + ön-kayıt (≈1 hafta)
- Harness telemetri yükseltmesi (R6) — **tüm yeni koşulardan önce**, yoksa koşular tekrarlanır.
- `paper/analysis_plan.md` **ön-kayıt**: hipotezler, metrikler, hücre tasarımı, karar kuralları,
  baseline adalet tasarımı. Koşulardan önce commit'lenir (hakemler bu disiplini övdü; sürdür).
- OpenAI + Anthropic hesap/kredi (~$20'şer başlangıç). **GÜVENLİK: anahtarlar yalnızca env
  var; asla chat'e/commit'e yapıştırılmaz (geçmişte 2 kez sızdı, ikisi de iptal edildi).**
- Açık model kararı: WSL2+RTX3050 (Qwen2.5-3B/7B Q4, bedava) vs hosted endpoint (~$20).
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
| Faz 1-3 API çağrıları (~10-15k çağrı, ucuz modeller) | $50-150 |
| Açık model (WSL2 lokal $0 / hosted) | $0-20 |
| Toplam tavan | **< $250** |

---

## 5. Riskler ve önceden verilen kararlar

1. **Sonuçlar hikâyeyi bozarsa** (ör. bir sağlayıcıda L1=0%): tez zaten "oranlar genellenmez,
   kontrat her koşulda tutar" — her sonuç teze hizmet eder. Bulgular olduğu gibi raporlanır.
2. **Guardrails adalet tuzağı:** stok validator'lar sadakat kontrol etmez; bizim doğrulayıcıyı
   onların döngüsüne gömmeden yapılan kıyas "kendinle kıyas" eleştirisi yer. İki-kollu tasarım şart.
3. **Bütçe/emek taşması:** R1'deki minimumlar bilinçli mütevazı; matris büyütme cazibesine
   direnç. Önce DoD, sonra süs.
4. **Yazarlık gerginliği:** Selim hoca "şimdi gönderme, güçlendir" demişti (14 Haz) — bu
   kampanya tam da onun istediği güçlendirme. Görüşmeye bu çerçeveyle gidilir; gizli gönderim
   ASLA (14 Haz'da tartışıldı ve reddedildi).
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
