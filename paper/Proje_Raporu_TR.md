# mlcompass — Projeyi Nasıl Yaptık?

**Teknik Anlatım ve Süreç Raporu**

*İstanbul Medipol Üniversitesi — Bilgisayar Mühendisliği — Bitirme Projesi*

**Proje Üyeleri:** Hakan Sabuniş · Yusuf Ünlü

**Tarih:** Haziran 2026

---

## 1. Projenin Kısa Özeti

mlcompass, makine öğrenmesi mühendislerinin tüm proje sürecinde
yanlarında olan, açık kaynak bir komut satırı asistanıdır. Veriye ilk
bakıştan modelin canlıya alınmasına kadar 11 ayrı komutla mühendise
tavsiye verir. Bunu yaparken, bugün popüler olan yapay zeka
yardımcıların en büyük zayıflığı olan **uydurma (hallucination)
problemini** çözmek için runtime düzeyinde zorlayıcı bir mimari
tasarladık.

Proje, Python ile yazıldı; PyPI üzerinden `pip install mlcompass`
komutuyla kurulabiliyor; MIT lisansı altında GitHub'da yayınlandı.
Test sürecinin sonunda **539 başarılı regresyon testi**, üç farklı
kalite kapısının (`ruff`, `ruff format`, `mypy --strict`) hepsinin
temiz geçtiği, **14 sürüm** ile mağaza-kalitesinde bir paket olarak
çıktı.

Bu doküman, akademik raporun teknik içeriğinden farklı olarak, **bu
projeyi gerçekten nasıl yaptığımızı** anlatıyor. Hangi kararları
neden aldığımızı, hangi araçları kullandığımızı, neyi yanlış yapıp
sonra düzelttiğimizi açık dille anlatmaya çalıştık. Hocamızın veya
benzer projeleri yapacak gelecek öğrencilerin, "ne öğrenebiliriz?"
sorusuna cevap bulması için yazıldı.

---

## 2. Problemden Yola Çıkış — Bu Projeyi Neden Yaptık?

### 2.1 Karşılaştığımız boşluk

Makine öğrenmesi öğrenirken her birimiz şu deneyimi yaşadık: elimizde
bir CSV dosyası var, modelimi eğittim, ama eğitim sırasında veya
sonrasında bir şey yanlış gitti — örneğin model çok yüksek skor
verdi ya da training loss düşerken validation loss artmaya başladı.
Bu noktada **"şimdi ne yapmalıyım?"** sorusuna cevap verecek tek bir
araç yoktu.

Mevcut araçlar tek tek güzel ama hiçbiri tavsiye vermiyor:

- **pandas-profiling** verinin istatistiklerini güzel gösteriyor ama
  "hangi sütun senin hedef değişkenin?" sorusunu sormuyor.
- **Weights & Biases** ve **TensorBoard** training run'larını
  takip ediyor ama "bu loss eğrisi overfitting'e benziyor" demiyor.
- **GitHub Copilot** ve **Cursor** kod yazıyor ama
  `optimizer=Adam(momentum=0.9)` gibi semantik olarak yanlış bir ML
  kodunu uyarısız yazıyor (Adam optimizer'ı `momentum` parametresi
  kabul etmiyor).

Daha kötüsü, son dönemde popüler olan **LLM tabanlı yardımcılar**
güzel bir tavsiye veriyor ama **çoğu zaman uyduruyor**. 2025'te
yalnızca arXiv, bioRxiv ve SSRN üzerinde 146.932 hayalî atıf tespit
edildi [Zhao ve diğ., 2026]. Kullanıcı bu uydurmalara güveniyor,
çünkü sistem kendinden emin yazıyor.

### 2.2 Çözmek istediğimiz sorun

İki ana hedef belirledik:

1. **Komple bir ML pipeline asistanı yapmak** — veriye ilk baktığımız
   andan, modeli canlıya aldığımız ana kadar her aşamada
   kullanıcının yanında olan bir araç.
2. **Yapay zekanın uydurmasının önüne fiziksel olarak geçmek** —
   sadece "uydurmamalısın" diye prompt vermek yetmez, runtime düzeyinde
   yapamayacak hale getirmek lazım.

Bu iki hedefin kesişiminde mlcompass doğdu.

---

## 3. Mimari Kararlar — Neden Böyle Tasarladık?

### 3.1 İki katmanlı beyin

Projenin **en temel mimari kararı** şu: bilgiyi üreten katmanı,
bilgiyi anlatan katmandan **fiziksel olarak ayırmak.**

Bu fikre nereden geldik? Şöyle bir gözlem yaptık: bir LLM bir CSV'ye
bakıp "bu sütun şüpheli" derken aslında iki şey yapıyor — (1) veriyi
analiz ediyor, (2) sonucu anlatıyor. Birinci adımdaki *analiz* deterministik
bir iş; pandas ile saniyeler içinde hesaplanır. İkinci adımdaki
*anlatım* doğal dil üretimi gerektiriyor; LLM bu işte iyi. **Ama LLM
bu iki işi tek seferde yapınca, anlatım aşamasında veriye ait olmayan
şeyler eklemeye başlıyor — uydurma orada doğuyor.**

Çözüm açıktı: iki işi iki katmana ayıralım. Veriye sadece pandas
baksın; LLM sadece pandas'ın sonucunu açıklasın.

```
src/mlcompass/tools/   ← Pure Python, no LLM
src/mlcompass/agents/  ← Claude, narrator only
```

Bu basit ayrımın iki büyük faydası oldu:

1. **Saf Python katmanı uydurma yapamaz**, çünkü metin üretmiyor.
   Sadece dictionary döndürüyor.
2. **LLM katmanı opsiyonel.** Kullanıcı `--no-llm` derse, sistem
   yine de çalışıyor — sadece deterministik analizi gösteriyor.
   API anahtarı olmayanlar bile kullanabiliyor.

### 3.2 Dört kullanıcı yüzeyi, tek tool katmanı

Hedeflediğimiz kullanıcı kitlesi geniş — terminal seven, sohbet
asistanı kullanan, otonom agent isteyen. Hepsi için ayrı araç yapmak
yerine, **aynı 11 aracı dört farklı yüzeyden ulaşılabilir hale
getirdik.**

Yüzeyler:

- **CLI** (terminal): Click kütüphanesi ile, shell script'ten veya
  CI pipeline'dan çağrılır.
- **MCP server**: Anthropic'in 2024'te tanıttığı Model Context
  Protocol üzerinden Claude Desktop, Cursor, Continue, Claude Code
  gibi sohbet asistanlarına bağlanır.
- **Self-driving agent**: `mlcompass agent "data.csv'mi production'a
  götür"` gibi tek cümlelik prompt'larla otonom çalışır.
- **Claude Code slash commands** (v0.8.0): `/mlc-advise data.csv` gibi
  tek tuşla ulaşılabilen komutlar.

### 3.3 Kalıcı proje hafızası — `.mlcompass/`

`.mlcompass/` klasörü, git'in `.git/` klasörü gibi proje köküne
yerleşen bir state directory. İçinde:

- `project.yaml` — proje meta-bilgisi
- `context.json` — şimdiye kadar verilen kararların kronolojik listesi
- `datasets/` — analiz edilen veri setlerinin fingerprint'leri
- `runs/` — training run config + metrik kayıtları
- `advice.log` — komut çağrılarının JSONL log'u
- `cache/` — LLM prompt-cache hit'leri için

Bu sayede `mlcompass deploy` komutu çalıştığında, sistem zaten hangi
dataset, hangi hedef sütun, hangi model, hangi metrik kararlarını
**hatırlıyor.** Tekrar tekrar bilgi vermek gerekmiyor.

---

## 4. Anti-Hallucination Kontratı — Asıl Teknik Katkımız

Projenin **en orijinal kısmı** burası. Üç katmandan oluşan ve her
katmanı tek başına zayıf, üçü birlikte sıkı olan bir kontrat.

### 4.1 Sistemin doğuşu

İlk versiyonumuzda Claude'a basit bir prompt yazdık: "Bu kanıt
dictionary'sine bak ve veri sızıntısı olup olmadığını söyle." Çalışıyordu —
genelde. Ama bir hafta sonra şunu fark ettik: bazen Claude veri
dictionary'sinde olmayan sütun isimlerinden bahsediyor. Örneğin
sentetik bir veri setimizde `log_target_v2` adında bir sütun varken
Claude *"target_score sütunu şüpheli"* yazıyordu. **`target_score`
hiçbir yerde yok.** Hayalî bir sütun. Ve Claude bunu büyük bir özgüvenle
söylüyordu.

Bunu prompt mühendisliği ile çözmeye çalıştık — daha katı talimatlar
verdik, JSON formatına soktuk, örnekler ekledik. Hallucination oranı
%8'den %1'e düştü. **Ama hala %1.** 100 vakanın 1'inde uyduruyordu.
Production ortamı için bu kabul edilemez bir oran.

Sonunda şunu anladık: **promptlar tavsiyedir; gerçekten zorlayıcı
yapı runtime'da olmalı.**

### 4.2 Üç katman

**Katman 1 — Saf Python kanıt üretici (`tools/leakage.py`).**
Predictions tablosunu okuyoruz, üç sinyal hesaplıyoruz:

1. **Pearson korelasyonu** (linear leak'leri yakalar)
2. **Spearman korelasyonu** (monoton transformasyon leak'lerini
   yakalar — `log(target)`, `sqrt(target)` gibi)
3. **Perfect-match oranı**: `y_pred == y_true` olan satırların yüzdesi

İkisinin maksimumunu alıyoruz: `max(|Pearson|, |Spearman|) > 0.97`
ise aday leak sütun. Bu eşik yüksek tutuldu çünkü gerçek mühendislik
ortamında 0.97 üstü korelasyon ya gerçek bir leak ya da hedefle
mükemmel doğrusal ilişki kuran bir feature anlamına gelir — her iki
durumda da kullanıcı bilmek ister.

**Önemli not:** Bu modülün hiç LLM çağrısı yok. Saf Python. Bu yüzden
asla uydurma yapmaz.

**Katman 2 — Katı prompt'lu Claude narrator (`agents/leakage_investigator.py`).**
Bu modül, Layer 1'in ürettiği dictionary'yi alıp Claude'a gönderiyor.
System prompt'ta üç açık talimat var:

1. *"Sadece bu dictionary'de geçen sütunlardan bahset."*
2. *"Emin değilsen 'cannot_determine' de, tahmin etme."*
3. *"Kod yaması önerme — sadece manuel kontrol önerebilirsin."*

Sentetik test setimizde, bu prompt ile hallucination oranı %8'den
%1'e düştü. İyileşme önemli ama yeterli değil.

**Katman 3 — Runtime şema sınırı.** Asıl marifet burada. Claude
cevabını "tool use" formatında veriyor — yani bir tool çağırıyor ve
parametreleri JSON ile dolduruyor. Bizim tool'umuzun input şeması
böyle:

```json
{
  "columns_referenced": {
    "type": "array",
    "items": {
      "type": "string",
      "enum": ["log_target_v2", "near_target_proxy", "feature_3", ...]
    }
  },
  "narration": { "type": "string" }
}
```

**Bu `enum` listesini runtime'da, kanıt dictionary'sindeki sütun
isimlerinden üretiyoruz.** Yani şema **çalışma anında** veriye göre
şekilleniyor.

Eğer Claude `revenue` gibi enum'da olmayan bir sütun ismi yazarsa,
Anthropic SDK `ToolInputValidationError` fırlatıyor. Biz bunu yakalayıp
agent'ı yeniden çağırıyoruz. **Claude'un fiziksel olarak hayalî bir
sütun ismiyle cevap göndermesi imkansız.**

Bu Layer 3 sayesinde hallucination oranı 200 örneklemde %0.0'a düştü
(Wilson 95% güven aralığı [0.00, 1.83]).

### 4.3 Sloganımız

Bu deneyimden çıkardığımız ders, projemizin sloganı oldu:

> **Promptlar tavsiyedir.** Şemalar **zorlamadır.**

Bir LLM sistemi tasarlarken, modelin "uymasını umduğunuz" şeyleri
runtime'da gerçekten **uymak zorunda bırakacak şekilde** yapılandırın.
Sadece nazikçe rica ederseniz, sistem %1'lik bir oranla rica
ettiğinizden farklı davranacaktır.

---

## 5. Kullandığımız Teknik Stack

Projeyi gerçekten makinece çalıştırmak için kullandığımız araçlar:

### 5.1 Çekirdek Python paketleri

| Paket | Ne için kullandık |
|---|---|
| **`pandas`** | CSV/Parquet okuma, kolon analizi |
| **`numpy`** | Sayısal hesaplamalar |
| **`scipy.stats`** | Pearson/Spearman korelasyon, PSI/KS/Chi-squared drift testleri |
| **`scikit-learn`** | Metrik hesaplama (AUC, F1, R² gibi) ve threshold sweep |
| **`Click`** | Komut satırı arayüzü |
| **`rich`** | Renkli terminal çıktısı, tablo render |
| **`pyyaml`** | `project.yaml` ve config dosyaları |
| **`tbparse`** | TensorBoard event file okuma |
| **`ast`** | Eğitim script'lerini statik analiz |
| **`pytest`** | Test framework (539 başarılı test) |

### 5.2 LLM entegrasyonu

| Paket | Ne için kullandık |
|---|---|
| **`anthropic`** | Doğrudan Anthropic API çağrıları, tool-use schema enforcement |
| **`mcp`** | Model Context Protocol server'ı |
| **`claude-agent-sdk`** | Claude Code CLI ile entegrasyon |
| **`agentlite`** | Custom Claude agent library (kendi yazdığımız) |

### 5.3 Kalite ve geliştirme araçları

| Araç | Amaç |
|---|---|
| **`ruff`** | Hem lint hem format — hızlı (Rust ile yazılmış) |
| **`mypy --strict`** | Statik type kontrolü, gizli `Any` yok |
| **`pytest-cov`** | Test coverage ölçümü |
| **`twine`** | PyPI'ya yayın |
| **`build`** | Wheel + sdist üretimi |

### 5.4 Dış sistemler

- **GitHub** — kaynak kod, GitHub Releases, GitHub Actions CI
- **PyPI** — paket dağıtımı, 14 sürüm yayınlandı
- **Anthropic API** — Claude 3.5 Sonnet (`claude-3-5-sonnet-20241022`)
- **Kaggle** — field test için 5 gerçek veri seti

---

## 6. Geliştirme Süreci — Faz Faz

Projeyi sıfırdan v0.8.1'e kadar 10 ana fazda geliştirdik. Her faz
küçük, kapalı bir iş paketi olarak kuruldu. Bu sayede her faz
sonunda çalışan bir versiyon vardı.

### Faz 1 — `init` ve `advise` (v0.1.0, Mayıs 2026)

İlk önce proje iskeletini kurduk: `mlcompass init` ile `.mlcompass/`
klasörünü oluşturmak, `mlcompass advise data.csv` ile veri setini
analiz etmek. Bu fazda öğrendiğimiz en önemli şey: **veri seti
analizinin yarısı, hedef sütununu doğru tahmin etmek.** Müşterinin
"hangi sütunu tahmin etmek istiyorum?" demesini beklemek yerine,
yaygın hedef isimlerinden (price, target, churn, survived, vs.)
bir liste tutuyoruz ve veriye bakıp tahmin ediyoruz.

### Faz 2 — `audit`, `watch`, `compare` (v0.2.0)

Eğitim öncesi script denetimi (`audit`), eğitim sırasında log izleme
(`watch`), iki eğitim koşusunu karşılaştırma (`compare`).
`audit` için Python'ın `ast` modülünü kullandık — 8 yaygın hatayı
otomatik tespit ediyor (random seed eksik, validation split yok,
`log(x)` epsilon clamp olmadan, vs.).

### Faz 3 — `evaluate` ve threshold optimizasyonu (v0.3.0)

Predictions tablosundan metrikleri hesaplama, threshold sweep,
confusion matrix. Bu faz, sonradan Faz 9'da çıkacak olan **leakage
investigation panel'inin** temel altyapısını kurdu.

### Faz 4 — `deploy` (v0.3.0)

Model dosyasını ve dependencies'i analiz edip deployment hazırlığı
kontrol etme. AWS Lambda için package size limit, Docker base image
seçimi, SageMaker constraint'leri gibi target-specific checkler.

### Faz 5 — `status` ve proje hafızası (v0.3.0)

Proje state'inin kullanıcıya gösterilmesi. Bu faz, `.mlcompass/`
klasörünün şeklini de netleştirdi.

### Faz 6 — MCP Server (v0.4.0)

`mlcompass-mcp` script'ini yazdık. Anthropic'in resmi `mcp` Python
paketini kullanarak, 8 mlcompass aracını JSON-RPC üzerinden Claude
Desktop, Cursor ve Claude Code'a açtık. Bu faz, bizi **multi-surface
mimari konusunda olgunlaştırdı** — aynı kapasitenin iki yüzey
üzerinden çıkması, parity sorununu doğurdu.

### Faz 7 — Self-driving Agent (v0.5.0)

`mlcompass agent` komutu. İki backend yazdık: birincisi Anthropic API'ye
direkt giden (`api`), ikincisi Claude Code CLI üzerinden geçen
(`claude-code`). Permission gating ekledik — agent sadece bir mutating
operasyon (`mlcompass_init`) için "y/N" onayı istiyor; gerisi
otomatik.

### Faz 8 — `monitor` ve `optimize` (v0.6.0)

Post-deployment drift detection (PSI/KS/Chi²) ve hyperparameter
search agent. Bu fazda **agent memory** ekledik — `--resume <run-id>`
ile session'lar arası devam edebilme.

### Faz 9 — Anti-Hallucination Contract (v0.7.0)

Projenin onur konuğu. `tools/leakage.py` ile deterministik kanıt
üretici, `agents/leakage_investigator.py` ile katı prompt'lu narrator,
ve runtime tool-use schema boundary. Üç katmanın hep birlikte
ablation deneyi N=200 üzerinde yapıldı.

Sonra v0.7.1, v0.7.2, v0.7.3 patch'leri ile Kaggle field test bug'larını
düzelttik (bkz. Bölüm 7).

### Faz 10 — Claude Code Slash Commands (v0.8.0)

11 hazır `.md` dosyası ve `mlcompass install-slash-commands` CLI
komutu. `mlc-advise`, `mlc-evaluate`, `mlc-leak` gibi tek tuşla
ulaşılabilen komutlar.

### v0.8.1 — Paper supporting materials

Akademik raporun reviewer feedback'i sonrası yapılan revize için
gerekli scriptler ve dokümanlar. Wilson 95% CI hesaplayan
reproducibility script, threat model dokümanı, known limitations
dokümanı, per-bug field test detayları.

---

## 7. Kaggle Field Tests — En Çok Öğrendiğimiz Yer

Sentetik unit testleri yazarken kendimizi çok iyi sandık. Yazdığımız
testleri **çeşitli edge case'leri kapsadığını** düşünüyorduk. Sonra
**bir Kaggle veri seti açtık** ve gerçek dünya unitery test'imizin
ne kadar dar olduğunu gösterdi.

### Test 1 — Telco Churn (baseline)

Bu testte hiçbir bug çıkmadı. Beklenen sonuç — Telco zaten geliştirme
sırasında kullandığımız veri setiydi.

### Test 2 — Ames House Prices (v0.7.0, 3 bug)

**30 saniye içinde üç bug:**

1. `mlcompass advise train.csv` çıktısı: *"hedef sütun bulunamadı."*
   Halbuki `SalePrice` apaçık hedef. Bizim hedef-isim listemizde
   sadece `price`, `target`, `cost` vardı. Kaggle'ın canonical hedef
   ismi olan `saleprice` listede yoktu.
2. `YearBuilt` sütunu ID-like (çoğunluk unique) heuristics'i ile
   ID sanıldı ve uyarı verildi.
3. `PoolQC` sütunu %96 NaN. IQR hesaplaması `nan` üretti, warning
   formatter crash etti.

Üçünü de düzelttik. Yeni regresyon testleri yazdık. v0.7.0
yayınlandı.

### Test 3 — Titanic (v0.7.1, 1 bug)

`Survived` binary hedef yüksek-güven listesinde yoktu. Eklerken bir
dizi başka canonical Kaggle hedef ismini de ekledik: `Outcome`,
`Diagnosis`, `Defaulted`, `Approved`, vs. Aynı akşam patch yayınlandı.

### Test 4 — Penguins (v0.7.2, 3 bug)

Bu test en zorluydu.

İlk bug: `species` 3-class hedefi binary olarak yanlış sınıflandırılıyordu.
Sebep: `_infer_task` fonksiyonunda class count check'i `nunique() == 2`
yazıyordu, halbuki `<= 2 if binary else <= N` olmalıydı. Bir karakter
düzeltme.

İkinci bug çok daha öğretici. Penguins pipeline'ını **Claude Code'un
MCP entegrasyonu** üzerinden çalıştırdık. Sonra `mlcompass_status`
dedik. Cevap **boş geldi.** Hiçbir karar listelenmedi.

Sebep şuydu: CLI handler'ımız her komutu **"ledger'a yaz" adımı ile
sarıyordu.** Yani `mlcompass advise` her çalıştığında, `.mlcompass/
context.json` dosyasına bir karar kaydı ekleniyordu. **Ama MCP server
tool fonksiyonları bu adımı atlıyordu** — doğrudan deterministik tool
fonksiyonlarını çağırıyor, ledger'a yazmıyorlardı.

Yani aynı capability iki yüzey üzerinden farklı davranıyordu. Bu
**bizi gerçekten şaşırttı** — kod aynı, davranış farklı.

Çözüm: `_persist_to_ledger` adında ortak bir helper fonksiyonu yazdık.
Hem CLI hem MCP onu çağırıyor. Ayrıca CLI/MCP parity test'leri
ekledik — her komut için "CLI'den çağrılınca ledger'a yazıyor mu?
MCP'den çağrılınca yazıyor mu?" diye explicit testler.

### Test 5 — Insurance Charges (v0.7.3, 4 bug)

`charges` regresyon hedefi düşük güvenle algılanıyordu — finans/sigorta
hedef isimleri listemizde yoktu. `charges`, `total_charges`,
`medical_cost`, `fee`, `tuition`, `expenses`, `arpu`, `spend`
ekledik.

Daha kötüsü: v0.7.2'deki MCP fix yetersiz çıktı. `_persist_to_ledger`
sadece `decisions[]` listesine ekleme yapıyordu ama active-state
field'larını (`project_type`, `target_column`, `active_dataset`)
güncellemiyordu. CLI bunları ayrı bir adımda update ediyordu — yine
parity gap. Düzelttik, `_persist_to_ledger`'a `state_updates`
parametresi ekledik.

### Çıkardığımız ders

**Gerçek veri setleri, sentetik test setlerinin asla yakalayamayacağı
bug'ları yakalıyor.** Bunun bir teorik açıklaması yok — sadece
production code path'leri unit test yazarının düşünmediği şekillerde
egzersize ettiriyor. Pratik sonuç: artık her release önce bir Kaggle
dataset'te dry-run yapılıyor. **"Field-test-then-patch"** ritüelimiz
bu testlerin ürünü.

---

## 8. Yazılım Mühendisliği Disiplinimiz

Bir capstone projesinde bile, gerçek bir paket gibi davrandık. Aşağıdaki
disiplinleri sıkı sıkı uyguladık:

### 8.1 Test-driven development

Her özellik regresyon testleriyle birlikte geldi. Final test seti
**539 başarılı test, 2 atlanmış (intentional)**. Test kategorileri:

- **Tool layer testleri**: Sentetik ve Kaggle CSV'leri üzerinde
  deterministic doğruluk kontrolü
- **Agent layer testleri**: Mock'lanmış Anthropic client ile — hızlı,
  offline, deterministic
- **MCP server testleri**: In-process JSON-RPC round-trip'leri
- **CLI integration testleri**: Click'in `CliRunner` helper'ı ile
- **Anti-hallucination contract testleri**: Narrator'a kanıt
  dictionary'sinde olmayan sütun ismini söylemeye prompt'la ve
  schema boundary'nin reddettiğini doğrula

### 8.2 Üç kalite kapısı

Her commit ve her release tag'inde üç kapı çalışıyor:

- **`ruff check`** — 6 rule grubu ile style + correctness lint
- **`ruff format --check`** — tutarlı format
- **`mypy --strict`** — 51 source file üzerinde tam statik typing,
  implicit `Any` yok

v0.1.0'dan v0.8.1'e kadar her release tag'inde üçü de temiz. Strict
mypy bizi production'a gitmeyecek iki bug'tan kurtardı (bir
`Optional[str]`'ın `Path()` constructor'ına geçirilmesi ve bir
list-of-dict'in tip imzasının sessizce kaymış olması).

### 8.3 Release ritüeli

14 release'i her birini 6 adımda yayınladık:

1. `__init__.py` ve `pyproject.toml`'da version bump
2. CHANGELOG entry — bot için değil, **insan için** yazılı
3. Tüm kalite kapıları (`pytest && ruff check && ruff format
   --check && mypy --strict src`)
4. Build (`python -m build`) + `twine check`
5. `git commit`, `git tag`, `git push`
6. `twine upload` ile PyPI'ya, sonra GitHub Release

**14 release'de sıfır post-release rollback.** Ritüel işe yarıyor.

### 8.4 CHANGELOG dürüstlüğü

CHANGELOG entry'lerini "neyi" yerine **"neden"** üzerine yazdık.
Yani sadece "v0.7.3'te bug fix" değil, "v0.7.3'te Field Test #5
Insurance Charges üzerinde tespit edilen şu davranışı şu sebeple
düzelttik" tarzında. Bunun bedeli release başına 20 ekstra dakika oldu.
Karşılığı: Field Test #5 sırasında target name listesinde regresyon
çıktığında, CHANGELOG'da "target name" arama yaparak o sütuna ne
zaman, neden eklenen tüm değişikliklerin nedenleriyle birlikte
listesini hemen bulduk. **CHANGELOG, proje hafızamıza dönüştü.**

---

## 9. Reviewer Geri Bildirimi ve Revizyon (v0.8.1)

Akademik raporu yazdıktan sonra, **3 peer-reviewer tarzı** geri
bildirim aldık (kendi içimizde role-play yaparak). Üç reviewer da
ortak şu zayıflıkları işaret etti:

1. **N=200 yetersiz, Wilson güven aralığı verilmemiş.**
2. **Tek LLM (Claude) üzerinde test edilmiş, cross-model generalization
   yok.**
3. **"Anti-hallucination contract" başlığı abartılı; mekanizma
   aslında Anthropic'in tool-use API'sinin standart bir özelliği —
   yeni olan kısım runtime-evidence-bound enum.**
4. **Outlines, LMQL, OpenAI strict mode gibi constrained-generation
   literatürü ile karşılaştırma yapılmamış.**

Bu eleştirilere **kod düzeyinde cevap vermek** için v0.8.1'i
çıkardık. Sadece paper revize etmek dürüstçe olmazdı — eleştirileri
gerçekten adres etmek için:

- **`scripts/reproduce_hallucination_ablation.py`** yazdık. Default
  mock mode'da deterministic olarak çalışıyor, paper'daki Tablo I'i
  Wilson 95% CI ile reproduce ediyor. `--mode live --n 2000` ile
  gerçek Anthropic API'ye çağrı yapıp güven aralıklarını sıkmak
  mümkün (~$48 maliyet).
- **`scripts/measure_latency.py`** yazdık. Layer 3 retry rate ve
  end-to-end latency ölçüyor.
- **`docs/THREAT_MODEL.md`** — kontratın neyi koruduğu, neyi korumadığı
  uzun form doküman.
- **`docs/KNOWN_LIMITATIONS.md`** — 8 maddelik limitations list,
  severity weighting ile.
- **`docs/FIELD_TEST_BUGS.md`** — 11 field-test bug'ının her birinin
  detayı: kategori, patch release, regression test referansı.

Sonra paper'ı revize ettik:

- Başlığı **"Schema-Bounded LLM Narrator"** olarak değiştirdik (daha
  dürüst).
- Wilson güven aralıklarını Tablo I'e ekledik.
- Constrained-generation literatürünü (referans [9]-[13]) ekledik.
- **Limitations and Residual Risks** bölümünü Bölüm V.D olarak
  açıkça yazdık — 5 maddelik liste.

Sonuçta üç reviewer'ın ortalama skoru **5.33'ten 7.0'a** çıktı.

---

## 10. Karşılaştığımız Zorluklar ve Çözümler

### 10.1 Prompt engineering, sıfır garantisini vermiyor

Anti-hallucination kontratının ilk versiyonu sadece prompt'a
dayanıyordu. Bir hafta uğraştık. Hallucination'ı %8'den %1'e
indirdik. **%1 yine de production için kabul edilemez.** Sonunda
Layer 3 (runtime schema enforcement) eklemek **fiziksel olarak**
imkansız hale getirdi. Çıkardığımız ders: kritik garantiler runtime
katmanında, prompt katmanında değil.

### 10.2 Surface parity gap (Field Test #4)

Bahsettiğimiz gibi, MCP server ve CLI farklı davranıyordu. **Aynı
fonksiyonu çağırıyor olmaları parity'yi garanti etmiyor** çünkü
CLI'in Click handler'ı bir "post-process" adımı ekliyordu, MCP
wrapper bunu atlıyordu. Lesson: bir capability iki yüzey üzerinden
çıkıyorsa, **parity bedava gelmiyor, açıkça test edilmesi gerekiyor.**

### 10.3 Sentetik testler real-world bug'ları bulamadı

11 Kaggle field-test bug'unun **hiçbiri** unit test suite'imiz
tarafından önceden öngörülmemişti. Bu gözlem, bizi field-test
ritüeline yönlendirdi.

### 10.4 Reviewer feedback'i ciddi alma

Paper'ı yazdıktan sonra "tamam, bitti" demek isterken **reviewer
yorumlarını kendimize uyguladık** ve aklımıza gelmemiş zayıflıkları
gördük. v0.8.1 sürecindeki revize ve kod ekleri, bu ciddiye
almanın sonucu. **Geri bildirimi kod düzeyinde adres etmek, paper
düzeyinde adres etmekten çok daha güçlü.**

---

## 11. Sonuçlar ve Metrikler

Proje sonuçları:

| Metrik | Değer |
|---|:-:|
| CLI komutları | 11 |
| MCP araçları | 8 |
| Claude Code slash komutları | 11 (`mlc-*`) |
| Kaynak dosyaları | 51 |
| Geçen test | 539 |
| Atlanan test (intentional) | 2 |
| Lint / format / type hata | 0 |
| PyPI release sayısı | 14 |
| Kaggle field test sayısı | 5 |
| Field-test bug yakalama | 11 |
| Patch release sayısı | 4 |

Phantom-column hallucination ablation sonuçları (N=200, Claude 3.5
Sonnet, Wilson 95% CI):

| Konfigürasyon | Oran | 95% CI |
|---|:-:|:-:|
| Layer 1 (no prompt, no schema) | 8.0% | [4.93, 12.66] |
| Layer 1+2 (strict prompt) | 1.0% | [0.27, 3.56] |
| Layer 1+2+3 (schema-bounded) | **0.0%** | [0.00, 1.83] |

Latency ve maliyet (Anthropic API, Claude 3.5 Sonnet, Aug 2025
fiyatlandırma):

| Metrik | Değer |
|---|:-:|
| Ortalama API çağrısı / `evaluate --llm` | 1.04 |
| End-to-end latency (mean ± std) | 2.1 ± 0.4 saniye |
| Çağrı başına maliyet | ~$0.008 |
| Layer 3 retry rate | %4 |

---

## 12. Gelecek Çalışmalar

Belgelenmiş olarak bilinen ama henüz adres edilmemiş 5 çalışma alanı
var (`docs/KNOWN_LIMITATIONS.md` içinde):

1. **Cross-model evaluation**: GPT-4, Llama-3, Gemini üzerinde aynı
   ablation'ı koşturup cross-provider rate'leri raporlama. v0.9'da
   OpenAI backend eklendikten sonra mümkün.
2. **Constrained-generation benchmark**: Outlines (Willard & Louf,
   2023) ve OpenAI strict mode ile direkt karşılaştırma. Llama-3-8B
   + Outlines kombinasyonu öncelikli.
3. **Geniş N (2000+) live run**: Wilson güven aralıklarını sıkmak
   için. Reproducibility script'i hazır; ~$48 maliyet.
4. **External user study**: Q3 2026'da bir Türk fintech şirketiyle
   görüşmeler ilerliyor. 3 internal projede mlcompass denecek.
5. **Plug-in system**: HuggingFace, PyTorch Lightning, vs. için
   domain-specific analyzer'ları üçüncü tarafların ekleyebilmesi.

---

## 13. Sonuç

mlcompass projesini yaparken **iki şey** öğrendik:

Birincisi **teknik**: LLM tabanlı sistemlerde garantiler runtime
katmanında verilmeli, prompt katmanında değil. *"Promptlar
tavsiyedir, şemalar zorlamadır."* Bu mantra, schema-bounded contract
tasarımının özü.

İkincisi **metodolojik**: gerçek veri setleri sentetik testlerin
asla yakalayamayacağı bug'ları yakalıyor. *"Real datasets find bugs
synthetic ones don't."* Bu yüzden field-test-then-patch ritüeli
projemizin en değerli engineering pratiği.

Proje açık kaynak. Kaynak kod, CHANGELOG, threat model, limitations
dokümanı, reproducibility scriptleri — hepsi GitHub'da. Üçüncü
şahısların kullanması, eleştirmesi, geliştirmesi için hazır.

Bu projeyi yapmak bize sadece teknik beceri kazandırmadı — bir
ürünü baştan sona, mühendislik disipliniyle yayınlamayı öğretti.
Bitirme projesi olarak yapılması gereken neyse onun çok ötesinde,
gerçek bir açık kaynak ürün ortaya çıkardık.

Teşekkürler.

---

**Hakan Sabuniş & Yusuf Ünlü**
*İstanbul Medipol Üniversitesi, Bilgisayar Mühendisliği*
*Haziran 2026*

**Kaynak kod:** <https://github.com/hakansabunis/mlcompass>
**Paket:** <https://pypi.org/project/mlcompass/>
**Lisans:** MIT
