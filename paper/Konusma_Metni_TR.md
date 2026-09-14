# mlcompass Sunum — Türkçe Konuşma Metni

**13 slayt · ~12-13 dakika konuşma + ~2 dakika Q&A**

Her slayt için ortalama 45-60 saniye konuşma. **Kalın** yazılı kısımlar
vurgu yapılacak yerler. *İtalik* yazılı kısımlar slaytta görünmeyen
hatırlatmalar.

---

## 🎬 Slayt 1 — Title (~30 saniye)

Merhaba arkadaşlar, ben Hakan Sabuniş, yanımdaki Yusuf Ünlü. Istanbul
Medipol Üniversitesi Bilgisayar Mühendisliği bitirme projemizi sunmak
için buradayız.

Projemizin adı **mlcompass** — Türkçesi pusula gibi, makine öğrenmesi
mühendislerinin yanında, projenin başından sonuna kadar onlara yol
gösteren bir araç. Açık kaynak, MIT lisansı altında, PyPI üzerinde
herkesin `pip install mlcompass` yazıp kullanabileceği canlı bir
paket.

Sunumda şunlardan bahsedeceğiz: önce çözmeye çalıştığımız problemi
açıklayacağız, sonra mimari kararlarımızı, ardından **anti-
hallucination kontratı** dediğimiz ana teknik katkımızı, deney
sonuçlarımızı, karşılaştığımız zorlukları, ve canlı bir demo da
yapacağız.

*[Slaytı açtıktan sonra 5 saniye sessiz dur ki dinleyici başlığı okusun.]*

---

## 🎬 Slayt 2 — The Problem (~60 saniye)

Şu anki makine öğrenmesi ekosistemi parçalanmış. Bir veri bilimcisi
verisine bakmak için pandas-profiling açıyor, modelini eğitirken
Weights & Biases ya da TensorBoard kullanıyor, kodunu yazarken Copilot
veya Cursor'dan yardım alıyor — **ama hiçbiri ona tavsiye vermiyor.**

pandas-profiling verinin dağılımını gösteriyor ama "hangi sütun senin
hedef değişkenin?" sorusunu bile sormuyor. Weights & Biases loss
eğrisini çiziyor ama "bu loss eğrisi overfitting'e benziyor" diyemiyor.
Copilot kod tamamlıyor ama `Adam optimizer'ına momentum=0.9` yazarsan —
ki bu **geçersiz bir parametre** — uyarmıyor.

Daha kötüsü, son dönemde popüler olan LLM tabanlı yardımcılar —
**hallucination yapıyorlar.** Var olmayan sütun isimleri uyduruyorlar,
gerçek olmayan problemler için çözümler öneriyorlar. 2025'te sadece
arXiv, bioRxiv ve SSRN'de **146 binden fazla hayali atıf** tespit
edildi. **Yanlış güvenle yanlış bilgi**, hiç bilgi olmamasından çok
daha tehlikeli.

İşte bu boşluğu doldurmak için mlcompass'i tasarladık.

*[Vurgu: 146 bin rakamı slaytın altında görünüyor — onu işaret et.]*

---

## 🎬 Slayt 3 — mlcompass in 30 Seconds (~50 saniye)

mlcompass'i bir cümleyle özetlersek: **Açıklamasına izin verilen ama
uydurmasına izin verilmeyen, ML projesinin başından sonuna kadar size
eşlik eden açık kaynak bir komut satırı aracı.**

11 komutumuz var — `init` ile yeni proje açıyorsunuz, `advise` ile
verinize bakıyoruz, `audit` ile eğitim scriptinizi denetliyoruz,
`watch` ile training log'unuzu canlı izliyoruz, `compare` ile iki
run'ı karşılaştırıyoruz, `evaluate` ile sonuçları yorumluyoruz,
`deploy` ile ne kadar production'a hazır olduğunuzu kontrol
ediyoruz. Artı `monitor`, `optimize`, `status`, `agent` komutları.

Ve bu 11 araç **dört farklı yüzey** üzerinden ulaşılabilir: terminal,
Claude Desktop / Cursor / Claude Code gibi sohbet asistanları içinden
**MCP server** olarak, otonom çalışan bir **self-driving agent**
olarak, ve Claude Code'un slash komut menüsünden.

En önemlisi: `.mlcompass/` klasöründe bir proje hafızası var — yarın
geri geldiğinizde sistem ne karar verdiğinizi hatırlıyor.

---

## 🎬 Slayt 4 — System Architecture (~60 saniye)

Sistemin mimarisi üç katmandan oluşuyor.

Yukarıda kullanıcıya bakan üç yüzey görüyorsunuz: solda terminal
**CLI**, ortada sohbet asistanlarına bağlanan **MCP Server**, sağda
otonom çalışan **self-driving agent**.

Ortada **iki katmanlı beyin** dediğimiz kritik kısım var. Mavi olan
üst katman **deterministic tool layer** — saf Python kodu, hiç yapay
zeka yok, veriye bakıyor ve yapılandırılmış bir dictionary üretiyor.
Mor olan alt katman ise opsiyonel **Claude tabanlı narrator** — bu
dictionary'yi doğal dilde açıklıyor.

Altta da **kalıcı proje hafızası** — `.mlcompass/` klasörü, tıpkı
git'in `.git/` klasörü gibi proje bilgisini günlerce taşıyor.

Bu mimariyi neden böyle kurduk? **Çünkü hayalî bilgi üretiminin
kaynağı, bilginin kendisini üretmek ile bilgiyi anlatmak işlemlerinin
karışmasıdır.** Biz bu ikisini katmanlara ayırdık. Veriyi sadece
deterministik kod okuyor, yapay zeka sadece sonucu anlatıyor —
hiçbir zaman tersi yok.

*[Konuş esnasında sağ ele ok kullanarak fragment'ler arasında geç.]*

---

## 🎬 Slayt 5 — The Two-Layer Brain (~55 saniye)

İki katmanı daha yakından açıklayayım.

Sol panelde **deterministic tool layer**. Kodu Python'da yazılmış,
pandas kullanarak CSV okuyor, Python'ın `ast` modülünü kullanarak
eğitim scriptini parse ediyor, NumPy ile Pearson ve Spearman
korelasyonlarını hesaplıyor. Çıktısı her zaman bir Python dictionary
— hiçbir zaman serbest metin değil. **Metin üretmediği için
hallucination yapması fiziksel olarak mümkün değil.**

Sağ panelde **Claude tabanlı narrator**. Bu katman, deterministik
katmanın ürettiği dictionary'yi alıyor ve doğal dilde açıklıyor.
Sadece kullanıcı `--llm` flag'ini geçtiğinde çalışıyor. Yani **API
anahtarı olmayan birisi bile** mlcompass'i kullanabilir — sadece
deterministik analiz görür.

Bir sonraki slaytta açıklayacağımız üç katmanlı kontrat, **yapay
zekanın yalnızca alt katmanın gördüğü şeyleri anlatabilmesini**
garanti ediyor.

---

## 🎬 Slayt 6 — The Anti-Hallucination Contract (~70 saniye)

Şimdi projenin **ana teknik katkısı** olan anti-hallucination
kontratını anlatıyorum. Üç katmandan oluşuyor — her biri tek
başına zayıf, üçü birlikte sıkı.

**Birinci katman: deterministik kanıt üretici.** `tools/leakage.py`
modülü, tahmin tablosunu okuyup yapılandırılmış bir kanıt
dictionary'si üretiyor. İçinde şüpheli metrik, aday leak sütunları,
korelasyonlar, mükemmel-eşleşme oranı var. Tüm bunlar saf Python ile
hesaplanıyor.

**İkinci katman: katı prompt'lu narrator.** Claude'a bu kanıtı
veriyoruz ve üç açık talimat ekliyoruz: **sadece kanıt
dictionary'sinde olan şeyleri söyle, emin değilsen 'belirlenemez'
de, kod yaması önerme — sadece manuel kontrol öner.**

**Üçüncü katman: runtime şema sınırı.** Claude'un cevabını verirken
kullandığı tool'un input şeması, kanıt dictionary'sindeki sütun
isimleriyle sınırlandırılmış bir enum içeriyor. **Eğer LLM dictionary
dışında bir sütun ismi yazarsa, Anthropic SDK bunu reddediyor ve
agent yeniden deniyor.**

Sloganımız şu: **Promptlar tavsiyedir, şemalar zorlamadır.** Yapay
zekayı sözle ikna etmek yerine, runtime'da fiziksel olarak engelliyoruz.

*[Üçüncü katmanı vurgularken bir an dur, izleyiciye düşünme süresi ver.]*

---

## 🎬 Slayt 7 — Worked Example: Ames House Prices (~55 saniye)

Somut bir örnek üzerinden gösteriyorum. Bir kullanıcı, ev fiyatı tahmin
etmek için eğitim verisine yanlışlıkla `log_price` sütununu ekliyor.
`log_price`, hedef değişken `saleprice`'ın logaritması — yani direkt
veri sızıntısı. Modelin R² değeri 1.000 çıkıyor — alarm zili.

**Birinci katman** çalışıyor: Pearson korelasyonu 0.94, Spearman
korelasyonu 1.00 hesaplıyor. Spearman'ı da hesaplamamızın sebebi şu —
logaritma gibi monoton dönüşümler Pearson'ı zayıflatıyor ama Spearman'ı
1.00'da bırakıyor. `log_price` aday sütun olarak işaretleniyor.

**İkinci katman** doğal dilde anlatıyor: *"log_price'ın Spearman'ı
1.00. Büyük ihtimalle dönüştürülmüş bir leak. Manuel olarak feature
pipeline'ınızı kontrol etmenizi öneririm."*

**Üçüncü katman** doğruluyor: `log_price` enum'da var, cevap
gönderiliyor. **Eğer LLM "revenue de leak olabilir" deseydi —
`revenue` veri setinde yok — SDK bu cevabı reddederdi ve yeniden
denerdi.** Bu Layer 3'ün gücü.

---

## 🎬 Slayt 8 — Results: Wilson 95% CI Ablation (~65 saniye)

Şimdi deney sonuçlarımız. **N=200 narrator cevabı**, Claude 3.5
Sonnet, sentetik bir regresyon veri seti üzerinde.

Birinci satır: **sadece Layer 1**, prompt yok, şema yok.
Hallucination oranı **%8.0**. Wilson 95% güven aralığı 4.93 ile
12.66 arasında. Yani sıfırdan istatistiksel olarak ayırt edilebiliyor.

İkinci satır: **Layer 1 + 2**, prompt eklendi. Hallucination oranı
**%1.0**'a düştü. Sekiz katlık iyileşme — ama hala sıfır değil.

Üçüncü satır: **Üç katman da aktif**, şema zorlaması ile. Gözlemlenen
oran **%0.0**. 200 örneklemde tek bir hallucination yok.

**Önemli dürüstlük noktası**: N=200'de Layer 2 ve Layer 3'ün güven
aralıkları üst üste biniyor. Yani istatistiksel olarak ayırt
edemiyoruz. Ama Layer 3'ün değeri ortalamada değil — **en kötü
durum garantisi sağlamasında**. Şema her halükarda hallucination'lı
cevabı reddediyor.

Çıkarım: **Prompt mühendisliği iyileştiriyor ama garantileyemiyor.
Sıfır garantisi için runtime şema zorlaması şart.**

---

## 🎬 Slayt 9 — Field Tests (~55 saniye)

Sadece sentetik veriyle yetinmedik. **Her release'i gerçek bir Kaggle
veri setinde test ettik.** Beş Kaggle field test, **11 gerçek bug**.

Test 1: Telco Churn — baseline, beklediğimiz gibi sıfır bug. Sistem
uçtan uca çalışıyor.

Test 2: Ames House Prices — **30 saniyede üç bug**. `SalePrice` hedef
isim listesinde yoktu, `YearBuilt` ID sanılıyordu, `PoolQC` %96 NaN
olduğu için crash veriyordu.

Test 3: Titanic — `Survived` hedef isim listesi eksik.

Test 4: Penguins — multiclass binary olarak sınıflandırılıyordu. Daha
zoru: **CLI üzerinden çalışırken proje hafızasına yazıyorduk ama MCP
üzerinden çalışırken yazmıyorduk** — surface'lar arasında driftlemişiz.

Test 5: Insurance Charges — `charges` regresyon hedefi düşük güvenle
algılanıyordu, finans hedef isimleri eksikti.

**Dersi şu**: Hayalimizden test senaryosu üretmek ile gerçek bir Kaggle
veri seti açmak farklı şeyler. Hiçbir bug bizim önceden öngörebildiğimiz
bir şey değildi. Her bug için yeni bir regresyon testi yazdık —
sistematik olarak öğrendik.

---

## 🎬 Slayt 10 — Live Demo (~90 saniye)

Şimdi terminale geçiyorum.

*[Terminal'i aç, ekrana yansıt.]*

İlk komut: `mlcompass init insurance-demo` — `.mlcompass/` klasörünü
oluşturuyor. Tıpkı `git init` gibi.

İkinci: `mlcompass advise data/insurance.csv` — Kaggle Insurance
Charges veri setine bakıyor. Hedef değişkeni `charges` olarak doğru
algılıyor, smoker sütununda dengesizlik olduğunu söylüyor.

Üçüncü: `mlcompass evaluate predictions.csv` — kasıtlı olarak leak
ekledim. **R² 1.000 görünce otomatik olarak leakage paneli açılıyor**
— bu projenin onur konuğu özelliği. Hangi sütunun şüpheli olduğunu,
korelasyon değerlerini, perfect-match oranını gösteriyor.

Dördüncü: `mlcompass status` — şimdiye kadar ne kararlar verdiğimizi
listeliyor. CLI ve MCP üzerinden yapılan tüm işlemler burada.

*[Demoyu 90 saniyede bitir. Çok uzatırsan toplam süre kayar.]*

*[Eğer terminal bağlantısı çökerse: "Önceden kaydedilmiş video var" deyip
mp4 dosyasını çal.]*

---

## 🎬 Slayt 11 — Challenges (~55 saniye)

Karşılaştığımız dört zorluk.

**Birincisi: Prompt mühendisliği yetmedi.** İlk versiyonda Claude'a
"hallucination yapma" derken bir hafta geçirdik. Oranı %8'den %1'e
indirdik ama bitiremedik. Çözüm: prompt katmanında değil, runtime
katmanında zorlamak.

**İkincisi: CLI ve MCP server arasında parity gap.** Aynı Python
fonksiyonlarını çağırıyorduk ama farklı davranıyorlardı çünkü CLI'da
"ledger'a yaz" adımı vardı, MCP'de yoktu. **Bir capability iki yüzey
üzerinden çıkıyorsa, parity bedava gelmiyor — açıkça testlenmesi
gerekiyor.**

**Üçüncüsü: Sentetik unit testler gerçek bug'ları bulamadı.** Beş
Kaggle veri seti açtık — 11 bug çıktı. Hiçbiri unit testlerimizde
önceden öngörülemezdi. Bu yüzden release ritüelimizi değiştirdik:
artık her release önce Kaggle'da deniyor.

**Dördüncüsü: Reviewer geri bildirimi.** Yazdığımız paper'a üç
peer-review tarzı yorum geldi — istatistiksel iddialarımız zayıftı,
baseline karşılaştırması yoktu, threat model gizliydi. **v0.8.1'i bu
geri bildirimleri kod düzeyinde adres etmek için** çıkardık. Wilson
CI'lar, reproducibility scriptleri, threat model dokümanı, limitations
dokümanı eklendi.

---

## 🎬 Slayt 12 — Limitations + Future Work (~50 saniye)

**Ölçmediğimiz şeyleri dürüstçe söylüyoruz.** Sol sütunda:

Sadece Claude 3.5 Sonnet ile test ettik. GPT-4, Llama-3, Gemini'de
nasıl davranır bilmiyoruz. Outlines veya OpenAI'ın strict mode'una
karşı direkt benchmark yapmadık. Harici kullanıcı çalışması yok —
sadece biz ikimiz ve bir kaç sınıf arkadaşımız test ettik.
Hallucination'ın diğer kategorileri — miscalibrated confidence,
prompt injection — projemizin kapsamı dışında.

Sağ sütunda **v0.9 ve v1.0 için planladıklarımız**:

OpenAI backend eklemek — aynı tabloyu GPT-4 + strict mode üzerinde
tekrar çalıştırmak. Llama-3-8B üzerinde Outlines ile karşılaştırma
yapmak. N'i 2000'e çıkararak güven aralıklarını sıkmak — bunu yapmak
için reproducibility scriptimiz hazır, `scripts/reproduce_
hallucination_ablation.py --mode live --n 2000`. Ve son olarak,
plug-in sistemi — üçüncü tarafların domain-specific analyzer ekleyebilmesi.

**Dürüstçe limitations vermek, papermızı saklamak değil
güçlendiriyor.** Akademik olgunluk sinyali bu.

---

## 🎬 Slayt 13 — Thank You + Q&A (~30 saniye)

Beni dinlediğiniz için teşekkür ederim.

Projenin tüm kaynak kodu **github.com/hakansabunis/mlcompass** üzerinde,
MIT lisansı altında açık. Kurmak için sadece `pip install mlcompass`
yazmanız yeterli. Paper'ımız repo'nun `paper/` klasöründe.

Sorularınızı almaya hazırım.

*[Q&A sırasında 2 dakika kalsın. Önemli muhtemel soruları hazırla:*

- *"Neden Claude kullandınız, GPT değil?" → "Anthropic'in tool-use
  API'sinin schema enforcement'ı diğerlerinden daha katı. Ama mekanizma
  generic — v0.9'da OpenAI backend ekliyoruz."*

- *"N=200 azlık değil mi?" → "Evet, paper'da da dürüstçe söyledik. Reproducibility
  scriptimiz N'i 2000'e çıkarmaya hazır. ~$48 maliyet, internal budget'ımız
  yetmedi."*

- *"Constrained generation literature ile farkı ne?" → "Outlines ve LMQL şemayı
  statik olarak deklare ediyor. Bizim katkı: şemayı runtime'da deterministik
  kanıttan üretiyoruz — enum domain veriye göre küçülüyor."*

- *"En zor anınız neydi?" → "Field Test #4, CLI/MCP parity gap. İki saatlik
  debug, sonunda öğrendiğimiz ders en genelleştirilebilir bulgumuz oldu."*

*]*

---

## 📝 Genel sunum tavsiyeleri

1. **Hız**: Her slayt 45-60 saniye. Toplam 12-13 dk. Q&A için 2 dk bırak.

2. **Vurgular**: "Anti-hallucination kontratı", "Promptlar tavsiyedir,
   şemalar zorlamadır", "11 gerçek bug", "%8.0 → %1.0 → %0.0" — bu
   sayıları yavaş ve net söyle.

3. **Body language**: Slayt geçişlerinde 2-3 saniye sessiz dur,
   dinleyici slaytı okusun.

4. **Demo backup**: Internet/terminal sorun çıkarırsa diye **MP4 ekran
   kaydı al** önceden. USB stick'te bulunsun.

5. **Türkçe-İngilizce karışım**: Teknik terim İngilizce kalsın
   (`hallucination`, `schema`, `Wilson CI`) — Türkçeye çevirmeye
   çalışırsan akıcılığı bozar.

6. **Q&A**: Bilmiyorsan **"Bu güzel bir soru, paper'da Section V.D'de
   tartıştığımız bir limitation. Şu an net bir cevabımız yok ama..."**
   diye dürüstçe söyle. Hoca dürüstlüğü değerlendirir.

İyi sunumlar! 🚀
