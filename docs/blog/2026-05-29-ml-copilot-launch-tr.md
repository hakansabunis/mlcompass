# ml-copilot v0.1: ML eğitiminin yanındaki AI mühendisi

*2026-05-29 — Hakan Sabunis*

Capstone projemde aynı hataları **defalarca** yaptım. Yanlış metric.
Eksik seed. Validation split çok küçük. Loss fonksiyonunda log(0)'dan
NaN. Her seferinde "bunu bir kıdemli yanımda olsa 30 saniyede yakalardı"
diyordum.

W&B, TensorBoard, MLflow var — ama hiçbiri **öneri vermiyor**, sadece
ne olduğunu kaydediyor. AutoML araçları diğer uçta: tamamen otonom
çalışıp size bir model fırlatıyorlar, neden o seçimleri yaptıklarını
öğrenemiyorsunuz.

Aradaki boşluğa **ml-copilot**'u yazdım. Bugün **v0.1** yayında.

```bash
pip install ml-copilot
ml-copilot init my-project
ml-copilot advise data.csv
```

## Ne yapıyor?

Veri CSV'nizi verirsiniz; ml-copilot şunları söyler:

1. **Hangi 3 modeli denemelisiniz** (gerçekçi metric aralıklarıyla)
2. **Hangi feature engineering'i yapmalısınız** (kolona göre özelleşmiş)
3. **Hangi pitfall'lardan kaçınmalısınız** (sınıf dengesizliği, eksik
   veri, yüksek kardinalite, vs.)

Örnek çıktı (`examples/customer_churn.csv` üzerinde):

```
📊 Veri analizi
  500 satır × 8 kolon
  Hedef:  churn (yüksek güven)
  Görev:  binary classification (0=%98, 1=%2)

⚠ Uyarılar
  • Sınıf dengesizliği (%1.6 azınlık). Accuracy KULLANMA — AUC/F1
    kullan. class_weight='balanced' veya focal loss düşün.

✨ Önerilen modeller
  • XGBoost              AUC 0.80–0.84
  • Logistic Regression  AUC 0.72–0.76  (yorumlanabilir baseline)
  • LightGBM             AUC 0.80–0.85  (XGB'dan hızlı bu ölçekte)

🔧 Feature engineering
  • signup_date → days_since_signup, dayofweek
  • country (30 kategori) → target encoding veya top-N
```

## Mimari

Önemli karar: **veri analizi tamamen deterministik (pandas)**. LLM'e
"bu kolon ne tipte?" diye sormuyoruz. Şema, eksik veri, outlier, sınıf
dengesi — hepsi Python. LLM sadece **bu yapılı analize bakıp önerileri
üretiyor**.

Bu karar üç şeyi getiriyor:

- **Hız** — bir LLM round-trip yerine yerel pandas
- **Maliyet** — sadece öneri kısmı LLM, gerisi bedava
- **Öngörülebilirlik** — analiz sonuçları her seferinde aynı

Altyapı olarak [agentlite](https://github.com/hakansabunis/agentlite)
kullanıyorum — Claude için kendi yazdığım ~2.000 satırlık küçük agent
kütüphanesi. Şu üç özelliği veriyor:

- **Prompt caching** otomatik açık (sistem prompt'u tekrar tekrar
  yollanmıyor)
- **Permission sistemi** birinci sınıf (kodunuza dokunan her tool önce
  izin sorar)
- **Sub-agent factory** — alt-agent'lar bağımsız context'lerde çalışıyor

## v0.1 yapısı

```
ml-copilot init <name>          Yeni proje başlat (.mlcopilot/)
ml-copilot advise <data> ...    Veri analizi + model + FE önerisi
```

Proje boyunca süren bir `.mlcopilot/` klasörü tutulur — git'in `.git/`'i
gibi. İçinde hangi veri seçildi, hangi model önerildi, kullanıcı ne
karar verdi hepsi saklı. Sonraki komutlar (`audit`, `watch`, `evaluate`,
`deploy`) bu bağlamı okuyup üzerine inşa edecek.

## Sonraki adımlar

v0.2'de eğitim sırasında **canlı izleme** geliyor:

- `ml-copilot audit <script>` — train.py'nin statik analizi (seed
  eksik mi, val split makul mu, loss fonksiyonu stabil mi)
- `ml-copilot watch <script>` — eğitim sürerken plateau / overfit /
  NaN tespit et, **izinli** olarak müdahale önerisi sun
- `ml-copilot compare run-a run-b` — iki eğitim arasında AI yorumu

v0.3'te `evaluate`, v0.4'te `deploy` kontrolleri ekleniyor.

Yol haritası tamamen [CHANGELOG.md](https://github.com/hakansabunis/ml-copilot/blob/main/CHANGELOG.md)'de,
tasarım kararları [ARCHITECTURE.md](https://github.com/hakansabunis/ml-copilot/blob/main/ARCHITECTURE.md)'de
yazılı.

## Dene

```bash
pip install ml-copilot
ml-copilot init demo
ml-copilot advise <senin-csv'n>
```

`ANTHROPIC_API_KEY` yoksa `--no-llm` flag'iyle sadece deterministik
analiz kısmını ücretsiz çalıştırabilirsin.

Hazır deneyecek veri yoksa `examples/` altında 3 sentetik dataset var:
Titanic-benzeri binary classification, ev fiyatı regression, telekom
churn (ağır dengesizlik).

## Geri bildirim

ml-copilot alfa aşamasında. En çok iterate edeceğim şey advisor'ın
prompt'u — denersen ve öneri tutmazsa, lütfen issue aç ve hangi veriyi
verdiğinde neyi beklediğini yaz.

GitHub: <https://github.com/hakansabunis/ml-copilot>

> *Kapsam içerisinde dipnot: ml-copilot'u TÜBİTAK 1001 başvurusu ya da
> şirkete ücretli SaaS olarak sunmak gibi bir niyetim şu an yok. Açık
> kaynak, MIT lisanslı, kendi ihtiyacımdan doğmuş bir alet. Sen de
> kullanışlı bulursan ne mutlu.*
