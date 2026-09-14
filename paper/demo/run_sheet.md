# 🎬 mlcompass Sınıf Demo — Çalıştırma Kılavuzu

Tek sayfalık talimat. Cep telefonunda / tablette aç, sahnede gör.
Sunum slayt 10'a (Live Demo) geldiğinde aşağıdaki adımları sırayla yap.

---

## 🚨 Sınıftan 15 dakika önce — warm-up

PowerShell aç, şu üç komutu çalıştır (kontrol amaçlı):

```powershell
cd $env:USERPROFILE\Desktop\mlcompass-demo
.\.venv\Scripts\Activate.ps1
mlcompass --version
claude mcp list
```

Beklenen:
- `mlcompass, version 0.8.1`
- `mlcompass` MCP server listede

İkisi de yeşilse demo'ya hazırsın. Yoksa şu Yedek Plan A'ya bak.

---

## 🎤 Sınyaf slayt 10'a geldiğinde

### 1. Sahneye terminale geç

- Alt+Tab ile PowerShell'e geç (önceden açtın, venv aktif)
- F11 ile **full-screen**
- Font size 22-24pt olduğundan emin ol (sınıf görsün)

### 2. Claude Code'u aç

```powershell
claude
```

Açıldığında **promptu temizle** (Ctrl+L). Boş bir Claude oturumu olsun.

### 3. Slash komutu #1 — `init` (15 saniye)

Yaz:

```
/mlc-init insurance-demo
```

**Söyle:**
> *"İlk komut. Slash menüsünde `mlc-` ön ekli 11 komut var, hepsi mlcompass'in araçlarını çağırıyor. `mlc-init` Claude Code'a 'mlcompass_init MCP tool'unu çağır' diyor. `.mlcompass/` klasörü oluşuyor — proje hafıza klasörümüz."*

### 4. Slash komutu #2 — `advise` (45 saniye)

Yaz:

```
/mlc-advise demo-data/insurance.csv
```

**Söyle:**
> *"İkinci komut. Veri setine bakıyor. Saf Python pandas çalışıyor — yapay zeka yok, hallucination olamaz. Hedef sütununu `charges` olarak doğru algıladı, regresyon problemi olduğunu çıkardı, smoker sütununda dengesizlik olduğunu söylüyor. Tüm bunlar deterministik."*

**Beklenen çıktıdan göster:**
- Shape: 1338 rows × 7 columns
- Target: `charges` (regression, high confidence)
- Warning: smoker imbalance
- Recommended models: XGBoost, LightGBM, Linear

### 5. Slash komutu #3 — **YILDIZ MOMENT** (75 saniye)

Yaz:

```
/mlc-evaluate demo-data/predictions_with_leak.csv
```

**Söyle (büyük vurgu — bu en önemli kısım):**
> *"Üçüncü komut, demoumuzun yıldız anı. Predictions tablosuna bakıyor."*

*[Çıktıyı bekle — birkaç saniye sonra leakage paneli açılacak]*

> *"İşte! R² 1.000 görür görmez **otomatik olarak leakage paneli** açıldı. `log_charges_leak` sütunu yakalandı, Spearman korelasyonu 1.00, perfect-match oranı 98.7%. Claude burada Claude session'ını kullanarak yorum yapıyor — **ama bizim runtime schema boundary'miz hala işliyor.** Claude sadece tool'un evidence dictionary'sinde gördüğü sütun isimlerini söyleyebilir. `revenue` uydurmak istese, JSON-schema enum reddederdi."*

> *"Sloganımız: **promptlar tavsiyedir, şemalar zorlamadır.**"*

### 6. Slash komutu #4 — `status` (30 saniye)

Yaz:

```
/mlc-status
```

**Söyle:**
> *"Son komut. Proje hafızası. Init, advise, evaluate kararları listede. Yarın geri gelsem hangi dataset üzerinde çalıştığımı hatırlıyor. Tıpkı git'in commit history'si gibi."*

### 7. Slayda dön

Alt+Tab ile sunuma geç, slayt 11 (Challenges) açılıyor. Demo bitti.

---

## 🚨 Yedek planlar

### Yedek Plan A — `mlcompass` veya `claude` çalışmıyor

Önce **venv aktif mi** kontrol et:

```powershell
cd $env:USERPROFILE\Desktop\mlcompass-demo
.\.venv\Scripts\Activate.ps1
mlcompass --version
```

Hala çalışmıyorsa CLI fallback'e geç (Yedek Plan B).

### Yedek Plan B — Doğrudan CLI demo (Claude Code yok)

`claude` çalışmıyorsa **PowerShell terminalinde direkt mlcompass komutları** çalıştır. Aynı sonuçları gösterir, Claude Code yorumu eksik olur:

```powershell
cd $env:USERPROFILE\Desktop\mlcompass-demo
mlcompass init insurance-demo
mlcompass advise demo-data/insurance.csv
mlcompass evaluate demo-data/predictions_with_leak.csv
mlcompass status
```

Söyle: *"Demoyu doğrudan CLI üzerinden gösteriyorum, Claude Code'la
aynı tool'lar zaten arka planda çalışıyor."*

### Yedek Plan C — Bilgisayar tamamen çakıldı

Önceden kayıt ettiğin **MP4 yedeği** USB stick'ten oynat:

> "Şu an teknik bir sorun var, demoyu önceden kaydettim, onu göstereyim."

Sınıf sorun yapmaz — gerçek dünyada demo'lar bazen patlar.

### Yedek Plan D — Yusuf'un bilgisayarına geç

Aynı kurulum Yusuf'un bilgisayarında da hazır olsun (önceden yap). Bilgisayar çakılırsa onun bilgisayarından devam et.

---

## 📋 Cep listesi (yazdır, yanına al)

```
[ ] PowerShell terminal açık, venv aktif (yeşil prompt)
[ ] mlcompass --version → 0.8.1
[ ] claude mcp list → mlcompass var
[ ] demo-data/insurance.csv var
[ ] demo-data/predictions_with_leak.csv var
[ ] Yedek MP4 USB stick'te
[ ] Telefon şarjda, hotspot hazır
[ ] Yusuf'un bilgisayarı aynı kurulumla hazır
[ ] Sunumun slayt 10'unda terminale geçeceğini biliyorsun
```

---

## 💡 Profesyonel hatırlatmalar

1. **Yavaş yaz** komutu — sınıf okusun, autocomplete ile düzelt acele etme
2. **Çıktıyı kendine de oku** — sınıfla senkron olsun
3. **R² 1.000'i parmakla işaret et** — leakage paneli açılınca dikkat ver
4. **"Schema reddederdi" cümlesini iki kez söyle** — bu paper'ın özü
5. Sonunda **"sorularınız var mı?"** demeden slayt 11'e geç — Q&A son slayt'ta

---

## 🎯 Beklenen çıktı örnekleri (sınıfı yönlendirmek için)

### `/mlc-advise` çıktısı (özet):
```
📊 Dataset analysis
   Path:    demo-data/insurance.csv
   Shape:   1338 rows × 7 columns
   Target:  charges (regression, high confidence)
   Task:    regression

⚠ Warnings
  • Smoker imbalance: 79.5% non-smoker, 20.5% smoker
  • Age range 18-64 looks like a valid feature

✨ Recommended models
  • XGBoost Regressor    R² 0.85-0.90
  • LightGBM Regressor   R² 0.85-0.90
  • Linear Regression    R² 0.75-0.80 (baseline)
```

### `/mlc-evaluate` çıktısı (özet):
```
📈 Evaluation metrics
   R²:    1.0000  ⚠ SUSPICIOUSLY PERFECT
   MAE:   0.001
   RMSE:  0.003

┌──── 🔬 Leakage investigation ────┐
│ Candidate leak: log_charges_leak │
│ Spearman ρ:    1.0000             │
│ Pearson ρ:     0.9412             │
│ max(|ρ_P|,|ρ_S|): 1.0000 ≥ 0.97  │
│ Perfect-match rate: 98.7%        │
└───────────────────────────────────┘

Narrator (under contract):
"log_charges_leak has Spearman 1.00 with the target —
almost certainly a transformed leak. Recommend manually
checking the feature pipeline."
```

Bu çıktıları görünce sınıf "vaay" der. İşaret et, vurgu yap.

---

Bol şans! 🚀
