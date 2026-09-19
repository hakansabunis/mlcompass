# Durum notu — 19 Eylül 2026

**Gönderilen dosya:** `paper/tse_latex/main.pdf` — IEEE iki sütun, 18 sayfa
(gövde 16'da bitiyor, kaynakça 17–18).

Springer uzun formu (`paper/emse_latex/`, 54 sayfa) bırakıldı. Hedef dergi
IEEE TSE, ve bundan sonra tek bakımı yapılan sürüm bu.

---

## Stanford (paperreview.ai) sonucu

**"I recommend acceptance."**

Önemli bir uyarı ile: o değerlendirme, aşağıdaki düzeltmelerden *önceki*
sürüme yazıldı. Övdüğü bulgulardan birini biz o tarihten sonra **geri çektik**.
Yani olumlu karar, kendi düzelttiğimiz hatayı içeren metne ait.

---

## Değerlendirmeden bu yana kapatılanlar

Stanford'un saydığı sekiz zayıflıktan dördü kapandı, biri kısmen:

| Zayıflık | Durum |
|---|---|
| Tek kanıt biçimi ölçülmüş; sınıf iddiası argümana dayanıyor | **Kapandı** — ikinci görev ölçüldü |
| Ölçek: binlerce kolonda Tier A ne olur, ölçülmemiş | **Kapandı** — 10'dan 1000 kolona ölçüldü |
| Kolon başına birden fazla istatistik ele alınmamış | **Kapandı** — contract genel katmana çıkarıldı |
| Eksik ilgili çalışma (ATLAS-RTC, LedgerMind, vb.) | **Kapandı** — beşi eklendi, dokuzu da arXiv'den tek tek doğrulandı |
| Düzeltmelerin yoğunluğu okuyucuyu yoruyor | **Kapandı** — "ne kurduk" paragrafı en başa alındı |
| Serbest metne kaçış ölçülmemiş | Kısmen — ölçülemediği ve *nedeninin biz olduğu* raporlandı |
| Karşılaştırma fiilen tek sağlayıcı | Açık |
| τ=0.005 toleransı gerekçesiz | Açık |

---

## İkinci görev: ne buldu

Aynı contract, farklı kanıt biçimi — bu kez veri profili (24 kolon, 13
istatistik, 196 ölçülmüş nicelik; leakage'da 10 / 1 / 10 idi).

Kısıtsız kol **%8.5** oranında hata veriyor (leakage'da %42). Oran zayıf.
**Ama hatanın türü aynı çıktı**, ve asıl bulgu bu:

- 3.277 yapılandırılmış iddianın 12'si yanlış sayı içeriyor.
- **11'i uydurma değil** — başka bir kolonun *gerçek* ölçümü, yanlış kolona
  yapıştırılmış (`cat_feature_07` → `num_feature_07`, aynı sayısal sonek).

Leakage gerçek bir **adı** yanlış alana koyuyordu; profil gerçek bir **ölçümü**
yanlış varlığa koyuyor. Aynı başarısızlık, kanıtın şekline göre yer
değiştiriyor. Sınıf iddiası için ikinci bir %42'den daha güçlü bir sonuç:
bir mühendisin narration'a bakarak yakalayamayacağı tek hata türü bu, çünkü
metindeki her rakam aracın gerçekten ürettiği bir rakam.

**Beklenmedik ikinci bulgu.** Sadece şema kısıtı olan kol (Tier A, doğrulama
yok) 17'den 12'ye iniyor — yani **neredeyse hiçbir şey almıyor**. Bir enum
kolon adını kısıtlayabilir, bir *sayıyı* kısıtlayamaz. Leakage'da bu kol sıfıra
iniyordu ve iki katmanı birbirinden ayırmak mümkün değildi; bu görev ayırıyor.

İki zorlama kolu (`contract`, `stress`) şu anda koşuyor; sonuçları yarın.

---

## Kendi enstrümanlarımızda bulunan hatalar

Toplam on. Son üçü bu hafta, hepsi biz raporlamadan önce yakalandı:

1. **Kontrol kolu puanlanmadan önce onarılıyordu.** 200 yanıtın 60'ının
   ihlalleri siliniyor, kol 0/200 okuyordu. İşaret, Tier B'si olmayan bir kolda
   "60 Tier B yakalaması" yazmasıydı.
2. **Sözlük kanıtın kelimelerini değiştiriyordu.** Profiler `iqr_count` diyor,
   bizim tanım `iqr_outliers` diyordu; model kanıtın kendi kelimesini yazınca
   "uydurma" sayılıyordu. Ölçülen hatanın üçte biri bizim sözlüğümüzdü.
3. **Ölçek tablosunu elle yazmışım**, düzeltmeden önceki koşudan; yedi satır da
   kaymış. Artık komut üretiyor, birebir yapıştırılıyor.

Ayrıca contract kolunun prompt'unda bir tasarım hatası 101/200'de yakalanıp
durduruldu — hiçbir sayı raporlanmadığı için ondalık listeye girmiyor.

On hatanın **ikisi**, savunduğumuz hipotezin *aleyhine* sonuç verdi. İkisi de
tam bunu aramak için kurulmuş prosedürlerden çıktı.

---

## Açıkça geri çektiğimiz iddia

Önceki sürüm, aşağı akıştaki kusur düşüşünü (31 → 8) aracın bulgularına
atfediyordu. Sadece altı puanlanan kuralın adını veren, script hakkında hiçbir
şey söylemeyen bir kol **1'e** indi. Atıf geri çekildi.

Ayakta kalan: pratikte alınan sonuç (çalışıyor **ve** kusursuz) — aracın
bulguları %64, jenerik liste %44, aralıklar örtüşüyor. Zayıf ayrım, ve zayıf
olduğunu yazıyoruz.

---

## Kalan iş

- İki zorlama kolu bitince ikinci görev bölümünün yazılması (yarın)
- Kör insan değerlendirmesi: aparat hazır, yazar olmayan iki kişi gerekiyor
- Zenodo DOI alınması. Depo public ve MIT (`github.com/hakansabunis/mlcompass`,
  PyPI'da 0.9.0), ama makale kalıcı bir arşiv bağlantısı vaat ediyor ve henüz
  yok. Gönderim öncesi kapatılmalı.
