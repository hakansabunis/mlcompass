# "Çalmasınlar" — neyin gerçekten koruduğu

Kısa cevap: **asıl risk kodun çalınması değil, makalenin önünün kesilmesi
(scooping).** Ve bu ikisi tamamen farklı önlemler istiyor. Aşağıda hangisinin
işe yaradığını, hangisinin tiyatro olduğunu ayırdım.

---

## 1. Lisansı değiştirmek istediğin şeyi yapmaz

Bunu en başa koyuyorum çünkü ilk akla gelen hamle bu, ve büyük ölçüde
kapanmış bir kapıyı kapatmak oluyor.

**PyPI'da 29 Mayıs'tan beri 14 sürüm MIT altında yayında.** MIT geri
alınamaz: o sürümleri indirmiş herkes, kodu alma, değiştirme, kapatma ve
satma hakkını **kalıcı olarak** elinde tutuyor. Bugün AGPL'e geçsen bile bu
sadece *bundan sonraki* sürümleri bağlar, ve isteyen 0.9.0'dan devam eder.

Ayrıca AGPL'in üç somut maliyeti var:

- Makale iki yerde "open-source (MIT)" diyor. Basılı metni değiştirmek gerekir.
- Birçok kurum AGPL'li kodu kullanmayı yasaklıyor — benimsenme düşer.
- Artifact evaluation süreçleri izin verici lisans bekler; replication
  package'ın AGPL olması hakemin işini zorlaştırır.

Ve karşılığında ne alırsın? Sadece "kodu alıp kapalı bir servis olarak
satan" senaryoya karşı koruma. Senin gerçek riskin bu değil.

**Önerim: MIT kalsın.** Katılmıyorsan söyle, AGPL'e veya çift lisansa
geçiririm — ama bu üç yazarı ve basılı metni ilgilendirdiği için senin
kararın, benim tek başıma yapacağım bir şey değil.

---

## 2. Asıl risk: makale yayımlanmadan deponun herkese açık olması

Depo public. İçinde yöntem, tüm veri, ve **makalenin LaTeX kaynağı** var.
Yani bugün biri deponu bulup aynı fikri daha hızlı yazabilir. Bu, akademide
gerçek ve sık görülen bir şey, ve lisansın buna hiçbir etkisi yok — fikirler
telif kapsamında değil.

### Çözüm: arXiv preprint. Bugün.

Standart, ücretsiz ve etkili koruma budur. arXiv'e yüklediğin an **tarihli,
herkese açık, değiştirilemez bir öncelik kaydı** oluşur. Sonradan biri aynı
iddiayı yaparsa, senin tarihin onunkinden önce olur ve bu tartışma biter.

- **IEEE preprint'e izin veriyor.** TSE'ye gönderirken arXiv'de preprint
  bulunması sorun değil; IEEE'nin yazar politikası bunu açıkça kabul ediyor.
  Kabul sonrası preprint'e DOI ve "accepted for publication" notu eklenir.
- Kategori: `cs.SE` (birincil), `cs.LG` ve `cs.CL` çapraz liste.
- Yükleyeceğin dosya: `paper/tse_latex/main.pdf` (veya kaynak + .bib).

Bunu yapana kadar depo public olduğu için savunmasızsın. **Bugün yapılacak
tek iş buysa, bu olmalı.**

---

## 3. Zenodo DOI — kodu ve veriyi tarihe bağlar

Makale zaten söz veriyor ("will be archived on Zenodo with a DOI"), yani
zaten yapılacak. İki işi birden görür: kalıcı arşiv + alıntılanabilir kayıt.

Adımlar (10 dakika):

1. zenodo.org → GitHub ile giriş
2. Settings → GitHub → `hakansabunis/mlcompass` anahtarını aç
3. GitHub'da bir **release** oluştur (ör. `v0.9.0`)
4. Zenodo otomatik arşivler ve DOI verir
5. DOI'yi `CITATION.cff`'e ve makalenin Data Availability bölümüne ekle

Release oluşturmadan Zenodo tetiklenmez — anahtarı açmak tek başına yetmiyor.

---

## 4. Atıf: zaten hazır, iki yeri düzelttim

`CITATION.cff` dosyası mevcut ve iyi durumda: üç yazar, kurum, ve makale
`preferred-citation` olarak kayıtlı. GitHub bunu okuyup depo sayfasında
"Cite this repository" düğmesi gösteriyor, yani kullanan birinin atıf
vermemesi için bahanesi kalmıyor.

Bugün düzelttiğim iki şey:

- `date-released` Haziran'da kalmıştı, Eylül yapıldı.
- Not, "iki manuscript aynı sayıları taşıyor" diyordu. **Artık taşımıyorlar** —
  EMSE dondu, TSE güncel. Anlaşmazlık hâlinde TSE'nin doğru olduğu yazıldı.

---

## Özet — ne yapılacak, hangi sırayla

| | İş | Süre | Kim |
|---|---|---|---|
| 1 | **arXiv preprint** — öncelik kaydı | 1 saat | Sen |
| 2 | GitHub release + Zenodo DOI | 10 dk | Sen |
| 3 | DOI'yi CITATION.cff ve makaleye işle | 5 dk | Ben |
| 4 | Lisans kararı (önerim: MIT kalsın) | — | Sen + yazarlar |

Birinci madde diğer üçünden daha önemli. Lisans tartışması olmadan da
yapılabilir ve bugün yapılabilir.

---

## Yapmadığım ve yapmaman gerektiğini düşündüğüm şey

**Depoyu private'a çekmek.** Makale "open-source" diyor ve bağlantı veriyor;
hakem tıklayıp 404 görürse bu, çalınma riskinden daha kesin bir zarar. Ayrıca
replication package'ın erişilebilir olması makalenin en güçlü yanlarından biri
— onu kapatmak, uğruna çalıştığımız şeyi geri almak olur.

Doğru hamle depoyu saklamak değil, **tarihi damgalamak.**
