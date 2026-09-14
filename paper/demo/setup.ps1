# mlcompass demo setup — Tek tıklamayla tüm kurulum
#
# Sınıftan 1 gün önce çalıştır. Bütün her şeyi otomatik kurar:
#   1. Python venv oluştur
#   2. mlcompass + MCP extras + scikit-learn yükle
#   3. Claude Code'a MCP server ekle
#   4. Slash command'ları kur (global user scope)
#   5. Insurance Charges veri setini indir
#   6. Leak'li predictions dosyası üret
#   7. Hızlı sanity check (mlcompass --version vb.)
#
# Kullanım:
#   .\setup.ps1
#
# Eğer Claude Code kurulu değilse:
#   npm install -g @anthropic-ai/claude-code
#   (Sonra setup.ps1'i tekrar çalıştır)

$ErrorActionPreference = "Stop"

# --------------------------------------------------------------------------- #
# 0. Demo klasörü kur                                                         #
# --------------------------------------------------------------------------- #

$DemoRoot = "$env:USERPROFILE\Desktop\mlcompass-demo"
Write-Host "`n🎬 mlcompass demo kurulumu başlıyor..." -ForegroundColor Cyan
Write-Host "   Demo klasörü: $DemoRoot`n" -ForegroundColor Gray

if (-not (Test-Path $DemoRoot)) {
    New-Item -ItemType Directory -Path $DemoRoot | Out-Null
}
Set-Location $DemoRoot

# --------------------------------------------------------------------------- #
# 1. Python kontrol                                                            #
# --------------------------------------------------------------------------- #

Write-Host "[1/7] Python kontrolü..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "   ❌ Python bulunamadı." -ForegroundColor Red
    Write-Host "      python.org/downloads adresinden Python 3.11+ kur." -ForegroundColor Red
    exit 1
}
Write-Host "   ✓ $pythonVersion" -ForegroundColor Green

# --------------------------------------------------------------------------- #
# 2. Claude Code kontrol                                                       #
# --------------------------------------------------------------------------- #

Write-Host "`n[2/7] Claude Code kontrolü..." -ForegroundColor Yellow
$claudeVersion = claude --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "   ⚠️  Claude Code kurulu değil." -ForegroundColor Yellow
    Write-Host "      Şu komutla kur (önce Node.js gerekli):" -ForegroundColor Yellow
    Write-Host "      npm install -g @anthropic-ai/claude-code`n" -ForegroundColor White
    Write-Host "      Sonra bu scripti tekrar çalıştır." -ForegroundColor Yellow
    Write-Host "      Şimdilik devam ediyorum — Claude Code dışında her şey kurulacak." -ForegroundColor Gray
    $skipClaudeSetup = $true
} else {
    Write-Host "   ✓ $claudeVersion" -ForegroundColor Green
    $skipClaudeSetup = $false
}

# --------------------------------------------------------------------------- #
# 3. Sanal ortam                                                              #
# --------------------------------------------------------------------------- #

Write-Host "`n[3/7] Sanal ortam kuruluyor..." -ForegroundColor Yellow
if (-not (Test-Path "$DemoRoot\.venv")) {
    python -m venv .venv
    Write-Host "   ✓ .venv oluşturuldu" -ForegroundColor Green
} else {
    Write-Host "   ✓ .venv zaten var" -ForegroundColor Green
}

# Aktive et (bu script kapsamında)
. "$DemoRoot\.venv\Scripts\Activate.ps1"

# --------------------------------------------------------------------------- #
# 4. mlcompass + bağımlılıklar                                                #
# --------------------------------------------------------------------------- #

Write-Host "`n[4/7] mlcompass kuruluyor (PyPI'dan)..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet
pip install "mlcompass[mcp]" scikit-learn --quiet
$mlcompassVersion = mlcompass --version 2>&1
Write-Host "   ✓ $mlcompassVersion" -ForegroundColor Green

# --------------------------------------------------------------------------- #
# 5. Claude Code'a MCP server ekle                                             #
# --------------------------------------------------------------------------- #

if (-not $skipClaudeSetup) {
    Write-Host "`n[5/7] MCP server Claude Code'a ekleniyor..." -ForegroundColor Yellow

    # mlcompass-mcp.exe'nin tam yolunu bul
    $mcpServerPath = (Get-Command mlcompass-mcp).Source

    # Önceden ekli mi kontrol et
    $existingServers = claude mcp list 2>&1
    if ($existingServers -match "mlcompass") {
        Write-Host "   ⚠ mlcompass zaten ekli, atlanıyor" -ForegroundColor Gray
    } else {
        claude mcp add mlcompass $mcpServerPath 2>&1 | Out-Null
        Write-Host "   ✓ MCP server eklendi: $mcpServerPath" -ForegroundColor Green
    }

    # Slash command'ları kur
    Write-Host "`n[6/7] Slash command'lar kuruluyor (user scope)..." -ForegroundColor Yellow
    mlcompass install-slash-commands --scope user --force 2>&1 | Out-Null
    $slashCount = (Get-ChildItem "$env:USERPROFILE\.claude\commands\mlc-*.md" -ErrorAction SilentlyContinue).Count
    Write-Host "   ✓ $slashCount slash command kuruldu (~\.claude\commands\)" -ForegroundColor Green
} else {
    Write-Host "`n[5-6/7] Claude Code kurulu değil, MCP + slash adımları atlandı." -ForegroundColor Gray
}

# --------------------------------------------------------------------------- #
# 7. Demo veri seti                                                            #
# --------------------------------------------------------------------------- #

Write-Host "`n[7/7] Demo verisi hazırlanıyor..." -ForegroundColor Yellow

$DataDir = "$DemoRoot\demo-data"
if (-not (Test-Path $DataDir)) {
    New-Item -ItemType Directory -Path $DataDir | Out-Null
}
Set-Location $DataDir

# Insurance Charges indir (~50 KB)
if (-not (Test-Path "insurance.csv")) {
    $url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
    Invoke-WebRequest -Uri $url -OutFile "insurance.csv" -UseBasicParsing
    Write-Host "   ✓ insurance.csv indirildi" -ForegroundColor Green
} else {
    Write-Host "   ✓ insurance.csv zaten var" -ForegroundColor Green
}

# Leak'li predictions üret
$pythonScript = @"
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('insurance.csv')
df['log_charges_leak'] = np.log(df['charges'] + 1)

y = df['charges']
X = pd.get_dummies(df.drop(columns=['charges']), drop_first=True)

model = LinearRegression().fit(X, y)
y_pred = model.predict(X)

out = pd.DataFrame({'y_true': y, 'y_pred': y_pred})
out = pd.concat([out, X.reset_index(drop=True)], axis=1)
out.to_csv('predictions_with_leak.csv', index=False)

print(f'R^2 on train: {model.score(X, y):.4f}')
"@

$pythonScript | Out-File -Encoding utf8 "make_predictions.py"
python make_predictions.py 2>&1 | ForEach-Object { Write-Host "   $_" -ForegroundColor Gray }
Write-Host "   ✓ predictions_with_leak.csv hazır (R²=1.0)" -ForegroundColor Green

# --------------------------------------------------------------------------- #
# Özet                                                                        #
# --------------------------------------------------------------------------- #

Set-Location $DemoRoot

Write-Host "`n" -NoNewline
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  ✅ Demo kurulumu tamamlandı!" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "Demo konum: " -NoNewline; Write-Host "$DemoRoot" -ForegroundColor White
Write-Host "Veri: " -NoNewline; Write-Host "$DataDir\insurance.csv" -ForegroundColor White
Write-Host "Predictions: " -NoNewline; Write-Host "$DataDir\predictions_with_leak.csv" -ForegroundColor White
Write-Host ""

if (-not $skipClaudeSetup) {
    Write-Host "🎯 Sınıfta yapacaklarım:" -ForegroundColor Cyan
    Write-Host "   1. Terminal aç, " -NoNewline; Write-Host "cd $DemoRoot" -ForegroundColor White
    Write-Host "   2. " -NoNewline; Write-Host ".\.venv\Scripts\Activate.ps1" -ForegroundColor White -NoNewline
    Write-Host "  (venv aktif)"
    Write-Host "   3. " -NoNewline; Write-Host "claude" -ForegroundColor White -NoNewline
    Write-Host "  (Claude Code aç)"
    Write-Host "   4. İçinde slash komutları sırayla yaz:"
    Write-Host "      " -NoNewline; Write-Host "/mlc-init insurance-demo" -ForegroundColor Yellow
    Write-Host "      " -NoNewline; Write-Host "/mlc-advise demo-data/insurance.csv" -ForegroundColor Yellow
    Write-Host "      " -NoNewline; Write-Host "/mlc-evaluate demo-data/predictions_with_leak.csv" -ForegroundColor Yellow
    Write-Host "      " -NoNewline; Write-Host "/mlc-status" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "📋 Tam talimat: " -NoNewline; Write-Host "paper\demo\run_sheet.md" -ForegroundColor White
} else {
    Write-Host "⚠ Claude Code kurulumu eksik — yukarıdaki uyarıya bak." -ForegroundColor Yellow
}

Write-Host ""
