# make.ps1 - PowerShell-обёртка целей Makefile для Windows без установленного make.
# Запуск из каталога infra/:  .\make.ps1 up   |   .\make.ps1 ps   |   .\make.ps1 down
# Те же цели, что в Makefile. Пути артефактов проверяются идемпотентно.

param(
    [Parameter(Position = 0)]
    [string]$Target = "help"
)

$ErrorActionPreference = "Stop"
# Каталог скрипта = infra/. Корень репозитория - на уровень выше.
$Infra = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $Infra
Set-Location $Infra

$Compose = @("compose", "--env-file", ".env", "-f", "docker-compose.yml")
$Onnx = Join-Path $Root "models\siglip2_vision.onnx"
$ImagesMarker = Join-Path $Root "data\images\.done"

function Invoke-DC { param([string[]]$Args) & docker @Compose @Args }

function Ensure-Env {
    if (-not (Test-Path ".env")) {
        Copy-Item ".env.example" ".env"
        Write-Host "infra/.env создан из .env.example"
    }
}

function Export-Onnx {
    # Идемпотентно: не пересобираем 370MB-модель, если она уже на месте.
    if (Test-Path $Onnx) {
        Write-Host "ONNX уже на месте: $Onnx (пропуск)"
    } else {
        Push-Location $Root; try { uv run python scripts/export_onnx.py } finally { Pop-Location }
    }
}

function Fetch-Images {
    # Идемпотентно: банк картинок качаем один раз (маркер .done).
    if (Test-Path $ImagesMarker) {
        Write-Host "Банк картинок уже на месте (пропуск)"
    } else {
        Push-Location $Root; try { uv run python scripts/fetch_demo_images.py } finally { Pop-Location }
    }
}

switch ($Target) {
    "help" {
        Write-Host "Цели: env, export-onnx, fetch-images, build, up, down, teardown, logs, ps, smoke, e2e, test"
    }
    "env"          { Ensure-Env }
    "export-onnx"  { Export-Onnx }
    "fetch-images" { Fetch-Images }
    "build"        { Invoke-DC @("build") }
    "up" {
        Ensure-Env; Export-Onnx; Fetch-Images
        Invoke-DC @("build")
        Invoke-DC @("up", "-d")
        Write-Host "Стек поднимается. Дождитесь healthy: .\make.ps1 ps. Первый запуск может занять несколько минут."
    }
    "down"     { Invoke-DC @("down") }
    "teardown" { Invoke-DC @("down", "-v") }
    "logs"     { Invoke-DC @("logs", "-f", "--tail=100") }
    "ps"       { Invoke-DC @("ps") }
    "smoke"    { & bash ./smoke.sh }
    "e2e"      { & bash ./e2e.sh }
    "test"     { Push-Location $Root; try { uv run ruff check .; uv run pytest -q } finally { Pop-Location } }
    default    { Write-Host "Неизвестная цель: $Target"; exit 1 }
}
