# v2.1-final overnight plan (PowerShell twin of run_night.sh). Resumable; a failed run is logged, not fatal.
#   powershell -ExecutionPolicy Bypass -File scripts\run_night.ps1
param([string]$Py = "python", [int[]]$Seeds = @(0, 1, 2))
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONUTF8 = "1"; $env:PYTHONUNBUFFERED = "1"
$Root = "results/v2.1-final"; $Log = "$Root/logs"; $Fails = "$Log/failures.txt"
New-Item -ItemType Directory -Force "$Log/runs" | Out-Null

function Note($msg) { $l = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $msg"; Write-Host $l; Add-Content "$Log/night.log" $l }

function Step($name, [string[]]$cmd) {
    if (Test-Path "$Log/$name.ok") { Note "skip $name (done)"; return $true }
    Note "start $name"
    & $Py @cmd *> "$Log/$name.log"
    if ($LASTEXITCODE -eq 0) { New-Item "$Log/$name.ok" -ItemType File -Force | Out-Null; Note "ok $name"; return $true }
    Note "FAILED $name (see $Log/$name.log)"; Add-Content $Fails $name; return $false
}

function Run($name, [string[]]$runArgs) {
    foreach ($chunk in 256, 128, 64) {
        & $Py scripts/run.py --resume @runArgs --set "dp.vmap_chunk=$chunk" *>> "$Log/runs/$name.log"
        if ($LASTEXITCODE -eq 0) { return }
        Note "retry $name (attempt failed, vmap_chunk=$chunk)"
    }
    Note "FAILED run $name"; Add-Content $Fails "run $name"
}

Note "=== night run start ==="
$EncAll = "[nih,mimic_ch,pc]"
if (-not (Step "extract" @("scripts/02_extract.py"))) { $EncAll = "[nih]"; Note "extraction failed -> nih-only fallback" }
if (-not (Step "sanity" @("scripts/03_sanity.py", "--set", "features.encoders=$EncAll", "--set", "sanity.shuffle_tol=0.10"))) { Note "SANITY FAILED - stopping"; exit 1 }

foreach ($enc in @("[nih]", $EncAll)) {
    $en = if ($enc -eq "[nih]") { "nih" } else { "all" }
    foreach ($head in "film", "linear") { foreach ($loss in "bce_pw", "asl") {
        $t = "trial_${en}_${head}_${loss}"
        Run $t @("--exp", "E2", "--seed", "0", "--tag", $t, "--set", "features.encoders=$enc", "--set", "model.head=$head", "--set", "model.loss=$loss")
    } }
    $t = "trial_${en}_film_bce_pw_ema"
    Run $t @("--exp", "E2", "--seed", "0", "--tag", $t, "--set", "features.encoders=$enc", "--set", "model.head=film", "--set", "model.loss=bce_pw", "--set", "model.ema=0.999")
}
if (-not (Step "freeze_model" @("scripts/freeze.py", "--stage", "model"))) { exit 1 }
$Frozen = "$Root/frozen.yaml"

foreach ($lr in "0.03", "0.1", "0.3") { foreach ($cb in @(@("1.0", "256"), @("0.5", "256"), @("1.0", "512"))) {
    $t = "dpsweep_lr${lr}_C$($cb[0])_B$($cb[1])"
    Run $t @("--exp", "E4", "--eps", "3", "--seed", "0", "--config", $Frozen, "--tag", $t, "--set", "dp.lr=$lr", "--set", "dp.clip=$($cb[0])", "--set", "dp.batch=$($cb[1])")
} }
if (-not (Step "freeze_dp" @("scripts/freeze.py", "--stage", "dp"))) { exit 1 }

foreach ($s in $Seeds) {
    $F = @("--seed", "$s", "--config", $Frozen, "--final")
    Run "E0_s$s" (@("--exp", "E0") + $F)
    Run "E1_s$s" (@("--exp", "E1") + $F)
    Run "E2_s$s" (@("--exp", "E2") + $F)
    Run "E3_s$s" (@("--exp", "E3") + $F)
    foreach ($e in 1, 3, 8) { Run "E4_eps${e}_s$s" (@("--exp", "E4", "--eps", "$e") + $F) }
    Run "E5_eps8_s$s" (@("--exp", "E5", "--eps", "8") + $F)
    Run "E7_eps3_s$s" (@("--exp", "E7", "--eps", "3") + $F)
    Run "E8_E3_int8_s$s" (@("--exp", "E8") + $F)
    Run "E8_eps3_int8_s$s" (@("--exp", "E8", "--eps", "3") + $F)
    Run "E8_eps3_topk_s$s" (@("--exp", "E8", "--eps", "3") + $F + @("--set", "comm.up=topk", "--set", "comm.down=fp16"))
}
Step "aggregate" @("scripts/aggregate.py") | Out-Null
Step "compare_scales" @("scripts/compare_scales.py") | Out-Null
Step "export_model" @("scripts/export_model.py") | Out-Null
Note "=== night run end ==="
