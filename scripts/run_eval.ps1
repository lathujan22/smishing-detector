# Waits for run_xlmr.ps1 to finish, then evaluates the saved best XLM-R checkpoints (no retraining).
$ErrorActionPreference = "Continue"
Set-Location D:\ladhu\smishing-detector
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONWARNINGS = "ignore"; $env:PYTHONUNBUFFERED = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"; $env:TOKENIZERS_PARALLELISM = "false"
$P = "results\v2\logs\progress.txt"
while (-not (Select-String -Path $P -Pattern "ALL DONE" -Quiet)) { Start-Sleep 30 }
foreach ($t in "binary", "three") {
  if (-not (Test-Path "results\v2\xlmr\${t}_lr2e-05_bs16_len128_seed42\metrics.csv")) {
    "START xlmr_${t}_eval $(Get-Date -Format s)" | Add-Content $P
    cmd /c "python -m src.v2.xlmr_train --task $t --lr 2e-5 --bs 8 --accum 2 --eval-only > results\v2\logs\xlmr_${t}_eval.log 2>&1"
    "END   xlmr_${t}_eval exit=$LASTEXITCODE $(Get-Date -Format s)" | Add-Content $P
  }
}
"EVAL DONE $(Get-Date -Format s)" | Add-Content $P
