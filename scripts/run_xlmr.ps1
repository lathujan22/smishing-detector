# XLM-R runs (memory-saving mode for a 6 GB laptop GPU); progress -> results\v2\logs\progress.txt
$ErrorActionPreference = "Continue"
Set-Location D:\ladhu\smishing-detector
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONWARNINGS = "ignore"; $env:PYTHONUNBUFFERED = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"; $env:TOKENIZERS_PARALLELISM = "false"
$P = "results\v2\logs\progress.txt"
$jobs = @(
  @("xlmr_binary",     "-m", "src.v2.xlmr_train", "--task", "binary", "--lr", "2e-5", "--bs", "8", "--accum", "2", "--grad-ckpt", "--epochs", "5"),
  @("xlmr_three",      "-m", "src.v2.xlmr_train", "--task", "three",  "--lr", "2e-5", "--bs", "8", "--accum", "2", "--grad-ckpt", "--epochs", "5"),
  @("dl_hybrid_three", "-m", "src.v2.dl_models", "--arch", "hybrid", "--task", "three")
)
foreach ($j in $jobs) {
  $name = $j[0]; $cmd = $j[1..($j.Length - 1)]
  "START $name $(Get-Date -Format s)" | Add-Content $P
  cmd /c "python $($cmd -join ' ') > results\v2\logs\$name.log 2>&1"
  "END   $name exit=$LASTEXITCODE $(Get-Date -Format s)" | Add-Content $P
}
"ALL DONE $(Get-Date -Format s)" | Add-Content $P
