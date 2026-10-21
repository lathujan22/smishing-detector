# Runs all V2 deep-learning experiments one after another; progress -> results\v2\logs\progress.txt
$ErrorActionPreference = "Continue"
Set-Location D:\ladhu\smishing-detector
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONWARNINGS = "ignore"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"; $env:TOKENIZERS_PARALLELISM = "false"
New-Item -ItemType Directory -Force results\v2\logs | Out-Null
$P = "results\v2\logs\progress.txt"
$jobs = @(
  @("dl_cnn_binary",     "-m", "src.v2.dl_models", "--arch", "cnn",    "--task", "binary"),
  @("dl_bilstm_binary",  "-m", "src.v2.dl_models", "--arch", "bilstm", "--task", "binary"),
  @("dl_hybrid_binary",  "-m", "src.v2.dl_models", "--arch", "hybrid", "--task", "binary"),
  @("xlmr_binary",       "-m", "src.v2.xlmr_train", "--task", "binary", "--lr", "2e-5", "--bs", "16"),
  @("xlmr_three",        "-m", "src.v2.xlmr_train", "--task", "three",  "--lr", "2e-5", "--bs", "16"),
  @("dl_hybrid_three",   "-m", "src.v2.dl_models", "--arch", "hybrid", "--task", "three")
)
"ALL START $(Get-Date -Format s)" | Add-Content $P
foreach ($j in $jobs) {
  $name = $j[0]; $cmd = $j[1..($j.Length - 1)]
  "START $name $(Get-Date -Format s)" | Add-Content $P
  & python @cmd *> "results\v2\logs\$name.log"
  "END   $name exit=$LASTEXITCODE $(Get-Date -Format s)" | Add-Content $P
}
"ALL DONE $(Get-Date -Format s)" | Add-Content $P
