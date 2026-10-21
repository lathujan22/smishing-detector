# Builds FINAL_DELIVERY\ from the project (copies only; nothing in the project is moved or changed).
$ErrorActionPreference = "Continue"
$P = "D:\ladhu\smishing-detector"; $F = "$P\FINAL_DELIVERY"
$dirs = "01_selected_models\best_ML_LinearSVC","01_selected_models\best_DL_XLM-R","02_all_models\dl","03_results","04_figures\v1_baseline","04_figures\scam_only","04_figures\three_class","04_figures\v2","05_code","06_web_app"
$dirs | % { New-Item -ItemType Directory -Force "$F\$_" | Out-Null }
Copy-Item "$P\models\v2\three_linearsvc.joblib" "$F\01_selected_models\best_ML_LinearSVC\" -Force
Copy-Item "$P\models\v2\xlmr\three_lr2e-05_bs16_len128_seed42\*" "$F\01_selected_models\best_DL_XLM-R\" -Force
Copy-Item "$P\models\v2\*.joblib" "$F\02_all_models\" -Force
Copy-Item "$P\models\v2\dl\*.pt" "$F\02_all_models\dl\" -Force
# results: tables / reports / configs only (no pred_* files, no message text)
robocopy "$P\results\v2" "$F\03_results" /E /XF pred_*.csv *.log app_test.txt /XD figures logs /NFL /NDL /NJH /NJS | Out-Null
Copy-Item "$P\results\*.md","$P\results\*.csv" "$F\03_results\v1_baseline_tables\" -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force "$F\03_results\v1_baseline_tables" | Out-Null
Get-ChildItem "$P\results\*.md","$P\results\*.csv" | Where-Object { $_.Name -notlike "pred_*" } | Copy-Item -Destination "$F\03_results\v1_baseline_tables\" -Force
Copy-Item "$P\results\figures\*.png" "$F\04_figures\v1_baseline\" -Force
Copy-Item "$P\results\scam_only\figures\*.png" "$F\04_figures\scam_only\" -Force
Copy-Item "$P\results\three_class\figures\*.png" "$F\04_figures\three_class\" -Force
Copy-Item "$P\results\v2\figures\*.png" "$F\04_figures\v2\" -Force
foreach ($d in "src","app","scripts","notebooks","docs","lexicons","tests") { if (Test-Path "$P\$d") { robocopy "$P\$d" "$F\05_code\$d" /E /XD __pycache__ /NFL /NDL /NJH /NJS | Out-Null } }
Copy-Item "$P\start_app.bat","$P\requirements.txt","$P\README.md" "$F\05_code\" -Force -ErrorAction SilentlyContinue
Copy-Item "$P\app\samples.json","$P\start_app.bat" "$F\06_web_app\" -Force
Copy-Item "$F\README.md" "$F\06_web_app\HOW_TO_RUN.md" -Force
$n = (Get-ChildItem $F -Recurse -File | Measure-Object).Count; $gb = (Get-ChildItem $F -Recurse -File | Measure-Object Length -Sum).Sum / 1GB
"FINAL_DELIVERY: $n files, {0:N2} GB" -f $gb
Get-ChildItem $F -Directory -Recurse | % { "{0,-60} {1,4} files" -f $_.FullName.Replace($F,""), (Get-ChildItem $_.FullName -File).Count }
