"""Shared V2 helpers: paths, label names, provenance map, checksums, run manifests."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "v2"                 # private (git-ignored via data/*)
RES = ROOT / "results" / "v2"               # aggregate results only (no message text except pred_* files)
MOD = ROOT / "models" / "v2"
SEED = 42
CLASS3 = {"ham": 0, "spam": 1, "smishing": 2}
CLASS_NAMES = ["normal", "promotional", "scam"]
LANG_FAMILY = {"en": "EN", "si": "SI", "si-en": "SI", "ta": "TA", "ta-en": "TA", "si-ta": "mixed", "mixed": "mixed"}

# provenance of every existing source (label origin and whether the text is synthetic)
PROVENANCE = {
    "uci": dict(label_provenance="public dataset label (UCI SMS Spam Collection, human)", is_synthetic=False,
                collection_method="public dataset", licence="CC BY 4.0 (UCI)"),
    "mendeley": dict(label_provenance="public dataset label (Mendeley SMS phishing dataset, human)", is_synthetic=False,
                     collection_method="public dataset", licence="CC BY 4.0 (Mendeley Data)"),
    "multilingual": dict(label_provenance="researcher-written template (synthetic text, label by design)", is_synthetic=True,
                         collection_method="template generation (src/gen_multilingual.py) + handwritten templates",
                         licence="own work"),
    "mt_nllb": dict(label_provenance="label inherited from English source; text machine-translated (NLLB)", is_synthetic=True,
                    collection_method="machine translation of training messages", licence="derived from source dataset"),
    "real_form": dict(label_provenance="researcher label of real SMS (respondent label kept separately)", is_synthetic=False,
                      collection_method="Google Form / own phones", licence="research consent"),
}


def md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def env_info() -> dict:
    import scipy, sklearn
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__}
    for mod in ["xgboost", "torch", "transformers"]:
        try:
            info[mod] = __import__(mod).__version__
        except ImportError:
            info[mod] = "not installed"
    return info


def write_manifest(out_dir: Path, name: str, config: dict, data_files: dict[str, Path]) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    man = {"run": name, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": SEED, "config": config,
           "data": {k: {"path": str(Path(p).relative_to(ROOT)), "md5": md5(p), "rows": int(sum(1 for _ in open(p, encoding="utf-8")) - 1)}
                    for k, p in data_files.items()},
           "environment": env_info()}
    (out_dir / f"{name}_manifest.json").write_text(json.dumps(man, indent=2, default=str), encoding="utf-8")
    return man
