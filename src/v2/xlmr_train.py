"""V2 XLM-R fine-tuning (binary scam vs non-scam, or three-class). Runs on GPU (Colab) with CPU fallback.

    python -m src.v2.xlmr_train --task binary --lr 2e-5 --bs 16
    python -m src.v2.xlmr_train --task three  --lr 2e-5 --bs 16
    python -m src.v2.xlmr_train --token-stats            # token length analysis only

Data: data/v2/train.csv, data/v2/test.csv, data/v2/registry.csv (role == real_benchmark).
Validation: one family-grouped 15% split of TRAIN (no shared near-duplicate families) - used for early stopping,
checkpoint selection and the binary decision threshold. Held-out test and real benchmark are evaluated once at the end.
Imbalance: class-weighted cross-entropy (weights from the training part only).
Reproducibility: fixed seeds (python, numpy, torch, cuda), deterministic data order per seed, config saved.
Outputs results/v2/xlmr/<run>/: config.json, history.csv (loss + validation metrics per epoch), metrics.csv,
        by_language.csv, pred_*.csv;  models/v2/xlmr/<run>/ best checkpoint (model + tokenizer).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.v2.common import CLASS3, DATA, MOD, RES, env_info, md5
from src.v2.evaluation import binary, bootstrap_ci, f1_threshold, per_language, pct


def set_seed(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def load(task):
    tr = pd.read_csv(DATA / "train.csv"); te = pd.read_csv(DATA / "test.csv")
    reg = pd.read_csv(DATA / "registry.csv"); rl = reg[reg.role == "real_benchmark"].reset_index(drop=True)
    lab = (lambda d: (d.label3 == "smishing").astype(int).to_numpy()) if task == "binary" else (lambda d: d.label3.map(CLASS3).to_numpy())
    return tr, te, rl, lab


def token_stats(model_name, max_len):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    tr, te, rl, _ = load("binary"); rows = []
    for name, d in [("train", tr), ("test", te), ("real", rl)]:
        n = np.array([len(tok(t)["input_ids"]) for t in d.text_model.astype(str)])
        for g in ["ALL", "EN", "SI", "TA", "mixed"]:
            k = np.ones(len(d), bool) if g == "ALL" else (d.lang_family.to_numpy() == g)
            if k.sum():
                rows.append({"set": name, "lang": g, "n": int(k.sum()), "mean": round(n[k].mean(), 1), "p50": int(np.percentile(n[k], 50)),
                             "p95": int(np.percentile(n[k], 95)), "max": int(n[k].max()), f"pct_over_{max_len}": round((n[k] > max_len).mean() * 100, 2)})
    out = RES / "xlmr"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "token_lengths.csv", index=False); print(pd.DataFrame(rows).to_string(index=False))


def main(a):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    set_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    run = a.run or f"{a.task}_lr{a.lr:g}_bs{a.bs * a.accum}_len{a.max_len}_seed{a.seed}"
    out, mdir = RES / "xlmr" / run, MOD / "xlmr" / run
    out.mkdir(parents=True, exist_ok=True); mdir.mkdir(parents=True, exist_ok=True)
    tr_all, te, rl, lab = load(a.task)
    strat = tr_all.class3 + "|" + tr_all.lang_family
    fit_idx, val_idx = next(StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=a.seed).split(tr_all, strat, tr_all.family_id))
    tr, va = tr_all.iloc[fit_idx].reset_index(drop=True), tr_all.iloc[val_idx].reset_index(drop=True)
    assert not set(tr.family_id) & set(va.family_id)
    n_cls = 2 if a.task == "binary" else 3
    ytr, yva, yte, yrl = lab(tr), lab(va), lab(te), lab(rl)

    tok = AutoTokenizer.from_pretrained(a.model)
    enc = lambda d: tok(d.text_model.astype(str).tolist(), truncation=True, max_length=a.max_len, padding="max_length", return_tensors="pt")
    def loader(d, y, shuffle):
        e = enc(d); ds = TensorDataset(e["input_ids"], e["attention_mask"], torch.tensor(y, dtype=torch.long))
        g = torch.Generator(); g.manual_seed(a.seed)
        return DataLoader(ds, batch_size=a.bs if shuffle else 64, shuffle=shuffle, generator=g)
    L = {"train": loader(tr, ytr, True), "val": loader(va, yva, False), "test": loader(te, yte, False), "real": loader(rl, yrl, False)}

    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=n_cls).to(dev)
    if a.grad_ckpt:
        model.gradient_checkpointing_enable()
    counts = np.bincount(ytr, minlength=n_cls); w = torch.tensor(len(ytr) / (n_cls * counts), dtype=torch.float, device=dev)
    lossf = torch.nn.CrossEntropyLoss(weight=w if a.class_weight else None)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = (len(L["train"]) // a.accum) * a.epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.cuda.amp.GradScaler(enabled=dev.type == "cuda")

    def predict(dl):
        model.eval(); P = []
        with torch.no_grad():
            for ids, att, _ in dl:
                with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                    logits = model(input_ids=ids.to(dev), attention_mask=att.to(dev)).logits
                P.append(torch.softmax(logits.float(), -1).cpu().numpy())
        return np.vstack(P)

    def val_score(P, y):
        if a.task == "binary":
            return binary(y, (P[:, 1] >= 0.5).astype(int), P[:, 1])["auprc"]
        from sklearn.metrics import f1_score
        return f1_score(y, P.argmax(1), average="macro")

    cfg = {**vars(a), "run": run, "device": str(dev), "val_size": len(va), "train_size": len(tr), "class_weights": w.tolist(),
           "selection": "validation AUPRC (binary) / macro-F1 (three) for early stopping + checkpoint",
           "data_md5": {k: md5(DATA / f) for k, f in [("train", "train.csv"), ("test", "test.csv"), ("registry", "registry.csv")]},
           "environment": env_info()}
    (out / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    hist, best, bad = [], -1, 0
    for ep in (range(1, a.epochs + 1) if not a.eval_only else []):
        model.train(); t0, tot = time.time(), 0.0
        opt.zero_grad()
        for step, (ids, att, y) in enumerate(L["train"], 1):
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                loss = lossf(model(input_ids=ids.to(dev), attention_mask=att.to(dev)).logits.float(), y.to(dev))
            scaler.scale(loss / a.accum).backward(); tot += loss.item()
            if step % a.accum == 0 or step == len(L["train"]):
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sch.step(); opt.zero_grad()
            if step % 100 == 0:
                print(f"  epoch {ep} step {step}/{len(L['train'])} loss {tot/step:.4f} ({time.time()-t0:.0f}s)", flush=True)
        Pv = predict(L["val"]); vs = val_score(Pv, yva)
        hist.append({"epoch": ep, "train_loss": tot / len(L["train"]), "val_metric": vs, "seconds": round(time.time() - t0)})
        print(f"epoch {ep}: train loss {tot/len(L['train']):.4f}  val {'AUPRC' if a.task=='binary' else 'macro-F1'} {vs:.4f}", flush=True)
        if vs > best + 1e-4:
            best, bad = vs, 0; model.save_pretrained(mdir); tok.save_pretrained(mdir)
        else:
            bad += 1
            if bad >= a.patience:
                print("early stopping"); break
    if hist:
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
    import gc
    del model, opt; gc.collect(); torch.cuda.empty_cache()           # free the training copy before reloading the best checkpoint
    model = AutoModelForSequenceClassification.from_pretrained(mdir).to(dev)
    P = {k: predict(L[k]) for k in ["val", "test", "real"]}
    rows, langs = [], []
    if a.task == "binary":
        thr = f1_threshold(yva, P["val"][:, 1]); cfg["threshold"] = thr
        for k, d, y in [("validation", va, yva), ("held-out test", te, yte), ("real benchmark", rl, yrl)]:
            s = P["val" if k == "validation" else ("test" if k.startswith("held") else "real")][:, 1]; p = (s >= thr).astype(int)
            r = {"model": f"XLM-R ({run})", "set": k, **pct(binary(y, p, s))}
            if k != "validation":
                r.update(bootstrap_ci(y, p, s)); langs.append(per_language(d.lang_family.values, y, p, s).assign(set=k))
            rows.append(r)
            pd.DataFrame({"msg_id": d.msg_id, "lang_family": d.lang_family, "class3": d.class3, "score": s, "pred": p}).to_csv(out / f"pred_{k.split()[0]}.csv", index=False)
    else:
        from src.v2.three_class import multi
        for k, d, y in [("validation", va, yva), ("held-out test", te, yte), ("real benchmark", rl, yrl)]:
            s = P["val" if k == "validation" else ("test" if k.startswith("held") else "real")]
            rows.append({"model": f"XLM-R ({run})", "set": k, **pct(multi(y, s.argmax(1), s))})
            pd.DataFrame({"msg_id": d.msg_id, "lang_family": d.lang_family, "class3": d.class3, "pred": s.argmax(1),
                          **{f"p_{c}": s[:, i] for i, c in enumerate(["normal", "promotional", "scam"])}}).to_csv(out / f"pred_{k.split()[0]}.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    if langs:
        pd.concat(langs).to_csv(out / "by_language.csv", index=False)
    (out / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["binary", "three"], default="binary")
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=128); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-class-weight", dest="class_weight", action="store_false")
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps (effective batch = bs * accum)")
    ap.add_argument("--grad-ckpt", dest="grad_ckpt", action="store_true", help="gradient checkpointing (less GPU memory)")
    ap.add_argument("--eval-only", dest="eval_only", action="store_true", help="skip training; evaluate the saved best checkpoint")
    ap.add_argument("--cpu", action="store_true"); ap.add_argument("--run", default=None)
    ap.add_argument("--token-stats", action="store_true")
    a = ap.parse_args()
    token_stats(a.model, a.max_len) if a.token_stats else main(a)
