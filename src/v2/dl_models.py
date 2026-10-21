"""V2 deep models from the proposal: CNN, BiLSTM, CNN-BiLSTM-Attention (trained from scratch, PyTorch).

    python -m src.v2.dl_models --arch cnn      --task binary
    python -m src.v2.dl_models --arch bilstm   --task binary
    python -m src.v2.dl_models --arch hybrid   --task binary      # CNN + BiLSTM + additive attention
    add --context to concatenate the 11 contextual risk features before the classifier

Tokens: XLM-R SentencePiece sub-words (multilingual, handles Sinhala / Tamil / romanised text); embeddings are learned
from scratch (no pre-training) - this is what separates these models from XLM-R fine-tuning.
Protocol identical to src/v2/xlmr_train.py: family-grouped validation split of TRAIN for early stopping, checkpoint
and threshold; class-weighted loss; fixed seeds; held-out test and real benchmark evaluated once at the end.
Outputs results/v2/dl/<run>/ (config, history, metrics, by_language, pred_*), models/v2/dl/<run>.pt
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.v2.common import CLASS3, DATA, MOD, RES, env_info, md5
from src.v2.evaluation import binary, bootstrap_ci, f1_threshold, per_language, pct
from src.v2.models import context_matrix


def build(arch, vocab, n_cls, n_ctx, emb=128, hid=128):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab, emb, padding_idx=1)
            self.drop = nn.Dropout(0.3)
            if arch in ("cnn", "hybrid"):
                self.convs = nn.ModuleList([nn.Conv1d(emb, hid, k, padding=k // 2) for k in (2, 3, 4, 5)])
            if arch == "bilstm":
                self.lstm = nn.LSTM(emb, hid, batch_first=True, bidirectional=True)
            if arch == "hybrid":
                self.lstm = nn.LSTM(hid * 4, hid, batch_first=True, bidirectional=True)
                self.att = nn.Sequential(nn.Linear(2 * hid, hid), nn.Tanh(), nn.Linear(hid, 1))
            out = {"cnn": hid * 4, "bilstm": 2 * hid, "hybrid": 2 * hid}[arch]
            self.fc = nn.Sequential(nn.Linear(out + n_ctx, hid), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hid, n_cls))

        def forward(self, ids, mask, ctx):
            x = self.drop(self.emb(ids))                                   # B, T, E
            m = mask.unsqueeze(-1).float()
            if arch == "cnn":
                c = torch.cat([torch.relu(cv(x.transpose(1, 2)))[:, :, :x.size(1)] for cv in self.convs], 1)  # B, 4H, T
                c = c.masked_fill(mask.unsqueeze(1) == 0, -1e4); h = c.max(2).values
            elif arch == "bilstm":
                o, _ = self.lstm(x); h = (o * m).sum(1) / m.sum(1).clamp(min=1)          # masked mean pooling
            else:
                c = torch.cat([torch.relu(cv(x.transpose(1, 2)))[:, :, :x.size(1)] for cv in self.convs], 1).transpose(1, 2)
                o, _ = self.lstm(self.drop(c)); a = self.att(o).squeeze(-1).masked_fill(mask == 0, -1e4)
                h = (torch.softmax(a, 1).unsqueeze(-1) * o).sum(1)                       # attention pooling
            if ctx is not None and ctx.size(1):
                h = torch.cat([h, ctx], 1)
            return self.fc(self.drop(h))
    return Net()


def main(a):
    import random
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoTokenizer

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    run = a.run or f"{a.arch}{'_ctx' if a.context else ''}_{a.task}_seed{a.seed}"
    out = RES / "dl" / run; out.mkdir(parents=True, exist_ok=True); (MOD / "dl").mkdir(parents=True, exist_ok=True)
    tr_all = pd.read_csv(DATA / "train.csv"); te = pd.read_csv(DATA / "test.csv")
    reg = pd.read_csv(DATA / "registry.csv"); rl = reg[reg.role == "real_benchmark"].reset_index(drop=True)
    lab = (lambda d: (d.label3 == "smishing").astype(int).to_numpy()) if a.task == "binary" else (lambda d: d.label3.map(CLASS3).to_numpy())
    fi, vi = next(StratifiedGroupKFold(7, shuffle=True, random_state=a.seed).split(tr_all, tr_all.class3 + "|" + tr_all.lang_family, tr_all.family_id))
    tr, va = tr_all.iloc[fi].reset_index(drop=True), tr_all.iloc[vi].reset_index(drop=True)
    n_cls = 2 if a.task == "binary" else 3
    tok = AutoTokenizer.from_pretrained("xlm-roberta-base")

    def loader(d, shuffle):
        e = tok(d.text_model.astype(str).tolist(), truncation=True, max_length=a.max_len, padding="max_length", return_tensors="pt")
        ctx = torch.tensor(context_matrix(d), dtype=torch.float) if a.context else torch.zeros(len(d), 0)
        ds = TensorDataset(e["input_ids"], e["attention_mask"], ctx, torch.tensor(lab(d), dtype=torch.long))
        g = torch.Generator(); g.manual_seed(a.seed)
        return DataLoader(ds, batch_size=a.bs if shuffle else 256, shuffle=shuffle, generator=g)
    L = {"train": loader(tr, True), "val": loader(va, False), "test": loader(te, False), "real": loader(rl, False)}

    n_ctx = context_matrix(tr).shape[1] if a.context else 0
    model = build(a.arch, tok.vocab_size, n_cls, n_ctx).to(dev)
    ytr = lab(tr); w = torch.tensor(len(ytr) / (n_cls * np.bincount(ytr, minlength=n_cls)), dtype=torch.float, device=dev)
    lossf = torch.nn.CrossEntropyLoss(weight=w); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)

    def predict(dl):
        model.eval(); P = []
        with torch.no_grad():
            for ids, m, c, _ in dl:
                P.append(torch.softmax(model(ids.to(dev), m.to(dev), c.to(dev)), -1).cpu().numpy())
        return np.vstack(P)

    def vscore(P, y):
        if a.task == "binary":
            return binary(y, (P[:, 1] >= .5).astype(int), P[:, 1])["auprc"]
        from sklearn.metrics import f1_score
        return f1_score(y, P.argmax(1), average="macro")

    hist, best, bad, ckpt = [], -1, 0, MOD / "dl" / f"{run}.pt"
    for ep in range(1, a.epochs + 1):
        model.train(); t0, tot = time.time(), 0.0
        for ids, m, c, y in L["train"]:
            opt.zero_grad(); loss = lossf(model(ids.to(dev), m.to(dev), c.to(dev)), y.to(dev)); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tot += loss.item()
        Pv = predict(L["val"]); vs = vscore(Pv, lab(va))
        hist.append({"epoch": ep, "train_loss": tot / len(L["train"]), "val_metric": vs, "seconds": round(time.time() - t0, 1)})
        print(f"epoch {ep}: loss {tot/len(L['train']):.4f}  val {vs:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        if vs > best + 1e-4:
            best, bad = vs, 0; torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= a.patience:
                print("early stopping"); break
    pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    P = {k: predict(L[k]) for k in ["val", "test", "real"]}
    rows, langs = [], []
    name = f"{a.arch.upper()}{' + context' if a.context else ''}"
    cfg = {**vars(a), "run": run, "device": str(dev), "params": sum(p.numel() for p in model.parameters()),
           "data_md5": {k: md5(DATA / f) for k, f in [("train", "train.csv"), ("test", "test.csv"), ("registry", "registry.csv")]},
           "environment": env_info()}
    if a.task == "binary":
        thr = f1_threshold(lab(va), P["val"][:, 1]); cfg["threshold"] = thr
        for k, d, key in [("validation", va, "val"), ("held-out test", te, "test"), ("real benchmark", rl, "real")]:
            y, s = lab(d), P[key][:, 1]; p = (s >= thr).astype(int)
            r = {"model": name, "set": k, **pct(binary(y, p, s))}
            if k != "validation":
                r.update(bootstrap_ci(y, p, s)); langs.append(per_language(d.lang_family.values, y, p, s).assign(model=name, set=k))
            rows.append(r)
            pd.DataFrame({"msg_id": d.msg_id, "lang_family": d.lang_family, "class3": d.class3, "score": s, "pred": p}).to_csv(out / f"pred_{key}.csv", index=False)
    else:
        from src.v2.three_class import multi
        for k, d, key in [("validation", va, "val"), ("held-out test", te, "test"), ("real benchmark", rl, "real")]:
            y, s = lab(d), P[key]
            rows.append({"model": name, "set": k, **pct(multi(y, s.argmax(1), s))})
            pd.DataFrame({"msg_id": d.msg_id, "lang_family": d.lang_family, "class3": d.class3, "pred": s.argmax(1),
                          **{f"p_{c}": s[:, i] for i, c in enumerate(["normal", "promotional", "scam"])}}).to_csv(out / f"pred_{key}.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    if langs:
        pd.concat(langs).to_csv(out / "by_language.csv", index=False)
    (out / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    print(pd.DataFrame(rows)[[c for c in ["set", "precision", "recall", "f1", "fpr", "auprc", "macro_f1"] if c in rows[0]]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", choices=["cnn", "bilstm", "hybrid"], default="hybrid")
    ap.add_argument("--task", choices=["binary", "three"], default="binary")
    ap.add_argument("--context", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--max_len", type=int, default=128); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true"); ap.add_argument("--run", default=None)
    main(ap.parse_args())
