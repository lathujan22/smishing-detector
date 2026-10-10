"""Training-time augmentation: text noise + SMOTE (applied to TRAINING data only).

add_noise(text, rng, p_char, p_drop)  realistic SMS noise on the masked text:
    - character typos in a word: delete / duplicate / swap neighbours
    - look-alike spellings scammers use to dodge filters: o->0, e->3, a->@, i->1, s->5
    - word dropout and neighbour swaps
    placeholders (<URL>, <PHONE>, <MONEY> ...) are never changed.
noisy_copies(texts, y, groups, ...)   one noisy copy per message, same label + same CV group
smote(X, y, k, seed)                  SMOTE on a (sparse) TF-IDF matrix: synthetic minority samples
                                      x_new = x + lambda * (x_nn - x) between a sample and one of its
                                      k nearest same-class neighbours (cosine). Balances classes 1:1.
"""
from __future__ import annotations

import re

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

LEET = {"o": "0", "e": "3", "a": "@", "i": "1", "s": "5", "l": "1"}
PLACEHOLDER = re.compile(r"^<[A-Z]+>$")
LATIN = re.compile(r"[a-z]")


def _typo(word: str, rng: np.random.RandomState) -> str:
    if len(word) < 3:
        return word
    i = rng.randint(0, len(word) - 1)
    op = rng.randint(4) if LATIN.search(word) else rng.randint(2)   # leet / swap only for Latin words
    if op == 0:
        return word[:i] + word[i + 1:]                               # delete
    if op == 1:
        return word[:i] + word[i] + word[i:]                         # duplicate
    if op == 2:
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]       # swap neighbours
    cands = [j for j, c in enumerate(word) if c in LEET]
    if not cands:
        return word
    j = cands[rng.randint(len(cands))]
    return word[:j] + LEET[word[j]] + word[j + 1:]                   # look-alike


def add_noise(text: str, rng: np.random.RandomState, p_char: float = 0.15, p_drop: float = 0.08,
              p_swap: float = 0.05) -> str:
    words = str(text).split()
    out = []
    for w in words:
        if PLACEHOLDER.match(w):
            out.append(w)
            continue
        if rng.rand() < p_drop and len(words) > 3:
            continue
        out.append(_typo(w, rng) if rng.rand() < p_char else w)
    for i in range(len(out) - 1):
        if rng.rand() < p_swap and not PLACEHOLDER.match(out[i]) and not PLACEHOLDER.match(out[i + 1]):
            out[i], out[i + 1] = out[i + 1], out[i]
    return " ".join(out) if out else str(text)


def noisy_copies(texts, y, groups=None, weights=None, seed: int = 42, copies: int = 1, **kw):
    """Original data + `copies` noisy versions of every message (labels, groups and weights repeated)."""
    rng = np.random.RandomState(seed)
    texts = list(texts)
    X = texts + [add_noise(t, rng, **kw) for _ in range(copies) for t in texts]
    rep = lambda a: None if a is None else np.concatenate([np.asarray(a)] * (copies + 1))
    return X, rep(y), rep(groups), rep(weights)


def smote(X, y, k: int = 5, seed: int = 42):
    """Oversample the minority class to 1:1 with SMOTE. Works on scipy sparse or dense matrices."""
    y = np.asarray(y)
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    minority = classes[np.argmin(counts)]
    n_new = counts.max() - counts.min()
    if n_new == 0:
        return X, y
    idx = np.where(y == minority)[0]
    Xm = X[idx]
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(idx)), metric="cosine", algorithm="brute").fit(Xm)
    neigh = nn.kneighbors(Xm, return_distance=False)[:, 1:]          # drop self
    base = rng.randint(0, len(idx), n_new)
    pick = neigh[base, rng.randint(0, neigh.shape[1], n_new)]
    lam = rng.rand(n_new)
    if sp.issparse(X):
        A, B = Xm[base], Xm[pick]
        new = A + sp.diags(lam) @ (B - A)
        return sp.vstack([X, new]).tocsr(), np.concatenate([y, np.full(n_new, minority)])
    new = Xm[base] + lam[:, None] * (Xm[pick] - Xm[base])
    return np.vstack([X, new]), np.concatenate([y, np.full(n_new, minority)])
