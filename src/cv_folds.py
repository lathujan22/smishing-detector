"""Per-fold 10-fold grouped CV table for the baselines.   python -m src.cv_folds  -> results/CV_RESULTS.md, cv_folds.csv, cv_summary.csv"""
import sys, time, numpy as np, pandas as pd
sys.path.insert(0, '.')
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from src.baselines import models, cv_groups, SEED
from src.evaluate import to_binary
from sklearn.metrics import make_scorer, confusion_matrix
def fpr(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel(); return fp / (fp + tn)
tr = pd.read_csv('data/processed/train_feat.csv')
X, y, g = tr.text_model.astype(str), to_binary(tr.label3), cv_groups(tr)
cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
folds = list(cv.split(X, y, g))
rows = []
for name, pipe in models().items():
    t = time.time()
    r = cross_validate(pipe, X, y, groups=g, cv=folds, scoring={'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall',
                       'f1': 'f1', 'roc_auc': 'roc_auc', 'fpr': make_scorer(fpr)})
    for i, (tri, vai) in enumerate(folds):
        rows.append({'model': name, 'fold': i + 1, 'n_val': len(vai), 'pos_val': int(y[vai].sum()),
                     **{k: round(r[f'test_{k}'][i] * 100, 2) for k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'fpr']}})
    print(name, 'done', round(time.time() - t), 's', flush=True)
df = pd.DataFrame(rows); df.to_csv('results/cv_folds.csv', index=False)
M = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'fpr']
summ = df.groupby('model')[M].agg(['mean', 'std']).round(2)
summ.columns = [f'{a}_{b}' for a, b in summ.columns]
summ = summ.reindex(['NaiveBayes', 'SVM', 'RandomForest']); summ.to_csv('results/cv_summary.csv')
md = ['# 10-fold grouped stratified cross-validation (training set, 9,447 msgs)\n',
      'StratifiedGroupKFold, seed 42 - all variants of a Sinhala/Tamil pattern, and each translation with its English source, stay in the same fold. Values in %. fpr = false-alarm rate.\n',
      '## Summary (mean ± std over 10 folds)\n']
t = pd.DataFrame({'Model': summ.index, **{m.upper() if m == 'fpr' else m.capitalize(): [f"{summ.loc[i, m+'_mean']:.2f} ± {summ.loc[i, m+'_std']:.2f}" for i in summ.index] for m in M}})
md.append(t.to_markdown(index=False))
for name in ['NaiveBayes', 'SVM', 'RandomForest']:
    d = df[df.model == name].drop(columns='model')
    d = pd.concat([d, pd.DataFrame([{'fold': 'mean', **d[M].mean().round(2)}, {'fold': 'std', **d[M].std().round(2)}])])
    md += [f'\n## {name} - per fold\n', d.to_markdown(index=False)]
open('results/CV_RESULTS.md', 'w', encoding='utf-8').write('\n'.join(md))
print('\n'.join(md))
