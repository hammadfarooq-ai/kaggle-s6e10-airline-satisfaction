# S6E10: TabICL v2 on raw features + Chris Deotte's 9 engineered features (Flight Distance group stats,
# count and target encoding, original-data teacher). Folds: StratifiedKFold(5, shuffle=True, random_state=42).
# Each fold writes own_tabicl_fe_f{k}.npz (val idx + val preds); RUN_TEST also writes the test prediction.
import os, sys, glob, time, subprocess
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import numpy as np, pandas as pd

FOLDS_TO_RUN = [0, 1, 2]
RUN_TEST = False
CTX = 200_000          # context rows sampled from the training part of each fold
N_EST = 4
TIME_LIMIT = 11.3 * 3600

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'tabicl'])
import torch, xgboost as xgb
from tabicl import TabICLClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold

T0 = time.time()
def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)

COMP = os.path.dirname(sorted(glob.glob('/kaggle/input/**/playground-series-s6e10/train.csv', recursive=True))[0])
ORIG = os.path.dirname(sorted(glob.glob('/kaggle/input/**/airline-passenger-satisfaction/train.csv', recursive=True))[0])
tr, te = pd.read_csv(f'{COMP}/train.csv'), pd.read_csv(f'{COMP}/test.csv')
og = pd.concat([pd.read_csv(f'{ORIG}/train.csv'), pd.read_csv(f'{ORIG}/test.csv')], ignore_index=True)
yo = (og['satisfaction'] == 'satisfied').astype(int).values
y = tr['satisfaction'].astype(int).values
NTR, NTE = len(tr), len(te)
FD = 'Flight Distance'
CAT = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
RAW = [c for c in tr.columns if c not in ('id', 'satisfaction')]

full = pd.concat([tr[RAW], te[RAW], og[RAW]], ignore_index=True)
for c in CAT:
    full[c] = pd.factorize(full[c], sort=True)[0]
N2 = NTR + NTE
base = full.iloc[:N2].reset_index(drop=True)

# label-free engineered features (train+test rows)
F = base.copy()
for c in ['Class', 'Type of Travel', 'Customer Type', 'Age', 'Online boarding', 'Inflight wifi service']:
    F[f'fd_mean_{c}'] = base.groupby(FD)[c].transform('mean').astype(np.float32)
F['fd_count'] = base.groupby(FD)[FD].transform('size').astype(np.float32)
XR = full.astype(np.float32).values
teacher = xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=7, subsample=0.8, colsample_bytree=0.6,
                            tree_method='hist', device='cuda', verbosity=0).fit(XR[N2:], yo)
F['og_teacher'] = teacher.predict_proba(XR[:N2])[:, 1].astype(np.float32)
fdk = pd.factorize(base[FD])[0]
X_all = F.values.astype(np.float32)
log('features', X_all.shape, 'teacher AUC', round(roc_auc_score(y, F['og_teacher'].values[:NTR]), 5))

def te_stat(idx, m=20):
    K = fdk.max() + 1
    s = np.bincount(fdk[idx], weights=y[idx], minlength=K); n = np.bincount(fdk[idx], minlength=K)
    return (s + m * y[idx].mean()) / (n + m)
def with_te(ctx, query):
    """FD target encoding: context rows cross-fitted inside the context, query rows from the whole context."""
    te_ctx = np.zeros(len(ctx), np.float32)
    for a, b in KFold(5, shuffle=True, random_state=0).split(ctx):
        te_ctx[b] = te_stat(ctx[a])[fdk[ctx[b]]]
    te_q = te_stat(ctx)[fdk[query]].astype(np.float32)
    return np.column_stack([X_all[ctx], te_ctx]), np.column_stack([X_all[query], te_q])

def predict(train_idx, query, seed):
    rng = np.random.default_rng(seed)
    pos, neg = train_idx[y[train_idx] == 1], train_idx[y[train_idx] == 0]
    k = min(CTX, len(train_idx)); kp = int(round(k * len(pos) / len(train_idx)))
    ctx = np.concatenate([rng.choice(pos, kp, replace=False), rng.choice(neg, k - kp, replace=False)])
    Xc, Xq = with_te(ctx, query)
    # kv_cache=True ran out of memory on a 16 GB T4 with a 200k context; estimators run one at a time
    clf = TabICLClassifier(n_estimators=N_EST, batch_size=1, device='cuda', random_state=seed, kv_cache=False, verbose=False)
    clf.fit(Xc, y[ctx])
    out = np.zeros(len(query))
    step = 50000
    for i in range(0, len(query), step):
        out[i:i + step] = clf.predict_proba(Xq[i:i + step])[:, 1]
        log(f'  {i + step}/{len(query)} queries')
    del clf; torch.cuda.empty_cache()
    return out

FOLDS = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(NTR), y))
fold_sec = 0.0                      # duration of the last fold, used to skip a fold that cannot finish
for f in FOLDS_TO_RUN:
    if time.time() - T0 + 1.1 * fold_sec > TIME_LIMIT:
        log('time limit, stop before fold', f); break
    a, b = FOLDS[f]
    t = time.time()
    p = predict(a, b, seed=f)
    fold_sec = time.time() - t
    np.savez_compressed(f'own_tabicl_fe_f{f}.npz', idx=b, pred=p.astype(np.float32))
    log(f'### fold {f} AUC {roc_auc_score(y[b], p):.6f}')
if RUN_TEST and time.time() - T0 + 2.3 * fold_sec < TIME_LIMIT:
    p = predict(np.arange(NTR), np.arange(NTR, N2), seed=99)
    np.savez_compressed('own_tabicl_fe_test.npz', pred=p.astype(np.float32))
    log('### test saved')
log('done')
