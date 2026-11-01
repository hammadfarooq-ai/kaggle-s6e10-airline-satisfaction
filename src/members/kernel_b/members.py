# S6E10 own stack members (GPU): TabM x2, XGBoost with nested target encoding, kNN.
# Folds: StratifiedKFold(5, shuffle=True, random_state=42), same as the public OOF libraries.
import os, sys, glob, time, subprocess
import numpy as np, pandas as pd

KAGGLE = os.path.exists('/kaggle/input')
DEBUG = os.environ.get('DBG') == '1'
if KAGGLE:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'tabm', 'rtdl_num_embeddings'])
import torch, torch.nn as nn
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold

T0 = time.time()
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
ONLY = 'tabm_fe,tabm_fe_s2'.split(',')
def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)

if KAGGLE:
    COMP = os.path.dirname(sorted(glob.glob('/kaggle/input/**/playground-series-s6e10/train.csv', recursive=True))[0])
    ORIG = os.path.dirname(sorted(glob.glob('/kaggle/input/**/airline-passenger-satisfaction/train.csv', recursive=True))[0])
else:
    COMP, ORIG = sys.argv[1], sys.argv[2]
tr, te = pd.read_csv(f'{COMP}/train.csv'), pd.read_csv(f'{COMP}/test.csv')
if DEBUG:
    tr, te = tr.sample(30000, random_state=0).reset_index(drop=True), te.iloc[:5000].copy()
og = pd.concat([pd.read_csv(f'{ORIG}/train.csv'), pd.read_csv(f'{ORIG}/test.csv')], ignore_index=True)
og['satisfaction'] = (og['satisfaction'] == 'satisfied').astype(int)
y = tr['satisfaction'].astype(int).values
NTR, NTE = len(tr), len(te)
log('train', tr.shape, 'test', te.shape, 'orig', og.shape, 'device', DEV)

RAT = ['Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking', 'Gate location',
       'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment', 'On-board service',
       'Leg room service', 'Baggage handling', 'Checkin service', 'Cleanliness']
CAT = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
NUM = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
RAW = NUM + RAT + CAT
FD = 'Flight Distance'

full = pd.concat([tr[RAW], te[RAW], og[RAW]], ignore_index=True)
full['arr_nan'] = full['Arrival Delay in Minutes'].isna().astype(np.int8)
full['Arrival Delay in Minutes'] = full['Arrival Delay in Minutes'].fillna(full['Departure Delay in Minutes'])
for c in CAT:
    full[c] = pd.factorize(full[c], sort=True)[0]
N2 = NTR + NTE
yo = og['satisfaction'].values
PRIOR = yo.mean()

def codes(cols):
    k = full[cols[0]].astype(np.int64).values
    for c in cols[1:]:
        v = pd.factorize(full[c])[0].astype(np.int64)
        k = k * (v.max() + 1) + v
    return pd.factorize(k)[0]

# ---- original-data lookups (no competition labels involved) ----
F = pd.DataFrame(index=range(N2))
OG_GROUPS = [[c] for c in RAW] + [
    ['Type of Travel', 'Class', 'Online boarding', 'Inflight entertainment'],
    ['Type of Travel', 'Class', 'Online boarding', 'Inflight wifi service'],
    ['Customer Type', 'Type of Travel', 'Online boarding', 'Inflight entertainment'],
    ['Type of Travel', 'Class', 'Online boarding', 'Inflight entertainment', 'Inflight wifi service'],
    ['Type of Travel', 'Class', 'Online boarding', 'Inflight entertainment', 'Seat comfort'],
    ['Type of Travel', 'Class', 'Online boarding', 'Inflight entertainment', 'Customer Type'],
    ['Type of Travel', 'Class', 'Customer Type', 'Inflight wifi service', 'Ease of Online booking'],
    ['Type of Travel', 'Class', 'Customer Type', 'Age'],
    ['Type of Travel', 'Class', FD],
]
for i, g in enumerate(OG_GROUPS):
    k = codes(g); K = k.max() + 1
    s = np.bincount(k[N2:], weights=yo, minlength=K); n = np.bincount(k[N2:], minlength=K)
    F[f'og{i}'] = ((s + 5 * PRIOR) / (n + 5))[k[:N2]].astype(np.float32)
    if len(g) > 1 or g[0] == FD:
        F[f'ogn{i}'] = np.log1p(n[k[:N2]]).astype(np.float32)

# teacher: XGBoost fitted on the original rows only
XR = full[RAW + ['arr_nan']].astype(np.float32).values
teacher = xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=7, subsample=0.8, colsample_bytree=0.6,
                            tree_method='hist', device=DEV, verbosity=0).fit(XR[N2:], yo)
F['og_teacher'] = teacher.predict(XR[:N2], output_margin=True).astype(np.float32)
log('teacher AUC on train', round(roc_auc_score(y, F['og_teacher'].values[:NTR]), 5))

# ---- route / count / digit features ----
base = full.iloc[:N2].reset_index(drop=True)
fdk = codes([FD])[:N2]
F['fd_cnt'] = np.log1p(np.bincount(fdk)[fdk]).astype(np.float32)
fd10 = pd.factorize(base[FD] // 10)[0]
F['fd10_cnt'] = np.log1p(np.bincount(fd10)[fd10]).astype(np.float32)
F['fd_mod10'] = (base[FD] % 10).astype(np.float32)
F['fd_mod100'] = (base[FD] % 100).astype(np.float32)
F['delay_diff'] = (base['Arrival Delay in Minutes'] - base['Departure Delay in Minutes']).astype(np.float32)
F['rat_mean'] = base[RAT].mean(1).astype(np.float32)
F['rat_std'] = base[RAT].std(1).astype(np.float32)
F['rat_zero'] = (base[RAT] == 0).sum(1).astype(np.float32)
F['rat_five'] = (base[RAT] == 5).sum(1).astype(np.float32)

# ---- nested target encoding on competition labels ----
TE_KEYS = [[FD], [FD, 'Class'], [FD, 'Type of Travel'], [FD, 'Customer Type'], ['Age', 'Type of Travel', 'Class'],
           ['Type of Travel', 'Class', 'Customer Type', 'Online boarding', 'Inflight wifi service'],
           ['Type of Travel', 'Class', 'Online boarding', 'Inflight entertainment', 'Seat comfort'],
           ['Type of Travel', 'Customer Type', 'Inflight wifi service', 'Ease of Online booking', 'Gate location'],
           ['Class', 'Inflight entertainment', 'Seat comfort', 'On-board service', 'Leg room service'],
           ['Type of Travel', 'Checkin service', 'Baggage handling', 'Cleanliness', 'Food and drink'],
           [FD // 1 if False else FD, 'Age']]
TE_CODES = [codes(g)[:N2] for g in TE_KEYS]
def te_stat(k, idx, m=20):
    K = k.max() + 1
    s = np.bincount(k[idx], weights=y[idx], minlength=K); n = np.bincount(k[idx], minlength=K)
    return (s + m * y[idx].mean()) / (n + m)
def te_features(trn):
    """TE columns for all N2 rows: rows in `trn` get inner-fold values, every other row gets the full-`trn` table."""
    out = np.zeros((N2, len(TE_CODES)), np.float32)
    for j, k in enumerate(TE_CODES):
        out[:, j] = te_stat(k, trn)[k]
        for a, b in KFold(5, shuffle=True, random_state=0).split(trn):
            out[trn[b], j] = te_stat(k, trn[a])[k[trn[b]]]
    return out

FOLDS = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(NTR), y))
Xnum_base = np.column_stack([base['Age'].values, np.log1p(base[FD].values),
                             np.log1p(base['Departure Delay in Minutes'].values),
                             np.log1p(base['Arrival Delay in Minutes'].values)]).astype(np.float32)
Xcat = base[RAT + CAT + ['arr_nan']].astype(np.int64).values
CARD = [int(Xcat[:, j].max()) + 1 for j in range(Xcat.shape[1])]
FE = F.values.astype(np.float32)
log('features', FE.shape, 'TE keys', len(TE_CODES))

def save(name, oof, test):
    np.savez_compressed(f'own_{name}.npz', oof=oof.astype(np.float32), test=test.astype(np.float32))
    log(f'### {name} OOF AUC {roc_auc_score(y, oof):.6f} saved')

# ================= TabM =================
def run_tabm(name, use_fe, seed=0, epochs=30, patience=4, bs=1024, lr=2e-3, wd=3e-4, k=32):
    import tabm, rtdl_num_embeddings as rne
    oof, test = np.zeros(NTR), np.zeros(NTE)
    for f, (a, b) in enumerate(FOLDS):
        torch.manual_seed(seed * 100 + f); np.random.seed(seed * 100 + f)
        Xn = np.column_stack([Xnum_base, FE, te_features(a)]) if use_fe else Xnum_base.copy()
        mu, sd = Xn[a].mean(0), Xn[a].std(0) + 1e-6
        Xn = (Xn - mu) / sd
        Xn_t, Xc_t = torch.tensor(Xn), torch.tensor(Xcat)
        yt = torch.tensor(y, dtype=torch.float32)
        bins = rne.compute_bins(Xn_t[a][:200000], n_bins=48)
        model = tabm.TabM.make(n_num_features=Xn.shape[1], cat_cardinalities=CARD, d_out=1, k=k,
                               num_embeddings=rne.PiecewiseLinearEmbeddings(bins, d_embedding=16, activation=False, version='B'),
                               ).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        ia, ib = torch.tensor(a), torch.tensor(b)
        def predict(idx):
            model.eval(); out = []
            with torch.no_grad():
                for i in range(0, len(idx), 8192):
                    j = idx[i:i + 8192]
                    with torch.autocast(DEV, enabled=DEV == 'cuda'):
                        p = model(Xn_t[j].to(DEV), Xc_t[j].to(DEV)).squeeze(-1)
                    out.append(torch.sigmoid(p.float()).mean(1).cpu())
            return torch.cat(out).numpy()
        scaler = torch.amp.GradScaler(enabled=DEV == 'cuda')
        best, bad, bo, bt = 0, 0, None, None
        for ep in range(epochs):
            model.train()
            perm = ia[torch.randperm(len(ia))]
            for i in range(0, len(perm) - bs // 2, bs):
                j = perm[i:i + bs]
                with torch.autocast(DEV, enabled=DEV == 'cuda'):
                    p = model(Xn_t[j].to(DEV), Xc_t[j].to(DEV)).squeeze(-1)          # (B, k)
                loss = nn.functional.binary_cross_entropy_with_logits(p.float(), yt[j].to(DEV)[:, None].expand(-1, p.shape[1]))
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            pv = predict(ib); s = roc_auc_score(y[b], pv)
            if s > best:
                best, bad, bo, bt = s, 0, pv, predict(torch.arange(NTR, N2))
            else:
                bad += 1
            log(f'{name} fold {f} ep {ep} val {s:.6f} best {best:.6f}')
            if bad >= patience:
                break
        oof[b] = bo; test += bt / len(FOLDS)
        del model, opt
        if DEV == 'cuda':
            torch.cuda.empty_cache()
    save(name, oof, test)

# ================= XGBoost =================
def run_xgb(name, seed=0):
    oof, test = np.zeros(NTR), np.zeros(NTE)
    Xb = np.column_stack([XR[:N2], FE])
    P = dict(objective='binary:logistic', eval_metric='auc', tree_method='hist', device=DEV, learning_rate=0.03,
             max_depth=8, min_child_weight=8, subsample=0.8, colsample_bytree=0.4, reg_lambda=3.0, max_bin=256, seed=seed)
    for f, (a, b) in enumerate(FOLDS):
        X = np.column_stack([Xb, te_features(a)])
        dt = xgb.DMatrix(X[a], y[a]); dv = xgb.DMatrix(X[b], y[b])
        m = xgb.train(P, dt, 200 if DEBUG else 6000, [(dv, 'v')], early_stopping_rounds=200, verbose_eval=False)
        it = (0, m.best_iteration + 1)
        oof[b] = m.predict(dv, iteration_range=it)
        test += m.predict(xgb.DMatrix(X[NTR:]), iteration_range=it) / len(FOLDS)
        log(f'{name} fold {f} val {roc_auc_score(y[b], oof[b]):.6f} it {m.best_iteration}')
    save(name, oof, test)

# ================= kNN (cuML) =================
def run_knn(name, k=300):
    from cuml.neighbors import NearestNeighbors
    R = base[RAT].values.astype(np.float32)
    oh = np.column_stack([(base[c].values == v).astype(np.float32) * 3.0 for c in CAT for v in sorted(base[c].unique())])
    X = np.column_stack([R, oh, (base['Age'].values / 15).astype(np.float32),
                         (np.log1p(base[FD].values) * 1.5).astype(np.float32), FE[:, F.columns.get_loc('og_teacher')] * 0.5])
    X = np.ascontiguousarray(X, dtype=np.float32)
    oof, test = np.zeros(NTR), np.zeros(NTE)
    def query(nn_, Q, ya):
        out = []
        for i in range(0, len(Q), 20000):
            d, ix = nn_.kneighbors(Q[i:i + 20000])
            d, ix = np.asarray(d), np.asarray(ix)
            w = 1.0 / (d + 1.0)
            out.append((w * ya[ix]).sum(1) / w.sum(1))
        return np.concatenate(out)
    for f, (a, b) in enumerate(FOLDS):
        nn_ = NearestNeighbors(n_neighbors=k).fit(X[a])
        oof[b] = query(nn_, X[b], y[a].astype(np.float32))
        test += query(nn_, X[NTR:], y[a].astype(np.float32)) / len(FOLDS)
        log(f'{name} fold {f} val {roc_auc_score(y[b], oof[b]):.6f}')
    oof, test = np.clip(oof, 1e-4, 1 - 1e-4), np.clip(test, 1e-4, 1 - 1e-4)
    save(name, oof, test)

JOBS = [('xgb_te', lambda: run_xgb('xgb_te')),
        ('tabm_fe', lambda: run_tabm('tabm_fe', True)),
        ('tabm_raw', lambda: run_tabm('tabm_raw', False, seed=1)),
        ('knn', lambda: run_knn('knn')),
        ('tabm_fe_s2', lambda: run_tabm('tabm_fe_s2', True, seed=2, k=32, lr=1e-3, bs=512))]
for nm, job in JOBS:
    if ONLY and nm not in ONLY:
        continue
    if time.time() - T0 > 9.5 * 3600:
        log('time budget reached, skipping', nm); continue
    try:
        job()
    except Exception as e:
        import traceback; traceback.print_exc()
        log(f'FAILED {nm}: {type(e).__name__}: {e}')
log('done')
