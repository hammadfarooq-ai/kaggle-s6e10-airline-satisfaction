# S6E10 stack6: v10's 246 members; LR famreg + LightGBM stacker (as v10) + MLP stackers, all nested on the same
# meta folds (seeds 7 and 13). MLP-A reads the 246 scaled logits; MLP-B reads logits + raw features.
# Blend weights are picked on the mean AUC of the two meta seeds.
# Each candidate group is screened with the nested family-wise LR on meta seeds 7 and 13; groups that raise the
# mean of both seeds are kept, then the LR + LightGBM blend is refit on the kept set.
import glob, json, os, time, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

INPUT = os.environ.get("INPUT_DIR", "/kaggle/input")
def find(pat, hint):
    hits = sorted(p for p in glob.glob(f"{INPUT}/**/{pat}", recursive=True) if hint in p)
    assert hits, (pat, hint)
    return hits

tr = pd.read_csv(find("train.csv", "playground-series-s6e10")[0])
te = pd.read_csv(find("test.csv", "playground-series-s6e10")[0])
y = tr["satisfaction"].astype(int).values

def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-15, 1 - 1e-15)
    return np.clip(np.log(p / (1 - p)), -30, 30)
def tologit(v):
    v = np.asarray(v, dtype=np.float64)
    return v if (v.min() < 0 or v.max() > 1) else logit(v)

names, cols, tcols = [], [], []
def add(name, o, t):
    o, t = np.asarray(o).ravel(), np.asarray(t).ravel()
    assert len(o) == len(y) and len(t) == len(te) and not np.isnan(o).any(), name
    names.append(name); cols.append(tologit(o)); tcols.append(tologit(t))

def by_id(path, ids, col):
    return pd.read_csv(path).set_index("id").loc[ids, col].values

# ---- the 245 members of Part 10 (the loaders are unchanged; see that version for the commentary)
for d, tag in ((find("oof_gp_cat_ctr.npy", "124-public")[0], ""), (find("oof_tabfm.npy", "tfm-oof")[0], "tfm_")):
    d = os.path.dirname(d)
    for f in sorted(glob.glob(f"{d}/oof_*.npy")):
        add(tag + os.path.basename(f)[4:-4], np.load(f), np.load(f.replace("oof_", "test_")))
d = os.path.dirname(find("oof_o_tabpfn35.npy", "golem")[0])
for n in ("o_tabpfn35", "p_lgbm_auxev", "q_lgbm_auxev_10fold"):
    add("gl2_" + n, np.load(f"{d}/oof_{n}.npy"), np.load(f"{d}/test_{n}.npy"))
d = os.path.dirname(find("preds_e1_allcat_cat.npz", "artifacts")[0])
for f in sorted(glob.glob(f"{d}/preds_*.npz")) + sorted(glob.glob(f"{d}/orig_*_pred.npz")):
    z = np.load(f)
    if "oof" in z and "test" in z and len(z["oof"]) == len(y) and not np.isnan(z["oof"]).any():
        add("st_" + os.path.basename(f)[:-4].replace("preds_", ""), z["oof"], z["test"])
d = os.path.dirname(find("oof_forge_cat.npy", "forge")[0])
for f in sorted(glob.glob(f"{d}/oof_forge_*.npy")):
    add("am_" + os.path.basename(f)[4:-4], np.load(f), np.load(f.replace("oof_", "test_")))
d = os.path.dirname(find("tabpfn_catfd10_oof.csv", "oof-dataset")[0])
for m in ("cat_v3", "tabpfn_catfd10", "tabpfn_raw", "tabpfn_te_cnt", "xgb_v3"):
    add("ab_" + m, by_id(f"{d}/{m}_oof.csv", tr["id"], "pred"), by_id(f"{d}/{m}_test.csv", te["id"], "pred"))
d = os.path.dirname(find("train_ids_folds.csv", "route-tfm-nn-oof")[0])
for m in ("cat_w1", "lgb_w1", "rmlp_f_nc", "tabicl_fdcat_c250k", "tabm_bcd_rc_v2", "tabpfn_catfd10", "xgb_w1"):
    add("km_" + m, np.load(f"{d}/oof_{m}.npy"), np.load(f"{d}/test_{m}.npy"))
d = os.path.dirname(find("lgb_cpu_strong.npz", "stack-oof-predictions")[0])
for m in ("lgb_cpu_strong", "xgb_A_auxall"):
    z = np.load(f"{d}/{m}.npz"); add("s7_" + m, z["oof"], z["test"])
for m in ("rmlp_route_ogaux", "xgb_route_ogaux"):
    z = np.load(find(f"{m}.npz", "route-id-teacher")[0]); add("kr_" + m, z["oof"], z["test"])
d = os.path.dirname(find("oof_members.csv", "encodings-catboost-realmlp")[0])
for m in ("lgb", "xgb", "cat", "cat_allcat", "realmlp"):
    add("kg_" + m, by_id(f"{d}/oof_members.csv", tr["id"], m), by_id(f"{d}/test_members.csv", te["id"], m))
d = os.path.dirname(find("oof_preds.csv", "realmlp-pytabkit")[0])
add("yk_realmlp", by_id(f"{d}/oof_preds.csv", tr["id"], "satisfaction"), by_id(f"{d}/submission.csv", te["id"], "satisfaction"))
d = os.path.dirname(find("oof_train.npy", "single-catboost-eda")[0])
add("hm_cat", np.load(f"{d}/oof_train.npy"), np.load(f"{d}/oof_test.npy"))
d = os.path.dirname(find("oof_cb_cat.npy", "catboost-all-categorical")[0])
add("dk_cbcat", np.load(f"{d}/oof_cb_cat.npy"), np.load(f"{d}/test_cb_cat.npy"))
z = np.load(find("oof_artifacts.npz", "original-prior-stack-v5")[0])
for m in ("lgbm_champion", "lgbm_te", "lgbm_te_rows", "xgb_gpu", "catboost_gpu"):
    add("av5_" + m, z["oof_" + m], z["test_" + m])
d = os.path.dirname(find("members.csv", "goodpjw2008-extra-members")[0])
for m in pd.read_csv(f"{d}/members.csv")["member"]:
    add("gpx_" + m, np.load(f"{d}/oof_{m}.npy"), np.load(f"{d}/test_{m}.npy"))
d = os.path.dirname(find("members.csv", "ablation-members")[0])
for m in pd.read_csv(f"{d}/members.csv")["member"]:
    add("ka_" + m, np.load(f"{d}/oof_{m}.npy"), np.load(f"{d}/test_{m}.npy"))
d = os.path.dirname(find("members.csv", "goodpjw2008-extra-members")[0])
add("gpx_tabpfn_ne4", np.load(f"{d}/oof_tabpfn_ne4.npy"), np.load(f"{d}/test_tabpfn_ne4.npy"))
NBASE = len(names)
print("members", NBASE, flush=True)

Z, ZT = np.column_stack(cols), np.column_stack(tcols)
print(f"{Z.shape[1]} members x {Z.shape[0]:,} OOF rows")
names = list(names)

def family(n):
    if "tabpfn" in n or n.startswith("pfn_"): return "tabpfn"
    if n.startswith("tfm_") or "tabicl" in n or "tabm" in n: return "tfm"
    if "realmlp" in n or "rmlp" in n or "mlp" in n: return "mlp"
    return "gbdt"
FAM = np.array([family(n) for n in names])
print({f: int((FAM == f).sum()) for f in ("gbdt", "mlp", "tabpfn", "tfm")}, flush=True)
SCALE = np.array([{"tabpfn": 3.0, "tfm": 1.5}.get(f, 1.0) for f in FAM])
import lightgbm as lgb

def lr_nested(Zm, seed):
    skf = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(Zm, y))
    o = np.zeros(len(y))
    def one(a, b):
        return b, LogisticRegression(C=0.003, max_iter=5000).fit(Zm[a], y[a]).decision_function(Zm[b])
    for b, v in Parallel(n_jobs=min(5, os.cpu_count()))(delayed(one)(a, b) for a, b in skf):
        o[b] = v
    return o, skf

P = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=2000, feature_fraction=0.5,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=os.cpu_count(), seed=0)
def gb_nested(Zm, skf):
    o, its = np.zeros(len(y)), []
    for a, b in skf:
        ai = np.random.default_rng(0).permutation(a); cut = int(len(ai) * 0.9)
        m = lgb.train(P, lgb.Dataset(Zm[ai[:cut]], y[ai[:cut]]), 4000,
                      valid_sets=[lgb.Dataset(Zm[ai[cut:]], y[ai[cut:]])], callbacks=[lgb.early_stopping(100, verbose=False)])
        its.append(m.best_iteration); o[b] = m.predict(Zm[b], raw_score=True, num_iteration=m.best_iteration)
    return o, its

ZS, ZTS = Z * SCALE, ZT * SCALE

import torch, torch.nn as nn
torch.set_num_threads(os.cpu_count())
# raw features for MLP-B: numeric standardized (log1p for distance/delays), categoricals one-hot
RAWC = [c for c in tr.columns if c not in ("id", "satisfaction")]
allr = pd.concat([tr[RAWC], te[RAWC]], ignore_index=True)
allr["Arrival Delay in Minutes"] = allr["Arrival Delay in Minutes"].fillna(0)
for c in ("Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"):
    allr[c] = np.log1p(allr[c].clip(lower=0))
cat = ["Gender", "Customer Type", "Type of Travel", "Class"]
X = pd.get_dummies(allr, columns=cat, dtype=np.float32).astype(np.float32).values
X = (X - X[:len(tr)].mean(0)) / (X[:len(tr)].std(0) + 1e-6)
XR, XRT = X[:len(tr)], X[len(tr):]
mu, sdv = ZS.mean(0), ZS.std(0) + 1e-6
ZN, ZTN = ((ZS - mu) / sdv).astype(np.float32), ((ZTS - mu) / sdv).astype(np.float32)

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, 1)                      # linear path (like the LR stacker)
        self.net = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))
    def forward(self, x):
        return (self.lin(x) + self.net(x)).squeeze(-1)

def mlp_fit_predict(Xa, ya, Xb_list, seed=0, epochs=6):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Xa)); cut = int(len(idx) * 0.9); tr_i, va_i = idx[:cut], idx[cut:]
    m = MLP(Xa.shape[1]); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt, yt = torch.from_numpy(Xa), torch.from_numpy(ya.astype(np.float32))
    best, best_state, bad = -1, None, 0
    steps = epochs * (len(tr_i) // 2048); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps + 10)
    for ep in range(epochs):
        m.train(); perm = rng.permutation(tr_i)
        for i in range(0, len(perm) - 2047, 2048):
            b = perm[i:i + 2048]
            loss = nn.functional.binary_cross_entropy_with_logits(m(Xt[b]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        m.eval()
        with torch.no_grad():
            pv = m(Xt[va_i]).numpy()
        a = roc_auc_score(ya[va_i], pv)
        if a > best: best, best_state, bad = a, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else: bad += 1
        if bad >= 2: break
    m.load_state_dict(best_state); m.eval()
    with torch.no_grad():
        return [m(torch.from_numpy(Xb)).numpy() for Xb in Xb_list], ep + 1

def mlp_nested(Xm, skf):
    o, eps = np.zeros(len(y)), []
    for a, b in skf:
        (p,), e = mlp_fit_predict(Xm[a], y[a], [Xm[b]]); o[b] = p; eps.append(e)
    return o, eps

t0 = time.time()
XA, XB = ZN, np.hstack([ZN, XR])
XAT, XBT = ZTN, np.hstack([ZTN, XRT])
res, EP = {}, {"A": [], "B": []}
ITS = []
for seed in (7, 13):
    lo, skf = lr_nested(ZS, seed)
    go, its = gb_nested(Z, skf); ITS += its
    ma, ea = mlp_nested(XA, skf); EP["A"] += ea
    mb, eb = mlp_nested(XB, skf); EP["B"] += eb
    res[seed] = (lo, go, ma, mb)
    print(f"seed {seed}: LR {roc_auc_score(y, lo):.6f} LGB {roc_auc_score(y, go):.6f} MLP-A {roc_auc_score(y, ma):.6f} "
          f"MLP-B {roc_auc_score(y, mb):.6f} epochs A {ea} B {eb} ({time.time() - t0:.0f}s)", flush=True)

def z(v): return (v - v.mean()) / v.std()
grid = []
for wg in (0.0, 0.1, 0.2, 0.3):
    for wa in (0.0, 0.1, 0.2, 0.3, 0.4):
        for wb in (0.0, 0.1, 0.2, 0.3, 0.4):
            if wg + wa + wb > 0.8: continue
            r = [roc_auc_score(y, (1 - wg - wa - wb) * z(lo) + wg * z(go) + wa * z(ma) + wb * z(mb))
                 for lo, go, ma, mb in (res[7], res[13])]
            grid.append((np.mean(r), r[0], r[1], wg, wa, wb))
grid.sort(reverse=True)
ref = [g for g in grid if g[3] == 0.2 and g[4] == 0 and g[5] == 0][0]
print(f"v10 recipe (LR 0.8 + LGB 0.2): seed7 {ref[1]:.6f} seed13 {ref[2]:.6f} mean {ref[0]:.6f}")
for g in grid[:12]:
    print(f"  LGB {g[3]:.1f} MLP-A {g[4]:.1f} MLP-B {g[5]:.1f}: seed7 {g[1]:.6f} seed13 {g[2]:.6f} mean {g[0]:.6f} "
          f"gain {(g[0] - ref[0]) * 1e6:+.1f}e-6 ({g[1] > ref[1]} / {g[2] > ref[2]})", flush=True)
best = grid[0]; wg, wa, wb = best[3:]

# test: full-data fits
lr_test = LogisticRegression(C=0.003, max_iter=5000).fit(ZS, y).decision_function(ZTS)
lr_oof_ref = res[7][0]
gb_test = lgb.train(P, lgb.Dataset(Z, y), int(np.mean(ITS) * 1.1)).predict(ZT, raw_score=True)
(ta,), _ = mlp_fit_predict(XA, y, [XAT]); (tb,), _ = mlp_fit_predict(XB, y, [XBT])
# z-scaling of test parts uses the OOF mean/std of seed 7 so test and OOF blends live on the same scale
def zt(t, ref_oof): return (t - ref_oof.mean()) / ref_oof.std()
lo7, go7, ma7, mb7 = res[7]
ids = te["id"].values
def save(fn, v):
    pd.DataFrame({"id": ids, "satisfaction": 1 / (1 + np.exp(-v))}).to_csv(fn, index=False)
save("submission.csv", (1 - wg - wa - wb) * zt(lr_test, lo7) + wg * zt(gb_test, go7) + wa * zt(ta, ma7) + wb * zt(tb, mb7))
save("submission_v10recipe.csv", 0.8 * zt(lr_test, lo7) + 0.2 * zt(gb_test, go7))
np.save("test_parts.npy", np.column_stack([lr_test, gb_test, ta, tb]))
np.save("oof_parts_seed7.npy", np.column_stack(res[7])); np.save("oof_parts_seed13.npy", np.column_stack(res[13]))
print("chosen", best, f"done {time.time() - t0:.0f}s")
