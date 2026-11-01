# S6E10 stack v10: the stack v9 pool (245 members, family-wise LR + LightGBM stacker) plus candidate members.
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
NBASE = len(names)

# ---- candidate members (each group is screened against the 245 base)
GROUPS = {}
def cand(group, name, o, t):
    before = len(names)
    try:
        add(name, o, t)
    except Exception as e:
        print("skip", name, repr(e)[:120]); return
    GROUPS.setdefault(group, []).append(before)
d = os.path.dirname(find("members.csv", "goodpjw2008-extra-members")[0])
if os.path.exists(f"{d}/oof_tabpfn_ne4.npy"):
    cand("gpx_tabpfn_ne4", "gpx_tabpfn_ne4", np.load(f"{d}/oof_tabpfn_ne4.npy"), np.load(f"{d}/test_tabpfn_ne4.npy"))
hits = [p for p in glob.glob(f"{INPUT}/**/07_XGBoost_oof.csv", recursive=True)]
if hits:
    d = os.path.dirname(hits[0])
    try:
        ts = pd.read_csv(f"{d}/07_submission.csv")
        cand("naj_xgb07", "naj_xgb07", by_id(f"{d}/07_XGBoost_oof.csv", tr["id"], "oof_pred"),
             ts.set_index("id").loc[te["id"]].iloc[:, 0].values)
    except Exception as e:
        print("skip naj_xgb07", repr(e)[:200])
# own TabICL v2 + FE (kernels s6e10-tabicl-fe-a / -b): per-fold files, assembled into one OOF/test pair
fold_files = {}
for p in glob.glob(f"{INPUT}/**/own_tabicl_fe_f*.npz", recursive=True):
    fold_files[os.path.basename(p)] = p
test_hits = glob.glob(f"{INPUT}/**/own_tabicl_fe_test.npz", recursive=True)
if len(fold_files) == 5 and test_hits:
    o = np.full(len(y), np.nan)
    for k in range(5):
        z = np.load(fold_files[f"own_tabicl_fe_f{k}.npz"]); o[z["idx"]] = z["pred"]
    print("own TabICL FE OOF AUC", round(roc_auc_score(y, o), 6) if not np.isnan(o).any() else "incomplete")
    cand("own_tabicl_fe", "own_tabicl_fe", o, np.load(test_hits[0])["pred"])
else:
    print("own TabICL FE not attached / incomplete:", sorted(fold_files), len(test_hits))
for g, ix in GROUPS.items():
    print(f"candidate {g}: " + ", ".join(f"{names[i]} OOF {roc_auc_score(y, cols[i]):.6f}" for i in ix))

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

t0 = time.time()
ZS, ZTS = Z * SCALE, ZT * SCALE
base_ix = list(range(NBASE))
def lr_both(ix):
    out = []
    for seed in (7, 13):
        o, _ = lr_nested(ZS[:, ix], seed); out.append(roc_auc_score(y, o))
    return out
ref = lr_both(base_ix)
print(f"LR famreg base {NBASE}: seed7 {ref[0]:.6f} seed13 {ref[1]:.6f} mean {np.mean(ref):.6f} ({time.time() - t0:.0f}s)", flush=True)
keep = []
for g, ix in GROUPS.items():
    r = lr_both(base_ix + ix)
    gain = np.mean(r) - np.mean(ref)
    print(f"  + {g:16s} seed7 {r[0]:.6f} seed13 {r[1]:.6f} gain {gain * 1e6:+.1f}e-6 "
          f"({r[0] - ref[0] > 0} / {r[1] - ref[1] > 0})", flush=True)
    if r[0] > ref[0] and r[1] > ref[1]:
        keep += ix
if len(keep) and len(GROUPS) > 1:
    r = lr_both(base_ix + keep)
    print(f"  kept {[names[i] for i in keep]}: seed7 {r[0]:.6f} seed13 {r[1]:.6f} gain {(np.mean(r) - np.mean(ref)) * 1e6:+.1f}e-6", flush=True)
sel = base_ix + keep
names = [names[i] for i in sel]
Z, ZT, ZS, ZTS = Z[:, sel], ZT[:, sel], ZS[:, sel], ZTS[:, sel]
print("final members:", len(names), flush=True)
W = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
table, ITS = {}, []
for seed in (7, 13):
    lo, skf = lr_nested(ZS, seed)
    go, its = gb_nested(Z, skf); ITS += its
    for w in W:
        table.setdefault(w, []).append(roc_auc_score(y, (1 - w) * lo + w * go))
    print(f"seed {seed}: LR famreg {roc_auc_score(y, lo):.6f}  LGB {roc_auc_score(y, go):.6f}  iters {its}  "
          f"({time.time() - t0:.0f}s)", flush=True)
for w in W:
    print(f"  LR*{1 - w:.2f} + LGB*{w:.2f}   seed7 {table[w][0]:.6f}  seed13 {table[w][1]:.6f}  mean {np.mean(table[w]):.6f}"
          f"  gain vs LR {(np.mean(table[w]) - np.mean(table[0.0])) * 1e6:+.1f}e-6", flush=True)
best_w = max(W, key=lambda w: np.mean(table[w]))
print("chosen weight (mean of two meta seeds):", best_w, flush=True)

lr_test = LogisticRegression(C=0.003, max_iter=5000).fit(ZS, y).decision_function(ZTS)
gb_test = lgb.train(P, lgb.Dataset(Z, y), int(np.mean(ITS) * 1.1)).predict(ZT, raw_score=True)
ids = te["id"].values
def save(fn, z):
    pd.DataFrame({"id": ids, "satisfaction": 1 / (1 + np.exp(-z))}).to_csv(fn, index=False)
save("submission.csv", (1 - best_w) * lr_test + best_w * gb_test)
save("submission_lr_famreg.csv", lr_test)
for w in (0.1, 0.2, 0.3):
    save(f"submission_lgb{int(w * 100):02d}.csv", (1 - w) * lr_test + w * gb_test)
print(f"done {time.time() - t0:.0f}s")
