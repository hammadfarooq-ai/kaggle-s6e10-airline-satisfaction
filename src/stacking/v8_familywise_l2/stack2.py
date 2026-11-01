# S6E10 stack v8: the 245-member public pool (loaders from goodpjw2008 Part 11), family-wise L2,
# nested CV on three meta seeds, then a rank average with the two published frozen stacks.
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

def _fold(Zm, a, b):
    return b, LogisticRegression(C=0.003, max_iter=5000).fit(Zm[a], y[a]).decision_function(Zm[b])
def nested(Zm, seed):
    skf = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(Zm, y))
    o = np.zeros(len(y))
    for b, v in Parallel(n_jobs=min(5, os.cpu_count()))(delayed(_fold)(Zm, a, b) for a, b in skf):
        o[b] = v
    return roc_auc_score(y, o), [roc_auc_score(y[b], o[b]) for _, b in skf]

t0 = time.time()
for seed in (7, 13):
    (s0, f0), (s1, f1) = nested(Z, seed), nested(Z * SCALE, seed)
    print(f"seed {seed:3d}  flat {s0:.6f}  family-reg {s1:.6f}  gain {(s1 - s0) * 1e6:+5.1f}e-6  "
          f"{sum(b > a for a, b in zip(f0, f1))}/5 folds up ({time.time() - t0:.0f}s)", flush=True)

def fit_full(Zm, ZTm):
    return LogisticRegression(C=0.003, max_iter=5000).fit(Zm, y).decision_function(ZTm)
flat_test = fit_full(Z, ZT)
fam_test = fit_full(Z * SCALE, ZT * SCALE)

r = lambda v: rankdata(v) / len(v)
ids = te["id"].values
def save(fn, v):
    pd.DataFrame({"id": ids, "satisfaction": v}).to_csv(fn, index=False)
save("submission.csv", 1 / (1 + np.exp(-fam_test)))
save("submission_flat.csv", 1 / (1 + np.exp(-flat_test)))

# published frozen stacks: goodpjw2008's 245 (LB 0.96186) and dariushafshar's 247 (LB 0.96187)
parts = {"mine_famreg": fam_test}
try:
    parts["goodpjw245"] = np.load(find("test_stack245.npy", "goodpjw2008-extra-members")[0]).ravel()
except Exception as e:
    print("goodpjw frozen stack missing:", e)
try:
    d = os.path.dirname(find("stack_oof.npy", "247-public-oofs")[0])
    parts["dariush247"] = by_id(f"{d}/submission.csv", te["id"], "satisfaction")
except Exception as e:
    print("dariush stack missing:", e)
R = {k: r(v) for k, v in parts.items()}
print(pd.DataFrame(R).corr(method="spearman").round(6).to_string())
save("submission_twin3.csv", np.mean(list(R.values()), axis=0))
if "dariush247" in R:
    save("submission_mine_dariush.csv", 0.5 * R["mine_famreg"] + 0.5 * R["dariush247"])
print(f"done {time.time() - t0:.0f}s")
