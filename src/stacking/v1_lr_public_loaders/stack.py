# S6E10 stack: public OOF libraries (+ own members when attached) -> nested LR on logits.
# Library loaders follow matterhorn3838's public-OOF stack (built on kratosyan's loader structure).
import glob, os, time
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

T0 = time.time()
def find_dir(slug):
    hits = [p for p in glob.glob(f'/kaggle/input/**/{slug}', recursive=True) if os.path.isdir(p)]
    assert hits, f'input not attached: {slug}'
    return sorted(hits, key=len)[0]

COMP = find_dir('playground-series-s6e10')
tr, te = pd.read_csv(f'{COMP}/train.csv'), pd.read_csv(f'{COMP}/test.csv')
y = tr['satisfaction'].astype(int).values
NTR = len(y)
OOF, TEST, SRC = {}, {}, {}

def add(name, o, t, src):
    o, t = np.asarray(o, float), np.asarray(t, float)
    assert o.shape == (NTR,) and t.shape == (len(te),), (name, o.shape, t.shape)
    assert np.isfinite(o).all() and np.isfinite(t).all(), name
    OOF[name], TEST[name], SRC[name] = o, t, src

def by_id(df, ids, col):
    return df.set_index('id')[col].loc[ids].values

def l_gp():
    d = find_dir('s6e10-tabpfn-route-categories-lb-0-96160')
    go, gt = pd.read_csv(f'{d}/oof_members.csv'), pd.read_csv(f'{d}/test_members.csv')
    assert (go['id'].values == tr['id'].values).all() and (gt['id'].values == te['id'].values).all()
    for c in go.columns[1:]:
        add('gp_' + c, go[c].values, gt[c].values, 'goodpjw P8')
def l_mg():
    d = find_dir('s6e10-oof-library')
    mo, mt = pd.read_parquet(f'{d}/oof_predictions.parquet'), pd.read_parquet(f'{d}/test_predictions.parquet')
    assert (mo['id'].values == tr['id'].values).all() and (mt['id'].values == te['id'].values).all()
    groups = {}
    for c in mo.columns:
        if c in ('id', 'fold', 'satisfaction'):
            continue
        groups.setdefault(c.rsplit('_s', 1)[0] if c.startswith('realmlp') else c, []).append(c)
    for g, cols in groups.items():
        add('mg_' + g, mo[cols].mean(1).values, mt[cols].mean(1).values, 'megayak')
def l_gl():
    d = find_dir('s6e10-golem-oof-library')
    for f in sorted(glob.glob(f'{d}/oof_*.npy')):
        k = os.path.basename(f)[4:-4]
        add('gl_' + k, np.load(f), np.load(f'{d}/test_{k}.npy'), 'golem')
def l_ar():
    d = find_dir('s6e10-oof-test-predictions')
    ao, at = pd.read_parquet(f'{d}/oof_predictions.parquet'), pd.read_parquet(f'{d}/test_predictions.parquet')
    assert (ao['id'].values == tr['id'].values).all() and (at['id'].values == te['id'].values).all()
    for c in ['lgb_5f', 'xgb_5f', 'cat_5f', 'realmlp_5f']:
        add('ar_' + c, ao[c].values, at[c].values, 'arhancanli12')
def l_sa():
    d = find_dir('s6e10-stack-oof-predictions')
    for f in sorted(glob.glob(f'{d}/*.npz')):
        z = np.load(f)
        add('sa_' + os.path.basename(f)[:-4], z['oof'], z['test'], 'sachith7')
def l_nj():
    d = find_dir('s6e10-oof')
    add('nj_01', by_id(pd.read_csv(f'{d}/01_blend_oof.csv'), tr['id'], 'satisfaction'),
        by_id(pd.read_csv(f'{d}/01_submission.csv'), te['id'], 'satisfaction'), 'najiama')
def l_pfn():
    for nm, slug in [('pfn_samanyu_route_og', 's6e10-tabpfn-route-og'), ('pfn_samanyu_seeds', 's6e10-tabpfn-seeds'),
                     ('pfn_hemingweb_raw', 's6e10-exp01-tabpfn35-full-context')]:
        d = find_dir(slug)
        add(nm, by_id(pd.read_csv(f'{d}/oof_exp01_tabpfn35.csv'), tr['id'], 'pred'),
            by_id(pd.read_csv(f'{d}/test_exp01_tabpfn35.csv'), te['id'], 'satisfaction'), 'TabPFN outputs')
def l_mi():
    d = find_dir('s6e10-analysis-first-equality-blocks-stack')
    mo2 = pd.read_parquet(f'{d}/oof_predictions.parquet').set_index('id').loc[tr['id']]
    mt2 = pd.read_parquet(f'{d}/test_predictions.parquet').set_index('id').loc[te['id']]
    for c in mo2.columns:
        add('mi_' + c, mo2[c].values, mt2[c].values, 'mitudru')
def l_wx():
    for slug, c in [('s6e10-catboost-rating-crosses-cv-0-96080', 'rating_segments'),
                    ('s6e10-lightgbm-context-stats-cv-0-96065', 'context_numeric')]:
        d = find_dir(slug)
        add('wx_' + c, by_id(pd.read_parquet(f'{d}/oof_predictions.parquet'), tr['id'], c),
            by_id(pd.read_parquet(f'{d}/test_predictions.parquet'), te['id'], c), 'wangxintong111')
def l_th():
    for slug, tag in [('s6e10-xgb-realmlp-catboost-lb-0-96123', 'thisray'), ('s6e10-honest-oof-stack', 'megayak honest')]:
        d = find_dir(slug)
        o = pd.read_csv(f'{d}/oof_members.csv').set_index('id').loc[tr['id']]
        t = pd.read_csv(f'{d}/test_members.csv').set_index('id').loc[te['id']]
        for c in t.columns:
            add(f'{tag.split()[0][:2]}{"h" if "honest" in tag else ""}_{c}', o[c].values, t[c].values, tag)
def l_bp():
    d = find_dir('s6e10-busyaprime-five-members-oof')
    for f in sorted(glob.glob(f'{d}/bp_*.npz')):
        z = np.load(f)
        add(os.path.basename(f)[:-4], z['oof'], z['test'], 'busyaprime')
def l_ab():
    d = find_dir('oof-dataset')
    for m in ['tabpfn_catfd10', 'tabpfn_raw', 'tabpfn_te_cnt', 'cat_v3', 'xgb_v3']:
        add('ab_' + m, by_id(pd.read_csv(f'{d}/{m}_oof.csv'), tr['id'], 'pred'),
            by_id(pd.read_csv(f'{d}/{m}_test.csv'), te['id'], 'pred'), 'abdullah ' + ('pfn' if 'tabpfn' in m else 'gbdt'))
def l_kx():
    d = find_dir('s6e10-xgboost-realmlp-route-id-teacher')
    for f in sorted(glob.glob(f'{d}/**/*_route_ogaux.npz', recursive=True)):
        z = np.load(f)
        add('kx_' + os.path.basename(f)[:-4], z['oof'], z['test'], 'kratosyan10f')
def l_hg():
    d = find_dir('single-catboost-eda-ps6e10')
    add('hg_cat', np.load(glob.glob(f'{d}/**/oof_train.npy', recursive=True)[0]),
        np.load(glob.glob(f'{d}/**/oof_test.npy', recursive=True)[0]), 'hermengardo')
def l_dp():
    d = find_dir('s6e10-numeric-keys-catboost-ctr')
    o = pd.read_csv(glob.glob(f'{d}/**/oof_predictions.csv', recursive=True)[0])
    tp = np.zeros(len(te)); n = 0
    for f in sorted(glob.glob(f'{d}/**/catboost_numeric_ctr_fold*.npz', recursive=True)):
        z = np.load(f)
        assert (z['test_ids'] == te['id'].values).all()
        tp += z['test_pred']; n += 1
    assert n == 5, n
    add('dp_catctr', by_id(o, tr['id'], 'catboost_numeric_ctr'), tp / n, 'denpugovkin')
def l_yk():
    d = find_dir('ps-s6-e10-realmlp-pytabkit')
    add('yk_realmlp', by_id(pd.read_csv(f'{d}/oof_preds.csv'), tr['id'], 'satisfaction'),
        by_id(pd.read_csv(f'{d}/submission.csv'), te['id'], 'satisfaction'), 'yekenot')
def l_st():
    d = find_dir('ps-s6e10-airline-satisfaction-artifacts')
    for f in sorted(glob.glob(f'{d}/**/preds_*.npz', recursive=True)):
        z = np.load(f)
        add('st_' + os.path.basename(f)[6:-4], z['oof'], z['test'], 'sadamtorres')
    for f in ['orig_cat_pred', 'orig_model_pred']:
        z = np.load(glob.glob(f'{d}/**/{f}.npz', recursive=True)[0])
        add('st_' + f, z['train'], z['test'], 'orig teacher')
def l_am():
    d = find_dir('s6e10-the-original-prior-stack-v5')
    z = np.load(glob.glob(f'{d}/**/oof_artifacts.npz', recursive=True)[0])
    for k in ['lgbm_champion', 'lgbm_te', 'lgbm_te_rows', 'xgb_gpu', 'catboost_gpu']:
        add('am_' + k, z['oof_' + k], z['test_' + k], 'amanatar')
def l_kk():
    d = find_dir('s6e10-encodings-catboost-realmlp-lb-0-961')
    o, t = pd.read_csv(f'{d}/oof_members.csv'), pd.read_csv(f'{d}/test_members.csv')
    assert (o['id'].values == tr['id'].values).all() and (t['id'].values == te['id'].values).all()
    for c in o.columns[1:]:
        add('kk_' + c, o[c].values, t[c].values, 'kagankoral')
def l_stacklevel():
    d = find_dir('s6e10-route-or-distance-both-lb-0-96169')
    z = np.load(glob.glob(f'{d}/**/oof_stacks.npz', recursive=True)[0])
    add('bp_stack53', z['oof53'], z['test53'], 'stacklevel')
    d = find_dir('s6e10-oof')
    for k in ['02', '03']:
        add('nj_' + k, by_id(pd.read_csv(f'{d}/{k}_blend_oof.csv'), tr['id'], 'satisfaction'),
            by_id(pd.read_csv(f'{d}/{k}_submission.csv'), te['id'], 'satisfaction'), 'stacklevel')
def l_cd():
    d = find_dir('s6e10-tfm-oof-and-test-predictions')
    for f in sorted(glob.glob(f'{d}/**/oof_*.npy', recursive=True)):
        m = os.path.basename(f)[4:-4]
        add('cd_' + m, np.load(f), np.load(f.replace('oof_', 'test_')), 'cdeotte tfm')
def l_km():   # koumeimaki: TabPFN / TabICL / TabM / NN / GBDT on shared folds
    d = find_dir('s6e10-route-tfm-nn-oof')
    ti = pd.read_csv(f'{d}/train_ids_folds.csv'); si = pd.read_csv(f'{d}/test_ids.csv')
    assert (ti['id'].values == tr['id'].values).all() and (si['id'].values == te['id'].values).all()
    for f in sorted(glob.glob(f'{d}/oof_*.npy')):
        k = os.path.basename(f)[4:-4]
        add('km_' + k, np.load(f), np.load(f'{d}/test_{k}.npy'), 'extra')
def l_fg():   # amanatar forge pairs
    d = find_dir('s6e10-forge-own-pairs')
    for f in sorted(glob.glob(f'{d}/oof_*.npy')):
        k = os.path.basename(f)[4:-4]
        add('fg_' + k, np.load(f), np.load(f'{d}/test_{k}.npy'), 'extra')
def l_dk():   # dmitriykolmikov: CatBoost with every column categorical
    d = find_dir('s6e10-catboost-all-categorical-oof-and-test')
    add('dk_cb_cat', np.load(f'{d}/oof_cb_cat.npy'), np.load(f'{d}/test_cb_cat.npy'), 'extra')
def l_own():
    # own members: any attached kernel output named s6e10-own-*, files own_<name>.npz with oof/test
    hits = [p for p in glob.glob('/kaggle/input/**/s6e10-own-*', recursive=True) if os.path.isdir(p)]
    n = 0
    for d in hits:
        for f in sorted(glob.glob(f'{d}/**/own_*.npz', recursive=True)):
            z = np.load(f)
            add(os.path.basename(f)[:-4], z['oof'], z['test'], 'own'); n += 1
    assert n, 'no own members attached'

SKIPPED = []
for fn in [l_gp, l_mg, l_gl, l_ar, l_sa, l_nj, l_pfn, l_mi, l_wx, l_th, l_bp, l_ab, l_kx, l_hg, l_dp, l_yk,
           l_st, l_am, l_kk, l_stacklevel, l_cd, l_km, l_dk]:
    before = dict(OOF)
    try:
        fn()
    except Exception as e:
        for k in set(OOF) - set(before):
            OOF.pop(k); TEST.pop(k); SRC.pop(k)
        SKIPPED.append(fn.__name__)
        print(f'skipped {fn.__name__}: {type(e).__name__}: {e}')
names = list(OOF)
print(len(names), 'members |', pd.Series(SRC).value_counts().to_dict())
print(f'loaded in {time.time() - T0:.0f}s; skipped libraries: {SKIPPED}', flush=True)

def logit(p):
    # members stored as logits are kept as they are
    if p.min() < 0 or p.max() > 1:
        return np.clip(p, -14, 14)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

Z = np.column_stack([logit(OOF[n]) for n in names]).astype(np.float32)
ZT = np.column_stack([np.clip(TEST[n], -14, 14) if (OOF[n].min() < 0 or OOF[n].max() > 1) else logit(TEST[n])
                      for n in names]).astype(np.float32)
single = pd.Series({n: roc_auc_score(y, Z[:, i]) for i, n in enumerate(names)}).sort_values(ascending=False)
print(single.head(15).round(6).to_string())
own = [n for n in names if SRC[n] == 'own']
if own:
    print('own members:\n' + single[own].round(6).to_string(), flush=True)

OOF.clear(); TEST.clear()   # logits are in Z / ZT; free the float64 copies
import gc; gc.collect()
META = list(StratifiedKFold(5, shuffle=True, random_state=7).split(Z, y))
def report(tag, o):
    s = roc_auc_score(y, o)
    print(f'{tag:34s} nested CV {s:.6f}  folds ' + ' '.join(f'{roc_auc_score(y[b], o[b]):.5f}' for _, b in META), flush=True)
    return s

def lr_stack(cols, C=0.003, test=True):
    o = np.zeros(NTR)
    for a, b in META:
        o[b] = LogisticRegression(C=C, max_iter=3000).fit(Z[a][:, cols], y[a]).decision_function(Z[b][:, cols])
    t = LogisticRegression(C=C, max_iter=3000).fit(Z[:, cols], y).decision_function(ZT[:, cols]) if test else None
    return o, t

RES = {}
pub = [i for i, n in enumerate(names) if SRC[n] != 'extra']
allc = list(range(len(names)))
ex = [n for n in names if SRC[n] == 'extra']
print('extra members:\n' + single[ex].round(6).to_string(), flush=True)
o, t = lr_stack(pub); RES['lr_public'] = (report(f'LR public ({len(pub)})', o), o, t)
if ex:
    km = pub + [i for i, n in enumerate(names) if n.startswith('km_')]
    o, t = lr_stack(km); RES['lr_km'] = (report(f'LR public+koumeimaki ({len(km)})', o), o, t)
    if len(allc) > len(km):
        o, t = lr_stack(allc); RES['lr_all'] = (report(f'LR public+koumeimaki+cb_cat ({len(allc)})', o), o, t)

# LightGBM stacker on the same logits, blended with the best LR
best_lr = max(RES, key=lambda k: RES[k][0])
lr_oof, lr_test = RES[best_lr][1], RES[best_lr][2]
cols = {'lr_all': allc, 'lr_km': km if ex else pub, 'lr_public': pub}[best_lr]
P = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=2000, feature_fraction=0.5,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=4, seed=0)
A, B = Z[:, cols], ZT[:, cols]
gb_oof, its = np.zeros(NTR), []
for a, b in META:
    ai = np.random.default_rng(0).permutation(a); cut = int(len(ai) * 0.9)
    m = lgb.train(P, lgb.Dataset(A[ai[:cut]], y[ai[:cut]]), 3000, valid_sets=[lgb.Dataset(A[ai[cut:]], y[ai[cut:]])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    its.append(m.best_iteration); gb_oof[b] = m.predict(A[b], raw_score=True, num_iteration=m.best_iteration)
gb_test = lgb.train(P, lgb.Dataset(A, y), int(np.mean(its) * 1.1)).predict(B, raw_score=True)
report('LightGBM stacker', gb_oof)
blend = {w: report(f'LR*{1 - w:.1f} + LGB*{w:.1f}', (1 - w) * lr_oof + w * gb_oof) for w in [0.1, 0.2, 0.3, 0.4, 0.5]}

def save(fname, z):
    pd.DataFrame({'id': te['id'], 'satisfaction': 1 / (1 + np.exp(-z))}).to_csv(fname, index=False)
save('submission.csv', lr_test)                       # plain LR, selected by nested CV only
save('submission_lr_lgb20.csv', 0.8 * lr_test + 0.2 * gb_test)
for k in RES:
    save(f'submission_{k}.csv', RES[k][2])
np.savez_compressed('stack_preds.npz', lr_oof=lr_oof.astype(np.float32), lr_test=lr_test.astype(np.float32),
                    gb_oof=gb_oof.astype(np.float32), gb_test=gb_test.astype(np.float32))
print('submitted variant:', best_lr, f'| total {time.time() - T0:.0f}s')
