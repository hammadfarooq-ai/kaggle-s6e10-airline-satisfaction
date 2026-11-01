# S6E10: Q-FT v2 style order-free transformer (after Chris Deotte, discussion 751334).
# Each of the 21 features is one token = column embedding + value embedding + periodic numeric embedding
# + 8 side features (value counts, original-data log-odds, Flight-Distance "route" profile).
# Two summary tokens (row statistics, original-data block) are added for fine-tuning.
# No positions are used, so the encoder reads the row as a set. Weights start random (Chris: pretrained ModernBERT
# scored the same as random init), then replaced-token-detection (RTD) pretraining on train + test + original
# features (no labels), then 5-fold fine-tuning on StratifiedKFold(5, shuffle=True, random_state=42).
# Outputs: qft2_f{k}.npz (val idx, val logits, test logits) per fold, oof_qft2.npy, test_qft2.npy (logits).
import os, glob, time, math
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

SMOKE = os.environ.get('SMOKE') == '1'
INPUT = os.environ.get('INPUT_DIR', '/kaggle/input')
OUT = os.environ.get('OUT_DIR', '/kaggle/working')
D, H, L, FF = (64, 4, 2, 128) if SMOKE else (768, 12, 12, 3072)
BATCH = 256 if SMOKE else 512
PRE_EPOCHS, FT_EPOCHS = (1, 1) if SMOKE else (3, 2)
LR_ENC, LR_TOK, LR_PRE = 5e-5, 1e-3, 2e-4
TIME_LIMIT = 11.2 * 3600
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
NGPU = torch.cuda.device_count()
AMP = DEV == 'cuda'
torch.manual_seed(0); np.random.seed(0)

T0 = time.time()
def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)

def find(name, hint):
    hits = sorted(p for p in glob.glob(f'{INPUT}/**/{name}', recursive=True) if hint in p.replace('\\', '/'))
    assert hits, (name, hint)
    return os.path.dirname(hits[0])

COMP = INPUT if SMOKE else find('sample_submission.csv', 'playground-series-s6e10')
ORIG = find('train.csv', 'orig') if SMOKE else find('train.csv', 'airline-passenger-satisfaction')
tr, te = pd.read_csv(f'{COMP}/train.csv'), pd.read_csv(f'{COMP}/test.csv')
og = pd.concat([pd.read_csv(f'{ORIG}/train.csv'), pd.read_csv(f'{ORIG}/test.csv')], ignore_index=True)
if SMOKE:
    tr, te, og = tr.sample(20000, random_state=0).reset_index(drop=True), te.iloc[:5000], og.iloc[:10000]
y = tr['satisfaction'].astype(int).values
yo = (og['satisfaction'] == 'satisfied').astype(int).values
NTR, NTE, NOG = len(tr), len(te), len(og)
N2 = NTR + NTE
FD = 'Flight Distance'
CAT = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
COLS = [c for c in tr.columns if c not in ('id', 'satisfaction')]
NUMC = ['Age', FD, 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
RAT = [c for c in COLS if c not in CAT + NUMC]
C = len(COLS)
log('rows', NTR, NTE, NOG, 'cols', C, 'device', DEV, NGPU)

full = pd.concat([tr[COLS], te[COLS], og[COLS]], ignore_index=True)
for c in CAT:
    full[c] = pd.factorize(full[c], sort=True)[0]
full = full.astype(np.float64).fillna(-1.0)
A = full.values                                     # (N2 + NOG, C) raw values
isnum = np.array([c not in CAT for c in COLS])

# ---- numeric value per token (standardized; log1p for distance and delays)
XN = A.copy()
for j, c in enumerate(COLS):
    if c in (FD, 'Departure Delay in Minutes', 'Arrival Delay in Minutes'):
        XN[:, j] = np.log1p(np.clip(XN[:, j], 0, None))
mu, sd = XN[:N2].mean(0), XN[:N2].std(0) + 1e-6
XN = ((XN - mu) / sd).astype(np.float32)

# ---- value vocabulary (values seen < 3 times in train+test share one "rare" id per column)
V = np.zeros(A.shape, np.int64); off = 0; val_side = []
p0 = yo.mean()
for j in range(C):
    codes, uniq = pd.factorize(A[:, j])
    cnt_tt = np.bincount(codes[:N2], minlength=len(uniq))
    cnt_og = np.bincount(codes[N2:], minlength=len(uniq))
    pos_og = np.bincount(codes[N2:], weights=yo, minlength=len(uniq))
    keep = cnt_tt >= 3
    remap = np.full(len(uniq), -1); remap[keep] = np.arange(keep.sum()); rare = keep.sum()
    remap[~keep] = rare
    nv = rare + 1
    ctt = np.bincount(remap, weights=cnt_tt, minlength=nv); cog = np.bincount(remap, weights=cnt_og, minlength=nv)
    pog = np.bincount(remap, weights=pos_og, minlength=nv)
    m = 20.0
    lo = np.log((pog + m * p0) / (cog - pog + m * (1 - p0)))
    s = np.column_stack([np.log1p(ctt), np.log1p(cog), np.log((cog + 1) / (ctt + 1)), lo])
    val_side.append(s)
    V[:, j] = remap[codes] + off; off += nv
VOCAB = off
VS = np.concatenate(val_side).astype(np.float64)
VS = ((VS - VS.mean(0)) / (VS.std(0) + 1e-6)).astype(np.float32)

# ---- route profile: per row and column, mean / std of the column over train+test rows with the same distance,
#      and the original data's mean for that distance (global mean when the distance is not in the original)
fd_all = A[:, COLS.index(FD)]
fdc, fdu = pd.factorize(fd_all)
K = len(fdu)
RM = np.zeros((len(A), C), np.float32); RS = np.zeros_like(RM); ORM = np.zeros_like(RM)
ntt = np.bincount(fdc[:N2], minlength=K).astype(np.float64)
nog = np.bincount(fdc[N2:], minlength=K).astype(np.float64)
for j in range(C):
    x = XN[:, j].astype(np.float64)
    s1 = np.bincount(fdc[:N2], weights=x[:N2], minlength=K); s2 = np.bincount(fdc[:N2], weights=x[:N2] ** 2, minlength=K)
    mean = np.where(ntt > 0, s1 / np.maximum(ntt, 1), 0.0)
    std = np.sqrt(np.maximum(np.where(ntt > 0, s2 / np.maximum(ntt, 1), 1.0) - mean ** 2, 0)) + 0.1
    so = np.bincount(fdc[N2:], weights=x[N2:], minlength=K)
    omean = np.where(nog > 0, so / np.maximum(nog, 1), x[N2:].mean())
    RM[:, j], RS[:, j], ORM[:, j] = mean[fdc], std[fdc], omean[fdc]

# ---- summary token 1: row statistics (label-free)
R = A[:, [COLS.index(c) for c in RAT]]
dep, arr = A[:, COLS.index('Departure Delay in Minutes')], A[:, COLS.index('Arrival Delay in Minutes')]
S1 = np.column_stack([R.sum(1), R.mean(1), R.std(1), R.min(1), R.max(1), (R == 0).sum(1), (R == 5).sum(1),
                      (R >= 4).sum(1), (R <= 2).sum(1), np.log1p(np.clip(dep, 0, None)), np.log1p(np.clip(arr, 0, None)),
                      np.sign(arr - dep) * np.log1p(np.abs(arr - dep)), dep > 0, arr > 0, arr < 0,
                      np.log1p(ntt[fdc]), np.log1p(nog[fdc])]).astype(np.float64)

# ---- summary token 2: original-data block (teacher trained on the original rows only + key-match statistics)
import lightgbm as lgb
teacher = lgb.LGBMClassifier(n_estimators=100 if SMOKE else 600, learning_rate=0.05, num_leaves=63,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1)
teacher.fit(A[N2:], yo)
t_logit = teacher.predict(A, raw_score=True)
log('teacher AUC on train', round(roc_auc_score(y, t_logit[:NTR]), 5))
blocks = [t_logit]
ci = {c: COLS.index(c) for c in COLS}
for keys in ([FD], [FD, 'Class'], [FD, 'Age'], [FD, 'Age', 'Class'], [FD, 'Type of Travel', 'Class']):
    kc = pd.factorize(pd.MultiIndex.from_arrays([A[:, ci[k]] for k in keys]))[0]
    nk = kc.max() + 1
    n_tt = np.bincount(kc[:N2], minlength=nk); n_o = np.bincount(kc[N2:], minlength=nk)
    p_o = np.bincount(kc[N2:], weights=yo, minlength=nk)
    m = 10.0
    blocks += [np.log1p(n_tt[kc]), np.log1p(n_o[kc]), np.log((p_o[kc] + m * p0) / (n_o[kc] - p_o[kc] + m * (1 - p0)))]
S2 = np.column_stack(blocks).astype(np.float64)
def stdz(M):
    return ((M - M[:N2].mean(0)) / (M[:N2].std(0) + 1e-6)).astype(np.float32)
S1, S2 = stdz(S1), stdz(S2)
log('features ready: vocab', VOCAB, 'S1', S1.shape, 'S2', S2.shape)

# ---- tensors on the first GPU
g = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=DEV)
TV, TX, TRM, TRS, TORM = g(V, torch.long), g(XN), g(RM), g(RS), g(ORM)
TVS, TS1, TS2 = g(VS), g(S1), g(S2)
ISNUM = g(isnum.astype(np.float32))

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.val = nn.Embedding(VOCAB, D)
        self.col = nn.Parameter(torch.randn(C, D) * 0.02)
        self.freq = nn.Parameter(torch.randn(C, 16))
        self.numw = nn.Parameter(torch.randn(C, 32, D) * 0.02)
        self.sidew = nn.Parameter(torch.randn(C, 8, D) * 0.02)
        self.norm = nn.RMSNorm(D)
        self.register_buffer('isnum', ISNUM.view(1, C, 1).clone())
        self.sum1 = nn.Sequential(nn.Linear(S1.shape[1], D), nn.GELU(), nn.Linear(D, D))
        self.sum2 = nn.Sequential(nn.Linear(S2.shape[1], D), nn.GELU(), nn.Linear(D, D))
        self.sumtype = nn.Parameter(torch.randn(2, D) * 0.02)
        layer = nn.TransformerEncoderLayer(D, H, FF, dropout=0.05, activation='gelu', batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, L, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(D)
        self.rtd = nn.Linear(D, 1)
        self.head = nn.Linear(D, 1)

    def forward(self, v, xn, side, s1=None, s2=None):
        # v (B,C) value ids, xn (B,C) numeric values, side (B,C,8)
        z = xn.unsqueeze(-1) * self.freq * (2 * math.pi)                       # (B,C,16)
        per = torch.cat([torch.sin(z), torch.cos(z)], -1)                         # (B,C,32)
        num = torch.einsum('bcf,cfd->bcd', per, self.numw) * self.isnum
        sd_ = torch.einsum('bcf,cfd->bcd', side, self.sidew)
        tok = self.norm(self.val(v) + self.col + num + sd_)
        if s1 is not None:
            st = torch.stack([self.sum1(s1), self.sum2(s2)], 1) + self.sumtype
            tok = torch.cat([tok, st], 1)
        h = self.out_norm(self.enc(tok))
        if s1 is None:
            return self.rtd(h).squeeze(-1)                                        # (B,C) replaced-token logits
        return self.head(h.mean(1)).squeeze(-1)                                   # (B,) label logit

def side_of(v, xn, rows):
    vs = TVS[v]                                                                   # (B,C,4)
    rm, rs, orm = TRM[rows], TRS[rows], TORM[rows]
    dev_ = torch.clamp((xn - rm) / rs, -5, 5)
    return torch.cat([vs, rm.unsqueeze(-1), rs.unsqueeze(-1), orm.unsqueeze(-1), dev_.unsqueeze(-1)], -1)

def wrap(m):
    return nn.DataParallel(m) if NGPU > 1 else m

def sched(opt, total):
    warm = max(1, int(0.03 * total))
    f = lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)

def run_steps(net, opt, sch, scaler, params, loss_fn, rows_all, epochs, tag):
    n = len(rows_all); steps = 0; total = epochs * (n // BATCH)
    for ep in range(epochs):
        perm = rows_all[torch.randperm(n, device=DEV)]
        run, cnt = 0.0, 0
        for i in range(0, n - BATCH + 1, BATCH):
            rows = perm[i:i + BATCH]
            with torch.autocast('cuda', dtype=torch.float16, enabled=AMP):
                loss = loss_fn(net, rows)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); sch.step()
            run += loss.item() if (steps % 50 == 0) else 0; cnt += steps % 50 == 0; steps += 1
            if steps % 500 == 0:
                log(f'  {tag} ep {ep} step {steps}/{total} loss~{run / max(cnt, 1):.4f}')
                run, cnt = 0.0, 0

# ---- 1. RTD pretraining on train + test + original features (no labels)
def rtd_loss(net, rows):
    v, xn = TV[rows], TX[rows]
    mask = torch.rand(v.shape, device=DEV) < 0.15
    src = rows[torch.randint(0, len(rows), v.shape, device=DEV)]                 # a random other row per cell
    cidx = torch.arange(C, device=DEV).expand_as(v)
    v2 = torch.where(mask, TV[src, cidx], v); x2 = torch.where(mask, TX[src, cidx], xn)
    mask = mask & (v2 != v)                                                       # identical values are not "replaced"
    logits = net(v2, x2, side_of(v2, x2, rows))
    return F.binary_cross_entropy_with_logits(logits.float(), mask.float())

base = Model().to(DEV)
log('params', sum(p.numel() for p in base.parameters()) / 1e6, 'M')
pre_params = [p for n_, p in base.named_parameters() if not n_.startswith(('sum1', 'sum2', 'sumtype', 'head'))]
opt = torch.optim.AdamW(pre_params, lr=LR_PRE, weight_decay=0.01)
all_rows = torch.arange(len(A), device=DEV)
sch = sched(opt, PRE_EPOCHS * (len(A) // BATCH))
scaler = torch.amp.GradScaler('cuda', enabled=AMP)
net = wrap(base); net.train()
run_steps(net, opt, sch, scaler, pre_params, rtd_loss, all_rows, PRE_EPOCHS, 'rtd')
PRE_STATE = {k: v.detach().clone() for k, v in base.state_dict().items()}
log('pretraining done')
del net, opt

# ---- 2. fine-tuning, 5 folds
def ft_loss_factory(yt):
    def f(net, rows):
        v, xn = TV[rows], TX[rows]
        logit = net(v, xn, side_of(v, xn, rows), TS1[rows], TS2[rows])
        return F.binary_cross_entropy_with_logits(logit.float(), yt[rows])
    return f

@torch.no_grad()
def predict(net, rows_np):
    net.eval(); out = []
    rows_all = torch.as_tensor(rows_np, device=DEV)
    for i in range(0, len(rows_all), 4096):
        rows = rows_all[i:i + 4096]; v, xn = TV[rows], TX[rows]
        with torch.autocast('cuda', dtype=torch.float16, enabled=AMP):
            out.append(net(v, xn, side_of(v, xn, rows), TS1[rows], TS2[rows]).float().cpu().numpy())
    net.train()
    return np.concatenate(out)

YT = torch.zeros(len(A), device=DEV); YT[:NTR] = g(y)
oof = np.full(NTR, np.nan); test_sum = np.zeros(NTE); nt = 0
skf = StratifiedKFold(5, shuffle=True, random_state=42)
fold_times = []
for k, (a, b) in enumerate(skf.split(np.zeros(NTR), y)):
    if SMOKE and k > 0:
        break
    fn = f'{OUT}/qft2_f{k}.npz'
    if fold_times and time.time() - T0 + 1.15 * max(fold_times) > TIME_LIMIT:
        log('time guard: stopping before fold', k); break
    t1 = time.time()
    m = Model().to(DEV); m.load_state_dict(PRE_STATE)
    enc = [p for n_, p in m.named_parameters() if n_.startswith('enc.')]
    rest = [p for n_, p in m.named_parameters() if not n_.startswith('enc.')]
    opt = torch.optim.AdamW([{'params': enc, 'lr': LR_ENC}, {'params': rest, 'lr': LR_TOK}], weight_decay=0.01)
    sch = sched(opt, FT_EPOCHS * (len(a) // BATCH))
    scaler = torch.amp.GradScaler('cuda', enabled=AMP)
    net = wrap(m); net.train()
    run_steps(net, opt, sch, scaler, list(m.parameters()), ft_loss_factory(YT), torch.as_tensor(a, device=DEV), FT_EPOCHS, f'fold{k}')
    pv = predict(net, b); pt = predict(net, np.arange(NTR, N2))
    oof[b] = pv; test_sum += pt; nt += 1
    np.savez(fn, idx=b, pred=pv, test=pt)
    fold_times.append(time.time() - t1)
    log(f'fold {k} AUC {roc_auc_score(y[b], pv):.6f} ({fold_times[-1]:.0f}s)')
    del net, m, opt; torch.cuda.empty_cache() if AMP else None

done = ~np.isnan(oof)
log('OOF AUC on finished folds', round(roc_auc_score(y[done], oof[done]), 6), 'rows', int(done.sum()))
if done.all():
    np.save(f'{OUT}/oof_qft2.npy', oof.astype(np.float32)); np.save(f'{OUT}/test_qft2.npy', (test_sum / nt).astype(np.float32))
    pd.DataFrame({'id': te['id'].values, 'satisfaction': 1 / (1 + np.exp(-test_sum / nt))}).to_csv(f'{OUT}/submission.csv', index=False)
log('done')
