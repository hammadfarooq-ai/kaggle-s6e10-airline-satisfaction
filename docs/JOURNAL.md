# S6E10 experiment journal (Kaggle Playground S6E10, Predicting Airline Satisfaction)

> **Note on paths.** This journal was written while working, before the repository was reorganised. Old paths map to
> the current layout like this: `stack/` -> `src/stacking/v1_lr_public_loaders/`, `stack2/` -> `v8_familywise_l2/`,
> `stack3/` -> `v9_famreg_lgb/`, `stack4/` -> `v10_candidates/`, `stack5/` -> `v11_candidate_screen/`,
> `stack6/` -> `v12_mlp_stacker/` (all under `src/stacking/`); `members*/` -> `src/members/`; `tfm*/` -> `src/tabicl_fe/`;
> `qft/` -> `src/qft_transformer/`; `subs/`, `qft_out/`, `tfm_out/` -> `artifacts/` (not in git).


- Competition: `playground-series-s6e10`, metric ROC-AUC, deadline 2026-10-31 23:59 UTC, 10 submissions/day.
- Kaggle user: `malikhammadfarooq`. CLI auth uses OAuth; if calls start failing with "Authentication required"
  or "Permission denied", run `kaggle auth login --force` (the browser approves it).
- Data: `data/` (train, test, sample), `data/orig/` (teejmahal20 original, 129,880 rows),
  `data/orig2/data.csv` (arseniyshutko 129k: row-for-row identical to teejmahal20, nothing new),
  `data/pub_subs/` (rabari9999 top18 public submission files).

## What is in the pool and what was measured

All public OOF libraries use `StratifiedKFold(5, shuffle=True, random_state=42)`; stacker meta folds use seed 7.

| stack | members | nested CV | public LB |
|---|---|---|---|
| LR C=0.003, matterhorn's public loaders (`stack/stack.py`) | 202 | 0.962171 | 0.96182 |
| + koumeimaki 7 members | 209 | 0.962181 | 0.96181 |
| 0.8 LR + 0.2 LightGBM stacker, 209 | 209 | 0.962187 | 0.96181 |
| rank blends with public submission files (`subs/blend/`) | – | no CV | 0.96178 – 0.96183 (best `blend_b`, `blend_c`) |
| stack v8: goodpjw2008's 245-member pool + family-wise L2 (`stack2/stack2.py`) | 245 | 0.962229 (seed 7) / 0.962231 (seed 13); flat 0.962222 | **0.96186** |
| stack v9 (`stack3/stack3.py`): 245 pool, 0.8 family-wise LR + 0.2 LightGBM stacker (weight picked on the mean of meta seeds 7 and 13) | 245 | **0.962242** (seed 7 0.962242 / seed 13 0.962243) | 0.96184 |
| rank mean: v8 + goodpjw2008 frozen 245 + dariushafshar 247 (`subs/v8/submission_twin3.csv`) | – | no CV | 0.96186 |

Kaggle kernels (private): `s6e10-stack-public-own` (v7 = 210 members), `s6e10-own-members-a/-b` (own GPU models),
`s6e10-stack-v8-famreg` (245 pool, family-wise L2, rank average with goodpjw2008's and dariushafshar's frozen stacks).

## Dead ends (measured, do not repeat)

- Own GPU members (XGBoost + nested TE 0.96102, TabM raw 0.95829, TabM FE 0.96051 / 0.96006, kNN 0.95609):
  each one or all together add 0 to -0.000006 to the stack CV.
- dmitriykolmikov all-categorical CatBoost: -0.000001. amanatar forge pairs: below koumeimaki-only.
- Original-data near-copy "leak": only 347 train rows (0.05%) match; stack delta -0.000001.
- Residual LightGBM on top of the stack with raw features, train+test counts, id: learns nothing.
- Residual autocorrelation by id order: none. Exact duplicate rows: none.
- Rank blends with public submission files: all within public-LB noise (+-0.00003).

## State of the board (2026-10-14)

Leader 0.96243, #2 0.96234, then a crowd at 0.96184 – 0.96188. Best public notebooks reach 0.96187 – 0.96188
(goodpjw2008 Part 11 twin stacks, dariushafshar 247 stack, ewtherka family-wise L2). Own rank ~110 / 1782 (2026-10-14)
with best public 0.96186 (stack v8). Best CV is stack v9 (0.962242), public 0.96184.
Final-selection suggestion: stack v9 (best CV) + stack v8 family-wise L2 (best public). Public LB is ~20% of test (~60k rows); near-identical stacks differ by +-0.00002 by chance.

## In progress when this file was written (2026-10-14)

- DONE: stack v8 (`stack2/`, kernel `s6e10-stack-v8-famreg`) submitted on 2026-10-14: family-wise L2 and the
  three-stack rank mean both scored public LB 0.96186; rank went from 201 to 97. `stack3/stack3.py` (v9, same 245 loaders + LGB blend) is now
  the base stack: add new members to it.
- `tfm/tfm.py` (copies in `tfm_a/`, `tfm_b/`): TabICL v2 + Chris Deotte's 9 FE features (discussion 749007),
  context 200k, n_estimators 4. Kernel A = folds 0-2, kernel B = folds 3-4 + test. Push was blocked by the
  account's 2-GPU-session limit (other competitions' kernels); push with `kaggle kernels push -p tfm_a`
  and `-p tfm_b` once a slot is free. Per-fold outputs `own_tabicl_fe_f{k}.npz` (idx, pred) and
  `own_tabicl_fe_test.npz`: assemble into one OOF/test pair and test it as a stack member.
  On 2026-10-14 the push failed with "Maximum weekly GPU quota of 30.00 hours reached" (the user's other
  competitions used it up). Kaggle's weekly GPU quota resets on Saturday; push tfm_a / tfm_b after that.

## Session 2026-10-17 (Saturday, GPU quota reset)

- Board 2026-10-17: #1 0.96284, #2 0.96280 (both jumped on 10-17, ~100 subs each), Prior 0.96234, then 0.96200,
  a crowd at 0.96190 - 0.96194. Own 0.96186 is now rank 175 / 2112 (score unchanged, others moved up).
- Discussion / notebooks checked: no public explanation of the 0.9628 jump. New threads: Chris Deotte
  "How to build a Tabular Transformer" (Qwen3-0.6B LoRA on row tokens, OOF 0.959795, raw features only) and
  Chris's reply in "Finetuning TFMs": fine-tuned TFMs do NOT add to a stack that already has the zero-shot TFMs.
  starkhushi "Everything I Measured That Didn't Help" + "Finding Each Passenger's Parent Row": near-copy parent
  labels (AUC 0.89 alone) add -0.000002 to a strong stack; parent's FD/Age +0.00002 (tree) / -0.00001 (linear);
  original rows appended -0.00025; original rate by key 0; residual LightGBM -0.00025. Do not repeat these.
  0.96190 notebooks (mayankraaz, yuvrajlamba01, yeashusemwal, rabari9999) are blends of public stacks/submission
  files, nothing outside the 245 pool.
- `cdeotte/s6e10-tfm-oof-and-test-predictions`: unchanged since 10-06 (no FE / fine-tuned OOFs yet).
- `goodpjw2008/s6e10-goodpjw2008-extra-members-oof` v4: adds `tabpfn_ne4` (TabPFN-3.5 n_est=4, OOF 0.961491, not in
  members.csv; goodpjw2008 measured +0.000000 on the 245 stack) and `twin_submission_1015.csv` (no OOF).
  `nina2025/*`, `rabari9999/s6e10-top18-artifacts`: submission files only, no OOF. `najiama/s6e10-oof`:
  07_XGBoost OOF 0.960875 (blend OOFs 01-06 are stack-level, do not use as members).
- `stack4/stack4.py` (kernel `s6e10-stack-v10-candidates`): stack v9 + candidate groups (gpx_tabpfn_ne4,
  naj_xgb07, own_tabicl_fe when attached), each screened with nested family-wise LR on meta seeds 7 and 13;
  a group is kept only if both seeds go up.
- RESULT stack v10 run 1 (24 min CPU): LR famreg base 245 = 0.962229 / 0.962231. + gpx_tabpfn_ne4: 0.962230 /
  0.962234 (+2.2e-6, both up, kept). + naj_xgb07: -1.0e-6 (dropped). Final 246, 0.8 LR + 0.2 LGB: 0.962243 /
  0.962243 vs v9 0.962242 / 0.962243 = +0.5e-6, i.e. noise. NOT submitted.
- Adversarial validation train vs test (LightGBM, 21 raw features): AUC 0.498, no shift. Dead end.
- TabICL FE kernels: v1 of kernel A died with CUDA OOM on the T4 (kv_cache=True, 200k context, batch_size 8).
  Fixed in `tfm*/tfm.py`: kv_cache=False, batch_size=1, query step 50k, expandable_segments, fold-time guard.
  A v2 pushed 10-17; B v2 waits for B v1 (old code, queued) to clear the 2-GPU-session limit.
- RESULT kernel A v2 (T4, 77 min for 3 folds, ~25 min/fold): fold AUC 0.959678 / 0.958643 / 0.960235,
  folds 0-2 together 0.959506 (zero-shot raw TabICLv2 in Chris's set: 0.958088). Files copied to `tfm_out/`.
- Kernel B v1 (old OOM code) sat QUEUED for 2+ h; the second GPU slot is held by the user's other kernel
  `arc3-bench-throughput` (also queued). B v2 is pushed by a retry loop as soon as a slot frees.

## Session 2026-10-18

- TabICL FE complete (kernel B v2, 126 min): folds 3/4 0.959678 / 0.959764, full OOF 0.959582; assembled pair in
  `tfm_out/oof_own_tabicl_fe.npy` / `test_own_tabicl_fe.npy`. In stack4 (248 candidates) it gives -2.2e-6
  (seed 7 down, seed 13 equal), so dropped. Stack result identical to v10 (246, 0.962243 / 0.962243). Dead end.
- Submissions 2026-10-18 (user asked to use all 10): stack v10 `subs/v10/submission.csv` public 0.96183.
  Rank blends in `subs/b18/` (no CV) with public 0.96190 files (nina2025 0.96190_a/_b/0.96190, mayankraaz output;
  these four have rank corr 0.99995+ with each other, ~0.9996 with our stacks):
  b1 v10/n190b 50/50 0.96187; b2 30/70 0.96188; b3 mean(v10,v8,n190a,n190b,may) 0.96188;
  b4 v10/n190b 20/80 **0.96189**; b5 v9/n190b 30/70 0.96188; b6 v10 30 + pub4 70 0.96188;
  b7 mean(v10,v8,twin3) 30 + pub4 70 **0.96189**; b8 v10 40 + pub4 60 0.96188; b9 v10 LR-only 30 + n190b 70 **0.96189**.
  All within public noise (+-0.00002); best public now 0.96189.
- Final-selection suggestion now: stack v10/v9 (best CV, own and honest) + b7 or b9 (best public, 0.96189).

## Session 2026-10-21

- Board: #1 0.96357, #2 0.96334, Prior 0.96242, Fritz Cremer 0.96240, then 0.96235 / 0.96228 / 0.96210 ...
  Public notebooks now 0.96208 (mayankraaz "LB Surface Ascent v2", yuvrajlamba01, nina2025 h-blend, rabari9999,
  avazaku, kirilltym). These are built by fitting a quadratic surface to dozens of PUBLIC-LB scores and stepping
  along it: public-LB overfitting by construction, private gain unknown.
- goodpjw2008 extra-members v5-v8 added only pinned public files (anchor_twin_dp30, hanifnoerrofiq agentic 0.96198,
  mayankraaz surface v2 0.96196 / v4 0.96208), stack182 OOF/test logits and the 16 old series stacks; no new
  single-model members. Copies in `data/pub_subs/g21/`. Chris's TFM dataset still unchanged.
- The 0.96208 file has rank corr only 0.9965 with stack v10 (older public files ~0.9996).
- Submitted 2026-10-21 (`subs/b21/`, no CV): c1 v10 20 + s4 80 = 0.96205; c2 v10 10 + s4 90 = **0.96207**;
  c3 b7 30 + s4 70 = 0.96204; c4 v10 20 + s4 50 + hanif 30 = 0.96203. Adding our stack lowers public, as expected
  for an LB-fitted file.
- Final selection: keep one honest CV file (stack v10, CV 0.962243, public 0.96183) as the safety slot; the second
  slot is a choice between b7 (0.96189, CV-family blend) and c2 (0.96207, mostly LB-surface file, higher shake-up risk).

## Session 2026-10-22

- Board: #1 0.96370, #2 0.96341, Fritz Cremer 0.96244, Prior 0.96242. Best public notebook rabari9999 0.96209
  (output identical to nina2025 `0.96209.csv`; rank corr 0.9998+ with the other LB-surface files, 0.9960 with v10).
  Files in `data/pub_subs/g22/`. No new OOF members anywhere.
- Submitted (`subs/b22/`, no CV): d1 mean(s4, n208b, rab) 0.96208; d2 v10 5 + rab 95 0.96208;
  d3 rab 50 + n208b 50 **0.96209** (new best public, equal to the public notebook).
- Final selection unchanged in spirit: stack v10 (CV) + one LB-surface file (d3 0.96209 or c2 0.96207).

## Session 2026-10-23

- Board: #1 0.96370, #2 0.96341, Fritz Cremer 0.96247. Best public notebook yuvrajlamba01 0.96213
  (`data/pub_subs/g23/yuv/submission.csv`); nina2025 added `0.96210.csv`. goodpjw2008 v9/v10 only re-host the 48
  series members (= the gp_* columns already in the pool) and old revisit stack files: nothing new for the stack.
- Submitted (`subs/b23/`, no CV): e1 y213 50 + n210 50 = 0.96212; e2 y213 70 + d3 30 = **0.96212**;
  e3 mean(y213, n210, d3) = 0.96211. Best own public now 0.96212 (all LB-surface files, rank corr 0.9998+).

## Session 2026-10-24

- Board: #1 Masaya Kawamata **0.96407** (jumped 10-24), #2 0.96341, Fritz Cremer 0.96247, Prior 0.96243, srikalyan
  0.96236, then 0.96228 and a big crowd at 0.96213. Public notebooks top out at 0.96213.
- The 0.96213 recipe (decoded by goodpjw2008 "Meta Stack on Public OOF", Part 4): 0.90 x nina2025 0.96210 rank file
  + 0.10 x mehrankazeminia TabPFN probability. goodpjw2008 measured that it LOSES in nested CV. All 0.96213 files
  (nina2025 `ps-s6e10-02` 0.96213a-d, yuvrajlamba01, goodpjw2008 meta) have rank corr 0.999997+ with each other and
  0.99999 with our e1; copies in `data/pub_subs/g24/`.
- goodpjw2008 "Auxiliary-Task Features" / "More Aux Carriers" (10-23) are CPU re-runs of old parts: their members are
  the gp_* columns already in the pool. Extra-members dataset 10-23 only added revisit submission csvs.
- Chris's TFM dataset still unchanged (10-06). Chris's new thread 751334 "Part 2 Tabular Transformer" (Q-FT v2:
  ModernBERT-base, order-free, side features, replaced-token-detection pretraining on train+test+orig): OOF 0.960798,
  ~50 A100-minutes; Naji reproduced on T4x2 in ~3.5 h (OOF 0.96075). No OOF published. A genuinely new family; would
  need our own GPU run (ask the user first).
- Jiatian Zhang2 thread 750479 (15-fold GPU tree stack, "ensemble 0.96238" = trees + Chris's TFMs): notebook output
  has only submission.csv (rank corr 0.983 with v10), no OOF. Not usable as a member.
- New OOF: `vigneshalugunulla/srim-s6e10-bespoke-artifacts` (10-23, no description, no public notebook):
  `oof_srim_lgb_bespoke` OOF 0.960943 (ids/labels align with train), `pillar1_oof_base252` (a stack, 0.962238, not a
  member), features `survey_nz_mean` (AUC 0.81), `zero_cnt`. Tested in `stack5/stack5.py` (kernel
  `s6e10-stack-v11-candidates`): base = v10's 246 members, candidates srim_lgb and srim_survey_mean.
- RESULT stack v11 (15 min CPU, outputs in `subs/v11/`): base 246 LR famreg 0.962230 / 0.962234.
  + srim_lgb 0.962228 / 0.962235 (-0.3e-6, seed 7 down) dropped; + srim_survey_mean 0.962230 / 0.962232 (-1.1e-6)
  dropped. Final = v10 exactly (246, 0.8 LR + 0.2 LGB, 0.962243 / 0.962243). Not submitted. Dead end.
- Submitted (`subs/b24/`, no CV): f1 = rank mean of five public 0.96213 files (y213, nina 0.96213a/c/d, goodpjw2008
  meta) = **0.96213** (rank corr 0.999994 with e1; +0.00001 over e1 is noise).
- Final selection (as of 10-24): slot 1 = stack v10 `subs/v10/submission.csv` (CV 0.962243, public 0.96183);
  slot 2 = f1 `subs/b24/f1_mean5_213.csv` (0.96213) or e1 (0.96212), equivalent. Slot 2 is an LB-surface file with no
  CV support; goodpjw2008 measured that its TabPFN injection loses in CV, so it carries shake-up risk.
- Rest of 10-24's submissions (`subs/b24/`, all 10 used; "pil" = vigneshalugunulla `pillar1_test_base252`, their
  252-member stack, OOF 0.962238 checked by us, rank corr 0.9998 with v10):
  Public: f6 f1 95/v10 5 = 0.96212; f2 90/10 = 0.96211; f3 80/20 = 0.96210;
  f4 70/30 = 0.96208; f5 50/50 = 0.96203; f8 f1 90 + pil 10 = 0.96211; f9 f1 80 + v10 10 + pil 10 = 0.96210;
  f7 v10 50 + pil 50 = 0.96184; f10 pillar1 alone = 0.96184 (v10 = 0.96183).
  Reading: public falls smoothly (~-0.00001 per 10% of CV stack mixed in) toward the CV stacks' 0.96183-0.96184.
  The LB-surface files sit ~0.0003 above every honest CV stack on public; whether that holds on private is unknown.
- User approved a Q-FT v2 GPU run. `qft/qft.py` (kernel `s6e10-qft2-transformer`, T4, no internet): order-free
  transformer, random init (Chris: pretrained = random), d768 / 12 heads / 12 layers / FF 3072 (~90M params; Chris
  used ModernBERT-base 22 layers), 21 feature tokens (value + column + periodic numeric + 8 side features: value
  counts tt/orig, ratio, orig log-odds, FD-route mean/std, orig route mean, deviation from route) + 2 summary tokens
  (17 row stats; orig LightGBM teacher + key-match counts/log-odds for FD, FD+Class, FD+Age, FD+Age+Class,
  FD+Travel+Class). RTD pretraining 3 epochs on train+test+orig (15% swaps), fine-tune 2 epochs per fold,
  SKF(5, 42), fp16, DataParallel if 2 GPUs. Outputs `qft2_f{k}.npz`, `oof_qft2.npy`, `test_qft2.npy` (logits).
  CPU smoke test (SMOKE=1, tiny model, 20k rows) ran end to end. `qft/push_when_free.sh` pushes it only after one of
  the two ARC3 GPU kernels (arc3-eval-tc-dprime, arc3-bench-throughput, both RUNNING on 10-24) has finished.
  Next: add it to stack5 as a candidate (family "tfm"), keep only if seeds 7 and 13 both go up.
- RESULT Q-FT v2 (pushed 04:52 UTC once one ARC3 kernel ended; 2x T4, 10,029 s = 2.8 h; RTD pretraining 59 min,
  ~21.5 min per fold): fold AUC 0.961044 / 0.960300 / 0.961969 / 0.961126 / 0.961238, **OOF 0.961116** (Chris's
  ModernBERT version 0.960798). Spearman with v10 test 0.984. Files in `qft_out/` (logits).
  Stack test: `stack5` (kernel s6e10-stack-v11-candidates v2, kernel source = the Q-FT kernel), candidate tfm_qft2.
- RESULT: base 246 LR famreg 0.962230 / 0.962234; + tfm_qft2 0.962230 / 0.962234 (+0.2e-6, seed 13 not up) →
  dropped; final stack = v10 again (0.962243 / 0.962243). Outputs `subs/v11b/`. Dead end: a strong single model
  (0.9611, Spearman 0.984 with v10) still adds nothing, same as Chris found for fine-tuned TFMs. Do not repeat
  (other Q-FT sizes / seeds are not worth another GPU run). Final selection unchanged: v10 + f1.

## Session 2026-10-25

- Board: #1 Masaya Kawamata 0.96416, #2 0.96352, Fritz Cremer 0.96247, Prior 0.96243, Aurrog 0.96240.
- New: nina2025 "h-blend (pure.rank's)" (now top public notebook by score) = rank corr 0.9999996 with our f1.
  denpugovkin "Learn the Ranking | RealMLP x Direct AUC" exp046: hybrid OOF 0.961582 (a small blend of goodpjw2008-style
  members; own folds). Screened in stack5 v3 (kernel source = their notebook, candidate denp_hybrid); kernel
  RESULT (`subs/v11c/`): base 246 LR famreg 0.962228 / 0.962236 (mean 0.962232, same as before); + denp_hybrid
  0.962228 / 0.962235 (-0.8e-6, both seeds not up) → dropped. Final stack = v10 again (0.8 LR + 0.2 LGB
  0.962241 / 0.962246, mean 0.962243). Dead end.
  10th slot: `subs/b25/g10_f1_85_tab15` (f1 85 + mehran TabPFN rank 15, LB-surface probe, made by
  `subs/b25/make_g10.py`) = **0.96213**. Up to 15% extra TabPFN the public stays flat at 0.96213. All 10 of 10-25 used.
- Submitted 9 (`subs/b25/`, no CV, all LB-surface): g1 nina_h, g2 nh 50 + f1 50, g3 mean of 6 x 0.96213 files,
  g4 nh 70 + gp213 30, g5 nh 70 + n213d 30, g6 nh 70 + n213a 30, g7 nh 80 + e1 20, g8 f1 95 + mehran TabPFN 5,
  g9 f1 90 + TabPFN 10: **all 0.96213**. The public surface is flat here; nothing above 0.96213 is reachable by blending
  these files. One submission of 10-25 was kept for the denp stack if it passes both seeds.

- Leak checks (local, 10-25): no exact duplicates train/train or train/test; 4 test and 17 train rows equal an original
  row. 38,893 test rows (13%) and 91,054 train rows have a train "FD-twin" (identical on all 20 columns except Flight
  Distance; median FD gap 1181, ids spread out). Twin leave-one-out label AUC 0.806 on those rows; Q-FT alone 0.8863,
  Q-FT + twin label 0.8861. Twins are just common profiles, not copies. Dead end.
- `stack6/stack6.py` (kernel `s6e10-stack6-mlp-stacker`, CPU): 246 members, LR famreg + LGB (v10) + MLP-A (logits)
  + MLP-B (logits + raw features) stackers nested on meta seeds 7/13, blend grid on z-scored parts.
  RESULT (27 min CPU, outputs + OOF parts in `subs/s6/`): seed 7 / 13: LR 0.962228 / 0.962236, LGB 0.962046 /
  0.962008, MLP-A 0.961978 / 0.961973, MLP-B 0.961911 / 0.961922. Every MLP weight lowers both seeds (best with
  MLP = LR 0.8 + LGB 0.1 + MLP-A 0.1, -5.1e-6). Best = v10 recipe (0.962241 / 0.962246). Dead end: raw features and
  non-linear stacking add nothing on top of LR + LGB.
- Reading of the board: #1 rose 0.96284 → 0.96357 → 0.96370 → 0.96407 → 0.96416 in steady daily steps of
  ~1e-4 with ~100 subs/day, the pattern of public-LB hill-climbing, not of a new model. Our honest CV ceiling is
  0.962243; ~15 single models / stackers added since 10-14 all gave 0 (±1e-6).

## Session 2026-10-26

- Board: #1 0.96416 (unchanged), #2 0.96352, Gray Robertson 0.96275 (up from 0.96228), Fritz Cremer 0.96248.
- Nothing new for the stack: goodpjw2008 extra-members 10-25 added only revisit2 submission csvs; goutamsharma12 v3
  (LGB/XGB TE 0.9603, Cat 0.9586) and kiza123123 (simplex 0.96116, no per-member test) are weak GBDTs. Not tested.
- Submitted 6 (`subs/b26/`): h1 Q-FT v2 alone **0.96101** (OOF 0.961116, CV); h6 stack6 v10 recipe 0.96185 (CV
  0.962243, = v10 0.96183 within noise); h5 v10 90 + Q-FT 10 = 0.96183; h2 f1 95 + Q-FT 5 = 0.96213;
  h3 f1 90 + Q-FT 10 = 0.96212; h4 f1 85 + Q-FT 15 = 0.96211 (Q-FT does not lift the LB-surface file, unlike TabPFN).
  Then the last 4 (no CV, LB-surface): i1 mean(f1, nina_h, g3) 0.96213; i2 f1 80 + TabPFN 20 = 0.96211 (TabPFN beyond
  15% starts to cost); i3 nina_h 85 + TabPFN 15 = 0.96213; i4 mean(f1, g9, g10) = **0.96214** (best public, +0.00001
  over f1 = noise). File `subs/b26/i4_mean_f1_g9_g10.csv`.

## Session 2026-10-27

- Board: #1 Masaya Kawamata 0.96427, #2 0.96352, Gray Robertson 0.96276, "Token it easy - LAB7" 0.96254, Fritz 0.96252.
- yuvrajlamba01 "Airline Passenger Satisfaction (0.96276+)": the 0.96276 is only the title's "target"; the file is
  0.98 x the 0.96213 anchor + 2% of RepLeaf / TabPFN / CatBoost / XGB / RealMLP (rank corr 0.99999 with f1).
  Submitted as-is (j1): **0.96212**. Not a new level. Copy in `data/pub_subs/g27/yuv276/`.
- masumhuq RealMLP-original (OOF 0.957252) and TabResNet (0.956939), OOF + test npy (`data/pub_subs/g27/`), Spearman
  ~0.93 with v10 test: screened in stack5 v4 (candidates mh_realmlp, mh_tabresnet).
  RESULT (`subs/v11d/`): + mh_realmlp 0.962232 / 0.962234 (+1.0e-6, seed 13 down); + mh_tabresnet 0.962230 /
  0.962232 (-1.2e-6, seed 13 down). Both dropped; final stack = v10 again. Dead end.
- Rest of 10-27 (`subs/b27/`, no CV, LB-surface): j2 mean(g9,g10) 0.96213; j3 f1 88 + TabPFN 12 0.96213;
  j4 mean(i4,i3) 0.96213; j5 mean(i4,y276) 0.96213; j6 i4 80 + g10 20 0.96213; j7 mean(g9,g10,i3) 0.96213;
  j8 nina_h 88 + TabPFN 12 **0.96214**; j9 mean(i4, gp213) 0.96213; j10 mean(f1,g9,g10,i3,y276) **0.96214**.
  All 10 used. The plateau is 0.96213-0.96214; best public stays i4 / j8 / j10 (0.96214).
- Final selection (10-27): slot 1 = stack v10 `subs/v10/submission.csv` (CV 0.962243, public 0.96183);
  slot 2 = i4 `subs/b26/i4_mean_f1_g9_g10.csv` (public 0.96214, no CV).

## Session 2026-10-28

- Board: #1 0.96437, #2 0.96352, nammovuivui 0.96297 (new #3), Gray Robertson 0.96276, LAB7 0.96255.
- Nothing new for the stack: areebbb "Crushing the Leaderboard" = 27 variants of nina_h (h-blend) +/- small shares of
  Chris's TabPFN-3.5 / Kumo-large / RealMLP / mega-stack (files in `data/pub_subs/g28/areeb/`); dennis6886
  `s6e10-best.csv` = 50/50 Mayank + Orange, public 0.96213; denpugovkin "hard labels" outputs are stack-level.
- Submitted 10 areebbb files as-is (`subs/b28/k_*`, no CV). Chris TabPFN-3.5 share in nina_h vs public:
  -10% (n02) 0.96214, -5% (n01) 0.96214, +3% (r12) 0.96212, +5% (r01) 0.96211, +10% (r02) 0.96209, +20% (r16)
  0.96202. Others: r14 +5% Kumo-large 0.96211, r05 +10% mega 0.96212, r09 rank mean of 6 top files 0.96211,
  r11 0.96209. A monotone public trend along the TabPFN-3.5 axis (adding it costs, removing it gains a little);
  untested further points: n08 (-15%, in the folder) and -20% / -30%. This is public-LB fitting; private unknown.

## Session 2026-10-29

- Board: #1 0.96443, nammovuivui 0.96365, Altunok 0.96352, LAB7 0.96329, Gray Robertson 0.96276.
- Public notebooks now 0.96215-0.96217, all on the same surface: anchor (nina_h / 0.96213 files) pushed AWAY from
  Chris's TFM members. goodpjw2008 meta 0.96216 = probe G (push away from xgb_base 4.7%, gen12 2.2%, realmlp 0.9%,
  toward gen3 3.3%); its log says the top files differ from the 0.96213 file by private sd ~1e-5 only.
  areebbb "Do You Need Less" 0.96217 / "More" 0.96215: n/p/q series = 1.0x anchor minus shares of TabPFN-3.5 / XGB /
  RealMLP npy from Chris's TFM set. Files in `data/pub_subs/g29/`. No new OOF members anywhere.
- Submitted 10 (`subs/b29/m_*`, no CV, public probes): q01 (-5% TabPFN -3% XGB -3% RealMLP) 0.96215; q02 (-5/-5 XGB)
  0.96215; q05 (-7/-3/-3) 0.96215; q03 (-5 TabPFN -5 RealMLP) 0.96214; q06 (-5/-5/-5) 0.96214; p01 (-7% TabPFN) 0.96214;
  n08 (-15% TabPFN) 0.96212 (past the summit); q07 ERROR (Kaggle rejected the file); G 50 + q01 50 **0.96216**;
  G 50 + q06 50 **0.96216**. Best public now 0.96216: `subs/b29/m_G50_q0150.csv`.
- Final selection (10-29): slot 1 = stack v10 (CV 0.962243, public 0.96183); slot 2 = `subs/b29/m_G50_q0150.csv`
  (public 0.96216, LB-surface, no CV). The public gain over f1 (0.96213) is 0.00003; goodpjw2008 estimates the
  private difference between these top files at ~1e-5, so slot 2's choice among them barely matters.

## Session 2026-10-30

- Board: #1 Masaya Kawamata 0.96451, #2 Altunok 0.96446, #3 nammovuivui 0.96432, LAB7 0.96371, Gray Robertson 0.96276.
- New public files (`data/pub_subs/g30/`): tkms0816 "Push Model Orthogonal Directions" 0.96223 (anchor pushed along 10
  member directions at once: xgb_base +6.0, gen12 +2.7, realmlp +3.1, gen3 -3.6, tp35 +2.8, ku -2.6, te50 +3.7,
  p2viewb +2.4, ens_v4 -5.3, ne4f -4.1 %; rank corr 0.99894 with f1), mayankraaz "Mirror Pair Climb" 0.96218,
  goodpjw2008 meta 0.96217. No new OOF members.
- Submitted 10 (`subs/b30/`, no CV, public probes): n1 tk223 as-is **0.96223**; n2 my218 0.96218; n3 gp217 0.96217;
  n4 tk 50 + my 50 0.96221; n5 tk 50 + gp 50 0.96221; n6 mean(tk,my,gp) 0.96221; n7 tk 70 + my 30 0.96222;
  n8 1.25 tk - 0.25 f1 **0.96223**; n9 1.5 tk - 0.5 f1 0.96222; n10 tk 80 + ours216 20 0.96222.
  Best public now 0.96223 (tk223 / n8); the summit along tk223's own direction is near 1.0-1.25.
- Final selection (10-30): slot 1 = stack v10 `subs/v10/submission.csv` (CV 0.962243, public 0.96183, honest);
  slot 2 = `subs/b30/n1_tk223.csv` (public 0.96223, a public-LB-fitted file, no CV, larger shake-up risk).
  Selection is done by hand on the competition's Submissions page (the CLI cannot mark final submissions).

## Session 2026-10-31 (deadline day, 23:59 UTC)

- Board 01:00 UTC: #1 nammovuivui 0.96473, Masaya Kawamata 0.96451, Altunok 0.96446, LAB7 0.96375, Gray Robertson 0.96316.
- New public files (`data/pub_subs/g31/`): areebbb "Endgame" 0.96225 (rank corr 0.99958 with tk223), subhanjan33
  "Visual Journey" 0.96224 (only `candidate_5b_submission.csv` in its output; 0.99996 with tk223), mayankraaz
  "Mirror Nudge Climb III" 0.96219. No new OOF members.
- Submitted 10 (`subs/b31/`, no CV, public probes): o1 Endgame as-is **0.96225**; o2 subhanjan 5b 0.96224;
  o3 E 50 + tk223 50 0.96224; o4 E 50 + S 50 **0.96225**; o5 mean(E,S,T) 0.96224; o6 E 70 + T 30 **0.96225**;
  o7 1.25 E - 0.25 f1 0.96224; o8 1.25 E - 0.25 T **0.96225**; o9 mean(E,S,T,M) 0.96224; o10 E 80 + S 20 **0.96225**.
- FINAL SELECTION (mark by hand on the Submissions page, "Use for Final Score"):
  slot 1 = stack v10 `subs/v10/submission.csv` (submission described "stack v10", CV 0.962243, public 0.96183);
  slot 2 = `subs/b31/o4_E50_S50.csv` (public 0.96225: mean of the two top public files, LB-fitted, no CV).
  The user confirmed on 10-31 that both are marked as final.

## RESULT (private LB, 2026-11-01, preliminary)

- Our rank **1047 / 3747**, counted private score **0.96212** (the o-series LB-surface file). Top private 0.96318
  (Chris Deotte); public leaders (0.964+) fell out of the top.
- Private scores of our own files: stack v10 **0.96238** (rank ~100 on the private LB, where 0.96238 sits at rank 100);
  best of all 116 subs 0.96239 (h5 v10 90 + Q-FT 10, f7 v10 + pillar1); every honest CV stack 0.96237-0.96238;
  f1 0.96223, i4-family 0.96225-0.96227, tk223 0.96213, o4 0.96212, Q-FT alone 0.96165.
- So the public-LB-fitted files lost ~0.00025 on private, exactly the risk noted all along, and the honest CV stack won.
  The counted score shows stack v10 was not among the two final selections (or Kaggle used other marked files).
- Lessons: trust nested CV over public digits; LB-surface files (built by fitting public scores) shake down;
  confirm the selected pair on the Submissions page (both ticks) before the deadline.

## Ideas not yet tried

- Chris Deotte discussion 749010: fine-tuned TFMs gain +1e-4 to +13e-4 as single models (TabPFN-3.5,
  Kumo large, TabICLv2 ...), but cost 5-37 GPU hours each. Watch `cdeotte/s6e10-tfm-oof-and-test-predictions`
  for an update with FE / fine-tuned OOFs and add them to the pool the moment they appear.
- Kumo-Tabular large + FE (CV 0.961105 per Chris) as another member, if the model is available on Kaggle.

- Read the competition Discussion tab for what the two leaders (0.96243 / 0.96234) found.
- ewtherka "L2 family-wise actually helps" (LB 0.96188) and rabari9999 roadmap 0.96188: compare their pools
  to the 245 pool and add any members it lacks.
- Fine-tuned tabular foundation model (TabPFN / TabICL fine-tuning on GPU) as a genuinely new family.
