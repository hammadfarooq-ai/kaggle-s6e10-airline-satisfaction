"""Build the README figures from results/submissions_scores.csv (our own 116 submissions).

Run from the repository root:  python src/analysis/make_figures.py
Writes PNGs to results/figures/.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/figures"
os.makedirs(OUT, exist_ok=True)

# Palette: validated categorical order (blue, orange, aqua, yellow) on a light surface.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
})

d = pd.read_csv("results/submissions_scores.csv")
d = d[d.status.str.contains("COMPLETE")].copy()
d["publicScore"] = pd.to_numeric(d.publicScore, errors="coerce")
d["privateScore"] = pd.to_numeric(d.privateScore, errors="coerce")
d = d.dropna(subset=["publicScore", "privateScore"])
d["date"] = pd.to_datetime(d.date)
d["description"] = d.description.fillna("")


def kind(r):
    f, s = r.fileName, r.description
    if r.publicScore < 0.9600 or f.startswith("h1_"):
        return "Single model"
    if "nested CV" in s or "stack v10" in s or f.startswith(("h5_", "h6_", "f7_", "f10_")):
        return "CV stack"
    if f.startswith(("b", "blend_", "submission_twin")) and "surface" not in s:
        return "Stack + older public stacks"
    return "Public-LB surface blend"


d["kind"] = d.apply(kind, axis=1)
STYLE = {  # color, marker
    "CV stack": (BLUE, "o"),
    "Stack + older public stacks": (AQUA, "s"),
    "Public-LB surface blend": (ORANGE, "^"),
    "Single model": (YELLOW, "D"),
}

# ---------------------------------------------------------------- 1. public vs private
p = d[d.publicScore > 0.9615]
fig, ax = plt.subplots(figsize=(9, 6.2), dpi=150)
for k, (c, m) in STYLE.items():
    s = p[p.kind == k]
    if len(s):
        ax.scatter(s.publicScore, s.privateScore, s=46, c=c, marker=m, edgecolor=SURFACE, linewidth=1.5,
                   label=f"{k} ({len(s)})", zorder=3)
v10 = p[p.description.str.contains("stack v10")].iloc[0]
o4 = p[p.fileName == "o4_E50_S50.csv"].iloc[0]
ax.annotate("stack v10 (best CV)\npublic 0.96183 -> private 0.96238\n= private rank ~100",
            (v10.publicScore, v10.privateScore), xytext=(0.96186, 0.96222), fontsize=9.5, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
ax.annotate("o4 (best public)\npublic 0.96225 -> private 0.96212\n= private rank 1047 (counted)",
            (o4.publicScore, o4.privateScore), xytext=(0.96189, 0.96206), fontsize=9.5, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
ax.set_xlabel("Public LB AUC (20% of test)")
ax.set_ylabel("Private LB AUC (80% of test)")
fig.text(0.02, 0.965, "Higher public score, lower private score", fontsize=14, fontweight="bold")
fig.text(0.02, 0.93, "Each point is one of our submissions (single-model baseline at 0.958 not shown)", fontsize=9.5, color=INK2)
ax.legend(frameon=False, loc="upper right", fontsize=9.5)
ax.ticklabel_format(useOffset=False)
fig.tight_layout(rect=(0, 0, 1, 0.91))
fig.savefig(f"{OUT}/public_vs_private.png")
plt.close(fig)

# ---------------------------------------------------------------- 2. timeline
# per day: the submission with the best PUBLIC score (the one a public-chaser would pick) and its private score
t = d[d.publicScore > 0.9615].copy()
t["day"] = t.date.dt.normalize()
t = t.sort_values(["day", "publicScore", "privateScore"]).groupby("day").tail(1)
fig, ax = plt.subplots(figsize=(10, 4.8), dpi=150)
ax.plot(t.day, t.publicScore, color=ORANGE, lw=2, marker="^", ms=7, mec=SURFACE, label="Public LB of the day's best-public file")
ax.plot(t.day, t.privateScore, color=BLUE, lw=2, marker="o", ms=7, mec=SURFACE, label="Private LB of that same file")
ax.axhline(0.96238, color=BLUE, lw=1, ls=(0, (4, 3)))
ax.text(t.day.iloc[-1], 0.962388, "stack v10 private 0.96238", fontsize=9, color=INK2, va="bottom", ha="right")
ax.set_ylabel("AUC")
fig.text(0.02, 0.955, "Chasing the public board cost private score", fontsize=14, fontweight="bold")
fig.text(0.02, 0.91, "Each day's best-public submission; the two lines cross on Oct 29", fontsize=9.5, color=INK2)
ax.legend(frameon=False, loc="lower right", fontsize=9.5)
ax.ticklabel_format(axis="y", useOffset=False)
fig.autofmt_xdate()
fig.tight_layout(rect=(0, 0, 1, 0.89))
fig.savefig(f"{OUT}/timeline.png")
plt.close(fig)

# ---------------------------------------------------------------- 3. stack CV progression
steps = [
    ("v1  LR on 202 public OOF", 0.962171),
    ("+ 7 koumeimaki members (209)", 0.962181),
    ("0.8 LR + 0.2 LGB stacker", 0.962187),
    ("v8  245 pool, family-wise L2", 0.962230),
    ("v9  + LGB stacker blend", 0.962242),
    ("v10 + TabPFN ne4 (246)", 0.962243),
]
fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
y = np.arange(len(steps))[::-1]
vals = np.array([v for _, v in steps])
ax.hlines(y, vals.min() - 0.00001, vals, color=GRID, lw=2)
ax.scatter(vals, y, s=70, c=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
for yi, (lab, v) in zip(y, steps):
    ax.text(v + 0.0000015, yi, f"{v:.6f}", va="center", fontsize=9.5, color=INK)
ax.set_yticks(y, [s for s, _ in steps])
ax.set_xlim(vals.min() - 0.00001, vals.max() + 0.000012)
ax.set_xlabel("Nested CV AUC of the stack")
fig.text(0.02, 0.95, "Stack versions: most of the gain came from family-wise L2", fontsize=13, fontweight="bold")
ax.grid(axis="y", visible=False)
ax.ticklabel_format(axis="x", useOffset=False)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(f"{OUT}/stack_progression.png")
plt.close(fig)

# ---------------------------------------------------------------- 4. candidate members screened
cand = [  # name, mean gain over seeds 7/13 (x1e-6), kept?
    ("TabPFN-3.5 ne4 (goodpjw2008)", 2.2, True),
    ("RealMLP original (masumhuq)", 1.0, False),
    ("Own Q-FT v2 transformer", 0.2, False),
    ("srim LightGBM", -0.3, False),
    ("denpugovkin RealMLP hybrid", -0.8, False),
    ("najiama XGBoost", -1.0, False),
    ("srim survey mean feature", -1.1, False),
    ("TabResNet (masumhuq)", -1.2, False),
    ("Own TabICL v2 + FE", -2.2, False),
    ("MLP stacker (best mix)", -5.1, False),
]
fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
y = np.arange(len(cand))[::-1]
g = np.array([c[1] for c in cand])
ax.barh(y, g, color=[BLUE if c[2] else ORANGE for c in cand], height=0.6, edgecolor=SURFACE, linewidth=2)
for yi, (n, v, k) in zip(y, cand):
    ax.text(v + (0.15 if v >= 0 else -0.15), yi, f"{v:+.1f}" + ("  kept" if k else ""), va="center",
            ha="left" if v >= 0 else "right", fontsize=9.5, color=INK)
ax.axvline(0, color=INK2, lw=1)
ax.set_yticks(y, [c[0] for c in cand])
ax.set_xlim(-6.8, 3.8)
ax.set_xlabel("Change in nested CV AUC (x 1e-6, mean of meta seeds 7 and 13)")
fig.text(0.02, 0.955, "Candidates screened against the 245/246-member stack", fontsize=13, fontweight="bold")
fig.text(0.02, 0.91, "Kept only if both meta seeds improve; only TabPFN ne4 passed", fontsize=9.5, color=INK2)
ax.grid(axis="y", visible=False)
fig.tight_layout(rect=(0, 0, 1, 0.89))
fig.savefig(f"{OUT}/candidate_screen.png")
plt.close(fig)

print(d.kind.value_counts().to_string())
print("figures written to", OUT)
