"""
Generate all three paper figures/tables:
  Figure 1  – Provider-level feature mention bar chart
  Figure 2  – Feature mention rate heatmap (sex & age)
  Table 1   – Faithfulness metrics by provider and demographic group (LaTeX)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from pathlib import Path

# ── styling ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

OUTPUT = Path("results/figures")
OUTPUT.mkdir(parents=True, exist_ok=True)

# ── load data ─────────────────────────────────────────────────────────────────
import json

df = pd.read_csv(
    "results/shap_metrics/sex/per_instance_all_metrics_credit_20260429_212427.csv"
)

df_feat_sex = pd.read_excel(
    "results/shap_metrics/sex/statistical_tests_feature_mentions_CORRECTED.xlsx",
    sheet_name="All Results",
)
df_feat_age = pd.read_excel(
    "results/shap_metrics/age/statistical_tests_feature_mentions_age_AGGREGATED.xlsx",
    sheet_name="All Results",
)

PROVIDERS = ["claude", "deepseek", "gemini", "grok", "mistral", "openai"]
PROVIDER_LABELS = ["Claude", "DeepSeek", "Gemini", "Grok", "Mistral", "OpenAI"]

# Compute correct non-SHAP and protected attr averages from raw extractions
protected_attrs = {"sex", "age", "foreign_worker"}
non_shap_avgs = {}
protected_avgs = {}

for prov in PROVIDERS:
    non_shap_counts = []
    prot_counts = []
    for idx in range(34):
        gt = json.load(open(f"results/ground_truth/credit/instance_{idx}.json"))
        shap_names = {f["name"] for f in gt["most_important_features"]}
        try:
            ext = json.load(open(f"results/extractions/majority/{prov}/instance_{idx}.json"))
        except FileNotFoundError:
            continue
        non_shap_counts.append(sum(
            1 for f in ext.get("features", [])
            if f.get("mentioned") == 1
            and f["name"] not in shap_names
            and f["name"] not in protected_attrs
        ))
        prot_counts.append(sum(
            1 for f in ext.get("features", [])
            if f.get("mentioned") == 1
            and f["name"] in protected_attrs
        ))
    non_shap_avgs[prov] = np.mean(non_shap_counts) if non_shap_counts else 0
    protected_avgs[prov] = np.mean(prot_counts) if prot_counts else 0

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 – Provider-level feature mention bar chart
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Figure 1 ...")

agg_non_shap  = [non_shap_avgs[p]  for p in PROVIDERS]
agg_protected = [protected_avgs[p] for p in PROVIDERS]

x = np.arange(len(PROVIDERS))
w = 0.38

fig, ax = plt.subplots(figsize=(7, 4))
bars1 = ax.bar(x - w / 2, agg_non_shap, w, label="Non-SHAP features mentioned",
               color="#9999ff", edgecolor="black", linewidth=0.7)
bars2 = ax.bar(x + w / 2, agg_protected, w,
               label="Protected attributes mentioned", color="#ccccff", edgecolor="black", linewidth=0.7)

ax.set_xticks(x)
ax.set_xticklabels(PROVIDER_LABELS)
ax.set_ylabel("Average per narrative")
ax.set_title("Provider-level narrative feature coverage")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

# annotate values on bars
for bar in bars1:
    if bar.get_height() > 0:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
for bar in bars2:
    if bar.get_height() > 0:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
fig.savefig(OUTPUT / "fig1_provider_coverage.pdf", bbox_inches="tight")
fig.savefig(OUTPUT / "fig1_provider_coverage.png", dpi=600, bbox_inches="tight")
plt.close()
print("  → saved fig1_provider_coverage.pdf/png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 – Feature mention heatmap (sex and age, side by side)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Figure 2 ...")

# Features to show – those with at least one noteworthy difference
FEATURES = [
    "people_liable", "number_credits", "housing",
    "employment_duration", "credit_history", "duration",
    "other_debtors", "property",
]
FEAT_LABELS = [
    "People liable", "Number of credits", "Housing",
    "Employment duration", "Credit history", "Loan duration",
    "Other debtors", "Property",
]

# Build sex matrix  ── male%, female%
sex_rows = df_feat_sex.set_index("feature")
sex_data = pd.DataFrame(index=FEATURES, columns=["Male", "Female"])
for f in FEATURES:
    if f in sex_rows.index:
        m = float(str(sex_rows.loc[f, "male_mean"]).replace("%", ""))
        fe = float(str(sex_rows.loc[f, "female_mean"]).replace("%", ""))
        sex_data.loc[f] = [m, fe]
    else:
        sex_data.loc[f] = [np.nan, np.nan]

# p-values for sex
sex_pvals = {}
for f in FEATURES:
    if f in sex_rows.index:
        sex_pvals[f] = sex_rows.loc[f, "p_value"]
    else:
        sex_pvals[f] = 1.0

# Build age matrix  ── young%, old%
age_rows = df_feat_age.set_index("Feature")
age_data = pd.DataFrame(index=FEATURES, columns=["Young\n(<32)", "Old\n(≥32)"])
for f in FEATURES:
    if f in age_rows.index:
        age_data.loc[f] = [age_rows.loc[f, "Young %"], age_rows.loc[f, "Old %"]]
    else:
        age_data.loc[f] = [np.nan, np.nan]

age_pvals = {}
for f in FEATURES:
    if f in age_rows.index:
        age_pvals[f] = age_rows.loc[f, "p-value"]
    else:
        age_pvals[f] = 1.0

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(9.5, 5), sharey=True)
fig.subplots_adjust(left=0.18, right=0.88, wspace=0.05)

cmap = mcolors.LinearSegmentedColormap.from_list(
    "tikz_blue", ["#ffffff", "#9999ff"]  # white -> TikZ blue!40
)
norm = mcolors.Normalize(vmin=0, vmax=100)

def draw_heatmap(ax, data, pvals, title):
    mat = data.values.astype(float)
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns)
    ax.set_yticks(range(len(FEAT_LABELS)))
    ax.set_yticklabels(FEAT_LABELS)
    ax.set_title(title, pad=8)

    for i, feat in enumerate(FEATURES):
        for j in range(len(data.columns)):
            val = mat[i, j]
            if np.isnan(val):
                continue
            p = pvals.get(feat, 1.0)
            star = "**" if p < 0.01 else ("*" if p < 0.05 else "")
            txt = f"{val:.2f}%"
            if star and j == 0:          # annotate star only once (left column)
                txt = f"{val:.2f}%{star}"
            color = "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color=color, fontweight="bold" if star else "normal")
    return im

im = draw_heatmap(axes[0], sex_data, sex_pvals, "By Sex")
im = draw_heatmap(axes[1], age_data, age_pvals, "By Age Group")

# shared colorbar – placed explicitly to avoid overlap
cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
cb = fig.colorbar(im, cax=cbar_ax)
cb.set_label("Mention rate (%)")

fig.suptitle("Mention rates of non-top-SHAP features across demographic groups\n(* p<0.05, ** p<0.01, aggregated across all providers)",
             fontsize=10, y=1.01)

plt.savefig(OUTPUT / "fig2_feature_heatmap.pdf", bbox_inches="tight")
plt.savefig(OUTPUT / "fig2_feature_heatmap.png", dpi=600, bbox_inches="tight")
plt.close()
print("  → saved fig2_feature_heatmap.pdf/png")

# ─────────────────────────────────────────────────────────────────────────────
# TABLE 1 – Faithfulness metrics by provider and demographic group (LaTeX)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Table 1 (LaTeX) ...")

metrics = ["rank_1_agreement", "rank_2_agreement", "rank_3_agreement",
           "sign_agreement_mean", "all_value_agreement_mean"]
metric_labels = ["R1", "R2", "R3", "Sign", "Value"]

rows = []
for prov in PROVIDERS:
    sub = df[df["provider"] == prov]
    male   = sub[sub["sex"] == "male"]
    female = sub[sub["sex"] == "female"]
    young  = sub[sub["age_group"] == "young"]
    old    = sub[sub["age_group"] == "old"]

    row = {"Provider": prov.capitalize()}
    for m, ml in zip(metrics, metric_labels):
        row[f"{ml}_M"]  = male[m].mean()
        row[f"{ml}_F"]  = female[m].mean()
        row[f"{ml}_Y"]  = young[m].mean()
        row[f"{ml}_O"]  = old[m].mean()
    rows.append(row)

# 4-way intersectional rank_2 row
groups = {
    "M-Y": df[df["group_4way"] == "male_young"]["rank_2_agreement"].mean(),
    "M-O": df[df["group_4way"] == "male_old"]["rank_2_agreement"].mean(),
    "F-Y": df[df["group_4way"] == "female_young"]["rank_2_agreement"].mean(),
    "F-O": df[df["group_4way"] == "female_old"]["rank_2_agreement"].mean(),
}

tbl = pd.DataFrame(rows).set_index("Provider")

# ── render LaTeX ──────────────────────────────────────────────────────────────
header_sex = " & ".join([f"\\multicolumn{{2}}{{c}}{{{ml}}}" for ml in metric_labels])
subhead = " & ".join(["M & F"] * len(metrics))

lines = []
lines.append(r"\begin{table*}[ht]")
lines.append(r"\centering")
lines.append(r"\small")
lines.append(r"\caption{Faithfulness metrics (\%) by provider and demographic group. "
             r"M=male, F=female, Y=young ($<32$), O=old ($\geq32$). "
             r"Bold values differ significantly from their counterpart ($p{<}0.05$, Mann-Whitney U).}")
lines.append(r"\label{tab:faithfulness}")
lines.append(r"\begin{tabular}{l" + "rr" * len(metrics) + "}")
lines.append(r"\toprule")
lines.append(r"\textbf{Provider} & " + " & ".join(
    [f"\\multicolumn{{2}}{{c}}{{\\textbf{{{ml}}}}}" for ml in metric_labels]) + r" \\")
lines.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}")
lines.append(r" & " + " & ".join(["M & F"] * len(metrics)) + r" \\")
lines.append(r"\midrule")

# sex rows
for prov in PROVIDERS:
    row = tbl.loc[prov.capitalize()]
    cells = []
    for ml in metric_labels:
        m_val = row[f"{ml}_M"]
        f_val = row[f"{ml}_F"]
        cells.append(f"{m_val:.0f}")
        cells.append(f"{f_val:.0f}")
    lines.append(f"\\textit{{{prov.capitalize()}}} & " + " & ".join(cells) + r" \\")

lines.append(r"\midrule")
lines.append(r"\textbf{Mean} & " + " & ".join([
    f"{tbl[f'{ml}_M'].mean():.0f} & {tbl[f'{ml}_F'].mean():.0f}"
    for ml in metric_labels
]).replace(" & ", " & ") + r" \\")

lines.append(r"\midrule")
lines.append(r"\multicolumn{" + str(1 + 2 * len(metrics)) + r"}{l}{\textit{Age comparison (Young / Old)}} \\")
lines.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}")
lines.append(r" & " + " & ".join(["Y & O"] * len(metrics)) + r" \\")

for prov in PROVIDERS:
    row = tbl.loc[prov.capitalize()]
    cells = []
    for ml in metric_labels:
        cells.append(f"{row[f'{ml}_Y']:.0f}")
        cells.append(f"{row[f'{ml}_O']:.0f}")
    lines.append(f"\\textit{{{prov.capitalize()}}} & " + " & ".join(cells) + r" \\")

lines.append(r"\midrule")
lines.append(r"\textbf{Mean} & " + " & ".join([
    f"{tbl[f'{ml}_Y'].mean():.0f} & {tbl[f'{ml}_O'].mean():.0f}"
    for ml in metric_labels
]) + r" \\")

lines.append(r"\midrule")
lines.append(r"\multicolumn{" + str(1 + 2 * len(metrics)) + r"}{l}{\textit{Rank-2 agreement by intersectional subgroup}} \\")
lines.append(r" & \multicolumn{" + str(2 * len(metrics)) + r"}{l}{"
             + f"M-Y: {groups['M-Y']:.0f}\\%,  M-O: {groups['M-O']:.0f}\\%,  "
             + f"F-Y: {groups['F-Y']:.0f}\\%,  \\textbf{{F-O: {groups['F-O']:.0f}\\%}}"
             + r"} \\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table*}")

latex_str = "\n".join(lines)

with open(OUTPUT / "table1_faithfulness.tex", "w") as f:
    f.write(latex_str)

print("  → saved table1_faithfulness.tex")
print()
print("── LaTeX snippet ──────────────────────────────────────────────────────")
print(latex_str)
print("───────────────────────────────────────────────────────────────────────")
print()
# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 – R2 faithfulness by provider × sex and age (grouped bar chart)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Figure 3 ...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
fig.subplots_adjust(wspace=0.08)

demos = [
    ("sex",       ["male",  "female"], ["Male",        "Female"    ], ["#9999ff", "#ccccff"]),
    ("age_group", ["young", "old"],    ["Young (<32)", "Old (≥32)" ], ["#9999ff", "#ccccff"]),
]

for ax, (col, groups, glabels, colors) in zip(axes, demos):
    x = np.arange(len(PROVIDERS))
    w = 0.38
    vals0 = [df[(df["provider"] == p) & (df[col] == groups[0])]["rank_2_agreement"].mean()
             for p in PROVIDERS]
    vals1 = [df[(df["provider"] == p) & (df[col] == groups[1])]["rank_2_agreement"].mean()
             for p in PROVIDERS]
    bars0 = ax.bar(x - w / 2, vals0, w, label=glabels[0],
                   color=colors[0], edgecolor="black", linewidth=0.7)
    bars1 = ax.bar(x + w / 2, vals1, w, label=glabels[1],
                   color=colors[1], edgecolor="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(PROVIDER_LABELS)
    ax.set_ylim(0, 115)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    for bar in list(bars0) + list(bars1):
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                    f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7)

axes[0].set_ylabel("R2 Rank Agreement (%)")
axes[0].set_title("By Sex")
axes[1].set_title("By Age Group")
fig.suptitle("Top-2 SHAP rank agreement by provider and demographic group",
             fontsize=11, y=1.02)

plt.tight_layout()
fig.savefig(OUTPUT / "fig3_faithfulness_r2.pdf", bbox_inches="tight")
fig.savefig(OUTPUT / "fig3_faithfulness_r2.png", dpi=600, bbox_inches="tight")
plt.close()
print("  → saved fig3_faithfulness_r2.pdf/png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 – Intersectional faithfulness (4-way subgroups)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Figure 4 ...")

group_keys   = ["male_young", "male_old", "female_young", "female_old"]
group_labels = ["Male–Young", "Male–Old", "Female–Young", "Female–Old"]
metrics_4   = ["rank_1_agreement", "rank_2_agreement", "rank_3_agreement"]
mlabels_4   = ["R1", "R2", "R3"]
colors_4    = ["#6666ee", "#9999ff", "#ccccff"]

n_m, n_g = len(metrics_4), len(group_keys)
total_w = 0.65
bar_w   = total_w / n_m
offsets = np.linspace(-(total_w - bar_w) / 2, (total_w - bar_w) / 2, n_m)
x = np.arange(n_g)

fig, ax = plt.subplots(figsize=(8, 4.5))

for i, (met, lab, col) in enumerate(zip(metrics_4, mlabels_4, colors_4)):
    vals = [df[df["group_4way"] == g][met].mean() for g in group_keys]
    bars = ax.bar(x + offsets[i], vals, bar_w, label=lab,
                  color=col, edgecolor="black", linewidth=0.7)
    for bar in bars:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                    f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7.5)

ax.set_xticks(x)
ax.set_xticklabels(group_labels)
ax.set_ylabel("Agreement (%)")
ax.set_ylim(0, 115)
ax.legend(frameon=False, title="Metric", title_fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
ax.set_title(
    "Faithfulness metrics by intersectional subgroup\n"
    "(Kruskal-Wallis R2: H=8.22, p=0.042)",
    fontsize=11
)

# annotate the F-O drop
fo_r2 = df[df["group_4way"] == "female_old"]["rank_2_agreement"].mean()
ax.annotate(
    "Significantly\nlower (p=0.042)",
    xy=(3 + offsets[1], fo_r2 + 1.5),
    xytext=(2.55, 72),
    arrowprops=dict(arrowstyle="->", color="black", lw=1),
    fontsize=8, ha="center", color="black",
)

plt.tight_layout()
fig.savefig(OUTPUT / "fig4_intersectional.pdf", bbox_inches="tight")
fig.savefig(OUTPUT / "fig4_intersectional.png", dpi=600, bbox_inches="tight")
plt.close()
print("  → saved fig4_intersectional.pdf/png")

print(f"All outputs written to: {OUTPUT.resolve()}")
