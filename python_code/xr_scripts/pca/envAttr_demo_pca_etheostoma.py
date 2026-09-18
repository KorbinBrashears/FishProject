# program		envAttr_demo_pca_etheostoma
# purpose	    PCA to ANOVA for environmental attributes to find signficant DRIVERS 
# usage         script
# notes         (1) this script adds abundance gradient to env. PCA plot for genus (top 4 labeled)
#               (2) prints PC1 and PC2 loadings
#               (3) does 1-way ANOVA, outputs scores for PC1/2 F(1,44) = [], p = []
#               (4) bar graph based on lower ANOVA p-value (>0.001-0.01-0.05-0.10<)
#               (5) ANOVA finds association of genus abundance and suggested PC drivers
#               (6) can run a post-hoc test (ex. Tukey's HSD) if p < .05 
# date			01/28/2026
# programmer    Xavier Ramirez

import datetime
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm
from statsmodels.formula.api import ols

# ============== COMMON INITIALIZATION =====================
date_o = datetime.datetime.today()
date_c = date_o.strftime('%m/%d/%Y')
programname_c = os.path.basename(__file__)
programName_c = os.path.dirname(os.path.abspath(__file__))
script_name = os.path.splitext(os.path.basename(__file__))[0]

fileName_c = 'data/seth_gov_envData10dtsMetrics_genusCount2026.csv'
programMsg_c = programName_c + ' (' + date_c + ')'
authorName_c = 'Xavier Ramirez'

fig_dir = os.path.abspath(os.path.join(programName_c, "..", "..", "..", "figures")) #NEW
os.makedirs(fig_dir, exist_ok=True) #NEW
figName_c = os.path.join(fig_dir, f"{script_name}_fig.png") #NEW

# ============== 0. Load and preprocess data ===============
df = pd.read_csv(fileName_c)                # load dataset
df.columns = df.columns.str.strip()         # clean column names
genus_col = "Etheostoma"                      # target genus for script

# ============== 1. Select environmental columns ===========
env_cols = df.columns[4:25]                 # 4:25 for environmental attributes
X = df[env_cols]                            # matrix for PCA input

# ============== 2. Select environmental columns ===========
scaler = StandardScaler()                   # standardize environmental attributes
X_scaled = scaler.fit_transform(X)          # ensures PCA is not dominated by large-scale variables

# ============== 3. Standardize the environmental variables =
pca = PCA()                                 # initialize PCA model
pca.fit(X_scaled)                           # fit PCA to standardized data

# ============== 4. Fit PCA (all components) ===============
pc_scores = pca.transform(X_scaled)         # compute each site's pc scores
df["PC1"] = pc_scores[:, 0]                 # store PC1 scores
df["PC2"] = pc_scores[:, 1]                 # store PC2 scores

# ============== 5. PCA loadings table =====================
loadings = pd.DataFrame(
    pca.components_.T,
    index=env_cols,
    columns=[f"PC{i+1}" for i in range(len(env_cols))])
print(loadings.iloc[:, :2])                 # print loadings for PC1–PC2

# ============== 6. One-Way ANOVA ==========================
def p_to_stars(p):                          # shows significance as stars
    if p < 0.001:
        return "***"                        # greatly statistically signficant
    elif p < 0.01:
        return "**"                         # statistically signficant
    elif p < 0.05:
        return "*"                          # barely signficant
    elif p < 0.10:
        return "·"                          # borderline suggestive
    else:
        return ""                           # no meaningful association
def p_to_phrase(p):                         # creates strings based on above comments
    if p < 0.001:
        return "Highly significant"
    elif p < 0.01:
        return "Significant"
    elif p < 0.05:
        return "Barely significant"
    elif p < 0.10:
        return "Borderline suggestive"
    else:
        return "No evidence, but"
model_pc1 = ols(f'{genus_col} ~ PC1', data=df).fit()
anova_pc1 = sm.stats.anova_lm(model_pc1, typ=2)
print("\nANOVA for abundance ~ PC1")
print(anova_pc1)
p_pc1 = anova_pc1["PR(>F)"].iloc[0]         # p-value for PC1
F_pc1 = anova_pc1["F"].iloc[0]              # f-score for PC1
stars_pc1 = p_to_stars(p_pc1)               # convert to stars

model_pc2 = ols(f'{genus_col} ~ PC2', data=df).fit()
anova_pc2 = sm.stats.anova_lm(model_pc2, typ=2)
print("\nANOVA for abundance ~ PC2")
print(anova_pc2)
p_pc2 = anova_pc2["PR(>F)"].iloc[0]         # p-value for PC2
F_pc2 = anova_pc2["F"].iloc[0]              # f-score for PC2
stars_pc2 = p_to_stars(p_pc2)               # convert to stars

# ============== 7. PCA plot colored by genus abundance =====
abundance = df[genus_col]                   # abundance for color gradient
top_sites = df.sort_values(by=genus_col, ascending=False).head(4) # txt top # populous sites

plt.figure(figsize=(12,6))                  # creates a nicely-sized figure
plt.subplot(121)
plt.scatter(
    pc_scores[:, 0],
    pc_scores[:, 1],
    c=abundance,
    cmap="viridis",
    s=50,
    alpha=0.8)
# ---- Residual-based outlier detection 
model_used = model_pc1 if p_pc1 <= p_pc2 else model_pc2 # choose  ANOVA model w/ stronger signal
residuals = model_used.resid
threshold = 2 * residuals.std()             # threshold: 2 standard deviations
outlier_mask = np.abs(residuals) > threshold # identify outlier sites
outliers = df[outlier_mask]

for i, row in outliers.iterrows():          # plot red dashed circles around outlier sites
    plt.scatter(row["PC1"], row["PC2"],
                facecolors='none', edgecolors='red',
                s=300, linewidths=1.5, linestyle='--',
                label="Residual outlier" if i == outliers.index[0] else None)
plt.legend(loc="upper left", fontsize=8)
plt.colorbar(label=f"{genus_col} abundance") # abundance legend
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
plt.title(f"PCA Colored by {genus_col} Abundance")
for i, row in top_sites.iterrows():         # runs top_sites onto subplot 121
    site_label = f"Site {i}"
    plt.text(
        row["PC1"] + 0.2,
        row["PC2"] + 0.2,
        site_label,
        fontsize=8,
        color="black",
        weight="bold")

# ============== 8. PCA loadings in +/- box plot ===========
if p_pc1 < 0.10 or p_pc2 < 0.10:
    pc_to_use = 0 if p_pc1 <= p_pc2 else 1  # choose PC with lower p-value
else:
    pc_to_use = 0                           # neither meaningful, default to PC1
p_selected = p_pc1 if pc_to_use == 0 else p_pc2 # for title of bar graph
stars = p_to_stars(p_selected)              # star notation for title
phrase = p_to_phrase(p_selected)            # phrase for title based on p-value

loadings_vec = pca.components_[pc_to_use]   # loadings for chosen PC
attributes = env_cols                       # environmental variable names
scores = df[f"PC{pc_to_use+1}"].values      # PC scores for direction test
abundances = df[genus_col].values           # abundance vector

abund_pos = abundances[scores > 0]          # split abundance by sign of PC score
abund_neg = abundances[scores < 0]
mean_pos = abund_pos.mean() if len(abund_pos) > 0 else 0
mean_neg = abund_neg.mean() if len(abund_neg) > 0 else 0
direction = "positive" if mean_pos > mean_neg else "negative"

sorted_idx = np.argsort(np.abs(loadings_vec))[::-1] # sort loading strength
sorted_loadings = loadings_vec[sorted_idx][::-1] # reversed for descending
sorted_attrs = attributes[sorted_idx][::-1] # reversed for descending
colors = ['#1f77b4' if val > 0 else '#d62728' for val in sorted_loadings]

plt.subplot(122)
plt.barh(sorted_attrs, sorted_loadings, color=colors) # horizontal barplot
plt.axvline(0, color='black', linewidth=1)  # zero line for reference
plt.title(
    f"PC{pc_to_use+1} Environmental Drivers {stars}\n"
    f"{phrase} toward the {direction} end")
plt.xlabel("Loading Strength")
plt.tight_layout()

# ============== Make the subplots look a little nicer =====
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.13, wspace=0.5, hspace=0.3)

# ============== Label plot edges ==========================
plt.subplot(position=[0.0500,    0.93,    0.02500,    0.02500]) # U-left
plt.axis('off')
plt.text(0,.5, programMsg_c, fontsize=7)

plt.subplot(position=[0.550,    0.93,    0.02500,    0.02500]) # U-right
plt.axis('off')
plt.text(0,.5, authorName_c, fontsize=7)

plt.subplot(position=[0.0500,    0.02,    0.02500,    0.02500]) # L-left
plt.axis('off')
plt.text(0,.5, fileName_c, fontsize=7)

plt.subplot(position=[0.4500,    0.02,    0.02500,    0.02500]) # L-right
plt.axis('off')
plt.text(0, .5,
    f"PC1: F(1,44) = {F_pc1:.2f}, p = {p_pc1:.4f}{stars_pc1}\n"
    f"PC2: F(1,44) = {F_pc2:.2f}, p = {p_pc2:.4f}{stars_pc2}", fontsize=9)
plt.savefig(figName_c)
plt.close()