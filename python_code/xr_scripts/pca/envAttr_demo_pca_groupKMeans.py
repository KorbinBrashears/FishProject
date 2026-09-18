# program		envAttr_demo_pca_groupKMeans.py
# purpose	    Using PCA->Kmeans to find groupings and clusterings
# usage         script
# notes         (1) 
# date			02/16/2026
# programmer    Xavier Ramirez

import datetime
import os
os.environ["OMP_NUM_THREADS"] = "1"
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib.patches import Ellipse
import seaborn as sns
from scipy.stats import chi2_contingency

# ============== COMMON INITIALIZATION =====================
date_o = datetime.datetime.today()
date_c = date_o.strftime('%m/%d/%Y')
programName_c = os.path.dirname(os.path.abspath(__file__)) #NEW
script_name = os.path.splitext(os.path.basename(__file__))[0] #NEW
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_seq_items", None)

fileName_c = 'data/seth_gov_envData10dtsMetrics_genusCount2026.csv'
programMsg_c = script_name + ' (' + date_c + ')' #changed from programName_c to script_name
authorName_c = 'Xavier Ramirez'

fig_dir = os.path.abspath(os.path.join(programName_c, "..", "..", "..", "figures")) #NEW
os.makedirs(fig_dir, exist_ok=True) #NEW
figName_c = os.path.join(fig_dir, f"{script_name}_fig.png") #NEW

# ============== 0. Load and preprocess data ===============
df = pd.read_csv(fileName_c)                # load dataset
df.columns = df.columns.str.strip()         # clean column names

# ============== 1. Select environmental columns ===========
env_cols = df.columns[4:25]                 # columns 4–25 are environmental
X_env = df[env_cols]

# ============== 2. Standardize the data ===================
scaler = StandardScaler()                   # standardize environmental attributes
X_env_scaled = scaler.fit_transform(X_env)  # ensures PCA is not dominated by large-scale variables

# ============== 3. Standardize the environmental variables =
pca_env = PCA()                             # initialize PCA model
pca_env.fit(X_env_scaled)                   # fit PCA to standardized data

# ============== 4. Fit PCA (all components) ===============
pc_scores = pca_env.transform(X_env_scaled) # compute each site's pc scores
df["PC1"] = pc_scores[:, 0]                 # store PC1 scores
df["PC2"] = pc_scores[:, 1]                 # store PC2 scores

plt.figure(figsize=(12,8))                  # creates a nicely-sized figure
# ============== 5. PC1 vs PC2 scatter plot (231)===========
pc_scores = pca_env.transform(X_env_scaled)
plt.subplot(231)
plt.scatter(pc_scores[:, 0], pc_scores[:, 1], alpha=0.7)
plt.xlabel(f"PC1 ({pca_env.explained_variance_ratio_[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({pca_env.explained_variance_ratio_[1]*100:.1f}% var)")
plt.title("Environmental PCA (PC1 vs PC2)")
plt.tight_layout()

# Cluster sites based on PC1 and PC2
cluster_colors = {
    0: 'orange',
    1: 'green',
    2: 'red'}
kmeans = KMeans(n_clusters=3, random_state=42).fit(pc_scores[:, :2])
df["EnvCluster"] = kmeans.labels_
# print sites in each cluster and their states
print("\nFull SiteN lists for EnvCluster:")
for cluster_id, sites in df.groupby("EnvCluster")["SiteN"].unique().items():
    print(f"EnvCluster {cluster_id}: {list(sites)}")

print("\nState counts for EnvCluster:")
print(df.groupby("EnvCluster")["State"].value_counts())

# Plot with cluster colors
plt.subplot(231)
for cluster_id in np.unique(df["EnvCluster"]):
    cluster_points = pc_scores[df["EnvCluster"] == cluster_id]
    color = cluster_colors[cluster_id]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                label=f"EnvCluster {cluster_id}", alpha=0.7, color=color)
    center = cluster_points.mean(axis=0)
    width = cluster_points[:, 0].ptp() * 1.2
    height = cluster_points[:, 1].ptp() * 1.2
    ellipse = Ellipse(center, width, height,
                      edgecolor=color, facecolor='none', linestyle='--', linewidth=1.5)
    plt.gca().add_patch(ellipse)
plt.legend()

# --------------------------------TIME SERIES METRICS SECTION---------------------------------
# ============== 6. Select metric columns ==================
timeMet = df.columns[59:202]                # columns 59-202 are time metrics
X_met = df[timeMet]

# ============== 7. Standardize the data ==================
scaler = StandardScaler()                   # standardize environmental attributes
X_met_scaled = scaler.fit_transform(X_met)  # ensures PCA is not dominated by large-scale variables

# ============== 8. Standardize the environmental variables 
pca_met = PCA()                             # initialize PCA model
pca_met.fit(X_met_scaled)                   # fit PCA to standardized data

# ============== 9. Fit PCA (all components) ==============
pc_met_scores = pca_met.transform(X_met_scaled) # compute each site's pc scores
df["PC1"] = pc_met_scores[:, 0]             # store PC1 scores
df["PC2"] = pc_met_scores[:, 1]             # store PC2 scores

# ============== 10. PC1 vs PC2 scatter plot (232)==========
pc_met_scores = pca_met.transform(X_met_scaled)
plt.subplot(232)
plt.scatter(pc_met_scores[:, 0], pc_met_scores[:, 1], alpha=0.7)
plt.xlabel(f"PC1 ({pca_met.explained_variance_ratio_[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({pca_met.explained_variance_ratio_[1]*100:.1f}% var)")
plt.title("Time Series Metrics PCA (PC1 vs PC2)")
plt.tight_layout()

# Cluster sites based on PC1 and PC2
cluster_colors = {
    0: 'orange',
    1: 'green',
    2: 'red'}
kmeans = KMeans(n_clusters=3, random_state=42).fit(pc_met_scores[:, :2])
df["MetCluster"] = kmeans.labels_
# print sites in each cluster and their states
print("\nFull SiteN lists for MetCluster:")
for cluster_id, sites in df.groupby("MetCluster")["SiteN"].unique().items():
    print(f"MetCluster {cluster_id}: {list(sites)}")

print("\nState counts for MetCluster:")
print(df.groupby("MetCluster")["State"].value_counts())

# Plot with cluster colors
plt.subplot(232)
for cluster_id in np.unique(df["MetCluster"]):
    cluster_points = pc_met_scores[df["MetCluster"] == cluster_id]
    color = cluster_colors[cluster_id]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                label=f"MetCluster {cluster_id}", alpha=0.7, color=color)
    center = cluster_points.mean(axis=0)
    width = cluster_points[:, 0].ptp() * 1.2
    height = cluster_points[:, 1].ptp() * 1.2
    ellipse = Ellipse(center, width, height,
                      edgecolor=color, facecolor='none', linestyle='--', linewidth=1.5)
    plt.gca().add_patch(ellipse)
plt.legend()

# --------------------------------COMBINED METRICS SECTION---------------------------------
# ============== 11. Select metric columns ==================
X_comb = df[list(env_cols) + list(timeMet)]             # 4-25, 59-202 are combined metrics 

# ============== 12. Standardize the data ==================
scaler = StandardScaler()                     # standardize environmental attributes
X_comb_scaled = scaler.fit_transform(X_comb)  # ensures PCA is not dominated by large-scale variables

# ============== 13. Standardize the environmental variables 
pca_comb = PCA()                              # initialize PCA model
pca_comb.fit(X_comb_scaled)                   # fit PCA to standardized data

# ============== 14. Fit PCA (all components) ==============
pc_comb_scores = pca_comb.transform(X_comb_scaled) # compute each site's pc scores
df["PC1"] = pc_comb_scores[:, 0]             # store PC1 scores
df["PC2"] = pc_comb_scores[:, 1]             # store PC2 scores

# ============== 15. PC1 vs PC2 scatter plot (233)==========
pc_comb_scores = pca_comb.transform(X_comb_scaled)
plt.subplot(233)
plt.scatter(pc_comb_scores[:, 0], pc_comb_scores[:, 1], alpha=0.7)
plt.xlabel(f"PC1 ({pca_comb.explained_variance_ratio_[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({pca_comb.explained_variance_ratio_[1]*100:.1f}% var)")
plt.title("Combined Metrics PCA (PC1 vs PC2)")
plt.tight_layout()

# Cluster sites based on PC1 and PC2
cluster_colors = {
    0: 'orange',
    1: 'green',
    2: 'red',}
kmeans = KMeans(n_clusters=3, random_state=42).fit(pc_comb_scores[:, :2])
df["CombCluster"] = kmeans.labels_
# print sites in each cluster and their states
print("\nFull SiteN lists for CombCluster:")
for cluster_id, sites in df.groupby("CombCluster")["SiteN"].unique().items():
    print(f"CombCluster {cluster_id}: {list(sites)}")

print("\nState counts for CombCluster:")
print(df.groupby("CombCluster")["State"].value_counts())

# Plot with cluster colors
plt.subplot(233)
for cluster_id in np.unique(df["CombCluster"]):
    cluster_points = pc_comb_scores[df["CombCluster"] == cluster_id]
    color = cluster_colors[cluster_id]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                label=f"CombCluster {cluster_id}", alpha=0.7, color=color)
    center = cluster_points.mean(axis=0)
    width = cluster_points[:, 0].ptp() * 1.2
    height = cluster_points[:, 1].ptp() * 1.2
    ellipse = Ellipse(center, width, height,
                      edgecolor=color, facecolor='none', linestyle='--', linewidth=1.5)
    plt.gca().add_patch(ellipse)
plt.legend()

# ============== 16. 2 Cluster comparison prints==========
# print percentages of states in each cluster using the csv file, for "accuracy"
def cluster_state_percentages(df, cluster_col):
    counts = df.groupby([cluster_col, "State"]).size().unstack(fill_value=0)
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100
    return percentages
print("\nEnvironmental Cluster State Percentages:")
print(cluster_state_percentages(df, "EnvCluster"))
print("\nTime-Series Cluster State Percentages:")
print(cluster_state_percentages(df, "MetCluster"))

# chi-square tests prints (look into how to interpret because idk yet)
env_table = pd.crosstab(df["EnvCluster"], df["State"])
met_table = pd.crosstab(df["MetCluster"], df["State"])
comb_table = pd.crosstab(df["CombCluster"], df["State"])

print("\nChi-square for Environmental Clusters vs State:")
print(chi2_contingency(env_table))

print("\nChi-square for Time-Series Clusters vs State:")
print(chi2_contingency(met_table))

print("\nChi-square for Combined Clusters vs State:")
print(chi2_contingency(comb_table))

# ============== 12. 3 Clusters Box Plots (234), (235) and (236)==========
plt.subplot(234)                            # environmental clusters by state     
env_counts = df.groupby(["EnvCluster", "State"]).size().reset_index(name="Count")
sns.barplot(data=env_counts, x="EnvCluster", y="Count", hue="State")
plt.title("Environmental Clusters by State")

plt.subplot(235)                            # time-series clusters by state
met_counts = df.groupby(["MetCluster", "State"]).size().reset_index(name="Count")
sns.barplot(data=met_counts, x="MetCluster", y="Count", hue="State")
plt.title("Time-Series Clusters by State")

plt.subplot(236)                            # combined clusters by state
Comb_counts = df.groupby(["CombCluster", "State"]).size().reset_index(name="Count")
sns.barplot(data=Comb_counts, x="CombCluster", y="Count", hue="State")
plt.title("Combined Clusters by State")

# ============= Make the subplots look a little nicer ================= 
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.13, wspace=0.3, hspace=0.3)

# ================= label plot edges ==================
plt.subplot(position=[0.0500,    0.93,    0.02500,    0.02500]) # U-left
plt.axis('off')
plt.text(0,.5, programMsg_c, fontsize=7)

plt.subplot(position=[0.550,    0.93,    0.02500,    0.02500]) # U-right
plt.axis('off')
plt.text(0,.5, authorName_c, fontsize=7)

plt.subplot(position=[0.0500,    0.02,    0.02500,    0.02500]) # L-left
plt.axis('off')
plt.text(0,.5, fileName_c, fontsize=7)

plt.subplot(position=[0.3500,    0.02,    0.02500,    0.02500]) # L-right
plt.axis('off')
# plt.text(0,.5, msg_plot_c, fontsize=8)

plt.savefig(figName_c)
plt.close()