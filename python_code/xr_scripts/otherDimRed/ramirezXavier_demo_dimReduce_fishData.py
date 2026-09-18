# program		ramirezXavier_demo_dimReduce_fishData
# purpose	    Demo dimensionality reduction visualization with fish data
# usage         script
# notes         (1) PCA focused assignment
#               (2) PCA on static env. attributes; PCA on time series metrics
#               (3) includes respective scree plots and PC# bar graphs
# date			02/1/2026
# programmer    Xavier Ramirez

import datetime
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib.patches import Ellipse

# ============== COMMON INITIALIZATION =====================
date_o = datetime.datetime.today()

date_c = date_o.strftime('%m/%d/%Y')
programName_c = os.path.dirname(os.path.abspath(__file__))
script_name = os.path.splitext(os.path.basename(__file__))[0]

fileName_c = 'data/seth_gov_envData10dtsMetrics_genusCount2026.csv'
programMsg_c = script_name + ' (' + date_c + ')'
authorName_c = 'Xavier Ramirez'

fig_dir = os.path.abspath(os.path.join(programName_c, "..", "..", "..", "figures")) #NEW
os.makedirs(fig_dir, exist_ok=True) #NEW
figName_c = os.path.join(fig_dir, f"{script_name}_fig.png") #NEW

# ============== 0. Load and preprocess data ===============
df = pd.read_csv(fileName_c)                # load dataset
df.columns = df.columns.str.strip()         # clean column names
genus_col = "Gambusia"                      # target genus for script

# ============== 1. Select environmental columns ===========
env_cols = df.columns[4:25]                 # columns 1–24 are environmental
X_env = df[env_cols]

# ============== 2. Standardize the data ===================
scaler = StandardScaler()                   # standardize environmental attributes
X_env_scaled = scaler.fit_transform(X_env)  # ensures PCA is not dominated by large-scale variables

# ============== 3. Standardize the environmental variables =
pca_env = PCA()                             # initialize PCA model
pca_env.fit(X_env_scaled)                   # fit PCA to standardized data

# ============== 4. Fit PCA (all components) ===============
pc_scores = pca_env.transform(X_env_scaled)         # compute each site's pc scores
df["PC1"] = pc_scores[:, 0]                 # store PC1 scores
df["PC2"] = pc_scores[:, 1]                 # store PC2 scores

plt.figure(figsize=(12,6))                  # creates a nicely-sized figure
# ============== 5. PC1 vs PC2 scatter plot (241)===========
pc_scores = pca_env.transform(X_env_scaled)
plt.subplot(241)
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
df["Cluster"] = kmeans.labels_
# Plot with cluster colors
plt.subplot(241)
for cluster_id in np.unique(df["Cluster"]):
    cluster_points = pc_scores[df["Cluster"] == cluster_id]
    color = cluster_colors[cluster_id]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                label=f"Cluster {cluster_id}", alpha=0.7, color=color)
    center = cluster_points.mean(axis=0)
    width = cluster_points[:, 0].ptp() * 1.2
    height = cluster_points[:, 1].ptp() * 1.2
    ellipse = Ellipse(center, width, height,
                      edgecolor=color, facecolor='none', linestyle='--', linewidth=1.5)
    plt.gca().add_patch(ellipse)
plt.legend()

# ============== 6. PCA loadings table (242)================
loadings = pd.DataFrame(
    pca_env.components_.T,
    index=env_cols,
    columns=[f"PC{i+1}" for i in range(len(env_cols))])
print(loadings.iloc[:, :2])                 # print loadings for PC1–PC2

plt.subplot(242)
plt.plot(
    range(1, len(pca_env.explained_variance_ratio_) + 1),
    pca_env.explained_variance_ratio_,
    marker='o'
)
plt.xlabel("Principal Component")
plt.ylabel("Proportion of Variance Explained")
plt.title("Scree Plot of Environmental PCA")
num_pcs = len(pca_env.explained_variance_ratio_)
plt.xticks(np.arange(1, num_pcs + 1, 5))    # ticks every 10 PCs

# ============== 7. PC1 Loadings Bar Plot (243)=============
pc1_vec = pca_env.components_[0]            # PC1 loadings
sorted_idx = np.argsort(np.abs(pc1_vec))[::-1]
sorted_loadings = pc1_vec[sorted_idx][::-1]
sorted_attrs = env_cols[sorted_idx][::-1]
colors = ['#1f77b4' if val > 0 else '#d62728' for val in sorted_loadings]
plt.subplot(243)
plt.barh(sorted_attrs, sorted_loadings, color=colors)
plt.axvline(0, color='black', linewidth=1)
plt.title("Environmental PCA – PC1 Drivers")
plt.xlabel("Loading Strength")
plt.tight_layout()

# ============== 8. PC2 Loadings Bar Plot (244)=============
pc2_vec = pca_env.components_[1]            # PC2 loadings
sorted_idx = np.argsort(np.abs(pc2_vec))[::-1]
sorted_loadings = pc2_vec[sorted_idx][::-1]
sorted_attrs = env_cols[sorted_idx][::-1]
colors = ['#1f77b4' if val > 0 else '#d62728' for val in sorted_loadings]
plt.subplot(244)
plt.barh(sorted_attrs, sorted_loadings, color=colors)
plt.axvline(0, color='black', linewidth=1)
plt.title("Environmental PCA – PC2 Drivers")
plt.xlabel("Loading Strength")
plt.tight_layout()

# TIME SERIES METRICS SECTION
# ============== 9. Select metric columns ==================
timeMet = df.columns[59:202]                # columns 59-202 are time metrics
X_met = df[timeMet]

# ============== 10. Standardize the data ==================
scaler = StandardScaler()                   # standardize environmental attributes
X_met_scaled = scaler.fit_transform(X_met)  # ensures PCA is not dominated by large-scale variables

# ============== 11. Standardize the environmental variables 
pca_met = PCA()                             # initialize PCA model
pca_met.fit(X_met_scaled)                   # fit PCA to standardized data

# ============== 12. Fit PCA (all components) ==============
pc_met_scores = pca_met.transform(X_met_scaled) # compute each site's pc scores
df["PC1"] = pc_met_scores[:, 0]                 # store PC1 scores
df["PC2"] = pc_met_scores[:, 1]                 # store PC2 scores

# ============== 13. PC1 vs PC2 scatter plot (245)==========
pc_met_scores = pca_met.transform(X_met_scaled)
plt.subplot(245)
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
df["Cluster"] = kmeans.labels_
# Plot with cluster colors
plt.subplot(245)
for cluster_id in np.unique(df["Cluster"]):
    cluster_points = pc_met_scores[df["Cluster"] == cluster_id]
    color = cluster_colors[cluster_id]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                label=f"Cluster {cluster_id}", alpha=0.7, color=color)
    center = cluster_points.mean(axis=0)
    width = cluster_points[:, 0].ptp() * 1.2
    height = cluster_points[:, 1].ptp() * 1.2
    ellipse = Ellipse(center, width, height,
                      edgecolor=color, facecolor='none', linestyle='--', linewidth=1.5)
    plt.gca().add_patch(ellipse)
plt.legend()

# ============== 14. PCA loadings table (246)===============
loadings = pd.DataFrame(
    pca_met.components_.T,
    index=timeMet,
    columns=[f"PC{i+1}" for i in range(pca_met.n_components_)]
)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(loadings.iloc[:, :2])                 # print loadings for PC1–PC2

plt.subplot(246)
plt.plot(
    range(1, len(pca_met.explained_variance_ratio_) + 1),
    pca_met.explained_variance_ratio_,
    marker='o'
)
plt.xlabel("Principal Component")
plt.ylabel("Proportion of Variance Explained")
plt.title("Scree Plot of Time Series Metrics PCA")
num_pcs = len(pca_met.explained_variance_ratio_)
plt.xticks(np.arange(1, num_pcs + 1, 5))        # ticks every 10 PCs

# ============== 15. PC1 Loadings Bar Plot (247)============
pc1_vec = pca_met.components_[0]                 # PC1 loadings
sorted_idx = np.argsort(np.abs(pc1_vec))[::-1]
top_idx = sorted_idx[:20]                        # take only the top 20
sorted_loadings = pc1_vec[top_idx][::-1]
sorted_attrs = timeMet[top_idx][::-1]
colors = ['#1f77b4' if val > 0 else '#d62728' for val in sorted_loadings]
plt.subplot(247)
plt.barh(sorted_attrs, sorted_loadings, color=colors)
plt.axvline(0, color='black', linewidth=1)
plt.title("Time Series Metrics PCA – PC1 Drivers")
plt.xlabel("Loading Strength")
plt.tight_layout()

# ============== 16. PC2 Loadings Bar Plot (248)============
pc2_vec = pca_met.components_[1]                 # PC2 loadings
sorted_idx = np.argsort(np.abs(pc2_vec))[::-1]
top_idx = sorted_idx[:20]
sorted_loadings = pc2_vec[top_idx][::-1]
sorted_attrs = timeMet[top_idx][::-1]
colors = ['#1f77b4' if val > 0 else '#d62728' for val in sorted_loadings]
plt.subplot(248)
plt.barh(sorted_attrs, sorted_loadings, color=colors)
plt.axvline(0, color='black', linewidth=1)
plt.title("Time Series Metrics PCA – PC2 Drivers")
plt.xlabel("Loading Strength")
plt.tight_layout()

# ============= Make the subplots look a little nicer ================= 
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.13, wspace=.65, hspace=0.3)

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
#plt.text(0,.5, msg_plot_c, fontsize=8)

plt.savefig(figName_c)
plt.close()