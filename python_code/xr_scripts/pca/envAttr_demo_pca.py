# program		envAttr_demo_pca
# purpose	    PCA for environmental attributes to find signficant PCs
# usage         script
# notes         (1) 
# date			01/27/2026
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
programname_c = os.path.basename(__file__)
programName_c = os.path.dirname(os.path.abspath(__file__))
script_name = os.path.splitext(os.path.basename(__file__))[0]

fileName_c = 'data/seth_gov_envData10dtsMetrics_genusCount2026.csv'
programMsg_c = programName_c + ' (' + date_c + ')'
authorName_c = 'Xavier Ramirez'

fig_dir = os.path.abspath(os.path.join(programName_c, "..", "..", "..", "figures")) #NEW
os.makedirs(fig_dir, exist_ok=True) #NEW
figName_c = os.path.join(fig_dir, f"{script_name}_fig.png") #NEW

# ========== Load and preprocess data ==============
df = pd.read_csv(fileName_c)
df.columns = df.columns.str.strip()


# ---- 1. Select environmental columns ----
env_cols = df.columns[4:25]     # example: columns 1–24 are environmental
X = df[env_cols]

# ---- 2. Standardize the environmental variables ----
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---- 3. Fit PCA (all components) ----
pca = PCA()
pca.fit(X_scaled)

# ---- 4. Scree Plot: variance explained by each PC ----
plt.figure(figsize=(12,6))     # creates a nicely-sized figure
plt.subplot(122)
plt.plot(
    range(1, len(pca.explained_variance_ratio_) + 1),
    pca.explained_variance_ratio_,
    marker='o'
)
plt.xlabel("Principal Component")
plt.ylabel("Proportion of Variance Explained")
plt.title("Scree Plot of Environmental PCA")
plt.xticks(range(1, len(pca.explained_variance_ratio_) + 1))
plt.tight_layout()

# ---- 5. PC1 vs PC2 scatter plot ----
pc_scores = pca.transform(X_scaled)

plt.subplot(121)
plt.scatter(pc_scores[:, 0], pc_scores[:, 1], alpha=0.7)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
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
plt.subplot(121)
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

# ---- 6. Optional: PCA loadings table ----
loadings = pd.DataFrame(
    pca.components_.T,
    index=env_cols,
    columns=[f"PC{i+1}" for i in range(len(env_cols))]
)

print(loadings.iloc[:, :2])  # show loadings for PC1–PC2

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
#plt.text(0,.5, msg_plot_c, fontsize=8)

plt.savefig(figName_c)
plt.close()