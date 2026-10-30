# program		ramirezXavier_demo_umapPlotMultiple_fishData.py
# purpose	    Demonstrate UMAP plot with Seth environmental data
# usage         script
# notes         (1) Using UMAP minus 1 feature to make a correlation matrix and compare which units are more correlated with each other.
# date			10/30/2026
# programmer    Xavier Ramirez

import datetime          # used for getting the date
import os                # used for getting the basic file name (returns lower case)
import win32api          # used for getting fileName with correct case
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import umap
from sklearn import preprocessing    # Import label encoder  
import matplotlib.patheffects as PathEffects    # use for fancy plot text for example
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

# ============== COMMON INITIALIZATION =====================
date_o = datetime.datetime.today()
date_c = date_o.strftime('%m/%d/%Y')
programName_c = os.path.dirname(os.path.abspath(__file__))
script_name = os.path.splitext(os.path.basename(__file__))[0]

fileName_c = 'data/seth_gov_envData10dtsMetrics_genusCount2026.csv'
programMsg_c = script_name + ' (' + date_c + ')'
authorName_c = 'Xavier Ramirez'

fig_dir = os.path.abspath(os.path.join(programName_c, "..", "..", "..", "figures"))
os.makedirs(fig_dir, exist_ok=True)
figName_c = os.path.join(fig_dir, f"{script_name}_fig.png")

# ================= get data ==================
sethEnvData_df = pd.read_csv((fileName_c))                 # load the sethEnv data into a pandas data frame
sethEnvData_num_df = sethEnvData_df.iloc[:, :26]
sethEnvData_num_df = sethEnvData_num_df.drop(['SiteN','State'], axis = 1)    # numerical data only
print(sethEnvData_num_df.shape)
sethEnvData_labels = sethEnvData_df['State']
label_encoder = preprocessing.LabelEncoder()  
sethEnvData_labels_v = label_encoder.fit_transform(sethEnvData_labels) 
nLabels = len(np.unique(sethEnvData_labels_v))

sethEnvDataLabelsNames_cv = label_encoder.inverse_transform(range(0,nLabels))

# ==================== scale data ===================
sc_z = StandardScaler()

# Standardize features for training and testing sets
featureNames_cv = sethEnvData_num_df.columns.tolist()
sethEnvData_num_df = sc_z.fit_transform(sethEnvData_num_df)

# ================ do UMAP ======================
# correlation metric -> more defined clusters than euclidean (default), hamming, chebyshev
# n_neighbors=15 (default) seems to yield more distinct clusters than range 5-20, although 5 OK also
# 5 OK with a larger number of epochs (1000-1500); 15 good with smaller # (500)
# also, 2, 3 generate handfull of tight clusters (maybe this is better?); 7 not bad either
# learn_rate = 0.5 seems to do better than 1.0 default
# default init seems better than pca
# min_dist default of 0.1 looks best
umapModel = umap.UMAP(n_components=2,metric='correlation',random_state=1,n_neighbors=7,n_epochs=500,learning_rate=0.5,min_dist=0.1,n_jobs = 1)
sethEnvData_umap = umapModel.fit_transform(sethEnvData_num_df)
D0_v = pdist(sethEnvData_umap)

print('type(isethEnvData_umap) = ',type(sethEnvData_umap))
print('sethEnvData_umap.shape = ',sethEnvData_umap.shape)

# loop for each feature for feature sensitivity
featureImpact_v = []
baseX_df = sethEnvData_df.iloc[:, :26]
baseX_df = baseX_df.drop(['SiteN','State'], axis=1)

for kc in range(len(featureNames_cv)):

    print('Removing:', featureNames_cv[kc])

    # remove one feature
    X_df = baseX_df.copy()
    X_df = X_df.drop(columns=[featureNames_cv[kc]])

    # scale
    X_v = StandardScaler().fit_transform(X_df)

    # run identical UMAP
    embedding_i = umap.UMAP(
        n_components=2,
        metric='correlation',
        random_state=1,
        n_neighbors=7,
        n_epochs=500,
        learning_rate=0.5,
        min_dist=0.1,
        n_jobs=1
    ).fit_transform(X_v)

    # compare against baseline
    Di_v = pdist(embedding_i)

    r = np.corrcoef(D0_v, Di_v)[0,1]

    impact = 1 - r
    print(featureNames_cv[kc], ' impact = ', impact)

    featureImpact_v.append(impact)

featureImpact_v = np.array(featureImpact_v)

sortIdx_v = np.argsort(featureImpact_v)

featureImpact_v = featureImpact_v[sortIdx_v]
featureNames_cv = np.array(featureNames_cv)[sortIdx_v]

# print table of features and their impact
print('\n=== Feature Ranking ===')

for name, impact in zip(featureNames_cv[::-1], featureImpact_v[::-1]):
    print(f'{name:25s} {impact:.3f}')

# ===================== PLOT RESULTS ===================
msg_parm_c = 'umapModel = umap(n_components=2'

fig = plt.figure(figsize=(12,5), dpi=200)
plt.rcParams.update({'font.size': 8})           # change default font size
plt.rcParams.update({'lines.markersize': 5})           # change default size of markers

# ============= Make the subplots look a little nicer ================= 
# umap plot 121
plt.subplot(121)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.10, wspace=0.2, hspace=0.3)

for kc in range(nLabels):
    plt.scatter(sethEnvData_umap[sethEnvData_labels_v == kc, 0],sethEnvData_umap[sethEnvData_labels_v == kc,1])
    xtext, ytext = np.median(sethEnvData_umap[sethEnvData_labels_v == kc,:], axis=0)
    txt = plt.annotate(sethEnvDataLabelsNames_cv[kc],(xtext,ytext),fontsize=9)
    txt.set_path_effects([PathEffects.Stroke(linewidth=5, foreground='w'),PathEffects.Normal()])

    idx = (sethEnvData_labels_v == kc)

    plt.scatter(
    sethEnvData_umap[idx,0],
    sethEnvData_umap[idx,1],
    c=sethEnvData_df.loc[idx,'pbottom_Mud'],
    cmap='viridis',
    s=60,
    alpha=0.8
)
plt.colorbar(label='Mud (%)')

for kc in range(nLabels):

    points = sethEnvData_umap[sethEnvData_labels_v == kc]

    if len(points) > 3:
        hull = ConvexHull(points)

        for simplex in hull.simplices:
            plt.plot(points[simplex,0],
                     points[simplex,1],
                     'k-', alpha=0.3)
plt.legend(frameon=True)
plt.title('UMAP Plot of sethEnv Data, Cat. by State')

# bar plot 122
plt.subplot(122)

plt.barh(
    featureNames_cv,
    featureImpact_v,
    color='steelblue'
)

plt.xlabel('UMAP Change (1-r)')
plt.title('Feature Sensitivity')
plt.grid(alpha=0.3)

# ================= label plot edges ==================
plt.subplot(position=[0.0500,    0.94,    0.02500,    0.02500]) # U-left
plt.axis('off')
plt.text(0,.5, programMsg_c, fontsize=8)

plt.subplot(position=[0.550,    0.94,    0.02500,    0.02500]) # U-right
plt.axis('off')
plt.text(0,.5, authorName_c, fontsize=8)

plt.subplot(position=[0.0500,    0.02,    0.02500,    0.02500]) # L-left
plt.axis('off')
plt.text(0,.5, fileName_c, fontsize=8)

plt.subplot(position=[0.3500,    0.02,    0.02500,    0.02500]) # L-right
plt.axis('off')
plt.text(0,.5, msg_parm_c, fontsize=8)

plt.savefig(figName_c)
plt.close()