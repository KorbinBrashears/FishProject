# program		timeAttr_demo_sumMetrics_siteVariable.py
# purpose	    Pick site, variable, # of splits; make time series plots changing over 24 years by # of splits
# usage         script
# notes         (1) 
# date			02/18/2026
# programmer    Xavier Ramirez

import datetime
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ============== COMMON INITIALIZATION =====================
date_o = datetime.datetime.today()
date_c = date_o.strftime('%m/%d/%Y')
programName_c = os.path.dirname(os.path.abspath(__file__)) #NEW
script_name = os.path.splitext(os.path.basename(__file__))[0] #NEW

fileName_c = 'data/gov_envData10dts_2025.csv'
programMsg_c = script_name + ' (' + date_c + ')'
authorName_c = 'Xavier Ramirez'

fig_dir = os.path.abspath(os.path.join(programName_c, "..", "..", "..", "figures")) #NEW
os.makedirs(fig_dir, exist_ok=True) #NEW
figName_c = os.path.join(fig_dir, f"{script_name} + _fig.png") #NEW

# ========== Load and preprocess data ==============
df = pd.read_csv(fileName_c)
df.columns = df.columns.str.strip()
sites = sorted({col.split('.')[0] for col in df.columns if col.startswith("site_")})

# ---- 1. Input site, variable, # of splits ----
site_analyze = 1                   # integer site number input
var_analyze = "maxT"                # variable name input
n_splits = 6                        # number of time windows input

col_name = f"site_{site_analyze - 1}.{var_analyze}" # site_#.variable
y = df[col_name].values

# ---- 2. Split into n_splits ----
def split_series(y, n_splits):
    length = len(y)
    split_size = length // n_splits
    splits = []

    for i in range(n_splits):
        start = i * split_size
        end = (i + 1) * split_size if i < n_splits - 1 else length
        splits.append(y[start:end])

    return splits

# ---- 3. Compute summary metrics for each split ----
def compute_trend(y):               # using LR for trend line
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    return model.coef_[0]

def compute_metrics(y):             # using np for other metrics
    return {
        "mean": np.mean(y),
        "median": np.median(y),
        "sd": np.std(y),
        "min": np.min(y),
        "max": np.max(y),
        "range": np.max(y) - np.min(y),
        "cv": np.std(y) / np.mean(y),
        "trend": compute_trend(y)
    }

splits = split_series(y, n_splits)
metrics_per_split = [compute_metrics(s) for s in splits]

# ---- 4. Plot metrics over time ----
metric_names = ["mean", "median", "sd", "min", "max", "range", "cv", "trend"]
plt.figure(figsize=(16, 8))

for i, metric in enumerate(metric_names, start=1):
    plt.subplot(2, 4, i)

    values = [m[metric] for m in metrics_per_split]
    plt.plot(range(1, n_splits + 1), values, marker="o")

    plt.title(metric)
    plt.xlabel("Time window")
    plt.ylabel("Value")
plt.suptitle(f"Site {site_analyze} - {var_analyze} Metrics Over Time", fontsize=14)

# ============= Make the subplots look a little nicer ================= 
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.13, wspace=0.4, hspace=0.3)

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

outname = f"site{site_analyze}_{var_analyze}_metricsChange_fig.png"
outfile = os.path.join(fig_dir, outname)

plt.savefig(outfile)
plt.close()
