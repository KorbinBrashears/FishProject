# program		demo_regression_envData10dtsMetrics.py
# purpose	    Demonstrate linear & quadratic regression on Boston House data
# usage         script
# notes         (1) Can select nX = 1 ['lstat'], 2 ['lstat','rm'], or all variables to predict 'mdev'
#               (2) Includes lstat vs mdev linear and quadratic fit plot also (not prediction on V data)
#               (3) Uses make_pipeline
# date			 2/26/2026 (significantly revised)
# programmer    Xavier Ramirez

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures # use this for polynomial regression
from sklearn.model_selection import train_test_split
import datetime          # used for getting the date
import os                # used for getting the basic file name (returns lower case)

# ============== COMMON INITIALIZATION =====================
date_o = datetime.datetime.today()
date_c = date_o.strftime('%m/%d/%Y')
programName_c = os.path.dirname(os.path.abspath(__file__)) #NEW
script_name = os.path.splitext(os.path.basename(__file__))[0] #NEW

fileName_c = 'data/seth_gov_envData10dtsMetrics_genusCount2026.csv'
programMsg_c = script_name + ' (' + date_c + ')'
authorName_c = 'Xavier Ramirez'

fig_dir = os.path.abspath(os.path.join(programName_c, "..", "..", "..", "figures")) #NEW
os.makedirs(fig_dir, exist_ok=True) #NEW
figName_c = os.path.join(fig_dir, f"{script_name}_fig.png") #NEW

# ================= GET ALL DATA ======================
data_df = pd.read_csv((fileName_c))
genusFind_v = data_df[['Etheostoma']]    # get median bostonHouse cost column
pLowerStatusA_v = data_df.iloc[:, 59:60]   # first predictor column

# =============== GET TRAIN, TEST DATA ==================
dataT_df, dataV_df = train_test_split(data_df,test_size = 0.5, random_state=42)
genusFindT_v = dataT_df[['Etheostoma']]
genusFindV_v = dataV_df[['Etheostoma']]

# X predictors: use all variables from columns 59–202
XdataT_df = dataT_df.iloc[:, 59:202]
XdataV_df = dataV_df.iloc[:, 59:202]

print('=========== CHECK SIZES OF DATA =============')
print('data_df shape = ',data_df.shape)
print('dataT_df shape = ',dataT_df.shape)
print('dataV_df shape = ',dataV_df.shape)
print('XdataV_df shape = ',XdataV_df.shape)

# =============== regression using polyfeatures with pipeline ======
pdeg = 1    # polynomial of degree 1 (linear)
pipeline_polyregression = make_pipeline(PolynomialFeatures(degree=pdeg), linear_model.LinearRegression())

pipeline_polyregression.fit(XdataT_df, genusFindT_v)
medianHouseCostPredictP1_v = pipeline_polyregression.predict(XdataV_df)
r2Test_p1 = pipeline_polyregression.score(XdataV_df,genusFindV_v)
r2Train_p1 = pipeline_polyregression.score(XdataT_df, genusFindT_v)
print('train p1 R2 = =', r2Train_p1)
print('validate p1 R2 = =', r2Test_p1)

pdeg = 2    # polynomial of degree 2 (quadratic)
pipeline_polyregression = make_pipeline(PolynomialFeatures(degree=pdeg), linear_model.LinearRegression())

pipeline_polyregression.fit(XdataT_df, genusFindT_v)
medianHouseCostPredictP2_v = pipeline_polyregression.predict(XdataV_df)
r2Test_p2 = pipeline_polyregression.score(XdataV_df,genusFindV_v)
r2Train_p2 = pipeline_polyregression.score(XdataT_df, genusFindT_v)
print('train p2 R2 = =', r2Train_p2)
print('validate p2 R2 = =', r2Test_p2)

# Get normalized RMSE error =======
rmsenTest_p1 = np.sqrt(1-r2Test_p1)
rmsenTest_p2 = np.sqrt(1-r2Test_p2)

# ================== PREPARE TO PLOT ===============
Lmsg1_c = 'R2 score; normalized RMSE = ' + '%1.2f' %r2Test_p1
Lmsg1_c = Lmsg1_c + '; ' + '%1.2f' %rmsenTest_p1 + ' (' + ' variables)'
Qmsg1_c = 'R2 score; normalized RMSE = ' + '%1.2f' %r2Test_p2
Qmsg1_c = Qmsg1_c + '; ' + '%1.2f' %rmsenTest_p2 + ' (' + ' variables)'

Lmsg_c = 'Linear Model: ' + Lmsg1_c
Qmsg_c = 'Quadratic Model: ' + Qmsg1_c

# ========= DO X,Y LINEAR AND QUADRATIC FIT ON ALL DATA FOR PLOT ONLY =====
pdeg = 1    # polynomial of degree 1 (linear)
pipeline_polyregression = make_pipeline(PolynomialFeatures(degree=pdeg), linear_model.LinearRegression())
pipeline_polyregression.fit(pLowerStatusA_v, genusFind_v)
genusPlotP1_v = pipeline_polyregression.predict(pLowerStatusA_v)

pdeg = 2    # polynomial of degree 2 (quadratic)
pipeline_polyregression = make_pipeline(PolynomialFeatures(degree=pdeg), linear_model.LinearRegression())
pipeline_polyregression.fit(pLowerStatusA_v, genusFind_v)
genusPlotP2_v = pipeline_polyregression.predict(pLowerStatusA_v)

# get sorting indices for quadratic and plot sorted values for plot
pLowerStatusA_vx = np.array(pLowerStatusA_v)
pLowerStatusA_vx = pLowerStatusA_vx.flatten() # convert to 1D before get sort indices
sort_indices = np.argsort(pLowerStatusA_vx)
genusPlotP2_vx = np.array(genusPlotP2_v)
genusPlotP2_vx = genusPlotP2_vx.flatten() # convert to 1d np array for plot with indices

# ===================== PLOT RESULTS ===================
plt.figure(num=1, figsize=(11.0, 5.5),dpi=200) # standard size for power point
plt.subplots_adjust(bottom=0.15, top=0.9, left=.05, right=0.95)
plt.rcParams.update({'font.size': 9})

plt.scatter(pLowerStatusA_v, genusFind_v, color='magenta', marker='o')
plt.plot(pLowerStatusA_v, genusPlotP1_v, color='black', linewidth=5)
plt.plot(pLowerStatusA_vx[sort_indices], genusPlotP2_vx[sort_indices],color='blue',linewidth=5)
plt.xlabel('Predictor Variable', fontsize=8)
plt.ylabel('Etheostoma Count', fontsize=8)
plt.title('Predictor vs Etheostoma Abundance', fontsize=8)
plt.legend(['All data','Linear Fit', 'Quadratic Fit'])
plt.text(15,47,Lmsg_c)
plt.text(15,44.5,Qmsg_c)


# ================= label plot edges ==================
plt.subplot(position=[0.0500,    0.94,    0.02500,    0.02500]) # U-left
plt.axis('off')
plt.text(0,.5, programMsg_c, fontsize=8)

plt.subplot(position=[0.550,    0.94,    0.02500,    0.02500]) # U-right
plt.axis('off')
plt.text(0,.5, authorName_c, fontsize=8)

plt.subplot(position=[0.0500,    0.0325,    0.02500,    0.02500]) # L-left
plt.axis('off')
plt.text(0,.5, fileName_c, fontsize=8)

plt.savefig(figName_c)

plt.close()