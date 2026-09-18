# program		ramirezXavier_demo_xgb_umap_vary_env_genera.py 
# purpose	    Demonstrate XGBoost model using UMAP visualization for seth_environmentalGenusCountData_nov2024.csv data set.
# usage         script
# notes         (1) this is very disorganized, and organization comes LAST 'round 
#               these parts, not really my worry nor forte at the moment, but we ball
#               (2) we are LOG-TRANSFORMING all data
#               (3) vary pipeline is biotic + abiotic attributes
#               (4) env pipeline is abiotic-only attributes
#               (5)
#               (6) ..._<genera>_fig1: visualizations of pipelines
#               (7) ..._<genera>_fig2: diagnotics print (for minimal console clutter)
#               (8) DO NOT RUN FULLY AS TO CLUTTER THE FIGURES FOLDER 18SEP2026
# date			01/19/2025
# programmer    Xavier Ramirez

import datetime
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from scipy.stats import zscore

import umap.umap_ as umap
import xgboost as xgb
import shap
plt.rcParams.update({'font.size': 8})

# ============== COMMON INITIALIZATION =====================
date_o = datetime.datetime.today()
date_c = date_o.strftime('%m/%d/%Y')

programName_c = os.path.dirname(os.path.abspath(__file__))
script_name = os.path.splitext(os.path.basename(__file__))[0]

ix = str.find(programName_c, '.')

fileName_c = 'data/seth_gov_envData10dtsMetrics_genusCount2026.csv'
programMsg_c = script_name + ' (' + date_c + ')'
authorName_c = 'Xavier Ramirez'

fig_dir = os.path.abspath(os.path.join(programName_c, "..", "..", "..", "figures")) #NEW
os.makedirs(fig_dir, exist_ok=True) #NEW
figName_c = os.path.join(fig_dir, f"{script_name} + _fig.png") #NEW

# ========== Load and preprocess data ==============
df = pd.read_csv(fileName_c)
df.columns = df.columns.str.strip()

# Drop non-numeric columns
non_numeric_cols = df.select_dtypes(include=['object']).columns
print("Dropping non-numeric columns:", non_numeric_cols.tolist())
df = df.drop(non_numeric_cols, axis=1)

# Fill missing numeric values
df = df.fillna(0)

# ================== GENUS LOOP SETUP ==================
# Choose genus columns by index
genus_columns = df.columns[25:54]   # 25-54 for all unless custom

# ================== GENUS LOOP START ==================
for target_genus in genus_columns:
    print(f"\n=== Running genus: {target_genus} ===")
    
    # Log-transform abundance
    df["target"] = np.log1p(df[target_genus])
    label_col = "target"
    
    # ================== PREDICTOR SETS ==================
    
    # vary: all numeric except metadata + target + target_genus
    exclude_cols_vary = [target_genus, "target", "SiteN", "State", "Lat", "Long"]
    predictor_cols_vary = [c for c in df.columns if c not in exclude_cols_vary]
    
    # env: explicit environmental predictors
    predictor_cols_env = [
        "DO", "pH",
        "avg_WW", "avg_Depth",
        "pflow_Pool", "pflow_Run", "pflow_Riffle",
        "pbottom_Mud", "pbottom_Sand", "pbottom_SmGravel",
        "pbottom_LgGravel", "pbottom_Cobble", "pbottom_Boulder", "pbottom_Bedrock",
        "pfloat_macrophytes", "pfloat_wood",
        "upstreamCumDA_km2", "slopePercent",
        "avgMonthFlow_cfs", "CV_flow", "maxMonthFlow_cfs", "minMonthFlow_cfs",
        "t_population", "t_species"
    ]
    
    # Ensure all env predictors exist
    missing_env = [c for c in predictor_cols_env if c not in df.columns]
    if missing_env:
        print("Missing env predictors:", missing_env)
        raise SystemExit
    
    pipelines = {
        "vary": predictor_cols_vary,
        "env": predictor_cols_env
    }
    
    # Train/validation split (shared)
    tts_ts = 0.10
    tts_rs = 24
    
    df_train, df_val = train_test_split(df, test_size=tts_ts, random_state=tts_rs)
    
    # ================== STORAGE FOR RESULTS ==================
    results = {}
    
    # ================== LOOP OVER PIPELINES ==================
    for name, predictor_cols in pipelines.items():
        print(f"\n=== Running pipeline: {name} for {target_genus} ===")
        # Design matrices
        X_train = df_train[predictor_cols].values
        y_train = df_train[label_col].values
    
        X_val = df_val[predictor_cols].values
        y_val = df_val[label_col].values
    
        # XGBoost model
        xgb_reg = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            reg_alpha=0.1,
            tree_method='hist',
            random_state=42
        )
    
        xgb_reg.fit(X_train, y_train)
    
        # Predictions (log-space)
        predict_T_log = xgb_reg.predict(X_train)
        predict_V_log = xgb_reg.predict(X_val)
    
        # Invert log-transform
        predict_T = np.expm1(predict_T_log)
        predict_V = np.expm1(predict_V_log)
        y_train_raw = np.expm1(y_train)
        y_val_raw = np.expm1(y_val)
    
        # Regression metrics in original scale
        mse_T = mean_squared_error(y_train_raw, predict_T)
        mse_V = mean_squared_error(y_val_raw, predict_V)
        r2_T = r2_score(y_train_raw, predict_T)
        r2_V = r2_score(y_val_raw, predict_V)
    
        # Cross-validation (log-space R²)
        scoresCV_v = cross_val_score(xgb_reg, X_train, y_train, cv=10, scoring='r2')
    
        # Full prediction for all rows (for residuals)
        X_full = df[predictor_cols].values
        y_pred_full_log = xgb_reg.predict(X_full)
        y_pred_full = np.expm1(y_pred_full_log)
    
        # Feature importance and top-10
        importance = xgb_reg.feature_importances_
        sorted_idx = np.argsort(importance)
        top10_idx = sorted_idx[-10:]
        top10_features = np.array(predictor_cols)[top10_idx]
    
        # Retrain on top-10 predictors
        X_train_top10 = df_train[top10_features].values
        X_val_top10 = df_val[top10_features].values
    
        xgb_reg_top10 = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            reg_alpha=0.1,
            tree_method='hist',
            random_state=42
        )
    
        xgb_reg_top10.fit(X_train_top10, y_train)
    
        # Predict using reduced model on full data
        X_full_top10 = df[top10_features].values
        predict_full_top10_log = xgb_reg_top10.predict(X_full_top10)
        predict_full_top10 = np.expm1(predict_full_top10_log)
    
        # UMAP on top-10 predictors
        umap_model_top10 = umap.UMAP(
            n_neighbors=20,
            min_dist=0.1,
            n_components=2,
            metric='euclidean',
            n_epochs=500,
            random_state=45
        )
    
        X_umap_top10 = umap_model_top10.fit_transform(X_full_top10)
    
        # Outlier detection on top-10 UMAP + residuals + attributes
        # UMAP spatial outliers
        nbrs = NearestNeighbors(n_neighbors=10).fit(X_umap_top10)
        distances, _ = nbrs.kneighbors(X_umap_top10)
        avg_distance = distances.mean(axis=1)
        umap_outlier_mask = avg_distance > np.percentile(avg_distance, 95)
    
        # Residual outliers (raw scale) using full model predictions
        residuals = np.expm1(df[label_col]) - y_pred_full
        residual_outlier_mask = np.abs(residuals) > np.percentile(np.abs(residuals), 95)
    
        # Attribute outliers (top-10 predictors only)
        X_key = df[top10_features]
        z_scores = np.abs(zscore(X_key))
        attribute_outlier_mask = (z_scores > 3).any(axis=1)
    
        # ===== SHAP on top-10 model =====
        background = shap.sample(X_full_top10, 100)
        
        explainer = shap.KernelExplainer(xgb_reg_top10.predict, background)
        shap_values = explainer.shap_values(X_full_top10, nsamples=200)
    
        results[name] = {
            "predictor_cols": predictor_cols,
            "importance": importance,
            "top10_idx": top10_idx,
            "top10_features": top10_features,
            "X_umap_top10": X_umap_top10,
            "y_pred_full_top10": predict_full_top10,
            "umap_outlier_mask": umap_outlier_mask,
            "residual_outlier_mask": residual_outlier_mask,
            "attribute_outlier_mask": attribute_outlier_mask,
            "mse_T": mse_T,
            "mse_V": mse_V,
            "r2_T": r2_T,
            "r2_V": r2_V,
            "cv_scores": scoresCV_v,
            "X_full_top10": X_full_top10,
            "shap_values": shap_values
        }
    
    # ======================== PLOT RESULTS ======================
    # ----------- FIGURE 1: VISUAL DASHBOARD (3×2) ---------------
    fig1_name = os.path.join(fig_dir, f"{script_name}_{target_genus}_fig1.png")
    plt.figure(num=1, figsize=(16, 12), dpi=200)
    
    # --- 321: vary UMAP (Top 10 Predictors Only) ---
    vary_res = results["vary"]
    plt.subplot(3, 2, 1)
    scatter_vary = plt.scatter(
        vary_res["X_umap_top10"][:, 0],
        vary_res["X_umap_top10"][:, 1],
        c=vary_res["y_pred_full_top10"],
        cmap='viridis',
        s=20
    )
    plt.scatter(
        vary_res["X_umap_top10"][vary_res["umap_outlier_mask"], 0],
        vary_res["X_umap_top10"][vary_res["umap_outlier_mask"], 1],
        edgecolor='red', facecolor='none', s=80, linewidth=1.5, label="UMAP Outliers"
    )
    plt.scatter(
        vary_res["X_umap_top10"][vary_res["residual_outlier_mask"], 0],
        vary_res["X_umap_top10"][vary_res["residual_outlier_mask"], 1],
        edgecolor='blue', facecolor='none', s=80, linewidth=1.5, label="Residual Outliers"
    )
    plt.scatter(
        vary_res["X_umap_top10"][vary_res["attribute_outlier_mask"], 0],
        vary_res["X_umap_top10"][vary_res["attribute_outlier_mask"], 1],
        edgecolor='green', facecolor='none', s=80, linewidth=1.5, label="Attribute Outliers"
    )
    plt.title(f"vary UMAP (Top 10 Predictors) — {target_genus}")
    plt.xlabel("UMAP-1", fontsize=6)
    plt.ylabel("UMAP-2", fontsize=6)
    
    cbar = plt.colorbar(scatter_vary, label=f"Predicted {target_genus} Count")
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label(f"Predicted {target_genus} Count", fontsize=6)
    
    # FINALIZED LEGEND PLACEMENT (outside bottom-left)
    plt.legend(loc='lower left', bbox_to_anchor=(-0.50, -0.45), fontsize=6)
    
    # --- 322: env UMAP (Top 10 Predictors Only) ---
    env_res = results["env"]
    plt.subplot(3, 2, 2)
    scatter_env = plt.scatter(
        env_res["X_umap_top10"][:, 0],
        env_res["X_umap_top10"][:, 1],
        c=env_res["y_pred_full_top10"],
        cmap='viridis',
        s=20
    )
    plt.scatter(
        env_res["X_umap_top10"][env_res["umap_outlier_mask"], 0],
        env_res["X_umap_top10"][env_res["umap_outlier_mask"], 1],
        edgecolor='red', facecolor='none', s=80, linewidth=1.5, label="UMAP Outliers"
    )
    plt.scatter(
        env_res["X_umap_top10"][env_res["residual_outlier_mask"], 0],
        env_res["X_umap_top10"][env_res["residual_outlier_mask"], 1],
        edgecolor='blue', facecolor='none', s=80, linewidth=1.5, label="Residual Outliers"
    )
    plt.scatter(
        env_res["X_umap_top10"][env_res["attribute_outlier_mask"], 0],
        env_res["X_umap_top10"][env_res["attribute_outlier_mask"], 1],
        edgecolor='green', facecolor='none', s=80, linewidth=1.5, label="Attribute Outliers"
    )
    plt.title(f"env UMAP (Top 10 Predictors) — {target_genus}")
    plt.xlabel("UMAP-1", fontsize=6)
    plt.ylabel("UMAP-2", fontsize=6)
    
    cbar = plt.colorbar(scatter_env, label=f"Predicted {target_genus} Count")
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label(f"Predicted {target_genus} Count", fontsize=6)
    
    # --- 323: vary XGBoost Feature Importance (Top 10) ---
    plt.subplot(3, 2, 3)
    plt.barh(
        vary_res["top10_features"],
        vary_res["importance"][vary_res["top10_idx"]]
    )
    plt.yticks(fontsize=6)
    plt.xticks(fontsize=6)
    plt.title("vary XGBoost Feature Importance (Top 10)")
    plt.xlabel("Importance Score", fontsize=6)
    
    # --- 324: env XGBoost Feature Importance (Top 10) ---
    plt.subplot(3, 2, 4)
    plt.barh(
        env_res["top10_features"],
        env_res["importance"][env_res["top10_idx"]]
    )
    plt.yticks(fontsize=6)
    plt.xticks(fontsize=6)
    plt.title("env XGBoost Feature Importance (Top 10)")
    plt.xlabel("Importance Score", fontsize=6)
    
    # --- 325: vary SHAP Summary (Top 10) ---
    plt.subplot(3, 2, 5)
    shap.summary_plot(
        vary_res["shap_values"],
        vary_res["X_full_top10"],
        feature_names=vary_res["top10_features"],
        show=False,
    )
    cb_ax = plt.gcf().axes[-1]
    cb_ax.tick_params(labelsize=6)
    cb_ax.set_ylabel("Feature value", fontsize=6)
    plt.tick_params(axis='y', labelsize=6)
    plt.xlabel("SHAP value (impact on model output)", fontsize=6)
    plt.xticks(fontsize=6)
    plt.title("vary SHAP Summary (Top 10)", pad=8)
    
    # --- 326: env SHAP Summary (Top 10) ---
    plt.subplot(3, 2, 6)
    shap.summary_plot(
        env_res["shap_values"],
        env_res["X_full_top10"],
        feature_names=env_res["top10_features"],
        show=False,
    )
    cb_ax = plt.gcf().axes[-1]
    cb_ax.tick_params(labelsize=6)
    cb_ax.set_ylabel("Feature value", fontsize=6)
    plt.tick_params(axis='y', labelsize=6)
    plt.xlabel("SHAP value (impact on model output)", fontsize=6)
    plt.xticks(fontsize=6)
    plt.title("env SHAP Summary (Top 10)", pad=8)
    
    # ======== Bottom Metadata Block ========
    plt.figtext(
        0.50, -0.02,
        f"Author: {authorName_c}\n"
        f"Data: {fileName_c}\n"
        f"Visualization for {target_genus} - {programMsg_c}",
        ha='center', va='bottom',
        fontsize=6,
    )
    
    plt.subplots_adjust(hspace=0.70, wspace=0.38)
    plt.savefig(fig1_name, bbox_inches='tight')
    plt.close()
    
    # ---------------- FIGURE 2: DIAGNOSTIC TEXT ----------------
    fig2_name = os.path.join(fig_dir, f"{script_name}_{target_genus}_fig2.png")
    plt.figure(num=2, figsize=(18, 8), dpi=200)
    
    # Build text blocks
    vary_text = (
        f"Pipeline: vary\n"
        f"Training MSE: {vary_res['mse_T']:.4f}\n"
        f"Validation MSE: {vary_res['mse_V']:.4f}\n"
        f"Training R²: {vary_res['r2_T']:.4f}\n"
        f"Validation R²: {vary_res['r2_V']:.4f}\n"
        f"CV R² (log-space): {np.round(vary_res['cv_scores'], 4)}\n"
        f"CV Mean: {np.mean(vary_res['cv_scores']):.4f}\n\n"
        f"Top 10 Predictors:\n" +
        "\n".join([f"  - {f}" for f in vary_res["top10_features"]]) +
        "\n\nUMAP Outliers:\n" +
        f"{np.where(vary_res['umap_outlier_mask'])[0]}\n\n"
        f"Residual Outliers:\n" +
        f"{np.where(vary_res['residual_outlier_mask'])[0]}\n\n"
        f"Attribute Outliers:\n" +
        f"{np.where(vary_res['attribute_outlier_mask'])[0]}\n"
    )
    
    env_text = (
        f"Pipeline: env\n"
        f"Training MSE: {env_res['mse_T']:.4f}\n"
        f"Validation MSE: {env_res['mse_V']:.4f}\n"
        f"Training R²: {env_res['r2_T']:.4f}\n"
        f"Validation R²: {env_res['r2_V']:.4f}\n"
        f"CV R² (log-space): {np.round(env_res['cv_scores'], 4)}\n"
        f"CV Mean: {np.mean(env_res['cv_scores']):.4f}\n\n"
        f"Top 10 Predictors:\n" +
        "\n".join([f"  - {f}" for f in env_res["top10_features"]]) +
        "\n\nUMAP Outliers:\n" +
        f"{np.where(env_res['umap_outlier_mask'])[0]}\n\n"
        f"Residual Outliers:\n" +
        f"{np.where(env_res['residual_outlier_mask'])[0]}\n\n"
        f"Attribute Outliers:\n" +
        f"{np.where(env_res['attribute_outlier_mask'])[0]}\n"
    )
    
    # Left panel (vary)
    ax1 = plt.subplot(1, 2, 1)
    ax1.text(0.02, 0.98, vary_text, va='top', family='monospace', 
             fontsize=10, wrap=True, transform=ax1.transAxes, 
             bbox=dict(facecolor='none', edgecolor='none')),
    ax1.axis('off')
    
    # Right panel (env)
    ax2 = plt.subplot(1, 2, 2)
    ax2.text(0.02, 0.98, env_text, va='top', family='monospace', 
             fontsize=10, wrap=True, transform=ax2.transAxes, 
             bbox=dict(facecolor='none', edgecolor='none'))
    ax2.axis('off')
    
    plt.figtext(
        0.50, -0.02,
        f"Author: {authorName_c}\n"
        f"Data: {fileName_c}\n"
        f"Diagnostics for {target_genus} — {programMsg_c}",
        ha='center', va='bottom',
        fontsize=9,
    )
    
    plt.subplots_adjust(wspace=0.30)
    plt.savefig(fig2_name, bbox_inches='tight')
    plt.close()