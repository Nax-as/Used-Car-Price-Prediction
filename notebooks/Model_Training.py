# %% [markdown]
# # Used Car Price Prediction — Model Training & Evaluation
# **Objective:** Train 8 regression models, compare them, tune the best one,
# and save it as a pickle file.

# %% [markdown]
# ## 1. Imports

# %%
import warnings, sys, os
warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor)
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.utils import save_object

# %% [markdown]
# ## 2. Data Ingestion & Transformation

# %%
ingestion = DataIngestion()
train_path, test_path = ingestion.initiate_data_ingestion("data/car_details.csv")

transformer = DataTransformation()
X_train, X_test, y_train, y_test, prep_path = \
    transformer.initiate_data_transformation(train_path, test_path)

print(f"X_train: {X_train.shape}  |  X_test: {X_test.shape}")
print(f"y_train range: {y_train.min():.2f} – {y_train.max():.2f}  (log scale)")

# %% [markdown]
# ## 3. Baseline Model Comparison

# %%
models = {
    "Linear Regression":  LinearRegression(),
    "Ridge Regression":   Ridge(),
    "Lasso Regression":   Lasso(max_iter=10_000),
    "Decision Tree":      DecisionTreeRegressor(random_state=42),
    "Random Forest":      RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Extra Trees":        ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting":  GradientBoostingRegressor(random_state=42),
    "XGBoost":            XGBRegressor(random_state=42, verbosity=0, n_jobs=-1),
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results.append({"Model": name, "R2": r2, "MAE": mae, "RMSE": rmse})
    print(f"  {name:25s}  R²={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}")

results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
print("\n", results_df.to_string(index=False))

# %%
# Visualise R² scores
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#16a34a" if r == results_df["R2"].max() else "#94a3b8"
          for r in results_df["R2"]]
bars = ax.barh(results_df["Model"], results_df["R2"], color=colors, edgecolor="white")
ax.set_xlim(0, 1.05)
ax.set_xlabel("R² Score (Test Set)", fontsize=11)
ax.set_title("Model Comparison — R² Scores", fontsize=13, fontweight="bold")
for bar, val in zip(bars, results_df["R2"]):
    ax.text(val + .005, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("../artifacts/model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Cross-Validation

# %%
best_name = results_df.iloc[0]["Model"]
best_base  = models[best_name]

cv_scores = cross_val_score(best_base, X_train, y_train, cv=5,
                            scoring="r2", n_jobs=-1)
print(f"\nCross-Validation R² ({best_name}):")
print(f"  Scores: {cv_scores.round(4)}")
print(f"  Mean: {cv_scores.mean():.4f}  ±  {cv_scores.std():.4f}")

# %% [markdown]
# ## 5. Hyperparameter Tuning

# %%
PARAM_GRIDS = {
    "Random Forest": {
        "n_estimators": [100, 200], "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    },
    "Extra Trees": {
        "n_estimators": [100, 200], "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    },
    "XGBoost": {
        "n_estimators": [100, 200], "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1], "subsample": [0.8, 1.0],
    },
    "Gradient Boosting": {
        "n_estimators": [100, 200], "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
    },
}

if best_name in PARAM_GRIDS:
    print(f"Tuning {best_name} …")
    grid = GridSearchCV(
        models[best_name], PARAM_GRIDS[best_name],
        cv=3, scoring="r2", n_jobs=-1, verbose=1,
    )
    grid.fit(X_train, y_train)
    best_model   = grid.best_estimator_
    best_params  = grid.best_params_
    print(f"Best params: {best_params}")
else:
    best_model = best_base

# %% [markdown]
# ## 6. Final Evaluation

# %%
y_pred_final = best_model.predict(X_test)

r2_final   = r2_score(y_test, y_pred_final)
mae_final  = mean_absolute_error(y_test, y_pred_final)
rmse_final = np.sqrt(mean_squared_error(y_test, y_pred_final))

print(f"\n=== Final Model: {best_name} ===")
print(f"  R² Score : {r2_final:.4f}")
print(f"  MAE      : {mae_final:.4f}")
print(f"  RMSE     : {rmse_final:.4f}")

# %%
# Actual vs Predicted plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(y_test, y_pred_final, alpha=.35, s=15, color="#16a34a")
mn, mx = y_test.min(), y_test.max()
axes[0].plot([mn, mx], [mn, mx], "r--", lw=1.5)
axes[0].set(title="Actual vs Predicted (log scale)",
            xlabel="Actual log(Price)", ylabel="Predicted log(Price)")

residuals = y_test - y_pred_final
axes[1].hist(residuals, bins=50, color="#4ade80", edgecolor="white")
axes[1].axvline(0, color="red", linestyle="--", lw=1.5)
axes[1].set(title="Residual Distribution", xlabel="Residuals", ylabel="Count")

plt.suptitle(f"{best_name} — Final Evaluation", fontsize=13)
plt.tight_layout()
plt.savefig("../artifacts/final_model_eval.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Feature importance (for tree-based models)
if hasattr(best_model, "feature_importances_"):
    feature_names = (
        ["km_driven","car_age","mileage_kmpl","engine_cc","max_power_bhp","seats"] +
        ["brand","model","fuel","seller_type","transmission","owner"]
    )
    imp = pd.Series(best_model.feature_importances_,
                    index=feature_names[:len(best_model.feature_importances_)]
                    ).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    imp.plot(kind="bar", ax=ax, color="#16a34a", edgecolor="white")
    ax.set_title("Feature Importance", fontsize=13, fontweight="bold")
    ax.set_ylabel("Importance Score")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig("../artifacts/feature_importance.png", dpi=150, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## 7. Save Final Model

# %%
save_object("artifacts/model.pkl", best_model)
print("Model saved to artifacts/model.pkl")

# %% [markdown]
# ## 8. Sample Predictions

# %%
sample_idx = np.random.choice(len(X_test), 8, replace=False)
sample_preds  = np.expm1(best_model.predict(X_test[sample_idx]))
sample_actual = np.expm1(y_test[sample_idx])

sample_df = pd.DataFrame({
    "Actual Price (₹)":    sample_actual.round(0),
    "Predicted Price (₹)": sample_preds.round(0),
    "Error %":             ((sample_preds - sample_actual) / sample_actual * 100).round(2),
})
print("\nSample Predictions:")
print(sample_df.to_string(index=False))
