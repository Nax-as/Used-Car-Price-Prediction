# %% [markdown]
# # Used Car Price Prediction — Exploratory Data Analysis
# **Dataset:** Cardekho Used Cars Dataset  
# **Objective:** Understand the data distribution, correlations, and feature importance through visualisations.

# %% [markdown]
# ## 1. Setup & Imports

# %%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("Set2")
pd.set_option("display.float_format", lambda x: f"{x:.2f}")

DATA_PATH = "../data/car_details.csv"
df_raw = pd.read_csv(DATA_PATH)
print(f"Shape: {df_raw.shape}")
df_raw.head()

# %% [markdown]
# ## 2. Dataset Overview

# %%
print("=== Column Info ===")
df_raw.info()

# %%
print("\n=== Missing Values ===")
missing = df_raw.isnull().sum()
print(missing[missing > 0])

# %%
print("\n=== Descriptive Statistics ===")
df_raw.describe()

# %% [markdown]
# ## 3. Data Cleaning

# %%
df = df_raw.copy()
df.drop_duplicates(inplace=True)
df.dropna(subset=["selling_price"], inplace=True)

# Feature engineering
df["brand"]   = df["name"].str.split().str[0]
df["model"]   = df["name"].str.split().str[1]
df["car_age"] = 2024 - df["year"]

def extract_num(s):
    return pd.to_numeric(
        s.astype(str).str.extract(r"([\d.]+)", expand=False), errors="coerce"
    )

df["mileage_kmpl"]  = extract_num(df["mileage"])
df["engine_cc"]     = extract_num(df["engine"])
df["max_power_bhp"] = extract_num(df["max_power"])

df.drop(columns=["name","year","mileage","engine","max_power","torque"],
        inplace=True)
print(f"Cleaned shape: {df.shape}")

# %% [markdown]
# ## 4. Univariate Analysis

# %%
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Univariate Analysis — Numerical Features", fontsize=15, y=1.01)

num_cols = ["selling_price","km_driven","mileage_kmpl","engine_cc","max_power_bhp","car_age"]
colors   = sns.color_palette("Set2", 6)

for ax, col, c in zip(axes.flat, num_cols, colors):
    data = df[col].dropna()
    ax.hist(data, bins=40, color=c, edgecolor="white", alpha=.85)
    ax.set_title(col, fontsize=11, fontweight="bold")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1e5:.0f}L" if col == "selling_price" else f"{x:,.0f}"))

plt.tight_layout()
plt.savefig("../artifacts/eda_univariate.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Log-transformed target
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.hist(df["selling_price"], bins=50, color="#16a34a", edgecolor="white")
ax1.set_title("Selling Price — Raw")
ax2.hist(np.log1p(df["selling_price"]), bins=50, color="#4ade80", edgecolor="white")
ax2.set_title("Selling Price — log(1+x)")
for ax in [ax1, ax2]: ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("../artifacts/eda_log_transform.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Categorical Feature Distributions

# %%
cat_cols = ["fuel","seller_type","transmission","owner"]
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle("Categorical Feature Distributions", fontsize=14)

for ax, col in zip(axes, cat_cols):
    counts = df[col].value_counts()
    ax.barh(counts.index, counts.values,
            color=sns.color_palette("Set2", len(counts)))
    ax.set_title(col.replace("_"," ").title(), fontweight="bold")
    ax.set_xlabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(v + 20, i, str(v), va="center", fontsize=8)

plt.tight_layout()
plt.savefig("../artifacts/eda_categorical.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Bivariate Analysis

# %%
# Selling price vs fuel type
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df.boxplot(column="selling_price", by="fuel", ax=axes[0], grid=False)
axes[0].set_title("Price vs Fuel Type"); axes[0].set_xlabel(""); axes[0].set_ylabel("Selling Price")
df.boxplot(column="selling_price", by="transmission", ax=axes[1], grid=False)
axes[1].set_title("Price vs Transmission"); axes[1].set_xlabel(""); axes[1].set_ylabel("")
plt.suptitle("")
plt.tight_layout()
plt.savefig("../artifacts/eda_bivariate_cat.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Scatter: KM driven vs Price
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(df["km_driven"], df["selling_price"],
                alpha=.3, s=15, color="#16a34a")
axes[0].set(title="KM Driven vs Price", xlabel="KM Driven", ylabel="Price")

axes[1].scatter(df["max_power_bhp"], df["selling_price"],
                alpha=.3, s=15, color="#166534")
axes[1].set(title="Max Power vs Price", xlabel="Max Power (bhp)", ylabel="Price")

plt.tight_layout()
plt.savefig("../artifacts/eda_scatter.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Multivariate Analysis

# %%
# Correlation heatmap
num_df = df[["selling_price","km_driven","car_age","mileage_kmpl",
             "engine_cc","max_power_bhp","seats"]].dropna()

corr = num_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdYlGn", center=0, linewidths=.5, ax=ax)
ax.set_title("Correlation Heatmap — Numerical Features", fontsize=13)
plt.tight_layout()
plt.savefig("../artifacts/eda_correlation.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Pairplot
pair_df = num_df[["selling_price","km_driven","max_power_bhp","engine_cc","car_age"]]
g = sns.pairplot(pair_df, diag_kind="kde", plot_kws={"alpha": .3, "s": 12},
                 diag_kws={"color": "#16a34a"})
g.fig.suptitle("Pairplot — Selected Numerical Features", y=1.02)
plt.savefig("../artifacts/eda_pairplot.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8. Top Brands by Volume and Median Price

# %%
top_brands = df["brand"].value_counts().head(15).index
brand_df   = df[df["brand"].isin(top_brands)]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

brand_df["brand"].value_counts().plot(kind="bar", ax=axes[0],
    color=sns.color_palette("Set2", 15), edgecolor="white")
axes[0].set_title("Top 15 Brands by Listing Count", fontweight="bold")
axes[0].tick_params(axis="x", rotation=45)

brand_df.groupby("brand")["selling_price"].median().reindex(
    brand_df["brand"].value_counts().head(15).index
).plot(kind="bar", ax=axes[1],
       color=sns.color_palette("muted", 15), edgecolor="white")
axes[1].set_title("Top 15 Brands by Median Price", fontweight="bold")
axes[1].tick_params(axis="x", rotation=45)
axes[1].yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"₹{x/1e5:.0f}L"))

plt.tight_layout()
plt.savefig("../artifacts/eda_brands.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 9. Price Trend over Car Age

# %%
age_price = df.groupby("car_age")["selling_price"].median().reset_index()
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(age_price["car_age"], age_price["selling_price"],
        marker="o", markersize=5, color="#16a34a", linewidth=2)
ax.fill_between(age_price["car_age"], age_price["selling_price"],
                alpha=.12, color="#16a34a")
ax.set(title="Median Selling Price vs Car Age",
       xlabel="Car Age (years)", ylabel="Median Price (₹)")
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"₹{x/1e5:.1f}L"))
plt.tight_layout()
plt.savefig("../artifacts/eda_age_trend.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 10. Key Insights Summary
# 
# | Insight | Detail |
# |---|---|
# | Target distribution | Right-skewed; log1p transform normalises it well |
# | Strongest positive correlator | `max_power_bhp` (r ≈ 0.76) |
# | Strongest negative correlator | `car_age` (r ≈ -0.52) and `km_driven` (r ≈ -0.45) |
# | Fuel type | Diesel cars command a price premium over petrol |
# | Transmission | Automatic cars are significantly more expensive |
# | Ownership | First-owner cars fetch the highest prices |
# | Brand | Luxury brands (BMW, Audi, Mercedes) have outlier high prices |

print("EDA Complete — figures saved to artifacts/")
