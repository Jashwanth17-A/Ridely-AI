import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# 1. PATHS
# ============================================================

OUTPUT_DIR = Path("outputs")

ML_FILE = OUTPUT_DIR / "ml_metrics.csv"
DL_FILE = OUTPUT_DIR / "dl_metrics.csv"


# ============================================================
# 2. LOAD RESULTS
# ============================================================

print("\nLoading model results...")

ml_df = pd.read_csv(ML_FILE)
dl_df = pd.read_csv(DL_FILE)

print("ML results loaded.")
print("DL results loaded.")


# ============================================================
# 3. COMBINE RESULTS
# ============================================================

all_results = pd.concat(
    [ml_df, dl_df],
    ignore_index=True
)

print("\nCombined Model Results:")
print(all_results.to_string(index=False))


# ============================================================
# 4. SAVE ALL MODEL RESULTS
# ============================================================

all_results.to_csv(
    OUTPUT_DIR / "all_model_metrics.csv",
    index=False
)


# ============================================================
# 5. CALCULATE AVERAGE PERFORMANCE
# ============================================================

summary = (
    all_results
    .groupby("Model")[["MAE", "RMSE", "R2"]]
    .mean()
    .reset_index()
)

print("\n======================================")
print("OVERALL MODEL COMPARISON")
print("======================================")

print(summary.to_string(index=False))


# ============================================================
# 6. RANK MODELS
# ============================================================

# Lower MAE is better
# Lower RMSE is better
# Higher R2 is better

summary["MAE_Rank"] = summary["MAE"].rank(
    ascending=True
)

summary["RMSE_Rank"] = summary["RMSE"].rank(
    ascending=True
)

summary["R2_Rank"] = summary["R2"].rank(
    ascending=False
)

summary["Overall_Rank"] = (
    summary["MAE_Rank"]
    + summary["RMSE_Rank"]
    + summary["R2_Rank"]
)

summary = summary.sort_values(
    "Overall_Rank"
)

print("\n======================================")
print("MODEL RANKING")
print("======================================")

print(
    summary[
        [
            "Model",
            "MAE",
            "RMSE",
            "R2",
            "Overall_Rank"
        ]
    ].to_string(index=False)
)


# ============================================================
# 7. SAVE FINAL COMPARISON
# ============================================================

summary.to_csv(
    OUTPUT_DIR / "final_model_comparison.csv",
    index=False
)


# ============================================================
# 8. BEST MODEL
# ============================================================

best_model = summary.iloc[0]["Model"]

print("\n======================================")
print("BEST OVERALL MODEL")
print("======================================")

print(f"Best Model: {best_model}")


# ============================================================
# 9. BEST MODEL FOR EACH PLATFORM
# ============================================================

platform_results = []

for platform in all_results["Platform"].unique():

    platform_df = all_results[
        all_results["Platform"] == platform
    ].copy()

    # Rank using RMSE
    platform_df = platform_df.sort_values(
        "RMSE",
        ascending=True
    )

    best = platform_df.iloc[0]

    platform_results.append({
        "Platform": platform,
        "Best_Model": best["Model"],
        "MAE": best["MAE"],
        "RMSE": best["RMSE"],
        "R2": best["R2"]
    })


best_platform_df = pd.DataFrame(platform_results)


print("\n======================================")
print("BEST MODEL BY PLATFORM")
print("======================================")

print(
    best_platform_df.to_string(index=False)
)


best_platform_df.to_csv(
    OUTPUT_DIR / "best_model_by_platform.csv",
    index=False
)


# ============================================================
# 10. CREATE COMPARISON GRAPH
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 5)
)


# ---------------- MAE ----------------

axes[0].bar(
    summary["Model"],
    summary["MAE"]
)

axes[0].set_title("MAE Comparison")
axes[0].set_xlabel("Model")
axes[0].set_ylabel("MAE")
axes[0].tick_params(axis="x", rotation=45)


# ---------------- RMSE ----------------

axes[1].bar(
    summary["Model"],
    summary["RMSE"]
)

axes[1].set_title("RMSE Comparison")
axes[1].set_xlabel("Model")
axes[1].set_ylabel("RMSE")
axes[1].tick_params(axis="x", rotation=45)


# ---------------- R2 ----------------

axes[2].bar(
    summary["Model"],
    summary["R2"]
)

axes[2].set_title("R² Comparison")
axes[2].set_xlabel("Model")
axes[2].set_ylabel("R²")
axes[2].tick_params(axis="x", rotation=45)


plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "final_model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 11. FINISHED
# ============================================================

print("\n======================================")
print("MODEL COMPARISON COMPLETED")
print("======================================")

print("\nGenerated files:")

print("1. outputs/all_model_metrics.csv")
print("2. outputs/final_model_comparison.csv")
print("3. outputs/best_model_by_platform.csv")
print("4. outputs/final_model_comparison.png")