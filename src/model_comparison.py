import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# FINAL MODEL COMPARISON
# ============================================================

data = {
    "Model": [
        "Linear Regression",
        "Random Forest",
        "Gradient Boosting",
        "Tuned Random Forest",
        "Tuned Gradient Boosting"
    ],

    "RMSE": [
        1.2918375121,
        0.6241682287,
        0.6347324303,
        0.6173962639,
        0.5963972347
    ],

    "R2": [
        0.6167578696,
        0.9105334315,
        0.9074793148,
        0.9124642500,
        0.9183175595
    ]
}


df = pd.DataFrame(data)

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

print(df.to_string(index=False))


# ============================================================
# SAVE RESULTS
# ============================================================

df.to_csv(
    "final_model_comparison.csv",
    index=False
)

print("\nSaved: final_model_comparison.csv")


# ============================================================
# RMSE COMPARISON
# ============================================================

plt.figure(figsize=(9, 6))

bars = plt.bar(
    df["Model"],
    df["RMSE"]
)

plt.ylabel("Test RMSE")
plt.xlabel("Model")
plt.title("Comparison of Machine Learning Models")

plt.xticks(
    rotation=25,
    ha="right"
)

plt.grid(
    axis="y",
    alpha=0.25
)

# Add values above bars
for bar, value in zip(bars, df["RMSE"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{value:.3f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    "final_model_rmse_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Saved: final_model_rmse_comparison.png")


# ============================================================
# R2 COMPARISON
# ============================================================

plt.figure(figsize=(9, 6))

bars = plt.bar(
    df["Model"],
    df["R2"]
)

plt.ylabel("Test $R^2$")
plt.xlabel("Model")
plt.title("Comparison of Model $R^2$ Scores")

plt.xticks(
    rotation=25,
    ha="right"
)

plt.ylim(0, 1)

plt.grid(
    axis="y",
    alpha=0.25
)

# Add values above bars
for bar, value in zip(bars, df["R2"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{value:.3f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    "final_model_r2_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Saved: final_model_r2_comparison.png")


# ============================================================
# BEST MODEL
# ============================================================

best_rmse = df.loc[df["RMSE"].idxmin()]

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print(f"Model : {best_rmse['Model']}")
print(f"RMSE  : {best_rmse['RMSE']:.4f}")
print(f"R²    : {best_rmse['R2']:.4f}")