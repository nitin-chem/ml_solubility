from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# FINAL MODEL PERFORMANCE FIGURE
# ============================================================

results = pd.read_csv(
    TABLES_DIR / "final_model_performance.csv"
)

print("Final model performance:")
print(results)


# ------------------------------------------------------------
# Create metric plot
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

plt.bar(
    results["Metric"],
    results["Value"]
)

plt.ylabel("Value")
plt.xlabel("Evaluation Metric")
plt.title("Final Gradient Boosting Model Performance")

# Add values above bars
for i, value in enumerate(results["Value"]):

    plt.text(
        i,
        value + 0.02,
        f"{value:.4f}",
        ha="center",
        fontsize=10
    )

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "final_model_performance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"\nSaved: {FIGURES_DIR / 'final_model_performance.png'}")