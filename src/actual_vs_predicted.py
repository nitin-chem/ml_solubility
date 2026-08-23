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
# FINAL ACTUAL VS PREDICTED PLOT
# ============================================================
results = pd.read_csv(
    TABLES_DIR / "final_model_predictions.csv"
)

actual = results["Actual_logS"]
predicted = results["Predicted_logS"]


# ============================================================
# CREATE PLOT
# ============================================================

plt.figure(figsize=(7, 7))

plt.scatter(
    actual,
    predicted,
    alpha=0.65,
    s=45
)

# Perfect prediction line
minimum = min(actual.min(), predicted.min())
maximum = max(actual.max(), predicted.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    linewidth=2
)

plt.xlabel("Experimental logS")
plt.ylabel("Predicted logS")

plt.title(
    "Actual vs Predicted Solubility\n"
    "Gradient Boosting Regressor"
)

plt.xlim(minimum, maximum)
plt.ylim(minimum, maximum)

plt.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "actual_vs_predicted_final.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"Saved: {FIGURES_DIR / 'actual_vs_predicted_final.png'}")