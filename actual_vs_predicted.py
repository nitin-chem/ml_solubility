import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# FINAL ACTUAL VS PREDICTED PLOT
# ============================================================

results = pd.read_csv("final_model_predictions.csv")

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
    "actual_vs_predicted_final.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Saved: actual_vs_predicted_final.png")