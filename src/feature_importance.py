import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================
# 1. Load trained model
# ============================================================

model = joblib.load("model.pkl")

print("Model loaded successfully.")


# ============================================================
# 2. Feature names
# ============================================================

descriptor_names = [
    "Molecular Weight",
    "LogP",
    "H-Bond Donors",
    "H-Bond Acceptors",
    "TPSA"
]

fingerprint_names = [
    f"Morgan_{i}"
    for i in range(512)
]

feature_names = descriptor_names + fingerprint_names


# ============================================================
# 3. Get feature importance
# ============================================================

importance = model.feature_importances_


print("\nTotal features:", len(importance))


# ============================================================
# 4. Check that feature names and importance match
# ============================================================

if len(importance) != len(feature_names):

    raise ValueError(
        f"Feature mismatch: model has {len(importance)} "
        f"features but feature_names has {len(feature_names)}"
    )


# ============================================================
# 5. Print top 20 features
# ============================================================

indices = np.argsort(importance)[::-1]

print("\n" + "=" * 50)
print("TOP 20 MOST IMPORTANT FEATURES")
print("=" * 50)

for rank, index in enumerate(indices[:20], start=1):

    print(
        f"{rank:2d}. "
        f"{feature_names[index]:20s} "
        f"{importance[index]:.6f}"
    )


# ============================================================
# 6. Plot importance of 5 molecular descriptors
# ============================================================

descriptor_importance = importance[:5]

sorted_indices = np.argsort(
    descriptor_importance
)[::-1]

sorted_names = [
    descriptor_names[i]
    for i in sorted_indices
]

sorted_values = [
    descriptor_importance[i]
    for i in sorted_indices
]


plt.figure(figsize=(9, 5))

plt.barh(
    sorted_names[::-1],
    sorted_values[::-1]
)

plt.xlabel("Feature Importance")
plt.ylabel("Molecular Descriptor")

plt.title(
    "Gradient Boosting Feature Importance"
)

plt.tight_layout()

plt.savefig(
    "feature_importance_descriptors.png",
    dpi=300
)

plt.show()


# ============================================================
# 7. Save all feature importance values
# ============================================================



importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

importance_df.to_csv(
    "feature_importance.csv",
    index=False
)

print("\nSaved:")
print("feature_importance.csv")
print("feature_importance_descriptors.png")

