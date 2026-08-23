import subprocess
import sys

steps = [
    "src/model.py",
    "src/evaluate.py",
    "src/model_comparison.py",
    "src/error.py",
    "src/feature_importance.py",
    "src/ad_threshold.py",
    "src/ad_threshold_validation.py",
    "src/ad_error_threshold.py",
    "src/esol_comparison.py",
    "src/shap_analysis.py",
    "src/final_results.py",
]

for step in steps:
    print(f"\nRunning {step}...")
    result = subprocess.run([sys.executable, step])

    if result.returncode != 0:
        print(f"FAILED: {step}")
        sys.exit(1)

print("\n================================")
print("PIPELINE COMPLETED SUCCESSFULLY")
print("================================")