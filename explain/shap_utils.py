"""
explain/shap_utils.py  —  SHAP utilities (Credit Card dataset)
"""
import shap, joblib, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_model_path = os.path.join(BASE_DIR, "model", "model.pkl")
_model      = joblib.load(_model_path)
explainer   = shap.TreeExplainer(_model)

with open(os.path.join(BASE_DIR, "model", "features.json")) as f:
    FEATURES = json.load(f)

# Human-readable labels for feature names
FEAT_LABELS = {
    "PAY_0":         "Last Payment Status",
    "PAY_1":         "Payment Status -1mo",
    "PAY_2":         "Payment Status -2mo",
    "util_ratio":    "Credit Utilisation",
    "late_months":   "Months with Late Pay",
    "LIMIT_BAL":     "Credit Limit",
    "BILL_AMT1":     "Last Bill Amount",
    "PAY_AMT1":      "Last Payment Amount",
    "repay_rate":    "Repayment Rate",
    "net_balance":   "Net Balance",
    "avg_bill_3m":   "Avg Bill (3 mo)",
    "avg_pay_3m":    "Avg Payment (3 mo)",
    "unused_credit": "Unused Credit",
    "AGE":           "Age",
    "EDUCATION":     "Education",
    "MARRIAGE":      "Marital Status",
}


def get_explanation(input_data: list) -> list[tuple]:
    arr = np.array([input_data], dtype=float)
    n   = _model.n_features_in_
    if   arr.shape[1] > n: arr = arr[:, :n]
    elif arr.shape[1] < n: arr = np.hstack([arr, np.zeros((1, n-arr.shape[1]))])

    sv = explainer.shap_values(arr)
    if isinstance(sv, list): sv = sv[1]
    vals  = sv.flatten()
    feats = FEATURES[:len(vals)]
    # Use human-readable labels
    labeled = [(FEAT_LABELS.get(f, f), v) for f, v in zip(feats, vals)]
    return sorted(labeled, key=lambda x: abs(x[1]), reverse=True)[:3]


def generate_plot(explanation: list) -> str:
    features = [x[0] for x in explanation]
    values   = [x[1] for x in explanation]
    colors   = ["#ef4444" if v > 0 else "#22c55e" for v in values]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(features, values, color=colors, edgecolor="white", height=0.5)
    ax.axvline(0, color="#64748b", linewidth=0.8)
    ax.set_xlabel("SHAP value  (positive = increases default risk)")
    ax.set_title("Top Feature Contributions")
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()
    out = os.path.join(BASE_DIR, "static", "plot.png")
    plt.savefig(out, dpi=100); plt.close()
    return "static/plot.png"


def generate_reason(exp: list) -> str:
    return "; ".join(
        f"{f} {'↑ risk' if v > 0 else '↓ risk'}" for f, v in exp[:2])
