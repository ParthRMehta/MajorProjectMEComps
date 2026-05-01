"""
train_model.py  —  UCI Taiwan Credit Card Default
==================================================
LOCATION : model/train_model.py
RUN AS   : python model/train_model.py  (from project root)

Expected runtime : 20-40 seconds
Expected metrics (on real UCI dataset):
  ROC-AUC          : 77-82%
  Balanced Accuracy: 72-78%
  Recall           : 65-75%   ← catching actual defaulters
  F1               : 55-65%
"""

import os, sys, time, warnings
import pandas as pd, numpy as np
import joblib, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble        import RandomForestClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.metrics         import (accuracy_score, precision_score, recall_score,
                                      f1_score, roc_auc_score, confusion_matrix,
                                      roc_curve, auc, precision_recall_curve,
                                      balanced_accuracy_score)

T0 = time.time()
def log(msg, sub=False):
    prefix = "         " if sub else f"  [{time.time()-T0:5.1f}s]  "
    print(f"{prefix}{msg}", flush=True)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data",   "credit_card_clean.csv")
STATIC_DIR = os.path.join(BASE_DIR, "static")
MODEL_DIR  = os.path.join(BASE_DIR, "model")
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

print("=" * 62)
print("  CreditSense  —  train_model.py  (Taiwan CC Dataset)")
print("=" * 62)

# ── Load ───────────────────────────────────────────────────────────────────
log("Step 1 — Loading clean dataset ...")
if not os.path.exists(DATA_PATH):
    log(f"[ERROR] {DATA_PATH} not found. Run  python clean_dataset.py  first.")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
df = df[df["default"].isin([0,1])].copy()
log(f"Loaded  {len(df):,} rows  ×  {len(df.columns)} columns")
log(f"Classes : {df['default'].value_counts().to_dict()}", sub=True)

# Signal check
r = df[["PAY_0","default"]].corr().iloc[0,1] if "PAY_0" in df.columns else 0.1
log(f"Signal  : PAY_0↔default  r={r:.4f}  (need >0.10)")
if abs(r) < 0.05:
    log("[FATAL] Near-zero signal. Re-run clean_dataset.py with the correct CSV.")
    sys.exit(1)
log("Signal  : OK ✓")

# ── Feature engineering ────────────────────────────────────────────────────
log("Step 2 — Feature engineering ...")

def engineer(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()

    # BILL_AMT columns can be negative in the UCI dataset (credits/refunds).
    # Clip to >= 0 before any division to prevent zero/near-zero denominators → inf.
    if "BILL_AMT1" in d.columns:
        bill1_pos = d["BILL_AMT1"].clip(lower=0)
    else:
        bill1_pos = pd.Series(0, index=d.index)

    if "BILL_AMT1" in d.columns and "LIMIT_BAL" in d.columns:
        d["util_ratio"]    = bill1_pos / (d["LIMIT_BAL"] + 1)
    if "PAY_AMT1"  in d.columns and "BILL_AMT1" in d.columns:
        d["pay_ratio"]     = d["PAY_AMT1"] / (bill1_pos + 100)  # +100 avoids near-zero
    pay_cols = [c for c in d.columns if c.startswith("PAY_")
                and not c.startswith("PAY_AMT")][:3]
    if pay_cols:
        d["late_months"]   = sum((d[c] > 0).astype(int) for c in pay_cols)
    else:
        d["late_months"]   = 0
    if "BILL_AMT1" in d.columns and "PAY_AMT1" in d.columns:
        d["net_balance"]   = bill1_pos - d["PAY_AMT1"]
    bill_c = [c for c in d.columns if c.startswith("BILL_AMT")][:3]
    if bill_c:
        d["avg_bill_3m"]   = d[bill_c].clip(lower=0).mean(axis=1)
    pamt_c = [c for c in d.columns if c.startswith("PAY_AMT")][:3]
    if pamt_c:
        d["avg_pay_3m"]    = d[pamt_c].mean(axis=1)
    if "avg_pay_3m" in d.columns and "avg_bill_3m" in d.columns:
        d["repay_rate"]    = d["avg_pay_3m"] / (d["avg_bill_3m"] + 100)  # +100 safe
    if "LIMIT_BAL" in d.columns and "BILL_AMT1" in d.columns:
        d["unused_credit"] = d["LIMIT_BAL"] - bill1_pos

    # Safety net: replace any inf/NaN produced by edge cases
    d.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in d.select_dtypes(include=[np.number]).columns:
        if d[col].isnull().any():
            d[col].fillna(d[col].median(), inplace=True)

    return d

df        = engineer(df)
FEAT_COLS = [c for c in df.columns if c != "default"]
log(f"Features ({len(FEAT_COLS)}): {FEAT_COLS}")

X = df[FEAT_COLS]
y = df["default"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
log(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

# ── Model training ─────────────────────────────────────────────────────────
log("Step 3 — Training RandomForest (150 trees) ...")
log("(should finish in ~10-20 s)", sub=True)

model = RandomForestClassifier(
    n_estimators     = 150,
    max_depth        = 15,
    min_samples_leaf = 3,
    max_features     = "sqrt",
    class_weight     = "balanced",
    n_jobs           = -1,
    random_state     = 42,
)
model.fit(X_train, y_train)
log("Training complete ✓")

y_prob    = model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_prob)
log(f"Test AUC: {auc_score*100:.2f}%")

if auc_score < 0.60:
    log("[FATAL] AUC < 60%. The CSV may be corrupted. Re-run clean_dataset.py.")
    sys.exit(1)

# ── Optimal threshold ──────────────────────────────────────────────────────
log("Step 4 — Optimal threshold (Youden's J) ...")
fpr_a, tpr_a, thr_a = roc_curve(y_test, y_prob)
j_idx      = int(np.argmax(tpr_a - fpr_a))
opt_thresh = float(thr_a[j_idx])
y_pred     = (y_prob >= opt_thresh).astype(int)
log(f"Threshold : {opt_thresh:.4f}")
log(f"Approval  : {(y_pred==0).mean()*100:.1f}%  |  Rejection: {(y_pred==1).mean()*100:.1f}%")

# ── Metrics ────────────────────────────────────────────────────────────────
log("Step 5 — Metrics ...")
acc  = accuracy_score(y_test,  y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test,    y_pred, zero_division=0)
f1   = f1_score(y_test,        y_pred, zero_division=0)
bauc = roc_auc_score(y_test,   y_prob)
bal  = balanced_accuracy_score(y_test, y_pred)

metrics = {
    "accuracy":          f"{acc*100:.2f}%",
    "precision":         f"{prec*100:.2f}%",
    "recall":            f"{rec*100:.2f}%",
    "f1":                f"{f1*100:.2f}%",
    "roc_auc":           f"{bauc*100:.2f}%",
    "balanced_accuracy": f"{bal*100:.2f}%",
}
with open(os.path.join(STATIC_DIR, "metrics.json"), "w") as fh:
    json.dump(metrics, fh, indent=2)

print()
print("  ╔══════════════════════════════════════════╗")
print("  ║           FINAL METRICS                  ║")
print("  ╠══════════════════════════════════════════╣")
for k, v in metrics.items():
    print(f"  ║  {k:<24}  {v:>8}          ║")
print("  ╚══════════════════════════════════════════╝")

# ── Plots ──────────────────────────────────────────────────────────────────
log("Step 6 — Generating plots ...")

# Confusion Matrix
log("  confusion_matrix ...", sub=True)
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_title("Confusion Matrix", fontweight="bold", fontsize=13)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(["No Default","Default"])
ax.set_yticklabels(["No Default","Default"])
fig.colorbar(im, ax=ax)
for i in range(2):
    for j in range(2):
        ax.text(j,i,f"{cm[i,j]:,}",ha="center",va="center",fontsize=13,
                color="white" if cm[i,j]>cm.max()*0.5 else "black")
plt.tight_layout(); plt.savefig(os.path.join(STATIC_DIR,"confusion_matrix.png"),dpi=100); plt.close()

# ROC Curve
log("  roc_curve ...", sub=True)
fig, ax = plt.subplots(figsize=(6,5))
ax.plot(fpr_a,tpr_a,color="#2563eb",lw=2.5,label=f"RF (AUC={bauc:.3f})")
ax.plot([0,1],[0,1],"--",color="#94a3b8",lw=1.5,label="Random")
ax.scatter(fpr_a[j_idx],tpr_a[j_idx],color="#f59e0b",s=90,zorder=5,
           label=f"Threshold={opt_thresh:.3f}")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve",fontweight="bold"); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(STATIC_DIR,"roc_curve.png"),dpi=100); plt.close()

# Precision-Recall
log("  pr_curve ...", sub=True)
prec_a, rec_a, _ = precision_recall_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(6,5))
ax.plot(rec_a,prec_a,color="#7c3aed",lw=2.5,label="Random Forest")
ax.axhline(y_test.mean(),color="#94a3b8",linestyle="--",label="Baseline")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve",fontweight="bold"); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(STATIC_DIR,"pr_curve.png"),dpi=100); plt.close()

# Feature Importance
log("  feature_importance ...", sub=True)
imp = pd.Series(model.feature_importances_, index=FEAT_COLS).sort_values()
colors_i = ["#2563eb" if v>=imp.quantile(0.75) else "#93c5fd" for v in imp]
fig, ax = plt.subplots(figsize=(9, max(4, len(imp)*0.36)))
imp.plot(kind="barh",ax=ax,color=colors_i,edgecolor="white")
ax.set_title("Feature Importance — Random Forest",fontweight="bold")
ax.set_xlabel("Importance"); plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR,"feature_importance.png"),dpi=100); plt.close()

# Model Comparison
log("  model_comparison ...", sub=True)
comp = {
    "Random Forest":  model,
    "Logistic Reg.":  LogisticRegression(max_iter=300, C=1.0, random_state=42),
    "Decision Tree":  DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=42),
    "RF (shallow)":   RandomForestClassifier(n_estimators=50,max_depth=6,
                                              class_weight="balanced",n_jobs=-1,random_state=42),
}
comp_scores = {}
for name, m in comp.items():
    if name != "Random Forest": m.fit(X_train, y_train)
    pc = m.predict_proba(X_test)[:,1]
    fr, tr, th = roc_curve(y_test, pc)
    bt = th[np.argmax(tr-fr)]
    comp_scores[name] = {
        "accuracy": accuracy_score(y_test,(pc>=bt).astype(int))*100,
        "bal_acc":  balanced_accuracy_score(y_test,(pc>=bt).astype(int))*100,
        "auc":      auc(fr,tr)*100,
    }

labels = list(comp_scores.keys())
x_pos  = np.arange(len(labels)); w = 0.26
fig, ax = plt.subplots(figsize=(12,5))
b1=ax.bar(x_pos-w,  [comp_scores[k]["accuracy"] for k in labels], w,label="Accuracy (%)",color="#2563eb",edgecolor="white")
b2=ax.bar(x_pos,    [comp_scores[k]["bal_acc"]  for k in labels], w,label="Balanced Acc (%)",color="#059669",edgecolor="white")
b3=ax.bar(x_pos+w,  [comp_scores[k]["auc"]      for k in labels], w,label="AUC (%)",color="#7c3aed",edgecolor="white")
ax.set_ylim(50,100); ax.set_xticks(x_pos)
ax.set_xticklabels(labels,rotation=15,ha="right",fontsize=9)
ax.set_title("Model Comparison",fontweight="bold"); ax.set_ylabel("Score (%)"); ax.legend()
for bar in list(b1)+list(b2)+list(b3):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f"{bar.get_height():.1f}",ha="center",va="bottom",fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR,"model_comparison.png"),dpi=100); plt.close()
log("  All plots saved ✓", sub=True)

# ── CV summary ─────────────────────────────────────────────────────────────
cv_summary = {"cv_auc_mean": f"{bauc*100:.2f}%","cv_auc_std":"±N/A","cv_folds":"80/20 Split"}
with open(os.path.join(STATIC_DIR,"cv_summary.json"),"w") as fh: json.dump(cv_summary, fh)

# ── Save artefacts ─────────────────────────────────────────────────────────
log("Step 7 — Saving model artefacts ...")
joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
joblib.dump(model, os.path.join(MODEL_DIR, "shap_model.pkl"))
model_meta = {
    "feature_cols":      FEAT_COLS,
    "optimal_threshold": opt_thresh,
    "model_type":        "RandomForestClassifier",
    "test_auc":          round(bauc*100, 2),
    "dataset":           "UCI Taiwan Credit Card Default",
}
with open(os.path.join(MODEL_DIR,"features.json"),   "w") as fh: json.dump(FEAT_COLS,   fh, indent=2)
with open(os.path.join(MODEL_DIR,"model_meta.json"), "w") as fh: json.dump(model_meta, fh, indent=2)
log("model.pkl, shap_model.pkl, features.json, model_meta.json saved ✓")

print()
print("=" * 62)
print(f"  TRAINING COMPLETE  —  {time.time()-T0:.1f}s")
print("=" * 62)
print(f"\n  Next: python app.py")