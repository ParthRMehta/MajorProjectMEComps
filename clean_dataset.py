"""
clean_dataset.py  —  UCI Taiwan Credit Card Default
====================================================
LOCATION : project root
RUN AS   : python clean_dataset.py

Dataset  : UCI Default of Credit Card Clients
Source   : https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
File     : UCI_Credit_Card.csv   (place in data/ folder)

Expected runtime : ~10 seconds
Expected output  : data/credit_card_clean.csv  (30 000 rows)

PREPROCESSING PIPELINE
-----------------------
1.  Load CSV
2.  Rename target column
3.  Remove invalid category codes
4.  Impute missing values (median)
5.  IQR outlier removal
6.  Feature engineering (8 derived features)
7.  Signal verification
8.  Feature selection (variance + correlation + target-correlation)
9.  Export clean CSV + feature charts

FEATURE SELECTION TECHNIQUES APPLIED
--------------------------------------
A.  Variance Threshold   — drop near-zero-variance columns
B.  Pearson Correlation  — drop one from each |r|>0.92 pair
C.  Target Correlation   — drop features with |r|<0.01 with target
D.  RF Embedded Imp.     — keep features with importance >= 0.005
E.  Final selection      — union of C∩D kept, plus MUST_KEEP core
"""

import os, sys, time, warnings
import pandas as pd
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")

T0 = time.time()
def log(msg, sub=False):
    prefix = "         " if sub else f"  [{time.time()-T0:5.1f}s]  "
    print(f"{prefix}{msg}", flush=True)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
STATIC_DIR = os.path.join(BASE_DIR, "static")
OUT_CSV    = os.path.join(DATA_DIR, "credit_card_clean.csv")
os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

print("=" * 62)
print("  CreditSense  —  clean_dataset.py  (Taiwan CC Dataset)")
print("=" * 62)

# ── Find input file ────────────────────────────────────────────────────────
CANDIDATES = [
    os.path.join(DATA_DIR, "UCI_Credit_Card.csv"),
    os.path.join(DATA_DIR, "default of credit card clients.csv"),
    os.path.join(DATA_DIR, "credit_card.csv"),
    os.path.join(DATA_DIR, "UCI_Credit_Card.xls"),
    os.path.join(BASE_DIR, "UCI_Credit_Card.csv"),
]

log("Searching for input file ...")
INPUT_FILE = None
for p in CANDIDATES:
    found = os.path.exists(p)
    log(f"{'✓ FOUND' if found else '✗ not found'}  {p}", sub=True)
    if found and INPUT_FILE is None:
        INPUT_FILE = p

if INPUT_FILE is None:
    print()
    print("  [ERROR] Dataset not found.")
    print()
    print("  HOW TO GET THE DATASET:")
    print("  1. Go to: https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset")
    print("     OR:     https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients")
    print("  2. Download  UCI_Credit_Card.csv")
    print("  3. Place it in the  data/  folder")
    print("  4. Re-run this script")
    sys.exit(1)

size_kb = os.path.getsize(INPUT_FILE) / 1024
log(f"Using: {os.path.basename(INPUT_FILE)}  ({size_kb:.0f} KB)")

# ── Step 1 – Load ──────────────────────────────────────────────────────────
log("Step 1 — Loading CSV ...")

# Handle XLS format if needed
if INPUT_FILE.endswith(".xls"):
    df = pd.read_excel(INPUT_FILE, header=1)   # row 0 is a second header in the XLS
else:
    df = pd.read_csv(INPUT_FILE)

log(f"Loaded {len(df):,} rows  ×  {len(df.columns)} columns")
log(f"Columns: {list(df.columns)}", sub=True)

# ── Step 2 – Standardise column names ─────────────────────────────────────
log("Step 2 — Standardising column names ...")

# Rename target
for possible_target in ["default.payment.next.month","default payment next month","y","Y"]:
    if possible_target in df.columns:
        df.rename(columns={possible_target: "default"}, inplace=True)
        break

if "default" not in df.columns:
    log(f"[ERROR] Target column not found. Available: {list(df.columns)}")
    sys.exit(1)

# Drop ID column if present
for id_col in ["ID","id"]:
    if id_col in df.columns:
        df.drop(columns=[id_col], inplace=True)

# Uppercase all column names for consistency
df.columns = [c.upper().replace(" ","_").replace(".","_") for c in df.columns]
df.rename(columns={"DEFAULT":"default"}, inplace=True)
log(f"Target  : 'default'  |  Features: {[c for c in df.columns if c!='default']}")

# ── Step 3 – Remove invalid category codes ─────────────────────────────────
log("Step 3 — Removing invalid category values ...")
before = len(df)
# EDUCATION: valid values 1,2,3,4 (0 and 5,6 are undocumented)
if "EDUCATION" in df.columns:
    df = df[df["EDUCATION"].isin([1,2,3,4])]
# MARRIAGE: valid values 1,2,3
if "MARRIAGE" in df.columns:
    df = df[df["MARRIAGE"].isin([1,2,3])]
log(f"Removed {before-len(df):,} rows with invalid codes  →  {len(df):,} remain")
vc = df["default"].value_counts()
log(f"Class 0 (No default)  : {vc.get(0,0):,}", sub=True)
log(f"Class 1 (Default)     : {vc.get(1,0):,}", sub=True)
log(f"Default rate          : {vc.get(1,0)/len(df)*100:.1f}%", sub=True)

# ── Step 4 – Convert dtypes ────────────────────────────────────────────────
log("Step 4 — Converting data types ...")
for col in df.columns:
    if col != "default":
        df[col] = pd.to_numeric(df[col], errors="coerce")
log(f"Nulls: {df.isnull().sum().sum()}")

# ── Step 5 – Impute missing values ─────────────────────────────────────────
log("Step 5 — Imputing missing values (median) ...")
for col in df.columns:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)
log(f"Nulls after imputation: {df.isnull().sum().sum()}")

# ── Step 6 – IQR outlier removal on continuous features ───────────────────
log("Step 6 — IQR outlier removal ...")
before = len(df)
for col in ["LIMIT_BAL","BILL_AMT1","BILL_AMT2","BILL_AMT3",
            "PAY_AMT1","PAY_AMT2","PAY_AMT3"]:
    if col in df.columns:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR    = Q3 - Q1
        df     = df[(df[col] >= Q1-3*IQR) & (df[col] <= Q3+3*IQR)]
log(f"Removed {before-len(df):,}  →  {len(df):,} remain")

# ── Step 7 – Feature engineering ──────────────────────────────────────────
log("Step 7 — Engineering features ...")

# NOTE: BILL_AMT columns can be negative in the UCI dataset (credits/refunds).
# When BILL_AMT1 = -1, using (BILL_AMT1 + 1) as a denominator produces 0 → inf.
# Fix: clip BILL amounts to >= 0 before any division, and add a safe epsilon.

BILL_AMT1_pos = df["BILL_AMT1"].clip(lower=0)

# Utilisation ratio — most informative single feature
# LIMIT_BAL is always >= 10 000 in this dataset, so +1 is safe
df["util_ratio"]   = BILL_AMT1_pos / (df["LIMIT_BAL"] + 1)

# Payment-to-bill ratio — repayment behaviour
# Use clipped bill (>=0) + 100 as denominator to avoid near-zero issues
df["pay_ratio"]    = df["PAY_AMT1"] / (BILL_AMT1_pos + 100)

# How many of the last 3 months had late payments
pay_cols = [c for c in df.columns if c.startswith("PAY_") and not c.startswith("PAY_AMT")]
if pay_cols:
    late_months = sum((df[c] > 0).astype(int) for c in pay_cols[:3])
else:
    late_months = pd.Series(0, index=df.index)
df["late_months"]  = late_months

# Net balance after payment
df["net_balance"]  = BILL_AMT1_pos - df["PAY_AMT1"]

# Average bill over last 3 months (clipped to >= 0)
bill_cols = [c for c in df.columns if c.startswith("BILL_AMT")][:3]
df["avg_bill_3m"]  = df[bill_cols].clip(lower=0).mean(axis=1)

# Average payment over last 3 months
pamt_cols = [c for c in df.columns if c.startswith("PAY_AMT")][:3]
df["avg_pay_3m"]   = df[pamt_cols].mean(axis=1)

# Repayment rate — safe denominator (+100 avoids near-zero for zero-bill months)
df["repay_rate"]   = df["avg_pay_3m"] / (df["avg_bill_3m"] + 100)

# Unused credit
df["unused_credit"]= df["LIMIT_BAL"] - BILL_AMT1_pos

log(f"Shape after engineering: {df.shape}")

# ── Safety net: replace any remaining inf/NaN introduced by engineering ────
inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
nan_count = df.isnull().sum().sum()
if inf_count > 0 or nan_count > 0:
    log(f"  Cleaning up {inf_count} inf and {nan_count} NaN values ...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
log(f"  inf/NaN after cleanup: {np.isinf(df.select_dtypes(include=[np.number])).sum().sum()}")

# ── Step 8 – Signal verification ──────────────────────────────────────────
log("Step 8 — Signal verification ...")
for col in ["PAY_0","util_ratio","late_months","LIMIT_BAL"]:
    if col in df.columns:
        r = df[[col,"default"]].corr().iloc[0,1]
        ok = "✓" if abs(r) > 0.05 else "✗ LOW"
        log(f"  {col:<18}  r={r:+.4f}  {ok}", sub=True)

r_main = df[["PAY_0","default"]].corr().iloc[0,1] if "PAY_0" in df.columns else 0
if abs(r_main) < 0.05:
    log("[ERROR] Near-zero signal. Check the input file format.")
    sys.exit(1)
log("Signal check PASSED ✓")

# ═══════════════════════════════════════════════════════════════════════════
# FEATURE SELECTION
# ═══════════════════════════════════════════════════════════════════════════
TARGET    = "default"
ALL_FEATS = [c for c in df.columns if c != TARGET]
X = df[ALL_FEATS].copy()
y = df[TARGET].copy()

log("Step 9 — Feature selection ...")

# A. Variance Threshold
log("  A — Variance Threshold (threshold=0.01) ...", sub=True)
vt   = VarianceThreshold(threshold=0.01)
vt.fit(X)
mask = vt.get_support()
drop_vt = [f for f,k in zip(ALL_FEATS, mask) if not k]
X = X.loc[:, mask]
log(f"     Dropped {len(drop_vt)}: {drop_vt or 'none'}", sub=True)

# B. Correlation Filter
log("  B — Correlation filter (|r|>0.92) ...", sub=True)
corr_m = X.corr().abs()
upper  = corr_m.where(np.triu(np.ones(corr_m.shape), k=1).astype(bool))
drop_c = [c for c in upper.columns if any(upper[c] > 0.92)]
X.drop(columns=drop_c, inplace=True)
log(f"     Dropped {len(drop_c)}: {drop_c or 'none'}", sub=True)

# Save correlation heatmap
fig, ax = plt.subplots(figsize=(14, 12))
corr_all = df[list(X.columns)].corr()
im = ax.imshow(corr_all, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(X.columns))); ax.set_yticks(range(len(X.columns)))
ax.set_xticklabels(X.columns, rotation=45, ha="right", fontsize=7)
ax.set_yticklabels(X.columns, fontsize=7)
fig.colorbar(im, ax=ax, shrink=0.6)
ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, "correlation_heatmap.png"), dpi=100); plt.close()

# C. Target Correlation Filter
log("  C — Target correlation filter (|r|<0.01 dropped) ...", sub=True)
ct = df[list(X.columns)+[TARGET]].corr()[TARGET].abs()
drop_t = [f for f in X.columns if ct.get(f, 0) < 0.01]
X.drop(columns=drop_t, inplace=True)
log(f"     Dropped {len(drop_t)}: {drop_t or 'none'}", sub=True)

# D. RF Embedded Importance (on 8 000 row sample for speed)
log("  D — RF Embedded Importance (8 000 row sample) ...", sub=True)
idx = np.random.RandomState(42).choice(len(X), min(8000, len(X)), replace=False)
rf_sel = RandomForestClassifier(n_estimators=100, max_depth=8,
                                  class_weight="balanced", random_state=42, n_jobs=-1)
rf_sel.fit(X.iloc[idx], y.iloc[idx])
rf_imp = pd.Series(rf_sel.feature_importances_, index=X.columns).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, max(4, len(rf_imp)*0.35)))
rf_imp.sort_values().plot(kind="barh", ax=ax, color="#7c3aed", edgecolor="white")
ax.set_title("Feature Importance (RF Selection)", fontweight="bold")
ax.set_xlabel("Importance"); plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, "rf_importance.png"), dpi=100); plt.close()

drop_rf = list(rf_imp[rf_imp < 0.005].index)
X.drop(columns=drop_rf, inplace=True)
log(f"     Dropped {len(drop_rf)} low-importance: {drop_rf or 'none'}", sub=True)

# E. Final — always keep core features
MUST_KEEP = ["LIMIT_BAL","AGE","EDUCATION","MARRIAGE","PAY_0","BILL_AMT1","PAY_AMT1"]
for f in MUST_KEEP:
    if f in df.columns and f not in X.columns:
        X[f] = df[f]

selected = list(X.columns)
log(f"  E — Final features ({len(selected)}): {selected}")

# ── MI chart ──────────────────────────────────────────────────────────────
from sklearn.feature_selection import mutual_info_classif
mi = pd.Series(
    mutual_info_classif(X.iloc[idx], y.iloc[idx], random_state=42),
    index=X.columns).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 4))
mi.plot(kind="bar", ax=ax, color="#2563eb", edgecolor="white")
ax.set_title("Mutual Information Scores", fontweight="bold")
ax.set_ylabel("MI Score"); plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, "mutual_information.png"), dpi=100); plt.close()

# ── Step 10 – Export ───────────────────────────────────────────────────────
log("Step 10 — Exporting clean CSV ...")
df_out = df[selected + [TARGET]].copy()
df_out.to_csv(OUT_CSV, index=False)
log(f"Saved  →  {OUT_CSV}")
log(f"Shape  :  {df_out.shape}  |  Classes: {df_out[TARGET].value_counts().to_dict()}")

print()
print("=" * 62)
print(f"  DONE  —  {time.time()-T0:.1f}s")
print("=" * 62)
print(f"\n  Next: python model/train_model.py")