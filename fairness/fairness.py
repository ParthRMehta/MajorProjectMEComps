"""
fairness/fairness.py  —  Credit Card Risk Flags
"""


def check_risk_flags(feature_map: dict) -> list[dict]:
    """Returns list of { label, level: warning|caution|ok, detail }."""
    flags = []

    pay_0         = feature_map.get("PAY_0", 0)
    late_months   = feature_map.get("late_months", 0)
    util_ratio    = feature_map.get("util_ratio", 0)
    limit_bal     = feature_map.get("LIMIT_BAL", 0)
    bill_amt1     = feature_map.get("BILL_AMT1", 0)
    pay_amt1      = feature_map.get("PAY_AMT1", 0)
    repay_rate    = feature_map.get("repay_rate", 1)

    # Payment delay — strongest predictor
    if pay_0 >= 3:
        flags.append({"label": "Severe Payment Delay", "level": "warning",
                      "detail": f"Last payment was {pay_0} months late — very high default indicator."})
    elif pay_0 == 2:
        flags.append({"label": "Payment 2 Months Late", "level": "warning",
                      "detail": "Last payment was 2 months overdue — elevated default risk."})
    elif pay_0 == 1:
        flags.append({"label": "Payment 1 Month Late", "level": "caution",
                      "detail": "Last payment was 1 month overdue — monitor repayment behaviour."})

    # Utilisation ratio
    if util_ratio > 0.9:
        flags.append({"label": "Critical Utilisation", "level": "warning",
                      "detail": f"Using {util_ratio*100:.1f}% of credit limit — severely over-extended."})
    elif util_ratio > 0.7:
        flags.append({"label": "High Utilisation", "level": "caution",
                      "detail": f"Using {util_ratio*100:.1f}% of credit limit — above comfortable 70%."})

    # Repayment rate
    if repay_rate < 0.05 and bill_amt1 > 0:
        flags.append({"label": "Very Low Repayment", "level": "warning",
                      "detail": f"Paid only {repay_rate*100:.1f}% of bill — minimum or missed payment."})
    elif repay_rate < 0.20 and bill_amt1 > 0:
        flags.append({"label": "Low Repayment Rate", "level": "caution",
                      "detail": f"Paid {repay_rate*100:.1f}% of outstanding bill — below recommended 20%."})

    # Late months count
    if late_months >= 2:
        flags.append({"label": "Recurring Late Payments", "level": "warning",
                      "detail": f"{late_months} of last 3 months had late payments — pattern of delay."})

    # Low credit limit with high bill
    if limit_bal > 0 and bill_amt1 > limit_bal * 0.95:
        flags.append({"label": "Bill Exceeds Credit Limit", "level": "warning",
                      "detail": "Outstanding bill is at or above the credit limit — over-limit risk."})

    if not flags:
        flags.append({"label": "No Risk Flags", "level": "ok",
                      "detail": "Payment history, utilisation, and repayment rate are all within healthy thresholds."})
    return flags
