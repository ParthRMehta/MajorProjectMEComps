"""
app.py  —  UCI Taiwan Credit Card Default
==========================================
LOCATION : project root
RUN AS   : python app.py

Form inputs (user-friendly):
  LIMIT_BAL  — Credit limit balance
  AGE        — Age of cardholder
  EDUCATION  — Education level (1-4)
  MARRIAGE   — Marital status (1-3)
  PAY_0      — Payment status last month (-2 to 8)
  BILL_AMT1  — Bill statement last month
  PAY_AMT1   — Payment amount last month
"""

from flask import Flask, render_template, request
import joblib, json, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, "templates"),
            static_folder  =os.path.join(BASE_DIR, "static"))

_model_path = os.path.join(BASE_DIR, "model", "model.pkl")
if not os.path.exists(_model_path):
    raise FileNotFoundError("model/model.pkl not found. Run  python model/train_model.py  first.")

model = joblib.load(_model_path)

with open(os.path.join(BASE_DIR, "model", "features.json")) as f:
    FEATURE_COLS = json.load(f)

OPT_THRESHOLD = 0.5
_meta = os.path.join(BASE_DIR, "model", "model_meta.json")
if os.path.exists(_meta):
    with open(_meta) as f:
        OPT_THRESHOLD = float(json.load(f).get("optimal_threshold", 0.5))

from explain.shap_utils import get_explanation, generate_plot, generate_reason
from fairness.fairness  import check_risk_flags

print(f"[app] Model     : {type(model).__name__}")
print(f"[app] Threshold : {OPT_THRESHOLD:.4f}")
print(f"[app] Features  : {FEATURE_COLS}")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        limit_bal = float(request.form["limit_bal"])
        age       = int(request.form["age"])
        education = int(request.form["education"])
        marriage  = int(request.form["marriage"])
        pay_0     = int(request.form["pay_0"])
        bill_amt1 = float(request.form["bill_amt1"])
        pay_amt1  = float(request.form["pay_amt1"])

        if limit_bal <= 0:          raise ValueError("Credit limit must be positive.")
        if not (18 <= age <= 100):  raise ValueError("Age must be between 18 and 100.")
        if education not in [1,2,3,4]: raise ValueError("Invalid education level.")
        if marriage   not in [1,2,3]:  raise ValueError("Invalid marriage status.")
        if bill_amt1  < 0:          raise ValueError("Bill amount cannot be negative.")
        if pay_amt1   < 0:          raise ValueError("Payment amount cannot be negative.")

    except (KeyError, ValueError) as exc:
        return render_template("error.html", error=str(exc)), 400

    # Feature engineering — mirrors train_model.py exactly
    # Clip bill amounts to >= 0 (negatives = credits in UCI dataset)
    bill_amt1_pos = max(bill_amt1, 0)

    util_ratio    = bill_amt1_pos / (limit_bal + 1)
    pay_ratio     = pay_amt1      / (bill_amt1_pos + 100)
    late_months   = int(pay_0 > 0)
    net_balance   = bill_amt1_pos - pay_amt1
    avg_bill_3m   = bill_amt1_pos
    avg_pay_3m    = pay_amt1
    repay_rate    = pay_amt1      / (bill_amt1_pos + 100)
    unused_credit = limit_bal     - bill_amt1_pos

    feature_map = {
        "LIMIT_BAL":     limit_bal,
        "AGE":           age,
        "EDUCATION":     education,
        "MARRIAGE":      marriage,
        "PAY_0":         pay_0,
        "PAY_1":         0,            # no history: assume paid duly
        "PAY_2":         0,
        "PAY_3":         0,
        "PAY_4":         0,
        "PAY_5":         0,
        "PAY_6":         0,
        "BILL_AMT1":     bill_amt1,
        "BILL_AMT2":     bill_amt1,    # approx with last month
        "BILL_AMT3":     bill_amt1,
        "BILL_AMT4":     bill_amt1,
        "BILL_AMT5":     bill_amt1,
        "BILL_AMT6":     bill_amt1,
        "PAY_AMT1":      pay_amt1,
        "PAY_AMT2":      pay_amt1,
        "PAY_AMT3":      pay_amt1,
        "PAY_AMT4":      pay_amt1,
        "PAY_AMT5":      pay_amt1,
        "PAY_AMT6":      pay_amt1,
        "SEX":           0,
        "util_ratio":    util_ratio,
        "pay_ratio":     pay_ratio,
        "late_months":   late_months,
        "net_balance":   net_balance,
        "avg_bill_3m":   avg_bill_3m,
        "avg_pay_3m":    avg_pay_3m,
        "repay_rate":    repay_rate,
        "unused_credit": unused_credit,
    }
    input_data = [feature_map.get(f, 0.0) for f in FEATURE_COLS]

    prob       = float(model.predict_proba([input_data])[0][1])
    pred       = int(prob >= OPT_THRESHOLD)
    result     = "Approved" if pred == 0 else "Rejected"
    confidence = round((1-prob)*100 if pred == 0 else prob*100, 1)

    pay_labels = {-2:"No Consumption",-1:"Paid in Full",0:"Revolving Credit",
                   1:"1 Month Late",2:"2 Months Late",3:"3 Months Late",
                   4:"4 Months Late",5:"5 Months Late",6:"6+ Months Late"}
    edu_labels = {1:"Graduate School",2:"University",3:"High School",4:"Other"}
    mar_labels = {1:"Married",2:"Single",3:"Other"}

    explanation = get_explanation(input_data)
    risk_flags  = check_risk_flags(feature_map)
    plot        = generate_plot(explanation)
    reason      = generate_reason(explanation)

    with open(os.path.join(BASE_DIR,"static","metrics.json")) as fh:
        metrics = json.load(fh)
    cv_path    = os.path.join(BASE_DIR,"static","cv_summary.json")
    cv_summary = json.load(open(cv_path)) if os.path.exists(cv_path) else {}

    return render_template(
        "result.html",
        prediction  = result,
        confidence  = f"{confidence:.1f}",
        prob_risk   = f"{prob*100:.1f}",
        explanation = explanation,
        risk_flags  = risk_flags,
        plot_path   = plot,
        reason      = reason,
        metrics     = metrics,
        cv_summary  = cv_summary,
        inputs      = {
            "Credit Limit":   f"${limit_bal:,.0f}",
            "Age":            str(age),
            "Education":      edu_labels.get(education, str(education)),
            "Marital Status": mar_labels.get(marriage,  str(marriage)),
            "Payment Status": pay_labels.get(pay_0,     str(pay_0)),
            "Last Bill":      f"${bill_amt1:,.0f}",
            "Last Payment":   f"${pay_amt1:,.0f}",
            "Utilisation":    f"{util_ratio*100:.1f}%",
        },
    )


if __name__ == "__main__":
    app.run(debug=True)