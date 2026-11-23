"""Freight rate prediction: time-based validation, final fit, predictions."""
import json, warnings
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
warnings.filterwarnings("ignore")

D = Path("data")
train = pd.read_csv(D / "train_test.csv", parse_dates=["date"])
val = pd.read_csv(D / "validation.csv", parse_dates=["date"])
dec = pd.read_csv(D / "december_chart_inputs.csv", parse_dates=["date"])

CITIES = sorted(set(train.pickup) | set(train.delivery) | set(val.pickup) | set(val.delivery))
CITY = {c: i for i, c in enumerate(CITIES)}
EQ = {"Dry Van": 0, "Flatbed": 1, "Reefer": 2}
COLS = ["log_dist", "weight", "equip", "dow", "pickup", "delivery", "quote_signal"]
PARAMS = dict(max_iter=800, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=20,
              categorical_features=[2, 4, 5], random_state=0)

def features(d):
    return pd.DataFrame({
        "log_dist": np.log(d.distance), "weight": d.weight.abs(),      # negative weights = sign errors
        "equip": d.equipment.map(EQ), "dow": d.date.dt.dayofweek,
        "pickup": d.pickup.map(CITY), "delivery": d.delivery.map(CITY),
        "quote_signal": d.quote_signal}, index=d.index)[COLS]

def fit(df):
    """Two-pass fit: rough rate/mile filter, then drop rows with large residuals."""
    rpm = df.posted_rate / df.distance
    a = df[(rpm > 1.2) & (rpm < 4.5)]
    m = HistGradientBoostingRegressor(**PARAMS).fit(features(a), np.log(a.posted_rate))
    resid = np.log(df.posted_rate) - m.predict(features(df))
    a = df[resid.abs() < 0.25]
    return HistGradientBoostingRegressor(**PARAMS).fit(features(a), np.log(a.posted_rate)), len(df) - len(a)

def metrics(y, p):
    return dict(MAE=mean_absolute_error(y, p), RMSE=float(np.sqrt(mean_squared_error(y, p))),
                MAPE=float(np.mean(np.abs(y - p) / y) * 100), MedAPE=float(np.median(np.abs(y - p) / y) * 100))

# ---- 1. Rolling-origin (time-based) validation -------------------------------
FOLDS = {"Fold 1: train Jan-Apr -> test May-Jun": ("2025-05-01", "2025-07-01"),
         "Fold 2: train Jan-Jun -> test Jul-Aug": ("2025-07-01", "2025-09-01"),
         "Fold 3: train Jan-Aug -> test Sep-Oct": ("2025-09-01", "2025-11-01")}
rows, last = [], None
for name, (s, e) in FOLDS.items():
    tr, te = train[train.date < s], train[(train.date >= s) & (train.date < e)]
    m, _ = fit(tr)
    p = np.exp(m.predict(features(te)))
    rpm = te.posted_rate / te.distance
    clean = (rpm > 1.2) & (rpm < 4.5)
    # naive baseline: distance x median rate/mile by equipment
    trr = tr.posted_rate / tr.distance
    base = tr[(trr > 1.2) & (trr < 4.5)].assign(r=trr).groupby("equipment").r.median()
    pb = te.distance * te.equipment.map(base)
    rows.append({"fold": name, "n_test": len(te),
                 **{f"model_{k}": v for k, v in metrics(te.posted_rate, p).items()},
                 **{f"baseline_{k}": v for k, v in metrics(te.posted_rate, pb).items()},
                 **{f"model_clean_{k}": v for k, v in metrics(te.posted_rate[clean], p[clean]).items()},
                 "pct_rows_outlier": float((~clean).mean() * 100)})
    last = (te, p)
cv = pd.DataFrame(rows); cv.to_csv("cv_results.csv", index=False)
print(cv.round(2).T.to_string())

# ---- 2. Final model on all labelled data ------------------------------------
model, dropped = fit(train)
print("rows dropped as corrupted in final fit:", dropped, f"({dropped/len(train):.2%})")

# ---- 3. Validation predictions ----------------------------------------------
vp = np.exp(model.predict(features(val)))
tmpl = pd.read_csv(D / "validation_predictions_template.csv")
out = tmpl[["load_id"]].merge(pd.DataFrame({"load_id": val.load_id, "predicted_rate": np.round(vp, 2)}), on="load_id", how="left")
assert out.predicted_rate.notna().all() and (out.predicted_rate > 0).all() and len(out) == 12000
out.to_csv("validation_predictions.csv", index=False)

# ---- 4. December chart: only the date varies --------------------------------
# December inputs have no quote_signal; use the observed daily average for that date from validation.csv
qs_daily = val.groupby("date").quote_signal.mean()
d = dec.copy()
d["quote_signal"] = d.date.map(qs_daily)
d["weight"] = d.weight.astype(float)
d["predicted_rate"] = np.round(np.exp(model.predict(features(d))), 2)
d[["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]].assign(date=lambda x: x.date.dt.strftime("%Y-%m-%d")).to_csv(D / "december_chart_inputs.csv", index=False)
print(d[["date", "predicted_rate"]].describe().T)

# ---- 5. Diagnostics for the report ------------------------------------------
te, p = last
pd.DataFrame({"actual": te.posted_rate.values, "pred": p, "date": te.date.values}).to_csv("holdout_fold3.csv", index=False)
json.dump({"dropped_rows": int(dropped)}, open("final_fit.json", "w"))
