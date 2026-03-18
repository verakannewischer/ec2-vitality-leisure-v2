"""
train_model.py  –  Vitality Leisure Park  –  Visitor Forecast Model (v2)
Run once offline:  python3 train_model.py

Improvements vs v1:
  - School & public holidays for NRW as binary features
  - Numeric weather: temp_c (float) and precip_mm (float) replace categories
  - Weather category kept as fallback one-hot (for days without numeric data)
  - Capacity max defined here for use in app.py
"""

import pandas as pd
import numpy as np
import json, joblib, holidays
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score

MAX_CAPACITY = 2400  # maximum comfortable daily capacity

# ── 1. Load & normalise ────────────────────────────────────────────────────
df = pd.read_excel("data.xlsx", parse_dates=["date"])

df["weather_condition"] = (
    df["weather_condition"].str.strip().str.lower()
    .replace({"bewölkt": "cloudy", "bewölkt ": "cloudy"})
)
df["temp_category"] = df["temp_category"].str.strip().str.lower()

print(f"Training rows  : {len(df)}")
print(f"Date range     : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"Weather values : {sorted(df['weather_condition'].unique())}")
print(f"Temp values    : {sorted(df['temp_category'].unique())}")

# ── 2. Merge numeric weather if available ─────────────────────────────────
try:
    wx_hist = pd.read_csv("weather_history.csv", parse_dates=["date"])
    wx_hist["date"] = wx_hist["date"].dt.normalize()
    df["date_norm"] = df["date"].dt.normalize()
    df = df.merge(wx_hist[["date", "temp_c", "precip_mm"]], 
                  left_on="date_norm", right_on="date", 
                  how="left", suffixes=("", "_wx"))
    df.drop(columns=["date_wx", "date_norm"], errors="ignore", inplace=True)
    has_numeric_weather = df["temp_c"].notna().sum() > 100
    print(f"Numeric weather: {df['temp_c'].notna().sum()} rows matched")
except FileNotFoundError:
    has_numeric_weather = False
    df["temp_c"]    = np.nan
    df["precip_mm"] = np.nan
    print("weather_history.csv not found — using category features only")

# Fill missing numeric weather with category-based estimates
temp_map_c = {"freezing": -3, "cool": 8, "mild": 14, "warm": 20, "hot": 28}
df["temp_c"]    = df["temp_c"].fillna(df["temp_category"].map(temp_map_c))
df["precip_mm"] = df["precip_mm"].fillna(
    df["weather_condition"].map({"sunny": 0, "cloudy": 0.5, "rainy": 8, "snowy": 5})
)

# ── 3. Public & school holidays (NRW) ─────────────────────────────────────
years = range(int(df["year"].min()), int(df["year"].max()) + 2)
nrw_holidays = holidays.Germany(state="NW", years=years)
holiday_dates = set(nrw_holidays.keys())

# NRW school holidays (official dates 2008–2025, summer + christmas + easter)
# Source: Schulministerium NRW
school_holiday_ranges = [
    # Format: (start, end) inclusive — major holiday periods only
    # Summer holidays (6 weeks)
    ("2008-06-23", "2008-08-05"), ("2009-06-29", "2009-08-11"),
    ("2010-07-05", "2010-08-17"), ("2011-07-11", "2011-08-23"),
    ("2012-07-09", "2012-08-21"), ("2013-06-27", "2013-08-06"),
    ("2014-07-07", "2014-08-19"), ("2015-06-29", "2015-08-11"),
    ("2016-06-27", "2016-08-09"), ("2017-06-26", "2017-08-08"),
    ("2018-07-16", "2018-08-28"), ("2019-07-15", "2019-08-27"),
    ("2020-06-29", "2020-08-11"), ("2021-06-28", "2021-08-10"),
    ("2022-06-27", "2022-08-09"), ("2023-06-22", "2023-08-01"),
    ("2024-07-08", "2024-08-20"), ("2025-06-23", "2025-08-05"),
    # Christmas holidays
    ("2008-12-22", "2009-01-06"), ("2009-12-23", "2010-01-06"),
    ("2010-12-24", "2011-01-08"), ("2011-12-23", "2012-01-07"),
    ("2012-12-21", "2013-01-04"), ("2013-12-23", "2014-01-03"),
    ("2014-12-22", "2015-01-06"), ("2015-12-23", "2016-01-06"),
    ("2016-12-23", "2017-01-06"), ("2017-12-22", "2018-01-05"),
    ("2018-12-21", "2019-01-04"), ("2019-12-23", "2020-01-06"),
    ("2020-12-23", "2021-01-06"), ("2021-12-24", "2022-01-07"),
    ("2022-12-23", "2023-01-06"), ("2023-12-22", "2024-01-05"),
    ("2024-12-23", "2025-01-06"),
    # Easter holidays (2 weeks)
    ("2008-03-17", "2008-04-01"), ("2009-04-06", "2009-04-18"),
    ("2010-03-29", "2010-04-10"), ("2011-04-18", "2011-05-02"),
    ("2012-04-02", "2012-04-14"), ("2013-03-25", "2013-04-06"),
    ("2014-04-14", "2014-04-26"), ("2015-03-30", "2015-04-11"),
    ("2016-03-21", "2016-04-02"), ("2017-04-10", "2017-04-22"),
    ("2018-03-26", "2018-04-07"), ("2019-04-15", "2019-04-27"),
    ("2020-04-06", "2020-04-18"), ("2021-03-29", "2021-04-10"),
    ("2022-04-11", "2022-04-23"), ("2023-04-03", "2023-04-15"),
    ("2024-03-25", "2024-04-06"), ("2025-04-14", "2025-04-26"),
]

school_holiday_dates = set()
for start, end in school_holiday_ranges:
    r = pd.date_range(start, end)
    school_holiday_dates.update(r.date)

df["is_public_holiday"] = df["date"].dt.date.apply(
    lambda d: 1 if d in holiday_dates else 0
)
df["is_school_holiday"] = df["date"].dt.date.apply(
    lambda d: 1 if d in school_holiday_dates else 0
)

print(f"Public holidays: {df['is_public_holiday'].sum()} days flagged")
print(f"School holidays: {df['is_school_holiday'].sum()} days flagged")

# ── 4. Cyclical & derived features ────────────────────────────────────────
df["month_sin"] = np.sin(2 * np.pi * df["month"]       / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"]       / 12)
df["wd_sin"]    = np.sin(2 * np.pi * df["weekday_num"] / 7)
df["wd_cos"]    = np.cos(2 * np.pi * df["weekday_num"] / 7)
df["doy_sin"]   = np.sin(2 * np.pi * df["day_of_year"] / 365)
df["doy_cos"]   = np.cos(2 * np.pi * df["day_of_year"] / 365)
df["season"]    = df["month"].map({
    12:0,1:0,2:0, 3:1,4:1,5:1, 6:2,7:2,8:2, 9:3,10:3,11:3
})

# ── 5. One-hot encodings ───────────────────────────────────────────────────
wd_dummies = pd.get_dummies(df["weekday_num"], prefix="wday").astype(int)
wx_dummies = pd.get_dummies(df["weather_condition"], prefix="wx").astype(int)
tc_dummies = pd.get_dummies(df["temp_category"], prefix="tc").astype(int)
df = pd.concat([df, wd_dummies, wx_dummies, tc_dummies], axis=1)

temp_order = {"freezing": 0, "cool": 1, "mild": 2, "warm": 3, "hot": 4}
df["temp_num"] = df["temp_category"].map(temp_order).fillna(1)

# ── 6. Feature list ───────────────────────────────────────────────────────
wday_cols = sorted([c for c in df.columns if c.startswith("wday_")])
wx_cols   = sorted([c for c in df.columns if c.startswith("wx_")])
tc_cols   = sorted([c for c in df.columns if c.startswith("tc_")])

BASE_FEATURES = [
    "weekday_num", "is_weekend", "month", "season",
    "month_sin", "month_cos",
    "wd_sin", "wd_cos",
    "doy_sin", "doy_cos",
    "temp_num",
    "temp_c", "precip_mm",           # NEW: numeric weather
    "is_public_holiday",              # NEW: public holiday flag
    "is_school_holiday",              # NEW: school holiday flag
]
FEATURES = BASE_FEATURES + wday_cols + wx_cols + tc_cols

assert len(FEATURES) == len(set(FEATURES)), "Duplicate features!"
print(f"\nTotal features : {len(FEATURES)}")

TARGET = "total_visitors"
X = df[FEATURES].fillna(0)
y = df[TARGET]

# ── 7. Train ──────────────────────────────────────────────────────────────
model = GradientBoostingRegressor(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.04,
    subsample=0.8,
    min_samples_leaf=5,
    random_state=42,
)
model.fit(X, y)

# ── 8. Evaluate ───────────────────────────────────────────────────────────
cv_mae    = -cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")
preds     = model.predict(X)
r2        = r2_score(y, preds)
train_mae = mean_absolute_error(y, preds)

print(f"\nCV MAE   : {cv_mae.mean():.0f} ± {cv_mae.std():.0f} visitors")
print(f"Train MAE: {train_mae:.0f}  |  R²: {r2:.3f}")
print(f"Mean visitors: {y.mean():.0f}")

# ── 9. Save ───────────────────────────────────────────────────────────────
joblib.dump(model, "model.joblib")

meta = {
    "features":              FEATURES,
    "wday_cols":             wday_cols,
    "wx_cols":               wx_cols,
    "tc_cols":               tc_cols,
    "temp_order":            temp_order,
    "cv_mae":                float(cv_mae.mean()),
    "r2":                    float(r2),
    "mean_visitors":         float(y.mean()),
    "std_visitors":          float(y.std()),
    "max_capacity":          MAX_CAPACITY,
    "weather_categories":    sorted(df["weather_condition"].unique().tolist()),
    "temp_categories":       sorted(df["temp_category"].unique().tolist()),
    "has_numeric_weather":   bool(has_numeric_weather),
}
with open("model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

monthly_avg = df.groupby("month")["total_visitors"].mean().to_dict()
with open("monthly_avg.json", "w") as f:
    json.dump({str(k): round(v, 1) for k, v in monthly_avg.items()}, f)

wd_avg = df.groupby("weekday_num")["total_visitors"].mean().to_dict()
with open("weekday_avg.json", "w") as f:
    json.dump({str(int(k)): round(v, 1) for k, v in wd_avg.items()}, f)

ym_avg = df.groupby(["year", "month"])["total_visitors"].mean().reset_index()
ym_avg.columns = ["year", "month", "avg_visitors"]
ym_avg.to_csv("yearmonth_avg.csv", index=False)

print("\nSaved: model.joblib, model_meta.json, monthly_avg.json, weekday_avg.json, yearmonth_avg.csv")
