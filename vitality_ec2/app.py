"""
Vitality Leisure Park — Web Application
Flask backend serving HTML frontend
"""

from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import json, joblib, requests, holidays
from datetime import date, timedelta
import os
import cohere

app = Flask(__name__)

# ── Load ML assets ────────────────────────────────────────────────────────────
model = joblib.load("model.joblib")
with open("model_meta.json")  as f: meta    = json.load(f)
with open("monthly_avg.json") as f: mon_avg = {int(k): v for k,v in json.load(f).items()}
with open("weekday_avg.json") as f: wd_avg  = {int(k): v for k,v in json.load(f).items()}
ym_df  = pd.read_csv("yearmonth_avg.csv")
raw_df = pd.read_excel("data.xlsx", parse_dates=["date"])

FEATURES     = meta["features"]
TEMP_ORDER   = meta["temp_order"]
MEAN_V       = meta["mean_visitors"]
MAX_CAPACITY = meta.get("max_capacity", 2400)

# ── RAG embeddings ────────────────────────────────────────────────────────────
with open("embeddings.json") as f:
    emb_data = json.load(f)
doc_chunks     = emb_data["chunks"]
doc_embeddings = np.array(emb_data["embeddings"], dtype="float32")

COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "")
co = cohere.ClientV2(COHERE_API_KEY) if COHERE_API_KEY else None

# ── Holiday helpers ───────────────────────────────────────────────────────────
def get_holiday_sets():
    yrs = range(2024, 2028)
    nrw = holidays.Germany(state="NW", years=yrs)
    pub = set(nrw.keys())
    ranges = [
        ("2024-06-27","2024-08-09"),("2024-10-14","2024-10-26"),
        ("2024-12-23","2025-01-06"),("2025-03-31","2025-04-12"),
        ("2025-06-23","2025-08-05"),("2025-10-06","2025-10-18"),
        ("2025-12-22","2026-01-05"),("2026-03-30","2026-04-11"),
        ("2026-06-22","2026-08-04"),
    ]
    sch = set()
    for s, e in ranges:
        for d in pd.date_range(s, e): sch.add(d.date())
    return pub, sch

pub_holidays, school_holidays = get_holiday_sets()

# ── Weather ───────────────────────────────────────────────────────────────────
LAT, LON = 52.0833, 8.75

def fetch_weather(days=14):
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}"
           f"&daily=weathercode,temperature_2m_max,temperature_2m_min,precipitation_sum"
           f"&timezone=Europe%2FBerlin&forecast_days={days}")
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        d  = r.json()["daily"]
        df = pd.DataFrame(d)
        df["date"]          = pd.to_datetime(df["time"])
        df["temp_avg"]      = (df["temperature_2m_max"] + df["temperature_2m_min"]) / 2
        df["temp_c"]        = df["temp_avg"]
        df["precip_mm"]     = df["precipitation_sum"].fillna(0)
        df["weather_label"] = df["weathercode"].apply(_wmo)
        df["temp_category"] = df["temp_avg"].apply(_tcat)
        return df
    except:
        return None

def _wmo(c):
    if c in [0,1]: return "sunny"
    if c in [2,3]: return "cloudy"
    if 51<=c<=69 or 80<=c<=82: return "rainy"
    if 71<=c<=77 or 85<=c<=86: return "snowy"
    return "cloudy"

def _tcat(t):
    if t < 0:  return "freezing"
    if t < 12: return "cool"
    if t < 18: return "mild"
    if t < 25: return "warm"
    return "hot"

WD_FULL  = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
WD_SHORT = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
MON_NAMES= {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
             7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

def predict(dt, wx, tc, temp_c=None, precip_mm=None):
    ts  = pd.Timestamp(dt)
    row = {f: 0 for f in FEATURES}
    row["weekday_num"]       = ts.dayofweek
    row["is_weekend"]        = int(ts.dayofweek >= 5)
    row["month"]             = ts.month
    row["season"]            = {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}[ts.month]
    row["month_sin"]         = np.sin(2*np.pi*ts.month/12)
    row["month_cos"]         = np.cos(2*np.pi*ts.month/12)
    row["wd_sin"]            = np.sin(2*np.pi*ts.dayofweek/7)
    row["wd_cos"]            = np.cos(2*np.pi*ts.dayofweek/7)
    row["doy_sin"]           = np.sin(2*np.pi*ts.day_of_year/365)
    row["doy_cos"]           = np.cos(2*np.pi*ts.day_of_year/365)
    row["temp_num"]          = TEMP_ORDER.get(tc, 1)
    row["is_public_holiday"] = int(dt in pub_holidays)
    row["is_school_holiday"] = int(dt in school_holidays)
    tf = {"freezing":-3,"cool":8,"mild":14,"warm":20,"hot":28}
    row["temp_c"]    = temp_c    if temp_c    is not None else tf.get(tc, 14)
    row["precip_mm"] = precip_mm if precip_mm is not None else (8 if wx=="rainy" else 0)
    if f"wday_{ts.dayofweek}" in row: row[f"wday_{ts.dayofweek}"] = 1
    if f"wx_{wx}"             in row: row[f"wx_{wx}"]             = 1
    if f"tc_{tc}"             in row: row[f"tc_{tc}"]             = 1
    X = pd.DataFrame([row])[FEATURES]
    return max(0, int(round(model.predict(X)[0])))

def crowd_level(v):
    if v < 900:  return "quiet",    "#5C8A3C", "Quiet"
    if v < 1350: return "moderate", "#8B6E4E", "Moderate"
    return               "busy",    "#8B3A2C", "Busy"

def build_forecast(n=14):
    wx_df = fetch_weather(n)
    rows  = []
    for i in range(n):
        d  = date.today() + timedelta(days=i)
        ts = pd.Timestamp(d)
        if wx_df is not None and i < len(wx_df):
            r    = wx_df.iloc[i]
            wx   = r["weather_label"]
            tc   = r["temp_category"]
            tmp  = float(r["temp_avg"])
            prec = float(r["precip_mm"])
        else:
            wx, tc, tmp, prec = "cloudy","mild", None, None
        v = predict(d, wx, tc, tmp, prec)
        cl, col, label = crowd_level(v)
        rows.append(dict(
            date=d.strftime("%a %d %b"),
            date_raw=d.isoformat(),
            weekday=WD_FULL[ts.dayofweek],
            weekday_short=WD_SHORT[ts.dayofweek],
            wx=wx, tc=tc,
            temp=f"{tmp:.0f}°C" if tmp else tc.title(),
            precip=f"{prec:.1f}mm" if prec else "—",
            visitors=v,
            cap_pct=min(100, round(v/MAX_CAPACITY*100, 1)),
            crowd_level=cl,
            crowd_color=col,
            crowd_label=label,
            is_public_holiday=int(d in pub_holidays),
            is_school_holiday=int(d in school_holidays),
        ))
    return rows

def retrieve_rag(query, top_k=5):
    if not co: return []
    q_resp = co.embed(texts=[query], model="embed-english-v3.0",
                      input_type="search_query", embedding_types=["float"])
    q_vec  = np.array(q_resp.embeddings.float_[0], dtype="float32")
    q_norm = q_vec / np.linalg.norm(q_vec)
    scores  = doc_embeddings @ q_norm
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(doc_chunks[i], float(scores[i])) for i in top_idx]

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    wx_df = fetch_weather(1)
    today_weather = None
    if wx_df is not None:
        t = wx_df.iloc[0]
        today_weather = {
            "label": t["weather_label"].title(),
            "temp": f"{t['temp_avg']:.0f}°C",
            "is_holiday": date.today() in pub_holidays,
            "is_school": date.today() in school_holidays,
        }
    return render_template("index.html", weather=today_weather,
                           cv_mae=meta["cv_mae"], r2=meta["r2"],
                           max_capacity=MAX_CAPACITY)

@app.route("/manager")
def manager():
    fc = build_forecast(14)
    fc7 = fc[:7]
    total7  = sum(r["visitors"] for r in fc7)
    avg7    = total7 / 7
    mon_ref = mon_avg.get(date.today().month, MEAN_V)
    avg_cap = sum(r["cap_pct"] for r in fc7) / 7
    peak    = max(fc7, key=lambda r: r["visitors"])
    quiet   = min(fc7, key=lambda r: r["visitors"])

    # Historical heatmap data
    raw_f = raw_df[~raw_df["year"].isin([2020,2021])].copy()
    raw_f["weekday_num"] = pd.to_datetime(raw_f["date"]).dt.dayofweek
    raw_f["month"]       = pd.to_datetime(raw_f["date"]).dt.month
    pivot = raw_f.pivot_table(values="total_visitors", index="weekday_num",
                               columns="month", aggfunc="mean")
    heatmap_data = {
        "days":   [WD_FULL[i] for i in pivot.index],
        "months": [MON_NAMES[c] for c in pivot.columns],
        "values": [[round(v, 0) for v in row] for row in pivot.values.tolist()],
    }

    import calendar
    today = date.today()
    monthly = []
    for i in range(6):
        m    = (today.month - 1 + i) % 12 + 1
        y    = today.year + ((today.month - 1 + i) // 12)
        avg  = mon_avg.get(m, MEAN_V)
        days = calendar.monthrange(y, m)[1]
        monthly.append({"label": f"{MON_NAMES[m]} {y}", "expected": int(avg * days)})

    return render_template("manager.html", fc7=fc7, fc=fc,
                           total7=total7, avg7=round(avg7), mon_ref=round(mon_ref),
                           avg_cap=round(avg_cap,1), peak=peak, quiet=quiet,
                           delta=avg7-mon_ref, max_capacity=MAX_CAPACITY,
                           heatmap=heatmap_data, monthly=monthly)

@app.route("/visitor")
def visitor():
    fc = build_forecast(14)
    return render_template("visitor.html", fc=fc, max_capacity=MAX_CAPACITY)

@app.route("/wellness")
def wellness():
    return render_template("wellness.html")

@app.route("/api/forecast")
def api_forecast():
    n = int(request.args.get("n", 14))
    return jsonify(build_forecast(n))

@app.route("/api/chat", methods=["POST"])
def api_chat():
    if not co:
        return jsonify({"reply": "Wellness Coach is not configured. Please set COHERE_API_KEY."})
    data     = request.json
    messages = data.get("messages", [])
    user_msg = messages[-1]["content"] if messages else ""

    # RAG retrieval
    retrieved = retrieve_rag(user_msg, top_k=5)
    rag_ctx   = ""
    if retrieved:
        rag_ctx = "\n\nRELEVANT FACILITY INFORMATION:\n"
        for chunk, _ in retrieved:
            rag_ctx += f"- {chunk.get('section','')}: {chunk['text']}\n"

    fc        = build_forecast(7)
    fc_lines  = "\n".join(
        f"- {r['weekday']} {r['date']}: {r['visitors']:,} visitors ({r['crowd_level']} crowd, {r['cap_pct']}% capacity), {r['wx']}, {r['temp']}"
        for r in fc
    )

    system = f"""You are a warm, expert wellness coach at Vitality Leisure Park.

7-DAY VISITOR FORECAST:
{fc_lines}
{rag_ctx}

Create personalised spa day plans. Be warm, concise, and grounded in the retrieved information.
Recommend specific menu items, fitness classes with real times, and the best day based on crowd levels.
Do not mention AI, Cohere, or RAG."""

    api_messages = [{"role": "system", "content": system}]
    for msg in messages:
        if msg["role"] in ["user", "assistant"]:
            api_messages.append({"role": msg["role"], "content": msg["content"]})

    try:
        response = co.chat(model="command-a-03-2025", messages=api_messages)
        reply = response.message.content[0].text
    except Exception as e:
        reply = f"Something went wrong: {e}"

    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
