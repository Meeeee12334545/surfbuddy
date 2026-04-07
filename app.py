import math
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------------------------
# SEQ Live Surf Cam Catalogue
# Comprehensive South East Queensland surf cam database.
#   youtube  : YouTube 24/7 live stream video ID (directly embeddable)
#   swellnet : Swellnet cam slug → swellnet.com/surfcams/{slug}
# ---------------------------------------------------------------------------

SEQ_CAMS = [
    # ── Gold Coast ──────────────────────────────────────────────────────────
    {"name": "Snapper Rocks",     "region": "Gold Coast",     "youtube": "VbnLxUlT4io", "swellnet": "snapper-rocks"},
    {"name": "Kirra Beach",       "region": "Gold Coast",     "swellnet": "kirra"},
    {"name": "Duranbah",          "region": "Gold Coast",     "swellnet": "duranbah"},
    {"name": "Coolangatta Beach", "region": "Gold Coast",     "swellnet": "coolangatta"},
    {"name": "Currumbin Alley",   "region": "Gold Coast",     "swellnet": "currumbin-alley"},
    {"name": "Palm Beach",        "region": "Gold Coast",     "swellnet": "palm-beach"},
    {"name": "Tugun",             "region": "Gold Coast",     "swellnet": "tugun"},
    {"name": "Narrowneck",        "region": "Gold Coast",     "swellnet": "narrowneck"},
    {"name": "Surfers Paradise",  "region": "Gold Coast",     "swellnet": "surfers-paradise"},
    {"name": "Broadbeach",        "region": "Gold Coast",     "swellnet": "broadbeach"},
    {"name": "Mermaid Beach",     "region": "Gold Coast",     "swellnet": "mermaid-beach"},
    {"name": "Miami Beach",       "region": "Gold Coast",     "swellnet": "miami"},
    {"name": "Burleigh Heads",    "region": "Gold Coast",     "swellnet": "burleigh-heads"},
    {"name": "South Stradbroke",  "region": "Gold Coast",     "swellnet": "south-stradbroke"},
    # ── Sunshine Coast ──────────────────────────────────────────────────────
    {"name": "Noosa Heads",        "region": "Sunshine Coast", "swellnet": "noosa-heads"},
    {"name": "Sunshine Beach",     "region": "Sunshine Coast", "swellnet": "sunshine-beach"},
    {"name": "Peregian Beach",     "region": "Sunshine Coast", "swellnet": "peregian"},
    {"name": "Coolum Beach",       "region": "Sunshine Coast", "swellnet": "coolum"},
    {"name": "Alexandra Headland", "region": "Sunshine Coast", "swellnet": "alex-headland"},
    {"name": "Mooloolaba",         "region": "Sunshine Coast", "swellnet": "mooloolaba"},
    {"name": "Maroochydore",       "region": "Sunshine Coast", "swellnet": "maroochydore"},
    {"name": "Kings Beach",        "region": "Sunshine Coast", "swellnet": "kings-beach"},
    {"name": "Caloundra",          "region": "Sunshine Coast", "swellnet": "caloundra"},
]

# ---------------------------------------------------------------------------
# Surf spot catalogue (used in Forecast tab)
# ---------------------------------------------------------------------------

SURF_SPOTS = {
    "Gold Coast": [
        {"name": "Snapper Rocks", "lat": -28.166, "lon": 153.546, "orientation": 90, "break_type": "Point"},
        {"name": "Kirra Beach", "lat": -28.162, "lon": 153.544, "orientation": 85, "break_type": "Beach"},
        {"name": "Burleigh Heads", "lat": -28.085, "lon": 153.453, "orientation": 70, "break_type": "Point"},
        {"name": "Coolangatta Beach", "lat": -28.163, "lon": 153.545, "orientation": 80, "break_type": "Beach"},
    ],
    "Sunshine Coast": [
        {"name": "Noosa Heads", "lat": -26.392, "lon": 153.091, "orientation": 60, "break_type": "Point"},
        {"name": "Mooloolaba Beach", "lat": -26.681, "lon": 153.119, "orientation": 75, "break_type": "Beach"},
        {"name": "Maroochydore", "lat": -26.655, "lon": 153.102, "orientation": 80, "break_type": "Beach"},
    ],
    "Northern NSW": [
        {"name": "Byron Bay – The Pass", "lat": -28.641, "lon": 153.628, "orientation": 45, "break_type": "Point"},
        {"name": "Lennox Head", "lat": -28.797, "lon": 153.588, "orientation": 60, "break_type": "Point"},
        {"name": "Ballina", "lat": -28.871, "lon": 153.563, "orientation": 90, "break_type": "Beach"},
    ],
}

# ---------------------------------------------------------------------------
# Data fetching – Open-Meteo (free, no API key)
# ---------------------------------------------------------------------------

MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"


@st.cache_data(ttl=900)
def fetch_marine(lat: float, lon: float) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "wave_height,wave_period,wave_direction",
        "forecast_days": 7,
        "timezone": "Australia/Sydney",
    }
    r = requests.get(MARINE_URL, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()["hourly"]
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    return df


@st.cache_data(ttl=900)
def fetch_weather(lat: float, lon: float) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "windspeed_10m,winddirection_10m",
        "forecast_days": 7,
        "timezone": "Australia/Sydney",
    }
    r = requests.get(WEATHER_URL, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()["hourly"]
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    return df


def get_forecast(lat: float, lon: float) -> pd.DataFrame:
    marine = fetch_marine(lat, lon)
    weather = fetch_weather(lat, lon)
    df = marine.merge(weather, on="time")
    return df


# ---------------------------------------------------------------------------
# Surf scoring engine
# ---------------------------------------------------------------------------

def _wave_height_score(h: float) -> float:
    """Score wave height 0–10. Sweet spot 0.8–2.5 m."""
    if h < 0.3:
        return 0.0
    if h < 0.8:
        return (h - 0.3) / 0.5 * 5
    if h <= 2.5:
        return 10.0 - (h - 0.8) / 1.7 * 2  # peaks at 0.8, gentle taper
    if h <= 4.0:
        return 8.0 - (h - 2.5) / 1.5 * 5
    return max(0.0, 3.0 - (h - 4.0))


def _period_score(p: float) -> float:
    """Score wave period 0–10. Sweet spot 10–16 s."""
    if p < 5:
        return 0.0
    if p < 10:
        return (p - 5) / 5 * 7
    if p <= 16:
        return 10.0
    return max(0.0, 10.0 - (p - 16) * 0.5)


def _wind_score(speed: float, wind_dir: float, break_orientation: float) -> float:
    """Score wind 0–10. Offshore = best, light = bonus."""
    # Offshore direction = break_orientation + 180 (wind blowing from sea → land)
    offshore_dir = (break_orientation + 180) % 360
    angle_diff = abs(wind_dir - offshore_dir)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
    # Offshore (diff < 45) scores best; onshore (diff > 135) scores worst
    direction_factor = max(0.0, 1.0 - angle_diff / 180)
    # Speed factor: calm is best
    if speed < 10:
        speed_factor = 1.0
    elif speed < 20:
        speed_factor = 1.0 - (speed - 10) / 20
    else:
        speed_factor = max(0.0, 0.5 - (speed - 20) / 40)
    return round(10 * direction_factor * speed_factor, 1)


def _swell_direction_score(wave_dir: float, break_orientation: float) -> float:
    """Score swell direction vs break orientation 0–10."""
    diff = abs(wave_dir - break_orientation)
    if diff > 180:
        diff = 360 - diff
    return max(0.0, 10.0 - diff / 18)


def surf_score(wave_height, wave_period, wave_direction, wind_speed, wind_direction, break_orientation) -> float:
    """Composite surf score 0–10."""
    h = _wave_height_score(float(wave_height))
    p = _period_score(float(wave_period))
    w = _wind_score(float(wind_speed), float(wind_direction), break_orientation)
    s = _swell_direction_score(float(wave_direction), break_orientation)
    return round((h * 0.35 + p * 0.25 + w * 0.25 + s * 0.15), 1)


def score_label(score: float) -> str:
    if score >= 8:
        return "🔥 Epic"
    if score >= 6:
        return "✅ Good"
    if score >= 4:
        return "🟡 Fair"
    return "❌ Poor"


def score_color(score: float) -> str:
    if score >= 8:
        return "#00cc44"
    if score >= 6:
        return "#66cc00"
    if score >= 4:
        return "#ffaa00"
    return "#ff4444"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def current_row(df: pd.DataFrame) -> pd.Series:
    """Return the row closest to now."""
    now = pd.Timestamp.now(tz="Australia/Sydney").tz_convert(df["time"].dt.tz)
    idx = (df["time"] - now).abs().idxmin()
    return df.loc[idx]


def best_window(df: pd.DataFrame, spot: dict) -> pd.DataFrame:
    """Return next 48 h rows sorted by surf score descending."""
    now = pd.Timestamp.now(tz="Australia/Sydney").tz_convert(df["time"].dt.tz)
    future = df[df["time"] >= now].head(48).copy()
    future["score"] = future.apply(
        lambda r: surf_score(
            r["wave_height"], r["wave_period"], r["wave_direction"],
            r["windspeed_10m"], r["winddirection_10m"], spot["orientation"],
        ),
        axis=1,
    )
    return future.sort_values("score", ascending=False).head(5)


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Surf Buddy 🏄", page_icon="🏄", layout="wide")

# Light, clean theme
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #f7f9fc; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e4e8ef;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e4e8ef;
        border-radius: 10px;
        padding: 14px 18px;
    }

    /* Headings */
    h1 { font-size: 2rem !important; font-weight: 700 !important; color: #1a1a2e !important; }
    h2, h3 { color: #1a1a2e !important; font-weight: 600 !important; }

    /* Tabs */
    [data-testid="stTabs"] button { font-weight: 500; }

    /* Caption / small text */
    .stCaption { color: #6c757d !important; }

    /* Divider */
    hr { border-color: #e4e8ef !important; }
</style>
""", unsafe_allow_html=True)

st.title("🏄 Surf Buddy")
st.caption('"Know before you go. Surf smarter. It\'s on."')

# Sidebar – region & spot selection
with st.sidebar:
    st.header("📍 Select a Spot")
    region = st.selectbox("Region", list(SURF_SPOTS.keys()))
    spots = SURF_SPOTS[region]
    spot_names = [s["name"] for s in spots]
    spot_name = st.selectbox("Surf Spot", spot_names)
    spot = next(s for s in spots if s["name"] == spot_name)

    st.divider()
    st.caption("Data: Open-Meteo Marine API (updated every 15 min)")

# ---------------------------------------------------------------------------
# Fetch data
# ---------------------------------------------------------------------------

with st.spinner("Fetching latest surf data…"):
    try:
        df = get_forecast(spot["lat"], spot["lon"])
    except Exception as e:
        st.error(f"Could not fetch forecast data: {e}")
        st.stop()

# Add computed score column for the full week
df["score"] = df.apply(
    lambda r: surf_score(
        r["wave_height"], r["wave_period"], r["wave_direction"],
        r["windspeed_10m"], r["winddirection_10m"], spot["orientation"],
    )
    if all(pd.notna(r[c]) for c in ["wave_height", "wave_period", "wave_direction", "windspeed_10m", "winddirection_10m"])
    else np.nan,
    axis=1,
)

row = current_row(df)
now_score = surf_score(
    row["wave_height"], row["wave_period"], row["wave_direction"],
    row["windspeed_10m"], row["winddirection_10m"], spot["orientation"],
)

# ---------------------------------------------------------------------------
# Current conditions banner
# ---------------------------------------------------------------------------

st.subheader(f"📡 {spot_name} — Right Now")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("🌊 Wave Height", f"{row['wave_height']:.1f} m")
col2.metric("⏱ Wave Period", f"{row['wave_period']:.0f} s")
col3.metric("💨 Wind Speed", f"{row['windspeed_10m']:.0f} km/h")
col4.metric("🧭 Wind Dir", f"{row['winddirection_10m']:.0f}°")
col5.metric("🏄 Surf Score", f"{now_score} / 10", delta=score_label(now_score))

# Colour-coded score bar
bar_html = f"""
<div style="background:{score_color(now_score)};border-radius:8px;padding:12px 20px;
            color:white;font-size:1.4rem;font-weight:bold;text-align:center;margin-top:8px;">
    {score_label(now_score)} &nbsp;|&nbsp; Score: {now_score} / 10
    &nbsp;·&nbsp; {spot['break_type']} Break
</div>
"""
st.markdown(bar_html, unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# Live Webcam
# ---------------------------------------------------------------------------

st.subheader("📷 Live Webcam")

cam = WEBCAMS.get(spot_name, {})
youtube_id = cam.get("youtube")
swellnet_url = cam.get("swellnet")

if youtube_id:
    components.iframe(
        f"https://www.youtube.com/embed/{youtube_id}?autoplay=1&mute=1&rel=0",
        height=420,
        scrolling=False,
    )
else:
    st.info(
        "No embedded live cam available for this spot yet. "
        "Tap the button below to watch on Swellnet. 🎥",
        icon="📹",
    )

if swellnet_url:
    st.link_button(f"🎥 Watch {spot_name} live cam on Swellnet →", swellnet_url)

st.divider()

# ---------------------------------------------------------------------------
# "Go Surf" recommendation
# ---------------------------------------------------------------------------

st.subheader("🚀 Go Surf Mode")

top = best_window(df, spot)
if not top.empty:
    best = top.iloc[0]
    best_time = best["time"]
    best_sc = best["score"]
    drive_buffer = 30  # minutes

    leave_time = best_time - pd.Timedelta(minutes=drive_buffer)
    st.success(
        f"**Best window:** {best_time.strftime('%A %d %b, %H:%M')}  |  "
        f"Score **{best_sc}/10** {score_label(best_sc)}  |  "
        f"Set your alarm for **{leave_time.strftime('%H:%M')}** 🤙"
    )

    with st.expander("Top 5 windows in next 48 h"):
        display = top[["time", "wave_height", "wave_period", "windspeed_10m", "score"]].copy()
        display.columns = ["Time", "Wave (m)", "Period (s)", "Wind (km/h)", "Score"]
        display["Time"] = display["Time"].dt.strftime("%a %d %b %H:%M")
        display["Score"] = display["Score"].apply(lambda x: f"{x} {score_label(x)}")
        st.dataframe(display, width='stretch', hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Forecast charts
# ---------------------------------------------------------------------------

st.subheader("📈 7-Day Forecast")

tab1, tab2, tab3, tab4 = st.tabs(["Surf Score", "Wave Height & Period", "Wind", "Swell Direction"])

with tab1:
    fig = px.area(
        df.dropna(subset=["score"]),
        x="time", y="score",
        color_discrete_sequence=["#0077b6"],
        labels={"time": "", "score": "Surf Score"},
    )
    fig.update_layout(yaxis_range=[0, 10], hovermode="x unified")
    fig.add_hline(y=6, line_dash="dash", line_color="green", annotation_text="Good")
    fig.add_hline(y=8, line_dash="dash", line_color="gold", annotation_text="Epic")
    st.plotly_chart(fig, width='stretch')

with tab2:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df["time"], y=df["wave_height"], name="Wave Height (m)",
        line=dict(color="#0096c7"), fill="tozeroy", fillcolor="rgba(0,150,199,0.15)",
    ))
    fig2.add_trace(go.Scatter(
        x=df["time"], y=df["wave_period"] / 10, name="Wave Period / 10 (s)",
        line=dict(color="#48cae4", dash="dot"),
        yaxis="y",
    ))
    fig2.update_layout(
        legend=dict(orientation="h"),
        hovermode="x unified",
        yaxis_title="Wave Height (m) | Period / 10 (s)",
    )
    st.plotly_chart(fig2, width='stretch')

with tab3:
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=df["time"], y=df["windspeed_10m"], name="Wind Speed (km/h)",
        line=dict(color="#e63946"), fill="tozeroy", fillcolor="rgba(230,57,70,0.12)",
    ))
    fig3.add_trace(go.Scatter(
        x=df["time"], y=df["winddirection_10m"], name="Wind Direction (°)",
        line=dict(color="#f4a261", dash="dot"),
        yaxis="y2",
    ))
    fig3.update_layout(
        yaxis=dict(title="Wind Speed (km/h)"),
        yaxis2=dict(title="Wind Direction (°)", overlaying="y", side="right", range=[0, 360]),
        legend=dict(orientation="h"),
        hovermode="x unified",
    )
    st.plotly_chart(fig3, width='stretch')

with tab4:
    fig4 = px.scatter_polar(
        df.dropna(subset=["wave_direction"]).head(168),
        r="wave_height",
        theta="wave_direction",
        color="score",
        color_continuous_scale="Blues",
        labels={"wave_direction": "Swell Direction (°)", "wave_height": "Height (m)", "score": "Score"},
    )
    fig4.update_layout(polar=dict(angularaxis=dict(direction="clockwise", rotation=90)))
    st.plotly_chart(fig4, width='stretch')

st.divider()

# ---------------------------------------------------------------------------
# Region dashboard – all spots scored
# ---------------------------------------------------------------------------

st.subheader(f"🗺️ {region} Spot Rankings")

rankings = []
for s in SURF_SPOTS[region]:
    try:
        sdf = get_forecast(s["lat"], s["lon"])
        srow = current_row(sdf)
        sc = surf_score(
            srow["wave_height"], srow["wave_period"], srow["wave_direction"],
            srow["windspeed_10m"], srow["winddirection_10m"], s["orientation"],
        )
        rankings.append({
            "Spot": s["name"],
            "Break": s["break_type"],
            "Wave (m)": round(srow["wave_height"], 1),
            "Period (s)": round(srow["wave_period"], 0),
            "Wind (km/h)": round(srow["windspeed_10m"], 0),
            "Score": sc,
            "Rating": score_label(sc),
        })
    except Exception:
        pass

if rankings:
    rank_df = pd.DataFrame(rankings).sort_values("Score", ascending=False).reset_index(drop=True)
    rank_df.index += 1

    def _color_score(val):
        return f"color: {score_color(val)}; font-weight: bold"

    st.dataframe(
        rank_df.style.map(_color_score, subset=["Score"]),
        width='stretch',
    )

st.caption("🌊 Surf Buddy — 100% free for all surfers. Data via Open-Meteo.")
