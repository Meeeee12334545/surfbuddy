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
    # ── Gold Coast (north → south) ──────────────────────────────────────────
    # Snapper Rocks has a confirmed free YouTube 24/7 live stream.
    # All other Gold Coast spots link to Swellnet (free low-res, no login).
    {"name": "South Stradbroke",  "region": "Gold Coast", "swellnet": "south-stradbroke"},
    {"name": "Duranbah",          "region": "Gold Coast", "swellnet": "duranbah"},
    {"name": "Snapper Rocks",     "region": "Gold Coast", "youtube": "VbnLxUlT4io",
     "swellnet": "snapper-rocks"},
    {"name": "Rainbow Bay",       "region": "Gold Coast", "swellnet": "rainbow-bay"},
    {"name": "Greenmount",        "region": "Gold Coast", "swellnet": "greenmount"},
    {"name": "Kirra Beach",       "region": "Gold Coast", "swellnet": "kirra"},
    {"name": "Coolangatta Beach", "region": "Gold Coast", "swellnet": "coolangatta"},
    {"name": "Currumbin Alley",   "region": "Gold Coast", "swellnet": "currumbin-alley"},
    {"name": "Palm Beach",        "region": "Gold Coast", "swellnet": "palm-beach"},
    {"name": "Tugun",             "region": "Gold Coast", "swellnet": "tugun"},
    {"name": "Burleigh Heads",    "region": "Gold Coast", "swellnet": "burleigh-heads"},
    {"name": "Miami Beach",       "region": "Gold Coast", "swellnet": "miami"},
    {"name": "Mermaid Beach",     "region": "Gold Coast", "swellnet": "mermaid-beach"},
    {"name": "Broadbeach",        "region": "Gold Coast", "swellnet": "broadbeach"},
    {"name": "Surfers Paradise",  "region": "Gold Coast", "swellnet": "surfers-paradise"},
    {"name": "Narrowneck",        "region": "Gold Coast", "swellnet": "narrowneck"},
    # ── Sunshine Coast (south → north) ──────────────────────────────────────
    # Noosa Surf Cam (noosasurfcam.com) is a free, no-login cam site for Noosa spots.
    # Surf Sunshine Coast (surfsunshinecoast.com.au) covers Sunshine Coast breaks.
    {"name": "Caloundra",          "region": "Sunshine Coast", "swellnet": "caloundra"},
    {"name": "Kings Beach",        "region": "Sunshine Coast", "swellnet": "kings-beach"},
    {"name": "Mooloolaba",         "region": "Sunshine Coast", "swellnet": "mooloolaba",
     "extra_url": "https://surfsunshinecoast.com.au/surf-cams/",
     "extra_label": "Surf SC Cams"},
    {"name": "Alexandra Headland", "region": "Sunshine Coast", "swellnet": "alex-headland",
     "extra_url": "https://surfsunshinecoast.com.au/surf-cams/",
     "extra_label": "Surf SC Cams"},
    {"name": "Maroochydore",       "region": "Sunshine Coast", "swellnet": "maroochydore"},
    {"name": "Coolum Beach",       "region": "Sunshine Coast", "swellnet": "coolum"},
    {"name": "Peregian Beach",     "region": "Sunshine Coast", "swellnet": "peregian"},
    {"name": "Sunshine Beach",     "region": "Sunshine Coast", "swellnet": "sunshine-beach",
     "extra_url": "https://noosasurfcam.com/sunshine-beach-surf-cam.html",
     "extra_label": "Noosa Surf Cam"},
    {"name": "Noosa Heads",        "region": "Sunshine Coast", "swellnet": "noosa-heads",
     "extra_url": "https://noosasurfcam.com/",
     "extra_label": "Noosa Surf Cam"},
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
# Cam gallery helpers
# ---------------------------------------------------------------------------

def _cam_card_html(cam: dict) -> str:
    name = cam["name"]
    region = cam["region"]
    swellnet_url = f"https://www.swellnet.com/surfcams/{cam['swellnet']}"
    extra_url = cam.get("extra_url", "")
    extra_label = cam.get("extra_label", "")

    badge_color = "#c1440e" if region == "Gold Coast" else "#1a7a4a"

    if cam.get("youtube"):
        media_html = f"""
        <div style="position:relative;">
          <span style="position:absolute;top:8px;left:8px;z-index:10;
                       background:#e63946;color:white;font-size:0.62rem;
                       font-weight:700;padding:2px 8px;border-radius:4px;
                       letter-spacing:0.08em;pointer-events:none;">&#9679; LIVE</span>
          <iframe
            src="https://www.youtube.com/embed/{cam['youtube']}?autoplay=0&mute=1&rel=0&modestbranding=1&playsinline=1"
            width="100%" height="185" frameborder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowfullscreen style="display:block;"></iframe>
        </div>"""
    else:
        gradient = (
            "linear-gradient(150deg,#023e8a 0%,#0096c7 55%,#48cae4 100%)"
            if region == "Gold Coast"
            else "linear-gradient(150deg,#1b4332 0%,#2d9596 55%,#48cae4 100%)"
        )
        media_html = f"""
        <div style="height:185px;background:{gradient};display:flex;flex-direction:column;
                    align-items:center;justify-content:center;color:white;">
          <div style="font-size:2.6rem;line-height:1">🌊</div>
          <div style="font-size:0.78rem;margin-top:10px;opacity:0.85;text-align:center;padding:0 14px">
            Click <strong>Watch Free</strong> below to view live
          </div>
        </div>"""

    extra_btn = ""
    if extra_url:
        extra_btn = f"""<a href="{extra_url}" target="_blank" rel="noopener"
           style="background:#1a3a5c;color:#90caf9;padding:6px 11px;border-radius:6px;
                  font-size:0.72rem;text-decoration:none;font-weight:600;
                  white-space:nowrap;border:1px solid #2d5986;">
           🔗 {extra_label}
        </a>"""

    return f"""
    <div style="border-radius:12px;overflow:hidden;background:#0f1e2e;
                box-shadow:0 4px 16px rgba(0,0,0,0.4);">
      {media_html}
      <div style="padding:10px 12px;display:flex;justify-content:space-between;
                  align-items:center;gap:6px;">
        <div style="min-width:0;flex:1;">
          <div style="color:#e8f4fd;font-weight:700;font-size:0.9rem;
                      white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{name}</div>
          <span style="display:inline-block;margin-top:3px;background:{badge_color};
                       color:white;font-size:0.62rem;font-weight:700;padding:2px 8px;
                       border-radius:4px;letter-spacing:0.04em;">{region.upper()}</span>
        </div>
        <div style="display:flex;gap:6px;flex-shrink:0;align-items:center;">
          {extra_btn}
          <a href="{swellnet_url}" target="_blank" rel="noopener"
             style="background:#0077b6;color:white;padding:7px 14px;border-radius:7px;
                    font-size:0.78rem;text-decoration:none;font-weight:700;
                    white-space:nowrap;">
            🎥 Watch Free
          </a>
        </div>
      </div>
    </div>"""


def build_cam_gallery(cams: list) -> tuple:
    """Return (html_string, height_px) for the responsive cam card grid."""
    n_rows = math.ceil(len(cams) / 3)
    height = n_rows * 275 + 40
    cards = "\n".join(_cam_card_html(c) for c in cams)
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:transparent;padding:4px 2px 8px}}
  .grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}
</style></head>
<body><div class="grid">{cards}</div></body></html>"""
    return html, height


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Surf Buddy 🏄", page_icon="🏄", layout="wide")

st.markdown("""
<style>
  .stApp { background-color: #0a1628; color: #e8f4fd; }
  [data-testid="stSidebar"] {
    background-color: #0f1e2e;
    border-right: 1px solid #1e3a5c;
  }
  [data-testid="stMetric"] {
    background: #0f1e2e;
    border: 1px solid #1e3a5c;
    border-radius: 10px;
    padding: 14px 18px;
  }
  [data-testid="stMetricValue"] { color: #e8f4fd !important; }
  [data-testid="stMetricLabel"] { color: #90caf9 !important; }
  h1 { font-size: 2rem !important; font-weight: 700 !important; color: #e8f4fd !important; }
  h2, h3 { color: #e8f4fd !important; font-weight: 600 !important; }
  p, li, .stMarkdown { color: #b0c8e0 !important; }
  [data-testid="stTabs"] button { font-weight: 600; color: #90caf9 !important; }
  [data-testid="stTabs"] button[aria-selected="true"] { color: #48cae4 !important; border-bottom-color: #48cae4 !important; }
  .stCaption { color: #5a8aa8 !important; }
  hr { border-color: #1e3a5c !important; }
  [data-testid="stRadio"] label { color: #b0c8e0 !important; }
  .stDataFrame { background: #0f1e2e; }
  [data-testid="stSuccess"] { background: #0d2b1a !important; border-color: #1a7a4a !important; }
  [data-testid="stSuccess"] p { color: #6fcf97 !important; }
  [data-testid="stExpander"] { background: #0f1e2e !important; border-color: #1e3a5c !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:12px 0 4px">
  <span style="font-size:2.2rem;font-weight:800;color:#e8f4fd;">🏄 Surf Buddy</span>
  <span style="font-size:0.95rem;color:#5a8aa8;margin-left:12px;">SEQ Live Surf Cams &amp; Forecast</span>
</div>
""", unsafe_allow_html=True)

# ── Top-level navigation tabs ────────────────────────────────────────────────
tab_cams, tab_forecast = st.tabs(["🎥  Live Cams — SEQ", "📈  Surf Forecast"])

# ============================================================================
# TAB 1 – LIVE CAMS GALLERY
# ============================================================================
with tab_cams:
    st.markdown("""
    <p style="color:#5a8aa8;font-size:0.85rem;margin:-4px 0 12px">
    All cams link to <strong style="color:#90caf9">Swellnet free tier</strong>
    (no subscription needed for basic viewing) or dedicated free cam sites.
    Snapper Rocks has an embedded live YouTube stream.
    </p>
    """, unsafe_allow_html=True)

    # Region filter
    region_filter = st.radio(
        "Filter:",
        ["🌏 All SEQ (25 cams)", "🟠 Gold Coast (16)", "🟢 Sunshine Coast (9)"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if "Gold Coast" in region_filter:
        visible_cams = [c for c in SEQ_CAMS if c["region"] == "Gold Coast"]
    elif "Sunshine Coast" in region_filter:
        visible_cams = [c for c in SEQ_CAMS if c["region"] == "Sunshine Coast"]
    else:
        visible_cams = SEQ_CAMS

    gallery_html, gallery_height = build_cam_gallery(visible_cams)
    components.html(gallery_html, height=gallery_height, scrolling=False)

# ============================================================================
# TAB 2 – SURF FORECAST
# ============================================================================
with tab_forecast:
    # Sidebar – spot selector (only meaningful in forecast context)
    with st.sidebar:
        st.markdown("### 📍 Select a Spot")
        st.caption("Used for the Forecast tab only.")
        region = st.selectbox("Region", list(SURF_SPOTS.keys()))
        spots = SURF_SPOTS[region]
        spot_names = [s["name"] for s in spots]
        spot_name = st.selectbox("Surf Spot", spot_names)
        spot = next(s for s in spots if s["name"] == spot_name)
        st.divider()
        st.caption("Data: Open-Meteo Marine API · updated every 15 min")

    # ── Fetch data ───────────────────────────────────────────────────────────
    with st.spinner("Fetching latest surf data…"):
        try:
            df = get_forecast(spot["lat"], spot["lon"])
        except Exception as e:
            st.error(f"Could not fetch forecast data: {e}")
            st.stop()

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

    # ── Current conditions ───────────────────────────────────────────────────
    st.subheader(f"📡 {spot_name} — Right Now")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🌊 Wave Height", f"{row['wave_height']:.1f} m")
    col2.metric("⏱ Wave Period", f"{row['wave_period']:.0f} s")
    col3.metric("💨 Wind Speed", f"{row['windspeed_10m']:.0f} km/h")
    col4.metric("🧭 Wind Dir", f"{row['winddirection_10m']:.0f}°")
    col5.metric("🏄 Surf Score", f"{now_score} / 10", delta=score_label(now_score))

    bar_html = f"""
    <div style="background:{score_color(now_score)};border-radius:8px;padding:12px 20px;
                color:white;font-size:1.35rem;font-weight:bold;text-align:center;margin-top:8px;">
        {score_label(now_score)} &nbsp;|&nbsp; Score: {now_score} / 10
        &nbsp;·&nbsp; {spot['break_type']} Break
    </div>"""
    st.markdown(bar_html, unsafe_allow_html=True)
    st.divider()

    # ── Go Surf recommendation ───────────────────────────────────────────────
    st.subheader("🚀 Go Surf Mode")

    top = best_window(df, spot)
    if not top.empty:
        best = top.iloc[0]
        best_time = best["time"]
        best_sc = best["score"]
        leave_time = best_time - pd.Timedelta(minutes=30)
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
            st.dataframe(display, use_container_width=True, hide_index=True)

    st.divider()

    # ── 7-Day forecast charts ────────────────────────────────────────────────
    st.subheader("📈 7-Day Forecast")

    fc1, fc2, fc3, fc4 = st.tabs(["Surf Score", "Wave Height & Period", "Wind", "Swell Direction"])

    with fc1:
        fig = px.area(
            df.dropna(subset=["score"]),
            x="time", y="score",
            color_discrete_sequence=["#0096c7"],
            labels={"time": "", "score": "Surf Score"},
        )
        fig.update_layout(
            yaxis_range=[0, 10], hovermode="x unified",
            paper_bgcolor="#0a1628", plot_bgcolor="#0f1e2e",
            font_color="#b0c8e0",
        )
        fig.add_hline(y=6, line_dash="dash", line_color="#6fcf97", annotation_text="Good")
        fig.add_hline(y=8, line_dash="dash", line_color="#f2c94c", annotation_text="Epic")
        st.plotly_chart(fig, use_container_width=True)

    with fc2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df["time"], y=df["wave_height"], name="Wave Height (m)",
            line=dict(color="#0096c7"), fill="tozeroy", fillcolor="rgba(0,150,199,0.18)",
        ))
        fig2.add_trace(go.Scatter(
            x=df["time"], y=df["wave_period"] / 10, name="Wave Period / 10 (s)",
            line=dict(color="#48cae4", dash="dot"),
        ))
        fig2.update_layout(
            legend=dict(orientation="h"), hovermode="x unified",
            yaxis_title="Wave Height (m) | Period / 10 (s)",
            paper_bgcolor="#0a1628", plot_bgcolor="#0f1e2e", font_color="#b0c8e0",
        )
        st.plotly_chart(fig2, use_container_width=True)

    with fc3:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=df["time"], y=df["windspeed_10m"], name="Wind Speed (km/h)",
            line=dict(color="#e63946"), fill="tozeroy", fillcolor="rgba(230,57,70,0.12)",
        ))
        fig3.add_trace(go.Scatter(
            x=df["time"], y=df["winddirection_10m"], name="Wind Direction (°)",
            line=dict(color="#f4a261", dash="dot"), yaxis="y2",
        ))
        fig3.update_layout(
            yaxis=dict(title="Wind Speed (km/h)"),
            yaxis2=dict(title="Wind Direction (°)", overlaying="y", side="right", range=[0, 360]),
            legend=dict(orientation="h"), hovermode="x unified",
            paper_bgcolor="#0a1628", plot_bgcolor="#0f1e2e", font_color="#b0c8e0",
        )
        st.plotly_chart(fig3, use_container_width=True)

    with fc4:
        fig4 = px.scatter_polar(
            df.dropna(subset=["wave_direction"]).head(168),
            r="wave_height", theta="wave_direction", color="score",
            color_continuous_scale="Blues",
            labels={"wave_direction": "Swell Direction (°)", "wave_height": "Height (m)", "score": "Score"},
        )
        fig4.update_layout(
            polar=dict(angularaxis=dict(direction="clockwise", rotation=90)),
            paper_bgcolor="#0a1628", font_color="#b0c8e0",
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()

    # ── Region rankings ──────────────────────────────────────────────────────
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
            use_container_width=True,
        )

st.caption("🌊 Surf Buddy — 100% free for all surfers. Data via Open-Meteo.")
