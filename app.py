import math
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
from streamlit_js_eval import get_geolocation

# ---------------------------------------------------------------------------
# GCCC Live Camera Catalogue
# Direct HLS streams from Gold Coast City Council CloudFront distribution.
# URL pattern:  HLS_BASE/{stream}.stream/playlist.m3u8
# ---------------------------------------------------------------------------

HLS_BASE = "https://d1nm4r8e5x1rwd.cloudfront.net/cw"
GCCC_PAGE = "https://www.goldcoast.qld.gov.au/Things-to-do/Gold-Coast-beaches/Beach-cameras"

# South → North ordering so the gallery reads geographically
SEQ_CAMS = [
    # ── Gold Coast – confirmed GCCC direct HLS streams ──────────────────────
    {"name": "Rainbow Bay",         "region": "Gold Coast", "stream": "rainbowbaycamera",          "lat": -28.168, "lon": 153.546},
    {"name": "Greenmount",          "region": "Gold Coast", "stream": "rainbowbaycamera",          "lat": -28.165, "lon": 153.544},
    {"name": "Kirra",               "region": "Gold Coast", "stream": "kirracamera",               "lat": -28.162, "lon": 153.544},
    {"name": "Tugun / Bilinga",     "region": "Gold Coast", "stream": "tuguncamera",               "lat": -28.145, "lon": 153.527},
    {"name": "Palm Beach",          "region": "Gold Coast", "stream": "currumbincamera",           "lat": -28.118, "lon": 153.472},
    {"name": "Burleigh Heads",      "region": "Gold Coast", "stream": "burleighhighcamera",        "lat": -28.085, "lon": 153.453},
    {"name": "Miami Beach",         "region": "Gold Coast", "stream": "miamicamera",               "lat": -28.056, "lon": 153.440},
    {"name": "Northcliffe",         "region": "Gold Coast", "stream": "surfersparadisehighcamera", "lat": -27.998, "lon": 153.428},
    {"name": "Narrowneck",          "region": "Gold Coast", "stream": "narrowneckhighcamera",      "lat": -27.984, "lon": 153.431},
    {"name": "Main Beach",          "region": "Gold Coast", "stream": "mainbeachcamera",           "lat": -27.971, "lon": 153.431},
    # ── Gold Coast – GCCC page link (stream slug not yet confirmed) ──────────
    {"name": "Snapper Rocks",       "region": "Gold Coast", "lat": -28.166, "lon": 153.546},
    {"name": "Coolangatta Beach",   "region": "Gold Coast", "lat": -28.163, "lon": 153.545},
    {"name": "Currumbin Alley",     "region": "Gold Coast", "lat": -28.158, "lon": 153.498},
    {"name": "Mermaid Beach",       "region": "Gold Coast", "lat": -28.038, "lon": 153.432},
    {"name": "Broadbeach",          "region": "Gold Coast", "lat": -28.023, "lon": 153.429},
    {"name": "Surfers Paradise",    "region": "Gold Coast", "lat": -27.998, "lon": 153.428},
    # ── Sunshine Coast ───────────────────────────────────────────────────────
    {"name": "Caloundra",           "region": "Sunshine Coast", "cam_url": "https://surfsunshinecoast.com.au/surf-cams/",                                 "lat": -26.800, "lon": 153.142},
    {"name": "Mooloolaba",          "region": "Sunshine Coast", "cam_url": "https://surfsunshinecoast.com.au/surf-cams/",                                 "lat": -26.681, "lon": 153.119},
    {"name": "Alexandra Headland",  "region": "Sunshine Coast", "cam_url": "https://surfsunshinecoast.com.au/surf-cams/",                                 "lat": -26.668, "lon": 153.113},
    {"name": "Coolum Beach",        "region": "Sunshine Coast", "cam_url": "https://surfsunshinecoast.com.au/surf-cams/",                                 "lat": -26.532, "lon": 153.088},
    {"name": "Sunshine Beach",      "region": "Sunshine Coast", "cam_url": "https://noosasurfcam.com/sunshine-beach-surf-cam.html",                      "lat": -26.410, "lon": 153.093},
    {"name": "Noosa Heads",         "region": "Sunshine Coast", "cam_url": "https://noosasurfcam.com/",                                                  "lat": -26.392, "lon": 153.091},
]

# ---------------------------------------------------------------------------
# Surf spot catalogue – used for forecasts & rankings
# ---------------------------------------------------------------------------

SURF_SPOTS = {
    "Gold Coast": [
        {"name": "Snapper Rocks",    "lat": -28.166, "lon": 153.546, "orientation": 75,  "break_type": "Point"},
        {"name": "Rainbow Bay",      "lat": -28.168, "lon": 153.546, "orientation": 78,  "break_type": "Point"},
        {"name": "Greenmount",       "lat": -28.165, "lon": 153.544, "orientation": 80,  "break_type": "Point"},
        {"name": "Kirra",            "lat": -28.162, "lon": 153.544, "orientation": 82,  "break_type": "Beach"},
        {"name": "Currumbin Alley",  "lat": -28.158, "lon": 153.498, "orientation": 70,  "break_type": "Alley"},
        {"name": "Palm Beach",       "lat": -28.118, "lon": 153.472, "orientation": 78,  "break_type": "Beach"},
        {"name": "Burleigh Heads",   "lat": -28.085, "lon": 153.453, "orientation": 72,  "break_type": "Point"},
        {"name": "Miami Beach",      "lat": -28.056, "lon": 153.440, "orientation": 75,  "break_type": "Beach"},
        {"name": "Mermaid Beach",    "lat": -28.038, "lon": 153.432, "orientation": 75,  "break_type": "Beach"},
        {"name": "Surfers Paradise", "lat": -27.998, "lon": 153.428, "orientation": 75,  "break_type": "Beach"},
        {"name": "Narrowneck",       "lat": -27.984, "lon": 153.431, "orientation": 78,  "break_type": "Beach"},
        {"name": "Main Beach",       "lat": -27.971, "lon": 153.431, "orientation": 78,  "break_type": "Beach"},
    ],
    "Sunshine Coast": [
        {"name": "Noosa Heads",        "lat": -26.392, "lon": 153.091, "orientation": 60, "break_type": "Point"},
        {"name": "Sunshine Beach",     "lat": -26.410, "lon": 153.093, "orientation": 75, "break_type": "Beach"},
        {"name": "Coolum Beach",       "lat": -26.532, "lon": 153.088, "orientation": 80, "break_type": "Beach"},
        {"name": "Alexandra Headland", "lat": -26.668, "lon": 153.113, "orientation": 68, "break_type": "Point"},
        {"name": "Mooloolaba Beach",   "lat": -26.681, "lon": 153.119, "orientation": 75, "break_type": "Beach"},
        {"name": "Caloundra",          "lat": -26.800, "lon": 153.142, "orientation": 90, "break_type": "Beach"},
    ],
    "Northern NSW": [
        {"name": "Tweed / D'bah",        "lat": -28.178, "lon": 153.554, "orientation": 88, "break_type": "Beach"},
        {"name": "Byron Bay – The Pass", "lat": -28.641, "lon": 153.628, "orientation": 45, "break_type": "Point"},
        {"name": "Lennox Head",          "lat": -28.797, "lon": 153.588, "orientation": 60, "break_type": "Point"},
        {"name": "Ballina",              "lat": -28.871, "lon": 153.563, "orientation": 90, "break_type": "Beach"},
    ],
}

# All spots across all regions flattened (for Near Me search)
ALL_SPOTS = [
    {**s, "region": region}
    for region, spots in SURF_SPOTS.items()
    for s in spots
]

# ---------------------------------------------------------------------------
# Data fetching – Open-Meteo (free, no API key)
# ---------------------------------------------------------------------------

MARINE_URL  = "https://marine-api.open-meteo.com/v1/marine"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"


@st.cache_data(ttl=900)
def fetch_marine(lat: float, lon: float) -> pd.DataFrame:
    params = {
        "latitude":  lat,
        "longitude": lon,
        "hourly": ",".join([
            "wave_height", "wave_period", "wave_direction",
            "swell_wave_height", "swell_wave_period", "swell_wave_direction",
            "wind_wave_height", "wind_wave_period",
            "sea_surface_temperature",
        ]),
        "forecast_days": 7,
        "timezone": "Australia/Sydney",
    }
    r = requests.get(MARINE_URL, params=params, timeout=10)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df


@st.cache_data(ttl=900)
def fetch_weather(lat: float, lon: float) -> pd.DataFrame:
    params = {
        "latitude":  lat,
        "longitude": lon,
        "hourly": "windspeed_10m,winddirection_10m,uv_index",
        "forecast_days": 7,
        "timezone": "Australia/Sydney",
    }
    r = requests.get(WEATHER_URL, params=params, timeout=10)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df


def get_forecast(lat: float, lon: float) -> pd.DataFrame:
    return fetch_marine(lat, lon).merge(fetch_weather(lat, lon), on="time")


@st.cache_data(ttl=3600)
def get_ip_location() -> tuple:
    """Rough location from IP — used as fallback when browser geolocation unavailable."""
    try:
        r = requests.get("https://ipapi.co/json/", timeout=5)
        d = r.json()
        return float(d["latitude"]), float(d["longitude"]), d.get("city", "Unknown")
    except Exception:
        return -28.0, 153.4, "Gold Coast (estimated)"  # sensible default


# ---------------------------------------------------------------------------
# Improved surf scoring engine
# ---------------------------------------------------------------------------

def _wave_height_score(h: float) -> float:
    """0–10. Sweet spot 0.8–2.5 m."""
    if h < 0.3:  return 0.0
    if h < 0.8:  return (h - 0.3) / 0.5 * 5
    if h <= 2.5: return 10.0 - (h - 0.8) / 1.7 * 2
    if h <= 4.0: return 8.0 - (h - 2.5) / 1.5 * 5
    return max(0.0, 3.0 - (h - 4.0))


def _period_score(p: float) -> float:
    """0–10. Sweet spot 10–16 s."""
    if p < 5:   return 0.0
    if p < 10:  return (p - 5) / 5 * 7
    if p <= 16: return 10.0
    return max(0.0, 10.0 - (p - 16) * 0.5)


def _wind_score(speed: float, wind_dir: float, orientation: float) -> float:
    """0–10. Offshore best, light best."""
    offshore = (orientation + 180) % 360
    diff = abs(wind_dir - offshore)
    if diff > 180: diff = 360 - diff
    dir_factor   = max(0.0, 1.0 - diff / 180)
    speed_factor = (1.0 if speed < 10 else
                    1.0 - (speed - 10) / 20 if speed < 20 else
                    max(0.0, 0.5 - (speed - 20) / 40))
    return round(10 * dir_factor * speed_factor, 1)


def _swell_dir_score(wave_dir: float, orientation: float) -> float:
    """0–10. Direct hit best."""
    diff = abs(wave_dir - orientation)
    if diff > 180: diff = 360 - diff
    return max(0.0, 10.0 - diff / 18)


# Break-type weight presets  (height, period, wind, direction)
_BREAK_WEIGHTS = {
    "Point": (0.30, 0.30, 0.25, 0.15),  # long period matters most
    "Alley": (0.35, 0.25, 0.20, 0.20),  # direction more important
    "Beach": (0.35, 0.25, 0.25, 0.15),  # balanced
}


def surf_score(
    wave_height, wave_period, wave_direction,
    wind_speed, wind_direction, orientation,
    break_type: str = "Beach",
    wind_wave_height: float = 0.0,
) -> float:
    """
    Composite surf score 0–10.
    Uses swell-specific height/period for accuracy, then applies a chop
    penalty proportional to wind-wave contamination.
    """
    h = _wave_height_score(float(wave_height))
    p = _period_score(float(wave_period))
    w = _wind_score(float(wind_speed), float(wind_direction), orientation)
    s = _swell_dir_score(float(wave_direction), orientation)

    ww, sw = float(wind_wave_height), float(wave_height)
    chop_penalty = min(0.20, (ww / max(sw, 0.1)) * 0.20)  # up to –20 %

    bw = _BREAK_WEIGHTS.get(break_type, _BREAK_WEIGHTS["Beach"])
    raw = h * bw[0] + p * bw[1] + w * bw[2] + s * bw[3]
    return round(raw * (1 - chop_penalty), 1)


def score_label(score: float) -> str:
    if score >= 8: return "🔥 Epic"
    if score >= 6: return "✅ Good"
    if score >= 4: return "🟡 Fair"
    return "❌ Poor"


def score_color(score: float) -> str:
    if score >= 8: return "#00cc44"
    if score >= 6: return "#66cc00"
    if score >= 4: return "#ffaa00"
    return "#ff4444"


def uv_label(uv: float) -> str:
    if uv < 3:  return "Low"
    if uv < 6:  return "Moderate"
    if uv < 8:  return "High"
    if uv < 11: return "Very High"
    return "Extreme ☠️"


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

_COMPASS = ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]


def compass(deg: float) -> str:
    return _COMPASS[round(float(deg) / 22.5) % 16]


def wind_relation(wind_dir: float, orientation: float) -> str:
    offshore = (orientation + 180) % 360
    diff = abs(wind_dir - offshore)
    if diff > 180: diff = 360 - diff
    if diff < 45:  return "🟢 Offshore"
    if diff < 90:  return "🟡 Cross-shore"
    return "🔴 Onshore"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def current_row(df: pd.DataFrame) -> pd.Series:
    now = pd.Timestamp.now(tz="Australia/Sydney").tz_convert(df["time"].dt.tz)
    return df.loc[(df["time"] - now).abs().idxmin()]


def best_window(df: pd.DataFrame, spot: dict) -> pd.DataFrame:
    now    = pd.Timestamp.now(tz="Australia/Sydney").tz_convert(df["time"].dt.tz)
    future = df[df["time"] >= now].head(48).copy()
    future["score"] = future.apply(
        lambda r: surf_score(
            r.get("swell_wave_height", r["wave_height"]),
            r.get("swell_wave_period", r["wave_period"]),
            r.get("swell_wave_direction", r["wave_direction"]),
            r["windspeed_10m"], r["winddirection_10m"], spot["orientation"],
            spot["break_type"],
            r.get("wind_wave_height", 0.0),
        ), axis=1,
    )
    return future.sort_values("score", ascending=False).head(5)


def _safe(row, col, fallback=None):
    """Safely read a column from a pandas Series, returning fallback if missing/NaN."""
    if col in row.index and pd.notna(row[col]):
        return row[col]
    return fallback


# ---------------------------------------------------------------------------
# Cam gallery helpers
# ---------------------------------------------------------------------------

_GALLERY_COLS   = 3
_CARD_HEIGHT_PX = 275
_GALLERY_PAD_PX = 40


def _cam_card_html(cam: dict, idx: int) -> str:
    name         = cam["name"]
    region       = cam["region"]
    badge_color  = "#c1440e" if region == "Gold Coast" else "#1a7a4a"

    if cam.get("stream"):
        stream_url = f"{HLS_BASE}/{cam['stream']}.stream/playlist.m3u8"
        media_html = f"""
        <div style="position:relative;">
          <span style="position:absolute;top:8px;left:8px;z-index:10;
                       background:#e63946;color:white;font-size:0.62rem;
                       font-weight:700;padding:2px 8px;border-radius:4px;
                       letter-spacing:0.08em;pointer-events:none;">&#9679; LIVE</span>
          <video id="cam{idx}" data-src="{stream_url}"
                 muted playsinline controls
                 style="width:100%;height:190px;background:#000;display:block;object-fit:cover;">
          </video>
        </div>"""
        action = ('<span style="background:#00cc44;color:#001a0a;padding:6px 12px;'
                  'border-radius:6px;font-size:0.72rem;font-weight:700;">✅ Direct Stream</span>')
    else:
        link_url  = cam.get("cam_url", GCCC_PAGE)
        btn_label = "🎥 Watch Free" if cam.get("cam_url") else "🎥 GCCC Live Cam"
        gradient  = (
            "linear-gradient(150deg,#023e8a 0%,#0096c7 55%,#48cae4 100%)"
            if region == "Gold Coast"
            else "linear-gradient(150deg,#1b4332 0%,#2d9596 55%,#48cae4 100%)"
        )
        media_html = f"""
        <div style="height:190px;background:{gradient};display:flex;flex-direction:column;
                    align-items:center;justify-content:center;color:white;">
          <div style="font-size:2.6rem;line-height:1">🌊</div>
          <div style="font-size:0.78rem;margin-top:10px;opacity:0.85;
                      text-align:center;padding:0 14px;">
            Click <strong>{btn_label.split()[-1]}</strong> below to view live
          </div>
        </div>"""
        action = (f'<a href="{link_url}" target="_blank" rel="noopener"'
                  f' style="background:#0077b6;color:white;padding:7px 14px;'
                  f'border-radius:7px;font-size:0.78rem;text-decoration:none;'
                  f'font-weight:700;white-space:nowrap;">{btn_label}</a>')

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
        <div style="flex-shrink:0;">{action}</div>
      </div>
    </div>"""


def build_cam_gallery(cams: list) -> tuple:
    """Return (html_string, height_px) for the responsive cam card grid."""
    n_rows = math.ceil(len(cams) / _GALLERY_COLS)
    height = n_rows * _CARD_HEIGHT_PX + _GALLERY_PAD_PX
    cards  = "\n".join(_cam_card_html(c, i) for i, c in enumerate(cams))

    hls_inits = "\n".join(
        "  initHLS('cam{i}', '{base}/{stream}.stream/playlist.m3u8');".format(
            i=i, base=HLS_BASE, stream=c["stream"]
        )
        for i, c in enumerate(cams) if c.get("stream")
    )

    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:transparent;padding:4px 2px 8px}}
  .grid{{display:grid;grid-template-columns:repeat({_GALLERY_COLS},1fr);gap:16px}}
</style>
</head><body><div class="grid">{cards}</div>
<script>
function initHLS(id, url) {{
  var v = document.getElementById(id);
  if (!v) return;
  if (typeof Hls !== 'undefined' && Hls.isSupported()) {{
    var hls = new Hls({{enableWorker:true, lowLatencyMode:true}});
    hls.loadSource(url);
    hls.attachMedia(v);
    hls.on(Hls.Events.ERROR, function(e, d) {{
      if (d.fatal) {{
        v.parentElement.innerHTML = '<div style="height:190px;background:#1a0010;'
          + 'display:flex;align-items:center;justify-content:center;color:#ff8080;'
          + 'font-size:0.8rem;text-align:center;padding:1rem;">'
          + '⚠️ Stream temporarily<br>unavailable</div>';
      }}
    }});
  }} else if (v.canPlayType('application/vnd.apple.mpegurl')) {{
    v.src = url;
  }}
}}
{hls_inits}
</script>
</body></html>"""
    return html, height


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Surf Buddy 🏄", page_icon="🏄", layout="wide")

st.markdown("""
<style>
  .stApp { background-color: #0a1628; color: #e8f4fd; }
  [data-testid="stSidebar"] { background-color: #0f1e2e; border-right: 1px solid #1e3a5c; }
  [data-testid="stMetric"] {
    background: #0f1e2e; border: 1px solid #1e3a5c;
    border-radius: 10px; padding: 14px 18px;
  }
  [data-testid="stMetricValue"] { color: #e8f4fd !important; }
  [data-testid="stMetricLabel"] { color: #90caf9 !important; }
  h1 { font-size: 2rem !important; font-weight: 700 !important; color: #e8f4fd !important; }
  h2, h3 { color: #e8f4fd !important; font-weight: 600 !important; }
  p, li, .stMarkdown { color: #b0c8e0 !important; }
  [data-testid="stTabs"] button { font-weight: 600; color: #90caf9 !important; }
  [data-testid="stTabs"] button[aria-selected="true"] {
    color: #48cae4 !important; border-bottom-color: #48cae4 !important;
  }
  .stCaption { color: #5a8aa8 !important; }
  hr { border-color: #1e3a5c !important; }
  [data-testid="stRadio"] label { color: #b0c8e0 !important; }
  .stDataFrame { background: #0f1e2e; }
  [data-testid="stSuccess"] { background: #0d2b1a !important; border-color: #1a7a4a !important; }
  [data-testid="stSuccess"] p { color: #6fcf97 !important; }
  [data-testid="stExpander"] { background: #0f1e2e !important; border-color: #1e3a5c !important; }
  [data-testid="stInfo"] { background: #0a1e38 !important; border-color: #1e3a5c !important; }
  [data-testid="stInfo"] p { color: #90caf9 !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:12px 0 4px">
  <span style="font-size:2.2rem;font-weight:800;color:#e8f4fd;">🏄 Surf Buddy</span>
  <span style="font-size:0.95rem;color:#5a8aa8;margin-left:12px;">
    SEQ Live Cams · Forecast · Near-Me Rankings
  </span>
</div>
""", unsafe_allow_html=True)

# ── Geolocation (browser first, IP fallback) ─────────────────────────────────
with st.sidebar:
    st.markdown("### 📍 Your Location")
    loc_data = get_geolocation()
    if loc_data and loc_data.get("coords"):
        user_lat = loc_data["coords"]["latitude"]
        user_lon = loc_data["coords"]["longitude"]
        st.success(f"GPS: {user_lat:.3f}°, {user_lon:.3f}°")
        loc_source = "GPS"
    else:
        user_lat, user_lon, loc_city = get_ip_location()
        st.info(f"Using approximate location: {loc_city}")
        loc_source = "IP"

    st.divider()
    st.markdown("### 📍 Forecast Spot")
    st.caption("Detailed forecast for a specific spot.")
    region_sel = st.selectbox("Region", list(SURF_SPOTS.keys()))
    spots      = SURF_SPOTS[region_sel]
    spot_name  = st.selectbox("Surf Spot", [s["name"] for s in spots])
    spot       = next(s for s in spots if s["name"] == spot_name)
    st.divider()
    st.markdown("#### 🌊 Tide Times")
    tide_urls = {
        "Gold Coast":    "https://www.bom.gov.au/australia/tides/#!/qld/southport",
        "Sunshine Coast":"https://www.bom.gov.au/australia/tides/#!/qld/mooloolaba",
        "Northern NSW":  "https://www.bom.gov.au/australia/tides/#!/nsw/ballina",
    }
    st.markdown(f"[📅 BOM Tide Predictions ↗]({tide_urls.get(region_sel, 'https://www.bom.gov.au/australia/tides/')})")
    st.divider()
    st.caption("Data: Open-Meteo · updated every 15 min")

# ── Top-level tabs ───────────────────────────────────────────────────────────
tab_cams, tab_nearme, tab_forecast = st.tabs([
    "🎥  Live Cams",
    "📍  Near Me",
    "📈  Surf Forecast",
])

# ============================================================================
# TAB 1 – LIVE CAMS GALLERY
# ============================================================================
with tab_cams:
    gc_direct = sum(1 for c in SEQ_CAMS if c["region"] == "Gold Coast" and c.get("stream"))
    gc_total  = sum(1 for c in SEQ_CAMS if c["region"] == "Gold Coast")
    sc_total  = sum(1 for c in SEQ_CAMS if c["region"] == "Sunshine Coast")

    st.markdown(
        f'<p style="color:#5a8aa8;font-size:0.85rem;margin:-4px 0 12px">'
        f'<strong style="color:#00cc44">&#9679; {gc_direct} direct live streams</strong> embedded '
        f'from <strong style="color:#90caf9">Gold Coast City Council (GCCC)</strong> cameras — '
        f'no login, no subscription. Other spots link to their free cam pages.</p>',
        unsafe_allow_html=True,
    )

    region_filter = st.radio(
        "Filter:",
        [f"🌏 All SEQ ({len(SEQ_CAMS)})",
         f"🟠 Gold Coast ({gc_total})",
         f"🟢 Sunshine Coast ({sc_total})"],
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
# TAB 2 – NEAR ME
# ============================================================================
with tab_nearme:
    st.subheader(f"📍 Best Surf Near You  ·  {loc_source} location")
    st.caption(
        f"Ranked by surf score right now, within 200 km of your position "
        f"({user_lat:.3f}°, {user_lon:.3f}°)."
    )

    near_rows = []
    progress  = st.progress(0, text="Checking all spots…")
    for i, s in enumerate(ALL_SPOTS):
        progress.progress((i + 1) / len(ALL_SPOTS), text=f"Checking {s['name']}…")
        try:
            sdf  = get_forecast(s["lat"], s["lon"])
            srow = current_row(sdf)
            sc   = surf_score(
                _safe(srow, "swell_wave_height", srow["wave_height"]),
                _safe(srow, "swell_wave_period", srow["wave_period"]),
                _safe(srow, "swell_wave_direction", srow["wave_direction"]),
                srow["windspeed_10m"], srow["winddirection_10m"],
                s["orientation"], s["break_type"],
                _safe(srow, "wind_wave_height", 0.0),
            )
            dist_km = haversine_km(user_lat, user_lon, s["lat"], s["lon"])
            near_rows.append({
                "Spot":      s["name"],
                "Region":    s["region"],
                "Break":     s["break_type"],
                "Dist (km)": round(dist_km, 1),
                "Wave (m)":  round(srow["wave_height"], 1),
                "Swell (m)": round(_safe(srow, "swell_wave_height", srow["wave_height"]), 1),
                "Period (s)":round(_safe(srow, "swell_wave_period", srow["wave_period"]), 0),
                "Wind":      f"{srow['windspeed_10m']:.0f} km/h {compass(srow['winddirection_10m'])}",
                "Wind qlty": wind_relation(srow["winddirection_10m"], s["orientation"]),
                "Score":     sc,
                "Rating":    score_label(sc),
                "_lat":      s["lat"],
                "_lon":      s["lon"],
            })
        except Exception:
            pass
    progress.empty()

    if near_rows:
        near_df = (
            pd.DataFrame(near_rows)
            .query("`Dist (km)` <= 200")
            .sort_values("Score", ascending=False)
            .reset_index(drop=True)
        )
        near_df.index += 1

        # ── Top pick hero card ───────────────────────────────────────────────
        if not near_df.empty:
            top = near_df.iloc[0]
            hero_color = score_color(top["Score"])
            st.markdown(f"""
<div style="background:{hero_color}22;border:2px solid {hero_color};border-radius:14px;
            padding:18px 24px;margin-bottom:16px;">
  <div style="font-size:1.6rem;font-weight:800;color:#e8f4fd;">
    🏆 Go to <span style="color:{hero_color}">{top['Spot']}</span>
  </div>
  <div style="color:#b0c8e0;margin-top:6px;font-size:0.95rem;">
    {top['Rating']} &nbsp;·&nbsp; Score <strong style="color:{hero_color}">{top['Score']}/10</strong>
    &nbsp;·&nbsp; {top['Break']} break
    &nbsp;·&nbsp; {top['Dist (km)']} km away
    &nbsp;·&nbsp; {top['Swell (m)']}m swell @ {top['Period (s)']:.0f}s
    &nbsp;·&nbsp; Wind {top['Wind']} {top['Wind qlty']}
  </div>
</div>""", unsafe_allow_html=True)

        # ── Full ranked table ────────────────────────────────────────────────
        display_df = near_df.drop(columns=["_lat", "_lon"])

        def _color_score(val):
            return f"color: {score_color(val)}; font-weight: bold"

        st.dataframe(
            display_df.style.map(_color_score, subset=["Score"]),
            use_container_width=True,
        )
    else:
        st.warning("Could not load any spot data. Please check your connection.")

# ============================================================================
# TAB 3 – SURF FORECAST
# ============================================================================
with tab_forecast:
    with st.spinner("Fetching latest surf data…"):
        try:
            df = get_forecast(spot["lat"], spot["lon"])
        except Exception as e:
            st.error(f"Could not fetch forecast data: {e}")
            st.stop()

    required_cols = ["wave_height", "wave_period", "wave_direction",
                     "windspeed_10m", "winddirection_10m"]
    df["score"] = df.apply(
        lambda r: surf_score(
            _safe(r, "swell_wave_height", r["wave_height"]),
            _safe(r, "swell_wave_period",  r["wave_period"]),
            _safe(r, "swell_wave_direction", r["wave_direction"]),
            r["windspeed_10m"], r["winddirection_10m"],
            spot["orientation"], spot["break_type"],
            _safe(r, "wind_wave_height", 0.0),
        ) if all(pd.notna(r[c]) for c in required_cols) else np.nan,
        axis=1,
    )

    row       = current_row(df)
    now_score = surf_score(
        _safe(row, "swell_wave_height", row["wave_height"]),
        _safe(row, "swell_wave_period",  row["wave_period"]),
        _safe(row, "swell_wave_direction", row["wave_direction"]),
        row["windspeed_10m"], row["winddirection_10m"],
        spot["orientation"], spot["break_type"],
        _safe(row, "wind_wave_height", 0.0),
    )

    # ── Current conditions ───────────────────────────────────────────────────
    st.subheader(f"📡 {spot_name} — Right Now")

    swell_h = _safe(row, "swell_wave_height", row["wave_height"])
    swell_p = _safe(row, "swell_wave_period",  row["wave_period"])
    swell_d = _safe(row, "swell_wave_direction", row["wave_direction"])
    water_t = _safe(row, "sea_surface_temperature")
    uv_val  = _safe(row, "uv_index")
    ww_h    = _safe(row, "wind_wave_height", 0.0)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("🌊 Wave Height",  f"{row['wave_height']:.1f} m")
    c2.metric("🌀 Swell",        f"{swell_h:.1f} m @ {swell_p:.0f} s")
    c3.metric("💨 Wind",
              f"{row['windspeed_10m']:.0f} km/h {compass(row['winddirection_10m'])}",
              delta=wind_relation(row["winddirection_10m"], spot["orientation"]),
              delta_color="off")
    c4.metric("🧭 Swell Dir",    f"{compass(swell_d)} ({swell_d:.0f}°)")
    c5.metric("🌡 Water Temp",   f"{water_t:.1f} °C" if water_t is not None else "—")
    c6.metric("☀️ UV Index",
              f"{uv_val:.0f} — {uv_label(uv_val)}" if uv_val is not None else "—")

    bar_color = score_color(now_score)
    st.markdown(f"""
<div style="background:{bar_color};border-radius:8px;padding:12px 20px;
            color:white;font-size:1.35rem;font-weight:bold;text-align:center;margin-top:8px;">
  {score_label(now_score)} &nbsp;|&nbsp; Score: {now_score} / 10
  &nbsp;·&nbsp; {spot['break_type']} Break
</div>
<div style="background:#0f1e2e;border:1px solid #1e3a5c;border-radius:8px;
            padding:10px 20px;margin-top:6px;display:flex;gap:24px;flex-wrap:wrap;
            font-size:0.85rem;color:#b0c8e0;">
  <span>Wind quality: <strong style="color:#e8f4fd">
    {wind_relation(row['winddirection_10m'], spot['orientation'])}</strong></span>
  <span>Chop: <strong style="color:#e8f4fd">{ww_h:.1f} m wind swell</strong></span>
  <span>Swell direction: <strong style="color:#e8f4fd">
    {compass(swell_d)} — {_swell_dir_score(swell_d, spot['orientation']):.0f}/10 for this break</strong></span>
</div>
""", unsafe_allow_html=True)
    st.divider()

    # ── Go Surf recommendation ───────────────────────────────────────────────
    st.subheader("🚀 Go Surf Mode")
    top5 = best_window(df, spot)
    if not top5.empty:
        best = top5.iloc[0]
        leave = best["time"] - pd.Timedelta(minutes=30)
        st.success(
            f"**Best window:** {best['time'].strftime('%A %d %b, %H:%M')}  |  "
            f"Score **{best['score']}/10** {score_label(best['score'])}  |  "
            f"Set your alarm for **{leave.strftime('%H:%M')}** 🤙"
        )
        with st.expander("Top 5 windows in next 48 h"):
            disp = top5[["time", "wave_height", "wave_period", "windspeed_10m", "score"]].copy()
            disp.columns = ["Time", "Wave (m)", "Period (s)", "Wind (km/h)", "Score"]
            disp["Time"]  = disp["Time"].dt.strftime("%a %d %b %H:%M")
            disp["Score"] = disp["Score"].apply(lambda x: f"{x} {score_label(x)}")
            st.dataframe(disp, use_container_width=True, hide_index=True)
    st.divider()

    # ── 7-Day forecast charts ────────────────────────────────────────────────
    st.subheader("📈 7-Day Forecast")
    fc1, fc2, fc3, fc4, fc5 = st.tabs([
        "Surf Score", "Wave Height & Period", "Wind", "Swell Direction", "Water & UV",
    ])

    chart_layout = dict(
        paper_bgcolor="#0a1628", plot_bgcolor="#0f1e2e",
        font_color="#b0c8e0", hovermode="x unified",
    )

    with fc1:
        fig = px.area(df.dropna(subset=["score"]), x="time", y="score",
                      color_discrete_sequence=["#0096c7"],
                      labels={"time": "", "score": "Surf Score"})
        fig.update_layout(yaxis_range=[0, 10], **chart_layout)
        fig.add_hline(y=6, line_dash="dash", line_color="#6fcf97", annotation_text="Good")
        fig.add_hline(y=8, line_dash="dash", line_color="#f2c94c", annotation_text="Epic")
        st.plotly_chart(fig, use_container_width=True)

    with fc2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df["time"], y=df["wave_height"], name="Total Wave (m)",
            line=dict(color="#0096c7"), fill="tozeroy", fillcolor="rgba(0,150,199,0.18)"))
        if "swell_wave_height" in df.columns:
            fig2.add_trace(go.Scatter(
                x=df["time"], y=df["swell_wave_height"], name="Swell Height (m)",
                line=dict(color="#48cae4", dash="dash")))
        fig2.add_trace(go.Scatter(
            x=df["time"], y=df["wave_period"] / 10, name="Period / 10 (s)",
            line=dict(color="#90e0ef", dash="dot")))
        fig2.update_layout(legend=dict(orientation="h"),
                           yaxis_title="Height (m) | Period / 10 (s)", **chart_layout)
        st.plotly_chart(fig2, use_container_width=True)

    with fc3:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=df["time"], y=df["windspeed_10m"], name="Wind Speed (km/h)",
            line=dict(color="#e63946"), fill="tozeroy", fillcolor="rgba(230,57,70,0.12)"))
        fig3.add_trace(go.Scatter(
            x=df["time"], y=df["winddirection_10m"], name="Wind Dir (°)",
            line=dict(color="#f4a261", dash="dot"), yaxis="y2"))
        fig3.update_layout(
            yaxis=dict(title="Wind Speed (km/h)"),
            yaxis2=dict(title="Wind Direction (°)", overlaying="y", side="right", range=[0, 360]),
            legend=dict(orientation="h"), **chart_layout)
        st.plotly_chart(fig3, use_container_width=True)

    with fc4:
        r_col = "swell_wave_height" if "swell_wave_height" in df.columns else "wave_height"
        fig4 = px.scatter_polar(
            df.dropna(subset=["wave_direction"]).head(168),
            r=r_col, theta="wave_direction", color="score",
            color_continuous_scale="Blues",
            labels={"wave_direction": "Swell Dir (°)", r_col: "Swell (m)", "score": "Score"})
        fig4.update_layout(
            polar=dict(angularaxis=dict(direction="clockwise", rotation=90)),
            paper_bgcolor="#0a1628", font_color="#b0c8e0")
        st.plotly_chart(fig4, use_container_width=True)

    with fc5:
        fig5 = go.Figure()
        if "sea_surface_temperature" in df.columns:
            fig5.add_trace(go.Scatter(
                x=df["time"], y=df["sea_surface_temperature"], name="Water Temp (°C)",
                line=dict(color="#48cae4"), fill="tozeroy",
                fillcolor="rgba(72,202,228,0.15)"))
        if "uv_index" in df.columns:
            fig5.add_trace(go.Scatter(
                x=df["time"], y=df["uv_index"], name="UV Index",
                line=dict(color="#f4a261"), yaxis="y2"))
        fig5.update_layout(
            yaxis=dict(title="Water Temp (°C)"),
            yaxis2=dict(title="UV Index", overlaying="y", side="right", range=[0, 14]),
            legend=dict(orientation="h"), **chart_layout)
        st.plotly_chart(fig5, use_container_width=True)

    st.divider()

    # ── Region rankings ──────────────────────────────────────────────────────
    st.subheader(f"🗺️ {region_sel} Spot Rankings — Right Now")

    rank_rows = []
    for s in SURF_SPOTS[region_sel]:
        try:
            sdf  = get_forecast(s["lat"], s["lon"])
            srow = current_row(sdf)
            sc   = surf_score(
                _safe(srow, "swell_wave_height", srow["wave_height"]),
                _safe(srow, "swell_wave_period",  srow["wave_period"]),
                _safe(srow, "swell_wave_direction", srow["wave_direction"]),
                srow["windspeed_10m"], srow["winddirection_10m"],
                s["orientation"], s["break_type"],
                _safe(srow, "wind_wave_height", 0.0),
            )
            rank_rows.append({
                "Spot":       s["name"],
                "Break":      s["break_type"],
                "Wave (m)":   round(srow["wave_height"], 1),
                "Swell (m)":  round(_safe(srow, "swell_wave_height", srow["wave_height"]), 1),
                "Period (s)": round(_safe(srow, "swell_wave_period", srow["wave_period"]), 0),
                "Wind":       f"{srow['windspeed_10m']:.0f} {compass(srow['winddirection_10m'])}",
                "Wind qlty":  wind_relation(srow["winddirection_10m"], s["orientation"]),
                "Score":      sc,
                "Rating":     score_label(sc),
            })
        except Exception:
            pass

    if rank_rows:
        rank_df = (pd.DataFrame(rank_rows)
                   .sort_values("Score", ascending=False)
                   .reset_index(drop=True))
        rank_df.index += 1

        def _color_score(val):
            return f"color: {score_color(val)}; font-weight: bold"

        st.dataframe(
            rank_df.style.map(_color_score, subset=["Score"]),
            use_container_width=True,
        )

st.caption("🌊 Surf Buddy — 100% free. Data: Open-Meteo · GCCC beach cameras.")
