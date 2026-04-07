import base64
import math
import time
from datetime import datetime

MAX_CHOP_PENALTY = 0.30   # maximum scoring reduction from wind chop/swell contamination
MAX_DISTANCE_KM  = 200    # radius used for "Near Me" spot search

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_js_eval import get_geolocation

# ---------------------------------------------------------------------------
# Live Camera Catalogue – only cams with confirmed HLS streams
# ---------------------------------------------------------------------------

HLS_BASE = "https://d1nm4r8e5x1rwd.cloudfront.net/cw"

# South → North ordering
SEQ_CAMS = [
    {"name": "Currumbin Alley",  "region": "Gold Coast", "stream": "rainbowbaycamera",       "lat": -28.158, "lon": 153.498},
    {"name": "Tugun",            "region": "Gold Coast", "stream": "tallebudgeracamera",      "lat": -28.145, "lon": 153.527},
    {"name": "Palm Beach South", "region": "Gold Coast", "stream": "palmbeachqldcamera",      "lat": -28.118, "lon": 153.472},
    {"name": "Tallebudgera",     "region": "Gold Coast", "stream": "tallebudgeracamera",      "lat": -28.075, "lon": 153.440},
    {"name": "Miami",            "region": "Gold Coast", "stream": "miamicamera",             "lat": -28.056, "lon": 153.440},
    {"name": "Surfers Paradise", "region": "Gold Coast", "stream": "surfersparadisecamera",   "lat": -27.998, "lon": 153.428},
    {"name": "The Spit North",   "region": "Gold Coast", "stream": "seawayspitcamera",        "lat": -27.957, "lon": 153.431},
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


def _fetch_with_retry(url: str, params: dict, max_retries: int = 3) -> requests.Response:
    """HTTP GET with exponential back-off — handles 429 rate-limit errors gracefully."""
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 429 and attempt < max_retries - 1:
                time.sleep(1.5 * (2 ** attempt))
                continue
            r.raise_for_status()
            return r
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(1.5)
                continue
            raise requests.exceptions.Timeout("Request timed out — please try again.")
    raise Exception("Data service is busy. Please wait a moment and refresh.")


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
    r = _fetch_with_retry(MARINE_URL, params)
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
    r = _fetch_with_retry(WEATHER_URL, params)
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
# Significantly improved surf scoring engine
# ---------------------------------------------------------------------------

def _wave_height_score(h: float, break_type: str = "Beach") -> float:
    """
    Score 0-10 based on wave height, calibrated per break type.
    Point breaks thrive on larger, more powerful swells.
    Beach breaks wash out when too big.
    """
    # (min_rideable, ideal_low, ideal_high, survivable_max)
    ranges = {
        "Point": (0.5, 1.0, 3.0, 5.5),
        "Alley": (0.4, 0.8, 2.5, 4.5),
        "Beach": (0.3, 0.6, 2.0, 4.0),
    }
    min_r, id_low, id_high, max_r = ranges.get(break_type, ranges["Beach"])
    if h < 0.2:      return 0.0
    if h < min_r:    return (h - 0.2) / (min_r - 0.2) * 3.0
    if h < id_low:   return 3.0 + (h - min_r) / (id_low - min_r) * 7.0
    if h <= id_high: return 10.0
    if h <= max_r:   return 10.0 - (h - id_high) / (max_r - id_high) * 6.5
    return max(0.0, 3.5 - (h - max_r) * 1.5)


def _period_score(p: float) -> float:
    """
    Score 0-10 based on swell period.
    Ground swell (>12 s) produces clean, powerful, well-organised waves.
    Short-period wind swell (<8 s) = choppy and disorganised.
    """
    if p < 4:    return 0.0
    if p < 7:    return (p - 4) / 3 * 3.0           # 4-7 s: poor wind swell
    if p < 10:   return 3.0 + (p - 7) / 3 * 3.5     # 7-10 s: mixed/short period
    if p < 13:   return 6.5 + (p - 10) / 3 * 2.5    # 10-13 s: decent ground swell
    if p <= 18:  return 9.0 + (p - 13) / 5 * 1.0    # 13-18 s: quality ground swell
    return 10.0                                        # >18 s: ultra-long period


def _wind_score(speed: float, wind_dir: float, orientation: float) -> float:
    """
    Score 0-10 based on wind speed and direction relative to the break.
    Offshore + calm = glassy perfection. Onshore + strong = blown out.
    """
    offshore_dir = (orientation + 180) % 360
    diff = abs(wind_dir - offshore_dir)
    if diff > 180: diff = 360 - diff

    # Direction component
    if diff <= 30:    dir_score = 1.00
    elif diff <= 60:  dir_score = 1.00 - (diff - 30) / 30 * 0.25   # 1.00 → 0.75
    elif diff <= 90:  dir_score = 0.75 - (diff - 60) / 30 * 0.30   # 0.75 → 0.45
    elif diff <= 120: dir_score = 0.45 - (diff - 90) / 30 * 0.25   # 0.45 → 0.20
    elif diff <= 150: dir_score = 0.20 - (diff - 120) / 30 * 0.15  # 0.20 → 0.05
    else:             dir_score = max(0.0, 0.05 - (diff - 150) / 30 * 0.05)

    # Speed component: calm is ideal; offshore wind at moderate speed grooms the face
    if speed < 5:    speed_mult = 1.05   # glassy bonus
    elif speed < 12: speed_mult = 1.00
    elif speed < 20: speed_mult = 1.00 - (speed - 12) / 8 * 0.25
    elif speed < 30: speed_mult = 0.75 - (speed - 20) / 10 * 0.40
    elif speed < 45: speed_mult = 0.35 - (speed - 30) / 15 * 0.30
    else:            speed_mult = 0.05

    # Offshore wind at moderate speed can actually help clean up the face
    if diff < 45 and 5 < speed < 20:
        speed_mult = max(speed_mult, 0.80)

    return min(10.0, round(10 * dir_score * speed_mult, 1))


def _swell_dir_score(wave_dir: float, orientation: float) -> float:
    """
    Score 0-10 based on swell direction vs break orientation.
    Direct hit = 10, heavily angled = diminishing returns, blocked = 0.
    """
    diff = abs(wave_dir - orientation)
    if diff > 180: diff = 360 - diff
    if diff <= 15:   return 10.0
    if diff <= 40:   return 10.0 - (diff - 15) / 25 * 2.0    # 10 → 8
    if diff <= 70:   return 8.0  - (diff - 40) / 30 * 3.5    # 8 → 4.5
    if diff <= 100:  return 4.5  - (diff - 70) / 30 * 3.5    # 4.5 → 1
    if diff <= 130:  return max(0.0, 1.0 - (diff - 100) / 30) # 1 → 0
    return 0.0


def _swell_energy_bonus(swell_height: float, swell_period: float) -> float:
    """
    Bonus 0.0–1.0 for high-energy ground swell.
    Wave power ∝ H² × T (based on linear wave theory).
    Rewards powerful, long-period swells that create quality surf.
    """
    if swell_period < 8 or swell_height < 0.4:
        return 0.0
    energy = (swell_height ** 2) * swell_period
    # Calibrated: 1 m @ 12 s = 12 (entry), 2 m @ 14 s = 56 (quality), 3 m @ 18 s = 162 (epic)
    if energy < 8:   return 0.0
    if energy < 30:  return (energy - 8) / 22 * 0.40
    if energy < 80:  return 0.40 + (energy - 30) / 50 * 0.40
    return min(1.0, 0.80 + (energy - 80) / 100 * 0.20)


def _chop_penalty(wind_wave_h: float, total_h: float, wind_speed: float) -> float:
    """
    Penalty 0.0–0.30 for choppiness.
    High wind-wave fraction + strong wind = heavily textured, unrideable surface.
    """
    ww_fraction = wind_wave_h / max(total_h, 0.1) if total_h > 0 else 0.0
    speed_chop   = max(0.0, (wind_speed - 18) / 60)
    return min(MAX_CHOP_PENALTY, ww_fraction * 0.22 + speed_chop * 0.12)


# Break-type weight presets  (height, period, wind, direction)
_BREAK_WEIGHTS = {
    "Point": {"height": 0.25, "period": 0.30, "wind": 0.25, "direction": 0.20},
    "Alley": {"height": 0.28, "period": 0.25, "wind": 0.22, "direction": 0.25},
    "Beach": {"height": 0.28, "period": 0.25, "wind": 0.27, "direction": 0.20},
}


def surf_score(
    wave_height, wave_period, wave_direction,
    wind_speed, wind_direction, orientation,
    break_type: str = "Beach",
    wind_wave_height: float = 0.0,
) -> float:
    """
    Composite surf score 0–10.
    Combines swell height (per break type), period (ground swell emphasis),
    wind quality, swell alignment, swell energy bonus, and surface chop penalty.
    """
    h_s = _wave_height_score(float(wave_height), break_type)
    p_s = _period_score(float(wave_period))
    w_s = _wind_score(float(wind_speed), float(wind_direction), orientation)
    d_s = _swell_dir_score(float(wave_direction), orientation)
    energy_bonus = _swell_energy_bonus(float(wave_height), float(wave_period))
    chop         = _chop_penalty(float(wind_wave_height), float(wave_height), float(wind_speed))
    bw   = _BREAK_WEIGHTS.get(break_type, _BREAK_WEIGHTS["Beach"])
    raw  = (h_s * bw["height"] + p_s * bw["period"] +
            w_s * bw["wind"]   + d_s * bw["direction"])
    return round(min(10.0, max(0.0, (raw + energy_bonus) * (1.0 - chop))), 1)


def get_score_breakdown(
    wave_height, wave_period, wave_direction,
    wind_speed, wind_direction, orientation,
    break_type: str = "Beach",
    wind_wave_height: float = 0.0,
) -> dict:
    """Return individual component scores for the detailed breakdown panel."""
    return {
        "height":    round(_wave_height_score(float(wave_height), break_type), 1),
        "period":    round(_period_score(float(wave_period)), 1),
        "wind":      round(_wind_score(float(wind_speed), float(wind_direction), orientation), 1),
        "direction": round(_swell_dir_score(float(wave_direction), orientation), 1),
        "energy":    round(_swell_energy_bonus(float(wave_height), float(wave_period)), 2),
        "chop_pct":  round(_chop_penalty(float(wind_wave_height), float(wave_height), float(wind_speed)) * 100, 0),
    }


def score_label(score: float) -> str:
    if score >= 9.0: return "🔥 Epic"
    if score >= 7.5: return "⭐ Very Good"
    if score >= 6.0: return "✅ Good"
    if score >= 4.5: return "🟡 Fair"
    if score >= 3.0: return "🟠 Poor"
    return "❌ Flat / Blown Out"


def score_color(score: float) -> str:
    if score >= 9.0: return "#7C3AED"   # purple  – epic
    if score >= 7.5: return "#059669"   # green   – very good
    if score >= 6.0: return "#0077B6"   # blue    – good
    if score >= 4.5: return "#D97706"   # amber   – fair
    if score >= 3.0: return "#EA580C"   # orange  – poor
    return "#DC2626"                     # red     – flat/blown out


def score_description(score: float, wave_height: float, wave_period: float,
                      wind_speed: float, wind_rel: str) -> str:
    """One-line human-readable summary of current conditions."""
    wave_str   = f"{wave_height:.1f} m @ {wave_period:.0f} s"
    wind_clean = wind_rel.split()[-1]
    if score >= 9.0:
        return f"World-class! {wave_str} — drop everything and paddle out. 🤙"
    if score >= 7.5:
        return f"Excellent! {wave_str}, {wind_clean} winds — definitely get in the water."
    if score >= 6.0:
        return f"Solid surf. {wave_str} with {wind_speed:.0f} km/h {wind_clean} wind."
    if score >= 4.5:
        return f"Average conditions. {wave_str} — worth it if you're keen."
    if score >= 3.0:
        return f"Poor surf. {wave_str} — only if you're desperate."
    return "Not worth it. Flat or blown-out conditions."


def uv_label(uv: float) -> str:
    if uv < 3:  return "Low 🟢"
    if uv < 6:  return "Moderate 🟡"
    if uv < 8:  return "High 🟠"
    if uv < 11: return "Very High 🔴"
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
    dlat       = math.radians(lat2 - lat1)
    dlon       = math.radians(lon2 - lon1)
    sin_dlat   = math.sin(dlat / 2)
    sin_dlon   = math.sin(dlon / 2)
    cos_lat1   = math.cos(math.radians(lat1))
    cos_lat2   = math.cos(math.radians(lat2))
    a = sin_dlat ** 2 + cos_lat1 * cos_lat2 * sin_dlon ** 2
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


def _safe_col(row, col, fallback=None):
    """Safely read a column from a pandas Series, returning fallback if missing/NaN."""
    if col in row.index and pd.notna(row[col]):
        return row[col]
    return fallback


# ---------------------------------------------------------------------------
# Cam gallery helpers  (iframe-safe HTML rendering)
# ---------------------------------------------------------------------------

_GALLERY_COLS   = 3
_CARD_HEIGHT_PX = 285
_GALLERY_PAD_PX = 50


def _cam_card_html(cam: dict, idx: int) -> str:
    name         = cam["name"]
    region       = cam["region"]
    badge_color  = "#b45309" if region == "Gold Coast" else "#047857"
    stream_url   = f"{HLS_BASE}/{cam['stream']}.stream/playlist.m3u8"

    return f"""
    <div style="border-radius:14px;overflow:hidden;background:#ffffff;
                box-shadow:0 4px 16px rgba(0,0,0,0.10);border:1px solid #e2e8f0;
                transition:transform 0.18s,box-shadow 0.18s;"
         onmouseover="this.style.transform='translateY(-3px)';this.style.boxShadow='0 8px 28px rgba(0,0,0,0.16)'"
         onmouseout="this.style.transform='';this.style.boxShadow='0 4px 16px rgba(0,0,0,0.10)'">
      <div style="position:relative;" id="wrap{idx}">
        <span style="position:absolute;top:10px;left:10px;z-index:10;
                     background:#dc2626;color:white;font-size:0.60rem;
                     font-weight:800;padding:3px 10px;border-radius:5px;
                     letter-spacing:0.10em;pointer-events:none;
                     box-shadow:0 2px 8px rgba(220,38,38,0.45);">● LIVE</span>
        <button onclick="goFS({idx})"
                style="position:absolute;top:10px;right:10px;z-index:10;
                       background:rgba(0,0,0,0.55);color:white;border:none;
                       border-radius:6px;padding:4px 10px;font-size:0.78rem;
                       cursor:pointer;backdrop-filter:blur(4px);" title="Full screen">⛶</button>
        <video id="cam{idx}" data-src="{stream_url}"
               muted playsinline controls
               style="width:100%;height:195px;background:#0a0a0a;display:block;object-fit:cover;">
        </video>
      </div>
      <div style="padding:12px 14px;display:flex;justify-content:space-between;
                  align-items:center;gap:8px;">
        <div style="min-width:0;flex:1;">
          <div style="color:#0f172a;font-weight:700;font-size:0.92rem;
                      white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{name}</div>
          <span style="display:inline-block;margin-top:5px;background:{badge_color};
                       color:white;font-size:0.60rem;font-weight:700;padding:2px 9px;
                       border-radius:4px;letter-spacing:0.06em;">{region.upper()}</span>
        </div>
        <div style="flex-shrink:0;">
          <span style="background:#ecfdf5;color:#059669;padding:5px 12px;
                       border-radius:6px;font-size:0.72rem;font-weight:700;
                       border:1px solid #a7f3d0;">✓ Live Stream</span>
        </div>
      </div>
    </div>"""


def build_cam_gallery(cams: list) -> tuple:
    """Return (html_string, height_px) for the responsive cam card grid."""
    n_rows = math.ceil(len(cams) / _GALLERY_COLS)
    height = n_rows * _CARD_HEIGHT_PX + _GALLERY_PAD_PX
    cards  = "\n".join(_cam_card_html(c, i) for i, c in enumerate(cams))

    hls_inits = "\n".join(
        f"  initHLS('cam{i}', '{HLS_BASE}/{c['stream']}.stream/playlist.m3u8');"
        for i, c in enumerate(cams) if c.get("stream")
    )

    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:transparent;padding:6px 4px 12px;
        font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif}}
  .grid{{display:grid;grid-template-columns:repeat({_GALLERY_COLS},1fr);gap:18px}}
  @media(max-width:600px){{.grid{{grid-template-columns:1fr}}}}
  @media(max-width:960px) and (min-width:601px){{.grid{{grid-template-columns:repeat(2,1fr)}}}}
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
        v.parentElement.innerHTML = '<div style="height:195px;background:#fef2f2;'
          + 'display:flex;align-items:center;justify-content:center;color:#991b1b;'
          + 'font-size:0.85rem;text-align:center;flex-direction:column;gap:8px;">'
          + '<span style=\\"font-size:1.8rem\\">📷</span>'
          + 'Stream temporarily unavailable</div>';
      }}
    }});
  }} else if (v.canPlayType('application/vnd.apple.mpegurl')) {{
    v.src = url;
  }}
}}
function goFS(idx) {{
  var wrap = document.getElementById('wrap' + idx);
  var el = wrap || document.getElementById('cam' + idx);
  if (!el) return;
  var fn = el.requestFullscreen || el.webkitRequestFullscreen || el.mozRequestFullScreen || el.msRequestFullscreen;
  if (fn) fn.call(el);
}}
{hls_inits}
</script>
</body></html>"""
    return html, height


def render_html_safe(html: str, height: int, scrolling: bool = False) -> None:
    """
    Render raw HTML using st.iframe (data URI) — avoids st.components.v1.html deprecation.
    Falls back to components.html for older Streamlit versions.
    """
    if hasattr(st, "iframe"):
        b64 = base64.b64encode(html.encode("utf-8")).decode("utf-8")
        st.iframe(src=f"data:text/html;base64,{b64}", height=height, scrolling=scrolling)
    else:
        import streamlit.components.v1 as _c
        _c.html(html, height=height, scrolling=scrolling)


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Surf Buddy 🏄", page_icon="🏄", layout="wide")

st.markdown("""
<style>
  /* ── Base ── */
  .stApp { background: #f0f4f8; }
  .block-container { padding: 1.5rem 2rem 3rem 2rem !important; max-width: 1400px; }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a2540 0%, #0c3060 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.08) !important;
  }
  section[data-testid="stSidebar"] > div:first-child {
    padding: 1.5rem 1.1rem 1rem 1.1rem !important;
  }
  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] span,
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] div { color: #cbd5e1 !important; }
  section[data-testid="stSidebar"] h3,
  section[data-testid="stSidebar"] h4 { color: #93c5fd !important; font-size: 0.88rem !important; letter-spacing: 0.02em; }
  section[data-testid="stSidebar"] a { color: #7dd3fc !important; }
  section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12) !important; }
  section[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.16) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
  }
  section[data-testid="stSidebar"] [data-testid="stSuccess"] {
    background: rgba(5,150,105,0.18) !important;
    border: 1px solid rgba(5,150,105,0.35) !important;
    border-radius: 8px !important;
  }
  section[data-testid="stSidebar"] [data-testid="stSuccess"] p { color: #6ee7b7 !important; }
  section[data-testid="stSidebar"] [data-testid="stInfo"] {
    background: rgba(59,130,246,0.18) !important;
    border: 1px solid rgba(59,130,246,0.35) !important;
    border-radius: 8px !important;
  }
  section[data-testid="stSidebar"] [data-testid="stInfo"] p { color: #93c5fd !important; }

  /* ── Metric cards ── */
  [data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
    padding: 18px 20px !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06) !important;
  }
  [data-testid="stMetricValue"] { color: #0f172a !important; font-size: 1.35rem !important; font-weight: 700 !important; }
  [data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.78rem !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.04em; }
  [data-testid="stMetricDelta"] { font-size: 0.80rem !important; }

  /* ── Tabs ── */
  [data-testid="stTabs"] button {
    font-weight: 600 !important;
    color: #64748b !important;
    font-size: 0.95rem !important;
    padding: 10px 22px !important;
    border-radius: 0 !important;
  }
  [data-testid="stTabs"] button[aria-selected="true"] {
    color: #0077b6 !important;
    border-bottom: 3px solid #0077b6 !important;
  }

  /* ── Typography ── */
  h2, h3 { color: #0f172a !important; font-weight: 700 !important; }
  p, li { color: #475569 !important; }
  .stMarkdown p { color: #475569 !important; }
  .stCaption, [data-testid="stCaptionContainer"] { color: #94a3b8 !important; font-size: 0.80rem !important; }
  hr { border-color: #e2e8f0 !important; }

  /* ── Alerts ── */
  [data-testid="stSuccess"] { background: #f0fdf4 !important; border: 1px solid #86efac !important; border-radius: 10px !important; }
  [data-testid="stSuccess"] p { color: #166534 !important; }
  [data-testid="stInfo"] { background: #eff6ff !important; border: 1px solid #93c5fd !important; border-radius: 10px !important; }
  [data-testid="stInfo"] p { color: #1e40af !important; }
  [data-testid="stWarning"] { background: #fffbeb !important; border: 1px solid #fcd34d !important; border-radius: 10px !important; }
  [data-testid="stWarning"] p { color: #92400e !important; }
  [data-testid="stError"] { background: #fef2f2 !important; border: 1px solid #fca5a5 !important; border-radius: 10px !important; }
  [data-testid="stError"] p { color: #991b1b !important; }

  /* ── Expander & Tables ── */
  [data-testid="stExpander"] { background: #ffffff !important; border: 1px solid #e2e8f0 !important; border-radius: 12px !important; }
  .stDataFrame { border-radius: 12px !important; overflow: hidden !important; }

  /* ── Progress bar ── */
  [data-testid="stProgress"] > div { background: #e2e8f0 !important; border-radius: 100px !important; }
  [data-testid="stProgress"] > div > div { background: linear-gradient(90deg,#0077b6,#0096c7) !important; border-radius: 100px !important; }

  /* ── Mobile ── */
  @media (max-width: 768px) {
    .block-container { padding: 1rem 0.75rem 2rem 0.75rem !important; }
    [data-testid="stMetricValue"] { font-size: 1.1rem !important; }
    [data-testid="stTabs"] button { padding: 8px 12px !important; font-size: 0.85rem !important; }
  }
</style>
""", unsafe_allow_html=True)

# ── Header banner ────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#023e8a 0%,#0077b6 60%,#0096c7 100%);
            border-radius:18px;padding:22px 28px;margin-bottom:22px;
            display:flex;align-items:center;gap:18px;
            box-shadow:0 6px 24px rgba(0,119,182,0.28);">
  <span style="font-size:3rem;line-height:1;">🏄</span>
  <div>
    <div style="color:#ffffff;font-size:1.85rem;font-weight:800;line-height:1.15;
                letter-spacing:-0.02em;">Surf Buddy</div>
    <div style="color:rgba(255,255,255,0.78);font-size:0.88rem;margin-top:5px;
                display:flex;gap:16px;flex-wrap:wrap;">
      <span>🎥 Gold Coast Live Cams</span>
      <span>·</span>
      <span>📍 Near Me Rankings</span>
      <span>·</span>
      <span>📈 7-Day Forecast</span>
    </div>
  </div>
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
    st.markdown(
        f'<p style="color:#4a6080;font-size:0.85rem;margin:-4px 0 12px">'
        f'<strong style="color:#1a7a4a">&#9679; {len(SEQ_CAMS)} live streams</strong> — '
        f'Gold Coast City Council (GCCC) direct HLS feeds. '
        f'Press <strong>⛶</strong> on any camera to go full screen.</p>',
        unsafe_allow_html=True,
    )

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
                _safe_col(srow, "swell_wave_height", srow["wave_height"]),
                _safe_col(srow, "swell_wave_period", srow["wave_period"]),
                _safe_col(srow, "swell_wave_direction", srow["wave_direction"]),
                srow["windspeed_10m"], srow["winddirection_10m"],
                s["orientation"], s["break_type"],
                _safe_col(srow, "wind_wave_height", 0.0),
            )
            dist_km = haversine_km(user_lat, user_lon, s["lat"], s["lon"])
            near_rows.append({
                "Spot":      s["name"],
                "Region":    s["region"],
                "Break":     s["break_type"],
                "Dist (km)": round(dist_km, 1),
                "Wave (m)":  round(srow["wave_height"], 1),
                "Swell (m)": round(_safe_col(srow, "swell_wave_height", srow["wave_height"]), 1),
                "Period (s)":round(_safe_col(srow, "swell_wave_period", srow["wave_period"]), 0),
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
            .query(f"`Dist (km)` <= {MAX_DISTANCE_KM}")
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
  <div style="font-size:1.6rem;font-weight:800;color:#1a2332;">
    🏆 Go to <span style="color:{hero_color}">{top['Spot']}</span>
  </div>
  <div style="color:#3a5068;margin-top:6px;font-size:0.95rem;">
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
            _safe_col(r, "swell_wave_height", r["wave_height"]),
            _safe_col(r, "swell_wave_period",  r["wave_period"]),
            _safe_col(r, "swell_wave_direction", r["wave_direction"]),
            r["windspeed_10m"], r["winddirection_10m"],
            spot["orientation"], spot["break_type"],
            _safe_col(r, "wind_wave_height", 0.0),
        ) if all(pd.notna(r[c]) for c in required_cols) else np.nan,
        axis=1,
    )

    row       = current_row(df)
    now_score = surf_score(
        _safe_col(row, "swell_wave_height", row["wave_height"]),
        _safe_col(row, "swell_wave_period",  row["wave_period"]),
        _safe_col(row, "swell_wave_direction", row["wave_direction"]),
        row["windspeed_10m"], row["winddirection_10m"],
        spot["orientation"], spot["break_type"],
        _safe_col(row, "wind_wave_height", 0.0),
    )

    # ── Current conditions ───────────────────────────────────────────────────
    st.subheader(f"📡 {spot_name} — Right Now")

    swell_h = _safe_col(row, "swell_wave_height", row["wave_height"])
    swell_p = _safe_col(row, "swell_wave_period",  row["wave_period"])
    swell_d = _safe_col(row, "swell_wave_direction", row["wave_direction"])
    water_t = _safe_col(row, "sea_surface_temperature")
    uv_val  = _safe_col(row, "uv_index")
    ww_h    = _safe_col(row, "wind_wave_height", 0.0)

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
<div style="background:#ffffff;border:1px solid #d0d8e4;border-radius:8px;
            padding:10px 20px;margin-top:6px;display:flex;gap:24px;flex-wrap:wrap;
            font-size:0.85rem;color:#3a5068;">
  <span>Wind quality: <strong style="color:#1a2332">
    {wind_relation(row['winddirection_10m'], spot['orientation'])}</strong></span>
  <span>Chop: <strong style="color:#1a2332">{ww_h:.1f} m wind swell</strong></span>
  <span>Swell direction: <strong style="color:#1a2332">
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
        paper_bgcolor="#f5f7fa", plot_bgcolor="#ffffff",
        font_color="#3a5068", hovermode="x unified",
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
            paper_bgcolor="#f5f7fa", font_color="#3a5068")
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
                _safe_col(srow, "swell_wave_height", srow["wave_height"]),
                _safe_col(srow, "swell_wave_period",  srow["wave_period"]),
                _safe_col(srow, "swell_wave_direction", srow["wave_direction"]),
                srow["windspeed_10m"], srow["winddirection_10m"],
                s["orientation"], s["break_type"],
                _safe_col(srow, "wind_wave_height", 0.0),
            )
            rank_rows.append({
                "Spot":       s["name"],
                "Break":      s["break_type"],
                "Wave (m)":   round(srow["wave_height"], 1),
                "Swell (m)":  round(_safe_col(srow, "swell_wave_height", srow["wave_height"]), 1),
                "Period (s)": round(_safe_col(srow, "swell_wave_period", srow["wave_period"]), 0),
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
