import base64
import math
import time
from datetime import datetime, timedelta, timezone

MAX_CHOP_PENALTY = 0.30   # maximum scoring reduction from wind chop/swell contamination
MAX_DISTANCE_KM  = 200    # radius used for "Near Me" spot search

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_js_eval import get_geolocation, streamlit_js_eval

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
def fetch_marine(lat: float, lon: float, tz: str = "UTC") -> pd.DataFrame:
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
        "timezone": tz,
    }
    r = _fetch_with_retry(MARINE_URL, params)
    df = pd.DataFrame(r.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(tz)
    return df


@st.cache_data(ttl=900)
def fetch_weather(lat: float, lon: float, tz: str = "UTC") -> pd.DataFrame:
    params = {
        "latitude":  lat,
        "longitude": lon,
        "hourly": "windspeed_10m,winddirection_10m,uv_index",
        "forecast_days": 7,
        "timezone": tz,
    }
    r = _fetch_with_retry(WEATHER_URL, params)
    df = pd.DataFrame(r.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(tz)
    return df


def get_forecast(lat: float, lon: float, tz: str = "UTC") -> pd.DataFrame:
    return fetch_marine(lat, lon, tz).merge(fetch_weather(lat, lon, tz), on="time")


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


def sun_times(lat: float, lon: float, local_date) -> tuple:
    """Return (sunrise, sunset) as UTC-aware datetimes for the given lat/lon/date.

    Uses the NOAA solar-position approximation (±5 min accuracy).
    Returns (None, None) under polar-night or midnight-sun conditions.
    """
    doy = local_date.timetuple().tm_yday
    gamma = 2 * math.pi / 365 * (doy - 1)

    # Equation of time (minutes)
    eot = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.040890 * math.sin(2 * gamma)
    )

    # Solar declination (radians)
    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)   + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2*gamma) + 0.000907 * math.sin(2*gamma)
        - 0.002697 * math.cos(3*gamma) + 0.001480 * math.sin(3*gamma)
    )

    lat_rad = math.radians(lat)
    cos_ha = (
        math.cos(math.radians(90.833)) / (math.cos(lat_rad) * math.cos(decl))
        - math.tan(lat_rad) * math.tan(decl)
    )

    if cos_ha > 1:
        return None, None   # polar night
    if cos_ha < -1:
        return None, None   # midnight sun

    ha_deg = math.degrees(math.acos(cos_ha))

    # Solar noon in minutes from midnight UTC
    solar_noon = 720 - (eot + 4 * lon)
    sunrise_min = solar_noon - ha_deg * 4
    sunset_min  = solar_noon + ha_deg * 4

    base = datetime(local_date.year, local_date.month, local_date.day, tzinfo=timezone.utc)
    return (
        base + timedelta(minutes=sunrise_min),
        base + timedelta(minutes=sunset_min),
    )


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
    now = pd.Timestamp.now(tz=df["time"].dt.tz)
    return df.loc[(df["time"] - now).abs().idxmin()]


def best_window(df: pd.DataFrame, spot: dict) -> pd.DataFrame:
    now    = pd.Timestamp.now(tz=df["time"].dt.tz)
    future = df[df["time"] >= now].head(48).copy()

    # Keep only daylight hours (sunrise − 30 min → sunset + 30 min).
    # Pre-compute sun times once per unique date to avoid redundant calculations.
    lat, lon = spot["lat"], spot["lon"]
    _sun_cache: dict = {}
    def _is_daylight(ts: pd.Timestamp) -> bool:
        local_date = ts.date()   # ts is already tz-aware in the user's local tz
        if local_date not in _sun_cache:
            _sun_cache[local_date] = sun_times(lat, lon, local_date)
        sr, ss = _sun_cache[local_date]
        if sr is None:
            return True   # midnight sun — always OK
        ts_utc = ts.tz_convert("UTC")
        sr_pd  = pd.Timestamp(sr)
        ss_pd  = pd.Timestamp(ss)
        return (ts_utc >= sr_pd - pd.Timedelta(minutes=30) and
                ts_utc <= ss_pd + pd.Timedelta(minutes=30))

    future = future[future["time"].apply(_is_daylight)]

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
# Travel-time calculator + "grab your board" messaging
# ---------------------------------------------------------------------------

# Average coastal road speed used for drive-time estimates (km/h)
_DRIVE_SPEED_KMH = 60
# Wax-up / gear prep time added on top of the drive (minutes)
_PREP_MINS = 20


def travel_time_mins(dist_km: float) -> int:
    """Return estimated one-way drive time in whole minutes."""
    return max(5, round((dist_km / _DRIVE_SPEED_KMH) * 60))


def leave_by_time(best_wave_time: pd.Timestamp, dist_km: float) -> pd.Timestamp:
    """Return the timestamp you need to walk out the door by."""
    return best_wave_time - pd.Timedelta(minutes=travel_time_mins(dist_km) + _PREP_MINS)


def go_surf_msg(score: float, spot_name: str, best_time: pd.Timestamp,
                leave_time: pd.Timestamp, drive_mins: int) -> str:
    """
    Generate a fun, surf-culture 'go surf' call-to-action.
    Returns an HTML string ready for st.markdown(unsafe_allow_html=True).
    """
    time_str  = best_time.strftime("%H:%M")
    leave_str = leave_time.strftime("%H:%M")
    hour      = leave_time.hour
    day_str   = leave_time.strftime("%A")

    # Time-of-day flavour
    if hour < 7:
        sesh_vibe = "dawn patrol 🌅"
    elif hour < 10:
        sesh_vibe = "morning sesh 🌤️"
    elif hour < 14:
        sesh_vibe = "midday rip ☀️"
    elif hour < 17:
        sesh_vibe = "arvo sesh 🌊"
    else:
        sesh_vibe = "sunset sesh 🌇"

    drive_str = f"{drive_mins} min drive" if drive_mins >= 10 else "just around the corner"

    if score >= 9.0:
        headline = f"🔥 This is FIRING — it's going OFF at {spot_name}!"
        cta      = (f"Drop everything. Grab your board, slap on some wax and "
                    f"<strong>be out the door by {leave_str}</strong>. "
                    f"It's a {drive_str} to be at the peak by {time_str}. DO NOT miss this {sesh_vibe}. 🤙")
    elif score >= 7.5:
        headline = f"⭐ {spot_name} is pumping — {sesh_vibe} is on!"
        cta      = (f"Grab your board and towel, double-check the wax and "
                    f"<strong>be out the door by {leave_str}</strong>. "
                    f"It's a {drive_str}, and the waves are waiting by {time_str}. Get some! 🏄")
    elif score >= 6.0:
        headline = f"✅ Solid surf at {spot_name} — worth the trip!"
        cta      = (f"Pack your board and a dry towel and "
                    f"<strong>leave by {leave_str}</strong> on {day_str}. "
                    f"It's a {drive_str} for a {time_str} splash. You won't regret it. 🤙")
    elif score >= 4.5:
        headline = f"🟡 {spot_name} — average but rideable."
        cta      = (f"If you're keen, pack your gear and "
                    f"<strong>aim to leave by {leave_str}</strong>. "
                    f"It's a {drive_str} to make the {time_str} window. Manage your expectations 😄")
    else:
        headline = f"🟠 {spot_name} — marginal at best."
        cta      = (f"Honestly, the couch is calling 🛋️ But if you're desperate, "
                    f"<strong>leave by {leave_str}</strong> ({drive_str}, "
                    f"peaking around {time_str}).")

    return headline, cta


# ---------------------------------------------------------------------------
# Detailed wave conditions analysis
# ---------------------------------------------------------------------------

def wave_conditions_analysis(
    wave_height: float,
    swell_height: float,
    swell_period: float,
    swell_dir: float,
    wind_speed: float,
    wind_dir: float,
    orientation: float,
    break_type: str,
    wind_wave_height: float,
    water_temp: float | None = None,
    tz: str = "UTC",
) -> dict:
    """
    Returns a comprehensive dict describing wave breaking character, surface texture,
    currents, hazards, skill level, and wetsuit recommendation.
    """
    result = {}

    # ── Face height (surfer's traditional call — approx 1.3× back height) ──
    face_h = wave_height * 1.3
    if face_h < 0.4:   face_size = "ankle high"
    elif face_h < 0.8: face_size = "knee to waist"
    elif face_h < 1.2: face_size = "waist to chest"
    elif face_h < 1.6: face_size = "chest to head high"
    elif face_h < 2.2: face_size = "head high"
    elif face_h < 3.0: face_size = "overhead"
    elif face_h < 4.5: face_size = "double overhead"
    else:              face_size = "triple overhead +"
    result["face_height"]    = f"{face_h:.1f} m  ({face_size})"
    result["face_size_label"] = face_size

    # ── Steepness — deep-water wavelength L₀ = 1.56 × T² ──
    wavelength = 1.56 * (swell_period ** 2) if swell_period > 0 else 1.0
    steepness  = swell_height / wavelength
    result["steepness"] = steepness

    # ── Wind relationship to break ──
    offshore_dir = (orientation + 180) % 360
    wind_diff    = abs(wind_dir - offshore_dir)
    if wind_diff > 180: wind_diff = 360 - wind_diff
    is_offshore = wind_diff < 45
    is_onshore  = wind_diff > 135

    # ── Breaking character ──
    if wave_height < 0.3:
        breaking        = "Flat — no rideable surf"
        breaking_detail = "The ocean is essentially flat. Great for swimming, absolutely not for surfing."
    elif is_onshore and wind_speed > 28:
        breaking        = "Blown Out — messy closeouts"
        breaking_detail = (f"Strong {wind_speed:.0f} km/h onshore wind is destroying wave shape, "
                           "creating disorganised, crumbling walls with no clean face.")
    elif swell_period >= 14 and steepness > 0.018 and (is_offshore or wind_speed < 10):
        if break_type == "Point":
            breaking        = "Long Walling Barrels — point-break perfection"
            breaking_detail = (f"Long-period ground swell ({swell_period:.0f}s) wrapping the point is "
                               "generating long, mechanical walls with hollow barrel sections. "
                               "Expect fast, critical take-offs and extended rides.")
        else:
            breaking        = "Hollow & Barreling — steep critical faces"
            breaking_detail = (f"Long-period swell ({swell_period:.0f}s) is throwing up hollow, "
                               "steep wave faces with real barrel potential. Quick feet required on the drop.")
    elif swell_period >= 10 and swell_height >= 0.5 and not is_onshore:
        if break_type == "Point":
            breaking        = "Long Peeling Walls — quality point-break surf"
            breaking_detail = (f"Ground swell ({swell_period:.0f}s) is wrapping nicely, producing "
                               "long, peeling walls ideal for drawn-out turns, cutbacks, and noserides.")
        else:
            breaking        = "Peaky & Punchy — defined beach-break peaks"
            breaking_detail = (f"Solid ground swell ({swell_period:.0f}s) is producing well-defined "
                               "peaks with good punch. Expect multiple A-frame peaks up and down the beach.")
    elif swell_period >= 7 and not is_onshore:
        breaking        = "Peaky & Fun — cruisy beach surf"
        breaking_detail = (f"Short-to-moderate period ({swell_period:.0f}s) swell is creating fun, "
                           "rideable peaks. Less power than ground swell but very accessible.")
    else:
        breaking        = "Crumbling / Sectiony — wind-swell jumble"
        breaking_detail = (f"Short-period wind swell ({swell_period:.0f}s) is producing soft, "
                           "crumbling waves that section quickly — hard to build speed or link turns.")
    result["breaking"]        = breaking
    result["breaking_detail"] = breaking_detail

    # ── Surface texture ──
    if wind_speed < 5:
        surface        = "🟦 Glassy"
        surface_detail = "Near-zero wind — mirror-smooth surface. The gold standard."
    elif is_offshore and wind_speed < 20:
        surface        = "🟩 Groomed"
        surface_detail = (f"Offshore wind ({wind_speed:.0f} km/h) is lightly feathering the lips "
                          "and grooming the wave faces. Excellent conditions.")
    elif wind_speed < 15:
        surface        = "🟨 Lightly Textured"
        surface_detail = (f"Light {wind_speed:.0f} km/h wind adds minor ripples to the surface — "
                          "still very clean and surfable.")
    elif wind_speed < 25:
        surface        = "🟧 Choppy"
        surface_detail = (f"{wind_speed:.0f} km/h wind is creating chop that makes paddling harder "
                          "and wave faces messier.")
    else:
        surface        = "🟥 Blown Out"
        surface_detail = f"Strong {wind_speed:.0f} km/h winds have badly degraded surface quality."
    result["surface"]        = surface
    result["surface_detail"] = surface_detail

    # ── Longshore drift ──
    swell_diff = swell_dir - orientation
    if swell_diff >  180: swell_diff -= 360
    if swell_diff < -180: swell_diff += 360
    if abs(swell_diff) < 20:
        longshore = "Minimal — swell is nearly square-on to the beach."
    elif swell_diff > 0:
        longshore = "Pushing RIGHT along the beach (looking shoreward) — drift from swell's left angle."
    else:
        longshore = "Pushing LEFT along the beach (looking shoreward) — drift from swell's right angle."
    result["longshore"] = longshore

    # ── Rip current risk ──
    if break_type == "Beach":
        if wave_height > 2.0 or (wave_height > 1.2 and is_onshore):
            rip_strength = "⚠️ High"
            rip_detail   = (f"Big beach-break surf ({wave_height:.1f}m) drives powerful rip channels. "
                            "Identify the darker, calmer water between the peaks before paddling out. "
                            "If caught in a rip: stay calm, conserve energy, paddle parallel to shore "
                            "or angle 45° across the rip — never straight against it.")
        elif wave_height > 0.8:
            rip_strength = "Moderate"
            rip_detail   = ("Moderate rip risk. Scan the beach from the sand — rips appear as dark, "
                            "rippled lanes with fewer breaking waves. Experienced surfers use them as "
                            "a free elevator out the back.")
        else:
            rip_strength = "Low"
            rip_detail   = ("Small surf means weaker rip currents. Still worth identifying channels "
                            "from the beach before paddling out.")
    elif break_type == "Point":
        rip_strength = "Low–Moderate"
        rip_detail   = ("Point breaks have a consistent sweep along the rocks. "
                        "Hug the rocks for an easy paddle-out — fighting the break zone wastes energy. "
                        "Watch for wash from sets on the inside section.")
    else:  # Alley / Creek
        rip_strength = "Moderate"
        rip_detail   = ("Creek/alley outlets create a steady outflowing current. "
                        "Paddle out alongside the channel rather than through the whitewash. "
                        "This current can strengthen significantly during ebb tide.")
    result["rip_strength"] = rip_strength
    result["rip_detail"]   = rip_detail

    # ── Hazard level ──
    if wave_height >= 3.0:
        hazard       = "🔴 High"
        hazard_color = "#DC2626"
    elif wave_height >= 1.8:
        hazard       = "🟠 Moderate–High"
        hazard_color = "#EA580C"
    elif wave_height >= 1.0:
        hazard       = "🟡 Moderate"
        hazard_color = "#D97706"
    else:
        hazard       = "🟢 Low"
        hazard_color = "#059669"
    hazard_notes = []
    if wave_height >= 3.0:
        hazard_notes.append(f"Large surf ({wave_height:.1f}m) — powerful hold-downs, reef/shore impact risk")
    if is_onshore and wind_speed > 30:
        hazard_notes.append(f"Strong onshore gale ({wind_speed:.0f} km/h) — return-to-shore is difficult")
    if "High" in rip_strength or "Moderate" in rip_strength:
        hazard_notes.append(f"Rip current risk: {rip_strength}")
    if water_temp is not None and water_temp < 17:
        hazard_notes.append(f"Cold water ({water_temp:.0f}°C) — cold shock risk, wear adequate rubber")
    result["hazard"]       = hazard
    result["hazard_color"] = hazard_color
    result["hazard_notes"] = hazard_notes

    # ── Skill level ──
    if wave_height < 0.5:
        skill        = "🟢 All Levels / Beginner"
        skill_detail = "Very small surf. Ideal for beginners on foam/soft-top boards with supervision."
    elif wave_height < 1.0 and not is_onshore:
        skill        = "🟢 Beginner–Intermediate"
        skill_detail = "Manageable size with decent shape. Fun for learners with basic pop-up and paddling skills."
    elif wave_height < 1.8 and not is_onshore:
        skill        = "🟡 Intermediate"
        skill_detail = (f"Solid {wave_height:.1f}m surf requires confident paddling, duck-diving, "
                        "and wave reading. Not ideal for beginners.")
    elif wave_height < 2.8 and (is_offshore or wind_speed < 15):
        skill        = "🟠 Advanced"
        skill_detail = (f"Powerful {wave_height:.1f}m surf demands strong paddling, reliable duck-diving, "
                        "and the ability to handle hold-downs and wipeouts.")
    else:
        skill        = "🔴 Expert Only"
        skill_detail = (f"Serious conditions. {wave_height:.1f}m surf with significant force — "
                        "experienced big-wave surfers only.")
    if is_onshore and wind_speed > 30:
        skill        = "🔴 Expert Only"
        skill_detail = "Blown-out conditions are dangerous regardless of size."
    result["skill"]        = skill
    result["skill_detail"] = skill_detail

    # ── Crowd estimate (time-of-day, user's local time) ──
    hour = pd.Timestamp.now(tz=tz).hour
    if 5 <= hour < 8:     crowd = "🌅 Dawn patrol — hardcore locals, light crowd"
    elif 8 <= hour < 11:  crowd = "☀️ Morning peak — busiest window, popular spots will be packed"
    elif 11 <= hour < 14: crowd = "🌤️ Midday — crowd thins as locals break for work/lunch"
    elif 14 <= hour < 17: crowd = "🌊 Arvo session — second wave of surfers, often windy"
    else:                 crowd = "🌇 Evening — quiet, sunset chasers only"
    result["crowd"] = crowd

    # ── Wetsuit recommendation ──
    if water_temp is not None:
        if water_temp >= 26:   wetsuit = "Board shorts / rashie 🩲"
        elif water_temp >= 23: wetsuit = "Springsuit or light 2mm shortie 🩱"
        elif water_temp >= 20: wetsuit = "2mm–3mm shortie or springsuit 🤿"
        elif water_temp >= 17: wetsuit = "3mm–4mm full suit 🏊"
        elif water_temp >= 14: wetsuit = "4mm–5mm full suit + boots 🧊"
        else:                  wetsuit = "5mm+ hooded full suit + boots + gloves ❄️"
    else:
        wetsuit = "Check local conditions — water temp unavailable"
    result["wetsuit"] = wetsuit

    return result


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
        st.iframe(src=f"data:text/html;base64,{b64}", height=height)
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

# ── Detect user's browser timezone ───────────────────────────────────────────
_raw_tz = streamlit_js_eval(js_expressions="Intl.DateTimeFormat().resolvedOptions().timeZone", key="tz_detect")
user_tz = _raw_tz if isinstance(_raw_tz, str) and _raw_tz else "UTC"
_tz_fallback = not (isinstance(_raw_tz, str) and _raw_tz)

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
    if _tz_fallback:
        st.caption("⚠️ Could not detect your timezone — times shown in UTC.")
    else:
        st.caption(f"🕐 Times shown in: **{user_tz}**")
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
        f'<p style="color:#64748b;font-size:0.88rem;margin:0 0 14px">'
        f'<strong style="color:#059669">● {len(SEQ_CAMS)} live streams</strong> — '
        f'Gold Coast City Council (GCCC) direct HLS feeds. '
        f'Tap <strong>⛶</strong> on any camera to go full screen.</p>',
        unsafe_allow_html=True,
    )

    visible_cams = SEQ_CAMS

    gallery_html, gallery_height = build_cam_gallery(visible_cams)
    render_html_safe(gallery_html, height=gallery_height, scrolling=False)

# ============================================================================
# TAB 2 – NEAR ME
# ============================================================================
with tab_nearme:
    _hdr, _btn = st.columns([6, 1])
    with _hdr:
        st.subheader(f"📍 Best Surf Near You  ·  {loc_source} location")
    with _btn:
        if st.button("🔄 Refresh", use_container_width=True, key="refresh_nearme"):
            st.cache_data.clear()
            st.rerun()
    st.caption(
        f"Ranked by surf score right now · within {MAX_DISTANCE_KM} km of "
        f"({user_lat:.3f}°, {user_lon:.3f}°) · drive times assume {_DRIVE_SPEED_KMH} km/h avg"
    )

    near_rows = []
    progress  = st.progress(0, text="Checking all spots…")
    for i, s in enumerate(ALL_SPOTS):
        progress.progress((i + 1) / len(ALL_SPOTS), text=f"Checking {s['name']}…")
        try:
            sdf  = get_forecast(s["lat"], s["lon"], user_tz)
            srow = current_row(sdf)
            sc   = surf_score(
                _safe_col(srow, "swell_wave_height", srow["wave_height"]),
                _safe_col(srow, "swell_wave_period", srow["wave_period"]),
                _safe_col(srow, "swell_wave_direction", srow["wave_direction"]),
                srow["windspeed_10m"], srow["winddirection_10m"],
                s["orientation"], s["break_type"],
                _safe_col(srow, "wind_wave_height", 0.0),
            )
            dist_km  = haversine_km(user_lat, user_lon, s["lat"], s["lon"])
            drive    = travel_time_mins(dist_km)
            near_rows.append({
                "Spot":       s["name"],
                "Region":     s["region"],
                "Break":      s["break_type"],
                "Dist (km)":  round(dist_km, 1),
                "Drive (min)":drive,
                "Wave (m)":   round(srow["wave_height"], 1),
                "Swell (m)":  round(_safe_col(srow, "swell_wave_height", srow["wave_height"]), 1),
                "Period (s)": round(_safe_col(srow, "swell_wave_period", srow["wave_period"]), 0),
                "Wind":       f"{srow['windspeed_10m']:.0f} km/h {compass(srow['winddirection_10m'])}",
                "Wind qlty":  wind_relation(srow["winddirection_10m"], s["orientation"]),
                "Score":      sc,
                "Rating":     score_label(sc),
                "_lat":       s["lat"],
                "_lon":       s["lon"],
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
            top        = near_df.iloc[0]
            hero_color = score_color(top["Score"])
            drive_mins = int(top["Drive (min)"])
            # Best window for the top spot to give an accurate leave time
            try:
                top_spot_obj  = next(s for s in ALL_SPOTS if s["name"] == top["Spot"])
                top_df        = get_forecast(top_spot_obj["lat"], top_spot_obj["lon"], user_tz)
                top_best5     = best_window(top_df, top_spot_obj)
                _fallback     = pd.Timestamp.now(tz=user_tz) + pd.Timedelta(hours=1)
                best_wave     = top_best5.iloc[0]["time"] if not top_best5.empty else _fallback
            except Exception:
                best_wave     = pd.Timestamp.now(tz=user_tz) + pd.Timedelta(hours=1)

            leave_t    = leave_by_time(best_wave, top["Dist (km)"])
            headline, cta = go_surf_msg(
                top["Score"], top["Spot"], best_wave, leave_t, drive_mins
            )

            st.markdown(f"""
<div style="background:linear-gradient(135deg,{hero_color}18,{hero_color}08);
            border:2px solid {hero_color};border-radius:18px;
            padding:22px 26px;margin-bottom:20px;
            box-shadow:0 4px 20px {hero_color}25;">
  <div style="font-size:1.45rem;font-weight:800;color:#0f172a;margin-bottom:8px;">
    {headline}
  </div>
  <div style="background:rgba(255,255,255,0.85);border-radius:12px;
              padding:14px 18px;margin:10px 0;font-size:0.98rem;
              color:#1e293b;line-height:1.6;">
    🏄 {cta}
  </div>
  <div style="display:flex;flex-wrap:wrap;gap:10px;margin-top:10px;">
    <span style="background:{hero_color};color:white;padding:5px 14px;
                 border-radius:8px;font-size:0.88rem;font-weight:700;">
      Score {top['Score']}/10 · {top['Rating']}
    </span>
    <span style="background:#f1f5f9;color:#475569;padding:5px 14px;
                 border-radius:8px;font-size:0.88rem;font-weight:600;">
      🚗 {drive_mins} min · {top['Dist (km)']} km
    </span>
    <span style="background:#f1f5f9;color:#475569;padding:5px 14px;
                 border-radius:8px;font-size:0.88rem;font-weight:600;">
      🌊 {top['Swell (m)']}m swell @ {top['Period (s)']:.0f}s
    </span>
    <span style="background:#f1f5f9;color:#475569;padding:5px 14px;
                 border-radius:8px;font-size:0.88rem;font-weight:600;">
      💨 {top['Wind']} {top['Wind qlty']}
    </span>
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
            df = get_forecast(spot["lat"], spot["lon"], user_tz)
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

    # Metrics row
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

    # Score banner
    bar_color = score_color(now_score)
    now_wind_rel = wind_relation(row["winddirection_10m"], spot["orientation"])
    cond_desc    = score_description(now_score, swell_h, swell_p,
                                     row["windspeed_10m"], now_wind_rel)
    bd = get_score_breakdown(
        swell_h, swell_p, swell_d,
        row["windspeed_10m"], row["winddirection_10m"],
        spot["orientation"], spot["break_type"], ww_h,
    )

    st.markdown(f"""
<div style="background:linear-gradient(135deg,{bar_color}ee,{bar_color}bb);
            border-radius:14px;padding:16px 22px;color:white;
            display:flex;align-items:center;justify-content:space-between;
            flex-wrap:wrap;gap:12px;margin-top:10px;
            box-shadow:0 4px 16px {bar_color}44;">
  <div>
    <div style="font-size:1.6rem;font-weight:900;letter-spacing:-0.02em;">
      {score_label(now_score)}&ensp;<span style="opacity:0.9">{now_score} / 10</span>
    </div>
    <div style="opacity:0.88;font-size:0.92rem;margin-top:4px;">{cond_desc}</div>
  </div>
  <div style="background:rgba(255,255,255,0.18);border-radius:10px;padding:10px 16px;
              font-size:0.85rem;line-height:1.8;min-width:200px;">
    <div>⛰ Wave Height &nbsp;<strong>{bd['height']}/10</strong></div>
    <div>⏱ Period Quality &nbsp;<strong>{bd['period']}/10</strong></div>
    <div>💨 Wind Quality &nbsp;<strong>{bd['wind']}/10</strong></div>
    <div>🧭 Swell Direction &nbsp;<strong>{bd['direction']}/10</strong></div>
    <div style="margin-top:4px;opacity:0.80;font-size:0.78rem;">
      ⚡ Energy bonus +{bd['energy']:.2f} &nbsp;·&nbsp; 🌀 Chop -{bd['chop_pct']:.0f}%
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:18px'></div>", unsafe_allow_html=True)

    # ── Detailed wave conditions analysis ───────────────────────────────────
    st.subheader("🔍 Detailed Conditions Analysis")
    cond = wave_conditions_analysis(
        wave_height       = row["wave_height"],
        swell_height      = swell_h,
        swell_period      = swell_p,
        swell_dir         = swell_d,
        wind_speed        = row["windspeed_10m"],
        wind_dir          = row["winddirection_10m"],
        orientation       = spot["orientation"],
        break_type        = spot["break_type"],
        wind_wave_height  = ww_h,
        water_temp        = water_t,
        tz                = user_tz,
    )

    # Row 1 — Breaking & Surface
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
<div style="background:white;border:1px solid #e2e8f0;border-radius:14px;
            padding:18px 20px;height:100%;box-shadow:0 1px 6px rgba(0,0,0,0.06);">
  <div style="font-size:0.78rem;font-weight:700;color:#64748b;text-transform:uppercase;
              letter-spacing:0.06em;margin-bottom:8px;">🌊 How Waves Are Breaking</div>
  <div style="font-size:1.05rem;font-weight:700;color:#0f172a;margin-bottom:6px;">
    {cond['breaking']}
  </div>
  <div style="font-size:0.88rem;color:#475569;line-height:1.6;">
    {cond['breaking_detail']}
  </div>
  <div style="margin-top:12px;padding-top:10px;border-top:1px solid #f1f5f9;
              font-size:0.85rem;color:#64748b;">
    📏 <strong>Face height:</strong> {cond['face_height']}
  </div>
</div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
<div style="background:white;border:1px solid #e2e8f0;border-radius:14px;
            padding:18px 20px;height:100%;box-shadow:0 1px 6px rgba(0,0,0,0.06);">
  <div style="font-size:0.78rem;font-weight:700;color:#64748b;text-transform:uppercase;
              letter-spacing:0.06em;margin-bottom:8px;">🪟 Surface Texture</div>
  <div style="font-size:1.05rem;font-weight:700;color:#0f172a;margin-bottom:6px;">
    {cond['surface']}
  </div>
  <div style="font-size:0.88rem;color:#475569;line-height:1.6;">
    {cond['surface_detail']}
  </div>
  <div style="margin-top:12px;padding-top:10px;border-top:1px solid #f1f5f9;
              font-size:0.85rem;color:#64748b;">
    🌀 Wind chop: <strong>{ww_h:.2f} m</strong> &nbsp;·&nbsp;
    Swell steepness: <strong>{'Hollow' if cond['steepness'] > 0.025 else 'Normal' if cond['steepness'] > 0.012 else 'Mellow'}</strong>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)

    # Row 2 — Currents
    col_c, col_d = st.columns(2)
    with col_c:
        rip_col = "#DC2626" if "High" in cond['rip_strength'] else "#D97706" if "Moderate" in cond['rip_strength'] else "#059669"
        st.markdown(f"""
<div style="background:white;border:1px solid #e2e8f0;border-radius:14px;
            padding:18px 20px;height:100%;box-shadow:0 1px 6px rgba(0,0,0,0.06);">
  <div style="font-size:0.78rem;font-weight:700;color:#64748b;text-transform:uppercase;
              letter-spacing:0.06em;margin-bottom:8px;">🌀 Rip Currents</div>
  <div style="display:inline-block;background:{rip_col}18;border:1px solid {rip_col}55;
              border-radius:8px;padding:4px 12px;font-size:0.88rem;font-weight:700;
              color:{rip_col};margin-bottom:10px;">Risk: {cond['rip_strength']}</div>
  <div style="font-size:0.88rem;color:#475569;line-height:1.6;">
    {cond['rip_detail']}
  </div>
</div>""", unsafe_allow_html=True)

    with col_d:
        st.markdown(f"""
<div style="background:white;border:1px solid #e2e8f0;border-radius:14px;
            padding:18px 20px;height:100%;box-shadow:0 1px 6px rgba(0,0,0,0.06);">
  <div style="font-size:0.78rem;font-weight:700;color:#64748b;text-transform:uppercase;
              letter-spacing:0.06em;margin-bottom:8px;">↔️ Longshore Drift & Current</div>
  <div style="font-size:0.88rem;color:#475569;line-height:1.6;margin-bottom:10px;">
    {cond['longshore']}
  </div>
  <div style="font-size:0.78rem;font-weight:700;color:#64748b;text-transform:uppercase;
              letter-spacing:0.06em;margin-bottom:6px;">📍 Crowd Forecast</div>
  <div style="font-size:0.88rem;color:#475569;">{cond['crowd']}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)

    # Row 3 — Who Should Surf + Gear
    col_e, col_f = st.columns(2)
    with col_e:
        haz_color = cond["hazard_color"]
        haz_notes_html = "".join(
            f"<div style='margin-top:6px;font-size:0.82rem;color:#475569;'>⚠️ {n}</div>"
            for n in cond["hazard_notes"]
        ) or "<div style='font-size:0.82rem;color:#059669;margin-top:6px;'>No major hazards identified.</div>"
        st.markdown(f"""
<div style="background:white;border:1px solid #e2e8f0;border-radius:14px;
            padding:18px 20px;height:100%;box-shadow:0 1px 6px rgba(0,0,0,0.06);">
  <div style="font-size:0.78rem;font-weight:700;color:#64748b;text-transform:uppercase;
              letter-spacing:0.06em;margin-bottom:8px;">🏄 Who Should Surf This?</div>
  <div style="font-size:1.0rem;font-weight:700;color:#0f172a;margin-bottom:6px;">{cond['skill']}</div>
  <div style="font-size:0.88rem;color:#475569;line-height:1.6;">{cond['skill_detail']}</div>
  <div style="margin-top:12px;padding-top:10px;border-top:1px solid #f1f5f9;">
    <div style="font-size:0.78rem;font-weight:700;color:#64748b;text-transform:uppercase;
                letter-spacing:0.06em;margin-bottom:4px;">
      Hazard Level: <span style="color:{haz_color}">{cond['hazard']}</span>
    </div>
    {haz_notes_html}
  </div>
</div>""", unsafe_allow_html=True)

    with col_f:
        st.markdown(f"""
<div style="background:white;border:1px solid #e2e8f0;border-radius:14px;
            padding:18px 20px;height:100%;box-shadow:0 1px 6px rgba(0,0,0,0.06);">
  <div style="font-size:0.78rem;font-weight:700;color:#64748b;text-transform:uppercase;
              letter-spacing:0.06em;margin-bottom:10px;">🧥 What To Wear</div>
  <div style="font-size:1.0rem;font-weight:700;color:#0f172a;margin-bottom:8px;">
    {cond['wetsuit']}
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:6px;">
    <div style="background:#f8fafc;border-radius:8px;padding:10px 12px;">
      <div style="font-size:0.72rem;color:#94a3b8;font-weight:600;text-transform:uppercase;">Water Temp</div>
      <div style="font-size:1.05rem;font-weight:700;color:#0f172a;margin-top:2px;">
        {f"{water_t:.1f} °C" if water_t is not None else "—"}
      </div>
    </div>
    <div style="background:#f8fafc;border-radius:8px;padding:10px 12px;">
      <div style="font-size:0.72rem;color:#94a3b8;font-weight:600;text-transform:uppercase;">UV Index</div>
      <div style="font-size:1.05rem;font-weight:700;color:#0f172a;margin-top:2px;">
        {f"{uv_val:.0f} — {uv_label(uv_val)}" if uv_val is not None else "—"}
      </div>
    </div>
    <div style="background:#f8fafc;border-radius:8px;padding:10px 12px;grid-column:1/-1;">
      <div style="font-size:0.72rem;color:#94a3b8;font-weight:600;text-transform:uppercase;">Break Type</div>
      <div style="font-size:0.95rem;font-weight:600;color:#0f172a;margin-top:2px;">
        {spot['break_type']} Break — facing {spot['orientation']}°
      </div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    st.divider()

    # ── Go Surf Mode ─────────────────────────────────────────────────────────
    st.subheader("🚀 Go Surf Mode")
    top5 = best_window(df, spot)
    if not top5.empty:
        best       = top5.iloc[0]
        dist_to_spot = haversine_km(user_lat, user_lon, spot["lat"], spot["lon"])
        leave_t    = leave_by_time(best["time"], dist_to_spot)
        drive_mins = travel_time_mins(dist_to_spot)
        headline, cta = go_surf_msg(
            best["score"], spot_name, best["time"], leave_t, drive_mins
        )
        go_color = score_color(best["score"])

        st.markdown(f"""
<div style="background:linear-gradient(135deg,{go_color}18,{go_color}08);
            border:2px solid {go_color};border-radius:18px;
            padding:22px 26px;margin-bottom:16px;
            box-shadow:0 4px 20px {go_color}25;">
  <div style="font-size:1.35rem;font-weight:800;color:#0f172a;margin-bottom:10px;">
    {headline}
  </div>
  <div style="background:rgba(255,255,255,0.90);border-radius:12px;
              padding:16px 20px;font-size:1.0rem;color:#1e293b;line-height:1.7;">
    🏄 {cta}
  </div>
  <div style="display:flex;flex-wrap:wrap;gap:10px;margin-top:14px;">
    <span style="background:{go_color};color:white;padding:6px 16px;
                 border-radius:8px;font-size:0.88rem;font-weight:700;">
      🌊 Score {best['score']}/10 · {score_label(best['score'])}
    </span>
    <span style="background:#f1f5f9;color:#334155;padding:6px 16px;
                 border-radius:8px;font-size:0.88rem;font-weight:600;">
      📅 {best['time'].strftime('%A %d %b @ %H:%M')}
    </span>
    <span style="background:#f1f5f9;color:#334155;padding:6px 16px;
                 border-radius:8px;font-size:0.88rem;font-weight:600;">
      🚗 {drive_mins} min · {dist_to_spot:.1f} km
    </span>
    <span style="background:#f1f5f9;color:#334155;padding:6px 16px;
                 border-radius:8px;font-size:0.88rem;font-weight:600;">
      🏖️ {spot['break_type']} break
    </span>
  </div>
</div>""", unsafe_allow_html=True)

        with st.expander("📋 Top 5 windows in the next 48 hours"):
            disp = top5[["time", "wave_height", "wave_period", "windspeed_10m", "score"]].copy()
            disp.columns = ["Time", "Wave (m)", "Period (s)", "Wind (km/h)", "Score"]
            disp["Time"]     = disp["Time"].dt.strftime("%a %d %b  %H:%M")
            leave_times      = top5["time"].apply(lambda t: leave_by_time(t, dist_to_spot).strftime("%H:%M"))
            disp["Leave By"] = leave_times.values
            disp["Score"]    = disp["Score"].apply(lambda x: f"{x}  {score_label(x)}")
            st.dataframe(disp, use_container_width=True, hide_index=True)
    st.divider()

    # ── 7-Day forecast charts ────────────────────────────────────────────────
    st.subheader("📈 7-Day Forecast")
    fc1, fc2, fc3, fc4, fc5 = st.tabs([
        "Surf Score", "Wave Height & Period", "Wind", "Swell Direction", "Water & UV",
    ])

    chart_layout = dict(
        paper_bgcolor="#f0f4f8", plot_bgcolor="#ffffff",
        font=dict(color="#1e293b", size=13),
        hovermode="x unified",
        margin=dict(t=40, b=40, l=60, r=130),
        xaxis=dict(
            tickfont=dict(size=12, color="#1e293b"),
            title_font=dict(size=13, color="#1e293b"),
        ),
    )

    with fc1:
        fig = px.area(df.dropna(subset=["score"]), x="time", y="score",
                      color_discrete_sequence=["#0077b6"],
                      labels={"time": "", "score": "Surf Score"})
        fig.update_layout(
            yaxis=dict(range=[0, 10],
                       tickfont=dict(size=12, color="#1e293b"),
                       title_font=dict(size=13, color="#1e293b")),
            **chart_layout)
        fig.add_hrect(y0=7.5, y1=10, fillcolor="#059669", opacity=0.06, line_width=0)
        fig.add_hline(y=6.0, line_dash="dot",  line_color="#D97706", line_width=2,
                      annotation_text="Good (6)", annotation_font_color="#D97706",
                      annotation_font_size=13, annotation_position="right")
        fig.add_hline(y=7.5, line_dash="dash", line_color="#059669", line_width=2,
                      annotation_text="Very Good (7.5)", annotation_font_color="#059669",
                      annotation_font_size=13, annotation_position="right")
        fig.add_hline(y=9.0, line_dash="dash", line_color="#7C3AED", line_width=2,
                      annotation_text="Epic (9)", annotation_font_color="#7C3AED",
                      annotation_font_size=13, annotation_position="right")
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
            line=dict(color="#7c3aed", dash="dot")))
        fig2.update_layout(
            legend=dict(orientation="h", font=dict(size=13, color="#1e293b")),
            yaxis=dict(title="Height (m) | Period / 10 (s)",
                       tickfont=dict(size=12, color="#1e293b"),
                       title_font=dict(size=13, color="#1e293b")),
            **chart_layout)
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
            yaxis=dict(title="Wind Speed (km/h)",
                       tickfont=dict(size=12, color="#1e293b"),
                       title_font=dict(size=13, color="#1e293b")),
            yaxis2=dict(title="Wind Direction (°)", overlaying="y", side="right",
                        range=[0, 360],
                        tickfont=dict(size=12, color="#1e293b"),
                        title_font=dict(size=13, color="#1e293b")),
            legend=dict(orientation="h", font=dict(size=13, color="#1e293b")),
            **chart_layout)
        st.plotly_chart(fig3, use_container_width=True)

    with fc4:
        r_col = "swell_wave_height" if "swell_wave_height" in df.columns else "wave_height"
        fig4 = px.scatter_polar(
            df.dropna(subset=["wave_direction"]).head(168),
            r=r_col, theta="wave_direction", color="score",
            color_continuous_scale="Blues",
            labels={"wave_direction": "Swell Dir (°)", r_col: "Swell (m)", "score": "Score"})
        fig4.update_layout(
            polar=dict(
                angularaxis=dict(direction="clockwise", rotation=90,
                                 tickfont=dict(size=12, color="#1e293b")),
                radialaxis=dict(tickfont=dict(size=11, color="#1e293b")),
            ),
            paper_bgcolor="#f5f7fa",
            font=dict(color="#1e293b", size=13))
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
            yaxis=dict(title="Water Temp (°C)",
                       tickfont=dict(size=12, color="#1e293b"),
                       title_font=dict(size=13, color="#1e293b")),
            yaxis2=dict(title="UV Index", overlaying="y", side="right", range=[0, 14],
                        tickfont=dict(size=12, color="#1e293b"),
                        title_font=dict(size=13, color="#1e293b")),
            legend=dict(orientation="h", font=dict(size=13, color="#1e293b")),
            **chart_layout)
        st.plotly_chart(fig5, use_container_width=True)

    st.divider()

    # ── Region rankings ──────────────────────────────────────────────────────
    st.subheader(f"🗺️ {region_sel} Spot Rankings — Right Now")

    rank_rows = []
    for s in SURF_SPOTS[region_sel]:
        try:
            sdf  = get_forecast(s["lat"], s["lon"], user_tz)
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
        # Add drive time and leave-by from user location
        rank_df["Drive (min)"] = rank_df.apply(
            lambda r: travel_time_mins(haversine_km(user_lat, user_lon,
                next(s for s in SURF_SPOTS[region_sel] if s["name"] == r["Spot"])["lat"],
                next(s for s in SURF_SPOTS[region_sel] if s["name"] == r["Spot"])["lon"])),
            axis=1,
        )

        def _color_score(val):
            return f"color: {score_color(val)}; font-weight: bold"

        st.dataframe(
            rank_df.style.map(_color_score, subset=["Score"]),
            use_container_width=True,
        )

st.caption("🌊 Surf Buddy — 100% free. Data: Open-Meteo · GCCC beach cameras.")
