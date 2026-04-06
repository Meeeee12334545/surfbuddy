# Surf Buddy 🏄

> "Know before you go. Surf smarter. It's on."

Surf Buddy is a real-time surf decision platform — your personal surf advisor that tells you exactly **where to surf**, **when to go**, and **whether it's worth it**.

**100% free for all users.** No premium tiers, no paywalls — every feature is available to everyone.

## Features

- **Live Surf Dashboard** — Auto-detect location, ranked nearby surf spots with Surf Score (0–10)
- **Surf Scoring Engine** — Real-time scores based on swell, wind, tide, and break orientation
- **"Go Surf" Mode** — One-tap recommendation for best spot, best time window, and when to leave
- **Surf Forecasting** — Hourly + 7-day forecasts with visual graphs for wind, swell, and tide
- **Surfcam Integration** — Aggregated live surf cams embedded per spot
- **Smart Notifications** — "It's ON" alerts, best window alerts, conditions improving, last chance, and daily summaries

## Stack

- **Frontend:** Next.js / React
- **Backend:** Python (FastAPI)
- **Database:** PostgreSQL (geo-enabled)
- **Queue:** Redis
- **Scraping:** Playwright

## Launch Regions

Phase 1: Gold Coast · Sunshine Coast · Northern NSW  
Phase 2: Sydney · Victoria · Western Australia

## Getting Started

```bash
# Frontend
cd frontend && npm install && npm run dev

# Backend
cd backend && pip install -r requirements.txt && uvicorn main:app --reload
```

Docker:

```bash
docker-compose up
```