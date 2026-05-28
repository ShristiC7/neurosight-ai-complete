# NeuroSight AI — Getting Started

## Your Three-Part Guide

| Document | What It Covers | Read When |
|---|---|---|
| **[Part 1 — Configuration](GUIDE_PART1_CONFIGURATION.md)** | Secret keys, .env files, third-party services | Before running anything |
| **[Part 2 — Datasets & Training](GUIDE_PART2_DATASETS_AND_TRAINING.md)** | Where to get data, how to train each model, sample data formats | After configuration |
| **[Part 3 — Testing](GUIDE_PART3_TESTING.md)** | Unit tests, API tests, WebSocket tests, browser tests | After models are trained |

---

## Absolute Minimum to Get the Dashboard Running

If you just want to see the app working as fast as possible, do only these steps:

**Step 1 — Generate keys (2 minutes)**
```bash
python3 -c "import secrets; print('SECRET_KEY=' + secrets.token_hex(32))"
python3 -c "import secrets; print('JWT_SECRET_KEY=' + secrets.token_hex(32))"
```

**Step 2 — Create `backend/.env`**

Copy `.env.example` to `backend/.env` and fill in the two keys above plus a Postgres password.

**Step 3 — Create `frontend/.env.local`**
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
```

**Step 4 — Start with Docker**
```bash
npm run docker:up
```

**Step 5 — Run database migrations**
```bash
docker exec -it neurosight-ai-backend-1 alembic upgrade head
```

**Step 6 — Train the two auto-generating models (no data needed)**
```bash
docker exec -it neurosight-ai-backend-1 python \
  /app/../ml-models/behavioral-analytics/src/train.py

docker exec -it neurosight-ai-backend-1 python \
  /app/../ml-models/productivity-predictor/src/train.py
```

**Step 7 — Open the app**
```
http://localhost:3000
```

Register an account → the dashboard is live. Eye fatigue and voice stress will use
heuristic scoring until you download datasets and train those models (Part 2).

---

## Model Training Priority

```
Priority 1 (do now, no data needed):
  ✅ Behavioral Autoencoder + Isolation Forest
  ✅ Productivity LSTM + XGBoost
  ✅ RL Agent (pre-train with simulation)

Priority 2 (do when ready, needs downloads):
  📥 Eye Fatigue CNN+LSTM  ← Download MRL Dataset (~1GB)
  📥 Voice Stress Transformer ← Download RAVDESS (~500MB) + CREMA-D (~1GB)
```

---

## Quick Commands Reference

```bash
# Start everything
npm run docker:up

# Run backend locally
cd backend && uvicorn app.main:app --reload

# Run frontend locally
cd frontend && npm run dev

# Run backend tests
cd backend && pytest tests/ -v

# Run frontend tests
cd frontend && npm run test

# Train behavioral model (no data needed)
cd ml-models/behavioral-analytics/src && python train.py

# Train productivity model (no data needed)
cd ml-models/productivity-predictor/src && python train.py

# Pre-train RL agent (no data needed)
cd ml-models/rl-agent/src && python simulate.py

# View API docs (dev mode only)
open http://localhost:8000/api/docs

# View Grafana metrics
open http://localhost:3001   # admin / neurosight_grafana

# View Celery tasks
open http://localhost:5555   # admin / neurosight_flower

# Run database migrations
cd backend && alembic upgrade head

# Rollback last migration
cd backend && alembic downgrade -1
```

---

## Architecture at a Glance

```
Your Browser
    │
    ├─── http://localhost:3000  (Next.js frontend)
    │         │
    │         └─ Talks to ──► localhost:8000  (FastAPI backend)
    │                                │
    │                    ┌───────────┼───────────────┐
    │                    │           │               │
    │              localhost:5432  localhost:6379  localhost:6333
    │               (PostgreSQL)   (Redis)         (Qdrant)
    │
    └─── ws://localhost:8000/ws  (Real-time WebSocket)
```

---

## Support Checklist — If Something Doesn't Work

1. **Check Docker is running:** `docker info`
2. **Check all services are up:** `docker compose ps`
3. **Check backend logs:** `docker compose logs backend --tail=50`
4. **Check `.env` exists** in `backend/` with all required keys
5. **Check `.env.local` exists** in `frontend/`
6. **Check migrations ran:** backend logs should show "Database tables initialized"
7. **See Part 3 — Common Issues** for specific error messages

---

*NeuroSight AI — Multimodal Cognitive Intelligence Platform*
