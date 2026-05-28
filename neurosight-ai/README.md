# 🧠 NeuroSight AI

**Multimodal AI-Powered Cognitive Fatigue & Productivity Intelligence Platform**

> Real-time detection of fatigue, stress, and burnout using computer vision, behavioral analytics, voice intelligence, and adaptive reinforcement learning.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend Layer                          │
│         Next.js 15 + React 19 + Tailwind v4                │
│    Real-time Dashboard · WebSocket Client · MediaPipe      │
└──────────────────────────┬──────────────────────────────────┘
                           │ REST + WebSocket
┌──────────────────────────▼──────────────────────────────────┐
│                     API Gateway                             │
│              FastAPI + JWT Auth + Rate Limiting             │
└──────┬─────────────┬──────────────┬───────────┬────────────┘
       │             │              │           │
  ┌────▼────┐  ┌─────▼────┐  ┌────▼───┐  ┌───▼──────────┐
  │   CV    │  │  Audio   │  │Behav.  │  │Recommendation│
  │ Service │  │ Service  │  │Service │  │   Engine RL  │
  └────┬────┘  └─────┬────┘  └────┬───┘  └───┬──────────┘
       │             │              │           │
┌──────▼─────────────▼──────────────▼───────────▼────────────┐
│                   ML Inference Layer                        │
│  Eye Fatigue CNN+LSTM · Voice Stress Transformer · XGBoost │
│         DQN RL Agent · Isolation Forest · Autoencoder      │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                     Data Layer                              │
│         PostgreSQL 17 · Redis 7 · Qdrant Vector DB         │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    MLOps / DevOps                           │
│         Docker · Kubernetes · GitHub Actions CI/CD          │
│              Prometheus · Grafana · Celery                  │
└─────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Layer | Technologies |
|---|---|
| **Frontend** | Next.js 15, React 19, TypeScript 5.8, Tailwind v4, Framer Motion, Recharts |
| **State** | Zustand 5, TanStack Query 5, Immer |
| **Backend** | FastAPI 0.115, Python 3.12, Uvicorn, Pydantic v2 |
| **ML/AI** | PyTorch 2.7, ONNX Runtime, MediaPipe, OpenCV, Librosa |
| **Models** | CNN+LSTM (fatigue), Transformer (voice), XGBoost (productivity), DQN (recommendations) |
| **Database** | PostgreSQL 17, Redis 7, Qdrant 1.14 |
| **Queue** | Celery 5.5, Redis broker |
| **DevOps** | Docker, Kubernetes, GitHub Actions, Prometheus, Grafana |

---

## Project Structure

```
neurosight-ai/
├── frontend/                    # Next.js 15 App Router
│   └── src/
│       ├── app/                 # Pages (App Router)
│       ├── components/          # UI components
│       │   ├── dashboard/       # Dashboard layout, KPI panels
│       │   ├── charts/          # Timeline, heatmap, gauge
│       │   ├── fatigue/         # Eye fatigue panel
│       │   ├── audio/           # Voice stress panel
│       │   └── behavioral/      # Behavioral analytics panel
│       ├── hooks/               # Custom React hooks
│       │   ├── use-fatigue-detection.ts  # MediaPipe webcam hook
│       │   ├── use-voice-stress.ts       # Web Audio API hook
│       │   ├── use-behavioral-analytics.ts
│       │   └── use-websocket.ts          # WS context + hooks
│       ├── store/               # Zustand stores
│       ├── lib/                 # API client, utilities
│       └── types/               # TypeScript types
│
├── backend/                     # FastAPI application
│   └── app/
│       ├── api/v1/endpoints/    # REST + WS endpoints
│       ├── core/                # Config, security, Redis, logging
│       ├── db/                  # SQLAlchemy models, migrations
│       ├── middleware/          # Rate limiter, metrics, request ID
│       ├── schemas/             # Pydantic schemas
│       ├── services/            # Business logic, ML integration
│       └── tasks/               # Celery background tasks
│
├── ml-models/                   # ML model code
│   ├── eye-fatigue/src/
│   │   └── model.py             # CNN + LSTM fatigue classifier
│   ├── voice-stress/src/
│   │   └── model.py             # CNN + Transformer stress model
│   ├── behavioral-analytics/src/
│   │   └── train.py             # Isolation Forest + Autoencoder
│   ├── productivity-predictor/src/
│   │   └── train.py             # LSTM + XGBoost ensemble
│   └── rl-agent/src/
│       └── agent.py             # Dueling DQN with PER
│
├── infrastructure/
│   ├── docker/
│   │   ├── docker-compose.yml   # Full stack local environment
│   │   ├── Dockerfile.backend   # Multi-stage Python image
│   │   └── Dockerfile.frontend  # Multi-stage Node image
│   ├── kubernetes/              # K8s deployments, HPA, services
│   └── monitoring/              # Prometheus + Grafana configs
│
└── .github/workflows/
    └── ci-cd.yml                # Full CI/CD pipeline
```

---

## Quick Start

### Prerequisites
- Node.js 22+, npm 10+
- Python 3.12+
- Docker + Docker Compose
- Git

### 1. Clone & Configure
```bash
git clone https://github.com/your-org/neurosight-ai.git
cd neurosight-ai
cp .env.example backend/.env
# Edit backend/.env with your values
```

### 2. Start with Docker (recommended)
```bash
npm run docker:up
# Services: Frontend :3000, Backend :8000, Grafana :3001
```

### 3. Local Development
```bash
# Terminal 1 — Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend
npm install
npm run dev

# Terminal 3 — Celery Worker
cd backend
celery -A app.core.celery_app worker --loglevel=info
```

### 4. Database Setup
```bash
cd backend
alembic upgrade head
```

---

## ML Model Training

```bash
# Train behavioral analytics models (Isolation Forest + Autoencoder)
cd ml-models/behavioral-analytics/src
python train.py

# Train productivity forecasting (LSTM + XGBoost)
cd ml-models/productivity-predictor/src
python train.py

# RL agent trains online from user feedback
# Initial weights are random — it learns from your data
```

---

## API Reference

Interactive docs available at `http://localhost:8000/api/docs` (dev only).

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/auth/register` | POST | Register user |
| `/api/v1/auth/login` | POST | Authenticate |
| `/api/v1/auth/refresh` | POST | Refresh JWT |
| `/api/v1/fatigue/metrics` | POST | Store fatigue data |
| `/api/v1/fatigue/analyze-frame` | POST | Server-side frame analysis |
| `/api/v1/audio/analyze` | POST | Voice stress analysis |
| `/api/v1/behavioral/metrics` | POST | Store behavioral data |
| `/api/v1/predictions/` | GET | Get productivity prediction |
| `/api/v1/recommendations/` | GET | Get AI recommendations |
| `/api/v1/ws` | WebSocket | Real-time bidirectional stream |

---

## ML Architecture Details

### Eye Fatigue Detection (CNN + LSTM)
- **Input**: 48×48 grayscale eye crops + EAR/MAR time series (30 frames)
- **CNN Branch**: Depthwise-separable convolutions → 128-dim embedding
- **LSTM Branch**: Bidirectional LSTM + attention → 128-dim embedding
- **Output**: 5-class drowsiness + continuous 0–100 fatigue score
- **Loss**: Focal Loss (handles class imbalance for rare critical events)

### Voice Stress Model (CNN + Transformer)
- **Input**: 128-bin mel spectrogram + 39-dim MFCC statistics
- **CNN**: 2D spectrogram encoder → temporal sequence
- **Transformer**: 4-layer encoder with sinusoidal positional encoding
- **Output**: 5-class emotion + continuous stress score
- **Training data**: RAVDESS, CREMA-D, custom samples

### Productivity Forecasting (LSTM + XGBoost Ensemble)
- **Input**: 15-dimensional feature vector × 30 time steps
- **LSTM**: Bidirectional with multi-head attention pooling
- **XGBoost**: Feature-based snapshot model (last time step)
- **Ensemble**: 60% XGBoost + 40% heuristic
- **Output**: Productivity score (0–100) + burnout probability (0–1)

### RL Recommendation Agent (Dueling DQN)
- **State space**: 12-dimensional (fatigue, stress, productivity, time features)
- **Action space**: 10 recommendation types
- **Algorithm**: Dueling DQN + Double DQN + Prioritized Experience Replay
- **Reward**: Accepted recommendations that improve subsequent metrics
- **Cooldowns**: Per-action minimum intervals to prevent spam

---

## Deployment

### Kubernetes
```bash
kubectl create namespace neurosight
kubectl apply -f infrastructure/kubernetes/
```

### CI/CD Pipeline
Push to `main` → Tests → Docker build → Push to GHCR → Deploy to staging → Smoke tests

---

## Success Metrics

| Metric | Target |
|---|---|
| Fatigue Detection Accuracy | >92% |
| Voice Stress F1-Score | >88% |
| Productivity Prediction RMSE | <10 points |
| Recommendation Acceptance Rate | >70% |
| Real-Time Inference Latency | <300ms |
| Burnout Prediction Precision | >85% |

---

## License

MIT © NeuroSight AI
