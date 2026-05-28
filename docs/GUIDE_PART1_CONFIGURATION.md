# NeuroSight AI — Next Steps Guide

## Part 1 of 3: Configuration & Required Keys

> **Read this first before running anything.**
> Every section below is something you must action before the platform will work.
> Items marked 🔴 are **blockers** — the app won't start without them.
> Items marked 🟡 are **important** — features will silently fail without them.
> Items marked 🟢 are **optional** — only needed for production or advanced features.

---

## 1.1 Generate Your Secret Keys

These are cryptographic keys you generate yourself — no sign-up needed.

### 🔴 SECRET_KEY (App encryption key)

Used to sign session cookies and internal tokens.

```bash
# Run this in your terminal — copy the output
python3 -c "import secrets; print(secrets.token_hex(32))"
```

Example output:
```
a3f8e2d1c4b5a6f7e8d9c0b1a2f3e4d5c6b7a8f9e0d1c2b3a4f5e6d7c8b9a0
```

Paste that into your `backend/.env`:
```env
SECRET_KEY=a3f8e2d1c4b5a6f7e8d9c0b1a2f3e4d5c6b7a8f9e0d1c2b3a4f5e6d7c8b9a0
```

---

### 🔴 JWT_SECRET_KEY (Token signing key)

Used to sign and verify all JWT access tokens. Must be different from SECRET_KEY.

```bash
# Generate a separate key
python3 -c "import secrets; print(secrets.token_hex(32))"
```

```env
JWT_SECRET_KEY=<paste-second-generated-value-here>
```

> ⚠️ **Never reuse SECRET_KEY and JWT_SECRET_KEY.** If one leaks, the other stays safe.

---

### 🔴 POSTGRES_PASSWORD

Pick any strong password for your local database. No restrictions on format.

```bash
# Suggestion
python3 -c "import secrets; print(secrets.token_urlsafe(20))"
```

```env
POSTGRES_USER=neurosight
POSTGRES_PASSWORD=<your-generated-password>
POSTGRES_DB=neurosight
```

> ⚠️ This same password must match in both `backend/.env` and `infrastructure/docker/docker-compose.yml` under the `postgres` service.

---

## 1.2 Your Complete `backend/.env` File

Create this file at `neurosight-ai/backend/.env`. Fill in each value:

```env
# ─── Core ─────────────────────────────────────────────────
ENVIRONMENT=development
DEBUG=true
SECRET_KEY=<generated-above>
API_PREFIX=/api/v1
HOST=0.0.0.0
PORT=8000

# ─── Database ─────────────────────────────────────────────
POSTGRES_HOST=localhost          # Use "postgres" when running in Docker
POSTGRES_PORT=5432
POSTGRES_DB=neurosight
POSTGRES_USER=neurosight
POSTGRES_PASSWORD=<generated-above>

# ─── Redis ────────────────────────────────────────────────
REDIS_HOST=localhost             # Use "redis" when running in Docker
REDIS_PORT=6379
REDIS_DB=0

# ─── Qdrant ───────────────────────────────────────────────
QDRANT_HOST=localhost            # Use "qdrant" when running in Docker
QDRANT_PORT=6333

# ─── JWT ──────────────────────────────────────────────────
JWT_SECRET_KEY=<second-generated-key>
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60
JWT_REFRESH_TOKEN_EXPIRE_DAYS=30

# ─── Celery ───────────────────────────────────────────────
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2

# ─── ML Models ────────────────────────────────────────────
MODEL_DIR=../ml-models
EYE_FATIGUE_MODEL_PATH=../ml-models/eye-fatigue/model.onnx
VOICE_STRESS_MODEL_PATH=../ml-models/voice-stress/model.onnx
PRODUCTIVITY_MODEL_PATH=../ml-models/productivity-predictor/src/xgboost.json
RL_AGENT_PATH=../ml-models/rl-agent/agent.zip
INFERENCE_DEVICE=cpu

# ─── CORS ─────────────────────────────────────────────────
ALLOWED_ORIGINS=["http://localhost:3000"]

# ─── Feature Flags ────────────────────────────────────────
ENABLE_VOICE_ANALYSIS=true
ENABLE_RL_RECOMMENDATIONS=true
ENABLE_EDGE_INFERENCE=false
ENABLE_EXPLAINABLE_AI=true

# ─── Monitoring ───────────────────────────────────────────
PROMETHEUS_ENABLED=true
```

> 📝 **Docker vs local:** When you run with `docker compose`, change the hostnames:
> - `POSTGRES_HOST=localhost` → `POSTGRES_HOST=postgres`
> - `REDIS_HOST=localhost` → `REDIS_HOST=redis`
> - `QDRANT_HOST=localhost` → `QDRANT_HOST=qdrant`
> - Celery URLs: `redis://localhost` → `redis://redis`

---

## 1.3 Your Complete `frontend/.env.local` File

Create this file at `neurosight-ai/frontend/.env.local`:

```env
# Points to your FastAPI backend
NEXT_PUBLIC_API_URL=http://localhost:8000

# WebSocket endpoint
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws

# App name (optional, shows in browser tab)
NEXT_PUBLIC_APP_NAME=NeuroSight AI
```

---

## 1.4 Third-Party Services (Optional)

These are only needed if you want specific features. The app works without all of them.

---

### 🟡 Sentry — Error Monitoring

Catches backend crashes and frontend errors in production.

**Steps:**
1. Go to [sentry.io](https://sentry.io) → Create a free account
2. Create a new project → choose **Python** (for backend)
3. Copy the DSN from Project Settings → Client Keys

```env
# backend/.env
SENTRY_DSN=https://abc123@o000000.ingest.sentry.io/0000000
```

4. Create a second project → choose **Next.js** (for frontend)

```env
# frontend/.env.local
NEXT_PUBLIC_SENTRY_DSN=https://xyz789@o000000.ingest.sentry.io/1111111
```

> 💡 Free tier gives you 5,000 errors/month — more than enough for development.

---

### 🟢 Qdrant Cloud — Managed Vector Database

If you don't want to run Qdrant locally via Docker.

**Steps:**
1. Go to [cloud.qdrant.io](https://cloud.qdrant.io) → Create free account
2. Create a cluster (free tier: 1GB storage)
3. Copy your **Cluster URL** and **API Key**

```env
# backend/.env
QDRANT_HOST=abc-xyz.aws.cloud.qdrant.io
QDRANT_PORT=6333
QDRANT_API_KEY=your-qdrant-cloud-api-key-here
```

> 💡 The local Docker Qdrant at port 6333 works fine for development.
> Only switch to cloud when deploying to production.

---

### 🟢 AWS Credentials — Cloud Deployment

Only needed when deploying the Kubernetes cluster to AWS EKS.

**Steps:**
1. Log into [AWS Console](https://console.aws.amazon.com)
2. Go to IAM → Create a new user → Attach policies:
   - `AmazonEKSClusterPolicy`
   - `AmazonRDSFullAccess`
   - `AmazonS3FullAccess`
   - `ElasticLoadBalancingFullAccess`
3. Create access keys for that user

```bash
# Configure AWS CLI
aws configure
# Enter: Access Key ID, Secret Access Key, region (e.g. us-east-1), output (json)
```

```env
# backend/.env (only for production)
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
AWS_DEFAULT_REGION=us-east-1
```

> ⚠️ Never commit AWS credentials to git. Use IAM roles in production instead.

---

### 🟢 GitHub Container Registry — Docker Image Push

For the CI/CD pipeline to push built Docker images.

**Steps:**
1. Go to GitHub → Settings → Developer Settings → Personal Access Tokens → Tokens (classic)
2. Generate token with scopes: `write:packages`, `read:packages`, `delete:packages`
3. Add to GitHub repo → Settings → Secrets and Variables → Actions

| Secret Name | Value |
|---|---|
| `GHCR_TOKEN` | Your personal access token |
| `KUBE_CONFIG_STAGING` | Base64-encoded kubeconfig for staging cluster |
| `POSTGRES_PASSWORD` | Same password as your .env |

---

### 🟢 OpenTelemetry — Distributed Tracing

For production tracing across microservices.

**Steps:**
1. Sign up for [Honeycomb](https://honeycomb.io) (free tier) or [Jaeger](https://jaegertracing.io) (self-hosted)
2. Copy API key/endpoint

```env
# backend/.env
OTEL_EXPORTER_ENDPOINT=https://api.honeycomb.io
OTEL_EXPORTER_HEADERS=x-honeycomb-team=your-api-key
```

---

## 1.5 Docker Compose Password Sync

**Important:** The Postgres password in `infrastructure/docker/docker-compose.yml` must match your `.env`.

Open `infrastructure/docker/docker-compose.yml` and find the postgres service:

```yaml
postgres:
  environment:
    - POSTGRES_PASSWORD=neurosight_dev_password   # ← change this
```

Change `neurosight_dev_password` to whatever you set in `POSTGRES_PASSWORD` in your `.env`.

---

## 1.6 Environment Checklist

Run through this before starting the app for the first time:

```
□ backend/.env created
□ SECRET_KEY generated and pasted (≥32 chars)
□ JWT_SECRET_KEY generated and pasted (≥32 chars, different from SECRET_KEY)
□ POSTGRES_PASSWORD set (same in .env AND docker-compose.yml)
□ frontend/.env.local created
□ NEXT_PUBLIC_API_URL set to http://localhost:8000
□ NEXT_PUBLIC_WS_URL set to ws://localhost:8000/ws
□ Docker Desktop is running (if using Docker)
□ Python 3.12+ installed (if running locally)
□ Node.js 22+ installed (check: node --version)
```

Once all boxes are checked, proceed to Part 3 (Testing Guide).

---

*Next: Part 2 — Sample Datasets & Model Training →*
