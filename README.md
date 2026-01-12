# Project Genesis - ML Trading Platform

An AI-powered trading analysis platform that provides real-time market predictions using machine learning models.

## Tech Stack

### Frontend
| Technology | Purpose |
|------------|---------|
| **Next.js 14** | React framework with App Router |
| **TypeScript** | Type-safe development |
| **Tailwind CSS** | Styling |
| **Canvas API** | High-performance chart rendering |

**Deployment:** [Vercel](https://vercel.com)

### Backend (ML Server)
| Technology | Purpose |
|------------|---------|
| **Python 3.11** | Runtime |
| **Flask** | REST API framework |
| **Gunicorn** | WSGI HTTP server |
| **scikit-learn** | ML model training |
| **XGBoost** | Gradient boosting models |
| **CatBoost** | Categorical boosting models |
| **pandas/numpy** | Data processing |

**Deployment:** [Render](https://render.com)

### Data
| Service | Purpose |
|---------|---------|
| **Polygon.io** | Real-time & historical market data |
| **Git LFS** | Large model file storage (~160MB) |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PROJECT GENESIS                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   FRONTEND (Vercel)              ML BACKEND (Render)        │
│   ┌──────────────────┐          ┌──────────────────┐       │
│   │  Next.js App     │  ─────►  │  Flask ML Server │       │
│   │  - Charts        │   API    │  - V6 Models     │       │
│   │  - UI Components │  calls   │  - Predictions   │       │
│   │  - Real-time data│          │  - Analysis      │       │
│   └──────────────────┘          └──────────────────┘       │
│           │                              │                  │
│           ▼                              ▼                  │
│   ┌──────────────────┐          ┌──────────────────┐       │
│   │   Polygon.io     │          │   Git LFS        │       │
│   │   Market Data    │          │   Model Storage  │       │
│   └──────────────────┘          └──────────────────┘       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Production URLs

| Service | URL |
|---------|-----|
| **ML Backend** | `https://project-genesis-6roa.onrender.com` |
| **Health Check** | `https://project-genesis-6roa.onrender.com/health` |

## Environment Variables

### Vercel (Frontend)
```env
NEXT_PUBLIC_ML_SERVER_URL=https://project-genesis-6roa.onrender.com
ML_SERVER_URL=https://project-genesis-6roa.onrender.com
NEXT_PUBLIC_POLYGON_API_KEY=your_polygon_api_key
NEXT_PUBLIC_POLYGON_PLAN=advanced
```

### Render (ML Backend)
```env
POLYGON_API_KEY=your_polygon_api_key
PORT=10000
```

## ML Models

The platform uses V6 time-split models for predictions:

| Model | Tickers | Purpose |
|-------|---------|---------|
| **Intraday V6** | SPY, QQQ, IWM | Same-day directional predictions |
| **Swing V6.1** | SPY, QQQ, IWM | 5-day and 10-day predictions |
| **3-Day Swing** | SPY, QQQ, IWM | 3-day directional predictions |
| **1-Day Swing** | SPY, QQQ, IWM | Next-day predictions |

Models are stored using Git LFS (~160MB total).

## Local Development

### Frontend
```bash
npm install
npm run dev
```

### ML Server
```bash
cd ml
pip install -r requirements.txt
python -m server.app
```

## API Endpoints

### Health Check
```
GET /health
```

### Trading Directions
```
GET /trading_directions?ticker=SPY
```

### Northstar (Multi-timeframe)
```
GET /northstar?ticker=SPY
```

### MTF Analysis
```
GET /mtf?ticker=SPY
```

## Deployment

### Frontend (Vercel)
- Automatically deploys from `main` branch
- Connected to GitHub repository

### ML Backend (Render)
- Automatically deploys from `main` branch
- Root directory: `ml`
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn server.app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 2`
- Git LFS files are automatically pulled

## Project Structure

```
mvp-trading-app/
├── app/                    # Next.js App Router pages
│   ├── api/               # API routes
│   └── ticker/            # Dynamic ticker pages
├── src/
│   ├── components/        # React components
│   │   └── ProfessionalChart/  # Canvas chart system
│   └── services/          # API services
├── ml/                    # Python ML backend
│   ├── server/           # Flask application
│   │   ├── app.py       # Main entry point
│   │   ├── models/      # Model loading
│   │   └── routes/      # API endpoints
│   ├── v6_models/       # Trained ML models (Git LFS)
│   └── requirements.txt # Python dependencies
└── docs/                 # Documentation
```
