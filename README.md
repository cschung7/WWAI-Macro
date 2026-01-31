# WWAI-Macro Landing Page

Unified landing page for the WWAI-Macro Global Macroeconomic Forecasting Platform, combining VAR and GNN models.

## Overview

This landing page serves as the entry point to two macroeconomic forecasting models:

| Model | Type | Key Features |
|-------|------|--------------|
| **VAR** | Vector Autoregression | Linear IRF, Regime Detection, Granger Causality |
| **GNN** | Graph Neural Network | Non-linear, 8-step Message Passing, 26 Countries |

## Quick Start

```bash
# Start all services
./start_all.sh

# Or start individually
npm run dev  # Starts on port 3801
```

## URLs

| Service | URL | Port |
|---------|-----|------|
| Landing Page | http://localhost:3801 | 3801 |
| GNN Dashboard | http://localhost:3789 | 3789 |
| GNN API | http://localhost:8005 | 8005 |
| VAR Dashboard | http://localhost:8012 | 8012 |

## Features

### Landing Page (`/`)
- Model comparison cards (VAR vs GNN)
- Quick scenario input with dual-model execution
- Bilingual support (English/Korean)
- Platform features overview

### Comparison Page (`/compare`)
- Side-by-side model results
- Run same scenario on both models
- Methodology comparison table
- Bilingual support

## Tech Stack

- **Framework**: Next.js 16
- **Styling**: Tailwind CSS + DaisyUI
- **Language**: TypeScript
- **i18n**: Built-in translations (EN/KO)

## Project Structure

```
wwai-macro-landing/
├── app/
│   ├── page.tsx           # Landing page
│   ├── compare/
│   │   └── page.tsx       # Model comparison
│   ├── translations.ts    # i18n strings
│   ├── layout.tsx         # Root layout
│   └── globals.css        # Global styles
├── package.json
├── start_all.sh           # Service startup script
└── README.md
```

## API Integration

### GNN API Endpoints
- `POST /api/gnn/generate-report` - Generate shock analysis report
- `POST /api/gnn/simulate` - Run shock simulation
- `GET /api/gnn/graph-structure` - Get country network

### VAR API Endpoints (Port 8012)
- `/api/var/simulate-fed-shock` - Fed rate shock simulation
- `/api/var/granger-causality` - Causality tests

## Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
npm start
```

## Related Projects

- `/mnt/nas/WWAI/WWAI-MACRO/WWAI-GNN` - GNN Model & API
- `/mnt/nas/WWAI/WWAI-MACRO/korea_macro_hrm` - VAR Model
- `/mnt/nas/WWAI/WWAI-MACRO/WWAI-GraphECast` - Original GraphECast

---

*WWAI-Macro Platform v1.0.0*
