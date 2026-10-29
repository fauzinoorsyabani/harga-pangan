# Harga Pangan Indonesia — Food Price Intelligence Platform

This repository contains the **Indonesian Food Commodity Price Intelligence Platform** analyzing average prices across 11 staple commodities (2023–2025):

- `notebooks/01_eksplorasi_data.ipynb`: Data loading, cleaning, and exploratory analysis
- `notebooks/02_analisis_tren.ipynb`: Year-over-year trend analysis, inflation rate computation
- `notebooks/03_clustering_komoditas.ipynb`: K-Means commodity clustering and price segmentation
- `notebooks/04_forecasting_harga.ipynb`: Time series forecasting with Linear Regression and statistical models
- `app.py`: Flask/Streamlit interactive dashboard for commodity price analytics
- `src/data_loader.py`: Data ingestion and preprocessing pipeline
- `src/analyzer.py`: Statistical analysis and ML model training
- `src/visualizer.py`: Publication-grade visualization generator

## Agent skills

### Issue tracker

Issues and specs are tracked as local markdown files under `.scratch/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five canonical triage roles (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout (`GLOSSARY.md` at repo root and `docs/adr/`). See `docs/agents/domain.md`.

### Installed Skill Suites (`.agents/skills/` & `.agents/plugins/`)

1. **Matt Pocock Engineering & Productivity Skills** (`mattpocock/skills`):
   - Engineering: `ask-matt`, `code-review`, `codebase-design`, `diagnosing-bugs`, `domain-modeling`, `grill-with-docs`, `implement`, `implement-spec`, `improve-codebase-architecture`, `pr`, `prototype`, `research`, `retro`, `setup-matt-pocock-skills`, `tdd`, `to-spec`, `to-tickets`, `triage`, `wayfinder`, `wizard`
   - Productivity: `grill-me`, `grilling`, `handoff`, `teach`, `to-questionnaire`, `wait-what`, `writing-for-agents`
2. **REA — Reverse Engineer Anything** (`morluto/rea`):
   - Skill: `reverse-engineer-anything` in `.agents/skills/reverse-engineer-anything`
   - MCP Server: `rea` configured in `~/.gemini/config/mcp_config.json`
3. **The Designer Skills Pack** (`Owl-Listener/designer-skills`):
   - 9 Plugins & 111 Design Skills installed in `.agents/plugins/`, `.agents/skills/`, and `.gemini/extensions/`

