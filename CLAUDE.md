# CLAUDE.md — AIOps Incident Cockpit (Premonition Detection Demo)

## Project Overview

Multi-site network fault detection and remediation system combining Root Cause Analysis (LogicalRCA), Digital Twin predictive failure detection, and AI-powered remediation planning. Predictive failures (marked with a crystal ball icon) are surfaced alongside real alarms — no UI changes required.

**Primary language:** Python 3.9+
**Frontend:** Streamlit (>=1.28.0)
**Documentation language:** Japanese (README, UI labels, comments)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

System package `graphviz` is required (listed in `packages.txt` for Streamlit Cloud).

## Project Structure

```
├── app.py                      # Streamlit entry point
├── inference_engine.py         # LogicalRCA v5.1 — cascade suppression, silent detection
├── digital_twin.py             # Digital Twin v3.1 — prediction bridge
├── alarm_generator.py          # Scenario-based alarm generation (18 scenarios)
├── network_ops.py              # LLM integration, remediation commands
├── registry.py                 # Multi-site topology manager
├── verifier.py                 # Log verification & validation
├── rate_limiter.py             # API rate limiting
├── mock_data_gen.py            # Mock data generation
├── streamlit_cache.py          # Caching wrapper
├── digital_twin_pkg/           # Digital Twin engine package (V45)
│   ├── engine.py               # Core prediction API (largest module)
│   ├── rules.py                # Escalation rule definitions (21+ rules)
│   ├── config.py               # Thresholds, retention, weights
│   ├── storage.py              # Persistence layer
│   ├── vector_store.py         # ChromaDB vector embeddings
│   ├── llm_client.py           # LLM integration
│   ├── llm_local.py            # Local LLM fallback
│   ├── bayesian.py             # Bayesian inference
│   ├── tuning.py               # Auto-tuning engine
│   ├── gnn.py                  # Graph neural network
│   └── audit.py                # Audit logging
├── ui/                         # Streamlit UI components
│   ├── cockpit.py              # Main incident cockpit (largest file)
│   ├── sidebar.py              # Scenario selection
│   ├── tuning.py               # Digital Twin tuning dashboard
│   ├── dashboard.py            # Site status board
│   ├── explanation_panel.py    # Explanation rendering
│   └── graph.py                # Topology visualization
├── utils/                      # Shared utilities
│   ├── state.py                # Session state management
│   ├── const.py                # Constants, scenario mappings
│   ├── helpers.py              # Helper functions
│   └── llm_helper.py           # LLM helper functions
├── topologies/                 # Site topology JSON definitions
├── configs/                    # Device configuration samples
└── tests/                      # Test suites
```

## Architecture

```
UI Layer (Streamlit)
  └─ cockpit.py, sidebar.py, tuning.py, dashboard.py
        │
Application Layer
  └─ inference_engine.py (LogicalRCA)
  └─ network_ops.py (LLM ops, remediation)
        │
Digital Twin Layer
  └─ digital_twin_pkg/engine.py
     ├─ BFS propagation (NetworkX)
     ├─ Embedding matching (sentence-transformers, all-MiniLM-L6-v2)
     ├─ Confidence scoring (Bayesian)
     └─ Escalation rules (keyword + semantic)
        │
Data & Storage Layer
  └─ storage.py, vector_store.py, registry.py
```

**Data flow:** Alarms → LogicalRCA.analyze() → cascade suppression + silent detection → Digital Twin predictions (rule matching → confidence scoring → BFS propagation → SPOF/HA adjustments) → display candidates → LLM report/remediation generation.

## Key Technologies

| Component | Technology |
|-----------|-----------|
| Web UI | Streamlit |
| Graph/topology | NetworkX (>=3.0) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector DB | ChromaDB (>=0.5.0) |
| LLM | Google Generative AI (Gemini/Gemma) |
| Device CLI | netmiko (>=4.0.0) |
| Data | pandas (>=2.0.0), NumPy (>=1.24.0) |
| Visualization | graphviz |
| ML backend | PyTorch (CPU-only — required for Streamlit Cloud 1GB limit) |

## Running Tests

Tests use Python's built-in `unittest`. No pytest configuration.

```bash
# Integration tests (inference_engine + digital_twin)
python tests/test_integration_v2.py

# Display logic tests (D1: status, D2: KPI)
python tests/test_d1_d2.py

# Full pipeline regression
python tests/test_full_regression.py

# Digital Twin unit tests
python tests/test_digital_twin_v2.py

# Run all tests
python -m unittest discover -s tests -v
```

## Linting & Formatting

No linting or formatting tools are configured. Code follows standard Python conventions with UTF-8 encoding declarations.

## Deployment

1. **Streamlit Community Cloud** — deploy directly from GitHub; set API keys via Secrets panel
2. **Airgap/Offline** — `deploy_airgap.sh` (3-phase: download → package → install with SHA-256 verification)
3. **Local** — `streamlit run app.py`

## Key Design Patterns & Conventions

- **Singleton caching** for ML models: `DigitalTwinEngine._model_loaded` prevents reloading across Streamlit reruns. Do not instantiate multiple engine instances.
- **Session state** (`st.session_state`) is the primary UI state store. See `utils/state.py`.
- **Topology as JSON**: Network topologies live in `topologies/*.json` and `topology.json`. Use `registry.py` to load/manage multi-site configs.
- **Hybrid matching**: Escalation rules use both keyword patterns (fast) and semantic embeddings (flexible). Thresholds: 0.40 embedding similarity, 0.50 final confidence.
- **No hardcoded API keys**: Use Streamlit Secrets (`st.secrets`) or environment variables.
- **CPU-only PyTorch**: The `requirements.txt` pins `--extra-index-url` to CPU wheels. Do not add GPU PyTorch — it exceeds the Streamlit Cloud memory limit.

## Important Thresholds (digital_twin_pkg/config.py)

- Minimum prediction confidence: 0.40
- History retention: 90 days
- Counter retention: 180 days
- BFS hop limit: 3 (default)
- SPOF confidence boost: +10%
- HA redundancy discount: -15%

## Supported Scenarios

18 alarm scenarios including: normal operation, WAN total outage, firewall failures, L2 switch silent failures, BGP flapping, power/FAN/memory issues, and complex multi-device failures. Scenarios are defined in `alarm_generator.py` and mapped in `utils/const.py`.

## Common Pitfalls

- **Model cold start**: First load of all-MiniLM-L6-v2 takes 5-10 seconds. Subsequent calls use the cached model (<0.1ms).
- **Memory**: Keep PyTorch CPU-only. The full stack (Streamlit + transformers + ChromaDB) must fit in 1GB for cloud deployment.
- **Topology changes**: When modifying `topologies/*.json`, ensure parent-child relationships and redundancy groups remain consistent. The BFS propagation and SPOF detection depend on correct topology structure.
- **LLM fallback**: If Google GenAI is unavailable, the system falls back to `digital_twin_pkg/llm_local.py`. Ensure fallback paths are tested.
- **Streamlit reruns**: Every widget interaction triggers a full script rerun. Use `st.session_state` and `@st.cache_resource` / `@st.cache_data` to avoid expensive recomputation.
