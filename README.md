# PCB Reflow Thermal Digital Twin
## Complete Project Structure

```
reflow_project/
│
├── inference_engine.py          ← (EXISTING) Model loading + ReflowProfile + predict()
├── reflow_advisor.py            ← (EXISTING) SAC305 rule validation + risk scoring
├── optimizer.py                 ← (EXISTING) GA + NSGA-II optimization
├── visualizer.py                ← (EXISTING) Publication figures
├── main.py                      ← (EXISTING) Stage 2 full pipeline runner
│
├── experimental_pipeline.py     ← (NEW) Stage 3: Exp → Optimize → Compare (CLI script)
│
├── app.py                       ← (NEW) Streamlit UI entry point
├── requirements.txt             ← (NEW) All dependencies
│
└── app_pages/
    ├── __init__.py
    ├── setup_tab.py             ← (NEW) Tab 1: uploads + profile input
    ├── run_tab.py               ← (NEW) Tab 2: run button + progress
    └── results_tab.py           ← (NEW) Tab 3: all figures + downloads
```

## How to run

### CLI (no UI):
```bash
python experimental_pipeline.py \
  --model  "saved_models_stage1/thermal_digital_twin (1).pth" \
  --pcb    "data/C1_PCB_Tmap_50x50.csv" \
  --die    "data/C1_DIE_Tmap_50x50.csv" \
  --train  "data/TRAINING_DATA.xlsx"
```

### Streamlit UI:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## What each existing file does (unchanged)
| File | Role |
|---|---|
| `inference_engine.py` | `InferenceEngine` class — loads `.pth`, runs `predict(ReflowProfile)` → `PredictionResult` (50×50 maps) |
| `reflow_advisor.py` | `ReflowAdvisor` — SAC305 rule checks, composite score, `evaluate(profile)` → `AdvisorReport` |
| `optimizer.py` | `ReflowOptimizer` — `run_ga()` + `run_nsga2()` → best profile + Pareto front |
| `visualizer.py` | `Visualizer` — `plot_thermal_comparison()`, `plot_reflow_profile()`, `plot_pareto_front()`, `plot_dashboard()` |
| `main.py` | Wires everything together for Stage 2 (baseline → optimize → figures) |

## What the new files add
| File | Role |
|---|---|
| `experimental_pipeline.py` | Stage 3: feed your experimental profile + FEA CSVs → optimize → Fig6 contours → % ΔT comparison |
| `app.py` + `app_pages/` | Streamlit UI wrapping `experimental_pipeline.py` logic with file upload + interactive parameters |
