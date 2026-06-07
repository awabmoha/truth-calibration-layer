# TCL-v0 Extended Validation Decision Note

Decision: `supports_continuing_tcl_v0`

Conservative TCL-v0 wins most primary criteria on most runs without high-confidence error regressions.

## Matrix Coverage

- Completed runs: 6
- Benchmark types: squad, triviaqa
- Models: /kaggle/input/models/google/gemma/transformers/2b-it/3, Qwen/Qwen2.5-0.5B-Instruct, microsoft/Phi-3.5-mini-instruct
- Model families: /kaggle/input/models/google/gemma/transformers/2b-it/3, microsoft/Phi-3.5-mini-instruct, qwen
- Matrix complete: True

## Run-Level Results

### freegpu-squad1000-Qwen_Qwen2.5-0.5B-Instruct-answermean

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Benchmark: `squad`
- Records/test: 1000 / 200
- Analysis dir: `tcl_experiments\runs\freegpu-squad1000-Qwen_Qwen2.5-0.5B-Instruct-answermean\analysis`
- Targeted manual review: `complete`
- Reviewed metrics applied: `True`
- Conservative TCL-v0 primary metrics: 2 wins, 2 losses, 0 ties vs raw
  - brier: raw=0.189956, conservative=0.195644, winner=raw
  - ece: raw=0.119559, conservative=0.15257, winner=raw
  - wrong_conf_ge_0_8: raw=30, conservative=10, winner=candidate
  - wrong_conf_ge_0_9: raw=11, conservative=2, winner=candidate

### freegpu-triviaqa1000-Qwen_Qwen2.5-0.5B-Instruct-answermean

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Benchmark: `triviaqa`
- Records/test: 1000 / 200
- Analysis dir: `tcl_experiments\runs\freegpu-triviaqa1000-Qwen_Qwen2.5-0.5B-Instruct-answermean\analysis`
- Targeted manual review: `complete`
- Reviewed metrics applied: `True`
- Conservative TCL-v0 primary metrics: 4 wins, 0 losses, 0 ties vs raw
  - brier: raw=0.255366, conservative=0.131989, winner=candidate
  - ece: raw=0.344028, conservative=0.0997398, winner=candidate
  - wrong_conf_ge_0_8: raw=9, conservative=1, winner=candidate
  - wrong_conf_ge_0_9: raw=3, conservative=0, winner=candidate

### freegpu-squad1000-microsoft_Phi-3.5-mini-instruct-answermean

- Model: `microsoft/Phi-3.5-mini-instruct`
- Benchmark: `squad`
- Records/test: 1000 / 200
- Analysis dir: `tcl_experiments\runs\freegpu-squad1000-microsoft_Phi-3.5-mini-instruct-answermean\analysis`
- Targeted manual review: `complete`
- Reviewed metrics applied: `True`
- Conservative TCL-v0 primary metrics: 2 wins, 2 losses, 0 ties vs raw
  - brier: raw=0.0293605, conservative=0.0464392, winner=raw
  - ece: raw=0.00921063, conservative=0.0404882, winner=raw
  - wrong_conf_ge_0_8: raw=5, conservative=4, winner=candidate
  - wrong_conf_ge_0_9: raw=3, conservative=2, winner=candidate

### freegpu-triviaqa1000-microsoft_Phi-3.5-mini-instruct-answermean

- Model: `microsoft/Phi-3.5-mini-instruct`
- Benchmark: `triviaqa`
- Records/test: 1000 / 200
- Analysis dir: `tcl_experiments\runs\freegpu-triviaqa1000-microsoft_Phi-3.5-mini-instruct-answermean\analysis`
- Targeted manual review: `complete`
- Reviewed metrics applied: `True`
- Conservative TCL-v0 primary metrics: 4 wins, 0 losses, 0 ties vs raw
  - brier: raw=0.278213, conservative=0.198426, winner=candidate
  - ece: raw=0.296857, conservative=0.152008, winner=candidate
  - wrong_conf_ge_0_8: raw=42, conservative=9, winner=candidate
  - wrong_conf_ge_0_9: raw=17, conservative=3, winner=candidate

### freegpu-squad1000-_kaggle_input_models_google_gemma_transformers_2b-it_3-answermean

- Model: `/kaggle/input/models/google/gemma/transformers/2b-it/3`
- Benchmark: `squad`
- Records/test: 1000 / 200
- Analysis dir: `tcl_experiments\runs\freegpu-squad1000-_kaggle_input_models_google_gemma_transformers_2b-it_3-answermean\analysis`
- Targeted manual review: `complete`
- Reviewed metrics applied: `True`
- Conservative TCL-v0 primary metrics: 4 wins, 0 losses, 0 ties vs raw
  - brier: raw=0.281062, conservative=0.13416, winner=candidate
  - ece: raw=0.238979, conservative=0.121125, winner=candidate
  - wrong_conf_ge_0_8: raw=62, conservative=10, winner=candidate
  - wrong_conf_ge_0_9: raw=33, conservative=5, winner=candidate

### freegpu-triviaqa1000-_kaggle_input_models_google_gemma_transformers_2b-it_3-answermean

- Model: `/kaggle/input/models/google/gemma/transformers/2b-it/3`
- Benchmark: `triviaqa`
- Records/test: 1000 / 200
- Analysis dir: `tcl_experiments\runs\freegpu-triviaqa1000-_kaggle_input_models_google_gemma_transformers_2b-it_3-answermean\analysis`
- Targeted manual review: `complete`
- Reviewed metrics applied: `True`
- Conservative TCL-v0 primary metrics: 4 wins, 0 losses, 0 ties vs raw
  - brier: raw=0.473393, conservative=0.197273, winner=candidate
  - ece: raw=0.546427, conservative=0.176408, winner=candidate
  - wrong_conf_ge_0_8: raw=76, conservative=10, winner=candidate
  - wrong_conf_ge_0_9: raw=19, conservative=2, winner=candidate

## Claim Boundary

This decision note only evaluates TCL-v0 confidence diagnostics. It does not validate the full Truth Calibration Layer theory, does not prove truthfulness, and does not show deployment readiness.
