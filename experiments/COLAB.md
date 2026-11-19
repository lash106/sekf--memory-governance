# Real-LLM loop in Google Colab

The provider is already OpenAI-compatible (`SEKF_PROVIDER=openai`, `SEKF_LLM_BASE_URL`, `SEKF_LLM_MODEL`, `SEKF_LLM_API_KEY`). Point the base URL at Gemini, Groq, or OpenRouter. Do not commit the key.

| Backend | `SEKF_LLM_BASE_URL` |
|---|---|
| Gemini OpenAI compatibility | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| Groq | `https://api.groq.com/openai/v1` |
| OpenRouter | `https://openrouter.ai/api/v1` |

Groq does not return `logprobs`, and a requested temperature of 0 may be sent as a tiny positive number by their gateway. Record the temperature the call actually used.

## Cells

1. Setup

```python
!git clone https://github.com/lash106/sekf--memory-governance.git
%cd sekf--memory-governance
!pip install -e .
```

2. Secrets. `google.colab.userdata` keeps the key out of the notebook.

```python
import os
from google.colab import userdata
os.environ["SEKF_LLM_API_KEY"] = userdata.get("SEKF_LLM_API_KEY")
os.environ["SEKF_PROVIDER"] = "openai"
os.environ["SEKF_LLM_BASE_URL"] = "https://generativelanguage.googleapis.com/v1beta/openai/"
os.environ["SEKF_LLM_MODEL"] = "gemini-2.0-flash"
os.environ["SEKF_HISTORIAN"] = "llm"
os.environ["SEKF_SYNTHESIZER"] = "llm"
os.environ["SEKF_MEMORY"] = "sql"
os.environ["SEKF_POLICY"] = "v1"  # set v2 only when you mean to
```

3. Smoke. `scripts/smoke_llm.py` refuses to run when the provider is fake or the key is missing, and it caps the number of calls.

```python
!python scripts/smoke_llm.py --max-calls 5
```

4. Threshold sweep with no extra model calls. Export cached `PolicyContext` rows to JSONL (`context` plus `gold_action`), then:

```python
!python scripts/sweep_policy_thresholds.py cached_contexts.jsonl
```

That sweep uses policy-v1. policy-v2 replaces support, refute, and source reliability from the risk tier, so sweeping those three on v2 would not change them.

5. Offline score and calibration. This does not call a model:

```python
!python scripts/run_offline_experiment.py --out experiments/offline_dev.json --limit 20
```

```python
from sekf.judge.calibration import fit_temperature, apply_temperature
temperature = fit_temperature(confidences, correct)
print(temperature, apply_temperature(0.91, temperature))
```

Misclassified rows in `experiments/offline_dev.json` under `policy_v2_per_case` are the export for a later fine-tune. This repository does not download or train a local model.
