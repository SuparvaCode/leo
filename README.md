# Leo

**An open-weight decision model.** Send a *state* (text or JSON) and typed questions (`choice`, `score`, `noul` yes/no); get a calibrated probability distribution for every question from one forward pass, with no text generation.

- **Weights:** [huggingface.co/Suparva/leo-1.7b](https://huggingface.co/Suparva/leo-1.7b) (Apache-2.0, Qwen3-1.7B base + LoRA)
- **Benchmarks:** [docs/RESULTS.md](docs/RESULTS.md) (every table generated from `results/`)
- **Design notes:** [docs/PLAN.md](docs/PLAN.md)

Leo speaks the `POST /v1/systemone` wire format that TypeSafe documents for its Jev models, so existing clients work after a base-URL change. It is an independent project: not affiliated with TypeSafe, none of its weights, and **no Jev outputs were used for training**. Jev was called live only to score it on identical requests.

## Quick start

```bash
pip install torch transformers peft safetensors numpy pydantic huggingface_hub
```

```python
import sys
from huggingface_hub import snapshot_download

path = snapshot_download("Suparva/leo-1.7b")
sys.path.insert(0, path)          # the model repo ships its own inference code
from leo.infer import Leo

leo = Leo.load(path, dtype="bf16")
print(leo.system_one(
    "I was charged twice this month, please refund one of them.",
    {"route": {"type": "choice", "instructions": "Which team handles this?",
               "criteria": {"billing": "charges, refunds", "technical": "bugs, outages", "other": None}},
     "urgent": {"type": "noul", "instructions": "Does the customer need an answer today?"}},
))
```

Or install this repo (`pip install -e .[serve]`) and run the HTTP server:

```bash
python -m leo.serve --model checkpoints/leo-1.7b-v4 --port 8000     # 127.0.0.1 only, no key
LEO_API_KEY=change-me python -m leo.serve --model <path> --host 0.0.0.0   # a key is required off loopback
```

## How it works

- A pretrained causal LM (Qwen3-1.7B-Base) run **prefill-only**, adapted with LoRA r16.
- The state is encoded once; every question is packed behind it under a **block-causal mask** with positions restarting after the state, so questions share the state but cannot see each other.
- Boundaries are **reserved marker embeddings** outside the vocabulary, so text in the state cannot forge them.
- A **listwise pointer head** scores each option's end marker against the question's decision marker.
- Trained with **proper scoring rules** (log loss + ranked probability score for `score`), then **temperature-calibrated** per question type and option-count bucket.

## Results (leo-1.7b-v4 vs live Jev 1.13, identical requests)

See [docs/RESULTS.md](docs/RESULTS.md) for the full tables, per-family and per-language breakdowns, and protocol. Short version: Leo matches Jev on the browser suite and on JevBench's easy and standard tiers, beats it on tweet topics, and is clearly behind on hard reasoning, knowledge-heavy exams and low-resource languages. The model card lists the known flaws.

## Repository layout

| path | what |
|---|---|
| `leo/schema.py`, `leo/render.py` | request validation (TypeSafe limits, 422s), answers, confidence formulas, JSON-state rendering |
| `leo/encode.py`, `leo/model.py` | markers, block-causal packing, pointer head |
| `leo/train.py`, `leo/calibrate.py`, `leo/select.py`, `leo/checkpoint.py` | training (resumable, data-parallel, `--init` warm start), calibration, checkpoint selection |
| `leo/infer.py`, `leo/serve.py`, `leo/client.py` | inference, HTTP server, client |
| `leo/data/` | dataset converters and code-labelled families (browser simulators, reasoning, multilingual, Mind2Web) |
| `leo/bench/` | held-out classification, JevBench, multilingual, probes, browser (jev-ultrafast) benchmarks |
| `scripts/` | data builds, Kaggle training, false-DONE probe, HF export |
| `results/` | benchmark outputs for Leo (Jev per-item responses are not published, only aggregates) |
| `tests/` | 56 unit tests (schema, confidence maths, mask isolation, packing parity, resume, data families) |

## Reproduce

Windows PowerShell shown; Linux is the same with `/` paths.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip install -e ".[data,serve,dev]"
.\.venv\Scripts\python.exe -m pytest -q

# data (v3 mixture, then the v4 continuation set)
python -m leo.data.mind2web                        # converts Mind2Web train steps once
python -m leo.data.build --out data/processed-v3 --recipe v3
python scripts/build_v4.py                         # 30k v3 replay + 9k evidence-required browser steps

# training on Kaggle's free 2x T4 (set KAGGLE_USERNAME; datasets and kernels are private)
python scripts/kaggle/kaggle_run.py bundle --data data/processed-v3
python scripts/kaggle/kaggle_run.py launch --kernel leo-train-a --run leo-1.7b-v3 --data processed-v3 --extra "--epochs 2 ..."
# or locally: python -m leo.train --data data/processed-v3 --base Qwen/Qwen3-1.7B-Base --out checkpoints/leo-1.7b-v3 --resume

# benchmarks
powershell -File scripts\bench_all.ps1 -Run checkpoints\leo-1.7b-v4 -Name leo-1.7b-v4 -Data data/processed-v4 -Multilingual -Browser
python scripts/done_probe.py --leo checkpoints/leo-1.7b-v4 --name leo-1.7b-v4
```

Benchmarks that call Jev need `TYPESAFE_API_KEY` in a git-ignored `.env` and your own TypeSafe account; responses are cached on disk so nothing is paid for twice. The browser benchmark needs [jev-ultrafast](https://github.com/browser-use/jev-ultrafast) cloned into `external/jev-ultrafast` and installed (`pip install -e external/jev-ultrafast`), plus Chrome.

## Licence

Code and weights: Apache-2.0 ([LICENSE](LICENSE)). Training datasets keep their own licences (listed in the model card); some are CC BY-SA or custom.
