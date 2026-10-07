# Jean

A small decision model based on **LiquidAI/LFM2.5-VL-3B**, with text and image inputs. This is our starting point; we'll grow it together.

There are four small modules:

- `model.py`: load Liquid, score the supplied choices in one forward pass, predict and save.
- `train.py`: plain supervised training and optional validation accuracy.
- `prepare.py`: fetch a tiny BoolQ sample from Hugging Face.
- `decision_index.py`: connect Jean to the official benchmark runner.

Jean keeps Liquid's standard architecture and vocabulary head. Choices use letter codes, binary decisions use yes/no, and ratings use digit levels. We read those answer-token logits and normalize over the available choices instead of generating text. All weights are trainable. For now, use one question per example.

## One dataset format

Each JSONL line uses the same fields:

```json
{"type":"choice","state":"Le client a été débité deux fois.","question":"Quelle équipe ?","choices":{"billing":"Facturation","support":"Assistance"},"target":"billing"}
```

`type` is required: explicitly choose `choice`, `noul`, or `score`:

| Type | `choices` keys | Output |
|---|---|---|
| `choice` | Any 2–1,728 IDs¹ | Winning ID and probabilities |
| `noul` | `yes` and `no` | `noul`: P(yes), plus probabilities |
| `score` | Ordered `0`, `1`, …, with 2–10 levels | `score`: expected level, plus probabilities |

¹ The current tokenizer provides 26 one-letter, 609 two-letter and 1,093 three-letter codes with a single-token form (possibly with a leading space). Jean discovers these when loading and skips combinations that split into several tokens. The whole input must still fit within 4,096 tokens.

For example, keeping the same `state`/`question` fields:

```json
{"type":"noul","state":"The customer asks for a refund.","question":"Is a refund requested?","choices":{"yes":"Yes","no":"No"},"target":"yes"}
{"type":"score","state":"The customer cannot access their account.","question":"How urgent?","choices":{"0":"Can wait","1":"Today","2":"Blocking"},"target":"2"}
```

`target` is the correct answer ID (a string for every type); omit it for prediction. Optional `images` contains paths relative to the dataset file. `state` can also be structured JSON. Add a new dataset by converting its examples to this format. Keep training and validation examples separate.

## Try it

```sh
uv sync
uv run python -m jean.prepare
uv run jean-train data/boolq/train.jsonl checkpoints/jean \
  --validation data/boolq/validation.jsonl
```

The binary sample takes the first 200 training and 50 validation examples from [google/boolq](https://huggingface.co/datasets/google/boolq), excluding any training passage shared with our validation sample. It is for testing the code, not measuring general decision ability. BoolQ uses CC BY-SA 3.0. Re-running preparation does not overwrite existing samples.

Prediction from Python:

```python
from jean.model import Jean

model = Jean.load("checkpoints/jean")  # Automatically selects CUDA, Metal, or CPU.
request = {"state": "The customer asks for a refund.", "question": "Is a refund requested?",
           "type": "noul", "choices": {"yes": "Yes", "no": "No"}}
print(model.predict(request))
```

Training uses the same cross entropy for all types; only ordinary choice options are shuffled. Validation reports accuracy of the most likely answer, including the most likely rating level.

Training defaults to CUDA. Full-weight AdamW needs roughly 50 GB before activations; use a remote GPU for actual training. On Apple silicon, ordinary PyTorch supports Metal through `device="mps"`; `Jean.load()` selects it automatically when available. `device="cpu"` also supports local inference. Our M5 Pro with 24 GiB of unified memory passed real text/image inference and a vocabulary-head backward check on Metal, but cannot fit full-weight AdamW training. The downloaded checkpoint is cached in `.cache/liquid` and reused automatically when running from this folder. It is approximately 6.3 GB to download; saved FP32 weights are approximately 12.5 GB. Data, downloaded weights and checkpoints are ignored by Git.

## Decision Index

Install the optional runner with `uv sync --extra benchmark`. The adapter supports the official text suite's `choice` and `noul` questions, answers each question separately, and leaves persistence, resume and scoring to [Decision Index](https://github.com/apolinario/decision-index). It does not evaluate images or `score` questions.

First prepare the official **0.3** suite following the upstream rebuild/import instructions, placing the imported suite in `benchmarks/suite-0.3`. The full rebuild needs approximately 7 GB of downloads and 17 GB of working space; prepare it remotely if laptop space is limited. Installing the runner alone downloads no benchmark data.

With that suite available, start with a sample:

```sh
uv run --extra benchmark python -m decision_index suite sample \
  --edition 0.3 --dir benchmarks/suite-0.3 \
  --n 100 --out benchmarks/sample-100.jsonl.gz

uv run --extra benchmark python -m decision_index pipeline \
  --engine jean.decision_index:JeanEngine --option device=mps \
  --edition 0.3 --suite-dir benchmarks/suite-0.3 \
  --rows benchmarks/sample-100.jsonl.gz --out runs/liquid-sample
```

For a full run, omit `--rows` and use a new output directory. For Jean's trained weights, add `--model checkpoints/jean`; use a separate directory for each checkpoint. CUDA machines use `--option device=cuda`; omitting the device selects it automatically.

The pipeline writes individual predictions, per-benchmark metrics and the public index into the output directory. Repeating the same command resumes it. A sampled run is a diagnostic, not a full leaderboard result; the private Full score requires the maintainers' evaluation. Options and inputs are never truncated: requests exceeding Jean's capacity are marked unsupported and penalized by the official scorer.

No RL framework yet.

```sh
uv run --extra benchmark pytest -q
uv run ruff check .
```

The forward pass has three steps: `prepare_input()` builds text/image tensors, the model runs once, and `answer_tokens()` identifies the vocabulary scores to read.

Local verification used the real checkpoint on Metal for text/image inference, 30 choices and a vocabulary-head backward pass. Full training and checkpoint roundtrips were tested with tiny random weights; the 3B model has not been trained yet.

Our code is MIT. Liquid weights use the license in `licenses/LIQUID_LICENSE`. Tiny-model tests use random weights and download nothing.
