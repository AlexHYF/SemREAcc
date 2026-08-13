# Running the XOR SemRE experiment

This harness compares two uses of the same local model:

1. **Direct:** ask whether the complete `given-name surname` string matches the
   crossed Chinese/Japanese predicate.
2. **Component SemRE:** ask the model the four reusable atomic predicates and
   evaluate
   `(ChineseGiven ∧ JapaneseSurname) ∨ (JapaneseGiven ∧ ChineseSurname)`.

The core run contains 4,000 full names and 800 unique atomic questions. Raw API
answers are appended to JSONL immediately, so an interrupted run resumes rather
than repeating completed requests. Invalid answers count as incorrect in the
primary accuracy metric and are also reported separately as coverage.

## Quick start on Lambda Cloud or RunPod

Start from a recent NVIDIA/CUDA image with Python 3.10--3.12, clone or upload
this repository, and run:

```bash
cd SemREAcc
scripts/bootstrap_vllm.sh
```

The bootstrap reads the maximum CUDA version reported by `nvidia-smi`. All
CUDA 12.x machines use the CUDA 12.9 vLLM wheel via NVIDIA's minor-version
compatibility; CUDA 13 machines use the CUDA 13 vLLM wheel. Override detection
explicitly when needed:

```bash
SEMRE_CUDA_VARIANT=cu129 scripts/bootstrap_vllm.sh  # RTX 4090 / CUDA 12.x
SEMRE_CUDA_VARIANT=cu130 scripts/bootstrap_vllm.sh  # B300 / CUDA 13
```

The wrapper records the selected serving backend inside `.venv`, so the same
`run_model_experiment.sh` command works for both. An explicit `cu124` option is
retained as a slow compatibility fallback using Hugging Face's lightweight
OpenAI-compatible `transformers serve`.

The CUDA 12.9 path uses the stable vLLM release; the CUDA 13.0 variant still
uses vLLM nightly. Transformers main is installed for the latest model
definitions used by Qwen3.5 and GLM-4.7-Flash.

Put Hugging Face downloads on the provider's persistent volume when possible.
For example, on a RunPod volume mounted at `/workspace`:

```bash
export HF_HOME=/workspace/huggingface-cache
```

On Lambda Cloud, an attached filesystem is normally mounted below
`/lambda/nfs/FILESYSTEM_NAME`, so the equivalent is:

```bash
export HF_HOME=/lambda/nfs/FILESYSTEM_NAME/huggingface-cache
```

See the providers' current storage references for lifecycle details:
[RunPod storage](https://docs.runpod.io/pods/storage/types) and
[Lambda Cloud storage](https://docs.lambda.ai/public-cloud/on-demand/#storage).

Then launch a server, wait for it to become ready, run the core experiment, and
stop the server with one command:

```bash
scripts/run_model_experiment.sh qwen35-0.8b --dataset core --concurrency 32
```

The first launch downloads the checkpoint. `SERVER_START_TIMEOUT` defaults to
two hours to accommodate large downloads. Set `HF_TOKEN` before launch if the
Hugging Face endpoint requires authentication.

Available model keys are:

```bash
.venv/bin/python scripts/serve_model.py --list
```

The configured keys are `qwen35-0.8b`, `qwen35-2b`, `qwen35-4b`,
`gemma3-4b`, `llama32-3b`, `phi4-mini-3.8b`, `ministral3-3b`,
`qwen35-9b`, `qwen35-27b`, `glm47-flash`, `qwen35-35b-a3b`,
`kimi-linear-48b-a3b`, and `glm45-air-fp8`.
The additional `qwen35-122b-a10b-fp8` key is the recommended large-model
checkpoint for a single B300.

## GPU count and memory

`configs/models.json` supplies a conservative default tensor-parallel count.
Override it to match the rented machine:

```bash
TP_SIZE=2 scripts/run_model_experiment.sh qwen35-27b --dataset core
```

The defaults are starting points, not guaranteed minimums: CUDA graphs, the
inference engine version, and each GPU's usable memory affect whether a model
fits. In particular, MoE *active* parameters describe compute per token, not
the number of weights that must reside in aggregate GPU memory.

| Model key | Stored precision | Default GPUs | Practical starting point |
|---|---:|---:|---|
| `qwen35-0.8b` / `2b` / `4b` / `9b` | BF16 | 1 | One modern 24--48 GB GPU |
| `gemma3-4b` / `llama32-3b` / `ministral3-3b` / `phi4-mini-3.8b` | BF16 | 1 | One modern 24--48 GB GPU |
| `qwen35-27b` | BF16 | 1 | One 80 GB GPU, or use `TP_SIZE=2` on smaller GPUs |
| `glm47-flash` | BF16 | 2 | Two GPUs; about 60 GB of weights in aggregate before overhead |
| `qwen35-35b-a3b` | BF16 | 2 | Two GPUs; about 70 GB of weights in aggregate before overhead |
| `kimi-linear-48b-a3b` | BF16 | 4 | Four GPUs, matching the publisher's vLLM example |
| `qwen35-122b-a10b-fp8` | FP8 | 1 | One B300 288 GB; text-only mode and 4K context |
| `glm45-air-fp8` | FP8 | 4 | Four 80 GB GPUs; test the dry launch before renting a long job |

The four-GPU Kimi default follows the publisher's
[vLLM deployment example](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct#deployment).
The conservative GLM recommendation is informed by the publisher's
[inference configurations](https://huggingface.co/zai-org/GLM-4.5-Air-FP8#system-requirements),
although this harness uses a much smaller context and batch than its
full-featured benchmark.

All launches use a 4,096-token maximum model length because this classification
task has very short prompts. This materially reduces KV-cache reservation. To
change a server option, pass it after `--` to `serve_model.py` in the two-terminal
workflow described below.

## Two-terminal or existing-server workflow

For easier server debugging, launch it separately:

```bash
TP_SIZE=1 .venv/bin/python scripts/serve_model.py qwen35-4b
```

In another terminal:

```bash
.venv/bin/python scripts/run_xor_experiment.py \
  --model-key qwen35-4b \
  --dataset core \
  --concurrency 32
```

The experiment client uses the OpenAI-compatible `/v1/chat/completions`
interface and has no third-party Python dependencies. Point it at SGLang, a
remote vLLM instance, or an already-running RunPod endpoint with:

```bash
OPENAI_BASE_URL=http://SERVER:PORT/v1 \
OPENAI_API_KEY=EMPTY \
.venv/bin/python scripts/run_xor_experiment.py \
  --model-key qwen35-4b --dataset core
```

If a server is already running locally, the one-command wrapper can reuse it:

```bash
USE_EXISTING_SERVER=1 OPENAI_BASE_URL=http://127.0.0.1:8000/v1 \
  scripts/run_model_experiment.sh qwen35-4b --dataset core
```

To append custom vLLM arguments, use the separate server command:

```bash
.venv/bin/python scripts/serve_model.py qwen35-4b -- \
  --enforce-eager
```

## Smoke test, full run, and recovery

Test the complete pipeline cheaply before committing to all 4,000 names:

```bash
scripts/run_model_experiment.sh qwen35-0.8b --limit 20 --concurrency 8
```

The limit selects only the atomic queries needed by those 20 names. Rerun the
same command without `--limit`; cached atoms and direct predictions are reused,
and the remaining rows are appended.

For the exhaustive 40,000-name set:

```bash
scripts/run_model_experiment.sh qwen35-0.8b --dataset full --concurrency 64
```

Network/server failures and unparseable outputs are recorded rather than
silently dropped. Retry just those records by adding `--rerun-failures`. If you
change a prompt or sampling setting, the runner refuses to mix the new protocol
with old cached answers; supply a new name such as `--run-name qwen35-4b-rep2`.

## Outputs and cross-model summary

Each run writes under `results/MODEL_KEY/`:

- `atomic.jsonl`: raw/cached component answers, timing, token use, and errors.
- `direct_core.jsonl` or `direct_full.jsonl`: raw complete-name answers.
- `predictions_core.csv`: gold, direct, and recombined SemRE predictions.
- `metrics_core.json`: accuracy, precision, recall, F1, false-positive rate,
  false-negative rate, coverage, pair-type breakdowns, 95% Wilson intervals,
  request statistics, and an exact paired McNemar test.
- `run_config.json`: the protocol fingerprint that prevents cache mixing.
- `dataset_config_core.json`: the dataset hash that prevents stale predictions
  from being reused after a data change.
- `server.log`: vLLM startup and inference logs when the wrapper is used.

After running several models, build the scaling table:

```bash
.venv/bin/python scripts/summarize_results.py --dataset core
```

This produces `results/summary_core.csv`, sorted by total parameter count. Keep
the raw JSONL files for auditing: romanized names can be cross-cultural, so the
model's disagreement with the source-list labels can reflect benchmark
ambiguity as well as classifier error.

## Three similarly sized small-model voting experiment

To test whether several small local models improve accuracy without confounding
the vote with a large capacity difference, use three instruction models from
different families but a narrow 3--4B parameter range:

| Model key | Checkpoint | Parameters |
|---|---|---:|
| `qwen35-4b` | `Qwen/Qwen3.5-4B` | 4.0B |
| `gemma3-4b` | `google/gemma-3-4b-it` | 4.0B |
| `llama32-3b` | `meta-llama/Llama-3.2-3B-Instruct` | 3.2B |

Each model independently runs the unchanged direct and component benchmarks on
all 4,000 core rows. The different model families make error diversity more
plausible, while their similar sizes make the majority vote easier to interpret.

The offline evaluator reports three pre-specified equal-vote ensembles:

1. `direct_majority`: majority vote over the three whole-name answers.
2. `component_majority`: each model evaluates the component SemRE, followed by
   a majority vote over the three final Boolean results.
3. `oracle_majority_component`: majority vote on each atomic name predicate,
   followed by evaluation of the component SemRE. This is the primary
   ensemble-LLM-oracle condition.

Use a strict two-of-three majority. An invalid answer is not silently converted
to `NO`; an ensemble answer is valid only when at least two models produce the
same Boolean label. No inference calls are needed after the three ordinary
model runs. Evaluate improvement against the best individual model, and inspect
accuracy, false-positive rate, exact paired McNemar comparisons, and pairwise
disagreement. Low disagreement means voting has little diversity to exploit.

Gemma 3 and Llama 3.2 are gated Hugging Face repositories. Accept the terms on
their checkpoint pages before starting, then export a Hugging Face read token
as `HF_TOKEN` (or authenticate with `hf auth login`).

On a CUDA 12.4 RTX 4090 RunPod, run the models sequentially with the
Transformers serving fallback:

```bash
cd /workspace/SemREAcc
export HF_HOME=/workspace/huggingface-cache
export HF_TOKEN=hf_your_read_token
SEMRE_CUDA_VARIANT=cu124 scripts/bootstrap_vllm.sh
scripts/run_small_model_ensemble.sh
```

The bootstrap command rebuilds the environment, so deactivate an already-active
`.venv` before running it. The ensemble wrapper downloads each missing model,
starts and stops its Transformers server, completes the same dataset, and then
performs the offline vote. Override concurrency if necessary with, for example,
`CONCURRENCY=16 scripts/run_small_model_ensemble.sh`.

The concise comparison is
`results/ensemble-qwen-gemma-llama/summary_core.csv`. The same directory also
contains detailed JSON metrics and row-level predictions. Package every input
and ensemble output for download with:

```bash
tar -czf ensemble-qwen-gemma-llama-results.tar.gz \
  results/ensemble-qwen35-4b \
  results/ensemble-gemma3-4b \
  results/ensemble-llama32-3b \
  results/ensemble-qwen-gemma-llama
```

## Reproducibility choices

- The default is greedy decoding (`temperature=0`, `seed=0`) with a 16-token
  ceiling and the same system instruction for both methods.
- Qwen3.5, GLM-4.7, and GLM-4.5 thinking is disabled through each model's
  chat-template request option. Gemma 3, Llama 3.2, Phi-4-mini, Ministral, and
  Kimi use their default instruction templates.
- Every unique atom is queried once per model and reused across names. There is
  no short-circuiting, so all models receive exactly the same atomic workload.
- Three-valued logic propagates an invalid atomic response only when the other
  Boolean input cannot determine the result.
