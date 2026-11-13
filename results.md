# GAIA validation results

Two arms on the same model, tools, prompts, and limits. **Baseline** is one ReAct agent. **Full** is the system as shipped: supervisor, runtime-authored subagents, Celery dispatch, the diagnose-and-retry loop, and PostgreSQL + ChromaDB memory. Every table and count below is computed by `bench/report.py` from per-task result files. Paragraphs marked *Checked by hand* come from replaying individual tasks.

## 1. Setup

| | |
|---|---|
| Model | `qwen2.5:7b` (7.6B parameters) via Ollama 0.33.0, OpenAI-compatible endpoint, temperature 0 |
| Quantization | Q4_K_M |
| Context window | 16,384 tokens (`qwen2.5-7b-ctx16k`: same weights, `num_ctx` raised from Ollama's 4,096 default) |
| Embeddings | `nomic-embed-text` for ChromaDB (full arm only) |
| Tools (both arms) | `read_file`, `write_file`, `list_files`, `web_search` (`ddgs`, snippets only, no page fetching) |
| Hardware | Apple M3 Pro, 18 GB RAM, macOS 26.6.2 |
| Tasks | 165 from the GAIA 2023 validation split; 106 runnable, 59 unsupported |
| Per-task timeout | 20 minutes, both arms |
| Max steps | 25 per agent (LangGraph default), both arms |
| Full-arm settings | repair budget 3 attempts per subtask, 4 Celery worker processes, 300 s wait per subtask (all system defaults) |
| Runs per task | 1 per arm, task order fixed by seed 20261109 |
| Date run | 2026-11-12 to 2026-11-13 |
| Code | commit `da846b7` |

## 2. Main table

| Arm | Level | Tasks | Ran | Correct | Accuracy | Unrecoverable | Wrong answer | Unsupported | Avg latency (ran) |
|---|---|---|---|---|---|---|---|---|---|
| baseline | all | 165 | 106 | 9 | 5.5% | 4 (2.4%) | 93 (56.4%) | 59 (35.8%) | 52s |
| baseline | L1 | 53 | 40 | 5 | 9.4% | 1 (1.9%) | 34 (64.2%) | 13 (24.5%) | 46s |
| baseline | L2 | 86 | 53 | 4 | 4.7% | 3 (3.5%) | 46 (53.5%) | 33 (38.4%) | 63s |
| baseline | L3 | 26 | 13 | 0 | 0.0% | 0 (0.0%) | 13 (50.0%) | 13 (50.0%) | 26s |
| full | all | 165 | 106 | 3 | 1.8% | 12 (7.3%) | 91 (55.2%) | 59 (35.8%) | 67s |
| full | L1 | 53 | 40 | 1 | 1.9% | 4 (7.5%) | 35 (66.0%) | 13 (24.5%) | 63s |
| full | L2 | 86 | 53 | 2 | 2.3% | 7 (8.1%) | 44 (51.2%) | 33 (38.4%) | 60s |
| full | L3 | 26 | 13 | 0 | 0.0% | 1 (3.8%) | 12 (46.2%) | 13 (50.0%) | 112s |

Rates are over all tasks in the row. Latency is averaged over tasks that ran; an unsupported task is never started.

Accuracy over runnable tasks only: baseline 8.5%, full 2.8%.

## 3. Headline

Unrecoverable failures: **4 → 12 of 165**, a relative increase of **200.0%** ((baseline − full) / baseline = -2.000).

Over runnable tasks only (106): baseline 3.8%, full 11.3%.

Task by task: 4 baseline failures did not fail in the full system, 12 tasks failed only in the full system, and 0 failed in both.

**Statistical significance.** An exact McNemar test on the paired tasks gives p = 0.077 for the difference in unrecoverable failures (4 vs 12 discordant tasks), and p = 0.146 for the difference in correct answers (correct only in baseline 9, only in full 3, in both 0). Neither reaches p < 0.05, so the direction is suggestive, not established, at this sample size.

## 4. Failure breakdown

| Sub-cause | Baseline | Full |
|---|---|---|
| empty_decomposition | 0 | 8 |
| empty_output | 1 | 0 |
| subtask_timeout | 0 | 4 |
| timeout | 3 | 0 |

Unsupported inputs (identical for both arms, decided before any run): attachment_not_text 33, needs_capability 26.

## 5. Trade-offs

| Per task that ran | Baseline | Full | Full / baseline |
|---|---|---|---|
| Latency | 52 s | 67 s | 1.3x |
| Tokens in | 3560 | 4701 | 1.3x |
| Tokens out | 246 | 867 | 3.5x |
| Model calls | 3.1 | 7.5 | 2.4x |

The full arm spawned 2.8 subagents per task on average and triggered at least one retry on 0 of 106 tasks.

## 6. Biggest fix

The most common baseline failure the full system did not repeat was **timeout** (3 of 4 such tasks): the run hit the 20-minute limit. Of the 4 tasks, **0 became correct**; the full system's outcomes were wrong_answer 4. A retry fired on 0 of them.

So the full system turned these failures into finished runs, not into right answers. None involved a retry, so the repair loop played no part. What differed was the prompt: the supervisor's decomposition gave each agent a different task description, and with those prompts these runs finished instead of stalling.

Example task: `17b5a6a3-bc87-42e8-b0fb-6ab0781ef2cc` (level 2). In the full run, the supervisor split it into 3 subtask(s); no retry was triggered, so the repair loop played no part.

*Checked by hand.* Baseline task `dc28cf18-6431-458b-83ef-64b3ce566c10` timed out with no completed model call. Its first model call is deterministic (temperature 0, no search results yet), so it was replayed directly against Ollama: within one turn the model emitted the same `web_search` call 51 times in 60 seconds and never ended the turn. Ollama sets no output cap by default, so that turn would continue until the 20-minute limit. The label `timeout` is correct; this is a model failure, not a harness one. The full system finished the same task, with a wrong answer.

## 7. What didn't improve

12 task(s) failed **only** in the full system:

- **empty_decomposition** (8): the supervisor's decompose call returned an empty subtask list (about 12 output tokens, the size of an empty list) and `decompose_goal` raises on an empty plan; decompose has no repair. Examples: `05407167-39ec-4d3a-a234-73a9120c325d`, `0a65cb96-cb6e-4a6a-8aae-c1084f613456`.
- **subtask_timeout** (4): a subagent ran past the supervisor's hard-coded 300 s wait for its Celery result; that wait raises and ends the whole run, although the subagent had not failed. Examples: `389793a7-ca17-4e82-81cb-2b3a2391b4b9`, `851e570a-e3de-4d84-bcfa-cc85578baa59`.

**The diagnose-and-retry loop fired on 0 of 106 tasks**, and 0 subtask(s) were ever marked failed. It only runs when a subagent raises an exception, and in this run none did: every full-arm failure happened in the supervisor itself (an empty plan, or the 300 s wait), which the loop does not cover.

Wrong answers were the dominant outcome in both arms (baseline 93, full 91). Retrying cannot fix a run that finishes confidently with the wrong answer: the repair loop only fires on an exception, never on a wrong result.

## 8. Limitations

- **Sample size.** 4 tasks, of which 106 ran. Per-level rows, level 3 especially, are too small to compare.
- **One run per task.** The 15 pilot tasks were run again here, a day apart, with identical settings. Outcomes changed on baseline 3 of 15, full 1 of 15 tasks. Search results drift and the full arm's subtasks finish in a different order each run, so temperature 0 is not deterministic end to end. Differences of a few tasks between the arms are inside this noise.
- **Validation split only.** These tasks are public; nothing here says how either arm does on the held-out test split.
- **Small local model.** A 7.6B model at 4-bit quantization. Absolute accuracy is far below published GAIA results and should not be compared with them.
- **Narrow tools.** -102 tasks could not be attempted at all. Search returns snippets and cannot open a page, which most GAIA tasks need.
- **What "retry" means.** Retries are an application-level diagnose-and-retry loop that runs inside each Celery task. The system has no Celery retry queues.
- **What the repair loop can see.** It fires only when a subagent raises. Tool errors are returned to the agent as messages and wrong answers finish normally, so neither triggers it.
- **Memory is order-dependent.** The full arm recalls earlier tasks' goals and answers into later decompositions, so its results depend on the task order, which was fixed by seed.
- **Unprotected steps.** The supervisor's own decompose and aggregate steps have no repair; a failure there ends the run.
- **One shared model server.** Four worker processes share one local model server, so subagents queue behind each other and the 300 s per-subtask wait is easier to hit than it would be with parallel model servers.

## 9. How to reproduce

```bash
# services and models
docker compose up -d postgres redis
ollama pull qwen2.5:7b && ollama pull nomic-embed-text
printf 'FROM qwen2.5:7b\nPARAMETER num_ctx 16384\n' > Modelfile && ollama create qwen2.5-7b-ctx16k -f Modelfile

# environment (keep the virtualenv outside any synced folder)
uv venv --python 3.9 ~/.venvs/agent-orchestration
VIRTUAL_ENV=~/.venvs/agent-orchestration uv pip install -e ".[dev]" pandas pyarrow ollama

# GAIA validation is read from $GAIA_DIR (default ~/gaia-test/2023/validation); it is never copied here
~/.venvs/agent-orchestration/bin/python -m pytest                      # harness and system tests
~/.venvs/agent-orchestration/bin/python -m bench.run --set smoke --run-tag smoke
~/.venvs/agent-orchestration/bin/python -m bench.run --set pilot --run-tag pilot
# caffeinate cannot stop sleep on lid-close; keep the Mac plugged in with the lid open
caffeinate -dims ~/.venvs/agent-orchestration/bin/python -m bench.run --set all --run-tag full
~/.venvs/agent-orchestration/bin/python -m bench.report results --repeat-dir results-pilot --write results.md
```

The run is resumable: rerun the same command and finished tasks are skipped. Per-task result files contain ground-truth answers and are gitignored.
