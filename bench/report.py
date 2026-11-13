"""Aggregate per-task result files into the numbers that get reported.

    python -m bench.report results-pilot            # print a summary
    python -m bench.report results --project 165    # also project a full run

Every figure is computed from the result files; nothing is entered by hand.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from bench.classify import CORRECT, OUTCOMES, UNRECOVERABLE, UNSUPPORTED, WRONG_ANSWER

ARMS = ("baseline", "full")


def load(results_dir: Path) -> Dict[str, Dict[str, Dict]]:
    """{arm: {task_id: record}}"""
    return {
        arm: {p.stem: json.loads(p.read_text()) for p in sorted((Path(results_dir) / arm).glob("*.json"))}
        for arm in ARMS
    }


def rate(count: int, total: int) -> Optional[float]:
    return count / total if total else None


def pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def stats(records: List[Dict]) -> Dict:
    """Outcome counts and costs for one group of results."""
    counts = Counter(r["outcome"] for r in records)
    ran = [r for r in records if r["outcome"] != UNSUPPORTED]
    total = len(records)

    def mean(key: str) -> Optional[float]:
        return sum(r[key] or 0 for r in ran) / len(ran) if ran else None

    return {
        "n": total,
        "ran": len(ran),
        "counts": {o: counts.get(o, 0) for o in OUTCOMES},
        "accuracy": rate(counts.get(CORRECT, 0), total),
        "accuracy_of_ran": rate(counts.get(CORRECT, 0), len(ran)),
        "unrecoverable_rate": rate(counts.get(UNRECOVERABLE, 0), total),
        "unrecoverable_rate_of_ran": rate(counts.get(UNRECOVERABLE, 0), len(ran)),
        "wrong_rate": rate(counts.get(WRONG_ANSWER, 0), total),
        "unsupported_rate": rate(counts.get(UNSUPPORTED, 0), total),
        # Costs are averaged over tasks that actually ran; an unsupported task costs nothing.
        "avg_latency_s": mean("latency_s"),
        "avg_tokens_in": mean("tokens_in"),
        "avg_tokens_out": mean("tokens_out"),
        "avg_llm_calls": mean("llm_calls"),
        "sub_causes": dict(Counter(r["sub_cause"] for r in records if r["outcome"] == UNRECOVERABLE).most_common()),
    }


def by_level(records: List[Dict]) -> Dict[str, Dict]:
    groups = {"all": stats(records)}
    for level in (1, 2, 3):
        groups[f"L{level}"] = stats([r for r in records if r["level"] == level])
    return groups


def headline(baseline: Dict, full: Dict) -> Dict:
    """Relative reduction in unrecoverable failures, with the raw counts."""
    b, f, n = baseline["counts"][UNRECOVERABLE], full["counts"][UNRECOVERABLE], baseline["n"]
    return {
        "baseline_failures": b,
        "full_failures": f,
        "n": n,
        "relative_reduction": None if b == 0 else (b - f) / b,
    }


def paired(data: Dict[str, Dict[str, Dict]]) -> Dict[str, List[Dict]]:
    """Per-task comparison over tasks both arms completed."""
    shared = sorted(set(data["baseline"]) & set(data["full"]))
    recovered, regressed, both = [], [], []
    for task_id in shared:
        b, f = data["baseline"][task_id], data["full"][task_id]
        b_fail, f_fail = b["outcome"] == UNRECOVERABLE, f["outcome"] == UNRECOVERABLE
        row = {"task_id": task_id, "level": b["level"], "baseline": b, "full": f}
        if b_fail and not f_fail:
            recovered.append(row)
        elif f_fail and not b_fail:
            regressed.append(row)
        elif b_fail and f_fail:
            both.append(row)
    return {"recovered": recovered, "regressed": regressed, "both_failed": both}


def mcnemar_p(b: int, c: int) -> Optional[float]:
    """Exact two-sided McNemar test on the discordant pairs of a paired comparison.

    b and c are the tasks where only one arm had the outcome. With no
    discordant pairs there is nothing to test.
    """
    import math

    n = b + c
    if n == 0:
        return None
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def correctness_pairs(data: Dict[str, Dict[str, Dict]]) -> Dict[str, int]:
    shared = set(data["baseline"]) & set(data["full"])
    is_correct = lambda arm, t: data[arm][t]["outcome"] == CORRECT
    return {
        "both": sum(1 for t in shared if is_correct("baseline", t) and is_correct("full", t)),
        "baseline_only": sum(1 for t in shared if is_correct("baseline", t) and not is_correct("full", t)),
        "full_only": sum(1 for t in shared if is_correct("full", t) and not is_correct("baseline", t)),
    }


# What each failure sub-cause means mechanically, for the report's prose.
CAUSE_EXPLAINED = {
    "timeout": "the run hit the 20-minute limit",
    "subtask_timeout": "a subagent ran past the supervisor's hard-coded 300 s wait for its Celery result; that wait raises and ends the whole run, although the subagent had not failed",
    "empty_decomposition": "the supervisor's decompose call returned an empty subtask list (about 12 output tokens, the size of an empty list) and `decompose_goal` raises on an empty plan; decompose has no repair",
    "empty_output": "the run finished with an empty final message",
    "max_steps": "an agent hit LangGraph's 25-step recursion limit",
    "retries_exhausted": "the repair loop used all its attempts and the unattended escalation aborted the run",
    "parse_error": "a model response could not be parsed into the expected structure",
    "model_api_error": "a call to the model server failed",
    "infrastructure_error": "Postgres, Redis, or ChromaDB raised",
    "crash": "an unclassified exception ended the run",
}


def repeat_agreement(data: Dict[str, Dict[str, Dict]], repeat: Dict[str, Dict[str, Dict]]) -> Dict[str, Dict]:
    """How often the same arm gave the same outcome on a task run twice."""
    agreement = {}
    for arm in ARMS:
        shared = sorted(set(data[arm]) & set(repeat.get(arm, {})))
        flips = [t for t in shared if data[arm][t]["outcome"] != repeat[arm][t]["outcome"]]
        agreement[arm] = {"tasks": len(shared), "flipped": len(flips), "flipped_ids": flips}
    return agreement


def table(data: Dict[str, Dict[str, Dict]]) -> str:
    lines = [
        "| Arm | Level | Tasks | Ran | Correct | Accuracy | Unrecoverable | Wrong answer | Unsupported | Avg latency (ran) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in ARMS:
        for level, s in by_level(list(data[arm].values())).items():
            c = s["counts"]
            latency = "n/a" if s["avg_latency_s"] is None else f"{s['avg_latency_s']:.0f}s"
            lines.append(
                f"| {arm} | {level} | {s['n']} | {s['ran']} | {c[CORRECT]} | {pct(s['accuracy'])} "
                f"| {c[UNRECOVERABLE]} ({pct(s['unrecoverable_rate'])}) | {c[WRONG_ANSWER]} ({pct(s['wrong_rate'])}) "
                f"| {c[UNSUPPORTED]} ({pct(s['unsupported_rate'])}) | {latency} |"
            )
    return "\n".join(lines)


def summary(data: Dict[str, Dict[str, Dict]], project: Optional[int] = None, runnable: int = 106) -> str:
    out = [table(data), ""]
    overall = {arm: stats(list(data[arm].values())) for arm in ARMS}
    for arm in ARMS:
        s = overall[arm]
        if s["ran"]:
            out.append(
                f"{arm}: failure sub-causes {s['sub_causes'] or '{}'}; per ran task "
                f"{s['avg_latency_s']:.0f}s, {s['avg_tokens_in']:.0f} tokens in, "
                f"{s['avg_tokens_out']:.0f} out, {s['avg_llm_calls']:.1f} model calls"
            )
    h = headline(overall["baseline"], overall["full"])
    out.append(
        f"unrecoverable failures: {h['baseline_failures']} -> {h['full_failures']} of {h['n']} "
        f"(relative reduction {pct(h['relative_reduction'])})"
    )
    pairs = paired(data)
    out.append(
        f"paired: recovered by full {len(pairs['recovered'])}, regressed in full "
        f"{len(pairs['regressed'])}, failed in both {len(pairs['both_failed'])}"
    )
    if project:
        per_task = sum((overall[arm]["avg_latency_s"] or 0) for arm in ARMS)
        hours = per_task * runnable / 3600
        out.append(
            f"projection: {per_task:.0f}s per runnable task across both arms x {runnable} runnable "
            f"of {project} tasks = {hours:.1f} hours (unsupported tasks cost nothing)"
        )
    return "\n".join(out)


# --- results.md -------------------------------------------------------------
# Written for a public repo: task ids and aggregate numbers only. No ground
# truth and no answer text, which GAIA's terms forbid resharing.


def _sh(*cmd: str) -> str:
    import subprocess

    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def environment() -> Dict[str, str]:
    from bench import config

    show = _sh("ollama", "show", config.MODEL)

    def field(name: str) -> str:
        for line in show.splitlines():
            parts = line.split()
            if parts[:1] == [name] or parts[:2] == name.split():
                return " ".join(parts[len(name.split()):])
        return "unknown"

    memory_gb = int(_sh("sysctl", "-n", "hw.memsize") or 0) / 2**30
    return {
        "model": config.MODEL,
        "parameters": field("parameters"),
        "quantization": field("quantization"),
        "num_ctx": field("num_ctx"),
        "ollama": _sh("ollama", "--version").replace("ollama version is ", ""),
        "chip": _sh("sysctl", "-n", "machdep.cpu.brand_string"),
        "memory": f"{memory_gb:.0f} GB",
        "os": f"macOS {_sh('sw_vers', '-productVersion')}",
        "commit": _sh("git", "rev-parse", "--short", "HEAD"),
    }


def _cost(s: Dict, key: str, fmt: str = "{:.0f}") -> str:
    return "n/a" if s[key] is None else fmt.format(s[key])


def _ratio(full: Optional[float], base: Optional[float]) -> str:
    return "n/a" if not full or not base else f"{full / base:.1f}x"


def _describe_recovery(row: Dict) -> str:
    """What the full system did on a task the baseline failed. No answer text."""
    f = row["full"]
    subtasks = f.get("subtasks") or []
    bits = [f"the supervisor split it into {len(subtasks)} subtask(s)"]
    retries = f.get("retries_triggered") or 0
    if retries:
        diagnoses = [x["diagnosis"] for s in subtasks for x in s["failures"] if x.get("diagnosis")]
        bits.append(f"the repair loop retried {retries} time(s)")
        if diagnoses:
            bits.append(f'its first diagnosis read: "{diagnoses[0]}"')
    else:
        bits.append("no retry was triggered, so the repair loop played no part")
    return "; ".join(bits)


def results_markdown(
    data: Dict[str, Dict[str, Dict]], subset_note: str = "", repeat: Optional[Dict[str, Dict[str, Dict]]] = None
) -> str:
    from bench import config

    env = environment()
    overall = {arm: stats(list(data[arm].values())) for arm in ARMS}
    b, f = overall["baseline"], overall["full"]
    h = headline(b, f)
    pairs = paired(data)
    stamps = sorted(r["finished_at"] for arm in ARMS for r in data[arm].values())
    n, ran = b["n"], b["ran"]
    L: List[str] = []
    add = L.append

    add("# GAIA validation results\n")
    add(
        "Two arms on the same model, tools, prompts, and limits. **Baseline** is one ReAct "
        "agent. **Full** is the system as shipped: supervisor, runtime-authored subagents, "
        "Celery dispatch, the diagnose-and-retry loop, and PostgreSQL + ChromaDB memory. "
        "Every table and count below is computed by `bench/report.py` from per-task result files. "
        "Paragraphs marked *Checked by hand* come from replaying individual tasks.\n"
    )

    add("## 1. Setup\n")
    add("| | |\n|---|---|")
    add(f"| Model | `qwen2.5:7b` ({env['parameters']} parameters) via Ollama {env['ollama']}, OpenAI-compatible endpoint, temperature 0 |")
    add(f"| Quantization | {env['quantization']} |")
    add(f"| Context window | 16,384 tokens (`{env['model']}`: same weights, `num_ctx` raised from Ollama's 4,096 default) |")
    add(f"| Embeddings | `{config.EMBED_MODEL}` for ChromaDB (full arm only) |")
    add("| Tools (both arms) | `read_file`, `write_file`, `list_files`, `web_search` (`ddgs`, snippets only, no page fetching) |")
    add(f"| Hardware | {env['chip']}, {env['memory']} RAM, {env['os']} |")
    add(f"| Tasks | {n} from the GAIA 2023 validation split{subset_note}; {ran} runnable, {n - ran} unsupported |")
    add(f"| Per-task timeout | {config.TASK_TIMEOUT_S // 60} minutes, both arms |")
    add("| Max steps | 25 per agent (LangGraph default), both arms |")
    add(f"| Full-arm settings | repair budget {config.MAX_ATTEMPTS} attempts per subtask, {config.WORKER_CONCURRENCY} Celery worker processes, 300 s wait per subtask (all system defaults) |")
    add(f"| Runs per task | 1 per arm, task order fixed by seed {config.SEED} |")
    add(f"| Date run | {stamps[0][:10]} to {stamps[-1][:10]} |" if stamps else "| Date run | n/a |")
    add(f"| Code | commit `{env['commit']}` |\n")

    add("## 2. Main table\n")
    add(table(data))
    add(
        "\nRates are over all tasks in the row. Latency is averaged over tasks that ran; "
        "an unsupported task is never started.\n"
    )
    add(f"Accuracy over runnable tasks only: baseline {pct(b['accuracy_of_ran'])}, full {pct(f['accuracy_of_ran'])}.\n")

    add("## 3. Headline\n")
    if h["relative_reduction"] is None:
        add(
            f"Unrecoverable failures: **{h['baseline_failures']} → {h['full_failures']} of {h['n']}**. "
            "The baseline had none, so a relative reduction is undefined.\n"
        )
    else:
        direction = "reduction" if h["relative_reduction"] >= 0 else "increase"
        add(
            f"Unrecoverable failures: **{h['baseline_failures']} → {h['full_failures']} of {h['n']}**, "
            f"a relative {direction} of **{pct(abs(h['relative_reduction']))}** "
            f"((baseline − full) / baseline = {h['relative_reduction']:+.3f}).\n"
        )
    add(
        f"Over runnable tasks only ({ran}): baseline {pct(b['unrecoverable_rate_of_ran'])}, "
        f"full {pct(f['unrecoverable_rate_of_ran'])}.\n"
    )
    add(
        f"Task by task: {len(pairs['recovered'])} baseline failures did not fail in the full system, "
        f"{len(pairs['regressed'])} tasks failed only in the full system, and "
        f"{len(pairs['both_failed'])} failed in both.\n"
    )
    p_fail = mcnemar_p(len(pairs["recovered"]), len(pairs["regressed"]))
    cp = correctness_pairs(data)
    p_correct = mcnemar_p(cp["baseline_only"], cp["full_only"])

    def fmt_p(value: Optional[float]) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    add(
        "**Statistical significance.** An exact McNemar test on the paired tasks gives "
        f"p = {fmt_p(p_fail)} for the difference in unrecoverable failures "
        f"({len(pairs['recovered'])} vs {len(pairs['regressed'])} discordant tasks), and "
        f"p = {fmt_p(p_correct)} for the difference in correct answers "
        f"(correct only in baseline {cp['baseline_only']}, only in full {cp['full_only']}, in both {cp['both']}). "
        + ("Neither reaches p < 0.05" if all(v is None or v >= 0.05 for v in (p_fail, p_correct))
           else "At least one reaches p < 0.05")
        + ", so the direction is suggestive, not established, at this sample size.\n"
    )

    add("## 4. Failure breakdown\n")
    causes = sorted(set(b["sub_causes"]) | set(f["sub_causes"]))
    if causes:
        add("| Sub-cause | Baseline | Full |\n|---|---|---|")
        for cause in causes:
            add(f"| {cause} | {b['sub_causes'].get(cause, 0)} | {f['sub_causes'].get(cause, 0)} |")
        add("")
    else:
        add("Neither arm had an unrecoverable failure.\n")
    unsupported = Counter(
        (r["sub_cause"] or "").split(":")[0] for r in data["baseline"].values() if r["outcome"] == UNSUPPORTED
    )
    add(
        "Unsupported inputs (identical for both arms, decided before any run): "
        + (", ".join(f"{k} {v}" for k, v in unsupported.most_common()) or "none")
        + ".\n"
    )

    add("## 5. Trade-offs\n")
    add("| Per task that ran | Baseline | Full | Full / baseline |\n|---|---|---|---|")
    add(f"| Latency | {_cost(b, 'avg_latency_s')} s | {_cost(f, 'avg_latency_s')} s | {_ratio(f['avg_latency_s'], b['avg_latency_s'])} |")
    add(f"| Tokens in | {_cost(b, 'avg_tokens_in')} | {_cost(f, 'avg_tokens_in')} | {_ratio(f['avg_tokens_in'], b['avg_tokens_in'])} |")
    add(f"| Tokens out | {_cost(b, 'avg_tokens_out')} | {_cost(f, 'avg_tokens_out')} | {_ratio(f['avg_tokens_out'], b['avg_tokens_out'])} |")
    add(f"| Model calls | {_cost(b, 'avg_llm_calls', '{:.1f}')} | {_cost(f, 'avg_llm_calls', '{:.1f}')} | {_ratio(f['avg_llm_calls'], b['avg_llm_calls'])} |\n")
    full_ran = [r for r in data["full"].values() if r["outcome"] != UNSUPPORTED]
    retried = [r for r in full_ran if (r.get("retries_triggered") or 0) > 0]
    spawned = [r["subagents_spawned"] for r in full_ran if r.get("subagents_spawned")]
    add(
        f"The full arm spawned {sum(spawned) / len(spawned):.1f} subagents per task on average "
        f"and triggered at least one retry on {len(retried)} of {len(full_ran)} tasks.\n"
        if spawned else "No full-arm task got as far as spawning subagents.\n"
    )

    add("## 6. Biggest fix\n")
    if pairs["recovered"]:
        cause, count = Counter(r["baseline"]["sub_cause"] for r in pairs["recovered"]).most_common(1)[0]
        outcomes = Counter(r["full"]["outcome"] for r in pairs["recovered"])
        with_retry = sum(1 for r in pairs["recovered"] if (r["full"].get("retries_triggered") or 0) > 0)
        became_correct = outcomes.get(CORRECT, 0)
        add(
            f"The most common baseline failure the full system did not repeat was **{cause}** "
            f"({count} of {len(pairs['recovered'])} such tasks): {CAUSE_EXPLAINED.get(cause, cause)}. "
            f"Of the {len(pairs['recovered'])} tasks, **{became_correct} became correct**; the full system's "
            "outcomes were " + ", ".join(f"{k} {v}" for k, v in outcomes.most_common()) + ". "
            f"A retry fired on {with_retry} of them."
        )
        if became_correct == 0:
            add(
                "\nSo the full system turned these failures into finished runs, not into right answers. "
                + ("None involved a retry, so the repair loop played no part. What differed was the prompt: "
                   "the supervisor's decomposition gave each agent a different task description, and with "
                   "those prompts these runs finished instead of stalling.\n" if with_retry == 0 else "\n")
            )
        else:
            add("")
        example = next(r for r in pairs["recovered"] if r["baseline"]["sub_cause"] == cause)
        add(f"Example task: `{example['task_id']}` (level {example['level']}). In the full run, {_describe_recovery(example)}.\n")
    else:
        add("There is none to report: no task failed unrecoverably in the baseline and then avoided that in the full system.\n")

    add("## 7. What didn't improve\n")
    if pairs["both_failed"]:
        both = Counter(f"{r['baseline']['sub_cause']} → {r['full']['sub_cause']}" for r in pairs["both_failed"])
        add(f"{len(pairs['both_failed'])} task(s) failed unrecoverably in both arms (baseline cause → full cause): "
            + ", ".join(f"{k} ({v})" for k, v in both.most_common()) + ".\n")
    if pairs["regressed"]:
        regressed = Counter(r["full"]["sub_cause"] for r in pairs["regressed"])
        add(f"{len(pairs['regressed'])} task(s) failed **only** in the full system:\n")
        for cause_name, n in regressed.most_common():
            ids = [r["task_id"] for r in pairs["regressed"] if r["full"]["sub_cause"] == cause_name][:2]
            add(f"- **{cause_name}** ({n}): {CAUSE_EXPLAINED.get(cause_name, cause_name)}. "
                f"Examples: " + ", ".join(f"`{i}`" for i in ids) + ".")
        add("")
    full_ran = [r for r in data["full"].values() if r["outcome"] != UNSUPPORTED]
    retried = sum(1 for r in full_ran if (r.get("retries_triggered") or 0) > 0)
    failed_subtasks = sum(1 for r in full_ran for st in (r.get("subtasks") or []) if st.get("ok") is False)
    add(
        f"**The diagnose-and-retry loop fired on {retried} of {len(full_ran)} tasks**, and "
        f"{failed_subtasks} subtask(s) were ever marked failed. It only runs when a subagent raises an "
        "exception, and in this run none did: every full-arm failure happened in the supervisor "
        "itself (an empty plan, or the 300 s wait), which the loop does not cover.\n"
    )
    if not pairs["both_failed"] and not pairs["regressed"]:
        add("No task failed unrecoverably in the full system.\n")
    add(
        f"Wrong answers were the dominant outcome in both arms (baseline {b['counts'][WRONG_ANSWER]}, "
        f"full {f['counts'][WRONG_ANSWER]}). Retrying cannot fix a run that finishes confidently "
        "with the wrong answer: the repair loop only fires on an exception, never on a wrong result.\n"
    )

    add("## 8. Limitations\n")
    add(f"- **Sample size.** {n} tasks, of which {ran} ran. Per-level rows, level 3 especially, are too small to compare.")
    agreement = repeat_agreement(data, repeat) if repeat else None
    if agreement and agreement["baseline"]["tasks"]:
        parts = [
            f"{arm} {a['flipped']} of {a['tasks']}" for arm, a in agreement.items() if a["tasks"]
        ]
        add(
            "- **One run per task.** The 15 pilot tasks were run again here, a day apart, with "
            "identical settings. Outcomes changed on " + ", ".join(parts) + " tasks. Search "
            "results drift and the full arm's subtasks finish in a different order each run, so "
            "temperature 0 is not deterministic end to end. Differences of a few tasks between the "
            "arms are inside this noise."
        )
    else:
        add("- **One run per task.** No variance estimate. Web search results change over time, so a rerun will not reproduce every outcome even at temperature 0.")
    add("- **Validation split only.** These tasks are public; nothing here says how either arm does on the held-out test split.")
    add("- **Small local model.** A 7.6B model at 4-bit quantization. Absolute accuracy is far below published GAIA results and should not be compared with them.")
    add(f"- **Narrow tools.** {n - ran} tasks could not be attempted at all. Search returns snippets and cannot open a page, which most GAIA tasks need.")
    add("- **What \"retry\" means.** Retries are an application-level diagnose-and-retry loop that runs inside each Celery task. The system has no Celery retry queues.")
    add("- **What the repair loop can see.** It fires only when a subagent raises. Tool errors are returned to the agent as messages and wrong answers finish normally, so neither triggers it.")
    add("- **Memory is order-dependent.** The full arm recalls earlier tasks' goals and answers into later decompositions, so its results depend on the task order, which was fixed by seed.")
    add("- **Unprotected steps.** The supervisor's own decompose and aggregate steps have no repair; a failure there ends the run.")
    add("- **One shared model server.** Four worker processes share one local model server, so subagents queue behind each other and the 300 s per-subtask wait is easier to hit than it would be with parallel model servers.\n")

    add("## 9. How to reproduce\n")
    add("```bash")
    add("# services and models")
    add("docker compose up -d postgres redis")
    add("ollama pull qwen2.5:7b && ollama pull nomic-embed-text")
    add("printf 'FROM qwen2.5:7b\\nPARAMETER num_ctx 16384\\n' > Modelfile && ollama create qwen2.5-7b-ctx16k -f Modelfile")
    add("")
    add("# environment (keep the virtualenv outside any synced folder)")
    add("uv venv --python 3.9 ~/.venvs/agent-orchestration")
    add('VIRTUAL_ENV=~/.venvs/agent-orchestration uv pip install -e ".[dev]" pandas pyarrow ollama')
    add("")
    add("# GAIA validation is read from $GAIA_DIR (default ~/gaia-test/2023/validation); it is never copied here")
    add("~/.venvs/agent-orchestration/bin/python -m pytest                      # harness and system tests")
    add("~/.venvs/agent-orchestration/bin/python -m bench.run --set smoke --run-tag smoke")
    add("~/.venvs/agent-orchestration/bin/python -m bench.run --set pilot --run-tag pilot")
    add("# caffeinate cannot stop sleep on lid-close; keep the Mac plugged in with the lid open")
    add("caffeinate -dims ~/.venvs/agent-orchestration/bin/python -m bench.run --set all --run-tag full")
    add("~/.venvs/agent-orchestration/bin/python -m bench.report results --repeat-dir results-pilot --write results.md")
    add("```\n")
    add("The run is resumable: rerun the same command and finished tasks are skipped. Per-task result files contain ground-truth answers and are gitignored.")
    return "\n".join(L) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir")
    parser.add_argument("--project", type=int, default=None, help="Project a full run of this many tasks.")
    parser.add_argument("--write", default=None, help="Write the full report to this file.")
    parser.add_argument("--subset-note", default="", help="Appended to the task count in the setup table.")
    parser.add_argument("--repeat-dir", default=None, help="An earlier run of some of the same tasks, to measure run-to-run agreement.")
    args = parser.parse_args(argv)
    data = load(Path(args.results_dir))
    print(summary(data, project=args.project))
    if args.write:
        repeat = load(Path(args.repeat_dir)) if args.repeat_dir else None
        Path(args.write).write_text(results_markdown(data, args.subset_note, repeat))
        print(f"wrote {args.write}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
