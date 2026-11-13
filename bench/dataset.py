"""Load GAIA validation from where it already sits, and decide what is runnable.

The ground truth is loaded here because the scorer needs it, but it never
leaves the driver process: the prompt built for the agent contains only the
question, the answer-format instruction, and the name of any attachment.
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from bench.config import GAIA_DIR, SEED

# Perception the system has no tool for. Matched against the annotators' own
# list of tools needed for the task, decided before any run.
PERCEPTION_KEYWORDS = ("video", "audio", "speech", "image", "ocr", "vision", "color recognition")

ANSWER_FORMAT = (
    "Finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. "
    "YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated "
    "list of numbers and/or strings. If you are asked for a number, don't use comma to write "
    "your number neither use units such as $ or percent sign unless specified otherwise. "
    "If you are asked for a string, don't use articles, neither abbreviations (e.g. for "
    "cities), and write the digits in plain text unless specified otherwise. If you are "
    "asked for a comma separated list, apply the above rules depending of whether the "
    "element to be put in the list is a number or a string."
)


@dataclass(frozen=True)
class Task:
    task_id: str
    question: str
    level: int
    ground_truth: str
    file_name: str
    annotator_tools: str

    @property
    def attachment(self) -> Optional[Path]:
        return GAIA_DIR / self.file_name if self.file_name else None


def load_tasks(gaia_dir: Path = GAIA_DIR) -> List[Task]:
    import pandas as pd

    frame = pd.read_parquet(Path(gaia_dir) / "metadata.parquet")
    return [
        Task(
            task_id=row["task_id"],
            question=row["Question"],
            level=int(row["Level"]),
            ground_truth=str(row["Final answer"]),
            file_name=row["file_name"] or "",
            annotator_tools=str((row["Annotator Metadata"] or {}).get("Tools", "")),
        )
        for _, row in frame.iterrows()
    ]


def unsupported_reason(task: Task) -> Optional[str]:
    """Why the system cannot take this task at all, or None if it can.

    Two rules, both independent of how either arm behaves:
      1. the attachment is not UTF-8 text, which is all the file tool reads;
      2. the task needs perception (image, audio, video) the system has no tool for.
    """
    attachment = task.attachment
    if attachment is not None:
        if not attachment.exists():
            return "attachment_missing"
        try:
            attachment.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"attachment_not_text:{attachment.suffix.lstrip('.').lower()}"

    tools = task.annotator_tools.lower()
    for keyword in PERCEPTION_KEYWORDS:
        if keyword in tools:
            return f"needs_capability:{keyword.split()[0]}"
    return None


def build_prompt(task: Task) -> str:
    """The exact text both arms receive. No ground truth, no annotator notes."""
    parts = [task.question]
    if task.file_name:
        parts.append(
            f'The file attached to this question is in your workspace as "{task.file_name}". '
            "Read it with the read_file tool."
        )
    parts.append(ANSWER_FORMAT)
    return "\n\n".join(parts)


def select(tasks: List[Task], name: str, seed: int = SEED) -> List[Task]:
    """A deterministic task list.

    all          every task, in one seeded order
    pilot        5 runnable tasks per level
    smoke        2 runnable level-1 tasks
    balanced:N   N runnable tasks per level, plus every unsupported task
    """
    rng = random.Random(seed)
    ordered = sorted(tasks, key=lambda t: t.task_id)
    rng.shuffle(ordered)
    if name == "all":
        return ordered

    runnable = [t for t in ordered if unsupported_reason(t) is None]

    def per_level(count: int, levels=(1, 2, 3)) -> List[Task]:
        picked = []
        for level in levels:
            picked.extend([t for t in runnable if t.level == level][:count])
        return picked

    if name == "smoke":
        return per_level(2, levels=(1,))
    if name == "pilot":
        chosen = {t.task_id for t in per_level(5)}
    elif name.startswith("balanced:"):
        chosen = {t.task_id for t in per_level(int(name.split(":", 1)[1]))}
        chosen |= {t.task_id for t in ordered if unsupported_reason(t) is not None}
    else:
        raise ValueError(f"Unknown task set {name!r}")
    # Keep the one seeded order, so a subset is a prefix-stable slice of "all".
    return [t for t in ordered if t.task_id in chosen]
