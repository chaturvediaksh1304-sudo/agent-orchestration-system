"""Tools available to subagents, confined to a workspace directory.

Every filesystem access routes through ``_resolve``. Paths here are authored by
a model, so that single choke point is the trust boundary for the whole module.
"""

from pathlib import Path
from typing import Dict

from langchain_core.tools import BaseTool, tool


def build_tools(workspace: Path) -> Dict[str, BaseTool]:
    """Build the tool registry bound to ``workspace``.

    Returned as a name -> tool mapping because the supervisor names tools as
    strings in each AgentSpec, and the subagent factory resolves them by name.
    """
    root = Path(workspace).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    def _resolve(path: str) -> Path:
        # resolve() collapses ".." and follows symlinks, so a link planted inside
        # the workspace can't be used to step outside it. No expanduser() here:
        # a "~" from the model stays a literal directory name inside the
        # workspace rather than reaching the real home directory.
        target = (root / path).resolve()
        if target != root and root not in target.parents:
            raise ValueError(
                f"Path {path!r} is outside the workspace and cannot be accessed."
            )
        return target

    @tool
    def read_file(path: str) -> str:
        """Read a UTF-8 text file from the workspace and return its contents."""
        return _resolve(path).read_text(encoding="utf-8")

    @tool
    def write_file(path: str, content: str) -> str:
        """Write UTF-8 text to a file in the workspace, creating parents as needed."""
        target = _resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to {path}."

    @tool
    def list_files() -> str:
        """List every file in the workspace, one workspace-relative path per line."""
        return "\n".join(
            sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
        )

    @tool
    def web_search(query: str) -> str:
        """Search the web and return the top results as title, URL, and snippet."""
        # Imported lazily so nothing loads a network client until a search runs.
        from ddgs import DDGS

        results = DDGS().text(query, max_results=5)
        return "\n\n".join(
            f"{r.get('title', '')}\n{r.get('href', '')}\n{r.get('body', '')}"
            for r in results
        ) or "No results."

    return {t.name: t for t in (read_file, write_file, list_files, web_search)}
