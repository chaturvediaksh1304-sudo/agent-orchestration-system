"""CLI entrypoint: a goal in, an aggregated result out.

Exit codes are distinct so the escalation flow is scriptable:
  0  the run completed
  1  a usage or configuration error
  2  the run was aborted (by a human, or unattended)
  3  the run is halted awaiting a human response
"""

import argparse
import sqlite3
import sys
import uuid
from pathlib import Path
from typing import List, Optional

from agent_orchestration import auth
from agent_orchestration.escalation import abort_handler, interactive_handler
from agent_orchestration.llm import DEFAULT_MODELS, build_llm
from agent_orchestration.memory import ConversationMemory
from agent_orchestration.state import EscalationDecision
from agent_orchestration.store import Store
from agent_orchestration.supervisor import build_supervisor
from agent_orchestration.tools import build_tools

EXIT_OK, EXIT_ERROR, EXIT_ABORTED, EXIT_PENDING = 0, 1, 2, 3


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orchestrate",
        description="Decompose a goal, dispatch subagents to do it, and report the result.",
    )
    parser.add_argument(
        "goal",
        nargs="?",
        help="What you want accomplished, or one of: login, logout, whoami.",
    )
    parser.add_argument(
        "--provider",
        default="anthropic",
        help=f"Model provider ({', '.join(DEFAULT_MODELS)}). Default: anthropic.",
    )
    parser.add_argument("--model", default=None, help="Override the provider's default model.")
    parser.add_argument(
        "--workspace",
        default="workspace",
        help="Directory the subagents' file tools are confined to. Default: ./workspace",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="PostgreSQL connection string. Defaults to $DATABASE_URL.",
    )
    parser.add_argument(
        "--memory", default=".orchestration/chroma", help="ChromaDB conversation-memory directory."
    )
    parser.add_argument(
        "--checkpoints",
        default=".orchestration/checkpoints.db",
        help="SQLite checkpoint database, used by --durable.",
    )
    parser.add_argument(
        "--no-memory", action="store_true", help="Run without persistent memory."
    )
    parser.add_argument(
        "--queue",
        action="store_true",
        help="Dispatch subagents to Celery workers. Requires Redis and a running worker.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Self-repair attempts per subtask before escalating. Default: 3.",
    )

    escalation = parser.add_argument_group("escalation")
    escalation.add_argument(
        "--durable",
        action="store_true",
        help="On escalation, halt and persist instead of prompting. Resume later with --respond.",
    )
    escalation.add_argument(
        "--thread", default=None, help="Thread id for --durable. Generated if omitted."
    )
    escalation.add_argument(
        "--non-interactive",
        action="store_true",
        help="Never prompt; abort on escalation. Implied when stdin is not a terminal.",
    )
    escalation.add_argument(
        "--list-escalations", action="store_true", help="List escalations and exit."
    )
    escalation.add_argument(
        "--respond", type=int, metavar="ID", help="Answer a pending escalation and resume its run."
    )
    escalation.add_argument(
        "--action", choices=("retry", "skip", "abort"), help="The answer, with --respond."
    )
    escalation.add_argument(
        "--guidance", default="", help="Instructions for a --action retry."
    )
    return parser


def _checkpointer(path: Path):
    from langgraph.checkpoint.sqlite import SqliteSaver

    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))


def _report(result) -> int:
    # Note: no `out=sys.stdout` default — that binds at definition time and
    # ignores any later redirection of sys.stdout.
    for item in result.get("results", []):
        mark = " " if item.ok else "!"
        print(
            f"{mark} [{item.subtask.spec.role}] {item.subtask.description}", file=sys.stderr
        )

    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print(payload["detail"], file=sys.stderr)
        print(
            f"\nHalted awaiting a human. Respond with:\n"
            f"  --respond {payload['escalation_id']} --action retry|skip|abort",
            file=sys.stderr,
        )
        return EXIT_PENDING

    if result.get("status") == "aborted":
        print("Run aborted; no result produced.", file=sys.stderr)
        return EXIT_ABORTED

    print(result["final"])
    return EXIT_OK


def _login() -> int:
    """Device flow: show a code, wait for the human to approve it elsewhere."""
    if not auth.configured():
        print(
            "No OAuth provider configured. Set OAUTH_CLIENT_ID (see README).",
            file=sys.stderr,
        )
        return EXIT_ERROR

    try:
        device = auth.request_device_code()
        print(f"Open {device.verification_uri} and enter code: {device.user_code}")
        token = auth.poll_for_token(device)
        user_id = auth.fetch_identity(token)
    except auth.AuthError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_ERROR

    auth.save_credentials(auth.Credentials(access_token=token, user_id=user_id))
    print(f"Logged in as {user_id}.")
    return EXIT_OK


def _identity() -> Optional[str]:
    """Who is running this, or None when auth isn't configured."""
    credentials = auth.load_credentials()
    return credentials.user_id if credentials else None


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)

    if args.goal == "login":
        return _login()
    if args.goal == "logout":
        print("Logged out." if auth.clear_credentials() else "Not logged in.")
        return EXIT_OK
    if args.goal == "whoami":
        user_id = _identity()
        print(user_id or "Not logged in.")
        return EXIT_OK if user_id else EXIT_ERROR

    if args.durable and args.no_memory:
        # Without the store the escalation is never logged, so the halted run
        # has no id to respond to and can never be resumed.
        print(
            "--durable needs the escalation log; it cannot be combined with --no-memory.",
            file=sys.stderr,
        )
        return EXIT_ERROR

    if args.list_escalations:
        for item in Store(args.database_url).load_escalations():
            print(
                f"#{item.id}  [{item.status}]  thread={item.thread_id}  {item.goal}",
                file=sys.stdout,
            )
        return EXIT_OK

    if args.respond is not None:
        return _respond(args, Store(args.database_url))

    if not args.goal:
        print("A goal is required (or use --respond / --list-escalations).", file=sys.stderr)
        return EXIT_ERROR

    # Auth gates the run only when a provider is configured, so the tool stays
    # usable standalone — which is the whole reason it's CLI-first.
    user_id = _identity()
    if auth.configured() and user_id is None:
        print("Not logged in. Run `orchestrate login` first.", file=sys.stderr)
        return EXIT_ERROR

    # Validate the provider before opening the database, so a bad --provider
    # reports itself rather than surfacing as a connection error.
    try:
        llm = build_llm(args.provider, args.model)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_ERROR

    store = None if args.no_memory else Store(args.database_url)

    unattended = args.non_interactive or not sys.stdin.isatty()
    thread = args.thread or uuid.uuid4().hex

    result = build_supervisor(
        llm,
        build_tools(Path(args.workspace)),
        store=store,
        memory=None if args.no_memory else ConversationMemory(Path(args.memory)),
        use_queue=args.queue,
        workspace=Path(args.workspace),
        provider=args.provider,
        model=args.model,
        max_attempts=args.max_attempts,
        handler=abort_handler if unattended else interactive_handler,
        use_interrupt=args.durable,
        checkpointer=_checkpointer(Path(args.checkpoints)) if args.durable else None,
        thread_id=thread,
        user_id=user_id,
    ).invoke({"goal": args.goal}, config={"configurable": {"thread_id": thread}})

    return _report(result)


def _respond(args, store: Store) -> int:
    """Answer a halted run and resume it from its checkpoint."""
    from langgraph.types import Command

    escalation = store.get_escalation(args.respond)
    if escalation is None:
        print(f"No escalation #{args.respond}.", file=sys.stderr)
        return EXIT_ERROR
    if escalation.status != "pending":
        print(
            f"Escalation #{args.respond} is already {escalation.status}.", file=sys.stderr
        )
        return EXIT_ERROR
    if not args.action:
        print("--respond needs --action retry|skip|abort.", file=sys.stderr)
        return EXIT_ERROR

    try:
        llm = build_llm(args.provider, args.model)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_ERROR

    decision = EscalationDecision(action=args.action, guidance=args.guidance)
    result = build_supervisor(
        llm,
        build_tools(Path(args.workspace)),
        store=store,
        memory=None if args.no_memory else ConversationMemory(Path(args.memory)),
        use_queue=args.queue,
        workspace=Path(args.workspace),
        provider=args.provider,
        model=args.model,
        max_attempts=args.max_attempts,
        use_interrupt=True,
        checkpointer=_checkpointer(Path(args.checkpoints)),
        thread_id=escalation.thread_id,
    ).invoke(
        Command(resume=decision.model_dump()),
        config={"configurable": {"thread_id": escalation.thread_id}},
    )

    return _report(result)


if __name__ == "__main__":
    sys.exit(main())
