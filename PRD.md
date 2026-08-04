# Agent Orchestration System — PRD

## One-liner
A multi-agent orchestration system with tool use, persistent memory, and human-in-the-loop escalation — for developers who want a real, non-demo pattern for building autonomous agent systems.

## Problem
Most agent projects that get shared publicly are single-agent demos — one model, one prompt, one tool call, done. They don't prove anything about how production agent systems actually work. This project demonstrates supervisor delegation, persistent memory across sessions, and a real self-repair + escalation loop — the parts that separate a toy from something that could actually ship.

## Target user
Open-source adopters and other developers who want to study or fork a real multi-agent orchestration pattern — not end consumers.

## MVP scope (must-have)
- Supervisor agent that decomposes a goal and delegates to dynamically dispatched subagents
- Tool use across subagents
- Persistent memory: PostgreSQL (task history, agent configs) + ChromaDB (conversation memory)
- Async task queue via Redis + Celery
- Self-repair loop: failing subagents auto-diagnose and retry/fix before escalating
- Human-in-the-loop escalation when self-repair fails repeatedly

## Explicitly out of scope (for now)
- None — full core loop ships in MVP, nothing held back for later phases.

## Success looks like
The system handles a real multi-step task end-to-end — supervisor delegates, subagents execute with tool use, memory persists across the run, and escalation fires correctly if something can't self-repair.
