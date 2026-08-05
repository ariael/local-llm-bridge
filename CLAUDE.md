# Working in this repo — local-model delegation policy

This project exposes a local Qwen model (on the GPU) as MCP delegation tools so an
**online Claude session pays zero Anthropic tokens** for bounded, mechanical work.
The local tokens are free. **Use them by default, not as a last resort.**

## Core rule: delegate reflexively (aggressive posture)

When a subtask fits the local model, route it there **without being asked** and
**without narrating a deliberation** — just do it, then verify the result. Being a
good orchestrator here means keeping the heavy, bulky, or repetitive work off your
own context and on the GPU.

### Send to the local model by default

- **Summarize / condense** a file, log, diff, or pasted blob.
- **Classify / label / triage** items; **extract fields to JSON** from text.
- **Rewrite / reformat / translate** boilerplate, docs, comments, config.
- **Bulk mechanical edits** across files in a folder (rename, restyle, convert).
- **Scaffold** files from a clear spec (multi-file is fine — this is the aggressive
  posture: attempt it locally first, then review).
- Anything where the output is **large** — write it to disk via `transform_file`
  or `run_local_agent`, so the bulk never enters your context.

### Keep for yourself

- Work needing **repo-wide judgement, architecture, or correctness under subtlety**.
- Anything that **touches this repo outside the sandbox with side effects** you
  can't cheaply verify (the agent is confined to `LOCAL_AGENT_ROOT`; that's on
  purpose — see below).
- Final review. **You always verify the local model's output** — an A3B model is
  fast and free but not reliable open-ended. Aggressive delegation is paid for by
  disciplined verification, never by trusting the result blind.

## Which tool

| Need | Tool |
|---|---|
| Compact answer back in context (summary, JSON, label) | `delegate` |
| **Many small items** (classify/summarize/extract a list) | `delegate_batch` |
| Large output → straight to a file, short status back | `transform_file` |
| Multi-step work in a sandbox (read/write/scaffold) | `run_local_agent` |
| Stage an input file into the sandbox | `copy_in` |
| Retrieve a result file from the sandbox | `copy_out` |
| Record if a result was usable (after verifying) | `mark_outcome` |
| See what's working / tokens saved | `local_stats` |

Prefer `delegate_batch` over a loop of `delegate` calls whenever you have several
similar items — one round-trip, one compact result array, far fewer context tokens.

## Moving files in and out of the sandbox

`run_local_agent` is hard-confined to `LOCAL_AGENT_ROOT` (default
`C:\AI\agent-sandbox`) and cannot see the rest of the disk — this is the primary
guardrail, don't try to work around it. Instead:

1. `copy_in(source_path, dest)` — stage the input(s) into the sandbox.
2. `run_local_agent(task=...)` — let the model work on them.
3. `copy_out(source, dest_path)` — pull the result back to where it belongs, then
   review it.

## Statistics — keep them meaningful

Every delegation is logged to `logs/local_llm_stats.jsonl` (tool, `kind`, task
snippet, status, tokens in/out, elapsed, estimated Anthropic tokens saved). This data
is how we tune the policy over time, so:

- **Pass a `kind`** on every delegation call ("summarize", "classify", "extract",
  "rewrite", "scaffold", …). It's what makes `by_kind` breakdowns — and thus "which
  task shapes work locally" — possible.
- **After you verify a result, call `mark_outcome`** (`accepted` / `rejected` /
  `redone`). Completion status alone doesn't tell us if the output was *good*; this
  is the signal that does. It's cheap and it's how the policy gets smarter.
- Check `local_stats` occasionally; if a `kind` shows a poor `accepted_rate`, stop
  routing that shape locally and note it here.
