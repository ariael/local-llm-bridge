#!/usr/bin/env python3
r"""
stats — persistent, append-only telemetry for every local-model delegation.

Why
---
The whole point of the bridge is spending *free* local-GPU tokens instead of paid
Anthropic tokens. To steer that (and to tune the auto-delegation policy over time)
we need to know, per call: which tool ran, on what KIND of task, whether it
completed, whether the result was actually USEFUL, how many local tokens it burned,
how long it took — and a rough estimate of the Anthropic tokens it *saved*.

Two record types, both one-JSON-object-per-line in the same JSONL file:
* ``type:"call"``    — one delegation (delegate / transform_file / run_local_agent
                       / delegate_batch). Written by ``record()``.
* ``type:"outcome"`` — a verdict Claude attaches AFTER verifying the result
                       (accepted / rejected / redone). Written by ``record_outcome()``.
                       Paired to the most recent call by ``summary()``.

Why two levels? "status" only says the call *completed* (``ok``/``done``). Whether
the output was good enough to keep is a separate signal — the one that actually
answers "what works and what doesn't". Outcomes are optional; when absent, summary
falls back to completion status.

Honest token savings (see ``_INPUT_OFFLOADED``)
-----------------------------------------------
For ``transform_file`` / ``run_local_agent`` the input (a file, a multi-step
scratchpad) is NOT in Claude's context, so both prompt AND completion were
offloaded → saved = prompt + completion. For ``delegate`` / ``delegate_batch``
Claude already holds the input it passed in, so only the *completion* is truly
saved. Counting prompt+completion for the latter would flatter the number.

Design: never raises into the caller (telemetry must not break a tool); no deps.
Location: LOCAL_LLM_STATS env, else <repo>/logs/local_llm_stats.jsonl.
"""

import json
import os
import threading
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATS_PATH = os.environ.get(
    "LOCAL_LLM_STATS", os.path.join(_REPO_ROOT, "logs", "local_llm_stats.jsonl")
)

# Completion statuses that count as a usable result (vs. failure/degraded).
_SUCCESS = {"ok", "done", "done_text"}
# Outcome verdicts that count as "the result was actually kept".
_ACCEPTED = {"accepted", "good"}
# Tools whose INPUT is not already in Claude's context, so the prompt tokens are a
# genuine saving too (not just the completion). See module docstring.
_INPUT_OFFLOADED = {"transform_file", "run_local_agent"}

_lock = threading.Lock()


def _append(rec):
    """Serialize one record and append it. Best-effort — never raises."""
    try:
        line = json.dumps(rec, ensure_ascii=False)
        with _lock:
            os.makedirs(os.path.dirname(STATS_PATH), exist_ok=True)
            with open(STATS_PATH, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception:
        pass  # telemetry must never break the tool it is measuring


def record(tool, task, status, prompt_tokens=0, completion_tokens=0,
           elapsed_s=0.0, kind="", extra=None):
    """Append one delegation (``call``) record. Best-effort — never raises."""
    try:
        p = int(prompt_tokens or 0)
        c = int(completion_tokens or 0)
        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "type": "call",
            "tool": tool,
            "kind": kind or "",
            "task": (task or "")[:200],   # snippet only — no bulky payloads on disk
            "status": status,
            "success": status in _SUCCESS,
            "prompt_tokens": p,
            "completion_tokens": c,
            "elapsed_s": round(float(elapsed_s or 0.0), 1),
            # See _INPUT_OFFLOADED: only count prompt tokens as "saved" when the
            # input wasn't already sitting in Claude's context.
            "est_tokens_saved": (p + c) if tool in _INPUT_OFFLOADED else c,
        }
        if extra:
            rec["extra"] = extra
        _append(rec)
    except Exception:
        pass


def record_outcome(outcome, note="", kind=""):
    """Append an ``outcome`` verdict for the most recent call. Best-effort.

    outcome: accepted | rejected | redone (free text tolerated). ``summary`` pairs
    this with the last logged call to compute an acceptance rate — the real
    "did it actually work" signal, distinct from completion status.
    """
    try:
        _append({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "type": "outcome",
            "outcome": (outcome or "").strip().lower(),
            "note": (note or "")[:200],
            "kind": kind or "",
        })
    except Exception:
        pass


def _iter_records():
    if not os.path.isfile(STATS_PATH):
        return
    with open(STATS_PATH, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def summary(limit=None):
    """Aggregate the log into a compact dict: totals, per-tool and per-kind
    breakdowns, completion success rate, ACCEPTANCE rate (from outcomes), tokens
    saved, and the most recent calls.

    Args:
        limit: if set, only the last `limit` call records are considered (outcomes
            are paired within that window).
    """
    records = list(_iter_records())

    # Pair each outcome to the most recent preceding call (last verdict wins).
    calls = []
    last_call = None
    for r in records:
        if r.get("type") == "outcome":
            if last_call is not None:
                last_call["_outcome"] = r.get("outcome", "")
        else:
            calls.append(r)
            last_call = r

    if limit:
        calls = calls[-int(limit):]

    if not calls:
        return {"calls": 0, "note": "no local-model calls logged yet",
                "stats_path": STATS_PATH}

    by_tool = {}
    by_kind = {}
    by_status = {}
    outcomes = {}
    ok = tok_in = tok_out = saved = 0
    with_outcome = accepted = 0
    total_time = 0.0

    for r in calls:
        t = r.get("tool", "?")
        bt = by_tool.setdefault(t, {"calls": 0, "success": 0,
                                    "tokens_in": 0, "tokens_out": 0, "time_s": 0.0})
        bt["calls"] += 1
        bt["success"] += 1 if r.get("success") else 0
        bt["tokens_in"] += r.get("prompt_tokens", 0)
        bt["tokens_out"] += r.get("completion_tokens", 0)
        bt["time_s"] += r.get("elapsed_s", 0.0)

        k = r.get("kind") or "unlabeled"
        bk = by_kind.setdefault(k, {"calls": 0, "success": 0, "with_outcome": 0, "accepted": 0})
        bk["calls"] += 1
        bk["success"] += 1 if r.get("success") else 0

        st = r.get("status", "?")
        by_status[st] = by_status.get(st, 0) + 1

        ok += 1 if r.get("success") else 0
        tok_in += r.get("prompt_tokens", 0)
        tok_out += r.get("completion_tokens", 0)
        saved += r.get("est_tokens_saved", 0)
        total_time += r.get("elapsed_s", 0.0)

        oc = r.get("_outcome")
        if oc:
            outcomes[oc] = outcomes.get(oc, 0) + 1
            with_outcome += 1
            bk["with_outcome"] += 1
            if oc in _ACCEPTED:
                accepted += 1
                bk["accepted"] += 1

    for bt in by_tool.values():
        bt["success_rate"] = round(bt["success"] / bt["calls"], 3)
        bt["time_s"] = round(bt["time_s"], 1)
    for bk in by_kind.values():
        bk["success_rate"] = round(bk["success"] / bk["calls"], 3)
        if bk["with_outcome"]:
            bk["accepted_rate"] = round(bk["accepted"] / bk["with_outcome"], 3)

    n = len(calls)
    out = {
        "calls": n,
        "success_rate": round(ok / n, 3),          # completion: did the call finish cleanly
        "tokens_in": tok_in,
        "tokens_out": tok_out,
        "est_anthropic_tokens_saved": saved,
        "total_local_time_s": round(total_time, 1),
        "by_tool": by_tool,
        "by_kind": by_kind,
        "by_status": by_status,
        "recent": [
            {"ts": r.get("ts"), "tool": r.get("tool"), "kind": r.get("kind"),
             "status": r.get("status"), "outcome": r.get("_outcome"),
             "task": (r.get("task") or "")[:80]}
            for r in calls[-10:]
        ],
        "stats_path": STATS_PATH,
    }
    # Acceptance = the real "did it actually work" signal, only over calls that got
    # a verdict. Absent until Claude starts calling mark_outcome after verifying.
    if with_outcome:
        out["quality"] = {
            "with_outcome": with_outcome,
            "accepted": accepted,
            "accepted_rate": round(accepted / with_outcome, 3),
            "by_outcome": outcomes,
        }
    else:
        out["quality"] = {"with_outcome": 0,
                          "note": "no outcomes recorded yet — call mark_outcome after verifying results"}
    return out
