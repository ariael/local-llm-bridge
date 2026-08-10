#!/usr/bin/env python3
"""
Smoke test for the local-llm-bridge MCP server.

Exercises the tools end-to-end against a running llama-server backend:
  1. health         — backend reachable, model served
  2. delegate       — a summary, a one-word classification
  3. delegate_batch — three classifications in one call
  4. transform_file — uppercases a temp file in place, verifies the result
  5. run_local_agent + copy_in/copy_out — agent writes a file in the sandbox;
                       verify we can stage a file in and pull the result out
  6. mark_outcome + local_stats — record a verdict, read the rolled-up stats

Prerequisites:
  * llama-server running (scripts\\Start-LlamaServer.ps1 -Commit)
  * pip install -r server/requirements.txt

Run from the repo root:
  python scripts/smoke_test.py

Exit code 0 = all checks passed, 1 = something failed. Honors LOCAL_LLM_BASE /
LOCAL_LLM_MODEL / LOCAL_LLM_TIMEOUT the same way the server does.
"""

import asyncio
import importlib.util
import json
import os
import sys
import tempfile

# Isolate side effects BEFORE the server module is imported (it reads these at
# import time): a scratch sandbox for the agent, and a scratch stats log so the
# smoke test never pollutes the real telemetry.
_SANDBOX = os.path.join(tempfile.gettempdir(), "local_llm_smoke_sandbox")
os.makedirs(_SANDBOX, exist_ok=True)
# Force (not setdefault): keep the test hermetic even if the user points
# LOCAL_AGENT_ROOT at a real project — we don't want to write test files there.
os.environ["LOCAL_AGENT_ROOT"] = _SANDBOX
os.environ["LOCAL_LLM_STATS"] = os.path.join(tempfile.gettempdir(), "local_llm_smoke_stats.jsonl")

# Load the server module by path so this works regardless of cwd / packaging.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SERVER = os.path.join(_HERE, "..", "server", "local_llm_mcp.py")
_spec = importlib.util.spec_from_file_location("local_llm_mcp", _SERVER)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
SRV = _mod.mcp


def _text(result):
    """Pull the first text block out of an mcp CallToolResult."""
    content = getattr(result, "content", None) or []
    for block in content:
        if hasattr(block, "text"):
            return block.text
    return repr(result)


async def _call(name, args):
    return _text(await SRV.call_tool(name, args))


async def main():
    failures = 0

    health = await _call("health", {})
    print("health      :", health)
    if not health.startswith("UP"):
        print("  ! backend not up — start scripts\\Start-LlamaServer.ps1 -Commit")
        return 1

    summary = await _call("delegate", {
        "task": "Summarize in one sentence.",
        "input_text": "llama.cpp runs LLMs on CPU and GPU, including Vulkan and ROCm backends for AMD cards.",
        "max_tokens": 120,
    })
    print("summary     :", summary.splitlines()[0])
    if len(summary.split("---")[0].strip()) < 10:
        print("  ! empty/short summary"); failures += 1

    classify = await _call("delegate", {
        "task": "Sentiment as exactly one word (positive/negative/neutral). Output only the word.",
        "input_text": "This finally works and it is fast.",
        "max_tokens": 10, "temperature": 0.0,
    })
    label = classify.split("---")[0].strip().lower()
    print("classify    :", label)
    if "positive" not in label:
        print("  ! expected 'positive'"); failures += 1

    batch = await _call("delegate_batch", {
        "shared_instruction": "Sentiment as exactly one word (positive/negative/neutral). Output only the word.",
        "tasks": ["I love it.", "This is terrible.", "It is a chair."],
        "max_tokens": 8, "temperature": 0.0, "kind": "classify",
    })
    try:
        bdata = json.loads(batch)
        labels = [str(r.get("result", "")).lower() for r in bdata.get("results", [])]
        print("batch       :", labels, "(failed=%s)" % bdata.get("failed"))
        if bdata.get("n") != 3 or bdata.get("failed"):
            print("  ! batch did not process all 3 items"); failures += 1
    except Exception as exc:
        print("  ! batch result not JSON:", exc, batch[:120]); failures += 1

    tmp = os.path.join(tempfile.gettempdir(), "local_llm_bridge_smoke.txt")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("hello world\nthis is a test\n")
    status = await _call("transform_file", {
        "instruction": "Uppercase every letter. Keep line breaks.",
        "path": tmp, "max_tokens": 200, "kind": "reformat",
    })
    print("transform   :", status.split("---")[0].strip())
    result = open(tmp, encoding="utf-8").read()
    if "HELLO WORLD" not in result:
        print("  ! file not uppercased:", repr(result)); failures += 1

    # copy_in -> run_local_agent -> copy_out round trip through the sandbox.
    src = os.path.join(tempfile.gettempdir(), "local_llm_smoke_input.txt")
    with open(src, "w", encoding="utf-8") as fh:
        fh.write("seed input\n")
    cin = await _call("copy_in", {"source_path": src, "dest": "input.txt"})
    print("copy_in     :", cin.split("(")[0].strip())
    if not cin.startswith("OK"):
        print("  ! copy_in failed"); failures += 1

    agent = await _call("run_local_agent", {
        "task": "Create a file named result.txt containing exactly the word DONE, then finish.",
        "max_steps": 6, "timeout_s": 120, "kind": "scaffold",
    })
    try:
        adata = json.loads(agent)
        print("agent       : status=%s files=%s" % (adata.get("status"), adata.get("files_changed")))
        if adata.get("status") not in ("done", "done_text", "max_steps"):
            print("  ! agent errored:", adata.get("summary")); failures += 1
    except Exception as exc:
        print("  ! agent result not JSON:", exc, agent[:120]); failures += 1

    # Pull whatever the agent created back out (result.txt if it made it).
    out = os.path.join(tempfile.gettempdir(), "local_llm_smoke_output.txt")
    if os.path.exists(out):
        os.remove(out)
    cout = await _call("copy_out", {"source": "result.txt", "dest_path": out, "overwrite": True})
    print("copy_out    :", cout.split("(")[0].strip())
    if cout.startswith("OK") and not os.path.isfile(out):
        print("  ! copy_out reported OK but file missing"); failures += 1

    # Record an outcome and confirm stats roll it up.
    await _call("mark_outcome", {"outcome": "accepted", "kind": "classify", "note": "smoke test"})
    stats_out = await _call("local_stats", {})
    try:
        sdata = json.loads(stats_out)
        print("stats       : calls=%s success_rate=%s saved=%s" % (
            sdata.get("calls"), sdata.get("success_rate"),
            sdata.get("est_anthropic_tokens_saved")))
        if not sdata.get("calls"):
            print("  ! stats show no calls"); failures += 1
        if sdata.get("quality", {}).get("with_outcome", 0) < 1:
            print("  ! outcome not recorded in stats"); failures += 1
    except Exception as exc:
        print("  ! stats not JSON:", exc, stats_out[:120]); failures += 1

    print("\n%s" % ("ALL CHECKS PASSED" if failures == 0 else "%d CHECK(S) FAILED" % failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
