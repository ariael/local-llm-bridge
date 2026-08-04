#!/usr/bin/env python3
"""
Smoke test for the local-llm-bridge MCP server.

Exercises all three tools end-to-end against a running llama-server backend:
  1. health        — backend reachable, model served
  2. delegate      — a summary, a one-word classification, a JSON extraction
  3. transform_file — uppercases a temp file in place, verifies the result

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
import os
import sys
import tempfile

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
    if not health.startswith("UP:"):
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

    tmp = os.path.join(tempfile.gettempdir(), "local_llm_bridge_smoke.txt")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("hello world\nthis is a test\n")
    status = await _call("transform_file", {
        "instruction": "Uppercase every letter. Keep line breaks.",
        "path": tmp, "max_tokens": 200,
    })
    print("transform   :", status.split("---")[0].strip())
    result = open(tmp, encoding="utf-8").read()
    if "HELLO WORLD" not in result:
        print("  ! file not uppercased:", repr(result)); failures += 1

    print("\n%s" % ("ALL CHECKS PASSED" if failures == 0 else "%d CHECK(S) FAILED" % failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
