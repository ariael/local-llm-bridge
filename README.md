# local-llm-bridge

Expose a **local Qwen model running on the GPU** as MCP tools that an **online
Claude Code** session can call. Claude stays the planner; the local model does
the bounded, mechanical subtasks — saving Anthropic tokens.

This is the productionized "delegation" layer. It is a sibling to the existing
`C:\AI\local-llm\` stack, which does a *different* thing (runs Claude Code
itself on Qwen via `ANTHROPIC_BASE_URL` + LiteLLM). See
[Relationship to `C:\AI\local-llm`](#relationship-to-cailocal-llm).

---

## Architecture

```
online Claude Code  (real Anthropic API — planner/orchestrator)
   │  calls MCP tool: delegate / transform_file / health
   ▼
local-llm MCP server  (server/local_llm_mcp.py, stdio)
   │  OpenAI Chat Completions  (NO LiteLLM on this path)
   ▼
llama-server  :8001  (Vulkan build, scripts/Start-LlamaServer.ps1)
   ▼
Radeon RX 7900 XTX  →  Qwen A3B (GGUF)
```

Key point: the MCP server speaks the OpenAI protocol directly to llama-server,
so **LiteLLM is not needed here**. LiteLLM only exists in `C:\AI\local-llm` to
translate Anthropic↔OpenAI for the "run Claude *on* Qwen" path.

---

## The tools

| Tool | Returns | Use for |
|---|---|---|
| `delegate(task, input_text?, max_tokens?, temperature?)` | the model's answer (goes into Claude's context) | **compact** results: summaries, classification, field extraction to JSON, short rewrites, Q&A over a snippet |
| `transform_file(instruction, path, output_path?, ...)` | a short status line only | **bulky** output: generated code, whole-file rewrites, bulk reformatting, translating a file — the big text is written to disk and never enters Claude's context |
| `health()` | up/down + served model id | quick backend check |

### Why two tools — the token-saving rule

An MCP tool **result still costs Claude tokens** (it flows back into context).
So the saving is real only when the result is small:

- Use `delegate` when you *want* the answer back and it's short.
- Use `transform_file` when the output is large — Qwen writes it to disk, Claude
  gets only `OK: wrote N chars to <path>`. This is where the biggest token
  savings come from.

Good candidates to delegate: summarization, classification/labelling, JSON
extraction, log parsing, boilerplate/scaffold generation, mechanical
refactors, whole-file translation. Keep architecture, hard reasoning, and final
review on Claude.

---

## Setup

1. **Install the MCP server dependency** (only `mcp`; HTTP uses the stdlib):
   ```bash
   pip install -r server/requirements.txt
   ```

2. **Start the local backend** (leave it running in its own window):
   ```powershell
   .\scripts\Start-LlamaServer.ps1 -Commit
   ```
   Dry-run first (`-DryRun`, the default) to see the exact command. Update
   `-ModelPath` when the Qwen model version changes.

3. **Register the MCP server** with whatever project you want to delegate from:
   copy the block in [`.mcp.json.example`](.mcp.json.example) into that
   project's `.mcp.json`, or run:
   ```bash
   claude mcp add local-llm -- python C:\GitHub\local-llm-bridge\server\local_llm_mcp.py
   ```

4. **Verify** from a Claude Code session: call the `health` tool — it should
   report `UP: ... serving local-model`.

---

## Production status & backlog

Done:
- [x] MCP server with `delegate` / `transform_file` / `health` (`server/local_llm_mcp.py`)
- [x] Standalone persistent backend launcher (`scripts/Start-LlamaServer.ps1`)
- [x] Registration example (`.mcp.json.example`)

To do:
- [ ] **Model check/update** — confirm the current Qwen A3B version on the box
      (LM Studio / HF). As of 2026-08 there is no "Qwen3.8"; current A3B-class
      releases are `Qwen3-30B-A3B-Instruct-2507`, `Qwen3-Coder-30B-A3B`, and
      `Qwen3-Next-80B-A3B`. Pick one, download the GGUF, update `-ModelPath`,
      then re-run `C:\AI\local-llm\Test-ToolCalling.ps1` to confirm tool-calling.
- [ ] **Run the backend as a service** (Windows scheduled task at logon, or NSSM)
      so the MCP tools always have a backend without a manual window.
- [ ] **Smoke test** the full chain: `health`, then a `delegate` summarize, then
      a `transform_file` on a scratch file; confirm token footer looks sane.
- [ ] **Delegation policy** — optionally add a short skill / CLAUDE.md snippet in
      consumer projects telling Claude *when* to reach for these tools.
- [ ] Consider a `classify`/`extract_json` convenience tool if `delegate` prompts
      get repetitive.

---

## Relationship to `C:\AI\local-llm`

| | `C:\AI\local-llm` (existing) | this repo (`local-llm-bridge`) |
|---|---|---|
| What runs the agent loop | **Qwen** (Claude Code pointed at local model) | **Claude** (real API) |
| Local model's role | *is* the assistant | a callable worker tool |
| Needs LiteLLM | yes (Anthropic→OpenAI translation) | no (MCP server speaks OpenAI) |
| Launch | `local-ai -Commit` | `Start-LlamaServer.ps1 -Commit` + MCP registration |
| Token cost | ~zero (fully offline) | Claude tokens for planning + tool round-trips; Qwen does the bulk work |

Both share the same llama.cpp Vulkan build and model files. The benchmark
rationale (measure prefill, Vulkan > HIP) lives in
`C:\AI\local-llm\reports\phase2-summary.md`.
