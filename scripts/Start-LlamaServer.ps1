<#
.SYNOPSIS
    Starts llama-server (Vulkan build) as a persistent local inference backend
    for the local-llm-bridge MCP server.

.DESCRIPTION
    This is the PRODUCTION backend launcher for the delegation use case: an
    online Claude Code session calls the local-llm MCP tools, which talk to this
    llama-server on http://127.0.0.1:8001/v1. Unlike C:\AI\local-llm\
    Start-ClaudeCodeLocal.ps1, this script:

      * starts ONLY llama-server (no LiteLLM proxy, no `claude`) — the MCP server
        speaks OpenAI Chat Completions directly, so LiteLLM is not needed here;
      * is meant to stay running in its own window (or as a scheduled task), so
        the MCP tools always have a backend to call.

    -DryRun (default) prints the command and starts nothing. -Commit runs it.

.PARAMETER ModelPath
    Full path to the GGUF model file. Default points at the current Qwen A3B
    model. UPDATE this (and re-run Test-ToolCalling in C:\AI\local-llm) when
    swapping models.

.PARAMETER MmprojPath
    Optional. Full path to the multimodal projector GGUF (mmproj-*.gguf) that
    belongs to -ModelPath. Set this to enable vision input; leave empty for a
    text-only backend. The projector is model-specific — a mismatched mmproj
    will fail to load or produce garbage.

.PARAMETER LlamaServerExe
    Path to the Vulkan llama-server.exe (chosen over HIP per the Phase 2
    benchmark in C:\AI\local-llm\reports\phase2-summary.md).

.PARAMETER ContextSize
    Context window in tokens. 32768 is Claude Code's floor; the delegation tools
    send short prompts, but keep headroom. Default 32768.

.PARAMETER Port
    llama-server port. Must match LOCAL_LLM_BASE in .mcp.json. Default 8001.

.EXAMPLE
    .\Start-LlamaServer.ps1 -DryRun

.EXAMPLE
    .\Start-LlamaServer.ps1 -Commit

.NOTES
    Target : Windows PowerShell 5.1. No PS7 syntax.
#>

[CmdletBinding(DefaultParameterSetName = "DryRun")]
param(
    [string] $ModelPath = "C:\AI Models\unsloth\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",

    [string] $MmprojPath = "",

    [string] $ModelAlias = "local-model",

    [string] $LlamaServerExe = "C:\AI\llama.cpp\vulkan-b10660\llama-server.exe",

    [int] $ContextSize = 32768,

    [int] $Port = 8001,

    [Parameter(ParameterSetName = "DryRun")]
    [switch] $DryRun,

    [Parameter(ParameterSetName = "Commit")]
    [switch] $Commit
)

# --- Declare everything up front (project convention) -----------------------
$isCommitRun = $false
if ($PSCmdlet.ParameterSetName -eq "Commit") {
    $isCommitRun = $true
}

$modelFileExists      = $false
$mmprojRequested      = $false
$llamaServerExeExists = $false
$serverArgs           = @()
$quotedArgs           = @()
$argText              = ""
$commandText          = ""

if ([string]::IsNullOrWhiteSpace($MmprojPath) -eq $false) {
    $mmprojRequested = $true
}

# --- Validate inputs --------------------------------------------------------
if (Test-Path -LiteralPath $ModelPath) {
    $modelFileExists = $true
}
else {
    Write-Host "ERROR: Model file not found: $ModelPath" -ForegroundColor Red
    Write-Host "Update -ModelPath (the current Qwen A3B model may have a new version)." -ForegroundColor Yellow
    exit 1
}

if ($mmprojRequested -eq $true) {
    if (Test-Path -LiteralPath $MmprojPath) {
        Write-Host "Vision enabled via mmproj: $MmprojPath" -ForegroundColor Cyan
    }
    else {
        Write-Host "ERROR: mmproj file not found: $MmprojPath" -ForegroundColor Red
        exit 1
    }
}

if (Test-Path -LiteralPath $LlamaServerExe) {
    $llamaServerExeExists = $true
}
else {
    Write-Host "ERROR: llama-server.exe not found: $LlamaServerExe" -ForegroundColor Red
    exit 1
}

# --- Build the command ------------------------------------------------------
# --cache-reuse is critical for repeated agent-style calls: it reuses the KV
# cache for a shared prefix instead of recomputing it every request.
$serverArgs = @(
    "--model", $ModelPath,
    "--alias", $ModelAlias,
    "--port", $Port,
    "--host", "127.0.0.1",
    "--ctx-size", $ContextSize,
    "--flash-attn", "on",
    "--n-gpu-layers", "99",
    "--cache-reuse", "256"
)

if ($mmprojRequested -eq $true) {
    $serverArgs += @("--mmproj", $MmprojPath)
}

# Quote any argument containing whitespace so the printed command can be copied
# and run by hand. The real invocation splats $serverArgs and needs no quoting.
foreach ($serverArg in $serverArgs) {
    $argText = [string]$serverArg
    if ($argText -match '\s') {
        $quotedArgs += '"' + $argText + '"'
    }
    else {
        $quotedArgs += $argText
    }
}

$commandText = '"' + $LlamaServerExe + '" ' + ($quotedArgs -join " ")

# --- Dry run ----------------------------------------------------------------
if ($isCommitRun -eq $false) {
    Write-Host "=== DRY RUN - nothing will be started ===" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "llama-server command:" -ForegroundColor Cyan
    Write-Host "   $commandText"
    Write-Host ""
    Write-Host "Health check will be at: http://127.0.0.1:$Port/health" -ForegroundColor Cyan
    Write-Host "MCP server should use  : http://127.0.0.1:$Port/v1  (LOCAL_LLM_BASE)" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Re-run with -Commit to actually start it." -ForegroundColor Yellow
    exit 0
}

# --- Commit: run in the foreground so this window IS the server -------------
Write-Host "Starting llama-server on 127.0.0.1:$Port ..." -ForegroundColor Cyan
Write-Host "Leave this window open. Ctrl+C stops the backend." -ForegroundColor Yellow
Write-Host ""
& $LlamaServerExe @serverArgs
