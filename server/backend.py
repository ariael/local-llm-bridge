#!/usr/bin/env python3
r"""
backend — on-demand lifecycle for the local llama-server.

The GPU is shared with games, so the model must NOT run as an always-on service.
Instead the backend is spun up only when Claude actually delegates work, and
torn down when idle or on request — freeing the card the rest of the time.

Behaviour
---------
* `ensure()` — called at the start of every delegation tool. If llama-server is
  already answering on the health port, returns immediately. Otherwise (and if
  autostart is on) it launches llama-server detached, waits for /health, and
  records the PID to a state file so a later session can stop it too.
* `stop()` — kills the backend we launched (by PID), freeing VRAM. Safe to call
  when nothing is running.
* An idle watchdog thread stops the backend after LOCAL_LLM_IDLE_STOP_S seconds
  with no delegation, so a forgotten backend doesn't sit on the GPU.
* `touch()` — bumps the idle timer; the tools call it on every use.

Everything keys off LLAMA_BASE (host+port parsed from it). The launch command
mirrors scripts/Start-LlamaServer.ps1.
"""

import os
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

LLAMA_BASE = os.environ.get("LOCAL_LLM_BASE", "http://127.0.0.1:8001/v1")
AUTOSTART = os.environ.get("LOCAL_LLM_AUTOSTART", "1") not in ("0", "false", "False", "")
IDLE_STOP_S = int(os.environ.get("LOCAL_LLM_IDLE_STOP_S", "600"))  # 0 disables
STARTUP_TIMEOUT_S = int(os.environ.get("LOCAL_LLM_STARTUP_TIMEOUT_S", "240"))

LLAMA_EXE = os.environ.get("LOCAL_LLM_LLAMA_EXE", r"C:\AI\llama.cpp\vulkan\llama-server.exe")
MODEL_PATH = os.environ.get("LOCAL_LLM_MODEL_PATH", r"C:\AI Models\unsloth\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf")
MODEL_ALIAS = os.environ.get("LOCAL_LLM_MODEL", "local-model")
CTX_SIZE = os.environ.get("LOCAL_LLM_CTX", "32768")

_PID_FILE = os.path.join(tempfile.gettempdir(), "local_llm_backend.pid")
_LOG_FILE = os.path.join(tempfile.gettempdir(), "local_llm_backend.log")

_lock = threading.Lock()
_last_used = 0.0
_watchdog_started = False


def _host_port():
    p = urlparse(LLAMA_BASE)
    return (p.hostname or "127.0.0.1", p.port or 8001)


def _health_ok(timeout=2):
    host, port = _host_port()
    try:
        with urllib.request.urlopen("http://%s:%d/health" % (host, port), timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _read_pid():
    try:
        with open(_PID_FILE) as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return None


def _write_pid(pid):
    try:
        with open(_PID_FILE, "w") as fh:
            fh.write(str(pid))
    except OSError:
        pass


def _clear_pid():
    try:
        os.remove(_PID_FILE)
    except OSError:
        pass


def touch():
    """Mark the backend as just-used (resets the idle-stop timer)."""
    global _last_used
    _last_used = time.time()


def status():
    """Return a short human-readable status line."""
    host, port = _host_port()
    up = _health_ok()
    pid = _read_pid()
    owned = " (ours, pid %d)" % pid if pid else " (started outside this tool)" if up else ""
    idle = ""
    if up and _last_used and IDLE_STOP_S:
        idle = ", idle %ds/%ds" % (int(time.time() - _last_used), IDLE_STOP_S)
    return "%s http://%s:%d%s%s" % ("UP" if up else "DOWN", host, port, owned, idle)


def _launch():
    """Start llama-server detached, so it survives this (MCP server) process."""
    host, port = _host_port()
    args = [
        LLAMA_EXE, "--model", MODEL_PATH, "--alias", MODEL_ALIAS,
        "--port", str(port), "--host", host, "--ctx-size", str(CTX_SIZE),
        "--flash-attn", "on", "--n-gpu-layers", "99", "--cache-reuse", "256",
    ]
    # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP so killing the MCP server does
    # not take the model down mid-request; we manage its lifetime explicitly.
    flags = 0
    if os.name == "nt":
        flags = 0x00000008 | 0x00000200
    logfh = open(_LOG_FILE, "w")
    proc = subprocess.Popen(args, stdout=logfh, stderr=subprocess.STDOUT,
                            creationflags=flags, close_fds=True)
    _write_pid(proc.pid)
    return proc.pid


def ensure():
    """Guarantee the backend is answering. Returns (ok, message)."""
    if _health_ok():
        _start_watchdog()
        touch()
        return True, "already up"
    if not AUTOSTART:
        return False, "backend is down and LOCAL_LLM_AUTOSTART is off — start scripts\\Start-LlamaServer.ps1"

    with _lock:
        if _health_ok():  # someone won the race
            touch()
            return True, "already up"
        if not os.path.isfile(LLAMA_EXE):
            return False, "llama-server not found: %s" % LLAMA_EXE
        if not os.path.isfile(MODEL_PATH):
            return False, "model not found: %s" % MODEL_PATH
        pid = _launch()
        started = time.time()
        while time.time() - started < STARTUP_TIMEOUT_S:
            if _health_ok():
                _start_watchdog()
                touch()
                return True, "started (pid %d) in %.0fs" % (pid, time.time() - started)
            time.sleep(2)
        return False, "backend did not become ready within %ds (see %s)" % (STARTUP_TIMEOUT_S, _LOG_FILE)


def stop(force=False):
    """Stop the backend we launched (frees the GPU). Returns a message.

    Only kills a process we started (tracked via the PID file). If the backend
    was started outside this tool (e.g. the PS script), refuses unless force.
    """
    pid = _read_pid()
    if pid is None:
        if _health_ok() and force:
            # No PID record but something is up and caller insists: kill by image.
            subprocess.run(["taskkill", "/IM", os.path.basename(LLAMA_EXE), "/F", "/T"],
                           capture_output=True)
            time.sleep(1)
            return "force-stopped by image name; GPU released" if not _health_ok() else "kill sent"
        if _health_ok() and not force:
            return "backend is up but was not started by this tool; call with force=true to kill it anyway"
        return "nothing to stop"
    # Kill the process tree so no child lingers.
    subprocess.run(["taskkill", "/PID", str(pid), "/F", "/T"], capture_output=True)
    _clear_pid()
    time.sleep(1)
    return "stopped (pid %d); GPU released" % pid if not _health_ok() else "kill sent to pid %d" % pid


def _start_watchdog():
    global _watchdog_started
    if _watchdog_started or IDLE_STOP_S <= 0:
        return
    _watchdog_started = True

    def _loop():
        global _watchdog_started
        while True:
            time.sleep(30)
            if not _health_ok():
                _watchdog_started = False
                return
            if _read_pid() is None:
                continue  # not ours to stop
            if _last_used and (time.time() - _last_used) > IDLE_STOP_S:
                stop()
                _watchdog_started = False
                return

    threading.Thread(target=_loop, name="backend-idle-watchdog", daemon=True).start()


if __name__ == "__main__":
    # Standalone CLI so a SessionEnd hook can free the GPU when Claude Code exits:
    #   python backend.py stop          (kills only the backend we started)
    #   python backend.py stop --force   (kill even if started elsewhere)
    #   python backend.py start | status
    import sys
    _action = sys.argv[1] if len(sys.argv) > 1 else "status"
    if _action == "stop":
        print(stop(force=("--force" in sys.argv)))
    elif _action == "start":
        ok, msg = ensure()
        print(("OK: " if ok else "FAILED: ") + msg)
    else:
        print(status())
