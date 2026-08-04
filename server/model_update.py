#!/usr/bin/env python3
r"""
model_update - check for and download a newer GGUF of the tracked model.

Two operations, matching the project's dry-run/commit + consent conventions:

* check()  - READ-ONLY. Ask the Hugging Face API which file matches the tracked
              repo + quant, compare its LFS hash against a local manifest, and
              report up-to-date / update-available / not-tracked. Safe to run on
              a schedule and exposed as the `check_model_update` MCP tool.
* apply()  - DOWNLOADS a large file (~20 GB), so it is a deliberate action:
              dry-run by default (prints repo/file/size/dest), and only fetches
              on commit=True. Resumable (HTTP Range), verifies size and - unless
              skipped - the SHA-256 against HF's LFS etag, then records a
              manifest. Downloading is gated on the caller's explicit go-ahead.

Config (env, overridable per call):
  LOCAL_LLM_HF_REPO   HF repo id to track, e.g. "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF"
  LOCAL_LLM_HF_QUANT  substring selecting the file, e.g. "UD-Q4_K_XL"
  LOCAL_LLM_MODEL_PATH / LOCAL_LLM_MODEL_DIR  where models live (dir is derived)

Stdlib only - no huggingface_hub needed.
"""

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

HF_REPO = os.environ.get("LOCAL_LLM_HF_REPO", "")
HF_QUANT = os.environ.get("LOCAL_LLM_HF_QUANT", "UD-Q4_K_XL")
MODEL_PATH = os.environ.get("LOCAL_LLM_MODEL_PATH", r"C:\AI Models\unsloth\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf")
MODEL_DIR = os.environ.get("LOCAL_LLM_MODEL_DIR", "") or os.path.dirname(MODEL_PATH)
MANIFEST = os.path.join(MODEL_DIR, ".model_manifest.json")

STATE = os.path.join(MODEL_DIR, ".model_check.json")
CHECK_INTERVAL_DAYS = int(os.environ.get("LOCAL_LLM_UPDATE_CHECK_DAYS", "30"))

_API = "https://huggingface.co/api/models/%s"
_RESOLVE = "https://huggingface.co/%s/resolve/main/%s"


def _api_info(repo):
    with urllib.request.urlopen(_API % repo, timeout=20) as r:
        return json.load(r)


def _pick_file(info, quant):
    """Choose the single .gguf sibling matching the quant. Returns filename or raises."""
    ggufs = [s["rfilename"] for s in info.get("siblings", [])
             if s.get("rfilename", "").lower().endswith(".gguf")]
    matches = [g for g in ggufs if quant.lower() in g.lower()]
    if not matches:
        raise RuntimeError("no .gguf matching quant %r in %s (have: %s)"
                           % (quant, info.get("id"), ", ".join(ggufs[:8])))
    # Reject multi-part models for now - auto-download of split files is out of scope.
    split = [m for m in matches if "-of-" in m]
    if split:
        raise RuntimeError("matched a split/multi-part model (%s); download manually" % split[0])
    if len(matches) > 1:
        matches.sort(key=len)  # prefer the plainest name
    return matches[0]


def _head(url):
    """HEAD the resolve URL (following redirects) → (size, sha256_hex or None)."""
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        h = r.headers
        size = int(h.get("X-Linked-Size") or h.get("Content-Length") or 0)
        etag = (h.get("X-Linked-Etag") or h.get("ETag") or "").strip('"')
        sha = etag if len(etag) == 64 and all(c in "0123456789abcdef" for c in etag) else None
        return size, sha


def _load_manifest():
    try:
        with open(MANIFEST, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def remote_target(repo=None, quant=None):
    """Resolve what HF currently offers for the tracked repo+quant."""
    repo = repo or HF_REPO
    quant = quant or HF_QUANT
    if not repo:
        raise RuntimeError("LOCAL_LLM_HF_REPO is not set - nothing to track yet.")
    info = _api_info(repo)
    fname = _pick_file(info, quant)
    size, sha = _head(_RESOLVE % (repo, fname))
    return {"repo": repo, "filename": fname, "size": size, "sha256": sha,
            "lastModified": info.get("lastModified"), "url": _RESOLVE % (repo, fname)}


def check(repo=None, quant=None):
    """Read-only: is a newer model available? Returns a human-readable verdict."""
    try:
        rt = remote_target(repo, quant)
    except Exception as exc:
        return "ERROR: %s" % exc
    man = _load_manifest()
    size_gb = rt["size"] / 1e9
    head = "tracked %s :: %s (%.1f GB)" % (rt["repo"], rt["filename"], size_gb)
    if man is None:
        installed = os.path.basename(MODEL_PATH) if os.path.isfile(MODEL_PATH) else "(none)"
        return ("%s\nNO MANIFEST yet - current file on disk: %s.\n"
                "Run apply (commit) once to record a baseline and enable diffing." % (head, installed))
    same = (man.get("sha256") and rt["sha256"] and man["sha256"] == rt["sha256"]) or \
           (man.get("filename") == rt["filename"] and man.get("size") == rt["size"])
    if same:
        return "%s\nUP TO DATE (installed %s)." % (head, man.get("filename"))
    return ("%s\nUPDATE AVAILABLE:\n  installed: %s (%s)\n  remote:    %s (%.1f GB)\n"
            "Run apply with commit=true to download." %
            (head, man.get("filename"), man.get("sha256", "?")[:12], rt["filename"], size_gb))


def _read_state():
    try:
        with open(STATE, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def maybe_check(interval_days=None, force=False):
    """Run check() at most once per interval (default monthly) and cache it.

    This is how the MCP server self-schedules the update check WITHOUT an
    external scheduler: it calls this on startup; the state file throttles it to
    ~monthly no matter how often the server restarts. Non-fatal and returns a
    dict {verdict, checked_at, age_days, fresh}.
    """
    interval_days = CHECK_INTERVAL_DAYS if interval_days is None else interval_days
    st = _read_state()
    now = time.time()
    age_days = ((now - st["checked_at"]) / 86400.0) if (st and st.get("checked_at")) else None

    if force or age_days is None or age_days >= interval_days:
        if not HF_REPO:
            return {"verdict": "not tracked (LOCAL_LLM_HF_REPO unset)",
                    "checked_at": None, "age_days": None, "fresh": False}
        verdict = check()
        # Never throttle on an error (network blip, bad repo id) — otherwise a
        # transient failure would suppress checks for a whole month. Only a real
        # verdict gets cached; errors are returned but retried next start.
        if not verdict.startswith("ERROR"):
            try:
                with open(STATE, "w", encoding="utf-8") as fh:
                    json.dump({"checked_at": now, "verdict": verdict}, fh, indent=2)
            except OSError:
                pass
        return {"verdict": verdict, "checked_at": now, "age_days": 0.0, "fresh": True}

    return {"verdict": st["verdict"], "checked_at": st["checked_at"],
            "age_days": round(age_days, 1), "fresh": False}


def _download_resumable(url, dest_part, expected_size, log=sys.stderr):
    have = os.path.getsize(dest_part) if os.path.exists(dest_part) else 0
    if have and have >= expected_size:
        return
    req = urllib.request.Request(url)
    if have:
        req.add_header("Range", "bytes=%d-" % have)
    mode = "ab" if have else "wb"
    with urllib.request.urlopen(req, timeout=60) as r, open(dest_part, mode) as fh:
        done = have
        last = time.time()
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if time.time() - last > 5:
                log.write("  %.1f / %.1f GB (%.0f%%)\n" %
                          (done / 1e9, expected_size / 1e9, 100 * done / expected_size))
                log.flush()
                last = time.time()


def _sha256(path, log=sys.stderr):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def apply(commit=False, repo=None, quant=None, verify_sha=True):
    """Dry-run by default. commit=True downloads the newer model (resumable)."""
    try:
        rt = remote_target(repo, quant)
    except Exception as exc:
        return "ERROR: %s" % exc
    dest = os.path.join(MODEL_DIR, rt["filename"])
    size_gb = rt["size"] / 1e9

    if not commit:
        return ("DRY RUN - would download:\n  repo: %s\n  file: %s\n  size: %.1f GB\n"
                "  from: %s\n  to:   %s\nRe-run with commit=true to download." %
                (rt["repo"], rt["filename"], size_gb, rt["url"], dest))

    if os.path.isfile(dest) and os.path.getsize(dest) == rt["size"]:
        result = "already present"
    else:
        os.makedirs(MODEL_DIR, exist_ok=True)
        part = dest + ".part"
        _download_resumable(rt["url"], part, rt["size"])
        if os.path.getsize(part) != rt["size"]:
            return "ERROR: size mismatch after download (%d != %d)" % (os.path.getsize(part), rt["size"])
        if verify_sha and rt["sha256"]:
            got = _sha256(part)
            if got != rt["sha256"]:
                return "ERROR: sha256 mismatch (%s != %s) - kept .part for retry" % (got[:12], rt["sha256"][:12])
        os.replace(part, dest)
        result = "downloaded and verified"

    manifest = {"repo": rt["repo"], "filename": rt["filename"], "size": rt["size"],
                "sha256": rt["sha256"], "lastModified": rt["lastModified"],
                "installed_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return ("OK: %s -> %s\nTo use it, point LOCAL_LLM_MODEL_PATH at:\n  %s\n"
            "(then stop_backend so the next start loads the new model)." %
            (result, rt["filename"], dest))


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "check"
    if action == "check":
        print(check())
    elif action == "apply":
        print(apply(commit=("--commit" in sys.argv), verify_sha=("--no-sha" not in sys.argv)))
    else:
        print("usage: model_update.py [check | apply [--commit] [--no-sha]]")
