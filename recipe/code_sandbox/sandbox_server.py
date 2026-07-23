#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A minimal, docker-free code-execution sandbox compatible with verl's SandboxFusion client.

verl's reward path (``verl/utils/reward_score/sandbox_fusion``) POSTs one HTTP request per
test case to a sandbox service and reads back the program's stdout / status. The official
service is the docker image https://github.com/bytedance/SandboxFusion ; on hosts without
docker (or firejail/nsjail/bwrap) we instead run this lightweight FastAPI server, which speaks
the exact same wire protocol and executes each submission in a resource-limited subprocess.

Wire protocol (must stay in sync with ``sandbox_fusion/utils.py``):

  Request  POST /run_code  (JSON):
      { "compile_timeout": int, "run_timeout": int, "code": str, "stdin": str|null,
        "memory_limit_MB": int, "language": str, "files": {}, "fetch_files": [] }

  Response (JSON):
      { "status": "Success" | "Failed" | "SandboxError",
        "compile_result": { "status", "execution_time", "stderr", "return_code" } | null,
        "run_result":     { "status": "Finished" | "TimeLimitExceeded" | "Error",
                            "stdout", "stderr", "return_code", "execution_time" } | null }

The client (``_process_single_case``) decides pass/fail itself by comparing
``run_result.stdout.rstrip("\\n")`` to the expected output. This server never sees the
expected output; it only runs code and reports what happened. ``fn_name`` (call-based)
problems are wrapped into a plain stdin->stdout ``__main__`` by the *client* before they reach
us, so no call-based logic is needed here.

Isolation here is best-effort (rlimits + timeout + optional privilege drop), NOT a security
boundary as strong as a container. Run it only on trusted training infrastructure.
"""

from __future__ import annotations

import argparse
import os
import resource
import signal
import subprocess
import sys
import tempfile
import time
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

# -----------------------------------------------------------------------------
# Configuration (populated from CLI / env in main(); read by request handlers)
# -----------------------------------------------------------------------------
PYTHON_INTERPRETER = os.environ.get("SANDBOX_PYTHON", sys.executable)
DROP_PRIVILEGES = os.environ.get("SANDBOX_DROP_PRIVILEGES", "false").lower() == "true"
# Fast-start submissions with `python -S` (skip the `site` module). On an anaconda
# interpreter that auto-loads hundreds of packages, `-S` cuts cold-start from ~83ms
# to ~31ms (~2.7x), which dominates wall-clock for short test cases. Trade-off:
# `-S` makes THIRD-PARTY packages (numpy, sortedcontainers, ...) unimportable; the
# stdlib (json/math/collections/itertools/re/sys/...) still imports fine. Set
# SANDBOX_SITE=1 to keep the full site (third-party importable) at the cost of speed.
USE_SITE = os.environ.get("SANDBOX_SITE", "0").lower() in ("1", "true")
# anyio threadpool size: Starlette dispatches sync `def` endpoints onto this pool,
# whose default cap is 40 — a per-worker concurrency ceiling. Raise it so a single
# worker can hold many in-flight subprocess executions (each releases the GIL while
# waiting on communicate()). Sized in main()/startup from cpu count.
THREAD_LIMIT = int(os.environ.get("SANDBOX_THREAD_LIMIT", "0") or 0)
# uid/gid to drop to when DROP_PRIVILEGES is on and we are root. 65534 == nobody/nogroup.
NOBODY_UID = int(os.environ.get("SANDBOX_NOBODY_UID", "65534"))
NOBODY_GID = int(os.environ.get("SANDBOX_NOBODY_GID", "65534"))
# Hard ceiling on captured stdout/stderr bytes returned to the client (avoid OOM on the driver).
MAX_OUTPUT_BYTES = int(os.environ.get("SANDBOX_MAX_OUTPUT_BYTES", str(1 * 1024 * 1024)))
# Absolute cap on memory_limit_MB the client may request (defensive).
MAX_MEMORY_LIMIT_MB = int(os.environ.get("SANDBOX_MAX_MEMORY_LIMIT_MB", str(8 * 1024)))

app = FastAPI(title="verl-code-sandbox", version="1.0")


@app.on_event("startup")
def _raise_thread_limit():
    """Raise the anyio threadpool cap so one worker can run many cases concurrently.

    Starlette runs sync `def` endpoints (like /run_code) on anyio's default thread
    limiter, capped at 40. Each case spends almost all its time blocked in
    subprocess.communicate() (GIL released), so a higher cap lets a single worker
    keep many subprocesses in flight. Defaults to max(256, cpu*4) unless
    SANDBOX_THREAD_LIMIT overrides it.
    """
    limit = THREAD_LIMIT if THREAD_LIMIT > 0 else max(256, (os.cpu_count() or 8) * 4)
    try:
        import anyio.to_thread

        anyio.to_thread.current_default_thread_limiter().total_tokens = limit
    except Exception:
        pass


class RunCodeRequest(BaseModel):
    code: str
    stdin: Optional[str] = None
    language: str = "python"
    run_timeout: int = 10
    compile_timeout: int = 10
    memory_limit_MB: int = 1024
    # Accepted for protocol compatibility; not used by this minimal sandbox.
    files: dict = {}
    fetch_files: list = []


def _make_preexec_fn(memory_limit_mb: int, cpu_seconds: int):
    """Build a child-process pre-exec hook that sandboxes the subprocess.

    Runs in the forked child *before* exec. Sets a new session (so a timeout can kill the
    whole process group), applies rlimits, and optionally drops root to an unprivileged uid.
    """

    def _preexec():
        # New session/process-group: lets us SIGKILL the entire tree on timeout.
        os.setsid()

        # Address-space (virtual memory) limit — the main defense against runaway allocation.
        if memory_limit_mb and memory_limit_mb > 0:
            mem_bytes = memory_limit_mb * 1024 * 1024
            try:
                resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            except (ValueError, OSError):
                pass

        # CPU-time limit (seconds): backstop for wall-clock timeout against busy loops that
        # somehow evade the subprocess timeout (e.g. ignoring signals).
        if cpu_seconds and cpu_seconds > 0:
            try:
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
            except (ValueError, OSError):
                pass

        # Cap the size of any file the program writes (also limits fd-based output blowups).
        try:
            fsize = max(MAX_OUTPUT_BYTES * 4, 16 * 1024 * 1024)
            resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
        except (ValueError, OSError):
            pass

        # Limit number of processes/threads the submission may spawn (fork-bomb guard).
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (256, 256))
        except (ValueError, OSError):
            pass

        # Disable core dumps.
        try:
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        except (ValueError, OSError):
            pass

        # Drop privileges last (after rlimits, which may need privilege on some systems).
        if DROP_PRIVILEGES and os.getuid() == 0:
            try:
                os.setgid(NOBODY_GID)
                os.setuid(NOBODY_UID)
            except OSError:
                pass

    return _preexec


def _truncate(data: bytes) -> str:
    """Decode + truncate captured output so a pathological program can't OOM the caller."""
    if data is None:
        return ""
    if len(data) > MAX_OUTPUT_BYTES:
        data = data[:MAX_OUTPUT_BYTES]
        suffix = b"\n...[truncated]"
        return (data + suffix).decode("utf-8", errors="replace")
    return data.decode("utf-8", errors="replace")


def _run_python(req: RunCodeRequest) -> dict:
    """Execute a python submission in an isolated subprocess and map to the wire response."""
    run_timeout = max(1, int(req.run_timeout or 10))
    memory_limit_mb = min(int(req.memory_limit_MB or 1024), MAX_MEMORY_LIMIT_MB)
    stdin_bytes = (req.stdin or "").encode("utf-8")

    # Write the code to a temp file in its own scratch dir; cwd is set there so any files the
    # program creates land in a directory we delete afterwards.
    with tempfile.TemporaryDirectory(prefix="sbx_") as workdir:
        code_path = os.path.join(workdir, "main.py")
        with open(code_path, "w", encoding="utf-8") as fh:
            fh.write(req.code)

        # CPU rlimit a touch above the wall-clock timeout so the subprocess `timeout=` is the
        # primary control and RLIMIT_CPU is only a backstop.
        preexec = _make_preexec_fn(memory_limit_mb, run_timeout + 2)
        # Default to `python -S` (skip site) for a ~2.7x faster cold start; SANDBOX_SITE=1
        # restores the full site so submissions that import third-party packages work.
        cmd = [PYTHON_INTERPRETER, code_path] if USE_SITE else [PYTHON_INTERPRETER, "-S", code_path]
        start = time.monotonic()
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=workdir,
                preexec_fn=preexec,
                env={
                    "PATH": "/usr/bin:/bin",
                    "PYTHONIOENCODING": "utf-8",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    # Keep the submission from importing the trainer's site-packages by accident.
                    "HOME": workdir,
                    "TMPDIR": workdir,
                },
            )
            try:
                stdout, stderr = proc.communicate(input=stdin_bytes, timeout=run_timeout)
                duration = time.monotonic() - start
            except subprocess.TimeoutExpired:
                # Kill the whole process group (setsid above), then reap.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.communicate()
                duration = time.monotonic() - start
                return {
                    "status": "Failed",
                    "compile_result": None,
                    "run_result": {
                        "status": "TimeLimitExceeded",
                        "stdout": "",
                        "stderr": f"Execution timed out after {run_timeout}s",
                        "return_code": None,
                        "execution_time": duration,
                    },
                }
        except Exception as exc:  # noqa: BLE001 — spawn failure is a sandbox-side error
            return {
                "status": "SandboxError",
                "compile_result": None,
                "run_result": {
                    "status": "Error",
                    "stdout": "",
                    "stderr": f"Sandbox failed to launch subprocess: {exc!r}",
                    "return_code": None,
                    "execution_time": time.monotonic() - start,
                },
            }

        return_code = proc.returncode
        # return_code < 0 means killed by signal N (e.g. -9 OOM via RLIMIT_AS, -24 RLIMIT_CPU).
        # The client treats any non-zero / non-Finished as a failed case, which is what we want.
        run_status = "Finished" if return_code == 0 else "Error"
        top_status = "Success" if return_code == 0 else "Failed"
        return {
            "status": top_status,
            "compile_result": None,
            "run_result": {
                "status": run_status,
                "stdout": _truncate(stdout),
                "stderr": _truncate(stderr),
                "return_code": return_code,
                "execution_time": duration,
            },
        }


def _run_bash(req: RunCodeRequest) -> dict:
    """Minimal bash support (some datasets use it); same isolation as python."""
    run_timeout = max(1, int(req.run_timeout or 10))
    memory_limit_mb = min(int(req.memory_limit_MB or 1024), MAX_MEMORY_LIMIT_MB)
    stdin_bytes = (req.stdin or "").encode("utf-8")
    with tempfile.TemporaryDirectory(prefix="sbx_") as workdir:
        script = os.path.join(workdir, "main.sh")
        with open(script, "w", encoding="utf-8") as fh:
            fh.write(req.code)
        preexec = _make_preexec_fn(memory_limit_mb, run_timeout + 2)
        start = time.monotonic()
        try:
            proc = subprocess.Popen(
                ["/bin/bash", script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=workdir,
                preexec_fn=preexec,
                env={"PATH": "/usr/bin:/bin", "HOME": workdir, "TMPDIR": workdir},
            )
            try:
                stdout, stderr = proc.communicate(input=stdin_bytes, timeout=run_timeout)
                duration = time.monotonic() - start
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.communicate()
                return {
                    "status": "Failed",
                    "compile_result": None,
                    "run_result": {
                        "status": "TimeLimitExceeded",
                        "stdout": "",
                        "stderr": f"Execution timed out after {run_timeout}s",
                        "return_code": None,
                        "execution_time": time.monotonic() - start,
                    },
                }
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "SandboxError",
                "compile_result": None,
                "run_result": {
                    "status": "Error",
                    "stdout": "",
                    "stderr": f"Sandbox failed to launch subprocess: {exc!r}",
                    "return_code": None,
                    "execution_time": time.monotonic() - start,
                },
            }
        rc = proc.returncode
        return {
            "status": "Success" if rc == 0 else "Failed",
            "compile_result": None,
            "run_result": {
                "status": "Finished" if rc == 0 else "Error",
                "stdout": _truncate(stdout),
                "stderr": _truncate(stderr),
                "return_code": rc,
                "execution_time": duration,
            },
        }


@app.post("/run_code")
def run_code(req: RunCodeRequest):
    lang = (req.language or "python").lower()
    if lang in ("python", "python3", "python_gpu", "pytest"):
        return _run_python(req)
    if lang == "bash":
        return _run_bash(req)
    # Unsupported language: report a clean SandboxError so the client records -1 for the case
    # rather than crashing. (LCB / Eurus are all python, so this is just defensive.)
    return {
        "status": "SandboxError",
        "compile_result": None,
        "run_result": {
            "status": "Error",
            "stdout": "",
            "stderr": f"Unsupported language: {req.language!r} (this sandbox supports python/bash)",
            "return_code": None,
            "execution_time": 0.0,
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "python": PYTHON_INTERPRETER,
        "drop_privileges": DROP_PRIVILEGES,
        "use_site": USE_SITE,
    }


def main():
    ap = argparse.ArgumentParser(description="verl-compatible code execution sandbox")
    ap.add_argument("--host", default=os.environ.get("SANDBOX_HOST", "127.0.0.1"),
                    help="bind address. Defaults to loopback only; this endpoint executes "
                         "arbitrary code, so do NOT bind it to 0.0.0.0 on shared networks "
                         "unless you have an external access control in front of it.")
    ap.add_argument("--port", type=int, default=int(os.environ.get("SANDBOX_PORT", "8080")))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SANDBOX_WORKERS", "64")),
                    help="uvicorn worker processes; size to absorb the client's concurrent "
                         "ThreadPool (reward_model.sandbox_fusion.max_concurrent).")
    ap.add_argument("--python", default=PYTHON_INTERPRETER,
                    help="interpreter used to run python submissions")
    ap.add_argument("--drop-privileges", action="store_true", default=DROP_PRIVILEGES,
                    help="setuid/setgid submissions to nobody (only effective when run as root)")
    args = ap.parse_args()

    # Propagate to module globals so uvicorn workers (which re-import this module) pick them up.
    os.environ["SANDBOX_PYTHON"] = args.python
    os.environ["SANDBOX_DROP_PRIVILEGES"] = "true" if args.drop_privileges else "false"

    import uvicorn

    # workers>1 requires an import string target.
    uvicorn.run(
        "sandbox_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
