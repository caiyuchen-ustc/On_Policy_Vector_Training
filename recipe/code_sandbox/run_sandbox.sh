#!/usr/bin/env bash
#
# Launch / stop the local code-execution sandbox used by the code RL reward.
#
#   bash run_sandbox.sh [start]   # start server, wait for health, print SANDBOX_FUSION_URL
#   bash run_sandbox.sh stop      # stop the server started by this script
#   bash run_sandbox.sh status    # show whether it is running + health
#   bash run_sandbox.sh bench     # fire a concurrent burst at /run_code, print req/s
#
# Tunables (env):
#   SANDBOX_HOST           bind host        (default 127.0.0.1 — loopback only; see security note)
#   SANDBOX_PORT           bind port        (default 8080)
#   SANDBOX_WORKERS        uvicorn workers  (default min(cpu,32) — real multi-process parallelism)
#   SANDBOX_SITE           0/1              (default 0; 1 = full `site`, third-party imports like
#                                            numpy work but submissions start ~2.7x slower)
#   SANDBOX_DROP_PRIVILEGES  true/false     (default false; setuid submissions to nobody)
#   SANDBOX_BENCH_N        bench requests   (default 1024)
#   SANDBOX_BENCH_CONC     bench concurrency(default 256)
#
# SECURITY: /run_code executes arbitrary code. Keep it bound to 127.0.0.1 (the default). The
# whole flow is loopback-only — training runs on the same node and uses http://127.0.0.1:PORT.
# Only change SANDBOX_HOST if you have your own network access control in front of it.
#
# After start, export the printed URL before launching training:
#   export SANDBOX_FUSION_URL=http://127.0.0.1:8080/run_code

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi
set -euo pipefail

source /apdcephfs_zwfy3/share_302867165/xxucaxu/anaconda3/bin/activate \
       /apdcephfs_zwfy3/share_302867165/xxucaxu/anaconda3/envs/verlopd

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${SCRIPT_DIR}/sandbox.pid"
LOG_FILE="${SCRIPT_DIR}/sandbox.log"

SANDBOX_HOST="${SANDBOX_HOST:-127.0.0.1}"
SANDBOX_PORT="${SANDBOX_PORT:-8080}"
# Default workers to min(cpu_count, 32): real multi-process parallelism. Capped at 32 because
# beyond that, simultaneous worker boot + fork-of-large-parent contention on a busy node tends
# to REDUCE throughput rather than help. Override SANDBOX_WORKERS to tune for your machine.
SANDBOX_WORKERS="${SANDBOX_WORKERS:-$(python3 -c 'import os; print(min(os.cpu_count() or 8, 32))')}"
SANDBOX_DROP_PRIVILEGES="${SANDBOX_DROP_PRIVILEGES:-false}"
SANDBOX_SITE="${SANDBOX_SITE:-0}"

HEALTH_URL="http://127.0.0.1:${SANDBOX_PORT}/health"
RUN_URL="http://127.0.0.1:${SANDBOX_PORT}/run_code"

ACTION="${1:-start}"

is_running() {
    [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null
}

stop_server() {
    if is_running; then
        local pid
        pid="$(cat "${PID_FILE}")"
        echo "Stopping sandbox (pid ${pid}) and its workers..."
        # Kill the whole process group (uvicorn master + workers).
        kill -TERM "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
        sleep 2
        kill -KILL "-${pid}" 2>/dev/null || true
    else
        echo "No running sandbox recorded at ${PID_FILE}."
    fi
    # Reap any orphaned sandbox uvicorn process still bound to this port (e.g. left over
    # from a crashed master or a stale pid file), so a subsequent start isn't blocked.
    local orphans
    orphans="$(pgrep -f "uvicorn sandbox_server:app .*--port ${SANDBOX_PORT}\b" 2>/dev/null || true)"
    if [ -n "${orphans}" ]; then
        echo "Reaping orphaned sandbox uvicorn on port ${SANDBOX_PORT}: ${orphans}"
        for opid in ${orphans}; do
            local opgid
            opgid="$(ps -o pgid= -p "${opid}" 2>/dev/null | tr -d ' ')"
            [ -n "${opgid}" ] && kill -KILL "-${opgid}" 2>/dev/null || true
            kill -KILL "${opid}" 2>/dev/null || true
        done
        sleep 1
    fi
    rm -f "${PID_FILE}"
}

case "${ACTION}" in
    stop)
        stop_server
        exit 0
        ;;
    status)
        if is_running; then
            echo "Sandbox RUNNING (pid $(cat "${PID_FILE}"))"
            echo "worker children: $(pgrep -P "$(cat "${PID_FILE}")" 2>/dev/null | wc -l)"
            curl -s -m 5 "${HEALTH_URL}" && echo
        else
            echo "Sandbox NOT running."
        fi
        exit 0
        ;;
    bench)
        # Fire a concurrent burst of trivial /run_code requests and report req/s.
        export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
        export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
        BENCH_N="${SANDBOX_BENCH_N:-1024}" BENCH_CONC="${SANDBOX_BENCH_CONC:-256}" \
        RUN_URL="${RUN_URL}" python3 - <<'PYBENCH'
import os, time, concurrent.futures, requests
URL = os.environ["RUN_URL"]
N = int(os.environ["BENCH_N"]); CONC = int(os.environ["BENCH_CONC"])
payload = {"compile_timeout": 10, "run_timeout": 10,
           "code": "print(sum(int(x) for x in input().split()))",
           "stdin": "2 3", "memory_limit_MB": 1024, "language": "python",
           "files": {}, "fetch_files": []}
def one():
    try:
        return requests.post(URL, json=payload, timeout=30).json()["run_result"]["stdout"].strip()
    except Exception as e:
        return f"ERR:{e}"
t = time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=CONC) as ex:
    outs = list(ex.map(lambda _: one(), range(N)))
dt = time.monotonic() - t
ok = sum(1 for o in outs if o == "5")
print(f"bench: N={N} conc={CONC}  {dt:.2f}s  ->  {N/dt:.0f} req/s   (ok={ok}/{N})")
PYBENCH
        exit 0
        ;;
    start|"")
        ;;
    *)
        echo "Usage: bash run_sandbox.sh [start|stop|status|bench]" >&2
        exit 1
        ;;
esac

if is_running; then
    echo "Sandbox already running (pid $(cat "${PID_FILE}"))."
    echo "SANDBOX_FUSION_URL=${RUN_URL}"
    exit 0
fi

# Disable proxies for loopback traffic so health/training requests aren't routed through
# the corporate proxy (which would fail for 127.0.0.1).
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"

echo "Starting sandbox on ${SANDBOX_HOST}:${SANDBOX_PORT} with ${SANDBOX_WORKERS} workers (SANDBOX_SITE=${SANDBOX_SITE})..."
# Launch via the uvicorn CLI from SCRIPT_DIR so workers can import "sandbox_server:app".
# This is what actually spawns N worker processes — `python sandbox_server.py` (which calls
# uvicorn.run(workers=N) internally) only produced a single process. The workers re-import the
# module and read SANDBOX_PYTHON / SANDBOX_SITE / SANDBOX_DROP_PRIVILEGES from the environment.
# setsid + own process group so `stop` can kill master+workers together.
cd "${SCRIPT_DIR}"
export SANDBOX_DROP_PRIVILEGES="${SANDBOX_DROP_PRIVILEGES}"
export SANDBOX_SITE="${SANDBOX_SITE}"
setsid nohup python3 -m uvicorn sandbox_server:app \
    --host "${SANDBOX_HOST}" \
    --port "${SANDBOX_PORT}" \
    --workers "${SANDBOX_WORKERS}" \
    --log-level warning \
    >"${LOG_FILE}" 2>&1 &
echo $! > "${PID_FILE}"

# Wait for /health to come up (up to ~60s; multi-worker boot can take a bit).
echo -n "Waiting for sandbox health"
for _ in $(seq 1 60); do
    if curl -s -m 3 -o /dev/null -w "%{http_code}" "${HEALTH_URL}" 2>/dev/null | grep -q 200; then
        echo " ... OK"
        SANDBOX_PID="$(cat "${PID_FILE}")"
        echo "Sandbox is up (pid ${SANDBOX_PID}, worker children: $(pgrep -P "${SANDBOX_PID}" 2>/dev/null | wc -l)). Log: ${LOG_FILE}"
        echo
        echo "Export this before launching training:"
        echo "  export SANDBOX_FUSION_URL=${RUN_URL}"
        exit 0
    fi
    echo -n "."
    sleep 1
done

echo
echo "ERROR: sandbox did not become healthy in time. Check ${LOG_FILE}:" >&2
tail -n 30 "${LOG_FILE}" >&2 || true
stop_server
exit 1
