# Code execution sandbox for DAPO code RL

A lightweight, **docker-free** code-execution service that the verl code reward talks to, so
test cases run **concurrently and off the training process** instead of serially inside it.

## Why this exists

When the `dapo` reward manager scores a code generation, `verl/utils/reward_score/__init__.py`
routes data sources in `{taco, apps, codeforces, codecontests, livecodebench}` to one of two
backends:

| Backend | When | Behavior |
|---|---|---|
| `prime_code` | `reward_model.sandbox_fusion.url` **unset** (default) | Runs every test case **serially, in-process** via `exec`. Simple, no setup, but slow and runs untrusted code in the training driver. |
| `sandbox_fusion` | `reward_model.sandbox_fusion.url` **set** | POSTs each test case to an HTTP sandbox and runs them **concurrently** (ThreadPool + a `max_concurrent` semaphore). Parallelizes across all CPUs and isolates execution. |

The official sandbox ([Bytedance SandboxFusion](https://github.com/bytedance/SandboxFusion))
is a Docker image. On hosts **without docker** (and without firejail/nsjail/bwrap), this
directory provides a drop-in replacement: a small FastAPI server speaking the exact same
`/run_code` protocol, executing each submission in a resource-limited subprocess.

It is fully verified to produce **identical scores to `prime_code`** on both stdin and
call-based (`fn_name`) problems — so turning it on changes throughput, not the reward.

## Files

- `sandbox_server.py` — the FastAPI service (`POST /run_code`, `GET /health`).
- `run_sandbox.sh` — start / stop / status helper; waits for health and prints the URL.

## End-to-end flow

```bash
cd /apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/recipe/code_sandbox

# 1. Start the sandbox on the training node (once). Prints the URL and waits for health.
bash run_sandbox.sh start
#   -> Export this before launching training:
#        export SANDBOX_FUSION_URL=http://127.0.0.1:8080/run_code

# 2. Export the URL, then launch any of the three code-RL modes.
export SANDBOX_FUSION_URL=http://127.0.0.1:8080/run_code
cd ../../examples/representation
bash code_rl_fullparam.sh        # full-parameter
#   or: bash code_rl_lora8.sh    # LoRA r=8
#   or: bash code_rl_single.sh   # single trainable vector (layers 0:20)

# 3. When finished, stop the sandbox.
cd ../../recipe/code_sandbox
bash run_sandbox.sh stop
```

### What happens during training

```
rollout generates code
  -> DAPORewardManager (reward_model.reward_manager=dapo)
     -> default_compute_score(data_source in {taco,apps,codeforces,codecontests,livecodebench})
        -> SANDBOX_FUSION_URL is set, so -> sandbox_fusion.compute_score
           -> for each test case (concurrently, bounded by max_concurrent):
              POST /run_code {code, stdin, run_timeout, memory_limit_MB, language}
              -> server runs code in an rlimited subprocess, returns {status, run_result{stdout,...}}
              -> client compares run_result.stdout.rstrip("\n") to the expected output
           -> pass ALL cases => 1.0, else 0.0   (then DAPO applies the overlong penalty)
```

`fn_name` (call-based) problems are wrapped into a stdin→stdout `__main__` by the verl **client**
before they reach the server, so the server only ever runs plain "read stdin / print stdout"
code — no special handling needed here.

If `SANDBOX_FUSION_URL` is left unset, training still works via the in-process `prime_code`
path. The sandbox is an opt-in accelerator/isolator, **not a hard dependency**.

## Configuration

`run_sandbox.sh` reads these env vars:

| Var | Default | Meaning |
|---|---|---|
| `SANDBOX_HOST` | `127.0.0.1` | Bind address. **Keep loopback** (see Security). |
| `SANDBOX_PORT` | `8080` | Port. The printed URL uses this. |
| `SANDBOX_WORKERS` | `64` | uvicorn worker processes. Size so workers ≳ the client's `max_concurrent`. |
| `SANDBOX_DROP_PRIVILEGES` | `false` | If `true` and running as root, `setuid`/`setgid` each submission to `nobody`. |

The training side (`code_rl.sh`) reads:

| Var | Default | Maps to |
|---|---|---|
| `SANDBOX_FUSION_URL` | *(empty)* | `reward_model.sandbox_fusion.url` (also the on/off switch) |
| `SANDBOX_MAX_CONCURRENT` | `256` | `reward_model.sandbox_fusion.max_concurrent` |
| `SANDBOX_MEMORY_LIMIT_MB` | `1024` | `reward_model.sandbox_fusion.memory_limit_mb` (per-submission memory cap) |

Tip: keep `SANDBOX_WORKERS` ≥ `SANDBOX_MAX_CONCURRENT / (number of reward worker processes)`
so the server isn't the bottleneck. With 384 cores, `SANDBOX_WORKERS=128` + `MAX_CONCURRENT=256`
is a reasonable starting point.

## Isolation (per submission)

Each `/run_code` request runs in its own subprocess with:
- `RLIMIT_AS` = `memory_limit_MB` (kills runaway allocation),
- `RLIMIT_CPU` = `run_timeout + 2` s (backstop for busy loops),
- wall-clock `timeout=run_timeout`; on expiry the **whole process group** (`os.setsid`) is SIGKILLed,
- `RLIMIT_NPROC` (fork-bomb guard), `RLIMIT_FSIZE` (output cap), core dumps disabled,
- a fresh temp `cwd`/`HOME`/`TMPDIR` (deleted after the run), a minimal `PATH`,
- optional drop to `nobody` (`SANDBOX_DROP_PRIVILEGES=true`).

## Security

`/run_code` **executes arbitrary code**. The server binds to `127.0.0.1` by default and the
entire flow is loopback-only — training runs on the same node. Do **not** bind it to `0.0.0.0`
on a shared network unless you put your own access control in front of it. The rlimit-based
isolation is best-effort hardening, not a container-grade security boundary.

## Troubleshooting

- **Didn't become healthy** → `tail recipe/code_sandbox/sandbox.log`. Usually a busy port
  (set `SANDBOX_PORT`) or the conda env not activating.
- **Reward is always 0 with the sandbox** → check the log for `Sandbox failed to launch` /
  `Unsupported language`; confirm `SANDBOX_FUSION_URL` ends in `/run_code`; confirm
  `no_proxy` includes `127.0.0.1` (the launcher sets this so loopback isn't sent to the
  corporate proxy).
- **Timeouts under load** → raise `SANDBOX_WORKERS`, or lower `SANDBOX_MAX_CONCURRENT`.
- **Verify by hand**:
  ```bash
  curl -s http://127.0.0.1:8080/health
  curl -s http://127.0.0.1:8080/run_code -H 'Content-Type: application/json' \
    -d '{"code":"print(int(input())*2)","stdin":"21\n","language":"python","run_timeout":5,"memory_limit_MB":512}'
  # -> {"status":"Success","run_result":{"status":"Finished","stdout":"42\n",...}}
  ```
