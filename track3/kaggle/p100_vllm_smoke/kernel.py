#!/usr/bin/env python3
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
import urllib.error
import zipfile
from pathlib import Path


WORK = Path("/kaggle/working")
SRC = Path(__file__).resolve().parent
REPO_RAW_BASE = os.environ.get("TRACK3_REPO_RAW_BASE", "https://raw.githubusercontent.com/voicon324/TheFlash_Viettel/main")
TRACE = WORK / "trace-round1.jsonl"
REPLAY = WORK / "replay_trace.py"
RESULT_ZIP = WORK / "track3_p100_vllm_smoke_result.zip"


def run(cmd: list[str], log: Path, timeout: int | None = None, check: bool = False) -> subprocess.CompletedProcess:
    started = time.time()
    with log.open("a", encoding="utf-8") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, timeout=timeout, check=False)
        f.write(f"\nexit_code={proc.returncode} elapsed_s={time.time() - started:.1f}\n")
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}")
    return proc


def pct(values: list[int], q: float) -> int:
    values = sorted(values)
    return values[round((len(values) - 1) * q)]


def ensure_source_files() -> None:
    downloads = {
        TRACE: f"{REPO_RAW_BASE}/track3/input/trace-round1.jsonl",
        REPLAY: f"{REPO_RAW_BASE}/track3/harness/replay_trace.py",
    }
    for path, url in downloads.items():
        if path.exists():
            continue
        with urllib.request.urlopen(url, timeout=120) as response:
            path.write_bytes(response.read())


def analyze_trace() -> dict:
    rows = [json.loads(line) for line in TRACE.read_text(encoding="utf-8").splitlines() if line.strip()]
    timestamps = [r["timestamp_ms"] for r in rows]
    char_lengths = []
    systems = []
    bodies = set()
    for r in rows:
        body = r["body"]
        bodies.add(json.dumps(body, sort_keys=True, ensure_ascii=False))
        messages = body.get("messages") or []
        char_lengths.append(len("\n".join(m.get("content", "") for m in messages)))
        systems.append(next((m.get("content", "") for m in messages if m.get("role") == "system"), ""))
    inter_arrival = [b - a for a, b in zip(timestamps, timestamps[1:])]
    return {
        "requests": len(rows),
        "timestamp_ms": {"first": min(timestamps), "last": max(timestamps), "span": max(timestamps) - min(timestamps)},
        "inter_arrival_ms": {
            "min": min(inter_arrival),
            "p50": statistics.median(inter_arrival),
            "p90": pct(inter_arrival, 0.9),
            "max": max(inter_arrival),
        },
        "prompt_chars": {
            "min": min(char_lengths),
            "p50": statistics.median(char_lengths),
            "p90": pct(char_lengths, 0.9),
            "max": max(char_lengths),
        },
        "unique_system_prompts": len(set(systems)),
        "first_system_prompt_chars": len(systems[0]) if systems else 0,
        "duplicate_bodies": len(rows) - len(bodies),
    }


def write_summary(summary: dict) -> None:
    (WORK / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def wait_for_server(url: str, timeout_s: int, summary: dict) -> bool:
    deadline = time.time() + timeout_s
    attempts = 0
    last_error = None
    while time.time() < deadline:
        attempts += 1
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if 200 <= response.status < 500:
                    summary["server_ready"] = {
                        "ready": True,
                        "attempts": attempts,
                        "elapsed_s": round(timeout_s - (deadline - time.time()), 1),
                        "status": response.status,
                    }
                    write_summary(summary)
                    return True
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
        time.sleep(5)
    summary["server_ready"] = {
        "ready": False,
        "attempts": attempts,
        "elapsed_s": timeout_s,
        "last_error": last_error,
    }
    write_summary(summary)
    return False


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    ensure_source_files()
    summary = {
        "platform": platform.platform(),
        "python": sys.version,
        "trace_analysis": analyze_trace(),
        "commands": {},
    }
    write_summary(summary)

    run(["nvidia-smi"], WORK / "nvidia-smi.log")
    summary["commands"]["nvidia_smi_log"] = "nvidia-smi.log"
    write_summary(summary)

    run([sys.executable, "-m", "pip", "install", "-U", "pip"], WORK / "pip_install.log", timeout=600)
    install = run(
        [sys.executable, "-m", "pip", "install", "aiohttp>=3.9", "vllm>=0.8.5"],
        WORK / "pip_install.log",
        timeout=3600,
    )
    summary["commands"]["pip_install_exit_code"] = install.returncode
    write_summary(summary)

    if install.returncode == 0 and os.environ.get("TRACK3_SKIP_VLLM", "0") != "1":
        server_log = (WORK / "vllm_server.log").open("w", encoding="utf-8")
        server_cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            os.environ.get("TRACK3_MODEL", "Qwen/Qwen3.5-2B"),
            "--served-model-name",
            "Qwen3.5-2B",
            "--dtype",
            "half",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--max-model-len",
            os.environ.get("TRACK3_MAX_MODEL_LEN", "8192"),
            "--gpu-memory-utilization",
            os.environ.get("TRACK3_GPU_MEMORY_UTIL", "0.90"),
            "--enable-prefix-caching",
        ]
        proc = subprocess.Popen(server_cmd, stdout=server_log, stderr=subprocess.STDOUT, text=True)
        summary["commands"]["vllm_pid"] = proc.pid
        write_summary(summary)
        try:
            if wait_for_server("http://127.0.0.1:8000/health", int(os.environ.get("TRACK3_SERVER_TIMEOUT", "600")), summary):
                replay = run(
                    [
                        sys.executable,
                        str(REPLAY),
                        "--trace",
                        str(TRACE),
                        "--endpoint",
                        "http://127.0.0.1:8000/v1/chat/completions",
                        "--output-dir",
                        str(WORK / "replay"),
                        "--limit",
                        os.environ.get("TRACK3_REPLAY_LIMIT", "8"),
                        "--time-scale",
                        "1",
                    ],
                    WORK / "replay.log",
                    timeout=1800,
                )
                summary["commands"]["replay_exit_code"] = replay.returncode
                replay_summary = WORK / "replay" / "summary.json"
                if replay_summary.exists():
                    summary["replay_summary"] = json.loads(replay_summary.read_text(encoding="utf-8"))
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            server_log.close()
            write_summary(summary)

    with zipfile.ZipFile(RESULT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in WORK.rglob("*"):
            if path == RESULT_ZIP or path.is_dir():
                continue
            zf.write(path, path.relative_to(WORK))
    print(f"wrote {RESULT_ZIP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
