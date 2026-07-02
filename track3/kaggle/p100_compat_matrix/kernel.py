#!/usr/bin/env python3
import json
import platform
import subprocess
import sys
import time
import zipfile
from pathlib import Path


WORK = Path("/kaggle/working")
RESULT_ZIP = WORK / "track3_p100_compat_matrix_result.zip"


TORCH_CANDIDATES = [
    {
        "name": "torch-2.6.0-cu124",
        "install": [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-cache-dir",
            "torch==2.6.0",
            "--index-url",
            "https://download.pytorch.org/whl/cu124",
        ],
    },
    {
        "name": "torch-2.5.1-cu121",
        "install": [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-cache-dir",
            "torch==2.5.1",
            "--index-url",
            "https://download.pytorch.org/whl/cu121",
        ],
    },
    {
        "name": "torch-2.3.1-cu121",
        "install": [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-cache-dir",
            "torch==2.3.1",
            "--index-url",
            "https://download.pytorch.org/whl/cu121",
        ],
    },
]


VLLM_CANDIDATES = [
    {
        "name": "vllm-0.6.6.post1-nodeps",
        "install": [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--no-deps", "vllm==0.6.6.post1"],
    },
    {
        "name": "vllm-0.5.5-nodeps",
        "install": [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--no-deps", "vllm==0.5.5"],
    },
]


def run(cmd: list[str], log: Path, timeout: int = 1200) -> dict:
    started = time.time()
    with log.open("a", encoding="utf-8") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, timeout=timeout, check=False)
        elapsed = time.time() - started
        f.write(f"\nexit_code={proc.returncode} elapsed_s={elapsed:.1f}\n")
    return {"exit_code": proc.returncode, "elapsed_s": elapsed}


def pyprobe(code: str, log: Path, timeout: int = 180) -> dict:
    return run([sys.executable, "-c", code], log, timeout=timeout)


TORCH_PROBE = r"""
import json, torch
out = {
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "available": torch.cuda.is_available(),
    "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
    "arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
}
x = torch.zeros((1024, 1024), device="cuda")
y = x + 1
torch.cuda.synchronize()
out["sum"] = float(y.sum().item())
print(json.dumps(out, indent=2))
"""


VLLM_PROBE = r"""
import json
import torch
import vllm
out = {
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "vllm": getattr(vllm, "__version__", "unknown"),
    "device_name": torch.cuda.get_device_name(0),
    "capability": torch.cuda.get_device_capability(0),
}
print(json.dumps(out, indent=2))
"""


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    summary = {"platform": platform.platform(), "python": sys.version, "torch_candidates": [], "vllm_candidates": []}
    run(["nvidia-smi"], WORK / "nvidia-smi.log", timeout=120)
    run([sys.executable, "-m", "pip", "install", "-U", "pip"], WORK / "pip.log", timeout=600)

    first_working_torch = None
    for candidate in TORCH_CANDIDATES:
        item = {"name": candidate["name"]}
        item["install"] = run(candidate["install"], WORK / f"{candidate['name']}.install.log", timeout=1800)
        if item["install"]["exit_code"] == 0:
            item["probe"] = pyprobe(TORCH_PROBE, WORK / f"{candidate['name']}.probe.log")
            if item["probe"]["exit_code"] == 0 and first_working_torch is None:
                first_working_torch = candidate["name"]
        summary["torch_candidates"].append(item)
        (WORK / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if first_working_torch:
            break

    if first_working_torch:
        for candidate in VLLM_CANDIDATES:
            item = {"name": candidate["name"], "base_torch": first_working_torch}
            item["install"] = run(candidate["install"], WORK / f"{candidate['name']}.install.log", timeout=1800)
            if item["install"]["exit_code"] == 0:
                item["probe"] = pyprobe(VLLM_PROBE, WORK / f"{candidate['name']}.probe.log")
            summary["vllm_candidates"].append(item)
            (WORK / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            if item.get("probe", {}).get("exit_code") == 0:
                break

    with zipfile.ZipFile(RESULT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in WORK.rglob("*"):
            if path == RESULT_ZIP or path.is_dir():
                continue
            zf.write(path, path.relative_to(WORK))
    print(f"wrote {RESULT_ZIP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
