#!/usr/bin/env python3
import argparse
import json
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRACK3_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = TRACK3_ROOT / "kaggle_runs"


def log(message: str) -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now(timezone.utc).isoformat()} {message}"
    with (RUN_ROOT / "progress.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)


def slugify(text: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return text[:55].strip("-")


def run(cmd: list[str], log_path: Path, env: dict[str, str], timeout: int | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=timeout, check=False)
        f.write(f"\nexit_code={proc.returncode}\n")
        return proc.returncode


def stage_job(config: dict, job: dict, account: dict) -> tuple[Path, str, str]:
    with open(account["kaggle_json"], "r", encoding="utf-8") as f:
        username = json.load(f)["username"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    slug = slugify(f"{job['job_id']}-{stamp}")
    kernel_id = f"{username}/{slug}"
    job_dir = RUN_ROOT / "jobs" / job["job_id"]
    kernel_dir = job_dir / "kernel"
    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
    shutil.copytree(REPO_ROOT / job["template_dir"], kernel_dir)
    shutil.copy2(TRACK3_ROOT / "input" / "trace-round1.jsonl", kernel_dir / "trace-round1.jsonl")
    shutil.copy2(TRACK3_ROOT / "harness" / "replay_trace.py", kernel_dir / "replay_trace.py")
    metadata_path = kernel_dir / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "id": kernel_id,
            "title": slug,
            "code_file": "kernel.py",
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": job.get("dataset_sources", []),
            "competition_sources": job.get("competition_sources", []),
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (job_dir / "url.txt").write_text(f"https://www.kaggle.com/code/{kernel_id}\n", encoding="utf-8")
    return kernel_dir, kernel_id, username


def private_kaggle_env(kaggle_json: str) -> tuple[tempfile.TemporaryDirectory, dict[str, str]]:
    tmp = tempfile.TemporaryDirectory(prefix="kaggle-config-")
    dst = Path(tmp.name) / "kaggle.json"
    shutil.copy2(kaggle_json, dst)
    dst.chmod(0o600)
    env = dict(**__import__("os").environ)
    env["KAGGLE_CONFIG_DIR"] = tmp.name
    return tmp, env


def write_dashboard(job_id: str, account: str, status: str, url: str, expected_zip: str) -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    body = "job_id\taccount\tstatus\turl\texpected_zip\tlast_checked\n"
    body += f"{job_id}\t{account}\t{status}\t{url}\t{expected_zip}\t{datetime.now(timezone.utc).isoformat()}\n"
    (RUN_ROOT / "dashboard.tsv").write_text(body, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=TRACK3_ROOT / "kaggle" / "config.example.json")
    parser.add_argument("--job-id", default="track3-p100-vllm-smoke")
    parser.add_argument("--poll-seconds", type=int, default=90)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    account = config["accounts"][0]
    job = next(j for j in config["jobs"] if j["job_id"] == args.job_id)
    job_dir = RUN_ROOT / "jobs" / job["job_id"]
    kernel_dir, kernel_id, _username = stage_job(config, job, account)
    slug = kernel_id.split("/", 1)[1]
    output_dir = job_dir / "output" / slug
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://www.kaggle.com/code/{kernel_id}"
    write_dashboard(job["job_id"], account["name"], "submitting", url, job["expected_zip"])
    log(f"submitting job={job['job_id']} account={account['name']} url={url}")

    tmp, env = private_kaggle_env(account["kaggle_json"])
    with tmp:
        code = run(
            ["kaggle", "kernels", "push", "-p", str(kernel_dir), "--accelerator", job.get("accelerator", "gpu")],
            job_dir / "push.log",
            env,
            timeout=600,
        )
        if code != 0:
            write_dashboard(job["job_id"], account["name"], "failed", url, job["expected_zip"])
            log(f"failed_push job={job['job_id']} code={code}")
            return code

        write_dashboard(job["job_id"], account["name"], "running", url, job["expected_zip"])
        log(f"running job={job['job_id']} url={url}")
        deadline = time.time() + job.get("timeout_minutes", 180) * 60
        while time.time() < deadline:
            time.sleep(args.poll_seconds)
            code = run(
                [
                    "kaggle",
                    "kernels",
                    "output",
                    kernel_id,
                    "-p",
                    str(output_dir),
                    "--file-pattern",
                    job.get("output_pattern", ".*"),
                    "-o",
                ],
                job_dir / "output.log",
                env,
                timeout=600,
            )
            expected = next(output_dir.rglob(job["expected_zip"]), None)
            if expected and expected.exists():
                try:
                    with zipfile.ZipFile(expected) as zf:
                        bad = zf.testzip()
                    if bad is None:
                        write_dashboard(job["job_id"], account["name"], "done", url, job["expected_zip"])
                        log(f"done job={job['job_id']} file={expected}")
                        return 0
                except zipfile.BadZipFile:
                    pass
            log(f"poll job={job['job_id']} output_code={code}")

    write_dashboard(job["job_id"], account["name"], "failed", url, job["expected_zip"])
    log(f"timeout job={job['job_id']}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
