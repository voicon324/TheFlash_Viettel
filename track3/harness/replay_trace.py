#!/usr/bin/env python3
import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import aiohttp


def load_trace(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = round((len(values) - 1) * q)
    return values[idx]


def extract_delta_text(payload: dict) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    delta = choice.get("delta") or {}
    if isinstance(delta, dict):
        return delta.get("content") or delta.get("reasoning_content") or ""
    text = choice.get("text")
    return text if isinstance(text, str) else ""


async def replay_one(session: aiohttp.ClientSession, endpoint: str, row: dict, start_ns: int, time_scale: float) -> dict:
    request_id = row.get("request_id")
    scheduled = start_ns + int(row["timestamp_ms"] / time_scale * 1_000_000)
    now = time.perf_counter_ns()
    if scheduled > now:
        await asyncio.sleep((scheduled - now) / 1e9)

    body = dict(row["body"])
    body["stream"] = True
    sent_ns = time.perf_counter_ns()
    first_ns = None
    last_token_ns = None
    gaps_ms = []
    text_parts = []
    error = None
    status = None

    try:
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=600)
        async with session.post(endpoint, json=body, timeout=timeout) as resp:
            status = resp.status
            async for raw in resp.content:
                for line in raw.decode("utf-8", errors="ignore").splitlines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        continue
                    try:
                        payload = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    token_text = extract_delta_text(payload)
                    if token_text:
                        t_ns = time.perf_counter_ns()
                        if first_ns is None:
                            first_ns = t_ns
                        if last_token_ns is not None:
                            gaps_ms.append((t_ns - last_token_ns) / 1e6)
                        last_token_ns = t_ns
                        text_parts.append(token_text)
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)

    finished_ns = time.perf_counter_ns()
    output_text = "".join(text_parts)
    ttft_ms = None if first_ns is None else (first_ns - sent_ns) / 1e6
    tbt_median_ms = statistics.median(gaps_ms) if gaps_ms else 0.0 if output_text else None
    effective = bool(output_text) and ttft_ms is not None and ttft_ms <= 2000 and tbt_median_ms is not None and tbt_median_ms <= 200

    return {
        "request_id": request_id,
        "status": status,
        "effective": effective,
        "ttft_ms": ttft_ms,
        "tbt_median_ms": tbt_median_ms,
        "token_chunks": len(text_parts),
        "output_chars": len(output_text),
        "latency_ms": (finished_ns - sent_ns) / 1e6,
        "error": error,
    }


def summarize(results: list[dict]) -> dict:
    ttft = [r["ttft_ms"] for r in results if r["ttft_ms"] is not None]
    tbt = [r["tbt_median_ms"] for r in results if r["tbt_median_ms"] is not None]
    effective = sum(1 for r in results if r["effective"])
    return {
        "requests": len(results),
        "effective": effective,
        "erc": effective / len(results) if results else 0.0,
        "ttft_ms": {
            "p50": percentile(ttft, 0.50),
            "p90": percentile(ttft, 0.90),
            "p95": percentile(ttft, 0.95),
            "max": max(ttft) if ttft else None,
        },
        "tbt_median_ms": {
            "p50": percentile(tbt, 0.50),
            "p90": percentile(tbt, 0.90),
            "p95": percentile(tbt, 0.95),
            "max": max(tbt) if tbt else None,
        },
        "errors": sum(1 for r in results if r["error"]),
        "non_200": sum(1 for r in results if r["status"] and r["status"] != 200),
    }


async def main_async(args: argparse.Namespace) -> int:
    rows = load_trace(args.trace, args.limit)
    headers = {}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    connector = aiohttp.TCPConnector(limit=args.connection_limit)
    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        start_ns = time.perf_counter_ns() + int(args.warmup_delay * 1e9)
        tasks = [asyncio.create_task(replay_one(session, args.endpoint, row, start_ns, args.time_scale)) for row in rows]
        results = await asyncio.gather(*tasks)

    summary = summarize(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "trace_results.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in sorted(results, key=lambda x: x["request_id"])) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=Path("track3/input/trace-round1.jsonl"))
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--output-dir", type=Path, default=Path("track3/results/local_replay"))
    parser.add_argument("--time-scale", type=float, default=1.0, help="1.0 preserves arrival timestamps; 10.0 replays 10x faster.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--connection-limit", type=int, default=256)
    parser.add_argument("--warmup-delay", type=float, default=1.0)
    args = parser.parse_args()
    if args.limit <= 0:
        args.limit = None
    return args


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async(parse_args())))
