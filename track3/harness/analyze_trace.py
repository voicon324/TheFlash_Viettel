#!/usr/bin/env python3
import argparse
import json
import statistics
from pathlib import Path


def pct(values: list[int], q: float) -> int:
    values = sorted(values)
    return values[round((len(values) - 1) * q)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=Path("track3/input/trace-round1.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("track3/results/trace_analysis.json"))
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.trace.read_text(encoding="utf-8").splitlines() if line.strip()]
    timestamps = [r["timestamp_ms"] for r in rows]
    char_lengths = []
    system_prompts = []
    body_fingerprints = set()
    max_tokens = []
    temperatures = []
    for r in rows:
        body = r["body"]
        body_fingerprints.add(json.dumps(body, sort_keys=True, ensure_ascii=False))
        max_tokens.append(body.get("max_tokens"))
        temperatures.append(body.get("temperature"))
        messages = body.get("messages") or []
        text = "\n".join(m.get("content", "") for m in messages)
        char_lengths.append(len(text))
        system_prompts.append(next((m.get("content", "") for m in messages if m.get("role") == "system"), ""))

    inter_arrival = [b - a for a, b in zip(timestamps, timestamps[1:])]
    analysis = {
        "requests": len(rows),
        "timestamp_ms": {"first": min(timestamps), "last": max(timestamps), "span": max(timestamps) - min(timestamps)},
        "inter_arrival_ms": {
            "min": min(inter_arrival),
            "p50": statistics.median(inter_arrival),
            "p90": pct(inter_arrival, 0.90),
            "max": max(inter_arrival),
        },
        "prompt_chars": {
            "min": min(char_lengths),
            "p50": statistics.median(char_lengths),
            "p90": pct(char_lengths, 0.90),
            "max": max(char_lengths),
        },
        "unique_system_prompts": len(set(system_prompts)),
        "first_system_prompt_chars": len(system_prompts[0]) if system_prompts else 0,
        "duplicate_bodies": len(rows) - len(body_fingerprints),
        "max_tokens_values": sorted({str(v) for v in max_tokens}),
        "temperature_values": sorted({str(v) for v in temperatures}),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(json.dumps(analysis, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
