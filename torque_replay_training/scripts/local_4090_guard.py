#!/usr/bin/env python3
"""Fail closed unless local physical GPU 0 is an idle RTX 4090."""

from __future__ import annotations

import csv
import io
import json
import subprocess


def _query(*arguments: str) -> list[list[str]]:
    result = subprocess.run(
        ["nvidia-smi", *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        row
        for row in csv.reader(io.StringIO(result.stdout), skipinitialspace=True)
        if row
    ]


def main() -> None:
    rows = _query(
        "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    )
    inventory = {int(row[0]): row for row in rows}
    if 0 not in inventory:
        raise SystemExit("physical GPU 0 was not reported by nvidia-smi")
    row = inventory[0]
    if "RTX 4090" not in row[2]:
        raise SystemExit(f"physical GPU 0 is not an RTX 4090: {row[2]}")

    try:
        processes = _query(
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        )
    except subprocess.CalledProcessError:
        processes = []
    compute = [
        {
            "pid": int(item[1]),
            "process_name": item[2],
            "memory_used_mib": int(item[3]),
        }
        for item in processes
        if len(item) >= 4 and item[0] == row[1]
    ]
    report = {
        "physical_gpu": 0,
        "name": row[2],
        "memory_used_mib": int(row[3]),
        "memory_total_mib": int(row[4]),
        "utilization_percent": int(row[5]),
        "compute_processes": compute,
        "ready": not compute,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if compute:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
