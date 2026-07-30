#!/usr/bin/env python3
"""Fail closed when any requested physical A100 GPU is already in use."""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
from typing import Any


ALLOWED_GPUS = {5}


def _nvidia_smi(*query: str) -> list[list[str]]:
    command = ["nvidia-smi", *query]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return [row for row in csv.reader(io.StringIO(result.stdout), skipinitialspace=True) if row]


def inspect_gpus(gpus: list[int], memory_limit_mib: int, utilization_limit: int) -> dict[str, Any]:
    rows = _nvidia_smi(
        "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    )
    inventory = {
        int(row[0]): {
            "index": int(row[0]),
            "uuid": row[1],
            "name": row[2],
            "memory_used_mib": int(row[3]),
            "memory_total_mib": int(row[4]),
            "utilization_percent": int(row[5]),
            "compute_processes": [],
        }
        for row in rows
    }
    try:
        process_rows = _nvidia_smi(
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        )
    except subprocess.CalledProcessError:
        process_rows = []
    by_uuid = {entry["uuid"]: entry for entry in inventory.values()}
    for row in process_rows:
        if len(row) < 4 or row[0] not in by_uuid:
            continue
        by_uuid[row[0]]["compute_processes"].append(
            {"pid": int(row[1]), "process_name": row[2], "memory_used_mib": int(row[3])}
        )

    selected = []
    for gpu in gpus:
        if gpu not in inventory:
            raise RuntimeError(f"physical GPU {gpu} was not reported by nvidia-smi")
        entry = inventory[gpu]
        reasons = []
        if entry["compute_processes"]:
            reasons.append("compute_process_present")
        if entry["memory_used_mib"] > memory_limit_mib:
            reasons.append(f"memory_used>{memory_limit_mib}MiB")
        if entry["utilization_percent"] > utilization_limit:
            reasons.append(f"utilization>{utilization_limit}%")
        entry["busy_reasons"] = reasons
        entry["free"] = not reasons
        selected.append(entry)
    return {
        "requested_physical_gpus": gpus,
        "allowed_physical_gpus": sorted(ALLOWED_GPUS),
        "memory_limit_mib": memory_limit_mib,
        "utilization_limit_percent": utilization_limit,
        "all_free": all(entry["free"] for entry in selected),
        "gpus": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs="+", type=int, default=[5])
    parser.add_argument("--memory-limit-mib", type=int, default=1024)
    parser.add_argument("--utilization-limit", type=int, default=10)
    args = parser.parse_args()
    if not args.gpus or not set(args.gpus).issubset(ALLOWED_GPUS):
        raise SystemExit("only physical GPU 5 is permitted")
    report = inspect_gpus(args.gpus, args.memory_limit_mib, args.utilization_limit)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["all_free"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
