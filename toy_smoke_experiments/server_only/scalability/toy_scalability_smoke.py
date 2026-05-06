#!/usr/bin/env python3
"""Minimal server-only smoke test for the MetaCS-FL experiment pipeline.

This script intentionally does not run the full scalability simulator. It only
validates that the server-only backend can:
  1. run a Python script on the server node;
  2. receive a --config-file argument;
  3. create an output directory;
  4. produce files that run_on_server_only.sh can collect.
"""

from __future__ import annotations

import argparse
import csv
import json
import socket
from configparser import ConfigParser
from datetime import datetime, timezone
from pathlib import Path


def parse_list(value: str, cast=int) -> list:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    if not value.strip():
        return []
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny server-only smoke experiment.")
    parser.add_argument("--config-file", required=True, help="Smoke config file.")
    args = parser.parse_args()

    config_path = Path(args.config_file)
    cfg = ConfigParser(interpolation=None)
    cfg.read(config_path)

    approach_name = cfg.get("Approach Settings", "approach_name", fallback="FedAvg")
    output_directory = Path(
        cfg.get(
            "Output Settings",
            "output_directory",
            fallback="results/toy_smoke_experiments/server_only/scalability/fedavg_smoke",
        )
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    num_clients_list = parse_list(
        cfg.get("Experiments Combinations Settings", "num_clients_list", fallback="[10]")
    )
    num_tasks_list = parse_list(
        cfg.get("Experiments Combinations Settings", "num_tasks_list", fallback="[10]")
    )
    num_rounds_list = parse_list(
        cfg.get("Experiments Combinations Settings", "num_rounds_list", fallback="[1]")
    )

    rows = []
    for num_clients in num_clients_list:
        for num_tasks in num_tasks_list:
            for num_rounds in num_rounds_list:
                rows.append(
                    {
                        "approach_name": approach_name,
                        "num_clients": num_clients,
                        "num_tasks": num_tasks,
                        "num_rounds": num_rounds,
                        "status": "completed",
                    }
                )

    csv_path = output_directory / "toy_scalability_smoke_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["approach_name", "num_clients", "num_tasks", "num_rounds", "status"],
        )
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "script": str(Path(__file__).as_posix()),
        "config_file": str(config_path),
        "output_directory": str(output_directory),
        "hostname": socket.gethostname(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "num_rows": len(rows),
    }
    with (output_directory / "toy_scalability_smoke_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Toy server-only smoke completed: {csv_path}")


if __name__ == "__main__":
    main()
