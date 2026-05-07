#!/usr/bin/env python3
"""
Generate explicit experiment manifests for MetaCS-FL campaigns.

This script defines experiment packs independently from the runner. The
runner should execute the generated manifest, not rediscover experiments
from the folder tree at runtime.

Currently supported pack:
  - fgcs_2026_experiments

This pack includes:
  - static_client_availability/performance_experiments
      50_clients, 100_clients
      cifar_10, fashion_mnist
      iid, non_iid
      fedavg, oort, mec, ecmtc, divfl, ecsm, metacsfl
      excludes emotion

  - static_client_availability/dp_impact_experiments
      100_clients/cifar_10/non_iid
      metacsfl, metacsfl_no_privacy

  - static_client_availability/scalability_experiments
      server-only:
      fedavg, oort, mec, ecmtc, divfl, ecsm,
      metacsfl_015, metacsfl_025, metacsfl_050

  - static_client_availability/sensitivity_experiments
      server-only:
      metacsfl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from configparser import ConfigParser
from dataclasses import asdict, dataclass
from pathlib import Path


SUPPORTED_PACKS = {"fgcs_2026_experiments"}


@dataclass(frozen=True)
class ExperimentRow:
    experiment_id: str
    backend: str
    group: str
    num_clients: str
    dataset: str
    distribution: str
    approach: str
    executor_cfg: str
    server_cfg: str
    client_cfg: str
    server_script: str
    server_config: str
    remote_output_dir: str


FIELDNAMES = [
    "experiment_id",
    "backend",
    "group",
    "num_clients",
    "dataset",
    "distribution",
    "approach",
    "executor_cfg",
    "server_cfg",
    "client_cfg",
    "server_script",
    "server_config",
    "remote_output_dir",
]


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()

    if normalized in {"1", "true", "yes", "y", "on"}:
        return True

    if normalized in {"0", "false", "no", "n", "off"}:
        return False

    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def read_output_directory_from_cfg(cfg_path: Path, fallback: str) -> str:
    if not cfg_path.is_file():
        return fallback

    parser = ConfigParser(interpolation=None)
    parser.read(cfg_path)

    candidates = [
        ("Output Settings", "output_directory"),
        ("Output Settings", "output_dir"),
        ("Output Settings", "results_directory"),
        ("Output Settings", "results_dir"),
        ("General Settings", "output_directory"),
        ("General Settings", "output_dir"),
    ]

    for section, option in candidates:
        if parser.has_option(section, option):
            value = parser.get(section, option).strip()
            if value:
                return value

    return fallback


def add_distributed_row(
    rows: list[ExperimentRow],
    *,
    experiment_id: str,
    group: str,
    num_clients: str,
    dataset: str,
    distribution: str,
    approach: str,
    executor_cfg: Path,
    server_cfg: Path,
    client_cfg: Path,
) -> None:
    rows.append(
        ExperimentRow(
            experiment_id=experiment_id,
            backend="distributed_flower",
            group=group,
            num_clients=num_clients,
            dataset=dataset,
            distribution=distribution,
            approach=approach,
            executor_cfg=str(executor_cfg),
            server_cfg=str(server_cfg),
            client_cfg=str(client_cfg),
            server_script="",
            server_config="",
            remote_output_dir="",
        )
    )


def add_server_only_row(
    rows: list[ExperimentRow],
    *,
    experiment_id: str,
    group: str,
    approach: str,
    server_script: Path,
    server_config: Path | None,
    remote_output_dir: str,
) -> None:
    rows.append(
        ExperimentRow(
            experiment_id=experiment_id,
            backend="server_only_python",
            group=group,
            num_clients="",
            dataset="",
            distribution="",
            approach=approach,
            executor_cfg="",
            server_cfg="",
            client_cfg="",
            server_script=str(server_script),
            server_config="" if server_config is None else str(server_config),
            remote_output_dir=remote_output_dir,
        )
    )


def build_fgcs_2026_experiments(experiments_root: Path) -> list[ExperimentRow]:
    rows: list[ExperimentRow] = []

    static_root = experiments_root / "static_client_availability"

    performance_root = static_root / "performance_experiments"
    dp_impact_root = static_root / "dp_impact_experiments"
    scalability_root = static_root / "scalability_experiments"
    sensitivity_root = static_root / "sensitivity_experiments"

    performance_client_systems = ["50_clients", "100_clients"]
    performance_datasets = ["cifar_10", "fashion_mnist"]
    performance_distributions = ["iid", "non_iid"]
    performance_approaches = [
        "fedavg",
        "oort",
        "mec",
        "ecmtc",
        "divfl",
        "ecsm",
        "metacsfl",
    ]

    for num_clients in performance_client_systems:
        for dataset in performance_datasets:
            for distribution in performance_distributions:
                base = performance_root / num_clients / dataset / distribution

                for approach in performance_approaches:
                    experiment_id = (
                        f"performance__{num_clients}__{dataset}__"
                        f"{distribution}__{approach}"
                    )

                    add_distributed_row(
                        rows,
                        experiment_id=experiment_id,
                        group="performance_experiments",
                        num_clients=num_clients,
                        dataset=dataset,
                        distribution=distribution,
                        approach=approach,
                        executor_cfg=base / f"{approach}.cfg",
                        server_cfg=base / f"{approach}_server.cfg",
                        client_cfg=base / "client.cfg",
                    )

    dp_impact_base = dp_impact_root / "100_clients" / "cifar_10" / "non_iid"
    dp_impact_approaches = ["metacsfl", "metacsfl_no_privacy"]

    for approach in dp_impact_approaches:
        experiment_id = (
            f"dp_impact__100_clients__cifar_10__non_iid__{approach}"
        )

        add_distributed_row(
            rows,
            experiment_id=experiment_id,
            group="dp_impact_experiments",
            num_clients="100_clients",
            dataset="cifar_10",
            distribution="non_iid",
            approach=approach,
            executor_cfg=dp_impact_base / f"{approach}.cfg",
            server_cfg=dp_impact_base / f"{approach}_server.cfg",
            client_cfg=dp_impact_base / "client.cfg",
        )

    scalability_script = scalability_root / "scalability_experiments.py"
    scalability_approaches = [
        "fedavg",
        "oort",
        "mec",
        "ecmtc",
        "divfl",
        "ecsm",
        "metacsfl_015",
        "metacsfl_025",
        "metacsfl_050",
    ]

    for approach in scalability_approaches:
        cfg = scalability_root / f"scalability_experiments_{approach}.cfg"
        fallback_output = (
            f"results/static_client_availability/"
            f"scalability_results/{approach}"
        )
        remote_output_dir = read_output_directory_from_cfg(cfg, fallback_output)

        add_server_only_row(
            rows,
            experiment_id=f"scalability__{approach}",
            group="scalability_experiments",
            approach=approach,
            server_script=scalability_script,
            server_config=cfg,
            remote_output_dir=remote_output_dir,
        )

    sensitivity_script = sensitivity_root / "metacsfl_sensitivity_experiments.py"

    add_server_only_row(
        rows,
        experiment_id="sensitivity__metacsfl",
        group="sensitivity_experiments",
        approach="metacsfl",
        server_script=sensitivity_script,
        server_config=None,
        remote_output_dir="results/static_client_availability/sensitivity_results",
    )

    return rows


def validate_rows(rows: list[ExperimentRow]) -> list[str]:
    problems: list[str] = []
    seen_ids: set[str] = set()

    for row in rows:
        if row.experiment_id in seen_ids:
            problems.append(f"Duplicate experiment_id: {row.experiment_id}")
        seen_ids.add(row.experiment_id)

        if row.backend == "distributed_flower":
            for label, value in [
                ("executor_cfg", row.executor_cfg),
                ("server_cfg", row.server_cfg),
                ("client_cfg", row.client_cfg),
            ]:
                if not value:
                    problems.append(
                        f"{row.experiment_id}: missing {label} path"
                    )
                    continue

                if not Path(value).is_file():
                    problems.append(
                        f"{row.experiment_id}: {label} not found: {value}"
                    )

        elif row.backend == "server_only_python":
            if not row.server_script:
                problems.append(
                    f"{row.experiment_id}: missing server_script path"
                )
            elif not Path(row.server_script).is_file():
                problems.append(
                    f"{row.experiment_id}: server_script not found: "
                    f"{row.server_script}"
                )

            if row.server_config and not Path(row.server_config).is_file():
                problems.append(
                    f"{row.experiment_id}: server_config not found: "
                    f"{row.server_config}"
                )

        else:
            problems.append(
                f"{row.experiment_id}: unsupported backend: {row.backend}"
            )

    return problems


def write_manifest(rows: list[ExperimentRow], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for row in rows:
            writer.writerow(asdict(row))


def write_summary(
    rows: list[ExperimentRow],
    output: Path,
    pack: str,
    experiments_root: Path,
    validation_problems: list[str],
) -> None:
    summary = {
        "pack": pack,
        "experiments_root": str(experiments_root),
        "total": len(rows),
        "by_backend": {},
        "by_group": {},
        "validation_problem_count": len(validation_problems),
        "validation_problems": validation_problems,
    }

    for row in rows:
        summary["by_backend"][row.backend] = (
            summary["by_backend"].get(row.backend, 0) + 1
        )
        summary["by_group"][row.group] = (
            summary["by_group"].get(row.group, 0) + 1
        )

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate explicit MetaCS-FL experiment pack manifests."
    )
    parser.add_argument(
        "--pack",
        required=True,
        choices=sorted(SUPPORTED_PACKS),
        help="Pack name to generate.",
    )
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=Path("experiments"),
        help="Path to the experiments root directory. Default: experiments",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output campaign_manifest.csv path.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional JSON summary path. Default: <output>.summary.json",
    )
    parser.add_argument(
        "--validate",
        type=parse_bool,
        default=True,
        help="Validate that all referenced files exist. Default: true",
    )

    args = parser.parse_args()

    experiments_root = args.experiments_root

    if args.pack == "fgcs_2026_experiments":
        rows = build_fgcs_2026_experiments(experiments_root)
    else:
        raise ValueError(f"Unsupported pack: {args.pack}")

    validation_problems = validate_rows(rows)

    summary_output = (
        args.summary_output
        if args.summary_output is not None
        else args.output.with_suffix(args.output.suffix + ".summary.json")
    )

    write_manifest(rows, args.output)
    write_summary(
        rows=rows,
        output=summary_output,
        pack=args.pack,
        experiments_root=experiments_root,
        validation_problems=validation_problems,
    )

    print(f"Pack: {args.pack}")
    print(f"Experiments root: {experiments_root}")
    print(f"Manifest: {args.output}")
    print(f"Summary: {summary_output}")
    print(f"Total experiments: {len(rows)}")

    by_backend: dict[str, int] = {}
    by_group: dict[str, int] = {}

    for row in rows:
        by_backend[row.backend] = by_backend.get(row.backend, 0) + 1
        by_group[row.group] = by_group.get(row.group, 0) + 1

    print("By backend:")
    for key, value in sorted(by_backend.items()):
        print(f"  {key}: {value}")

    print("By group:")
    for key, value in sorted(by_group.items()):
        print(f"  {key}: {value}")

    if validation_problems:
        print()
        print("Validation problems:", file=sys.stderr)
        for problem in validation_problems:
            print(f"  - {problem}", file=sys.stderr)

        if args.validate:
            sys.exit(1)


if __name__ == "__main__":
    main()
