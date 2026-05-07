#!/usr/bin/env python3
"""
Safely merge distributed MetaCS-FL output folders gathered from multiple nodes.

Expected gathered structure:

    gathered_results/<RUN_ID>/
        node_<node_a>/output_execution/exec_a_1/...
        node_<node_b>/output_execution/exec_a_1/...
        node_<node_c>/output_execution/exec_b_1/...
        distributed_run_manifest.txt
        nodes_<RUN_ID>.txt

The script preserves the relative path after each node_* folder. For example:

    node_paradoxe-16/output_execution/exec_a_1/output/individual_fit_metrics_history.csv
    node_paradoxe-17/output_execution/exec_a_1/output/individual_fit_metrics_history.csv

becomes:

    merged/output_execution/exec_a_1/output/individual_fit_metrics_history.csv

CSV files with the same relative path are appended row-wise when their headers match.
If a merged CSV contains a column named 'comm_round', rows are sorted by it ascending.
Non-CSV files with the same relative path are copied once if identical; otherwise they
are written to a conflict folder to avoid data loss.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_NODE_PREFIX = "node_"
CONFLICT_DIR_NAME = "_merge_conflicts"
REPORT_FILE_NAME = "merge_report.json"


@dataclass(frozen=True)
class SourceFile:
    node_name: str
    source_path: Path
    relative_path: Path


def parse_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in value)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def discover_node_dirs(gathered_root: Path, node_prefix: str = DEFAULT_NODE_PREFIX) -> list[Path]:
    return sorted(p for p in gathered_root.iterdir() if p.is_dir() and p.name.startswith(node_prefix))


def discover_source_files(gathered_root: Path, node_prefix: str = DEFAULT_NODE_PREFIX) -> dict[Path, list[SourceFile]]:
    grouped: dict[Path, list[SourceFile]] = {}
    node_dirs = discover_node_dirs(gathered_root, node_prefix=node_prefix)

    for node_dir in node_dirs:
        node_name = node_dir.name
        for source_path in sorted(p for p in node_dir.rglob("*") if p.is_file()):
            relative_path = source_path.relative_to(node_dir)
            grouped.setdefault(relative_path, []).append(
                SourceFile(node_name=node_name, source_path=source_path, relative_path=relative_path)
            )

    return grouped


def copy_top_level_metadata(gathered_root: Path, output_root: Path, node_prefix: str = DEFAULT_NODE_PREFIX) -> list[str]:
    copied: list[str] = []
    metadata_dir = output_root / "_gather_metadata"

    for item in sorted(gathered_root.iterdir()):
        if item.is_dir() and item.name.startswith(node_prefix):
            continue

        target = metadata_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
            copied.append(str(target.relative_to(output_root)))
        elif item.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
            copied.append(str(target.relative_to(output_root)))

    return copied


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return [], []
        header = list(reader.fieldnames)
        rows = []
        for row in reader:
            # csv.DictReader may create a None key when a row has too many columns.
            if None in row:
                raise ValueError(f"Malformed CSV row in {path}: too many columns: {row[None]}")
            rows.append({k: ("" if v is None else v) for k, v in row.items()})
        return header, rows


def numeric_or_text(value: str):
    value = "" if value is None else str(value).strip()
    if value == "":
        return (1, 0, "")
    try:
        return (0, float(value), "")
    except ValueError:
        return (0, 0, value)


def sort_rows_if_possible(header: list[str], rows: list[dict[str, str]]) -> list[dict[str, str]]:
    lower_to_original = {h.lower(): h for h in header}

    primary_sort_columns = []
    for candidate in ["comm_round", "round", "server_round"]:
        if candidate in lower_to_original:
            primary_sort_columns.append(lower_to_original[candidate])
            break

    secondary_candidates = ["client_id", "client", "phase", "timestamp", "time"]
    for candidate in secondary_candidates:
        if candidate in lower_to_original:
            primary_sort_columns.append(lower_to_original[candidate])

    if not primary_sort_columns:
        return rows

    return sorted(rows, key=lambda row: tuple(numeric_or_text(row.get(col, "")) for col in primary_sort_columns))


def write_csv(path: Path, header: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def copy_conflict_files(files: list[SourceFile], output_root: Path, reason: str) -> list[str]:
    copied_paths: list[str] = []
    for sf in files:
        target = output_root / CONFLICT_DIR_NAME / reason / sf.relative_path.parent / f"{sf.relative_path.name}.from_{safe_name(sf.node_name)}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sf.source_path, target)
        copied_paths.append(str(target.relative_to(output_root)))
    return copied_paths


def merge_csv_group(
    relative_path: Path,
    files: list[SourceFile],
    output_root: Path,
    add_source_node: bool,
    deduplicate_rows: bool,
) -> dict:
    non_empty: list[tuple[SourceFile, list[str], list[dict[str, str]]]] = []
    empty_files: list[str] = []

    for sf in files:
        header, rows = read_csv(sf.source_path)
        if not header:
            empty_files.append(str(sf.source_path))
            continue
        non_empty.append((sf, header, rows))

    if not non_empty:
        # All files were empty/headerless. Copy the first one as-is.
        target = output_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(files[0].source_path, target)
        return {
            "action": "copied_empty_or_headerless_csv",
            "relative_path": str(relative_path),
            "sources": [str(sf.source_path) for sf in files],
            "output": str(target),
            "empty_or_headerless_sources": empty_files,
        }

    reference_header = non_empty[0][1]
    mismatched = [(sf, header) for sf, header, _ in non_empty if header != reference_header]
    if mismatched:
        copied = copy_conflict_files(files, output_root, reason="csv_header_mismatch")
        return {
            "action": "conflict_csv_header_mismatch",
            "relative_path": str(relative_path),
            "sources": [str(sf.source_path) for sf in files],
            "conflict_copies": copied,
            "reference_header": reference_header,
            "mismatched_headers": [
                {"source": str(sf.source_path), "header": header} for sf, header in mismatched
            ],
        }

    output_header = list(reference_header)
    if add_source_node and "source_node" not in output_header:
        output_header.append("source_node")

    merged_rows: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()

    for sf, _, rows in non_empty:
        for row in rows:
            row_out = dict(row)
            if add_source_node:
                row_out["source_node"] = sf.node_name

            if deduplicate_rows:
                # Deduplicate using the output header so source_node participates only when explicitly requested.
                key = tuple(row_out.get(col, "") for col in output_header)
                if key in seen:
                    continue
                seen.add(key)

            merged_rows.append(row_out)

    merged_rows = sort_rows_if_possible(output_header, merged_rows)

    target = output_root / relative_path
    write_csv(target, output_header, merged_rows)

    return {
        "action": "merged_csv",
        "relative_path": str(relative_path),
        "sources": [str(sf.source_path) for sf in files],
        "output": str(target),
        "source_count": len(files),
        "row_count": len(merged_rows),
        "sorted_by": "comm_round_or_round_if_present",
        "empty_or_headerless_sources": empty_files,
    }


def merge_non_csv_group(relative_path: Path, files: list[SourceFile], output_root: Path) -> dict:
    if len(files) == 1:
        target = output_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(files[0].source_path, target)
        return {
            "action": "copied_unique_file",
            "relative_path": str(relative_path),
            "sources": [str(files[0].source_path)],
            "output": str(target),
        }

    hashes = {file_sha256(sf.source_path) for sf in files}
    if len(hashes) == 1:
        target = output_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(files[0].source_path, target)
        return {
            "action": "copied_identical_duplicate_file_once",
            "relative_path": str(relative_path),
            "sources": [str(sf.source_path) for sf in files],
            "output": str(target),
            "source_count": len(files),
        }

    copied = copy_conflict_files(files, output_root, reason="non_csv_name_conflict")
    return {
        "action": "conflict_non_csv_name_conflict",
        "relative_path": str(relative_path),
        "sources": [str(sf.source_path) for sf in files],
        "conflict_copies": copied,
    }


def merge_distributed_outputs(
    gathered_root: Path,
    output_root: Path,
    node_prefix: str = DEFAULT_NODE_PREFIX,
    add_source_node: bool = False,
    deduplicate_rows: bool = False,
    clean_output: bool = False,
) -> dict:
    gathered_root = gathered_root.resolve()
    output_root = output_root.resolve()

    if not gathered_root.is_dir():
        raise FileNotFoundError(f"Gathered root does not exist or is not a directory: {gathered_root}")

    if clean_output and output_root.exists():
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    node_dirs = discover_node_dirs(gathered_root, node_prefix=node_prefix)
    if not node_dirs:
        raise RuntimeError(f"No node folders found in {gathered_root} with prefix '{node_prefix}'")

    report: dict = {
        "gathered_root": str(gathered_root),
        "output_root": str(output_root),
        "node_prefix": node_prefix,
        "nodes": [p.name for p in node_dirs],
        "add_source_node": add_source_node,
        "deduplicate_rows": deduplicate_rows,
        "clean_output": clean_output,
        "top_level_metadata_copied": [],
        "files": [],
        "summary": {},
    }

    report["top_level_metadata_copied"] = copy_top_level_metadata(gathered_root, output_root, node_prefix=node_prefix)

    grouped = discover_source_files(gathered_root, node_prefix=node_prefix)

    for relative_path, files in sorted(grouped.items(), key=lambda item: str(item[0])):
        try:
            if relative_path.suffix.lower() == ".csv":
                entry = merge_csv_group(
                    relative_path=relative_path,
                    files=files,
                    output_root=output_root,
                    add_source_node=add_source_node,
                    deduplicate_rows=deduplicate_rows,
                )
            else:
                entry = merge_non_csv_group(relative_path, files, output_root)
        except Exception as e:
            copied = copy_conflict_files(files, output_root, reason="merge_exception")
            entry = {
                "action": "conflict_merge_exception",
                "relative_path": str(relative_path),
                "sources": [str(sf.source_path) for sf in files],
                "error": repr(e),
                "conflict_copies": copied,
            }
        report["files"].append(entry)

    summary: dict[str, int] = {}
    for entry in report["files"]:
        action = entry.get("action", "unknown")
        summary[action] = summary.get(action, 0) + 1
    report["summary"] = summary

    report_path = output_root / REPORT_FILE_NAME
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safely merge distributed MetaCS-FL gathered_results folders."
    )
    parser.add_argument(
        "gathered_root",
        type=Path,
        help="Path to gathered_results/<RUN_ID> containing node_* folders.",
    )
    parser.add_argument(
        "output_root",
        type=Path,
        help="Path where the merged output should be written.",
    )
    parser.add_argument(
        "--node-prefix",
        type=str,
        default=DEFAULT_NODE_PREFIX,
        help="Prefix used by gathered node folders. Default: node_",
    )
    parser.add_argument(
        "--add-source-node",
        type=parse_bool,
        default=False,
        help="Add a source_node column to merged CSV rows. Default: false",
    )
    parser.add_argument(
        "--deduplicate-rows",
        type=parse_bool,
        default=False,
        help="Remove exact duplicate CSV rows after merging. Default: false",
    )
    parser.add_argument(
        "--clean-output",
        type=parse_bool,
        default=False,
        help="Delete output_root before writing merged results. Default: false",
    )

    args = parser.parse_args()

    report = merge_distributed_outputs(
        gathered_root=args.gathered_root,
        output_root=args.output_root,
        node_prefix=args.node_prefix,
        add_source_node=args.add_source_node,
        deduplicate_rows=args.deduplicate_rows,
        clean_output=args.clean_output,
    )

    print("Merged distributed outputs successfully.")
    print(f"Gathered root: {report['gathered_root']}")
    print(f"Output root:   {report['output_root']}")
    print("Summary:")
    for action, count in sorted(report["summary"].items()):
        print(f"  {action}: {count}")
    print(f"Report: {Path(report['output_root']) / REPORT_FILE_NAME}")


if __name__ == "__main__":
    main()
