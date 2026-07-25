#!/usr/bin/env python3
"""Standalone validator for a precomputed topology_step fixture package.

The validation logic is data driven.  Fixture-specific expected counts and
paths live only in package_manifest.json under ``fixture_expectations``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TOPOLOGY_FIELDS = {
    "topology_id",
    "topology_step_id",
    "step_order",
    "parent_topology_step_id",
    "assembly_cycle_id",
    "operation_type",
    "stage_id",
    "input_subassembly_id",
    "result_subassembly_id",
    "added_part_ids",
    "removed_part_ids",
    "activated_interface_ids",
    "deactivated_interface_ids",
    "activated_boundary_ids",
    "deactivated_boundary_ids",
    "activated_load_ids",
    "removed_load_ids",
    "activated_joint_ids",
    "deactivated_joint_ids",
    "operator_set_id",
    "solve_required",
    "reference_state_id",
    "measurement_checkpoint_id",
    "notes",
}


def read_rows(root: Path, relative: str) -> list[dict[str, str]]:
    with (root / relative).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def split_ids(value: object) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in str(value).split(";") if item.strip()]


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    detail: str,
    *,
    blocking: bool = True,
) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "status": "PASS" if passed else ("FAIL" if blocking else "WARN"),
            "blocking": bool(blocking),
            "detail": detail,
        }
    )


def _edges(interfaces: Iterable[dict[str, str]]) -> set[frozenset[str]]:
    return {
        frozenset((row.get("part_i", ""), row.get("part_j", "")))
        for row in interfaces
        if row.get("part_i") and row.get("part_j")
    }


def _path_exists(path: list[str], edges: set[frozenset[str]]) -> bool:
    return len(path) >= 2 and all(
        frozenset((left, right)) in edges for left, right in zip(path, path[1:])
    )


def _array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _manifest_layout_ok(
    row: dict[str, str], array: np.ndarray, vector_dimension: int
) -> bool:
    row_layout = str(row.get("row_layout_id_optional", "")).strip()
    column_layout = str(row.get("column_layout_id_optional", "")).strip()
    if array.ndim >= 1 and array.shape[0] == vector_dimension and not row_layout:
        return False
    if array.ndim >= 2 and array.shape[1] == vector_dimension and not column_layout:
        return False
    return True


def validate_package(root: Path, *, include_attachment_check: bool = True) -> dict[str, Any]:
    root = root.resolve()
    checks: list[dict[str, Any]] = []
    manifest = json.loads((root / "package_manifest.json").read_text(encoding="utf-8"))
    expected = manifest.get("fixture_expectations", {})

    parts = read_rows(root, "I0/part.csv")
    interfaces = read_rows(root, "I0/interface.csv")
    domains = read_rows(root, "I_Gamma/contact_domain.csv")
    points = read_rows(root, "I_Gamma/contact_point.csv")
    route = read_rows(root, "I0/assembly_topology.csv")
    layout = read_rows(root, "matrices/vector_layout.csv")
    part_ids = {row.get("part_id", "") for row in parts}
    interface_ids = {row.get("interface_id", "") for row in interfaces}
    domain_ids = {row.get("contact_domain_id", "") for row in domains}
    edge_set = _edges(interfaces)

    endpoints_ok = all(
        row.get("part_i") in part_ids and row.get("part_j") in part_ids
        for row in interfaces
    )
    _check(
        checks,
        "interface endpoint foreign keys",
        endpoints_ok,
        f"parts={len(parts)}, interfaces={len(interfaces)}",
    )
    for field, actual in (
        ("part_count", len(parts)),
        ("interface_count", len(interfaces)),
        ("topology_step_count", len(route)),
    ):
        wanted = expected.get(field)
        _check(
            checks,
            f"fixture expectation: {field}",
            wanted is None or int(wanted) == actual,
            f"expected={wanted}, actual={actual}",
        )

    serial_paths = expected.get("serial_paths", [])
    serial_ok = all(_path_exists(list(path), edge_set) for path in serial_paths)
    _check(
        checks,
        "serial paths from actual part-interface graph",
        serial_ok,
        json.dumps(serial_paths, ensure_ascii=False),
    )
    parallel_specs = expected.get("parallel_paths", [])
    parallel_ok = True
    parallel_detail: list[dict[str, Any]] = []
    for spec in parallel_specs:
        paths = [list(path) for path in spec.get("paths", [])]
        path_status = [_path_exists(path, edge_set) for path in paths]
        endpoints = list(spec.get("endpoints", []))
        same_endpoints = all(
            len(path) >= 2 and {path[0], path[-1]} == set(endpoints) for path in paths
        )
        distinct = len({tuple(path) for path in paths}) == len(paths) and len(paths) >= 2
        ok = all(path_status) and same_endpoints and distinct
        parallel_ok &= ok
        parallel_detail.append(
            {"endpoints": endpoints, "paths": paths, "path_status": path_status}
        )
    _check(
        checks,
        "parallel direct and bridge paths from actual graph",
        parallel_ok,
        json.dumps(parallel_detail, ensure_ascii=False),
    )

    domain_to_interface = {
        row.get("contact_domain_id", ""): row.get("interface_id", "") for row in domains
    }
    point_count_by_domain = {domain_id: 0 for domain_id in domain_ids}
    point_fk_ok = True
    for point in points:
        domain_id = point.get("contact_domain_id", "")
        point_fk_ok &= domain_id in domain_ids
        point_count_by_domain[domain_id] = point_count_by_domain.get(domain_id, 0) + 1
    point_count_by_interface: dict[str, int] = {}
    for domain_id, count in point_count_by_domain.items():
        interface_id = domain_to_interface.get(domain_id, "")
        point_count_by_interface[interface_id] = (
            point_count_by_interface.get(interface_id, 0) + count
        )
    _check(
        checks,
        "contact point foreign keys and per-interface grouping",
        point_fk_ok and set(domain_to_interface.values()) <= interface_ids,
        json.dumps(point_count_by_interface, ensure_ascii=False, sort_keys=True),
    )
    expected_points = expected.get("contact_points_per_interface")
    counts_ok = expected_points is None or (
        set(point_count_by_interface) == interface_ids
        and all(count == int(expected_points) for count in point_count_by_interface.values())
    )
    _check(
        checks,
        "fixture expectation: contact points per interface",
        counts_ok,
        f"expected={expected_points}, actual="
        f"{json.dumps(point_count_by_interface, ensure_ascii=False, sort_keys=True)}",
    )

    intervals = sorted(
        (int(row["start_index"]), int(row["end_index"]), row) for row in layout
    )
    covered = [
        index
        for start, end, _ in intervals
        for index in range(start, end + 1)
    ]
    vector_dimension = len(points)
    layout_ok = (
        covered == list(range(vector_dimension))
        and len(set(covered)) == vector_dimension
        and all(
            row.get("object_id", "") in interface_ids
            and row.get("contact_domain_id", "") in domain_ids
            and end - start + 1
            == point_count_by_domain.get(row.get("contact_domain_id", ""), -1)
            for start, end, row in intervals
        )
    )
    _check(
        checks,
        "vector layout contiguous, unique and data-linked",
        layout_ok,
        f"dimension={vector_dimension}, intervals={[(a, b) for a, b, _ in intervals]}",
    )
    expected_dimension = expected.get("vector_dimension")
    _check(
        checks,
        "fixture expectation: vector dimension",
        expected_dimension is None or int(expected_dimension) == vector_dimension,
        f"expected={expected_dimension}, actual={vector_dimension}",
    )

    route_ids = [row.get("topology_step_id", "") for row in route]
    route_id_set = set(route_ids)
    orders = [int(row.get("step_order", "0")) for row in route]
    parent_ok = len(route_ids) == len(route_id_set) and orders == sorted(orders)
    for index, row in enumerate(route):
        parent = row.get("parent_topology_step_id", "")
        if index == 0:
            parent_ok &= not parent
        else:
            parent_ok &= parent in set(route_ids[:index])
    _check(
        checks,
        "topology route IDs, order and parent chain",
        parent_ok,
        f"steps={route_ids}",
    )
    route_fk_ok = all(
        set(split_ids(row.get("added_part_ids"))) <= part_ids
        and set(split_ids(row.get("removed_part_ids"))) <= part_ids
        and set(split_ids(row.get("activated_interface_ids"))) <= interface_ids
        and set(split_ids(row.get("deactivated_interface_ids"))) <= interface_ids
        for row in route
    )
    _check(
        checks,
        "topology route part/interface foreign keys",
        route_fk_ok,
        "all activated/deactivated IDs resolve",
    )
    multi_activation = [
        row["topology_step_id"]
        for row in route
        if len(split_ids(row.get("activated_interface_ids"))) > 1
    ]
    expected_multi = expected.get("minimum_multi_interface_activation_steps", 0)
    _check(
        checks,
        "same-step multi-interface activation from route data",
        len(multi_activation) >= int(expected_multi),
        f"expected_min={expected_multi}, actual={multi_activation}",
    )

    manifest_rows = read_rows(root, "matrices/matrix_manifest.csv")
    manifest_keys = [row.get("npz_key", "") for row in manifest_rows]
    with np.load(root / "matrices/multi_part_matrices.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    key_ok = (
        len(manifest_keys) == len(set(manifest_keys))
        and set(manifest_keys) == set(arrays)
    )
    _check(
        checks,
        "MatrixManifest and NPZ key set",
        key_ok,
        f"manifest={len(manifest_keys)}, npz={len(arrays)}, unique={len(set(manifest_keys))}",
    )
    manifest_errors: list[str] = []
    for row in manifest_rows:
        key = row.get("npz_key", "")
        if key not in arrays:
            manifest_errors.append(f"{key}:missing")
            continue
        array = arrays[key]
        try:
            declared_shape = tuple(int(value) for value in json.loads(row.get("shape", "[]")))
        except (TypeError, ValueError, json.JSONDecodeError):
            declared_shape = ()
        dtype_ok = str(row.get("dtype", "")).strip() == str(array.dtype)
        layout_row_ok = _manifest_layout_ok(row, array, vector_dimension)
        if declared_shape != array.shape or not dtype_ok or not layout_row_ok:
            manifest_errors.append(
                f"{key}:shape={declared_shape}/{array.shape},"
                f"dtype={row.get('dtype')}/{array.dtype},layout={layout_row_ok}"
            )
    _check(
        checks,
        "MatrixManifest shape, dtype and row/column layout",
        not manifest_errors,
        f"errors={manifest_errors[:10]}",
    )

    operator_errors: list[str] = []
    cross_norms: dict[str, dict[str, float]] = {}
    layout_by_interface = {
        row["object_id"]: np.arange(
            int(row["start_index"]), int(row["end_index"]) + 1, dtype=int
        )
        for row in layout
    }
    active_interfaces: set[str] = set()
    solved_steps = []
    for row in route:
        active_interfaces.difference_update(split_ids(row.get("deactivated_interface_ids")))
        active_interfaces.update(split_ids(row.get("activated_interface_ids")))
        if not truthy(row.get("solve_required")):
            continue
        solved_steps.append(row["topology_step_id"])
        operator = row.get("operator_set_id", "")
        required_keys = {
            name: f"{name}_{operator}"
            for name in ("Q", "W_STRUCT", "CN", "W_TOTAL", "U_FREE")
        }
        if any(key not in arrays for key in required_keys.values()):
            operator_errors.append(f"{row['topology_step_id']}:missing operator arrays")
            continue
        q = np.asarray(arrays[required_keys["Q"]], dtype=float)
        ws = np.asarray(arrays[required_keys["W_STRUCT"]], dtype=float)
        cn = np.asarray(arrays[required_keys["CN"]], dtype=float)
        wt = np.asarray(arrays[required_keys["W_TOTAL"]], dtype=float)
        shape_ok = (
            q.shape == (vector_dimension,)
            and ws.shape == cn.shape == wt.shape == (vector_dimension, vector_dimension)
        )
        relation_ok = shape_ok and np.allclose(wt, ws + cn, atol=1e-12)
        symmetric_ok = shape_ok and np.allclose(ws, ws.T, atol=1e-12)
        if not (shape_ok and relation_ok and symmetric_ok):
            operator_errors.append(
                f"{row['topology_step_id']}:shape={shape_ok},relation={relation_ok},"
                f"symmetric={symmetric_ok}"
            )
        step_norms: dict[str, float] = {}
        ordered_active = sorted(active_interfaces)
        for index, left in enumerate(ordered_active):
            for right in ordered_active[index + 1 :]:
                if left not in layout_by_interface or right not in layout_by_interface:
                    operator_errors.append(
                        f"{row['topology_step_id']}:layout missing {left}|{right}"
                    )
                    continue
                norm = float(
                    np.linalg.norm(
                        ws[
                            np.ix_(
                                layout_by_interface[left],
                                layout_by_interface[right],
                            )
                        ]
                    )
                )
                step_norms[f"{left}|{right}"] = norm
        cross_norms[row["topology_step_id"]] = step_norms
    _check(
        checks,
        "precomputed topology operator completeness and identity",
        not operator_errors,
        f"solved_steps={solved_steps}, errors={operator_errors[:10]}",
    )
    require_cross = bool(expected.get("require_nonzero_cross_blocks", False))
    cross_ok = not require_cross or all(
        norm > 1e-12 for norms in cross_norms.values() for norm in norms.values()
    )
    _check(
        checks,
        "active W_struct cross-interface block norms",
        cross_ok,
        json.dumps(cross_norms, ensure_ascii=False, sort_keys=True),
    )

    oracle = read_rows(root, "validation/topology_step_lcp_oracle.csv")
    oracle_steps = {row.get("topology_step_id", "") for row in oracle}
    oracle_ok = oracle_steps == set(solved_steps) and all(
        float(row.get("equilibrium_residual", "inf")) <= 1e-9
        and float(row.get("complementarity_residual", "inf")) <= 1e-8
        for row in oracle
    )
    _check(
        checks,
        "topology_step independent LCP oracle registration",
        oracle_ok,
        f"oracle_steps={sorted(oracle_steps)}, solved_steps={solved_steps}",
    )

    dictionary = read_rows(root, "field_dictionary.csv")
    dictionary_pairs = {
        (row.get("file_path", ""), row.get("field_name", "")) for row in dictionary
    }
    csv_pairs: set[tuple[str, str]] = set()
    for csv_path in root.rglob("*.csv"):
        relative = csv_path.relative_to(root).as_posix()
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            fields = csv.DictReader(handle).fieldnames or []
        csv_pairs.update((relative, field) for field in fields)
    dictionary_ok = csv_pairs <= dictionary_pairs
    topology_dictionary_ok = {
        ("I0/assembly_topology.csv", field) for field in TOPOLOGY_FIELDS
    } <= dictionary_pairs
    required_dictionary_columns = {
        "file_path",
        "object_name",
        "field_name",
        "data_type",
        "required",
        "cardinality",
        "unit",
        "enum_or_format",
        "key_semantics",
        "missing_handling",
        "description",
        "example_value",
    }
    dictionary_columns_ok = (
        bool(dictionary)
        and required_dictionary_columns <= set(dictionary[0])
    )
    _check(
        checks,
        "field_dictionary covers actual CSV schema and TopologyStepSpec",
        dictionary_ok and topology_dictionary_ok and dictionary_columns_ok,
        f"missing={sorted(csv_pairs - dictionary_pairs)[:20]}",
    )

    object_map = read_rows(root, "object_file_map.csv")
    required_objects = {
        "AssemblyTopology",
        "TopologyStepSpec",
        "TopologyStepResult",
        "ConnectionLockHistory",
        "ReleaseHistoryRecord",
        "TopologyStepLcpOracle",
        "TopologyStepOperatorMatrices",
        "TopologyStepExecutionReport",
        "ValidationResults",
    }
    map_names = {row.get("object_name", "") for row in object_map}
    map_file_ok = True
    for row in object_map:
        if truthy(row.get("is_runtime_result")):
            continue
        relative = row.get("file_path", "")
        map_file_ok &= bool(relative) and (root / relative).exists()
    _check(
        checks,
        "object_file_map formal objects and files",
        required_objects <= map_names and map_file_ok,
        f"missing_objects={sorted(required_objects - map_names)}",
    )

    truth_ok = (
        manifest.get("data_nature") == "SYNTHETIC_NUMERICAL_CONSISTENCY_CASE"
        and manifest.get("engineering_claim_allowed") is False
        and "fixture_expectations" in manifest
    )
    _check(
        checks,
        "synthetic-data truthfulness boundary",
        truth_ok,
        f"data_nature={manifest.get('data_nature')}, "
        f"engineering_claim_allowed={manifest.get('engineering_claim_allowed')}",
    )

    if include_attachment_check:
        stored_path = root / "validation" / "test_results.json"
        run_log = read_rows(root, "validation/run_log.csv")
        quality_gate = read_rows(root, "validation/quality_gate.csv")
        stored = (
            json.loads(stored_path.read_text(encoding="utf-8"))
            if stored_path.exists()
            else {}
        )
        stored_core = [
            (item.get("name"), item.get("status"), item.get("detail"))
            for item in stored.get("checks", [])
            if item.get("name") != "validation attachments match current package"
        ]
        current_core = [
            (item["name"], item["status"], item["detail"]) for item in checks
        ]
        attachment_ok = (
            stored.get("package_id") == root.name
            and stored.get("matrix_manifest_count") == len(manifest_rows)
            and stored.get("npz_key_count") == len(arrays)
            and stored_core == current_core
            and bool(run_log)
            and run_log[0].get("input_package_id") == root.name
            and int(run_log[0].get("matrix_manifest_count", "-1"))
            == len(manifest_rows)
            and int(run_log[0].get("npz_key_count", "-1")) == len(arrays)
            and bool(quality_gate)
            and quality_gate[0].get("target_object_ids") == root.name
            and quality_gate[0].get("pass_fail") == "PASS"
        )
        _check(
            checks,
            "validation attachments match current package",
            attachment_ok,
            f"matrix_manifest={len(manifest_rows)}, npz={len(arrays)}, "
            f"stored_status={stored.get('status')}",
        )

    blocking_failures = sum(
        item["status"] == "FAIL" and item["blocking"] for item in checks
    )
    passed = sum(item["status"] == "PASS" for item in checks)
    return {
        "package": root.name,
        "package_id": root.name,
        "status": "PASS" if blocking_failures == 0 else "FAIL",
        "passed": passed,
        "total": len(checks),
        "blocking_fail_count": blocking_failures,
        "matrix_manifest_count": len(manifest_rows),
        "npz_key_count": len(arrays),
        "checks": checks,
        "matrix_digests": {
            key: _array_digest(value) for key, value in sorted(arrays.items())
        },
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# topology_step package self-validation",
        "",
        f"- Package: `{report['package_id']}`",
        f"- Status: **{report['status']}**",
        f"- Checks: {report['passed']}/{report['total']} PASS",
        f"- Blocking failures: {report['blocking_fail_count']}",
        f"- MatrixManifest rows: {report['matrix_manifest_count']}",
        f"- NPZ keys: {report['npz_key_count']}",
        "",
        "| Check | Status | Blocking | Detail |",
        "|---|---:|---:|---|",
    ]
    for item in report["checks"]:
        detail = str(item["detail"]).replace("|", "/").replace("\n", " ")
        lines.append(
            f"| {item['name']} | {item['status']} | "
            f"{str(item['blocking']).lower()} | {detail} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    report = validate_package(args.root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.write_report:
        validation = args.root.resolve() / "validation"
        (validation / "test_results.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (validation / "TEST_RESULTS.md").write_text(
            markdown_report(report), encoding="utf-8"
        )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
