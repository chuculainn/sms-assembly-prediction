from __future__ import annotations

from collections import defaultdict, deque
from itertools import combinations, islice, product

import networkx as nx
import numpy as np
import pandas as pd

from .data_loader import SMSPackage
from .schema_adapter import parse_literal, to_vector


def is_multi_part_package(pkg: SMSPackage) -> bool:
    return pkg.package_type == "V25_MULTI_PART"


def vector_layout(pkg: SMSPackage) -> pd.DataFrame:
    layout = pkg.raw_tables.get("matrices/vector_layout.csv", pd.DataFrame()).copy()
    if layout.empty:
        cp = pkg.contact_points.copy()
        if "interface_id" not in cp.columns:
            return pd.DataFrame()
        rows = []
        start = 0
        for order, (interface_id, block) in enumerate(cp.groupby("interface_id", sort=False)):
            rows.append({
                "block_order": order,
                "object_type": "Interface",
                "object_id": interface_id,
                "contact_domain_id": ";".join(block.get("contact_domain_id", pd.Series(dtype=str)).astype(str).unique()),
                "start_index": start,
                "end_index": start + len(block) - 1,
            })
            start += len(block)
        return pd.DataFrame(rows)
    for col in ("block_order", "start_index", "end_index"):
        layout[col] = pd.to_numeric(layout[col], errors="coerce").astype("Int64")
    return layout.sort_values("block_order").reset_index(drop=True)


def topology_summary(pkg: SMSPackage) -> dict[str, object]:
    graph = assembly_graph(pkg)
    connectivity = _connectivity_graph(graph)
    part_ids = list(graph.nodes)
    components = nx.number_connected_components(graph) if part_ids else 0
    cycle_rank = max(0, graph.number_of_edges() - graph.number_of_nodes() + components) if part_ids else 0
    max_degree = max((degree for _, degree in graph.degree), default=0)
    max_connectivity_degree = max((degree for _, degree in connectivity.degree), default=0)
    shared = [node for node, degree in graph.degree if degree >= 2]
    flexible = []
    part_rows = pkg.parts.set_index("part_id", drop=False) if "part_id" in pkg.parts.columns else pd.DataFrame()
    for node in shared:
        row = part_rows.loc[node] if not part_rows.empty and node in part_rows.index else pd.Series(dtype=object)
        descriptor = " ".join(str(row.get(c, "")) for c in (
            "role_in_assembly", "rigid_flexible_flag", "flexibility_class", "structural_type"
        )).lower()
        if "flex" in descriptor or "柔" in descriptor:
            flexible.append(node)
    return {
        "part_count": len(part_ids),
        "interface_count": len(pkg.interfaces),
        "connected_components": components,
        "connected": bool(part_ids) and components == 1,
        "cycle_rank": cycle_rank,
        "has_closed_or_parallel_path": cycle_rank > 0,
        "has_serial_path": len(part_ids) >= 3 and max_connectivity_degree >= 2,
        "max_part_degree": max_degree,
        "shared_part_count": len(shared),
        "shared_part_ids": ";".join(shared),
        "shared_flexible_part_count": len(flexible),
        "shared_flexible_part_ids": ";".join(flexible),
    }


def assembly_graph(pkg: SMSPackage) -> nx.MultiGraph:
    """Build a lossless assembly graph with one keyed edge per interface."""
    graph = nx.MultiGraph()
    for _, row in pkg.parts.iterrows():
        part_id = str(row.get("part_id", ""))
        if part_id:
            graph.add_node(part_id, **{str(k): row.get(k) for k in row.index})
    domains = pkg.raw_tables.get("I_Gamma/contact_domain.csv", pd.DataFrame())
    domain_map = {}
    if {"interface_id", "contact_domain_id"} <= set(domains.columns):
        domain_map = domains.groupby("interface_id")["contact_domain_id"].apply(
            lambda s: ";".join(s.dropna().astype(str).unique())
        ).to_dict()
    joints = pkg.raw_tables.get("I0/joint_definition.csv", pd.DataFrame())
    joint_map = {}
    if {"interface_id", "joint_id"} <= set(joints.columns):
        joint_map = joints.groupby("interface_id")["joint_id"].apply(
            lambda s: ";".join(s.dropna().astype(str).unique())
        ).to_dict()
    for _, row in pkg.interfaces.iterrows():
        interface_id = str(row.get("interface_id", ""))
        a, b = str(row.get("part_i", "")), str(row.get("part_j", ""))
        if a and b:
            attrs = {str(k): row.get(k) for k in row.index}
            attrs.update({
                "interface_id": interface_id,
                "contact_domain_id": domain_map.get(interface_id, ""),
                "joint_id": joint_map.get(interface_id, ""),
            })
            graph.add_edge(a, b, key=interface_id, **attrs)
    return graph


def _connectivity_graph(graph: nx.MultiGraph) -> nx.Graph:
    """Return a simple node-connectivity view while retaining edges in ``graph``."""
    simple = nx.Graph()
    simple.add_nodes_from(graph.nodes(data=True))
    simple.add_edges_from((a, b) for a, b in graph.edges())
    return simple


def _edge_interface_ids(graph: nx.MultiGraph, a: str, b: str) -> list[str]:
    edge_data = graph.get_edge_data(a, b, default={})
    return [
        str(data.get("interface_id", key))
        for key, data in sorted(edge_data.items(), key=lambda item: str(item[0]))
    ]


def _path_interface_combinations(graph: nx.MultiGraph, path: list[str]) -> list[tuple[str, ...]]:
    choices = [_edge_interface_ids(graph, a, b) for a, b in zip(path, path[1:])]
    if not choices or any(not choice for choice in choices):
        return []
    return list(product(*choices))


def assembly_path_summary(pkg: SMSPackage, max_paths_per_pair: int = 2) -> pd.DataFrame:
    """Describe serial paths, independent parallel alternatives and cycle basis.

    Only the first ``max_paths_per_pair`` shortest simple paths are retained for
    each endpoint pair, which keeps the report bounded for large cyclic graphs.
    """
    graph = assembly_graph(pkg)
    connectivity = _connectivity_graph(graph)
    rows: list[dict[str, object]] = []
    if graph.number_of_nodes() == 0:
        return pd.DataFrame()
    for a, b in combinations(sorted(connectivity.nodes), 2):
        if not nx.has_path(connectivity, a, b):
            continue
        node_paths = list(islice(nx.shortest_simple_paths(connectivity, a, b), max(1, max_paths_per_pair)))
        interface_routes: list[tuple[list[str], tuple[str, ...]]] = []
        for path in node_paths:
            interface_routes.extend((path, route) for route in _path_interface_combinations(graph, path))
        serial_rank = 0
        for path, route in interface_routes:
            if len(path) >= 3:
                serial_rank += 1
                rows.append({
                    "path_type": "SERIAL",
                    "endpoint_i": a,
                    "endpoint_j": b,
                    "path_rank": serial_rank,
                    "part_ids": ";".join(path),
                    "interface_ids": ";".join(route),
                    "edge_count": len(path) - 1,
                    "description": "长度不少于2个接口的装配传递路径",
                })
        if len(interface_routes) >= 2:
            for rank, (path, route) in enumerate(interface_routes, 1):
                rows.append({
                    "path_type": "PARALLEL",
                    "endpoint_i": a,
                    "endpoint_j": b,
                    "path_rank": rank,
                    "part_ids": ";".join(path),
                    "interface_ids": ";".join(route),
                    "edge_count": len(route),
                    "description": "相同端点之间由节点路径或平行接口形成的独立装配路径",
                })
    cycle_rank = 0
    for a, b in connectivity.edges():
        interface_ids = _edge_interface_ids(graph, a, b)
        for left, right in combinations(interface_ids, 2):
            cycle_rank += 1
            rows.append({
                "path_type": "CYCLE",
                "endpoint_i": a,
                "endpoint_j": a,
                "path_rank": cycle_rank,
                "part_ids": ";".join([a, b, a]),
                "interface_ids": ";".join([left, right]),
                "edge_count": 2,
                "description": "同一零件对之间的两个独立接口形成的二边闭环",
            })
    for cycle in nx.cycle_basis(connectivity):
        closed = cycle + [cycle[0]] if cycle else cycle
        for route in _path_interface_combinations(graph, closed):
            cycle_rank += 1
            rows.append({
                "path_type": "CYCLE",
                "endpoint_i": cycle[0] if cycle else "",
                "endpoint_j": cycle[0] if cycle else "",
                "path_rank": cycle_rank,
                "part_ids": ";".join(closed),
                "interface_ids": ";".join(route),
                "edge_count": len(cycle),
                "description": "NetworkX cycle_basis识别的节点闭环及其完整接口组合",
            })
    return pd.DataFrame(rows).drop_duplicates(
        subset=["path_type", "endpoint_i", "endpoint_j", "path_rank", "part_ids"]
    ).reset_index(drop=True)


def coupling_block_summary(pkg: SMSPackage, stage_id: str) -> pd.DataFrame:
    layout = vector_layout(pkg)
    matrix = pkg.matrices.get(f"W_struct__{stage_id}")
    if matrix is None:
        topology = pkg.raw_tables.get("I0/assembly_topology.csv", pd.DataFrame())
        row = topology[topology.get("topology_step_id", pd.Series(dtype=str)).astype(str).eq(str(stage_id))] if not topology.empty else pd.DataFrame()
        if not row.empty:
            operator_set_id = str(row.iloc[0].get("operator_set_id", ""))
            matrix = pkg.matrices.get(f"W_STRUCT_{operator_set_id}")
    W = np.asarray(matrix if matrix is not None else np.empty((0, 0)), dtype=float)
    if layout.empty or W.ndim != 2:
        return pd.DataFrame()
    rows = []
    for i, left in layout.iterrows():
        li = slice(int(left["start_index"]), int(left["end_index"]) + 1)
        for j, right in layout.iterrows():
            if j < i:
                continue
            rj = slice(int(right["start_index"]), int(right["end_index"]) + 1)
            block = W[li, rj]
            norm = float(np.linalg.norm(block))
            left_diag = W[li, li]
            right_diag = W[rj, rj]
            scale = float(np.sqrt(np.linalg.norm(left_diag) * np.linalg.norm(right_diag)))
            relative = norm / scale if scale > 0 else np.nan
            max_abs = float(np.max(np.abs(block))) if block.size else 0.0
            signed_sum = float(np.sum(block)) if block.size else 0.0
            rows.append({
                "stage_id": stage_id,
                "interface_i": left.get("object_id", left.get("contact_domain_id", i)),
                "interface_j": right.get("object_id", right.get("contact_domain_id", j)),
                "block_shape": str(block.shape),
                "frobenius_norm": norm,
                "relative_coupling_strength": relative,
                "max_absolute_value": max_abs,
                "signed_sum": signed_sum,
                "coupling_sign": "POSITIVE" if signed_sum > 1e-15 else ("NEGATIVE" if signed_sum < -1e-15 else "MIXED_OR_ZERO"),
                "zero_block": bool(max_abs <= 1e-12),
                "cross_interface": bool(i != j),
                "coupled_flag": bool(i == j or norm > 1e-12),
            })
    return pd.DataFrame(rows)


def interface_stage_summary(pkg: SMSPackage, result: dict[str, dict]) -> pd.DataFrame:
    cp = pkg.contact_points.reset_index(drop=True)
    if "interface_id" not in cp.columns:
        return pd.DataFrame()
    rows = []
    for result_key, stage in result.items():
        lam = np.asarray(stage["solution"].lambda_n, dtype=float)
        gap = np.asarray(stage["solution"].gap_g, dtype=float)
        pressure = np.asarray(stage["pressure"], dtype=float)
        for interface_id, indices in cp.groupby("interface_id", sort=False).groups.items():
            idx = np.asarray(list(indices), dtype=int)
            p_values = pressure[idx]
            g_values = gap[idx]
            finite_p = p_values[np.isfinite(p_values)]
            finite_g = g_values[np.isfinite(g_values)]
            rows.append({
                "topology_step_id": stage.get("topology_step_id", result_key),
                "stage_id": stage.get("stage_id", result_key),
                "interface_id": interface_id,
                "contact_point_count": len(idx),
                "active_count": int(np.sum(lam[idx] > 1e-9)),
                "lambda_sum_N": float(lam[idx].sum()),
                "pressure_max_MPa": float(np.max(finite_p)) if finite_p.size else np.nan,
                "gap_min_mm": float(np.min(finite_g)) if finite_g.size else np.nan,
                "gap_mean_mm": float(np.mean(finite_g)) if finite_g.size else np.nan,
                "parent_interface_state_id": _parent_interface_state_id(pkg, str(stage.get("stage_id", result_key)), str(interface_id)),
                "joint_lock_history_id": ";".join(stage.get("stage_state").connection_lock_history_ids) if stage.get("stage_state") is not None else _interface_lock_history_id(pkg, str(stage.get("stage_id", result_key)), str(interface_id)),
                "active_interface": str(interface_id) in set(stage.get("active_interface_ids", [])),
                "state_source": "RUNTIME_GLOBAL_COUPLED_LCP",
            })
    return pd.DataFrame(rows)


def _parent_interface_state_id(pkg: SMSPackage, stage_id: str, interface_id: str) -> str:
    states = pkg.raw_tables.get("I_stage/interface_stage_state.csv", pd.DataFrame())
    if states.empty:
        return ""
    rows = states[
        states.get("stage_id", pd.Series(dtype=str)).astype(str).eq(stage_id)
        & states.get("interface_id", pd.Series(dtype=str)).astype(str).eq(interface_id)
    ]
    return str(rows.iloc[0].get("parent_interface_state_id_optional", "")) if not rows.empty else ""


def _interface_lock_history_id(pkg: SMSPackage, stage_id: str, interface_id: str) -> str:
    states = pkg.raw_tables.get("I_stage/interface_stage_state.csv", pd.DataFrame())
    if not states.empty:
        rows = states[
            states.get("stage_id", pd.Series(dtype=str)).astype(str).eq(stage_id)
            & states.get("interface_id", pd.Series(dtype=str)).astype(str).eq(interface_id)
        ]
        if not rows.empty:
            value = rows.iloc[0].get("joint_lock_history_id_optional", rows.iloc[0].get("joint_lock_history_id", ""))
            if pd.notna(value):
                return str(value)
    return ""


def interface_stage_state_table(pkg: SMSPackage, result: dict[str, dict], stage_id: str) -> pd.DataFrame:
    summary = interface_stage_summary(pkg, result)
    if summary.empty:
        return summary
    selector = "topology_step_id" if "topology_step_id" in summary.columns else "stage_id"
    current = summary[summary[selector].astype(str) == str(stage_id)].copy()
    stage_ids = list(result)
    pos = stage_ids.index(stage_id) if stage_id in stage_ids else 0
    previous = summary[summary[selector].astype(str) == stage_ids[pos - 1]].set_index("interface_id") if pos > 0 else pd.DataFrame()
    stage_type = str(result[stage_id].get("stage_name", stage_id)).upper()
    active_declared = set(result[stage_id].get("active_interface_ids", []))
    joint_definitions = pkg.raw_tables.get("I0/joint_definition.csv", pd.DataFrame())
    joint_interfaces = set(joint_definitions.get("interface_id", pd.Series(dtype=str)).dropna().astype(str))
    statuses = []
    for _, row in current.iterrows():
        previous_active = int(previous.loc[row["interface_id"], "active_count"]) if not previous.empty and row["interface_id"] in previous.index else 0
        if str(row["interface_id"]) not in active_declared:
            status = "NOT_ACTIVATED"
        elif "JOIN" in stage_type and str(row["interface_id"]) in joint_interfaces:
            status = "JOIN_LOCKED"
        elif "RELEASE" in stage_type and row.get("joint_lock_history_id", ""):
            status = "RETAINED_AFTER_RELEASE"
        elif int(row["active_count"]) > 0:
            status = "ACTIVE_CONTACT"
        elif previous_active > 0:
            status = "SEPARATED"
        else:
            status = "ACTIVE_NO_CONTACT"
        statuses.append(status)
    current["edge_state"] = statuses
    return current


def kcp_contribution_path(pkg: SMSPackage, kcp_id: str) -> dict[str, set[str]]:
    parts: set[str] = set()
    interfaces: set[str] = set()
    stages: set[str] = set()
    definitions = pkg.kcp_kcm
    rows = definitions[definitions.get("feature_id", pd.Series(dtype=str)).astype(str).eq(str(kcp_id))]
    if not rows.empty:
        row = rows.iloc[0]
        target = str(row.get("target_part_or_interface", ""))
        if target in set(pkg.parts.get("part_id", pd.Series(dtype=str)).astype(str)):
            parts.add(target)
        if target in set(pkg.interfaces.get("interface_id", pd.Series(dtype=str)).astype(str)):
            interfaces.add(target)
        stage = row.get("stage_id_optional", row.get("stage_id", ""))
        if pd.notna(stage) and str(stage):
            stages.add(str(stage))
    records = pkg.raw_tables.get("prediction/contribution_record.csv", pd.DataFrame())
    for _, row in records.iterrows():
        target_kcp_ids = {
            item.strip() for item in str(row.get("target_kcp_id", "")).split(";") if item.strip()
        }
        if str(kcp_id) not in target_kcp_ids:
            continue
        source = str(row.get("source_id", ""))
        if source in set(pkg.parts.get("part_id", pd.Series(dtype=str)).astype(str)):
            parts.add(source)
        if source in set(pkg.interfaces.get("interface_id", pd.Series(dtype=str)).astype(str)):
            interfaces.add(source)
        stage = row.get("origin_stage_id_optional", "")
        if pd.notna(stage) and str(stage):
            stages.add(str(stage))
    if not interfaces and "J_INTERFACE_ALL" in pkg.matrices:
        layout = vector_layout(pkg)
        definitions_kcp = definitions[definitions.get("feature_role", pd.Series(dtype=str)).astype(str).eq("KCP")]
        order = definitions_kcp.get("feature_id", pd.Series(dtype=str)).astype(str).tolist()
        if kcp_id in order:
            weights = np.asarray(pkg.matrices["J_INTERFACE_ALL"], dtype=float)[order.index(kcp_id)]
            for _, block in layout.iterrows():
                sl = slice(int(block["start_index"]), int(block["end_index"]) + 1)
                if np.any(np.abs(weights[sl]) > 1e-15):
                    interfaces.add(str(block["object_id"]))
    graph = assembly_graph(pkg)
    for interface_id in list(interfaces):
        for a, b, data in graph.edges(data=True):
            if str(data.get("interface_id", "")) == interface_id:
                parts.update([a, b])
    return {"parts": parts, "interfaces": interfaces, "stages": stages}


def coupling_ablation_comparison(
    pkg: SMSPackage,
    result: dict[str, dict],
    stage_id: str,
    threshold: float = 0.05,
) -> dict[str, object]:
    """Zero cross-interface W_struct blocks for a diagnostic-only comparison."""
    from .kcp import extract_kcp
    from .lcp_solver import solve_lcp_active_set

    layout = vector_layout(pkg)
    if layout.empty:
        raise ValueError("耦合对照试算需要显式 VectorLayout。")
    full_stage = result[stage_id]
    W_struct_diag = np.asarray(full_stage["W_struct"], dtype=float).copy()
    for i, left in layout.iterrows():
        li = slice(int(left["start_index"]), int(left["end_index"]) + 1)
        for j, right in layout.iterrows():
            if i == j:
                continue
            rj = slice(int(right["start_index"]), int(right["end_index"]) + 1)
            W_struct_diag[li, rj] = 0.0
    W_total_diag = W_struct_diag + np.asarray(full_stage["Cn"], dtype=float)
    diagnostic_solution = solve_lcp_active_set(np.asarray(full_stage["q"], dtype=float), W_total_diag)
    area = pkg.contact_points["area_weight"].to_numpy(float)
    diagnostic_stage = dict(full_stage)
    diagnostic_stage.update({
        "W_struct": W_struct_diag,
        "W_total": W_total_diag,
        "solution": diagnostic_solution,
        "pressure": diagnostic_solution.lambda_n / area,
        "local_compression": np.asarray(full_stage["Cn"], dtype=float) @ diagnostic_solution.lambda_n,
        "result_role": "DIAGNOSTIC_CROSS_BLOCK_ABLATION_NOT_ENGINEERING_RESULT",
    })
    diagnostic_result = dict(result)
    diagnostic_result[stage_id] = diagnostic_stage
    full_sol = full_stage["solution"]
    point = pkg.contact_points[[c for c in ("candidate_id", "interface_id", "local_index") if c in pkg.contact_points.columns]].copy()
    point["stage_id"] = stage_id
    point["lambda_full_N"] = full_sol.lambda_n
    point["lambda_ablation_N"] = diagnostic_solution.lambda_n
    point["lambda_change_N"] = diagnostic_solution.lambda_n - full_sol.lambda_n
    point["gap_full_mm"] = full_sol.gap_g
    point["gap_ablation_mm"] = diagnostic_solution.gap_g
    point["gap_change_mm"] = diagnostic_solution.gap_g - full_sol.gap_g
    point["active_full"] = np.asarray(full_sol.lambda_n) > 1e-9
    point["active_ablation"] = np.asarray(diagnostic_solution.lambda_n) > 1e-9
    point["active_changed"] = point["active_full"] != point["active_ablation"]

    kcp_full = extract_kcp(pkg, result)[["kcp_id", "predicted_value", "unit"]].rename(columns={"predicted_value": "full_coupling_value"})
    kcp_diag = extract_kcp(pkg, diagnostic_result)[["kcp_id", "predicted_value"]].rename(columns={"predicted_value": "ablation_value"})
    kcp_compare = kcp_full.merge(kcp_diag, on="kcp_id", how="outer")
    kcp_compare["change"] = kcp_compare["ablation_value"] - kcp_compare["full_coupling_value"]
    denom = kcp_compare["full_coupling_value"].abs().clip(lower=1e-12)
    kcp_compare["relative_change"] = kcp_compare["change"].abs() / denom
    kcp_compare["result_role"] = "DIAGNOSTIC_NOT_FORMAL_ENGINEERING_RESULT"

    lambda_scale = max(float(np.max(np.abs(full_sol.lambda_n), initial=0.0)), 1e-12)
    gap_scale = max(float(np.max(np.abs(full_sol.gap_g), initial=0.0)), 1e-12)
    pressure_full = np.asarray(full_stage["pressure"], dtype=float)
    pressure_diag = np.asarray(diagnostic_stage["pressure"], dtype=float)
    metrics = {
        "stage_id": stage_id,
        "lambda_max_absolute_change_N": float(np.max(np.abs(diagnostic_solution.lambda_n - full_sol.lambda_n), initial=0.0)),
        "lambda_relative_change": float(np.max(np.abs(diagnostic_solution.lambda_n - full_sol.lambda_n), initial=0.0) / lambda_scale),
        "gap_max_absolute_change_mm": float(np.max(np.abs(diagnostic_solution.gap_g - full_sol.gap_g), initial=0.0)),
        "gap_relative_change": float(np.max(np.abs(diagnostic_solution.gap_g - full_sol.gap_g), initial=0.0) / gap_scale),
        "active_set_changed_count": int(point["active_changed"].sum()),
        "max_pressure_full_MPa": float(np.max(pressure_full, initial=0.0)),
        "max_pressure_ablation_MPa": float(np.max(pressure_diag, initial=0.0)),
        "max_pressure_change_MPa": float(np.max(pressure_diag, initial=0.0) - np.max(pressure_full, initial=0.0)),
        "max_kcp_relative_change": float(kcp_compare["relative_change"].max()) if not kcp_compare.empty else 0.0,
        "warning_threshold": float(threshold),
        "warning_flag": False,
        "result_role": "DIAGNOSTIC_NOT_FORMAL_ENGINEERING_RESULT",
    }
    metrics["warning_flag"] = bool(max(
        metrics["lambda_relative_change"], metrics["gap_relative_change"], metrics["max_kcp_relative_change"]
    ) > threshold or metrics["active_set_changed_count"] > 0)
    return {
        "summary": pd.DataFrame([metrics]),
        "point_comparison": point,
        "kcp_comparison": kcp_compare,
        "diagnostic_result": diagnostic_result,
    }


def state_lineage_summary(pkg: SMSPackage) -> pd.DataFrame:
    aggregate = pkg.raw_tables.get("I_stage/stage_state_snapshot.csv", pd.DataFrame())
    if aggregate.empty:
        return pd.DataFrame()
    part_states = pkg.raw_tables.get("I_stage/part_stage_state.csv", pd.DataFrame())
    interface_states = pkg.raw_tables.get("I_stage/interface_stage_state.csv", pd.DataFrame())
    rows = []
    for _, row in aggregate.iterrows():
        stage_id = str(row.get("stage_id", ""))
        rows.append({
            "stage_id": stage_id,
            "stage_state_snapshot_id": row.get("stage_state_snapshot_id", ""),
            "parent_stage_state_id": row.get("parent_stage_state_id_optional", ""),
            "part_state_count": int((part_states.get("stage_id", pd.Series(dtype=str)).astype(str) == stage_id).sum()),
            "interface_state_count": int((interface_states.get("stage_id", pd.Series(dtype=str)).astype(str) == stage_id).sum()),
            "quality_flag": row.get("quality_flag", ""),
        })
    return pd.DataFrame(rows)


def contribution_ledger_summary(pkg: SMSPackage) -> dict[str, object]:
    records = pkg.raw_tables.get("prediction/contribution_record.csv", pd.DataFrame())
    prediction = pkg.raw_tables.get("prediction/kcp_prediction_result.csv", pd.DataFrame())
    if records.empty or prediction.empty:
        return {"available": False, "status": "NOT_AVAILABLE"}
    prediction_id_field = "prediction_result_id" if "prediction_result_id" in prediction.columns else "prediction_id"
    record_prediction_field = "consumed_by_prediction_id" if "consumed_by_prediction_id" in records.columns else "prediction_id"
    group_rows: list[dict[str, object]] = []
    total_duplicates = 0
    covered_record_indices: set[object] = set()
    for prediction_index, prediction_row in prediction.iterrows():
        prediction_id = str(prediction_row.get(prediction_id_field, prediction_index))
        prediction_kcp_ids = [
            item.strip() for item in str(
                prediction_row.get("kcp_ids", prediction_row.get("target_kcp_id", ""))
            ).split(";") if item.strip()
        ]
        predicted_values = to_vector(prediction_row.get("predicted_values"), length=None, default=np.nan)
        record_prediction_ids = records.get(record_prediction_field, pd.Series("", index=records.index)).fillna("").astype(str)
        explicit_match = record_prediction_ids.eq(prediction_id)
        blank_prediction = record_prediction_ids.str.strip().eq("")
        relevant = records[explicit_match | blank_prediction].copy()
        if prediction_kcp_ids and "target_kcp_id" in relevant.columns:
            prediction_targets = set(prediction_kcp_ids)
            relevant = relevant[
                relevant["target_kcp_id"].fillna("").astype(str).map(
                    lambda value: bool(prediction_targets.intersection(
                        item.strip() for item in value.split(";") if item.strip()
                    ))
                )
            ]
        prediction_sample = str(prediction_row.get("sample_id", "")).strip()
        if prediction_sample:
            relevant = relevant[relevant.get("sample_id", pd.Series("", index=relevant.index)).astype(str).eq(prediction_sample)]
            sample_ids = [prediction_sample]
        else:
            sample_ids = relevant.get("sample_id", pd.Series("", index=relevant.index)).fillna("").astype(str).unique().tolist()
            sample_ids = sample_ids or [""]
        for sample_id in sample_ids:
            scoped = relevant[
                relevant.get("sample_id", pd.Series("", index=relevant.index)).fillna("").astype(str).eq(sample_id)
            ].copy()
            covered_record_indices.update(scoped.index.tolist())
            keys = []
            for _, row in scoped.iterrows():
                target_scope = ";".join(sorted(
                    item.strip() for item in str(row.get("target_kcp_id", "")).split(";") if item.strip()
                ))
                keys.append((
                    sample_id, prediction_id, target_scope,
                    row.get("source_class", ""), row.get("source_id", ""),
                    row.get("origin_stage_id_optional", ""), row.get("increment_definition_id", ""),
                ))
            duplicate_count = len(keys) - len(set(keys))
            total_duplicates += duplicate_count
            reconstructed = np.zeros(len(prediction_kcp_ids), dtype=float)
            vector_error = False
            for _, row in scoped.iterrows():
                targets = [item.strip() for item in str(row.get("target_kcp_id", "")).split(";") if item.strip()]
                vector = to_vector(row.get("contribution_vector"), length=None, default=0.0)
                for kcp_index, target_kcp_id in enumerate(prediction_kcp_ids):
                    if target_kcp_id not in targets:
                        continue
                    if vector.size == len(prediction_kcp_ids):
                        reconstructed[kcp_index] += vector[kcp_index]
                    elif vector.size == len(targets):
                        reconstructed[kcp_index] += vector[targets.index(target_kcp_id)]
                    elif vector.size == 1:
                        reconstructed[kcp_index] += vector[0]
                    else:
                        vector_error = True
            comparable = predicted_values.size == reconstructed.size and not vector_error
            reconstruction_error = (
                float(np.max(np.abs(reconstructed - predicted_values))) if comparable else np.inf
            )
            status = "PASS" if duplicate_count == 0 and reconstruction_error <= 1e-9 else "FAIL"
            group_rows.append({
                "sample_id": sample_id,
                "prediction_id": prediction_id,
                "target_kcp_ids": ";".join(prediction_kcp_ids),
                "record_count": len(scoped),
                "unique_key_count": len(set(keys)),
                "duplicate_count": duplicate_count,
                "reconstructed_values": reconstructed.tolist(),
                "predicted_values": predicted_values.tolist(),
                "reconstruction_error": reconstruction_error,
                "status": status,
            })
    groups = pd.DataFrame(group_rows)
    uncovered_count = len(records.index.difference(pd.Index(covered_record_indices)))
    max_error = float(groups["reconstruction_error"].max()) if not groups.empty else np.inf
    all_pass = not groups.empty and groups["status"].eq("PASS").all() and uncovered_count == 0
    return {
        "available": True,
        "record_count": len(records),
        "prediction_group_count": len(groups),
        "unique_key_count": len(records) - total_duplicates,
        "duplicate_count": total_duplicates,
        "uncovered_record_count": uncovered_count,
        "reconstruction_error": max_error,
        "group_results": groups,
        "status": "PASS" if all_pass else "FAIL",
    }
