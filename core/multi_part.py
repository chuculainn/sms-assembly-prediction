from __future__ import annotations

from collections import defaultdict, deque

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
    part_ids = pkg.parts.get("part_id", pd.Series(dtype=str)).dropna().astype(str).tolist()
    edges = []
    for _, row in pkg.interfaces.iterrows():
        a, b = str(row.get("part_i", "")), str(row.get("part_j", ""))
        if a and b:
            edges.append((a, b))
    graph: dict[str, set[str]] = defaultdict(set)
    for part_id in part_ids:
        graph[part_id]
    for a, b in edges:
        graph[a].add(b)
        graph[b].add(a)
    seen: set[str] = set()
    components = 0
    for node in graph:
        if node in seen:
            continue
        components += 1
        queue = deque([node])
        seen.add(node)
        while queue:
            current = queue.popleft()
            for nxt in graph[current]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
    cycle_rank = max(0, len(edges) - len(part_ids) + components) if part_ids else 0
    max_degree = max((len(graph[p]) for p in graph), default=0)
    return {
        "part_count": len(part_ids),
        "interface_count": len(edges),
        "connected_components": components,
        "connected": bool(part_ids) and components == 1,
        "cycle_rank": cycle_rank,
        "has_closed_or_parallel_path": cycle_rank > 0,
        "has_serial_path": len(part_ids) >= 3 and max_degree >= 2,
        "max_part_degree": max_degree,
    }


def coupling_block_summary(pkg: SMSPackage, stage_id: str) -> pd.DataFrame:
    layout = vector_layout(pkg)
    W = np.asarray(pkg.matrices.get(f"W_struct__{stage_id}", np.empty((0, 0))), dtype=float)
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
            rows.append({
                "stage_id": stage_id,
                "interface_i": left.get("object_id", left.get("contact_domain_id", i)),
                "interface_j": right.get("object_id", right.get("contact_domain_id", j)),
                "block_shape": str(block.shape),
                "frobenius_norm": norm,
                "cross_interface": bool(i != j),
                "coupled_flag": bool(i == j or norm > 1e-12),
            })
    return pd.DataFrame(rows)


def interface_stage_summary(pkg: SMSPackage, result: dict[str, dict]) -> pd.DataFrame:
    cp = pkg.contact_points.reset_index(drop=True)
    if "interface_id" not in cp.columns:
        return pd.DataFrame()
    rows = []
    for stage_id, stage in result.items():
        lam = np.asarray(stage["solution"].lambda_n, dtype=float)
        gap = np.asarray(stage["solution"].gap_g, dtype=float)
        pressure = np.asarray(stage["pressure"], dtype=float)
        for interface_id, indices in cp.groupby("interface_id", sort=False).groups.items():
            idx = np.asarray(list(indices), dtype=int)
            rows.append({
                "stage_id": stage_id,
                "interface_id": interface_id,
                "contact_point_count": len(idx),
                "active_count": int(np.sum(lam[idx] > 1e-9)),
                "lambda_sum_N": float(lam[idx].sum()),
                "pressure_max_MPa": float(pressure[idx].max(initial=0.0)),
                "gap_min_mm": float(gap[idx].min(initial=np.inf)),
                "gap_mean_mm": float(gap[idx].mean()),
            })
    return pd.DataFrame(rows)


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
    keys = []
    total = None
    for _, row in records.iterrows():
        key = (
            row.get("sample_id", ""), row.get("source_class", ""), row.get("source_id", ""),
            row.get("origin_stage_id_optional", ""), row.get("increment_definition_id", ""),
        )
        keys.append(key)
        vec = to_vector(row.get("contribution_vector"), length=None, default=0.0)
        total = vec.copy() if total is None else total + vec
    predicted = to_vector(prediction.iloc[0].get("predicted_values"), length=None, default=np.nan)
    reconstruction_error = float(np.max(np.abs(total - predicted))) if total is not None and total.size == predicted.size else np.inf
    return {
        "available": True,
        "record_count": len(records),
        "unique_key_count": len(set(keys)),
        "duplicate_count": len(keys) - len(set(keys)),
        "reconstruction_error": reconstruction_error,
        "status": "PASS" if len(keys) == len(set(keys)) and reconstruction_error <= 1e-9 else "FAIL",
    }
