from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_loader import SMSPackage, get_stage_ids
from .multi_part import assembly_graph, coupling_block_summary, topology_summary, vector_layout
from .schema_adapter import parse_literal


COLUMNS = [
    "check_item", "status", "file", "field", "object_id", "detail", "suggestion", "blocking"
]


def _issue(
    rows: list[dict[str, Any]],
    check_item: str,
    status: str,
    file: str,
    field: str = "",
    object_id: str = "",
    detail: str = "",
    suggestion: str = "",
    blocking: bool = False,
) -> None:
    rows.append({
        "check_item": check_item,
        "status": status,
        "file": file,
        "field": field,
        "object_id": object_id,
        "detail": detail,
        "suggestion": suggestion,
        "blocking": bool(blocking and status == "FAIL"),
    })


def _table(pkg: SMSPackage, name: str) -> pd.DataFrame:
    return pkg.raw_tables.get(name, pd.DataFrame())


def _check_pk(
    rows: list[dict[str, Any]], pkg: SMSPackage, file: str, key: str, required: bool = True
) -> None:
    table = _table(pkg, file)
    if table.empty:
        _issue(
            rows, f"主键:{key}", "FAIL" if required else "WARN", file, key,
            detail="表缺失或为空", suggestion=f"提供包含唯一 {key} 的表。", blocking=required,
        )
        return
    if key not in table.columns:
        _issue(rows, f"主键:{key}", "FAIL", file, key, detail="字段缺失", suggestion=f"增加 {key} 字段。", blocking=True)
        return
    values = table[key]
    blank = values.isna() | values.astype(str).str.strip().eq("")
    duplicates = values[~blank].astype(str).duplicated(keep=False)
    if blank.any() or duplicates.any():
        ids = values[blank | duplicates].astype(str).tolist()
        _issue(
            rows, f"主键:{key}", "FAIL", file, key, object_id=";".join(ids[:10]),
            detail=f"空值={int(blank.sum())}, 重复行={int(duplicates.sum())}",
            suggestion="补齐主键并确保全表唯一。", blocking=True,
        )
    else:
        _issue(rows, f"主键:{key}", "PASS", file, key, detail=f"rows={len(table)}")


def _check_fk_values(
    rows: list[dict[str, Any]], source: pd.DataFrame, source_file: str, field: str,
    valid: set[str], target_name: str, *, allow_blank: bool = False, blocking: bool = True,
) -> None:
    if source.empty:
        return
    if field not in source.columns:
        _issue(rows, f"外键:{field}", "FAIL", source_file, field, detail="字段缺失", suggestion=f"增加指向 {target_name} 的字段。", blocking=blocking)
        return
    bad = []
    for _, row in source.iterrows():
        value = row.get(field)
        if pd.isna(value) or not str(value).strip():
            if not allow_blank:
                bad.append((str(row.iloc[0]), "<blank>"))
            continue
        for item in str(value).split(";"):
            if item.strip() and item.strip() not in valid:
                bad.append((str(row.iloc[0]), item.strip()))
    if bad:
        _issue(
            rows, f"外键:{field}", "FAIL", source_file, field,
            object_id=";".join(f"{oid}->{value}" for oid, value in bad[:10]),
            detail=f"无法解析外键数量={len(bad)}，目标={target_name}",
            suggestion=f"修正为已存在的 {target_name} ID；不要静默创建占位对象。", blocking=blocking,
        )
    else:
        _issue(rows, f"外键:{field}", "PASS", source_file, field, detail=f"全部可解析到 {target_name}")


def _metadata_truth(value: object) -> tuple[bool | None, str]:
    parsed = parse_literal(value, {})
    if isinstance(parsed, dict):
        nature = str(parsed.get("data_nature", ""))
        allowed = parsed.get("engineering_claim_allowed")
        return bool(allowed) if isinstance(allowed, bool) else None, nature
    return None, ""


def _lineage_check(
    rows: list[dict[str, Any]], pkg: SMSPackage, file: str, id_field: str, parent_field: str
) -> None:
    table = _table(pkg, file)
    if table.empty:
        return
    if id_field not in table.columns or parent_field not in table.columns:
        _issue(rows, f"父状态链:{id_field}", "FAIL", file, f"{id_field},{parent_field}", detail="状态ID或父ID字段缺失", suggestion="按修订稿补齐父状态链字段。", blocking=True)
        return
    ids = set(table[id_field].dropna().astype(str))
    parent_map = {
        str(row[id_field]): str(row[parent_field])
        for _, row in table.iterrows()
        if pd.notna(row.get(parent_field)) and str(row.get(parent_field)).strip()
    }
    missing = [(child, parent) for child, parent in parent_map.items() if parent not in ids]
    cycle_nodes: set[str] = set()
    for start in ids:
        seen: set[str] = set()
        node = start
        while node in parent_map:
            if node in seen:
                cycle_nodes.update(seen)
                break
            seen.add(node)
            node = parent_map[node]
    if missing or cycle_nodes:
        _issue(
            rows, f"父状态链:{id_field}", "FAIL", file, parent_field,
            object_id=";".join(sorted(cycle_nodes)[:10]),
            detail=f"缺失父状态={missing[:5]}, 环节点={sorted(cycle_nodes)[:10]}",
            suggestion="父状态必须存在且形成从前序阶段到后续阶段的无环链。", blocking=True,
        )
    else:
        _issue(rows, f"父状态链:{id_field}", "PASS", file, parent_field, detail=f"states={len(ids)}, parent_links={len(parent_map)}")


def _raw_npz(pkg: SMSPackage) -> dict[str, np.ndarray]:
    for name in ("multi_part_matrices.npz", "default_matrices.npz", "E1_matrices.npz"):
        path = pkg.root / "matrices" / name
        if path.exists():
            with np.load(path, allow_pickle=False) as archive:
                return {key: archive[key].copy() for key in archive.files}
    return {}


def validate_package_detailed(pkg: SMSPackage, lcp_tolerance: float = 1e-8) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    v25 = str(pkg.package_type).startswith("V25")
    multipart = pkg.package_type == "V25_MULTI_PART"

    if not v25:
        _issue(rows, "旧E1兼容模式", "PASS", "package_manifest.json", detail="使用旧E1适配层；多零件扩展表不作为必需输入。")
        return pd.DataFrame(rows, columns=COLUMNS)

    point_table = _table(pkg, "I_Gamma/contact_point.csv")
    point_pk = "point_id" if "point_id" in point_table.columns else "candidate_id"
    topology_table = _table(pkg, "I0/assembly_topology.csv")
    boundary_table = _table(pkg, "I_stage/boundary_item.csv")
    load_table = _table(pkg, "I_stage/load_item.csv")
    boundary_pk = "boundary_id" if "boundary_id" in boundary_table.columns else "boundary_item_id"
    load_pk = "load_id" if "load_id" in load_table.columns else "load_item_id"
    specs = [
        ("I0/part.csv", "part_id", True),
        ("I0/interface.csv", "interface_id", True),
        ("I_Gamma/contact_domain.csv", "contact_domain_id", True),
        ("I_Gamma/contact_point.csv", point_pk, True),
        ("I_stage/stage_input.csv", "stage_input_id", True),
        ("I_stage/boundary_item.csv", boundary_pk, True),
        ("I_stage/load_item.csv", load_pk, True),
    ]
    if multipart:
        specs += [
            ("I0/assembly_topology.csv", "topology_step_id", True),
            ("I0/joint_definition.csv", "joint_id", True),
            ("I0/subassembly.csv", "subassembly_id", True),
            ("I0/subassembly_membership.csv", "membership_id", True),
            ("I0/stage_definition.csv", "stage_id", True),
            ("I_stage/part_stage_state.csv", "part_stage_state_id", True),
            ("I_stage/interface_stage_state.csv", "interface_stage_state_id", True),
        ]
    for spec in specs:
        _check_pk(rows, pkg, *spec)
    if not multipart:
        topology_fields = [field for field in ("topology_id", "assembly_step") if field in topology_table.columns]
        duplicate_topology = topology_table.duplicated(topology_fields, keep=False) if len(topology_fields) == 2 else pd.Series([True] * len(topology_table))
        _issue(
            rows, "复合主键:topology_id+assembly_step", "PASS" if not topology_table.empty and not duplicate_topology.any() else "FAIL",
            "I0/assembly_topology.csv", "topology_id,assembly_step",
            object_id=";".join(topology_table.loc[duplicate_topology, "topology_id"].astype(str).tolist()[:10]) if "topology_id" in topology_table else "",
            detail=f"rows={len(topology_table)}, duplicate_rows={int(duplicate_topology.sum())}",
            suggestion="基础V2.5包使用 topology_id + assembly_step 复合唯一键；多零件修订包使用 topology_step_id。",
            blocking=True,
        )

    parts = _table(pkg, "I0/part.csv")
    interfaces = _table(pkg, "I0/interface.csv")
    domains = _table(pkg, "I_Gamma/contact_domain.csv")
    points = _table(pkg, "I_Gamma/contact_point.csv")
    part_ids = set(parts.get("part_id", pd.Series(dtype=str)).dropna().astype(str))
    interface_ids = set(interfaces.get("interface_id", pd.Series(dtype=str)).dropna().astype(str))
    domain_ids = set(domains.get("contact_domain_id", pd.Series(dtype=str)).dropna().astype(str))
    stage_ids = set(pkg.stage_plan.get("stage_id", pd.Series(dtype=str)).dropna().astype(str))
    _check_fk_values(rows, interfaces, "I0/interface.csv", "part_i", part_ids, "Part")
    _check_fk_values(rows, interfaces, "I0/interface.csv", "part_j", part_ids, "Part")
    same = interfaces[
        interfaces.get("part_i", pd.Series(dtype=str)).astype(str).eq(
            interfaces.get("part_j", pd.Series(dtype=str)).astype(str)
        )
    ] if not interfaces.empty else pd.DataFrame()
    _issue(
        rows, "接口两端不同零件", "PASS" if same.empty else "FAIL", "I0/interface.csv", "part_i,part_j",
        object_id=";".join(same.get("interface_id", pd.Series(dtype=str)).astype(str).tolist()),
        detail="无自环接口" if same.empty else f"自环接口数量={len(same)}",
        suggestion="将 part_i/part_j 修正为两个不同且已存在的零件。" if not same.empty else "", blocking=True,
    )
    _check_fk_values(rows, domains, "I_Gamma/contact_domain.csv", "interface_id", interface_ids, "Interface")
    _check_fk_values(rows, points, "I_Gamma/contact_point.csv", "contact_domain_id", domain_ids, "ContactDomain")

    if not domains.empty and "contact_domain_id" in domains.columns:
        mapping_count = domains.groupby("contact_domain_id")["interface_id"].nunique() if "interface_id" in domains.columns else pd.Series(dtype=int)
        bad = mapping_count[mapping_count != 1]
        _issue(
            rows, "ContactDomain唯一关联Interface", "PASS" if bad.empty else "FAIL",
            "I_Gamma/contact_domain.csv", "contact_domain_id,interface_id",
            object_id=";".join(bad.index.astype(str).tolist()), detail=f"异常域数量={len(bad)}",
            suggestion="每个 contact_domain_id 只能出现一个 interface_id。", blocking=True,
        )
    if not points.empty and {"contact_domain_id", "local_index"} <= set(points.columns):
        bad_domains = []
        for domain_id, group in points.groupby("contact_domain_id"):
            numeric = pd.to_numeric(group["local_index"], errors="coerce")
            actual = sorted(numeric.dropna().astype(int).tolist())
            expected = list(range(len(group)))
            if numeric.isna().any() or actual != expected:
                bad_domains.append(str(domain_id))
        _issue(
            rows, "ContactPoint.local_index域内连续唯一", "PASS" if not bad_domains else "FAIL",
            "I_Gamma/contact_point.csv", "contact_domain_id,local_index",
            object_id=";".join(bad_domains), detail=f"异常域数量={len(bad_domains)}",
            suggestion="每个接触域内使用从0开始、连续且不重复的 local_index。", blocking=True,
        )

    if multipart:
        joints = _table(pkg, "I0/joint_definition.csv")
        memberships = _table(pkg, "I0/subassembly_membership.csv")
        subassemblies = _table(pkg, "I0/subassembly.csv")
        topology = _table(pkg, "I0/assembly_topology.csv")
        _check_fk_values(rows, joints, "I0/joint_definition.csv", "interface_id", interface_ids, "Interface")
        _check_fk_values(rows, joints, "I0/joint_definition.csv", "part_i", part_ids, "Part")
        _check_fk_values(rows, joints, "I0/joint_definition.csv", "part_j", part_ids, "Part")
        _check_fk_values(rows, joints, "I0/joint_definition.csv", "activation_stage_id", stage_ids, "StageDefinition")
        interface_endpoints = {
            str(row.get("interface_id")): {str(row.get("part_i")), str(row.get("part_j"))}
            for _, row in interfaces.iterrows()
        }
        mismatched_joints = []
        for _, joint in joints.iterrows():
            interface_id = str(joint.get("interface_id", ""))
            if interface_id in interface_endpoints and {
                str(joint.get("part_i", "")), str(joint.get("part_j", ""))
            } != interface_endpoints[interface_id]:
                mismatched_joints.append(str(joint.get("joint_id", "")))
        _issue(
            rows, "JointDefinition端点与Interface一致", "PASS" if not mismatched_joints else "FAIL",
            "I0/joint_definition.csv", "interface_id,part_i,part_j", object_id=";".join(mismatched_joints),
            detail=f"不一致连接数量={len(mismatched_joints)}",
            suggestion="连接两端必须与其引用接口的两端零件集合一致。", blocking=True,
        )
        subassembly_ids = set(subassemblies.get("subassembly_id", pd.Series(dtype=str)).dropna().astype(str))
        joint_ids = set(joints.get("joint_id", pd.Series(dtype=str)).dropna().astype(str))
        _check_fk_values(rows, memberships, "I0/subassembly_membership.csv", "subassembly_id", subassembly_ids, "Subassembly")
        _check_fk_values(rows, memberships, "I0/subassembly_membership.csv", "part_id", part_ids, "Part")
        _check_fk_values(rows, subassemblies, "I0/subassembly.csv", "parent_subassembly_id", subassembly_ids, "Subassembly", allow_blank=True)
        _check_fk_values(rows, subassemblies, "I0/subassembly.csv", "member_part_ids", part_ids, "Part")
        _check_fk_values(rows, subassemblies, "I0/subassembly.csv", "active_interface_ids", interface_ids, "Interface", allow_blank=True)
        _check_fk_values(rows, subassemblies, "I0/subassembly.csv", "active_joint_ids", joint_ids, "JointDefinition", allow_blank=True)
        membership_mismatch = []
        for _, subassembly in subassemblies.iterrows():
            subassembly_id = str(subassembly.get("subassembly_id", ""))
            declared = {item for item in str(subassembly.get("member_part_ids", "")).split(";") if item}
            normalized = set(
                memberships.loc[
                    memberships.get("subassembly_id", pd.Series(dtype=str)).astype(str).eq(subassembly_id), "part_id"
                ].dropna().astype(str)
            ) if not memberships.empty and "part_id" in memberships.columns else set()
            if declared != normalized:
                membership_mismatch.append(subassembly_id)
        _issue(
            rows, "Subassembly成员快照与Membership一致", "PASS" if not membership_mismatch else "FAIL",
            "I0/subassembly.csv + I0/subassembly_membership.csv", "member_part_ids,part_id",
            object_id=";".join(membership_mismatch), detail=f"不一致子装配体数量={len(membership_mismatch)}",
            suggestion="分号成员快照必须与规范化Membership逐行关系完全一致。", blocking=True,
        )
        _check_fk_values(rows, topology, "I0/assembly_topology.csv", "stage_id", stage_ids, "StageDefinition")
        _check_fk_values(rows, topology, "I0/assembly_topology.csv", "activated_interface_ids", interface_ids, "Interface", allow_blank=True)
        _check_fk_values(rows, topology, "I0/assembly_topology.csv", "part_in", part_ids, "Part", allow_blank=True)
        _check_fk_values(rows, topology, "I0/assembly_topology.csv", "existing_subassembly", subassembly_ids, "Subassembly", allow_blank=True)
        _check_fk_values(rows, topology, "I0/assembly_topology.csv", "result_subassembly", subassembly_ids, "Subassembly")
        topology_step_ids = set(topology.get("topology_step_id", pd.Series(dtype=str)).dropna().astype(str))
        _check_fk_values(rows, topology, "I0/assembly_topology.csv", "predecessor_step_id", topology_step_ids, "TopologyStep", allow_blank=True)
        _check_fk_values(rows, topology, "I0/assembly_topology.csv", "successor_step_id", topology_step_ids, "TopologyStep", allow_blank=True)

        stage_definitions = _table(pkg, "I0/stage_definition.csv")
        _check_fk_values(rows, stage_definitions, "I0/stage_definition.csv", "subassembly_id", subassembly_ids, "Subassembly")
        _check_fk_values(rows, stage_definitions, "I0/stage_definition.csv", "active_part_ids", part_ids, "Part")
        _check_fk_values(rows, stage_definitions, "I0/stage_definition.csv", "active_interface_ids", interface_ids, "Interface")

        part_states = _table(pkg, "I_stage/part_stage_state.csv")
        interface_states = _table(pkg, "I_stage/interface_stage_state.csv")
        aggregate_states = _table(pkg, "I_stage/stage_state_snapshot.csv")
        _check_fk_values(rows, part_states, "I_stage/part_stage_state.csv", "stage_id", stage_ids, "StageDefinition")
        _check_fk_values(rows, part_states, "I_stage/part_stage_state.csv", "subassembly_id", subassembly_ids, "Subassembly")
        _check_fk_values(rows, part_states, "I_stage/part_stage_state.csv", "part_id", part_ids, "Part")
        _check_fk_values(rows, interface_states, "I_stage/interface_stage_state.csv", "stage_id", stage_ids, "StageDefinition")
        _check_fk_values(rows, interface_states, "I_stage/interface_stage_state.csv", "subassembly_id", subassembly_ids, "Subassembly")
        _check_fk_values(rows, interface_states, "I_stage/interface_stage_state.csv", "interface_id", interface_ids, "Interface")
        _check_fk_values(rows, interface_states, "I_stage/interface_stage_state.csv", "contact_domain_id", domain_ids, "ContactDomain")
        _check_fk_values(rows, aggregate_states, "I_stage/stage_state_snapshot.csv", "stage_id", stage_ids, "StageDefinition")

        boundaries = _table(pkg, "I_stage/boundary_item.csv")
        loads = _table(pkg, "I_stage/load_item.csv")
        stage_inputs = _table(pkg, "I_stage/stage_input.csv")
        boundary_ids = set(boundaries.get(boundary_pk, pd.Series(dtype=str)).dropna().astype(str))
        load_ids = set(loads.get(load_pk, pd.Series(dtype=str)).dropna().astype(str))
        _check_fk_values(rows, stage_inputs, "I_stage/stage_input.csv", "stage_id", stage_ids, "StageDefinition")
        _check_fk_values(rows, stage_inputs, "I_stage/stage_input.csv", "boundary_item_ids", boundary_ids, "BoundaryItem", allow_blank=True)
        _check_fk_values(rows, stage_inputs, "I_stage/stage_input.csv", "load_item_ids", load_ids, "LoadItem", allow_blank=True)
        _check_fk_values(rows, stage_inputs, "I_stage/stage_input.csv", "active_joint_ids", joint_ids, "JointDefinition", allow_blank=True)

    layout = vector_layout(pkg)
    if multipart:
        layout_file = "matrices/vector_layout.csv"
        if layout.empty:
            _issue(rows, "VectorLayout覆盖", "FAIL", layout_file, detail="布局缺失", suggestion="提供显式公共 VectorLayout。", blocking=True)
        else:
            intervals = [(int(r.start_index), int(r.end_index)) for _, r in layout.iterrows()]
            covered = [idx for start, end in intervals for idx in range(start, end + 1)]
            contiguous = covered == list(range(len(points))) and len(set(covered)) == len(covered)
            _issue(
                rows, "VectorLayout连续无重叠且全覆盖", "PASS" if contiguous else "FAIL", layout_file,
                "start_index,end_index", detail=f"intervals={intervals}, contact_points={len(points)}",
                suggestion="索引必须从0连续覆盖全部接触点，块之间不得遗漏或重叠。", blocking=True,
            )
            bad_blocks = []
            for _, block in layout.iterrows():
                domain = str(block.get("contact_domain_id", ""))
                span = int(block["end_index"]) - int(block["start_index"]) + 1
                count = int((points.get("contact_domain_id", pd.Series(dtype=str)).astype(str) == domain).sum())
                if domain not in domain_ids or span != count:
                    bad_blocks.append(str(block.get("object_id", domain)))
            _issue(
                rows, "VectorLayout对象与接触点对应", "PASS" if not bad_blocks else "FAIL", layout_file,
                "object_id,contact_domain_id", object_id=";".join(bad_blocks), detail=f"异常块数量={len(bad_blocks)}",
                suggestion="每个块必须引用唯一接触域/接口，且块长度等于该域接触点数。", blocking=True,
            )

    topo = topology_summary(pkg)
    _issue(rows, "装配图连通", "PASS" if topo["connected"] else "FAIL", "I0/interface.csv", detail=json.dumps(topo, ensure_ascii=False), suggestion="补齐连接边或拆分为明确的独立装配问题。", blocking=True)
    _issue(rows, "存在串联路径", "PASS" if topo["has_serial_path"] else "WARN", "I0/interface.csv", detail=f"max_degree={topo['max_part_degree']}", suggestion="若研究对象应包含串联传递，请检查接口拓扑。")
    _issue(rows, "存在并联/闭环路径", "PASS" if topo["has_closed_or_parallel_path"] else "WARN", "I0/interface.csv", detail=f"cycle_rank={topo['cycle_rank']}", suggestion="无闭环可以是合法配置；若案例声称并联/闭环，请补齐接口。")
    _issue(rows, "共享柔性零件", "PASS" if topo["shared_flexible_part_count"] else "WARN", "I0/part.csv", "role_in_assembly/flexibility_class", object_id=str(topo["shared_flexible_part_ids"]), detail=f"shared={topo['shared_part_ids']}", suggestion="为共享零件提供明确刚柔属性字段，避免仅由材料名称推断。")

    raw = _raw_npz(pkg)
    manifest = _table(pkg, "matrices/matrix_manifest.csv")
    topology_operator_mode = (
        str(pkg.manifest.get("operator_mode", "")).upper()
        == "PRECOMPUTED_TOPOLOGY_STEP_OPERATOR"
    )
    if manifest.empty:
        _issue(rows, "MatrixManifest与NPZ一致", "FAIL", "matrices/matrix_manifest.csv", detail="矩阵清单缺失", suggestion="提供每个NPZ数组的key、shape和布局引用。", blocking=True)
    else:
        manifest_errors = []
        manifest_keys = manifest.get("npz_key", pd.Series(dtype=str)).astype(str).tolist()
        duplicate_keys = (
            sorted(
                set(
                    manifest.loc[
                        manifest["npz_key"].astype(str).duplicated(keep=False),
                        "npz_key",
                    ].astype(str)
                )
            )
            if "npz_key" in manifest
            else ["<missing npz_key column>"]
        )
        missing_manifest_keys = sorted(set(raw) - set(manifest_keys))
        extra_manifest_keys = sorted(set(manifest_keys) - set(raw))
        for _, item in manifest.iterrows():
            key = str(item.get("npz_key", ""))
            declared = parse_literal(item.get("shape"), [])
            declared_shape = tuple(int(value) for value in declared) if isinstance(declared, (list, tuple)) else ()
            if key not in raw:
                manifest_errors.append(f"{key}:missing")
                continue
            array = raw[key]
            dtype_value = str(item.get("dtype", "")).strip()
            dtype_ok = (
                not dtype_value and not topology_operator_mode
            ) or dtype_value == str(array.dtype)
            row_layout = str(item.get("row_layout_id_optional", "")).strip()
            column_layout = str(item.get("column_layout_id_optional", "")).strip()
            row_layout_ok = (
                not topology_operator_mode
                or array.ndim < 1
                or array.shape[0] != len(points)
                or bool(row_layout)
            )
            column_layout_ok = (
                not topology_operator_mode
                or array.ndim < 2
                or array.shape[1] != len(points)
                or bool(column_layout)
            )
            if (
                tuple(array.shape) != declared_shape
                or not dtype_ok
                or not row_layout_ok
                or not column_layout_ok
            ):
                manifest_errors.append(
                    f"{key}:shape={declared_shape}/{array.shape},"
                    f"dtype={dtype_value or '<blank>'}/{array.dtype},"
                    f"layout={row_layout_ok and column_layout_ok}"
                )
        manifest_errors.extend(f"duplicate:{key}" for key in duplicate_keys)
        manifest_errors.extend(
            f"unregistered_npz:{key}" for key in missing_manifest_keys
        )
        manifest_errors.extend(f"missing_npz:{key}" for key in extra_manifest_keys)
        _issue(
            rows, "MatrixManifest与NPZ一致", "PASS" if not manifest_errors else "FAIL", "matrices/matrix_manifest.csv",
            "npz_key,shape,dtype,row_layout_id_optional,column_layout_id_optional",
            object_id=";".join(manifest_errors[:10]),
            detail=f"manifest_rows={len(manifest)}, npz_keys={len(raw)}, errors={len(manifest_errors)}",
            suggestion="清单必须逐 key 覆盖 NPZ，shape/dtype 与实际数组一致；全局向量轴必须登记 VectorLayout。", blocking=True,
        )
    m = len(points)
    cn = np.asarray(raw.get("CN_ALL", raw.get("CN_DEFAULT", pkg.matrices.get("Cn_local", np.empty((0, 0))))), dtype=float)
    for sid in get_stage_ids(pkg):
        suffixes = [sid.removeprefix("S_"), sid]
        def pick(prefix: str) -> np.ndarray | None:
            for suffix in suffixes:
                key = f"{prefix}_{suffix}"
                if key in raw:
                    return np.asarray(raw[key], dtype=float)
            internal = {
                "W_STRUCT": f"W_struct__{sid}", "W_TOTAL": f"W_total__{sid}", "Q": f"q__{sid}"
            }.get(prefix)
            return np.asarray(pkg.matrices[internal], dtype=float) if internal and internal in pkg.matrices else None

        q, ws, wt, lam, gap, pressure = (
            pick("Q"), pick("W_STRUCT"), pick("W_TOTAL"), pick("LAMBDA"), pick("GAP"), pick("PRESSURE")
        )
        objects = {"q": q, "W_struct": ws, "Cn": cn, "W_total": wt, "lambda": lam, "gap": gap, "pressure": pressure}
        for name, arr in objects.items():
            if arr is None and name in {"lambda", "gap", "pressure"}:
                _issue(rows, f"阶段数组:{name}:{sid}", "WARN", "matrices/*.npz", name, object_id=sid, detail="包内未提供预计算解；运行时可重算。", suggestion="如需基准一致性验证，请提供该数组。")
                continue
            expected = (m, m) if name in {"W_struct", "Cn", "W_total"} else (m,)
            ok = arr is not None and arr.shape == expected and np.isfinite(arr).all()
            _issue(
                rows, f"阶段数组维度与有限性:{name}:{sid}", "PASS" if ok else "FAIL", "matrices/*.npz", name,
                object_id=sid, detail=f"shape={None if arr is None else arr.shape}, expected={expected}",
                suggestion="修正 MatrixManifest/NPZ，使维度与 VectorLayout 全局维数一致且数值有限。", blocking=True,
            )
        if ws is not None and ws.shape == (m, m):
            symmetric = np.allclose(ws, ws.T, atol=1e-10)
            _issue(rows, f"W_struct对称性:{sid}", "PASS" if symmetric else "FAIL", "matrices/*.npz", "W_struct", object_id=sid, detail=f"max_asym={float(np.max(np.abs(ws-ws.T))):.3e}", suggestion="检查结构凝聚和布局行列顺序。", blocking=True)
        if wt is not None and ws is not None and cn.shape == (m, m) and wt.shape == (m, m):
            error = float(np.max(np.abs(wt - (ws + cn))))
            _issue(rows, f"W_total=W_struct+Cn:{sid}", "PASS" if error <= 1e-12 else "FAIL", "matrices/*.npz", "W_total", object_id=sid, detail=f"max_error={error:.3e}", suggestion="正式求解必须按相同 VectorLayout 逐元素相加。", blocking=True)
        if q is not None and wt is not None and lam is not None and gap is not None and all(a.shape == ((m,) if a.ndim == 1 else (m, m)) for a in (q, wt, lam, gap)):
            equilibrium = float(np.max(np.abs(gap - (q + wt @ lam))))
            comp = float(np.max(np.abs(lam * gap)))
            nonnegative = float(min(np.min(lam), np.min(gap))) >= -lcp_tolerance
            ok = equilibrium <= lcp_tolerance and comp <= lcp_tolerance and nonnegative
            _issue(rows, f"LCP非负/平衡/互补:{sid}", "PASS" if ok else "FAIL", "matrices/*.npz", "q,W_total,lambda,gap", object_id=sid, detail=f"equilibrium={equilibrium:.3e}, complementarity={comp:.3e}, min={min(np.min(lam), np.min(gap)):.3e}", suggestion="用完整全局 W_total 统一重求，并核对符号约定。", blocking=True)
        if multipart:
            blocks = coupling_block_summary(pkg, sid)
            cross = blocks[blocks.get("cross_interface", pd.Series(dtype=bool))] if not blocks.empty else pd.DataFrame()
            for _, block in cross.iterrows():
                status = "WARN" if bool(block["zero_block"]) else "PASS"
                _issue(rows, f"交叉柔度块:{sid}", status, "matrices/*.npz", "W_struct", object_id=f"{block['interface_i']}|{block['interface_j']}", detail=f"shape={block['block_shape']}, norm={block['frobenius_norm']:.3e}, relative={block['relative_coupling_strength']:.3e}", suggestion="零块可能合法；若接口共享柔性零件或处于闭环，应核对是否误删交叉块。")

    if multipart:
        _lineage_check(rows, pkg, "I_stage/stage_state_snapshot.csv", "stage_state_snapshot_id", "parent_stage_state_id_optional")
        _lineage_check(rows, pkg, "I_stage/part_stage_state.csv", "part_stage_state_id", "parent_part_state_id_optional")
        _lineage_check(rows, pkg, "I_stage/interface_stage_state.csv", "interface_stage_state_id", "parent_interface_state_id_optional")
        locks = _table(pkg, "I_stage/connection_lock_history.csv")
        releases = _table(pkg, "I_stage/release_history_record.csv")
        lock_ids = set(locks.get("lock_history_id", pd.Series(dtype=str)).dropna().astype(str))
        inherited = set(releases.get("lock_history_id", pd.Series(dtype=str)).dropna().astype(str))
        missing_locks = inherited - lock_ids
        ok = bool(lock_ids) and not releases.empty and not missing_locks
        _issue(rows, "JOIN锁定历史被RELEASE继承", "PASS" if ok else "FAIL", "I_stage/release_history_record.csv", "lock_history_id", object_id=";".join(sorted(missing_locks)), detail=f"join_locks={len(lock_ids)}, release_records={len(releases)}", suggestion="RELEASE记录必须引用JOIN实际生成的ConnectionLockHistory。", blocking=True)

    records = _table(pkg, "prediction/contribution_record.csv")
    if not records.empty:
        # The multipart revision replaces the ledger source key explicitly.
        # Base V2.5 scopes source consumption to one KCP path, so converter
        # records may legitimately repeat a source for different target KCPs.
        key_fields = ["sample_id", "source_class", "source_id", "origin_stage_id_optional", "increment_definition_id"]
        if not multipart:
            key_fields.insert(1, "target_kcp_id")
        missing_fields = [field for field in key_fields if field not in records.columns]
        duplicates = records.duplicated(key_fields, keep=False) if not missing_fields else pd.Series([True] * len(records))
        _issue(rows, "贡献账本唯一键", "PASS" if not missing_fields and not duplicates.any() else "FAIL", "prediction/contribution_record.csv", ",".join(key_fields), object_id=";".join(records.loc[duplicates, "source_id"].astype(str).tolist()[:10]) if "source_id" in records else "", detail=f"missing_fields={missing_fields}, duplicate_rows={int(duplicates.sum())}", suggestion="同一KCP路径（基础V2.5）或多零件修订唯一键内的来源只能登记一次，共享零件和并联路径不得重复累计。", blocking=True)

    nature = str(pkg.manifest.get("data_nature", pkg.manifest.get("row_mode", ""))).upper()
    engineering_allowed = pkg.manifest.get("engineering_claim_allowed")
    synthetic = "SYNTH" in nature or "DEMO" in nature or "PLACEHOLDER" in nature
    if synthetic and engineering_allowed is False:
        truth_status, blocking = "PASS", False
    elif synthetic:
        truth_status, blocking = "FAIL", True
    else:
        truth_status, blocking = ("PASS", False) if engineering_allowed is True else ("WARN", False)
    _issue(
        rows, "数据真实性标记", truth_status, "package_manifest.json", "data_nature,engineering_claim_allowed",
        detail=f"data_nature={nature}, engineering_claim_allowed={engineering_allowed}",
        suggestion="合成/占位数据必须显式设置 engineering_claim_allowed=false，并持续显示真实性声明。",
        blocking=blocking,
    )
    if topology_operator_mode:
        dictionary_path = pkg.root / "field_dictionary.csv"
        object_map_path = pkg.root / "object_file_map.csv"
        dictionary = (
            pd.read_csv(dictionary_path, encoding="utf-8-sig", dtype=str).fillna("")
            if dictionary_path.exists()
            else pd.DataFrame()
        )
        required_dictionary_columns = {
            "file_path", "object_name", "field_name", "data_type", "required",
            "cardinality", "unit", "enum_or_format", "key_semantics",
            "missing_handling", "description", "example_value",
        }
        actual_fields: set[tuple[str, str]] = set()
        for csv_path in pkg.root.rglob("*.csv"):
            relative = csv_path.relative_to(pkg.root).as_posix()
            csv_columns = pd.read_csv(
                csv_path, encoding="utf-8-sig", nrows=0
            ).columns.astype(str)
            actual_fields.update((relative, field) for field in csv_columns)
        dictionary_fields = (
            set(
                zip(
                    dictionary.get("file_path", pd.Series(dtype=str)).astype(str),
                    dictionary.get("field_name", pd.Series(dtype=str)).astype(str),
                )
            )
            if not dictionary.empty
            else set()
        )
        missing_fields = sorted(actual_fields - dictionary_fields)
        dictionary_ok = (
            not dictionary.empty
            and required_dictionary_columns <= set(dictionary.columns)
            and not missing_fields
        )
        _issue(
            rows, "field_dictionary与实际CSV字段一致",
            "PASS" if dictionary_ok else "FAIL", "field_dictionary.csv",
            "file_path,field_name", object_id=";".join(
                f"{path}:{field}" for path, field in missing_fields[:10]
            ),
            detail=f"actual_fields={len(actual_fields)}, dictionary_fields={len(dictionary_fields)}, missing={len(missing_fields)}",
            suggestion="按当前包内实际 CSV schema 重建字段字典，禁止仅保留示例值。",
            blocking=True,
        )

        object_map = (
            pd.read_csv(object_map_path, encoding="utf-8-sig", dtype=str).fillna("")
            if object_map_path.exists()
            else pd.DataFrame()
        )
        required_objects = {
            "AssemblyTopology", "TopologyStepSpec", "TopologyStepResult",
            "ConnectionLockHistory", "ReleaseHistoryRecord",
            "TopologyStepLcpOracle", "TopologyStepOperatorMatrices",
            "TopologyStepExecutionReport", "ValidationResults",
        }
        mapped_objects = set(
            object_map.get("object_name", pd.Series(dtype=str)).astype(str)
        )
        missing_objects = sorted(required_objects - mapped_objects)
        missing_files = []
        for _, item in object_map.iterrows():
            runtime = str(item.get("is_runtime_result", "")).strip().lower() in {
                "1", "true", "yes",
            }
            relative = str(item.get("file_path", "")).strip()
            if not runtime and (not relative or not (pkg.root / relative).exists()):
                missing_files.append(relative or "<blank>")
        object_map_ok = (
            not object_map.empty and not missing_objects and not missing_files
        )
        _issue(
            rows, "object_file_map正式对象可解析",
            "PASS" if object_map_ok else "FAIL", "object_file_map.csv",
            "object_name,file_path,is_runtime_result",
            object_id=";".join((missing_objects + missing_files)[:10]),
            detail=f"mapped={len(mapped_objects)}, missing_objects={len(missing_objects)}, missing_files={len(missing_files)}",
            suggestion="登记正式输入、算子、oracle、执行报告与验证结果；非运行时文件必须实际存在。",
            blocking=True,
        )

        results_path = pkg.root / "validation" / "test_results.json"
        markdown_path = pkg.root / "validation" / "TEST_RESULTS.md"
        run_log_path = pkg.root / "validation" / "run_log.csv"
        quality_path = pkg.root / "validation" / "quality_gate.csv"
        try:
            stored_results = json.loads(results_path.read_text(encoding="utf-8"))
            markdown = markdown_path.read_text(encoding="utf-8")
            run_log = pd.read_csv(
                run_log_path, encoding="utf-8-sig", dtype=str
            ).fillna("")
            quality_gate = pd.read_csv(
                quality_path, encoding="utf-8-sig", dtype=str
            ).fillna("")
            stored_matrix_count = int(stored_results.get("matrix_manifest_count", -1))
            stored_npz_count = int(stored_results.get("npz_key_count", -1))
            attachments_ok = (
                stored_results.get("package_id") == pkg.root.name
                and stored_results.get("status") == "PASS"
                and stored_matrix_count == len(manifest)
                and stored_npz_count == len(raw)
                and f"Checks: {stored_results.get('passed')}/{stored_results.get('total')} PASS" in markdown
                and not run_log.empty
                and str(run_log.iloc[0].get("input_package_id", "")) == pkg.root.name
                and int(run_log.iloc[0].get("matrix_manifest_count", "-1")) == len(manifest)
                and int(run_log.iloc[0].get("npz_key_count", "-1")) == len(raw)
                and not quality_gate.empty
                and str(quality_gate.iloc[0].get("target_object_ids", "")) == pkg.root.name
                and str(quality_gate.iloc[0].get("pass_fail", "")) == "PASS"
            )
            attachment_detail = (
                f"stored={stored_matrix_count}/{stored_npz_count}, "
                f"actual={len(manifest)}/{len(raw)}"
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            attachments_ok = False
            attachment_detail = f"{type(exc).__name__}: {exc}"
        _issue(
            rows, "validation静态附件与当前包一致",
            "PASS" if attachments_ok else "FAIL", "validation/*",
            "package_id,status,matrix_manifest_count,npz_key_count",
            detail=attachment_detail,
            suggestion="用当前生成器重新计算 JSON/Markdown/run_log/quality_gate，不得复制旧包统计。",
            blocking=True,
        )
    base = pd.DataFrame(rows, columns=COLUMNS)
    from .topology_step import validate_topology_steps
    topology = validate_topology_steps(pkg).copy()
    if not topology.empty:
        topology["file"] = "I0/assembly_topology.csv"
        topology["field"] = "topology_step"
        topology["object_id"] = ""
        topology["suggestion"] = "按topology_step实现合同修正路线、父链、外键或算子引用。"
        topology = topology[COLUMNS]
        base = pd.concat([base, topology], ignore_index=True)
    return base


def has_blocking_failures(validation: pd.DataFrame) -> bool:
    if validation.empty:
        return False
    blocking = validation.get(
        "blocking", pd.Series(False, index=validation.index, dtype="boolean")
    ).astype("boolean").fillna(False)
    return bool((validation.get("status", pd.Series(dtype=str)).astype(str).eq("FAIL") & blocking).any())


def data_truthfulness_statement(pkg: SMSPackage) -> str:
    nature = str(pkg.manifest.get("data_nature", pkg.manifest.get("row_mode", "UNSPECIFIED")))
    allowed = pkg.manifest.get("engineering_claim_allowed")
    lines = [
        f"package={pkg.root.name}",
        f"package_type={pkg.package_type}",
        f"data_nature={nature}",
        f"engineering_claim_allowed={allowed}",
    ]
    if "SYNTH" in nature.upper() or "DEMO" in nature.upper() or allowed is False:
        lines.append("仅用于数值一致性与软件联调，不代表真实工程预测结果。")
        lines.append("所有运行时求解、耦合消融和KCP结果均不得用于工程精度声明。")
    else:
        lines.append("工程使用仍需核对CAD/FE、测量、试验、验证角色与适用域证据。")
    lines.append("source_roles=ORIGINAL_INPUT;PACKAGE_PRECOMPUTED;RUNTIME_CALCULATION;DIAGNOSTIC_COMPARISON")
    return "\n".join(lines) + "\n"
