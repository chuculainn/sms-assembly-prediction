from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .data_loader import SMSPackage
from .lcp_solver import LCPSolution, solve_lcp_active_set
from .multi_part import vector_layout
from .schema_adapter import parse_literal
from .stage_state import StageState


PRECOMPUTED_OPERATOR = "PRECOMPUTED_TOPOLOGY_STEP_OPERATOR"
LEGACY_OPERATOR = "LEGACY_STAGE_OPERATOR"
LEGACY_FALLBACK_REASON = "LEGACY_STAGE_COMPATIBILITY"
RETAIN_THROUGH_RELEASE = "RETAIN_THROUGH_RELEASE"
REMOVE_AT_RELEASE = "REMOVE_AT_RELEASE"
SUPPORTED_RETENTION_RULES = {RETAIN_THROUGH_RELEASE, REMOVE_AT_RELEASE}


class TopologyStepValidationError(ValueError):
    """Raised before execution when the topology-step quality gate blocks solving."""


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return not str(value).strip()


def _text(value: object, default: str = "") -> str:
    return default if _is_blank(value) else str(value).strip()


def _ids(value: object) -> list[str]:
    if _is_blank(value):
        return []
    parsed = parse_literal(value, None)
    if isinstance(parsed, (list, tuple, set, np.ndarray)):
        values = parsed
    else:
        values = str(value).replace("|", ";").split(";")
    return list(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))


def _bool(value: object, default: bool = False) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if _is_blank(value):
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "是"}


def _first(row: pd.Series, names: Iterable[str], default: object = "") -> object:
    for name in names:
        if name in row.index and not _is_blank(row.get(name)):
            return row.get(name)
    return default


@dataclass(frozen=True)
class TopologyStepSpec:
    topology_id: str
    topology_step_id: str
    step_order: int
    parent_topology_step_id: str | None
    assembly_cycle_id: str | None
    operation_type: str
    stage_id: str | None
    input_subassembly_id: str | None
    result_subassembly_id: str
    added_part_ids: tuple[str, ...]
    removed_part_ids: tuple[str, ...]
    activated_interface_ids: tuple[str, ...]
    deactivated_interface_ids: tuple[str, ...]
    activated_boundary_ids: tuple[str, ...]
    deactivated_boundary_ids: tuple[str, ...]
    activated_load_ids: tuple[str, ...]
    removed_load_ids: tuple[str, ...]
    activated_joint_ids: tuple[str, ...]
    deactivated_joint_ids: tuple[str, ...]
    operator_set_id: str | None
    solve_required: bool
    reference_state_id: str
    measurement_checkpoint_id: str | None
    notes: str
    adapter_source: str = "ASSEMBLY_TOPOLOGY_TABLE"

    def to_record(self) -> dict[str, Any]:
        record = dict(self.__dict__)
        for field in (
            "added_part_ids", "removed_part_ids", "activated_interface_ids",
            "deactivated_interface_ids", "activated_boundary_ids",
            "deactivated_boundary_ids", "activated_load_ids", "removed_load_ids",
            "activated_joint_ids", "deactivated_joint_ids",
        ):
            record[field] = ";".join(getattr(self, field))
        return record


def _real_topology_table(table: pd.DataFrame) -> bool:
    required = {"topology_step_id", "solve_required", "operator_set_id"}
    return not table.empty and required <= set(table.columns) and bool(
        {"step_order", "assembly_step"} & set(table.columns)
    )


def _spec_from_row(row: pd.Series, position: int) -> TopologyStepSpec:
    step_id = _text(_first(row, ("topology_step_id", "step_id")))
    parent = _text(_first(row, ("parent_topology_step_id", "predecessor_step_id", "predecessor")))
    stage_id = _text(_first(row, ("stage_id",)))
    operation_type = _text(_first(row, ("operation_type", "stage_type"), stage_id or "INIT")).upper()
    order_value = _first(row, ("step_order", "assembly_step", "operation_order"), position + 1)
    try:
        step_order = int(float(order_value))
    except (TypeError, ValueError):
        step_order = position + 1
    return TopologyStepSpec(
        topology_id=_text(_first(row, ("topology_id",)), "TOPOLOGY_DEFAULT"),
        topology_step_id=step_id,
        step_order=step_order,
        parent_topology_step_id=parent or None,
        assembly_cycle_id=_text(_first(row, ("assembly_cycle_id",))) or None,
        operation_type=operation_type,
        stage_id=stage_id or None,
        input_subassembly_id=_text(_first(row, ("input_subassembly_id", "existing_subassembly"))) or None,
        result_subassembly_id=_text(_first(row, ("result_subassembly_id", "result_subassembly")), f"ASM_AFTER_{step_id}"),
        added_part_ids=tuple(_ids(_first(row, ("added_part_ids", "part_in")))),
        removed_part_ids=tuple(_ids(_first(row, ("removed_part_ids", "part_out")))),
        activated_interface_ids=tuple(_ids(_first(row, ("activated_interface_ids", "interface_id")))),
        deactivated_interface_ids=tuple(_ids(_first(row, ("deactivated_interface_ids",)))),
        activated_boundary_ids=tuple(_ids(_first(row, ("activated_boundary_ids",)))),
        deactivated_boundary_ids=tuple(_ids(_first(row, ("deactivated_boundary_ids",)))),
        activated_load_ids=tuple(_ids(_first(row, ("activated_load_ids",)))),
        removed_load_ids=tuple(_ids(_first(row, ("removed_load_ids", "deactivated_load_ids")))),
        activated_joint_ids=tuple(_ids(_first(row, ("activated_joint_ids",)))),
        deactivated_joint_ids=tuple(_ids(_first(row, ("deactivated_joint_ids",)))),
        operator_set_id=_text(_first(row, ("operator_set_id",))) or None,
        solve_required=_bool(_first(row, ("solve_required",)), default=True),
        reference_state_id=_text(_first(row, ("reference_state_id",)), "PARENT_RUNTIME_STATE"),
        measurement_checkpoint_id=_text(_first(row, ("measurement_checkpoint_id",))) or None,
        notes=_text(_first(row, ("notes", "metadata"))),
    )


def _table_row_for_stage(table: pd.DataFrame, stage_id: str) -> pd.Series:
    if table.empty or "stage_id" not in table.columns:
        return pd.Series(dtype=object)
    rows = table[table["stage_id"].astype(str).eq(str(stage_id))]
    return rows.iloc[0] if not rows.empty else pd.Series(dtype=object)


def _legacy_specs(pkg: SMSPackage, topology_id: str | None = None) -> list[TopologyStepSpec]:
    stage_inputs = pkg.raw_tables.get("I_stage/stage_input.csv", pd.DataFrame())
    joints = pkg.raw_tables.get("I0/joint_definition.csv", pd.DataFrame())
    part_ids = tuple(pkg.parts.get("part_id", pd.Series(dtype=str)).dropna().astype(str))
    interface_ids = tuple(pkg.interfaces.get("interface_id", pd.Series(dtype=str)).dropna().astype(str))
    joint_ids = tuple(joints.get("joint_id", pd.Series(dtype=str)).dropna().astype(str))
    previous_boundaries: set[str] = set()
    previous_loads: set[str] = set()
    previous_step: str | None = None
    specs: list[TopologyStepSpec] = []
    legacy_names = {
        "LOCATE": "LEGACY_TS_LOCATE", "CLAMP": "LEGACY_TS_CLAMP",
        "JOIN": "LEGACY_TS_JOIN", "RELEASE": "LEGACY_TS_RELEASE",
    }
    for position, (_, stage) in enumerate(pkg.stage_plan.sort_values("operation_order").iterrows()):
        stage_id = str(stage.get("stage_id"))
        operation = str(stage.get("operation_type", stage_id)).upper()
        step_id = legacy_names.get(operation, f"LEGACY_TS_{position + 1:03d}")
        stage_input = _table_row_for_stage(stage_inputs, stage_id)
        boundaries = set(_ids(stage_input.get("boundary_item_ids")))
        loads = set(_ids(stage_input.get("load_item_ids")))
        is_first = position == 0
        specs.append(TopologyStepSpec(
            topology_id=topology_id or "LEGACY_FOUR_STAGE_TOPOLOGY",
            topology_step_id=step_id,
            step_order=position + 1,
            parent_topology_step_id=previous_step,
            assembly_cycle_id="LEGACY_SINGLE_CYCLE",
            operation_type=operation,
            stage_id=stage_id,
            input_subassembly_id=None if is_first else f"LEGACY_ASM_{position:03d}",
            result_subassembly_id=f"LEGACY_ASM_{position + 1:03d}",
            added_part_ids=part_ids if is_first else (),
            removed_part_ids=(),
            activated_interface_ids=interface_ids if is_first else (),
            deactivated_interface_ids=(),
            activated_boundary_ids=tuple(sorted(boundaries - previous_boundaries)),
            deactivated_boundary_ids=tuple(sorted(previous_boundaries - boundaries)),
            activated_load_ids=tuple(sorted(loads - previous_loads)),
            removed_load_ids=tuple(sorted(previous_loads - loads)),
            activated_joint_ids=joint_ids if operation == "JOIN" else (),
            deactivated_joint_ids=(),
            operator_set_id=stage_id,
            solve_required=True,
            reference_state_id=_text(stage_input.get("reference_state_id"), "PARENT_RUNTIME_STATE"),
            measurement_checkpoint_id=None,
            notes="旧数据包四阶段兼容路线；不代表真实多轮工艺路线。",
            adapter_source="LEGACY_STAGE_ADAPTER",
        ))
        previous_boundaries, previous_loads = boundaries, loads
        previous_step = step_id
    return specs


def load_topology_steps(pkg: SMSPackage, topology_id: str | None = None) -> list[TopologyStepSpec]:
    table = pkg.raw_tables.get("I0/assembly_topology.csv", pd.DataFrame())
    if not _real_topology_table(table):
        return _legacy_specs(pkg, topology_id)
    selected = table.copy()
    if topology_id is not None:
        selected = selected[selected.get("topology_id", pd.Series(dtype=str)).astype(str).eq(str(topology_id))]
    specs = [_spec_from_row(row, position) for position, (_, row) in enumerate(selected.iterrows())]
    return sorted(specs, key=lambda item: (item.step_order, item.topology_step_id))


def uses_precomputed_topology_operators(pkg: SMSPackage) -> bool:
    """Return True only for a real route backed by step-specific operators."""
    specs = load_topology_steps(pkg)
    return bool(specs) and any(
        spec.adapter_source != "LEGACY_STAGE_ADAPTER" for spec in specs
    )


def topology_step_table(pkg: SMSPackage, topology_id: str | None = None) -> pd.DataFrame:
    return pd.DataFrame([spec.to_record() for spec in load_topology_steps(pkg, topology_id)])


def _operator_key(pkg: SMSPackage, prefix: str, operator_set_id: str) -> str | None:
    candidates = (
        f"{prefix}_{operator_set_id}", f"{prefix}__{operator_set_id}",
        f"{prefix}_{operator_set_id.removeprefix('S_')}",
    )
    for key in candidates:
        if key in pkg.matrices:
            return key
    internal_prefix = {"Q": "q", "W_STRUCT": "W_struct", "W_TOTAL": "W_total", "CN": "Cn"}.get(prefix)
    internal = f"{internal_prefix}__{operator_set_id}" if internal_prefix else None
    return internal if internal and internal in pkg.matrices else None


def _gate(rows: list[dict[str, Any]], name: str, ok: bool, detail: str, *, blocking: bool = True) -> None:
    rows.append({
        "check_item": name,
        "status": "PASS" if ok else "FAIL",
        "detail": detail,
        "blocking": bool(blocking and not ok),
    })


def validate_topology_steps(
    pkg: SMSPackage,
    specs: list[TopologyStepSpec] | None = None,
    topology_id: str | None = None,
) -> pd.DataFrame:
    specs = specs if specs is not None else load_topology_steps(pkg, topology_id)
    legacy = bool(specs) and all(s.adapter_source == "LEGACY_STAGE_ADAPTER" for s in specs)
    rows: list[dict[str, Any]] = []
    ids = [spec.topology_step_id for spec in specs]
    orders = [spec.step_order for spec in specs]
    _gate(rows, "topology_step_id唯一", bool(ids) and len(ids) == len(set(ids)) and all(ids), f"steps={len(ids)}")
    _gate(rows, "同一路线step_order不重复", len(orders) == len(set(orders)), f"orders={orders}")
    _gate(rows, "步骤按step_order稳定排序", orders == sorted(orders), f"orders={orders}")
    id_to_pos = {step_id: i for i, step_id in enumerate(ids)}
    missing_parents = [s.topology_step_id for s in specs if s.parent_topology_step_id and s.parent_topology_step_id not in id_to_pos]
    _gate(rows, "parent_topology_step_id可解析", not missing_parents, f"missing={missing_parents}")
    future = [s.topology_step_id for i, s in enumerate(specs) if s.parent_topology_step_id and id_to_pos.get(s.parent_topology_step_id, i) >= i]
    _gate(rows, "不引用未来步骤", not future, f"future_references={future}")
    parent_map = {s.topology_step_id: s.parent_topology_step_id for s in specs if s.parent_topology_step_id}
    cycle_nodes: set[str] = set()
    for start in ids:
        seen: set[str] = set()
        node: str | None = start
        while node in parent_map:
            if node in seen:
                cycle_nodes.update(seen)
                break
            seen.add(node)
            node = parent_map.get(node)
    _gate(rows, "父步骤链无环", not cycle_nodes, f"cycle_nodes={sorted(cycle_nodes)}")
    linear_chain_errors = [
        spec.topology_step_id
        for position, spec in enumerate(specs)
        if (position == 0 and spec.parent_topology_step_id is not None)
        or (position > 0 and spec.parent_topology_step_id != specs[position - 1].topology_step_id)
    ]

    part_ids = set(pkg.parts.get("part_id", pd.Series(dtype=str)).dropna().astype(str))
    interface_rows = pkg.interfaces.set_index("interface_id", drop=False) if "interface_id" in pkg.interfaces.columns else pd.DataFrame()
    interface_ids = set(interface_rows.index.astype(str)) if not interface_rows.empty else set()
    boundaries = pkg.raw_tables.get("I_stage/boundary_item.csv", pd.DataFrame())
    loads = pkg.raw_tables.get("I_stage/load_item.csv", pd.DataFrame())
    joints = pkg.raw_tables.get("I0/joint_definition.csv", pd.DataFrame())
    boundary_field = "boundary_id" if "boundary_id" in boundaries.columns else "boundary_item_id"
    load_field = "load_id" if "load_id" in loads.columns else "load_item_id"
    boundary_ids = set(boundaries.get(boundary_field, pd.Series(dtype=str)).dropna().astype(str))
    load_ids = set(loads.get(load_field, pd.Series(dtype=str)).dropna().astype(str))
    joint_ids = set(joints.get("joint_id", pd.Series(dtype=str)).dropna().astype(str))
    bad_parts: list[str] = []
    bad_interfaces: list[str] = []
    bad_endpoints: list[str] = []
    bad_boundaries: list[str] = []
    bad_loads: list[str] = []
    bad_joints: list[str] = []
    active_parts: set[str] = set()
    active_interfaces: set[str] = set()
    duplicate_activation_ids: list[str] = []
    multi_activation_steps: dict[str, list[str]] = {}
    nonsolve_mechanical_changes: list[str] = []
    for spec in specs:
        bad_parts.extend(pid for pid in (*spec.added_part_ids, *spec.removed_part_ids) if pid not in part_ids)
        active_parts.difference_update(spec.removed_part_ids)
        active_parts.update(spec.added_part_ids)
        bad_interfaces.extend(iid for iid in (*spec.activated_interface_ids, *spec.deactivated_interface_ids) if iid not in interface_ids)
        active_interfaces.difference_update(spec.deactivated_interface_ids)
        active_interfaces.update(spec.activated_interface_ids)
        for interface_id in active_interfaces:
            if interface_id not in interface_ids:
                continue
            row = interface_rows.loc[interface_id]
            endpoints = {str(row.get("part_i", "")), str(row.get("part_j", ""))}
            if not endpoints <= active_parts:
                bad_endpoints.append(f"{spec.topology_step_id}:{interface_id}")
        bad_boundaries.extend(i for i in (*spec.activated_boundary_ids, *spec.deactivated_boundary_ids) if i not in boundary_ids)
        bad_loads.extend(i for i in (*spec.activated_load_ids, *spec.removed_load_ids) if i not in load_ids)
        bad_joints.extend(i for i in (*spec.activated_joint_ids, *spec.deactivated_joint_ids) if i not in joint_ids)
        if len(spec.activated_interface_ids) > 1:
            multi_activation_steps[spec.topology_step_id] = list(spec.activated_interface_ids)
        if len(spec.activated_interface_ids) != len(set(spec.activated_interface_ids)):
            duplicate_activation_ids.append(spec.topology_step_id)
        if (
            spec.parent_topology_step_id
            and not spec.solve_required
            and any((
                spec.added_part_ids, spec.removed_part_ids,
                spec.activated_interface_ids, spec.deactivated_interface_ids,
                spec.activated_boundary_ids, spec.deactivated_boundary_ids,
                spec.activated_load_ids, spec.removed_load_ids,
                spec.activated_joint_ids, spec.deactivated_joint_ids,
            ))
        ):
            nonsolve_mechanical_changes.append(spec.topology_step_id)
    _gate(rows, "added_part_ids零件存在", not bad_parts, f"invalid={sorted(set(bad_parts))}")
    _gate(rows, "activated_interface_ids接口存在", not bad_interfaces, f"invalid={sorted(set(bad_interfaces))}")
    _gate(rows, "活动接口两端属于当前子装配体", not bad_endpoints, f"invalid={bad_endpoints}")
    multi_activation_ok = not duplicate_activation_ids and all(
        all(interface_id in interface_ids for interface_id in activated)
        for activated in multi_activation_steps.values()
    )
    _gate(
        rows,
        "同一步允许多接口同时激活",
        multi_activation_ok,
        f"steps={multi_activation_steps}; duplicate_ids={duplicate_activation_ids}",
    )
    # Historical StageInput tables contain stage-scoped boundary/load labels that
    # predate the canonical definition tables.  The explicit legacy adapter must
    # preserve those already-supported packages; real topology_step routes remain
    # subject to strict foreign-key validation.
    _gate(rows, "BoundaryItem引用可解析", legacy or not bad_boundaries, f"invalid={sorted(set(bad_boundaries))}; legacy_adapter={legacy}")
    _gate(rows, "LoadItem引用可解析", legacy or not bad_loads, f"invalid={sorted(set(bad_loads))}; legacy_adapter={legacy}")
    _gate(rows, "JointDefinition引用可解析", legacy or not bad_joints, f"invalid={sorted(set(bad_joints))}; legacy_adapter={legacy}")
    subassembly_errors: list[str] = []
    spec_by_id = {spec.topology_step_id: spec for spec in specs}
    for spec in specs:
        if not spec.result_subassembly_id:
            subassembly_errors.append(f"{spec.topology_step_id}:blank_result")
        if spec.parent_topology_step_id and spec.input_subassembly_id:
            parent_spec = spec_by_id.get(spec.parent_topology_step_id)
            if parent_spec is not None and spec.input_subassembly_id != parent_spec.result_subassembly_id:
                subassembly_errors.append(
                    f"{spec.topology_step_id}:input={spec.input_subassembly_id}:parent_result={parent_spec.result_subassembly_id}"
                )
    _gate(rows, "result_subassembly_id可构造", not subassembly_errors, f"errors={subassembly_errors}")
    missing_operators = []
    for spec in specs:
        if not spec.solve_required:
            continue
        if not spec.operator_set_id:
            missing_operators.append(f"{spec.topology_step_id}:blank")
            continue
        if spec.adapter_source == "LEGACY_STAGE_ADAPTER":
            ok = f"W_struct__{spec.operator_set_id}" in pkg.matrices and f"q__{spec.operator_set_id}" in pkg.matrices
        else:
            ok = _operator_key(pkg, "W_STRUCT", spec.operator_set_id) is not None and _operator_key(pkg, "Q", spec.operator_set_id) is not None
        if not ok:
            missing_operators.append(f"{spec.topology_step_id}:{spec.operator_set_id}")
    _gate(rows, "solve_required时operator_set_id可解析", not missing_operators, f"missing={missing_operators}")
    _gate(rows, "legacy fallback显式标记", (not legacy) or all(s.operator_set_id for s in specs), f"legacy={legacy}; fallback_reason={LEGACY_FALLBACK_REASON if legacy else ''}")
    layout = vector_layout(pkg)
    m = len(pkg.contact_points)
    covered = [
        index
        for _, block in layout.iterrows()
        for index in range(int(block["start_index"]), int(block["end_index"]) + 1)
    ] if not layout.empty else []
    layout_interface_ids = set(layout.get("object_id", pd.Series(dtype=str)).dropna().astype(str))
    referenced_interfaces = {
        interface_id
        for spec in specs
        for interface_id in (*spec.activated_interface_ids, *spec.deactivated_interface_ids)
    }
    layout_ok = covered == list(range(m)) and referenced_interfaces <= layout_interface_ids
    endpoint_groups: dict[tuple[str, str], list[str]] = {}
    if not pkg.interfaces.empty:
        for _, interface in pkg.interfaces.iterrows():
            endpoints = tuple(sorted((str(interface.get("part_i", "")), str(interface.get("part_j", "")))))
            endpoint_groups.setdefault(endpoints, []).append(str(interface.get("interface_id", "")))
    parallel_groups = {
        "|".join(endpoints): interface_group
        for endpoints, interface_group in endpoint_groups.items()
        if len(interface_group) > 1
    }
    parallel_loss = {
        endpoints: sorted(set(interface_group) - layout_interface_ids)
        for endpoints, interface_group in parallel_groups.items()
        if set(interface_group) - layout_interface_ids
    }
    _gate(
        rows,
        "平行接口不覆盖或丢失",
        not parallel_loss and len(interface_ids) == len(pkg.interfaces),
        f"parallel_groups={parallel_groups}; missing_from_layout={parallel_loss}; "
        f"interface_rows={len(pkg.interfaces)}; unique_interface_ids={len(interface_ids)}",
    )
    manifest = pkg.raw_tables.get("matrices/matrix_manifest.csv", pd.DataFrame())
    manifest_key_field = "npz_key" if "npz_key" in manifest.columns else "matrix_id"
    manifest_by_key = manifest.set_index(manifest_key_field, drop=False) if manifest_key_field in manifest.columns else pd.DataFrame()
    operator_shape_errors: list[str] = []
    for spec in specs:
        if not spec.solve_required or spec.adapter_source == "LEGACY_STAGE_ADAPTER" or not spec.operator_set_id:
            continue
        keys = {
            "Q": _operator_key(pkg, "Q", spec.operator_set_id),
            "W_STRUCT": _operator_key(pkg, "W_STRUCT", spec.operator_set_id),
            "CN": _operator_key(pkg, "CN", spec.operator_set_id),
            "W_TOTAL": _operator_key(pkg, "W_TOTAL", spec.operator_set_id),
        }
        expected = {"Q": (m,), "W_STRUCT": (m, m), "CN": (m, m), "W_TOTAL": (m, m)}
        for role, key in keys.items():
            if key is None:
                if role in {"Q", "W_STRUCT"} or (role == "CN" and keys["W_TOTAL"] is None):
                    operator_shape_errors.append(f"{spec.topology_step_id}:{role}:missing")
                continue
            actual_shape = tuple(np.asarray(pkg.matrices[key]).shape)
            if actual_shape != expected[role]:
                operator_shape_errors.append(f"{spec.topology_step_id}:{role}:{actual_shape}")
            if manifest_by_key.empty or key not in manifest_by_key.index:
                operator_shape_errors.append(f"{spec.topology_step_id}:{role}:manifest_missing")
                continue
            manifest_row = manifest_by_key.loc[key]
            if isinstance(manifest_row, pd.DataFrame):
                manifest_row = manifest_row.iloc[0]
            declared = parse_literal(manifest_row.get("shape"), [])
            try:
                declared_shape = tuple(int(value) for value in declared)
            except (TypeError, ValueError):
                declared_shape = ()
            if declared_shape != actual_shape:
                operator_shape_errors.append(f"{spec.topology_step_id}:{role}:manifest_shape={declared_shape}")
            declared_dtype = _text(manifest_row.get("dtype"))
            actual_dtype = str(np.asarray(pkg.matrices[key]).dtype)
            if declared_dtype and declared_dtype != actual_dtype:
                operator_shape_errors.append(
                    f"{spec.topology_step_id}:{role}:manifest_dtype={declared_dtype}:actual={actual_dtype}"
                )
            row_layout = _text(manifest_row.get("row_layout_id_optional"))
            column_layout = _text(manifest_row.get("column_layout_id_optional"))
            if actual_shape and actual_shape[0] == m and not row_layout:
                operator_shape_errors.append(f"{spec.topology_step_id}:{role}:row_layout_missing")
            if len(actual_shape) == 2 and actual_shape[1] == m and not column_layout:
                operator_shape_errors.append(f"{spec.topology_step_id}:{role}:column_layout_missing")
    _gate(
        rows,
        "MatrixManifest与VectorLayout维度一致",
        layout_ok and not operator_shape_errors,
        f"layout_blocks={len(layout)}, dimension={m}, covered={covered}, operator_errors={operator_shape_errors}",
    )
    cross_block_metrics: list[str] = []
    cross_block_errors: list[str] = []
    active_for_step: set[str] = set()
    require_nonzero_cross = bool(
        pkg.manifest.get("fixture_expectations", {}).get("require_nonzero_cross_blocks", False)
    )
    layout_by_interface = {
        str(block.get("object_id", "")): np.arange(
            int(block["start_index"]), int(block["end_index"]) + 1
        )
        for _, block in layout.iterrows()
    } if not layout.empty else {}
    for spec in specs:
        active_for_step.difference_update(spec.deactivated_interface_ids)
        active_for_step.update(spec.activated_interface_ids)
        if not spec.solve_required or spec.adapter_source == "LEGACY_STAGE_ADAPTER":
            continue
        ws_key = _operator_key(pkg, "W_STRUCT", spec.operator_set_id or "")
        if ws_key is None:
            continue
        W_struct = np.asarray(pkg.matrices[ws_key], dtype=float)
        ordered_interfaces = [
            interface_id for interface_id in layout_by_interface if interface_id in active_for_step
        ]
        for position, first in enumerate(ordered_interfaces):
            for second in ordered_interfaces[position + 1:]:
                block = W_struct[np.ix_(layout_by_interface[first], layout_by_interface[second])]
                norm = float(np.linalg.norm(block))
                cross_block_metrics.append(
                    f"{spec.topology_step_id}:{first}<->{second}:{norm:.6e}"
                )
                if not np.isfinite(norm) or (require_nonzero_cross and norm <= 1e-12):
                    cross_block_errors.append(
                        f"{spec.topology_step_id}:{first}<->{second}:{norm}"
                    )
    _gate(
        rows,
        "活动子矩阵保留跨接口交叉块",
        layout_ok and not cross_block_errors,
        f"require_nonzero={require_nonzero_cross}; norms={cross_block_metrics}; "
        f"errors={cross_block_errors}; extraction=np.ix_",
    )
    retention_bad = []
    if not joints.empty and "retention_rule" in joints.columns:
        for _, joint in joints.iterrows():
            joint_id = _text(joint.get("joint_id"))
            rule = _text(joint.get("retention_rule")).upper()
            if rule not in SUPPORTED_RETENTION_RULES:
                retention_bad.append(f"{joint_id}:{rule or 'BLANK'}")
    _gate(rows, "JOIN和RELEASE retention_rule可解析", not retention_bad, f"invalid={retention_bad}")
    _gate(
        rows,
        "非求解步骤不改变机械活动集合",
        not nonsolve_mechanical_changes,
        f"invalid_steps={nonsolve_mechanical_changes}",
    )
    _gate(
        rows,
        "状态父链定义完整",
        not missing_parents and not future and not cycle_nodes and not linear_chain_errors,
        f"links={len(parent_map)}, linear_chain_errors={linear_chain_errors}",
    )
    nature = str(pkg.manifest.get("data_nature", "")).upper()
    synthetic = "SYNTH" in nature or pkg.manifest.get("engineering_claim_allowed") is False
    truth_ok = (not synthetic) or pkg.manifest.get("engineering_claim_allowed") is False
    _gate(rows, "合成数据engineering_claim_allowed=false", truth_ok, f"data_nature={nature}; engineering_claim_allowed={pkg.manifest.get('engineering_claim_allowed')}")
    return pd.DataFrame(rows)


def _active_indices(pkg: SMSPackage, active_interface_ids: Iterable[str]) -> np.ndarray:
    layout = vector_layout(pkg)
    active = set(str(value) for value in active_interface_ids)
    indices: list[int] = []
    for _, block in layout.iterrows():
        object_id = str(block.get("object_id", ""))
        if object_id in active:
            indices.extend(range(int(block["start_index"]), int(block["end_index"]) + 1))
    return np.asarray(sorted(set(indices)), dtype=int)


def _step_operator(pkg: SMSPackage, spec: TopologyStepSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    if spec.operator_set_id is None:
        raise TopologyStepValidationError(f"{spec.topology_step_id} 缺少 operator_set_id")
    q_key = _operator_key(pkg, "Q", spec.operator_set_id)
    ws_key = _operator_key(pkg, "W_STRUCT", spec.operator_set_id)
    cn_key = _operator_key(pkg, "CN", spec.operator_set_id)
    wt_key = _operator_key(pkg, "W_TOTAL", spec.operator_set_id)
    if q_key is None or ws_key is None:
        raise TopologyStepValidationError(f"{spec.topology_step_id} 无法解析预计算 q/W_struct")
    q = np.asarray(pkg.matrices[q_key], dtype=float).reshape(-1)
    W_struct = np.asarray(pkg.matrices[ws_key], dtype=float)
    if cn_key is not None:
        Cn = np.asarray(pkg.matrices[cn_key], dtype=float)
    elif wt_key is not None:
        Cn = np.asarray(pkg.matrices[wt_key], dtype=float) - W_struct
    else:
        Cn = np.asarray(pkg.matrices["Cn_local"], dtype=float)
        cn_key = "Cn_local"
    W_total = W_struct + Cn
    return q, W_struct, Cn, W_total, {
        "q_key": q_key, "W_struct_key": ws_key, "Cn_key": cn_key or "",
        "W_total_key": wt_key or "RUNTIME_W_STRUCT_PLUS_CN",
    }


def _update_set(parent: Iterable[str], added: Iterable[str], removed: Iterable[str]) -> list[str]:
    values = set(str(item) for item in parent)
    values.difference_update(str(item) for item in removed)
    values.update(str(item) for item in added)
    return sorted(values)


def _interface_states(
    pkg: SMSPackage,
    parent: StageState | None,
    active_interfaces: list[str],
    lam: np.ndarray,
    gap: np.ndarray,
    pressure: np.ndarray,
    compression: np.ndarray,
    state_id: str,
) -> dict[str, dict[str, Any]]:
    layout = vector_layout(pkg)
    active = set(active_interfaces)
    states: dict[str, dict[str, Any]] = {}
    for _, block in layout.iterrows():
        interface_id = str(block.get("object_id", ""))
        idx = np.arange(int(block["start_index"]), int(block["end_index"]) + 1)
        previous = parent.interface_state.get(interface_id, {}) if parent is not None else {}
        states[interface_id] = {
            "interface_state_id": f"{state_id}_IF_{interface_id}",
            "parent_interface_state_id": previous.get("interface_state_id", ""),
            "active_interface": interface_id in active,
            "global_indices": idx.tolist(),
            "gap": gap[idx].tolist(),
            "lambda_n": lam[idx].tolist(),
            "pressure": pressure[idx].tolist(),
            "local_compression": compression[idx].tolist(),
            "contact_mode_global_indices": [int(i) for i in idx if np.isfinite(lam[i]) and lam[i] > 1e-9],
            "data_source": "RUNTIME_TOPOLOGY_STEP_GLOBAL_COUPLED_LCP",
        }
    return states


def _new_part_state(pkg: SMSPackage, part_id: str, state_id: str) -> dict[str, Any]:
    priors = pkg.raw_tables.get("I0/sms_prior.csv", pd.DataFrame())
    rows = priors[
        priors.get("part_id", pd.Series(dtype=str)).astype(str).eq(str(part_id))
    ] if not priors.empty else pd.DataFrame()
    prior = rows.iloc[0] if not rows.empty else pd.Series(dtype=object)
    return {
        "part_state_id": f"{state_id}_PART_{part_id}",
        "parent_part_state_id": "",
        "initialization_source": "ASSEMBLY_BEFORE_SMS",
        "sms_source_part_id": part_id,
        "sms_prior_found": not rows.empty,
        "sms_prior_id": _text(prior.get("sms_prior_id")),
        "sms_prior_type": _text(prior.get("prior_type")),
        "sms_basis_id": _text(prior.get("basis_id")),
        "sms_alpha_mean": parse_literal(prior.get("alpha_mean"), []),
        "sms_alpha_covariance_id": _text(prior.get("alpha_covariance_id")),
        "sms_source_sample_ids": _text(prior.get("source_sample_ids")),
        "initialization_count": 1,
        "data_source": "PACKAGE_ASSEMBLY_BEFORE_SMS",
    }


def _joint_history(
    pkg: SMSPackage,
    spec: TopologyStepSpec,
    sample_id: str,
    state_id: str,
    active_joint_ids: list[str],
    local_compression: np.ndarray,
    active_contact_indices: list[int],
) -> dict[str, Any]:
    joints = pkg.raw_tables.get("I0/joint_definition.csv", pd.DataFrame())
    loads = pkg.raw_tables.get("I_stage/load_item.csv", pd.DataFrame())
    joint_rows = joints[joints.get("joint_id", pd.Series(dtype=str)).astype(str).isin(spec.activated_joint_ids)] if not joints.empty else pd.DataFrame()
    preload_ids = [
        load_id for load_id in spec.activated_load_ids
        if not loads.empty and not loads[
            loads.get("load_id", pd.Series(dtype=str)).astype(str).eq(load_id)
            & loads.get("load_type", pd.Series(dtype=str)).astype(str).str.contains("PRELOAD", case=False, na=False)
        ].empty
    ]
    return {
        "lock_history_id": f"LOCK_{sample_id}_{spec.topology_step_id}",
        "sample_id": sample_id,
        "topology_step_id": spec.topology_step_id,
        "join_stage_id": spec.stage_id or "",
        "joint_ids": ";".join(spec.activated_joint_ids),
        "active_joint_ids_after_step": ";".join(active_joint_ids),
        "locked_reference": json.dumps(local_compression.tolist()),
        "locked_reference_source": state_id,
        "preload_actual_source_ids": ";".join(preload_ids),
        "joint_stiffness_ids": ";".join(joint_rows.get("stiffness_matrix_id", pd.Series(dtype=str)).dropna().astype(str)),
        "locked_contact_mode": ";".join(str(i) for i in active_contact_indices),
        "quality_flag": "PASS",
    }


def _release_history(
    pkg: SMSPackage,
    spec: TopologyStepSpec,
    sample_id: str,
    lock_ids: list[str],
    retained_joint_ids: list[str],
    removed_joint_ids: list[str],
) -> dict[str, Any]:
    return {
        "release_history_id": f"RELEASE_{sample_id}_{spec.topology_step_id}",
        "sample_id": sample_id,
        "topology_step_id": spec.topology_step_id,
        "release_stage_id": spec.stage_id or "",
        "lock_history_ids": ";".join(lock_ids),
        "removed_boundary_ids": ";".join(spec.deactivated_boundary_ids),
        "removed_load_ids": ";".join(spec.removed_load_ids),
        "retained_joint_ids": ";".join(retained_joint_ids),
        "removed_joint_ids": ";".join(removed_joint_ids),
        "active_joint_ids_after_step": ";".join(retained_joint_ids),
        "retention_rule": "JOINT_DEFINITION_RETENTION_RULE",
        "quality_flag": "PASS",
    }


def _apply_release_retention(
    pkg: SMSPackage,
    spec: TopologyStepSpec,
    active_joint_ids: list[str],
) -> tuple[list[str], list[str]]:
    """Apply explicit deactivation first, then each remaining joint's release rule."""
    joints = pkg.raw_tables.get("I0/joint_definition.csv", pd.DataFrame())
    retained: list[str] = []
    removed = list(spec.deactivated_joint_ids)
    for joint_id in active_joint_ids:
        row = joints[
            joints.get("joint_id", pd.Series(dtype=str)).astype(str).eq(joint_id)
        ] if not joints.empty else pd.DataFrame()
        if row.empty:
            raise TopologyStepValidationError(f"{joint_id} 缺少 JointDefinition")
        rule = _text(row.iloc[0].get("retention_rule")).upper()
        if rule == RETAIN_THROUGH_RELEASE:
            retained.append(joint_id)
        elif rule == REMOVE_AT_RELEASE:
            removed.append(joint_id)
        else:
            raise TopologyStepValidationError(
                f"{joint_id} 未知 retention_rule={rule or 'BLANK'}"
            )
    return sorted(set(retained)), sorted(set(removed))


def _execute_step(
    pkg: SMSPackage,
    spec: TopologyStepSpec,
    sample_id: str,
    parent: StageState | None,
    *,
    eps: float,
    legacy_options: dict[str, Any],
) -> dict[str, Any]:
    parent_parts = parent.active_part_ids if parent is not None else []
    parent_interfaces = parent.active_interface_ids if parent is not None else []
    parent_boundaries = parent.active_boundary_ids if parent is not None else []
    parent_loads = parent.active_load_ids if parent is not None else []
    parent_joints = parent.active_joint_ids if parent is not None else []
    active_parts = _update_set(parent_parts, spec.added_part_ids, spec.removed_part_ids)
    active_interfaces = _update_set(parent_interfaces, spec.activated_interface_ids, spec.deactivated_interface_ids)
    active_boundaries = _update_set(parent_boundaries, spec.activated_boundary_ids, spec.deactivated_boundary_ids)
    active_loads = _update_set(parent_loads, spec.activated_load_ids, spec.removed_load_ids)
    active_joints = _update_set(parent_joints, spec.activated_joint_ids, spec.deactivated_joint_ids)
    if not spec.solve_required and parent is not None:
        # A non-solving event may add trace metadata, but cannot silently replace
        # the parent's mechanical topology or response.
        active_parts = list(parent.active_part_ids)
        active_interfaces = list(parent.active_interface_ids)
        active_boundaries = list(parent.active_boundary_ids)
        active_loads = list(parent.active_load_ids)
        active_joints = list(parent.active_joint_ids)
    active_indices = _active_indices(pkg, active_interfaces)
    m = len(pkg.contact_points)
    active_mask = np.zeros(m, dtype=bool)
    active_mask[active_indices] = True
    operator_source = LEGACY_OPERATOR if spec.adapter_source == "LEGACY_STAGE_ADAPTER" else PRECOMPUTED_OPERATOR
    fallback_flag = operator_source == LEGACY_OPERATOR
    matrix_keys: dict[str, str] = {}
    lcp_call_count = 0
    legacy_stage_result: dict[str, Any] | None = None
    if spec.solve_required:
        if fallback_flag:
            from .stage_solver import run_stage
            stage = run_stage(pkg, str(spec.stage_id), eps=eps, **legacy_options)
            legacy_stage_result = stage
            q_full = np.asarray(stage["q"], dtype=float)
            W_struct_full = np.asarray(stage["W_struct"], dtype=float)
            Cn_full = np.asarray(stage["Cn"], dtype=float)
            W_total_full = np.asarray(stage["W_total"], dtype=float)
            active_solution = stage["solution"]
            # Legacy routes activate the full historic interface set, so this is
            # already the one global solve used by the previous implementation.
            active_indices = np.arange(m, dtype=int)
            active_mask[:] = True
            lambda_active = np.asarray(active_solution.lambda_n, dtype=float)
            gap_active = np.asarray(active_solution.gap_g, dtype=float)
            lcp_call_count = 1
            matrix_keys = {
                "q_key": f"q__{spec.stage_id}", "W_struct_key": f"W_struct__{spec.stage_id}",
                "Cn_key": "Cn_local/runtime_substitution", "W_total_key": "RUNTIME_W_STRUCT_PLUS_CN",
            }
        else:
            q_full, W_struct_full, Cn_full, W_total_full, matrix_keys = _step_operator(pkg, spec)
            q_active = q_full[active_indices]
            W_active = W_total_full[np.ix_(active_indices, active_indices)]
            active_solution = solve_lcp_active_set(q_active, W_active, eps=eps)
            lambda_active = np.asarray(active_solution.lambda_n, dtype=float)
            gap_active = np.asarray(active_solution.gap_g, dtype=float)
            lcp_call_count = 1
        lambda_full = np.zeros(m, dtype=float)
        gap_full = np.full(m, np.nan, dtype=float)
        lambda_full[active_indices] = lambda_active
        gap_full[active_indices] = gap_active
        pressure = np.full(m, np.nan, dtype=float)
        area = pkg.contact_points["area_weight"].to_numpy(dtype=float)
        pressure[active_indices] = lambda_active / area[active_indices]
        local_compression = np.full(m, np.nan, dtype=float)
        local_compression[active_indices] = Cn_full[np.ix_(active_indices, active_indices)] @ lambda_active
        global_active_contacts = [int(active_indices[i]) for i in active_solution.active_indices]
        full_solution = LCPSolution(
            lambda_n=lambda_full,
            gap_g=gap_full,
            active_indices=global_active_contacts,
            inactive_indices=[i for i in range(m) if i not in global_active_contacts],
            residuals=dict(active_solution.residuals),
            iteration_count=active_solution.iteration_count,
            convergence_status=active_solution.convergence_status,
            active_set_trace=list(active_solution.active_set_trace),
        )
        solve_status = "CONVERGED" if full_solution.convergence_status == "CONVERGED" else full_solution.convergence_status
    elif parent is None:
        q_full = np.full(m, np.nan, dtype=float)
        W_struct_full = np.zeros((m, m), dtype=float)
        Cn_full = np.zeros((m, m), dtype=float)
        W_total_full = np.zeros((m, m), dtype=float)
        lambda_active = np.array([], dtype=float)
        gap_active = np.array([], dtype=float)
        lambda_full = np.zeros(m, dtype=float)
        gap_full = np.full(m, np.nan, dtype=float)
        pressure = np.full(m, np.nan, dtype=float)
        local_compression = np.full(m, np.nan, dtype=float)
        full_solution = LCPSolution(
            lambda_n=lambda_full, gap_g=gap_full, active_indices=[], inactive_indices=list(range(m)),
            residuals={"equilibrium_residual": 0.0, "gap_violation": 0.0, "force_violation": 0.0, "complementarity_residual": 0.0},
            iteration_count=0, convergence_status="NOT_REQUIRED", active_set_trace=[],
        )
        solve_status = "NOT_REQUIRED"
        mechanical_state_action = "INITIALIZE_EMPTY"
        not_required_reason = "NO_PARENT_MECHANICAL_STATE"
    else:
        q_full = np.full(m, np.nan, dtype=float)
        W_struct_full = np.zeros((m, m), dtype=float)
        Cn_full = np.zeros((m, m), dtype=float)
        W_total_full = np.zeros((m, m), dtype=float)
        lambda_full = np.asarray(parent.lambda_n, dtype=float).copy()
        gap_full = np.asarray(parent.gap, dtype=float).copy()
        pressure = np.asarray(parent.pressure, dtype=float).copy()
        local_compression = np.asarray(parent.local_compression, dtype=float).copy()
        active_mask = np.asarray(parent.active_index_mask, dtype=bool).copy()
        active_indices = np.flatnonzero(active_mask)
        lambda_active = lambda_full[active_indices].copy()
        gap_active = gap_full[active_indices].copy()
        full_solution = LCPSolution(
            lambda_n=lambda_full,
            gap_g=gap_full,
            active_indices=list(parent.active_set),
            inactive_indices=[i for i in range(m) if i not in set(parent.active_set)],
            residuals=dict(parent.physical_residuals),
            iteration_count=0,
            convergence_status="NOT_REQUIRED",
            active_set_trace=[],
        )
        solve_status = "NOT_REQUIRED"
        mechanical_state_action = "INHERIT_PARENT_UNCHANGED"
        not_required_reason = "NON_SOLVING_EVENT_WITH_PARENT"

    if spec.solve_required:
        mechanical_state_action = "SOLVE_GLOBAL_LCP"
        not_required_reason = ""

    state_id = f"STATE_{sample_id}_{spec.topology_step_id}"
    part_state = deepcopy(parent.part_state) if parent is not None else {}
    for part_id in spec.removed_part_ids:
        part_state.pop(part_id, None)
    for part_id in spec.added_part_ids:
        if part_id not in part_state:
            part_state[part_id] = _new_part_state(pkg, part_id, state_id)
    for part_id, value in part_state.items():
        if part_id not in spec.added_part_ids:
            previous_id = str(value.get("part_state_id", ""))
            value["parent_part_state_id"] = previous_id
            value["part_state_id"] = f"{state_id}_PART_{part_id}"
            value["data_source"] = "INHERITED_PARENT_ASSEMBLY_STATE"

    lock_ids = list(parent.connection_lock_history_ids) if parent is not None else []
    release_ids = list(parent.release_history_ids) if parent is not None else []
    new_locks: list[dict[str, Any]] = []
    new_releases: list[dict[str, Any]] = []
    release_removed_joint_ids: list[str] = []
    if spec.operation_type == "JOIN" and spec.activated_joint_ids:
        lock = _joint_history(pkg, spec, sample_id, state_id, active_joints, np.nan_to_num(local_compression), full_solution.active_indices)
        new_locks.append(lock)
        lock_ids.append(lock["lock_history_id"])
    if spec.operation_type == "RELEASE":
        active_joints, release_removed_joint_ids = _apply_release_retention(
            pkg, spec, active_joints
        )
        release = _release_history(
            pkg, spec, sample_id, lock_ids, active_joints,
            release_removed_joint_ids,
        )
        new_releases.append(release)
        release_ids.append(release["release_history_id"])

    parent_response = parent.contact_structural_response if parent is not None else np.zeros(m)
    response = W_struct_full @ lambda_full if spec.solve_required else parent_response.copy()
    gap_increment = (
        np.zeros(m, dtype=float)
        if not spec.solve_required and parent is not None
        else (
            np.nan_to_num(gap_full) - np.nan_to_num(parent.gap)
            if parent is not None else np.nan_to_num(gap_full)
        )
    )
    joint_lock_state = deepcopy(parent.joint_lock_state) if parent is not None else {}
    if new_locks:
        joint_lock_state = dict(new_locks[-1])
    elif joint_lock_state:
        joint_lock_state["inherited_from_stage_state_id"] = parent.stage_state_id if parent is not None else ""
    if new_releases:
        joint_lock_state["active_joint_ids_after_release"] = list(active_joints)
        joint_lock_state["removed_joint_ids_at_release"] = list(release_removed_joint_ids)
    layout = vector_layout(pkg)
    layout_id = _text(layout.iloc[0].get("vector_layout_id"), "LEGACY_IMPLICIT_LAYOUT") if not layout.empty else "LEGACY_IMPLICIT_LAYOUT"
    state = StageState(
        stage_state_id=state_id,
        stage_id=str(spec.stage_id or spec.operation_type),
        stage_type=spec.operation_type,
        parent_stage_id=parent.stage_id if parent is not None else None,
        parent_stage_state_id=parent.stage_state_id if parent is not None else None,
        part_state=part_state,
        interface_state=_interface_states(pkg, parent, active_interfaces, lambda_full, gap_full, pressure, local_compression, state_id),
        contact_structural_response=response,
        contact_structural_response_increment=response - parent_response,
        gap=gap_full,
        gap_increment=gap_increment,
        lambda_n=lambda_full,
        pressure=pressure,
        local_compression=local_compression,
        active_set=list(full_solution.active_indices),
        boundary_state={"activated": list(spec.activated_boundary_ids), "deactivated": list(spec.deactivated_boundary_ids), "active": active_boundaries},
        load_state={"activated": list(spec.activated_load_ids), "deactivated": list(spec.removed_load_ids), "removed": list(spec.removed_load_ids), "active": active_loads},
        joint_lock_state=joint_lock_state,
        reference_state=spec.reference_state_id,
        vector_layout_id=layout_id,
        data_source=operator_source,
        fallback_flag=fallback_flag,
        input_sources=[matrix_keys.get("q_key", ""), matrix_keys.get("W_struct_key", ""), matrix_keys.get("Cn_key", "")],
        physical_residuals={key: float(value) for key, value in full_solution.residuals.items()},
        notes=[
            "预计算步骤算子用于运行时确定性统一LCP；未在线重建全阶FE算子。",
            "已有子装配体继承父状态；仅本步新加入零件读取装配前SMS。",
        ],
        sample_id=sample_id,
        topology_id=spec.topology_id,
        topology_step_id=spec.topology_step_id,
        parent_topology_step_id=spec.parent_topology_step_id,
        assembly_cycle_id=spec.assembly_cycle_id or "",
        operation_type=spec.operation_type,
        current_subassembly_id=spec.result_subassembly_id,
        active_part_ids=active_parts,
        active_interface_ids=active_interfaces,
        active_joint_ids=active_joints,
        active_boundary_ids=active_boundaries,
        active_load_ids=active_loads,
        operator_set_id=spec.operator_set_id or "",
        operator_source=operator_source,
        solve_status=solve_status,
        fallback_reason=LEGACY_FALLBACK_REASON if fallback_flag else "",
        active_index_mask=active_mask.tolist(),
        lambda_active=lambda_active,
        gap_active=gap_active,
        connection_lock_history_ids=lock_ids,
        release_history_ids=release_ids,
        mechanical_state_action=mechanical_state_action,
        not_required_reason=not_required_reason,
        quality_flag="PASS" if solve_status in {"CONVERGED", "NOT_REQUIRED"} else "FAIL",
    )
    trace = {
        "trace_id": f"TRACE_{sample_id}_{spec.topology_step_id}",
        "sample_id": sample_id,
        "topology_id": spec.topology_id,
        "topology_step_id": spec.topology_step_id,
        "parent_state_id": state.parent_stage_state_id or "",
        "operator_set_id": spec.operator_set_id or "",
        "operator_source": operator_source,
        "q_key": matrix_keys.get("q_key", ""),
        "W_struct_key": matrix_keys.get("W_struct_key", ""),
        "Cn_key": matrix_keys.get("Cn_key", ""),
        "W_total_source": "W_struct + Cn",
        "active_interface_ids": active_interfaces,
        "active_indices": active_indices.tolist(),
        "lcp_call_count": lcp_call_count,
        "mechanical_state_action": mechanical_state_action,
        "not_required_reason": not_required_reason,
        "parameter_effective": bool(fallback_flag),
        "parameter_mode": (
            "LEGACY_RUNTIME_RECONSTRUCTION"
            if fallback_flag else "PRECOMPUTED_TOPOLOGY_STEP_OPERATOR_DISABLED"
        ),
        "lambda_active": lambda_active.tolist(),
        "gap_active": gap_active.tolist(),
        "pressure_active": pressure[active_indices].tolist() if active_indices.size else [],
        "local_compression_active": local_compression[active_indices].tolist() if active_indices.size else [],
        "complementarity_residual": full_solution.residuals.get("complementarity_residual", 0.0),
        "quality_flag": state.quality_flag,
    }
    return {
        "stage_id": state.stage_id,
        "stage_name": spec.operation_type,
        "topology_step_id": spec.topology_step_id,
        "topology_step_spec": spec,
        "q": q_full,
        "q_active": q_full[active_indices] if spec.solve_required else np.array([], dtype=float),
        "W_total": W_total_full,
        "W_active": W_total_full[np.ix_(active_indices, active_indices)] if spec.solve_required else np.empty((0, 0)),
        "W_struct": W_struct_full,
        "Cn": Cn_full,
        "solution": full_solution,
        "lambda_active": lambda_active,
        "gap_active": gap_active,
        "lambda_full": lambda_full,
        "gap_full": gap_full,
        "active_index_mask": active_mask,
        "active_interface_ids": active_interfaces,
        "inactive_interface_ids": sorted(set(pkg.interfaces.get("interface_id", pd.Series(dtype=str)).astype(str)) - set(active_interfaces)),
        "pressure": pressure,
        "local_compression": local_compression,
        "stage_state": state,
        "contact_trace": trace,
        "connection_lock_history": new_locks,
        "release_history": new_releases,
        "operator_source": operator_source,
        "operator_set_id": spec.operator_set_id or "",
        "solve_status": solve_status,
        "fallback_flag": fallback_flag,
        "fallback_reason": LEGACY_FALLBACK_REASON if fallback_flag else "",
        "lcp_call_count": lcp_call_count,
        "runtime_scales": {
            "sms_scale": float(legacy_options.get("sms_scale", 1.0)) if fallback_flag else 1.0,
            "closure_scale": float(legacy_options.get("closure_scale", 1.0)) if fallback_flag else 1.0,
            "cn_scale": float(legacy_options.get("cn_scale", 1.0)) if fallback_flag else 1.0,
        },
        "runtime_parameter_effective": bool(fallback_flag),
        "runtime_parameter_mode": (
            "LEGACY_RUNTIME_RECONSTRUCTION"
            if fallback_flag else "PRECOMPUTED_TOPOLOGY_STEP_OPERATOR_DISABLED"
        ),
        "substitution_components": legacy_stage_result.get("substitution_components", pd.DataFrame()) if legacy_stage_result is not None else pd.DataFrame(),
        "sms_rebuild": legacy_stage_result.get("sms_rebuild") if legacy_stage_result is not None else None,
        "extended_lcp": legacy_stage_result.get("extended_lcp") if legacy_stage_result is not None else None,
        "force_nonuniqueness": legacy_stage_result.get("force_nonuniqueness") if legacy_stage_result is not None else None,
        "tangential_ncp": legacy_stage_result.get("tangential_ncp", pd.DataFrame()) if legacy_stage_result is not None else pd.DataFrame(),
    }


def run_topology_steps(
    pkg: SMSPackage,
    topology_id: str | None = None,
    sample_id: str = "SAMPLE_001",
    *,
    sms_scale: float = 1.0,
    closure_scale: float = 1.0,
    cn_scale: float = 1.0,
    eps: float = 1e-9,
    substitution_settings: Any = None,
    sms_mapping_settings: Any = None,
    overconstraint_settings: Any = None,
    tangential_settings: Any = None,
) -> dict[str, dict[str, Any]]:
    specs = load_topology_steps(pkg, topology_id)
    validation = validate_topology_steps(pkg, specs, topology_id)
    blocking = validation[validation["status"].eq("FAIL") & validation["blocking"].astype(bool)]
    if not blocking.empty:
        raise TopologyStepValidationError(blocking[["check_item", "detail"]].to_string(index=False))
    options = {
        "sms_scale": sms_scale,
        "closure_scale": closure_scale,
        "cn_scale": cn_scale,
        "substitution_settings": substitution_settings,
        "sms_mapping_settings": sms_mapping_settings,
        "overconstraint_settings": overconstraint_settings,
        "tangential_settings": tangential_settings,
    }
    result: dict[str, dict[str, Any]] = {}
    states: dict[str, StageState] = {}
    for spec in specs:
        parent = states.get(spec.parent_topology_step_id) if spec.parent_topology_step_id else None
        step_result = _execute_step(pkg, spec, sample_id, parent, eps=eps, legacy_options=options)
        result[spec.topology_step_id] = step_result
        states[spec.topology_step_id] = step_result["stage_state"]
    return result


def topology_step_execution_table(result: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for key, step in result.items():
        state: StageState = step["stage_state"]
        rows.append({
            "sample_id": state.sample_id, "topology_id": state.topology_id,
            "topology_step_id": state.topology_step_id or key,
            "step_order": step["topology_step_spec"].step_order,
            "parent_topology_step_id": state.parent_topology_step_id or "",
            "parent_state_id": state.parent_stage_state_id or "",
            "assembly_cycle_id": state.assembly_cycle_id,
            "operation_type": state.operation_type, "stage_id": state.stage_id,
            "input_subassembly_id": step["topology_step_spec"].input_subassembly_id or "",
            "result_subassembly_id": state.current_subassembly_id,
            "added_part_ids": ";".join(step["topology_step_spec"].added_part_ids),
            "active_part_ids": ";".join(state.active_part_ids),
            "active_interface_ids": ";".join(state.active_interface_ids),
            "active_boundary_ids": ";".join(state.active_boundary_ids),
            "active_load_ids": ";".join(state.active_load_ids),
            "active_joint_ids": ";".join(state.active_joint_ids),
            "operator_set_id": state.operator_set_id, "operator_source": state.operator_source,
            "solve_status": state.solve_status,
            "mechanical_state_action": state.mechanical_state_action,
            "not_required_reason": state.not_required_reason,
            "lcp_call_count": step["lcp_call_count"],
            "parameter_effective": step.get("runtime_parameter_effective", True),
            "parameter_mode": step.get("runtime_parameter_mode", ""),
            "fallback_flag": state.fallback_flag, "fallback_reason": state.fallback_reason,
            "quality_flag": state.quality_flag,
        })
    return pd.DataFrame(rows)


def topology_step_state_lineage_table(result: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for step in result.values():
        state: StageState = step["stage_state"]
        rows.append({
            "state_id": state.stage_state_id, "topology_step_id": state.topology_step_id,
            "parent_topology_step_id": state.parent_topology_step_id or "",
            "parent_state_id": state.parent_stage_state_id or "",
            "reference_state_id": state.reference_state,
            "connection_lock_history_ids": ";".join(state.connection_lock_history_ids),
            "release_history_ids": ";".join(state.release_history_ids),
        })
    return pd.DataFrame(rows)


def topology_step_operator_usage_table(result: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for step in result.values():
        trace = step["contact_trace"]
        rows.append({key: trace.get(key, "") for key in (
            "topology_step_id", "operator_set_id", "operator_source", "q_key",
            "W_struct_key", "Cn_key", "W_total_source", "lcp_call_count",
            "parameter_effective", "parameter_mode", "quality_flag",
        )})
    return pd.DataFrame(rows)


def topology_step_contact_summary_table(result: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for step in result.values():
        state: StageState = step["stage_state"]
        rows.append({
            "topology_step_id": state.topology_step_id,
            "active_interface_ids": ";".join(step["active_interface_ids"]),
            "inactive_interface_ids": ";".join(step["inactive_interface_ids"]),
            "active_index_count": int(np.sum(step["active_index_mask"])),
            "contact_count": len(step["solution"].active_indices),
            "lambda_sum_N": float(np.sum(step["lambda_active"])) if len(step["lambda_active"]) else 0.0,
            "complementarity_residual": step["solution"].residuals.get("complementarity_residual", 0.0),
            "solve_status": step["solve_status"],
        })
    return pd.DataFrame(rows)


def connection_lock_history_table(result: dict[str, dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([row for step in result.values() for row in step.get("connection_lock_history", [])])


def release_history_table(result: dict[str, dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([row for step in result.values() for row in step.get("release_history", [])])
