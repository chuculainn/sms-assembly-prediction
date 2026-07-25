from __future__ import annotations

from io import BytesIO
import json
import zipfile

import pandas as pd

from .data_loader import SMSPackage
from .multi_part import (
    assembly_path_summary,
    coupling_ablation_comparison,
    coupling_block_summary,
    interface_stage_summary,
    topology_summary,
)
from .package_validator import data_truthfulness_statement, validate_package_detailed
from .stage_solver import point_result_table, stage_summary_table
from .stage_state import stage_transition_runtime_table
from .topology_step import (
    connection_lock_history_table,
    release_history_table,
    topology_step_contact_summary_table,
    topology_step_execution_table,
    topology_step_operator_usage_table,
    topology_step_state_lineage_table,
    validate_topology_steps,
)


def runtime_state_lineage(result: dict[str, dict]) -> pd.DataFrame:
    runtime = stage_transition_runtime_table(result)
    if runtime.empty:
        return runtime
    return runtime[[
        "stage_state_id", "stage_id", "stage_type", "parent_stage_id",
        "parent_stage_state_id", "joint_lock_history_id", "joint_lock_source",
        "data_source", "fallback_flag",
    ]].copy()


def coupling_ablation_export(pkg: SMSPackage, result: dict[str, dict], threshold: float = 0.05) -> pd.DataFrame:
    rows = []
    if pkg.raw_tables.get("matrices/vector_layout.csv", pd.DataFrame()).empty:
        return pd.DataFrame()
    for stage_id in result:
        if not bool(pd.Series(result[stage_id].get("active_index_mask", [])).all()):
            continue
        try:
            rows.append(coupling_ablation_comparison(pkg, result, stage_id, threshold)["summary"])
        except (KeyError, ValueError):
            continue
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_runtime_report_zip(
    pkg: SMSPackage,
    result: dict[str, dict],
    kcp_validation: pd.DataFrame,
    traces: list[dict],
    *,
    mc_stats: pd.DataFrame | None = None,
    mc_samples: pd.DataFrame | None = None,
    physical_report: dict | None = None,
    ablation_threshold: float = 0.05,
) -> bytes:
    stage_summary = stage_summary_table(result)
    points = point_result_table(pkg, result)
    interface_summary = interface_stage_summary(pkg, result)
    coupling = pd.concat(
        [coupling_block_summary(pkg, stage_id) for stage_id in result], ignore_index=True
    ) if not pkg.raw_tables.get("matrices/vector_layout.csv", pd.DataFrame()).empty else pd.DataFrame()
    topology = pd.DataFrame([topology_summary(pkg)])
    paths = assembly_path_summary(pkg)
    transitions = stage_transition_runtime_table(result)
    lineage = runtime_state_lineage(result)
    validation_summary = validate_package_detailed(pkg)
    ablation = coupling_ablation_export(pkg, result, ablation_threshold)
    truth = data_truthfulness_statement(pkg)
    step_execution = topology_step_execution_table(result)
    step_validation = validate_topology_steps(pkg)
    subassembly_history = step_execution[[
        column for column in (
            "sample_id", "topology_id", "topology_step_id", "step_order",
            "result_subassembly_id", "active_part_ids", "active_interface_ids",
            "active_joint_ids", "active_boundary_ids", "active_load_ids",
        ) if column in step_execution.columns
    ]].copy()
    step_lineage = topology_step_state_lineage_table(result)
    operator_usage = topology_step_operator_usage_table(result)
    step_contact = topology_step_contact_summary_table(result)
    lock_history = connection_lock_history_table(result)
    release_history = release_history_table(result)

    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        csvs = {
            "stage_summary.csv": stage_summary,
            "point_results.csv": points,
            "kcp_validation.csv": kcp_validation,
            "topology_summary.csv": topology,
            "assembly_path_summary.csv": paths,
            "stage_transition_runtime.csv": transitions,
            "interface_stage_summary.csv": interface_summary,
            "cross_interface_coupling_blocks.csv": coupling,
            "coupling_ablation_comparison.csv": ablation,
            "state_lineage.csv": lineage,
            "validation_summary.csv": validation_summary,
            "topology_step_execution.csv": step_execution,
            "topology_step_validation.csv": step_validation,
            "active_subassembly_history.csv": subassembly_history,
            "topology_step_state_lineage.csv": step_lineage,
            "topology_step_operator_usage.csv": operator_usage,
            "topology_step_contact_summary.csv": step_contact,
            "connection_lock_history.csv": lock_history,
            "release_history.csv": release_history,
        }
        for name, frame in csvs.items():
            archive.writestr(name, frame.to_csv(index=False).encode("utf-8-sig"))
        archive.writestr("data_truthfulness_statement.txt", truth.encode("utf-8"))
        archive.writestr("contact_computation_trace.json", json.dumps(traces, ensure_ascii=False, indent=2))
        if mc_stats is not None and not mc_stats.empty:
            archive.writestr("monte_carlo_stats.csv", mc_stats.to_csv(index=False).encode("utf-8-sig"))
        if mc_samples is not None and not mc_samples.empty:
            archive.writestr("monte_carlo_samples.csv", mc_samples.to_csv(index=False).encode("utf-8-sig"))
        if physical_report is not None:
            archive.writestr(
                "physical_consistency_overall.json",
                json.dumps(physical_report.get("overall", {}), ensure_ascii=False, indent=2),
            )
            for name in ("stage_summary", "check_details", "kcp_anomalies"):
                frame = physical_report.get(name)
                if isinstance(frame, pd.DataFrame):
                    archive.writestr(
                        f"physical_consistency_{name}.csv",
                        frame.to_csv(index=False).encode("utf-8-sig"),
                    )
    return buf.getvalue()

