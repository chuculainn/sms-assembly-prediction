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
from .stage_measurement_update import StageMeasurementUpdateResult
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
    columns = [
        "stage_state_id", "stage_id", "stage_type", "parent_stage_id",
        "parent_stage_state_id", "joint_lock_history_id", "joint_lock_source",
        "data_source", "fallback_flag",
        "state_role", "measurement_checkpoint_id", "measurement_update_id",
        "predicted_state_id", "posterior_parent_state_id",
        "effective_state_id", "measurement_update_status",
        "posterior_accepted", "covariance_trace",
        "state_correction_norm", "rollback_record_id",
    ]
    return runtime[
        [column for column in columns if column in runtime.columns]
    ].copy()


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


def measurement_update_report_tables(
    pkg: SMSPackage,
    result: dict[str, dict],
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """Build durable posterior-update report artifacts from runtime results."""
    updates: list[StageMeasurementUpdateResult] = [
        item["measurement_update"]
        for item in result.values()
        if isinstance(item.get("measurement_update"), StageMeasurementUpdateResult)
    ]
    checkpoints = pkg.raw_tables.get(
        "I_meas/measurement_checkpoint.csv", pd.DataFrame()
    ).copy()
    observation_map = pkg.raw_tables.get(
        "I_meas/measurement_observation_map.csv", pd.DataFrame()
    ).copy()
    summary_rows: list[dict] = []
    innovation_rows: list[dict] = []
    decision_rows: list[dict] = []
    resolve_rows: list[dict] = []
    snapshot_rows: list[dict] = []
    comparison_rows: list[dict] = []
    covariance_rows: list[dict] = []
    rollback_rows: list[dict] = []
    lineage_rows: list[dict] = []
    traces: list[dict] = []

    for update in updates:
        record = update.to_record()
        trace = dict(update.trace)
        summary_rows.append({
            **record,
            "prior_residual_norm": trace.get("prior_residual_norm"),
            "posterior_residual_norm": trace.get("posterior_residual_norm"),
            "posterior_linearized_residual_norm": trace.get(
                "posterior_linearized_residual_norm"
            ),
            "weighted_residual_prior_physical": trace.get(
                "weighted_residual_prior_physical"
            ),
            "weighted_residual_posterior_physical": trace.get(
                "weighted_residual_posterior_physical"
            ),
            "physical_residual_improved": trace.get(
                "physical_residual_improved"
            ),
            "linearization_error_norm": trace.get(
                "linearization_error_norm"
            ),
            "acceptance_basis": trace.get("acceptance_basis"),
            "measurement_covariance_source": trace.get(
                "measurement_covariance_source",
                update.measurement_covariance_source,
            ),
            "P_prior_trace": trace.get("P_prior_trace"),
            "P_posterior_trace": trace.get("P_posterior_trace"),
            "parameter_frozen": trace.get("parameter_frozen", True),
            "sms_frozen": trace.get("sms_frozen", True),
            "engineering_claim_allowed": trace.get(
                "engineering_claim_allowed",
                bool(pkg.manifest.get("engineering_claim_allowed", False)),
            ),
        })
        innovation_rows.extend(update.observation_records)
        for observation_record in update.observation_records:
            comparison_rows.append({
                "record_type": "OBSERVATION",
                "update_id": update.update_id,
                "checkpoint_id": update.checkpoint_id,
                "measurement_id": observation_record.get(
                    "measurement_id", ""
                ),
                "observation_class": observation_record.get(
                    "observation_class", ""
                ),
                "observed_quantity": observation_record.get(
                    "observed_quantity", ""
                ),
                "z_measured": observation_record.get("z_measured"),
                "z_predicted_prior_physical": observation_record.get(
                    "z_pred_prior_physical"
                ),
                "z_predicted_posterior_linearized": (
                    observation_record.get(
                        "z_pred_posterior_linearized"
                    )
                ),
                "z_predicted_posterior_physical": (
                    observation_record.get(
                        "z_pred_posterior_physical"
                    )
                ),
                "residual_prior_physical": observation_record.get(
                    "prior_residual_physical"
                ),
                "residual_posterior_linearized": (
                    observation_record.get(
                        "posterior_residual_linearized"
                    )
                ),
                "residual_posterior_physical": observation_record.get(
                    "posterior_residual_physical"
                ),
                "linearization_error": observation_record.get(
                    "linearization_error"
                ),
            })
        if update.decision_record is not None:
            decision_rows.append(update.decision_record.to_record())
        resolve_rows.append({
            "update_id": update.update_id,
            "checkpoint_id": update.checkpoint_id,
            **update.resolve_requirement.to_record(),
            "resolve_lcp_call_count": update.resolve_lcp_call_count,
        })
        for role, state in (
            ("PREDICTED", next(
                (
                    step.get("predicted_stage_state")
                    for step in result.values()
                    if getattr(
                        step.get("predicted_stage_state"),
                        "stage_state_id",
                        "",
                    ) == update.predicted_state_id
                ),
                None,
            )),
            ("POSTERIOR", update.posterior_state),
        ):
            if state is not None:
                snapshot_rows.append({
                    **state.to_record(),
                    "report_state_role": role,
                    "update_id": update.update_id,
                    "checkpoint_id": update.checkpoint_id,
                })
                lineage_rows.append({
                    "checkpoint_id": update.checkpoint_id,
                    "update_id": update.update_id,
                    "state_role": role,
                    "state_id": state.stage_state_id,
                    "parent_state_id": state.parent_stage_state_id or "",
                    "predicted_state_id": update.predicted_state_id,
                    "posterior_state_id": update.posterior_state_id,
                    "effective_state_id": update.effective_state_id,
                    "posterior_accepted": update.posterior_accepted,
                })
        dimension = max(len(update.eta_prior), len(update.eta_posterior))
        basis = pkg.raw_tables.get(
            "I_stage/state_update_basis.csv", pd.DataFrame()
        )
        basis = basis[
            basis.get(
                "update_state_layout_id", pd.Series(dtype=str)
            ).astype(str).eq(
                str(
                    getattr(
                        update.posterior_state,
                        "update_state_layout_id",
                        "",
                    )
                )
            )
        ] if not basis.empty else basis
        basis = basis.sort_values(
            "component_order", kind="stable"
        ) if not basis.empty and "component_order" in basis else basis
        prior_std = (
            pd.Series(update.P_prior.diagonal() ** 0.5)
            if update.P_prior.shape == (dimension, dimension)
            else pd.Series([float("nan")] * dimension)
        )
        posterior_std = (
            pd.Series(update.P_posterior.diagonal() ** 0.5)
            if update.P_posterior.shape == (dimension, dimension)
            else pd.Series([float("nan")] * dimension)
        )
        for index in range(dimension):
            item = basis.iloc[index] if index < len(basis) else {}
            eta_prior = float(update.eta_prior[index])
            eta_posterior = float(update.eta_posterior[index])
            comparison_rows.append({
                "record_type": "STATE_COMPONENT",
                "update_id": update.update_id,
                "checkpoint_id": update.checkpoint_id,
                "component_order": index,
                "component_id": item.get("component_id", f"component_{index}"),
                "component_type": item.get("component_type", ""),
                "target_object": item.get("target_object_id", ""),
                "eta_prior": eta_prior,
                "eta_posterior": eta_posterior,
                "posterior_increment": eta_posterior - eta_prior,
                "prior_std": float(prior_std.iloc[index]),
                "posterior_std": float(posterior_std.iloc[index]),
                "uncertainty_reduction": float(
                    prior_std.iloc[index] - posterior_std.iloc[index]
                ),
            })
        covariance_rows.append({
            "update_id": update.update_id,
            "checkpoint_id": update.checkpoint_id,
            "predicted_state_id": update.predicted_state_id,
            "posterior_state_id": update.posterior_state_id,
            "prior_trace": (
                float(update.P_prior.trace()) if update.P_prior.size else None
            ),
            "posterior_trace": (
                float(update.P_posterior.trace())
                if update.P_posterior.size else None
            ),
            "trace_reduction": (
                float(update.P_prior.trace() - update.P_posterior.trace())
                if update.P_prior.size and update.P_posterior.size else None
            ),
            "covariance_source": (
                getattr(update.posterior_state, "covariance_source", "")
                if update.posterior_state is not None else ""
            ),
        })
        if update.rollback_record is not None:
            rollback_rows.append(update.rollback_record.to_record())
        traces.append({
            "update": record,
            "decision": (
                update.decision_record.to_record()
                if update.decision_record is not None else {}
            ),
            "resolve_requirement": update.resolve_requirement.to_record(),
            "rollback": (
                update.rollback_record.to_record()
                if update.rollback_record is not None else {}
            ),
            "trace": trace,
        })

    tables = {
        "measurement_checkpoint_table.csv": checkpoints,
        "measurement_update_summary.csv": pd.DataFrame(summary_rows),
        "measurement_innovation.csv": pd.DataFrame(innovation_rows),
        "measurement_observation_map.csv": observation_map,
        "update_decision_record.csv": pd.DataFrame(decision_rows),
        "resolve_requirement.csv": pd.DataFrame(resolve_rows),
        "posterior_state_snapshot.csv": pd.DataFrame(snapshot_rows),
        "predicted_posterior_comparison.csv": pd.DataFrame(comparison_rows),
        "stage_covariance_trace.csv": pd.DataFrame(covariance_rows),
        "update_rollback_record.csv": pd.DataFrame(rollback_rows),
        "posterior_state_lineage.csv": pd.DataFrame(lineage_rows),
    }
    return tables, traces


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
    measurement_tables, measurement_traces = measurement_update_report_tables(
        pkg, result
    )

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
            **measurement_tables,
        }
        for name, frame in csvs.items():
            archive.writestr(name, frame.to_csv(index=False).encode("utf-8-sig"))
        archive.writestr("data_truthfulness_statement.txt", truth.encode("utf-8"))
        archive.writestr("contact_computation_trace.json", json.dumps(traces, ensure_ascii=False, indent=2))
        archive.writestr(
            "measurement_update_trace.json",
            json.dumps(measurement_traces, ensure_ascii=False, indent=2),
        )
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

