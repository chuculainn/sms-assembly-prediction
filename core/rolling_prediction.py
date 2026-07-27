from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .data_loader import SMSPackage
from .kcp import extract_kcp
from .multi_part import vector_layout
from .stage_state import StageState
from .topology_step import (
    TopologyStepSpec,
    TopologyStepValidationError,
    load_topology_steps,
    run_topology_steps_from_state,
)


PLAN_TABLE = "I_pred/rolling_prediction_plan.csv"
SAMPLE_SET_TABLE = "I_pred/virtual_sms_sample_set.csv"
SAMPLE_TABLE = "I_pred/virtual_sms_sample.csv"
COMPONENT_TABLE = "I_pred/virtual_sms_component.csv"
COEFFICIENT_TABLE = "I_pred/virtual_sms_coefficients.csv"
ASSIGNMENT_TABLE = "I_pred/future_sms_assignment.csv"
MAPPING_TABLE = "I_pred/sms_operator_mapping.csv"
SCENARIO_TABLE = "I_pred/future_process_scenario.csv"
KCP_CONFIG_TABLE = "I_pred/rolling_kcp_config.csv"
MATRIX_MANIFEST_TABLE = "matrices/matrix_manifest.csv"

REQUIRE_ACCEPTED_POSTERIOR = "REQUIRE_ACCEPTED_POSTERIOR"
PREDICTED_CUTOFF_BASELINE = "PREDICTED_CUTOFF_BASELINE"
DELTA_FROM_OPERATOR_REFERENCE = "DELTA_FROM_OPERATOR_REFERENCE"
EXPLICIT_SAMPLE_NATURE = "EXPLICIT_SYNTHETIC_VIRTUAL_SMS_LIBRARY"
DETERMINISTIC_VALUES = "DETERMINISTIC_EXPLICIT_VALUES"
DETERMINISTIC_BASELINE = "DETERMINISTIC_BASELINE"
SUPPORTED_AGGREGATION_POLICY = (
    "ONE_SMS_PER_PART_PLUS_ONE_INCREMENT_PER_INTERFACE_STAGE"
)
BASELINE_REQUIRED = "POSTERIOR_AND_PREDICTED_CUTOFF"
BASELINE_POSTERIOR_ONLY = "POSTERIOR_ONLY"
SUPPORTED_BASELINE_POLICIES = {
    BASELINE_REQUIRED,
    PREDICTED_CUTOFF_BASELINE,
    BASELINE_POSTERIOR_ONLY,
}
FAIL_RUN_IF_ANY_FORMAL_SAMPLE_FAILS = (
    "FAIL_RUN_IF_ANY_FORMAL_SAMPLE_FAILS"
)
SUPPORTED_FAILURE_POLICIES = {
    FAIL_RUN_IF_ANY_FORMAL_SAMPLE_FAILS,
}
EFFECTIVE_MAPPING = "EFFECTIVE_MAPPING"
EXPLICIT_ZERO_NO_EFFECT = "EXPLICIT_ZERO_NO_EFFECT"
SUPPORTED_MAPPING_ROLES = {
    EFFECTIVE_MAPPING,
    EXPLICIT_ZERO_NO_EFFECT,
}
DIRECT_KCP_MAPPING_ROLE = "SMS_TO_KCP_" + "DIRECT"
PASS_QUALITY_FLAGS = {"PASS"}
EMPIRICAL_FRACTION_LABEL = (
    "EMPIRICAL_SAMPLE_FRACTION_NOT_FAILURE_PROBABILITY"
)
DIRECT_SMS_ADD = "ADD_DIRECT_SMS_CONTRIBUTION"
DIRECT_SMS_ALREADY_INCLUDED = (
    "DIRECT_SMS_ALREADY_INCLUDED_IN_FINAL_STATE"
)
CONTRIBUTION_LEDGER_COLUMNS = [
    "sample_id",
    "source_state_role",
    "source_class",
    "source_id",
    "origin_stage_id_optional",
    "increment_definition_id",
    "target_kcp_id",
    "contribution_vector",
]
BASELINE_COMPARISON_COLUMNS = [
    "virtual_sms_sample_id",
    "kcp_id",
    "predicted_cutoff_result",
    "posterior_cutoff_result",
    "posterior_minus_predicted",
    "source_predicted_state_id",
    "source_posterior_state_id",
    "posterior_quality_status",
    "baseline_quality_status",
    "comparison_status",
    "missing_side",
    "posterior_failure_reason",
    "baseline_failure_reason",
    "quality_status",
]
ROLLING_KCP_COLUMNS = [
    "kcp_id",
    "feature_type",
    "stage_id",
    "predicted_value",
    "unit",
    "nominal_value",
    "lower_tol",
    "upper_tol",
    "description",
    "sms_contribution",
    "contact_contribution",
    "other_contribution",
    "projection_matrix_id",
    "virtual_sms_direct_contribution_candidate",
    "virtual_sms_direct_contribution",
    "source_state_role",
    "virtual_sms_sample_id",
    "aggregation_policy",
    "final_state_includes_direct_sms_geometry",
    "direct_sms_aggregation_action",
    "double_count_status",
    "tolerance_status",
    "quality_status",
    "sample_quality_status",
]
ROLLING_KCP_SUMMARY_COLUMNS = [
    "kcp_id",
    "count",
    "descriptive_mean",
    "descriptive_std",
    "empirical_min",
    "empirical_max",
    "empirical_p05",
    "empirical_p50",
    "empirical_p95",
    "tolerance_exceedance_count",
    "tolerance_exceedance_fraction",
    "fraction_semantics",
    "probability_interpretation_allowed",
    "engineering_claim_allowed",
]


class RollingPredictionValidationError(ValueError):
    """Raised when a formal rolling quality gate blocks execution."""


class RollingPredictionRuntimeError(RuntimeError):
    """Raised when the rolling solve or KCP path cannot produce a valid result."""

    def __init__(self, message: str, failure_type: str = "RUNTIME") -> None:
        super().__init__(message)
        self.failure_type = failure_type


def _blank(value: object) -> bool:
    return value is None or (
        isinstance(value, float) and np.isnan(value)
    ) or not str(value).strip()


def _text(value: object, default: str = "") -> str:
    return default if _blank(value) else str(value).strip()


def _bool(value: object, default: bool = False) -> bool:
    if _blank(value):
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "是"}


def _strict_bool(value: object, field_name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = "" if _blank(value) else str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise RollingPredictionValidationError(
        f"{field_name}必须是严格布尔值true/false，实际={value!r}"
    )


def _ids(value: object) -> tuple[str, ...]:
    if _blank(value):
        return ()
    if isinstance(value, (list, tuple, set)):
        raw = list(value)
    else:
        raw = str(value).replace(",", ";").split(";")
    return tuple(dict.fromkeys(str(item).strip() for item in raw if str(item).strip()))


def _table(pkg: SMSPackage, name: str) -> pd.DataFrame:
    return pkg.raw_tables.get(name, pd.DataFrame()).copy()


@dataclass(frozen=True)
class RollingPredictionPlan:
    rolling_plan_id: str
    topology_id: str
    source_checkpoint_id: str
    source_topology_step_id: str
    source_state_policy: str
    source_posterior_state_id_optional: str
    prediction_start_step_id: str
    prediction_end_step_id_optional: str
    virtual_sms_sample_set_id: str
    future_part_ids: tuple[str, ...]
    future_process_scenario_id: str
    kcp_set_id: str
    baseline_comparison_policy: str
    aggregation_policy: str
    failure_policy: str
    active_flag: bool
    engineering_claim_allowed: bool
    quality_flag: str
    notes: str


@dataclass(frozen=True)
class VirtualSMSSampleSet:
    sample_set_id: str
    sample_set_name: str
    sample_nature: str
    sample_count: int
    sms_layout_id: str
    reference_sms_id: str
    generation_method: str
    probability_interpretation_allowed: bool
    engineering_claim_allowed: bool
    source: str
    quality_flag: str
    notes: str


@dataclass(frozen=True)
class VirtualSMSSample:
    virtual_sms_sample_id: str
    sample_set_id: str
    part_id: str
    sms_layout_id: str
    reference_sms_id: str
    coefficient_source: str
    sample_order: int
    quality_flag: str
    source: str
    notes: str


@dataclass(frozen=True)
class VirtualSMSComponent:
    sms_layout_id: str
    component_order: int
    component_id: str
    component_type: str
    part_id: str
    unit: str
    reference_state_id: str
    quality_flag: str
    description: str


@dataclass(frozen=True)
class FutureSMSAssignment:
    assignment_id: str
    rolling_plan_id: str
    virtual_sms_sample_id: str
    part_id: str
    first_effective_topology_step_id: str
    last_effective_topology_step_id_optional: str
    mapping_semantics: str
    reference_sms_id: str
    quality_flag: str


@dataclass(frozen=True)
class SMSOperatorMappingSpec:
    mapping_id: str
    rolling_plan_id: str
    part_id: str
    operator_set_id: str
    matrix_id: str
    row_vector_layout_id: str
    column_sms_layout_id: str
    mapping_semantics: str
    reference_sms_id: str
    first_valid_step_id: str
    last_valid_step_id_optional: str
    mapping_role: str
    derivation_source: str
    quality_flag: str


@dataclass(frozen=True)
class FutureProcessScenario:
    future_process_scenario_id: str
    scenario_type: str
    q_correction_matrix_id_optional: str
    parameter_override_allowed: bool
    probability_weight_optional: float | None
    probability_interpretation_allowed: bool
    quality_flag: str


@dataclass
class RollingSampleResult:
    rolling_run_id: str
    rolling_plan_id: str
    virtual_sms_sample_id: str
    source_checkpoint_id: str
    source_predicted_state_id: str
    source_posterior_state_id: str
    effective_source_state_id: str
    source_state_role: str
    prediction_start_step_id: str
    prediction_end_step_id: str
    future_part_ids: tuple[str, ...]
    sms_coefficients: dict[str, list[float]]
    sms_reference_coefficients: dict[str, list[float]]
    sms_delta_coefficients: dict[str, list[float]]
    future_process_scenario_id: str
    step_results: dict[str, dict[str, Any]]
    final_state_id: str
    kcp_prediction_result: pd.DataFrame
    contribution_ledger: pd.DataFrame
    contact_mode_signature: str
    quality_status: str
    failure_reason: str
    trace: dict[str, Any]
    double_count_fail_count: int = 0
    application_fail_count: int = 0
    kcp_fail_count: int = 0
    authoritative_failure_reason: str = ""


@dataclass
class RollingPredictionSummary:
    rolling_run_id: str
    rolling_plan_id: str
    sample_count: int
    success_count: int
    failure_count: int
    baseline_attempt_count: int
    baseline_success_count: int
    baseline_failure_count: int
    baseline_quality_status: str
    kcp_summary: pd.DataFrame
    contact_mode_counts: pd.DataFrame
    probability_interpretation_allowed: bool
    engineering_claim_allowed: bool
    quality_status: str


@dataclass
class RollingPredictionRunResult:
    rolling_run_id: str
    plan: RollingPredictionPlan
    source_posterior_state: StageState
    source_predicted_state: StageState
    sample_results: list[RollingSampleResult]
    sample_failures: list[RollingSampleResult]
    predicted_baseline_results: list[RollingSampleResult]
    predicted_baseline_failures: list[RollingSampleResult]
    kcp_predictions: pd.DataFrame
    baseline_comparison: pd.DataFrame
    descriptive_summary: RollingPredictionSummary
    contact_mode_summary: pd.DataFrame
    quality_gates: pd.DataFrame
    trace: dict[str, Any]
    quality_status: str
    authoritative_failure_reason: str
    status_summary: dict[str, Any]


def has_rolling_prediction_plans(pkg: SMSPackage) -> bool:
    table = _table(pkg, PLAN_TABLE)
    if table.empty:
        return False
    active = table.get("active_flag", pd.Series(True, index=table.index))
    return bool(active.map(lambda value: _bool(value, True)).any())


def load_rolling_prediction_plans(pkg: SMSPackage) -> list[RollingPredictionPlan]:
    rows: list[RollingPredictionPlan] = []
    for _, row in _table(pkg, PLAN_TABLE).iterrows():
        rows.append(RollingPredictionPlan(
            rolling_plan_id=_text(row.get("rolling_plan_id")),
            topology_id=_text(row.get("topology_id")),
            source_checkpoint_id=_text(row.get("source_checkpoint_id")),
            source_topology_step_id=_text(row.get("source_topology_step_id")),
            source_state_policy=_text(row.get("source_state_policy")),
            source_posterior_state_id_optional=_text(
                row.get("source_posterior_state_id_optional")
            ),
            prediction_start_step_id=_text(row.get("prediction_start_step_id")),
            prediction_end_step_id_optional=_text(
                row.get("prediction_end_step_id_optional")
            ),
            virtual_sms_sample_set_id=_text(row.get("virtual_sms_sample_set_id")),
            future_part_ids=_ids(row.get("future_part_ids")),
            future_process_scenario_id=_text(
                row.get("future_process_scenario_id")
            ),
            kcp_set_id=_text(row.get("kcp_set_id")),
            baseline_comparison_policy=_text(
                row.get("baseline_comparison_policy")
            ),
            aggregation_policy=_text(row.get("aggregation_policy")),
            failure_policy=_text(row.get("failure_policy")),
            active_flag=_bool(row.get("active_flag"), True),
            engineering_claim_allowed=_bool(
                row.get("engineering_claim_allowed"), False
            ),
            quality_flag=_text(row.get("quality_flag")),
            notes=_text(row.get("notes")),
        ))
    return rows


def load_virtual_sms_sample_sets(pkg: SMSPackage) -> list[VirtualSMSSampleSet]:
    out: list[VirtualSMSSampleSet] = []
    for _, row in _table(pkg, SAMPLE_SET_TABLE).iterrows():
        out.append(VirtualSMSSampleSet(
            sample_set_id=_text(row.get("sample_set_id")),
            sample_set_name=_text(row.get("sample_set_name")),
            sample_nature=_text(row.get("sample_nature")),
            sample_count=int(float(row.get("sample_count", 0))),
            sms_layout_id=_text(row.get("sms_layout_id")),
            reference_sms_id=_text(row.get("reference_sms_id")),
            generation_method=_text(row.get("generation_method")),
            probability_interpretation_allowed=_bool(
                row.get("probability_interpretation_allowed"), False
            ),
            engineering_claim_allowed=_bool(
                row.get("engineering_claim_allowed"), False
            ),
            source=_text(row.get("source")),
            quality_flag=_text(row.get("quality_flag")),
            notes=_text(row.get("notes")),
        ))
    return out


def load_virtual_sms_samples(pkg: SMSPackage) -> list[VirtualSMSSample]:
    out: list[VirtualSMSSample] = []
    for position, (_, row) in enumerate(_table(pkg, SAMPLE_TABLE).iterrows()):
        out.append(VirtualSMSSample(
            virtual_sms_sample_id=_text(row.get("virtual_sms_sample_id")),
            sample_set_id=_text(row.get("sample_set_id")),
            part_id=_text(row.get("part_id")),
            sms_layout_id=_text(row.get("sms_layout_id")),
            reference_sms_id=_text(row.get("reference_sms_id")),
            coefficient_source=_text(row.get("coefficient_source")),
            sample_order=int(float(row.get("sample_order", position))),
            quality_flag=_text(row.get("quality_flag")),
            source=_text(row.get("source")),
            notes=_text(row.get("notes")),
        ))
    return out


def load_virtual_sms_components(pkg: SMSPackage) -> list[VirtualSMSComponent]:
    out: list[VirtualSMSComponent] = []
    for _, row in _table(pkg, COMPONENT_TABLE).iterrows():
        out.append(VirtualSMSComponent(
            sms_layout_id=_text(row.get("sms_layout_id")),
            component_order=int(float(row.get("component_order", 0))),
            component_id=_text(row.get("component_id")),
            component_type=_text(row.get("component_type")),
            part_id=_text(row.get("part_id")),
            unit=_text(row.get("unit")),
            reference_state_id=_text(row.get("reference_state_id")),
            quality_flag=_text(row.get("quality_flag")),
            description=_text(row.get("description")),
        ))
    return out


def load_future_sms_assignments(pkg: SMSPackage) -> list[FutureSMSAssignment]:
    out: list[FutureSMSAssignment] = []
    for _, row in _table(pkg, ASSIGNMENT_TABLE).iterrows():
        out.append(FutureSMSAssignment(
            assignment_id=_text(row.get("assignment_id")),
            rolling_plan_id=_text(row.get("rolling_plan_id")),
            virtual_sms_sample_id=_text(row.get("virtual_sms_sample_id")),
            part_id=_text(row.get("part_id")),
            first_effective_topology_step_id=_text(
                row.get("first_effective_topology_step_id")
            ),
            last_effective_topology_step_id_optional=_text(
                row.get("last_effective_topology_step_id_optional")
            ),
            mapping_semantics=_text(row.get("mapping_semantics")),
            reference_sms_id=_text(row.get("reference_sms_id")),
            quality_flag=_text(row.get("quality_flag")),
        ))
    return out


def load_sms_operator_mappings(pkg: SMSPackage) -> list[SMSOperatorMappingSpec]:
    out: list[SMSOperatorMappingSpec] = []
    for _, row in _table(pkg, MAPPING_TABLE).iterrows():
        out.append(SMSOperatorMappingSpec(
            mapping_id=_text(row.get("mapping_id")),
            rolling_plan_id=_text(row.get("rolling_plan_id")),
            part_id=_text(row.get("part_id")),
            operator_set_id=_text(row.get("operator_set_id")),
            matrix_id=_text(row.get("matrix_id")),
            row_vector_layout_id=_text(row.get("row_vector_layout_id")),
            column_sms_layout_id=_text(row.get("column_sms_layout_id")),
            mapping_semantics=_text(row.get("mapping_semantics")),
            reference_sms_id=_text(row.get("reference_sms_id")),
            first_valid_step_id=_text(row.get("first_valid_step_id")),
            last_valid_step_id_optional=_text(
                row.get("last_valid_step_id_optional")
            ),
            mapping_role=_text(row.get("mapping_role")),
            derivation_source=_text(row.get("derivation_source")),
            quality_flag=_text(row.get("quality_flag")),
        ))
    return out


def load_future_process_scenarios(
    pkg: SMSPackage,
) -> list[FutureProcessScenario]:
    out: list[FutureProcessScenario] = []
    for _, row in _table(pkg, SCENARIO_TABLE).iterrows():
        weight = row.get("probability_weight_optional")
        out.append(FutureProcessScenario(
            future_process_scenario_id=_text(
                row.get("future_process_scenario_id")
            ),
            scenario_type=_text(row.get("scenario_type")),
            q_correction_matrix_id_optional=_text(
                row.get("q_correction_matrix_id_optional")
            ),
            parameter_override_allowed=_bool(
                row.get("parameter_override_allowed"), False
            ),
            probability_weight_optional=(
                None if _blank(weight) else float(weight)
            ),
            probability_interpretation_allowed=_bool(
                row.get("probability_interpretation_allowed"), False
            ),
            quality_flag=_text(row.get("quality_flag")),
        ))
    return out


def _gate(
    rows: list[dict[str, Any]],
    name: str,
    passed: bool,
    detail: str,
    *,
    blocking: bool = True,
) -> None:
    rows.append({
        "check_item": name,
        "status": "PASS" if passed else "FAIL",
        "blocking": bool(blocking),
        "detail": detail,
    })


def _matrix_and_manifest(
    pkg: SMSPackage, matrix_id: str
) -> tuple[np.ndarray, pd.Series]:
    manifest = _table(pkg, MATRIX_MANIFEST_TABLE)
    matches = manifest[
        manifest.get("matrix_id", pd.Series(dtype=str)).astype(str).eq(matrix_id)
        | manifest.get("npz_key", pd.Series(dtype=str)).astype(str).eq(matrix_id)
    ]
    if matches.empty:
        raise RollingPredictionValidationError(
            f"MatrixManifest缺少matrix_id={matrix_id}"
        )
    row = matches.iloc[0]
    key = _text(row.get("npz_key"), matrix_id)
    if key not in pkg.matrices:
        raise RollingPredictionValidationError(f"NPZ缺少key={key}")
    return np.asarray(pkg.matrices[key], dtype=float), row


def _quality_pass(value: object) -> bool:
    return _text(value).upper() in PASS_QUALITY_FLAGS


def _plan_end_step(
    plan: RollingPredictionPlan,
    specs: list[TopologyStepSpec],
) -> str:
    if plan.prediction_end_step_id_optional:
        return plan.prediction_end_step_id_optional
    return max(specs, key=lambda item: item.step_order).topology_step_id


def _kcp_ids_for_set(
    pkg: SMSPackage,
    kcp_set_id: str,
) -> list[str]:
    definitions = pkg.kcp_kcm.copy()
    if definitions.empty or "kcp_set_id" not in definitions.columns:
        return []
    role = definitions.get(
        "feature_role", pd.Series("KCP", index=definitions.index)
    ).astype(str)
    selected = definitions[
        role.eq("KCP")
        & definitions["kcp_set_id"].astype(str).eq(kcp_set_id)
    ]
    id_column = (
        "feature_id" if "feature_id" in selected.columns else "kcp_id"
    )
    if id_column not in selected.columns:
        return []
    return selected[id_column].astype(str).tolist()


def _kcp_direct_semantics(
    configs: pd.DataFrame,
) -> tuple[dict[str, bool], list[str]]:
    field_name = "final_state_includes_direct_sms_geometry"
    if field_name not in configs.columns:
        return {}, [f"缺少字段{field_name}"]
    semantics: dict[str, bool] = {}
    errors: list[str] = []
    grouped: dict[str, set[bool]] = {}
    for index, row in configs.iterrows():
        part_id = _text(row.get("part_id"))
        try:
            value = _strict_bool(row.get(field_name), field_name)
        except RollingPredictionValidationError as exc:
            errors.append(f"row={index}:{exc}")
            continue
        grouped.setdefault(part_id, set()).add(value)
    for part_id, values in grouped.items():
        if len(values) != 1:
            errors.append(
                f"part_id={part_id}:冲突声明={sorted(values)}"
            )
        else:
            semantics[part_id] = next(iter(values))
    return semantics, errors


def _source_linkage_evidence(
    pkg: SMSPackage,
    plan: RollingPredictionPlan,
    topology_result: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    checkpoints = _table(pkg, "I_meas/measurement_checkpoint.csv")
    matches = checkpoints[
        checkpoints.get(
            "checkpoint_id", pd.Series(dtype=str)
        ).astype(str).eq(plan.source_checkpoint_id)
    ]
    evidence: dict[str, Any] = {
        "plan_topology_id": plan.topology_id,
        "checkpoint_topology_id": "",
        "plan_source_step_id": plan.source_topology_step_id,
        "checkpoint_topology_step_id": "",
        "measurement_update_id": "",
        "measurement_update_checkpoint_id": "",
        "measurement_update_posterior_state_id": "",
        "actual_source_state_id": "",
        "actual_source_state_role": "",
        "actual_source_state_checkpoint_id": "",
        "actual_source_state_sample_id": "",
        "source_linkage_status": "FAIL",
        "failure_reasons": [],
    }
    reasons: list[str] = evidence["failure_reasons"]
    if len(matches) != 1:
        reasons.append(
            f"source checkpoint匹配数必须为1，实际={len(matches)}"
        )
        return evidence
    checkpoint = matches.iloc[0]
    checkpoint_topology_id = _text(checkpoint.get("topology_id"))
    checkpoint_step_id = _text(checkpoint.get("topology_step_id"))
    evidence["checkpoint_topology_id"] = checkpoint_topology_id
    evidence["checkpoint_topology_step_id"] = checkpoint_step_id
    if checkpoint_topology_id != plan.topology_id:
        reasons.append(
            "checkpoint.topology_id与plan.topology_id不一致"
        )
    if checkpoint_step_id != plan.source_topology_step_id:
        reasons.append(
            "checkpoint.topology_step_id与plan source step不一致"
        )
    if topology_result is None:
        evidence["source_linkage_status"] = (
            "PASS" if not reasons else "FAIL"
        )
        return evidence

    source = topology_result.get(plan.source_topology_step_id)
    if source is None:
        reasons.append("plan source topology step没有运行结果")
        return evidence
    update = source.get("measurement_update")
    posterior = source.get("posterior_stage_state")
    actual = source.get("stage_state")
    predicted = source.get("predicted_stage_state")
    if update is None:
        reasons.append("source step缺少measurement update result")
        return evidence

    evidence.update({
        "measurement_update_id": _text(
            getattr(update, "update_id", "")
        ),
        "measurement_update_checkpoint_id": _text(
            getattr(update, "checkpoint_id", "")
        ),
        "measurement_update_posterior_state_id": _text(
            getattr(update, "posterior_state_id", "")
        ),
        "actual_source_state_id": _text(
            getattr(actual, "stage_state_id", "")
        ),
        "actual_source_state_role": _text(
            getattr(actual, "state_role", "")
        ),
        "actual_source_state_checkpoint_id": _text(
            getattr(actual, "source_checkpoint_id", "")
            or getattr(actual, "measurement_checkpoint_id", "")
        ),
        "actual_source_state_sample_id": _text(
            getattr(actual, "sample_id", "")
        ),
    })
    if _text(getattr(update, "checkpoint_id", "")) != plan.source_checkpoint_id:
        reasons.append("measurement update checkpoint_id不一致")
    if (
        _text(getattr(update, "topology_step_id", ""))
        != plan.source_topology_step_id
    ):
        reasons.append("measurement update topology_step_id不一致")
    if not bool(getattr(update, "posterior_accepted", False)):
        reasons.append("measurement update posterior_accepted不是true")
    if _text(getattr(update, "quality_flag", "")).upper() != "PASS":
        reasons.append("measurement update quality_flag不是PASS")
    if getattr(update, "rollback_record", None) is not None:
        reasons.append("measurement update存在rollback替代posterior")
    posterior_state_id = _text(getattr(update, "posterior_state_id", ""))
    effective_state_id = _text(getattr(update, "effective_state_id", ""))
    if not posterior_state_id:
        reasons.append("measurement update posterior_state_id为空")
    if effective_state_id != posterior_state_id:
        reasons.append(
            "measurement update effective_state_id未指向accepted posterior"
        )
    if posterior is None or actual is None:
        reasons.append("source step缺少posterior/actual source state")
        return evidence
    actual_state_id = _text(getattr(actual, "stage_state_id", ""))
    posterior_actual_id = _text(
        getattr(posterior, "stage_state_id", "")
    )
    if posterior_actual_id != posterior_state_id:
        reasons.append("posterior state ID与measurement update不一致")
    if actual_state_id != posterior_state_id:
        reasons.append("实际source state不是accepted posterior")
    if _text(getattr(actual, "effective_state_id", "")) != posterior_state_id:
        reasons.append("实际source effective_state_id不一致")
    if _text(getattr(actual, "state_role", "")) != "POSTERIOR":
        reasons.append("实际source state role不是POSTERIOR")
    if not bool(getattr(actual, "posterior_accepted", False)):
        reasons.append("实际source state未标记posterior accepted")
    if _text(getattr(actual, "quality_flag", "")).upper() != "PASS":
        reasons.append("实际source state quality_flag不是PASS")
    if _text(getattr(actual, "topology_id", "")) != plan.topology_id:
        reasons.append("实际source state topology_id不一致")
    if (
        _text(getattr(actual, "topology_step_id", ""))
        != plan.source_topology_step_id
    ):
        reasons.append("实际source state topology_step_id不一致")
    if (
        _text(getattr(actual, "measurement_checkpoint_id", ""))
        != plan.source_checkpoint_id
        or _text(getattr(actual, "source_checkpoint_id", ""))
        != plan.source_checkpoint_id
    ):
        reasons.append("实际source state checkpoint_id不一致")
    if (
        _text(getattr(actual, "measurement_update_id", ""))
        != _text(getattr(update, "update_id", ""))
    ):
        reasons.append("实际source state measurement_update_id不一致")
    if (
        plan.source_posterior_state_id_optional
        and actual_state_id != plan.source_posterior_state_id_optional
    ):
        reasons.append(
            "plan source_posterior_state_id与实际source state不一致"
        )
    update_posterior = getattr(update, "posterior_state", None)
    sample_ids = {
        _text(getattr(item, "sample_id", ""))
        for item in (actual, posterior, update_posterior, predicted)
        if item is not None
    }
    if len(sample_ids) != 1 or "" in sample_ids:
        reasons.append(
            f"source predicted/posterior sample_id不闭合: {sorted(sample_ids)}"
        )
    evidence["source_linkage_status"] = (
        "PASS" if not reasons else "FAIL"
    )
    return evidence


def validate_rolling_prediction_package(
    pkg: SMSPackage,
    topology_result: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    if not has_rolling_prediction_plans(pkg):
        return pd.DataFrame(
            columns=["check_item", "status", "blocking", "detail"]
        )
    rows: list[dict[str, Any]] = []
    plans = [plan for plan in load_rolling_prediction_plans(pkg) if plan.active_flag]
    ids = [plan.rolling_plan_id for plan in plans]
    _gate(rows, "rolling_plan_id唯一", len(ids) == len(set(ids)) and bool(ids), str(ids))
    specs = load_topology_steps(pkg)
    spec_by_id = {spec.topology_step_id: spec for spec in specs}
    order = {spec.topology_step_id: spec.step_order for spec in specs}
    part_ids = set(pkg.parts.get("part_id", pd.Series(dtype=str)).astype(str))
    checkpoint_table = _table(pkg, "I_meas/measurement_checkpoint.csv")
    sets = {item.sample_set_id: item for item in load_virtual_sms_sample_sets(pkg)}
    samples = load_virtual_sms_samples(pkg)
    components = load_virtual_sms_components(pkg)
    assignments = load_future_sms_assignments(pkg)
    mappings = load_sms_operator_mappings(pkg)
    scenarios = {
        item.future_process_scenario_id: item
        for item in load_future_process_scenarios(pkg)
    }
    coefficients = _table(pkg, COEFFICIENT_TABLE)
    kcp_configs = _table(pkg, KCP_CONFIG_TABLE)
    manifest_table = _table(pkg, MATRIX_MANIFEST_TABLE)
    layout = vector_layout(pkg)
    row_layout_id = (
        _text(layout.iloc[0].get("vector_layout_id"))
        if not layout.empty and "vector_layout_id" in layout.columns else ""
    )
    dimension = len(pkg.contact_points)

    for plan in plans:
        _gate(
            rows, f"{plan.rolling_plan_id}:plan quality",
            _quality_pass(plan.quality_flag),
            f"quality_flag={plan.quality_flag or 'BLANK'}",
        )
        _gate(
            rows, f"{plan.rolling_plan_id}:source policy",
            plan.source_state_policy == REQUIRE_ACCEPTED_POSTERIOR,
            plan.source_state_policy,
        )
        checkpoint_matches = checkpoint_table[
            checkpoint_table.get("checkpoint_id", pd.Series(dtype=str))
            .astype(str).eq(plan.source_checkpoint_id)
        ]
        _gate(
            rows, f"{plan.rolling_plan_id}:source checkpoint",
            len(checkpoint_matches) == 1,
            f"matches={len(checkpoint_matches)}",
        )
        source_linkage = _source_linkage_evidence(
            pkg, plan, topology_result
        )
        _gate(
            rows,
            f"{plan.rolling_plan_id}:source checkpoint/topology linkage",
            source_linkage["source_linkage_status"] == "PASS",
            json.dumps(
                source_linkage, ensure_ascii=False, sort_keys=True
            ),
        )
        _gate(
            rows, f"{plan.rolling_plan_id}:start/end step",
            plan.source_topology_step_id in order
            and plan.prediction_start_step_id in order
            and order.get(plan.source_topology_step_id, np.inf)
            < order.get(plan.prediction_start_step_id, -np.inf)
            and (
                not plan.prediction_end_step_id_optional
                or (
                    plan.prediction_end_step_id_optional in order
                    and order[plan.prediction_end_step_id_optional]
                    >= order[plan.prediction_start_step_id]
                )
            ),
            (
                f"source={plan.source_topology_step_id}, "
                f"start={plan.prediction_start_step_id}, "
                f"end={plan.prediction_end_step_id_optional}"
            ),
        )
        _gate(
            rows, f"{plan.rolling_plan_id}:aggregation policy",
            plan.aggregation_policy == SUPPORTED_AGGREGATION_POLICY,
            plan.aggregation_policy,
        )
        _gate(
            rows, f"{plan.rolling_plan_id}:baseline policy",
            plan.baseline_comparison_policy in SUPPORTED_BASELINE_POLICIES,
            plan.baseline_comparison_policy,
        )
        _gate(
            rows, f"{plan.rolling_plan_id}:failure policy",
            plan.failure_policy in SUPPORTED_FAILURE_POLICIES,
            plan.failure_policy,
        )
        _gate(
            rows, f"{plan.rolling_plan_id}:future parts",
            bool(plan.future_part_ids)
            and set(plan.future_part_ids) <= part_ids,
            str(plan.future_part_ids),
        )
        sample_set = sets.get(plan.virtual_sms_sample_set_id)
        _gate(
            rows, f"{plan.rolling_plan_id}:sample set",
            sample_set is not None,
            plan.virtual_sms_sample_set_id,
        )
        if sample_set is None:
            continue
        _gate(
            rows, f"{plan.rolling_plan_id}:sample set quality",
            _quality_pass(sample_set.quality_flag),
            f"quality_flag={sample_set.quality_flag}",
        )
        _gate(
            rows, f"{plan.rolling_plan_id}:truth labels",
            sample_set.sample_nature == EXPLICIT_SAMPLE_NATURE
            and sample_set.generation_method == DETERMINISTIC_VALUES
            and not sample_set.probability_interpretation_allowed
            and not sample_set.engineering_claim_allowed
            and not plan.engineering_claim_allowed,
            (
                f"nature={sample_set.sample_nature}, "
                f"generation={sample_set.generation_method}, "
                "probability=false, engineering=false"
            ),
        )
        sample_rows = [
            item for item in samples
            if item.sample_set_id == sample_set.sample_set_id
        ]
        sample_ids = sorted({
            item.virtual_sms_sample_id for item in sample_rows
        })
        _gate(
            rows, f"{plan.rolling_plan_id}:sample count",
            len(sample_ids) == sample_set.sample_count,
            f"declared={sample_set.sample_count}, actual={len(sample_ids)}",
        )
        _gate(
            rows, f"{plan.rolling_plan_id}:sample row quality",
            bool(sample_rows)
            and all(_quality_pass(item.quality_flag) for item in sample_rows),
            f"rows={len(sample_rows)}",
        )
        scenario = scenarios.get(plan.future_process_scenario_id)
        _gate(
            rows, f"{plan.rolling_plan_id}:deterministic scenario",
            scenario is not None
            and scenario.scenario_type == DETERMINISTIC_BASELINE
            and not scenario.parameter_override_allowed
            and not scenario.probability_interpretation_allowed,
            plan.future_process_scenario_id,
        )
        _gate(
            rows, f"{plan.rolling_plan_id}:scenario quality",
            scenario is not None
            and _quality_pass(scenario.quality_flag),
            (
                "missing" if scenario is None
                else f"quality_flag={scenario.quality_flag}"
            ),
        )
        effective_end = _plan_end_step(plan, specs)
        plan_kcp_ids = _kcp_ids_for_set(pkg, plan.kcp_set_id)
        _gate(
            rows, f"{plan.rolling_plan_id}:KCP set",
            bool(plan_kcp_ids) and len(plan_kcp_ids) == len(set(plan_kcp_ids)),
            f"kcp_set_id={plan.kcp_set_id}, kcp_ids={plan_kcp_ids}",
        )
        plan_kcp_configs = kcp_configs[
            kcp_configs.get(
                "rolling_plan_id", pd.Series(dtype=str)
            ).astype(str).eq(plan.rolling_plan_id)
        ].copy()
        _gate(
            rows, f"{plan.rolling_plan_id}:KCP config",
            not plan_kcp_configs.empty
            and plan_kcp_configs.get(
                "kcp_set_id", pd.Series(dtype=str)
            ).astype(str).eq(plan.kcp_set_id).all()
            and plan_kcp_configs.get(
                "aggregation_policy", pd.Series(dtype=str)
            ).astype(str).eq(plan.aggregation_policy).all()
            and plan_kcp_configs.get(
                "quality_flag", pd.Series(dtype=str)
            ).map(_quality_pass).all(),
            f"rows={len(plan_kcp_configs)}",
        )
        direct_semantics, direct_semantic_errors = (
            _kcp_direct_semantics(plan_kcp_configs)
            if not plan_kcp_configs.empty
            else ({}, ["KCP config为空"])
        )
        _gate(
            rows,
            f"{plan.rolling_plan_id}:KCP direct geometry semantics",
            bool(direct_semantics) and not direct_semantic_errors,
            json.dumps(
                {
                    "by_part": direct_semantics,
                    "errors": direct_semantic_errors,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
        _gate(
            rows,
            f"{plan.rolling_plan_id}:KCP direct geometry aggregation compatibility",
            not plan_kcp_configs.empty
            and plan.aggregation_policy == SUPPORTED_AGGREGATION_POLICY
            and plan_kcp_configs.get(
                "aggregation_policy", pd.Series(dtype=str)
            ).astype(str).eq(SUPPORTED_AGGREGATION_POLICY).all()
            and plan_kcp_configs.get(
                "direct_sms_matrix_id_optional",
                pd.Series(dtype=str),
            ).map(lambda value: bool(_text(value))).all(),
            (
                f"policy={plan.aggregation_policy}, "
                f"semantics={direct_semantics}"
            ),
        )
        if not plan_kcp_configs.empty:
            duplicate_columns = [
                "rolling_plan_id", "part_id", "kcp_set_id",
                "direct_sms_matrix_id_optional", "aggregation_policy",
            ]
            duplicate_count = int(
                plan_kcp_configs[duplicate_columns]
                .astype(str).duplicated().sum()
            )
            _gate(
                rows, f"{plan.rolling_plan_id}:KCP contribution uniqueness",
                duplicate_count == 0,
                f"duplicate_count={duplicate_count}",
                # A duplicate is evaluated by the authoritative contribution
                # ledger so the sample/run/CLI runtime status is preserved.
                blocking=False,
            )
        for part_id in plan.future_part_ids:
            part_components = sorted(
                [
                    item for item in components
                    if item.part_id == part_id
                    and item.sms_layout_id == sample_set.sms_layout_id
                ],
                key=lambda item: item.component_order,
            )
            component_orders = [item.component_order for item in part_components]
            _gate(
                rows, f"{plan.rolling_plan_id}:{part_id}:component layout",
                component_orders == list(range(len(part_components)))
                and bool(part_components),
                str(component_orders),
            )
            _gate(
                rows, f"{plan.rolling_plan_id}:{part_id}:component quality",
                bool(part_components)
                and all(
                    _quality_pass(item.quality_flag)
                    for item in part_components
                ),
                f"rows={len(part_components)}",
            )
            first_add_candidates = [
                spec for spec in specs if part_id in spec.added_part_ids
            ]
            first_add_step = (
                min(first_add_candidates, key=lambda item: item.step_order)
                .topology_step_id
                if first_add_candidates else ""
            )
            _gate(
                rows, f"{plan.rolling_plan_id}:{part_id}:first add step",
                bool(first_add_step)
                and first_add_step == plan.prediction_start_step_id,
                (
                    f"first_add={first_add_step}, "
                    f"rolling_start={plan.prediction_start_step_id}"
                ),
            )
            for sample_id in sample_ids:
                sample_part = [
                    item for item in sample_rows
                    if item.virtual_sms_sample_id == sample_id
                    and item.part_id == part_id
                ]
                coeff = coefficients[
                    coefficients.get(
                        "virtual_sms_sample_id", pd.Series(dtype=str)
                    ).astype(str).eq(sample_id)
                    & coefficients.get(
                        "part_id", pd.Series(dtype=str)
                    ).astype(str).eq(part_id)
                ]
                coeff_order = (
                    pd.to_numeric(
                        coeff.get("component_order", pd.Series(dtype=float)),
                        errors="coerce",
                    ).astype("Int64").tolist()
                )
                component_by_order = {
                    item.component_order: item for item in part_components
                }
                coeff_units_ok = all(
                    _text(row.get("unit"))
                    == component_by_order.get(
                        int(float(row.get("component_order", -1)))
                    ).unit
                    for _, row in coeff.iterrows()
                    if int(float(row.get("component_order", -1)))
                    in component_by_order
                )
                coeff_reference_ok = (
                    not coeff.empty
                    and coeff.get(
                        "reference_sms_id", pd.Series(dtype=str)
                    ).astype(str).eq(sample_set.reference_sms_id).all()
                )
                coeff_quality_ok = (
                    not coeff.empty
                    and coeff.get(
                        "quality_flag", pd.Series(dtype=str)
                    ).map(_quality_pass).all()
                )
                sample_reference_ok = (
                    len(sample_part) == 1
                    and sample_part[0].reference_sms_id
                    == sample_set.reference_sms_id
                    and sample_part[0].sms_layout_id
                    == sample_set.sms_layout_id
                    and all(
                        item.reference_state_id
                        == sample_set.reference_sms_id
                        for item in part_components
                    )
                )
                _gate(
                    rows,
                    f"{plan.rolling_plan_id}:{sample_id}:{part_id}:coefficients",
                    len(sample_part) == 1
                    and coeff_order == component_orders
                    and np.isfinite(
                        pd.to_numeric(
                            coeff.get("value", pd.Series(dtype=float)),
                            errors="coerce",
                        )
                    ).all(),
                    f"orders={coeff_order}",
                )
                _gate(
                    rows,
                    f"{plan.rolling_plan_id}:{sample_id}:{part_id}:"
                    "coefficient unit/reference/quality",
                    coeff_units_ok
                    and coeff_reference_ok
                    and coeff_quality_ok
                    and sample_reference_ok,
                    (
                        f"unit={coeff_units_ok}, reference="
                        f"{coeff_reference_ok and sample_reference_ok}, "
                        f"quality={coeff_quality_ok}"
                    ),
                )
                assignment = [
                    item for item in assignments
                    if item.rolling_plan_id == plan.rolling_plan_id
                    and item.virtual_sms_sample_id == sample_id
                    and item.part_id == part_id
                ]
                assignment_item = (
                    assignment[0] if len(assignment) == 1 else None
                )
                last_step = (
                    assignment_item.last_effective_topology_step_id_optional
                    if assignment_item is not None else ""
                )
                assignment_interval_ok = (
                    assignment_item is not None
                    and assignment_item.first_effective_topology_step_id
                    in order
                    and (not last_step or last_step in order)
                    and order.get(
                        assignment_item.first_effective_topology_step_id,
                        np.inf,
                    ) <= order.get(last_step or effective_end, -np.inf)
                    and order.get(
                        assignment_item.first_effective_topology_step_id,
                        -np.inf,
                    ) >= order.get(plan.prediction_start_step_id, np.inf)
                    and order.get(last_step or effective_end, np.inf)
                    <= order.get(effective_end, -np.inf)
                    and assignment_item.first_effective_topology_step_id
                    == first_add_step
                    and (not last_step or last_step == effective_end)
                )
                _gate(
                    rows,
                    f"{plan.rolling_plan_id}:{sample_id}:{part_id}:assignment",
                    assignment_item is not None
                    and assignment_item.mapping_semantics
                    == DELTA_FROM_OPERATOR_REFERENCE
                    and assignment_item.reference_sms_id
                    == sample_set.reference_sms_id
                    and _quality_pass(assignment_item.quality_flag)
                    and assignment_interval_ok,
                    (
                        f"matches={len(assignment)}, first="
                        f"{getattr(assignment_item, 'first_effective_topology_step_id', '')}, "
                        f"last={last_step}, expected_first={first_add_step}, "
                        f"expected_end={effective_end}"
                    ),
                )
            future_solve_specs = [
                spec for spec in specs
                if spec.solve_required
                and order[spec.topology_step_id]
                >= order.get(first_add_step, np.inf)
                and order[spec.topology_step_id]
                <= order.get(effective_end, -np.inf)
            ]
            for spec in future_solve_specs:
                matching = [
                    item for item in mappings
                    if item.rolling_plan_id == plan.rolling_plan_id
                    and item.part_id == part_id
                    and item.operator_set_id == str(spec.operator_set_id or "")
                ]
                mapping_ok = len(matching) == 1
                detail = f"step={spec.topology_step_id}, matches={len(matching)}"
                if mapping_ok:
                    item = matching[0]
                    try:
                        matrix, manifest = _matrix_and_manifest(
                            pkg, item.matrix_id
                        )
                        mapping_ok = (
                            matrix.shape == (dimension, len(part_components))
                            and item.row_vector_layout_id == row_layout_id
                            and item.column_sms_layout_id
                            == sample_set.sms_layout_id
                            and item.mapping_semantics
                            == DELTA_FROM_OPERATOR_REFERENCE
                            and _text(
                                manifest.get("row_layout_id_optional")
                            ) == row_layout_id
                            and _text(
                                manifest.get("column_layout_id_optional")
                            ) == sample_set.sms_layout_id
                            and item.reference_sms_id
                            == sample_set.reference_sms_id
                            and _text(
                                manifest.get("reference_sms_id_optional")
                            ) == sample_set.reference_sms_id
                            and _text(
                                manifest.get("mapping_semantics_optional")
                            ) == DELTA_FROM_OPERATOR_REFERENCE
                            and item.mapping_role
                            in SUPPORTED_MAPPING_ROLES
                            and _text(
                                manifest.get("mapping_role_optional")
                            ) == item.mapping_role
                            and _quality_pass(item.quality_flag)
                            and _quality_pass(manifest.get("quality_flag"))
                            and bool(item.derivation_source)
                            and bool(_text(
                                manifest.get("derivation_source")
                            ))
                            and _step_in_range(
                                spec.topology_step_id,
                                item.first_valid_step_id,
                                item.last_valid_step_id_optional,
                                order,
                            )
                        )
                        norm = float(np.linalg.norm(matrix))
                        role_ok = (
                            (
                                item.mapping_role == EFFECTIVE_MAPPING
                                and norm > 1e-14
                            )
                            or (
                                item.mapping_role
                                == EXPLICIT_ZERO_NO_EFFECT
                                and norm <= 1e-14
                            )
                        )
                        mapping_ok = mapping_ok and role_ok
                        detail += f", shape={matrix.shape}"
                        detail += (
                            f", role={item.mapping_role}, norm={norm:.3e}"
                        )
                    except Exception as exc:
                        mapping_ok = False
                        detail += f", error={exc}"
                _gate(
                    rows,
                    f"{plan.rolling_plan_id}:{part_id}:{spec.topology_step_id}:G_SMS",
                    mapping_ok,
                    detail,
                )
            part_kcp_configs = plan_kcp_configs[
                plan_kcp_configs.get(
                    "part_id", pd.Series(dtype=str)
                ).astype(str).eq(part_id)
            ]
            direct_ok = not part_kcp_configs.empty
            direct_detail: list[str] = []
            for _, config in part_kcp_configs.iterrows():
                matrix_id = _text(
                    config.get("direct_sms_matrix_id_optional")
                )
                try:
                    direct_matrix, direct_manifest = _matrix_and_manifest(
                        pkg, matrix_id
                    )
                    direct_ok = direct_ok and (
                        direct_matrix.shape
                        == (len(plan_kcp_ids), len(part_components))
                        and _text(
                            direct_manifest.get(
                                "row_layout_id_optional"
                            )
                        ) == plan.kcp_set_id
                        and _text(
                            direct_manifest.get(
                                "column_layout_id_optional"
                            )
                        ) == sample_set.sms_layout_id
                        and _text(
                            direct_manifest.get(
                                "mapping_role_optional"
                            )
                        ) == DIRECT_KCP_MAPPING_ROLE
                        and _text(
                            direct_manifest.get(
                                "reference_sms_id_optional"
                            )
                        ) == sample_set.reference_sms_id
                        and _quality_pass(
                            direct_manifest.get("quality_flag")
                        )
                    )
                    direct_detail.append(
                        f"{matrix_id}:{direct_matrix.shape}"
                    )
                except Exception as exc:
                    direct_ok = False
                    direct_detail.append(f"{matrix_id}:{exc}")
            _gate(
                rows, f"{plan.rolling_plan_id}:{part_id}:KCP direct mapping",
                direct_ok,
                "; ".join(direct_detail),
            )

        if topology_result is not None:
            source = topology_result.get(plan.source_topology_step_id, {})
            posterior = source.get("posterior_stage_state")
            _gate(
                rows, f"{plan.rolling_plan_id}:runtime posterior",
                source_linkage["source_linkage_status"] == "PASS"
                and posterior is not None
                and bool(getattr(posterior, "posterior_accepted", False))
                and str(getattr(posterior, "state_role", "")) == "POSTERIOR"
                and str(getattr(posterior, "quality_flag", "")) == "PASS",
                json.dumps(
                    source_linkage, ensure_ascii=False, sort_keys=True
                ),
            )
    return pd.DataFrame(rows)


def _canonical(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            "shape": list(contiguous.shape),
            "dtype": str(contiguous.dtype),
            "bytes_sha256": hashlib.sha256(
                contiguous.tobytes()
            ).hexdigest(),
        }
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
        return {
            "columns": list(frame.columns),
            "dtypes": [str(frame[column].dtype) for column in frame.columns],
            "records": [
                [_canonical(item) for item in row]
                for row in frame.itertuples(index=False, name=None)
            ],
        }
    if isinstance(value, pd.Series):
        return _canonical(value.to_dict())
    if is_dataclass(value):
        return _canonical(asdict(value))
    if isinstance(value, dict):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if not callable(item)
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        _canonical(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def package_input_hash(pkg: SMSPackage) -> str:
    return stable_hash({
        "tables": {
            key: value
            for key, value in pkg.raw_tables.items()
            if not key.replace("\\", "/").startswith("validation/")
        },
        "matrices": pkg.matrices,
        "manifest": pkg.manifest,
    })


_PACKAGE_HASH_EXCLUDED = {
    "validation/test_results.json",
    "validation/TEST_RESULTS.md",
    "validation/run_log.csv",
    "validation/quality_gate.csv",
    "validation/validation_result.csv",
}


def stable_package_file_hash(root: Path) -> str:
    """Hash immutable package files by relative path, length and raw bytes.

    Runtime/validation attachments that contain the hash or regenerated status
    are excluded to avoid self-reference.  The algorithm never parses CSV/JSON,
    so Python, pandas and locale differences cannot alter the digest.
    """
    root = Path(root).resolve()
    digest = hashlib.sha256()
    paths = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        parts = set(path.relative_to(root).parts)
        if (
            relative in _PACKAGE_HASH_EXCLUDED
            or ".git" in parts
            or "__pycache__" in parts
            or path.suffix.lower() in {".pyc", ".tmp", ".log"}
        ):
            continue
        paths.append((relative, path))
    for relative, path in sorted(paths):
        payload = path.read_bytes()
        file_digest = hashlib.sha256(payload).hexdigest()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(payload)).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _coefficient_vectors(
    pkg: SMSPackage,
    plan: RollingPredictionPlan,
    sample_id: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    samples = load_virtual_sms_samples(pkg)
    components = load_virtual_sms_components(pkg)
    coefficients = _table(pkg, COEFFICIENT_TABLE)
    values: dict[str, np.ndarray] = {}
    references: dict[str, np.ndarray] = {}
    deltas: dict[str, np.ndarray] = {}
    for part_id in plan.future_part_ids:
        sample_meta = [
            item for item in samples
            if item.virtual_sms_sample_id == sample_id
            and item.part_id == part_id
        ]
        if len(sample_meta) != 1:
            raise RollingPredictionValidationError(
                f"sample/part记录不唯一: {sample_id}/{part_id}"
            )
        layout_id = sample_meta[0].sms_layout_id
        order = [
            item.component_order for item in sorted(
                [
                    item for item in components
                    if item.sms_layout_id == layout_id
                    and item.part_id == part_id
                ],
                key=lambda item: item.component_order,
            )
        ]
        if order != list(range(len(order))) or not order:
            raise RollingPredictionValidationError(
                f"SMS component order不连续: {part_id}/{order}"
            )

        def vector_for(target_sample: str) -> np.ndarray:
            rows = coefficients[
                coefficients.get(
                    "virtual_sms_sample_id", pd.Series(dtype=str)
                ).astype(str).eq(target_sample)
                & coefficients.get(
                    "part_id", pd.Series(dtype=str)
                ).astype(str).eq(part_id)
            ].copy()
            rows["_order"] = pd.to_numeric(
                rows.get("component_order"), errors="coerce"
            )
            rows = rows.sort_values("_order", kind="stable")
            if rows["_order"].astype("Int64").tolist() != order:
                raise RollingPredictionValidationError(
                    f"coefficient order不匹配: {target_sample}/{part_id}"
                )
            vector = pd.to_numeric(rows["value"], errors="coerce").to_numpy(float)
            if vector.shape != (len(order),) or not np.isfinite(vector).all():
                raise RollingPredictionValidationError(
                    f"coefficient vector无效: {target_sample}/{part_id}"
                )
            return vector

        reference_candidates = sorted(
            {
                item.virtual_sms_sample_id for item in samples
                if item.sample_set_id == sample_meta[0].sample_set_id
                and item.part_id == part_id
                and item.coefficient_source == "REFERENCE_SMS"
            }
        )
        if len(reference_candidates) != 1:
            raise RollingPredictionValidationError(
                f"reference sample不唯一: {part_id}/{reference_candidates}"
            )
        values[part_id] = vector_for(sample_id)
        references[part_id] = vector_for(reference_candidates[0])
        deltas[part_id] = values[part_id] - references[part_id]
    return values, references, deltas


def _step_in_range(
    step_id: str,
    first: str,
    last: str,
    order: dict[str, int],
) -> bool:
    return (
        step_id in order
        and first in order
        and order[step_id] >= order[first]
        and (not last or (last in order and order[step_id] <= order[last]))
    )


def _rolling_contexts(
    pkg: SMSPackage,
    plan: RollingPredictionPlan,
    sample_id: str,
    rolling_run_id: str,
    specs: list[TopologyStepSpec],
    deltas: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    order = {spec.topology_step_id: spec.step_order for spec in specs}
    assignments = load_future_sms_assignments(pkg)
    mappings = load_sms_operator_mappings(pkg)
    scenarios = {
        item.future_process_scenario_id: item
        for item in load_future_process_scenarios(pkg)
    }
    scenario = scenarios[plan.future_process_scenario_id]
    dimension = len(pkg.contact_points)
    contexts: dict[str, dict[str, Any]] = {}
    for spec in specs:
        if not spec.solve_required:
            continue
        if not _step_in_range(
            spec.topology_step_id,
            plan.prediction_start_step_id,
            plan.prediction_end_step_id_optional,
            order,
        ):
            continue
        by_part: dict[str, np.ndarray] = {}
        traces: list[dict[str, Any]] = []
        for part_id, delta in deltas.items():
            assignment = [
                item for item in assignments
                if item.rolling_plan_id == plan.rolling_plan_id
                and item.virtual_sms_sample_id == sample_id
                and item.part_id == part_id
            ]
            if len(assignment) != 1:
                raise RollingPredictionValidationError(
                    f"assignment不唯一: {sample_id}/{part_id}"
                )
            assignment_item = assignment[0]
            if not _step_in_range(
                spec.topology_step_id,
                assignment_item.first_effective_topology_step_id,
                assignment_item.last_effective_topology_step_id_optional,
                order,
            ):
                continue
            mapping = [
                item for item in mappings
                if item.rolling_plan_id == plan.rolling_plan_id
                and item.part_id == part_id
                and item.operator_set_id == str(spec.operator_set_id or "")
                and _step_in_range(
                    spec.topology_step_id,
                    item.first_valid_step_id,
                    item.last_valid_step_id_optional,
                    order,
                )
            ]
            if len(mapping) != 1:
                raise RollingPredictionValidationError(
                    f"G_SMS mapping不唯一: {part_id}/{spec.topology_step_id}"
                )
            mapping_item = mapping[0]
            matrix, manifest = _matrix_and_manifest(pkg, mapping_item.matrix_id)
            if matrix.shape != (dimension, delta.size):
                raise RollingPredictionValidationError(
                    f"G_SMS shape={matrix.shape}, 期望={(dimension, delta.size)}"
                )
            correction = matrix @ delta
            by_part[part_id] = correction
            traces.append({
                "rolling_run_id": rolling_run_id,
                "virtual_sms_sample_id": sample_id,
                "part_id": part_id,
                "topology_step_id": spec.topology_step_id,
                "operator_set_id": str(spec.operator_set_id or ""),
                "matrix_id": mapping_item.matrix_id,
                "reference_sms_id": mapping_item.reference_sms_id,
                "delta_alpha": delta.tolist(),
                "q_sms_correction": correction.tolist(),
                "q_sms_correction_norm": float(np.linalg.norm(correction)),
                "application_count": 1,
                "double_count_status": "PASS",
                "mapping_role": mapping_item.mapping_role,
                "declared_zero_effect": (
                    mapping_item.mapping_role
                    == EXPLICIT_ZERO_NO_EFFECT
                ),
                "matrix_norm": float(np.linalg.norm(matrix)),
                "manifest_mapping_role": _text(
                    manifest.get("mapping_role_optional")
                ),
            })
        process = np.zeros(dimension, dtype=float)
        if scenario.q_correction_matrix_id_optional:
            matrix, _ = _matrix_and_manifest(
                pkg, scenario.q_correction_matrix_id_optional
            )
            process = np.asarray(matrix, dtype=float).reshape(-1)
            if process.shape != (dimension,):
                raise RollingPredictionValidationError(
                    "future process correction维度不匹配"
                )
        contexts[spec.topology_step_id] = {
            "rolling_run_id": rolling_run_id,
            "virtual_sms_sample_id": sample_id,
            "q_virtual_sms_correction_by_part": by_part,
            "q_future_process_correction": process,
            "sms_application_trace": traces,
        }
    return contexts


def _expected_application_keys(
    plan: RollingPredictionPlan,
    sample_id: str,
    specs: list[TopologyStepSpec],
    assignments: list[FutureSMSAssignment],
) -> set[tuple[str, str, str]]:
    order = {spec.topology_step_id: spec.step_order for spec in specs}
    end_step = _plan_end_step(plan, specs)
    expected: set[tuple[str, str, str]] = set()
    for part_id in plan.future_part_ids:
        matches = [
            item for item in assignments
            if item.rolling_plan_id == plan.rolling_plan_id
            and item.virtual_sms_sample_id == sample_id
            and item.part_id == part_id
        ]
        if len(matches) != 1:
            raise RollingPredictionValidationError(
                f"assignment不唯一: {sample_id}/{part_id}"
            )
        first = matches[0].first_effective_topology_step_id
        last = matches[0].last_effective_topology_step_id_optional or end_step
        for spec in specs:
            if (
                spec.solve_required
                and _step_in_range(
                    spec.topology_step_id, first, last, order
                )
                and _step_in_range(
                    spec.topology_step_id,
                    plan.prediction_start_step_id,
                    end_step,
                    order,
                )
            ):
                expected.add((
                    sample_id, part_id, spec.topology_step_id
                ))
    return expected


def _check_application_completeness(
    plan: RollingPredictionPlan,
    sample_id: str,
    specs: list[TopologyStepSpec],
    step_results: dict[str, dict[str, Any]],
    assignments: list[FutureSMSAssignment],
) -> tuple[bool, dict[str, Any]]:
    expected = _expected_application_keys(
        plan, sample_id, specs, assignments
    )
    traces = [
        trace
        for item in step_results.values()
        for trace in item.get("contact_trace", {}).get(
            "sms_application_trace", []
        )
    ]
    counts: dict[tuple[str, str, str], int] = {}
    invalid_count_rows = []
    role_mismatches = []
    for trace in traces:
        key = (
            _text(trace.get("virtual_sms_sample_id")),
            _text(trace.get("part_id")),
            _text(trace.get("topology_step_id")),
        )
        count = int(trace.get("application_count", 0))
        counts[key] = counts.get(key, 0) + count
        if count != 1:
            invalid_count_rows.append({
                "key": key, "application_count": count,
            })
        role = _text(trace.get("mapping_role"))
        manifest_role = _text(trace.get("manifest_mapping_role"))
        matrix_norm = float(trace.get("matrix_norm", np.inf))
        role_ok = (
            role == manifest_role
            and (
                (role == EFFECTIVE_MAPPING and matrix_norm > 1e-14)
                or (
                    role == EXPLICIT_ZERO_NO_EFFECT
                    and matrix_norm <= 1e-14
                    and float(
                        trace.get("q_sms_correction_norm", np.inf)
                    ) <= 1e-14
                )
            )
        )
        if not role_ok:
            role_mismatches.append({
                "key": key,
                "mapping_role": role,
                "manifest_mapping_role": manifest_role,
                "matrix_norm": matrix_norm,
            })
    actual = set(counts)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    duplicates = sorted(
        key for key, count in counts.items() if count != 1
    )
    passed = not (
        missing
        or unexpected
        or duplicates
        or invalid_count_rows
        or role_mismatches
    )
    return passed, {
        "status": "PASS" if passed else "FAIL",
        "expected_count": len(expected),
        "actual_count": len(traces),
        "expected_keys": sorted(expected),
        "actual_counts": [
            {"key": key, "count": count}
            for key, count in sorted(counts.items())
        ],
        "missing": missing,
        "unexpected": unexpected,
        "duplicate_or_invalid": duplicates,
        "invalid_count_rows": invalid_count_rows,
        "mapping_role_mismatches": role_mismatches,
    }


def _cross_block_norms(
    pkg: SMSPackage, step_result: dict[str, Any]
) -> dict[str, float]:
    layout = vector_layout(pkg)
    W = np.asarray(step_result.get("W_struct"), dtype=float)
    active = set(step_result.get("active_interface_ids", []))
    rows = [
        row for _, row in layout.iterrows()
        if str(row.get("object_id", "")) in active
    ]
    out: dict[str, float] = {}
    for left_index, left in enumerate(rows):
        li = slice(int(left["start_index"]), int(left["end_index"]) + 1)
        for right in rows[left_index + 1:]:
            ri = slice(int(right["start_index"]), int(right["end_index"]) + 1)
            key = f"{left.get('object_id')}|{right.get('object_id')}"
            out[key] = float(np.linalg.norm(W[li, ri]))
    return out


def _kcp_with_ledger(
    pkg: SMSPackage,
    plan: RollingPredictionPlan,
    sample_id: str,
    source_role: str,
    result: dict[str, dict[str, Any]],
    deltas: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    kcp = extract_kcp(pkg, result).copy().reset_index(drop=True)
    kcp_ids = _kcp_ids_for_set(pkg, plan.kcp_set_id)
    if not kcp_ids:
        raise RollingPredictionRuntimeError(
            f"KCP set无有效定义: {plan.kcp_set_id}", "KCP"
        )
    kcp = (
        kcp[kcp["kcp_id"].astype(str).isin(kcp_ids)]
        .assign(
            _kcp_order=lambda frame: frame["kcp_id"].astype(str).map(
                {item: index for index, item in enumerate(kcp_ids)}
            )
        )
        .sort_values("_kcp_order", kind="stable")
        .drop(columns="_kcp_order")
        .reset_index(drop=True)
    )
    if kcp["kcp_id"].astype(str).tolist() != kcp_ids:
        raise RollingPredictionRuntimeError(
            f"KCP set提取不完整: expected={kcp_ids}, "
            f"actual={kcp['kcp_id'].astype(str).tolist()}",
            "KCP",
        )
    if kcp.empty or not np.isfinite(
        pd.to_numeric(kcp["predicted_value"], errors="coerce")
    ).all():
        raise RollingPredictionRuntimeError(
            "正式KCP入口未返回有限结果", "KCP"
        )
    base = pd.to_numeric(kcp["predicted_value"], errors="coerce").to_numpy(float)
    direct_candidate_total = np.zeros(len(kcp), dtype=float)
    direct_applied_total = np.zeros(len(kcp), dtype=float)
    ledger_rows: list[dict[str, Any]] = [{
        "sample_id": sample_id,
        "source_state_role": source_role,
        "source_class": "FINAL_ABSOLUTE_CONTACT_STATE",
        "source_id": result[list(result)[-1]]["stage_state"].stage_state_id,
        "origin_stage_id_optional": list(result)[-1],
        "increment_definition_id": "FORMAL_EXTRACT_KCP_CONTACT_AND_BASE_TERMS",
        "target_kcp_id": ";".join(kcp["kcp_id"].astype(str)),
        "contribution_vector": base.tolist(),
    }]
    configs = _table(pkg, KCP_CONFIG_TABLE)
    configs = configs[
        configs.get("rolling_plan_id", pd.Series(dtype=str))
        .astype(str).eq(plan.rolling_plan_id)
        & configs.get("kcp_set_id", pd.Series(dtype=str))
        .astype(str).eq(plan.kcp_set_id)
        & configs.get("aggregation_policy", pd.Series(dtype=str))
        .astype(str).eq(plan.aggregation_policy)
    ]
    if configs.empty:
        raise RollingPredictionRuntimeError(
            f"KCP config为空: {plan.rolling_plan_id}/{plan.kcp_set_id}",
            "KCP",
        )
    semantics_by_part, semantic_errors = _kcp_direct_semantics(configs)
    if semantic_errors:
        raise RollingPredictionValidationError(
            "KCP direct geometry语义无效: "
            + "; ".join(semantic_errors)
        )
    direct_actions: list[dict[str, Any]] = []
    for _, row in configs.iterrows():
        part_id = _text(row.get("part_id"))
        matrix_id = _text(row.get("direct_sms_matrix_id_optional"))
        if not matrix_id:
            continue
        if part_id not in deltas:
            raise RollingPredictionValidationError(
                f"KCP direct mapping引用非future part: {part_id}"
            )
        matrix, _ = _matrix_and_manifest(pkg, matrix_id)
        if matrix.shape != (len(kcp), deltas[part_id].size):
            raise RollingPredictionValidationError(
                f"KCP SMS matrix shape={matrix.shape}"
            )
        contribution = matrix @ deltas[part_id]
        direct_candidate_total += contribution
        included = semantics_by_part[part_id]
        action = (
            DIRECT_SMS_ALREADY_INCLUDED if included else DIRECT_SMS_ADD
        )
        direct_actions.append({
            "part_id": part_id,
            "matrix_id": matrix_id,
            "final_state_includes_direct_sms_geometry": included,
            "aggregation_action": action,
            "candidate_contribution": contribution.tolist(),
            "applied_contribution": (
                np.zeros_like(contribution).tolist()
                if included else contribution.tolist()
            ),
        })
        if not included:
            direct_applied_total += contribution
            ledger_rows.append({
                "sample_id": sample_id,
                "source_state_role": source_role,
                "source_class": "FUTURE_SMS_DIRECT_GEOMETRY",
                "source_id": part_id,
                "origin_stage_id_optional": plan.prediction_start_step_id,
                "increment_definition_id": "DELTA_FROM_OPERATOR_REFERENCE",
                "target_kcp_id": ";".join(kcp["kcp_id"].astype(str)),
                "contribution_vector": contribution.tolist(),
            })
    semantic_values = set(semantics_by_part.values())
    final_state_includes = (
        next(iter(semantic_values)) if len(semantic_values) == 1 else None
    )
    kcp["virtual_sms_direct_contribution_candidate"] = (
        direct_candidate_total
    )
    kcp["virtual_sms_direct_contribution"] = direct_applied_total
    kcp["predicted_value"] = base + direct_applied_total
    kcp["source_state_role"] = source_role
    kcp["virtual_sms_sample_id"] = sample_id
    kcp["aggregation_policy"] = plan.aggregation_policy
    kcp["final_state_includes_direct_sms_geometry"] = (
        final_state_includes
    )
    kcp["direct_sms_aggregation_action"] = (
        DIRECT_SMS_ALREADY_INCLUDED
        if final_state_includes is True
        else DIRECT_SMS_ADD
        if final_state_includes is False
        else "MIXED_BY_PART"
    )
    kcp["double_count_status"] = "PASS"
    values = pd.to_numeric(kcp["predicted_value"], errors="coerce")
    lower = pd.to_numeric(kcp.get("lower_tol"), errors="coerce")
    upper = pd.to_numeric(kcp.get("upper_tol"), errors="coerce")
    below = lower.notna() & values.lt(lower)
    above = upper.notna() & values.gt(upper)
    kcp["tolerance_status"] = np.where(
        below | above, "EXCEED", "WITHIN_OR_NOT_DEFINED"
    )
    kcp["quality_status"] = "PASS"
    ledger = pd.DataFrame(
        ledger_rows, columns=CONTRIBUTION_LEDGER_COLUMNS
    )
    keys = ledger[[
        "sample_id", "source_state_role", "source_class", "source_id",
        "origin_stage_id_optional", "increment_definition_id",
        "target_kcp_id",
    ]].astype(str).agg("|".join, axis=1)
    config_duplicate_columns = [
        "rolling_plan_id",
        "part_id",
        "kcp_set_id",
        "direct_sms_matrix_id_optional",
        "aggregation_policy",
        "final_state_includes_direct_sms_geometry",
    ]
    config_duplicate_count = int(
        configs[config_duplicate_columns].astype(str).duplicated().sum()
    )
    ledger_duplicate_count = int(keys.duplicated().sum())
    duplicate_count = ledger_duplicate_count + config_duplicate_count
    reconstructed = np.sum(
        np.asarray(ledger["contribution_vector"].tolist(), dtype=float),
        axis=0,
    )
    reconstruction_error = float(np.max(
        np.abs(reconstructed - values.to_numpy(float))
    ))
    status = (
        "PASS"
        if duplicate_count == 0 and reconstruction_error <= 1e-12
        else "FAIL"
    )
    kcp["double_count_status"] = status
    kcp["quality_status"] = status
    kcp["sample_quality_status"] = status
    return kcp, ledger, {
        "aggregation_policy": plan.aggregation_policy,
        "kcp_set_id": plan.kcp_set_id,
        "kcp_config_count": len(configs),
        "duplicate_count": duplicate_count,
        "ledger_duplicate_count": ledger_duplicate_count,
        "config_duplicate_count": config_duplicate_count,
        "reconstruction_error": reconstruction_error,
        "double_count_status": status,
        "kcp_quality_status": status,
        "double_count_fail_count": int(status == "FAIL"),
        "direct_sms_applied": bool(
            np.any(np.abs(direct_applied_total) > 0.0)
        ),
        "direct_sms_candidate_contribution":
            direct_candidate_total.tolist(),
        "direct_sms_applied_contribution":
            direct_applied_total.tolist(),
        "direct_sms_actions": direct_actions,
        "final_state_includes_direct_sms_geometry":
            final_state_includes,
        "final_state_includes_direct_sms_geometry_by_part":
            semantics_by_part,
        "direct_sms_aggregation_action": (
            DIRECT_SMS_ALREADY_INCLUDED
            if final_state_includes is True
            else DIRECT_SMS_ADD
            if final_state_includes is False
            else "MIXED_BY_PART"
        ),
    }


def _run_sample_branch(
    pkg: SMSPackage,
    topology_result: dict[str, dict[str, Any]],
    specs: list[TopologyStepSpec],
    plan: RollingPredictionPlan,
    rolling_run_id: str,
    sample_id: str,
    source_state: StageState,
    source_role: str,
    source_predicted_state_id: str,
    source_posterior_state_id: str,
) -> RollingSampleResult:
    values, references, deltas = _coefficient_vectors(
        pkg, plan, sample_id
    )
    contexts = _rolling_contexts(
        pkg, plan, sample_id, rolling_run_id, specs, deltas
    )
    source_result = topology_result[plan.source_topology_step_id]
    branch_sample_id = (
        f"{rolling_run_id}__{sample_id}__{source_role}"
    )
    step_results = run_topology_steps_from_state(
        pkg,
        specs,
        source_state,
        source_result,
        plan.prediction_start_step_id,
        prediction_end_step_id=(
            plan.prediction_end_step_id_optional or None
        ),
        sample_id=branch_sample_id,
        rolling_context_by_step=contexts,
    )
    final_step_id = list(step_results)[-1]
    final_state: StageState = step_results[final_step_id]["stage_state"]
    physical_ok = all(
        str(item.get("solve_status", "")).upper()
        in {"CONVERGED", "NOT_REQUIRED"}
        and int(item.get("lcp_call_count", 0))
        == (1 if item["topology_step_spec"].solve_required else 0)
        and float(
            item["solution"].residuals.get(
                "complementarity_residual", np.inf
            )
        ) <= 1e-8
        for item in step_results.values()
    )
    if not physical_ok:
        raise RollingPredictionRuntimeError(
            "rolling physical/LCP quality gate failed", "PHYSICAL"
        )
    sms_trace = [
        trace
        for item in step_results.values()
        for trace in item["contact_trace"].get(
            "sms_application_trace", []
        )
    ]
    application_ok, application_trace = _check_application_completeness(
        plan,
        sample_id,
        specs,
        step_results,
        load_future_sms_assignments(pkg),
    )
    if not application_ok:
        raise RollingPredictionRuntimeError(
            "virtual SMS application完整性失败: "
            + json.dumps(
                {
                    "missing": application_trace["missing"],
                    "unexpected": application_trace["unexpected"],
                    "duplicate_or_invalid":
                        application_trace["duplicate_or_invalid"],
                    "mapping_role_mismatches":
                        application_trace["mapping_role_mismatches"],
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "APPLICATION",
        )
    kcp, ledger, ledger_trace = _kcp_with_ledger(
        pkg, plan, sample_id, source_role, step_results, deltas
    )
    double_count_fail_count = int(
        ledger_trace["double_count_status"] == "FAIL"
    )
    kcp_fail_count = int(
        ledger_trace["kcp_quality_status"] == "FAIL"
    )
    sample_status = (
        "PASS"
        if double_count_fail_count == 0 and kcp_fail_count == 0
        else "FAIL"
    )
    sample_failure_reason = ""
    if sample_status == "FAIL":
        sample_failure_reason = (
            "KCP contribution double-count/quality failure: "
            f"duplicate_count={ledger_trace['duplicate_count']}, "
            "reconstruction_error="
            f"{ledger_trace['reconstruction_error']:.3e}"
        )
    contact_signature = ";".join(
        str(index) for index in final_state.active_set
    )
    trace = {
        "source_state_role": source_role,
        "source_state_id": source_state.stage_state_id,
        "sample_state_branch_ids": [
            item["stage_state"].stage_state_id
            for item in step_results.values()
        ],
        "q_decomposition": [
            {
                "topology_step_id": step_id,
                "operator_set_id": item.get("operator_set_id", ""),
                "q_operator_base_norm": item["contact_trace"].get(
                    "q_operator_base_norm", 0.0
                ),
                "posterior_correction_norm": item["contact_trace"].get(
                    "q_posterior_state_correction_norm", 0.0
                ),
                "virtual_sms_correction_norm": item["contact_trace"].get(
                    "q_virtual_sms_correction_norm", 0.0
                ),
                "process_correction_norm": item["contact_trace"].get(
                    "q_future_process_correction_norm", 0.0
                ),
                "q_effective_norm": item["contact_trace"].get(
                    "q_effective_norm", 0.0
                ),
                "lcp_call_count": item.get("lcp_call_count", 0),
                "active_interface_ids": item.get(
                    "active_interface_ids", []
                ),
                "cross_block_norms": _cross_block_norms(pkg, item),
            }
            for step_id, item in step_results.items()
        ],
        "sms_application_trace": sms_trace,
        "application_completeness": application_trace,
        "ledger": ledger_trace,
        "probability_interpretation_allowed": False,
        "engineering_claim_allowed": False,
    }
    return RollingSampleResult(
        rolling_run_id=rolling_run_id,
        rolling_plan_id=plan.rolling_plan_id,
        virtual_sms_sample_id=sample_id,
        source_checkpoint_id=plan.source_checkpoint_id,
        source_predicted_state_id=source_predicted_state_id,
        source_posterior_state_id=source_posterior_state_id,
        effective_source_state_id=source_state.stage_state_id,
        source_state_role=source_role,
        prediction_start_step_id=plan.prediction_start_step_id,
        prediction_end_step_id=final_step_id,
        future_part_ids=plan.future_part_ids,
        sms_coefficients={
            key: value.tolist() for key, value in values.items()
        },
        sms_reference_coefficients={
            key: value.tolist() for key, value in references.items()
        },
        sms_delta_coefficients={
            key: value.tolist() for key, value in deltas.items()
        },
        future_process_scenario_id=plan.future_process_scenario_id,
        step_results=step_results,
        final_state_id=final_state.stage_state_id,
        kcp_prediction_result=kcp,
        contribution_ledger=ledger,
        contact_mode_signature=contact_signature,
        quality_status=sample_status,
        failure_reason=sample_failure_reason,
        trace=trace,
        double_count_fail_count=double_count_fail_count,
        application_fail_count=0,
        kcp_fail_count=kcp_fail_count,
        authoritative_failure_reason=sample_failure_reason,
    )


def _failed_sample(
    plan: RollingPredictionPlan,
    rolling_run_id: str,
    sample_id: str,
    source_predicted_state_id: str,
    source_posterior_state_id: str,
    source_role: str,
    exc: Exception,
) -> RollingSampleResult:
    failure_type = _text(getattr(exc, "failure_type", "RUNTIME")).upper()
    reason = f"{type(exc).__name__}: {exc}"
    return RollingSampleResult(
        rolling_run_id=rolling_run_id,
        rolling_plan_id=plan.rolling_plan_id,
        virtual_sms_sample_id=sample_id,
        source_checkpoint_id=plan.source_checkpoint_id,
        source_predicted_state_id=source_predicted_state_id,
        source_posterior_state_id=source_posterior_state_id,
        effective_source_state_id="",
        source_state_role=source_role,
        prediction_start_step_id=plan.prediction_start_step_id,
        prediction_end_step_id=plan.prediction_end_step_id_optional,
        future_part_ids=plan.future_part_ids,
        sms_coefficients={},
        sms_reference_coefficients={},
        sms_delta_coefficients={},
        future_process_scenario_id=plan.future_process_scenario_id,
        step_results={},
        final_state_id="",
        kcp_prediction_result=pd.DataFrame(
            columns=ROLLING_KCP_COLUMNS
        ),
        contribution_ledger=pd.DataFrame(
            columns=CONTRIBUTION_LEDGER_COLUMNS
        ),
        contact_mode_signature="",
        quality_status="FAIL",
        failure_reason=reason,
        trace={
            "failure_isolated": True,
            "failure_type": failure_type,
            "source_state_role": source_role,
        },
        double_count_fail_count=int(failure_type == "DOUBLE_COUNT"),
        application_fail_count=int(failure_type == "APPLICATION"),
        kcp_fail_count=int(failure_type == "KCP"),
        authoritative_failure_reason=reason,
    )


def _summarize(
    rolling_run_id: str,
    plan: RollingPredictionPlan,
    requested_sample_ids: list[str],
    results: list[RollingSampleResult],
    failures: list[RollingSampleResult],
    baseline_results: list[RollingSampleResult],
    baseline_failures: list[RollingSampleResult],
    quality_status: str,
) -> RollingPredictionSummary:
    frames = [
        item.kcp_prediction_result.assign(
            virtual_sms_sample_id=item.virtual_sms_sample_id
        )
        for item in results
        if item.quality_status == "PASS"
    ]
    predictions = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    if not predictions.empty:
        for kcp_id, group in predictions.groupby("kcp_id", sort=False):
            values = pd.to_numeric(group["predicted_value"], errors="coerce")
            exceed = group["tolerance_status"].astype(str).eq("EXCEED")
            rows.append({
                "kcp_id": kcp_id,
                "count": int(values.count()),
                "descriptive_mean": float(values.mean()),
                "descriptive_std": (
                    float(values.std(ddof=1)) if values.count() > 1 else 0.0
                ),
                "empirical_min": float(values.min()),
                "empirical_max": float(values.max()),
                "empirical_p05": float(values.quantile(0.05)),
                "empirical_p50": float(values.quantile(0.50)),
                "empirical_p95": float(values.quantile(0.95)),
                "tolerance_exceedance_count": int(exceed.sum()),
                "tolerance_exceedance_fraction": float(exceed.mean()),
                "fraction_semantics": EMPIRICAL_FRACTION_LABEL,
                "probability_interpretation_allowed": False,
                "engineering_claim_allowed": False,
            })
    modes = pd.Series(
        [item.contact_mode_signature for item in results],
        dtype=str,
    ).value_counts(dropna=False)
    mode_frame = pd.DataFrame({
        "contact_mode_signature": modes.index.astype(str),
        "sample_count": modes.to_numpy(int),
    })
    return RollingPredictionSummary(
        rolling_run_id=rolling_run_id,
        rolling_plan_id=plan.rolling_plan_id,
        sample_count=len(set(requested_sample_ids)),
        success_count=len(results),
        failure_count=len(failures),
        baseline_attempt_count=(
            len(baseline_results) + len(baseline_failures)
        ),
        baseline_success_count=len(baseline_results),
        baseline_failure_count=len(baseline_failures),
        baseline_quality_status=(
            "NOT_APPLICABLE"
            if plan.baseline_comparison_policy == BASELINE_POSTERIOR_ONLY
            else "PASS" if not baseline_failures else "FAIL"
        ),
        kcp_summary=pd.DataFrame(
            rows, columns=ROLLING_KCP_SUMMARY_COLUMNS
        ),
        contact_mode_counts=mode_frame,
        probability_interpretation_allowed=False,
        engineering_claim_allowed=False,
        quality_status=quality_status,
    )


def _source_states(
    pkg: SMSPackage,
    topology_result: dict[str, dict[str, Any]],
    plan: RollingPredictionPlan,
) -> tuple[StageState, StageState]:
    linkage = _source_linkage_evidence(pkg, plan, topology_result)
    if linkage["source_linkage_status"] != "PASS":
        raise RollingPredictionValidationError(
            "rolling source linkage未闭合: "
            + json.dumps(
                linkage, ensure_ascii=False, sort_keys=True
            )
        )
    source = topology_result.get(plan.source_topology_step_id)
    if source is None:
        raise RollingPredictionValidationError(
            f"source topology step不存在: {plan.source_topology_step_id}"
        )
    posterior = source.get("posterior_stage_state")
    predicted = source.get("predicted_stage_state")
    if posterior is None or predicted is None:
        raise RollingPredictionValidationError(
            "source step未同时保留PREDICTED/POSTERIOR状态"
        )
    if (
        not posterior.posterior_accepted
        or posterior.state_role != "POSTERIOR"
        or posterior.quality_flag != "PASS"
        or posterior.source_checkpoint_id != plan.source_checkpoint_id
    ):
        raise RollingPredictionValidationError(
            "source posterior未通过accepted/role/quality/checkpoint门"
        )
    if (
        plan.source_posterior_state_id_optional
        and posterior.stage_state_id
        != plan.source_posterior_state_id_optional
    ):
        raise RollingPredictionValidationError(
            "source posterior state ID与计划不一致"
        )
    unexpected = set(plan.future_part_ids) & set(posterior.active_part_ids)
    if unexpected:
        raise RollingPredictionValidationError(
            f"future parts已在source subassembly中: {sorted(unexpected)}"
        )
    return posterior, predicted


def _authoritative_rolling_status(
    plan: RollingPredictionPlan,
    requested_sample_ids: list[str],
    results: list[RollingSampleResult],
    failures: list[RollingSampleResult],
    baseline_results: list[RollingSampleResult],
    baseline_failures: list[RollingSampleResult],
    *,
    immutable: bool,
) -> dict[str, Any]:
    requested_count = len(set(requested_sample_ids))
    baseline_required = (
        plan.baseline_comparison_policy != BASELINE_POSTERIOR_ONLY
    )
    double_count_fail_count = sum(
        item.double_count_fail_count
        for item in [*results, *failures]
    )
    application_fail_count = sum(
        item.application_fail_count
        for item in [*results, *failures]
    )
    kcp_fail_count = sum(
        item.kcp_fail_count
        for item in [*results, *failures]
    )
    formal_failed = (
        bool(failures)
        or len(results) != requested_count
        or any(item.quality_status != "PASS" for item in results)
    )
    baseline_failed = (
        baseline_required
        and (
            bool(baseline_failures)
            or len(baseline_results) != requested_count
            or any(
                item.quality_status != "PASS"
                for item in baseline_results
            )
        )
    )
    reasons: list[str] = []
    if formal_failed:
        reasons.append(
            "one or more formal POSTERIOR samples failed"
        )
    if baseline_failed:
        reasons.append(
            "required PREDICTED cutoff baseline is incomplete or failed"
        )
    if double_count_fail_count:
        reasons.append(
            f"KCP double-count failures={double_count_fail_count}"
        )
    if application_fail_count:
        reasons.append(
            f"SMS application failures={application_fail_count}"
        )
    if kcp_fail_count:
        reasons.append(f"KCP quality failures={kcp_fail_count}")
    if not immutable:
        reasons.append("topology/package immutability failure")
    run_status = (
        "PASS"
        if not formal_failed and not baseline_failed and immutable
        else "FAIL"
    )
    return {
        "run_quality_status": run_status,
        "authoritative_failure_reason": "; ".join(reasons),
        "formal_sample_count": requested_count,
        "formal_success_count": len(results),
        "formal_failure_count": len(failures),
        "baseline_attempt_count": (
            len(baseline_results) + len(baseline_failures)
        ),
        "baseline_success_count": len(baseline_results),
        "baseline_failure_count": len(baseline_failures),
        "baseline_quality_status": (
            "NOT_APPLICABLE"
            if not baseline_required
            else "PASS" if not baseline_failed else "FAIL"
        ),
        "double_count_fail_count": double_count_fail_count,
        "application_fail_count": application_fail_count,
        "kcp_fail_count": kcp_fail_count,
        "immutability_fail_count": int(not immutable),
        "immutability_status": "PASS" if immutable else "FAIL",
        "policy_validation_status": "PASS",
        "probability_interpretation_allowed": False,
        "engineering_claim_allowed": False,
    }


def run_posterior_virtual_sms_rolling_prediction(
    package: SMSPackage,
    topology_result: dict[str, dict[str, Any]],
    rolling_plan_id: str,
    virtual_sms_sample_ids: Iterable[str] | None = None,
    include_predicted_cutoff_baseline: bool = True,
) -> RollingPredictionRunResult:
    plans = {
        plan.rolling_plan_id: plan
        for plan in load_rolling_prediction_plans(package)
        if plan.active_flag
    }
    if rolling_plan_id not in plans:
        raise RollingPredictionValidationError(
            f"rolling plan不存在或未启用: {rolling_plan_id}"
        )
    plan = plans[rolling_plan_id]
    checks = validate_rolling_prediction_package(package, topology_result)
    blocking = checks[
        checks["status"].eq("FAIL") & checks["blocking"].astype(bool)
    ]
    if not blocking.empty:
        raise RollingPredictionValidationError(
            blocking[["check_item", "detail"]].to_string(index=False)
        )
    posterior, predicted = _source_states(package, topology_result, plan)
    source_linkage = _source_linkage_evidence(
        package, plan, topology_result
    )
    specs = load_topology_steps(package, plan.topology_id)
    rolling_run_id = f"ROLL__{plan.rolling_plan_id}"
    samples = [
        item for item in load_virtual_sms_samples(package)
        if item.sample_set_id == plan.virtual_sms_sample_set_id
        and item.part_id in plan.future_part_ids
    ]
    sample_order = {
        item.virtual_sms_sample_id: item.sample_order for item in samples
    }
    available = sorted(set(sample_order), key=lambda key: (sample_order[key], key))
    requested = list(virtual_sms_sample_ids) if virtual_sms_sample_ids is not None else available
    if len(requested) != len(set(requested)) or not set(requested) <= set(available):
        raise RollingPredictionValidationError(
            f"virtual SMS sample选择无效: {requested}"
        )
    baseline_required = (
        plan.baseline_comparison_policy != BASELINE_POSTERIOR_ONLY
    )
    if baseline_required and not include_predicted_cutoff_baseline:
        raise RollingPredictionValidationError(
            "baseline policy要求PREDICTED cutoff对照，不能由调用参数关闭"
        )
    run_baseline = baseline_required

    topology_hashes_before = {
        key: stable_hash(value)
        for key, value in sorted(topology_result.items())
    }
    source_hashes_before = {
        "posterior": stable_hash(posterior),
        "predicted": stable_hash(predicted),
        "complete_topology_result": stable_hash(topology_result),
        "historical_topology": stable_hash(topology_result),
        "topology_step_hashes": topology_hashes_before,
        "package": package_input_hash(package),
    }
    results: list[RollingSampleResult] = []
    failures: list[RollingSampleResult] = []
    baselines: list[RollingSampleResult] = []
    baseline_failures: list[RollingSampleResult] = []
    for sample_id in requested:
        try:
            result = _run_sample_branch(
                package, topology_result, specs, plan, rolling_run_id,
                sample_id, deepcopy(posterior), "POSTERIOR",
                predicted.stage_state_id, posterior.stage_state_id,
            )
            if result.quality_status == "PASS":
                result.trace["source_linkage"] = deepcopy(source_linkage)
                results.append(result)
            else:
                result.trace["source_linkage"] = deepcopy(source_linkage)
                failures.append(result)
        except Exception as exc:
            failures.append(_failed_sample(
                plan, rolling_run_id, sample_id,
                predicted.stage_state_id, posterior.stage_state_id,
                "POSTERIOR", exc,
            ))
        if run_baseline:
            try:
                baseline = _run_sample_branch(
                    package, topology_result, specs, plan, rolling_run_id,
                    sample_id, deepcopy(predicted), "PREDICTED",
                    predicted.stage_state_id, posterior.stage_state_id,
                )
                if baseline.quality_status == "PASS":
                    baseline.trace["source_linkage"] = deepcopy(
                        source_linkage
                    )
                    baselines.append(baseline)
                else:
                    baseline.trace["source_linkage"] = deepcopy(
                        source_linkage
                    )
                    baseline_failures.append(baseline)
            except Exception as exc:
                baseline_failures.append(_failed_sample(
                    plan, rolling_run_id, sample_id,
                    predicted.stage_state_id, posterior.stage_state_id,
                    "PREDICTED", exc,
                ))

    topology_hashes_after = {
        key: stable_hash(value)
        for key, value in sorted(topology_result.items())
    }
    source_hashes_after = {
        "posterior": stable_hash(posterior),
        "predicted": stable_hash(predicted),
        "complete_topology_result": stable_hash(topology_result),
        "historical_topology": stable_hash(topology_result),
        "topology_step_hashes": topology_hashes_after,
        "package": package_input_hash(package),
    }
    immutable = source_hashes_before == source_hashes_after
    changed_objects = sorted(
        {
            key
            for key in set(topology_hashes_before)
            | set(topology_hashes_after)
            if topology_hashes_before.get(key)
            != topology_hashes_after.get(key)
        }
    )
    for key in ("posterior", "predicted", "package"):
        if source_hashes_before[key] != source_hashes_after[key]:
            changed_objects.append(key)
    changed_objects = sorted(set(changed_objects))
    posterior_frames = [
        item.kcp_prediction_result for item in results
    ]
    kcp_predictions = (
        pd.concat(posterior_frames, ignore_index=True)
        if posterior_frames else pd.DataFrame(
            columns=ROLLING_KCP_COLUMNS
        )
    )
    comparison_rows: list[dict[str, Any]] = []
    formal_by_sample = {
        item.virtual_sms_sample_id: item
        for item in [*results, *failures]
    }
    baseline_by_sample = {
        item.virtual_sms_sample_id: item
        for item in [*baselines, *baseline_failures]
    }
    if run_baseline:
        for sample_id in requested:
            posterior_result = formal_by_sample.get(sample_id)
            baseline = baseline_by_sample.get(sample_id)
            kcp_ids = _kcp_ids_for_set(package, plan.kcp_set_id)
            for kcp_id in kcp_ids:
                posterior_row = (
                    posterior_result.kcp_prediction_result[
                        posterior_result.kcp_prediction_result[
                            "kcp_id"
                        ].astype(str).eq(kcp_id)
                    ]
                    if posterior_result is not None
                    and not posterior_result.kcp_prediction_result.empty
                    else pd.DataFrame()
                )
                baseline_row = (
                    baseline.kcp_prediction_result[
                        baseline.kcp_prediction_result[
                            "kcp_id"
                        ].astype(str).eq(kcp_id)
                    ]
                    if baseline is not None
                    and not baseline.kcp_prediction_result.empty
                    else pd.DataFrame()
                )
                posterior_ok = (
                    posterior_result is not None
                    and posterior_result.quality_status == "PASS"
                    and not posterior_row.empty
                )
                baseline_ok = (
                    baseline is not None
                    and baseline.quality_status == "PASS"
                    and not baseline_row.empty
                )
                posterior_value = (
                    float(posterior_row.iloc[0]["predicted_value"])
                    if posterior_ok else np.nan
                )
                predicted_value = (
                    float(baseline_row.iloc[0]["predicted_value"])
                    if baseline_ok else np.nan
                )
                comparison_status = (
                    "PASS"
                    if posterior_ok and baseline_ok
                    else "POSTERIOR_FAILED"
                    if not posterior_ok and baseline_ok
                    else "PREDICTED_FAILED"
                    if posterior_ok and not baseline_ok
                    else "BOTH_FAILED"
                )
                comparison_rows.append({
                    "virtual_sms_sample_id": sample_id,
                    "kcp_id": kcp_id,
                    "predicted_cutoff_result": predicted_value,
                    "posterior_cutoff_result": posterior_value,
                    "posterior_minus_predicted":
                        (
                            posterior_value - predicted_value
                            if posterior_ok and baseline_ok else np.nan
                        ),
                    "source_predicted_state_id": predicted.stage_state_id,
                    "source_posterior_state_id": posterior.stage_state_id,
                    "posterior_quality_status": (
                        posterior_result.quality_status
                        if posterior_result is not None else "FAIL"
                    ),
                    "baseline_quality_status": (
                        baseline.quality_status
                        if baseline is not None else "NOT_RUN"
                    ),
                    "comparison_status": comparison_status,
                    "missing_side": (
                        "" if comparison_status == "PASS"
                        else comparison_status
                    ),
                    "posterior_failure_reason": (
                        posterior_result.failure_reason
                        if posterior_result is not None else "missing"
                    ),
                    "baseline_failure_reason": (
                        baseline.failure_reason
                        if baseline is not None else "not run"
                    ),
                    "quality_status": comparison_status,
                })
    comparison = pd.DataFrame(
        comparison_rows, columns=BASELINE_COMPARISON_COLUMNS
    )
    status_summary = _authoritative_rolling_status(
        plan,
        requested,
        results,
        failures,
        baselines,
        baseline_failures,
        immutable=immutable,
    )
    plan_kcp_configs = _table(package, KCP_CONFIG_TABLE)
    plan_kcp_configs = plan_kcp_configs[
        plan_kcp_configs.get(
            "rolling_plan_id", pd.Series(dtype=str)
        ).astype(str).eq(plan.rolling_plan_id)
        & plan_kcp_configs.get(
            "kcp_set_id", pd.Series(dtype=str)
        ).astype(str).eq(plan.kcp_set_id)
    ]
    direct_semantics, _ = _kcp_direct_semantics(plan_kcp_configs)
    direct_values = set(direct_semantics.values())
    direct_included = (
        next(iter(direct_values)) if len(direct_values) == 1 else None
    )
    direct_action = (
        DIRECT_SMS_ALREADY_INCLUDED
        if direct_included is True
        else DIRECT_SMS_ADD
        if direct_included is False
        else "MIXED_BY_PART"
    )
    status_summary.update({
        "source_linkage_status": source_linkage[
            "source_linkage_status"
        ],
        "final_state_includes_direct_sms_geometry": direct_included,
        "final_state_includes_direct_sms_geometry_by_part":
            direct_semantics,
        "direct_sms_aggregation_action": direct_action,
    })
    run_quality = status_summary["run_quality_status"]
    summary = _summarize(
        rolling_run_id,
        plan,
        requested,
        results,
        failures,
        baselines,
        baseline_failures,
        run_quality,
    )
    run_gates = pd.concat([
        checks,
        pd.DataFrame([
            {
                "check_item": "source/package不可变",
                "status": "PASS" if immutable else "FAIL",
                "blocking": True,
                "detail": json.dumps({
                    "before": source_hashes_before,
                    "after": source_hashes_after,
                    "changed_objects": changed_objects,
                }, ensure_ascii=False, sort_keys=True),
            },
            {
                "check_item": "formal samples全部成功",
                "status": (
                    "PASS"
                    if status_summary["formal_failure_count"] == 0
                    and status_summary["formal_success_count"]
                    == status_summary["formal_sample_count"]
                    else "FAIL"
                ),
                "blocking": True,
                "detail": (
                    f"requested={status_summary['formal_sample_count']}, "
                    f"success={status_summary['formal_success_count']}, "
                    f"failure={status_summary['formal_failure_count']}"
                ),
            },
            {
                "check_item": "predicted baseline policy",
                "status": status_summary["baseline_quality_status"],
                "blocking": True,
                "detail": (
                    f"policy={plan.baseline_comparison_policy}, "
                    f"attempt={status_summary['baseline_attempt_count']}, "
                    f"success={status_summary['baseline_success_count']}, "
                    f"failure={status_summary['baseline_failure_count']}"
                ),
            },
            {
                "check_item": "SMS application completeness",
                "status": (
                    "PASS"
                    if status_summary["application_fail_count"] == 0
                    else "FAIL"
                ),
                "blocking": True,
                "detail": (
                    "failure_count="
                    f"{status_summary['application_fail_count']}"
                ),
            },
            {
                "check_item": "KCP contribution double-count",
                "status": (
                    "PASS"
                    if status_summary["double_count_fail_count"] == 0
                    else "FAIL"
                ),
                "blocking": True,
                "detail": (
                    "failure_count="
                    f"{status_summary['double_count_fail_count']}"
                ),
            },
            {
                "check_item": "authoritative rolling status",
                "status": run_quality,
                "blocking": True,
                "detail": (
                    status_summary["authoritative_failure_reason"]
                    or "all formal and required baseline branches passed"
                ),
            },
        ]),
    ], ignore_index=True)
    trace = {
        "rolling_run_id": rolling_run_id,
        "rolling_plan_id": plan.rolling_plan_id,
        "sample_order": requested,
        "source_state_hashes_before": source_hashes_before,
        "source_state_hashes_after": source_hashes_after,
        "immutability_status": "PASS" if immutable else "FAIL",
        "immutability_changed_objects": changed_objects,
        "sample_branch_ids": {
            item.virtual_sms_sample_id: item.trace.get(
                "sample_state_branch_ids", []
            )
            for item in results
        },
        "failure_isolation": [
            {
                "virtual_sms_sample_id": item.virtual_sms_sample_id,
                "reason": item.failure_reason,
            }
            for item in failures
        ],
        "baseline_failure_isolation": [
            {
                "virtual_sms_sample_id": item.virtual_sms_sample_id,
                "source_state_role": item.source_state_role,
                "reason": item.failure_reason,
            }
            for item in baseline_failures
        ],
        "source_linkage": source_linkage,
        "plan_topology_id": source_linkage["plan_topology_id"],
        "checkpoint_topology_id":
            source_linkage["checkpoint_topology_id"],
        "plan_source_step_id": source_linkage["plan_source_step_id"],
        "checkpoint_topology_step_id":
            source_linkage["checkpoint_topology_step_id"],
        "measurement_update_id":
            source_linkage["measurement_update_id"],
        "measurement_update_posterior_state_id":
            source_linkage["measurement_update_posterior_state_id"],
        "actual_source_state_id":
            source_linkage["actual_source_state_id"],
        "source_linkage_status":
            source_linkage["source_linkage_status"],
        "final_state_includes_direct_sms_geometry": direct_included,
        "final_state_includes_direct_sms_geometry_by_part":
            direct_semantics,
        "direct_sms_aggregation_action": direct_action,
        "authoritative_status": status_summary,
        "authoritative_failure_reason":
            status_summary["authoritative_failure_reason"],
        "probability_interpretation_allowed": False,
        "engineering_claim_allowed": False,
        "quality_status": run_quality,
    }
    return RollingPredictionRunResult(
        rolling_run_id=rolling_run_id,
        plan=plan,
        source_posterior_state=posterior,
        source_predicted_state=predicted,
        sample_results=results,
        sample_failures=failures,
        predicted_baseline_results=baselines,
        predicted_baseline_failures=baseline_failures,
        kcp_predictions=kcp_predictions,
        baseline_comparison=comparison,
        descriptive_summary=summary,
        contact_mode_summary=summary.contact_mode_counts,
        quality_gates=run_gates,
        trace=trace,
        quality_status=run_quality,
        authoritative_failure_reason=
            status_summary["authoritative_failure_reason"],
        status_summary=status_summary,
    )
