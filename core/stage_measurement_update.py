from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from .data_loader import SMSPackage
from .lcp_solver import LCPSolution, solve_lcp_active_set
from .multi_part import vector_layout
from .stage_state import StageState


ALLOWED_UPDATE_DATA_ROLES = frozenset({"CALIBRATE", "UPDATE"})
EVALUATION_ONLY_DATA_ROLES = frozenset({"VALIDATE"})
EXCLUDED_UPDATE_DATA_ROLES = frozenset({"IDENTIFY", "MONITOR", "EXCLUDE"})
ALLOWED_STATE_UPDATE_TARGETS = frozenset({
    "STAGE_STATE",
    "POSE",
    "POSE_BIAS",
    "GAP_BIAS",
    "ACTUAL_LOAD_CORRECTION",
    "PROCESS_LOAD",
    "LOCATOR_ZERO_BIAS",
    "OVERCONSTRAINT_STATE",
})
FROZEN_PARAMETER_TARGETS = frozenset({
    "SMS",
    "CN",
    "CT",
    "MU",
    "BETA_R",
    "JOINT_STIFFNESS",
    "INTERFACE_PARAMETER",
    "REBOUND_PARAMETER",
    "LOCATOR_COMPLIANCE",
})
SUPPORTED_OBSERVED_QUANTITIES = frozenset({
    "GAP_G", "LAMBDA_N", "PRESSURE_P_N", "LOCAL_COMPRESSION_W_N",
})
SUPPORTED_VECTOR_SOURCES = frozenset({"RUNTIME_STAGE_STATE"})
OBSERVATION_UNITS = {
    "GAP_G": {"MM"},
    "LAMBDA_N": {"N"},
    "PRESSURE_P_N": {"MPA", "N/MM^2", "N/MM2"},
    "LOCAL_COMPRESSION_W_N": {"MM"},
}


class StageMeasurementUpdateValidationError(ValueError):
    """Raised for a blocking measurement-update package or governance error."""


class StageMeasurementUpdateRuntimeError(RuntimeError):
    """Raised when the posterior numerical or physical update cannot be accepted."""


def _array(value: Any, *, ndim: int = 1) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return np.empty((0, 0), dtype=float) if ndim == 2 else np.array([], dtype=float)
    if ndim == 2 and array.ndim == 1:
        return array.reshape(1, -1)
    return array.copy()


def _vector_record(value: Any) -> list[float]:
    return _array(value).reshape(-1).tolist()


def _matrix_record(value: Any) -> list[list[float]]:
    matrix = _array(value, ndim=2)
    return matrix.tolist()


@dataclass(frozen=True)
class MeasurementCheckpointSpec:
    checkpoint_id: str
    topology_id: str
    topology_step_id: str
    source_topology_step_id: str
    measurement_set_id: str
    update_config_id: str
    reference_state_id: str
    missing_measurement_policy: str
    rollback_policy: str
    active_flag: bool
    notes: str = ""

    def to_record(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class StateUpdateBasisItem:
    update_state_layout_id: str
    component_order: int
    component_id: str
    component_type: str
    target_object_type: str
    target_object_id: str
    unit: str
    description: str = ""

    def to_record(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class MeasurementObservationSpec:
    observation_id: str
    checkpoint_id: str
    measurement_id: str
    observation_order: int
    observed_quantity: str
    vector_source: str
    global_index_optional: int | None
    target_object_id_optional: str | None
    unit: str
    coordinate_system_id: str
    reference_state_id: str
    sensitivity_row_index: int
    quality_flag: str

    def to_record(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class MeasurementUpdateConfig:
    update_config_id: str
    update_state_layout_id: str
    prior_mean_matrix_id: str
    prior_covariance_matrix_id: str
    observation_jacobian_matrix_id: str
    measurement_covariance_matrix_id_optional: str | None
    state_to_q_mapping_rule: str
    algorithm: str
    regularization: float
    nis_threshold: float
    covariance_floor: float
    resolve_policy: str
    parameter_update_allowed: bool
    quality_flag: str
    reference_state_id: str = ""
    allow_diagonal_covariance_fallback: bool = False
    physical_residual_threshold: float = np.inf
    individual_degradation_tolerance: float = 1.0

    def to_record(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class UpdateDecisionRecord:
    update_id: str
    checkpoint_id: str
    measurement_ids: list[str]
    accepted_measurement_ids: list[str]
    rejected_measurement_ids: list[str]
    evaluation_measurement_ids: list[str]
    skipped_measurement_ids: list[str]
    blocked_measurement_ids: list[str]
    data_roles: list[str]
    update_targets: list[str]
    allowed_state_components: list[str]
    frozen_parameter_ids: list[str]
    decision: str
    decision_reason: str
    identifiability_summary: str
    quality_flag: str

    def to_record(self) -> dict[str, Any]:
        record = dict(self.__dict__)
        for field_name in (
            "measurement_ids", "accepted_measurement_ids", "rejected_measurement_ids",
            "evaluation_measurement_ids", "skipped_measurement_ids",
            "blocked_measurement_ids",
            "data_roles", "update_targets", "allowed_state_components",
            "frozen_parameter_ids",
        ):
            record[field_name] = ";".join(getattr(self, field_name))
        return record


@dataclass(frozen=True)
class ReSolveRequirement:
    level: str
    changed_objects: tuple[str, ...]
    source_operator_set_id: str
    reason: str
    required_matrix_keys: tuple[str, ...]
    quality_flag: str

    def to_record(self) -> dict[str, Any]:
        record = dict(self.__dict__)
        record["changed_objects"] = ";".join(self.changed_objects)
        record["required_matrix_keys"] = ";".join(self.required_matrix_keys)
        return record


@dataclass
class UpdateRollbackRecord:
    rollback_record_id: str
    checkpoint_id: str
    predicted_state_id: str
    attempted_posterior_state_id: str
    failure_stage: str
    failure_reason: str
    retained_effective_state_id: str
    measurement_ids: list[str]
    quality_flag: str

    def to_record(self) -> dict[str, Any]:
        record = dict(self.__dict__)
        record["measurement_ids"] = ";".join(self.measurement_ids)
        return record


@dataclass
class StageMeasurementUpdateResult:
    update_id: str
    checkpoint_id: str
    topology_step_id: str
    source_topology_step_id: str
    predicted_state_id: str
    posterior_state_id: str
    effective_state_id: str
    measurement_ids: list[str]
    z_measured: np.ndarray
    z_predicted_prior: np.ndarray
    z_predicted_posterior: np.ndarray
    innovation: np.ndarray
    normalized_innovation: np.ndarray
    eta_prior: np.ndarray
    eta_posterior: np.ndarray
    P_prior: np.ndarray
    P_posterior: np.ndarray
    kalman_gain: np.ndarray
    nis: float
    q_correction_prior: np.ndarray
    q_correction_posterior: np.ndarray
    resolve_requirement: ReSolveRequirement
    resolve_lcp_call_count: int
    physical_residuals: dict[str, float]
    posterior_accepted: bool
    rollback_record: UpdateRollbackRecord | None
    quality_flag: str
    trace: dict[str, Any] = field(default_factory=dict)
    decision_record: UpdateDecisionRecord | None = None
    posterior_state: Any | None = None
    measurement_source: str = "PACKAGE"
    observation_records: list[dict[str, Any]] = field(default_factory=list)
    q_posterior: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    W_total: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))
    solution: LCPSolution | None = None
    lambda_full: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    gap_full: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    pressure: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    local_compression: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    active_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    evaluation_measurement_ids: list[str] = field(default_factory=list)
    skipped_measurement_ids: list[str] = field(default_factory=list)
    z_predicted_prior_physical: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )
    z_predicted_posterior_linearized: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )
    z_predicted_posterior_physical: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )
    residual_prior_physical: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )
    residual_posterior_linearized: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )
    residual_posterior_physical: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )
    standardized_residual_prior_physical: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )
    standardized_residual_posterior_physical: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )
    weighted_residual_prior_physical: float = np.nan
    weighted_residual_posterior_physical: float = np.nan
    linearization_error: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )
    physical_residual_improved: bool = False
    measurement_covariance_source: str = ""
    q_operator_base: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )
    q_source_effective: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )
    eta_increment: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )
    q_update_increment: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )
    q_effective_after_update: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )
    q_base_semantics: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "update_id": self.update_id,
            "checkpoint_id": self.checkpoint_id,
            "topology_step_id": self.topology_step_id,
            "source_topology_step_id": self.source_topology_step_id,
            "predicted_state_id": self.predicted_state_id,
            "posterior_state_id": self.posterior_state_id,
            "effective_state_id": self.effective_state_id,
            "measurement_ids": ";".join(self.measurement_ids),
            "z_measured": _vector_record(self.z_measured),
            "z_predicted_prior": _vector_record(
                self.z_predicted_prior_physical
                if self.z_predicted_prior_physical.size
                else self.z_predicted_prior
            ),
            "z_predicted_posterior": _vector_record(
                self.z_predicted_posterior_physical
                if self.z_predicted_posterior_physical.size
                else self.z_predicted_posterior
            ),
            "z_predicted_prior_physical": _vector_record(
                self.z_predicted_prior_physical
            ),
            "z_predicted_posterior_linearized": _vector_record(
                self.z_predicted_posterior_linearized
            ),
            "z_predicted_posterior_physical": _vector_record(
                self.z_predicted_posterior_physical
            ),
            "residual_prior_physical": _vector_record(
                self.residual_prior_physical
            ),
            "residual_posterior_linearized": _vector_record(
                self.residual_posterior_linearized
            ),
            "residual_posterior_physical": _vector_record(
                self.residual_posterior_physical
            ),
            "standardized_residual_prior_physical": _vector_record(
                self.standardized_residual_prior_physical
            ),
            "standardized_residual_posterior_physical": _vector_record(
                self.standardized_residual_posterior_physical
            ),
            "weighted_residual_prior_physical": float(
                self.weighted_residual_prior_physical
            ),
            "weighted_residual_posterior_physical": float(
                self.weighted_residual_posterior_physical
            ),
            "linearization_error": _vector_record(self.linearization_error),
            "physical_residual_improved": bool(
                self.physical_residual_improved
            ),
            "measurement_covariance_source": (
                self.measurement_covariance_source
            ),
            "innovation": _vector_record(self.innovation),
            "normalized_innovation": _vector_record(self.normalized_innovation),
            "eta_prior": _vector_record(self.eta_prior),
            "eta_posterior": _vector_record(self.eta_posterior),
            "P_prior": _matrix_record(self.P_prior),
            "P_posterior": _matrix_record(self.P_posterior),
            "kalman_gain": _matrix_record(self.kalman_gain),
            "nis": float(self.nis),
            "q_correction_prior": _vector_record(self.q_correction_prior),
            "q_correction_posterior": _vector_record(self.q_correction_posterior),
            "q_operator_base": _vector_record(self.q_operator_base),
            "q_source_effective": _vector_record(self.q_source_effective),
            "eta_increment": _vector_record(self.eta_increment),
            "q_update_increment": _vector_record(self.q_update_increment),
            "q_effective_after_update": _vector_record(
                self.q_effective_after_update
            ),
            "q_base_semantics": self.q_base_semantics,
            "resolve_level": self.resolve_requirement.level,
            "resolve_lcp_call_count": int(self.resolve_lcp_call_count),
            "physical_residuals": dict(self.physical_residuals),
            "posterior_accepted": bool(self.posterior_accepted),
            "rollback_record_id": (
                self.rollback_record.rollback_record_id if self.rollback_record else ""
            ),
            "quality_flag": self.quality_flag,
            "measurement_source": self.measurement_source,
            "evaluation_measurement_ids": ";".join(
                self.evaluation_measurement_ids
            ),
            "skipped_measurement_ids": ";".join(
                self.skipped_measurement_ids
            ),
        }


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


def _bool(value: object, default: bool = False) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if _is_blank(value):
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "是"}


def _float(value: object, default: float = 0.0) -> float:
    if _is_blank(value):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise StageMeasurementUpdateValidationError(
            f"无法解析浮点值：{value!r}"
        ) from exc


def _int_optional(value: object) -> int | None:
    if _is_blank(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError) as exc:
        raise StageMeasurementUpdateValidationError(
            f"无法解析整数索引：{value!r}"
        ) from exc


def _first(row: pd.Series, *names: str, default: object = "") -> object:
    for name in names:
        if name in row.index and not _is_blank(row.get(name)):
            return row.get(name)
    return default


def _table(pkg: SMSPackage, *names: str) -> pd.DataFrame:
    for name in names:
        frame = pkg.raw_tables.get(name)
        if frame is not None and not frame.empty:
            return frame.copy()
    for name in names:
        if name in pkg.raw_tables:
            return pkg.raw_tables[name].copy()
    return pd.DataFrame()


def has_measurement_checkpoints(pkg: SMSPackage) -> bool:
    table = _table(pkg, "I_meas/measurement_checkpoint.csv")
    return not table.empty


def load_measurement_checkpoints(
    pkg: SMSPackage,
    topology_id: str | None = None,
) -> list[MeasurementCheckpointSpec]:
    table = _table(pkg, "I_meas/measurement_checkpoint.csv")
    specs: list[MeasurementCheckpointSpec] = []
    for _, row in table.iterrows():
        spec = MeasurementCheckpointSpec(
            checkpoint_id=_text(_first(row, "checkpoint_id", "measurement_checkpoint_id")),
            topology_id=_text(_first(row, "topology_id"), "TOPOLOGY_DEFAULT"),
            topology_step_id=_text(_first(row, "topology_step_id")),
            source_topology_step_id=_text(_first(row, "source_topology_step_id")),
            measurement_set_id=_text(_first(row, "measurement_set_id")),
            update_config_id=_text(_first(row, "update_config_id", "state_update_config_id")),
            reference_state_id=_text(_first(row, "reference_state_id")),
            missing_measurement_policy=_text(
                _first(row, "missing_measurement_policy"), "BLOCK"
            ).upper(),
            rollback_policy=_text(
                _first(row, "rollback_policy"), "ROLLBACK_TO_PREDICTED"
            ).upper(),
            active_flag=_bool(_first(row, "active_flag"), True),
            notes=_text(_first(row, "notes", "metadata")),
        )
        if topology_id is None or spec.topology_id == topology_id:
            specs.append(spec)
    return specs


def load_state_update_basis(
    pkg: SMSPackage,
    update_state_layout_id: str | None = None,
) -> list[StateUpdateBasisItem]:
    table = _table(pkg, "I_stage/state_update_basis.csv")
    items: list[StateUpdateBasisItem] = []
    for position, (_, row) in enumerate(table.iterrows()):
        layout_id = _text(_first(row, "update_state_layout_id"))
        if update_state_layout_id is not None and layout_id != update_state_layout_id:
            continue
        items.append(StateUpdateBasisItem(
            update_state_layout_id=layout_id,
            component_order=int(_float(_first(row, "component_order"), position)),
            component_id=_text(_first(row, "component_id")),
            component_type=_text(_first(row, "component_type")).upper(),
            target_object_type=_text(_first(row, "target_object_type")),
            target_object_id=_text(_first(row, "target_object_id")),
            unit=_text(_first(row, "unit")),
            description=_text(_first(row, "description", "metadata")),
        ))
    return sorted(items, key=lambda item: (item.component_order, item.component_id))


def load_measurement_observations(
    pkg: SMSPackage,
    checkpoint_id: str | None = None,
) -> list[MeasurementObservationSpec]:
    table = _table(pkg, "I_meas/measurement_observation_map.csv")
    observations: list[MeasurementObservationSpec] = []
    for position, (_, row) in enumerate(table.iterrows()):
        checkpoint = _text(_first(row, "checkpoint_id", "measurement_checkpoint_id"))
        if checkpoint_id is not None and checkpoint != checkpoint_id:
            continue
        observations.append(MeasurementObservationSpec(
            observation_id=_text(_first(row, "observation_id"), f"OBS_{position:04d}"),
            checkpoint_id=checkpoint,
            measurement_id=_text(_first(row, "measurement_id", "record_id")),
            observation_order=int(_float(_first(row, "observation_order"), position)),
            observed_quantity=_text(
                _first(row, "observed_quantity", "measurement_type", "feature_definition_id")
            ).upper(),
            vector_source=_text(_first(row, "vector_source")).upper(),
            global_index_optional=_int_optional(
                _first(row, "global_index_optional", "global_index")
            ),
            target_object_id_optional=(
                _text(_first(row, "target_object_id_optional", "target_object_id", "object_id"))
                or None
            ),
            unit=_text(_first(row, "unit")),
            coordinate_system_id=_text(_first(row, "coordinate_system_id")),
            reference_state_id=_text(_first(row, "reference_state_id")),
            sensitivity_row_index=int(
                _float(_first(row, "sensitivity_row_index"), position)
            ),
            quality_flag=_text(_first(row, "quality_flag"), "PASS").upper(),
        ))
    return sorted(
        observations,
        key=lambda item: (item.observation_order, item.observation_id),
    )


def load_measurement_update_configs(
    pkg: SMSPackage,
) -> dict[str, MeasurementUpdateConfig]:
    table = _table(pkg, "I_meas/measurement_update_config.csv")
    configs: dict[str, MeasurementUpdateConfig] = {}
    for _, row in table.iterrows():
        config = MeasurementUpdateConfig(
            update_config_id=_text(
                _first(row, "update_config_id", "state_update_config_id")
            ),
            update_state_layout_id=_text(_first(row, "update_state_layout_id")),
            prior_mean_matrix_id=_text(_first(row, "prior_mean_matrix_id")),
            prior_covariance_matrix_id=_text(
                _first(row, "prior_covariance_matrix_id")
            ),
            observation_jacobian_matrix_id=_text(
                _first(row, "observation_jacobian_matrix_id", "H_matrix_id")
            ),
            measurement_covariance_matrix_id_optional=(
                _text(_first(row, "measurement_covariance_matrix_id_optional", "R_matrix_id"))
                or None
            ),
            state_to_q_mapping_rule=_text(
                _first(row, "state_to_q_mapping_rule")
            ),
            algorithm=_text(
                _first(row, "algorithm"), "LINEAR_GAUSSIAN_JOSEPH"
            ).upper(),
            regularization=_float(_first(row, "regularization"), 0.0),
            nis_threshold=_float(_first(row, "nis_threshold"), np.inf),
            covariance_floor=_float(_first(row, "covariance_floor"), 0.0),
            resolve_policy=_text(
                _first(row, "resolve_policy"), "RESOLVE_LCP"
            ).upper(),
            parameter_update_allowed=_bool(
                _first(row, "parameter_update_allowed"), False
            ),
            quality_flag=_text(_first(row, "quality_flag"), "PASS").upper(),
            reference_state_id=_text(_first(row, "reference_state_id")),
            allow_diagonal_covariance_fallback=_bool(
                _first(row, "allow_diagonal_covariance_fallback"), False
            ),
            physical_residual_threshold=_float(
                _first(row, "physical_residual_threshold"), np.inf
            ),
            individual_degradation_tolerance=_float(
                _first(row, "individual_degradation_tolerance"), 1.0
            ),
        )
        if config.update_config_id in configs:
            raise StageMeasurementUpdateValidationError(
                f"measurement_update_config 主键重复：{config.update_config_id}"
            )
        configs[config.update_config_id] = config
    return configs


def _matrix_manifest(pkg: SMSPackage) -> pd.DataFrame:
    return _table(pkg, "matrices/matrix_manifest.csv")


def _matrix_key(pkg: SMSPackage, matrix_id_or_key: str) -> str:
    if matrix_id_or_key in pkg.matrices:
        return matrix_id_or_key
    manifest = _matrix_manifest(pkg)
    if not manifest.empty:
        candidates = manifest[
            manifest.get("matrix_id", pd.Series(dtype=str)).astype(str).eq(
                str(matrix_id_or_key)
            )
        ]
        if not candidates.empty:
            key = _text(candidates.iloc[0].get("npz_key"))
            if key in pkg.matrices:
                return key
    raise StageMeasurementUpdateValidationError(
        f"矩阵无法解析：{matrix_id_or_key}"
    )


def load_matrix(pkg: SMSPackage, matrix_id_or_key: str) -> tuple[np.ndarray, str]:
    key = _matrix_key(pkg, matrix_id_or_key)
    value = np.asarray(pkg.matrices[key], dtype=float).copy()
    if not np.all(np.isfinite(value)):
        raise StageMeasurementUpdateValidationError(f"矩阵包含非有限值：{key}")
    return value, key


def _matrix_manifest_row(
    pkg: SMSPackage,
    matrix_id_or_key: str,
) -> pd.Series:
    manifest = _matrix_manifest(pkg)
    if manifest.empty:
        return pd.Series(dtype=object)
    mask = pd.Series(False, index=manifest.index)
    for field_name in ("matrix_id", "npz_key"):
        if field_name in manifest.columns:
            mask |= manifest[field_name].astype(str).eq(
                str(matrix_id_or_key)
            )
    rows = manifest[mask]
    return rows.iloc[0] if not rows.empty else pd.Series(dtype=object)


def resolve_state_to_q_matrix(
    pkg: SMSPackage,
    config: MeasurementUpdateConfig,
    checkpoint_id: str,
    operator_set_id: str,
) -> tuple[np.ndarray, str]:
    rule = config.state_to_q_mapping_rule
    if not rule:
        raise StageMeasurementUpdateValidationError(
            f"{config.update_config_id} 缺少 state_to_q_mapping_rule"
        )
    try:
        matrix_id = rule.format(
            checkpoint_id=checkpoint_id,
            operator_set_id=operator_set_id,
        )
    except (KeyError, ValueError) as exc:
        raise StageMeasurementUpdateValidationError(
            f"无效 state_to_q_mapping_rule={rule}"
        ) from exc
    return load_matrix(pkg, matrix_id)


def _symmetric_psd(
    matrix: np.ndarray,
    *,
    name: str,
    tolerance: float = 1e-10,
    require_positive: bool = False,
) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise StageMeasurementUpdateValidationError(
            f"{name} 必须为方阵，实际 shape={matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)):
        raise StageMeasurementUpdateValidationError(f"{name} 包含非有限值")
    if not np.allclose(matrix, matrix.T, atol=tolerance, rtol=0.0):
        raise StageMeasurementUpdateValidationError(f"{name} 不对称")
    eigenvalues = np.linalg.eigvalsh((matrix + matrix.T) / 2.0)
    minimum = float(np.min(eigenvalues)) if eigenvalues.size else 0.0
    if require_positive and minimum <= tolerance:
        raise StageMeasurementUpdateValidationError(
            f"{name} 非正定，最小特征值={minimum:.6g}"
        )
    if minimum < -tolerance:
        raise StageMeasurementUpdateValidationError(
            f"{name} 非半正定，最小特征值={minimum:.6g}"
        )
    return (matrix + matrix.T) / 2.0


def linear_gaussian_update(
    eta_prior: np.ndarray,
    P_prior: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
    z_measured: np.ndarray,
    z_predicted_prior: np.ndarray,
    *,
    regularization: float = 0.0,
    covariance_floor: float = 0.0,
) -> dict[str, np.ndarray | float | str]:
    eta_prior = np.asarray(eta_prior, dtype=float).reshape(-1)
    P_prior = _symmetric_psd(P_prior, name="P_prior")
    H = np.asarray(H, dtype=float)
    R = _symmetric_psd(R, name="R", require_positive=True)
    z_measured = np.asarray(z_measured, dtype=float).reshape(-1)
    z_predicted_prior = np.asarray(z_predicted_prior, dtype=float).reshape(-1)
    state_dimension = eta_prior.size
    observation_dimension = z_measured.size
    if P_prior.shape != (state_dimension, state_dimension):
        raise StageMeasurementUpdateValidationError(
            f"P_prior shape={P_prior.shape} 与 eta 维数={state_dimension} 不一致"
        )
    if H.shape != (observation_dimension, state_dimension):
        raise StageMeasurementUpdateValidationError(
            f"H shape={H.shape}，期望 {(observation_dimension, state_dimension)}"
        )
    if R.shape != (observation_dimension, observation_dimension):
        raise StageMeasurementUpdateValidationError(
            f"R shape={R.shape}，期望 {(observation_dimension, observation_dimension)}"
        )
    if z_predicted_prior.shape != z_measured.shape:
        raise StageMeasurementUpdateValidationError(
            "z_measured 与 z_predicted_prior 维数不一致"
        )
    if not all(np.all(np.isfinite(value)) for value in (
        eta_prior, H, z_measured, z_predicted_prior,
    )):
        raise StageMeasurementUpdateValidationError("更新输入包含非有限值")
    if regularization < 0.0:
        raise StageMeasurementUpdateValidationError("regularization 不得小于 0")
    innovation = z_measured - z_predicted_prior
    innovation_covariance = H @ P_prior @ H.T + R
    if regularization > 0.0:
        innovation_covariance = (
            innovation_covariance
            + float(regularization) * np.eye(observation_dimension)
        )
        solve_method = "CHOLESKY_WITH_CONFIGURED_REGULARIZATION"
    else:
        solve_method = "CHOLESKY"
    innovation_covariance = _symmetric_psd(
        innovation_covariance,
        name="innovation_covariance",
        require_positive=True,
    )
    try:
        factor = np.linalg.cholesky(innovation_covariance)
        solved_cross = np.linalg.solve(
            factor.T,
            np.linalg.solve(factor, H @ P_prior),
        )
        solved_innovation = np.linalg.solve(
            factor.T,
            np.linalg.solve(factor, innovation),
        )
    except np.linalg.LinAlgError:
        solve_method = "NUMPY_LINALG_SOLVE"
        try:
            solved_cross = np.linalg.solve(
                innovation_covariance, H @ P_prior
            )
            solved_innovation = np.linalg.solve(
                innovation_covariance, innovation
            )
        except np.linalg.LinAlgError as exc:
            raise StageMeasurementUpdateRuntimeError(
                "创新协方差线性系统求解失败"
            ) from exc
    kalman_gain = solved_cross.T
    eta_posterior = eta_prior + kalman_gain @ innovation
    identity = np.eye(state_dimension)
    joseph_left = identity - kalman_gain @ H
    P_posterior = (
        joseph_left @ P_prior @ joseph_left.T
        + kalman_gain @ R @ kalman_gain.T
    )
    P_posterior = (P_posterior + P_posterior.T) / 2.0
    covariance_floor_applied = False
    if covariance_floor > 0.0 and P_posterior.size:
        eigenvalues, eigenvectors = np.linalg.eigh(P_posterior)
        if float(np.min(eigenvalues)) < covariance_floor:
            P_posterior = (
                eigenvectors
                @ np.diag(np.maximum(eigenvalues, covariance_floor))
                @ eigenvectors.T
            )
            P_posterior = (P_posterior + P_posterior.T) / 2.0
            covariance_floor_applied = True
    _symmetric_psd(P_posterior, name="P_posterior")
    z_predicted_posterior = (
        z_predicted_prior + H @ (eta_posterior - eta_prior)
    )
    normalized_innovation = innovation / np.sqrt(
        np.diag(innovation_covariance)
    )
    nis = float(innovation @ solved_innovation)
    return {
        "eta_posterior": eta_posterior,
        "P_posterior": P_posterior,
        "kalman_gain": kalman_gain,
        "innovation": innovation,
        "innovation_covariance": innovation_covariance,
        "normalized_innovation": normalized_innovation,
        "nis": nis,
        "z_predicted_posterior": z_predicted_posterior,
        "solve_method": solve_method,
        "covariance_form": "JOSEPH",
        "covariance_floor_applied": covariance_floor_applied,
    }


def _measurement_table(pkg: SMSPackage) -> pd.DataFrame:
    table = _table(
        pkg,
        "I_meas/measurement_record.csv",
        "I_meas/process_record.csv",
    )
    if table.empty and not pkg.process_record.empty:
        table = pkg.process_record.copy()
    if table.empty:
        return table
    normalized = table.copy()
    if "measurement_id" not in normalized.columns:
        if "record_id" in normalized.columns:
            normalized["measurement_id"] = normalized["record_id"]
        else:
            normalized["measurement_id"] = [
                f"LEGACY_PROCESS_RECORD_{index:04d}"
                for index in range(len(normalized))
            ]
    if "measurement_type" not in normalized.columns:
        normalized["measurement_type"] = normalized.get(
            "feature_definition_id", ""
        )
    for column, default in (
        ("data_role", ""),
        ("update_target", ""),
        ("quality_flag", "PASS"),
        ("reference_state_id_optional", ""),
        ("coordinate_system_id", ""),
        ("standard_uncertainty", np.nan),
        ("covariance_block_id_optional", ""),
    ):
        if column not in normalized.columns:
            normalized[column] = default
    return normalized


def _apply_measurement_override(
    package_measurements: pd.DataFrame,
    override: pd.DataFrame | None,
    checkpoint_id: str,
) -> tuple[pd.DataFrame, str]:
    if override is None:
        return package_measurements.copy(), "PACKAGE"
    runtime = override.copy()
    if "measurement_id" not in runtime.columns and "record_id" in runtime.columns:
        runtime["measurement_id"] = runtime["record_id"]
    required = {"measurement_id", "value"}
    if not required <= set(runtime.columns):
        raise StageMeasurementUpdateValidationError(
            f"运行时测量覆盖缺少字段：{sorted(required - set(runtime.columns))}"
        )
    if "checkpoint_id" in runtime.columns:
        runtime = runtime[
            runtime["checkpoint_id"].astype(str).eq(str(checkpoint_id))
        ]
    if runtime.empty:
        raise StageMeasurementUpdateValidationError(
            f"运行时覆盖不包含 checkpoint={checkpoint_id} 的记录"
        )
    ids = runtime["measurement_id"].astype(str)
    if ids.duplicated().any():
        raise StageMeasurementUpdateValidationError("运行时覆盖 measurement_id 重复")
    package_ids = set(package_measurements["measurement_id"].astype(str))
    unknown = sorted(set(ids) - package_ids)
    if unknown:
        raise StageMeasurementUpdateValidationError(
            f"运行时覆盖包含未知 measurement_id={unknown}"
        )
    allowed_override_fields = {
        "measurement_id", "record_id", "checkpoint_id",
        "value", "standard_uncertainty",
    }
    governance_fields = {
        "data_role", "update_target", "quality_flag", "reference_state_id_optional",
        "coordinate_system_id", "unit", "measurement_set_id",
    }
    supplied_governance = governance_fields & set(runtime.columns)
    if supplied_governance:
        raise StageMeasurementUpdateValidationError(
            "运行时覆盖不得修改治理字段："
            + ", ".join(sorted(supplied_governance))
        )
    unexpected = set(runtime.columns) - allowed_override_fields
    if unexpected:
        raise StageMeasurementUpdateValidationError(
            "运行时覆盖包含不支持字段：" + ", ".join(sorted(unexpected))
        )
    merged = package_measurements.copy()
    for _, row in runtime.iterrows():
        measurement_id = str(row["measurement_id"])
        value = _float(row["value"])
        if not np.isfinite(value):
            raise StageMeasurementUpdateValidationError(
                f"{measurement_id} 覆盖值非有限"
            )
        mask = merged["measurement_id"].astype(str).eq(measurement_id)
        merged.loc[mask, "value"] = value
        if "standard_uncertainty" in runtime.columns and not _is_blank(
            row.get("standard_uncertainty")
        ):
            uncertainty = _float(row.get("standard_uncertainty"))
            if not np.isfinite(uncertainty) or uncertainty <= 0.0:
                raise StageMeasurementUpdateValidationError(
                    f"{measurement_id} 覆盖 standard_uncertainty 必须大于 0"
                )
            merged.loc[mask, "standard_uncertainty"] = uncertainty
    return merged, "RUNTIME_OVERRIDE"


def validate_runtime_measurement_override(
    package: SMSPackage,
    override: pd.DataFrame,
) -> pd.DataFrame:
    """Validate a runtime CSV without mutating or replacing package tables."""
    measurements = _measurement_table(package)
    checkpoints = [
        checkpoint
        for checkpoint in load_measurement_checkpoints(package)
        if checkpoint.active_flag
    ]
    if not checkpoints:
        raise StageMeasurementUpdateValidationError(
            "当前数据包没有启用的 measurement checkpoint"
        )
    for checkpoint in checkpoints:
        _apply_measurement_override(
            measurements,
            override,
            checkpoint.checkpoint_id,
        )
    return override.copy()


def _observation_interface_id(
    layout: pd.DataFrame,
    global_index: int,
) -> str:
    candidates = layout[
        (pd.to_numeric(layout["start_index"], errors="coerce") <= global_index)
        & (pd.to_numeric(layout["end_index"], errors="coerce") >= global_index)
    ]
    if len(candidates) != 1:
        raise StageMeasurementUpdateValidationError(
            f"global_index={global_index} 无法唯一映射到 VectorLayout"
        )
    return _text(candidates.iloc[0].get("object_id"))


def _stage_state_field(stage_state: Any, *names: str) -> Any:
    if isinstance(stage_state, dict):
        for name in names:
            if name in stage_state:
                return stage_state[name]
        return None
    for name in names:
        if hasattr(stage_state, name):
            return getattr(stage_state, name)
    return None


def extract_observation_vector(
    stage_state: Any,
    observation_specs: list[MeasurementObservationSpec],
    vector_layout_frame: pd.DataFrame,
) -> np.ndarray:
    """Extract governed physical observations from one runtime stage state.

    The same ordered extractor is used for prior, post-LCP, VALIDATE and
    independent-oracle comparisons. It intentionally supports only the first
    formally governed normal-contact quantities.
    """
    if vector_layout_frame.empty:
        raise StageMeasurementUpdateValidationError(
            "缺少公共 VectorLayout"
        )
    layout = vector_layout_frame.copy()
    required_layout = {"object_id", "start_index", "end_index"}
    if not required_layout <= set(layout.columns):
        raise StageMeasurementUpdateValidationError(
            "VectorLayout 缺少字段："
            + ", ".join(sorted(required_layout - set(layout.columns)))
        )
    layout_ids = (
        layout.get("vector_layout_id", pd.Series(dtype=str))
        .dropna().astype(str).unique().tolist()
    )
    state_layout_id = _text(
        _stage_state_field(stage_state, "vector_layout_id")
    )
    if (
        state_layout_id
        and layout_ids
        and state_layout_id not in set(layout_ids)
    ):
        raise StageMeasurementUpdateValidationError(
            f"stage state VectorLayout={state_layout_id}"
            f" 与 observation layout={layout_ids} 不一致"
        )
    active_interfaces = set(
        str(value)
        for value in (
            _stage_state_field(stage_state, "active_interface_ids") or []
        )
    )
    quantity_fields = {
        "GAP_G": ("gap_full", "gap_g", "gap"),
        "LAMBDA_N": ("lambda_full", "lambda_n"),
        "PRESSURE_P_N": ("pressure", "pressure_p_n"),
        "LOCAL_COMPRESSION_W_N": (
            "local_compression", "local_compression_w_n",
        ),
    }
    output: list[float] = []
    for observation in observation_specs:
        quantity = observation.observed_quantity
        if quantity not in SUPPORTED_OBSERVED_QUANTITIES:
            raise StageMeasurementUpdateValidationError(
                f"不支持 observed_quantity={quantity}"
            )
        if observation.vector_source not in SUPPORTED_VECTOR_SOURCES:
            raise StageMeasurementUpdateValidationError(
                f"不支持 vector_source={observation.vector_source or '<blank>'}"
            )
        observation_unit = observation.unit.upper()
        if (
            not observation_unit
            or observation_unit not in OBSERVATION_UNITS[quantity]
        ):
            raise StageMeasurementUpdateValidationError(
                f"{observation.observation_id} unit={observation.unit or '<blank>'}"
                f" 不支持 observed_quantity={quantity}"
            )
        index = observation.global_index_optional
        if index is None:
            raise StageMeasurementUpdateValidationError(
                f"{observation.observation_id} 缺少 global_index"
            )
        interface_id = _observation_interface_id(layout, index)
        if (
            observation.target_object_id_optional
            and observation.target_object_id_optional != interface_id
        ):
            raise StageMeasurementUpdateValidationError(
                f"{observation.observation_id} object="
                f"{observation.target_object_id_optional}"
                f" 与 index 所属接口={interface_id} 不一致"
            )
        if active_interfaces and interface_id not in active_interfaces:
            raise StageMeasurementUpdateValidationError(
                f"{observation.observation_id} 指向非活动接口={interface_id}"
            )
        values_raw = _stage_state_field(
            stage_state, *quantity_fields[quantity]
        )
        if values_raw is None:
            raise StageMeasurementUpdateValidationError(
                f"stage state 缺少 {quantity} 物理向量"
            )
        values = np.asarray(values_raw, dtype=float).reshape(-1)
        if index < 0 or index >= values.size:
            raise StageMeasurementUpdateValidationError(
                f"{observation.observation_id} global_index={index}"
                f" 超出 [0,{values.size})"
            )
        value = float(values[index])
        if not np.isfinite(value):
            raise StageMeasurementUpdateValidationError(
                f"{observation.observation_id} 物理观测值非有限"
            )
        output.append(value)
    return np.asarray(output, dtype=float)


def _residual_metrics(
    z_measured: np.ndarray,
    z_predicted: np.ndarray,
    covariance: np.ndarray,
) -> dict[str, Any]:
    residual = (
        np.asarray(z_measured, dtype=float).reshape(-1)
        - np.asarray(z_predicted, dtype=float).reshape(-1)
    )
    covariance = _symmetric_psd(
        covariance, name="measurement_covariance", require_positive=True
    )
    if covariance.shape != (residual.size, residual.size):
        raise StageMeasurementUpdateValidationError(
            f"measurement covariance shape={covariance.shape}"
            f" 与 residual 维数={residual.size} 不一致"
        )
    try:
        factor = np.linalg.cholesky(covariance)
        whitened = np.linalg.solve(factor, residual)
    except np.linalg.LinAlgError as exc:
        raise StageMeasurementUpdateValidationError(
            "measurement covariance Cholesky 求解失败"
        ) from exc
    standardized = residual / np.sqrt(np.diag(covariance))
    return {
        "residual": residual,
        "raw_norm": float(np.linalg.norm(residual)),
        "standardized": standardized,
        "weighted": float(whitened @ whitened),
    }


def _measurement_covariance(
    pkg: SMSPackage,
    config: MeasurementUpdateConfig,
    measurements: pd.DataFrame,
    *,
    total_observation_count: int | None = None,
) -> tuple[np.ndarray, str, str]:
    block_ids = [
        _text(value)
        for value in measurements.get(
            "covariance_block_id_optional", pd.Series(dtype=object)
        )
        if not _is_blank(value)
    ]
    unique_blocks = list(dict.fromkeys(block_ids))
    configured_block = config.measurement_covariance_matrix_id_optional or ""
    if configured_block and unique_blocks and unique_blocks != [configured_block]:
        raise StageMeasurementUpdateValidationError(
            f"measurement covariance config={configured_block}"
            f" 与 records={unique_blocks} 不一致"
        )
    declared_block = configured_block or (
        unique_blocks[0] if len(unique_blocks) == 1 else ""
    )
    if unique_blocks and (
        len(unique_blocks) != 1 or len(block_ids) != len(measurements)
    ):
        raise StageMeasurementUpdateValidationError(
            "covariance_block_id_optional 必须对当前更新观测完整且唯一"
        )
    if declared_block:
        matrix, key = load_matrix(pkg, declared_block)
        manifest_row = _matrix_manifest_row(pkg, declared_block)
        if manifest_row.empty:
            raise StageMeasurementUpdateValidationError(
                f"R[{declared_block}] 缺少 MatrixManifest"
            )
        row_layout_id = _text(
            manifest_row.get("row_layout_id_optional")
        )
        column_layout_id = _text(
            manifest_row.get("column_layout_id_optional")
        )
        if (
            not row_layout_id
            or not column_layout_id
            or row_layout_id != column_layout_id
        ):
            raise StageMeasurementUpdateValidationError(
                f"R[{declared_block}] row/column layout 不完整或不一致"
            )
        h_manifest_row = _matrix_manifest_row(
            pkg, config.observation_jacobian_matrix_id
        )
        h_row_layout_id = _text(
            h_manifest_row.get("row_layout_id_optional")
        )
        if not h_row_layout_id or h_row_layout_id != row_layout_id:
            raise StageMeasurementUpdateValidationError(
                f"R[{declared_block}] layout={row_layout_id}"
                f" 与 H layout={h_row_layout_id or '<blank>'} 不一致"
            )
        matrix = _symmetric_psd(
            matrix, name=f"R[{declared_block}]", require_positive=True
        )
        allowed_dimensions = {len(measurements)}
        if total_observation_count is not None:
            allowed_dimensions.add(int(total_observation_count))
        if (
            matrix.ndim != 2
            or matrix.shape[0] != matrix.shape[1]
            or matrix.shape[0] not in allowed_dimensions
        ):
            raise StageMeasurementUpdateValidationError(
                f"R[{declared_block}] shape={matrix.shape}"
                f"，期望维数={sorted(allowed_dimensions)}"
            )
        return matrix, key, "FULL_COVARIANCE_BLOCK"
    if not config.allow_diagonal_covariance_fallback:
        raise StageMeasurementUpdateValidationError(
            "未声明完整 covariance block，且配置未允许对角回退"
        )
    uncertainties = pd.to_numeric(
        measurements["standard_uncertainty"], errors="coerce"
    ).to_numpy(dtype=float)
    if (
        uncertainties.size != len(measurements)
        or not np.all(np.isfinite(uncertainties))
        or np.any(uncertainties <= 0.0)
    ):
        raise StageMeasurementUpdateValidationError(
            "standard_uncertainty 必须逐观测存在、有限且大于 0"
        )
    return (
        np.diag(uncertainties**2),
        "DIAGONAL_STANDARD_UNCERTAINTY",
        "DIAGONAL_STANDARD_UNCERTAINTY",
    )


def _frame_hash(frame: pd.DataFrame) -> str:
    if frame.empty:
        return hashlib.sha256(b"EMPTY").hexdigest()
    normalized = frame.fillna("").astype(str)
    payload = normalized.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _named_table_hashes(
    pkg: SMSPackage,
    predicate: Any,
) -> dict[str, str]:
    return {
        path: _frame_hash(pkg.raw_tables[path])
        for path in sorted(pkg.raw_tables)
        if predicate(path.lower())
    }


def _named_matrix_hashes(
    pkg: SMSPackage,
    predicate: Any,
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for key in sorted(pkg.matrices):
        if not predicate(key.lower()):
            continue
        array = np.asarray(pkg.matrices[key])
        payload = (
            str(array.dtype).encode("utf-8")
            + repr(array.shape).encode("utf-8")
            + np.ascontiguousarray(array).tobytes()
        )
        hashes[key] = hashlib.sha256(payload).hexdigest()
    return hashes


def _parameter_and_sms_hashes(pkg: SMSPackage) -> dict[str, Any]:
    sms_tables = _named_table_hashes(
        pkg, lambda path: "sms" in path
    )
    parameter_tables = _named_table_hashes(
        pkg,
        lambda path: (
            path.startswith("parameter_library/")
            or path in {
                "i0/material.csv",
                "i0/joint_definition.csv",
                "i_gamma/interface_parameter.csv",
            }
            or any(
                token in path
                for token in (
                    "compliance", "cn", "ct", "mu", "beta_r",
                    "joint_stiffness",
                )
            )
        ),
    )
    manifest_tables = _named_table_hashes(
        pkg, lambda path: path == "matrices/matrix_manifest.csv"
    )
    frozen_matrices = _named_matrix_hashes(
        pkg,
        lambda key: any(
            token in key
            for token in (
                "w_struct", "cn", "ct", "mu", "beta_r",
                "joint_stiffness",
            )
        ),
    )
    parameter_payload = json.dumps(
        {
            "tables": parameter_tables,
            "matrices": frozen_matrices,
            "manifest": manifest_tables,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    sms_payload = json.dumps(
        sms_tables, ensure_ascii=False, sort_keys=True
    )
    frozen_objects = {
        **{f"TABLE::{key}": value for key, value in parameter_tables.items()},
        **{f"TABLE::{key}": value for key, value in sms_tables.items()},
        **{f"TABLE::{key}": value for key, value in manifest_tables.items()},
        **{f"MATRIX::{key}": value for key, value in frozen_matrices.items()},
    }
    return {
        "parameter_snapshot_hash": hashlib.sha256(
            parameter_payload.encode("utf-8")
        ).hexdigest(),
        "sms_snapshot_hash": hashlib.sha256(
            sms_payload.encode("utf-8")
        ).hexdigest(),
        "frozen_object_hashes": frozen_objects,
    }


def _transfer_table_row(
    pkg: SMSPackage,
    source_layout_id: str,
    target_topology_step_id: str,
) -> pd.Series:
    table = _table(pkg, "I_stage/stage_covariance_transfer.csv")
    if table.empty:
        return pd.Series(dtype=object)
    mask = pd.Series(True, index=table.index)
    if "update_state_layout_id" in table.columns:
        mask &= table["update_state_layout_id"].astype(str).eq(
            str(source_layout_id)
        )
    target_field = (
        "to_topology_step_id"
        if "to_topology_step_id" in table.columns
        else "target_topology_step_id"
    )
    if target_field in table.columns:
        mask &= table[target_field].astype(str).eq(
            str(target_topology_step_id)
        )
    rows = table[mask]
    return rows.iloc[0] if not rows.empty else pd.Series(dtype=object)


def propagate_low_dimensional_state(
    pkg: SMSPackage,
    parent_state: StageState | None,
    target_topology_step_id: str,
    *,
    fallback_layout_id: str = "",
) -> dict[str, Any]:
    if (
        parent_state is None
        or np.asarray(parent_state.state_correction_vector).size == 0
    ):
        return {
            "eta": np.array([], dtype=float),
            "P": np.empty((0, 0), dtype=float),
            "update_state_layout_id": fallback_layout_id,
            "covariance_transfer_id": "",
            "covariance_source": "NOT_AVAILABLE",
            "F": np.empty((0, 0), dtype=float),
            "Q": np.empty((0, 0), dtype=float),
            "trace": {"policy": "NO_PARENT_POSTERIOR"},
        }
    eta_parent = np.asarray(
        parent_state.state_correction_vector, dtype=float
    ).reshape(-1)
    P_parent = _symmetric_psd(
        np.asarray(parent_state.state_covariance, dtype=float),
        name="parent_state_covariance",
    )
    if P_parent.shape != (eta_parent.size, eta_parent.size):
        raise StageMeasurementUpdateValidationError(
            "父状态 eta/P 维数不一致"
        )
    layout_id = parent_state.update_state_layout_id or fallback_layout_id
    row = _transfer_table_row(pkg, layout_id, target_topology_step_id)
    if row.empty:
        F = np.eye(eta_parent.size)
        Q = np.zeros_like(P_parent)
        transfer_id = (
            f"DEFAULT_IDENTITY_ZERO__{parent_state.topology_step_id}"
            f"__{target_topology_step_id}"
        )
        covariance_source = "DEFAULT_IDENTITY_F_ZERO_Q"
        F_key = "IDENTITY"
        Q_key = "ZERO"
    else:
        transfer_id = _text(
            _first(row, "covariance_transfer_id", "transfer_id")
        )
        F_id = _text(_first(row, "state_jacobian_F_matrix_id", "F_matrix_id"))
        Q_id = _text(_first(row, "process_noise_Q_matrix_id", "Q_matrix_id"))
        F, F_key = load_matrix(pkg, F_id)
        Q, Q_key = load_matrix(pkg, Q_id)
        covariance_source = "PACKAGE_STAGE_COVARIANCE_TRANSFER"
    if F.shape != (eta_parent.size, eta_parent.size):
        raise StageMeasurementUpdateValidationError(
            f"F shape={F.shape} 与父状态维数={eta_parent.size} 不一致"
        )
    Q = _symmetric_psd(Q, name="Q")
    if Q.shape != P_parent.shape:
        raise StageMeasurementUpdateValidationError(
            f"Q shape={Q.shape} 与 P shape={P_parent.shape} 不一致"
        )
    eta = F @ eta_parent
    P = F @ P_parent @ F.T + Q
    P = _symmetric_psd(P, name="propagated_P")
    return {
        "eta": eta,
        "P": P,
        "update_state_layout_id": layout_id,
        "covariance_transfer_id": transfer_id,
        "covariance_source": covariance_source,
        "F": F,
        "Q": Q,
        "trace": {
            "policy": covariance_source,
            "F_key": F_key,
            "Q_key": Q_key,
            "source_state_id": parent_state.stage_state_id,
            "target_topology_step_id": target_topology_step_id,
        },
    }


def effective_q_from_state(
    pkg: SMSPackage,
    parent_state: StageState | None,
    operator_set_id: str,
    q_base: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    q_base = np.asarray(q_base, dtype=float).reshape(-1)
    if (
        parent_state is None
        or np.asarray(parent_state.state_correction_vector).size == 0
        or not parent_state.source_checkpoint_id
    ):
        return q_base.copy(), {
            "applied": False,
            "q_base_key": "",
            "G_q_key": "",
            "eta_source_state_id": "",
            "q_correction_norm": 0.0,
            "effective_q_hash": hashlib.sha256(q_base.tobytes()).hexdigest(),
        }
    checkpoints = {
        spec.checkpoint_id: spec for spec in load_measurement_checkpoints(pkg)
    }
    checkpoint = checkpoints.get(parent_state.source_checkpoint_id)
    if checkpoint is None:
        raise StageMeasurementUpdateValidationError(
            f"父状态 checkpoint 无法解析：{parent_state.source_checkpoint_id}"
        )
    configs = load_measurement_update_configs(pkg)
    config = configs.get(checkpoint.update_config_id)
    if config is None:
        raise StageMeasurementUpdateValidationError(
            f"checkpoint={checkpoint.checkpoint_id} 的 update_config 无法解析"
        )
    G_q, G_q_key = resolve_state_to_q_matrix(
        pkg, config, checkpoint.checkpoint_id, operator_set_id
    )
    eta = np.asarray(parent_state.state_correction_vector, dtype=float).reshape(-1)
    if G_q.shape != (q_base.size, eta.size):
        raise StageMeasurementUpdateValidationError(
            f"G_q shape={G_q.shape}，期望 {(q_base.size, eta.size)}"
        )
    correction = G_q @ eta
    effective_q = q_base + correction
    return effective_q, {
        "applied": True,
        "q_base_key": "",
        "G_q_key": G_q_key,
        "eta_source_state_id": parent_state.stage_state_id,
        "q_correction_norm": float(np.linalg.norm(correction)),
        "effective_q_hash": hashlib.sha256(
            effective_q.astype(np.float64).tobytes()
        ).hexdigest(),
    }


def _gate(
    rows: list[dict[str, Any]],
    check_item: str,
    ok: bool,
    detail: str,
    *,
    blocking: bool = True,
) -> None:
    rows.append({
        "check_item": check_item,
        "status": "PASS" if ok else "FAIL",
        "detail": detail,
        "blocking": bool(blocking and not ok),
    })


def validate_measurement_update_package(
    pkg: SMSPackage,
    topology_specs: list[Any] | None = None,
) -> pd.DataFrame:
    checkpoints = load_measurement_checkpoints(pkg)
    if not checkpoints:
        return pd.DataFrame(columns=["check_item", "status", "detail", "blocking"])
    if topology_specs is None:
        from .topology_step import load_topology_steps
        topology_specs = load_topology_steps(pkg)
    rows: list[dict[str, Any]] = []
    route_ids = [str(spec.topology_step_id) for spec in topology_specs]
    route_positions = {step_id: index for index, step_id in enumerate(route_ids)}
    checkpoint_ids = [spec.checkpoint_id for spec in checkpoints]
    _gate(
        rows, "measurement checkpoint主键唯一",
        len(checkpoint_ids) == len(set(checkpoint_ids)) and all(checkpoint_ids),
        f"checkpoint_ids={checkpoint_ids}",
    )
    target_steps = [spec.topology_step_id for spec in checkpoints if spec.active_flag]
    _gate(
        rows, "checkpoint与topology_step唯一对应",
        len(target_steps) == len(set(target_steps)),
        f"topology_step_ids={target_steps}",
    )
    missing_steps = [
        spec.checkpoint_id for spec in checkpoints
        if spec.topology_step_id not in route_positions
        or spec.source_topology_step_id not in route_positions
    ]
    _gate(
        rows, "checkpoint步骤外键可解析",
        not missing_steps, f"invalid={missing_steps}",
    )
    bad_order = [
        spec.checkpoint_id for spec in checkpoints
        if spec.topology_step_id in route_positions
        and spec.source_topology_step_id in route_positions
        and route_positions[spec.source_topology_step_id]
        >= route_positions[spec.topology_step_id]
    ]
    _gate(
        rows, "source_topology_step位于checkpoint之前",
        not bad_order, f"invalid={bad_order}",
    )
    route_checkpoint_map = {
        str(spec.topology_step_id): _text(spec.measurement_checkpoint_id)
        for spec in topology_specs
    }
    mismatch = [
        spec.checkpoint_id for spec in checkpoints
        if route_checkpoint_map.get(spec.topology_step_id, "")
        != spec.checkpoint_id
    ]
    _gate(
        rows, "route measurement_checkpoint_id一致",
        not mismatch, f"invalid={mismatch}",
    )
    configs = load_measurement_update_configs(pkg)
    missing_configs = [
        spec.checkpoint_id for spec in checkpoints
        if spec.update_config_id not in configs
    ]
    _gate(
        rows, "measurement update config外键可解析",
        not missing_configs, f"invalid={missing_configs}",
    )
    measurements = _measurement_table(pkg)
    measurement_ids = (
        measurements.get("measurement_id", pd.Series(dtype=str))
        .dropna().astype(str).tolist()
    )
    _gate(
        rows, "measurement_id唯一",
        len(measurement_ids) == len(set(measurement_ids)),
        f"count={len(measurement_ids)}",
    )
    future_mapping_errors: list[str] = []
    matrix_errors: list[str] = []
    governance_errors: list[str] = []
    covariance_errors: list[str] = []
    route_by_id = {
        str(spec.topology_step_id): spec for spec in topology_specs
    }
    measurement_lookup = {
        _text(row.get("measurement_id")): row
        for _, row in measurements.iterrows()
    }
    package_sample_id = _text(pkg.manifest.get("sample_id"))
    for checkpoint in checkpoints:
        config = configs.get(checkpoint.update_config_id)
        if config is None:
            continue
        basis = load_state_update_basis(pkg, config.update_state_layout_id)
        observations = load_measurement_observations(pkg, checkpoint.checkpoint_id)
        observation_ids = [
            observation.observation_id for observation in observations
        ]
        if (
            len(observation_ids) != len(set(observation_ids))
            or any(not value for value in observation_ids)
        ):
            governance_errors.append(
                f"{checkpoint.checkpoint_id}:OBSERVATION_ID_NOT_UNIQUE"
            )
        target_spec = route_by_id.get(checkpoint.topology_step_id)
        target_stage_id = _text(
            getattr(target_spec, "stage_id", "")
        )
        if (
            not checkpoint.reference_state_id
            or not config.reference_state_id
            or checkpoint.reference_state_id
            != config.reference_state_id
        ):
            governance_errors.append(
                f"{checkpoint.checkpoint_id}:REFERENCE_STATE_CONFIG_MISMATCH"
            )
        checkpoint_measurements: list[pd.Series] = []
        for observation in observations:
            row = measurement_lookup.get(observation.measurement_id)
            if row is None:
                governance_errors.append(
                    f"{checkpoint.checkpoint_id}:"
                    f"{observation.measurement_id}:MISSING"
                )
                continue
            checkpoint_measurements.append(row)
            role_raw = _text(row.get("data_role")).upper()
            if role_raw not in {
                "CALIBRATE", "UPDATE", "VALIDATE", "IDENTIFY",
                "MONITOR", "EXCLUDE",
            }:
                governance_errors.append(
                    f"{checkpoint.checkpoint_id}:"
                    f"{observation.measurement_id}:DATA_ROLE_INVALID"
                )
            if observation.vector_source not in SUPPORTED_VECTOR_SOURCES:
                governance_errors.append(
                    f"{checkpoint.checkpoint_id}:"
                    f"{observation.measurement_id}:VECTOR_SOURCE_INVALID"
                )
            if (
                not observation.reference_state_id
                or observation.reference_state_id
                != checkpoint.reference_state_id
            ):
                governance_errors.append(
                    f"{checkpoint.checkpoint_id}:"
                    f"{observation.measurement_id}:OBS_REFERENCE_MISMATCH"
                )
            record_reference = _text(
                _first(
                    row,
                    "reference_state_id_optional",
                    "reference_state_id",
                )
            )
            if (
                not record_reference
                or record_reference != checkpoint.reference_state_id
            ):
                governance_errors.append(
                    f"{checkpoint.checkpoint_id}:"
                    f"{observation.measurement_id}:RECORD_REFERENCE_MISMATCH"
                )
            if (
                package_sample_id
                and _text(row.get("sample_id")) != package_sample_id
            ):
                governance_errors.append(
                    f"{checkpoint.checkpoint_id}:"
                    f"{observation.measurement_id}:SAMPLE_ID_MISMATCH"
                )
            if (
                target_stage_id
                and _text(row.get("stage_id")) != target_stage_id
            ):
                governance_errors.append(
                    f"{checkpoint.checkpoint_id}:"
                    f"{observation.measurement_id}:STAGE_MISMATCH"
                )
        try:
            eta, _ = load_matrix(pkg, config.prior_mean_matrix_id)
            P, _ = load_matrix(pkg, config.prior_covariance_matrix_id)
            H, _ = load_matrix(pkg, config.observation_jacobian_matrix_id)
            dimension = len(basis)
            if eta.reshape(-1).size != dimension:
                raise StageMeasurementUpdateValidationError(
                    f"eta维数={eta.size}，basis维数={dimension}"
                )
            _symmetric_psd(P, name="P_prior")
            if P.shape != (dimension, dimension):
                raise StageMeasurementUpdateValidationError(
                    f"P shape={P.shape}，期望 {(dimension, dimension)}"
                )
            if H.shape != (len(observations), dimension):
                raise StageMeasurementUpdateValidationError(
                    f"H shape={H.shape}，期望 {(len(observations), dimension)}"
                )
        except StageMeasurementUpdateValidationError as exc:
            matrix_errors.append(f"{checkpoint.checkpoint_id}:{exc}")
        update_rows = [
            row
            for row in checkpoint_measurements
            if _text(row.get("data_role")).upper()
            in {"CALIBRATE", "UPDATE"}
        ]
        if update_rows:
            try:
                _measurement_covariance(
                    pkg,
                    config,
                    pd.DataFrame(update_rows).reset_index(drop=True),
                    total_observation_count=len(observations),
                )
            except StageMeasurementUpdateValidationError as exc:
                covariance_errors.append(
                    f"{checkpoint.checkpoint_id}:{exc}"
                )
        current_position = route_positions.get(checkpoint.topology_step_id, -1)
        affected_specs = [
            spec for index, spec in enumerate(topology_specs)
            if (
                index >= current_position
                and bool(spec.solve_required)
                and spec.operator_set_id
            )
        ]
        source_spec = next(
            (
                spec for spec in topology_specs
                if str(spec.topology_step_id) == checkpoint.source_topology_step_id
            ),
            None,
        )
        if source_spec is not None and source_spec.operator_set_id:
            affected_specs = [source_spec, *affected_specs]
        seen_operator_sets: set[str] = set()
        for spec in affected_specs:
            operator_set_id = str(spec.operator_set_id)
            if operator_set_id in seen_operator_sets:
                continue
            seen_operator_sets.add(operator_set_id)
            try:
                G_q, _ = resolve_state_to_q_matrix(
                    pkg, config, checkpoint.checkpoint_id, operator_set_id
                )
                if G_q.shape != (len(pkg.contact_points), len(basis)):
                    raise StageMeasurementUpdateValidationError(
                        f"G_q shape={G_q.shape}"
                    )
            except StageMeasurementUpdateValidationError as exc:
                future_mapping_errors.append(
                    f"{checkpoint.checkpoint_id}:{operator_set_id}:{exc}"
                )
    _gate(
        rows, "P/H矩阵和update state layout一致",
        not matrix_errors, f"invalid={matrix_errors}",
    )
    _gate(
        rows, "checkpoint及后续步骤G_q完整",
        not future_mapping_errors, f"invalid={future_mapping_errors}",
    )
    _gate(
        rows, "measurement治理字段完整一致",
        not governance_errors, f"invalid={governance_errors}",
    )
    _gate(
        rows, "measurement covariance来源可解析",
        not covariance_errors, f"invalid={covariance_errors}",
    )
    return pd.DataFrame(rows)


def _posterior_interface_states(
    pkg: SMSPackage,
    predicted_state: StageState,
    posterior_state_id: str,
    lambda_full: np.ndarray,
    gap_full: np.ndarray,
    pressure: np.ndarray,
    local_compression: np.ndarray,
) -> dict[str, dict[str, Any]]:
    states = deepcopy(predicted_state.interface_state)
    layout = vector_layout(pkg)
    for _, row in layout.iterrows():
        interface_id = _text(row.get("object_id"))
        if interface_id not in set(predicted_state.active_interface_ids):
            continue
        start = int(row.get("start_index", 0))
        end = int(row.get("end_index", start))
        indices = np.arange(start, end + 1, dtype=int)
        previous = states.get(interface_id, {})
        previous_id = _text(previous.get("interface_state_id"))
        states[interface_id] = {
            **previous,
            "interface_state_id": f"{posterior_state_id}_INTERFACE_{interface_id}",
            "parent_interface_state_id": previous_id,
            "lambda_n": lambda_full[indices].tolist(),
            "gap_g": gap_full[indices].tolist(),
            "pressure_p_n": pressure[indices].tolist(),
            "local_compression_w_n": local_compression[indices].tolist(),
            "active_set_local_indices": [
                int(local_index)
                for local_index, global_index in enumerate(indices)
                if lambda_full[global_index] > 0.0
            ],
            "data_source": "MEASUREMENT_POSTERIOR_RESOLVE_LCP",
        }
    return states


def _rollback_result(
    *,
    checkpoint: MeasurementCheckpointSpec,
    predicted_state: StageState,
    update_id: str,
    attempted_posterior_state_id: str,
    measurement_ids: list[str],
    failure_stage: str,
    failure_reason: str,
    decision_record: UpdateDecisionRecord | None = None,
    measurement_source: str = "PACKAGE",
    trace: dict[str, Any] | None = None,
) -> StageMeasurementUpdateResult:
    rollback = UpdateRollbackRecord(
        rollback_record_id=f"ROLLBACK_{update_id}",
        checkpoint_id=checkpoint.checkpoint_id,
        predicted_state_id=predicted_state.stage_state_id,
        attempted_posterior_state_id=attempted_posterior_state_id,
        failure_stage=failure_stage,
        failure_reason=failure_reason,
        retained_effective_state_id=predicted_state.stage_state_id,
        measurement_ids=list(measurement_ids),
        quality_flag="FAIL",
    )
    empty_vector = np.array([], dtype=float)
    empty_matrix = np.empty((0, 0), dtype=float)
    return StageMeasurementUpdateResult(
        update_id=update_id,
        checkpoint_id=checkpoint.checkpoint_id,
        topology_step_id=checkpoint.topology_step_id,
        source_topology_step_id=checkpoint.source_topology_step_id,
        predicted_state_id=predicted_state.stage_state_id,
        posterior_state_id=attempted_posterior_state_id,
        effective_state_id=predicted_state.stage_state_id,
        measurement_ids=list(measurement_ids),
        z_measured=empty_vector,
        z_predicted_prior=empty_vector,
        z_predicted_posterior=empty_vector,
        innovation=empty_vector,
        normalized_innovation=empty_vector,
        eta_prior=empty_vector,
        eta_posterior=empty_vector,
        P_prior=empty_matrix,
        P_posterior=empty_matrix,
        kalman_gain=empty_matrix,
        nis=np.nan,
        q_correction_prior=empty_vector,
        q_correction_posterior=empty_vector,
        resolve_requirement=ReSolveRequirement(
            level="NONE",
            changed_objects=(),
            source_operator_set_id="",
            reason=failure_reason,
            required_matrix_keys=(),
            quality_flag="FAIL",
        ),
        resolve_lcp_call_count=0,
        physical_residuals={},
        posterior_accepted=False,
        rollback_record=rollback,
        quality_flag="FAIL",
        trace={
            **(trace or {}),
            "status": "POSTERIOR_REJECTED_ROLLBACK",
            "failure_stage": failure_stage,
            "failure_reason": failure_reason,
        },
        decision_record=decision_record,
        posterior_state=None,
        measurement_source=measurement_source,
    )


def _evaluation_only_result(
    *,
    checkpoint: MeasurementCheckpointSpec,
    predicted_state: StageState,
    update_id: str,
    evaluation_measurement_ids: list[str],
    skipped_measurement_ids: list[str],
    observation_records: list[dict[str, Any]],
    decision_record: UpdateDecisionRecord,
    measurement_source: str,
    hashes_before: dict[str, Any],
    hashes_after: dict[str, Any],
) -> StageMeasurementUpdateResult:
    empty_vector = np.array([], dtype=float)
    empty_matrix = np.empty((0, 0), dtype=float)
    return StageMeasurementUpdateResult(
        update_id=update_id,
        checkpoint_id=checkpoint.checkpoint_id,
        topology_step_id=checkpoint.topology_step_id,
        source_topology_step_id=checkpoint.source_topology_step_id,
        predicted_state_id=predicted_state.stage_state_id,
        posterior_state_id="",
        effective_state_id=predicted_state.stage_state_id,
        measurement_ids=(
            list(evaluation_measurement_ids)
            + list(skipped_measurement_ids)
        ),
        z_measured=empty_vector,
        z_predicted_prior=empty_vector,
        z_predicted_posterior=empty_vector,
        innovation=empty_vector,
        normalized_innovation=empty_vector,
        eta_prior=empty_vector,
        eta_posterior=empty_vector,
        P_prior=empty_matrix,
        P_posterior=empty_matrix,
        kalman_gain=empty_matrix,
        nis=np.nan,
        q_correction_prior=empty_vector,
        q_correction_posterior=empty_vector,
        resolve_requirement=ReSolveRequirement(
            level="NONE",
            changed_objects=(),
            source_operator_set_id="",
            reason="NO_UPDATE_OBSERVATION",
            required_matrix_keys=(),
            quality_flag="PASS",
        ),
        resolve_lcp_call_count=0,
        physical_residuals={},
        posterior_accepted=False,
        rollback_record=None,
        quality_flag="PASS",
        trace={
            "status": "EVALUATION_ONLY",
            "decision_basis": "NO_UPDATE_OBSERVATION",
            "update_measurement_count": 0,
            "evaluation_measurement_count": len(
                evaluation_measurement_ids
            ),
            "skipped_measurement_count": len(skipped_measurement_ids),
            "parameter_frozen": (
                hashes_before["parameter_snapshot_hash"]
                == hashes_after["parameter_snapshot_hash"]
            ),
            "sms_frozen": (
                hashes_before["sms_snapshot_hash"]
                == hashes_after["sms_snapshot_hash"]
            ),
            "hashes_before": hashes_before,
            "hashes_after": hashes_after,
        },
        decision_record=decision_record,
        posterior_state=None,
        measurement_source=measurement_source,
        observation_records=observation_records,
        evaluation_measurement_ids=list(evaluation_measurement_ids),
        skipped_measurement_ids=list(skipped_measurement_ids),
    )


def run_stage_measurement_update(
    *,
    package: SMSPackage,
    checkpoint_id: str,
    predicted_step_result: dict[str, Any],
    previous_results: dict[str, dict[str, Any]],
    predicted_state: StageState,
    measurement_override: pd.DataFrame | None = None,
    eps: float = 1e-9,
) -> StageMeasurementUpdateResult:
    checkpoints = {
        spec.checkpoint_id: spec
        for spec in load_measurement_checkpoints(package)
        if spec.active_flag
    }
    checkpoint = checkpoints.get(checkpoint_id)
    if checkpoint is None:
        raise StageMeasurementUpdateValidationError(
            f"measurement checkpoint 无法解析：{checkpoint_id}"
        )
    update_id = (
        f"MEAS_UPDATE_{predicted_state.sample_id}_{checkpoint.checkpoint_id}"
    )
    posterior_state_id = (
        f"STATE_{predicted_state.sample_id}_"
        f"{checkpoint.topology_step_id}_POSTERIOR"
    )
    hashes_before = _parameter_and_sms_hashes(package)
    observations = load_measurement_observations(
        package, checkpoint.checkpoint_id
    )
    measurement_ids = [item.measurement_id for item in observations]
    decision_record: UpdateDecisionRecord | None = None
    measurement_source = "PACKAGE"
    attempt_trace: dict[str, Any] = {}
    try:
        if predicted_state.topology_step_id != checkpoint.topology_step_id:
            raise StageMeasurementUpdateValidationError(
                "predicted state 与 checkpoint topology_step 不一致"
            )
        source_result = previous_results.get(checkpoint.source_topology_step_id)
        if source_result is None:
            raise StageMeasurementUpdateValidationError(
                f"source_topology_step 不存在：{checkpoint.source_topology_step_id}"
            )
        configs = load_measurement_update_configs(package)
        config = configs.get(checkpoint.update_config_id)
        if config is None:
            raise StageMeasurementUpdateValidationError(
                f"update_config 不存在：{checkpoint.update_config_id}"
            )
        if config.parameter_update_allowed:
            raise StageMeasurementUpdateValidationError(
                "本轮 parameter_update_allowed 必须为 false"
            )
        if config.quality_flag != "PASS":
            raise StageMeasurementUpdateValidationError(
                f"update_config quality_flag={config.quality_flag}"
            )
        source_state = source_result.get("stage_state")
        reference_values = {
            "checkpoint": checkpoint.reference_state_id,
            "config": config.reference_state_id,
            "predicted_state": predicted_state.reference_state,
            "source_state": _text(
                getattr(source_state, "reference_state", "")
            ),
        }
        if (
            any(not value for value in reference_values.values())
            or len(set(reference_values.values())) != 1
        ):
            raise StageMeasurementUpdateValidationError(
                f"REFERENCE_STATE_MISMATCH:{reference_values}"
            )
        basis = load_state_update_basis(
            package, config.update_state_layout_id
        )
        if not basis:
            raise StageMeasurementUpdateValidationError(
                f"状态更新基为空：{config.update_state_layout_id}"
            )
        if not observations:
            raise StageMeasurementUpdateValidationError(
                f"checkpoint={checkpoint.checkpoint_id} 没有 observation map"
            )
        if len(measurement_ids) != len(set(measurement_ids)):
            raise StageMeasurementUpdateValidationError(
                "observation map measurement_id 重复"
            )
        all_measurements = _measurement_table(package)
        measurements = all_measurements[
            all_measurements["measurement_id"].astype(str).isin(measurement_ids)
        ].copy()
        measurements, measurement_source = _apply_measurement_override(
            measurements, measurement_override, checkpoint.checkpoint_id
        )
        measurement_lookup = {
            str(row["measurement_id"]): row
            for _, row in measurements.iterrows()
        }
        update_observations: list[MeasurementObservationSpec] = []
        evaluation_observations: list[MeasurementObservationSpec] = []
        skipped_observations: list[MeasurementObservationSpec] = []
        rejected_ids: list[str] = []
        blocked_ids: list[str] = []
        rejection_reasons: list[str] = []
        blocking_reasons: list[str] = []
        update_rows: list[pd.Series] = []
        evaluation_rows: list[pd.Series] = []
        data_roles: list[str] = []
        update_targets: list[str] = []
        layout = vector_layout(package)
        for observation in observations:
            row = measurement_lookup.get(observation.measurement_id)
            if row is None:
                rejected_ids.append(observation.measurement_id)
                blocked_ids.append(observation.measurement_id)
                rejection_reasons.append(
                    f"{observation.measurement_id}:MISSING"
                )
                blocking_reasons.append(
                    f"{observation.measurement_id}:MISSING"
                )
                continue
            role_raw = _text(row.get("data_role")).upper()
            role = "CALIBRATE" if role_raw == "UPDATE" else role_raw
            target = _text(row.get("update_target")).upper()
            data_roles.append(role_raw)
            update_targets.append(target)
            reason = ""
            if not role_raw:
                reason = "DATA_ROLE_MISSING"
            elif role not in {
                "CALIBRATE", "VALIDATE", "IDENTIFY", "MONITOR", "EXCLUDE",
            }:
                reason = f"DATA_ROLE_{role_raw}_NOT_ALLOWED"
            if (
                not reason
                and role == "CALIBRATE"
                and target in FROZEN_PARAMETER_TARGETS
            ):
                reason = f"UPDATE_TARGET_{target}_FROZEN"
            elif (
                not reason
                and role == "CALIBRATE"
                and target not in ALLOWED_STATE_UPDATE_TARGETS
            ):
                reason = f"UPDATE_TARGET_{target}_NOT_ALLOWED"
            if (
                not reason
                and role in {"CALIBRATE", "VALIDATE"}
                and _text(row.get("quality_flag"), "PASS").upper() != "PASS"
            ):
                reason = f"QUALITY_{_text(row.get('quality_flag')).upper()}"
            elif (
                not reason
                and role in {"CALIBRATE", "VALIDATE"}
                and observation.quality_flag != "PASS"
            ):
                reason = f"OBSERVATION_MAP_QUALITY_{observation.quality_flag}"
            if (
                checkpoint.measurement_set_id
                and not reason
                and _text(row.get("measurement_set_id"))
                != checkpoint.measurement_set_id
            ):
                reason = "MEASUREMENT_SET_MISMATCH"
            sample_id = _text(row.get("sample_id"))
            if (
                not reason
                and role in {"CALIBRATE", "VALIDATE"}
                and (
                    not sample_id
                    or sample_id != predicted_state.sample_id
                )
            ):
                reason = "SAMPLE_ID_MISMATCH"
            reference = _text(
                _first(row, "reference_state_id_optional", "reference_state_id")
            )
            if (
                not reason
                and role in {"CALIBRATE", "VALIDATE"}
                and (
                    not reference
                    or not observation.reference_state_id
                    or reference != checkpoint.reference_state_id
                    or observation.reference_state_id
                    != checkpoint.reference_state_id
                )
            ):
                reason = "REFERENCE_STATE_MISMATCH"
            measurement_stage = _text(row.get("stage_id"))
            if (
                not reason
                and role in {"CALIBRATE", "VALIDATE"}
                and (
                    not measurement_stage
                    or measurement_stage != predicted_state.stage_id
                )
            ):
                reason = "STAGE_CHECKPOINT_MISMATCH"
            coordinate = _text(row.get("coordinate_system_id"))
            if (
                not reason
                and role in {"CALIBRATE", "VALIDATE"}
                and observation.coordinate_system_id
                and coordinate != observation.coordinate_system_id
            ):
                reason = "COORDINATE_SYSTEM_MISMATCH"
            unit = _text(row.get("unit")).upper()
            observation_unit = observation.unit.upper()
            if (
                not reason
                and role in {"CALIBRATE", "VALIDATE"}
                and unit != observation_unit
            ):
                reason = "UNIT_MISMATCH"
            if (
                not reason
                and role in {"CALIBRATE", "VALIDATE"}
                and observation_unit
                not in OBSERVATION_UNITS.get(
                    observation.observed_quantity, {observation_unit}
                )
            ):
                reason = "OBSERVATION_UNIT_NOT_SUPPORTED"
            if (
                not reason
                and role in {"CALIBRATE", "VALIDATE"}
                and observation.vector_source not in SUPPORTED_VECTOR_SOURCES
            ):
                reason = "VECTOR_SOURCE_NOT_SUPPORTED"
            record_object_id = _text(
                _first(
                    row,
                    "interface_id_optional",
                    "object_id",
                )
            )
            if (
                not reason
                and role in {"CALIBRATE", "VALIDATE"}
                and observation.target_object_id_optional
                and record_object_id
                != observation.target_object_id_optional
            ):
                reason = "OBJECT_INTERFACE_MISMATCH"
            if not reason and role in {"CALIBRATE", "VALIDATE"}:
                try:
                    extract_observation_vector(
                        source_result, [observation], layout
                    )
                except StageMeasurementUpdateValidationError as exc:
                    reason = str(exc)
            if reason:
                rejected_ids.append(observation.measurement_id)
                blocked_ids.append(observation.measurement_id)
                rejection_reasons.append(
                    f"{observation.measurement_id}:{reason}"
                )
                if role != "VALIDATE":
                    blocking_reasons.append(
                        f"{observation.measurement_id}:{reason}"
                    )
            elif role == "CALIBRATE":
                update_observations.append(observation)
                update_rows.append(row)
            elif role == "VALIDATE":
                evaluation_observations.append(observation)
                evaluation_rows.append(row)
            else:
                skipped_observations.append(observation)
        allow_partial = checkpoint.missing_measurement_policy in {
            "SKIP", "SKIP_BAD", "ALLOW_PARTIAL",
        }
        decision = "ACCEPT_FOR_STATE_UPDATE"
        decision_reason = "CALIBRATE measurements passed governance gates"
        if blocking_reasons and not allow_partial:
            decision = "REJECT"
            decision_reason = ";".join(blocking_reasons)
        if not update_observations and decision != "REJECT":
            decision = "EVALUATION_ONLY"
            decision_reason = "NO_UPDATE_OBSERVATION"
        decision_record = UpdateDecisionRecord(
            update_id=update_id,
            checkpoint_id=checkpoint.checkpoint_id,
            measurement_ids=measurement_ids,
            accepted_measurement_ids=[
                item.measurement_id for item in update_observations
            ],
            rejected_measurement_ids=rejected_ids,
            evaluation_measurement_ids=[
                item.measurement_id for item in evaluation_observations
            ],
            skipped_measurement_ids=[
                item.measurement_id for item in skipped_observations
            ],
            blocked_measurement_ids=blocked_ids,
            data_roles=data_roles,
            update_targets=update_targets,
            allowed_state_components=[item.component_id for item in basis],
            frozen_parameter_ids=sorted(FROZEN_PARAMETER_TARGETS),
            decision=decision,
            decision_reason=decision_reason,
            identifiability_summary=(
                f"update={len(update_observations)};"
                f" evaluation={len(evaluation_observations)};"
                f" skipped={len(skipped_observations)};"
                f" state_dimension={len(basis)}"
            ),
            quality_flag="FAIL" if decision == "REJECT" else "PASS",
        )
        if decision == "REJECT":
            raise StageMeasurementUpdateValidationError(decision_reason)
        evaluation_ids = [
            item.measurement_id for item in evaluation_observations
        ]
        skipped_ids = [
            item.measurement_id for item in skipped_observations
        ]
        if decision == "EVALUATION_ONLY":
            observation_records: list[dict[str, Any]] = []
            for observation, row in zip(
                evaluation_observations, evaluation_rows
            ):
                value = extract_observation_vector(
                    source_result, [observation], layout
                )[0]
                measured = _float(row.get("value"))
                observation_records.append({
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "measurement_id": observation.measurement_id,
                    "observation_class": "EVALUATION_ONLY",
                    "observed_quantity": observation.observed_quantity,
                    "z_measured": measured,
                    "z_pred_prior_physical": value,
                    "z_pred_posterior_physical": value,
                    "prior_residual_physical": measured - value,
                    "posterior_residual_physical": measured - value,
                    "data_role": "VALIDATE",
                    "quality_flag": "PASS",
                })
            for observation in skipped_observations:
                observation_records.append({
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "measurement_id": observation.measurement_id,
                    "observation_class": "SKIPPED_IDENTIFY",
                    "data_role": _text(
                        measurement_lookup[
                            observation.measurement_id
                        ].get("data_role")
                    ).upper(),
                    "quality_flag": "SKIPPED",
                })
            hashes_after = _parameter_and_sms_hashes(package)
            return _evaluation_only_result(
                checkpoint=checkpoint,
                predicted_state=predicted_state,
                update_id=update_id,
                evaluation_measurement_ids=evaluation_ids,
                skipped_measurement_ids=skipped_ids,
                observation_records=observation_records,
                decision_record=decision_record,
                measurement_source=measurement_source,
                hashes_before=hashes_before,
                hashes_after=hashes_after,
            )
        accepted_frame = pd.DataFrame(update_rows).reset_index(drop=True)
        accepted = update_observations
        accepted_ids = [item.measurement_id for item in accepted]
        z_measured = pd.to_numeric(
            accepted_frame["value"], errors="coerce"
        ).to_numpy(dtype=float)
        if not np.all(np.isfinite(z_measured)):
            raise StageMeasurementUpdateValidationError(
                "measurement value 包含非有限值"
            )
        z_predicted_prior = extract_observation_vector(
            source_result, accepted, layout
        )
        eta_matrix, eta_key = load_matrix(
            package, config.prior_mean_matrix_id
        )
        P_matrix, P_key = load_matrix(
            package, config.prior_covariance_matrix_id
        )
        H_full, H_key = load_matrix(
            package, config.observation_jacobian_matrix_id
        )
        eta_prior = eta_matrix.reshape(-1)
        P_prior = P_matrix
        if np.asarray(predicted_state.state_correction_vector).size:
            if (
                predicted_state.update_state_layout_id
                != config.update_state_layout_id
            ):
                raise StageMeasurementUpdateValidationError(
                    "predicted state update layout 与 checkpoint config 不一致"
                )
            eta_prior = np.asarray(
                predicted_state.state_correction_vector, dtype=float
            ).reshape(-1)
            P_prior = np.asarray(
                predicted_state.state_covariance, dtype=float
            )
        row_indices = np.array(
            [item.sensitivity_row_index for item in accepted], dtype=int
        )
        if np.any(row_indices < 0) or np.any(row_indices >= H_full.shape[0]):
            raise StageMeasurementUpdateValidationError(
                "sensitivity_row_index 超出 H 行范围"
            )
        H = H_full[row_indices, :]
        R_full, R_key, covariance_source = _measurement_covariance(
            package,
            config,
            accepted_frame,
            total_observation_count=len(observations),
        )
        if R_full.shape == (len(observations), len(observations)):
            observation_positions = {
                item.measurement_id: position
                for position, item in enumerate(observations)
            }
            selected = np.array(
                [observation_positions[item.measurement_id] for item in accepted],
                dtype=int,
            )
            R = R_full[np.ix_(selected, selected)]
        else:
            R = R_full
        update = linear_gaussian_update(
            eta_prior, P_prior, H, R,
            z_measured, z_predicted_prior,
            regularization=config.regularization,
            covariance_floor=config.covariance_floor,
        )
        eta_posterior = np.asarray(update["eta_posterior"], dtype=float)
        P_posterior = np.asarray(update["P_posterior"], dtype=float)
        z_predicted_posterior_linearized = np.asarray(
            update["z_predicted_posterior"], dtype=float
        )
        prior_metrics = _residual_metrics(
            z_measured, z_predicted_prior, R
        )
        linearized_metrics = _residual_metrics(
            z_measured, z_predicted_posterior_linearized, R
        )
        attempt_trace.update({
            "acceptance_basis": "POST_LCP_PHYSICAL_OBSERVATION",
            "z_predicted_prior_physical": _vector_record(
                z_predicted_prior
            ),
            "z_predicted_posterior_linearized": _vector_record(
                z_predicted_posterior_linearized
            ),
            "residual_prior_physical": _vector_record(
                prior_metrics["residual"]
            ),
            "residual_posterior_linearized": _vector_record(
                linearized_metrics["residual"]
            ),
            "prior_residual_norm": prior_metrics["raw_norm"],
            "posterior_linearized_residual_norm": (
                linearized_metrics["raw_norm"]
            ),
            "weighted_residual_prior_physical": (
                prior_metrics["weighted"]
            ),
            "measurement_covariance_source": covariance_source,
        })
        nis = float(update["nis"])
        if nis > config.nis_threshold:
            raise StageMeasurementUpdateRuntimeError(
                f"NIS={nis:.6g} 超过阈值={config.nis_threshold:.6g}"
            )
        source_operator_set_id = _text(
            source_result.get("operator_set_id")
        )
        G_q, G_q_key = resolve_state_to_q_matrix(
            package, config, checkpoint.checkpoint_id,
            source_operator_set_id,
        )
        q_source_effective = np.asarray(
            source_result["q"], dtype=float
        ).reshape(-1)
        if G_q.shape != (q_source_effective.size, eta_prior.size):
            raise StageMeasurementUpdateValidationError(
                f"G_q shape={G_q.shape}"
                f"，期望 {(q_source_effective.size, eta_prior.size)}"
            )
        q_correction_prior = G_q @ eta_prior
        q_correction_posterior = G_q @ eta_posterior
        eta_increment = eta_posterior - eta_prior
        q_update_increment = G_q @ eta_increment
        q_posterior = q_source_effective + q_update_increment
        q_operator_base = q_source_effective - q_correction_prior
        resolve_requirement = ReSolveRequirement(
            level=(
                "NONE"
                if config.resolve_policy == "NONE"
                else "RESOLVE_LCP"
            ),
            changed_objects=("q",),
            source_operator_set_id=source_operator_set_id,
            reason="LOW_DIMENSIONAL_STATE_CORRECTION_CHANGED_Q",
            required_matrix_keys=(
                _text(source_result["contact_trace"].get("W_struct_key")),
                _text(source_result["contact_trace"].get("Cn_key")),
                G_q_key,
            ),
            quality_flag="PASS",
        )
        active_mask = np.asarray(
            source_result["active_index_mask"], dtype=bool
        )
        active_indices = np.flatnonzero(active_mask)
        W_struct = np.asarray(source_result["W_struct"], dtype=float)
        Cn = np.asarray(source_result["Cn"], dtype=float)
        W_total = W_struct + Cn
        resolve_lcp_call_count = 0
        if resolve_requirement.level == "RESOLVE_LCP":
            try:
                active_solution = solve_lcp_active_set(
                    q_posterior[active_indices],
                    W_total[np.ix_(active_indices, active_indices)],
                    eps=eps,
                )
            except RuntimeError as exc:
                raise StageMeasurementUpdateRuntimeError(
                    f"posterior LCP execution failed: {exc}"
                ) from exc
            resolve_lcp_call_count = 1
            if active_solution.convergence_status != "CONVERGED":
                raise StageMeasurementUpdateRuntimeError(
                    f"posterior LCP status={active_solution.convergence_status}"
                )
            lambda_full = np.zeros_like(q_posterior)
            gap_full = np.full_like(q_posterior, np.nan)
            lambda_full[active_indices] = active_solution.lambda_n
            gap_full[active_indices] = active_solution.gap_g
            pressure = np.full_like(q_posterior, np.nan)
            area = package.contact_points["area_weight"].to_numpy(dtype=float)
            pressure[active_indices] = (
                active_solution.lambda_n / area[active_indices]
            )
            local_compression = np.full_like(q_posterior, np.nan)
            local_compression[active_indices] = (
                Cn[np.ix_(active_indices, active_indices)]
                @ active_solution.lambda_n
            )
            global_active = [
                int(active_indices[index])
                for index in active_solution.active_indices
            ]
            full_solution = LCPSolution(
                lambda_n=lambda_full,
                gap_g=gap_full,
                active_indices=global_active,
                inactive_indices=[
                    index for index in range(q_posterior.size)
                    if index not in set(global_active)
                ],
                residuals=dict(active_solution.residuals),
                iteration_count=active_solution.iteration_count,
                convergence_status=active_solution.convergence_status,
                active_set_trace=list(active_solution.active_set_trace),
            )
        else:
            lambda_full = np.asarray(
                source_result["lambda_full"], dtype=float
            ).copy()
            gap_full = np.asarray(
                source_result["gap_full"], dtype=float
            ).copy()
            pressure = np.asarray(
                source_result["pressure"], dtype=float
            ).copy()
            local_compression = np.asarray(
                source_result["local_compression"], dtype=float
            ).copy()
            full_solution = deepcopy(source_result["solution"])
        residuals = {
            key: float(value)
            for key, value in full_solution.residuals.items()
        }
        active_lambda = lambda_full[active_indices]
        active_gap = gap_full[active_indices]
        complementarity = float(
            residuals.get("complementarity_residual", np.inf)
        )
        if (
            np.min(active_lambda, initial=0.0) < -eps
            or np.min(active_gap, initial=0.0) < -eps
            or complementarity > max(eps * 10.0, 1e-8)
        ):
            raise StageMeasurementUpdateRuntimeError(
                "posterior LCP 物理质量门失败"
            )
        posterior_physical_state = {
            "vector_layout_id": predicted_state.vector_layout_id,
            "active_interface_ids": list(
                predicted_state.active_interface_ids
            ),
            "gap_full": gap_full,
            "lambda_full": lambda_full,
            "pressure": pressure,
            "local_compression": local_compression,
        }
        z_predicted_posterior_physical = extract_observation_vector(
            posterior_physical_state, accepted, layout
        )
        posterior_metrics = _residual_metrics(
            z_measured, z_predicted_posterior_physical, R
        )
        weighted_tolerance = max(
            1e-12,
            1e-9 * max(1.0, prior_metrics["weighted"]),
        )
        physical_residual_improved = bool(
            posterior_metrics["weighted"]
            <= prior_metrics["weighted"] + weighted_tolerance
        )
        attempt_trace.update({
            "z_predicted_posterior_physical": _vector_record(
                z_predicted_posterior_physical
            ),
            "residual_posterior_physical": _vector_record(
                posterior_metrics["residual"]
            ),
            "posterior_residual_norm": posterior_metrics["raw_norm"],
            "weighted_residual_posterior_physical": (
                posterior_metrics["weighted"]
            ),
            "physical_residual_improved": physical_residual_improved,
            "linearization_error": _vector_record(
                z_predicted_posterior_physical
                - z_predicted_posterior_linearized
            ),
        })
        if not physical_residual_improved:
            raise StageMeasurementUpdateRuntimeError(
                "POSTERIOR_PHYSICAL_WEIGHTED_RESIDUAL_NOT_IMPROVED:"
                f" prior={prior_metrics['weighted']:.12g};"
                f" post={posterior_metrics['weighted']:.12g}"
            )
        maximum_standardized = float(
            np.max(
                np.abs(posterior_metrics["standardized"]),
                initial=0.0,
            )
        )
        if maximum_standardized > config.physical_residual_threshold:
            raise StageMeasurementUpdateRuntimeError(
                "POSTERIOR_PHYSICAL_RESIDUAL_THRESHOLD_FAILED:"
                f" max_standardized={maximum_standardized:.12g};"
                f" threshold={config.physical_residual_threshold:.12g}"
            )
        severe_degradation = (
            np.abs(posterior_metrics["standardized"])
            > (
                np.abs(prior_metrics["standardized"])
                + config.individual_degradation_tolerance
            )
        ) & (
            np.abs(posterior_metrics["standardized"])
            > config.physical_residual_threshold
        )
        if np.any(severe_degradation):
            raise StageMeasurementUpdateRuntimeError(
                "POSTERIOR_INDIVIDUAL_PHYSICAL_RESIDUAL_SEVERE_DEGRADATION:"
                f" indices={np.flatnonzero(severe_degradation).tolist()}"
            )
        linearization_error = (
            z_predicted_posterior_physical
            - z_predicted_posterior_linearized
        )
        posterior_state = deepcopy(predicted_state)
        posterior_state.stage_state_id = posterior_state_id
        posterior_state.parent_stage_state_id = predicted_state.stage_state_id
        posterior_state.parent_stage_id = predicted_state.stage_id
        posterior_state.interface_state = _posterior_interface_states(
            package, predicted_state, posterior_state_id,
            lambda_full, gap_full, pressure, local_compression,
        )
        posterior_state.contact_structural_response = W_struct @ lambda_full
        posterior_state.contact_structural_response_increment = (
            posterior_state.contact_structural_response
            - predicted_state.contact_structural_response
        )
        posterior_state.gap = gap_full
        posterior_state.gap_increment = (
            np.nan_to_num(gap_full) - np.nan_to_num(predicted_state.gap)
        )
        posterior_state.lambda_n = lambda_full
        posterior_state.pressure = pressure
        posterior_state.local_compression = local_compression
        posterior_state.active_set = list(full_solution.active_indices)
        posterior_state.physical_residuals = residuals
        posterior_state.solve_status = (
            "CONVERGED"
            if resolve_requirement.level == "RESOLVE_LCP"
            else predicted_state.solve_status
        )
        posterior_state.mechanical_state_action = (
            "MEASUREMENT_POSTERIOR_RESOLVE_LCP"
            if resolve_requirement.level == "RESOLVE_LCP"
            else "MEASUREMENT_POSTERIOR_NO_RESOLVE"
        )
        posterior_state.not_required_reason = ""
        posterior_state.quality_flag = "PASS"
        posterior_state.state_role = "POSTERIOR"
        posterior_state.measurement_checkpoint_id = checkpoint.checkpoint_id
        posterior_state.measurement_update_id = update_id
        posterior_state.predicted_state_id = predicted_state.stage_state_id
        posterior_state.posterior_parent_state_id = predicted_state.stage_state_id
        posterior_state.effective_state_id = posterior_state_id
        posterior_state.update_state_layout_id = (
            config.update_state_layout_id
        )
        posterior_state.state_correction_vector = eta_posterior
        posterior_state.state_covariance = P_posterior
        posterior_state.measurement_ids = accepted_ids
        posterior_state.measurement_update_status = "POSTERIOR_ACCEPTED"
        posterior_state.posterior_accepted = True
        posterior_state.rollback_record_id = ""
        posterior_state.covariance_source = (
            "LINEAR_GAUSSIAN_JOSEPH_POSTERIOR"
        )
        posterior_state.source_checkpoint_id = checkpoint.checkpoint_id
        posterior_state.lambda_active = lambda_full[active_indices]
        posterior_state.gap_active = gap_full[active_indices]
        hashes_after = _parameter_and_sms_hashes(package)
        if hashes_before != hashes_after:
            before_objects = hashes_before.get("frozen_object_hashes", {})
            after_objects = hashes_after.get("frozen_object_hashes", {})
            changed_objects = sorted(
                key
                for key in set(before_objects) | set(after_objects)
                if before_objects.get(key) != after_objects.get(key)
            )
            raise StageMeasurementUpdateRuntimeError(
                "FROZEN_PARAMETER_OR_SMS_CHANGED:"
                + ",".join(changed_objects)
            )
        observation_records = []
        innovation = np.asarray(update["innovation"], dtype=float)
        normalized_innovation = np.asarray(
            update["normalized_innovation"], dtype=float
        )
        for position, observation in enumerate(accepted):
            row = accepted_frame.iloc[position]
            observation_records.append({
                "checkpoint_id": checkpoint.checkpoint_id,
                "measurement_id": observation.measurement_id,
                "observed_quantity": observation.observed_quantity,
                "object_id": observation.target_object_id_optional or "",
                "global_index": observation.global_index_optional,
                "observation_class": "UPDATE",
                "z_pred_prior": z_predicted_prior[position],
                "z_pred_prior_physical": z_predicted_prior[position],
                "z_measured": z_measured[position],
                "innovation": innovation[position],
                "standard_uncertainty": row.get("standard_uncertainty"),
                "normalized_innovation": normalized_innovation[position],
                "z_pred_posterior": (
                    z_predicted_posterior_physical[position]
                ),
                "z_pred_posterior_linearized": (
                    z_predicted_posterior_linearized[position]
                ),
                "z_pred_posterior_physical": (
                    z_predicted_posterior_physical[position]
                ),
                "prior_residual_physical": (
                    prior_metrics["residual"][position]
                ),
                "posterior_residual_linearized": (
                    linearized_metrics["residual"][position]
                ),
                "posterior_residual_physical": (
                    posterior_metrics["residual"][position]
                ),
                "posterior_residual": (
                    posterior_metrics["residual"][position]
                ),
                "linearization_error": linearization_error[position],
                "data_role": row.get("data_role", ""),
                "update_target": row.get("update_target", ""),
                "quality_flag": row.get("quality_flag", ""),
            })
        for observation, row in zip(
            evaluation_observations, evaluation_rows
        ):
            z_prior_evaluation = extract_observation_vector(
                source_result, [observation], layout
            )[0]
            z_post_evaluation = extract_observation_vector(
                posterior_physical_state, [observation], layout
            )[0]
            measured_evaluation = _float(row.get("value"))
            observation_records.append({
                "checkpoint_id": checkpoint.checkpoint_id,
                "measurement_id": observation.measurement_id,
                "observed_quantity": observation.observed_quantity,
                "object_id": observation.target_object_id_optional or "",
                "global_index": observation.global_index_optional,
                "observation_class": "EVALUATION_ONLY",
                "z_measured": measured_evaluation,
                "z_pred_prior_physical": z_prior_evaluation,
                "z_pred_posterior_physical": z_post_evaluation,
                "prior_residual_physical": (
                    measured_evaluation - z_prior_evaluation
                ),
                "posterior_residual_physical": (
                    measured_evaluation - z_post_evaluation
                ),
                "data_role": "VALIDATE",
                "update_target": row.get("update_target", ""),
                "quality_flag": "PASS",
            })
        for observation in skipped_observations:
            observation_records.append({
                "checkpoint_id": checkpoint.checkpoint_id,
                "measurement_id": observation.measurement_id,
                "observation_class": "SKIPPED_IDENTIFY",
                "data_role": _text(
                    measurement_lookup[
                        observation.measurement_id
                    ].get("data_role")
                ).upper(),
                "quality_flag": "SKIPPED",
            })
        trace = {
            "status": "POSTERIOR_ACCEPTED",
            "acceptance_basis": "POST_LCP_PHYSICAL_OBSERVATION",
            "checkpoint_id": checkpoint.checkpoint_id,
            "measurement_source": measurement_source,
            "measurement_ids": accepted_ids,
            "update_measurement_count": len(accepted_ids),
            "evaluation_measurement_count": len(evaluation_ids),
            "skipped_measurement_count": len(skipped_ids),
            "eta_prior_key": eta_key,
            "P_prior_key": P_key,
            "H_key": H_key,
            "R_key": R_key,
            "measurement_covariance_source": covariance_source,
            "G_q_key": G_q_key,
            "q_operator_base_key": source_result["contact_trace"].get(
                "q_key", ""
            ),
            "q_base_semantics": (
                "CURRENT_CHECKPOINT_SOURCE_Q_IS_EFFECTIVE;"
                " APPLY_G_Q_TO_ETA_INCREMENT"
            ),
            "operator_set_id": source_operator_set_id,
            "update_state_layout_id": config.update_state_layout_id,
            "solve_method": update["solve_method"],
            "covariance_form": update["covariance_form"],
            "covariance_floor_applied": update["covariance_floor_applied"],
            "prior_residual_norm": prior_metrics["raw_norm"],
            "posterior_residual_norm": posterior_metrics["raw_norm"],
            "posterior_linearized_residual_norm": (
                linearized_metrics["raw_norm"]
            ),
            "weighted_residual_prior_physical": (
                prior_metrics["weighted"]
            ),
            "weighted_residual_posterior_physical": (
                posterior_metrics["weighted"]
            ),
            "physical_residual_improved": physical_residual_improved,
            "maximum_standardized_posterior_residual": (
                maximum_standardized
            ),
            "physical_residual_threshold": (
                config.physical_residual_threshold
            ),
            "linearization_error_norm": float(
                np.linalg.norm(linearization_error)
            ),
            "P_prior_trace": float(np.trace(P_prior)),
            "P_posterior_trace": float(np.trace(P_posterior)),
            "nis": nis,
            "nis_threshold": config.nis_threshold,
            "q_correction_prior_norm": float(
                np.linalg.norm(q_correction_prior)
            ),
            "q_correction_posterior_norm": float(
                np.linalg.norm(q_correction_posterior)
            ),
            "eta_increment_norm": float(np.linalg.norm(eta_increment)),
            "q_update_increment_norm": float(
                np.linalg.norm(q_update_increment)
            ),
            "resolve_lcp_call_count": resolve_lcp_call_count,
            "active_interface_ids": list(
                predicted_state.active_interface_ids
            ),
            "active_indices": active_indices.tolist(),
            "cross_interface_blocks_retained": True,
            "parameter_frozen": True,
            "sms_frozen": True,
            "hashes_before": hashes_before,
            "hashes_after": hashes_after,
            "parameter_snapshot_hash_before": (
                hashes_before["parameter_snapshot_hash"]
            ),
            "parameter_snapshot_hash_after": (
                hashes_after["parameter_snapshot_hash"]
            ),
            "sms_snapshot_hash_before": (
                hashes_before["sms_snapshot_hash"]
            ),
            "sms_snapshot_hash_after": (
                hashes_after["sms_snapshot_hash"]
            ),
            "engineering_claim_allowed": bool(
                package.manifest.get("engineering_claim_allowed", False)
            ),
        }
        return StageMeasurementUpdateResult(
            update_id=update_id,
            checkpoint_id=checkpoint.checkpoint_id,
            topology_step_id=checkpoint.topology_step_id,
            source_topology_step_id=checkpoint.source_topology_step_id,
            predicted_state_id=predicted_state.stage_state_id,
            posterior_state_id=posterior_state_id,
            effective_state_id=posterior_state_id,
            measurement_ids=accepted_ids,
            z_measured=z_measured,
            z_predicted_prior=z_predicted_prior,
            z_predicted_posterior=z_predicted_posterior_physical,
            innovation=innovation,
            normalized_innovation=normalized_innovation,
            eta_prior=eta_prior,
            eta_posterior=eta_posterior,
            P_prior=P_prior,
            P_posterior=P_posterior,
            kalman_gain=np.asarray(update["kalman_gain"], dtype=float),
            nis=nis,
            q_correction_prior=q_correction_prior,
            q_correction_posterior=q_correction_posterior,
            resolve_requirement=resolve_requirement,
            resolve_lcp_call_count=resolve_lcp_call_count,
            physical_residuals=residuals,
            posterior_accepted=True,
            rollback_record=None,
            quality_flag="PASS",
            trace=trace,
            decision_record=decision_record,
            posterior_state=posterior_state,
            measurement_source=measurement_source,
            observation_records=observation_records,
            q_posterior=q_posterior,
            W_total=W_total,
            solution=full_solution,
            lambda_full=lambda_full,
            gap_full=gap_full,
            pressure=pressure,
            local_compression=local_compression,
            active_indices=active_indices,
            evaluation_measurement_ids=evaluation_ids,
            skipped_measurement_ids=skipped_ids,
            z_predicted_prior_physical=z_predicted_prior,
            z_predicted_posterior_linearized=(
                z_predicted_posterior_linearized
            ),
            z_predicted_posterior_physical=(
                z_predicted_posterior_physical
            ),
            residual_prior_physical=prior_metrics["residual"],
            residual_posterior_linearized=(
                linearized_metrics["residual"]
            ),
            residual_posterior_physical=posterior_metrics["residual"],
            standardized_residual_prior_physical=(
                prior_metrics["standardized"]
            ),
            standardized_residual_posterior_physical=(
                posterior_metrics["standardized"]
            ),
            weighted_residual_prior_physical=(
                prior_metrics["weighted"]
            ),
            weighted_residual_posterior_physical=(
                posterior_metrics["weighted"]
            ),
            linearization_error=linearization_error,
            physical_residual_improved=physical_residual_improved,
            measurement_covariance_source=covariance_source,
            q_operator_base=q_operator_base,
            q_source_effective=q_source_effective,
            eta_increment=eta_increment,
            q_update_increment=q_update_increment,
            q_effective_after_update=q_posterior,
            q_base_semantics=(
                "CURRENT_CHECKPOINT_SOURCE_Q_IS_EFFECTIVE;"
                " APPLY_G_Q_TO_ETA_INCREMENT"
            ),
        )
    except (
        StageMeasurementUpdateValidationError,
        StageMeasurementUpdateRuntimeError,
        np.linalg.LinAlgError,
        KeyError,
        ValueError,
    ) as exc:
        hashes_after = _parameter_and_sms_hashes(package)
        before_objects = hashes_before.get("frozen_object_hashes", {})
        after_objects = hashes_after.get("frozen_object_hashes", {})
        changed_objects = sorted(
            key
            for key in set(before_objects) | set(after_objects)
            if before_objects.get(key) != after_objects.get(key)
        )
        return _rollback_result(
            checkpoint=checkpoint,
            predicted_state=predicted_state,
            update_id=update_id,
            attempted_posterior_state_id=posterior_state_id,
            measurement_ids=measurement_ids,
            failure_stage=type(exc).__name__,
            failure_reason=str(exc),
            decision_record=decision_record,
            measurement_source=measurement_source,
            trace={
                **attempt_trace,
                "hashes_before": hashes_before,
                "hashes_after": hashes_after,
                "changed_frozen_objects": changed_objects,
                "parameter_frozen": (
                    hashes_before["parameter_snapshot_hash"]
                    == hashes_after["parameter_snapshot_hash"]
                ),
                "sms_frozen": (
                    hashes_before["sms_snapshot_hash"]
                    == hashes_after["sms_snapshot_hash"]
                ),
            },
        )
