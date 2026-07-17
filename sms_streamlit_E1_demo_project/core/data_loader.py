from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schema_adapter import (
    adapt_v25_package,
    detect_package_type,
    scan_package_files,
    load_all_raw_tables,
)


@dataclass
class SMSPackage:
    root: Path
    manifest: dict[str, Any]
    parts: pd.DataFrame
    interfaces: pd.DataFrame
    contact_points: pd.DataFrame
    gap_field: pd.DataFrame
    interface_parameters: pd.DataFrame
    stage_plan: pd.DataFrame
    process_record: pd.DataFrame
    kcp_kcm: pd.DataFrame
    condensed_operator: pd.DataFrame
    validation_kcp: pd.DataFrame
    matrices: dict[str, np.ndarray]
    package_type: str = 'E1_LEGACY'
    data_overview: pd.DataFrame = field(default_factory=pd.DataFrame)
    raw_tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    raw_json: dict[str, Any] = field(default_factory=dict)


def read_json(path: Path) -> dict[str, Any] | list[Any]:
    with path.open('r', encoding='utf-8-sig') as f:
        return json.load(f)


def _load_e1_legacy_package(root: Path, manifest: dict[str, Any], package_type: str, overview: pd.DataFrame) -> SMSPackage:
    parts = pd.read_csv(root / 'I0' / 'part_table.csv')
    interfaces = pd.read_csv(root / 'I0' / 'interface_table.csv')
    contact_points = pd.read_csv(root / 'I_Gamma' / 'contact_points.csv')
    gap_field = pd.read_csv(root / 'I_Gamma' / 'gap_field.csv')
    interface_parameters = pd.read_csv(root / 'I_Gamma' / 'interface_parameter.csv')
    stage_plan = pd.read_csv(root / 'I_stage' / 'stage_plan.csv')
    process_record = pd.read_csv(root / 'I_stage' / 'process_record.csv')
    kcp_kcm = pd.read_csv(root / 'I_key' / 'KCP_KCM_list.csv')
    condensed_operator = pd.read_csv(root / 'I_red' / 'condensed_operator.csv')
    validation_kcp = pd.read_csv(root / 'validation' / 'validation_kcp.csv')

    matrix_npz = np.load(root / 'matrices' / 'E1_matrices.npz', allow_pickle=False)
    matrices = {k: matrix_npz[k].copy() for k in matrix_npz.files}
    matrix_npz.close()
    raw_tables, raw_json = load_all_raw_tables(root)

    return SMSPackage(
        root=root,
        manifest=manifest,
        parts=parts,
        interfaces=interfaces,
        contact_points=contact_points,
        gap_field=gap_field,
        interface_parameters=interface_parameters,
        stage_plan=stage_plan.sort_values('operation_order').reset_index(drop=True),
        process_record=process_record,
        kcp_kcm=kcp_kcm,
        condensed_operator=condensed_operator,
        validation_kcp=validation_kcp,
        matrices=matrices,
        package_type=package_type,
        data_overview=overview,
        raw_tables=raw_tables,
        raw_json=raw_json,
    )


def load_package(root: str | Path) -> SMSPackage:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f'找不到数据包目录：{root}')

    manifest_path = root / 'package_manifest.json'
    manifest = read_json(manifest_path) if manifest_path.exists() else {'package_name': root.name}
    if not isinstance(manifest, dict):
        manifest = {'package_name': root.name, 'manifest_raw': manifest}
    package_type = detect_package_type(root)
    overview = scan_package_files(root)

    if package_type.startswith('V25'):
        adapted = adapt_v25_package(root, manifest=manifest)
        return SMSPackage(
            root=root,
            manifest=adapted['manifest'],
            parts=adapted['parts'],
            interfaces=adapted['interfaces'],
            contact_points=adapted['contact_points'],
            gap_field=adapted['gap_field'],
            interface_parameters=adapted['interface_parameters'],
            stage_plan=adapted['stage_plan'],
            process_record=adapted['process_record'],
            kcp_kcm=adapted['kcp_kcm'],
            condensed_operator=adapted['condensed_operator'],
            validation_kcp=adapted['validation_kcp'],
            matrices=adapted['matrices'],
            package_type=package_type,
            data_overview=overview,
            raw_tables=adapted.get('raw_tables', {}),
            raw_json=adapted.get('raw_json', {}),
        )

    if package_type == 'E1_LEGACY':
        return _load_e1_legacy_package(root, manifest, package_type, overview)

    raise ValueError(
        '无法识别数据包类型：需要旧 E1 包（I0/part_table.csv + matrices/E1_matrices.npz）'
        '、V2.5 包（I0/part.csv + matrices/default_matrices.npz）'
        '或多零件 V2.5 包（I0/part.csv + matrices/multi_part_matrices.npz）。'
    )


def get_stage_ids(pkg: SMSPackage) -> list[str]:
    return pkg.stage_plan.sort_values('operation_order')['stage_id'].astype(str).tolist()
