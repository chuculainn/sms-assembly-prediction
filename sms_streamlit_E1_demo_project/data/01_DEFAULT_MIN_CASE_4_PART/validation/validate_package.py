#!/usr/bin/env python3
"""Validate the synthetic multi-part serial/parallel data package."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


STAGES = ["S_LOCATE_01", "S_CLAMP_02", "S_JOIN_03", "S_RELEASE_04"]
INTERFACES = ["G_PANEL_RIB", "G_RIB_SPAR", "G_PANEL_BRACKET", "G_RIB_BRACKET"]
DOMAINS = ["CD_PANEL_RIB", "CD_RIB_SPAR", "CD_PANEL_BRACKET", "CD_RIB_BRACKET"]


def rows(root: Path, rel: str) -> list[dict[str, str]]:
    with (root / rel).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def split_ids(value: str) -> list[str]:
    return [x for x in value.split(";") if x]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str) -> None:
        checks.append((name, bool(condition), detail))

    parts = rows(root, "I0/part.csv")
    interfaces = rows(root, "I0/interface.csv")
    part_ids = {x["part_id"] for x in parts}
    edge_set = {frozenset((x["part_i"], x["part_j"])) for x in interfaces}
    check("四零件", len(parts) == 4, f"part_count={len(parts)}")
    check("四接口", len(interfaces) == 4, f"interface_count={len(interfaces)}")
    check("接口端点外键", all(x["part_i"] in part_ids and x["part_j"] in part_ids for x in interfaces), "all interface endpoints resolve")
    check("串联路径A-B-C", frozenset(("P_PANEL_A", "P_RIB_B")) in edge_set and frozenset(("P_RIB_B", "P_SPAR_C")) in edge_set, "A-B and B-C exist")
    check("并联路径A-B与A-D-B", all(frozenset(e) in edge_set for e in [("P_PANEL_A", "P_RIB_B"), ("P_PANEL_A", "P_BRACKET_D"), ("P_BRACKET_D", "P_RIB_B")]), "direct and bridge paths exist")

    domains = rows(root, "I_Gamma/contact_domain.csv")
    points = rows(root, "I_Gamma/contact_point.csv")
    point_count = {d: 0 for d in DOMAINS}
    for point in points:
        point_count[point["contact_domain_id"]] = point_count.get(point["contact_domain_id"], 0) + 1
    check("每接口三个接触点", len(domains) == 4 and all(point_count.get(d) == 3 for d in DOMAINS), json.dumps(point_count, ensure_ascii=False))
    layout = rows(root, "matrices/vector_layout.csv")
    layout_ok = [(int(x["start_index"]), int(x["end_index"])) for x in layout] == [(0,2),(3,5),(6,8),(9,11)]
    check("12维向量布局", layout_ok, "AB[0:3], BC[3:6], AD[6:9], BD[9:12]")

    manifest_rows = rows(root, "matrices/matrix_manifest.csv")
    npz = np.load(root / "matrices/multi_part_matrices.npz")
    manifest_ok = True
    for item in manifest_rows:
        key = item["npz_key"]
        expected = tuple(json.loads(item["shape"]))
        manifest_ok &= key in npz.files and npz[key].shape == expected
    check("矩阵清单与NPZ一致", manifest_ok, f"manifest={len(manifest_rows)}, npz={len(npz.files)}")

    matrix_ok = True
    coupling_ok = True
    relation_ok = True
    lcp_ok = True
    for stage in STAGES:
        suffix = stage.removeprefix("S_")
        ws = npz[f"W_STRUCT_{suffix}"]
        wt = npz[f"W_TOTAL_{suffix}"]
        cn = npz["CN_ALL"]
        matrix_ok &= ws.shape == (12,12) and np.allclose(ws, ws.T, atol=1e-12) and np.linalg.eigvalsh(ws).min() > 0
        relation_ok &= np.allclose(wt, ws + cn, atol=1e-12)
        coupling_ok &= all(np.linalg.norm(ws[a*3:(a+1)*3, b*3:(b+1)*3]) > 1e-10 for a in range(4) for b in range(a+1,4))
        q = npz[f"Q_{suffix}"]; lam = npz[f"LAMBDA_{suffix}"]; gap = npz[f"GAP_{suffix}"]
        lcp_ok &= np.all(lam >= -1e-9) and np.all(gap >= -1e-9) and np.allclose(gap, q + wt @ lam, atol=1e-9) and np.max(np.abs(lam * gap)) <= 1e-8
    check("W_struct对称正定", matrix_ok, "all four 12x12 operators")
    check("W_total=W_struct+Cn", relation_ok, "all four stages")
    check("跨接口交叉块非零", coupling_ok, "all six interface block pairs coupled")
    check("四阶段LCP互补性", lcp_ok, "lambda>=0, gap>=0, lambda*gap≈0")

    transitions = rows(root, "I_stage/stage_transition_record.csv")
    expected_transitions = list(zip(STAGES[:-1], STAGES[1:]))
    actual_transitions = [(x["from_stage_id"], x["to_stage_id"]) for x in transitions]
    check("三条阶段转移完整", actual_transitions == expected_transitions, str(actual_transitions))
    boundary_ids = {x["boundary_id"] for x in rows(root, "I_stage/boundary_item.csv")}
    load_ids = {x["load_id"] for x in rows(root, "I_stage/load_item.csv")}
    inputs = rows(root, "I_stage/stage_input.csv")
    input_fk_ok = all(set(split_ids(x["boundary_item_ids"])) <= boundary_ids and set(split_ids(x["load_item_ids"])) <= load_ids for x in inputs)
    check("阶段边界载荷外键", input_fk_ok, "all StageInput references resolve")
    state_rows = rows(root, "I_stage/stage_state_snapshot.csv")
    state_ids = {x["stage_state_snapshot_id"] for x in state_rows}
    state_chain_ok = len(state_rows) == 4 and all(not x["parent_stage_state_id_optional"] or x["parent_stage_state_id_optional"] in state_ids for x in state_rows)
    check("阶段状态父链", state_chain_ok, "four aggregate snapshots")
    part_states = rows(root, "I_stage/part_stage_state.csv")
    interface_states = rows(root, "I_stage/interface_stage_state.csv")
    check("逐零件逐接口状态", len(part_states) == 16 and len(interface_states) == 16, f"part_states={len(part_states)}, interface_states={len(interface_states)}")

    joint_ids = {x["joint_id"] for x in rows(root, "I0/joint_definition.csv")}
    lock = rows(root, "I_stage/connection_lock_history.csv")[0]
    release = rows(root, "I_stage/release_history_record.csv")[0]
    lock_ok = set(split_ids(lock["joint_ids"])) == joint_ids and float(json.loads(lock["joint_stiffness"])[next(iter(joint_ids))]) > 0 and release["lock_history_id"] == lock["lock_history_id"]
    check("JOIN锁定与RELEASE继承", lock_ok, "four non-zero-stiffness joints retained")

    contrib = rows(root, "prediction/contribution_record.csv")
    keys = [(x["sample_id"], x["source_class"], x["source_id"], x["origin_stage_id_optional"], x["increment_definition_id"]) for x in contrib]
    ledger = rows(root, "prediction/deformation_contribution_ledger.csv")[0]
    pred = np.asarray(json.loads(rows(root, "prediction/kcp_prediction_result.csv")[0]["predicted_values"]), dtype=float)
    summed = sum((np.asarray(json.loads(x["contribution_vector"]), dtype=float) for x in contrib), np.zeros(3))
    ledger_ok = len(keys) == len(set(keys)) and set(split_ids(ledger["contribution_record_ids"])) == {x["contribution_id"] for x in contrib} and np.allclose(pred, summed, atol=1e-9)
    check("贡献账本唯一且可重构", ledger_ok, f"records={len(contrib)}")

    manifest = json.loads((root / "package_manifest.json").read_text(encoding="utf-8"))
    check("真实性声明", manifest.get("data_nature") == "SYNTHETIC_NUMERICAL_CONSISTENCY_CASE" and manifest.get("engineering_claim_allowed") is False, "synthetic fixture, no engineering claim")
    legacy_tokens = ["P_PART_A", "P_PART_B", "G_PART_A_PART_B", "CD_G_PART_A_PART_B", "MODE_DEFAULT", "LOCK_DEFAULT"]
    corpus = "\n".join(path.read_text(encoding="utf-8-sig", errors="ignore") for path in root.rglob("*.csv"))
    check("无旧案例核心ID残留", not any(token in corpus for token in legacy_tokens), "legacy identifiers absent")

    passed = sum(1 for _, ok, _ in checks if ok)
    report = {"package": str(root), "passed": passed, "total": len(checks), "status": "PASS" if passed == len(checks) else "FAIL", "checks": [{"name": n, "passed": ok, "detail": d} for n, ok, d in checks]}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.write_report:
        (root / "validation/test_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        lines = ["# 多零件数据包自动测试结果", "", f"结论：**{report['status']}**（{passed}/{len(checks)}）", "", "| 检查项 | 结果 | 说明 |", "|---|---:|---|"]
        for name, ok, detail in checks:
            lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail.replace('|','/')} |")
        (root / "validation/TEST_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
