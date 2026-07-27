# 04_POSTERIOR_VIRTUAL_SMS_ROLLING_MIN_CASE validation

- Status: PASS
- Passed: 67/67
- Checks: 67/67 PASS
- Blocking failures: 0
- MatrixManifest rows: 173
- NPZ keys: 173

| Check | Status | Blocking | Detail |
|---|---:|---:|---|
| interface endpoint foreign keys | PASS | true | parts=4, interfaces=4 |
| fixture expectation: part_count | PASS | true | expected=4, actual=4 |
| fixture expectation: interface_count | PASS | true | expected=4, actual=4 |
| fixture expectation: topology_step_count | PASS | true | expected=14, actual=14 |
| serial paths from actual part-interface graph | PASS | true | [["P_A", "P_B", "P_C"]] |
| parallel direct and bridge paths from actual graph | PASS | true | [{"endpoints": ["P_A", "P_B"], "paths": [["P_A", "P_B"], ["P_A", "P_D", "P_B"]], "path_status": [true, true]}] |
| contact point foreign keys and per-interface grouping | PASS | true | {"G_AB": 3, "G_AD": 3, "G_BC": 3, "G_DB": 3} |
| fixture expectation: contact points per interface | PASS | true | expected=3, actual={"G_AB": 3, "G_AD": 3, "G_BC": 3, "G_DB": 3} |
| vector layout contiguous, unique and data-linked | PASS | true | dimension=12, intervals=[(0, 2), (3, 5), (6, 8), (9, 11)] |
| fixture expectation: vector dimension | PASS | true | expected=12, actual=12 |
| topology route IDs, order and parent chain | PASS | true | steps=['TS000', 'TS101', 'TS102', 'TS103', 'TS104', 'TS201', 'TS202', 'TS203', 'TS204', 'TS205', 'TS301', 'TS302', 'TS303', 'TS304'] |
| topology route part/interface foreign keys | PASS | true | all activated/deactivated IDs resolve |
| same-step multi-interface activation from route data | PASS | true | expected_min=1, actual=['TS301'] |
| MatrixManifest and NPZ key set | PASS | true | manifest=173, npz=173, unique=173 |
| MatrixManifest shape, dtype and row/column layout | PASS | true | errors=[] |
| precomputed topology operator completeness and identity | PASS | true | solved_steps=['TS101', 'TS102', 'TS103', 'TS104', 'TS201', 'TS202', 'TS203', 'TS204', 'TS301', 'TS302', 'TS303', 'TS304'], errors=[] |
| active W_struct cross-interface block norms | PASS | true | {"TS101": {}, "TS102": {}, "TS103": {}, "TS104": {}, "TS201": {"G_AB\|G_BC": 0.00020759561652404902}, "TS202": {"G_AB\|G_BC": 0.00017645627404544166}, "TS203": {"G_AB\|G_BC": 0.00013493715074063185}, "TS204": {"G_AB\|G_BC": 0.00015569671239303677}, "TS301": {"G_AB\|G_AD": 0.00012666424120484835, "G_AB\|G_BC": 0.00020759561652404902, "G_AB\|G_DB": 0.00012186771516689728, "G_AD\|G_BC": 6.733453794302e-05, "G_AD\|G_DB": 7.904416487002694e-05, "G_BC\|G_DB": 0.0001103934780682265}, "TS302": {"G_AB\|G_AD": 0.0001076646050241211, "G_AB\|G_BC": 0.00017645627404544166, "G_AB\|G_DB": 0.00010358755789186268, "G_AD\|G_BC": 5.7234357251567e-05, "G_AD\|G_DB": 6.71875401395229e-05, "G_BC\|G_DB": 9.383445635799252e-05}, "TS303": {"G_AB\|G_AD": 8.233175678315143e-05, "G_AB\|G_BC": 0.00013493715074063185, "G_AB\|G_DB": 7.921401485848322e-05, "G_AD\|G_BC": 4.376744966296301e-05, "G_AD\|G_DB": 5.137870716551752e-05, "G_BC\|G_DB": 7.175576074434723e-05}, "TS304": {"G_AB\|G_AD": 9.499818090363627e-05, "G_AB\|G_BC": 0.00015569671239303677, "G_AB\|G_DB": 9.140078637517296e-05, "G_AD\|G_BC": 5.0500903457265e-05, "G_AD\|G_DB": 5.928312365252021e-05, "G_BC\|G_DB": 8.279510855116987e-05}} |
| topology_step independent LCP oracle registration | PASS | true | oracle_steps=['TS101', 'TS102', 'TS103', 'TS104', 'TS201', 'TS202', 'TS203', 'TS204', 'TS301', 'TS302', 'TS303', 'TS304'], solved_steps=['TS101', 'TS102', 'TS103', 'TS104', 'TS201', 'TS202', 'TS203', 'TS204', 'TS301', 'TS302', 'TS303', 'TS304'] |
| field_dictionary covers actual CSV schema and TopologyStepSpec | PASS | true | missing=[] |
| object_file_map formal objects and files | PASS | true | missing_objects=[] |
| synthetic-data truthfulness boundary | PASS | true | data_nature=SYNTHETIC_NUMERICAL_CONSISTENCY_CASE, engineering_claim_allowed=False |
| stage measurement update required files | PASS | true | missing=[] |
| checkpoint route relation and explicit source step | PASS | true | checkpoints=['MCP_AFTER_ABC'], invalid=[] |
| measurement/config/checkpoint foreign keys | PASS | true | measurements=['MEAS_GAP_AB_AFTER_RELEASE', 'MEAS_FORCE_AB_AFTER_RELEASE'], observations=['MEAS_GAP_AB_AFTER_RELEASE', 'MEAS_FORCE_AB_AFTER_RELEASE'], configs=['MUCFG_AFTER_ABC'] |
| measurement data_role and frozen parameter governance | PASS | true | only CALIBRATE/UPDATE state targets; parameter_update_allowed=false |
| posterior P/H/R/G_q matrix completeness | PASS | true | mapping_count=5, errors=[] |
| stage covariance transfer F/Q registration | PASS | true | transfers=4 |
| independent posterior and LCP oracle | PASS | true | oracle_rows=1, expected_status=POSTERIOR_ACCEPTED |
| oracle physical observation fields | PASS | true | missing=[] |
| post-LCP physical residual improvement | PASS | true | raw=0.691859980228161->9.323377868711058e-06; weighted=54150.25458676836->2.101731704278381e-05 |
| H independent physical finite difference | PASS | true | source=INDEPENDENT_GLOBAL_LCP_CENTRAL_FINITE_DIFFERENCE; method=CENTRAL_DIFFERENCE; stable=True |
| oracle uses unified observation extractor | PASS | true | extractor=core.stage_measurement_update.extract_observation_vector |
| parameter SMS Cn W_struct frozen package hash | PASS | true | before=d4c3b204faadef449662341c9c8d39b1688eebd8a43fd1ce83211d4623fdf16b; after=d4c3b204faadef449662341c9c8d39b1688eebd8a43fd1ce83211d4623fdf16b |
| single posterior global LCP call | PASS | true | resolve_lcp_call_count=1 |
| posterior covariance trace reduction | PASS | true | trace=0.000625->0.00022627875879919823 |
| measurement update objects registered | PASS | true | missing=[] |
| measurement truthfulness boundary | PASS | true | data_nature=SYNTHETIC_NUMERICAL_CONSISTENCY_CASE, measurement_data_nature=SYNTHETIC_NUMERICAL_CONSISTENCY_CASE, engineering_claim_allowed=False |
| rolling prediction required files | PASS | true | missing=[] |
| rolling plan primary key and active row | PASS | true | plans=['ROLL_PLAN_ABC_TO_ABCD'] |
| explicit virtual SMS sample cardinality | PASS | true | declared=5, rows=5 |
| component layout and coefficient completeness | PASS | true | component_orders=[0, 1], counts={'VSMS_COMBO': 2, 'VSMS_M1_NEG': 2, 'VSMS_M1_POS': 2, 'VSMS_M2_POS': 2, 'VSMS_REF': 2} |
| single zero reference SMS sample | PASS | true | reference_ids=['VSMS_REF'], values=[0.0, 0.0] |
| sample assignment keys unique and complete | PASS | true | assignment_rows=5 |
| G_SMS manifest coverage and column layout | PASS | true | G_SMS_P_D_OP_TS301:(12, 2), G_SMS_P_D_OP_TS302:(12, 2), G_SMS_P_D_OP_TS303:(12, 2), G_SMS_P_D_OP_TS304:(12, 2) |
| synthetic descriptive truth boundary | PASS | true | probability=false, engineering=false, explicit deterministic samples |
| direct SMS final-state boolean semantics | PASS | true | tokens=['false'], values=[False] |
| plan checkpoint topology and step linkage | PASS | true | plan_topology=TOPOLOGY_STEP_MIN_CASE, checkpoint_topology=TOPOLOGY_STEP_MIN_CASE, plan_step=TS205, checkpoint_step=TS205 |
| production static rolling quality gates | PASS | true | pass=45/45 |
| source package immutable during rolling run | PASS | true | before=6db1f993c031861d818902b091d578349bbfe38f6e328be26ed16d693e29db6f, after=6db1f993c031861d818902b091d578349bbfe38f6e328be26ed16d693e29db6f |
| accepted posterior is formal source | PASS | true | state_id=STATE_SAMPLE_001_TS205_POSTERIOR |
| runtime posterior source linkage closure | PASS | true | {"actual_source_state_checkpoint_id": "MCP_AFTER_ABC", "actual_source_state_id": "STATE_SAMPLE_001_TS205_POSTERIOR", "actual_source_state_role": "POSTERIOR", "actual_source_state_sample_id": "SAMPLE_001", "checkpoint_topology_id": "TOPOLOGY_STEP_MIN_CASE", "checkpoint_topology_step_id": "TS205", "failure_reasons": [], "measurement_update_checkpoint_id": "MCP_AFTER_ABC", "measurement_update_id": "MEAS_UPDATE_SAMPLE_001_MCP_AFTER_ABC", "measurement_update_posterior_state_id": "STATE_SAMPLE_001_TS205_POSTERIOR", "plan_source_step_id": "TS205", "plan_topology_id": "TOPOLOGY_STEP_MIN_CASE", "source_linkage_status": "PASS"} |
| all explicit samples complete with isolated branches | PASS | true | success=5, failure=0, baseline_success=5, baseline_failure=0 |
| expected SMS application matrix complete | PASS | true | applications=20, expected=20 |
| independent oracle q decomposition | PASS | true | matched=40/40 |
| independent oracle global LCP solutions | PASS | true | matched=40/40, one_call_per_step=true |
| oracle implementation independence declarations | PASS | true | active-set enumeration and direct KCP formula are independent |
| independent KCP oracle agreement | PASS | true | rows=30, max_abs_error=1.021e-12 |
| reference sample has zero virtual SMS correction | PASS | true | reference_count=1 |
| non-reference SMS sensitivity is observable | PASS | true | distinct_kcp_vectors=5 |
| coupled W_struct cross-interface blocks preserved | PASS | true | checked_steps=20, max_norm=2.075956e-04 |
| KCP contribution ledger and double-count gates | PASS | true | samples=5 |
| runtime direct SMS aggregation semantics | PASS | true | included=False, action=ADD_DIRECT_SMS_CONTRIBUTION |
| direct SMS contribution ledger action | PASS | true | expected_action=ADD_DIRECT_SMS_CONTRIBUTION |
| predicted-cutoff comparison is complete | PASS | true | policy=POSTERIOR_AND_PREDICTED_CUTOFF, rows=15 |
| descriptive summary oracle and probability labels | PASS | true | kcp_ids=['KCP_END_GAP', 'KCP_STEP_AB', 'KCP_PARALLEL_BALANCE'] |
| runtime rolling quality gates | PASS | true | pass=52/52 |
| validation attachments match current rolling package | PASS | true | stored_status=PASS |
