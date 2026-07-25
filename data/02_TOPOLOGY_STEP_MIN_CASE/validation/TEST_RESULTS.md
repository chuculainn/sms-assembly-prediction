# topology_step package self-validation

- Package: `02_TOPOLOGY_STEP_MIN_CASE`
- Status: **PASS**
- Checks: 22/22 PASS
- Blocking failures: 0
- MatrixManifest rows: 156
- NPZ keys: 156

| Check | Status | Blocking | Detail |
|---|---:|---:|---|
| interface endpoint foreign keys | PASS | true | parts=4, interfaces=4 |
| fixture expectation: part_count | PASS | true | expected=4, actual=4 |
| fixture expectation: interface_count | PASS | true | expected=4, actual=4 |
| fixture expectation: topology_step_count | PASS | true | expected=13, actual=13 |
| serial paths from actual part-interface graph | PASS | true | [["P_A", "P_B", "P_C"]] |
| parallel direct and bridge paths from actual graph | PASS | true | [{"endpoints": ["P_A", "P_B"], "paths": [["P_A", "P_B"], ["P_A", "P_D", "P_B"]], "path_status": [true, true]}] |
| contact point foreign keys and per-interface grouping | PASS | true | {"G_AB": 3, "G_AD": 3, "G_BC": 3, "G_DB": 3} |
| fixture expectation: contact points per interface | PASS | true | expected=3, actual={"G_AB": 3, "G_AD": 3, "G_BC": 3, "G_DB": 3} |
| vector layout contiguous, unique and data-linked | PASS | true | dimension=12, intervals=[(0, 2), (3, 5), (6, 8), (9, 11)] |
| fixture expectation: vector dimension | PASS | true | expected=12, actual=12 |
| topology route IDs, order and parent chain | PASS | true | steps=['TS000', 'TS101', 'TS102', 'TS103', 'TS104', 'TS201', 'TS202', 'TS203', 'TS204', 'TS301', 'TS302', 'TS303', 'TS304'] |
| topology route part/interface foreign keys | PASS | true | all activated/deactivated IDs resolve |
| same-step multi-interface activation from route data | PASS | true | expected_min=1, actual=['TS301'] |
| MatrixManifest and NPZ key set | PASS | true | manifest=156, npz=156, unique=156 |
| MatrixManifest shape, dtype and row/column layout | PASS | true | errors=[] |
| precomputed topology operator completeness and identity | PASS | true | solved_steps=['TS101', 'TS102', 'TS103', 'TS104', 'TS201', 'TS202', 'TS203', 'TS204', 'TS301', 'TS302', 'TS303', 'TS304'], errors=[] |
| active W_struct cross-interface block norms | PASS | true | {"TS101": {}, "TS102": {}, "TS103": {}, "TS104": {}, "TS201": {"G_AB/G_BC": 0.00020759561652404902}, "TS202": {"G_AB/G_BC": 0.00017645627404544166}, "TS203": {"G_AB/G_BC": 0.00013493715074063185}, "TS204": {"G_AB/G_BC": 0.00015569671239303677}, "TS301": {"G_AB/G_AD": 0.00012666424120484835, "G_AB/G_BC": 0.00020759561652404902, "G_AB/G_DB": 0.00012186771516689728, "G_AD/G_BC": 6.733453794302e-05, "G_AD/G_DB": 7.904416487002694e-05, "G_BC/G_DB": 0.0001103934780682265}, "TS302": {"G_AB/G_AD": 0.0001076646050241211, "G_AB/G_BC": 0.00017645627404544166, "G_AB/G_DB": 0.00010358755789186268, "G_AD/G_BC": 5.7234357251567e-05, "G_AD/G_DB": 6.71875401395229e-05, "G_BC/G_DB": 9.383445635799252e-05}, "TS303": {"G_AB/G_AD": 8.233175678315143e-05, "G_AB/G_BC": 0.00013493715074063185, "G_AB/G_DB": 7.921401485848322e-05, "G_AD/G_BC": 4.376744966296301e-05, "G_AD/G_DB": 5.137870716551752e-05, "G_BC/G_DB": 7.175576074434723e-05}, "TS304": {"G_AB/G_AD": 9.499818090363627e-05, "G_AB/G_BC": 0.00015569671239303677, "G_AB/G_DB": 9.140078637517296e-05, "G_AD/G_BC": 5.0500903457265e-05, "G_AD/G_DB": 5.928312365252021e-05, "G_BC/G_DB": 8.279510855116987e-05}} |
| topology_step independent LCP oracle registration | PASS | true | oracle_steps=['TS101', 'TS102', 'TS103', 'TS104', 'TS201', 'TS202', 'TS203', 'TS204', 'TS301', 'TS302', 'TS303', 'TS304'], solved_steps=['TS101', 'TS102', 'TS103', 'TS104', 'TS201', 'TS202', 'TS203', 'TS204', 'TS301', 'TS302', 'TS303', 'TS304'] |
| field_dictionary covers actual CSV schema and TopologyStepSpec | PASS | true | missing=[] |
| object_file_map formal objects and files | PASS | true | missing_objects=[] |
| synthetic-data truthfulness boundary | PASS | true | data_nature=SYNTHETIC_NUMERICAL_CONSISTENCY_CASE, engineering_claim_allowed=False |
| validation attachments match current package | PASS | true | matrix_manifest=156, npz=156, stored_status=PASS |
