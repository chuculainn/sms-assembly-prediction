# 03_STAGE_MEASUREMENT_UPDATE_MIN_CASE

This package is a deterministic synthetic numerical-consistency case for a two-dimensional low-dimensional stage-state posterior update. TS205 is a non-solving MEASURE step whose checkpoint explicitly uses TS204 as its mechanical source. Two synthetic process measurements drive a one-pass linear Gaussian update with Joseph covariance, an explicit state-to-q mapping, and one coupled global LCP re-solve. The accepted posterior is propagated into TS301-TS304 with explicit identity F and zero Q matrices.

`engineering_claim_allowed=false`. The measurements are not factory data, the sensitivities are not online FE derivatives, and the package does not validate engineering accuracy, identify parameters, or update SMS/Cn/Ct/mu/beta_r/joint stiffness.
