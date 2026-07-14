# P0 — Pre-architecture hardening

P0 blocks checkpoint 22.3.2b until CP22.3.2b reports explicit evidence for:

- event, magnitude and direction metrics by series type and horizon;
- MC-Dropout coverage, width and interval score by type/horizon;
- an independently trained, same-seed residual enabled/disabled ablation;
- early-stopping patience diagnosed from the productive objective curve;
- a version-locked, independently executable validation environment.

The component comparison inside one fitted residual model is not presented as a
retrained ablation. Productive MAC3 runs must preserve the same split, seed and
candidate and vary only `USE_LOCAL_RESIDUAL_DECODER`.
