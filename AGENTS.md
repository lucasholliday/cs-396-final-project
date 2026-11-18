# Project instructions

Read `prompt.md` completely, then `PROJECT_STATUS.md` and `DATA_ISSUES.md` before resuming.
Raw `Club Data/`, `Video Data/`, `Watch Data/`, and `trackman_shot_labels.xlsx` are immutable.
Never publish recordings, push Git, or overwrite original labels without authorization.
Do not train until critical shot identity, timing, and label ambiguities have been clarified by the user.
Session names are association evidence, not proof of individual shot alignment. Preserve provisional IDs and confidence.
Keep shot windows and all modalities in the same evaluation fold; fit preprocessing within folds.
Separate pre-impact prediction from post-impact estimation; never use TrackMan outcomes as inference inputs.
Save code, configurations, provenance, measured results, and cached intermediates in this workspace.
Update status, issues, experiment log, and reproduction commands at each milestone and before stopping.
Do not claim a completed experiment without executed results. Mark unperformed checks explicitly.
Before Git operations verify the repository root is this workspace. A local repository on `main` was initialized with approval; an unrelated home-directory repository also exists and must never be staged into.

## Resume
Follow the next three tasks in `PROJECT_STATUS.md`. Record user clarifications in `DATA_ISSUES.md` before resolving manifest associations. The initial audit is read-only and does not establish physical shot identities.

## Scientific scope of current artifacts
`annotations/shot_review.csv` accepts shot identity/order for whole-swing benchmarks, not precise impacts or point localization. Ball CSVs are unvalidated detector proposals and must not enter benchmark inputs until independently reviewed. Video baseline features use watch-assisted clips and include follow-through: never label these as autonomous camera-only or practice-swing predictions. Current nested carry results are primary; fixed-grid minima are retrospective. Club acceleration is saturated and excluded from initial predictors. Preserve failed methods and cohort sizes.

Use `.venv/bin/python`; the isolated environment avoids incompatible system binary packages. Read reproduction sequence in README. Re-running initial audit resets enriched manifest; subsequent event/alignment/feature steps restore it, with source-hash-checked annotations. Do not rerun costly decoding when cache signatures match.

Early-window ablation is not verified pre-impact: session offsets differ, and actual practice swings are absent. AI native-frame point review includes explicit uncertainty and needs a second reviewer; six close points do not validate full tracks. Viewer top proposals are not approved phase labels. Current research delivery status is in reports/research_summary.md; do not describe the long-term product as complete.
