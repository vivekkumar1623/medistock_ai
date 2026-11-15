# Architecture Decision Record

Every non-trivial decision lives here with its reason and trade-off. This file
is the skeleton of the technical report and the interview cheat-sheet: if you
cannot point to an entry that justifies a choice, the choice is not defensible
yet.

Format: **D-XXX — title**, then Context / Decision / Consequences / Revisit.

---

## Phase 1 — Foundation

### D-001 — Repository root is `medistock-ai/`, not the workspace folder
**Context.** The workspace directory is named `Project Algions AI`; the graded
artifact is a repository called `medistock-ai`. Mixing the two produces paths
like `Project Algions AI/app/...` in the report and clone instructions.
**Decision.** All code lives under `medistock-ai/`. The parent folder is scratch space.
**Consequences.** Commands must be run from `medistock-ai/`. Fresh-clone
instructions are identical to the workspace layout except for one directory level.
**Revisit.** Never.

### D-002 — One artifact bundle is the only source of truth for serving
**Context.** The classic integration failure in this project type is the API and
the dashboard computing demand differently, so KPI cards disagree with charts.
**Decision.** `app/services/artifact_store.py` defines the `ArtifactBundle`
contract. API services and the dashboard both call `ArtifactStore.load()`. No
endpoint may compute a prediction from raw CSVs at request time.
**Consequences.** Training must write the bundle before the API can serve
(503 until then, by design). Adding a derived number means adding it to the
bundle, not recomputing it per view.
**Revisit.** No. This is the load-bearing decision of the whole build.

### D-003 — The bundle is checksummed, versioned, and horizon-locked
**Context.** "Load the saved model" is easy to fake: load a model, ignore the
feature spec, guess the horizon. That yields plausible but wrong numbers.
**Decision.** `manifest.json` records contract version, artifact version,
prediction origin, horizon, dataset fingerprint, library versions, and SHA-256
per file. `ArtifactStore.verify()` enforces all of it; a horizon that disagrees
with configuration is a hard error, not a warning.
**Consequences.** Startup costs one checksum pass (milliseconds for a few MB);
retraining must rewrite the manifest; moving a file without updating the
manifest fails loudly.
**Revisit.** If artifacts grow beyond a few hundred MB, make checksum
verification lazy/off-by-default behind a config flag.

### D-004 — Domain exceptions, translated once at the web edge
**Context.** Raising `HTTPException` inside services couples business logic to
FastAPI and makes the ML code untestable in isolation.
**Decision.** `app/core/exceptions.py` defines `MedistockError` subclasses with
`status_code` and `error_code`. `app/api/errors.py` is the only place that knows
about HTTP.
**Consequences.** Every failure returns one envelope:
`{"error": {"code", "message", "details"}}`. Frontend error handling is a single
branch. Tests assert domain exceptions without a web client.
**Revisit.** No.

### D-005 — Stub endpoints return 503, never fake data
**Context.** Stub-first development (dashboard written before the model exists)
usually tempts you to return `{"predicted_demand": 120}` to make the UI green.
That is fabricated output and violates the project's own hard rules.
**Decision.** Stub routes declare the real `response_model` and raise
`FeatureNotAvailableError` (503 `feature_not_available`) with the phase that will
implement them. The OpenAPI contract is complete from Phase 1; behaviour is honest.
**Consequences.** The dashboard must implement an error state early (good — the
master prompt requires error states). The Phase-N test "no fake numbers" is trivial.
**Revisit.** No.

### D-006 — Configuration is one typed object, env-prefixed, path-resolved
**Context.** Scattered `os.getenv` calls with relative paths break when the
process starts from a different working directory (IDE, Docker, Task Scheduler).
**Decision.** `app/core/config.py` is the only reader of the environment, uses
prefix `MEDITSTOCK_`, validates types/bounds, and resolves relative paths
against the repository root. `get_settings()` is cached.
**Consequences.** Tests call `get_settings.cache_clear()` after monkeypatching.
A typo in a variable name fails silently as "default used" — acceptable, because
`.env.example` is the documented contract and `pydantic` rejects bad *values*.
**Revisit.** Add a `Settings.model_config["validate_default"] = True` sweep if we
ever see a default that violates its own constraint.

### D-007 — Dependencies are range-pinned; exact versions recorded in the manifest
**Context.** Hard pins rot and block security updates; no pins make a "works on
my machine" demo impossible to reproduce.
**Decision.** Compatible-release ranges in `requirements.txt`; the actually
installed versions of pandas/numpy/scikit-learn/python are written into
`manifest.json` at training time.
**Consequences.** A reviewer can reconstruct the training environment from the
bundle even if `requirements.txt` has moved on. Breaking-major upgrades are
blocked by the upper bounds.
**Revisit.** After the final demo, freeze `requirements.lock.txt` with `pip freeze`
and document it as the submitted environment.

### D-008 — WAPE is computed as `sum(|e|) / sum(|y|)` with a documented zero-guard
**Context.** MAPE explodes on zero-demand days, which this dataset deliberately
contains. A metric that is undefined for ~20% of rows cannot be the headline.
**Decision.** Report MAE, RMSE and WAPE. WAPE denominator is `sum(|actual|)`;
rows where the denominator is 0 for an entire evaluation slice are reported as
`None` rather than 0.0 or infinity.
**Consequences.** The comparison table needs a "n/a" state. The report must say
why WAPE is weighted by volume (it is not a per-product average).
**Revisit.** Confirm implementation in Phase 5; this entry is the pre-registration.

### D-009 — No CORS middleware
**Context.** Streamlit calls the API server-side with `httpx`, not from the
browser. Adding CORS "just in case" widens the attack surface for nothing.
**Decision.** No CORS configuration. If a browser client is ever added, it gets
an explicit allow-list, never `*`.
**Consequences.** A browser-based client will fail until configured — acceptable.
**Revisit.** Only when a real browser client exists.

---

## Phase 4–9 — Modelling and serving

### D-010 — The API serves the bundle; the database is the catalogue and audit store
**Context.** Two stores of the same facts (SQLite and the bundle's parquet snapshots) invite
divergence — the classic "KPI card disagrees with chart" bug.
**Decision.** All *analytical* answers (forecast, expiry risk, coverage, recommendations) are
computed from the artifact bundle through one service stack. The database holds the
operational catalogue (products, sales, inventory, batches) and persists a scoring snapshot
for audit/history. A test asserts the two agree on stock and batch quantities.
**Consequences.** Serving has exactly one source of truth for numbers. The DB is not on the
hot path, so a read-only deployment could omit it. Persisted scores can drift from the live
bundle after a retrain until re-persisted — documented in `LIMITATIONS.md` §4.4.
**Revisit.** If the operational catalogue becomes the master (edits in the UI), the bundle
must be regenerated from it and the direction of truth flips.

### D-011 — Inventory levels are NOT demand-model features
**Context.** Only one inventory snapshot exists (at 2025-10-01). Training origins run from
2025-01-01, so stock at those origins is unknowable.
**Decision.** Exclude stock from the feature set. Stock enters the *decision* layer (expiry
risk, replenishment) only.
**Consequences.** A little signal is lost, and an entire leakage class is removed. A reviewer
asking "why isn't stock a feature?" gets a principled answer instead of a shrug.
**Revisit.** Only when historical stock snapshots exist.

### D-012 — The serving layer imports `ml.features`
**Context.** The standard alternative is to re-implement feature construction for inference.
That is the canonical source of train/serve skew: the model receives slightly different
inputs and degrades silently.
**Decision.** `app/services/forecast_service.py` imports `ml.features.build`. The same code
builds training rows and serving rows.
**Consequences.** The API package depends on the `ml` package. Accepted deliberately: sharing
the implementation is the only way to make skew structurally impossible. It is also why the
architecture doc's "API must not import ml" rule has this single, documented exception.
**Revisit.** No. Splitting the feature module into a neutral package would be cosmetic.

### D-013 — Selection by mean rolling-origin WAPE, not one validation window
**Context.** The first implementation selected on a single 600-row validation window. It chose
Random Forest over HistGradientBoosting by **0.001 WAPE** — indistinguishable from noise —
while rolling-origin folds consistently favoured HGB.
**Decision.** Select by the mean WAPE of three expanding-window folds that end before the test
period. The code asserts those folds never include test origins.
**Consequences.** Selection is more stable and the reasoning is auditable
(`selection_scores` in `metrics.json`). Cost: 3× more fits and a selection statistic that is
less familiar than "validation WAPE".
**Revisit.** With more origins, use more folds (5–8) and report the fold variance.

### D-014 — One-at-a-time ablation instead of SHAP
**Context.** SHAP adds a dependency, a background-sampling design choice, and a computation
whose assumptions are hard to defend in a 10-minute viva.
**Decision.** Per-prediction attribution replaces one feature at a time with the population
median/mode and records the change in forecast. Each number is a real model output
difference. The method is labelled in the API (`one_at_a_time_ablation_vs_population_reference`)
and the dashboard states it is not a Shapley value.
**Consequences.** No new dependency, exact for the model, fast (29 predictions per product).
Limitation: contributions are not additive and are measured relative to the population, not
to a model expectation — stated wherever they are shown.
**Revisit.** If additive attributions are required for a regulatory narrative, add SHAP with
a documented background sample.

---

## Open decisions (owner: later phases)

| ID | Question | Decides in | Default if unresolved |
|---|---|---|---|
| O-01 | Seasonal-naive window: 7-day or 30-day naive? | Phase 4 | 30-day seasonal naive (matches horizon) |
| O-02 | Rolling-origin folds: how many, what spacing? | Phase 4 | 4 folds, 30-day step |
| O-03 | Explainability: SHAP or permutation importance + manual worked example? | Phase 7 | Permutation importance (Tier-3 cut rule) |
| O-04 | Safety stock formula: `z * sigma_d * sqrt(L)` or service-level table? | Phase 7 | `z * sigma_d * sqrt(L)` with `z=1.65` (95%) |
| O-05 | Alembic migrations vs `create_all` for the demo DB | Phase 8 | `create_all` + README note (pre-committed cut) |
| O-06 | Commit the trained bundle to git? | Phase 4 | Yes, if `< 10 MB`; else document regeneration |
