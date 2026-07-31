# Energy System Simulator Remediation and Expansion Prompts

This is the second-generation development prompt pack for the current `energy-system-simulator` repository. It is based on the audited `0.1.0` source archive, not on an assumed future implementation.

The current repository is a working deterministic mixed-integer baseline with one solar plant, one wind farm, one thermal unit, one battery, imports, aggregate delivery losses and capacity, reporting, and a standalone DC power-flow utility. The audited example solves successfully, preserves energy balance, and passes the installed test suite. It still has release-blocking defects and has not implemented the broader roadmap.

This document therefore has three layers:

1. **FIX prompts** correct the exact defects identified in the final audit and produce a trustworthy `0.1.1` or `0.2.0` baseline.
2. **CORE prompts** generalise the model into a research-grade multi-asset electricity-system simulator.
3. **ADV prompts** extend the repository into security, multi-energy, climate, infrastructure-planning, and operational-service domains.

Do not run all prompts as one task. Give a coding agent one major prompt at a time, review the mathematical formulation and diff, and merge only after the milestone gate passes.

## Audited starting state

| Area | Current result | Required action |
|---|---|---|
| Core MILP | Feasible and optimal for the 336-hour example | Preserve while fixing terminal semantics |
| Tests | 22 passed in an installed environment | Add missing regression and stress cases |
| Coverage | Approximately 84.8% branch-aware | Raise coverage around formulation and status handling |
| Balance | Residuals near numerical precision | Keep explicit reconciliation gates |
| Licensing metadata | Unresolved owner and contact placeholders | Block publication until resolved |
| Version reporting | `0.0.0+unknown` from an uninstalled source tree | Establish one authoritative version source |
| Minimum up/down | Initial residual obligations implemented | Add terminal residual obligations or a terminal policy |
| Configuration | Typed dataclasses but fixed single-asset schema | Reject unknown fields, then migrate to versioned portfolios |
| Network | Aggregate delivery model plus standalone DC power flow | Clarify semantics, then integrate nodal optimisation |
| Documentation | Baseline audit mixes historical and current statements | Mark snapshots and publish current model status |
| Roadmap | Prompts were not included in the uploaded repository | Commit this file and maintain milestone status |

## How to use the prompts

1. Create a branch using the branch name suggested by the selected prompt.
2. Paste the **Global implementation contract**, followed by exactly one major prompt.
3. Require the coding agent to inspect the current branch before proposing changes.
4. Reject changes that weaken constraints, hide non-optimal results, remove tests, or alter units without an explicit mathematical justification.
5. Require a small deterministic reproducer for every defect.
6. Merge FIX-00 through FIX-10 before CORE-01.
7. Treat ADV prompts as optional model families. They are not all required for a defensible version 1.0.

## Global implementation contract

Prepend this contract to every major implementation prompt.

```text
You are extending the existing Energy System Simulator repository. Inspect the current repository before editing it. Work from the actual implementation and preserve supported behaviour unless this prompt explicitly changes it.

Engineering contract
- Keep source code, configuration, documentation, comments, tests, CLI text, and commit messages in English.
- Use Python 3.11 or later and retain the existing Poetry-based workflow unless the prompt explicitly changes packaging.
- Prefer mathematically explicit, physically interpretable, deterministic models. Do not introduce machine learning merely to forecast or approximate a component that can be modelled transparently.
- Keep strict type annotations on public and internal interfaces. Add concise docstrings where behaviour, units, or semantics are not obvious.
- Preserve the `src/` package layout. Keep ingestion, domain types, formulation, solver adapters, simulation orchestration, accounting, and reporting separate.
- Express units in names whenever practical: `_mw`, `_mwh`, `_hours`, `_eur_per_mwh`, `_tonnes_per_mwh`, and similar.
- Never silently clip invalid inputs, ignore unknown configuration fields, replace missing data with zero, or present a missing incumbent as a valid solution.
- Avoid TODO-only code, mock production paths, dead interfaces, and broad rewrites that discard working functionality.
- Keep dependencies narrow. Explain and justify every new runtime dependency.
- Do not add a web application, distributed system, Kubernetes, database, machine-learning framework, or cloud deployment unless the selected prompt explicitly asks for it.

Mathematical contract
- Define every set, index, parameter, decision variable, objective term, and constraint added or changed.
- State whether each power quantity is generator-side, bus-injection-side, network-flow-side, or end-user-side.
- Check dimensional consistency for arbitrary configured time-step duration.
- Preserve exact logical relationships for commitment, startup, shutdown, charge modes, scenario selection, and investment decisions.
- State and test initial and terminal conditions. Do not allow finite-horizon boundary effects to remain implicit.
- Add physical, energy, cost, emissions, and settlement reconciliation checks as applicable.
- Distinguish optimal, feasible non-optimal, infeasible, unbounded, numerical failure, interrupted, and missing-solution states.
- Do not loosen feasibility tolerances to make tests pass without numerical evidence.

Testing contract
- Add a regression test that fails on the pre-change implementation for every corrected defect.
- Add normal, boundary, invalid-input, and invariant tests for each new capability.
- Use small deterministic mathematical fixtures with hand-computable expected results.
- Keep the example integration test, but do not rely on the example as the only validation case.
- Run the repository's supported verification commands. At minimum report pytest, Ruff, strict mypy, package build, and one clean-install smoke test when those tools are available.
- Do not delete or weaken a test merely because the new implementation fails it. Explain intentional semantic changes and replace obsolete tests with stronger ones.

Documentation and delivery contract
- Before coding, summarise the relevant current architecture, confirmed defect or capability gap, equations affected, and expected files to change.
- Update mathematical documentation, configuration documentation, output schema, README, changelog, and migration notes where applicable.
- Add or update a milestone record in `DEVELOPMENT_PROMPTS.md` or the repository roadmap.
- At completion report: files changed, equations and semantics, migration impact, commands executed, exact test results, benchmark impact, unresolved limitations, and the next safe milestone.
- Do not claim completion when a required command could not be run. State the limitation precisely.
```

## Completion response required from the coding agent

Require this response format after every major prompt:

```text
1. Baseline inspected
2. Defect or capability implemented
3. Mathematical formulation
4. Configuration and schema changes
5. Files changed
6. Regression and invariant tests added
7. Commands executed and exact results
8. Runtime and formulation-size impact
9. Backward-compatibility impact
10. Known limitations
11. Recommended next prompt
```

## Roadmap status

| ID | Milestone | Suggested branch | Gate | Status |
|---|---|---|---|---|
| FIX-00 | Preserve audited baseline and roadmap | `chore/audit-baseline-gate` | Required | Complete |
| FIX-01 | Resolve legal and citation metadata | `fix/release-metadata` | Required before publication | Complete |
| FIX-02 | Clean-checkout packaging and version authority | `fix/package-versioning` | Required | Complete |
| FIX-03 | Terminal commitment obligations | `fix/terminal-commitment` | Required | Complete |
| FIX-04 | Missing duration and initial-state tests | `test/time-state-regressions` | Required | Complete |
| FIX-05 | Solver termination regression suite | `test/solver-statuses` | Required | Complete |
| FIX-06 | Startup and shutdown ramp semantics | `fix/transition-ramps` | Required | Complete |
| FIX-07 | Strict configuration and unknown fields | `fix/config-strictness` | Required | Complete |
| FIX-08 | Numerical tolerance policy | `refactor/numerical-policy` | Required | Complete |
| FIX-09 | Network utility semantics | `fix/network-semantics` | Required | Not started |
| FIX-10 | Stress example and baseline release gate | `release/0.1.1` | Required | Not started |
| CORE-01–CORE-24 | Research-grade electricity-system roadmap | See each prompt | Sequential by dependency | Not started |
| ADV-01–ADV-15 | Optional post-core model families | See each prompt | Select by research question | Not started |

---

# FIX-00 — Preserve the audited baseline and commit the roadmap

**Suggested branch:** `chore/audit-baseline-gate`

**Objective:** Turn the current audited state into a reproducible reference before any correction changes results.

```text
Apply the global implementation contract.

Create an immutable baseline record for the current audited 0.1.0 repository. Do not change optimisation semantics in this prompt.

Tasks
1. Add this prompt pack as `DEVELOPMENT_PROMPTS.md` at the repository root.
2. Create `docs/audits/final-audit-0.1.0.md` recording the observed baseline:
   - example periods, objective, renewable share, emissions, imports, and unserved energy;
   - variable counts by type, constraint count, and matrix non-zero count;
   - test count and measured coverage;
   - maximum source balance, delivered-demand, battery-state, and objective-reconciliation residuals;
   - known defects from FIX-01 through FIX-09.
3. Store expected numerical values in one versioned machine-readable fixture used only for regression comparison. Use tolerances rather than brittle string comparisons.
4. Add a `baseline` verification command or Make target that runs the example and compares it with the fixture.
5. Make output ordering deterministic. Do not compare generated timestamps or machine-specific paths as numerical model outputs.
6. Add a roadmap status table and mark only truly completed work as complete.
7. Add a policy explaining when a baseline fixture may be updated: only after a documented semantic change and reviewer approval.

Tests
- Baseline comparison passes on two consecutive runs.
- A deliberately modified expected objective fails with a clear message.
- Manifest fields that are intentionally non-deterministic are excluded from equality checks.

Acceptance criteria
- The current solver result can be reproduced and compared with one command.
- The audit is clearly labelled as the pre-remediation state.
- No mathematical result changes in this prompt.
```

# FIX-01 — Resolve licensing, citation, and release metadata

**Suggested branch:** `fix/release-metadata`

**Objective:** Prevent publication of an artefact with unresolved legal-owner and citation placeholders.

```text
Apply the global implementation contract.

Harden repository metadata without changing licence terms or offering legal interpretations.

Tasks
1. Search every tracked text file for unresolved tokens including `[FULL LEGAL NAME]`, `[CONTACT EMAIL]`, `TODO`, `TBD`, example domains, and placeholder author fields.
2. Replace owner and contact placeholders with values explicitly supplied through repository configuration or environment during this task. Do not invent a legal name or address. When the values are unavailable, make the validation fail and report the required inputs rather than guessing.
3. Make `scripts/validate_licensing.py` reject unresolved placeholders and inconsistent owner/contact metadata across `LICENSE`, `LICENSING.md`, `COMMERCIAL-LICENSE.md`, `NOTICE`, release metadata, and package metadata.
4. Complete `CITATION.cff` with a proper citation name, repository title, version, licence identifier, and release date when known. Validate it structurally.
5. Ensure `pyproject.toml`, README citation instructions, CITATION file, licence documents, and release metadata agree on project name and version.
6. Add a `release-metadata` verification target and run it in CI before package build.
7. Document that changing licence terms requires separate legal review. This prompt may correct placeholders and inconsistencies only.

Tests
- The current placeholder state is represented by a failing regression fixture.
- Each prohibited placeholder token causes validation failure.
- Mismatched owner, email, version, or licence identifier causes failure.
- A complete consistent fixture passes.

Acceptance criteria
- No unresolved release placeholder remains in tracked files.
- A future placeholder cannot pass CI.
- No licence clause is silently rewritten.
```

# FIX-02 — Establish clean-checkout execution and one version authority

**Suggested branch:** `fix/package-versioning`

**Objective:** Make supported development, source-tree, wheel, and manifest version behaviour explicit and reproducible.

```text
Apply the global implementation contract.

Fix packaging and version reporting without abandoning the `src/` layout.

Tasks
1. Decide and document the supported clean-checkout workflow. At minimum, a clean environment must be able to run `poetry install`, the test suite, the CLI, and the example.
2. Add Make targets or scripts for:
   - dependency installation;
   - tests in the Poetry environment;
   - editable-install smoke test;
   - wheel build and wheel-install smoke test;
   - complete verification.
3. Establish `pyproject.toml` as the authoritative project version unless a stronger standard mechanism is justified. Remove duplicate hard-coded version values or add an automated consistency test.
4. Implement a shared `get_package_version()` helper:
   - use installed package metadata when available;
   - use a deterministic source-tree fallback tied to the authoritative version;
   - never report `0.0.0+unknown` for a valid repository checkout;
   - preserve an honest unknown result for an arbitrary copied module without project metadata.
5. Use the helper in manifests, CLI `--version`, reports, and any citation generation.
6. Add CI that builds a wheel, installs it into a fresh environment, imports the package, runs `energy-sim --version`, validates a config, and executes a reduced simulation.
7. Do not hide packaging problems by depending only on `PYTHONPATH=src`. If pytest source-path configuration is added, retain the wheel smoke test.

Tests
- Installed package version equals the authoritative project version.
- Source-tree fallback returns the same version.
- A mismatched duplicate version fails a consistency test.
- Wheel installation can import all public modules and execute the CLI.

Acceptance criteria
- Manifests report the correct version in installed and source-tree development modes.
- The documented clean-checkout command works.
- Package-build failures are caught in CI.
```

# FIX-03 — Define and enforce terminal commitment obligations

**Suggested branch:** `fix/terminal-commitment`

**Objective:** Eliminate the finite-horizon bias that allows a thermal unit to start or stop without satisfying residual minimum up or down time after the final period.

```text
Apply the global implementation contract.

Correct terminal minimum up/down semantics in the single-unit commitment model.

Before coding
1. Reproduce the confirmed defect with a three-period fixture where a unit with a three-hour minimum up time starts in the final period.
2. Write the current transition and minimum-duration equations, including the initial residual treatment.
3. Compare the following terminal policies:
   - `forbid_incomplete_transitions`;
   - `carry_residual_obligations`;
   - `fixed_terminal_commitment`;
   - `terminal_cost_approximation`.
4. Select an explicit default suitable for a standalone finite-horizon run. Reserve carry-forward state for rolling-horizon work.

Implementation
1. Add a typed terminal commitment configuration with documented modes.
2. In strict standalone mode, prevent startups or shutdowns whose required minimum duration cannot be completed within the horizon.
3. In carry-forward mode, return terminal commitment state, output, consecutive on/off duration, and residual minimum up/down obligations as typed result state.
4. Validate incompatible terminal settings early.
5. Ensure transition, ramp, minimum-duration, and terminal equations are jointly consistent.
6. Add terminal-state fields to the manifest and summary.
7. Update `docs/model.md` with equations and a timeline example.

Tests
- Final-period startup defect is reproduced and then prevented in strict mode.
- Final-period shutdown with residual minimum down time is treated symmetrically.
- Carry-forward mode exports the correct residual hours.
- A transition exactly at the last feasible period remains allowed.
- Sub-hourly horizons use the documented conservative duration conversion.
- Existing example remains feasible and its changed result, if any, is explained.

Acceptance criteria
- No transition can evade minimum-duration requirements through the horizon boundary.
- The terminal policy is explicit in configuration and output metadata.
- Rolling-horizon state can be implemented later without redesigning result types.
```

# FIX-04 — Complete time-step and initial-state regression coverage

**Suggested branch:** `test/time-state-regressions`

**Objective:** Commit the missing tests for arbitrary time resolution and both initial commitment states.

```text
Apply the global implementation contract.

Add a compact mathematical regression matrix for duration, ramp, energy, and initial-state semantics. Change production code only when a new test reveals a confirmed defect.

Required fixtures
1. Equivalent 15-minute, 30-minute, and 60-minute cases with analytically consistent energy and cost totals.
2. Initially-on unit with a residual minimum up obligation.
3. Initially-off unit with a residual minimum down obligation.
4. Minimum durations that are exact multiples of the step and durations that require conservative ceiling.
5. Ramping expressed in MW/hour across every time resolution.
6. Battery state transition, throughput cost, import cost, curtailment penalty, lost-load cost, and emissions scaling by time-step duration.
7. Exact, minimum, cyclic, and free battery terminal modes.

Test design
- Use parameterised tests where the same invariant applies across resolutions.
- Hand-calculate expected values in test comments or helper functions.
- Assert decisions when the optimum is unique; otherwise assert invariant quantities and objective bounds.
- Avoid comparing large example outputs as a substitute for small proofs.

Acceptance criteria
- The committed suite covers 0.25 h, 0.5 h, and 1.0 h steps.
- Both residual initial up and residual initial down obligations are tested.
- All cost and energy terms demonstrate correct time scaling.
- Coverage increases specifically in duration and initial-condition branches.
```

# FIX-05 — Add a complete solver-termination regression suite

**Suggested branch:** `test/solver-statuses`

**Objective:** Prove that every solver termination state is mapped and reported honestly.

```text
Apply the global implementation contract.

Harden solver termination handling and test it independently from large optimisation cases.

Tasks
1. Create a small solver adapter boundary so SciPy result interpretation can be tested with typed result fixtures without monkey-patching unrelated internals.
2. Define domain statuses for:
   - optimal;
   - feasible incumbent with time or node limit;
   - infeasible;
   - unbounded;
   - infeasible or unbounded when the backend cannot distinguish;
   - numerical or solver error;
   - interrupted;
   - no incumbent available.
3. Keep raw backend status and message in diagnostics.
4. Require the explicit `allow_non_optimal_solution` policy before returning a feasible incumbent from a limit-reached solve.
5. Reject a limit-reached result with no incumbent even when non-optimal results are allowed.
6. Calculate objective bound, absolute gap, and relative gap only when mathematically defined. Handle zero and negative objective values safely.
7. Ensure reports and manifests never label a feasible incumbent as optimal.
8. Add deterministic unit tests using result fixtures for every status.
9. Add at least one real small infeasible formulation. Add a real unbounded test at the solver-adapter level if the production formulation is structurally bounded.
10. Add a deliberately tiny time-limit integration test only when it is stable across supported CI environments; otherwise test the mapping with fixtures and document why.

Acceptance criteria
- Every backend status has one documented domain mapping.
- Missing incumbent and feasible incumbent are distinguished.
- Non-optimal policy is enforced consistently in CLI, Python API, and reports.
- Tests cover optimal, limit reached, infeasible, unbounded, error, and missing-solution paths.
```

# FIX-06 — Make startup and shutdown ramp semantics explicit

**Suggested branch:** `fix/transition-ramps`

**Objective:** Validate or correct transition-ramp equations and prevent configurations that make legitimate starts or stops impossible without explanation.

```text
Apply the global implementation contract.

Review startup, shutdown, and normal ramp constraints as one coherent formulation.

Tasks
1. Write the current equations for normal ramp up/down, startup ramp, shutdown ramp, minimum stable output, and transitions from the initial state.
2. Construct hand-computable tests for:
   - off to minimum output startup;
   - on at minimum output to shutdown;
   - startup limited below minimum stable output;
   - shutdown limited below minimum stable output;
   - startup followed by normal ramping;
   - initially-on output transition at period zero.
3. Decide whether `startup_ramp_mw` and `shutdown_ramp_mw` represent:
   - maximum output in the transition period; or
   - maximum change from the adjacent period.
   Use one definition consistently and document it.
4. Add semantic validation. A configuration that cannot start because the startup limit is below minimum stable output must either be rejected or explicitly supported through a multi-period startup trajectory. Do not leave accidental infeasibility.
5. Add exact constraints preventing simultaneous startup and shutdown if not already implied strongly enough.
6. Include transition-ramp feasibility diagnostics in configuration validation and infeasibility guidance.
7. Update equations, configuration docs, and example comments.

Acceptance criteria
- Startup and shutdown tests demonstrate the selected interpretation.
- Invalid transition parameters fail before solver construction unless multi-period trajectories are intentionally supported.
- No large-M relaxation makes normal ramps ineffective.
- Initial and terminal transitions use the same physical semantics as internal transitions.
```

# FIX-07 — Reject unknown configuration fields and improve error paths

**Suggested branch:** `fix/config-strictness`

**Objective:** Prevent misspelled or obsolete YAML fields from being silently ignored.

```text
Apply the global implementation contract.

Make the current configuration parser strict before migrating to a portfolio schema.

Tasks
1. Define the allowed key set for every current configuration section and for the root mapping.
2. Reject unknown fields by default with an error containing the complete dotted path and a sorted list of allowed keys.
3. Detect likely misspellings and provide at most one safe suggestion using deterministic string similarity. Never auto-correct configuration.
4. Detect duplicate YAML keys. Use a safe loader strategy that raises rather than accepting the last duplicate silently.
5. Distinguish missing required keys, unknown keys, wrong scalar types, invalid ranges, and cross-field inconsistency.
6. Preserve path resolution relative to the configuration file and test Windows- and POSIX-style input safely.
7. Add a schema version field or reserve it explicitly for CORE-01. Unknown future versions must fail clearly.
8. Add CLI validation output suitable for humans and a structured JSON error mode suitable for automation.
9. Document deprecation policy for fields that will be replaced by the list-based schema.

Tests
- Misspelled `maximum_output_mw` fails and suggests the correct key.
- Unknown root and nested keys fail.
- Duplicate keys fail.
- Boolean values do not pass numeric validation.
- All example configurations remain valid.

Acceptance criteria
- No unrecognised YAML field is silently ignored.
- Errors identify the exact field path.
- Parser behaviour is deterministic and fully tested.
```

# FIX-08 — Consolidate numerical tolerances and residual policy

**Suggested branch:** `refactor/numerical-policy`

**Objective:** Replace declared-but-unused or scattered tolerances with one tested numerical policy.

```text
Apply the global implementation contract.

Audit every comparison, rounding operation, solver residual, integrality check, and report reconciliation.

Tasks
1. Inventory all numeric constants and literal tolerances in source and tests, including currently unused feasibility, integrality, and reporting constants.
2. Define a typed immutable numerical policy with separate meanings for:
   - primal feasibility in power equations;
   - energy reconciliation;
   - integrality interpretation;
   - objective reconciliation in EUR;
   - non-negativity cleanup for solver noise;
   - report display rounding.
3. Explain why absolute or relative tolerance is used for each quantity. Use scale-aware checks where system size can vary substantially.
4. Route invariant checks and result extraction through shared helpers. Do not use report-rounding tolerances to validate physics.
5. Permit clipping only for solver noise inside a documented tolerance and preserve the raw value in diagnostics when useful.
6. Remove unused constants or make them operational with tests.
7. Add residual summaries including maximum absolute residual, scale-normalised residual, affected period, and affected equation family.
8. Add tests near, below, and above each tolerance boundary.
9. Add a numerical-scaling note for very small and very large systems.

Acceptance criteria
- Every tolerance has one semantic purpose and at least one test.
- No unused declared tolerance remains.
- Physical validation remains stricter than display rounding.
- Failure messages identify the equation family and worst period.
```

# FIX-09 — Clarify DC power-flow and network-capacity semantics

**Suggested branch:** `fix/network-semantics`

**Objective:** Prevent users from mistaking the standalone DC power-flow calculation for capacity-constrained optimal power flow.

```text
Apply the global implementation contract.

Audit and clarify both current network representations: aggregate delivery constraints in dispatch and standalone DC power flow.

Tasks
1. Document the exact aggregate-network equations, including source-side transfer capacity, delivery loss, end-user demand, network-capacity shedding, and any ordering assumptions.
2. Review the standalone `dc_power_flow` utility. State clearly whether it:
   - solves bus angles for specified injections;
   - checks but does not enforce line ratings;
   - assumes a connected network and one slack bus;
   - omits losses and reactive power.
3. Rename functions, result fields, or documentation when current names imply optimisation or capacity enforcement that does not occur.
4. Add explicit post-solution overload diagnostics with line ID, MW flow, rating, utilisation, and overload amount.
5. Add validation for disconnected islands, duplicate line IDs, self-loops, zero or invalid reactance, missing slack bus, multiple slack buses, and net-injection imbalance.
6. Decide whether line overload should raise, warn, or return an invalid-state result. Make the policy explicit and configurable.
7. Do not implement integrated DC optimal power flow here; that belongs in CORE-08. Keep this prompt focused on correctness and semantics.

Tests
- A two-bus hand calculation matches the analytical flow.
- A three-bus case checks sign conventions.
- An overloaded line is detected but not falsely described as constrained.
- Disconnected and imbalanced cases fail precisely.

Acceptance criteria
- Public names and docs do not imply that standalone power flow enforces ratings.
- Aggregate and nodal network concepts are clearly distinguished.
- All network assumptions and invalid states are tested.
```

# FIX-10 — Add stress cases and cut the corrected baseline release

**Suggested branch:** `release/0.1.1`

**Objective:** Prove the corrected baseline under normal and stressed operation, then prepare a release without expanding to multiple assets.

```text
Apply the global implementation contract.

Create a deterministic stress-example suite and release gate for the corrected single-system baseline.

Required stress cases
1. Renewable surplus causing positive curtailment.
2. Import-constrained peak demand causing battery discharge.
3. Transfer-capacity bottleneck causing network-capacity shedding.
4. Generation shortage causing dispatch load shedding.
5. Thermal startup and shutdown with non-zero costs.
6. Battery charge and discharge across multiple periods with exact exclusivity.
7. Carbon price changing the relative use of imports and thermal generation when economically feasible.
8. A deliberately infeasible configuration caught before or by the solver with a precise status.
9. A terminal-commitment case exercising residual obligations.

Deliverables
- Add small scenario configurations and generated input fixtures under `examples/stress/` or a similarly clear structure.
- Add one command that runs all stress cases and writes a compact comparison table.
- Add quantitative expected checks for energy balance, cost decomposition, curtailment, shedding type, commitment transitions, and terminal state.
- Correct historical documentation. Rename the old baseline audit as a dated snapshot or add a prominent historical banner. Create a current `docs/model-status.md`.
- Update README capability and limitation tables.
- Build source distribution and wheel, install the wheel in a clean environment, run all supported examples, and validate release metadata.
- Release as `0.1.1` when changes are corrections only, or `0.2.0` if public semantics or schema change materially. Explain the version choice.

Acceptance criteria
- Every major decision and penalty variable is positive in at least one committed stress case.
- All cost and energy reconciliation checks pass.
- The package can be installed and executed from a wheel.
- No unresolved audit blocker remains.
- CORE-01 can start from a documented stable baseline.
```

---

# Core electricity-system development roadmap

The following CORE prompts retain and refine the original expansion roadmap. Execute them only after FIX-10 passes.

# CORE-01 — Typed configuration and domain schema

**Objective:** Replace the single-asset configuration shape with an extensible, validated schema while keeping the existing example usable.

```text
Apply the global engineering contract.

Refactor configuration and core domain types so the simulator can support portfolios of assets and multiple network representations without turning configuration parsing into untyped dictionary manipulation.

Design requirements
1. Introduce explicit identifiers for assets, buses, zones, fuels, and scenarios. Identifiers must be non-empty, unique within scope, stable in outputs, and validated before optimisation.
2. Define typed domain objects for:
   - simulation settings;
   - solver settings;
   - buses and lines;
   - renewable generators;
   - thermal generators;
   - storage units;
   - hydro units reserved for a later prompt;
   - import or interconnector resources;
   - demand nodes or sectors;
   - penalties and policy parameters;
   - paths and output settings.
3. Decide whether to remain with dataclasses plus explicit parsing or adopt a narrowly scoped validation library. Prefer no new dependency unless it materially reduces risk. Document the decision.
4. Replace the single `thermal`, `battery`, `solar`, and `wind` sections with list-based asset sections, but maintain backward compatibility for the 0.1 configuration through either:
   - a migration function; or
   - support for both schemas with a deprecation warning.
5. Add a version field to configuration files. Create `docs/configuration.md` and document schema versions and migration rules.
6. Validate cross-references:
   - every asset bus exists;
   - every line endpoint exists;
   - every demand series maps to a defined bus or aggregate system;
   - capacities and efficiencies are physically valid;
   - identifiers are unique;
   - initial states are within limits;
   - time-series keys exist in the input data contract.
7. Separate parsing, validation, and semantic cross-validation. Error messages must identify the exact path, for example `thermal_generators[1].maximum_output_mw`.
8. Add configuration serialisation to canonical YAML or JSON so a resolved configuration can be written into the output manifest.
9. Add a `energy-sim migrate-config` CLI command that converts a legacy example configuration to the new schema and preserves meaning.
10. Create at least three compact configurations:
   - backward-compatible single-system example;
   - two thermal units plus two renewable generators;
   - a deliberately invalid fixture used in tests.

Acceptance criteria
- Legacy configuration loads with a precise deprecation message or migrates deterministically.
- New list-based configurations are fully typed and cross-validated.
- Resolved configuration output is canonical and reproducible.
- No optimisation module reads raw YAML mappings.
- Configuration tests cover duplicate IDs, missing buses, invalid efficiencies, inconsistent initial states, unknown fields, and migration correctness.
```

---

---

# CORE-02 — Multi-asset portfolio architecture

**Objective:** Generalise the simulation engine around an asset portfolio before changing the optimisation mathematics substantially.

```text
Apply the global engineering contract.

Create a multi-asset portfolio architecture. This prompt should generalise generation and storage orchestration, but it should not yet implement the complete multi-unit commitment formulation requested in CORE-03.

Tasks
1. Introduce a small, explicit interface or protocol for time-dependent assets. Avoid a deep inheritance hierarchy. The architecture must make it easy to distinguish:
   - availability-only resources;
   - dispatchable resources;
   - intertemporal resources;
   - demand and network components.
2. Refactor solar and wind models so multiple named plants can be evaluated independently from their own configuration and input columns.
3. Return labelled tables with a stable column convention. Prefer a tidy internal representation with dimensions such as `timestamp`, `asset_id`, `variable`, and `value`, while allowing wide CSV exports for human use.
4. Add an `AssetRegistry` or equivalent that resolves assets, validates IDs and buses, and constructs model objects from configuration.
5. Separate renewable availability calculation from dispatch decisions. Persist available output by asset, used output by asset, and curtailment by asset.
6. Refactor the simulation engine into explicit stages:
   - load and validate time-series data;
   - resolve model configuration;
   - calculate exogenous availability and demand;
   - build optimisation inputs;
   - solve;
   - validate solution invariants;
   - calculate accounting metrics;
   - report outputs.
7. Remove assumptions in reporting that there is exactly one solar plant, one wind farm, one thermal unit, or one battery.
8. Add stable result-domain types rather than passing loosely specified DataFrames between all layers. DataFrames may remain the external tabular representation.
9. Add tests with at least two solar assets and two wind assets using distinct capacities and weather columns. Confirm that aggregation equals the sum of asset-level outputs.
10. Preserve a simple programmatic API for the current single-system example.

Acceptance criteria
- The engine calculates and reports multiple renewable assets by ID.
- Asset-level and aggregate renewable energy reconcile exactly within tolerance.
- Reporting code has no hard-coded dependency on a specific renewable asset count.
- Domain types make units and dimensions clear.
- The next prompt can introduce generator-indexed MILP variables without another engine rewrite.
```

---

---

# CORE-03 — Multi-unit commitment

**Objective:** Replace the single-thermal-unit formulation with a generator-indexed mixed-integer unit-commitment model.

```text
Apply the global engineering contract.

Implement multi-unit commitment for an arbitrary finite set of thermal generators. Preserve single-unit compatibility and all invariants from Prompt 01.

Mathematical scope
For each thermal unit g and period t, implement at least:
- commitment status `u[g,t]`;
- startup decision `y[g,t]`;
- shutdown decision `z[g,t]`;
- output `p[g,t]`;
- minimum and maximum stable output;
- ramp-up and ramp-down limits;
- startup and shutdown ramp limits;
- minimum up and minimum down duration;
- initial commitment, output, and elapsed state duration;
- variable, no-load, startup, shutdown, carbon, and optional fuel costs.

Implementation tasks
1. Replace fixed contiguous variable blocks with a general indexed-variable registry capable of named dimensions. It must expose deterministic indices and reversible extraction into labelled arrays or DataFrames.
2. Separate formulation building into focused components. The monolithic unit-commitment file should not continue growing without structure. Suggested modules include variables, bounds, balance constraints, thermal constraints, storage constraints, and objective construction.
3. Build sparse constraints efficiently without dense matrices.
4. Support heterogeneous generators with independent parameters.
5. Add optional must-run and availability-factor time series by generator.
6. Add planned derating and forced availability as an exogenous maximum-capacity multiplier in `[0,1]`.
7. Preserve imports, renewable curtailment, storage, and load shedding.
8. Export asset-level commitment, startup, shutdown, output, capacity available, capacity factor, cost, and emissions.
9. Add system-level metrics:
   - total starts and shutdowns;
   - online capacity;
   - unused committed capacity;
   - thermal fleet capacity factor;
   - generation by unit;
   - marginal versus inframarginal operating cost summaries.
10. Add tests for merit-order dispatch, minimum-output effects, ramp bottlenecks, startup-cost trade-offs, residual initial minimum up/down time, unavailable units, and identical-unit symmetry.
11. Add a two-unit example where a low-variable-cost high-startup-cost baseload unit competes with a higher-variable-cost flexible peaker.

Acceptance criteria
- Single-unit results remain compatible within numerical tolerance.
- Multi-unit outputs satisfy per-generator constraints and system balance.
- Tests demonstrate a case where the optimal solution keeps a unit online because restarting is more expensive.
- Tests demonstrate a peaker starting because the baseload unit is ramp-constrained.
- The formulation reports variable and constraint counts by component.
- Documentation includes the full generator-indexed equations.
```

---

---

# CORE-04 — Fuel, heat-rate, and emissions modelling

**Objective:** Replace constant thermal variable cost with transparent fuel and efficiency models.

```text
Apply the global engineering contract.

Extend thermal generation with explicit fuel consumption, piecewise-linear heat-rate curves, and richer emissions accounting.

Requirements
1. Introduce typed fuel definitions with fuel ID, price in EUR per thermal MWh or an explicitly documented alternative unit, lower heating value when needed, direct CO2 factor, and optional non-CO2 factors.
2. Allow each thermal generator to reference one fuel and define a piecewise-linear incremental heat-rate or input-output curve.
3. Formulate piecewise generation exactly using incremental segment variables. Do not use an opaque polynomial inside a linear MILP.
4. Enforce ordered segment usage where necessary. Explain whether convex segment costs make explicit ordering constraints redundant and test the chosen formulation.
5. Calculate:
   - fuel input MWh-thermal;
   - electrical efficiency by period;
   - fuel cost;
   - direct CO2 emissions;
   - optional methane, NOx, and SOx accounting as reported quantities;
   - carbon price cost;
   - startup fuel and startup emissions by startup category if configured.
6. Retain a constant marginal-cost compatibility mode for simple teaching examples.
7. Add hot, warm, and cold startup categories based on prior downtime. This requires explicit downtime-state tracking or a mathematically equivalent formulation. Keep the implementation auditable and document its size implications.
8. Add fuel-price time series support, with scalar fallback.
9. Add marginal cost diagnostics by generator and segment.
10. Include tests for segment dispatch, efficiency, fuel-cost reconciliation, carbon-price effects, startup-category selection, and time-varying fuel prices.

Acceptance criteria
- Fuel input and electrical output are dimensionally consistent.
- The sum of segment output equals generator output above any configured minimum-output block.
- Reported fuel and carbon costs reconcile with the objective.
- A test shows carbon price changing the merit order between two fuels.
- Documentation explains heat rate, efficiency, and why a piecewise-linear model is used.
```

---

---

# CORE-05 — Generalised storage portfolio

**Objective:** Support multiple storage technologies with explicit operating modes and degradation accounting.

```text
Apply the global engineering contract.

Generalise storage from one battery to a portfolio of named storage assets connected to buses or the aggregate system.

Required model
For each storage unit s and period t, support:
- charge power;
- discharge power;
- stored energy;
- charging and discharging efficiencies;
- standing self-discharge;
- minimum and maximum energy;
- independent charge and discharge power limits;
- exact mutually exclusive operating modes;
- initial and terminal conditions;
- availability time series;
- variable throughput cost.

Extensions
1. Support batteries and pumped storage using the same general energy-storage equations, while allowing technology-specific metadata.
2. Add optional minimum charge or discharge power when a mode is active.
3. Add optional ramp limits for charge and discharge power.
4. Add cycle and degradation metrics:
   - total charged and discharged energy;
   - equivalent full cycles;
   - round-trip losses;
   - average and maximum depth of discharge;
   - time at minimum and maximum state of charge;
   - throughput-based degradation cost.
5. Add a piecewise-linear degradation cost based on energy throughput or depth-of-discharge bands. Keep it optional and document the approximation.
6. Support fixed final, minimum final, cyclic, and free terminal states per asset.
7. Ensure storage cannot create energy and that self-discharge and efficiencies are applied over the correct time interval.
8. Add tests for multiple storage assets, self-discharge, exclusivity, terminal modes, power limits, energy limits, and degradation-cost trade-offs.
9. Add an example where a short-duration battery and long-duration pumped storage respond differently to a renewable surplus.

Acceptance criteria
- Storage energy reconciliation is reported by asset and system.
- Multiple storage technologies can operate simultaneously without column collisions.
- Exact mode prevents simultaneous charging and discharging even under negative prices.
- All storage objective components reconcile.
- The formulation remains linear mixed-integer.
```

---

---

# CORE-06 — Reservoir hydro

**Objective:** Add reservoir hydropower as an intertemporal dispatchable renewable resource.

```text
Apply the global engineering contract.

Implement reservoir hydropower and optional run-of-river generation. Keep the first hydro model linear and transparent.

Reservoir model requirements
For each reservoir h and period t, include:
- storage volume or energy-equivalent reservoir state;
- natural inflow;
- turbine release;
- spill;
- hydroelectric generation;
- minimum and maximum storage;
- turbine capacity;
- spill capacity if finite;
- initial and terminal reservoir conditions;
- optional environmental minimum release;
- optional evaporation or standing loss;
- water-balance constraints.

Modelling decisions
1. Choose either physical water units with a constant conversion coefficient or direct energy-equivalent storage. Support explicit units and document the choice.
2. Support constant turbine conversion efficiency in the first version. Design extension points for head-dependent efficiency but do not add nonlinear optimisation.
3. Add run-of-river plants whose available output depends on inflow and has no substantial intertemporal storage.
4. Add optional water value as a terminal or operating opportunity cost.
5. Add cascade support only if it can be implemented cleanly: upstream releases become delayed downstream inflows. Otherwise provide the typed data model and defer optimisation support explicitly.
6. Report inflow, release, spill, reservoir level, generation, capacity factor, water losses, and terminal value.
7. Add tests for water conservation, spill under full-reservoir conditions, drought dispatch, environmental flow, terminal storage, and competition with thermal generation.
8. Add a synthetic seasonal example showing reservoir conservation across low-price and high-price periods.

Acceptance criteria
- Water balance reconciles by reservoir and horizon.
- Hydro cannot generate without sufficient release or inflow.
- Terminal storage choices react correctly to water value.
- Spillage is non-negative and occurs only when economically or physically necessary.
- Model equations and units are documented.
```

---

---

# CORE-07 — Sector demand and demand response

**Objective:** Model consumption explicitly by sector and add mathematically controlled flexibility.

```text
Apply the global engineering contract.

Replace a single aggregate demand series with a demand portfolio while retaining an aggregate compatibility mode.

Demand representation
1. Support named demand entities or sectors, each assigned to a bus or aggregate zone.
2. Support fixed demand time series for residential, commercial, industrial, transport, and other configured sectors without hard-coding those names.
3. Distinguish:
   - inflexible demand;
   - curtailable voluntary demand;
   - shiftable demand;
   - deferrable tasks with energy requirements and time windows;
   - involuntary load shedding as a final reliability variable.
4. Implement demand response through linear constraints:
   - voluntary curtailment bounds and utility-loss cost;
   - load shifting with energy conservation over a configured window or horizon;
   - maximum upward and downward adjustment;
   - rebound or recovery where configured;
   - task energy completion before a deadline.
5. Prevent demand response from becoming free energy. Report baseline demand, adjusted demand, voluntary curtailment, shifted-in energy, shifted-out energy, task completion, and involuntary shedding separately.
6. Add sector-specific value of lost load and voluntary curtailment cost.
7. Add temperature-sensitive demand as an optional deterministic preprocessing model using transparent heating-degree and cooling-degree functions. Do not add machine learning.
8. Add electric-vehicle charging as deferrable demand with arrival, departure, charging-power, and required-energy constraints.
9. Add tests for energy-conserving shifts, deadline completion, unavailable windows, sector prioritisation under scarcity, and separation of voluntary and involuntary curtailment.
10. Add an example with residential demand, an industrial flexible load, and an EV fleet.

Acceptance criteria
- Aggregate demand equals the sum of demand entities before and after flexibility, with explicit adjustments.
- Shiftable demand conserves energy over the declared window.
- EV charging meets required energy when physically feasible and reports unmet task energy otherwise.
- Sector-specific scarcity costs produce the expected prioritisation.
- All demand adjustments are visible in outputs and cost reconciliation.
```

---

---

# CORE-08 — Integrated nodal DC optimal power flow

**Objective:** Integrate network constraints into dispatch rather than solving DC power flow only after the fact.

```text
Apply the global engineering contract.

Implement a nodal DC optimal-power-flow formulation integrated with unit commitment, renewables, storage, imports, hydro, and demand.

Network model
For each bus n, line l, and period t, include:
- voltage phase angle at non-slack buses;
- directed line flow;
- nodal power balance;
- line susceptance relation;
- symmetric thermal flow limits;
- one reference angle per connected network island;
- asset-to-bus mappings.

Requirements
1. Keep the existing aggregated network model as a supported mode.
2. Validate network topology before building the optimisation:
   - unknown buses;
   - self-loops;
   - duplicate line IDs;
   - non-positive capacities or susceptances;
   - disconnected components;
   - isolated buses with demand or assets.
3. Decide how to handle multiple connected islands. Either reject them with a clear error or create one slack bus per island. Document the decision.
4. Use consistent sign conventions and document them.
5. Integrate generator output, renewable use, storage charge/discharge, imports, demand response, hydro, and load shedding into nodal balances.
6. Add line-outage availability multipliers reserved for later contingency analysis.
7. Export bus angles, nodal injections, line flows, absolute and signed utilisation, congestion hours, and overload residual checks.
8. Add nodal load shedding rather than a single system-wide shedding variable.
9. Add tests for:
   - a two-bus constrained line causing local scarcity;
   - unconstrained network equivalence to copper-plate dispatch;
   - loop flows in a three-bus mesh;
   - disconnected network rejection;
   - flow sign and capacity limits;
   - asset relocation changing dispatch.
10. Add a compact three-bus example with renewable generation remote from demand and a congested corridor.

Acceptance criteria
- Every bus balances within tolerance in every period.
- Line flow equations and limits are satisfied.
- The unconstrained nodal model matches the aggregate copper-plate result within tolerance.
- Congestion can cause renewable curtailment in one bus and thermal generation or load shedding in another.
- The standalone `solve_dc_power_flow` utility is either reused coherently, repositioned as validation, or deprecated with documentation.
```

---

---

# CORE-09 — Operating reserves

**Objective:** Add reserve adequacy to unit commitment and dispatch.

```text
Apply the global engineering contract.

Implement operating-reserve products in the deterministic optimisation model.

Scope
1. Add upward and downward reserve variables by eligible asset and period.
2. Support at least:
   - spinning upward reserve from online thermal units;
   - downward reserve from online thermal units;
   - storage upward reserve from discharge headroom and stored energy;
   - storage downward reserve from charge headroom and available storage space;
   - optional demand-response reserve;
   - optional import reserve only when explicitly configured.
3. Define system reserve requirements as:
   - fixed MW;
   - percentage of demand;
   - percentage of renewable availability;
   - largest online contingency approximation;
   - additive combinations of the above.
4. Enforce capacity headroom, ramp deliverability, energy-duration, commitment, and network deliverability constraints where relevant.
5. Add reserve shortfall variables with a high explicit penalty rather than forcing generic infeasibility.
6. Add reserve procurement costs by asset and product.
7. Report procured reserve, requirement, shortfall, provider mix, reserve cost, and hours of inadequacy.
8. Add tests demonstrating:
   - additional commitment caused by reserve requirements;
   - storage reserve limited by state of charge;
   - ramp-limited reserve;
   - reserve shortfall under insufficient capacity;
   - zero requirement reproducing the prior solution.
9. Document the distinction between reserve capacity in MW and dispatched energy in MWh.

Acceptance criteria
- Reserve provision never exceeds physical headroom or energy deliverability.
- Reserve requirements reconcile by period.
- Zero-reserve configuration is backward compatible.
- Reserve costs reconcile with the objective.
- Results distinguish energy dispatch from held reserve.
```

---

---

# CORE-10 — Renewable model extensions

**Objective:** Improve renewable realism without obscuring the physics.

```text
Apply the global engineering contract.

Extend solar and wind availability models while preserving the current simple models as named options.

Solar requirements
1. Separate installed DC capacity, inverter AC capacity, module performance ratio, inverter efficiency, clipping, temperature derating, degradation, and availability.
2. Support tilted-plane irradiance only when the input data supplies it or through a documented deterministic transposition method. Do not invent missing meteorological data silently.
3. Support snow or soiling loss only as explicit input factors or transparent configured schedules.
4. Report DC potential, AC potential, clipping loss, temperature loss, other derating, and final availability.

Wind requirements
1. Support wind-speed adjustment from measurement height to hub height using a configurable power-law or logarithmic profile.
2. Support air-density correction from pressure and temperature when inputs are provided.
3. Support turbine-count and per-turbine rated capacity explicitly.
4. Allow tabulated manufacturer power curves with monotonicity and range validation.
5. Support wake and electrical losses as explicit factors, not hidden constants.
6. Report gross potential, wake loss, electrical loss, availability loss, high-wind shutdown, and final availability.

Cross-cutting requirements
1. Add asset availability and maintenance schedules.
2. Preserve deterministic operation and unit transparency.
3. Add tests against hand-calculated cases and curve boundary conditions.
4. Add documentation explaining which model is appropriate for teaching, planning, or engineering approximation.
5. Add example plots of power curves and solar derating decomposition.

Acceptance criteria
- Existing simple solar and wind modes remain available.
- Every derating is separately reported and reconciles from gross to net availability.
- Tabulated wind curves are validated and interpolated deterministically.
- No output silently exceeds configured AC or rated capacity.
```

---

---

# CORE-11 — Rolling-horizon simulation

**Objective:** Run long horizons efficiently while preserving intertemporal state consistency.

```text
Apply the global engineering contract.

Implement rolling-horizon optimisation for long simulations.

Requirements
1. Add configuration for:
   - optimisation window length;
   - implementation step length;
   - look-ahead overlap;
   - terminal treatment;
   - whether perfect foresight or forecast inputs are used.
2. At each window:
   - build the optimisation using the current state;
   - solve the full look-ahead window;
   - retain only the implementation segment;
   - transfer commitment status, elapsed up/down duration, output, storage state, reservoir state, deferrable-demand state, and other relevant state to the next window.
3. Avoid duplicated timestamps and ensure complete horizon coverage.
4. Preserve minimum up/down obligations across window boundaries.
5. Preserve storage and reservoir accounting across boundaries.
6. Support deterministic checkpoint files so an interrupted simulation can resume exactly.
7. Report per-window solver status, objective, runtime, gaps, transferred states, and any fallback behaviour.
8. Add an optional perfect-foresight full-horizon comparison for small fixtures.
9. Add tests for:
   - rolling horizon equivalent to full horizon when the window covers all periods;
   - state continuity;
   - startup near a boundary;
   - storage use near a boundary;
   - resume from checkpoint;
   - no missing or duplicated periods.
10. Document the horizon effect and why rolling-horizon solutions may differ from full-horizon optimum.

Acceptance criteria
- A one-year hourly synthetic case can run in bounded windows.
- State transitions are exact across windows.
- The result metadata records every subproblem status.
- Checkpoint resume reproduces an uninterrupted run.
- Boundary tests cover thermal, storage, hydro, and flexible demand states that exist at implementation time.
```

---

---

# CORE-12 — Outages and reliability simulation

**Objective:** Add probabilistic adequacy analysis around the deterministic dispatch model.

```text
Apply the global engineering contract.

Implement a Monte Carlo reliability layer for generation, storage, network, and import outages. The dispatch model itself remains optimisation-based; random sampling creates availability scenarios.

Requirements
1. Define typed outage models using forced-outage rate and mean time to repair, or transition probabilities derived from them.
2. Generate deterministic pseudo-random outage trajectories from an explicit seed.
3. Support independent and optionally grouped common-cause outages.
4. Apply outage states as time-varying availability multipliers for thermal units, renewable balance-of-plant, storage power converters, lines, and interconnectors.
5. Run sequential Monte Carlo simulations that preserve chronology and intertemporal states.
6. Compute at least:
   - loss-of-load probability;
   - loss-of-load expectation in hours/year;
   - expected unserved energy;
   - expected demand not served;
   - frequency and duration of scarcity events;
   - distribution and confidence intervals across replications;
   - contribution to unserved energy by outage type when attributable.
7. Add convergence diagnostics and confidence-interval stopping criteria as an optional mode.
8. Parallel execution may be added using the Python standard library, but deterministic seeds and result ordering must be preserved.
9. Add tests for seed reproducibility, zero-outage equivalence, complete asset outage, repair transitions, and confidence-interval calculations.
10. Add a compact reliability example comparing the base system with an additional peaker or battery.

Acceptance criteria
- Re-running with the same seed produces identical outage paths and metrics.
- Zero forced-outage rates reproduce deterministic availability.
- Reliability metrics use correct units and definitions.
- Failed replications are surfaced, not silently discarded.
- Documentation distinguishes deterministic load shedding from probabilistic adequacy estimates.
```

---

---

# CORE-13 — Scenario-based uncertainty and stochastic dispatch

**Objective:** Represent forecast uncertainty through explicit scenarios, not black-box prediction.

```text
Apply the global engineering contract.

Implement scenario-based stochastic dispatch as a separate model family. Preserve deterministic dispatch unchanged.

Scope
1. Define scenario sets with IDs, probabilities, and scenario-specific renewable availability, demand, import prices, inflows, and outages.
2. Validate that probabilities are non-negative and sum to one within tolerance.
3. Implement a two-stage stochastic formulation:
   - first-stage unit commitment decisions shared across scenarios for a configurable commitment horizon;
   - second-stage dispatch, storage, curtailment, imports, reserves, and shedding by scenario;
   - non-anticipativity constraints for decisions that must be shared.
4. Minimise expected cost. Report expected cost and scenario cost distribution.
5. Add expected-value and perfect-information benchmark calculations:
   - expected-value solution;
   - expected result of using the expected-value solution;
   - wait-and-see solution;
   - value of the stochastic solution;
   - expected value of perfect information.
6. Add optional Conditional Value at Risk with configurable confidence level and risk-aversion weight using a linear formulation.
7. Keep scenario generation outside the optimiser. Provide deterministic synthetic scenario generation based on configured perturbations and seed.
8. Add tests for probability weighting, non-anticipativity, one-scenario equivalence, value-of-information identities, and CVaR behaviour.
9. Document formulation size growth and practical scenario-count limits.

Acceptance criteria
- One scenario reproduces deterministic results.
- First-stage decisions are identical across scenarios where required.
- Expected objective equals probability-weighted scenario accounting plus first-stage costs.
- VSS and EVPI calculations are named, defined, and tested.
- CVaR increases protection against high-cost scenarios in a designed fixture.
```

---

---

# CORE-14 — Prices and market settlement

**Objective:** Add economically meaningful prices and settlements without confusing them with production costs.

```text
Apply the global engineering contract.

Implement market-price and settlement analysis for continuous dispatch and, with care, for mixed-integer unit commitment.

Requirements
1. Clearly distinguish:
   - system production cost;
   - objective penalties;
   - energy price;
   - nodal or zonal marginal price;
   - uplift or make-whole payment;
   - consumer payment;
   - generator revenue;
   - congestion rent;
   - scarcity rent.
2. For a continuous economic-dispatch or fixed-commitment model, obtain marginal prices from balance-constraint dual variables using a solver backend that exposes reliable duals.
3. Do not label raw MILP duals as market-clearing prices. For unit commitment, implement a documented pricing approach such as fixing integer decisions and re-solving the linear dispatch problem.
4. For nodal networks, calculate locational marginal prices and decompose them into energy and congestion components when supported by the formulation.
5. Add scarcity pricing through reserve shortage or load-shedding penalties, with explicit caps where configured.
6. Implement generator settlement:
   - energy revenue;
   - reserve revenue;
   - variable cost;
   - startup and no-load cost;
   - gross margin;
   - make-whole payment when market revenue does not recover committed cost under the selected rule.
7. Calculate consumer payments and congestion rent. Reconcile money flows and explain any residual caused by uplift or policy charges.
8. Add tests for uncongested uniform price, congested nodal-price separation, scarcity price, fixed-commitment pricing, and make-whole payment.
9. Add a dedicated `docs/market-model.md` warning that settlement rules vary by actual market design.

Acceptance criteria
- Price calculations use a solver path with verified dual sign conventions.
- MILP objective values are never misrepresented as prices.
- Settlement components reconcile under documented rules.
- A congested network example produces different nodal prices.
- Market outputs are optional and do not alter physical dispatch unless market-specific bidding is enabled.
```

---

---

# CORE-15 — Capacity-expansion planning

**Objective:** Add long-term investment decisions while retaining explicit operational constraints.

```text
Apply the global engineering contract.

Create a separate capacity-expansion model that selects new generation, storage, and network capacity while representing system operation over selected periods.

Scope
1. Keep operational simulation and planning as separate model entry points sharing domain components.
2. Add candidate investments for:
   - solar capacity;
   - wind capacity;
   - thermal capacity by technology;
   - storage power and energy capacity;
   - interconnector capacity;
   - transmission-line capacity or candidate lines.
3. Include annualised capital cost, fixed operation and maintenance cost, variable operation cost, fuel, emissions, startup, curtailment, and reliability penalties.
4. Support existing fixed assets and candidate assets.
5. Support representative periods with explicit weights. Validate that weighted hours represent the intended year.
6. Preserve chronology for storage and unit commitment within representative blocks. Document the limits of representative-period methods for seasonal storage.
7. Add policy constraints:
   - renewable-energy share;
   - emissions cap;
   - firm-capacity or planning-reserve margin;
   - maximum technology build;
   - minimum domestic generation;
   - carbon price.
8. Add discounted multi-year planning with build years and asset lifetime only after a robust single-year model exists. Keep the first implementation reviewable.
9. Report selected capacities, annualised costs, generation mix, curtailment, emissions, adequacy, and shadow prices when available from a continuous relaxation.
10. Add tests for investment thresholds, carbon-policy effects, storage power-versus-energy trade-off, and representative-period weights.
11. Add a small example in which renewable investment plus storage competes with a thermal candidate.

Acceptance criteria
- Investment variables have correct units and annualisation.
- Zero candidate capacity reproduces operational behaviour for the selected periods.
- Policy constraints are tested with hand-checkable cases.
- Planning outputs separate capital and operating costs.
- Documentation states the limitations of representative periods and perfect foresight.
```

---

---

# CORE-16 — Scenario runner and sensitivity analysis

**Objective:** Make policy and engineering experiments reproducible without editing configuration files manually.

```text
Apply the global engineering contract.

Implement a scenario-experiment framework around the simulator.

Requirements
1. Define a base configuration plus declarative parameter overrides.
2. Support one-factor sweeps, Cartesian grids, and explicitly enumerated scenarios.
3. Validate override paths against the typed configuration schema.
4. Generate stable scenario IDs from canonical parameter values.
5. Add a CLI command such as `energy-sim run-scenarios` that:
   - loads an experiment YAML;
   - resolves scenarios;
   - avoids accidental overwrite;
   - records manifests and checksums;
   - optionally resumes completed runs;
   - aggregates summaries into one table.
6. Add deterministic parallel execution with configurable worker count.
7. Add sensitivity outputs:
   - parameter values;
   - objective and cost components;
   - emissions;
   - renewable share;
   - curtailment;
   - unserved energy;
   - thermal starts;
   - storage cycles;
   - congestion and reserve metrics where available.
8. Add finite-difference local sensitivity for continuous scalar parameters, clearly labelled as numerical sensitivity rather than a derivative of a non-smooth MILP value function.
9. Add plotting utilities for response curves, heat maps, and trade-off frontiers. Do not hard-code colours unless existing project conventions require them.
10. Add tests for scenario expansion, stable IDs, resume behaviour, invalid override paths, and deterministic parallel results.
11. Create example experiments for carbon price, renewable capacity, storage size, thermal fuel price, and network capacity.

Acceptance criteria
- Every scenario is reproducible from its manifest.
- Parameter overrides never mutate the base configuration object.
- Aggregated results include solver status and exclude no failures silently.
- Re-running an experiment can skip verified completed scenarios safely.
- Charts are generated from tabular aggregate outputs, not directly from hidden state.
```

---

---

# CORE-17 — Public-data adapters and provenance

**Objective:** Add robust ingestion of public energy and weather data while keeping the simulator independent of any one provider.

```text
Apply the global engineering contract.

Create a data-adapter layer for public datasets. Do not hard-code live network access into the optimisation engine.

Requirements
1. Define a provider-neutral canonical time-series schema for demand, renewable weather drivers, prices, imports, outages, and hydro inflows.
2. Add adapter interfaces that transform provider-specific files or API responses into the canonical schema.
3. Keep network downloads in explicit CLI commands or scripts. Simulation runs must use local immutable input snapshots.
4. Add provenance metadata for every dataset:
   - provider;
   - source URL or dataset identifier;
   - retrieval timestamp;
   - licence;
   - original timezone;
   - transformation steps;
   - checksum;
   - missing-data treatment;
   - temporal aggregation.
5. Implement robust timezone and daylight-saving handling. Store canonical timestamps in UTC and preserve local-time metadata.
6. Implement deterministic aggregation and resampling with unit-aware rules. Power averages and energy sums must not be confused.
7. Add missing-data policies that are explicit and configurable: reject, bounded interpolation, forward fill with limit, or marked missing. Never silently fill long gaps.
8. Add at least one file-based adapter for a European demand dataset and one weather-data adapter, using small committed fixtures rather than downloading large datasets in tests.
9. Add a data-validation report containing coverage, gaps, duplicates, extremes, and energy totals before and after transformation.
10. Add tests for DST transitions, duplicate timestamps, missing intervals, unit conversion, power-versus-energy resampling, checksums, and provenance serialisation.

Acceptance criteria
- Optimisation modules do not import provider-specific code.
- Data snapshots and manifests are sufficient to reproduce a simulation.
- DST days with 23 or 25 local hours are handled correctly.
- Missing-data treatment is visible in provenance and validation reports.
- Tests do not depend on live external services.
```

---

---

# CORE-18 — Verification and benchmark suite

**Objective:** Demonstrate that the mathematical implementation is correct on analytically or independently solvable cases.

```text
Apply the global engineering contract.

Build a verification and validation suite beyond ordinary unit tests.

Required benchmark classes
1. Single-period economic dispatch with a hand-calculated merit order.
2. Single-unit commitment with a hand-enumerated small horizon.
3. Two-unit startup-cost example with an independently enumerated optimum.
4. Battery arbitrage with a known analytical solution under simple prices.
5. Reservoir allocation with a known two-period water-value solution.
6. Two-bus DC optimal power flow with a known congestion result.
7. Three-bus loop-flow case checked against an independent matrix calculation.
8. Reserve procurement with known headroom limits.
9. Demand shifting with an exact energy-conservation result.
10. Reliability metrics from a small enumerated outage-state model.

Implementation tasks
- Create `tests/benchmarks/` and document each mathematical case.
- For very small MILPs, implement an independent brute-force enumerator in test code to verify the optimiser result. Keep it separate from production formulation code.
- Add property-based tests only if introducing the dependency is justified; otherwise implement deterministic parameterised invariants.
- Add snapshot tests for stable output schemas, not for floating-point values that should be checked numerically.
- Add regression fixtures for every previously discovered model bug.
- Add performance budgets for representative CI-size models. Use generous thresholds that catch accidental order-of-magnitude regressions without creating flaky tests.
- Add a numerical-tolerance policy and test it across supported solver backends.
- Produce `docs/verification.md` mapping equations to tests.

Acceptance criteria
- Every major constraint family has at least one direct verification case.
- Small MILP optimums match independent enumeration.
- Verification documentation maps model components to tests and known limitations.
- Performance tests are deterministic enough for CI.
- Coverage emphasises mathematical branches, not only lines executed.
```

---

---

# CORE-19 — Solver abstraction and performance

**Objective:** Decouple formulation from one solver interface and improve scalability without sacrificing clarity.

```text
Apply the global engineering contract.

Refactor solver interaction behind a narrow backend interface and improve formulation performance.

Requirements
1. Separate model formulation data from solver invocation. Define typed structures for objective, bounds, integrality, sparse constraints, variable metadata, and solution metadata.
2. Preserve SciPy `milp` as the default open-source-compatible backend.
3. Add at most one additional backend in this prompt, selected for a clear reason such as dual access or improved MILP diagnostics. Make it optional and isolate the dependency in an extra Poetry group.
4. Define a capability matrix covering MILP, LP duals, warm starts, time limits, MIP gaps, node counts, infeasibility diagnostics, and solution pools.
5. Ensure backend-specific statuses map into project-level statuses without losing information.
6. Add formulation-build profiling by component.
7. Improve sparse matrix construction and avoid Python loops where safe, but do not replace readable formulations with opaque vectorisation.
8. Add optional variable and constraint name export for debugging.
9. Add LP or MPS export when supported, with stable names and a CLI option.
10. Implement warm-start support only for backends that genuinely support it. Never claim SciPy supports a capability that it does not expose.
11. Add tests comparing feasible solutions and objective values across available backends on small models.
12. Benchmark formulation and solve time for increasing horizon and asset counts. Produce tables, not unverifiable claims.

Acceptance criteria
- Core formulation code does not import a specific solver package directly.
- SciPy remains functional as the default.
- Optional backend absence produces a clear installation message.
- Status and gap reporting remain honest and backend-neutral.
- Benchmarks identify scaling in periods, units, storage assets, buses, and scenarios.
```

---

---

# CORE-20 — Reporting and diagnostics

**Objective:** Turn model outputs into auditable engineering and policy reports.

```text
Apply the global engineering contract.

Redesign reporting around a stable result schema and diagnostic checks.

Requirements
1. Define versioned output tables for:
   - system time series;
   - asset time series;
   - bus time series;
   - line time series;
   - cost components;
   - emissions;
   - reliability events;
   - solver diagnostics;
   - summary metrics.
2. Write a data dictionary for every output column, including units, sign convention, and aggregation rule.
3. Add plots for:
   - stacked dispatch and demand;
   - renewable availability, use, and curtailment;
   - thermal commitment and output by unit;
   - storage charge, discharge, and state;
   - reservoir levels and spill;
   - bus prices and line congestion when available;
   - reserve requirement and provision;
   - unserved-energy events;
   - cost and emissions decomposition;
   - duration curves.
4. Add automatic diagnostics that flag:
   - balance residuals;
   - bound violations;
   - simultaneous incompatible modes;
   - suspiciously frequent load shedding;
   - terminal-state violations;
   - solver non-optimal status;
   - missing time periods;
   - asset availability inconsistencies.
5. Generate a self-contained Markdown report referencing output CSV and PNG files. HTML may be optional, but do not add a heavy dashboard framework.
6. Add a comparison report for two or more scenario outputs with absolute and percentage differences.
7. Ensure plotting functions handle any number of assets and long horizons without unreadable legends. Aggregate or facet sensibly.
8. Add tests for metric definitions, output schema, diagnostics, empty optional components, and scenario comparison.

Acceptance criteria
- Every reported aggregate can be reproduced from versioned output tables.
- Reports prominently display solver status and balance diagnostics.
- Cost, energy, and emissions decompositions reconcile.
- Reporting handles aggregate, nodal, deterministic, rolling-horizon, and scenario-run outputs with graceful omission of unavailable components.
```

---

---

# CORE-21 — CLI and public Python API hardening

**Objective:** Make the simulator reliable as both a command-line tool and a Python library.

```text
Apply the global engineering contract.

Stabilise the public interfaces.

CLI requirements
1. Provide consistent commands for:
   - validate configuration;
   - validate data;
   - migrate configuration;
   - simulate;
   - run rolling horizon;
   - run scenario experiment;
   - run reliability study;
   - run capacity planning;
   - compare outputs;
   - export formulation;
   - show version and capabilities.
2. Use meaningful exit codes for success, invalid configuration, invalid data, infeasible model, solver failure, and partial feasible result.
3. Add structured logging with configurable verbosity. Keep normal output concise.
4. Support `--dry-run` to validate and report model size without solving.
5. Support command-line overrides only through validated configuration paths and record them in the manifest.
6. Never overwrite outputs unless `--overwrite` or `--resume` is explicitly supplied.

Python API requirements
1. Define a small supported public surface in package `__init__` exports.
2. Add typed functions or classes for loading configuration, validating data, building a model, solving, and obtaining results.
3. Avoid forcing users to invoke the CLI internally.
4. Document lifecycle and exceptions.
5. Add examples showing in-memory DataFrames and file-based runs.
6. Mark internal modules clearly and avoid accidental API promises.
7. Add API compatibility tests and CLI integration tests using subprocess invocation.

Acceptance criteria
- CLI errors are actionable and return non-zero status.
- `--dry-run` reports dimensions without calling the solver.
- Python users can run a small simulation without temporary configuration files.
- Public objects are typed and documented.
- Output overwrite protection is tested.
```

---

---

# CORE-22 — Iberian system case study

**Objective:** Build a reproducible, openly sourced Portugal–Spain case study without pretending to reproduce the full real grid.

```text
Apply the global engineering contract.

Create a research-oriented Iberian electricity-system case study using public data snapshots and clearly stated approximations.

Scope decisions
1. Define an explicit study period and temporal resolution.
2. Start with a zonal Portugal–Spain model or a small transparent bus aggregation. Do not claim full nodal fidelity without full network data.
3. Include, where data and licensing permit:
   - Portuguese and Spanish demand;
   - solar and wind availability;
   - hydro inflows or hydro generation approximation;
   - thermal fleets grouped by technology or selected representative units;
   - cross-border interconnection;
   - import or export prices;
   - carbon prices and fuel assumptions.
4. Store small derived datasets or provide reproducible download scripts. Respect source licences and do not commit restricted data.
5. Add a provenance document with exact providers, dataset IDs, retrieval dates, transformations, missing-data rules, and caveats.
6. Calibrate parameters only through transparent calculations. Do not use machine learning.
7. Validate annual or monthly energy totals against published aggregates when possible.
8. Define experiments such as:
   - increased solar capacity;
   - increased interconnection;
   - thermal-unit retirement;
   - drought hydro conditions;
   - battery additions;
   - carbon-price changes;
   - high electrification demand.
9. Produce a case-study report with assumptions, model topology, validation, baseline results, sensitivity results, and limitations.
10. Ensure the case study is separate from core package tests and that a small synthetic fixture remains the CI default.

Acceptance criteria
- Every external datum has provenance and a licence note.
- The case study is reproducible from documented commands.
- Modelled energy totals are compared with external aggregates and discrepancies are discussed.
- Conclusions are framed as results of an approximation, not as exact operational history.
- Portugal and Spain flows use a documented sign convention.
```

---

---

# CORE-23 — Research experiment framework

**Objective:** Make the repository suitable for producing defensible technical reports or papers.

```text
Apply the global engineering contract.

Create a research-experiment layer that separates hypotheses, experimental design, execution, and result interpretation.

Requirements
1. Add an `experiments/` structure with one directory per study containing:
   - research question;
   - hypotheses;
   - model version;
   - data version and checksums;
   - base configuration;
   - scenario definitions;
   - execution script;
   - expected outputs;
   - analysis notebook or script;
   - limitations.
2. Add experiment manifests that freeze package version, Git commit, configuration, dataset checksums, seed, solver backend, solver options, and machine information.
3. Add a command to reproduce an experiment from its manifest and verify hashes.
4. Require pre-specified metrics and comparisons before execution.
5. Support paired scenario comparisons using identical stochastic seeds where appropriate.
6. Add uncertainty intervals for Monte Carlo outputs and sensitivity ranges for deterministic assumptions.
7. Separate descriptive results from causal claims. Add a documentation template that explicitly states what the model can and cannot identify.
8. Add automated tables suitable for LaTeX and Markdown, with units in headers and stable rounding rules.
9. Add figure metadata and captions generated from experiment definitions.
10. Create one complete example experiment, such as the value of storage under increasing renewable penetration, with a mathematically grounded design and no machine learning.

Acceptance criteria
- A fresh checkout can reproduce the example experiment from one documented command after data preparation.
- Manifests detect changed inputs or configurations.
- Tables and figures are generated from result files, not manually entered values.
- The experiment document distinguishes assumptions, model outputs, interpretation, and limitations.
```

---

---

# CORE-24 — Version 1.0 release hardening

**Objective:** Prepare a stable research and teaching release after the chosen roadmap scope is complete.

```text
Apply the global engineering contract.

Prepare version 1.0.0 without adding major new model scope.

Tasks
1. Audit the public Python API, CLI, configuration schema, and output schema for consistency.
2. Remove or formally deprecate obsolete interfaces. Document every breaking change from 0.1.0.
3. Freeze configuration schema version 1 and output schema version 1.
4. Complete README sections for installation, quick start, model selection, examples, limitations, citation, and contribution.
5. Add a documentation index linking architecture, mathematical model, configuration, data contract, outputs, verification, market model, reliability, and case studies.
6. Ensure all examples run from a clean checkout.
7. Raise and enforce test coverage based on mature measured coverage, targeting at least 90% for core formulation and accounting modules.
8. Run static analysis, tests, benchmark suite, example simulations, package build, wheel installation test, and citation metadata validation.
9. Add semantic versioning and release notes.
10. Add contribution guidelines, code of conduct, security policy, issue templates, and pull-request template.
11. Add a compatibility matrix for Python and optional solver backends.
12. Verify that all committed data have acceptable licences and provenance.
13. Add an archive-ready citation and reproducibility checklist.
14. Produce `docs/release-validation-1.0.md` with exact commands, environment, results, known limitations, and unresolved risks.

Acceptance criteria
- Source distribution and wheel build and install cleanly in a fresh environment.
- All supported examples and verification cases pass.
- Public interfaces and schemas are documented and versioned.
- No known balance, cost-reconciliation, unit, or state-transition defect remains unresolved without explicit documentation.
- Release notes state clearly what the simulator is and is not intended to model.
```

---

---

# Advanced post-core development roadmap

# ADV-01 — Security-constrained unit commitment with N-1 contingencies

**Suggested branch:** `feat/security-constrained-uc`

**Objective:** Extend nodal commitment and dispatch so selected single-component contingencies are feasible or explicitly penalised.

```text
Apply the global implementation contract.

Implement a deterministic security-constrained unit commitment model after CORE-08 integrated nodal DC optimal power flow and CORE-09 reserves are stable.

Requirements
1. Define a contingency set for line outages, generator outages, and optionally interconnector outages.
2. Separate base-case commitment decisions from contingency redispatch decisions.
3. Implement one of two transparent formulations and document the trade-off:
   - explicit post-contingency DC flow constraints for every selected contingency; or
   - PTDF/LODF-based linear security constraints for line contingencies, validated against explicit solves.
4. Bound contingency redispatch by available headroom, footroom, reserve products, ramp capability, and outage status.
5. Prevent the failed component from contributing in its contingency state.
6. Support hard security and penalised emergency actions such as contingency shedding or rating exceedance with very high explicit costs.
7. Add contingency screening so harmless contingencies may be omitted only after a documented conservative test.
8. Report the binding contingency, overloaded element, redispatch, reserve deployment, and security cost.
9. Preserve base-case energy and cost accounting separately from contingency feasibility metrics.

Tests
- Three-bus line outage with an analytical alternative path.
- Generator outage requiring reserve deployment.
- Insecure case that is feasible in ordinary UC but infeasible under N-1.
- LODF formulation matches explicit outage flow within tolerance where assumptions hold.

Acceptance criteria
- Every included N-1 contingency is checked.
- Security actions and costs are not mixed silently into base dispatch.
- Model size growth is benchmarked by periods, contingencies, buses, and lines.
```

# ADV-02 — Frequency adequacy, inertia, and primary response

**Suggested branch:** `feat/frequency-adequacy`

**Objective:** Add transparent frequency-security proxies suitable for planning and commitment studies without pretending to perform full electromagnetic-transient simulation.

```text
Apply the global implementation contract.

Add a frequency-adequacy module linked to commitment and reserve decisions.

Requirements
1. Define synchronous inertia in MW·s or an equivalent consistently documented unit for each online thermal or hydro unit.
2. Represent the largest credible infeed loss endogenously or through a conservative configured value.
3. Add constraints or ex-post checks for:
   - minimum system inertia;
   - rate-of-change-of-frequency proxy;
   - primary frequency response availability;
   - response delivery time;
   - quasi-steady-state frequency balance.
4. Model battery or inverter-based fast frequency response separately from sustained spinning reserve.
5. Do not label these linear proxies as dynamic frequency simulation. Document assumptions and validity range.
6. Add optional synthetic-inertia capability with energy and power limits.
7. Report inertia, largest loss, RoCoF proxy, response requirement, response provision by asset, and scarcity periods.
8. Create a low-inertia high-renewables example where an energy-feasible dispatch is frequency-insecure.

Tests
- Hand-calculated two-generator inertia case.
- Largest-unit commitment changes the credible loss.
- Battery fast response is limited by power and state of charge.
- Removing a synchronous unit violates inertia even when energy balance remains feasible.

Acceptance criteria
- Units and equations are explicit.
- Frequency proxies are clearly separated from reserves and energy.
- The simulator reports the limitation that it is not a transient stability tool.
```

# ADV-03 — AC power-flow validation bridge

**Suggested branch:** `feat/ac-validation`

**Objective:** Validate selected DC-dispatch solutions against nonlinear AC power flow without immediately replacing the optimisation model with AC optimal power flow.

```text
Apply the global implementation contract.

Create an optional post-dispatch AC validation layer.

Requirements
1. Define buses with voltage magnitude limits, voltage angle, active and reactive demand, shunts, generator reactive limits, transformer taps, and line impedance where available.
2. Keep DC optimisation and AC validation as separate model layers with an explicit mapping.
3. Implement or integrate a narrowly scoped, well-maintained AC power-flow solver only after documenting the dependency decision.
4. For selected periods, solve AC power flow using dispatched active power and report:
   - convergence status;
   - bus voltage violations;
   - reactive-power limit violations;
   - branch MVA overloads;
   - active losses;
   - mismatch from the DC approximation.
5. Add a period-selection policy based on peak demand, peak renewables, congestion, and user-selected timestamps.
6. Never silently correct dispatch to force AC convergence. Return a validation failure with diagnostics.
7. Add a small standard network fixture with published or hand-verifiable values and provenance.

Acceptance criteria
- DC dispatch can be validated on selected periods from one command.
- Non-convergence and violations are explicit.
- No claim of AC feasibility is made for unvalidated periods.
```

# ADV-04 — Distribution feeders, distributed energy resources, and hosting capacity

**Suggested branch:** `feat/distribution-der`

**Objective:** Add a distribution-level model for radial feeders, rooftop solar, batteries, flexible load, voltage constraints, and hosting-capacity studies.

```text
Apply the global implementation contract.

Implement a separate distribution-system model family rather than overloading the transmission DC model.

Requirements
1. Represent radial buses, branches, resistance, reactance, thermal ratings, voltage bounds, substation import, fixed load, rooftop PV, behind-the-meter battery, and controllable demand.
2. Use a documented linearised DistFlow or second-order cone formulation. State all approximations.
3. Preserve phase and unit conventions. A single-phase equivalent is acceptable initially if clearly stated.
4. Include active and reactive flows, voltage-drop constraints, approximate losses, DER curtailment, and substation exchange.
5. Add hosting-capacity analysis that maximises additional DER capacity subject to voltage and thermal limits.
6. Separate customer-side and grid-side battery accounting.
7. Report voltage profiles, branch loading, losses, curtailment, reverse flow, and hosting capacity.
8. Provide a small radial feeder example and validate against an independent nonlinear power-flow result for selected cases.

Acceptance criteria
- Voltage and thermal constraints bind in deterministic fixtures.
- Hosting capacity has a reproducible optimum.
- Distribution outputs cannot be confused with transmission DC outputs.
```

# ADV-05 — Electric vehicles, heat pumps, and flexible electrification

**Suggested branch:** `feat/flexible-electrification`

**Objective:** Model sector electrification as constrained energy services rather than fixed demand multipliers.

```text
Apply the global implementation contract.

Add transparent flexible-load assets for electric vehicles and heat pumps.

Requirements
1. Electric vehicles:
   - arrival and departure periods;
   - initial and required departure energy;
   - charging power and efficiency;
   - optional vehicle-to-grid with separate efficiency and degradation cost;
   - fleet aggregation with availability fractions.
2. Heat pumps:
   - ambient-temperature-dependent coefficient of performance;
   - building thermal state or heat-storage state;
   - comfort bounds;
   - heat demand and thermal losses;
   - backup heat source with explicit cost and emissions.
3. Do not use machine learning to forecast behaviour in this prompt. Use supplied deterministic or scenario-based profiles.
4. Co-optimise flexible electrification with generation, network, reserves, and storage where available.
5. Report delivered mobility energy, unmet departure requirements, comfort violations, electrical demand, shifted energy, peak contribution, and emissions.
6. Add rebound and synchronisation stress cases.

Acceptance criteria
- Energy-service constraints are satisfied, not merely aggregate daily energy.
- Flexibility cannot use vehicles when absent or violate comfort silently.
- Results quantify peak-shifting benefits and new demand requirements.
```

# ADV-06 — Hydrogen production, storage, and reconversion

**Suggested branch:** `feat/hydrogen-system`

**Objective:** Add hydrogen as an explicitly lossy multi-period energy carrier for sector coupling and long-duration storage studies.

```text
Apply the global implementation contract.

Implement an optional hydrogen subsystem.

Requirements
1. Add electrolyser assets with electrical input, hydrogen output, minimum load if applicable, ramping, efficiency or piecewise-linear efficiency, startup logic only when justified, and variable cost.
2. Add hydrogen storage with inventory, charge/discharge rates, losses, initial and terminal inventory, and capacity.
3. Add exogenous hydrogen demand with optional shortage penalty.
4. Add fuel-cell or hydrogen-turbine reconversion with explicit efficiency, capacity, cost, and emissions assumptions.
5. Keep units explicit, selecting kg, MWh lower-heating-value, or another single canonical internal unit with conversion utilities.
6. Prevent free energy through inconsistent heating-value or efficiency conversions.
7. Report electricity consumed, hydrogen produced, stored, curtailed, delivered, reconverted, round-trip efficiency, and marginal system value.
8. Add a renewable-surplus case and a long-duration deficit case.

Acceptance criteria
- Hydrogen mass or energy balance reconciles every period and over the horizon.
- Conversion losses and costs are explicit.
- The model does not imply hydrogen is emission-free unless the configured electricity and process assumptions support that statement.
```

# ADV-07 — Combined heat and power and district heating

**Suggested branch:** `feat/chp-heat-network`

**Objective:** Add coupled electricity and heat production with thermal storage and district-heat demand.

```text
Apply the global implementation contract.

Implement a multi-energy heat subsystem with combined heat and power units.

Requirements
1. Represent heat demand, heat-only boilers, electric boilers, heat pumps, thermal storage, and CHP units.
2. Model CHP feasible operating regions with a convex polytope or documented piecewise-linear extraction/condensing representation.
3. Include fuel use, electrical output, heat output, efficiencies, costs, and emissions without double counting.
4. Add heat-network losses using a transparent aggregate approximation initially.
5. Enforce heat balance and thermal-storage state transitions independently from electrical balance.
6. Co-optimise heat and electricity while preserving separate price and emissions accounting.
7. Report heat by source, electricity by CHP, fuel use, heat dumping, unmet heat, storage operation, and coupling effects.

Acceptance criteria
- CHP cannot independently choose impossible maximum heat and power outputs.
- Fuel and emissions reconcile across both products.
- A small fixture demonstrates the electricity–heat trade-off.
```

# ADV-08 — Transmission expansion planning

**Suggested branch:** `feat/transmission-expansion`

**Objective:** Co-optimise candidate transmission investment with generation and storage planning.

```text
Apply the global implementation contract.

Add a transmission-expansion model family after capacity expansion and nodal DCOPF are validated.

Requirements
1. Define existing lines and candidate corridors with discrete circuits or continuous capacity options, capital cost, lifetime, availability date, and maximum build.
2. Link investment decisions to line capacity and susceptance using a mathematically valid linear formulation. Avoid multiplying a binary build variable by an unrestricted angle difference without proper reformulation.
3. Support multiple planning periods and discounted costs where CORE capacity expansion supports them.
4. Add optional right-of-way, regional build limits, and mutually exclusive candidate designs.
5. Include representative-period weighting carefully and document chronology limitations.
6. Report build decisions, congestion relief, renewable curtailment reduction, reliability impact, and annualised cost.
7. Add a three-zone case where the least-cost solution can be verified manually.

Acceptance criteria
- Candidate lines carry no flow before construction.
- Investment and operational costs reconcile.
- Big-M values, if used, are derived from defensible angle bounds and tested for sensitivity.
```

# ADV-09 — Distribution reinforcement and non-wires alternatives

**Suggested branch:** `feat/distribution-planning`

**Objective:** Compare feeder reinforcement with storage, demand flexibility, and DER curtailment as explicit alternatives.

```text
Apply the global implementation contract.

Extend the distribution model into a planning formulation.

Requirements
1. Add candidate branch upgrades, transformer upgrades, voltage-control devices, grid batteries, and contracted flexibility.
2. Annualise investment costs consistently and preserve operational chronology.
3. Model discrete upgrade sizes and mutually exclusive alternatives.
4. Quantify non-wires alternatives under the same reliability and voltage constraints as traditional reinforcement.
5. Add optional limits on curtailment hours, customer flexibility calls, and unmet demand.
6. Report selected upgrades, deferred investment, utilisation, voltage performance, curtailment, and total cost.
7. Add a radial feeder case where storage can defer but not permanently eliminate a capacity upgrade.

Acceptance criteria
- Planning decisions have physically valid operational consequences.
- Cost comparison uses the same horizon and discount assumptions.
- The model distinguishes temporary flexibility from durable network capacity.
```

# ADV-10 — Weather-year ensembles and climate stress testing

**Suggested branch:** `feat/climate-stress`

**Objective:** Evaluate system performance across multiple weather years and transparent climate perturbations.

```text
Apply the global implementation contract.

Build a weather-ensemble framework without claiming climate projections beyond supplied data.

Requirements
1. Support multiple aligned weather years containing wind speed, irradiance, temperature, hydro inflow, and demand-temperature drivers where available.
2. Preserve chronology within each year and record data provenance and timezone handling.
3. Add deterministic perturbation scenarios such as temperature shift, drought multiplier, wind-resource multiplier, solar-resource multiplier, and compound stress events.
4. Keep physical transformations separate from raw data.
5. Run operational or planning models across all years with equal, configured, or probabilistic weights.
6. Report distribution of cost, emissions, curtailment, unserved energy, reserve scarcity, storage cycles, and capacity adequacy.
7. Add worst-year and regret metrics, but do not collapse all outcomes into one mean.
8. Add validation that leap years, daylight-saving transitions, missing hours, and duplicated timestamps are handled explicitly.

Acceptance criteria
- Every result is traceable to a weather-year and transformation manifest.
- Chronology and timezones are preserved.
- Compound stress outcomes can be compared with the historical baseline reproducibly.
```

# ADV-11 — Calibration and empirical validation

**Suggested branch:** `feat/model-calibration-validation`

**Objective:** Estimate transparent model parameters from observed data and assess whether simulated operations reproduce measurable system behaviour.

```text
Apply the global implementation contract.

Create a calibration and validation workflow that remains separate from optimisation.

Requirements
1. Define calibratable parameters such as aggregate losses, availability, renewable performance ratio, heat-rate segment coefficients, demand-temperature response, and storage efficiency only when data support them.
2. Use transparent estimation methods: constrained least squares, robust regression, maximum likelihood for explicit distributions, or Bayesian estimation with stated priors. Do not introduce black-box ML.
3. Separate calibration periods from validation periods chronologically.
4. Add parameter bounds derived from physics and source documentation.
5. Quantify uncertainty and parameter correlation where possible.
6. Validate simulated generation shares, ramp distributions, start counts, prices where available, emissions, imports, curtailment, and demand using clearly defined metrics.
7. Do not tune penalty values merely to fit observed dispatch without documenting the behavioural interpretation.
8. Produce calibration manifests, residual diagnostics, and out-of-sample reports.

Acceptance criteria
- Parameters are estimated from identified data columns with units and provenance.
- Validation is out of sample.
- The report separates parameter uncertainty, structural model error, and data error.
```

# ADV-12 — Lifecycle emissions, carbon budgets, and policy constraints

**Suggested branch:** `feat/carbon-policy`

**Objective:** Extend operational emissions into configurable lifecycle and cumulative-policy accounting without obscuring the distinction between them.

```text
Apply the global implementation contract.

Add an emissions and carbon-policy model layer.

Requirements
1. Distinguish direct operational, upstream fuel, construction embodied, and decommissioning emissions.
2. Keep operational emissions time-dependent and lifecycle investment emissions associated with capacity decisions.
3. Add pollutant types with separate units and factors where justified, including CO2-equivalent only when the aggregation factors are supplied.
4. Support annual or cumulative carbon budgets, emissions intensity standards, renewable-energy constraints, and carbon prices as separate policy instruments.
5. Prevent double counting when imported electricity has an emissions factor and carbon cost already embedded in a price scenario.
6. Report emissions by source, scope, period, pollutant, and policy instrument.
7. Add marginal-abatement-cost scenario generation based on repeated transparent optimisations.
8. Document that lifecycle factors are assumptions with provenance, not universal constants.

Acceptance criteria
- Every tonne in summary output reconciles to asset, import, or investment components.
- Carbon budgets bind in deterministic fixtures.
- Direct and lifecycle emissions remain separately visible.
```

# ADV-13 — Water use and the water–energy nexus

**Suggested branch:** `feat/water-energy-nexus`

**Objective:** Quantify water withdrawals and consumption for thermal and hydro assets and introduce water constraints where relevant.

```text
Apply the global implementation contract.

Add optional water accounting and scarcity constraints.

Requirements
1. Define withdrawal and consumptive-use factors separately for thermal cooling technologies, fuel pathways, and other relevant assets.
2. Allow output-dependent piecewise-linear water intensity where supported.
3. Link hydro reservoir releases to energy and downstream minimum-flow requirements without counting the same release twice.
4. Add regional water availability constraints and seasonal scarcity prices or penalties.
5. Preserve units such as cubic metres per MWh and cubic metres per period.
6. Report withdrawal, consumption, return flow when modelled, water cost, and energy curtailed due to water limits.
7. Add a drought case where water constraints alter thermal or hydro dispatch.

Acceptance criteria
- Water balance and energy balance are independently reconciled.
- Withdrawal and consumption are not treated as synonyms.
- Water factors and regional mappings have provenance.
```

# ADV-14 — Decomposition and scalable optimisation

**Suggested branch:** `feat/optimisation-decomposition`

**Objective:** Scale long-horizon, stochastic, or expansion models without weakening the mathematical problem.

```text
Apply the global implementation contract.

Implement one justified decomposition method after profiling identifies a real scaling bottleneck.

Candidate methods
- rolling horizon with overlap for operational chronology;
- Benders decomposition for investment master and operational subproblems;
- progressive hedging for scenario-based decisions;
- Lagrangian relaxation for selected coupling constraints;
- column-and-constraint generation for robust optimisation.

Tasks
1. Select one method based on the actual target model structure. Explain why the assumptions hold.
2. Define master variables, subproblem variables, coupling constraints, bounds, and convergence test mathematically.
3. Implement deterministic iteration logs with lower bound, upper bound, gap, cut count, and subproblem statuses.
4. Preserve a monolithic formulation for small verification cases.
5. Compare decomposed and monolithic optimum on all small benchmark fixtures.
6. Add checkpoint and resume support only through serialisable model state, not opaque solver objects.
7. Benchmark scaling and memory with periods, assets, scenarios, or candidates.
8. Never claim exactness for a heuristic rolling horizon; report its status separately.

Acceptance criteria
- Decomposition matches the monolithic optimum on verification cases within tolerance when the method is exact.
- Convergence and termination are honest.
- Measured performance improvement is reported, not assumed.
```

# ADV-15 — Operational service, job API, and observability

**Suggested branch:** `feat/simulation-service`

**Objective:** Expose stable simulation jobs through a backend service after the Python API and schemas are versioned.

```text
Apply the global implementation contract.

Build an optional backend service around the existing simulator. Do not move optimisation logic into HTTP handlers.

Requirements
1. Use a small typed API framework only after justifying the dependency.
2. Expose versioned endpoints for:
   - health and package version;
   - configuration validation;
   - job submission from a validated configuration reference;
   - job status and solver diagnostics;
   - result manifest and output retrieval;
   - cancellation where the solver backend supports it safely.
3. Keep execution in an explicit job-runner abstraction. A local process or bounded worker is sufficient; do not introduce Kubernetes.
4. Define idempotency, job IDs, state transitions, output directories, retention, and failure semantics.
5. Apply strict size, path, and input validation. Do not permit arbitrary filesystem reads or shell commands.
6. Add structured logs with job ID, model version, solver status, runtime, and formulation size.
7. Add metrics for job count, failure count, solve time, queue time, and active jobs.
8. Add API schema tests, job-state tests, and an end-to-end reduced simulation.
9. Keep CLI and direct Python API fully functional without the service dependency through an optional dependency group.

Acceptance criteria
- HTTP handling contains no duplicated model equations.
- Jobs are reproducible from stored manifests.
- Failure and non-optimal states are exposed honestly.
- The service remains optional and locally runnable.
```

# Reusable review prompts

## REVIEW-A — Audit one completed milestone

```text
Inspect the implementation of milestone [ID] against its original prompt. Do not trust the changelog or agent summary. Trace every acceptance criterion to code, documentation, tests, and executable evidence. Produce a table with: requirement, evidence, status, defect, and required correction. Re-run the smallest relevant tests and one end-to-end case. Distinguish complete, partially complete, implemented but untested, documented only, and missing.
```

## REVIEW-B — Mathematical formulation review

```text
Review [MODEL COMPONENT] independently of coding style. Enumerate sets, parameters, variables, units, objective terms, bounds, initial conditions, terminal conditions, and constraints. Check signs, time-step scaling, logical equivalence, dimensional consistency, edge-horizon behaviour, numerical scaling, and accounting. Construct at least three hand-computable counterexample attempts. Implement changes only after the written review identifies a confirmed defect.
```

## REVIEW-C — Regression-preserving refactor

```text
Refactor [COMPONENT] without changing supported model semantics. First freeze characterisation fixtures for decisions, objective components, physical balances, diagnostics, and output schemas. Keep a before-and-after comparison tool. Any changed result must be explained by an intentional semantic correction and accompanied by a regression test and migration note.
```

## REVIEW-D — Performance investigation

```text
Profile [MODEL OR CASE] before optimising it. Measure formulation construction, sparse-matrix assembly, presolve, solve time, nodes, incumbent, bound, gap, memory, variables by type, constraints by family, and non-zero count. Propose only changes that preserve the mathematical problem or label heuristics honestly. Demonstrate before-and-after results on identical hardware and inputs.
```

## REVIEW-E — Research-result readiness

```text
Assess whether experiment [NAME] is ready to support a paper, report, or public claim. Verify data provenance, configuration and output hashes, model version, hypothesis pre-specification, sensitivity analysis, benchmark validation, solver status, uncertainty treatment, limitations, and reproducibility from a clean checkout. Reject causal or policy conclusions that are not identified by the model design.
```

# Recommended execution sequence

## Mandatory correction sequence

1. FIX-00 baseline preservation.
2. FIX-01 release metadata.
3. FIX-02 packaging and version authority.
4. FIX-03 terminal commitment.
5. FIX-04 duration and initial-state regressions.
6. FIX-05 solver statuses.
7. FIX-06 transition ramps.
8. FIX-07 strict configuration.
9. FIX-08 numerical policy.
10. FIX-09 network semantics.
11. FIX-10 stress suite and corrected baseline release.

## Core research-grade sequence

1. CORE-01 through CORE-04 establish typed portfolios and multi-unit thermal operation.
2. CORE-05 through CORE-09 add fuels, storage, hydro, demand flexibility, nodal networks, and reserves.
3. CORE-10 through CORE-12 extend renewable detail, long-horizon execution, and reliability.
4. CORE-13 through CORE-15 add uncertainty and market accounting only after deterministic verification.
5. CORE-16 capacity expansion should start only after operational constraints are stable.
6. CORE-17 through CORE-23 provide scenarios, data provenance, benchmarks, solvers, reporting, public API, and the Iberian case study.
7. CORE-24 prepares the stable 1.0 release.

## Selecting advanced prompts

- Use ADV-01 and ADV-02 for system-security questions.
- Use ADV-03 and ADV-04 for electrical-network fidelity.
- Use ADV-05 through ADV-07 for sector coupling.
- Use ADV-08 and ADV-09 for grid investment.
- Use ADV-10 through ADV-13 for climate, empirical, carbon, and water research.
- Use ADV-14 only after measured performance warrants decomposition.
- Use ADV-15 only after the Python API and output schemas are stable.

## Minimum defensible stopping point

A strong first research release does not require every advanced model. A defensible stopping point is:

- all FIX prompts;
- CORE-01 through CORE-09;
- CORE-11 rolling horizon;
- CORE-17 scenario runner;
- CORE-18 public-data provenance;
- CORE-19 benchmark suite;
- CORE-21 reporting;
- CORE-22 public API;
- one carefully scoped CORE-23 empirical case study;
- CORE-24 release hardening.

## Definition of done

A milestone is not complete merely because code exists. It is complete only when:

- the mathematical semantics are documented;
- configuration and outputs are versioned where affected;
- normal, boundary, invalid-input, regression, and invariant tests pass;
- objective, physical, emissions, and settlement accounting reconcile where applicable;
- solver termination is reported honestly;
- package and example execution are reproducible;
- source distribution and wheel remain installable;
- public documentation describes limitations;
- the milestone status table is updated with evidence;
- no placeholder, silent fallback, untested branch, or knowingly ambiguous boundary condition remains.
