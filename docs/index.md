# Documentation Index

Start here for the 1.0 documentation set.

| Area | Document |
|---|---|
| Architecture | [architecture.md](architecture.md) |
| Mathematical model | [model.md](model.md) |
| Configuration schemas | [configuration.md](configuration.md) |
| Data contract | [data-contract.md](data-contract.md) |
| CLI and Python API | [api-cli.md](api-cli.md) |
| Output tables and diagnostics | [reporting.md](reporting.md) |
| Verification and benchmarks | [verification.md](verification.md) |
| Solver backends and LP export | [solver-backends.md](solver-backends.md) |
| Renewable models | [renewable-models.md](renewable-models.md) |
| Rolling horizon | [rolling-horizon.md](rolling-horizon.md) |
| Reliability studies | [reliability.md](reliability.md) |
| Frequency adequacy proxies | [frequency-adequacy.md](frequency-adequacy.md) |
| Stochastic dispatch | [stochastic-dispatch.md](stochastic-dispatch.md) |
| Market model | [market-model.md](market-model.md) |
| Capacity expansion | [capacity-expansion.md](capacity-expansion.md) |
| Security-constrained dispatch checks | [security-constrained-dispatch.md](security-constrained-dispatch.md) |
| AC validation bridge | [ac-validation.md](ac-validation.md) |
| Distribution feeder studies | [distribution-feeder.md](distribution-feeder.md) |
| Hydrogen subsystem | [hydrogen.md](hydrogen.md) |
| District heat and CHP | [heat.md](heat.md) |
| Scenario experiments | [scenario-experiments.md](scenario-experiments.md) |
| Research experiments | [research-experiments.md](research-experiments.md) |
| Public-data adapters | [public-data-adapters.md](public-data-adapters.md) |
| Committed data provenance | [data-provenance-inventory.md](data-provenance-inventory.md) |
| Iberian case study | [../case_studies/iberia/README.md](../case_studies/iberia/README.md) |
| Model status and limitations | [model-status.md](model-status.md) |
| Compatibility matrix | [compatibility.md](compatibility.md) |
| Release checklist | [release-checklist.md](release-checklist.md) |
| 1.1.2 release validation | [release-validation-1.1.2.md](release-validation-1.1.2.md) |
| 1.1.1 release validation | [release-validation-1.1.1.md](release-validation-1.1.1.md) |
| 1.1 release validation | [release-validation-1.1.md](release-validation-1.1.md) |
| 1.0 release validation | [release-validation-1.0.md](release-validation-1.0.md) |

## Release Schema Policy

Energy System Simulator 1.0 supports aggregate configuration schema 1 and typed
portfolio configuration schema 2. Unknown future configuration versions fail
with an explicit error. Versioned output tables use schema version 1 and remain
the stable audit interface for the 1.0 line.
