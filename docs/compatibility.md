# Compatibility Matrix

## Python

| Python | Support |
|---|---|
| 3.11 | Supported and tested in CI |
| 3.12 | Supported and tested in CI |
| 3.13 | Supported and tested in CI |
| 3.14+ | Not supported by the 1.0 dependency constraints |

The package metadata declares `python = ">=3.11,<3.14"`.

## Solver Backends

| Backend | Status | Notes |
|---|---|---|
| `scipy.optimize.milp` | Supported default | Uses SciPy's HiGHS MILP interface. |
| LP export | Supported diagnostic path | `energy-sim export-model --format lp` writes the formulation for external inspection. |
| Commercial MILP solvers | Not integrated | Use LP export externally if a separate solver workflow is required. |

## Operating Systems

The codebase is pure Python and CI runs on Ubuntu. The 1.0 release-validation
run also passed on Windows. File paths in manifests and reports are normalized
where they are intended to be committed.

