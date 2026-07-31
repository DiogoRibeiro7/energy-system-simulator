# Mathematical model

## Sets and periods

The model is solved over periods \(t=1,\ldots,T\), with a constant period length \(\Delta t\) in hours.

## Distribution representation

Let \(\eta_n=1-\lambda\) be network delivery efficiency and \(F^{\max}\) the source-side transfer capacity. End-user demand \(d_t\) is split into:

\[
\bar d_t = \min(d_t,\eta_nF^{\max}),
\]

\[
d_t^{\mathrm{network\ shed}} = d_t-\bar d_t,
\]

and the source-equivalent demand passed to the dispatch model is

\[
g_t=\frac{\bar d_t}{\eta_n}.
\]

The ordering is deliberate: transfer capacity is applied first to end-user
demand through the maximum deliverable demand \(\eta_nF^{\max}\). Any excess is
reported as network-capacity shedding before dispatch optimisation. The dispatch
model then serves the remaining source-equivalent demand \(g_t\) with generation,
storage, imports, or source-side load shedding.

## Renewable generation

Let \(K^{\mathrm{ren}}\) be the configured renewable asset set. Each asset \(k\)
has exogenous availability \(a_{k,t}\) from its configured weather columns.
Available renewable production passed to the current aggregate dispatch
formulation is:

\[
a_t^{\mathrm{ren}}=\sum_{k\in K^{\mathrm{ren}}} a_{k,t}.
\]

The used renewable variable satisfies

\[
0\le r_t\le a_t^{\mathrm{ren}}.
\]

Aggregate curtailment is \(a_t^{\mathrm{ren}}-r_t\). Until renewable dispatch is
indexed in the optimisation, asset-level used output is reported by availability
share:

\[
r_{k,t}=r_t\frac{a_{k,t}}{a_t^{\mathrm{ren}}}
\]

when \(a_t^{\mathrm{ren}}>0\), and zero otherwise. Asset-level curtailment is
\(a_{k,t}-r_{k,t}\). These reported asset quantities reconcile exactly to the
aggregate dispatch variables.

## Power balance

All dispatch variables are represented on the source side:

\[
r_t+\sum_{g\in G}p_{g,t}+d_t^{\mathrm{bat}}+i_t+\ell_t
=g_t+c_t^{\mathrm{bat}}.
\]

Here \(\ell_t\) is source-equivalent involuntary load shedding. Delivered involuntary shedding is \(\eta_n\ell_t\).

## Thermal generators

Let \(G\) be the configured thermal generator set. For each generator \(g\in G\)
and period \(t\), commitment status \(u_{g,t}\), startup \(y_{g,t}\), shutdown
\(z_{g,t}\), and output \(p_{g,t}\) are indexed by generator. The power balance
uses total thermal output:

\[
r_t+\sum_{g\in G}p_{g,t}+d_t^{\mathrm{bat}}+i_t+\ell_t
=g_t+c_t^{\mathrm{bat}}.
\]

Generator availability is an exogenous multiplier
\(\alpha_{g,t}\in[0,1]\), combining the static `availability_factor` and an
optional configured availability time-series column. Output bounds are:

\[
P_g^{\min}u_{g,t}\le p_{g,t}\le
\alpha_{g,t}P_g^{\max}u_{g,t}.
\]

When `must_run` is true, \(u_{g,t}=1\) for every period. Start-up and shutdown
variables \(y_{g,t},z_{g,t}\in\{0,1\}\) satisfy

\[
u_{g,t}-u_{g,t-1}=y_{g,t}-z_{g,t}.
\]

For \(t=1\), \(u_{g,t-1}\) is the configured initial commitment state for unit
\(g\).

They are mutually exclusive in each period:

\[
y_{g,t} + z_{g,t} \le 1.
\]

Minimum up and down durations are configured in hours and converted to periods
with a conservative ceiling rule:

\[
N_g^{\uparrow}=\left\lceil H_g^{\uparrow}/\Delta t\right\rceil,\quad
N_g^{\downarrow}=\left\lceil H_g^{\downarrow}/\Delta t\right\rceil.
\]

Residual initial up-time and down-time obligations are enforced at the start of
the horizon from the configured initial state. If the unit is initially on, the
first

\[
\left\lceil\max(0,H^{\uparrow}-H^{\uparrow}_0)/\Delta t\right\rceil
\]

periods are forced on. If the unit is initially off, the corresponding residual
minimum down periods are forced off.

Ramping constraints use explicit start-up and shutdown ramp limits. The
configuration interprets `startup_ramp_mw` as the maximum output allowed in a
startup period. It interprets `shutdown_ramp_mw` as the maximum output in the
period immediately before a shutdown, including the configured initial output
when the unit shuts down in period 1.

\[
p_{g,t}-p_{g,t-1}\le
R_g^{\uparrow}\Delta t\,u_{g,t-1}+S_g^{\uparrow}y_{g,t},
\]

\[
p_{g,t-1}-p_{g,t}\le
R_g^{\downarrow}\Delta t\,u_{g,t}+S_g^{\downarrow}z_{g,t}.
\]

Because this formulation does not model multi-period startup or shutdown
trajectories, both transition limits must be at least \(P^{\min}\). Otherwise a
unit could be configured so that it cannot physically move between off and its
minimum stable output.

Minimum up and down times are imposed through rolling sums of recent starts and
shutdowns:

\[
\sum_{k=\max(1,t-N_g^{\uparrow}+1)}^t y_{g,k} \le u_{g,t},
\]

\[
\sum_{k=\max(1,t-N_g^{\downarrow}+1)}^t z_{g,k} \le 1-u_{g,t}.
\]

### Terminal commitment policy

Minimum up/down equations only look backward from modeled periods, so a
transition close to the horizon end needs an explicit terminal policy. The
thermal configuration supports:

- `forbid_incomplete_transitions`: the default for standalone finite-horizon
  studies. A startup or shutdown is allowed only if its full minimum-duration
  window fits inside the modeled horizon.
- `carry_residual_obligations`: allows terminal transitions and reports the
  remaining minimum up/down obligations for a later rolling-horizon solve.
- `fixed_terminal_commitment`: applies the strict transition rule and fixes
  \(u_T\) to `terminal_on`.

The unsupported `terminal_cost_approximation` policy was considered but not
implemented because no calibrated terminal value is available in the standalone
model.

For the strict and fixed policies, transitions are forbidden when they cannot
complete their minimum-duration windows:

\[
y_{g,t}=0 \quad \forall g,\ t>T-N_g^{\uparrow}+1,
\]

\[
z_{g,t}=0 \quad \forall g,\ t>T-N_g^{\downarrow}+1.
\]

For example, in a three-period hourly horizon with \(H^{\uparrow}=3\), a startup
in period 1 is feasible because periods 1, 2, and 3 complete the obligation. A
startup in period 2 or 3 is forbidden in strict standalone mode. In carry-forward
mode, a period-3 startup is feasible and the result reports two residual
minimum-up hours.

## Battery

The state of charge evolves as

\[
e_t=e_{t-1}+\eta_c c_t^{\mathrm{bat}}\Delta t
-\frac{d_t^{\mathrm{bat}}\Delta t}{\eta_d}.
\]

Power and energy bounds are applied each period. A binary charge-mode variable
\(m_t^{\mathrm{bat}}\in\{0,1\}\) enforces exact charge/discharge exclusivity:

\[
c_t^{\mathrm{bat}}\le P^{\max}_{\mathrm{bat}}m_t^{\mathrm{bat}},
\]

\[
d_t^{\mathrm{bat}}\le P^{\max}_{\mathrm{bat}}(1-m_t^{\mathrm{bat}}).
\]

The terminal state of charge supports four modes: minimum final state, exact
final state, cyclic final state equal to the initial state, and unconstrained.

## Objective

The objective minimizes:

\[
\sum_t \Delta t\left[
C^{\mathrm{var}}p_t+C^{\mathrm{imp}}i_t
+C^{\mathrm{lost}}\ell_t
+C^{\mathrm{bat}}(c_t^{\mathrm{bat}}+d_t^{\mathrm{bat}})
+C^{\mathrm{CO2}}(\gamma_p p_t+\gamma_i i_t)
\right]
\]

plus no-load, start-up, shutdown, and renewable-curtailment costs.

Reported cost components reconcile the solver objective into thermal variable,
thermal no-load, startup, shutdown, import energy, battery throughput, thermal
carbon, import carbon, renewable curtailment, dispatch load-shedding, and
network-capacity load-shedding costs. The simulator raises an error if the
reported component sum diverges from the objective beyond the configured
tolerance.

## Solver status and reconciliation

The simulator separates domain solver status from raw backend status. Domain
statuses are `optimal`, `feasible_limit`, `infeasible`, `unbounded`,
`infeasible_or_unbounded`, `solver_error`, `interrupted`, and `no_incumbent`.
The manifest also records the SciPy/HiGHS backend status code, backend status
name, and termination message.

A feasible non-optimal incumbent from a time or node limit is accepted only when
explicitly enabled in configuration. A limit-reached solve without an incumbent
is reported as `no_incumbent` and rejected. Objective bounds, absolute gaps, and
relative gaps are reported only when the backend provides finite values and the
relative-gap denominator is meaningful.

Per-period reconciliation columns report source balance residual, delivered
demand balance residual, battery energy residual, curtailment residual, network
losses, and unserved energy.

## Numerical Policy

The simulator uses an immutable numerical policy with separate absolute
tolerances for primal power feasibility, energy reconciliation, binary
integrality, objective reconciliation in EUR, near-zero non-negativity cleanup,
report rounding, timestamp spacing, and DC power-balance checks. Physical
validation uses feasibility and reconciliation tolerances; report rounding is
not used to validate equations.

Residual summaries report the equation family, worst period, maximum absolute
residual, local scale, and scale-normalised residual. For very small systems,
absolute tolerances dominate because relative residuals can be unstable near
zero. For very large systems, the scale-normalised residual helps identify
whether a small absolute residual is material relative to the period's dispatch
scale.

## Standalone DC Power Flow

The `solve_dc_power_flow` utility is separate from the aggregate dispatch
network. It solves bus voltage angles and line flows for specified fixed
injections on a connected, lossless DC network with exactly one slack bus
provided by the `slack_bus` argument. Positive injections represent generation;
negative injections represent load. Net injections must balance before solving.

Line ratings are checked after the power flow is solved. They are not enforced
and the function does not perform capacity-constrained optimal power flow. The
default overload policy is `report`, which returns per-line diagnostics with
line ID, MW flow, rating, utilisation, overload amount, and overloaded flag. The
optional `raise` policy fails when any overload is detected. The utility omits
losses, reactive power, voltage magnitudes, unit dispatch, and redispatch.

## Interpretation limits

The aggregated network is an energy-planning approximation. It does not represent voltage, reactive power, phase imbalance, protection, or transient stability. The DC power-flow module provides a separate linear nodal approximation for transmission-oriented experiments.
