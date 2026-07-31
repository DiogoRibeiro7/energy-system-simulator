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

## Renewable generation

Available renewable production is the sum of solar and wind availability:

\[
a_t^{\mathrm{ren}}=a_t^{\mathrm{solar}}+a_t^{\mathrm{wind}}.
\]

The used renewable variable satisfies

\[
0\le r_t\le a_t^{\mathrm{ren}}.
\]

Curtailment is \(a_t^{\mathrm{ren}}-r_t\).

## Power balance

All dispatch variables are represented on the source side:

\[
r_t+p_t+d_t^{\mathrm{bat}}+i_t+\ell_t
=g_t+c_t^{\mathrm{bat}}.
\]

Here \(\ell_t\) is source-equivalent involuntary load shedding. Delivered involuntary shedding is \(\eta_n\ell_t\).

## Thermal generator

For commitment status \(u_t\in\{0,1\}\):

\[
P^{\min}u_t\le p_t\le P^{\max}u_t.
\]

Start-up and shutdown variables \(y_t,z_t\in\{0,1\}\) satisfy

\[
u_t-u_{t-1}=y_t-z_t.
\]

They are mutually exclusive in each period:

\[
y_t + z_t \le 1.
\]

Minimum up and down durations are configured in hours and converted to periods
with a conservative ceiling rule:

\[
N^{\uparrow}=\left\lceil H^{\uparrow}/\Delta t\right\rceil,\quad
N^{\downarrow}=\left\lceil H^{\downarrow}/\Delta t\right\rceil.
\]

Residual initial up-time and down-time obligations are enforced at the start of
the horizon from the configured initial state.

Ramping constraints use explicit start-up and shutdown ramp limits:

\[
p_t-p_{t-1}\le R^{\uparrow}\Delta t\,u_{t-1}+S^{\uparrow}y_t,
\]

\[
p_{t-1}-p_t\le R^{\downarrow}\Delta t\,u_t+S^{\downarrow}z_t.
\]

Minimum up and down times are imposed through rolling sums of recent starts and
shutdowns.

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

The simulator reports solver status, primal objective, objective bound, absolute
gap, relative gap, runtime, and node count when the SciPy HiGHS result exposes
them. A feasible non-optimal incumbent is accepted only when explicitly enabled
in configuration.

Per-period reconciliation columns report source balance residual, delivered
demand balance residual, battery energy residual, curtailment residual, network
losses, and unserved energy.

## Interpretation limits

The aggregated network is an energy-planning approximation. It does not represent voltage, reactive power, phase imbalance, protection, or transient stability. The DC power-flow module provides a separate linear nodal approximation for transmission-oriented experiments.
