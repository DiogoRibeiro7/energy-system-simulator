from __future__ import annotations

# ruff: noqa: E501
import json
import math
from collections.abc import Mapping
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pandas as pd

MAX_DASHBOARD_POINTS = 2000


def write_dashboard(
    output_directory: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Write a self-contained HTML dashboard for a simulation output directory."""
    output = Path(output_directory)
    path = Path(output_path) if output_path is not None else output / "dashboard.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _dashboard_payload(output)
    path.write_text(_render_dashboard(payload), encoding="utf-8")
    return path


def write_dashboard_app(
    output_directory: str | Path,
    app_directory: str | Path | None = None,
) -> Path:
    """Write a structured local dashboard app for a simulation output directory."""
    output = Path(output_directory)
    app = Path(app_directory) if app_directory is not None else output / "dashboard"
    app.mkdir(parents=True, exist_ok=True)
    payload = _dashboard_payload(output)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    (app / "index.html").write_text(_render_dashboard_app_index(), encoding="utf-8")
    (app / "styles.css").write_text(_dashboard_styles(), encoding="utf-8")
    (app / "data.json").write_text(encoded + "\n", encoding="utf-8")
    (app / "data.js").write_text(
        "window.ENERGY_DASHBOARD_DATA = "
        + json.dumps(payload, allow_nan=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    (app / "app.js").write_text(_dashboard_app_js(), encoding="utf-8")
    return app / "index.html"


def serve_dashboard_app(
    output_directory: str | Path,
    app_directory: str | Path | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    """Serve the structured dashboard app with the Python standard library."""
    index = write_dashboard_app(output_directory, app_directory)
    handler = partial(SimpleHTTPRequestHandler, directory=str(index.parent))
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{server.server_port}/"
    print(f"Dashboard serving: {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard server stopped.")
    finally:
        server.server_close()


def _dashboard_payload(output: Path) -> dict[str, Any]:
    system_path = _first_existing(output / "system_timeseries_v1.csv", output / "timeseries.csv")
    if system_path is None:
        raise FileNotFoundError(
            f"No system time-series table found in {output}. "
            "Expected system_timeseries_v1.csv or timeseries.csv."
        )
    frame = pd.read_csv(system_path)
    if "timestamp" not in frame:
        raise ValueError(f"{system_path} does not contain a timestamp column")

    summary = _read_json(output / "summary.json")
    diagnostics = _read_json(output / "diagnostics.json")
    sampled = _sample_frame(frame)
    costs = _cost_components(output, summary)
    metric_names = (
        "objective_eur",
        "unserved_energy_mwh",
        "renewable_share_of_primary_generation",
        "renewable_used_mwh",
        "thermal_generation_mwh",
        "imports_mwh",
        "battery_discharge_mwh",
        "total_emissions_tonnes",
    )
    metrics = [
        {
            "name": name,
            "label": _metric_label(name),
            "value": _summary_number(summary, name),
            "unit": _metric_unit(name),
        }
        for name in metric_names
        if _summary_number(summary, name) is not None
    ]
    return {
        "schema_version": 1,
        "output_directory": str(output),
        "source_table": system_path.name,
        "periods": len(frame),
        "displayed_periods": len(sampled),
        "downsampled": len(sampled) != len(frame),
        "metrics": metrics,
        "costs": costs,
        "diagnostics": diagnostics.get("findings", []),
        "diagnostic_status": diagnostics.get("status", "unknown"),
        "series": {
            "timestamp": _column_values(sampled, "timestamp"),
            "end_user_demand_mw": _number_values(sampled, "end_user_demand_mw"),
            "renewable_used_mw": _number_values(sampled, "renewable_used_mw"),
            "thermal_output_mw": _number_values(sampled, "thermal_output_mw"),
            "imports_mw": _number_values(sampled, "imports_mw"),
            "battery_charge_mw": _number_values(sampled, "battery_charge_mw"),
            "battery_discharge_mw": _number_values(sampled, "battery_discharge_mw"),
            "battery_soc_mwh": _number_values(sampled, "battery_soc_mwh"),
            "total_load_shed_mw": _number_values(sampled, "total_load_shed_mw"),
            "thermal_emissions_tonnes": _number_values(sampled, "thermal_emissions_tonnes"),
            "import_emissions_tonnes": _number_values(sampled, "import_emissions_tonnes"),
        },
    }


def _first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _sample_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if len(frame) <= MAX_DASHBOARD_POINTS:
        return frame
    step = math.ceil(len(frame) / MAX_DASHBOARD_POINTS)
    sampled = frame.iloc[::step].copy()
    if sampled.index[-1] != frame.index[-1]:
        sampled = pd.concat([sampled, frame.tail(1)])
    return sampled.reset_index(drop=True)


def _column_values(frame: pd.DataFrame, column: str) -> list[str]:
    if column not in frame:
        return []
    return [str(value) for value in frame[column].tolist()]


def _number_values(frame: pd.DataFrame, column: str) -> list[float]:
    if column not in frame:
        return []
    values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return [float(value) for value in values.tolist()]


def _summary_number(summary: Mapping[str, Any], name: str) -> float | None:
    value = summary.get(name)
    if not isinstance(value, int | float):
        return None
    return float(value)


def _cost_components(output: Path, summary: Mapping[str, Any]) -> list[dict[str, float | str]]:
    path = output / "cost_components_v1.csv"
    if path.exists():
        table = pd.read_csv(path)
        if {"component", "value_eur"} <= set(table.columns):
            table_records: list[dict[str, float | str]]
            table_records = [
                {"component": str(row["component"]), "value_eur": float(row["value_eur"])}
                for _, row in table.iterrows()
                if float(row["value_eur"]) != 0.0
            ]
            return _top_costs(table_records)
    costs = summary.get("cost_components_eur", {})
    if not isinstance(costs, Mapping):
        return []
    summary_records: list[dict[str, float | str]]
    summary_records = [
        {"component": str(component), "value_eur": float(value)}
        for component, value in costs.items()
        if isinstance(value, int | float) and float(value) != 0.0
    ]
    return _top_costs(summary_records)


def _top_costs(
    records: list[dict[str, float | str]], *, limit: int = 12
) -> list[dict[str, float | str]]:
    ordered = sorted(records, key=lambda row: abs(float(row["value_eur"])), reverse=True)
    if len(ordered) <= limit:
        return ordered
    keep = ordered[: limit - 1]
    other = sum(float(row["value_eur"]) for row in ordered[limit - 1 :])
    keep.append({"component": "other", "value_eur": other})
    return keep


def _metric_label(name: str) -> str:
    return {
        "objective_eur": "Objective",
        "unserved_energy_mwh": "Unserved Energy",
        "renewable_share_of_primary_generation": "Renewable Share",
        "renewable_used_mwh": "Renewable Used",
        "thermal_generation_mwh": "Thermal Generation",
        "imports_mwh": "Imports",
        "battery_discharge_mwh": "Battery Discharge",
        "total_emissions_tonnes": "Total Emissions",
    }.get(name, name.replace("_", " ").title())


def _metric_unit(name: str) -> str:
    if name.endswith("_eur"):
        return "EUR"
    if name.endswith("_mwh"):
        return "MWh"
    if name.endswith("_tonnes"):
        return "tonnes"
    if "share" in name:
        return "%"
    return ""


def _render_dashboard_app_index() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Energy System Dashboard</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <aside class="sidebar">
    <div>
      <p class="eyebrow">Energy System Simulator</p>
      <h1>Dashboard</h1>
    </div>
    <nav class="nav" aria-label="Dashboard views">
      <button type="button" data-view="overview" aria-current="page">Overview</button>
      <button type="button" data-view="dispatch">Dispatch</button>
      <button type="button" data-view="storage">Storage</button>
      <button type="button" data-view="emissions">Emissions</button>
      <button type="button" data-view="costs">Costs</button>
    </nav>
    <div class="source" id="source"></div>
  </aside>
  <main class="workspace">
    <header class="topbar">
      <div>
        <h2 id="viewTitle">Overview</h2>
        <p id="viewSubtitle"></p>
      </div>
      <div class="status" id="diagnosticStatus"><span></span><strong></strong></div>
    </header>
    <section class="metrics" id="metrics"></section>
    <section class="layout">
      <div class="panel primary">
        <div class="panel-head">
          <h3 id="chartTitle">System Dispatch</h3>
          <div class="readout" id="readout"></div>
        </div>
        <svg id="mainChart" class="chart" role="img" aria-labelledby="chartTitle"></svg>
        <div class="legend" id="legend"></div>
      </div>
      <div class="panel">
        <div class="panel-head">
          <h3>Cost Breakdown</h3>
        </div>
        <svg id="costChart" class="chart compact" role="img" aria-label="Cost breakdown"></svg>
      </div>
      <div class="panel">
        <div class="panel-head">
          <h3>Diagnostics</h3>
        </div>
        <ul class="diagnostics" id="diagnostics"></ul>
      </div>
      <div class="panel">
        <div class="panel-head">
          <h3>Files</h3>
        </div>
        <div class="files" id="files"></div>
      </div>
    </section>
  </main>
  <script src="data.js"></script>
  <script src="app.js"></script>
</body>
</html>
"""


def _dashboard_styles() -> str:
    return """:root {
  --bg: #eef2f4;
  --surface: #ffffff;
  --surface-alt: #f8fafb;
  --ink: #172026;
  --muted: #62717c;
  --line: #d8e0e5;
  --accent: #0e7c7b;
  --renewable: #2f9e44;
  --thermal: #4263eb;
  --imports: #e67700;
  --storage: #9c36b5;
  --shed: #c92a2a;
  --emissions: #495057;
  color-scheme: light;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  min-height: 100vh;
  display: grid;
  grid-template-columns: 248px minmax(0, 1fr);
  background: var(--bg);
  color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.sidebar {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  gap: 24px;
  padding: 22px 18px;
  border-right: 1px solid var(--line);
  background: var(--surface);
}
.eyebrow {
  margin: 0 0 6px;
  color: var(--muted);
  font-size: 12px;
  text-transform: uppercase;
}
h1, h2, h3, p { margin-top: 0; }
h1 { margin-bottom: 0; font-size: 26px; letter-spacing: 0; }
h2 { margin-bottom: 4px; font-size: 22px; letter-spacing: 0; }
h3 { margin-bottom: 0; font-size: 15px; letter-spacing: 0; }
.nav { display: grid; gap: 6px; }
.nav button {
  min-height: 38px;
  border: 1px solid transparent;
  border-radius: 7px;
  background: transparent;
  color: var(--muted);
  cursor: pointer;
  font: inherit;
  text-align: left;
  padding: 8px 10px;
}
.nav button[aria-current="page"] {
  border-color: var(--line);
  background: var(--surface-alt);
  color: var(--ink);
  font-weight: 650;
}
.source {
  margin-top: auto;
  padding-top: 16px;
  border-top: 1px solid var(--line);
  color: var(--muted);
  font-size: 12px;
  overflow-wrap: anywhere;
}
.workspace {
  min-width: 0;
  padding: 18px;
}
.topbar {
  min-height: 64px;
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 18px;
  margin-bottom: 14px;
}
.topbar p { color: var(--muted); margin-bottom: 0; }
.status {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 7px 10px;
  border: 1px solid var(--line);
  border-radius: 999px;
  background: var(--surface);
  color: var(--muted);
  white-space: nowrap;
}
.status span {
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: var(--accent);
}
.status.error span { background: var(--shed); }
.metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 10px;
  margin-bottom: 14px;
}
.metric, .panel {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--surface);
}
.metric {
  min-height: 84px;
  padding: 12px 14px;
}
.metric small {
  display: block;
  color: var(--muted);
  font-size: 12px;
  text-transform: uppercase;
}
.metric strong {
  display: block;
  margin-top: 8px;
  font-size: 21px;
  white-space: nowrap;
}
.metric span {
  color: var(--muted);
  font-size: 12px;
  font-weight: 500;
}
.layout {
  display: grid;
  grid-template-columns: minmax(0, 2fr) minmax(320px, 1fr);
  gap: 14px;
}
.panel {
  min-width: 0;
  padding: 14px;
}
.panel.primary { grid-row: span 2; }
.panel-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  min-height: 28px;
  margin-bottom: 8px;
}
.chart {
  width: 100%;
  height: 430px;
  display: block;
}
.chart.compact { height: 250px; }
.legend, .files {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 14px;
  color: var(--muted);
  font-size: 12px;
}
.legend-item {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.swatch {
  width: 10px;
  height: 10px;
  border-radius: 2px;
}
.readout {
  color: var(--muted);
  font-size: 12px;
  text-align: right;
}
.diagnostics {
  margin: 0;
  padding: 0;
  list-style: none;
  display: grid;
  gap: 8px;
}
.diagnostics li {
  padding-top: 8px;
  border-top: 1px solid var(--line);
  color: var(--muted);
}
svg text { fill: var(--muted); font-size: 11px; }
.axis { stroke: var(--line); stroke-width: 1; }
.grid-line { stroke: var(--line); stroke-width: 1; opacity: 0.7; }
.hover-line { stroke: var(--ink); stroke-width: 1; opacity: 0.35; pointer-events: none; }
@media (max-width: 980px) {
  body { grid-template-columns: 1fr; }
  .sidebar { min-height: auto; border-right: 0; border-bottom: 1px solid var(--line); }
  .nav { grid-template-columns: repeat(5, minmax(0, 1fr)); }
  .nav button { text-align: center; }
  .layout { grid-template-columns: 1fr; }
  .panel.primary { grid-row: auto; }
  .topbar { flex-direction: column; }
}
@media (max-width: 640px) {
  .nav { grid-template-columns: 1fr 1fr; }
  .workspace { padding: 12px; }
  .chart { height: 330px; }
}
"""


def _dashboard_app_js() -> str:
    return """const data = window.ENERGY_DASHBOARD_DATA;
const palette = {
  demand: "#172026",
  renewable: "#2f9e44",
  thermal: "#4263eb",
  imports: "#e67700",
  storage: "#9c36b5",
  shed: "#c92a2a",
  emissions: "#495057"
};
const viewMap = {
  overview: {
    title: "Overview",
    subtitle: "System-level operating results and diagnostic status.",
    chart: "System Dispatch",
    unit: "MW",
    series: [
      ["end_user_demand_mw", "Demand", palette.demand],
      ["renewable_used_mw", "Renewable", palette.renewable],
      ["thermal_output_mw", "Thermal", palette.thermal],
      ["imports_mw", "Imports", palette.imports]
    ]
  },
  dispatch: {
    title: "Dispatch",
    subtitle: "Demand, supply, imports, and scarcity by period.",
    chart: "System Dispatch",
    unit: "MW",
    series: [
      ["end_user_demand_mw", "Demand", palette.demand],
      ["renewable_used_mw", "Renewable", palette.renewable],
      ["thermal_output_mw", "Thermal", palette.thermal],
      ["imports_mw", "Imports", palette.imports],
      ["total_load_shed_mw", "Load shed", palette.shed]
    ]
  },
  storage: {
    title: "Storage",
    subtitle: "Battery charge, discharge, and state of charge.",
    chart: "Storage Operation",
    unit: "MW / MWh",
    series: [
      ["battery_charge_mw", "Charge", palette.imports],
      ["battery_discharge_mw", "Discharge", palette.storage],
      ["battery_soc_mwh", "State of charge", palette.renewable]
    ]
  },
  emissions: {
    title: "Emissions",
    subtitle: "Thermal and import emissions by period.",
    chart: "Emissions",
    unit: "tonnes",
    series: [
      ["thermal_emissions_tonnes", "Thermal", palette.emissions],
      ["import_emissions_tonnes", "Imports", palette.imports]
    ]
  },
  costs: {
    title: "Costs",
    subtitle: "Objective cost composition and operating drivers.",
    chart: "System Dispatch",
    unit: "MW",
    series: [
      ["thermal_output_mw", "Thermal", palette.thermal],
      ["imports_mw", "Imports", palette.imports],
      ["battery_discharge_mw", "Battery discharge", palette.storage]
    ]
  }
};
function formatNumber(value, unit) {
  if (value === null || value === undefined) return "";
  if (unit === "%") return (value * 100).toFixed(1);
  if (Math.abs(value) >= 1000000) {
    return value.toLocaleString(undefined, { maximumFractionDigits: 1 });
  }
  if (Math.abs(value) >= 1000) {
    return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
  }
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
}
function setMetrics() {
  document.getElementById("metrics").innerHTML = data.metrics.map(metric => `
    <article class="metric">
      <small>${metric.label}</small>
      <strong>${formatNumber(metric.value, metric.unit)} <span>${metric.unit}</span></strong>
    </article>
  `).join("");
}
function setSource() {
  const sample = data.downsampled ? `Showing ${data.displayed_periods.toLocaleString()} sampled periods.` : "Showing every period.";
  document.getElementById("source").textContent = `${data.periods.toLocaleString()} periods. ${sample} Source: ${data.source_table}`;
}
function activeSeries(view) {
  return view.series
    .map(([key, label, color]) => [key, label, color, data.series[key] || []])
    .filter(item => item[3].length);
}
function drawLineChart(view) {
  const svg = document.getElementById("mainChart");
  const width = svg.clientWidth || 900;
  const height = svg.clientHeight || 420;
  const margin = { top: 18, right: 18, bottom: 38, left: 54 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const series = activeSeries(view);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = "";
  if (!series.length) {
    svg.innerHTML = `<text x="${width / 2}" y="${height / 2}" text-anchor="middle">No data</text>`;
    return;
  }
  const values = series.flatMap(item => item[3]);
  const maxY = Math.max(...values, 1);
  const minY = Math.min(0, ...values);
  const spanY = maxY - minY || 1;
  const count = Math.max(data.series.timestamp.length - 1, 1);
  const x = index => margin.left + innerW * index / count;
  const y = value => margin.top + innerH - ((value - minY) / spanY * innerH);
  for (let tick = 0; tick <= 4; tick += 1) {
    const yy = margin.top + innerH * tick / 4;
    const value = maxY - spanY * tick / 4;
    svg.insertAdjacentHTML("beforeend", `<line class="grid-line" x1="${margin.left}" y1="${yy}" x2="${width - margin.right}" y2="${yy}"></line>`);
    svg.insertAdjacentHTML("beforeend", `<text x="${margin.left - 8}" y="${yy + 4}" text-anchor="end">${formatNumber(value, "")}</text>`);
  }
  svg.insertAdjacentHTML("beforeend", `<line class="axis" x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}"></line>`);
  svg.insertAdjacentHTML("beforeend", `<line class="axis" x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}"></line>`);
  series.forEach(([key, label, color, values]) => {
    const path = values.map((value, index) => `${index ? "L" : "M"}${x(index).toFixed(2)},${y(value).toFixed(2)}`).join(" ");
    svg.insertAdjacentHTML("beforeend", `<path d="${path}" fill="none" stroke="${color}" stroke-width="2" vector-effect="non-scaling-stroke"></path>`);
  });
  const hover = document.createElementNS("http://www.w3.org/2000/svg", "line");
  hover.setAttribute("class", "hover-line");
  hover.setAttribute("y1", String(margin.top));
  hover.setAttribute("y2", String(height - margin.bottom));
  hover.style.display = "none";
  svg.appendChild(hover);
  svg.onmousemove = event => {
    const rect = svg.getBoundingClientRect();
    const ratio = Math.min(Math.max((event.clientX - rect.left - margin.left) / innerW, 0), 1);
    const index = Math.round(ratio * Math.max(data.series.timestamp.length - 1, 0));
    hover.style.display = "block";
    hover.setAttribute("x1", String(x(index)));
    hover.setAttribute("x2", String(x(index)));
    document.getElementById("readout").textContent = [
      data.series.timestamp[index],
      ...series.map(([key, label, color, values]) => `${label}: ${formatNumber(values[index], view.unit)}`)
    ].join(" | ");
  };
  svg.onmouseleave = () => { hover.style.display = "none"; };
  document.getElementById("legend").innerHTML = series.map(([key, label, color]) =>
    `<span class="legend-item"><span class="swatch" style="background:${color}"></span>${label}</span>`
  ).join("");
}
function drawCosts() {
  const svg = document.getElementById("costChart");
  const width = svg.clientWidth || 420;
  const height = svg.clientHeight || 250;
  const margin = { top: 6, right: 18, bottom: 24, left: 150 };
  const rows = data.costs || [];
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = "";
  if (!rows.length) {
    svg.innerHTML = `<text x="${width / 2}" y="${height / 2}" text-anchor="middle">No cost data</text>`;
    return;
  }
  const innerW = width - margin.left - margin.right;
  const rowH = (height - margin.top - margin.bottom) / rows.length;
  const barH = Math.max(12, rowH - 4);
  const maxValue = Math.max(...rows.map(row => Math.abs(row.value_eur)), 1);
  rows.forEach((row, index) => {
    const y = margin.top + index * rowH;
    const w = Math.abs(row.value_eur) / maxValue * innerW;
    svg.insertAdjacentHTML("beforeend", `<text x="${margin.left - 8}" y="${y + barH * 0.75}" text-anchor="end">${row.component.replaceAll("_", " ")}</text>`);
    svg.insertAdjacentHTML("beforeend", `<rect x="${margin.left}" y="${y}" width="${w}" height="${barH}" fill="${palette.thermal}" rx="3"></rect>`);
  });
}
function setDiagnostics() {
  const status = document.getElementById("diagnosticStatus");
  status.classList.toggle("error", data.diagnostic_status === "error");
  status.querySelector("strong").textContent = `Diagnostics: ${data.diagnostic_status}`;
  const list = document.getElementById("diagnostics");
  if (!data.diagnostics.length) {
    list.innerHTML = "<li>No diagnostic findings.</li>";
    return;
  }
  list.innerHTML = data.diagnostics.map(item =>
    `<li><strong>${item.severity}</strong> ${item.check}: ${item.message}</li>`
  ).join("");
}
function setFiles() {
  const files = ["dashboard.html", "dashboard/index.html", "report.md", "summary.json", "system_timeseries_v1.csv", "asset_timeseries_v1.csv", "data_dictionary.csv"];
  document.getElementById("files").innerHTML = files.map(file =>
    `<span class="legend-item"><span class="swatch" style="background:var(--accent)"></span>${file}</span>`
  ).join("");
}
function setView(name) {
  const view = viewMap[name];
  document.getElementById("viewTitle").textContent = view.title;
  document.getElementById("viewSubtitle").textContent = view.subtitle;
  document.getElementById("chartTitle").textContent = view.chart;
  document.querySelectorAll("[data-view]").forEach(button => {
    button.setAttribute("aria-current", button.dataset.view === name ? "page" : "false");
  });
  drawLineChart(view);
}
setMetrics();
setSource();
setDiagnostics();
setFiles();
drawCosts();
setView("overview");
document.querySelectorAll("[data-view]").forEach(button => {
  button.addEventListener("click", () => setView(button.dataset.view));
});
window.addEventListener("resize", () => {
  const selected = document.querySelector("[data-view][aria-current='page']");
  setView(selected ? selected.dataset.view : "overview");
  drawCosts();
});
"""


def _render_dashboard(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, allow_nan=False, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Energy System Dashboard</title>
<style>
:root {{
  color-scheme: light;
  --bg: #f5f7f8;
  --panel: #ffffff;
  --ink: #172026;
  --muted: #5b6871;
  --line: #d9e0e4;
  --accent: #0e7c7b;
  --renewable: #2f9e44;
  --thermal: #4c6ef5;
  --imports: #f08c00;
  --storage: #ae3ec9;
  --shed: #e03131;
  --emissions: #495057;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
header {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 18px 24px;
  border-bottom: 1px solid var(--line);
  background: var(--panel);
}}
h1 {{
  margin: 0;
  font-size: 22px;
  font-weight: 650;
  letter-spacing: 0;
}}
.meta {{
  color: var(--muted);
  font-size: 13px;
  text-align: right;
}}
main {{
  max-width: 1480px;
  margin: 0 auto;
  padding: 18px 18px 28px;
}}
.metrics {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 10px;
  margin-bottom: 14px;
}}
.metric, .panel {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
}}
.metric {{
  padding: 12px 14px;
  min-height: 82px;
}}
.metric-label {{
  color: var(--muted);
  font-size: 12px;
  text-transform: uppercase;
}}
.metric-value {{
  margin-top: 8px;
  font-size: 21px;
  font-weight: 680;
  white-space: nowrap;
}}
.metric-unit {{
  margin-left: 4px;
  color: var(--muted);
  font-size: 12px;
  font-weight: 500;
}}
.grid {{
  display: grid;
  grid-template-columns: minmax(0, 2fr) minmax(300px, 1fr);
  gap: 14px;
}}
.panel {{
  min-width: 0;
  padding: 14px;
}}
.panel h2 {{
  margin: 0 0 10px;
  font-size: 15px;
  font-weight: 650;
}}
.toolbar {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 10px;
  margin-bottom: 8px;
}}
.tabs {{
  display: inline-flex;
  gap: 4px;
  padding: 3px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #eef2f4;
}}
button {{
  min-height: 30px;
  border: 0;
  border-radius: 6px;
  background: transparent;
  color: var(--muted);
  cursor: pointer;
  padding: 5px 10px;
  font: inherit;
}}
button[aria-selected="true"] {{
  background: var(--panel);
  color: var(--ink);
  box-shadow: 0 1px 2px rgb(0 0 0 / 8%);
}}
.chart {{
  width: 100%;
  height: 390px;
  display: block;
}}
.chart.small {{
  height: 235px;
}}
.legend {{
  display: flex;
  flex-wrap: wrap;
  gap: 10px 14px;
  color: var(--muted);
  font-size: 12px;
}}
.legend-item {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
}}
.swatch {{
  width: 10px;
  height: 10px;
  border-radius: 2px;
}}
.readout {{
  color: var(--muted);
  font-size: 12px;
  min-height: 18px;
}}
.status {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: var(--muted);
}}
.status-dot {{
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: var(--accent);
}}
.status.error .status-dot {{ background: var(--shed); }}
.diagnostics {{
  display: grid;
  gap: 8px;
  margin: 0;
  padding: 0;
  list-style: none;
}}
.diagnostics li {{
  border-top: 1px solid var(--line);
  padding-top: 8px;
  color: var(--muted);
}}
.empty {{
  color: var(--muted);
  padding: 26px 0;
  text-align: center;
}}
svg text {{
  fill: var(--muted);
  font-size: 11px;
}}
.axis {{ stroke: var(--line); stroke-width: 1; }}
.grid-line {{ stroke: var(--line); stroke-width: 1; opacity: 0.7; }}
.hover-line {{ stroke: var(--ink); stroke-width: 1; opacity: 0.35; pointer-events: none; }}
@media (max-width: 900px) {{
  header {{ align-items: flex-start; flex-direction: column; }}
  .meta {{ text-align: left; }}
  .grid {{ grid-template-columns: 1fr; }}
  .toolbar {{ align-items: flex-start; flex-direction: column; }}
  .chart {{ height: 330px; }}
}}
</style>
</head>
<body>
<header>
  <h1>Energy System Dashboard</h1>
  <div class="meta" id="runMeta"></div>
</header>
<main>
  <section class="metrics" id="metrics"></section>
  <section class="grid">
    <div class="panel">
      <div class="toolbar">
        <h2 id="chartTitle">System Dispatch</h2>
        <div class="tabs" role="tablist" aria-label="Dashboard view">
          <button type="button" data-view="dispatch" aria-selected="true">Dispatch</button>
          <button type="button" data-view="storage" aria-selected="false">Storage</button>
          <button type="button" data-view="emissions" aria-selected="false">Emissions</button>
        </div>
      </div>
      <svg class="chart" id="mainChart" role="img" aria-labelledby="chartTitle"></svg>
      <div class="legend" id="legend"></div>
      <div class="readout" id="readout"></div>
    </div>
    <div class="panel">
      <h2>Cost Breakdown</h2>
      <svg class="chart small" id="costChart" role="img" aria-label="Cost breakdown"></svg>
    </div>
    <div class="panel">
      <h2>Diagnostics</h2>
      <div class="status" id="diagnosticStatus"><span class="status-dot"></span><span></span></div>
      <ul class="diagnostics" id="diagnostics"></ul>
    </div>
    <div class="panel">
      <h2>Output Files</h2>
      <div id="files" class="legend"></div>
    </div>
  </section>
</main>
<script>
const data = {encoded};
const palette = {{
  demand: "#172026",
  renewable: "#2f9e44",
  thermal: "#4c6ef5",
  imports: "#f08c00",
  storage: "#ae3ec9",
  shed: "#e03131",
  emissions: "#495057"
}};
const views = {{
  dispatch: {{
    title: "System Dispatch",
    unit: "MW",
    series: [
      ["end_user_demand_mw", "Demand", palette.demand],
      ["renewable_used_mw", "Renewable", palette.renewable],
      ["thermal_output_mw", "Thermal", palette.thermal],
      ["imports_mw", "Imports", palette.imports],
      ["total_load_shed_mw", "Load shed", palette.shed]
    ]
  }},
  storage: {{
    title: "Storage Operation",
    unit: "MW / MWh",
    series: [
      ["battery_charge_mw", "Charge", palette.imports],
      ["battery_discharge_mw", "Discharge", palette.storage],
      ["battery_soc_mwh", "State of charge", palette.renewable]
    ]
  }},
  emissions: {{
    title: "Emissions",
    unit: "tonnes",
    series: [
      ["thermal_emissions_tonnes", "Thermal", palette.emissions],
      ["import_emissions_tonnes", "Imports", palette.imports]
    ]
  }}
}};
function number(value, unit) {{
  if (value === null || value === undefined) return "";
  if (unit === "%") return (value * 100).toFixed(1);
  if (Math.abs(value) >= 1000000) return value.toLocaleString(undefined, {{ maximumFractionDigits: 1 }});
  if (Math.abs(value) >= 1000) return value.toLocaleString(undefined, {{ maximumFractionDigits: 0 }});
  return value.toLocaleString(undefined, {{ maximumFractionDigits: 2 }});
}}
function setMetrics() {{
  const root = document.getElementById("metrics");
  root.innerHTML = data.metrics.map(metric => `
    <div class="metric">
      <div class="metric-label">${{metric.label}}</div>
      <div class="metric-value">${{number(metric.value, metric.unit)}}<span class="metric-unit">${{metric.unit}}</span></div>
    </div>`).join("");
  document.getElementById("runMeta").textContent =
    `${{data.periods.toLocaleString()}} periods from ${{data.source_table}}` +
    (data.downsampled ? `, showing ${{data.displayed_periods.toLocaleString()}}` : "");
}}
function pointsFor(seriesNames) {{
  return seriesNames
    .map(([key, label, color]) => [key, label, color, data.series[key] || []])
    .filter(item => item[3].length);
}}
function drawLineChart(svg, view) {{
  const width = svg.clientWidth || 900;
  const height = svg.clientHeight || 360;
  const margin = {{ top: 18, right: 18, bottom: 38, left: 54 }};
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const series = pointsFor(view.series);
  svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);
  svg.innerHTML = "";
  if (!series.length) {{
    svg.innerHTML = `<text x="${{width / 2}}" y="${{height / 2}}" text-anchor="middle">No data</text>`;
    return;
  }}
  const maxY = Math.max(...series.flatMap(item => item[3]), 1);
  const minY = Math.min(0, ...series.flatMap(item => item[3]));
  const spanY = maxY - minY || 1;
  const x = index => margin.left + (innerW * index / Math.max(data.series.timestamp.length - 1, 1));
  const y = value => margin.top + innerH - ((value - minY) / spanY * innerH);
  for (let tick = 0; tick <= 4; tick++) {{
    const yy = margin.top + innerH * tick / 4;
    const value = maxY - spanY * tick / 4;
    svg.insertAdjacentHTML("beforeend", `<line class="grid-line" x1="${{margin.left}}" y1="${{yy}}" x2="${{width - margin.right}}" y2="${{yy}}"></line>`);
    svg.insertAdjacentHTML("beforeend", `<text x="${{margin.left - 8}}" y="${{yy + 4}}" text-anchor="end">${{number(value, "")}}</text>`);
  }}
  svg.insertAdjacentHTML("beforeend", `<line class="axis" x1="${{margin.left}}" y1="${{height - margin.bottom}}" x2="${{width - margin.right}}" y2="${{height - margin.bottom}}"></line>`);
  svg.insertAdjacentHTML("beforeend", `<line class="axis" x1="${{margin.left}}" y1="${{margin.top}}" x2="${{margin.left}}" y2="${{height - margin.bottom}}"></line>`);
  series.forEach(([key, label, color, values]) => {{
    const d = values.map((value, index) => `${{index ? "L" : "M"}}${{x(index).toFixed(2)}},${{y(value).toFixed(2)}}`).join(" ");
    svg.insertAdjacentHTML("beforeend", `<path d="${{d}}" fill="none" stroke="${{color}}" stroke-width="2" vector-effect="non-scaling-stroke"></path>`);
  }});
  const hover = document.createElementNS("http://www.w3.org/2000/svg", "line");
  hover.setAttribute("class", "hover-line");
  hover.setAttribute("y1", String(margin.top));
  hover.setAttribute("y2", String(height - margin.bottom));
  hover.style.display = "none";
  svg.appendChild(hover);
  svg.onmousemove = event => {{
    const rect = svg.getBoundingClientRect();
    const ratio = Math.min(Math.max((event.clientX - rect.left - margin.left) / innerW, 0), 1);
    const index = Math.round(ratio * Math.max(data.series.timestamp.length - 1, 0));
    hover.style.display = "block";
    hover.setAttribute("x1", String(x(index)));
    hover.setAttribute("x2", String(x(index)));
    document.getElementById("readout").textContent = [
      data.series.timestamp[index],
      ...series.map(([key, label, color, values]) => `${{label}} ${{number(values[index], view.unit)}}`)
    ].join(" | ");
  }};
  svg.onmouseleave = () => {{ hover.style.display = "none"; }};
  document.getElementById("legend").innerHTML = series.map(([key, label, color]) =>
    `<span class="legend-item"><span class="swatch" style="background:${{color}}"></span>${{label}}</span>`
  ).join("");
}}
function drawCosts() {{
  const svg = document.getElementById("costChart");
  const width = svg.clientWidth || 420;
  const height = svg.clientHeight || 230;
  const margin = {{ top: 6, right: 18, bottom: 24, left: 150 }};
  const rows = data.costs || [];
  svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);
  svg.innerHTML = "";
  if (!rows.length) {{
    svg.innerHTML = `<text x="${{width / 2}}" y="${{height / 2}}" text-anchor="middle">No cost data</text>`;
    return;
  }}
  const innerW = width - margin.left - margin.right;
  const barH = Math.max(12, (height - margin.top - margin.bottom) / rows.length - 4);
  const maxValue = Math.max(...rows.map(row => Math.abs(row.value_eur)), 1);
  rows.forEach((row, index) => {{
    const y = margin.top + index * ((height - margin.top - margin.bottom) / rows.length);
    const w = Math.abs(row.value_eur) / maxValue * innerW;
    svg.insertAdjacentHTML("beforeend", `<text x="${{margin.left - 8}}" y="${{y + barH * 0.75}}" text-anchor="end">${{row.component.replaceAll("_", " ")}}</text>`);
    svg.insertAdjacentHTML("beforeend", `<rect x="${{margin.left}}" y="${{y}}" width="${{w}}" height="${{barH}}" fill="${{palette.thermal}}" rx="3"></rect>`);
  }});
}}
function setDiagnostics() {{
  const status = document.getElementById("diagnosticStatus");
  const text = status.querySelector("span:last-child");
  status.classList.toggle("error", data.diagnostic_status === "error");
  text.textContent = `Status: ${{data.diagnostic_status}}`;
  const list = document.getElementById("diagnostics");
  if (!data.diagnostics.length) {{
    list.innerHTML = `<li>No diagnostic findings.</li>`;
    return;
  }}
  list.innerHTML = data.diagnostics.map(item =>
    `<li><strong>${{item.severity}}</strong> ${{item.check}}: ${{item.message}}</li>`
  ).join("");
}}
function setFiles() {{
  document.getElementById("files").innerHTML = [
    "report.md",
    "summary.json",
    "system_timeseries_v1.csv",
    "asset_timeseries_v1.csv",
    "data_dictionary.csv"
  ].map(name => `<span class="legend-item"><span class="swatch" style="background:var(--accent)"></span>${{name}}</span>`).join("");
}}
function setView(name) {{
  const view = views[name];
  document.getElementById("chartTitle").textContent = view.title;
  document.querySelectorAll("[data-view]").forEach(button => {{
    button.setAttribute("aria-selected", String(button.dataset.view === name));
  }});
  drawLineChart(document.getElementById("mainChart"), view);
}}
setMetrics();
setDiagnostics();
setFiles();
drawCosts();
setView("dispatch");
document.querySelectorAll("[data-view]").forEach(button => {{
  button.addEventListener("click", () => setView(button.dataset.view));
}});
window.addEventListener("resize", () => {{
  const selected = document.querySelector("[data-view][aria-selected='true']");
  setView(selected ? selected.dataset.view : "dispatch");
  drawCosts();
}});
</script>
</body>
</html>
"""
