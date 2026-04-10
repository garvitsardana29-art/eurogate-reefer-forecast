const zoneDetails = {
  overview: {
    title: "Terminal Forecast",
    description:
      "One combined hourly reefer load forecast for the terminal.",
    facts: [
      ["Target", "Combined terminal reefer electricity usage"],
      ["Training window", "Model fit uses history before the first target hour"],
      ["Signals", "Container count, setpoint mix, stack tiers, hardware mix"],
    ],
  },
  zentralgate: {
    title: "Zentralgate",
    description:
      "Weather reference point used for terminal context.",
    facts: [
      ["Role", "Temperature, wind, and wind-direction source"],
      ["Chart", "Changes this side panel, not the terminal forecast chart"],
    ],
  },
  vc_halle3: {
    title: "VC Halle 3",
    description:
      "Second weather reference point in the same terminal package.",
    facts: [
      ["Role", "Temperature, wind, and wind-direction source"],
      ["Use", "Provides a second environmental anchor for context"],
    ],
  },
};

const modelDisplay = {
  blend: { key: "blend", name: "Strict Day-Ahead Blend", color: "#145b62", p90Color: "#d26842" },
  baseline: { key: "baseline", name: "Recursive Baseline Reference", color: "#577590", p90Color: "#b08968" },
  xgb: { key: "xgb", name: "Recursive XGBoost Reference", color: "#0d7f72", p90Color: "#ef8354" },
};

const zoneSelect = document.getElementById("zone-select");
const mapNodes = Array.from(document.querySelectorAll(".map-node"));
const zoneTitle = document.getElementById("zone-title");
const zoneDescription = document.getElementById("zone-description");
const zoneFacts = document.getElementById("zone-facts");
const modelSelect = document.getElementById("model-select");
const chartMetrics = document.getElementById("chart-metrics");
const svg = document.getElementById("forecast-chart");
const heroCombinedScore = document.getElementById("hero-combined-score");
const heroScoreNote = document.getElementById("hero-score-note");
const heroMaeAll = document.getElementById("hero-mae-all");
const heroMaePeak = document.getElementById("hero-mae-peak");
const heroPinball = document.getElementById("hero-pinball");
const heroCoverage = document.getElementById("hero-coverage");
const statHistoryHours = document.getElementById("stat-history-hours");
const statHistoryRange = document.getElementById("stat-history-range");
const statAvgLoad = document.getElementById("stat-avg-load");
const statAvgActive = document.getElementById("stat-avg-active");
const statPeakActive = document.getElementById("stat-peak-active");
function updateZone(zoneKey) {
  const zone = zoneDetails[zoneKey];
  if (!zone) return;

  zoneTitle.textContent = zone.title;
  zoneDescription.textContent = zone.description;
  zoneFacts.innerHTML = zone.facts
    .map(([label, text]) => `<li><strong>${label}</strong><span>${text}</span></li>`)
    .join("");

  zoneSelect.value = zoneKey;
  mapNodes.forEach((node) => {
    node.classList.toggle("active", node.dataset.zone === zoneKey);
  });
}

zoneSelect.addEventListener("change", (event) => {
  updateZone(event.target.value);
});

mapNodes.forEach((node) => {
  node.addEventListener("click", () => updateZone(node.dataset.zone));
});

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const cols = line.split(",");
    return Object.fromEntries(headers.map((header, idx) => [header, cols[idx] ?? ""]));
  });
}

function mean(values) {
  if (!values.length) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function pinballLoss(actual, predQ, q) {
  if (actual >= predQ) return q * (actual - predQ);
  return (1 - q) * (predQ - actual);
}

function percentile(values, q) {
  if (!values.length) return 0;
  const ordered = [...values].sort((a, b) => a - b);
  const idx = (ordered.length - 1) * q;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return ordered[lo];
  const frac = idx - lo;
  return ordered[lo] * (1 - frac) + ordered[hi] * frac;
}

function polylinePath(points) {
  return points
    .map((point, idx) => `${idx === 0 ? "M" : "L"} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`)
    .join(" ");
}

function computeMetrics(dataRows, modelKey) {
  const actualSeries = dataRows.map((row) => row.actual);
  const predSeries = dataRows.map((row) => row[modelKey]);
  const p90Series = dataRows.map((row) => row[`${modelKey}_p90`]);
  const peakThreshold = percentile(actualSeries, 0.9);
  const maeAll = mean(actualSeries.map((value, idx) => Math.abs(value - predSeries[idx])));
  const peakErrors = actualSeries
    .map((value, idx) => ({ value, err: Math.abs(value - predSeries[idx]) }))
    .filter((item) => item.value >= peakThreshold)
    .map((item) => item.err);
  const maePeak = mean(peakErrors);
  const pinball = mean(actualSeries.map((value, idx) => pinballLoss(value, p90Series[idx], 0.9)));
  const combined = 0.5 * maeAll + 0.3 * maePeak + 0.2 * pinball;
  const coverage = mean(actualSeries.map((value, idx) => (value <= p90Series[idx] ? 1 : 0)));
  return { maeAll, maePeak, pinball, combined, coverage };
}

function updateHeroMetrics(dataRows) {
  const metrics = computeMetrics(dataRows, "blend");
  heroCombinedScore.textContent = metrics.combined.toFixed(3);
  heroMaeAll.textContent = `${metrics.maeAll.toFixed(3)} kW`;
  heroMaePeak.textContent = `${metrics.maePeak.toFixed(3)} kW`;
  heroPinball.textContent = metrics.pinball.toFixed(3);
  heroCoverage.textContent = metrics.coverage.toFixed(3);
  heroScoreNote.textContent = "Computed from the current strict final CSV and visible target block";
}

function updateHistoryStats(hourlyRows, firstTargetTimestamp) {
  const observed = hourlyRows.filter((row) => row.is_observed_hour === "1");
  const trainingRows = observed.filter((row) => row.timestamp_utc < firstTargetTimestamp);
  if (!trainingRows.length) return;

  const timestamps = trainingRows.map((row) => row.timestamp_utc).sort();
  const loads = trainingRows.map((row) => Number(row.terminal_total_kw)).filter((value) => !Number.isNaN(value));
  const actives = trainingRows
    .map((row) => Number(row.active_container_count))
    .filter((value) => !Number.isNaN(value));

  statHistoryHours.textContent = `${trainingRows.length.toLocaleString("en-US")} hours`;
  statHistoryRange.textContent = `${timestamps[0].slice(0, 10)} to ${timestamps[timestamps.length - 1].slice(0, 10)}`;
  statAvgLoad.textContent = `${mean(loads).toFixed(3)} kW`;
  statAvgActive.textContent = mean(actives).toFixed(3);
  statPeakActive.textContent = `Peak observed count: ${Math.max(...actives).toFixed(0)}`;
}

function buildChart(dataRows, modelKey) {
  const model = modelDisplay[modelKey];
  const actualSeries = dataRows.map((row) => row.actual);
  const predSeries = dataRows.map((row) => row[model.key]);
  const p90Series = dataRows.map((row) => row[`${model.key}_p90`]);

  const values = [...actualSeries, ...predSeries, ...p90Series];
  const minValue = Math.min(...values) * 0.96;
  const maxValue = Math.max(...values) * 1.03;

  const width = 980;
  const height = 360;
  const margin = { top: 20, right: 28, bottom: 42, left: 54 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  const x = (idx) => margin.left + (idx / (dataRows.length - 1)) * innerWidth;
  const y = (value) =>
    margin.top + innerHeight - ((value - minValue) / (maxValue - minValue || 1)) * innerHeight;

  const actualPoints = actualSeries.map((value, idx) => ({ x: x(idx), y: y(value) }));
  const predPoints = predSeries.map((value, idx) => ({ x: x(idx), y: y(value) }));
  const p90Points = p90Series.map((value, idx) => ({ x: x(idx), y: y(value) }));

  const yTicks = Array.from({ length: 5 }, (_, idx) => minValue + ((maxValue - minValue) * idx) / 4);
  const xTickIndices = [0, 48, 96, 144, 192, dataRows.length - 1];

  const grid = yTicks
    .map((tick) => {
      const tickY = y(tick);
      return `
        <line x1="${margin.left}" y1="${tickY}" x2="${width - margin.right}" y2="${tickY}" stroke="rgba(23,33,38,0.08)" />
        <text x="${margin.left - 12}" y="${tickY + 4}" text-anchor="end" font-size="11" fill="#6b7a82">${tick.toFixed(0)}</text>
      `;
    })
    .join("");

  const xTicks = xTickIndices
    .map((idx) => {
      const tickX = x(idx);
      const label = dataRows[idx].timestamp.slice(5, 16).replace("T", " ");
      return `
        <line x1="${tickX}" y1="${height - margin.bottom}" x2="${tickX}" y2="${height - margin.bottom + 6}" stroke="rgba(23,33,38,0.14)" />
        <text x="${tickX}" y="${height - 12}" text-anchor="middle" font-size="11" fill="#6b7a82">${label}</text>
      `;
    })
    .join("");

  svg.innerHTML = `
    <rect x="0" y="0" width="${width}" height="${height}" fill="transparent"></rect>
    ${grid}
    <line x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}" stroke="rgba(23,33,38,0.18)" />
    <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}" stroke="rgba(23,33,38,0.18)" />
    ${xTicks}
    <path d="${polylinePath(p90Points)}" fill="none" stroke="${model.p90Color}" stroke-width="2.2" stroke-dasharray="6 6"></path>
    <path d="${polylinePath(predPoints)}" fill="none" stroke="${model.color}" stroke-width="3"></path>
    <path d="${polylinePath(actualPoints)}" fill="none" stroke="#172126" stroke-width="2.8"></path>
  `;

  const metrics = computeMetrics(dataRows, model.key);

  chartMetrics.innerHTML = `
    <span><strong>${model.name}</strong></span>
    <span>MAE All <strong>${metrics.maeAll.toFixed(3)}</strong></span>
    <span>MAE Peak <strong>${metrics.maePeak.toFixed(3)}</strong></span>
    <span>Pinball P90 <strong>${metrics.pinball.toFixed(3)}</strong></span>
    <span>P90 Coverage <strong>${metrics.coverage.toFixed(3)}</strong></span>
    <span>Combined <strong>${metrics.combined.toFixed(3)}</strong></span>
  `;
}

async function loadData() {
  const [actualCsv, baselineCsv, xgbCsv, blendCsv] = await Promise.all([
    fetch("../hourly_terminal_dataset.csv").then((res) => res.text()),
    fetch("../predictions.csv").then((res) => res.text()),
    fetch("../predictions_xgb_recursive.csv").then((res) => res.text()),
    fetch("../predictions_strict_day_ahead_blend.csv").then((res) => res.text()),
  ]);

  const actualRows = parseCsv(actualCsv)
    .filter((row) => row.is_observed_hour === "1")
    .filter((row) => row.timestamp_utc >= "2026-01-01T00:00:00Z" && row.timestamp_utc <= "2026-01-10T06:00:00Z")
    .map((row) => ({
      timestamp: row.timestamp_utc,
      actual: Number(row.terminal_total_kw),
    }));

  const byTs = Object.fromEntries(actualRows.map((row) => [row.timestamp, { ...row }]));

  const attachPredictions = (text, key) => {
    parseCsv(text).forEach((row) => {
      if (!byTs[row.timestamp_utc]) return;
      byTs[row.timestamp_utc][key] = Number(row.pred_power_kw);
      byTs[row.timestamp_utc][`${key}_p90`] = Number(row.pred_p90_kw);
    });
  };

  attachPredictions(baselineCsv, "baseline");
  attachPredictions(xgbCsv, "xgb");
  attachPredictions(blendCsv, "blend");

  const rows = Object.values(byTs).sort((a, b) => a.timestamp.localeCompare(b.timestamp));
  return { rows, hourlyRows: parseCsv(actualCsv) };
}

async function init() {
  updateZone("overview");

  try {
    const { rows, hourlyRows } = await loadData();
    updateHeroMetrics(rows);
    updateHistoryStats(hourlyRows, rows[0].timestamp);
    buildChart(rows, "blend");
    modelSelect.addEventListener("change", (event) => buildChart(rows, event.target.value));
  } catch (error) {
    chartMetrics.innerHTML = `<span>Could not load CSV data. Run a local server like <strong>python3 -m http.server 8000</strong> from the participant package folder.</span>`;
    console.error(error);
  }
}

init();
