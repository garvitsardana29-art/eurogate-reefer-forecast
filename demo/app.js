const zoneDetails = {
  overview: {
    title: "Terminal Forecast",
    description:
      "The challenge predicts one combined hourly reefer electricity load series for the terminal. This is a terminal-wide forecast, not a separate forecast for each point on the map.",
    facts: [
      ["Forecast target", "Combined hourly reefer electricity usage for the full terminal"],
      ["What the map means", "The extra points are weather reference locations inside the same terminal context"],
      ["Observed history", "8,403 hourly rows from January 2025 to January 2026"],
      ["Operational signals", "Container count, setpoint mix, stack-tier proxies, and hardware mix"],
    ],
  },
  zentralgate: {
    title: "Zentralgate",
    description:
      "Zentralgate is one of the named weather reference locations in the package. It does not have its own separate forecast curve; it provides environmental context for the terminal-wide model.",
    facts: [
      ["Role in package", "Temperature, wind, and wind-direction reference source"],
      ["How to read this", "Switching here changes the context panel, not the global terminal forecast chart"],
      ["Model note", "Weather was tested but was not the dominant driver of the final selected forecast"],
    ],
  },
  vc_halle3: {
    title: "VC Halle 3",
    description:
      "VC Halle 3 is the second named weather reference location in the package and offers another environmental view of the same terminal conditions.",
    facts: [
      ["Role in package", "Temperature, wind, and wind-direction reference source"],
      ["Why it matters", "Helps describe spatial weather context inside the same terminal"],
      ["Model note", "Used during feature testing and for explaining the package structure"],
    ],
  },
};

const modelDisplay = {
  blend: { key: "blend", name: "Strict Day-Ahead Blend", color: "#145b62", p90Color: "#d26842" },
  baseline: { key: "baseline", name: "Baseline Recursive", color: "#577590", p90Color: "#b08968" },
  xgb: { key: "xgb", name: "Recursive XGBoost", color: "#0d7f72", p90Color: "#ef8354" },
};

const zoneSelect = document.getElementById("zone-select");
const mapNodes = Array.from(document.querySelectorAll(".map-node"));
const zoneTitle = document.getElementById("zone-title");
const zoneDescription = document.getElementById("zone-description");
const zoneFacts = document.getElementById("zone-facts");
const modelSelect = document.getElementById("model-select");
const chartMetrics = document.getElementById("chart-metrics");
const svg = document.getElementById("forecast-chart");
const uploadForm = document.getElementById("upload-form");
const uploadStatus = document.getElementById("upload-status");
const uploadResult = document.getElementById("upload-result");

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

  const peakThreshold = percentile(actualSeries, 0.9);
  const maeAll = mean(actualSeries.map((value, idx) => Math.abs(value - predSeries[idx])));
  const peakErrors = actualSeries
    .map((value, idx) => ({ value, err: Math.abs(value - predSeries[idx]) }))
    .filter((item) => item.value >= peakThreshold)
    .map((item) => item.err);
  const maePeak = mean(peakErrors);
  const pinball = mean(actualSeries.map((value, idx) => pinballLoss(value, p90Series[idx], 0.9)));
  const combined = 0.5 * maeAll + 0.3 * maePeak + 0.2 * pinball;

  chartMetrics.innerHTML = `
    <span><strong>${model.name}</strong></span>
    <span>MAE All <strong>${maeAll.toFixed(3)}</strong></span>
    <span>MAE Peak <strong>${maePeak.toFixed(3)}</strong></span>
    <span>Pinball P90 <strong>${pinball.toFixed(3)}</strong></span>
    <span>Combined <strong>${combined.toFixed(3)}</strong></span>
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
  return rows;
}

async function init() {
  updateZone("overview");

  try {
    const health = await fetch("/api/health").then((res) => res.json());
    uploadStatus.innerHTML = `<p>Backend status: <strong>${health.status}</strong> · ${health.timestamp_utc}</p>`;
  } catch (error) {
    uploadStatus.innerHTML = `<p>Backend status: <strong>offline</strong>. Start the backend with <code>uvicorn backend_api:app --host 0.0.0.0 --port 8000</code>.</p>`;
  }

  try {
    const rows = await loadData();
    buildChart(rows, "blend");
    modelSelect.addEventListener("change", (event) => buildChart(rows, event.target.value));
  } catch (error) {
    chartMetrics.innerHTML = `<span>Could not load CSV data. Run a local server like <strong>python3 -m http.server 8000</strong> from the participant package folder.</span>`;
    console.error(error);
  }

  uploadForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const formData = new FormData(uploadForm);
    uploadResult.innerHTML = "<p>Uploading files and running the forecast pipeline...</p>";

    try {
      const response = await fetch("/api/jobs", {
        method: "POST",
        body: formData,
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload.detail, null, 2));
      }

      const preview = (payload.preview || [])
        .map(
          (row) =>
            `${row.timestamp_utc}, ${row.pred_power_kw}, ${row.pred_p90_kw}`
        )
        .join("\n");

      uploadResult.innerHTML = `
        <p><strong>Job completed:</strong> ${payload.job_id}</p>
        <p><strong>Rows written:</strong> ${payload.row_count}</p>
        <p><a href="${payload.output_url}" target="_blank" rel="noreferrer">Download generated predictions</a></p>
        <pre>${preview || "No preview available."}</pre>
      `;
    } catch (error) {
      uploadResult.innerHTML = `<p><strong>Run failed.</strong></p><pre>${String(error.message || error)}</pre>`;
    }
  });
}

init();
