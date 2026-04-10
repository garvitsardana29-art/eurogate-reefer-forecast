const uploadForm = document.getElementById("upload-form");
const uploadStatus = document.getElementById("upload-status");
const uploadResult = document.getElementById("upload-result");

async function initRerunPage() {
  try {
    const health = await fetch("/api/health").then((res) => res.json());
    uploadStatus.innerHTML = `<p>Backend status: <strong>${health.status}</strong> · ${health.timestamp_utc}</p>`;
  } catch (error) {
    uploadStatus.innerHTML = `<p>Backend status: <strong>offline</strong>. Start the backend with <code>uvicorn backend_api:app --host 0.0.0.0 --port 8000</code>.</p>`;
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
        .map((row) => `${row.timestamp_utc}, ${row.pred_power_kw}, ${row.pred_p90_kw}`)
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

initRerunPage();
