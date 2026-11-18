const API_URL = window.PATHSNAP_API_URL || "http://localhost:8000";

const tripSelect = document.querySelector("#trip");
const predictButton = document.querySelector("#predict");
const statusElement = document.querySelector("#status");
const metricsElement = document.querySelector("#metrics");
const canvas = document.querySelector("#canvas");
const context = canvas.getContext("2d");

let latestResponse = null;

function setStatus(message) {
  statusElement.textContent = message;
}

async function apiRequest(path, options) {
  let response;
  try {
    response = await fetch(`${API_URL}${path}`, options);
  } catch (error) {
    throw new Error(`Cannot reach the backend at ${API_URL}. Start it on Laptop 1 and configure PATHSNAP_API_URL.`);
  }
  let body = {};
  try { body = await response.json(); } catch (_) {}
  if (!response.ok) {
    const detail = typeof body.detail === "string" ? body.detail : "The request failed.";
    throw new Error(detail);
  }
  return body;
}

async function loadTrips() {
  try {
    const data = await apiRequest("/trips");
    tripSelect.innerHTML = "";
    if (!data.trip_ids.length) {
      tripSelect.innerHTML = '<option>No trips available</option>';
      setStatus("No evaluation trips were found in the configured partition.");
      return;
    }
    for (const tripId of data.trip_ids) {
      const option = document.createElement("option");
      option.value = tripId;
      option.textContent = tripId;
      tripSelect.appendChild(option);
    }
    predictButton.disabled = false;
    setStatus("Select a trip and run prediction.");
  } catch (error) {
    tripSelect.innerHTML = '<option>Unavailable</option>';
    setStatus(error.message);
  }
}

function fitSize() {
  const rect = canvas.parentElement.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = rect.width * ratio;
  canvas.height = rect.height * ratio;
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  drawMap();
}

function projectPoints(points) {
  if (!points.length) return [];
  const xs = points.map(point => point.lon);
  const ys = points.map(point => point.lat);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const rect = canvas.getBoundingClientRect();
  const pad = 50;
  const scale = Math.min(
    (rect.width - pad * 2) / Math.max(maxX - minX, 0.000001),
    (rect.height - pad * 2) / Math.max(maxY - minY, 0.000001)
  );
  const offsetX = (rect.width - (maxX - minX) * scale) / 2;
  const offsetY = (rect.height - (maxY - minY) * scale) / 2;
  return points.map(point => ({
    x: offsetX + (point.lon - minX) * scale,
    y: rect.height - offsetY - (point.lat - minY) * scale,
  }));
}

function drawMap() {
  const rect = canvas.getBoundingClientRect();
  context.clearRect(0, 0, rect.width, rect.height);
  if (!latestResponse) return;

  const nodes = latestResponse.map_geometry.nodes;
  const projected = projectPoints(nodes);
  context.strokeStyle = "#3b3b40";
  context.lineWidth = 1;
  context.beginPath();
  for (const arc of latestResponse.map_geometry.arcs) {
    const from = projected[arc.from_node];
    const to = projected[arc.to_node];
    if (!from || !to) continue;
    context.moveTo(from.x, from.y);
    context.lineTo(to.x, to.y);
  }
  context.stroke();

  const gps = projectPoints(latestResponse.gps_observations);
  context.fillStyle = "#ff8066";
  for (const point of gps) {
    context.beginPath();
    context.arc(point.x, point.y, 2.2, 0, Math.PI * 2);
    context.fill();
  }

  if (latestResponse.predicted_route.length) {
    context.strokeStyle = "#d6b46a";
    context.lineWidth = 4;
    context.lineJoin = "round";
    context.beginPath();
    let started = false;
    for (const arcId of latestResponse.predicted_route) {
      const arc = latestResponse.map_geometry.arcs.find(item => item.arc_id === arcId);
      if (!arc) continue;
      const from = projected[arc.from_node];
      const to = projected[arc.to_node];
      if (!from || !to) continue;
      if (!started) context.moveTo(from.x, from.y);
      else context.lineTo(from.x, from.y);
      context.lineTo(to.x, to.y);
      started = true;
    }
    context.stroke();
  }
}

function showMetrics(metrics) {
  metricsElement.innerHTML = "";
  if (!metrics) return;
  for (const [name, value] of Object.entries(metrics)) {
    const item = document.createElement("div");
    item.className = "metric";
    item.textContent = `${name}: ${typeof value === "number" ? value : JSON.stringify(value)}`;
    metricsElement.appendChild(item);
  }
}

predictButton.addEventListener("click", async () => {
  predictButton.disabled = true;
  latestResponse = null;
  metricsElement.innerHTML = "";
  setStatus("Loading prediction…");
  try {
    latestResponse = await apiRequest("/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({trip_id: tripSelect.value}),
    });
    setStatus(`Trip ${latestResponse.trip_id}: ${latestResponse.predicted_route.length} predicted arcs.`);
    showMetrics(latestResponse.evaluation_metrics);
    drawMap();
  } catch (error) {
    setStatus(error.message);
  } finally {
    predictButton.disabled = false;
  }
});

window.addEventListener("resize", fitSize);
fitSize();
loadTrips();
