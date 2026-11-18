const params = new URLSearchParams(window.location.search);
let API_URL = (params.get("api") || localStorage.getItem("pathsnap_api") ||
  window.PATHSNAP_API_URL || "http://localhost:8000").replace(/\/+$/, "");
const apiInput = document.querySelector("#api");
apiInput.value = API_URL;
apiInput.addEventListener("change", () => {
  API_URL = apiInput.value.trim().replace(/\/+$/, "");
  localStorage.setItem("pathsnap_api", API_URL);
  predictButton.disabled = true;
  loadTrips();
});

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
    const wanted = params.get("trip");  // optional: ?trip=00000052 auto-runs (demo screenshots)
    if (wanted && [...tripSelect.options].some(o => o.value === wanted)) {
      tripSelect.value = wanted;
      predictButton.click();
    }
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
  const projectedList = projectPoints(nodes);
  const projected = {};  // keyed by node_id (the API sends only nearby nodes)
  nodes.forEach((node, i) => { projected[node.node_id] = projectedList[i]; });
  context.strokeStyle = "#2c2c31";
  context.lineWidth = 0.6;
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
    context.lineWidth = 3;
    context.lineJoin = "round";
    context.beginPath();
    let started = false;
    const arcById = new Map(latestResponse.map_geometry.arcs.map(item => [item.arc_id, item]));
    for (const arcId of latestResponse.predicted_route) {
      const arc = arcById.get(arcId);
      if (!arc) continue;
      const from = projected[arc.from_node];
      const to = projected[arc.to_node];
      if (!from || !to) continue;
      context.moveTo(from.x, from.y);  // each arc drawn on its own: no invented connectors
      context.lineTo(to.x, to.y);
      started = true;
    }
    context.stroke();
  }
}

const METRIC_LABELS = {
  observation_accuracy: "GPS points on the right road",
  observations_correct: "Points correct",
  observations_total: "Points in trip (all)",
  unmatched_observations: "Points with no predicted road",
  route_precision: "Route precision",
  route_recall: "Route recall",
};

function showMetrics(metrics) {
  metricsElement.innerHTML = "";
  if (!metrics) return;
  for (const [name, value] of Object.entries(metrics)) {
    const item = document.createElement("div");
    item.className = "metric";
    const isRatio = typeof value === "number" && /accuracy|precision|recall/.test(name);
    const shown = isRatio ? (value * 100).toFixed(1) + "%" : (typeof value === "number" ? value : JSON.stringify(value));
    item.textContent = `${METRIC_LABELS[name] || name.replace(/_/g, " ")}: ${shown}`;
    metricsElement.appendChild(item);
  }
  const help = document.createElement("div");
  help.className = "metric";
  help.style.gridColumn = "1 / -1";
  help.style.color = "#aaa7a0";
  help.textContent = "Accuracy: share of ALL GPS points whose predicted road equals the reference road (unmatched points count as wrong). " +
    "Route precision / recall compare the SET of predicted roads with the set of reference roads (direction matters, order and repeats ignored); they are not route-sequence accuracy.";
  metricsElement.appendChild(help);
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
    setStatus(`Trip ${latestResponse.trip_id}: ${latestResponse.predicted_route.length} predicted arcs, ${latestResponse.n_no_candidates} of ${latestResponse.gps_observations.length} GPS points had no candidate road within 50 m.`);
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
