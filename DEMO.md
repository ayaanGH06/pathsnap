# Demo guide

## Roles
- Laptop 1: runs the API and holds the dataset. Everyone's console logs show on this screen.
- Laptop 2: only runs the UI (`http.server`) and a browser. It needs no dataset and no model.
- Both laptops on the same Wi-Fi/hotspot. Some campus Wi-Fi isolates clients; if so, use a phone hotspot.

## Setup (about 5 minutes, do it before the slot)
1. Laptop 1: install per README, set `PATHSNAP_DATA_ROOT`, start the API on `0.0.0.0:8000`.
2. Find Laptop 1's address: macOS `ipconfig getifaddr en0`; Windows `ipconfig` (IPv4); Linux `hostname -I`.
3. Allow inbound TCP 8000 in Laptop 1's firewall (Windows: allow Python on private networks; macOS: allow incoming for Python).
4. From Laptop 2, open `http://<laptop1-ip>:8000/health` in a browser. It must show `"status":"ok"`. If not, stop here: same network? firewall? wrong IP?
5. Laptop 2: `python -m http.server 5173 --directory ui` (inside a clone, no install needed), open `http://localhost:5173`.
6. Type `http://<laptop1-ip>:8000` in the API address box. The trip list must fill in. Pick a trip, Predict route.
7. Show the Laptop 1 console: one log line per request, plus the evaluation line.
Use `http://`, not `https://`. Browser origins outside localhost/private networks are blocked on purpose.

## Flow (about 3 minutes)
1. 00000052: smallest trip, 100% pointwise. Shows the full chain working.
2. 00000007: 89.1%. Route looks right but errors are mostly the reverse direction of the correct road (reverse twin arcs). Honest weak spot of the decoder.
3. 00000077: weakest trip, 86.2%, long drive with a loop.
4. Mention a bigger one (00000056, 3958 points, about 2 s) to show speed.
For the grading set use `PATHSNAP_PARTITION=evaluator_test` (folder `<DATA_ROOT>/evaluator_test/`); results are in `results/api/api_check_evaluator_test.json` (94.72% pooled over 19,674 observations).
Numbers above are from the real run in `results/api/api_check_student_test.json` (pooled 93.52% pointwise over 19,160 observations, 10 trips).

## Who explains what
- A: data and features. 50 m candidates, 10 features, why a point can have no candidate (such a point is never dropped: the frozen decoder fills it from the nearest decoded neighbour within 30 s, otherwise it counts wrong).
- B: model. Saved HistGradientBoosting scores each candidate; chosen by trip-grouped cross-validation.
- C: decoder. HMM/Viterbi picks a connected arc sequence; main weakness is reverse-direction arcs.
- D: API and UI. Prediction first, reference route read afterwards, only for accuracy.

## Say honestly
- Metric is per GPS point against the project's reference labels, not full-route sequence accuracy.
- Decoder tuned on training trips only; student-test and evaluator-test numbers were not used for tuning.
- Student-test 93.52% and evaluator-test 94.72% are two separate runs of the same frozen model; per-trip accuracy on evaluator ranges 88.5% to 99.7%.
- Predicted line is drawn from road segments; the road geometry is only the supplied nodes/arcs.

## If the network fails
- Run both on one laptop: API on 8000, UI on 5173, API address `http://localhost:8000`. State plainly that this is a one-machine fallback.
- Hotspot instead of shared Wi-Fi; or connect with a cable and use the Ethernet IP.
- Last resort: `python scripts/check_all_trips.py` in the console shows all ten trips through the API, with accuracy.
