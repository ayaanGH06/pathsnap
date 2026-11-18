# Task 1 findings: two-trip loading sanity check

## Trip 00000052
- Counts: 262 GPS points, 1770 nodes, 3577 arcs, 95 reference arcs (matches brief: True)
- Arc endpoints reference valid node ids: True
- Route arc ids valid: True
- Timestamps non-decreasing: True; step sizes in seconds: {'1': 257, '2': 4}
- Consecutive reference arcs connect: True (0 disconnections)
- Track lon/lat range: [35.858383, 35.896431] x [56.795125, 56.827472]

## Trip 00000007
- Counts: 1354 GPS points, 2061 nodes, 4369 arcs, 123 reference arcs (matches brief: True)
- Arc endpoints reference valid node ids: True
- Route arc ids valid: True
- Timestamps non-decreasing: True; step sizes in seconds: {'1': 1353}
- Consecutive reference arcs connect: True (0 disconnections)
- Track lon/lat range: [77.122052, 77.271115] x [43.388762, 43.496413]

## Notes for the team
- The two trips are in different parts of the world (see lon/lat ranges above),
  so one shared projected CRS such as EPSG:32643 cannot cover both. This script
  uses a per-trip local metre projection. Flag to A for the shared projection (A5).
- Timestamp step sizes above show any sampling gaps for A's data-quality notes.
