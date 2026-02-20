## Flutter ↔ Python API Contract

This document describes how the Flutter client communicates with the Python (Flask) backend in this project. Any changes to request/response shapes, endpoints, or serialization MUST update this document and the opposite side (frontend/backend) in the same change.

### Base URL
- Default in Flutter: relative (same-origin), e.g. `/images`, `/distributions`, `/charts`
- Override at build time: `--dart-define=API_BASE_URL=https://your-host` (or another absolute/relative prefix)

### Content Type
- Requests: `application/json`
- Responses: `application/json`

---

## Endpoints

### 1) POST /images
Generates a grid of manipulated face images.

Flutter call site: `lib/features/face_generation/data/datasources/face_manipulation_api_datasource.dart`
Backend handler: `app.py@app.route('/images', methods=['POST'])`

Request body
```
{
  "manipulated_dimensions": [
    {
      "name": "dominant",          // string; must match Flutter enum names (see below)
      "strength": 25.0,             // double; max absolute strength for this dimension
      "n_levels": 5,                // int; number of levels along this dimension
      "range_start": 0.0,           // double; optional, range filter hint (currently unused by /images)
      "range_end": 1.0              // double; optional, range filter hint (currently unused by /images)
    },
    // ... 1–3 dimensions supported
  ],
  "truncation_psi": 0.6,            // double; sampling temperature
  "num_faces": 100,                 // int; currently ignored by backend (/images sets num_faces=1 internally)
  "preserve_identity": false,       // bool; currently not used by backend synthesis logic
  "change_face": false,             // bool; when true, backend resamples the base face latent
  "mode": "shape",                 // "shape" | "color" | "both"; maps to W-slice on backend
  "filters": {                      // optional; currently ignored by /images (used by /distributions)
    "dominant": [0.2, 0.8],
    "trustworthy": [0.1, 0.9]
  },
  "controlled_variables": [         // optional; list of dimension names; currently unused by backend
    "attractive"
  ],
  "max_steps": 40                   // optional; controls backend direction strength schedule
}
```

Notes
- Backend converts each dimension to a list of levels using `linspace(-strength, strength, n_levels)`.
- Backend uses `mode` to select which W layers are affected: `shape -> [0..9)`, `color -> [9..end)`, `both -> [0..end)`.
- If `change_face` is missing, backend defaults it to true.
- `filters` and `controlled_variables` are accepted by Flutter but are currently not applied in `/images` generation.
- `num_faces` is currently overridden to 1 on the backend.

Response body
- Nested arrays of base64-encoded images (WEBP by default, PNG/JPG via query string), with depth equal to the number of manipulated dimensions.

Examples
- 1D (K=1):
```
[
  "<b64>", "<b64>", ...
]
```
- 2D (K=2):
```
[
  ["<b64>", "<b64>", ...],
  ["<b64>", "<b64>", ...],
  ...
]
```
- 3D (K=3):
```
[
  [ ["<b64>", ...], ["<b64>", ...], ... ],
  [ ["<b64>", ...], ["<b64>", ...], ... ],
  ...
]
```

Client decoding behavior
- For K=1: iterate list and decode each base64 string.
- For K=2: iterate rows and columns to flatten.
- For K=3: iterate depth slices, rows, then columns to flatten in the order used in the app.

Response format/quality override
- Query params: `?format=png|jpg|webp&quality=90`

Errors
- Non-200 results should be treated as failures on the client. The backend prints diagnostics and may return 400 for malformed requests.

---

### 2) POST /distributions
Returns normalized histograms over validation ratings for requested variables and optional filters.

Flutter call site: `lib/features/face_generation/data/datasources/distributions_api_datasource.dart`
Backend handler: `app.py@app.route('/distributions', methods=['POST'])`

Request body
```
{
  "filters": {                  // optional; map dim -> [lo, hi] in [0,1]
    "dominant": [0.2, 0.8]
  },
  "num_points": 100,            // optional; default 100
  "variables": ["dominant"]    // optional; if omitted, backend returns all available
}
```

Response body
```
{
  "distributions": {
    "dominance": [0.0, 0.01, 0.07, ...]  // length == num_points
  }
}
```

Name mapping
- Backend is tolerant to small naming differences and will map some common variants:
  - "trustworthy" -> "trustworthiness"
  - "dominant" -> "dominance"
  - appends "ness" or converts "ant"->"ance" and "y"->"iness" when possible

Errors
- On failure, backend returns `{ "error": "..." }` with HTTP 400.

---

### 3) GET /charts
Serves a self-contained HTML page with a responsive 2x10 grid of mocked Plotly charts, intended to be embedded via an iframe in the Flutter web app for prototyping/visualization.

Flutter call site: `lib/features/face_generation/presentation/widgets/plotly_iframe_panel.dart`
Backend handler: `app.py@app.route('/charts')`

Request
```
GET /charts[?ts=1699999999]
```

Query params
- `ts`: optional cache-buster (integer/string). When present, the client sets the iframe `src` to `/charts?ts=<value>` to force reloads.

Response
- Content-Type: `text/html`
- Body: standalone HTML that loads Plotly from CDN and renders 20 small line charts in a CSS grid. The page listens to window resize events and calls `Plotly.Plots.resize` for responsiveness.

Notes
- This endpoint is not part of the JSON API; it returns HTML, not JSON.
- Used only by the web client embedding an iframe; native/mobile builds do not use it.
- The charts are mocked/demo visuals and do not reflect backend data.

Errors
- On failure, the server will return a standard HTTP error response (e.g., 5xx). The Flutter client should handle iframe load failures gracefully (e.g., by hiding the panel or showing a placeholder).

---

## Data Models (Flutter)

`FaceManipulationRequest` → JSON
```
{
  "manipulated_dimensions": [ManipulatedDimension.toJson()],
  "truncation_psi": double,
  "num_faces": int,
  "preserve_identity": bool,
  "change_face": bool,
  "mode": "shape"|"color"|"both",
  "filters": { name: [double, double] }?,
  "controlled_variables": [name]? 
}
```

`ManipulatedDimension` → JSON
```
{
  "name": string,        // from Flutter enum ManipulatedDimensionName
  "strength": double,
  "n_levels": int,
  "range_start": double,
  "range_end": double
}
```

`ManipulatedDimensionName` (Flutter enum; key subset)
```
trustworthy, attractive, dominant, smart, age, gender, weight, typical, happy,
familiar, outgoing, memorable, wellGroomed, longHaired, smug, dorky, skinColor,
hairColor, alert, cute, privileged, liberal, asian, middleEastern, hispanic,
islander, native, black, white, looksLikeYou, gay, electable, godly, outdoors
```

---

## Cross‑Side Change Protocol

When changing the API request/response or endpoint behavior on either side:
1) Update this file: `docs/flutter_python_api.md` with the exact new contract.
2) Update the other side to match:
   - If editing Flutter (datasources/entities), update Python Flask handlers to parse/produce the new fields.
   - If editing Python (Flask routes/shape), update Flutter datasources/entities and client decoding.
3) Build and verify:
   - `flutter build web --release`
   - Manually test `/images` and `/distributions` with representative payloads.

Affected code paths
- Frontend:
  - `lib/features/face_generation/data/datasources/face_manipulation_api_datasource.dart`
  - `lib/features/face_generation/data/datasources/distributions_api_datasource.dart`
  - `lib/features/face_generation/domain/entities/**`
- Backend:
  - `app.py` (Flask routes `/images`, `/distributions` and `parse_config`)
  - `gan_backend.py` (generation internals)

---

## Known Current Limitations
- `/images` currently ignores `num_faces`, `filters`, and `controlled_variables`.
- Base64 output format defaults to WEBP; adjust with query parameters.
- Large requests may be GPU/CPU intensive; server serializes access using a lock.


