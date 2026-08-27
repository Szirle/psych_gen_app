## Flutter ↔ Python API Contract

This document describes how the Flutter client communicates with the Python (Flask) backend in this project. Any changes to request/response shapes, endpoints, or serialization MUST update this document and the opposite side (frontend/backend) in the same change.

### Base URL
- Default in Flutter: relative (same-origin), e.g. `/images`, `/distributions`, `/charts`
- Override at build time: `--dart-define=API_BASE_URL=https://your-host` (or another absolute/relative prefix)

### Content Type
- JSON endpoints use `application/json` for requests and responses.
- `/charts` returns `text/html`.

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
      "range_start": 0.0,           // double; retained for client state, unused by /images
      "range_end": 1.0              // double; retained for client state, unused by /images
    },
    // ... 1–3 dimensions supported
  ],
  "truncation_psi": 0.6,            // double; sampling temperature
  "num_faces": 100,                 // int; dataset-only setting, ignored by preview generation
  "preserve_identity": false,       // bool; reserved, ignored by preview generation
  "change_face": false,             // bool; when true, backend resamples the base face latent
  "mode": "shape",                 // "shape" | "color" | "both"; maps to W-slice on backend
  "filters": {                      // optional; selects a stored base-face latent
    "dominant": [0.2, 0.8],
    "trustworthy": [0.1, 0.9]
  },
  "controlled_variables": [         // optional; reserved, ignored by preview generation
    "attractive"
  ],
  "max_steps": 40                   // optional; controls backend direction strength schedule
}
```

Notes
- Backend converts each dimension to a list of levels using `linspace(-strength, strength, n_levels)`.
- Backend uses `mode` to select which W layers are affected: `shape -> [0..9)`, `color -> [9..end)`, `both -> [0..end)`.
- Requests must contain 1–3 distinct dimensions. Each `n_levels` must be 2–5, `truncation_psi` must be 0.1–1.0, and `max_steps` must be 1–100.
- Flutter camelCase names are normalized to the dataset's dash-separated names, such as `wellGroomed` → `well-groomed`.
- If `change_face` is missing, the backend defaults it to true. When filters are active, it chooses a different eligible stored latent when more than one exists.
- Filters are inclusive ranges over average empirical ratings. The selected photo must have both ratings and a latent in `photo_to_coords.pkl`.
- With `change_face=false`, the current base is retained while it remains eligible. If the filters change and exclude it, another eligible base is selected.
- `num_faces`, `preserve_identity`, and `controlled_variables` remain reserved and are not applied to preview generation.

Response body
- Nested arrays of base64-encoded images (WEBP by default, PNG/JPG via query string), with depth equal to the number of manipulated dimensions.
- For 1D, the response is `[dimension0]`.
- For 2D, the response is in UI row-major order: `[dimension1][dimension0]`. The second variable selects the row and the first variable selects the column. This is important for unequal grids such as 3×4.
- For 3D, the response retains model order `[dimension0][dimension1][dimension2]`; Flutter converts it to depth-slice display order with dimension 0 changing fastest.

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
- For 1D, decode the list directly.
- For 2D, flatten the response rows directly; each row contains all first-variable/x-axis levels.
- For 3D, traverse dimensions from last to first while indexing the response in model order.

Response format/quality override
- Query params: `?format=png|jpg|webp&quality=90`; quality must be 1–100.

Errors
- Validation failures return HTTP 400 with `{ "error": "...", "details": ... }`.
- When filters match no stored latent, the endpoint returns HTTP 422 with the same error shape.
- Unexpected generation failures return a non-diagnostic HTTP 500 error to the client.

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
    "dominant": [0.0, 0.01, 0.07, ...]  // length == num_points
  }
}
```

Name mapping
- Names are matched to the actual dataset keys after converting camelCase to dash-separated lowercase.
- Response keys are converted back to Flutter camelCase.
- Unsupported requested variables or filter keys produce a 400 response rather than being silently omitted.
- If filters are absent, histograms use all ratings available for each requested variable. If filters match zero photos, every requested histogram contains only zeroes.

Errors
- `num_points` must be an integer from 2 through 1000.
- Validation failures return `{ "error": "...", "details": ... }` with HTTP 400.

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
1) Update this file: `docs/API_COMMUNICATION.md` with the exact new contract.
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
- `/images` ignores `num_faces`, `preserve_identity`, and `controlled_variables`.
- Dataset export is not implemented.
- The charts iframe is disabled in the active UI and `/charts` contains mocked prototype data.
- Base64 output format defaults to WEBP; adjust with query parameters.
- Large requests may be GPU/CPU intensive; server serializes access using a lock.
