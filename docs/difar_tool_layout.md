# DIFAR Tool Layout Proposal

This document proposes a maintainable way to add a DIFAR-processing tool to the refactored Analyze app.

## 1) Where the DIFAR tool should live

### UI category
Put DIFAR under **Detection & Classification Tools**.

Reasoning:
- DIFAR is primarily about bearing extraction, direction finding, and event-level interpretation.
- It matches existing detection workflows (`Active Sonar`, `Cepstrum Analysis`, `Event Clustering`) rather than waveform file utilities.

### Code module
Create a new mixin module:

- `tools_difar.py`
- class: `DifarToolsMixin`

Then wire it into `MainWindow` alongside the other tool mixins.

## 2) Suggested internal structure for `tools_difar.py`

Keep one public popup entrypoint and split heavy logic into private helpers.

```python
class DifarToolsMixin:
    def difar_processing_popup(self):
        """Main dialog: input selection, parameters, run/export actions."""

    # --- signal preparation ---
    def _difar_get_selected_signal(self): ...
    def _difar_bandlimit(self, x, fs, low_hz, high_hz): ...

    # --- DIFAR core estimation ---
    def _difar_compute_bearing_series(self, x, fs, params): ...
    def _difar_confidence(self, bearing_series, snr_series): ...

    # --- post-processing ---
    def _difar_smooth_track(self, bearing_series, method="median"): ...
    def _difar_detect_stable_segments(self, bearing_series, conf_series): ...

    # --- outputs ---
    def _difar_plot_results(self, t, bearing, conf): ...
    def _difar_export_csv(self, output_path, rows): ...
```

Design principles:
- Keep UI orchestration in one method (`difar_processing_popup`).
- Keep numerics pure-ish (accept arrays + params, return arrays/records).
- Reuse shared utility functions where possible (filtering, FFT helpers).

## 3) Add a lightweight analysis core module (optional but recommended)

If DIFAR math grows beyond ~150 lines, move algorithmic pieces into:

- `difar_core.py`

Example contents:
- `DifarConfig` dataclass for algorithm parameters.
- `compute_bearing_series(signal, fs, config)`
- `compute_confidence(...)`
- `segment_tracks(...)`

Then the mixin becomes UI + I/O glue only.

## 4) UI flow in the popup

Use a 3-panel workflow:

1. **Input panel**
   - Channel selector with explicit channel-number mapping (OMNI/X/Y/Z)
   - Time window (start/end)
   - WAV start datetime (UTC, date + time) for GPS/ship-track alignment
   - Calibration CSV import (supports `x/y phase` + `z phase` file style) to DB
   - Compass import (constant heading or heading vs time)
   - Preset picker (e.g., Low-Frequency / Broadband)

2. **Processing panel**
   - Band-pass range
   - Frame length / overlap
   - Smoothing window
   - Confidence threshold

3. **Results panel**
   - Bearing vs time plot
   - Confidence vs time plot
   - Table of stable segments (start, end, bearing, confidence)
   - Buttons: `Export CSV`, `Save Figure`

This keeps DIFAR usable for quick-look analysis and export-ready workflows.

## 5) Integration points in `main_app_refactored.py`

Add in three places:

1. **Import**
   - `from tools_difar import DifarToolsMixin`

2. **MainWindow inheritance**
   - Add `DifarToolsMixin` into the `MainWindow(...)` mixin list.

3. **Tool dropdown wiring**
   - In `update_tool_list()`: add `"DIFAR Processing"` under `"Detection & Classification Tools"`.
   - In `on_tool_selected()`: map it to `self.difar_processing_popup()`.

## 6) Data/logging/output conventions

Follow existing app conventions:
- Save generated artifacts into project subfolders (`analysis/` style flow).
- Save each analyzed DIFAR run (metadata + time-series JSON) to database table `difar_results`.
- Log summary metrics to database with method name `DIFAR Processing`.
- Store minimal reproducibility metadata with exports:
  - wav start datetime (UTC)
  - sample rate
  - selected channel mapping (omni/x/y/z)
  - compass heading source (fixed or time-varying)
  - calibration file/version used
  - processing window
  - filter bounds
  - smoothing and thresholds

## 7) MVP implementation checklist

1. Create `tools_difar.py` and `DifarToolsMixin` skeleton.
2. Add popup with parameter form + run button.
3. Implement bearing series calculation (single method first).
4. Plot bearing/confidence and enable CSV export.
5. Wire into category/tool dropdown.
6. Add DB logging entry for each run.
7. Validate on known DIFAR sample files.

## 8) Future-proofing

After MVP, consider:
- Batch DIFAR processing for folders/playlists.
- Bearing track clustering / target separation.
- Optional map overlay when GPS track is loaded.
- Preset library per platform/sensor.

---

In short: add DIFAR as a dedicated mixin under Detection tools, keep popup/UI in the mixin, keep math in helper functions (or `difar_core.py` if it grows), and wire it through the same dropdown + logging patterns used by existing tools.


## 9) Static map bearing display

For static maps (non-animated), show bearing-vs-time by:
- Sensor marker at buoy location
- Decimated bearing rays from marker
- Color gradient or periodic labels to encode time progression

This keeps one map frame readable while still conveying time sequence.


## Current implementation note

The DIFAR popup should expose both:
- Calibration import controls, and
- Processing-run controls (WAV input, calibration selection, start UTC, optional compass CSV, optional output CSV, optional sensor lat/lon for static rays, optional "Show DIFAR rays on Chart tab").
