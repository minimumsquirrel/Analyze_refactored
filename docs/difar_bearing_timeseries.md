# Processing WAV data as a time series with bearing (DIFAR)

Use `difar_core.py` when your WAV contains DIFAR channels such as:

- `OMNI` (pressure/reference)
- `X` (particle velocity)
- `Y` (particle velocity)
- optional `Z` (particle velocity)

## Core idea
For each frame of audio:
1. (Optional) band-pass filter channels.
2. Convert channel volts to physical units using calibration curves:
   - X/Y/Z with sensitivity in **dB re 1V/(m/s)**
   - OMNI with pressure sensitivity (commonly **dB re 1V/µPa**)
3. Compute sample-wise direction from X/Y:
   - `theta(t) = atan2(Y(t), X(t))`
4. Circular-average `theta(t)` in each frame to get sensor-frame bearing.
5. Rotate by digital-compass heading to get true-north bearing.
6. Compute confidence and intensity metrics.

## Output time series
- `time_s` (relative frame time)
- `timestamp_utc` (absolute UTC datetime per frame, if `start_time_utc` is set)
- `bearing_sensor_deg` (sensor-frame)
- `bearing_true_deg` (compass-referenced true bearing)
- `confidence`
- `snr_db`
- `intensity_motion_db_re_1_mps` (if X/Y are calibrated)
- `intensity_pressure_db_re_1_Pa` (if OMNI is calibrated)

## Import calibration CSV into database
Processed DIFAR outputs can also be persisted to SQLite table `difar_results` from the popup.

You can import calibration sets directly to SQLite using:
- `import_difar_calibration_csv_to_db(db_path, csv_path, calibration_name)`
- `load_difar_calibration_from_db(db_path, calibration_name)`

Expected CSV columns (either style):
- `frequency, x, y, z, omni, x_phase, y_phase, z_phase, omni_phase`
- `frequency, x, y, z, omni, x/y phase, z phase`

For your file style, `x/y phase` is applied to both X and Y channels,
`z phase` is applied to Z, and OMNI phase defaults to `0 deg` if not present.

`x/y/z` are particle-motion sensitivities (dB re 1V/m/s), `omni`
is pressure sensitivity, and phase columns are in degrees.

The phase responses are used during processing to apply first-order
phase correction (at the analysis band center) before bearing estimation.


## Channel mapping in the app
The DIFAR popup now lets you assign file channel numbers to `OMNI`, `X`, `Y`, and optional `Z`
so files that are not in a fixed 1/2/3/4 order can still be processed correctly.

- Mapping inputs are **1-based** in the UI (human-friendly).
- Internally they are converted to 0-based indices for processing.
- Set `Z` to `unused` when your file has no Z channel.

## Compass reference for true north
Provide compass heading so bearings are map-referenced:
- Heading is degrees clockwise from true north.
- You can provide one constant heading, or a time series of heading values.

## Start time: include date + time
For map/GPS correlation, set `start_time_utc` to the UTC datetime of sample 0,
including **date and time**.

## Static map display of bearing time series
A static map cannot animate time directly, so a practical approach is:
1. Plot the sensor location marker.
2. Draw decimated bearing vectors from sensor location.
3. Encode time by color gradient (early blue → late red) or by sampling labels (e.g., every 1 min).

Use `bearing_series_static_map_vectors(...)` to compute vector endpoints for
this display.

In the app popup, use **Calibration Import...** for calibration files, then enable
**Display DIFAR on Chart map (sensor + rays)** to push these rays onto the existing
Chart map overlay immediately after processing.

On Folium maps, click the DIFAR buoy marker to open a target-bearing popup
showing latest bearing and a compact bearing dial.

## Can calibration give distance?
Usually not from a single DIFAR buoy alone. With calibration, you can reliably
get:
- **bearing**
- **calibrated intensity level** (motion and/or pressure)

Range/distance generally needs additional assumptions or data such as source
level, propagation model/environment, or multi-sensor geometry/TDOA.


## Tuning when simulated/straight tracks look off

If bearings are consistently offset or unstable on straight tracks, use the DIFAR popup tuning controls:
- **Swap X/Y**, **Invert X**, **Invert Y** to match simulator/sensor axis conventions.
- **Bearing Offset (deg)** to remove constant bias.
- **Directional Gate Percentile** to ignore weak vector samples in each frame.
- **Bearing Smooth Frames** to stabilize track display for steady headings.


### Why you may see both left and right bearing lines
A common DIFAR artifact is a ±180° directional ambiguity, which can appear as
paired left/right lines on straight tracks.

The tool now includes **Resolve ±180° ambiguity** with optional
**OMNI + phase disambiguation**.

When enabled, the solver first uses an active-intensity proxy from
phase-corrected OMNI×X and OMNI×Y to pick between bearing and bearing+180,
then applies continuity against the previous frame as a fallback stabilizer.


## Project-scoped ray storage and map visibility
Processed DIFAR ray overlays are persisted in SQLite table `difar_map_rays`
(project-scoped via `project_id`, plus optional `run_id` reference).

In the Chart tab, **Show DIFAR Rays** controls whether saved DIFAR rays are rendered.
When enabled, the map loads saved rays for the active project and draws sensor marker(s)
plus time-coloured rays.
