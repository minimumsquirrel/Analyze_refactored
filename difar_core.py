#!/usr/bin/env python3
"""Core DIFAR bearing time-series processing utilities.

This module provides reusable processing for turning multichannel WAV files
(OMNI + directional channels) into bearing-vs-time outputs and calibrated
intensity metrics suitable for GPS/map correlation workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Literal
import csv
import json
import re
from datetime import datetime, timezone, timedelta

import numpy as np
import soundfile as sf
from scipy.interpolate import interp1d
from scipy.signal import butter, sosfiltfilt, hilbert


CalibrationUnit = Literal["V_per_mps", "V_per_uPa", "V_per_Pa"]


@dataclass
class ChannelCalibration:
    """Frequency-dependent channel sensitivity.

    sensitivity_db is interpreted as 20*log10(V / physical_unit), where the
    physical unit is chosen by `unit`.
    """

    freq_hz: np.ndarray
    sensitivity_db: np.ndarray
    unit: CalibrationUnit
    phase_response_deg: Optional[np.ndarray] = None


@dataclass
class DifarCalibration:
    """Calibration bundle for DIFAR channels."""

    omni: Optional[ChannelCalibration] = None
    x: Optional[ChannelCalibration] = None
    y: Optional[ChannelCalibration] = None
    z: Optional[ChannelCalibration] = None


@dataclass
class CompassReference:
    """Compass data used to rotate sensor-frame bearings into true-north bearings.

    heading is interpreted as degrees clockwise from true north.
    - If `time_s` is provided, heading is interpolated by frame center time.
    - If `time_s` is None, heading_deg is treated as constant over file.
    """

    heading_deg: np.ndarray
    time_s: Optional[np.ndarray] = None


def _heading_at(compass: Optional[CompassReference], t_s: float, eps: float = 1e-12) -> float:
    """Return compass heading (deg true) at time t_s."""
    if compass is None:
        return 0.0
    h = np.asarray(compass.heading_deg, dtype=np.float64)
    if h.size == 0:
        return 0.0
    if compass.time_s is None:
        return float(h[0])
    tt = np.asarray(compass.time_s, dtype=np.float64)
    if tt.size != h.size:
        raise ValueError("Compass time_s and heading_deg must have the same length.")
    if tt.size == 1:
        return float(h[0])
    # unwrap heading for stable interpolation across 0/360
    h_unwrap = np.unwrap(np.deg2rad(h))
    h_interp = np.interp(float(t_s), tt, h_unwrap, left=h_unwrap[0], right=h_unwrap[-1])
    return float((np.rad2deg(h_interp) + 360.0) % 360.0)


@dataclass
class DifarConfig:
    """Configuration for DIFAR bearing extraction.

    Attributes:
        omni_channel: Channel index containing omni/pressure channel.
        x_channel: Channel index containing x directional channel.
        y_channel: Channel index containing y directional channel.
        z_channel: Optional channel index containing z directional channel.
        frame_seconds: Frame size (seconds) for averaging bearings.
        hop_seconds: Hop size (seconds) between successive frames.
        bandpass_hz: Optional (low, high) bandpass limits in Hz.
        filter_order: Butterworth filter order when bandpass_hz is set.
        eps: Small constant to avoid divide-by-zero and NaNs.
        start_time_utc: Optional UTC datetime for sample index 0.
            Must include calendar date for robust GPS/ship alignment.
        calibration: Optional DIFAR channel calibration bundle.
        compass: Optional compass reference to rotate sensor-frame bearing
            into true-north bearing.
        swap_xy: Swap X/Y channels before bearing estimation (sensor convention fix).
        invert_x: Multiply X by -1 before bearing estimation.
        invert_y: Multiply Y by -1 before bearing estimation.
        bearing_offset_deg: Additional fixed offset applied to sensor bearing (deg).
        min_directional_percentile: Optional percentile gate [0,100) to ignore
            low-magnitude directional samples inside each frame.
        bearing_smooth_frames: Optional moving-average smoothing window in frames
            (applied circularly to sensor bearing).
        resolve_180_ambiguity: If True, choose between bearing and bearing+180
            based on continuity with previous frame (helps avoid left/right flips).
    """

    omni_channel: int = 0
    x_channel: int = 1
    y_channel: int = 2
    z_channel: Optional[int] = None
    frame_seconds: float = 1.0
    hop_seconds: float = 0.25
    bandpass_hz: Optional[Tuple[float, float]] = (20.0, 500.0)
    filter_order: int = 4
    eps: float = 1e-12
    start_time_utc: Optional[datetime] = None
    calibration: Optional[DifarCalibration] = None
    compass: Optional[CompassReference] = None
    swap_xy: bool = False
    invert_x: bool = False
    invert_y: bool = False
    bearing_offset_deg: float = 0.0
    min_directional_percentile: float = 0.0
    bearing_smooth_frames: int = 1
    resolve_180_ambiguity: bool = True


def _normalize_start_time_utc(start_time_utc: Optional[datetime]) -> Optional[datetime]:
    """Normalize to timezone-aware UTC datetime (or None)."""
    if start_time_utc is None:
        return None
    if start_time_utc.year < 1970:
        raise ValueError("start_time_utc must include a valid calendar date (year >= 1970).")
    if start_time_utc.tzinfo is None:
        return start_time_utc.replace(tzinfo=timezone.utc)
    return start_time_utc.astimezone(timezone.utc)


def _bandpass(x: np.ndarray, fs: float, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    nyq = fs * 0.5
    low = max(low_hz / nyq, 1e-6)
    high = min(high_hz / nyq, 0.999999)
    if not (0.0 < low < high < 1.0):
        raise ValueError(f"Invalid bandpass bounds: low={low_hz}, high={high_hz}, fs={fs}")
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, x)


def _circular_mean_deg(theta_deg: np.ndarray, eps: float = 1e-12) -> Tuple[float, float]:
    """Return circular mean (deg, 0..360) and resultant length confidence [0,1]."""
    if theta_deg.size == 0:
        return float("nan"), 0.0
    theta_rad = np.deg2rad(theta_deg)
    z = np.exp(1j * theta_rad)
    zbar = np.mean(z)
    conf = float(np.abs(zbar))
    if conf < eps:
        return float("nan"), 0.0
    mean_rad = np.angle(zbar)
    mean_deg = (np.rad2deg(mean_rad) + 360.0) % 360.0
    return float(mean_deg), conf


def _weighted_circular_mean_deg(theta_deg: np.ndarray, w: np.ndarray, eps: float = 1e-12) -> Tuple[float, float]:
    """Weighted circular mean (deg, 0..360) and weighted resultant confidence [0,1]."""
    if theta_deg.size == 0:
        return float("nan"), 0.0
    ww = np.asarray(w, dtype=float)
    if ww.size != theta_deg.size:
        raise ValueError("weights must match theta size")
    ww = np.maximum(ww, 0.0)
    sw = float(np.sum(ww))
    if sw <= eps:
        return _circular_mean_deg(theta_deg, eps=eps)
    theta_rad = np.deg2rad(theta_deg)
    z = np.exp(1j * theta_rad)
    zbar = np.sum(z * ww) / sw
    conf = float(np.abs(zbar))
    if conf < eps:
        return float("nan"), 0.0
    mean_deg = (np.rad2deg(np.angle(zbar)) + 360.0) % 360.0
    return float(mean_deg), conf




def _ang_diff_deg(a: float, b: float) -> float:
    """Smallest absolute angular difference in degrees."""
    d = (float(a) - float(b) + 180.0) % 360.0 - 180.0
    return abs(d)


def _resolve_180_by_continuity(curr_deg: float, prev_deg: Optional[float]) -> float:
    """Pick curr or curr+180 based on proximity to previous bearing."""
    if prev_deg is None or not np.isfinite(prev_deg):
        return float(curr_deg % 360.0)
    a = float(curr_deg % 360.0)
    b = float((curr_deg + 180.0) % 360.0)
    return a if _ang_diff_deg(a, prev_deg) <= _ang_diff_deg(b, prev_deg) else b

def _smooth_circular_series_deg(theta_deg: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or theta_deg.size == 0:
        return theta_deg
    w = int(max(1, window))
    pad = w // 2
    ang = np.deg2rad(theta_deg)
    x = np.cos(ang)
    y = np.sin(ang)
    kernel = np.ones(w, dtype=float) / float(w)
    x2 = np.convolve(np.pad(x, (pad, pad), mode='edge'), kernel, mode='valid')
    y2 = np.convolve(np.pad(y, (pad, pad), mode='edge'), kernel, mode='valid')
    return (np.rad2deg(np.arctan2(y2, x2)) + 360.0) % 360.0


def _sensitivity_linear_at(cal: ChannelCalibration, f_hz: float) -> float:
    if cal.freq_hz.size == 0 or cal.sensitivity_db.size == 0:
        raise ValueError("Calibration arrays cannot be empty.")
    if cal.freq_hz.size != cal.sensitivity_db.size:
        raise ValueError("Calibration freq_hz and sensitivity_db must have same length.")
    interp = interp1d(cal.freq_hz, cal.sensitivity_db, kind="linear", fill_value="extrapolate")
    sens_db = float(interp(f_hz))
    return 10.0 ** (sens_db / 20.0)


def _phase_deg_at(cal: ChannelCalibration, f_hz: float) -> float:
    """Interpolate phase response at frequency; returns 0 when absent."""
    if cal.phase_response_deg is None:
        return 0.0
    ph = np.asarray(cal.phase_response_deg, dtype=float)
    if ph.size == 0:
        return 0.0
    if cal.freq_hz.size != ph.size:
        raise ValueError("Calibration freq_hz and phase_response_deg must have same length.")
    interp = interp1d(cal.freq_hz, ph, kind="linear", fill_value="extrapolate")
    return float(interp(f_hz))


def _apply_phase_correction(
    x_physical: np.ndarray,
    fs: float,
    bandpass_hz: Optional[Tuple[float, float]],
    cal: Optional[ChannelCalibration],
) -> np.ndarray:
    """Apply first-order phase correction using calibration phase at band center."""
    if cal is None:
        return x_physical
    if bandpass_hz is None:
        center_hz = fs * 0.25
    else:
        center_hz = 0.5 * (float(bandpass_hz[0]) + float(bandpass_hz[1]))
    ph_deg = _phase_deg_at(cal, center_hz)
    if abs(ph_deg) < 1e-12:
        return x_physical
    xa = hilbert(x_physical)
    corrected = xa * np.exp(-1j * np.deg2rad(ph_deg))
    return np.real(corrected)


def _calibrate_channel_volts_to_physical(
    x_volts: np.ndarray,
    fs: float,
    bandpass_hz: Optional[Tuple[float, float]],
    cal: Optional[ChannelCalibration],
) -> np.ndarray:
    """Convert channel from volts to physical units using calibration at a band center.

    For narrowband DIFAR workflows, we estimate sensitivity at the band center.
    """
    if cal is None:
        return x_volts

    if bandpass_hz is None:
        center_hz = fs * 0.25
    else:
        center_hz = 0.5 * (float(bandpass_hz[0]) + float(bandpass_hz[1]))

    sens_linear = _sensitivity_linear_at(cal, center_hz)  # V / (physical unit)
    if sens_linear <= 0:
        raise ValueError("Calibration sensitivity must be positive.")

    physical = x_volts / sens_linear
    if cal.unit == "V_per_uPa":
        return physical * 1e-6  # convert uPa -> Pa
    return physical


def load_difar_calibration_json(json_path: str) -> DifarCalibration:
    """Load DIFAR calibration curves from JSON.

    Expected JSON format:
    {
      "omni": {"unit": "V_per_uPa", "points": [[f_hz, sens_db], ...]},
      "x":    {"unit": "V_per_mps", "points": [[f_hz, sens_db], ...]},
      "y":    {"unit": "V_per_mps", "points": [[f_hz, sens_db], ...]},
      "z":    {"unit": "V_per_mps", "points": [[f_hz, sens_db], ...]}
    }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    def _parse_one(name: str) -> Optional[ChannelCalibration]:
        block = raw.get(name)
        if not block:
            return None
        points = block.get("points", [])
        if not points:
            return None
        arr = np.asarray(points, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"Invalid calibration points format for '{name}'.")
        phase = block.get("phase_deg")
        phase_arr = None if phase is None else np.asarray(phase, dtype=float)
        return ChannelCalibration(
            freq_hz=arr[:, 0],
            sensitivity_db=arr[:, 1],
            unit=block.get("unit", "V_per_mps"),
            phase_response_deg=phase_arr,
        )

    return DifarCalibration(
        omni=_parse_one("omni"),
        x=_parse_one("x"),
        y=_parse_one("y"),
        z=_parse_one("z"),
    )




def load_compass_csv(csv_path: str) -> CompassReference:
    """Load compass heading from CSV.

    CSV formats supported:
    - heading_deg
    - time_s,heading_deg
    """
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        fields = r.fieldnames or []
        if "heading_deg" not in fields:
            raise ValueError("Compass CSV must include 'heading_deg' column.")
        has_time = "time_s" in fields
        h = []
        t = []
        for row in r:
            h.append(float(row["heading_deg"]))
            if has_time:
                t.append(float(row["time_s"]))
    heading = np.asarray(h, dtype=np.float64)
    if has_time:
        return CompassReference(heading_deg=heading, time_s=np.asarray(t, dtype=np.float64))
    return CompassReference(heading_deg=heading)


def compute_bearing_time_series(data: np.ndarray, fs: float, cfg: DifarConfig | None = None) -> Dict[str, np.ndarray]:
    """Compute bearing-vs-time and confidence from OMNI/X/Y(/Z) channel data.

    Returns keys:
    - time_s
    - timestamp_utc (optional, if start_time_utc set)
    - bearing_sensor_deg
    - bearing_true_deg
    - confidence
    - snr_db
    - intensity_motion_db_re_1_mps (directional channels, if calibrated)
    - intensity_pressure_db_re_1_Pa (omni channel, if calibrated)
    """
    cfg = cfg or DifarConfig()

    x = np.asarray(data)
    if x.ndim == 1:
        raise ValueError("DIFAR bearing extraction requires at least 3 channels (OMNI, X, Y).")
    if x.ndim != 2:
        raise ValueError(f"Expected data with ndim=2, got {x.ndim}")

    n, ch = x.shape
    required_idx = [cfg.omni_channel, cfg.x_channel, cfg.y_channel]
    if cfg.z_channel is not None:
        required_idx.append(cfg.z_channel)
    needed = max(required_idx)
    if ch <= needed:
        raise ValueError(f"Data has {ch} channels but needs channel index {needed}. Check mapping.")

    omni = x[:, cfg.omni_channel].astype(np.float64, copy=False)
    x_ch = x[:, cfg.x_channel].astype(np.float64, copy=False)
    y_ch = x[:, cfg.y_channel].astype(np.float64, copy=False)
    z_ch = x[:, cfg.z_channel].astype(np.float64, copy=False) if cfg.z_channel is not None else None

    if cfg.bandpass_hz is not None:
        low, high = cfg.bandpass_hz
        omni = _bandpass(omni, fs, low, high, cfg.filter_order)
        x_ch = _bandpass(x_ch, fs, low, high, cfg.filter_order)
        y_ch = _bandpass(y_ch, fs, low, high, cfg.filter_order)
        if z_ch is not None:
            z_ch = _bandpass(z_ch, fs, low, high, cfg.filter_order)

    cal = cfg.calibration
    if cal is not None:
        omni = _calibrate_channel_volts_to_physical(omni, fs, cfg.bandpass_hz, cal.omni)
        x_ch = _calibrate_channel_volts_to_physical(x_ch, fs, cfg.bandpass_hz, cal.x)
        y_ch = _calibrate_channel_volts_to_physical(y_ch, fs, cfg.bandpass_hz, cal.y)
        if z_ch is not None:
            z_ch = _calibrate_channel_volts_to_physical(z_ch, fs, cfg.bandpass_hz, cal.z)

        # Apply phase correction using calibration phase response at band center
        omni = _apply_phase_correction(omni, fs, cfg.bandpass_hz, cal.omni)
        x_ch = _apply_phase_correction(x_ch, fs, cfg.bandpass_hz, cal.x)
        y_ch = _apply_phase_correction(y_ch, fs, cfg.bandpass_hz, cal.y)
        if z_ch is not None:
            z_ch = _apply_phase_correction(z_ch, fs, cfg.bandpass_hz, cal.z)

    frame_n = max(1, int(round(cfg.frame_seconds * fs)))
    hop_n = max(1, int(round(cfg.hop_seconds * fs)))

    t_out, b_sensor_out, b_true_out, c_out, s_out, ts_out = [], [], [], [], [], []
    motion_db_out, pressure_db_out = [], []
    start_utc = _normalize_start_time_utc(cfg.start_time_utc)

    for start in range(0, max(1, n - frame_n + 1), hop_n):
        stop = min(start + frame_n, n)
        if stop - start < 4:
            continue

        x_f = x_ch[start:stop]
        y_f = y_ch[start:stop]
        om_f = omni[start:stop]

        if cfg.swap_xy:
            x_f, y_f = y_f, x_f
        if cfg.invert_x:
            x_f = -x_f
        if cfg.invert_y:
            y_f = -y_f

        inst_theta = (np.rad2deg(np.arctan2(y_f, x_f)) + 360.0) % 360.0
        mag = np.sqrt(x_f * x_f + y_f * y_f)
        if float(cfg.min_directional_percentile) > 0.0:
            q = float(np.clip(cfg.min_directional_percentile, 0.0, 99.9))
            thr = float(np.percentile(mag, q))
            mask = (mag >= thr)
            if np.any(mask):
                inst_theta = inst_theta[mask]
                mag = mag[mask]

        bearing_sensor_deg, conf = _weighted_circular_mean_deg(inst_theta, mag, eps=cfg.eps)
        bearing_sensor_deg = (bearing_sensor_deg + float(cfg.bearing_offset_deg)) % 360.0
        if bool(cfg.resolve_180_ambiguity):
            prev_b = (b_sensor_out[-1] if len(b_sensor_out) > 0 else None)
            bearing_sensor_deg = _resolve_180_by_continuity(bearing_sensor_deg, prev_b)

        directional_power = float(np.mean(x_f * x_f + y_f * y_f))
        omni_power = float(np.mean(om_f * om_f))
        snr_db = 10.0 * np.log10((directional_power + cfg.eps) / (omni_power + cfg.eps))

        t_center = (start + 0.5 * (stop - start)) / fs

        t_out.append(t_center)
        b_sensor_out.append(bearing_sensor_deg)
        heading_deg = _heading_at(cfg.compass, t_center, eps=cfg.eps)
        b_true_out.append((bearing_sensor_deg + heading_deg) % 360.0)
        c_out.append(conf)
        s_out.append(snr_db)
        if start_utc is not None:
            ts_out.append((start_utc + timedelta(seconds=float(t_center))).isoformat())

        if cal is not None and (cal.x is not None or cal.y is not None):
            vrms_xy = float(np.sqrt(np.mean(x_f * x_f + y_f * y_f) / 2.0))
            motion_db_out.append(20.0 * np.log10(vrms_xy + cfg.eps))
        if cal is not None and cal.omni is not None:
            prms_pa = float(np.sqrt(np.mean(om_f * om_f)))
            pressure_db_out.append(20.0 * np.log10(prms_pa + cfg.eps))

    b_sensor_arr = np.asarray(b_sensor_out, dtype=np.float64)
    if int(cfg.bearing_smooth_frames) > 1:
        b_sensor_arr = _smooth_circular_series_deg(b_sensor_arr, int(cfg.bearing_smooth_frames))
        # recompute true-bearing with same heading sequence
        if len(b_true_out) == len(b_sensor_arr):
            b_true_out = [((float(b_sensor_arr[i]) + _heading_at(cfg.compass, float(t_out[i]), eps=cfg.eps)) % 360.0) for i in range(len(b_sensor_arr))]

    out: Dict[str, np.ndarray] = {
        "time_s": np.asarray(t_out, dtype=np.float64),
        "bearing_sensor_deg": b_sensor_arr,
        "bearing_true_deg": np.asarray(b_true_out, dtype=np.float64),
        "confidence": np.asarray(c_out, dtype=np.float64),
        "snr_db": np.asarray(s_out, dtype=np.float64),
    }
    if start_utc is not None:
        out["timestamp_utc"] = np.asarray(ts_out, dtype=object)
    if motion_db_out:
        out["intensity_motion_db_re_1_mps"] = np.asarray(motion_db_out, dtype=np.float64)
    if pressure_db_out:
        out["intensity_pressure_db_re_1_Pa"] = np.asarray(pressure_db_out, dtype=np.float64)
    return out


def process_wav_to_bearing_time_series(
    wav_path: str,
    cfg: DifarConfig | None = None,
    export_csv_path: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Load WAV and compute DIFAR bearing time series."""
    data, fs = sf.read(wav_path, always_2d=True)
    out = compute_bearing_time_series(data=data, fs=fs, cfg=cfg)

    if export_csv_path:
        with open(export_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            columns = ["time_s"]
            if "timestamp_utc" in out:
                columns.append("timestamp_utc")
            columns.extend(["bearing_sensor_deg", "bearing_true_deg", "confidence", "snr_db"])
            if "intensity_motion_db_re_1_mps" in out:
                columns.append("intensity_motion_db_re_1_mps")
            if "intensity_pressure_db_re_1_Pa" in out:
                columns.append("intensity_pressure_db_re_1_Pa")
            w.writerow(columns)

            row_count = len(out["time_s"])
            for i in range(row_count):
                row = []
                for col in columns:
                    val = out[col][i]
                    if isinstance(val, (np.floating, float)):
                        row.append(float(val))
                    else:
                        row.append(str(val))
                w.writerow(row)

    return out


def import_difar_calibration_csv_to_db(
    db_path: str,
    csv_path: str,
    calibration_name: str,
) -> int:
    """Import DIFAR calibration CSV to SQLite.

    Accepts either of these CSV shapes:
      1) frequency, x, y, z, omni, x_phase, y_phase, z_phase, omni_phase
      2) frequency, x, y, z, omni, x/y phase, z phase

    In shape (2), x/y phase is applied to both x and y. Missing omni phase is
    treated as 0 deg.

    Returns number of imported rows.
    """
    import sqlite3

    def _canon(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())

    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        raw_fields = r.fieldnames or []
        canon_to_raw = {_canon(c): c for c in raw_fields}

        # Required amplitude fields
        required = ["frequency", "x", "y", "z", "omni"]
        for req in required:
            if req not in canon_to_raw:
                raise ValueError(
                    "Calibration CSV must include frequency, x, y, z, omni columns."
                )

        # Phase aliases
        x_phase_key = (
            canon_to_raw.get("xphase")
            or canon_to_raw.get("xyphase")
            or canon_to_raw.get("xyphase")
        )
        y_phase_key = canon_to_raw.get("yphase")
        z_phase_key = canon_to_raw.get("zphase")
        omni_phase_key = canon_to_raw.get("omniphase")

        if x_phase_key is None and y_phase_key is None and z_phase_key is None and omni_phase_key is None:
            raise ValueError(
                "Calibration CSV must include phase columns. Supported: "
                "x_phase/y_phase/z_phase/omni_phase OR x/y phase + z phase."
            )

        for row in r:
            rr = {str(k).strip(): v for k, v in row.items()}
            try:
                freq = float(rr.get(canon_to_raw["frequency"]))
                xdb = float(rr.get(canon_to_raw["x"]))
                ydb = float(rr.get(canon_to_raw["y"]))
                zdb = float(rr.get(canon_to_raw["z"]))
                odb = float(rr.get(canon_to_raw["omni"]))

                # x/y shared phase support
                xph = float(rr.get(x_phase_key)) if x_phase_key is not None else 0.0
                yph = float(rr.get(y_phase_key)) if y_phase_key is not None else xph
                zph = float(rr.get(z_phase_key)) if z_phase_key is not None else 0.0
                oph = float(rr.get(omni_phase_key)) if omni_phase_key is not None else 0.0
            except Exception as e:
                raise ValueError(f"Invalid row in calibration CSV: {row}") from e

            rows.append((freq, xdb, ydb, zdb, odb, xph, yph, zph, oph))

    rows.sort(key=lambda t: t[0])
    if not rows:
        raise ValueError("Calibration CSV contains no data rows.")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS difar_calibration_sets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            calibration_name TEXT UNIQUE,
            created_utc TEXT,
            freq_json TEXT,
            x_json TEXT,
            y_json TEXT,
            z_json TEXT,
            omni_json TEXT,
            x_phase_json TEXT,
            y_phase_json TEXT,
            z_phase_json TEXT,
            omni_phase_json TEXT
        )
        """
    )

    # Ensure phase columns exist for older DBs
    cur.execute("PRAGMA table_info(difar_calibration_sets)")
    existing_cols = {r[1] for r in cur.fetchall()}
    for col in ("x_phase_json", "y_phase_json", "z_phase_json", "omni_phase_json"):
        if col not in existing_cols:
            cur.execute(f"ALTER TABLE difar_calibration_sets ADD COLUMN {col} TEXT")

    freq = [r[0] for r in rows]
    xvals = [r[1] for r in rows]
    yvals = [r[2] for r in rows]
    zvals = [r[3] for r in rows]
    ovals = [r[4] for r in rows]
    xphase = [r[5] for r in rows]
    yphase = [r[6] for r in rows]
    zphase = [r[7] for r in rows]
    ophase = [r[8] for r in rows]

    created = datetime.now(timezone.utc).isoformat()
    cur.execute(
        """
        INSERT OR REPLACE INTO difar_calibration_sets
        (calibration_name, created_utc, freq_json, x_json, y_json, z_json, omni_json, x_phase_json, y_phase_json, z_phase_json, omni_phase_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            calibration_name,
            created,
            json.dumps(freq),
            json.dumps(xvals),
            json.dumps(yvals),
            json.dumps(zvals),
            json.dumps(ovals),
            json.dumps(xphase),
            json.dumps(yphase),
            json.dumps(zphase),
            json.dumps(ophase),
        ),
    )
    conn.commit()
    conn.close()
    return len(rows)


def load_difar_calibration_from_db(db_path: str, calibration_name: str) -> DifarCalibration:
    """Load DIFAR calibration from SQLite into `DifarCalibration`."""
    import sqlite3

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(difar_calibration_sets)")
    cols = {r[1] for r in cur.fetchall()}
    has_xp = "x_phase_json" in cols
    has_yp = "y_phase_json" in cols
    has_zp = "z_phase_json" in cols
    has_op = "omni_phase_json" in cols

    cur.execute(
        f"""
        SELECT freq_json, x_json, y_json, z_json, omni_json,
               {('x_phase_json' if has_xp else 'NULL')} as x_phase_json,
               {('y_phase_json' if has_yp else 'NULL')} as y_phase_json,
               {('z_phase_json' if has_zp else 'NULL')} as z_phase_json,
               {('omni_phase_json' if has_op else 'NULL')} as omni_phase_json
        FROM difar_calibration_sets
        WHERE calibration_name=?
        """,
        (calibration_name,),
    )
    row = cur.fetchone()
    conn.close()
    if row is None:
        raise ValueError(f"Calibration not found: {calibration_name}")

    freq = np.asarray(json.loads(row[0]), dtype=float)
    x = np.asarray(json.loads(row[1]), dtype=float)
    y = np.asarray(json.loads(row[2]), dtype=float)
    z = np.asarray(json.loads(row[3]), dtype=float)
    omni = np.asarray(json.loads(row[4]), dtype=float)
    xph = np.asarray(json.loads(row[5]), dtype=float) if row[5] else None
    yph = np.asarray(json.loads(row[6]), dtype=float) if row[6] else None
    zph = np.asarray(json.loads(row[7]), dtype=float) if row[7] else None
    oph = np.asarray(json.loads(row[8]), dtype=float) if row[8] else None

    return DifarCalibration(
        x=ChannelCalibration(freq_hz=freq, sensitivity_db=x, unit="V_per_mps", phase_response_deg=xph),
        y=ChannelCalibration(freq_hz=freq, sensitivity_db=y, unit="V_per_mps", phase_response_deg=yph),
        z=ChannelCalibration(freq_hz=freq, sensitivity_db=z, unit="V_per_mps", phase_response_deg=zph),
        omni=ChannelCalibration(freq_hz=freq, sensitivity_db=omni, unit="V_per_uPa", phase_response_deg=oph),
    )


def bearing_series_static_map_vectors(
    sensor_lat: float,
    sensor_lon: float,
    bearing_true_deg: np.ndarray,
    time_s: np.ndarray,
    scale_m: float = 1000.0,
    every_n: int = 20,
) -> Dict[str, np.ndarray]:
    """Build static-map vector endpoints for bearing-vs-time display.

    A static map cannot animate time; this returns decimated vectors where each
    vector represents bearing at a sampled timestamp. Render with color-by-time
    or labels to convey temporal order.
    """
    b = np.asarray(bearing_true_deg, dtype=float)
    t = np.asarray(time_s, dtype=float)
    if b.size != t.size:
        raise ValueError("bearing_true_deg and time_s must have same length")
    if b.size == 0:
        return {"lat2": np.asarray([]), "lon2": np.asarray([]), "time_s": np.asarray([]), "bearing_true_deg": np.asarray([])}

    idx = np.arange(0, b.size, max(1, int(every_n)))
    br = np.deg2rad(b[idx])

    # local tangent-plane approximation
    d_north = scale_m * np.cos(br)
    d_east = scale_m * np.sin(br)

    m_per_deg_lat = 111_320.0
    m_per_deg_lon = max(1e-6, 111_320.0 * np.cos(np.deg2rad(sensor_lat)))

    lat2 = sensor_lat + d_north / m_per_deg_lat
    lon2 = sensor_lon + d_east / m_per_deg_lon

    return {
        "lat2": lat2,
        "lon2": lon2,
        "time_s": t[idx],
        "bearing_true_deg": b[idx],
    }
