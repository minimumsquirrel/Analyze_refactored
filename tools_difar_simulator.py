#!/usr/bin/env python3
"""DIFAR synthetic scenario simulator tool window."""

import json
import math
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
from scipy.io.wavfile import write as wav_write

from shared import DB_FILENAME

DARK_QSS = """
* { font-family: Segoe UI; font-size: 10pt; }
QWidget { background-color: #19232D; color: #E6E6E6; }
QGroupBox { border: 1px solid #32414B; border-radius: 8px; margin-top: 10px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #E6E6E6; }
QLabel { color: #E6E6E6; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
  background-color: #22303C; border: 1px solid #32414B; border-radius: 6px; padding: 4px;
  selection-background-color: #3B82F6;
}
QPushButton {
  background-color: #2B3B49; border: 1px solid #32414B; border-radius: 8px; padding: 6px 10px;
}
QPushButton:hover { background-color: #34495A; }
QPushButton:pressed { background-color: #22303C; }
QCheckBox { spacing: 8px; }
QPlainTextEdit {
  background-color: #22303C; border: 1px solid #32414B; border-radius: 6px; color: #E6E6E6;
}
QScrollArea { border: none; }
"""

EARTH_RADIUS_M = 6371000.0
KTS_TO_MPS = 0.514444


def meters_to_latlon_offsets(lat_deg: float, east_m: float, north_m: float) -> Tuple[float, float]:
    lat = math.radians(lat_deg)
    dlat = (north_m / EARTH_RADIUS_M) * (180.0 / math.pi)
    dlon = (east_m / (EARTH_RADIUS_M * math.cos(lat))) * (180.0 / math.pi)
    return dlat, dlon


def bearing_range_to_enu(bearing_deg: float, range_m: float) -> Tuple[float, float]:
    br = math.radians(bearing_deg)
    north = range_m * math.cos(br)
    east = range_m * math.sin(br)
    return east, north


def decimal_degrees_to_nmea_lat(lat_deg: float) -> Tuple[str, str]:
    hemi = "N" if lat_deg >= 0 else "S"
    lat_abs = abs(lat_deg)
    deg = int(lat_abs)
    minutes = (lat_abs - deg) * 60.0
    return f"{deg:02d}{minutes:07.4f}", hemi


def decimal_degrees_to_nmea_lon(lon_deg: float) -> Tuple[str, str]:
    hemi = "E" if lon_deg >= 0 else "W"
    lon_abs = abs(lon_deg)
    deg = int(lon_abs)
    minutes = (lon_abs - deg) * 60.0
    return f"{deg:03d}{minutes:07.4f}", hemi


def nmea_checksum(sentence_wo_dollar: str) -> str:
    csum = 0
    for ch in sentence_wo_dollar:
        csum ^= ord(ch)
    return f"{csum:02X}"


def make_gpgga(timestamp_s: float, lat_deg: float, lon_deg: float, fix_quality: int = 1, num_sats: int = 8, hdop: float = 0.9,
               altitude_m: float = 0.0, geoid_sep_m: float = 0.0) -> str:
    utc = time.gmtime(timestamp_s)
    frac = timestamp_s - int(timestamp_s)
    msec = min(999, int(round(frac * 1000.0)))
    hhmmss = f"{utc.tm_hour:02d}{utc.tm_min:02d}{utc.tm_sec:02d}.{msec:03d}"
    lat_str, lat_hemi = decimal_degrees_to_nmea_lat(lat_deg)
    lon_str, lon_hemi = decimal_degrees_to_nmea_lon(lon_deg)
    body = (
        f"GPGGA,{hhmmss},{lat_str},{lat_hemi},{lon_str},{lon_hemi},"
        f"{fix_quality},{num_sats:02d},{hdop:.1f},{altitude_m:.1f},M,{geoid_sep_m:.1f},M,,"
    )
    return f"${body}*{nmea_checksum(body)}"


@dataclass
class CalData:
    freq_hz: np.ndarray
    omni_db_v_per_upa: np.ndarray
    x_db_v_per_ms: np.ndarray
    y_db_v_per_ms: np.ndarray
    z_db_v_per_ms: np.ndarray
    xy_phase_deg: np.ndarray
    z_phase_deg: np.ndarray


@dataclass
class ScenarioResult:
    wav_path: str
    gga_path: str
    debug_csv_path: str
    meta_path: str
    east_m: np.ndarray
    north_m: np.ndarray
    time_s: np.ndarray
    range_m: np.ndarray
    bearing_deg: np.ndarray


def load_m20105_cal_csv(path: str) -> CalData:
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    required = ["frequency", "x", "y", "z", "omni", "x/y phase", "z phase"]
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"Calibration file is missing required columns: {missing}")
    return CalData(
        freq_hz=df[cols["frequency"]].to_numpy(dtype=float),
        x_db_v_per_ms=df[cols["x"]].to_numpy(dtype=float),
        y_db_v_per_ms=df[cols["y"]].to_numpy(dtype=float),
        z_db_v_per_ms=df[cols["z"]].to_numpy(dtype=float),
        omni_db_v_per_upa=df[cols["omni"]].to_numpy(dtype=float),
        xy_phase_deg=df[cols["x/y phase"]].to_numpy(dtype=float),
        z_phase_deg=df[cols["z phase"]].to_numpy(dtype=float),
    )


def interp_1d(x: np.ndarray, y: np.ndarray, xq: float) -> float:
    return float(np.interp(xq, x, y))


def db_v_per_upa_to_v_per_pa(db_v_per_upa: float) -> float:
    return (10.0 ** (db_v_per_upa / 20.0)) * 1e6


def db_v_per_ms_to_v_per_ms(db_v_per_ms: float) -> float:
    return 10.0 ** (db_v_per_ms / 20.0)


def absorption_db_per_km_thorp(f_khz: float) -> float:
    f2 = f_khz * f_khz
    return (0.11 * f2 / (1 + f2)) + (44 * f2 / (4100 + f2)) + (2.75e-4 * f2) + 0.003


def transmission_loss_db(range_m: float, freq_hz: float, use_absorption: bool) -> float:
    r = max(range_m, 1.0)
    tl = 20.0 * math.log10(r)
    if use_absorption:
        tl += absorption_db_per_km_thorp(freq_hz / 1000.0) * (r / 1000.0)
    return tl


def rl_from_sl(sl_db_re_1upa_1m: float, range_m: float, freq_hz: float, use_absorption: bool) -> float:
    return sl_db_re_1upa_1m - transmission_loss_db(range_m, freq_hz, use_absorption)


def pressure_pa_from_rl_db(rl_db_re_1upa: float) -> float:
    upa = 10.0 ** (rl_db_re_1upa / 20.0)
    return upa * 1e-6


def colored_noise(n: int, fs: int, alpha: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    shape = np.ones_like(freqs)
    shape[1:] = 1.0 / (freqs[1:] ** (alpha / 2.0))
    X *= shape
    y = np.fft.irfft(X, n=n)
    y /= np.max(np.abs(y)) + 1e-12
    return y


def apply_wave_amplitude_modulation(sig: np.ndarray, fs: int, strength: float, seed: int) -> np.ndarray:
    mod = colored_noise(sig.size, fs, alpha=2.0, seed=seed)
    win = max(1, int(fs * 0.5))
    if win > 1:
        mod = np.convolve(mod, np.ones(win, dtype=float) / float(win), mode="same")
    mod /= np.max(np.abs(mod)) + 1e-12
    return sig * (1.0 + strength * mod)


def apply_multipath(sig: np.ndarray, fs: int, delay_s: float, atten_db: float) -> np.ndarray:
    delay_samples = int(round(delay_s * fs))
    if delay_samples <= 0 or delay_samples >= sig.size:
        return sig
    out = sig.copy()
    out[delay_samples:] += (10.0 ** (-atten_db / 20.0)) * sig[:-delay_samples]
    return out


def vessel_cavitation_source(n: int, fs: int, seed: int, low_hz: float = 30.0, high_hz: float = 6000.0,
                             alpha: float = 1.2) -> np.ndarray:
    x = colored_noise(n, fs, alpha=alpha, seed=seed)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mask = (freqs >= low_hz) & (freqs <= min(high_hz, fs / 2.0 - 1.0))
    X[~mask] = 0.0
    y = np.fft.irfft(X, n=n)
    y /= np.max(np.abs(y)) + 1e-12
    return y


def base_signal(t: np.ndarray, sig_type: str, f0_hz: float, bw_hz: float, fs: int, seed: int) -> np.ndarray:
    if sig_type == "tone":
        return np.sin(2.0 * np.pi * f0_hz * t)
    if sig_type == "am_tone":
        return (0.5 * (1.0 + np.sin(2.0 * np.pi * 0.3 * t))) * np.sin(2.0 * np.pi * f0_hz * t)
    if sig_type == "band_noise":
        x = colored_noise(t.size, fs, alpha=0.0, seed=seed)
        X = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(t.size, d=1.0 / fs)
        lo = max(1.0, f0_hz - bw_hz / 2.0)
        hi = f0_hz + bw_hz / 2.0
        X[~((freqs >= lo) & (freqs <= hi))] = 0.0
        y = np.fft.irfft(X, n=t.size)
        y /= np.max(np.abs(y)) + 1e-12
        return y
    if sig_type == "vessel_noise":
        return vessel_cavitation_source(t.size, fs, seed=seed)
    raise ValueError(f"Unsupported signal type: {sig_type}")


def apply_doppler_tone_envelope(t: np.ndarray, f0_hz: float, sound_speed_ms: float, vr_mps: np.ndarray,
                                env_sig: np.ndarray) -> np.ndarray:
    factor = sound_speed_ms / np.maximum(1e-6, sound_speed_ms - vr_mps)
    inst_freq = f0_hz * factor
    phase = 2.0 * np.pi * np.cumsum(inst_freq) * (t[1] - t[0])
    carrier = np.sin(phase)
    env = np.abs(env_sig)
    env /= np.max(env) + 1e-12
    out = carrier * np.maximum(env, 0.05)
    out /= np.max(np.abs(out)) + 1e-12
    return out


def narrowband_phase_shifted_carrier(t: np.ndarray, f0_hz: float, phase_rad: float) -> np.ndarray:
    return np.sin(2.0 * np.pi * f0_hz * t + phase_rad)


def pattern_circle(t: np.ndarray, period_s: float, range_m: float, bearing0_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    return (bearing0_deg + 360.0 * (t / period_s)) % 360.0, np.full_like(t, float(range_m))


def pattern_backforth(t: np.ndarray, swing_deg: float, rate_hz: float, range_m: float,
                      bearing_center_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    return (bearing_center_deg + swing_deg * np.sin(2.0 * np.pi * rate_hz * t)) % 360.0, np.full_like(t, float(range_m))


def pattern_straight_pass(t: np.ndarray, closest_approach_m: float, speed_mps: float,
                          bearing0_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    t_mid = 0.5 * (t[0] + t[-1])
    along = (t - t_mid) * speed_mps
    cross = np.full_like(t, closest_approach_m)
    rng = np.sqrt(along * along + cross * cross)
    bearing = (np.degrees(np.arctan2(along, cross)) + bearing0_deg) % 360.0
    return bearing, rng


def pattern_racetrack(t: np.ndarray, long_m: float, short_m: float, speed_mps: float, bearing0_deg: float,
                      center_east_m: float = 0.0, center_north_m: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    perimeter = 2.0 * (long_m + short_m)
    s = np.mod(speed_mps * (t - t[0]), perimeter)
    e = np.zeros_like(t)
    n = np.zeros_like(t)
    for i, si in enumerate(s):
        if si < long_m:
            e[i], n[i] = -long_m / 2.0 + si, short_m / 2.0
        elif si < long_m + short_m:
            e[i], n[i] = long_m / 2.0, short_m / 2.0 - (si - long_m)
        elif si < 2 * long_m + short_m:
            e[i], n[i] = long_m / 2.0 - (si - (long_m + short_m)), -short_m / 2.0
        else:
            e[i], n[i] = -long_m / 2.0, -short_m / 2.0 + (si - (2 * long_m + short_m))
    ang = math.radians(bearing0_deg)
    er = e * math.cos(ang) - n * math.sin(ang) + center_east_m
    nr = e * math.sin(ang) + n * math.cos(ang) + center_north_m
    rng = np.sqrt(er * er + nr * nr)
    bearing = (np.degrees(np.arctan2(er, nr)) + 360.0) % 360.0
    return bearing, rng


def pattern_spiral(t: np.ndarray, r_start_m: float, r_end_m: float, revolutions: float,
                   bearing0_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    frac = (t - t[0]) / max(1e-9, (t[-1] - t[0]))
    rng = r_start_m + (r_end_m - r_start_m) * frac
    bearing = (bearing0_deg + 360.0 * revolutions * frac) % 360.0
    return bearing, rng


def choose_motion(t: np.ndarray, pattern_name: str, bearing0_deg: float, range_m: float, period_s: float, swing_deg: float,
                  rate_hz: float, speed_mps: float, racetrack_long_m: float, racetrack_short_m: float,
                  spiral_r_start_m: float, spiral_r_end_m: float, spiral_revs: float,
                  racetrack_center_east_m: float = 0.0, racetrack_center_north_m: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    if pattern_name == "circle":
        return pattern_circle(t, period_s, range_m, bearing0_deg)
    if pattern_name == "back_forth":
        return pattern_backforth(t, swing_deg, rate_hz, range_m, bearing0_deg)
    if pattern_name == "straight_pass":
        return pattern_straight_pass(t, range_m, speed_mps, bearing0_deg)
    if pattern_name == "racetrack":
        return pattern_racetrack(t, racetrack_long_m, racetrack_short_m, speed_mps, bearing0_deg,
                                 center_east_m=racetrack_center_east_m, center_north_m=racetrack_center_north_m)
    if pattern_name == "spiral":
        return pattern_spiral(t, spiral_r_start_m, spiral_r_end_m, spiral_revs, bearing0_deg)
    raise ValueError(f"Unsupported pattern: {pattern_name}")


def enforce_vp(data: np.ndarray, max_vp: float, mode: str) -> Tuple[np.ndarray, float, float]:
    peak_before = float(np.max(np.abs(data))) if data.size else 0.0
    if peak_before <= max_vp:
        return data, 1.0, peak_before
    if mode == "clip":
        return np.clip(data, -max_vp, max_vp), 1.0, peak_before
    scale = max_vp / peak_before
    return data * scale, scale, peak_before


def generate_scenario(out_prefix: str, cal: CalData, fs: int, duration_s: float, sensor_lat: float, sensor_lon: float,
                      pattern_name: str, bearing0_deg: float, range_m: float, period_s: float, swing_deg: float,
                      rate_hz: float, speed_mps: float, racetrack_long_m: float, racetrack_short_m: float,
                      spiral_r_start_m: float, spiral_r_end_m: float, spiral_revs: float, elevation_deg: float,
                      sig_type: str, f0_hz: float, bw_hz: float, sl_db_re_1upa_1m: float,
                      level_mode: str = "physical_sl", constant_rl_db_re_1upa: float = 120.0,
                      use_absorption: bool = False, ambient_noise_enable: bool = False,
                      ambient_noise_db_rel: float = -20.0, ambient_noise_color_alpha: float = 1.0,
                      wave_mod_enable: bool = False, wave_mod_strength: float = 0.25,
                      doppler_enable: bool = False, sound_speed_ms: float = 1500.0,
                      multipath_enable: bool = False, multipath_delay_ms: float = 12.0,
                      multipath_atten_db: float = 10.0, cavitation_mix_enable: bool = False,
                      cavitation_mix_db_rel: float = -12.0, max_vp: float = 10.0, vp_mode: str = "scale",
                      start_epoch_s: float = 0.0, seed: int = 1234, water_density_kgm3: float = 1025.0,
                      output_track_hz: float = 5.0, export_mode: str = "raw_volts", target_full_scale: float = 0.8,
                      racetrack_center_east_m: float = 0.0, racetrack_center_north_m: float = 0.0) -> ScenarioResult:
    n = int(fs * duration_s)
    if n < 2:
        raise ValueError("Duration/sample rate combination produces too few samples.")

    t = np.arange(n, dtype=float) / fs
    dt = 1.0 / fs
    bearing_deg, r_m = choose_motion(t=t, pattern_name=pattern_name, bearing0_deg=bearing0_deg, range_m=range_m,
                                     period_s=period_s, swing_deg=swing_deg, rate_hz=rate_hz, speed_mps=speed_mps,
                                     racetrack_long_m=racetrack_long_m, racetrack_short_m=racetrack_short_m,
                                     spiral_r_start_m=spiral_r_start_m, spiral_r_end_m=spiral_r_end_m,
                                     spiral_revs=spiral_revs, racetrack_center_east_m=racetrack_center_east_m,
                                     racetrack_center_north_m=racetrack_center_north_m)
    vr_mps = np.gradient(r_m, dt)

    src = base_signal(t, sig_type, f0_hz, bw_hz, fs, seed)
    if doppler_enable and sig_type in ("tone", "am_tone", "band_noise"):
        src = apply_doppler_tone_envelope(t, f0_hz, sound_speed_ms, vr_mps, src)
    if wave_mod_enable and wave_mod_strength > 0.0:
        src = apply_wave_amplitude_modulation(src, fs, wave_mod_strength, seed + 101)

    if level_mode == "constant_received_level":
        rl_db = np.full_like(r_m, float(constant_rl_db_re_1upa), dtype=float)
    else:
        rl_db = np.array([rl_from_sl(sl_db_re_1upa_1m, float(r), f0_hz, use_absorption) for r in r_m], dtype=float)

    p_pa = np.array([pressure_pa_from_rl_db(v) for v in rl_db], dtype=float)
    omni_db = interp_1d(cal.freq_hz, cal.omni_db_v_per_upa, f0_hz)
    sx_db = interp_1d(cal.freq_hz, cal.x_db_v_per_ms, f0_hz)
    sy_db = interp_1d(cal.freq_hz, cal.y_db_v_per_ms, f0_hz)
    sz_db = interp_1d(cal.freq_hz, cal.z_db_v_per_ms, f0_hz)
    xy_phase = math.radians(interp_1d(cal.freq_hz, cal.xy_phase_deg, f0_hz))
    z_phase = math.radians(interp_1d(cal.freq_hz, cal.z_phase_deg, f0_hz))

    omni_v_per_pa = db_v_per_upa_to_v_per_pa(omni_db)
    sx_v_per_ms = db_v_per_ms_to_v_per_ms(sx_db)
    sy_v_per_ms = db_v_per_ms_to_v_per_ms(sy_db)
    sz_v_per_ms = db_v_per_ms_to_v_per_ms(sz_db)

    theta = np.deg2rad(bearing_deg)
    phi = math.radians(elevation_deg)
    pressure_wave = p_pa * src
    particle_vel = pressure_wave / (water_density_kgm3 * sound_speed_ms)

    ux = particle_vel * np.cos(theta) * np.cos(phi)
    uy = particle_vel * np.sin(theta) * np.cos(phi)
    uz = particle_vel * np.sin(phi)

    v_omni = pressure_wave * omni_v_per_pa
    v_x = ux * sx_v_per_ms
    v_y = uy * sy_v_per_ms
    v_z = uz * sz_v_per_ms

    if sig_type in ("tone", "am_tone", "band_noise"):
        carrier_y = narrowband_phase_shifted_carrier(t, f0_hz, 0.0)
        carrier_x = narrowband_phase_shifted_carrier(t, f0_hz, xy_phase)
        carrier_z = narrowband_phase_shifted_carrier(t, f0_hz, z_phase)
        amp_x = np.abs(ux) * sx_v_per_ms
        amp_y = np.abs(uy) * sy_v_per_ms
        amp_z = np.abs(uz) * sz_v_per_ms
        sign_x = np.sign(np.cos(theta))
        sign_y = np.sign(np.sin(theta))
        sign_z = np.sign(np.sin(phi)) if abs(np.sin(phi)) > 1e-12 else np.ones_like(theta)
        v_x = sign_x * amp_x * carrier_x
        v_y = sign_y * amp_y * carrier_y
        v_z = sign_z * amp_z * carrier_z

    if multipath_enable:
        delay_s = max(0.0, multipath_delay_ms / 1000.0)
        v_omni = apply_multipath(v_omni, fs, delay_s, multipath_atten_db)
        v_x = apply_multipath(v_x, fs, delay_s, multipath_atten_db)
        v_y = apply_multipath(v_y, fs, delay_s, multipath_atten_db)
        v_z = apply_multipath(v_z, fs, delay_s, multipath_atten_db)

    if cavitation_mix_enable:
        cav = vessel_cavitation_source(n, fs, seed + 303)
        sig_rms = np.std(v_omni) + 1e-12
        cav_scaled = cav * ((sig_rms * (10.0 ** (cavitation_mix_db_rel / 20.0))) / (np.std(cav) + 1e-12))
        v_omni, v_x, v_y, v_z = v_omni + cav_scaled, v_x + 0.2 * cav_scaled, v_y + 0.2 * cav_scaled, v_z + 0.1 * cav_scaled

    if ambient_noise_enable:
        amb = colored_noise(n, fs, ambient_noise_color_alpha, seed + 707)
        sig_rms = np.std(v_omni) + 1e-12
        amb_scaled = amb * ((sig_rms * (10.0 ** (ambient_noise_db_rel / 20.0))) / (np.std(amb) + 1e-12))
        rng = np.random.default_rng(seed + 999)
        v_omni = v_omni + amb_scaled
        v_x = v_x + amb_scaled * rng.uniform(0.8, 1.2)
        v_y = v_y + amb_scaled * rng.uniform(0.8, 1.2)
        v_z = v_z + amb_scaled * rng.uniform(0.8, 1.2)

    data = np.vstack([v_omni, v_x, v_y, v_z]).T.astype(np.float64)
    data, limit_scale_applied, peak_before = enforce_vp(data, max_vp, vp_mode)
    channel_peaks = {"omni_peak_v": float(np.max(np.abs(v_omni))) if v_omni.size else 0.0,
                     "x_peak_v": float(np.max(np.abs(v_x))) if v_x.size else 0.0,
                     "y_peak_v": float(np.max(np.abs(v_y))) if v_y.size else 0.0,
                     "z_peak_v": float(np.max(np.abs(v_z))) if v_z.size else 0.0}

    wav_peak_before = float(np.max(np.abs(data))) if data.size else 0.0
    wav_scale_applied = 1.0
    wav_data = data.copy()
    if export_mode == "normalize_to_full_scale" and wav_peak_before > 0.0:
        wav_scale_applied = target_full_scale / wav_peak_before
        wav_data = data * wav_scale_applied

    wav_path = f"{out_prefix}.wav"
    wav_write(wav_path, fs, wav_data.astype(np.float32))

    ts = start_epoch_s + t
    east = np.zeros_like(t)
    north = np.zeros_like(t)
    lat = np.zeros_like(t)
    lon = np.zeros_like(t)
    for i in range(n):
        e, nn = bearing_range_to_enu(float(bearing_deg[i]), float(r_m[i]))
        east[i], north[i] = e, nn
        dlat, dlon = meters_to_latlon_offsets(sensor_lat, e, nn)
        lat[i], lon[i] = sensor_lat + dlat, sensor_lon + dlon

    de = np.gradient(east, dt)
    dn = np.gradient(north, dt)
    sog = np.sqrt(de * de + dn * dn)
    cog = (np.degrees(np.arctan2(de, dn)) + 360.0) % 360.0
    step = max(1, int(round(fs / output_track_hz)))

    gga_path = f"{out_prefix}_track_gpgga.txt"
    with open(gga_path, "w", encoding="utf-8") as f:
        for i in range(0, n, step):
            f.write(make_gpgga(ts[i], lat[i], lon[i]) + "\n")

    debug_csv_path = f"{out_prefix}_track_debug.csv"
    with open(debug_csv_path, "w", encoding="utf-8") as f:
        f.write("timestamp_s,lat,lon,east_m,north_m,sog_mps,cog_deg,range_m,bearing_deg\n")
        for i in range(0, n, step):
            f.write(f"{ts[i]:.3f},{lat[i]:.8f},{lon[i]:.8f},{east[i]:.3f},{north[i]:.3f},{sog[i]:.3f},{cog[i]:.2f},{r_m[i]:.2f},{bearing_deg[i]:.2f}\n")

    meta = {
        "fs": fs, "duration_s": duration_s,
        "timing": {"start_epoch_s": start_epoch_s, "wav_duration_s": duration_s, "wav_samples": n, "sample_rate_hz": fs,
                   "gps_output_rate_hz": output_track_hz, "gps_points_written": int(math.ceil(n / step))},
        "channels_order": ["omni", "x", "y", "z"], "sensor_lat": sensor_lat, "sensor_lon": sensor_lon,
        "pattern": pattern_name,
        "pattern_params": {"bearing0_deg": bearing0_deg, "range_m": range_m, "period_s": period_s, "swing_deg": swing_deg,
                           "rate_hz": rate_hz, "speed_mps": speed_mps, "speed_kts": speed_mps / KTS_TO_MPS,
                           "racetrack_long_m": racetrack_long_m, "racetrack_short_m": racetrack_short_m,
                           "racetrack_center_east_m": racetrack_center_east_m, "racetrack_center_north_m": racetrack_center_north_m,
                           "spiral_r_start_m": spiral_r_start_m, "spiral_r_end_m": spiral_r_end_m, "spiral_revs": spiral_revs,
                           "elevation_deg": elevation_deg},
        "source": {"sig_type": sig_type, "f0_hz": f0_hz, "bw_hz": bw_hz, "sl_db_re_1upa_1m": sl_db_re_1upa_1m,
                   "level_mode": level_mode, "constant_rl_db_re_1upa": constant_rl_db_re_1upa},
        "calibration": {"omni_units": "dB re 1 V/uPa", "vector_units": "dB re 1 V/(m/s)", "xy_phase_units": "deg",
                        "z_phase_units": "deg", "interpolated_values": {"omni_db_v_per_upa": omni_db, "x_db_v_per_ms": sx_db,
                                                                             "y_db_v_per_ms": sy_db, "z_db_v_per_ms": sz_db,
                                                                             "xy_phase_deg": math.degrees(xy_phase), "z_phase_deg": math.degrees(z_phase)}},
        "physics": {"sound_speed_ms": sound_speed_ms, "water_density_kgm3": water_density_kgm3,
                    "model": "Plane-wave approximation: particle_velocity = pressure / (rho*c)"},
        "ocean_effects": {"use_absorption": use_absorption, "ambient_noise_enable": ambient_noise_enable,
                          "ambient_noise_db_rel": ambient_noise_db_rel, "ambient_noise_color_alpha": ambient_noise_color_alpha,
                          "wave_mod_enable": wave_mod_enable, "wave_mod_strength": wave_mod_strength,
                          "doppler_enable": doppler_enable, "multipath_enable": multipath_enable,
                          "multipath_delay_ms": multipath_delay_ms, "multipath_atten_db": multipath_atten_db,
                          "cavitation_mix_enable": cavitation_mix_enable, "cavitation_mix_db_rel": cavitation_mix_db_rel},
        "voltage_limit": {"max_vp": max_vp, "mode": vp_mode, "peak_before": peak_before, "scale_applied": limit_scale_applied},
        "wav_export": {"mode": export_mode, "target_full_scale": target_full_scale,
                       "peak_before_export_scaling": wav_peak_before, "wav_scale_applied": wav_scale_applied},
        "channel_peaks_raw_volts": channel_peaks,
        "files": {"wav": os.path.abspath(wav_path), "track_gpgga": os.path.abspath(gga_path),
                  "track_debug_csv": os.path.abspath(debug_csv_path)},
    }

    meta_path = f"{out_prefix}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return ScenarioResult(wav_path, gga_path, debug_csv_path, meta_path, east, north, t, r_m, bearing_deg)


class GenerateThread(QtCore.QThread):
    finished_ok = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, kwargs: dict, parent=None):
        super().__init__(parent)
        self.kwargs = kwargs

    def run(self):
        try:
            self.finished_ok.emit(generate_scenario(**self.kwargs))
        except Exception as exc:
            self.failed.emit(str(exc))


class DifarSimWindow(QtWidgets.QMainWindow):
    def __init__(self, project_id: Optional[int] = None, output_dir: Optional[str] = None, db_path: Optional[str] = None,
                 host_window=None):
        super().__init__()
        self.setWindowTitle("DIFAR Synthetic Generator — PyQt5 / M20-105")
        self.resize(1580, 940)
        self.cal: Optional[CalData] = None
        self.cal_path: Optional[str] = None
        self._gen_thread: Optional[GenerateThread] = None
        self.project_id = project_id
        self.output_dir = output_dir
        self.db_path = db_path or DB_FILENAME
        self.host_window = host_window

        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10); root.setSpacing(10)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal); root.addWidget(splitter)

        left_wrap = QtWidgets.QWidget(); left_layout = QtWidgets.QVBoxLayout(left_wrap); left_layout.setSpacing(10)
        scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(left_wrap); scroll.setMinimumWidth(760)
        splitter.addWidget(scroll)

        right_widget = QtWidgets.QWidget(); right = QtWidgets.QVBoxLayout(right_widget); right.setSpacing(10)
        splitter.addWidget(right_widget); splitter.setStretchFactor(0, 0); splitter.setStretchFactor(1, 1); splitter.setSizes([760, 820])

        self.build_calibration_group(left_layout)
        self.build_scenario_group(left_layout)
        self.build_motion_group(left_layout)
        self.build_source_group(left_layout)
        self.build_effects_group(left_layout)
        self.build_output_group(left_layout)
        left_layout.addStretch(1)

        pg.setConfigOptions(antialias=True)
        self.track_plot = pg.PlotWidget(title="GPS Track (ENU meters)")
        self.track_plot.showGrid(x=True, y=True, alpha=0.25)
        self.track_plot.setLabel("bottom", "East (m)")
        self.track_plot.setLabel("left", "North (m)")
        self.track_curve = self.track_plot.plot([], [], pen=pg.mkPen(width=2))
        self.track_plot.addItem(pg.ScatterPlotItem([0], [0], size=12, brush=pg.mkBrush(255, 255, 255)))
        right.addWidget(self.track_plot, 2)

        self.range_plot = pg.PlotWidget(title="Range to Vector Sensor vs Time")
        self.range_plot.showGrid(x=True, y=True, alpha=0.25)
        self.range_plot.setLabel("bottom", "Time (s)")
        self.range_plot.setLabel("left", "Range (m)")
        self.range_curve = self.range_plot.plot([], [], pen=pg.mkPen(width=2))
        right.addWidget(self.range_plot, 1)

        self.log = QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(200)
        right.addWidget(self.log, 0)
        self.append_log("Ready.")

    def append_log(self, text: str):
        self.log.appendPlainText(text)

    def build_calibration_group(self, parent_layout):
        gb = QtWidgets.QGroupBox("Sensor Calibration"); g = QtWidgets.QGridLayout(gb)
        self.le_cal = QtWidgets.QLineEdit(); self.le_cal.setPlaceholderText("Load M20-105 Cal.csv")
        btn = QtWidgets.QPushButton("Load Calibration CSV"); btn.clicked.connect(self.on_load_cal)
        g.addWidget(QtWidgets.QLabel("Calibration File"), 0, 0); g.addWidget(self.le_cal, 0, 1); g.addWidget(btn, 0, 2)
        parent_layout.addWidget(gb)

    def build_scenario_group(self, parent_layout):
        gb = QtWidgets.QGroupBox("Scenario"); g = QtWidgets.QGridLayout(gb)
        self.ds_lat = QtWidgets.QDoubleSpinBox(); self.ds_lat.setRange(-90, 90); self.ds_lat.setDecimals(7); self.ds_lat.setValue(46.0)
        self.ds_lon = QtWidgets.QDoubleSpinBox(); self.ds_lon.setRange(-180, 180); self.ds_lon.setDecimals(7); self.ds_lon.setValue(-60.0)
        self.sb_fs = QtWidgets.QSpinBox(); self.sb_fs.setRange(8000, 384000); self.sb_fs.setValue(48000)
        self.ds_dur = QtWidgets.QDoubleSpinBox(); self.ds_dur.setRange(1, 36000); self.ds_dur.setDecimals(2); self.ds_dur.setValue(120.0)
        self.ds_track_hz = QtWidgets.QDoubleSpinBox(); self.ds_track_hz.setRange(0.1, 50.0); self.ds_track_hz.setDecimals(2); self.ds_track_hz.setValue(5.0)
        g.addWidget(QtWidgets.QLabel("Sensor Lat"), 0, 0); g.addWidget(self.ds_lat, 0, 1); g.addWidget(QtWidgets.QLabel("Sensor Lon"), 0, 2); g.addWidget(self.ds_lon, 0, 3)
        g.addWidget(QtWidgets.QLabel("Sample Rate (Hz)"), 1, 0); g.addWidget(self.sb_fs, 1, 1); g.addWidget(QtWidgets.QLabel("Duration (s)"), 1, 2); g.addWidget(self.ds_dur, 1, 3)
        g.addWidget(QtWidgets.QLabel("Track Output Rate (Hz)"), 2, 0); g.addWidget(self.ds_track_hz, 2, 1)
        parent_layout.addWidget(gb)

    def build_motion_group(self, parent_layout):
        gb = QtWidgets.QGroupBox("Vessel Motion"); v = QtWidgets.QVBoxLayout(gb); v.setSpacing(8)
        top = QtWidgets.QGridLayout()
        self.cb_pattern = QtWidgets.QComboBox(); self.cb_pattern.addItems(["circle", "back_forth", "straight_pass", "racetrack", "spiral"]); self.cb_pattern.currentTextChanged.connect(self.on_pattern_changed)
        self.ds_bearing0 = QtWidgets.QDoubleSpinBox(); self.ds_bearing0.setRange(0, 360); self.ds_bearing0.setDecimals(2); self.ds_bearing0.setValue(0.0)
        self.ds_range = QtWidgets.QDoubleSpinBox(); self.ds_range.setRange(1, 100000); self.ds_range.setDecimals(2); self.ds_range.setValue(1000.0)
        self.ds_speed_kts = QtWidgets.QDoubleSpinBox(); self.ds_speed_kts.setRange(0.0, 80.0); self.ds_speed_kts.setDecimals(2); self.ds_speed_kts.setValue(10.0)
        self.ds_elev = QtWidgets.QDoubleSpinBox(); self.ds_elev.setRange(-90.0, 90.0); self.ds_elev.setDecimals(2); self.ds_elev.setValue(0.0)
        top.addWidget(QtWidgets.QLabel("Pattern"), 0, 0); top.addWidget(self.cb_pattern, 0, 1, 1, 3)
        self.lbl_range = QtWidgets.QLabel("Range (m)")
        top.addWidget(QtWidgets.QLabel("Start Bearing (deg)"), 1, 0); top.addWidget(self.ds_bearing0, 1, 1); top.addWidget(self.lbl_range, 1, 2); top.addWidget(self.ds_range, 1, 3)
        top.addWidget(QtWidgets.QLabel("Speed (kts)"), 2, 0); top.addWidget(self.ds_speed_kts, 2, 1); top.addWidget(QtWidgets.QLabel("Elevation (deg)"), 2, 2); top.addWidget(self.ds_elev, 2, 3)
        v.addLayout(top)
        self.pattern_stack = QtWidgets.QStackedWidget(); v.addWidget(self.pattern_stack)
        page_circle = QtWidgets.QWidget(); g1 = QtWidgets.QGridLayout(page_circle)
        self.ds_period = QtWidgets.QDoubleSpinBox(); self.ds_period.setRange(1, 36000); self.ds_period.setDecimals(2); self.ds_period.setValue(120.0)
        g1.addWidget(QtWidgets.QLabel("Circle Period (s)"), 0, 0); g1.addWidget(self.ds_period, 0, 1); g1.setColumnStretch(2, 1); self.pattern_stack.addWidget(page_circle)
        page_bf = QtWidgets.QWidget(); g2 = QtWidgets.QGridLayout(page_bf)
        self.ds_swing = QtWidgets.QDoubleSpinBox(); self.ds_swing.setRange(0, 180); self.ds_swing.setDecimals(2); self.ds_swing.setValue(60.0)
        self.ds_rate = QtWidgets.QDoubleSpinBox(); self.ds_rate.setRange(0.001, 5.0); self.ds_rate.setDecimals(4); self.ds_rate.setValue(0.03)
        g2.addWidget(QtWidgets.QLabel("Swing (deg)"), 0, 0); g2.addWidget(self.ds_swing, 0, 1); g2.addWidget(QtWidgets.QLabel("Rate (Hz)"), 1, 0); g2.addWidget(self.ds_rate, 1, 1); g2.setColumnStretch(2, 1); self.pattern_stack.addWidget(page_bf)
        page_sp = QtWidgets.QWidget(); g3 = QtWidgets.QGridLayout(page_sp); g3.addWidget(QtWidgets.QLabel("Uses Range as closest approach.\nUses Speed (kts)."), 0, 0, 1, 2); g3.setColumnStretch(2, 1); self.pattern_stack.addWidget(page_sp)
        page_rt = QtWidgets.QWidget(); g4 = QtWidgets.QGridLayout(page_rt)
        self.ds_rt_long = QtWidgets.QDoubleSpinBox(); self.ds_rt_long.setRange(1, 100000); self.ds_rt_long.setDecimals(1); self.ds_rt_long.setValue(2000.0)
        self.ds_rt_short = QtWidgets.QDoubleSpinBox(); self.ds_rt_short.setRange(1, 100000); self.ds_rt_short.setDecimals(1); self.ds_rt_short.setValue(800.0)
        self.ds_rt_center_e = QtWidgets.QDoubleSpinBox(); self.ds_rt_center_e.setRange(-100000, 100000); self.ds_rt_center_e.setDecimals(1); self.ds_rt_center_e.setValue(2000.0)
        self.ds_rt_center_n = QtWidgets.QDoubleSpinBox(); self.ds_rt_center_n.setRange(-100000, 100000); self.ds_rt_center_n.setDecimals(1); self.ds_rt_center_n.setValue(0.0)
        g4.addWidget(QtWidgets.QLabel("Long (m)"), 0, 0); g4.addWidget(self.ds_rt_long, 0, 1); g4.addWidget(QtWidgets.QLabel("Short (m)"), 1, 0); g4.addWidget(self.ds_rt_short, 1, 1)
        g4.addWidget(QtWidgets.QLabel("Center East Offset (m)"), 2, 0); g4.addWidget(self.ds_rt_center_e, 2, 1); g4.addWidget(QtWidgets.QLabel("Center North Offset (m)"), 3, 0); g4.addWidget(self.ds_rt_center_n, 3, 1); g4.setColumnStretch(2, 1); self.pattern_stack.addWidget(page_rt)
        page_spiral = QtWidgets.QWidget(); g5 = QtWidgets.QGridLayout(page_spiral)
        self.ds_sp_r0 = QtWidgets.QDoubleSpinBox(); self.ds_sp_r0.setRange(1, 100000); self.ds_sp_r0.setDecimals(1); self.ds_sp_r0.setValue(3000.0)
        self.ds_sp_r1 = QtWidgets.QDoubleSpinBox(); self.ds_sp_r1.setRange(1, 100000); self.ds_sp_r1.setDecimals(1); self.ds_sp_r1.setValue(500.0)
        self.ds_sp_rev = QtWidgets.QDoubleSpinBox(); self.ds_sp_rev.setRange(0.1, 100); self.ds_sp_rev.setDecimals(2); self.ds_sp_rev.setValue(3.0)
        g5.addWidget(QtWidgets.QLabel("Start Range r0 (m)"), 0, 0); g5.addWidget(self.ds_sp_r0, 0, 1); g5.addWidget(QtWidgets.QLabel("End Range r1 (m)"), 1, 0); g5.addWidget(self.ds_sp_r1, 1, 1); g5.addWidget(QtWidgets.QLabel("Revolutions"), 2, 0); g5.addWidget(self.ds_sp_rev, 2, 1); g5.setColumnStretch(2, 1); self.pattern_stack.addWidget(page_spiral)
        parent_layout.addWidget(gb); self.on_pattern_changed(self.cb_pattern.currentText())

    def build_source_group(self, parent_layout):
        gb = QtWidgets.QGroupBox("Acoustic Source"); g = QtWidgets.QGridLayout(gb)
        self.cb_sig = QtWidgets.QComboBox(); self.cb_sig.addItems(["tone", "am_tone", "band_noise", "vessel_noise"])
        self.ds_f0 = QtWidgets.QDoubleSpinBox(); self.ds_f0.setRange(1, 200000); self.ds_f0.setDecimals(2); self.ds_f0.setValue(800.0)
        self.ds_bw = QtWidgets.QDoubleSpinBox(); self.ds_bw.setRange(1, 50000); self.ds_bw.setDecimals(2); self.ds_bw.setValue(300.0)
        self.ds_sl = QtWidgets.QDoubleSpinBox(); self.ds_sl.setRange(80, 250); self.ds_sl.setDecimals(2); self.ds_sl.setValue(175.0)
        self.cb_level_mode = QtWidgets.QComboBox(); self.cb_level_mode.addItems(["physical_sl", "constant_received_level"]); self.cb_level_mode.currentTextChanged.connect(self.on_level_mode_changed)
        self.ds_constant_rl = QtWidgets.QDoubleSpinBox(); self.ds_constant_rl.setRange(60, 220); self.ds_constant_rl.setDecimals(2); self.ds_constant_rl.setValue(120.0)
        self.ds_c = QtWidgets.QDoubleSpinBox(); self.ds_c.setRange(1200, 1700); self.ds_c.setDecimals(1); self.ds_c.setValue(1500.0)
        self.ds_rho = QtWidgets.QDoubleSpinBox(); self.ds_rho.setRange(900, 1200); self.ds_rho.setDecimals(1); self.ds_rho.setValue(1025.0)
        g.addWidget(QtWidgets.QLabel("Signal Type"), 0, 0); g.addWidget(self.cb_sig, 0, 1, 1, 3)
        g.addWidget(QtWidgets.QLabel("Center Frequency (Hz)"), 1, 0); g.addWidget(self.ds_f0, 1, 1); g.addWidget(QtWidgets.QLabel("Bandwidth (Hz)"), 1, 2); g.addWidget(self.ds_bw, 1, 3)
        g.addWidget(QtWidgets.QLabel("Level Mode"), 2, 0); g.addWidget(self.cb_level_mode, 2, 1); self.lbl_sl = QtWidgets.QLabel("Source Level (dB re 1 µPa @1m)"); g.addWidget(self.lbl_sl, 2, 2); g.addWidget(self.ds_sl, 2, 3)
        g.addWidget(QtWidgets.QLabel("Sound Speed (m/s)"), 3, 0); g.addWidget(self.ds_c, 3, 1); self.lbl_constant_rl = QtWidgets.QLabel("Constant RL (dB re 1 µPa)"); g.addWidget(self.lbl_constant_rl, 3, 2); g.addWidget(self.ds_constant_rl, 3, 3)
        g.addWidget(QtWidgets.QLabel("Water Density (kg/m³)"), 4, 0); g.addWidget(self.ds_rho, 4, 1)
        parent_layout.addWidget(gb); self.on_level_mode_changed(self.cb_level_mode.currentText())

    def build_effects_group(self, parent_layout):
        gb = QtWidgets.QGroupBox("Ocean / Realistic Effects"); g = QtWidgets.QGridLayout(gb)
        self.ck_abs = QtWidgets.QCheckBox("Absorption (Thorp)")
        self.ck_amb = QtWidgets.QCheckBox("Ambient Noise"); self.ds_amb_db = QtWidgets.QDoubleSpinBox(); self.ds_amb_db.setRange(-60, 20); self.ds_amb_db.setDecimals(1); self.ds_amb_db.setValue(-20.0)
        self.ds_amb_alpha = QtWidgets.QDoubleSpinBox(); self.ds_amb_alpha.setRange(0.0, 2.5); self.ds_amb_alpha.setDecimals(2); self.ds_amb_alpha.setValue(1.0)
        self.ck_wave = QtWidgets.QCheckBox("Wave AM Modulation"); self.ds_wave = QtWidgets.QDoubleSpinBox(); self.ds_wave.setRange(0.0, 1.0); self.ds_wave.setDecimals(2); self.ds_wave.setValue(0.25)
        self.ck_dopp = QtWidgets.QCheckBox("Doppler (tone / narrowband)")
        self.ck_mp = QtWidgets.QCheckBox("Multipath Reflection"); self.ds_mp_delay = QtWidgets.QDoubleSpinBox(); self.ds_mp_delay.setRange(0.1, 5000); self.ds_mp_delay.setDecimals(1); self.ds_mp_delay.setValue(12.0)
        self.ds_mp_att = QtWidgets.QDoubleSpinBox(); self.ds_mp_att.setRange(0.0, 60.0); self.ds_mp_att.setDecimals(1); self.ds_mp_att.setValue(10.0)
        self.ck_cav = QtWidgets.QCheckBox("Cavitation / Vessel Broadband Mix"); self.ds_cav_db = QtWidgets.QDoubleSpinBox(); self.ds_cav_db.setRange(-60, 20); self.ds_cav_db.setDecimals(1); self.ds_cav_db.setValue(-12.0)
        g.addWidget(self.ck_abs, 0, 0, 1, 2)
        g.addWidget(self.ck_amb, 1, 0); g.addWidget(QtWidgets.QLabel("Ambient Level (dB rel RMS)"), 1, 1); g.addWidget(self.ds_amb_db, 1, 2); g.addWidget(QtWidgets.QLabel("Ambient Color α"), 1, 3); g.addWidget(self.ds_amb_alpha, 1, 4)
        g.addWidget(self.ck_wave, 2, 0); g.addWidget(QtWidgets.QLabel("Wave Strength"), 2, 1); g.addWidget(self.ds_wave, 2, 2)
        g.addWidget(self.ck_dopp, 3, 0)
        g.addWidget(self.ck_mp, 4, 0); g.addWidget(QtWidgets.QLabel("Delay (ms)"), 4, 1); g.addWidget(self.ds_mp_delay, 4, 2); g.addWidget(QtWidgets.QLabel("Attenuation (dB)"), 4, 3); g.addWidget(self.ds_mp_att, 4, 4)
        g.addWidget(self.ck_cav, 5, 0); g.addWidget(QtWidgets.QLabel("Cavitation Level (dB rel RMS)"), 5, 1); g.addWidget(self.ds_cav_db, 5, 2)
        parent_layout.addWidget(gb)

    def build_output_group(self, parent_layout):
        gb = QtWidgets.QGroupBox("Output"); g = QtWidgets.QGridLayout(gb)
        self.le_out = QtWidgets.QLineEdit("difar_sim")
        self.ds_vp = QtWidgets.QDoubleSpinBox(); self.ds_vp.setRange(0.1, 50.0); self.ds_vp.setDecimals(2); self.ds_vp.setValue(10.0)
        self.cb_vp_mode = QtWidgets.QComboBox(); self.cb_vp_mode.addItems(["scale", "clip"])
        self.cb_export_mode = QtWidgets.QComboBox(); self.cb_export_mode.addItems(["raw_volts", "normalize_to_full_scale"])
        self.btn_generate = QtWidgets.QPushButton("Generate WAV + GPGGA"); self.btn_generate.clicked.connect(self.on_generate)
        g.addWidget(QtWidgets.QLabel("Output Prefix"), 0, 0); g.addWidget(self.le_out, 0, 1, 1, 3)
        g.addWidget(QtWidgets.QLabel("Max Input (Vp)"), 1, 0); g.addWidget(self.ds_vp, 1, 1); g.addWidget(QtWidgets.QLabel("Limit Mode"), 1, 2); g.addWidget(self.cb_vp_mode, 1, 3)
        g.addWidget(QtWidgets.QLabel("WAV Export Mode"), 2, 0); g.addWidget(self.cb_export_mode, 2, 1, 1, 3)
        g.addWidget(self.btn_generate, 3, 0, 1, 4)
        parent_layout.addWidget(gb)

    def on_pattern_changed(self, name: str):
        self.pattern_stack.setCurrentIndex({"circle": 0, "back_forth": 1, "straight_pass": 2, "racetrack": 3, "spiral": 4}.get(name, 0))
        self.lbl_range.setText("Closest Approach (m)" if name == "straight_pass" else "Range (m)")

    def on_level_mode_changed(self, mode: str):
        is_physical = mode == "physical_sl"
        self.lbl_sl.setVisible(is_physical); self.ds_sl.setVisible(is_physical)
        self.lbl_constant_rl.setVisible(not is_physical); self.ds_constant_rl.setVisible(not is_physical)

    def on_load_cal(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Calibration CSV", "", "CSV Files (*.csv)")
        if not path:
            return
        try:
            self.cal = load_m20105_cal_csv(path)
            self.cal_path = path
            self.le_cal.setText(path)
            self.append_log(f"Loaded calibration: {path}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Calibration Error", str(exc))

    def on_generate(self):
        if self.cal is None:
            QtWidgets.QMessageBox.warning(self, "Missing Calibration", "Load the M20-105 calibration CSV first.")
            return
        if self._gen_thread is not None and self._gen_thread.isRunning():
            QtWidgets.QMessageBox.information(self, "Generation Running", "A scenario is already being generated.")
            return
        self.btn_generate.setEnabled(False)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        out_name = os.path.basename(self.le_out.text().strip() or "difar_sim")
        out_prefix = out_name
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            out_prefix = os.path.join(self.output_dir, out_name)

        kwargs = dict(out_prefix=out_prefix, cal=self.cal, fs=int(self.sb_fs.value()),
                      duration_s=float(self.ds_dur.value()), sensor_lat=float(self.ds_lat.value()), sensor_lon=float(self.ds_lon.value()),
                      pattern_name=self.cb_pattern.currentText(), bearing0_deg=float(self.ds_bearing0.value()), range_m=float(self.ds_range.value()),
                      period_s=float(self.ds_period.value()), swing_deg=float(self.ds_swing.value()), rate_hz=float(self.ds_rate.value()),
                      speed_mps=float(self.ds_speed_kts.value()) * KTS_TO_MPS, racetrack_long_m=float(self.ds_rt_long.value()),
                      racetrack_short_m=float(self.ds_rt_short.value()), spiral_r_start_m=float(self.ds_sp_r0.value()),
                      spiral_r_end_m=float(self.ds_sp_r1.value()), spiral_revs=float(self.ds_sp_rev.value()), elevation_deg=float(self.ds_elev.value()),
                      sig_type=self.cb_sig.currentText(), f0_hz=float(self.ds_f0.value()), bw_hz=float(self.ds_bw.value()),
                      sl_db_re_1upa_1m=float(self.ds_sl.value()), level_mode=self.cb_level_mode.currentText(),
                      constant_rl_db_re_1upa=float(self.ds_constant_rl.value()), use_absorption=self.ck_abs.isChecked(),
                      ambient_noise_enable=self.ck_amb.isChecked(), ambient_noise_db_rel=float(self.ds_amb_db.value()),
                      ambient_noise_color_alpha=float(self.ds_amb_alpha.value()), wave_mod_enable=self.ck_wave.isChecked(),
                      wave_mod_strength=float(self.ds_wave.value()), doppler_enable=self.ck_dopp.isChecked(), sound_speed_ms=float(self.ds_c.value()),
                      multipath_enable=self.ck_mp.isChecked(), multipath_delay_ms=float(self.ds_mp_delay.value()),
                      multipath_atten_db=float(self.ds_mp_att.value()), cavitation_mix_enable=self.ck_cav.isChecked(),
                      cavitation_mix_db_rel=float(self.ds_cav_db.value()), max_vp=float(self.ds_vp.value()), vp_mode=self.cb_vp_mode.currentText(),
                      start_epoch_s=time.time(), seed=int(time.time()) & 0xFFFF, water_density_kgm3=float(self.ds_rho.value()),
                      output_track_hz=float(self.ds_track_hz.value()), export_mode=self.cb_export_mode.currentText(), target_full_scale=0.8,
                      racetrack_center_east_m=float(self.ds_rt_center_e.value()), racetrack_center_north_m=float(self.ds_rt_center_n.value()))
        self._gen_thread = GenerateThread(kwargs, self)
        self._gen_thread.finished_ok.connect(self._on_generate_done)
        self._gen_thread.failed.connect(self._on_generate_error)
        self._gen_thread.finished.connect(self._on_generate_thread_finished)
        self._gen_thread.start()

    @staticmethod
    def _ensure_chart_track_tables(conn):
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS gps_tracks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER,
                name TEXT,
                source_file TEXT,
                color TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS gps_track_points (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER NOT NULL,
                point_index INTEGER,
                timestamp_utc TEXT,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                elevation_m REAL,
                FOREIGN KEY(track_id) REFERENCES gps_tracks(id) ON DELETE CASCADE
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_gps_tracks_project ON gps_tracks(project_id, created_at)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_gps_track_points_track ON gps_track_points(track_id, point_index)")
        conn.commit()

    def _save_debug_track_to_db(self, result):
        if self.project_id is None:
            self.append_log("GPS DB note: no selected project, skipping GPS track DB save.")
            return
        if not os.path.isfile(result.debug_csv_path):
            return

        points = []
        try:
            df = pd.read_csv(result.debug_csv_path)
            for i, row in df.iterrows():
                ts_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(row["timestamp_s"])))
                points.append((i, ts_iso, float(row["lat"]), float(row["lon"]), None))
        except Exception as exc:
            self.append_log(f"GPS DB warning: failed reading debug track CSV: {exc}")
            return

        if not points:
            self.append_log("GPS DB note: no track points parsed; skipping GPS track DB save.")
            return

        try:
            conn = sqlite3.connect(self.db_path)
            self._ensure_chart_track_tables(conn)
            cur = conn.cursor()
            base = os.path.splitext(os.path.basename(result.gga_path))[0]
            track_name = f"DIFAR Sim: {base}"
            color = "#03DFE2"
            cur.execute(
                "INSERT INTO gps_tracks (project_id, name, source_file, color) VALUES (?, ?, ?, ?)",
                (int(self.project_id), track_name, os.path.abspath(result.gga_path), color),
            )
            track_id = cur.lastrowid
            cur.executemany(
                "INSERT INTO gps_track_points (track_id, point_index, timestamp_utc, latitude, longitude, elevation_m) VALUES (?, ?, ?, ?, ?, ?)",
                [(track_id, idx, ts, lat, lon, ele) for idx, ts, lat, lon, ele in points],
            )
            conn.commit()
            conn.close()
            self.append_log(f"GPS track saved to project DB (track_id={track_id}, points={len(points)}).")
            if self.host_window is not None and hasattr(self.host_window, "refresh_chart_tracks"):
                try:
                    self.host_window.refresh_chart_tracks(select_id=track_id)
                except Exception:
                    pass
        except Exception as exc:
            self.append_log(f"GPS DB warning: {exc}")

    def _on_generate_thread_finished(self):
        try:
            QtWidgets.QApplication.restoreOverrideCursor()
        except Exception:
            pass
        self.btn_generate.setEnabled(True)
        self._gen_thread = None

    def _on_generate_done(self, result):
        if self.cal_path:
            try:
                with open(result.meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta["calibration_file"] = os.path.abspath(self.cal_path)
                with open(result.meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
            except Exception as exc:
                self.append_log(f"Meta patch warning: {exc}")
        self.track_curve.setData(result.east_m, result.north_m)
        self.track_plot.enableAutoRange()
        self.range_curve.setData(result.time_s, result.range_m)
        self.range_plot.enableAutoRange()
        self.append_log(f"Done. WAV: {os.path.abspath(result.wav_path)}")
        self.append_log(f"Done. GPGGA: {os.path.abspath(result.gga_path)}")
        self.append_log(f"Done. Debug CSV: {os.path.abspath(result.debug_csv_path)}")
        self.append_log(f"Done. Meta: {os.path.abspath(result.meta_path)}")
        self._save_debug_track_to_db(result)

    def _on_generate_error(self, msg: str):
        QtWidgets.QMessageBox.critical(self, "Generation Error", msg)
        self.append_log(f"Error: {msg}")


def launch_difar_simulator(parent=None, calibration_csv: Optional[str] = None, project_id: Optional[int] = None,
                           output_dir: Optional[str] = None, db_path: Optional[str] = None,
                           host_window=None) -> DifarSimWindow:
    window = DifarSimWindow(project_id=project_id, output_dir=output_dir, db_path=db_path, host_window=host_window)
    window.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
    if parent is not None:
        window.setParent(parent, QtCore.Qt.Window)
        window.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, True)
        window.setWindowFlag(QtCore.Qt.Tool, True)
    if calibration_csv and os.path.isfile(calibration_csv):
        window.cal = load_m20105_cal_csv(calibration_csv)
        window.cal_path = calibration_csv
        window.le_cal.setText(calibration_csv)
        window.append_log(f"Loaded calibration: {calibration_csv}")
    window.show()
    window.raise_()
    window.activateWindow()
    return window


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(DARK_QSS)
    w = launch_difar_simulator()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
