#!/usr/bin/env python3
import sys
import os
import shutil
from datetime import datetime, timezone
import json
import csv
import math
import sqlite3
import contextlib
import numpy as np
import numpy.fft as fft
import matplotlib
import soundfile as sf
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import RectangleSelector, SpanSelector
from matplotlib.figure import Figure
import scipy.signal.windows as windows
from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, welch, wiener, hilbert, periodogram, find_peaks

# --- Safe filter wrappers to avoid padlen errors ---
import numpy as _np
from scipy.signal import filtfilt as _filtfilt_orig, sosfiltfilt as _sosfiltfilt_orig

def safe_filtfilt(b, a, x, **kwargs):
    """Robust filtfilt that auto-adjusts padlen for short vectors."""
    x = _np.asarray(x)
    axis = kwargs.get('axis', -1)
    n = x.shape[axis]
    padlen = kwargs.pop('padlen', None)
    padtype = kwargs.get('padtype', 'odd')
    if padlen is None:
        padlen = 3 * (max(len(a), len(b)) - 1)
    if n <= padlen:
        if n > 1:
            kwargs['padlen'] = max(1, n - 1)
        else:
            kwargs['padtype'] = None
            kwargs.pop('padlen', None)
    try:
        return _filtfilt_orig(b, a, x, **kwargs)
    except Exception:
        # last resort: no padding
        kwargs['padtype'] = None
        kwargs.pop('padlen', None)
        try:
            return _filtfilt_orig(b, a, x, **kwargs)
        except Exception:
            return x

def safe_sosfiltfilt(sos, x, **kwargs):
    """Robust sosfiltfilt that auto-adjusts padlen for short vectors."""
    x = _np.asarray(x)
    axis = kwargs.get('axis', -1)
    n = x.shape[axis]
    padlen = kwargs.pop('padlen', None)
    padtype = kwargs.get('padtype', 'odd')
    # Estimate effective order: 2 per biquad stage
    eff_order = 2 * (len(sos) if hasattr(sos, '__len__') else 1)
    if padlen is None:
        padlen = 3 * max(1, eff_order - 1)
    if n <= padlen:
        if n > 1:
            kwargs['padlen'] = max(1, n - 1)
        else:
            kwargs['padtype'] = None
            kwargs.pop('padlen', None)
    try:
        return _sosfiltfilt_orig(sos, x, **kwargs)
    except Exception:
        # last resort: no padding
        kwargs['padtype'] = None
        kwargs.pop('padlen', None)
        try:
            return _sosfiltfilt_orig(sos, x, **kwargs)
        except Exception:
            return x
from scipy.fft import rfft, rfftfreq
import pandas as pd
import pywt
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QSplashScreen, QDialog, QFormLayout, QLineEdit, QPushButton, QVBoxLayout, QMessageBox
from PyQt5.QtCore import QUrl, Qt, QTimer
from PyQt5.QtGui import QDesktopServices, QFont, QPixmap
import qdarkstyle
import pyqtgraph as pg
import tempfile

try:
    import folium
except Exception:
    folium = None

try:
    from PyQt5 import QtWebEngineWidgets
except Exception:
    QtWebEngineWidgets = None
import xml.etree.ElementTree as ET
from PyQt5.QtMultimedia import QSound
import joblib
from sklearn.cluster import KMeans
import importlib
import uuid, hashlib
import json, base64, re
# keep this as a normal (unicode) string literal:
with open("public.pem", "rb") as f:
    PUBLIC_PEM = f.read()
__version__ = "1.0.0"
# immediately load it
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding

PUBLIC_KEY = serialization.load_pem_public_key(PUBLIC_PEM)
import librosa

# ── Shared utilities ────────────────────────────────────────────────────────
from shared import (
    DB_FILENAME, safe_filtfilt, safe_sosfiltfilt, multitaper_psd, bandpass_filter,
    lighten_color, load_help_text, load_hydrophone_curves, save_hydrophone_curve,
    import_hydrophone_curve_file, init_db, get_setting, set_setting,
    log_measurement, fetch_logs, fetch_archived_logs, archive_log_entry,
    unarchive_log_entry, log_spl_calculation, fetch_spl_calculations,
    fetch_spl_archived_calculations, update_spl_calculation, archive_spl_calculation,
    unarchive_spl_calculation, load_or_convert_model, TrimDialog, MplCanvas,
)
# ── Tool module imports ─────────────────────────────────────────────────────
from tools_wav_file import WavFileToolsMixin
from tools_measurement import MeasurementToolsMixin
from tools_modelling import ModellingToolsMixin
from tools_detection import DetectionToolsMixin
from tools_database import DatabaseToolsMixin
# ───────────────────────────────────────────────────────────────────────────



# === Multichannel helpers (inserted) ===
def _selected_channel_indices(self):
    """Return list of selected channel indices (0-based) from UI if present;
    else if data is 2D, all columns; else [0]."""
    try:
        cbs = getattr(self, "channel_checkboxes", None) or getattr(self, "channel_checks", None)
        if cbs:
            idxs = [i for i, cb in enumerate(cbs) if getattr(cb, "isChecked", lambda: False)()]
            if idxs:
                return idxs
    except Exception:
        pass
    x = getattr(self, "full_data", None)
    if x is not None and getattr(x, "ndim", 1) == 2 and x.shape[1] >= 1:
        return list(range(int(x.shape[1])))
    return [0]

def _per_channel_basename(self):
    import os
    base, ext = os.path.splitext(getattr(self, "file_name", "measurement"))
    return base, (ext if ext else "")


REQUEST_FILE = "license.req"
LICENSE_FILE = "license.dat"


# High‐resolution multiplier for slider values
# timeline/audio state

# ---------------------
# Helper Functions
# ---------------------


def lighten_color(color, amount=0.4):
    color = color.lstrip('#')
    r, g, b = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f'#{r:02X}{g:02X}{b:02X}'


def load_help_text():
    HELP_FILENAME = "help.txt"
    try:
        with open(HELP_FILENAME, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ("Help file not found. Please refer to the documentation.\n\n"
                "Analyze.exe is a comprehensive tool for analyzing WAV audio files. "
                "For further assistance, please contact support.")

def compute_hwid():
    # pick e.g. MAC + machine UUID
    mac = uuid.getnode()
    try:
        machine_uuid = open('/sys/class/dmi/id/product_uuid').read().strip()
    except Exception:
        machine_uuid = "unknown"
    raw = f"{mac:012X}|{machine_uuid}"
    return hashlib.sha256(raw.encode('utf8')).hexdigest()


def verify_signature(lic):
    sig = base64.b64decode(lic.pop("signature"))
    raw = json.dumps(lic, sort_keys=True).encode("utf8")
    try:
        PUBLIC_KEY.verify(
           sig,
           raw,
           padding.PKCS1v15(),
           hashes.SHA256()
        )
        return True
    except:
        return False
# ---------------------
# Signal Processing Functions
# ---------------------

def multitaper_psd(segment, fs, NW=3, Kmax=None):
    M = len(segment)
    if Kmax is None:
        Kmax = int(2 * NW - 1)
    tapers = windows.dpss(M, NW, Kmax)
    psd_accum = 0
    for taper in tapers:
        tapered_signal = segment * taper
        fft_result = np.fft.rfft(tapered_signal)
        psd = np.abs(fft_result) ** 2
        psd_accum += psd
    psd_avg = psd_accum / Kmax
    freqs = np.fft.rfftfreq(M, d=1/fs)
    return freqs, psd_avg


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return safe_sosfiltfilt(sos, data)

# ---------------------
# Hydrophone Curve Functions
# ---------------------

def import_hydrophone_curve_file(file_path, curve_name):
    freqs = []
    sens = []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    curve_started = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if "CURVE" in line:
            curve_started = True
            continue
        if not curve_started:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            f_khz = float(parts[0])
            s_db = float(parts[1])
            freqs.append(f_khz * 1000)
            sens.append(s_db)
        except ValueError:
            continue
    if not freqs:
        raise ValueError("No valid data found.")
    freqs = np.array(freqs)
    sens = np.array(sens)
    min_freq = int(np.min(freqs))
    max_freq = int(np.max(freqs))
    new_freqs = np.arange(min_freq, max_freq + 1, 1)
    new_sens = np.interp(np.log10(new_freqs), np.log10(freqs), sens)
    return min_freq, max_freq, new_sens.tolist()


def save_hydrophone_curve(curve_name, file_name, min_freq, max_freq, sens_list):
    conn = sqlite3.connect(DB_FILENAME)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO hydrophone_curves (curve_name, file_name, min_frequency, max_frequency, sensitivity_json) VALUES (?, ?, ?, ?, ?)",
        (curve_name, file_name, min_freq, max_freq, json.dumps(sens_list))
    )
    conn.commit()
    conn.close()


def load_hydrophone_curves():
    conn = sqlite3.connect(DB_FILENAME)
    cur = conn.cursor()
    cur.execute("SELECT id, curve_name, min_frequency, max_frequency, sensitivity_json FROM hydrophone_curves")
    rows = cur.fetchall()
    conn.close()
    curves = {}
    for row in rows:
        curve_id, curve_name, min_freq, max_freq, sens_json = row
        curves[curve_id] = {
            "curve_name": curve_name,
            "min_freq": min_freq,
            "max_freq": max_freq,
            "sensitivity": json.loads(sens_json)
        }
    return curves

# ---------------------
# SQLite Database Functions
# ---------------------
DB_FILENAME = "analysis_log.db"

def init_db():
    conn = sqlite3.connect(DB_FILENAME)
    cur = conn.cursor()
    tables = [
        ("measurements",
         "id INTEGER PRIMARY KEY AUTOINCREMENT, file_name TEXT, method TEXT, target_frequency REAL, start_time REAL, end_time REAL, window_length REAL, max_voltage REAL, bandwidth REAL, measured_voltage REAL, filter_applied INTEGER, screenshot TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP"),
        ("archive",
         "id INTEGER PRIMARY KEY AUTOINCREMENT, file_name TEXT, method TEXT, target_frequency REAL, start_time REAL, end_time REAL, window_length REAL, max_voltage REAL, bandwidth REAL, measured_voltage REAL, filter_applied INTEGER, screenshot TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP"),
        ("hydrophone_curves",
         "id INTEGER PRIMARY KEY AUTOINCREMENT, curve_name TEXT, file_name TEXT, min_frequency INTEGER, max_frequency INTEGER, sensitivity_json TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP"),
        ("spl_calculations",
         "id INTEGER PRIMARY KEY AUTOINCREMENT, file_name TEXT, voltage_log_id INTEGER, hydrophone_curve TEXT, target_frequency REAL, rms_voltage REAL, spl REAL, start_time REAL, end_time REAL, window_length REAL, max_voltage REAL, bandwidth REAL, screenshot TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP"),
        ("spl_archive",
         "id INTEGER PRIMARY KEY AUTOINCREMENT, file_name TEXT, voltage_log_id INTEGER, hydrophone_curve TEXT, target_frequency REAL, rms_voltage REAL, spl REAL, start_time REAL, end_time REAL, window_length REAL, max_voltage REAL, bandwidth REAL, screenshot TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP"),
        ("tvr_curves",
         "id INTEGER PRIMARY KEY AUTOINCREMENT, curve_name TEXT, file_name TEXT, min_frequency INTEGER, max_frequency INTEGER, tvr_json TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP"),
        ("spectrograms",
         "id INTEGER PRIMARY KEY AUTOINCREMENT, file_name TEXT, project_id INTEGER, start_sample INTEGER, end_sample INTEGER, start_time REAL, end_time REAL, sample_rate REAL, image_path TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(project_id) REFERENCES projects(id)"),
        ("spectrogram_annotations",
         "id INTEGER PRIMARY KEY AUTOINCREMENT, spectrogram_id INTEGER, x0 REAL, y0 REAL, x1 REAL, y1 REAL, label TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(spectrogram_id) REFERENCES spectrograms(id)"),
        ("marine_call_library",
         "id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER, name TEXT NOT NULL, fmin_hz REAL, fmax_hz REAL, min_duration_s REAL, max_duration_s REAL, notes TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(project_id) REFERENCES projects(id)"),
        ("gps_tracks",
         "id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER, name TEXT NOT NULL, source_file TEXT, color TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(project_id) REFERENCES projects(id)"),
        ("gps_track_points",
         "id INTEGER PRIMARY KEY AUTOINCREMENT, track_id INTEGER NOT NULL, point_index INTEGER, timestamp_utc TEXT, latitude REAL NOT NULL, longitude REAL NOT NULL, elevation_m REAL, FOREIGN KEY(track_id) REFERENCES gps_tracks(id) ON DELETE CASCADE"),
        ("app_settings",
         "key TEXT PRIMARY KEY, value TEXT")
    ]
    for name, cols in tables:
        cur.execute(f"CREATE TABLE IF NOT EXISTS {name} ({cols})")
    conn.commit()

    # ── add misc column if missing ────────────────────────────────────────
    # ── add misc column if missing ────────────────────────────────────────
    cur.execute("PRAGMA table_info(measurements)")
    cols = [r[1] for r in cur.fetchall()]
    if 'misc' not in cols:
        cur.execute("ALTER TABLE measurements ADD COLUMN misc REAL")
        cur.execute("ALTER TABLE archive      ADD COLUMN misc REAL")

    # ── add distance, notes, and near-field columns to SPL tables if missing ─
    _spl_extra_cols = [
        ("distance",     "REAL"),
        ("notes",        "REAL"),
        ("spl_nf",       "REAL"),
        ("nf_enabled",   "INTEGER"),
        ("nf_delta_db",  "REAL"),
        ("nf_radius_m",  "REAL"),
        ("nf_range_m",   "REAL"),
        ("nf_c_ms",      "REAL"),
        ("nf_c_source",  "TEXT"),
        ("nf_c_depth_m", "REAL"),
        ("nf_ctd_id",    "INTEGER"),
    ]
    for tbl in ("spl_calculations", "spl_archive"):
        cur.execute(f"PRAGMA table_info({tbl})")
        existing = [r[1] for r in cur.fetchall()]
        for col, typ in _spl_extra_cols:
            if col not in existing:
                cur.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} {typ}")

    # ── add channel + SPL metadata to spectrograms ───────────────────────
    cur.execute("PRAGMA table_info(spectrograms)")
    spec_cols = [r[1] for r in cur.fetchall()]
    if "channel_index" not in spec_cols:
        cur.execute("ALTER TABLE spectrograms ADD COLUMN channel_index INTEGER")
    if "channel_name" not in spec_cols:
        cur.execute("ALTER TABLE spectrograms ADD COLUMN channel_name TEXT")
    if "hydrophone_curve" not in spec_cols:
        cur.execute("ALTER TABLE spectrograms ADD COLUMN hydrophone_curve TEXT")
    if "distance" not in spec_cols:
        cur.execute("ALTER TABLE spectrograms ADD COLUMN distance REAL")
    if "spl_db" not in spec_cols:
        cur.execute("ALTER TABLE spectrograms ADD COLUMN spl_db REAL")
    if "spl_freq" not in spec_cols:
        cur.execute("ALTER TABLE spectrograms ADD COLUMN spl_freq REAL")

    # ── indexes for spectrogram storage ───────────────────────────────────
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_spectrograms_file_project "
        "ON spectrograms(file_name, project_id, start_sample, end_sample)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_spectrogram_annotations_spec "
        "ON spectrogram_annotations(spectrogram_id)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_marine_call_library_scope "
        "ON marine_call_library(project_id, fmin_hz, fmax_hz)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_gps_tracks_project "
        "ON gps_tracks(project_id, created_at)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_gps_track_points_track "
        "ON gps_track_points(track_id, point_index)"
    )


def get_setting(key, default=None):
    conn = sqlite3.connect(DB_FILENAME)
    cur = conn.cursor()
    cur.execute("SELECT value FROM app_settings WHERE key=?", (key,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else default


def set_setting(key, value):
    conn = sqlite3.connect(DB_FILENAME)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO app_settings (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
    conn.commit()
    conn.close()

def log_measurement(
    file_name, method, target_frequency,
    start_time, end_time, window_length,
    max_voltage, bandwidth, measured_voltage,
    filter_applied, screenshot,
    misc=None                            # ← new parameter
):
    conn = sqlite3.connect(DB_FILENAME)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO measurements
          (file_name, method, target_frequency,
           start_time, end_time, window_length,
           max_voltage, bandwidth, measured_voltage,
           filter_applied, screenshot, misc)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            file_name, method,    target_frequency,
            start_time, end_time, window_length,
            max_voltage, bandwidth, measured_voltage,
            int(filter_applied), screenshot, misc
        )
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id



def fetch_logs():
    conn = sqlite3.connect(DB_FILENAME)
    cur = conn.cursor()
    cur.execute("SELECT id, file_name, method, target_frequency, start_time, end_time, window_length, max_voltage, bandwidth, measured_voltage, filter_applied, screenshot, timestamp FROM measurements ORDER BY id DESC")
    rows = cur.fetchall()
    conn.close()
    return rows


def fetch_archived_logs():
    conn = sqlite3.connect(DB_FILENAME)
    cur = conn.cursor()
    cur.execute("SELECT id, file_name, method, target_frequency, start_time, end_time, window_length, max_voltage, bandwidth, measured_voltage, filter_applied, screenshot, timestamp FROM archive ORDER BY id DESC")
    rows = cur.fetchall()
    conn.close()
    return rows


def archive_log_entry(entry_id):
    conn = sqlite3.connect(DB_FILENAME)
    cur = conn.cursor()
    cur.execute("SELECT file_name, method, target_frequency, start_time, end_time, window_length, max_voltage, bandwidth, measured_voltage, filter_applied, screenshot FROM measurements WHERE id = ?", (entry_id,))
    row = cur.fetchone()
    if row:
        cur.execute(
            "INSERT INTO archive (file_name, method, target_frequency, start_time, end_time, window_length, max_voltage, bandwidth, measured_voltage, filter_applied, screenshot) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            row
        )
        cur.execute("DELETE FROM measurements WHERE id = ?", (entry_id,))
        conn.commit()
    conn.close()


def unarchive_log_entry(entry_id):
    conn = sqlite3.connect(DB_FILENAME)
    cur = conn.cursor()
    cur.execute("SELECT file_name, method, target_frequency, start_time, end_time, window_length, max_voltage, bandwidth, measured_voltage, filter_applied, screenshot FROM archive WHERE id = ?", (entry_id,))
    row = cur.fetchone()
    if row:
        cur.execute(
            "INSERT INTO measurements (file_name, method, target_frequency, start_time, end_time, window_length, max_voltage, bandwidth, measured_voltage, filter_applied, screenshot) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            row
        )
        cur.execute("DELETE FROM archive WHERE id = ?", (entry_id,))
        conn.commit()
    conn.close()


def log_spl_calculation(file_name, voltage_log_id, hydrophone_curve, target_frequency, rms_voltage, spl, start_time, end_time, window_length, max_voltage, bandwidth, screenshot):
    conn = sqlite3.connect(DB_FILENAME)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO spl_calculations (file_name, voltage_log_id, hydrophone_curve, target_frequency, rms_voltage, spl, start_time, end_time, window_length, max_voltage, bandwidth, screenshot) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (file_name, voltage_log_id, hydrophone_curve, target_frequency, rms_voltage, spl, start_time, end_time, window_length, max_voltage, bandwidth, screenshot)
    )
    conn.commit()
    conn.close()


def fetch_spl_calculations():
    conn = sqlite3.connect(DB_FILENAME)
    cur = conn.cursor()
    cur.execute("SELECT id, file_name, voltage_log_id, hydrophone_curve, target_frequency, rms_voltage, spl, start_time, end_time, window_length, max_voltage, bandwidth, screenshot, timestamp FROM spl_calculations ORDER BY id DESC")
    rows = cur.fetchall()
    conn.close()
    return rows


def fetch_spl_archived_calculations():
    conn = sqlite3.connect(DB_FILENAME)
    cur = conn.cursor()
    cur.execute("SELECT id, file_name, voltage_log_id, hydrophone_curve, target_frequency, rms_voltage, spl, start_time, end_time, window_length, max_voltage, bandwidth, screenshot, timestamp FROM spl_archive ORDER BY id DESC")
    rows = cur.fetchall()
    conn.close()
    return rows


def update_spl_calculation(entry_id, file_name, voltage_log_id, hydrophone_curve, target_frequency, rms_voltage, spl, start_time, end_time, window_length, max_voltage, bandwidth, screenshot):
    conn = sqlite3.connect(DB_FILENAME)
    cur = conn.cursor()
    cur.execute(
        "UPDATE spl_calculations SET file_name = ?, voltage_log_id = ?, hydrophone_curve = ?, target_frequency = ?, rms_voltage = ?, spl = ?, start_time = ?, end_time = ?, window_length = ?, max_voltage = ?, bandwidth = ?, screenshot = ? WHERE id = ?",
        (file_name, voltage_log_id, hydrophone_curve, target_frequency, rms_voltage, spl, start_time, end_time, window_length, max_voltage, bandwidth, screenshot, entry_id)
    )
    conn.commit()
    conn.close()


def archive_spl_calculation(entry_id):
    conn = sqlite3.connect(DB_FILENAME)
    cur = conn.cursor()
    cur.execute("SELECT file_name, voltage_log_id, hydrophone_curve, target_frequency, rms_voltage, spl, start_time, end_time, window_length, max_voltage, bandwidth, screenshot FROM spl_calculations WHERE id = ?", (entry_id,))
    row = cur.fetchone()
    if row:
        cur.execute(
            "INSERT INTO spl_archive (file_name, voltage_log_id, hydrophone_curve, target_frequency, rms_voltage, spl, start_time, end_time, window_length, max_voltage, bandwidth, screenshot) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            row
        )
        cur.execute("DELETE FROM spl_calculations WHERE id = ?", (entry_id,))
        conn.commit()
    conn.close()


def unarchive_spl_calculation(entry_id):
    conn = sqlite3.connect(DB_FILENAME)
    cur = conn.cursor()
    cur.execute("SELECT file_name, voltage_log_id, hydrophone_curve, target_frequency, rms_voltage, spl, start_time, end_time, window_length, max_voltage, bandwidth, screenshot FROM spl_archive WHERE id = ?", (entry_id,))
    row = cur.fetchone()
    if row:
        cur.execute(
            "INSERT INTO spl_calculations (file_name, voltage_log_id, hydrophone_curve, target_frequency, rms_voltage, spl, start_time, end_time, window_length, max_voltage, bandwidth, screenshot) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            row
        )
        cur.execute("DELETE FROM spl_archive WHERE id = ?", (entry_id,))
        conn.commit()
    conn.close()

def load_or_convert_model(model_dir, cache_dir="model_cache"):
        """
        Attempts to load a native Keras model (.keras or .h5).  
        If the directory contains a legacy SavedModel, converts it once into .keras, caches it, and returns the model.
        """
        os.makedirs(cache_dir, exist_ok=True)
        base = os.path.basename(os.path.normpath(model_dir))
        cache_path = os.path.join(cache_dir, f"{base}.keras")

        # 1) If we have a cached .keras file, load that directly
        if os.path.exists(cache_path):
            return tf.keras.models.load_model(cache_path)

        # 2) Try loading directly (works for .keras and .h5)
        try:
            return tf.keras.models.load_model(model_dir)
        except Exception as e:
            msg = str(e)
            # 3) If it fails due to legacy SavedModel format, convert
            if "legacy savedmodel format is not supported" in msg.lower():
                print(f"Converting legacy SavedModel at '{model_dir}' to .keras format...")
                # load under TF2.x
                legacy_model = tf.keras.models.load_model(model_dir, compile=False)
                print(f"Saving converted model to '{cache_path}'...")
                legacy_model.save(cache_path, save_format="keras")
                return legacy_model
            else:
                # re-raise other errors
                raise

# ---------------------------------------------------
# 1) TRIM DIALOG CLASS 
# ---------------------------------------------------
class TrimDialog(QtWidgets.QDialog):
    """
    Dialog that asks for:
      - Trim Start (seconds to remove from the beginning)
      - Trim End   (seconds to remove from the end)
    Returns (start_trim, end_trim) if accepted, or raises ValueError if invalid.
    """
    def __init__(self, parent=None, max_duration=0.0):
        super().__init__(parent)
        self.log_current_page = 0
        self.log_entries_per_page = 50
        
        self.setWindowTitle("Trim WAV File")
        self.setModal(True)

        # Line edits for “seconds to trim from start” and “seconds to trim from end”
        self.start_edit = QtWidgets.QLineEdit("0.0")
        self.end_edit   = QtWidgets.QLineEdit("0.0")
        self.start_edit.setFixedWidth(80)
        self.end_edit.setFixedWidth(80)

        form = QtWidgets.QFormLayout()
        form.addRow("Trim Start (s):", self.start_edit)
        form.addRow("Trim End   (s):", self.end_edit)

        # Show the total file duration for reference
        info = QtWidgets.QLabel(f"(File length ≈ {max_duration:.2f} s)")
        form.addRow(info)

        # OK / Cancel buttons
        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)

        vbox = QtWidgets.QVBoxLayout(self)
        vbox.addLayout(form)
        vbox.addWidget(btn_box)

    def values(self):
        """
        Parse and return (start_trim, end_trim) as floats.
        Raises ValueError if the text cannot be converted.
        """
        start = float(self.start_edit.text())
        end   = float(self.end_edit.text())
        return start, end
    

# ---------------------
# PyQt5 Application Classes
# ---------------------
class MplCanvas(QtWidgets.QWidget):
    """
    Drop-in canvas replacement using pyqtgraph for the main waveform and FFT plots.
    Provides a GraphicsLayoutWidget with named plot items so existing code using
    self.canvas stays compatible.
    """

    def _style_plot(self, plot, title='', xlabel='', ylabel=''):
        # Base styling only (theme-specific colors are applied by apply_plot_theme)
        plot.setTitle(title)
        plot.setLabel('bottom', xlabel)
        plot.setLabel('left',   ylabel)
        plot.showGrid(x=True, y=True, alpha=0.2)

        # Remove extra padding inside the plot frame
        plot.getViewBox().setContentsMargins(0, 0, 0, 0)
        plot.setContentsMargins(0, 0, 0, 0)

    def refresh_theme(self):
        """Sync pyqtgraph background with the current Qt palette."""
        try:
            pal = self.palette()
            bg = pal.color(QtGui.QPalette.Window).name()
            self.glw.setBackground(bg)
        except Exception:
            pass

    def apply_plot_theme(self, theme: str):
        """Make axes/grid readable in light vs dark themes."""
        theme = (theme or "dark").strip().lower()
        is_light = (theme == "light")

        axis_color = "#000000" if is_light else "#FFFFFF"
        grid_alpha = 0.35 if is_light else 0.2

        for plot in (getattr(self, "plot_waveform", None), getattr(self, "plot_fft", None)):
            if plot is None:
                continue
            try:
                plot.getAxis('bottom').setTextPen(pg.mkPen(axis_color))
                plot.getAxis('left').setTextPen(pg.mkPen(axis_color))
            except Exception:
                pass
            try:
                plot.setTitle(plot.titleLabel.text, color=axis_color)
            except Exception:
                pass
            try:
                plot.showGrid(x=True, y=True, alpha=grid_alpha)
            except Exception:
                pass

    def __init__(self, parent=None, width=8, height=4, dpi=100):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.glw = pg.GraphicsLayoutWidget()
        pal = self.palette()
        bg = pal.color(QtGui.QPalette.Window).name()
        self.glw.setBackground(bg)
        layout.addWidget(self.glw)

        # Single plot (waveform mode)
        self.plot_waveform = self.glw.addPlot(row=0, col=0)
        self._style_plot(self.plot_waveform, 'Raw Waveform', 'Time (s)', 'Amplitude')

        # FFT plot (hidden until FFT mode)
        self.plot_fft = self.glw.addPlot(row=1, col=0)
        self._style_plot(self.plot_fft, 'Spectrum', 'Frequency (Hz)', 'Magnitude')
        self.plot_fft.setVisible(False)

        # Keep pyqtgraph native context menu/actions available on right-click
        for _p in (self.plot_waveform, self.plot_fft):
            try:
                _p.setMenuEnabled(True)
                _p.getViewBox().setMenuEnabled(True)
            except Exception:
                pass
        # Waveform fills all space; FFT row collapses when hidden
        self.glw.ci.layout.setRowStretchFactor(0, 1)
        self.glw.ci.layout.setRowStretchFactor(1, 0)
        # Remove internal padding from the GraphicsLayout container
        self.glw.ci.layout.setContentsMargins(0, 0, 0, 0)
        self.glw.ci.layout.setSpacing(0)

        # Compatibility shim: matplotlib-style .fig attribute
        self.fig = None
        self.ax  = None

        # Linear region item for span selection (replaces SpanSelector)
        self._region = None
        self._region_callback = None

    
def _style_plot(self, plot, title='', xlabel='', ylabel=''):
    # Base styling only (theme-specific colors are applied by apply_plot_theme)
    plot.setTitle(title)
    plot.setLabel('bottom', xlabel)
    plot.setLabel('left',   ylabel)
    plot.showGrid(x=True, y=True, alpha=0.2)

    # Remove extra padding inside the plot frame
    plot.getViewBox().setContentsMargins(0, 0, 0, 0)
    plot.setContentsMargins(0, 0, 0, 0)

def refresh_theme(self):
    """Sync pyqtgraph background with the current Qt palette."""
    try:
        pal = self.palette()
        bg = pal.color(QtGui.QPalette.Window).name()
        self.glw.setBackground(bg)
    except Exception:
        pass

def apply_plot_theme(self, theme: str):
    """Make axes/grid readable in light vs dark themes."""
    theme = (theme or "dark").strip().lower()
    is_light = (theme == "light")

    axis_color = "#000000" if is_light else "#FFFFFF"
    grid_alpha = 0.35 if is_light else 0.2

    for plot in (self.plot_waveform, self.plot_fft):
        if plot is None:
            continue
        try:
            plot.getAxis('bottom').setTextPen(pg.mkPen(axis_color))
            plot.getAxis('left').setTextPen(pg.mkPen(axis_color))
        except Exception:
            pass
        try:
            # Title + labels color (pyqtgraph supports color kw)
            plot.setTitle(plot.titleLabel.text, color=axis_color)
        except Exception:
            pass
        try:
            plot.showGrid(x=True, y=True, alpha=grid_alpha)
        except Exception:
            pass
    def draw(self):
        """Compat stub — pyqtgraph auto-redraws."""
        pass

    def draw_idle(self):
        """Compat stub — pyqtgraph auto-redraws."""
        pass

    def mpl_connect(self, event, callback):
        """Compat stub — pyqtgraph click events are wired separately."""
        pass

    def install_region(self, callback):
        """Install a LinearRegionItem for span selection on the waveform plot."""
        if self._region is not None:
            try:
                self.plot_waveform.removeItem(self._region)
            except Exception:
                pass
        self._region = pg.LinearRegionItem(
            brush=pg.mkBrush(255, 0, 0, 60),
            pen=pg.mkPen('#FF4444', width=1),
        )
        self._region_callback = callback
        self._region.sigRegionChangeFinished.connect(self._on_region_changed)
        self.plot_waveform.addItem(self._region)

    def _on_region_changed(self):
        if self._region_callback:
            xmin, xmax = self._region.getRegion()
            self._region_callback(xmin, xmax)

    def remove_region(self):
        if self._region is not None:
            try:
                self.plot_waveform.removeItem(self._region)
            except Exception:
                pass
            self._region = None
# ---------------------
# Matplotlib canvas for spectrogram tab (keeps full matplotlib API)
# ---------------------
class MplSpecCanvas(FigureCanvas):
    """Matplotlib-based canvas used exclusively for the spectrogram tab."""
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = plt.Figure(figsize=(width, height), dpi=dpi, facecolor="#000000")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#000000")
        self.ax.tick_params(axis="x", colors="white")
        self.ax.tick_params(axis="y", colors="white")
        self.ax.xaxis.label.set_color("white")
        self.ax.yaxis.label.set_color("white")
        for spine in self.ax.spines.values():
            spine.set_edgecolor("white")
        super().__init__(self.fig)
        self.setParent(parent)


# --- Ensure ChannelSelectorDialog is defined early ---
try:
    ChannelSelectorDialog
except NameError:
    from PyQt5 import QtWidgets

    class ChannelSelectorDialog(QtWidgets.QDialog):
        """Configure per-channel acquisition roles and scaling."""

        MODES = [
            ("Raw Voltage", "raw"),
            ("Hydrophone", "hydrophone"),
            ("Voltage Probe", "voltage_probe"),
            ("Current Probe", "current_probe"),
        ]

        def __init__(self, parent, channel_names, mask, configs=None, hydrophone_curves=None):
            super().__init__(parent)
            self.setWindowTitle("Select Channels")
            self.setModal(True)
            self._rows = []
            hydrophone_curves = hydrophone_curves or {}
            configs = configs or []

            layout = QtWidgets.QVBoxLayout(self)
            grid = QtWidgets.QGridLayout()
            grid.addWidget(QtWidgets.QLabel("Use"), 0, 0)
            grid.addWidget(QtWidgets.QLabel("Channel"), 0, 1)
            grid.addWidget(QtWidgets.QLabel("Name"), 0, 2)
            grid.addWidget(QtWidgets.QLabel("Type"), 0, 3)
            grid.addWidget(QtWidgets.QLabel("Hydrophone"), 0, 4)
            grid.addWidget(QtWidgets.QLabel("Distance (m)"), 0, 5)
            grid.addWidget(QtWidgets.QLabel("Depth (m)"), 0, 6)
            grid.addWidget(QtWidgets.QLabel("Scale"), 0, 7)

            for i, name in enumerate(channel_names):
                row_cfg = configs[i] if i < len(configs) else {}
                enabled = mask[i] if i < len(mask) else True

                enable_cb = QtWidgets.QCheckBox()
                enable_cb.setChecked(enabled)

                name_edit = QtWidgets.QLineEdit()
                prior_name = row_cfg.get("name") if isinstance(row_cfg, dict) else None
                name_edit.setText(prior_name or name)
                name_edit.setPlaceholderText(name)

                type_combo = QtWidgets.QComboBox()
                for label, val in self.MODES:
                    type_combo.addItem(label, val)
                initial_mode = row_cfg.get("mode") if isinstance(row_cfg, dict) else None
                if initial_mode:
                    idx = max(0, type_combo.findData(initial_mode))
                    type_combo.setCurrentIndex(idx)

                hydro_combo = QtWidgets.QComboBox()
                hydro_combo.addItem("None", None)
                for curve in hydrophone_curves.values():
                    hydro_combo.addItem(curve.get("curve_name", ""), curve.get("curve_name"))
                if isinstance(row_cfg, dict) and row_cfg.get("hydrophone_curve"):
                    idx = hydro_combo.findData(row_cfg.get("hydrophone_curve"))
                    if idx >= 0:
                        hydro_combo.setCurrentIndex(idx)

                dist_edit = QtWidgets.QLineEdit()
                dist_val = row_cfg.get("distance") if isinstance(row_cfg, dict) else None
                dist_edit.setText("" if dist_val in (None, "") else str(dist_val))
                dist_edit.setPlaceholderText("Optional")

                depth_edit = QtWidgets.QLineEdit()
                depth_val = row_cfg.get("depth") if isinstance(row_cfg, dict) else None
                depth_edit.setText("" if depth_val in (None, "") else str(depth_val))
                depth_edit.setPlaceholderText("Optional")

                scale_edit = QtWidgets.QLineEdit()
                scale_val = row_cfg.get("scale") if isinstance(row_cfg, dict) else None
                scale_edit.setText("" if scale_val in (None, "") else str(scale_val))
                scale_edit.setPlaceholderText("e.g. 100 for 1:100")

                self._rows.append(
                    {
                        "enable": enable_cb,
                        "name": name_edit,
                        "type": type_combo,
                        "hydro": hydro_combo,
                        "dist": dist_edit,
                        "depth": depth_edit,
                        "scale": scale_edit,
                    }
                )

                grid.addWidget(enable_cb, i + 1, 0)
                grid.addWidget(QtWidgets.QLabel(name), i + 1, 1)
                grid.addWidget(name_edit, i + 1, 2)
                grid.addWidget(type_combo, i + 1, 3)
                grid.addWidget(hydro_combo, i + 1, 4)
                grid.addWidget(dist_edit, i + 1, 5)
                grid.addWidget(depth_edit, i + 1, 6)
                grid.addWidget(scale_edit, i + 1, 7)

                def _on_mode_change(idx, row= self._rows[-1]):
                    mode = row["type"].itemData(idx)
                    is_hydro = mode == "hydrophone"
                    is_probe = mode in ("voltage_probe", "current_probe")
                    row["hydro"].setEnabled(is_hydro)
                    row["dist"].setEnabled(is_hydro)
                    row["depth"].setEnabled(is_hydro)
                    row["scale"].setEnabled(is_probe)

                type_combo.currentIndexChanged.connect(_on_mode_change)
                _on_mode_change(type_combo.currentIndex())

            layout.addLayout(grid)
            btns = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
            )
            btns.accepted.connect(self.accept)
            btns.rejected.connect(self.reject)
            layout.addWidget(btns)

        def mask(self):
            return [row["enable"].isChecked() for row in self._rows]

        def names(self):
            names = []
            for idx, row in enumerate(self._rows):
                txt = row["name"].text().strip()
                names.append(txt if txt else f"Ch {idx+1}")
            return names

        def configs(self):
            out = []
            for row in self._rows:
                out.append(
                    {
                        "mode": row["type"].currentData(),
                        "hydrophone_curve": row["hydro"].currentData(),
                        "distance": row["dist"].text(),
                        "depth": row["depth"].text(),
                        "scale": row["scale"].text(),
                        "name": row["name"].text(),
                    }
                )
            return out





class ChartTabAdapter:
    """Compatibility adapter exposing a set_project API for the embedded Chart tab."""
    def __init__(self, main_window):
        self.main_window = main_window

    def set_project(self, project_id=None, project_name=None):
        try:
            self.main_window.current_project_id = project_id
            self.main_window.current_project_name = project_name
        except Exception:
            pass
        if hasattr(self.main_window, "refresh_chart_tracks"):
            self.main_window.refresh_chart_tracks()

class MainWindow(
    QtWidgets.QMainWindow,
    WavFileToolsMixin,
    MeasurementToolsMixin,
    ModellingToolsMixin,
    DetectionToolsMixin,
    DatabaseToolsMixin,
):


    def apply_ui_theme(self, theme_name: str):
        """Apply UI theme ('dark' or 'light') and persist setting."""
        theme = (theme_name or "dark").strip().lower()
        app = QtWidgets.QApplication.instance()
        if app is None:
            return

        try:
            set_setting("ui_theme", theme)
        except Exception:
            pass

        if theme == "light":
            # Soft light (not blinding)
            try:
                app.setStyle("Fusion")
            except Exception:
                pass
            app.setStyleSheet("")
            pal = QtGui.QPalette()
            pal.setColor(QtGui.QPalette.Window, QtGui.QColor("#E9EDF2"))
            pal.setColor(QtGui.QPalette.Base, QtGui.QColor("#F5F7FA"))
            pal.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor("#EEF2F6"))
            pal.setColor(QtGui.QPalette.Text, QtGui.QColor("#1B2430"))
            pal.setColor(QtGui.QPalette.WindowText, QtGui.QColor("#1B2430"))
            pal.setColor(QtGui.QPalette.Button, QtGui.QColor("#E3E8EF"))
            pal.setColor(QtGui.QPalette.ButtonText, QtGui.QColor("#1B2430"))
            pal.setColor(QtGui.QPalette.Highlight, QtGui.QColor(getattr(self, "app_accent_color", "#03DFE2")))
            pal.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#FFFFFF"))
            app.setPalette(pal)
        else:
            # Dark (qdarkstyle)
            try:
                app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
            except Exception:
                pass

        # Sync plot backgrounds + axis colors
        try:
            if hasattr(self, "canvas") and self.canvas is not None:
                if hasattr(self.canvas, "refresh_theme"):
                    self.canvas.refresh_theme()
                if hasattr(self.canvas, "apply_plot_theme"):
                    self.canvas.apply_plot_theme(theme)
        except Exception:
            pass

        # Force redraw
        try:
            self.update_main_waveform_plot()
        except Exception:
            pass
        try:
            self.update_fft_plot()
        except Exception:
            pass

        try:
            self.refresh_chart_theme()
            self._plot_selected_gps_tracks()
        except Exception:
            pass

    def _ensure_main_axis(self):
        """
        Pyqtgraph version — returns the waveform PlotItem.
        Kept for compat with code that calls self._ensure_main_axis().
        """
        if not hasattr(self, 'canvas') or self.canvas is None:
            return None
        self.ax_main = self.canvas.plot_waveform
        return self.ax_main


    def update_main_waveform_plot(self):
        """
        Draw the raw waveform using pyqtgraph on self.canvas.plot_waveform.
        Handles mono and multi-channel (stacked lanes).
        """
        import numpy as np
        if not hasattr(self, 'canvas') or self.canvas is None:
            return
        pw = getattr(self.canvas, 'plot_waveform', None)
        if pw is None:
            return
        pw.clear()

        sr = None
        for k in ('sample_rate', 'samplerate', 'sr'):
            v = getattr(self, k, None)
            if isinstance(v, (int, float)) and v > 0:
                sr = int(v)
                break

        data = getattr(self, 'full_data', None)
        if sr is None or data is None or getattr(data, 'size', 0) == 0:
            pw.setTitle('No audio loaded', color='#FFFFFF')
            return

        # Channel selection
        def _selected():
            if hasattr(self, 'selected_channel_indices') and callable(self.selected_channel_indices):
                sel = self.selected_channel_indices()
                if sel:
                    return [int(x) for x in sel]
            if getattr(data, 'ndim', 1) > 1:
                return list(range(data.shape[1]))
            return [0]

        sel = _selected()
        X = np.asarray(data).reshape(-1, 1) if getattr(data, 'ndim', 1) == 1 else np.asarray(data)
        if X.ndim != 2:
            pw.setTitle('Unsupported data shape', color='#FFFFFF')
            return

        C = X.shape[1]
        sel = [ch for ch in sel if 0 <= ch < C]
        if not sel:
            pw.setTitle('No channels selected', color='#FFFFFF')
            return

        N = X.shape[0]
        t = np.arange(N, dtype=float) / float(sr)

        # Decimate to ~200k points
        max_pts = 200_000
        step = int(np.ceil(max(1, N * max(1, len(sel)) / max_pts)))
        t_plot = t[::step]

        names  = getattr(self, 'channel_names', None)
        multi  = len(sel) > 1
        palette = list((getattr(self, 'color_options', None) or {'Teal': '#03DFE2'}).values())
        base_color = getattr(self, 'graph_color', palette[0])
        try:
            base_idx = palette.index(base_color)
        except ValueError:
            base_idx = 0

        if not multi:
            ch = sel[0]
            y = X[::step, ch]
            theme = get_setting("ui_theme", "dark")
            if str(theme).lower() == "light":
                pen = pg.mkPen(color="#1B2430", width=1.8)
            else:
                pen = pg.mkPen(color=base_color, width=1.5)
            pw.plot(t_plot, y, pen=pen)
            axis_color = '#000000' if str(get_setting('ui_theme','dark')).lower() == 'light' else '#FFFFFF'
            pw.setLabel('left', 'Amplitude', color=axis_color)
        else:
            band_gap = 1.2
            n_sel = len(sel)
            ticks = []
            for idx, ch in enumerate(sel):
                y = X[::step, ch].astype(float)
                y -= np.mean(y)
                rng = float(np.max(y) - np.min(y))
                y_scaled = (y - (np.min(y) + np.max(y)) / 2.0) / (rng or 1.0)
                center = (n_sel - 1 - idx) * band_gap
                label = names[ch] if names and ch < len(names) else f'Ch {ch+1}'
                ticks.append((center, label))
                col = palette[(base_idx + idx) % len(palette)]
                pen = pg.mkPen(color=col, width=1.5)
                pw.plot(t_plot, y_scaled + center, pen=pen)

            ax_left = pw.getAxis('left')
            ax_left.setTicks([ticks])
            axis_color = '#000000' if str(get_setting('ui_theme','dark')).lower() == 'light' else '#FFFFFF'
            pw.setLabel('left', 'Channels', color=axis_color)

        axis_color = '#000000' if str(get_setting('ui_theme','dark')).lower() == 'light' else '#FFFFFF'
        pw.setTitle('Raw Waveform', color=axis_color)
        pw.setLabel('bottom', 'Time (s)', color=axis_color)
        if t_plot.size > 0:
            pw.setXRange(t_plot[0], t_plot[-1], padding=0)
        pw.enableAutoRange(axis='y', enable=True)

        # Restore span region if present
        if getattr(self, '_span_region', None) is not None:
            try:
                pw.addItem(self._span_region)
            except Exception:
                pass


    def _ensure_channel_info(self):
        """Ensure channel count, names, and mask exist."""
        if getattr(self, 'full_data', None) is None:
            return
        try:
            if getattr(self.full_data, 'ndim', 1) > 1:
                self.channels = int(self.full_data.shape[1])
            else:
                self.channels = 1
        except Exception:
            self.channels = 1
        if not hasattr(self, 'channel_names') or len(getattr(self, 'channel_names', [])) != self.channels:
            self.channel_names = [f"Ch {i+1}" for i in range(self.channels)]
        if not hasattr(self, 'channel_mask') or len(getattr(self, 'channel_mask', [])) != self.channels:
            self.channel_mask = [True] * self.channels

        # Keep per-channel configuration aligned with channel count
        if not hasattr(self, "channel_configs") or len(getattr(self, "channel_configs", [])) != self.channels:
            existing = getattr(self, "channel_configs", []) or []
            merged = []
            for i in range(self.channels):
                base = {"mode": "raw", "hydrophone_curve": None, "distance": None, "depth": None, "scale": 1.0, "name": None}
                if i < len(existing) and isinstance(existing[i], dict):
                    base.update({k: v for k, v in existing[i].items() if v is not None})
                merged.append(base)
            self.channel_configs = merged

        # Sync configured names back to channel_names for display
        cfg_names = []
        for i in range(self.channels):
            cfg = self.channel_configs[i] if i < len(self.channel_configs) else {}
            nm = None
            if isinstance(cfg, dict):
                nm = cfg.get("name")
            cfg_names.append(nm if nm else f"Ch {i+1}")
        self.channel_names = cfg_names

    def _default_hydrophone_depth_m(self):
        """Return a single configured hydrophone depth in meters, or None."""
        self._ensure_channel_info()
        depths = []
        for cfg in getattr(self, "channel_configs", []) or []:
            if not isinstance(cfg, dict):
                continue
            if cfg.get("mode") != "hydrophone":
                continue
            val = cfg.get("depth")
            if val in (None, ""):
                continue
            try:
                depths.append(float(val))
            except Exception:
                continue
        if len(depths) == 1:
            return depths[0]
        return None

    def get_channel_data(self, ch_index: int):
        """
        Return a 1-D numpy array for the given channel index.
        Works for mono and multi-channel full_data.
        """
        import numpy as np

        data = getattr(self, "full_data", None)
        if data is None:
            return None

        arr = np.asarray(data)
        if arr.ndim == 1:
            # mono
            return arr if ch_index == 0 else None

        if arr.ndim == 2 and 0 <= ch_index < arr.shape[1]:
            return arr[:, ch_index]

        return None


    def channel_file_label(self, ch_index: int) -> str:
        """
        Build file_name with _chN suffix for logging: myfile_ch1.wav, etc.
        """
        import os
        base_name = getattr(self, "file_name", "") or "measurement"
        base, ext = os.path.splitext(base_name)
        cfg = self._get_channel_config(ch_index) or {}
        alias = cfg.get("name") if isinstance(cfg, dict) else None
        if not alias and hasattr(self, "channel_names") and 0 <= ch_index < len(self.channel_names):
            alias = self.channel_names[ch_index]
        if alias:
            import re
            safe = re.sub(r"[^A-Za-z0-9]+", "_", str(alias)).strip("_")
        else:
            safe = ""
        suffix = f"_ch{ch_index+1}"
        if safe:
            suffix += f"_{safe}"
        return f"{base}{suffix}{ext}"


    def selected_channel_indices(self):
        """Return indices of currently selected channels."""
        self._ensure_channel_info()
        return [i for i, on in enumerate(getattr(self, 'channel_mask', [])) if on]

    def open_channel_selector(self):
        """Show the checkbox dialog; refresh plots on accept."""
        self._ensure_channel_info()
        dlg = ChannelSelectorDialog(
            self,
            getattr(self, 'channel_names', []),
            getattr(self, 'channel_mask', []),
            getattr(self, 'channel_configs', []),
            getattr(self, 'hydrophone_curves', {}),
        )
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.channel_mask = dlg.mask()
            try:
                self.channel_configs = dlg.configs()
                self.channel_names = dlg.names()
            except Exception:
                pass
            if hasattr(self, '_populate_spec_channel_combo'):
                try:
                    self._populate_spec_channel_combo()
                except Exception:
                    pass
            if hasattr(self, 'update_main_waveform_plot'):
                try:
                    self.update_main_waveform_plot()
                except Exception:
                    pass

            if getattr(self, 'fft_mode', False) and hasattr(self, 'update_fft_plot'):
                try:
                    self.update_fft_plot()
                except Exception:
                    pass

    def multi_channel_rms_popup(self):
        """Compute windowed RMS vs time for each selected channel and overlay a plot."""
        if getattr(self, 'full_data', None) is None:
            QtWidgets.QMessageBox.critical(self, "Error", "No file loaded.")
            return
        data2d = self.full_data
        sr = self.sample_rate
        chans = self.selected_channel_indices() if hasattr(self, 'selected_channel_indices') else [0]
        # Config dialog
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Multi‑Channel RMS")
        form = QtWidgets.QFormLayout(dlg)
        win_s = QtWidgets.QDoubleSpinBox(); win_s.setDecimals(3); win_s.setRange(0.001, 60.0); win_s.setValue(0.100)
        step_s = QtWidgets.QDoubleSpinBox(); step_s.setDecimals(3); step_s.setRange(0.001, 60.0); step_s.setValue(0.050)
        form.addRow("Window (s)", win_s)
        form.addRow("Step (s)",   step_s)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        form.addRow(btns)
        btns.accepted.connect(dlg.accept); btns.rejected.connect(dlg.reject)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        W = int(max(1, round(win_s.value()*sr)))
        S = int(max(1, round(step_s.value()*sr)))
        fig = plt.figure(figsize=(9,4))
        ax = fig.add_subplot(111)
        import itertools
        color_cycle = itertools.cycle(["#03DFE2","#C8B6FF","#77DD77","#FFC8A2","#FD8A8A","#FF69B4","#FFFFB5","#6495ED"])
        for ch in chans:
            ch_data = data2d if data2d.ndim==1 else data2d[:, ch]
            n = len(ch_data)
            idxs = list(range(0, n-W+1, S))
            rms = []
            times = []
            for i in idxs:
                seg = ch_data[i:i+W]
                rms.append(float(np.sqrt(np.mean(np.square(seg)))))
                times.append((i + W/2)/sr)
            ax.plot(times, rms, label=f"Ch {ch+1}", linewidth=1.2)
        ax.set_title("Windowed RMS vs Time")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("RMS (V)")
        ax.grid(True, alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.show()

    def __init__(self):
        super().__init__()
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setFocus()
        self.installEventFilter(self)
        self._hub = None
        self._tf  = None

        self.total_frames = 0
        self.samplerate = 0
        self.channels = 0

        # fft params (seconds)
        self.fft_window_length = 1.0   # set your default
        self.fft_start_time = 0.0

        # time scaling for sliders (avoid 32-bit overflow)
        self.TIME_MULTIPLIER = 1000
        #self.load_yamnet() 
        
        init_db()
        self.conn = sqlite3.connect(DB_FILENAME)
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id   INTEGER PRIMARY KEY,
            name TEXT UNIQUE
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS project_items (
            project_id INTEGER,
            file_name  TEXT,
            method     TEXT,
            FOREIGN KEY(project_id) REFERENCES projects(id)
        )
        """)
        self.conn.commit()
        self.setWindowTitle("Waveform Analyzer")
        self.resize(1200, 800)

        # State variables:
        self.full_data = None
        self.full_time = None
        self.decimated_data = None
        self.decimated_time = None
        self.current_project_name = None
        self.sample_rate = None
        self.original_dtype = None
        self.last_region = None
        self.fft_mode = False
        self.fft_start_time = 0.0
        self.fft_window_length = 0.3
        self.file_name = ""
        self.current_file_path = ""
        self.current_project_dir = None
        self.default_data_dir = get_setting("default_data_dir", None)
        self.hydrophone_curves = load_hydrophone_curves()
        self.app_accent_color = "#03DFE2"
        self.graph_color = self.app_accent_color
        self.spectral_method = "FFT"
        self.apply_filter = True
        self.pulse_indices = np.array([])
        self.current_pulse_index = None

        self.setup_ui()

    
        try:
            self.apply_ui_theme(get_setting("ui_theme", "dark"))
        except Exception:
            pass

    def _styled_label(self, text):
        lbl = QtWidgets.QLabel(text)
        lbl.setStyleSheet("color: white;")
        return lbl
    
    def startup_license_check(self):
        hwid = compute_hwid()

        # Try loading an existing license
        try:
            with open(LICENSE_FILE, 'rb') as f:
                lic_blob = f.read()
            lic = json.loads(base64.b64decode(lic_blob))
            if lic['hwid'] != hwid or not verify_signature(lic):
                raise ValueError("Invalid license")
            return  # license is good, continue startup
        except Exception:
            pass  # fall through to request dialog

        # Build modal dialog with NO close button
        dlg = QDialog(self)
        dlg.setWindowTitle("BlackFish Acoustics — License Required")
        dlg.setWindowModality(Qt.ApplicationModal)
        dlg.setWindowFlags(dlg.windowFlags() & ~Qt.WindowCloseButtonHint)

        form = QFormLayout(dlg)
        name_e  = QLineEdit()
        email_e = QLineEdit()
        form.addRow("Name:",      name_e)
        form.addRow("Email:",     email_e)
        form.addRow("Machine ID:", QLineEdit(hwid, readOnly=True))
        req_btn = QPushButton("Generate Request Code")
        form.addRow(req_btn)

        # If they somehow reject (Esc), quit immediately
        dlg.rejected.connect(QApplication.quit)

        def on_request():
            name, email = name_e.text().strip(), email_e.text().strip()
            if not name or not email:
                QMessageBox.warning(dlg, "Missing", "Please enter name and email.")
                return

            payload = {
                "name":    name,
                "email":   email,
                "hwid":    hwid,
                "reason":  "no-license" if not os.path.exists(LICENSE_FILE) else "bad-license",
                "version": __version__,  # make sure __version__ is defined at module top
                "ts":      datetime.now(timezone.utc).isoformat()
            }
            blob = base64.b64encode(json.dumps(payload).encode("utf8"))
            with open(REQUEST_FILE, "wb") as reqf:
                reqf.write(blob)

            QMessageBox.information(
                dlg,
                "Request Generated",
                f"Please email {REQUEST_FILE} to sales@blackfishacoustics.com.\n\n"
                "The application will now close."
            )

            dlg.accept()
            QApplication.quit()
            sys.exit(0)

        req_btn.clicked.connect(on_request)

        # Block here—no way around it
        dlg.exec_()

        # Fallback guard
        QApplication.quit()
        sys.exit(1)
    
    
    def nf_axial_piston_delta_db(f_Hz, radius_m, range_m, c=1500.0):
        """
        Axial baffled-piston near-field factor:
            F = | 2*sin(psi/2) / psi | ,  psi = k*a^2/(2R),  k = 2*pi*f/c
        Returns ΔSPL_NF in dB to SUBTRACT from measured SPL at R.
        """
        f = np.asarray(f_Hz, dtype=float)
        k = 2.0*np.pi*f/float(c)
        a = float(radius_m)
        R = float(range_m)
        psi = k*(a*a)/(2.0*R)
        F = np.ones_like(psi, dtype=float)
        nz = np.abs(psi) > 1e-12
        F[nz] = np.abs(2.0*np.sin(psi[nz]*0.5)/psi[nz])
        F = np.maximum(F, 1e-9)  # numeric safety
        return 20.0*np.log10(F)

    def _ctd_get_c_ms_at_depth(ctd_id, depth_m):
        """
        Return (c_ms, depth_used, source_str) from ctd_profiles at nearest depth to depth_m.
        Requires ctd_profiles table (present in your app).
        """
        try:
            # Reuse your DB path logic if available; else assume DB_FILENAME in scope.
            path = DB_FILENAME if 'DB_FILENAME' in globals() else os.path.join(os.path.abspath(os.getcwd()), "analyze_qt.db")
            conn = sqlite3.connect(path); cur = conn.cursor()
            cur.execute("SELECT depth_json, sound_speed_json, name, dt_utc FROM ctd_profiles WHERE id=?", (ctd_id,))
            row = cur.fetchone(); conn.close()
        except Exception:
            row = None
        if not row or row[0] is None or row[1] is None:
            return None, None, None
        depth = np.array(json.loads(row[0]), dtype=float)
        c_ms  = np.array(json.loads(row[1]), dtype=float)
        if depth.size == 0 or c_ms.size == 0:
            return None, None, None
        d = float(max(depth_m, 0.0))
        idx = int(np.argmin(np.abs(depth - d)))
        return float(c_ms[idx]), float(depth[idx]), f"ctd:{int(ctd_id)}@{float(depth[idx]):.2f}m"

    def _ensure_spl_nf_columns():
        table = "measurements"

        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        for tbl in ("spl_calculations", "spl_archive"):
            cur.execute(f"PRAGMA table_info({tbl})")
            cols = [r[1] for r in cur.fetchall()]
            def _add(col, typ):
                if col not in cols:
                    cur.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} {typ}")
            _add("distance",      "REAL")    # if not already present
            _add("spl_nf",        "REAL")    # SPL with NF correction
            _add("nf_enabled",    "INTEGER") # 0/1
            _add("nf_delta_db",   "REAL")    # ΔSPL_NF applied (dB)
            _add("nf_radius_m",   "REAL")    # piston radius a (m)
            _add("nf_range_m",    "REAL")    # hydrophone range R (m)
            _add("nf_c_ms",       "REAL")    # sound speed used (m/s)
            _add("nf_c_source",   "TEXT")    # "manual" or "ctd:<id>@<depth>m"
            _add("nf_c_depth_m",  "REAL")    # CTD depth used (m)
            _add("nf_ctd_id",     "INTEGER") # reference to ctd_profiles.id
        conn.commit()
        conn.close()

    # Call this once during app init (e.g., in MainWindow.__init__ or before SPL UI opens)
    try:
        _ensure_spl_nf_columns()
    except Exception as _e:
        # non-fatal; keeps older DBs working
        pass


    def load_yamnet(self):  # Make sure this is NOT nested inside __init__
        try:
            print("Loading YAMNet...")
            self.yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
            print("YAMNet loaded successfully.")
        except Exception as e:
            print("YAMNet loading failed:", e)
            self.yamnet_model = None

    def show_help(self):
        help_text = load_help_text()
        help_dialog = QtWidgets.QDialog(self)
        help_dialog.setWindowTitle("Help")
        help_dialog.resize(600, 600)
        layout = QtWidgets.QVBoxLayout(help_dialog)
        text_edit = QtWidgets.QTextEdit()
        text_edit.setText(help_text)
        text_edit.setReadOnly(True)
        layout.addWidget(text_edit)
        help_dialog.exec_()
    
    def import_curve(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Hydrophone Curve File",
            "",
            "Text Files (*.txt);;All Files (*)"
        )
        if not file_path:
            return

        curve_name, ok = QtWidgets.QInputDialog.getText(
            self,
            "Curve Name",
            "Enter a name for this hydrophone curve:"
        )
        if not ok or not curve_name:
            return

        try:
            min_freq, max_freq, sens_list = import_hydrophone_curve_file(file_path, curve_name)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Error",
                f"Failed to import hydrophone curve:\n{e}"
            )
            return

        save_hydrophone_curve(curve_name, os.path.basename(file_path), min_freq, max_freq, sens_list)
        QtWidgets.QMessageBox.information(self, "Success", "Hydrophone curve imported successfully.")

        # Refresh the combo
        self.hydrophone_curves = load_hydrophone_curves()
        if self.hydrophone_combo is not None:
            self.hydrophone_combo.clear()
            self.hydrophone_combo.addItem("None")
        for curve in self.hydrophone_curves.values():
            if self.hydrophone_combo is not None:
                self.hydrophone_combo.addItem(curve["curve_name"])
            
    def on_log_item_changed(self, item):
        # Called when a cell in the “Logs” table is edited
        if not hasattr(self, 'log_table') or self.log_table.signalsBlocked():
            return

        row, col = item.row(), item.column()
        id_item = self.log_table.item(row, 0)
        if not id_item:
            return
        entry_id = int(id_item.text())

        cols = [
            "id", "file_name", "method", "target_frequency", "start_time", "end_time",
            "window_length", "max_voltage", "bandwidth", "measured_voltage",
            "filter_applied", "screenshot", "misc", "timestamp"
        ]
        db_col = cols[col]

        # Don’t allow editing of primary key or timestamp
        if db_col in ("id", "timestamp"):
            return

        new_text = item.text().strip()

        # 1) “None” → store SQL NULL
        if new_text.lower() == 'none':
            new_val = None

        # 2) purely numeric columns **(exclude ‘misc’ here!)**
        elif db_col in (
            "target_frequency", "start_time", "end_time",
            "window_length", "max_voltage", "bandwidth",
            "measured_voltage"
        ):
            try:
                new_val = float(new_text)
            except ValueError:
                QtWidgets.QMessageBox.warning(self, "Error", f"Invalid number for {db_col}.")
                return

        # 3) boolean flag
        elif db_col == "filter_applied":
            try:
                new_val = int(new_text)
            except ValueError:
                QtWidgets.QMessageBox.warning(self, "Error", f"Invalid integer for {db_col}.")
                return

        # 4) everything else (including misc) is free‐form text
        else:
            new_val = new_text

        # Update the database
        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        cur.execute(f"UPDATE measurements SET {db_col} = ? WHERE id = ?", (new_val, entry_id))
        conn.commit()
        conn.close()

    def refresh_spl_log(self):
        """Compatibility shim: always use paginated view."""
        self.populate_spl_table()

    def on_tab_changed(self, index):
        tab_name = self.tabs.tabText(index)
        if tab_name == "Logs":
            self.request_logs_refresh()
        elif tab_name == "Advanced":
            QtCore.QTimer.singleShot(0, self.update_advanced_plots)
        elif tab_name == "SPL":
            # OLD: QtCore.QTimer.singleShot(0, self.refresh_spl_log)
            QtCore.QTimer.singleShot(0, self.populate_spl_table)  # ← paginated


    def _normalize_debounce_ms(self, debounce_ms):
        """Return a safe integer delay for log refresh scheduling."""

        try:
            return int(debounce_ms)
        except (TypeError, ValueError):
            return 120

    def _on_log_sigfigs_changed(self, value):
        """Update the significant-figure preference and refresh the current page."""

        try:
            sig_figs = max(1, int(value))
        except (TypeError, ValueError):
            sig_figs = 3

        self.log_sig_figs = sig_figs
        # No need to rebuild filters—just re-render the page with new formatting.
        self.request_logs_refresh(immediate=True, refresh_filters=False)

    def _format_log_value(self, value):
        """Format numeric values with the configured significant figures."""

        if value is None:
            return ""

        sig_figs = getattr(self, "log_sig_figs", 3) or 3

        try:
            num = float(value)
            if math.isfinite(num):
                return f"{num:.{sig_figs}g}"
        except (TypeError, ValueError):
            pass

        return str(value)

    def request_logs_refresh(self, debounce_ms=120, immediate=False, refresh_filters=True, *_args, **_kwargs):
        """Debounce log refreshes to avoid synchronous UI stalls.

        All refreshes are funneled through a single QTimer so they always occur
        after returning to the event loop, avoiding synchronous work while
        switching tabs or completing measurements.
        """

        # Guard against signals sending string payloads (e.g., currentTextChanged)
        debounce_ms_int = self._normalize_debounce_ms(debounce_ms)

        # Remember whether this refresh needs to rebuild the filter combos. If any
        # queued request asks for filters, keep that requirement until the refresh
        # runs so paging-only calls can remain light-weight.
        refresh_filters_flag = bool(refresh_filters)
        pending_flag = getattr(self, "_logs_refresh_refresh_filters", None)
        if pending_flag is None:
            self._logs_refresh_refresh_filters = refresh_filters_flag
        else:
            self._logs_refresh_refresh_filters = pending_flag or refresh_filters_flag

        force_immediate = bool(immediate) or debounce_ms_int <= 0
        timer = self._ensure_logs_refresh_timer()

        # If a refresh is already running, remember to reschedule once it
        # completes instead of piling more synchronous work onto the UI thread.
        if getattr(self, "_logs_refresh_running", False):
            setattr(self, "_logs_refresh_reschedule_needed", True)
            if force_immediate:
                setattr(self, "_logs_refresh_reschedule_immediate", True)
            return

        # If a timer is already queued and no immediate run is required, leave
        # it alone; otherwise, coalesce bursts by restarting with the shortest
        # delay requested so far.
        if timer.isActive():
            if force_immediate:
                timer.start(0)
            return

        setattr(self, "_logs_refresh_pending", True)

        delay = 0 if force_immediate else debounce_ms_int
        timer.start(delay)

    def _ensure_logs_refresh_timer(self):
        """Create the debounced refresh timer lazily if needed."""

        timer = getattr(self, "_logs_refresh_timer", None)
        if timer is None:
            timer = QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.setInterval(120)
            timer.timeout.connect(self._perform_logs_refresh)
            self._logs_refresh_timer = timer

        return timer

    def _ensure_logs_indexes(self):
        """Create indexes that keep log queries snappy under project filters."""

        if getattr(self, "_logs_indexes_ready", False):
            return

        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()

        cur.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_measurements_file_method
                ON measurements(file_name, method);
            CREATE INDEX IF NOT EXISTS idx_measurements_file_method_id
                ON measurements(file_name, method, id);
            CREATE INDEX IF NOT EXISTS idx_archive_file_method
                ON archive(file_name, method);
            CREATE INDEX IF NOT EXISTS idx_archive_file_method_id
                ON archive(file_name, method, id);
            CREATE INDEX IF NOT EXISTS idx_project_items_file_method
                ON project_items(file_name, method);
            CREATE INDEX IF NOT EXISTS idx_project_items_project
                ON project_items(project_id);
            CREATE INDEX IF NOT EXISTS idx_project_items_project_file_method
                ON project_items(project_id, file_name, method);
            CREATE INDEX IF NOT EXISTS idx_projects_name
                ON projects(name);
            """
        )

        conn.commit()
        conn.close()

        self._logs_indexes_ready = True

    def _get_project_id(self, project_name):
        """Resolve a project name to its id using the indexed projects table."""

        if not project_name:
            return None

        cache = getattr(self, "_project_id_cache", None)
        if cache is None:
            cache = {}
            self._project_id_cache = cache

        if project_name in cache:
            return cache[project_name]

        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        cur.execute("SELECT id FROM projects WHERE name = ?", (project_name,))
        row = cur.fetchone()
        conn.close()

        pid = row[0] if row else None
        cache[project_name] = pid
        return pid

    def _perform_logs_refresh(self):
        # If the Logs UI has been destroyed, bail out quietly
        if not hasattr(self, "log_table"):
            return

        # Clear the pending flag and mark the refresh as running to prevent
        # re-entrancy while we update the UI.
        if hasattr(self, "_logs_refresh_pending"):
            self._logs_refresh_pending = False
        self._logs_refresh_running = True

        self.log_table.setUpdatesEnabled(False)
        try:
            refresh_filters_flag = getattr(self, "_logs_refresh_refresh_filters", True)
            self._logs_refresh_refresh_filters = False

            self._ensure_logs_indexes()
            if refresh_filters_flag:
                self.update_log_filter_options()
                self.update_method_filter_options()
            self.populate_log_table()
        finally:
            self.log_table.setUpdatesEnabled(True)
            self.log_table.viewport().update()
            self._logs_refresh_running = False

            # If another request arrived while we were refreshing, schedule a
            # follow-up pass using the most urgent timing requested.
            if getattr(self, "_logs_refresh_reschedule_needed", False):
                immediate = getattr(self, "_logs_refresh_reschedule_immediate", False)
                self._logs_refresh_reschedule_needed = False
                self._logs_refresh_reschedule_immediate = False
                self.request_logs_refresh(immediate=immediate)

    def refresh_projects(self):
        """
        Refresh all project-related drop-downs from the projects table.
        Safely ignores any widgets that have been deleted.
        """
        cur = self.conn.cursor()
        cur.execute("SELECT name FROM projects ORDER BY name")
        names = [n for (n,) in cur.fetchall()]

        # Stats tab project pickers
        if hasattr(self, "proj_cb") and self.proj_cb is not None:
            try:
                self.proj_cb.blockSignals(True)
                self.proj_cb.clear()
                for n in names:
                    self.proj_cb.addItem(n)
                self.proj_cb.blockSignals(False)
            except RuntimeError:
                self.proj_cb = None  # C++ side was deleted

        if hasattr(self, "matrix_proj_cb") and self.matrix_proj_cb is not None:
            try:
                self.matrix_proj_cb.blockSignals(True)
                self.matrix_proj_cb.clear()
                for n in names:
                    self.matrix_proj_cb.addItem(n)
                self.matrix_proj_cb.blockSignals(False)
            except RuntimeError:
                self.matrix_proj_cb = None

        # Top-level project selector next to "Select File"
        if hasattr(self, "project_combo") and self.project_combo is not None:
            try:
                current = self.project_combo.currentText()
                self.project_combo.blockSignals(True)
                self.project_combo.clear()
                self.project_combo.addItem("(No project)")
                for n in names:
                    self.project_combo.addItem(n)
                self.project_combo.addItem("➕ Add project…")
                idx = self.project_combo.findText(current)
                if idx < 0:
                    idx = 0
                self.project_combo.setCurrentIndex(idx)
                self.project_combo.blockSignals(False)
            except RuntimeError:
                self.project_combo = None

        # Logs tab project filter
        if hasattr(self, "log_project_cb") and self.log_project_cb is not None:
            try:
                current = self.log_project_cb.currentText()
                self.log_project_cb.blockSignals(True)
                self.log_project_cb.clear()
                self.log_project_cb.addItem("All Projects")
                for n in names:
                    self.log_project_cb.addItem(n)
                idx = self.log_project_cb.findText(current)
                if idx < 0:
                    idx = 0
                self.log_project_cb.setCurrentIndex(idx)
                self.log_project_cb.blockSignals(False)
            except RuntimeError:
                self.log_project_cb = None

    def on_project_changed(self, index: int):
        """
        Handler for the top-level project dropdown next to 'Select File'.
        Keeps Stats, Projects, and Logs project selectors in sync.
        """
        # If the combobox itself is gone, bail out
        if not hasattr(self, "project_combo") or self.project_combo is None:
            return

        try:
            text = self.project_combo.currentText()
        except RuntimeError:
            # Underlying C++ object was deleted
            self.project_combo = None
            return

        # ─────────────────────────────
        # "Add project…" special case
        # ─────────────────────────────
        if text.startswith("➕"):
            name, ok = QtWidgets.QInputDialog.getText(
                self, "New Project", "Enter new project name:"
            )
            if ok and name.strip():
                name = name.strip()
                try:
                    cur = self.conn.cursor()
                    cur.execute("INSERT INTO projects (name) VALUES (?)", (name,))
                    self.conn.commit()
                except sqlite3.IntegrityError:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Duplicate Project",
                        "A project with that name already exists."
                    )

                # Refresh all project combos and select the new one
                self.refresh_projects()
                if self.project_combo is not None:
                    try:
                        idx = self.project_combo.findText(name)
                        if idx >= 0:
                            self.project_combo.setCurrentIndex(idx)
                    except RuntimeError:
                        self.project_combo = None
            else:
                # User cancelled: revert to previous selection or "(No project)"
                if self.project_combo is not None:
                    try:
                        self.project_combo.blockSignals(True)
                        fallback = self.current_project_name or "(No project)"
                        idx = self.project_combo.findText(fallback)
                        if idx < 0:
                            idx = 0
                        self.project_combo.setCurrentIndex(idx)
                        self.project_combo.blockSignals(False)
                    except RuntimeError:
                        self.project_combo = None
            return  # done handling the special entry

        # ─────────────────────────────
        # Normal selection
        # ─────────────────────────────
        if text == "(No project)":
            self.current_project_name = None
        else:
            self.current_project_name = text
        self.current_project_id = self._get_project_id(self.current_project_name)

        # Keep Stats tab project combo in sync
        if hasattr(self, "proj_cb") and self.proj_cb is not None:
            try:
                i = self.proj_cb.findText(text)
                if i >= 0:
                    self.proj_cb.setCurrentIndex(i)
            except RuntimeError:
                self.proj_cb = None

        # Keep Projects matrix tab project combo in sync
        if hasattr(self, "matrix_proj_cb") and self.matrix_proj_cb is not None:
            try:
                i = self.matrix_proj_cb.findText(text)
                if i >= 0:
                    self.matrix_proj_cb.setCurrentIndex(i)
            except RuntimeError:
                self.matrix_proj_cb = None

        # Keep Logs tab project filter in sync
        if hasattr(self, "log_project_cb") and self.log_project_cb is not None:
            try:
                target = "All Projects" if self.current_project_name is None else self.current_project_name
                i = self.log_project_cb.findText(target)
                if i >= 0:
                    blocker = QtCore.QSignalBlocker(self.log_project_cb)
                    self.log_project_cb.setCurrentIndex(i)
                    del blocker
            except RuntimeError:
                self.log_project_cb = None

        # Ensure a storage folder exists for the active project
        self._ensure_project_directory()

        # If a file is already loaded, mirror it into the project folder
        if getattr(self, "current_file_path", None):
            self._cache_file_into_project(self.current_file_path)

        # Refresh logs when project changes
        try:
            if hasattr(self, "request_logs_refresh"):
                self.request_logs_refresh()
        except RuntimeError:
            # If any logs widgets have been destroyed, just ignore
            pass

        # Refresh SPL tab filters/table when project changes
        try:
            if hasattr(self, "spl_table"):
                self.update_spl_filter_options()
                self.update_spl_method_filter_options()
                self.spl_page = 1
                self.populate_spl_table()
        except RuntimeError:
            pass

        if hasattr(self, "_refresh_spectrogram_gallery"):
            try:
                self._refresh_spectrogram_gallery()
            except Exception:
                pass

        if hasattr(self, "refresh_chart_tracks"):
            try:
                self.refresh_chart_tracks()
            except Exception:
                pass

        if hasattr(self, "gps_ctd_tab") and self.gps_ctd_tab is not None:
            try:
                self.gps_ctd_tab.set_project(self.current_project_id, self.current_project_name)
            except Exception:
                pass


    def _attach_measurement_to_current_project(self, file_name: str, method: str):
        """
        If a project is selected, ensure (file_name, method) is linked in project_items.
        """
        if not getattr(self, "current_project_name", None):
            return

        pid = getattr(self, "current_project_id", None)
        if pid is None:
            cur = self.conn.cursor()
            cur.execute("SELECT id FROM projects WHERE name=?", (self.current_project_name,))
            row = cur.fetchone()
            if not row:
                return
            pid = row[0]
            self.current_project_id = pid

        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR IGNORE INTO project_items (project_id, file_name, method)
            VALUES (?, ?, ?)
            """,
            (pid, file_name, method),
        )
        self.conn.commit()

    def _update_default_dir_display(self):
        if hasattr(self, "default_dir_edit"):
            text = self.default_data_dir or "(not set)"
            self.default_dir_edit.setText(text)

    def choose_default_data_dir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Default Data Folder",
            self.default_data_dir or os.path.expanduser("~"),
        )
        if not path:
            return

        self.default_data_dir = path
        set_setting("default_data_dir", path)
        self._update_default_dir_display()
        self._ensure_project_directory()

    def _ensure_project_directory(self):
        base = self.default_data_dir
        proj = getattr(self, "current_project_name", None)
        if not base or not proj:
            self.current_project_dir = None
            return None

        safe_proj = re.sub(r"[^\w\-\. ]+", "_", proj).strip() or "project"
        project_dir = os.path.join(base, safe_proj)
        os.makedirs(project_dir, exist_ok=True)

        for sub in ("originals", "spectrograms", "screenshots", "modified", "exports"):
            os.makedirs(os.path.join(project_dir, sub), exist_ok=True)

        self.current_project_dir = project_dir
        return project_dir

    def _project_subdir(self, name=None):
        base = self._ensure_project_directory()
        if not base:
            return None
        if not name:
            return base
        path = os.path.join(base, name)
        os.makedirs(path, exist_ok=True)
        return path

    def _dialog_default_dir(self, category=None):
        if category:
            path = self._project_subdir(category)
            if path:
                return path
        if self.default_data_dir:
            os.makedirs(self.default_data_dir, exist_ok=True)
            return self.default_data_dir
        return ""

    def _cache_file_into_project(self, filepath):
        proj_dir = self._ensure_project_directory()
        if not proj_dir or not filepath:
            return filepath

        originals_dir = self._project_subdir("originals")
        try:
            os.makedirs(originals_dir, exist_ok=True)
            dest = os.path.join(originals_dir, os.path.basename(filepath))
            if not os.path.exists(dest):
                shutil.copy2(filepath, dest)
            self.current_file_path = dest
            return dest
        except Exception:
            return filepath

    def _coerce_float(self, value, default=None):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _get_selected_hydrophone_curve(self):
        """Return the active hydrophone curve info, or None if unselected."""
        combo = getattr(self, "hydrophone_combo", None)
        if combo is None:
            return None

        try:
            choice = combo.currentText()
        except Exception:
            return None

        if not choice or choice.strip() == "None":
            return None

        for info in getattr(self, "hydrophone_curves", {}).values():
            if info.get("curve_name") == choice:
                return info
        return None

    def _get_hydrophone_curve_by_name(self, curve_name):
        if not curve_name:
            return None
        for info in getattr(self, "hydrophone_curves", {}).values():
            if info.get("curve_name") == curve_name:
                return info
        return None

    def _channel_index_from_file_name(self, file_name):
        try:
            match = re.search(r"_ch(\d+)", file_name or "", re.IGNORECASE)
            if match:
                idx = int(match.group(1)) - 1
                return idx if idx >= 0 else None
        except Exception:
            return None
        return None

    def _get_channel_config(self, channel_index):
        if channel_index is None:
            return None
        cfgs = getattr(self, "channel_configs", None)
        if not cfgs:
            return None
        if 0 <= channel_index < len(cfgs):
            return cfgs[channel_index]
        return None

    def _normalize_channel_index(self, channel_index, file_name=None):
        if channel_index is not None:
            return channel_index
        inferred = self._channel_index_from_file_name(file_name)
        if inferred is not None:
            return inferred
        if getattr(self, "channels", 0):
            return 0
        return None

    def _prepare_channel_measurement(self, measured_voltage, misc, channel_index):
        cfg = self._get_channel_config(channel_index) or {}
        mode = cfg.get("mode") or "raw"

        # Parse incoming voltage for scaling; fall back to raw if parsing fails
        parsed_voltage = self._coerce_float(measured_voltage, measured_voltage)
        misc_out = misc
        hydro_curve_name = None
        distance = None

        if mode == "hydrophone":
            hydro_curve_name = cfg.get("hydrophone_curve") or None
            if not hydro_curve_name:
                combo_curve = self._get_selected_hydrophone_curve()
                if combo_curve:
                    hydro_curve_name = combo_curve.get("curve_name")
            distance = self._coerce_float(cfg.get("distance"))
            if misc_out in (None, ""):
                misc_out = f"Hydrophone: {hydro_curve_name}" if hydro_curve_name else "Hydrophone"
            # Surface the entered distance in the misc column for traceability
            if distance is not None:
                tag = f"Distance: {distance:g} m"
                misc_out = tag if misc_out in (None, "") else f"{misc_out} | {tag}"
        elif mode in ("voltage_probe", "current_probe"):
            scale = self._coerce_float(cfg.get("scale"), 1.0) or 1.0
            if isinstance(parsed_voltage, (int, float)):
                parsed_voltage = parsed_voltage * scale
            label = "Voltage" if mode == "voltage_probe" else "Current"
            if misc_out in (None, ""):
                misc_out = f"{label} probe x{scale:g}"
        else:
            # Allow the legacy hydrophone combo to act as a fallback when no per-channel hydrophone is configured
            combo_curve = self._get_selected_hydrophone_curve()
            if combo_curve:
                hydro_curve_name = combo_curve.get("curve_name")

        return parsed_voltage, misc_out, hydro_curve_name, distance

    def _estimate_dominant_frequency(self, start_time, end_time, window_length, channel_index=None):
        """
        Estimate dominant frequency (highest-energy FFT peak) from the selected time window.
        Uses currently-loaded waveform self.full_data. Supports multichannel arrays.
        If start/end are missing, uses the last window_length seconds.
        """

        # --- get sample rate ---
        try:
            fs = float(getattr(self, "wav_sample_rate", None) or getattr(self, "sample_rate", None) or 0)
        except Exception:
            return None
        if fs <= 0:
            return None

        # --- get waveform from currently loaded data ---
        try:
            x = self.full_data
        except Exception:
            x = None
        if x is None:
            return None

        xa = np.asarray(x)

        # --- pick channel if multichannel ---
        if xa.ndim == 2:
            ch = 0 if channel_index is None else int(channel_index)
            ch = max(0, min(ch, xa.shape[1] - 1)) if xa.shape[0] >= xa.shape[1] else max(0, min(ch, xa.shape[0] - 1))

            # assume shape (N, C) when N >= C, else (C, N)
            sig = xa[:, ch] if xa.shape[0] >= xa.shape[1] else xa[ch, :]
        elif xa.ndim == 1:
            sig = xa
        else:
            sig = xa.reshape(-1)

        if sig is None or len(sig) < 32:
            return None

        # --- parse window times ---
        try:
            t0 = float(start_time) if start_time is not None else None
            t1 = float(end_time) if end_time is not None else None
        except Exception:
            t0, t1 = None, None

        # fallback window length
        try:
            win_s = float(window_length) if window_length is not None else 1.0
        except Exception:
            win_s = 1.0
        if win_s <= 0:
            win_s = 1.0

        # If no explicit start/end, use last win_s seconds
        if t0 is None or t1 is None:
            total_s = len(sig) / fs
            t1 = total_s
            t0 = max(0.0, t1 - win_s)

        if t1 <= t0:
            return None

        i0 = int(max(0, t0 * fs))
        i1 = int(min(len(sig), t1 * fs))
        seg = np.asarray(sig[i0:i1], dtype=np.float64)

        if seg.size < 64:
            return None

        # remove DC
        seg = seg - np.mean(seg)

        # FFT (windowed)
        win = np.hanning(seg.size)
        y = seg * win
        Y = np.fft.rfft(y)
        mag = np.abs(Y)
        freqs = np.fft.rfftfreq(len(y), d=1.0 / fs)

        if mag.size < 2:
            return None

        mag[0] = 0.0  # ignore DC
        k = int(np.argmax(mag))
        dom_f = float(freqs[k])

        if not np.isfinite(dom_f) or dom_f <= 0:
            return None

        return dom_f

    def _compute_dominant_freq_from_buffer(self, x, fs, fmin=0.0, fmax=None):
        """Compute dominant frequency from a buffer via FFT peak. Returns None on failure."""
        try:
            xa = np.asarray(x)
            if xa.ndim == 2:
                xa = xa[:, 0] if xa.shape[0] >= xa.shape[1] else xa[0, :]
            xa = xa.reshape(-1)
            if xa.size < 64:
                return None
            fs = float(fs)
            if not np.isfinite(fs) or fs <= 0:
                return None
            xa = xa.astype(np.float64, copy=False)
            xa = xa - np.mean(xa)
            win = np.hanning(xa.size)
            y = xa * win
            Y = np.fft.rfft(y)
            mag = np.abs(Y)
            freqs = np.fft.rfftfreq(y.size, d=1.0 / fs)
            if mag.size < 2:
                return None
            mag[0] = 0.0
            if fmax is None:
                fmax = fs / 2.0
            mask = (freqs >= float(fmin)) & (freqs <= float(fmax))
            if mask.sum() >= 2:
                k = int(np.argmax(mag[mask]))
                dom_f = float(freqs[mask][k])
            else:
                dom_f = float(freqs[int(np.argmax(mag))])
            if not np.isfinite(dom_f) or dom_f <= 0:
                return None
            return dom_f
        except Exception:
            return None







    def _log_spl_for_measurement(
        self,
        voltage_log_id,
        file_name,
        target_frequency,
        measured_voltage,
        start_time,
        end_time,
        window_length,
        max_voltage,
        bandwidth,
        screenshot,
        hydrophone_curve_name=None,
        distance=None,
        channel_index=None,
    ):
        """If a hydrophone is selected, log SPL alongside the voltage entry."""
        if voltage_log_id is None:
            return

        curve = self._get_hydrophone_curve_by_name(hydrophone_curve_name) or self._get_selected_hydrophone_curve()
        if not curve:
            return

        try:
            vrms = float(measured_voltage)
        except (TypeError, ValueError):
            return
        # Determine frequency for curve lookup / logging.
        # If "Auto freq (FFT)" is enabled, derive from dominant FFT peak in the selected window.
        # Determine frequency for curve lookup / logging
        freq = None
        try:
            auto_on = bool(getattr(self, "auto_freq_cb", None) and self.auto_freq_cb.isChecked())
        except Exception:
            auto_on = False

        if auto_on:
            try:
                freq = self._estimate_dominant_frequency(start_time, end_time, window_length, channel_index=channel_index)
            except Exception:
                freq = None

        if freq is None:
            try:
                freq = float(target_frequency)
            except (TypeError, ValueError):
                return

        # IMPORTANT: make the computed freq become the logged “target_frequency”
        target_frequency = freq

        try:
            self.log(
                f"[SPL AUTO FREQ] auto={auto_on}  freq_used={freq:.2f} Hz  "
                f"start={start_time} end={end_time} win={window_length}  file={file_name}"
            )
        except Exception:
            pass


        try:
            auto_on = bool(getattr(self, "auto_freq_cb", None) and self.auto_freq_cb.isChecked())
        except Exception:
            auto_on = False

        if auto_on:
            try:
                freq = self._estimate_dominant_frequency(start_time, end_time, window_length, channel_index=channel_index)
            except Exception:
                freq = None

        if freq is None:
            try:
                freq = float(target_frequency)
            except (TypeError, ValueError):
                return

        sens_list = curve.get("sensitivity") or []
        if not sens_list:
            return

        min_f = int(curve.get("min_freq", 0) or 0)
        idx = int(round(freq)) - min_f
        idx = max(0, min(idx, len(sens_list) - 1))
        try:
            sensitivity_db = float(sens_list[idx])
        except (TypeError, ValueError):
            return

        spl_val = 20.0 * np.log10(max(vrms, 1e-12)) - sensitivity_db

        # Apply distance correction if provided (convert measured level at d m to 1 m equivalent)
        if distance is not None:
            try:
                d = float(distance)
                if d > 0:
                    spl_val += 20.0 * np.log10(d)
            except (TypeError, ValueError):
                pass

        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO spl_calculations (
                file_name, voltage_log_id, hydrophone_curve,
                target_frequency, rms_voltage, spl, spl_nf,
                start_time, end_time, window_length,
                max_voltage, bandwidth, screenshot, distance,
                nf_enabled, nf_delta_db, nf_radius_m, nf_range_m, nf_c_ms,
                nf_c_source, nf_c_depth_m, nf_ctd_id
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                file_name,
                int(voltage_log_id),
                curve.get("curve_name"),
                freq,
                vrms,
                spl_val,
                spl_val,
                None if start_time is None else float(start_time),
                None if end_time is None else float(end_time),
                None if window_length is None else float(window_length),
                None if max_voltage is None else float(max_voltage),
                None if bandwidth is None else float(bandwidth),
                screenshot or "",
                None if distance is None else float(distance),
                0,
                0.0,
                0.0,
                0.0,
                0.0,
                "",
                None,
                None,
            ),
        )
        conn.commit()
        conn.close()

    def log_measurement_with_project(
        self,
        file_name,
        method,
        target_frequency,
        start_time,
        end_time,
        window_length,
        max_voltage,
        bandwidth,
        measured_voltage,
        filter_applied,
        screenshot,
        misc=None,
        channel_index=None,
    ):
        """
        Wrapper around log_measurement() that:

        1. Requires a project to be selected.
        2. Logs the measurement to the DB.
        3. Attaches (file_name, method) to the current project in project_items.
        """

        # 1) Enforce project selection
        if not getattr(self, "current_project_name", None):
            QtWidgets.QMessageBox.warning(
                self,
                "No Project Selected",
                "Please select a project in the 'Project' dropdown next to "
                "'Select File' before taking measurements.",
            )
            return  # do NOT log anything

        ch_idx = self._normalize_channel_index(channel_index, file_name)
        processed_voltage, misc_out, hydro_curve_name, distance = self._prepare_channel_measurement(
            measured_voltage, misc, ch_idx
        )

        # 2) Log the measurement as usual
        entry_id = log_measurement(
            file_name,
            method,
            target_frequency,
            start_time,
            end_time,
            window_length,
            max_voltage,
            bandwidth,
            processed_voltage,
            filter_applied,
            screenshot,
            misc_out,
        )

        # 3) Attach this (file, method) to the active project
        self._attach_measurement_to_current_project(file_name, method)

        # 4) If a hydrophone is selected, log SPL with the voltage entry
        self._log_spl_for_measurement(
            entry_id,
            file_name,
            target_frequency,
            processed_voltage,
            start_time,
            end_time,
            window_length,
            max_voltage,
            bandwidth,
            screenshot,
            hydrophone_curve_name=hydro_curve_name,
            distance=distance,
            channel_index=ch_idx,
        )

        return entry_id






    def manage_projects(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Manage Projects")
        dlg.resize(800, 500)
        layout = QtWidgets.QVBoxLayout(dlg)

        # Top: Create / select project
        top_bar = QtWidgets.QHBoxLayout()
        top_bar.addWidget(QtWidgets.QLabel("Project:"))
        proj_combo = QtWidgets.QComboBox()
        proj_combo.setMinimumWidth(200)
        top_bar.addWidget(proj_combo)
        new_proj_btn = QtWidgets.QPushButton("New Project…")
        top_bar.addWidget(new_proj_btn)
        top_bar.addStretch()
        layout.addLayout(top_bar)

        # Middle: file lists
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        all_list = QtWidgets.QListWidget()
        all_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        all_list.setStyleSheet("color: white; background-color: #2b2b2b;")
        proj_list = QtWidgets.QListWidget()
        proj_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        proj_list.setStyleSheet("color: white; background-color: #2b2b2b;")
        split.addWidget(all_list)
        split.addWidget(proj_list)
        layout.addWidget(split)

        # Bottom: add/remove/close buttons
        btn_row = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("Add →")
        remove_btn = QtWidgets.QPushButton("← Remove")
        close_btn = QtWidgets.QPushButton("Close")
        btn_row.addWidget(add_btn)
        btn_row.addWidget(remove_btn)
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        # Populate project list
        cur = self.conn.cursor()
        cur.execute("SELECT name FROM projects ORDER BY name")
        projects = [r[0] for r in cur.fetchall()]
        proj_combo.addItems(projects)

        def refresh_lists():
            proj_name = proj_combo.currentText()
            if not proj_name:
                return

            cur.execute("SELECT id FROM projects WHERE name=?", (proj_name,))
            row = cur.fetchone()
            if not row:
                return
            pid = row[0]

            # Get all measurement files
            cur.execute("SELECT DISTINCT file_name, method FROM measurements")
            all_items = set(cur.fetchall())

            # Get items in current project
            cur.execute("SELECT file_name, method FROM project_items WHERE project_id=?", (pid,))
            proj_items = set(cur.fetchall())

            remaining_items = all_items - proj_items

            all_list.clear()
            proj_list.clear()

            for fn, m in sorted(remaining_items):
                all_list.addItem(f"{fn} | {m}")
            for fn, m in sorted(proj_items):
                proj_list.addItem(f"{fn} | {m}")

        proj_combo.currentIndexChanged.connect(refresh_lists)

        def on_add():
            proj_name = proj_combo.currentText()
            if not proj_name:
                return
            cur.execute("SELECT id FROM projects WHERE name=?", (proj_name,))
            pid = cur.fetchone()[0]
            for item in all_list.selectedItems():
                fn, m = item.text().split(" | ")
                cur.execute("INSERT INTO project_items (project_id, file_name, method) VALUES (?, ?, ?)", (pid, fn, m))
            self.conn.commit()
            refresh_lists()

        def on_remove():
            proj_name = proj_combo.currentText()
            if not proj_name:
                return
            cur.execute("SELECT id FROM projects WHERE name=?", (proj_name,))
            pid = cur.fetchone()[0]
            for item in proj_list.selectedItems():
                fn, m = item.text().split(" | ")
                cur.execute("DELETE FROM project_items WHERE project_id=? AND file_name=? AND method=?", (pid, fn, m))
            self.conn.commit()
            refresh_lists()

        def on_new_project():
            name, ok = QtWidgets.QInputDialog.getText(dlg, "New Project", "Enter new project name:")
            if ok and name:
                try:
                    cur.execute("INSERT INTO projects (name) VALUES (?)", (name,))
                    self.conn.commit()
                    proj_combo.addItem(name)
                    proj_combo.setCurrentText(name)
                except sqlite3.IntegrityError:
                    QtWidgets.QMessageBox.warning(dlg, "Duplicate", "Project name already exists.")

        new_proj_btn.clicked.connect(on_new_project)
        add_btn.clicked.connect(on_add)
        remove_btn.clicked.connect(on_remove)
        close_btn.clicked.connect(dlg.accept)

        refresh_lists()
        dlg.exec_()
        self.refresh_projects()



    def setup_ui(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        main_layout.setSpacing(8)
        self.loaded_file = None

        # ────────────────────────────────────────────────────────────────────────
        # Top controls row: [ Select File ] [Category combo] [Tool combo]
        #                   [Graph Color combo] [Help button]
        # ── Row 1: File, folder, project, channels, tools, graph colour ─────────
        row1 = QtWidgets.QHBoxLayout()
        row1.setSpacing(6)
        main_layout.addLayout(row1)

        self.open_button = QtWidgets.QPushButton("Select File")
        self.open_button.clicked.connect(self.open_file)
        self.open_button.setStyleSheet(
          "background-color: #03DFE2;"
            "color: black;"
            "border-radius: 4px;"
            "padding: 4px 8px;"
        )
        row1.addWidget(self.open_button)

        row1.addWidget(QtWidgets.QLabel("Data Folder:"))
        self.default_dir_edit = QtWidgets.QLineEdit()
        self.default_dir_edit.setReadOnly(True)
        self.default_dir_edit.setFixedWidth(240)
        self._update_default_dir_display()
        row1.addWidget(self.default_dir_edit)
        self.default_dir_btn = QtWidgets.QPushButton("Default Folder…")
        self.default_dir_btn.clicked.connect(self.choose_default_data_dir)
        row1.addWidget(self.default_dir_btn)

        row1.addWidget(QtWidgets.QLabel("Project:"))
        self.project_combo = QtWidgets.QComboBox()
        self.project_combo.setMinimumWidth(180)
        self.project_combo.addItem("(No project)")
        self.project_combo.addItem("➕ Add project…")
        self.project_combo.currentIndexChanged.connect(self.on_project_changed)
        row1.addWidget(self.project_combo)

        self.channels_btn = QtWidgets.QPushButton("Channels…")
        self.channels_btn.setToolTip("Choose which channels to display and configure sensor types")
        self.channels_btn.clicked.connect(self.open_channel_selector)
        row1.addWidget(self.channels_btn)

        self.category_combo = QtWidgets.QComboBox()
        self.category_combo.addItems([
            "WAV File Tools",
            "Measurement Tools",
            "Modelling & Plotting Tools",
            "Detection & Classification Tools",
            "Database Tools"
        ])
        self.category_combo.currentIndexChanged.connect(self.update_tool_list)
        self.category_combo.setFixedWidth(200)
        row1.addWidget(self.category_combo)

        self.tool_combo = QtWidgets.QComboBox()
        self.tool_combo.setFixedWidth(250)
        self.tool_combo.currentIndexChanged.connect(self.on_tool_selected)
        row1.addWidget(self.tool_combo)
        self.update_tool_list(0)

        self.color_combo = QtWidgets.QComboBox()
        self.color_options = {
            "Blue": "#A8C8FF", "Green": "#AAFFC3", "Purple": "#DDD0FF",
            "Orange": "#FFD9B5", "Pink": "#FFB3D9", "Red": "#FFAAAA",
            "Teal": "#6FECEF", "White": "#FFFFFF", "Yellow": "#FFFF99"
        }
        for key in self.color_options:
            self.color_combo.addItem(key)
        self.color_combo.setCurrentText("Teal")
        self.color_combo.currentIndexChanged.connect(self.change_color)
        self.color_combo.setFixedWidth(100)
        row1.addWidget(QtWidgets.QLabel("Graph Color:"))
        row1.addWidget(self.color_combo)

        row1.addStretch()

        # Help button on row 1 right side
        self.help_button = QtWidgets.QPushButton("Help")
        self.help_button.clicked.connect(self.show_help)
        row1.addWidget(self.help_button)



        # ── Channel mode compat attribute ──────────────────────────────────────
        self.channel_mode = None  # removed from UI
        self.channel_picker = QtWidgets.QListWidget()
        self.channel_picker.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.channel_picker.setMaximumHeight(80)
        self.channel_picker.setStyleSheet(
            "QListWidget{background:#111;color:white;border:1px solid #333;}"
        )
        # Put the picker under the top controls for compactness
        main_layout.addWidget(self.channel_picker)
        # Hide inline channel picker; use the popup dialog instead
        try:
            self.channel_picker.setVisible(False)
            self.channel_picker.setMaximumHeight(0)
        except Exception:
            pass

        # ────────────────────────────────────────────────────────────────────────
        # Tabs: Analysis / SPL / Spectrogram / Logs  (Advanced is currently unused)
        # ────────────────────────────────────────────────────────────────────────
        self.tabs = QtWidgets.QTabWidget()
        main_layout.addWidget(self.tabs)

        # --- Analysis Tab -----------------------------------------------------
        self.analysis_tab = QtWidgets.QWidget()
        self.setup_analysis_tab()
        self.tabs.addTab(self.analysis_tab, "Analysis")

        # --- SPL Tab ----------------------------------------------------------
        self.spl_tab = QtWidgets.QWidget()
        self.setup_spl_tab()
        self.tabs.addTab(self.spl_tab, "SPL")

        # --- Spectrogram Tab -------------------------------------------------
        self.spectrogram_tab = QtWidgets.QWidget()
        self.setup_spectrogram_tab()
        self.tabs.addTab(self.spectrogram_tab, "Spectrogram")

        # --- Chart Tab --------------------------------------------------------
        self.chart_tab = QtWidgets.QWidget()
        self.setup_chart_tab()
        self.tabs.addTab(self.chart_tab, "Chart")
        self.gps_ctd_tab = ChartTabAdapter(self)

        # --- Logs Tab ---------------------------------------------------------
        self.logs_tab = QtWidgets.QWidget()
        self.setup_logs_tab()
        self.tabs.addTab(self.logs_tab, "Logs")

        self.tabs.currentChanged.connect(self.on_tab_changed)

        # --- Summary & Stats Tab removed per request ---
        self.proj_cb = None
        self.stat_cb = None
        self.plot_cb = None
        self.stats_table = None
        self.stats_fig = None
        self.stats_canvas = None

        # Defensive cleanup in case a stale build path added the old tab
        try:
            for _i in range(self.tabs.count() - 1, -1, -1):
                _label = self.tabs.tabText(_i) or ""
                if "summary" in _label.lower() and "stats" in _label.lower():
                    self.tabs.removeTab(_i)
        except Exception:
            pass

        # --- Matrix Tab ---------------------------------------------------------
        matrix_tab = QtWidgets.QWidget()
        matrix_layout = QtWidgets.QVBoxLayout(matrix_tab)

        # Top control bar
        top_controls = QtWidgets.QHBoxLayout()
        top_controls.addWidget(QtWidgets.QLabel("Project:"))
        self.matrix_proj_cb = QtWidgets.QComboBox()
        top_controls.addWidget(self.matrix_proj_cb)
        matrix_manage_btn = QtWidgets.QPushButton("Manage Projects…")
        matrix_manage_btn.clicked.connect(self.manage_projects)
        top_controls.addWidget(matrix_manage_btn)
        refresh_btn = QtWidgets.QPushButton("Refresh Matrix")
        refresh_btn.clicked.connect(self.build_file_measurement_matrix)
        top_controls.addWidget(refresh_btn)
        top_controls.addStretch()
        matrix_layout.addLayout(top_controls)

        # Matrix table
        self.matrix_table = QtWidgets.QTableWidget()
        matrix_layout.addWidget(self.matrix_table)

        # Add to main tabs
        self.tabs.addTab(matrix_tab, "Projects")
        # populate the project list
        self.refresh_projects()


# --- Below, still inside MainWindow class ----------------------------------

    
    def compute_stats(self):
        import scipy.stats as ss
        cur = self.conn.cursor()
        proj = self.proj_cb.currentText()
        if not proj:
            QtWidgets.QMessageBox.warning(self, "No Project", "Please select a project first.")
            return
        cur.execute("SELECT id FROM projects WHERE name=?", (proj,))
        row = cur.fetchone()
        if not row:
            QtWidgets.QMessageBox.information(self, "Empty Project", "That project has no items.")
            return
        pid = row[0]
        cur.execute("SELECT file_name, method FROM project_items WHERE project_id=?", (pid,))
        items = cur.fetchall()
        if not items:
            QtWidgets.QMessageBox.information(self, "Empty Project", "No files/methods in this project.")
            return

        # build WHERE clause without leading AND
        clauses = []
        params = []
        for fn, m in items:
            clauses.append("(file_name=? AND method=?)")
            params.extend([fn, m])
        where_expr = " OR ".join(clauses)
        sql = ("SELECT measured_voltage FROM measurements "
                + "WHERE " + where_expr)
        cur.execute(sql, params)
        rows = cur.fetchall()
        if not rows:
            QtWidgets.QMessageBox.information(self, "No Data", "No entries for that project.")
            return

        arr = np.array([r[0] for r in rows], dtype=float)
        # compute stats
        stats = {
            "Mean": arr.mean(),
            "Median": np.median(arr),
            "Min": arr.min(),
            "Max": arr.max(),
            "Count": len(arr)
        }
        # populate stats table
        self.stats_table.setRowCount(len(stats))
        for i, (k, v) in enumerate(stats.items()):
            self.stats_table.setItem(i, 0, QtWidgets.QTableWidgetItem(k))
            self.stats_table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{v:.3f}"))
        self.stats_table.resizeColumnsToContents()

        # plotting section follows unchanged...
        choice = self.plot_cb.currentText()
        self.stats_fig.clear()
        ax = self.stats_fig.add_subplot(111, facecolor='#19232D')
        ax.tick_params(colors='white')
        color = getattr(self, 'graphColorDropdown', None).currentText() if hasattr(self,'graphColorDropdown') else '#03DFE2'
        # ... rest of plotting logic ...
        self.stats_canvas.draw()

    def build_file_measurement_matrix(self):
        cur = self.conn.cursor()
        
        # 1. Get all files associated with the current project
        proj = self.proj_cb.currentText()
        if not proj:
            QtWidgets.QMessageBox.warning(self, "No Project", "Select a project first.")
            return
        cur.execute("SELECT id FROM projects WHERE name=?", (proj,))
        row = cur.fetchone()
        if not row:
            QtWidgets.QMessageBox.information(self, "No Project", "Invalid project selected.")
            return
        pid = row[0]
        cur.execute("SELECT DISTINCT file_name FROM project_items WHERE project_id=?", (pid,))
        file_names = [r[0] for r in cur.fetchall()]
        if not file_names:
            QtWidgets.QMessageBox.information(self, "Empty", "No files in this project.")
            return

        # 2. Get all distinct methods from measurements
        cur.execute("SELECT DISTINCT method FROM measurements")
        methods = [
            "Auto Analysis", "Waveform", "Ambient Noise", "Peak Prominences Analysis", "Interval Analysis", "Depth Sounder Analysis",
            "Slope De-Clipper", "Find Peaks", "Short-Time RMS", "Crest Factor", "Octave-Band Analysis",
            "SNR Estimator", "LFM Analysis", "LFM Batch Analysis", "HFM Analysis", "Multi Frequency Analysis",
            "SPL Transmit Analysis", "Hydrophone Calibration", "Duty Cycle Analysis",
            "Active Sonar", "Cepstrum Analysis", "Event Clustering"
        ]


        # 3. Build table
        self.matrix_table.clear()
        self.matrix_table.setRowCount(len(file_names))
        self.matrix_table.setColumnCount(len(methods) + 1)  # +1 for "File Name"
        self.matrix_table.setHorizontalHeaderLabels(["File Name"] + methods)

        for row, fname in enumerate(file_names):
            # File name cell
            item = QtWidgets.QTableWidgetItem(fname)
            item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            self.matrix_table.setItem(row, 0, item)

            for col, method in enumerate(methods, start=1):
                # Check for measurements in measurements table
                cur.execute("SELECT COUNT(*) FROM measurements WHERE file_name=? AND method=?", (fname, method))
                count1 = cur.fetchone()[0]

                # Check for SPL-specific methods in spl_calculations
                cur.execute("SELECT COUNT(*) FROM measurements WHERE file_name=? AND method=?", (fname, method))
                count2 = cur.fetchone()[0]

                count = count1 + count2

                exists = count > 0
                btn = QtWidgets.QPushButton("View" if exists else "Run")

                # Style the button based on whether results exist
                if exists:
                    color_hex = self.color_options.get(self.color_combo.currentText(), "#03DFE2")
                    btn.setStyleSheet(f"""
                        QPushButton {{
                            background-color: {color_hex};
                            color: black;
                            font-weight: bold;
                            border-radius: 4px;
                        }}
                    """)
                else:
                    btn.setStyleSheet("""
                        QPushButton {
                            background-color: #555;
                            color: white;
                            border-radius: 4px;
                        }
                    """)

                btn.clicked.connect(lambda _, f=fname, m=method, exists=exists: self._handle_matrix_button(f, m, exists))
                self.matrix_table.setCellWidget(row, col, btn)

        self.matrix_table.resizeColumnsToContents()

    def _handle_matrix_button(self, file_name, method, exists):
        if exists:
            self.view_measurement_results(file_name, method)
        else:
            popup_map = {
                "Ambient Noise": self.ambient_noise_analysis,
                "Peak Prominences Analysis": self.peak_prominences_popup,
                "Interval Analysis": self.interval_analysis_popup,
                "Depth Sounder Analysis": self.depth_sounder_popup,
                "Slope De-Clipper": self.slope_declipper_popup,
                "Find Peaks": self.find_peaks_analysis,
                "Short-Time RMS": self.short_time_rms_popup,
                "Octave-Band Analysis": self.octave_band_analysis_popup,
                "SNR Estimator": self.snr_estimator_popup,
                "LFM Analysis": self.lfm_pulse_analysis,
                "LFM Batch Analysis": self.lfm_pulse_batch_analysis,
                "HFM Analysis": self.hfm_pulse_analysis,
                "Multi Frequency Analysis": self.multi_freq_analysis,
                "SPL Transmit Analysis": self.spl_from_voltage_popup,
                "Hydrophone Calibration": self.generate_hydrophone_calibration_popup,
                "Duty Cycle Analysis": self.duty_cycle_analysis_popup,
                "Active Sonar": self.active_sonar_popup,
                "Cepstrum Analysis": self.cepstrum_analysis
            }

            if method not in popup_map:
                QtWidgets.QMessageBox.information(self, "Unsupported", f"No tool popup defined for: {method}")
                return

            # If file is already loaded, skip prompt
            
            if self.loaded_file and os.path.basename(self.loaded_file) == file_name:
                popup_map[method]()
                return

            # Ask user to locate the file
            QtWidgets.QMessageBox.information(self, "File Required",
                f"The analysis for '{method}' requires loading '{file_name}'. Please locate it.")
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, f"Locate {file_name}", "", "WAV Files (*.wav);;All Files (*)")
            if not path:
                return

            # Load the file
            try:
                self.loaded_file(path)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load file:\n{e}")
                return

            # Run tool
            popup_map[method]()

    def _load_file_by_name(self, fname):
        search_roots = []
        proj_root = self._project_subdir("originals")
        if proj_root:
            search_roots.append(proj_root)
        if self.default_data_dir:
            search_roots.append(self.default_data_dir)
        search_roots.append(".")

        for root in search_roots:
            for walk_root, _, files in os.walk(root):
                if fname in files:
                    full_path = os.path.join(walk_root, fname)
                    self.load_file(full_path)
                    return
        QtWidgets.QMessageBox.warning(self, "File Not Found", f"Cannot locate: {fname}")

    def view_measurement_results(self, file_name, method):
        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        cur.execute("""
            SELECT target_frequency, start_time, end_time, window_length,
                max_voltage, bandwidth, measured_voltage, misc, timestamp
            FROM measurements
            WHERE file_name=? AND method=?
            ORDER BY timestamp ASC
        """, (file_name, method))
        rows = cur.fetchall()
        conn.close()

        if not rows:
            QtWidgets.QMessageBox.information(self, "No Results", "No data available.")
            return

        # ─── Main Window ──────────────────────────────────────────────────────
        win = QtWidgets.QMainWindow(self)
        win.setWindowTitle(f"{method} — {file_name}")
        win.resize(1000, 700)
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        win.setCentralWidget(central)

        # ─── Control Buttons ──────────────────────────────────────────────────
        top_controls = QtWidgets.QHBoxLayout()

        toggle_btn = QtWidgets.QPushButton("Toggle Markers")
        logscale_cb = QtWidgets.QCheckBox("Log Scale (Y)")

        save_btn = QtWidgets.QPushButton("Save Graph…")
        export_btn = QtWidgets.QPushButton("Export CSV…")

        mode_group = QtWidgets.QButtonGroup(win)
        mode_time = QtWidgets.QRadioButton("Time vs Value")
        mode_freq = QtWidgets.QRadioButton("Freq vs Value")
        mode_time.setChecked(True)
        mode_group.addButton(mode_time)
        mode_group.addButton(mode_freq)

        for w in (toggle_btn, mode_time, mode_freq, logscale_cb, save_btn, export_btn):
            top_controls.addWidget(w)
        top_controls.addStretch()
        layout.addLayout(top_controls)

        # ─── Graph Area ───────────────────────────────────────────────────────
        fig = Figure(facecolor='#19232D')
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        layout.addWidget(canvas)

        # ─── Extract & Prepare Data ───────────────────────────────────────────
        freqs = [r[0] for r in rows]
        times = [r[1] for r in rows]
        y_vals = [r[6] if r[6] is not None else r[7] for r in rows]

        color_hex = self.color_options.get(self.color_combo.currentText(), "#03DFE2")
        plot_obj = None
        markers_on = True

        def update_plot():
            ax.clear()
            ax.set_facecolor("#000")
            ax.tick_params(colors='white')
            ax.set_title(f"{method} Results", color="white")
            ax.set_ylabel("Measured Value", color="white")
            ax.set_yscale("log" if logscale_cb.isChecked() else "linear")
            for spine in ax.spines.values():
                spine.set_edgecolor("white")

            x_vals = times if mode_time.isChecked() else freqs
            x_label = "Start Time (s)" if mode_time.isChecked() else "Frequency (Hz)"
            ax.set_xlabel(x_label, color="white")

            nonlocal plot_obj
            marker = 'o' if markers_on else ''
            plot_obj = ax.plot(x_vals, y_vals, color=color_hex, linestyle='-', marker=marker)[0]
            canvas.draw()

        update_plot()

        toggle_btn.clicked.connect(lambda: toggle_marker())
        logscale_cb.toggled.connect(update_plot)
        mode_time.toggled.connect(update_plot)
        mode_freq.toggled.connect(update_plot)

        def toggle_marker():
            nonlocal markers_on
            markers_on = not markers_on
            update_plot()

        # ─── Save Graph Handler ──────────────────────────────────────────────
        def save_graph():
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                win, "Save Graph as JPEG", "", "JPEG Files (*.jpg *.jpeg)"
            )
            if path:
                fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
                QtWidgets.QMessageBox.information(win, "Saved", f"Graph saved to:\n{path}")

        save_btn.clicked.connect(save_graph)

        # ─── Export CSV Handler ──────────────────────────────────────────────
        def export_csv():
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                win, "Export CSV", "", "CSV Files (*.csv)"
            )
            if path:
                with open(path, "w", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "Target Freq", "Start", "End", "Window", "Max V",
                        "Bandwidth", "Measured", "Misc", "Timestamp"
                    ])
                    for row in rows:
                        writer.writerow(row)
                QtWidgets.QMessageBox.information(win, "Exported", f"CSV saved to:\n{path}")

        export_btn.clicked.connect(export_csv)

        # ─── Table Area ──────────────────────────────────────────────────────
        tbl = QtWidgets.QTableWidget(len(rows), 9)
        tbl.setHorizontalHeaderLabels([
            "Target Freq", "Start", "End", "Window", "Max V",
            "Bandwidth", "Measured", "Misc", "Timestamp"
        ])
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                item = QtWidgets.QTableWidgetItem(f"{val:.6f}" if isinstance(val, float) else str(val))
                tbl.setItem(r, c, item)
        tbl.resizeColumnsToContents()
        layout.addWidget(tbl)

        win.show()






    def octave_band_analysis_popup(self):
        """
        Octave-Band Analysis (multi-channel, project-aware).

        • Requires a project selection.
        • For each selected channel:
            - Computes Welch PSD on that channel (in volts)
            - Integrates PSD over each band (1-octave or 1/3-octave) to get VRMS
            - Converts VRMS → dB (20*log10(VRMS)) stored in misc
        • Plot:
            - Fullscreen-style dialog with a grouped bar chart:
              each band has one bar per selected channel
        • Logging:
            - One measurement per channel per band with _chN suffix in file_name
            - Screenshot of the multi-channel bar plot saved and attached
        • Extras:
            - Save Graph… (JPG / PNG)
            - Data table of band values
            - Export CSV of table
        """
        import numpy as np
        import os
        import csv
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from scipy.signal import welch

        # --- Require project selection ---
        if not getattr(self, "current_project_name", None):
            QtWidgets.QMessageBox.warning(
                self, "Project Required",
                "Please select a Project before running Octave-Band Analysis."
            )
            return

        # --- Basic checks ---
        if getattr(self, "full_data", None) is None:
            QtWidgets.QMessageBox.critical(self, "Error", "Load a WAV file first.")
            return
        if not hasattr(self, "sample_rate") or self.sample_rate <= 0:
            QtWidgets.QMessageBox.critical(self, "Error", "Invalid sample rate.")
            return

        # === 1) Band type dialog ===
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Octave-Band Analysis")
        layout = QtWidgets.QVBoxLayout(dlg)

        form = QtWidgets.QFormLayout()
        bw_combo = QtWidgets.QComboBox()
        bw_combo.addItems(["1-Octave", "1/3-Octave"])
        form.addRow("Band Type:", bw_combo)
        layout.addLayout(form)

        btn_row = QtWidgets.QHBoxLayout()
        compute_btn = QtWidgets.QPushButton("Compute")
        cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_row.addStretch()
        btn_row.addWidget(compute_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        params = {}

        def on_compute():
            params["band_type"] = bw_combo.currentText()
            dlg.accept()

        compute_btn.clicked.connect(on_compute)
        cancel_btn.clicked.connect(dlg.reject)

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        band_type = params["band_type"]

        # === 2) Prepare data as (N, C) and convert to volts ===
        data = np.asarray(self.full_data)
        sr = float(self.sample_rate)

        if data.ndim == 1:
            data_2d = data.reshape(-1, 1)
        else:
            data_2d = data
        n_samples, n_ch = data_2d.shape

        # Counts → volts using max_voltage_entry and original_dtype
        if np.issubdtype(self.original_dtype, np.integer):
            try:
                vmax = float(self.max_voltage_entry.text())
            except Exception:
                vmax = 1.0
            conv = vmax / np.iinfo(self.original_dtype).max
        else:
            conv = 1.0
            vmax = 0.0

        data_2d = data_2d.astype(np.float64) * conv

        # Selected channels
        if hasattr(self, "selected_channel_indices") and callable(self.selected_channel_indices):
            sel_channels = self.selected_channel_indices()
        else:
            sel_channels = list(range(n_ch))
        sel_channels = [ch for ch in sel_channels if 0 <= ch < n_ch]
        if not sel_channels:
            sel_channels = [0]

        # === 3) Build center frequencies ===
        nyq = sr / 2.0
        centers = []
        if band_type == "1-Octave":
            f = 10.0
            while f <= nyq:
                centers.append(f)
                f *= 2.0
        else:  # 1/3-Octave
            f = 10.0
            step = 2 ** (1.0 / 3.0)
            while f <= nyq:
                centers.append(f)
                f *= step

        if not centers:
            QtWidgets.QMessageBox.information(
                self, "Octave-Band Analysis",
                "No octave bands fall below Nyquist."
            )
            return

        centers = np.array(centers, dtype=float)
        band_low = centers / (2.0 ** 0.5)
        band_high = centers * (2.0 ** 0.5)

        # === 4) Compute Welch PSD per channel and band VRMS ===
        vrms_by_ch = {}
        db_by_ch = {}

        nperseg = min(1024, n_samples if n_samples > 0 else 1024)

        for ch in sel_channels:
            x = data_2d[:, ch]
            freqs, Pxx = welch(x, fs=sr, nperseg=nperseg)

            ch_vrms = []
            ch_db = []

            for f0 in centers:
                f_low = f0 / (2 ** 0.5)
                f_high = f0 * (2 ** 0.5)
                mask = (freqs >= f_low) & (freqs <= f_high)

                if mask.any():
                    area = float(np.trapz(Pxx[mask], freqs[mask]))
                    if area > 0.0:
                        vrms = float(np.sqrt(area))
                        lvl_db = float(20.0 * np.log10(vrms))
                    else:
                        vrms = float("nan")
                        lvl_db = float("nan")
                else:
                    vrms = float("nan")
                    lvl_db = float("nan")

                ch_vrms.append(vrms)
                ch_db.append(lvl_db)

            vrms_by_ch[ch] = np.array(ch_vrms, dtype=float)
            db_by_ch[ch] = np.array(ch_db, dtype=float)

        # === 5) Build fullscreen-style results dialog with grouped bars ===
        fig, ax = plt.subplots(facecolor="#19232D")
        ax.set_facecolor("#000000")

        # X positions for bands
        num_bands = len(centers)
        x = np.arange(num_bands)

        # Grouped bar width
        num_ch = len(sel_channels)
        bar_group_width = 0.8
        bar_width = bar_group_width / max(1, num_ch)

        # Colors from your palette
        color_keys = list(self.color_options.keys())
        color_vals = [self.color_options[k] for k in color_keys]
        num_colors = len(color_vals) if color_vals else 1

        for idx, ch in enumerate(sel_channels):
            offset = (idx - (num_ch - 1) / 2.0) * bar_width
            col = color_vals[idx % num_colors] if color_vals else self.graph_color
            ax.bar(
                x + offset,
                vrms_by_ch[ch],
                width=bar_width * 0.95,
                label=f"Ch {ch+1}",
                color=col,
            )

        # Axis labels and formatting
        band_labels = [f"{int(round(f))}" for f in centers]
        ax.set_xticks(x)
        ax.set_xticklabels(band_labels, rotation=45, ha="right", color="white")

        ax.set_title(
            f"{band_type} Band VRMS (All Selected Channels)",
            color="white",
        )
        ax.set_xlabel("Center Frequency (Hz)", color="white")
        ax.set_ylabel("VRMS (V)", color="white")
        ax.tick_params(colors="white")

        for spine in ax.spines.values():
            spine.set_edgecolor("white")

        if num_ch > 1:
            leg = ax.legend()
            for txt in leg.get_texts():
                txt.set_color("white")

        canvas = FigureCanvas(fig)

        # === 6) Results dialog (graph + table + buttons) ===
        win = QtWidgets.QDialog(self)
        win.setWindowTitle("Octave-Band Results")
        vbox = QtWidgets.QVBoxLayout(win)
        vbox.addWidget(canvas)

        # Data table label
        table_label = QtWidgets.QLabel("Octave-band values (per channel):")
        table_label.setStyleSheet("color: white; font-weight: bold;")
        vbox.addWidget(table_label)

        # Data table
        table = QtWidgets.QTableWidget()
        num_cols = 3 + 2 * num_ch  # center, low, high + (VRMS, dB) per channel
        table.setRowCount(num_bands)
        table.setColumnCount(num_cols)

        headers = ["Center Hz", "Low Hz", "High Hz"]
        for idx, ch in enumerate(sel_channels):
            headers.append(f"Ch {ch+1} VRMS (V)")
            headers.append(f"Ch {ch+1} dB re 1 V")
        table.setHorizontalHeaderLabels(headers)

        # Fill table
        for i in range(num_bands):
            table.setItem(i, 0, QtWidgets.QTableWidgetItem(f"{centers[i]:.3f}"))
            table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{band_low[i]:.3f}"))
            table.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{band_high[i]:.3f}"))
            col = 3
            for ch in sel_channels:
                v = vrms_by_ch[ch][i]
                d = db_by_ch[ch][i]
                v_str = "" if not np.isfinite(v) else f"{v:.6g}"
                d_str = "" if not np.isfinite(d) else f"{d:.3f}"
                table.setItem(i, col,   QtWidgets.QTableWidgetItem(v_str))
                table.setItem(i, col+1, QtWidgets.QTableWidgetItem(d_str))
                col += 2

        table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch
        )
        table.setStyleSheet(
            "QTableWidget { background-color: #202020; color: white; gridline-color: #404040; }"
            "QHeaderView::section { background-color: #333333; color: white; }"
        )
        vbox.addWidget(table)

        # Buttons
        btns = QtWidgets.QHBoxLayout()
        btns.addStretch()
        save_graph_btn = QtWidgets.QPushButton("Save Graph…")
        export_csv_btn = QtWidgets.QPushButton("Export CSV")
        save_store_btn = QtWidgets.QPushButton("Save & Store")
        close_btn = QtWidgets.QPushButton("Close")
        for b in (save_graph_btn, export_csv_btn, save_store_btn, close_btn):
            btns.addWidget(b)
        vbox.addLayout(btns)

        # Make the dialog large / fullscreen-ish
        screen = QtWidgets.QApplication.desktop().availableGeometry(win)
        win.resize(int(screen.width() * 0.9), int(screen.height() * 0.9))
        win.showMaximized()

        screenshot_path = ""

        # --- Save Graph to JPG/PNG ---
        def on_save_graph():
            nonlocal screenshot_path
            base_path = getattr(self, "wav_file_path", "") or getattr(
                self, "current_file_path", ""
            ) or ""
            if not base_path:
                base_path = self.file_name or "octave_output"

            base_dir = os.path.dirname(base_path)
            if not base_dir:
                base_dir = os.getcwd()

            root_name = os.path.splitext(os.path.basename(base_path))[0]
            default_path = os.path.join(base_dir, f"{root_name}_octaveVRMS_multi.jpg")

            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                win,
                "Save Graph",
                default_path,
                "JPEG Files (*.jpg *.jpeg);;PNG Files (*.png);;All Files (*)",
            )
            if not path:
                return

            try:
                fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
                screenshot_path = path
                QtWidgets.QMessageBox.information(
                    win, "Saved", f"Graph saved to:\n{path}"
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    win, "Save Graph", f"Could not save graph:\n{e}"
                )

        # --- Export table to CSV ---
        def on_export_csv():
            base_path = getattr(self, "wav_file_path", "") or getattr(
                self, "current_file_path", ""
            ) or ""
            if not base_path:
                base_path = self.file_name or "octave_output"

            base_dir = os.path.dirname(base_path)
            if not base_dir:
                base_dir = os.getcwd()

            root_name = os.path.splitext(os.path.basename(base_path))[0]
            default_path = os.path.join(base_dir, f"{root_name}_octave_bands.csv")

            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                win,
                "Export CSV",
                default_path,
                "CSV Files (*.csv);;All Files (*)",
            )
            if not path:
                return

            try:
                with open(path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(headers)
                    for i in range(num_bands):
                        row = [
                            f"{centers[i]:.9f}",
                            f"{band_low[i]:.9f}",
                            f"{band_high[i]:.9f}",
                        ]
                        for ch in sel_channels:
                            v = vrms_by_ch[ch][i]
                            d = db_by_ch[ch][i]
                            row.append("" if not np.isfinite(v) else f"{v:.9g}")
                            row.append("" if not np.isfinite(d) else f"{d:.6f}")
                        w.writerow(row)

                QtWidgets.QMessageBox.information(
                    win, "Export CSV", f"Saved:\n{path}"
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    win, "Export CSV", f"Failed to save CSV:\n{e}"
                )

        # --- Save & Store to DB (per channel, per band) ---
        def on_save_and_store():
            nonlocal screenshot_path

            # Ensure we have at least some screenshot path
            if not screenshot_path:
                base_path = getattr(self, "wav_file_path", "") or getattr(
                    self, "current_file_path", ""
                ) or ""
                if not base_path:
                    base_path = self.file_name or "octave_output"

                base_dir = os.path.dirname(base_path)
                if not base_dir:
                    base_dir = os.getcwd()

                root_name = os.path.splitext(os.path.basename(base_path))[0]
                out_path = os.path.join(base_dir, f"{root_name}_octaveVRMS_multi.png")
                try:
                    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
                    screenshot_path = out_path
                except Exception as e:
                    QtWidgets.QMessageBox.warning(
                        win, "Save Graph", f"Could not save default screenshot:\n{e}"
                    )
                    screenshot_path = ""

            if not hasattr(self, "log_measurement_with_project"):
                QtWidgets.QMessageBox.warning(
                    win, "Logging", "log_measurement_with_project() is not available."
                )
                return

            try:
                vmax_local = float(self.max_voltage_entry.text())
            except Exception:
                vmax_local = vmax

            total_rows = 0
            for ch in sel_channels:
                # per-channel file name & method
                if n_ch > 1:
                    try:
                        fname = self.channel_file_label(ch)
                    except Exception:
                        fname = f"{self.file_name}_ch{ch+1}"
                    method_name = f"{band_type} Octave-Band VRMS_ch{ch+1}"
                else:
                    fname = self.file_name
                    method_name = f"{band_type} Octave-Band VRMS"

                for idx_band, f0 in enumerate(centers):
                    vrms_val = float(vrms_by_ch[ch][idx_band])
                    lvl_db = float(db_by_ch[ch][idx_band])
                    if not np.isfinite(vrms_val):
                        continue

                    f_low = float(band_low[idx_band])
                    f_high = float(band_high[idx_band])
                    bw = f_high - f_low

                    # Use 0 for time-related fields (no time window here)
                    self.log_measurement_with_project(
                        fname,
                        method_name,
                        float(f0),         # target_frequency
                        0.0,               # start_time
                        0.0,               # end_time
                        0.0,               # window_length
                        vmax_local,        # max_voltage
                        bw,                # bandwidth (Hz)
                        vrms_val,          # measured_voltage (VRMS)
                        False,             # filter_applied
                        screenshot_path,   # screenshot
                        misc=lvl_db,       # store dB level
                    )
                    total_rows += 1

            QtWidgets.QMessageBox.information(
                win,
                "Stored",
                f"Stored {total_rows} octave-band value(s) across {len(sel_channels)} channel(s).",
            )
            win.accept()

        save_graph_btn.clicked.connect(on_save_graph)
        export_csv_btn.clicked.connect(on_export_csv)
        save_store_btn.clicked.connect(on_save_and_store)
        close_btn.clicked.connect(win.reject)

        win.exec_()
        plt.close(fig)


    def ctd_import_popup(self):
        """
        CTD Import & Sound-Speed Profile  — with metadata prompt + DB save
        - Reads Sea-Bird .cnv and CSV with headers
        - Computes c(z) with Mackenzie (1981) if not supplied
        - Prompts for Name / Date-Time / GPS and saves to sqlite `ctd_profiles`
        - Can set as 'active' (self.ctd_profile) for other tools (e.g., Wenz)
        """
        from PyQt5 import QtWidgets, QtCore, QtGui
        import os, csv, re, json, sqlite3, numpy as np, ast
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

        # --- DB setup ----------------------------------------------------------
        def _db_path():
            try:
                from analyze_qt import DB_FILENAME
                return DB_FILENAME
            except Exception:
                return os.path.join(os.path.abspath(os.getcwd()), "analyze_qt.db")

        def _ensure_ctd_table(conn):
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ctd_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    dt_utc TEXT,
                    latitude REAL,
                    longitude REAL,
                    notes TEXT,
                    project_id INTEGER,
                    source TEXT,
                    depth_json TEXT NOT NULL,
                    temp_json TEXT,
                    sal_json TEXT,
                    sound_speed_json TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            try:
                cur.execute("ALTER TABLE ctd_profiles ADD COLUMN notes TEXT")
            except Exception:
                pass
            try:
                cur.execute("ALTER TABLE ctd_profiles ADD COLUMN project_id INTEGER")
            except Exception:
                pass
            cur.execute("CREATE INDEX IF NOT EXISTS idx_ctd_profiles_dt ON ctd_profiles(dt_utc)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_ctd_profiles_name ON ctd_profiles(name)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_ctd_profiles_project ON ctd_profiles(project_id)")
            conn.commit()

        def _save_profile_to_db(name, dt_iso_utc, lat, lon, notes, project_id, source, depth, temp, sal, c_ms):
            path = _db_path()
            conn = sqlite3.connect(path)
            _ensure_ctd_table(conn)
            cur = conn.cursor()
            as_list = lambda a: [] if a is None else [float(x) for x in np.asarray(a).ravel()]
            cur.execute("""
                INSERT INTO ctd_profiles
                (name, dt_utc, latitude, longitude, notes, project_id, source,
                depth_json, temp_json, sal_json, sound_speed_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                name, dt_iso_utc, None if lat is None else float(lat),
                None if lon is None else float(lon),
                notes or "",
                project_id,
                source or "",
                json.dumps(as_list(depth)),
                json.dumps(as_list(temp) if temp is not None else None),
                json.dumps(as_list(sal) if sal is not None else None),
                json.dumps(as_list(c_ms) if c_ms is not None else None),
            ))
            conn.commit()
            conn.close()

        def _list_ctd_profiles(project_id=None):
            path = _db_path()
            conn = sqlite3.connect(path)
            _ensure_ctd_table(conn)
            cur = conn.cursor()
            if project_id:
                cur.execute("""
                    SELECT c.id, c.name, c.dt_utc, c.latitude, c.longitude, c.notes,
                           p.name as project_name
                    FROM ctd_profiles c
                    LEFT JOIN projects p ON c.project_id = p.id
                    WHERE c.project_id = ?
                    ORDER BY c.dt_utc DESC, c.id DESC
                """, (project_id,))
            else:
                cur.execute("""
                    SELECT c.id, c.name, c.dt_utc, c.latitude, c.longitude, c.notes,
                           p.name as project_name
                    FROM ctd_profiles c
                    LEFT JOIN projects p ON c.project_id = p.id
                    ORDER BY c.dt_utc DESC, c.id DESC
                """)
            rows = cur.fetchall()
            conn.close()
            return rows

        def _list_projects():
            path = _db_path()
            conn = sqlite3.connect(path)
            cur = conn.cursor()
            cur.execute("SELECT id, name FROM projects ORDER BY name")
            rows = cur.fetchall()
            conn.close()
            return rows

        def _delete_ctd_profile(ctd_id):
            path = _db_path()
            conn = sqlite3.connect(path)
            _ensure_ctd_table(conn)
            cur = conn.cursor()
            cur.execute("DELETE FROM ctd_profiles WHERE id = ?", (ctd_id,))
            conn.commit()
            conn.close()

        def _load_ctd_profile(ctd_id):
            """
            Loads CTD arrays from ctd_profiles table columns:
            depth_json (required), temp_json, sal_json, sound_speed_json
            Returns normalized dict:
            {"depth_m": D, "temp_C": T, "sal_ppt": S, "pH": None, "c_ms": C}
            """
            try:
                conn = sqlite3.connect(_db_path()); cur = conn.cursor()
                cur.execute("""
                    SELECT depth_json, temp_json, sal_json, sound_speed_json
                    FROM ctd_profiles
                    WHERE id=?
                """, (ctd_id,))
                row = cur.fetchone()
                conn.close()
            except Exception:
                row = None

            if not row:
                return None

            depth_txt, temp_txt, sal_txt, c_txt = row

            def _parse_arr(txt):
                if txt is None:
                    return None
                txt = str(txt).strip()
                if not txt:
                    return None
                try:
                    obj = json.loads(txt)
                except Exception:
                    try:
                        obj = ast.literal_eval(txt)
                    except Exception:
                        return None

                # Accept: [1,2,3] or {"data":[...]} or {"values":[...]}
                if isinstance(obj, dict):
                    for k in ("data", "values", "array", "points"):
                        if k in obj:
                            obj = obj[k]
                            break

                try:
                    a = np.asarray(obj, dtype=float)
                except Exception:
                    return None

                if a.ndim != 1:
                    # If stored as [[depth,temp],...], take first column
                    if a.ndim == 2 and a.shape[1] >= 1:
                        a = np.asarray(a[:, 0], dtype=float)
                    else:
                        return None

                a = a[np.isfinite(a)]
                return a if a.size >= 2 else None

            D = _parse_arr(depth_txt)
            T = _parse_arr(temp_txt)
            S = _parse_arr(sal_txt)
            C = _parse_arr(c_txt)

            if D is None or D.size < 3:
                return None

            # If T missing, we can't compute depth guidance; still return what we have.
            # But the "ideal depth" feature requires temp.
            # We'll handle that upstream.
            if T is not None:
                n = min(D.size, T.size)
                D = D[:n]; T = T[:n]
                if S is not None:
                    S = S[:n] if S.size >= n else None
                if C is not None:
                    C = C[:n] if C.size >= n else None

                # sort by depth
                idx = np.argsort(D)
                D = D[idx]; T = T[idx]
                if S is not None: S = S[idx]
                if C is not None: C = C[idx]

            return {"depth_m": D, "temp_C": T, "sal_ppt": S, "pH": None, "c_ms": C}


        # --- UI skeleton -------------------------------------------------------
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Import CTD / Plot Profiles")
        dlg.setStyleSheet("background:#19232D; color:white;")
        dlg.resize(1100, 700)
        vbox = QtWidgets.QVBoxLayout(dlg); vbox.setContentsMargins(10,10,10,10); vbox.setSpacing(8)

        tabs = QtWidgets.QTabWidget()
        vbox.addWidget(tabs)

        # --- Tab 1: Import -----------------------------------------------------
        import_tab = QtWidgets.QWidget()
        tabs.addTab(import_tab, "Import")
        import_layout = QtWidgets.QVBoxLayout(import_tab)
        import_layout.setContentsMargins(10, 10, 10, 10)
        import_layout.setSpacing(8)

        top = QtWidgets.QHBoxLayout()
        pick_btn = QtWidgets.QPushButton("Open CTD (.cnv / .csv)")
        pick_btn.setStyleSheet("background:#3E6C8A;color:white;padding:6px 10px;border-radius:4px;")
        browse_btn = QtWidgets.QPushButton("Browse Saved Casts")
        browse_btn.setStyleSheet("background:#3E6C8A;color:white;padding:6px 10px;border-radius:4px;")
        file_lbl = QtWidgets.QLabel("(no file)"); file_lbl.setStyleSheet("color:#A0C4FF;")
        top.addWidget(pick_btn); top.addWidget(browse_btn); top.addSpacing(10); top.addWidget(file_lbl); top.addStretch()

        opt = QtWidgets.QHBoxLayout()
        use_c_from_file = QtWidgets.QCheckBox("Use sound speed column if present"); use_c_from_file.setChecked(True)
        flip_depth = QtWidgets.QCheckBox("Depth down"); flip_depth.setChecked(True)
        opt.addWidget(use_c_from_file); opt.addSpacing(12); opt.addWidget(flip_depth); opt.addStretch()

        import_layout.addLayout(top)
        import_layout.addLayout(opt)

        fig = Figure(facecolor="#19232D")
        axT = fig.add_subplot(131); axS = fig.add_subplot(132, sharey=axT); axC = fig.add_subplot(133, sharey=axT)
        def _style_axes(ax, xlab, ylab=None):
            ax.set_facecolor("#19232D")
            for s in ax.spines.values(): s.set_color("white")
            ax.tick_params(colors="white")
            ax.grid(True, ls="--", alpha=0.35, color="gray")
            ax.set_xlabel(xlab, color="white")
            if ylab: ax.set_ylabel(ylab, color="white")
        _style_axes(axT, "Temperature (°C)", "Depth (m)")
        _style_axes(axS, "Salinity (PSU)")
        _style_axes(axC, "Sound speed c (m/s)")
        fig.tight_layout()
        canvas = FigureCanvas(fig)
        import_layout.addWidget(canvas, 1)

        # Bottom row
        bot = QtWidgets.QHBoxLayout()
        save_btn = QtWidgets.QPushButton("Save Plot")
        save_btn.setStyleSheet("background:#3E6C8A;color:white;padding:6px 10px;border-radius:4px;")
        csv_btn = QtWidgets.QPushButton("Export CSV")
        csv_btn.setStyleSheet("background:#3E6C8A;color:white;padding:6px 10px;border-radius:4px;")
        set_active_btn = QtWidgets.QPushButton("Set As Active Profile")
        set_active_btn.setStyleSheet("background:#6EEB83;color:#111;padding:6px 12px;border-radius:6px;font-weight:bold;")
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.setStyleSheet("background:#3E6C8A;color:white;padding:6px 10px;border-radius:4px;")
        for b in (save_btn, csv_btn, set_active_btn, close_btn): bot.addWidget(b)
        bot.addStretch(); import_layout.addLayout(bot)

        # --- Tab 2: Browse -----------------------------------------------------
        browse_tab = QtWidgets.QWidget()
        tabs.addTab(browse_tab, "Browse Casts")
        browse_layout = QtWidgets.QVBoxLayout(browse_tab)
        browse_layout.setContentsMargins(10, 10, 10, 10)
        browse_layout.setSpacing(8)

        browse_top = QtWidgets.QHBoxLayout()
        browse_top.addWidget(QtWidgets.QLabel("Project:"))
        project_cb = QtWidgets.QComboBox()
        project_cb.setMinimumWidth(200)
        browse_top.addWidget(project_cb)
        browse_top.addStretch()
        browse_layout.addLayout(browse_top)

        table = QtWidgets.QTableWidget(0, 6)
        table.setHorizontalHeaderLabels(["Name", "Date (UTC)", "Latitude", "Longitude", "Notes", "Project"])
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        table.horizontalHeader().setStretchLastSection(True)
        browse_layout.addWidget(table)

        browse_btns = QtWidgets.QHBoxLayout()
        load_btn = QtWidgets.QPushButton("Load")
        export_btn = QtWidgets.QPushButton("Export CSV")
        export_all_btn = QtWidgets.QPushButton("Export Project CSVs")
        delete_btn = QtWidgets.QPushButton("Delete")
        refresh_btn = QtWidgets.QPushButton("Refresh")
        for b in (load_btn, export_btn, export_all_btn, delete_btn, refresh_btn):
            browse_btns.addWidget(b)
        browse_btns.addStretch()
        browse_layout.addLayout(browse_btns)

        # --- Helpers: parsing & physics ---------------------------------------
        def _mackenzie_c_ms(T, S, z):
            T = np.asarray(T, float); S = np.asarray(S, float); z = np.asarray(z, float)
            return (1448.96 + 4.591*T - 5.304e-2*T**2 + 2.374e-4*T**3
                    + 1.340*(S - 35.0) + 1.630e-2*z + 1.675e-7*z**2
                    - 1.025e-2*T*(S - 35.0) - 7.139e-13*T*z**3)

        def _pressure_to_depth_m(p_dbar):
            return np.asarray(p_dbar, float) * 1.0197

        def _read_cnv(path):
            names, units, data = [], [], []
            with open(path, "r", errors="ignore") as fh:
                lines = fh.read().splitlines()
            for ln in lines:
                if ln.startswith("# name ") and "=" in ln:
                    right = ln.split("=", 1)[1].strip()
                    nm = right.split(":")[0].strip()
                    m = re.search(r"\[(.+?)\]", ln)
                    u = m.group(1) if m else ""
                    names.append(nm); units.append(u)
            start_idx = 0
            for i, ln in enumerate(lines):
                if (ln and not ln.startswith("#") and not ln.startswith("*")):
                    start_idx = i; break
            for ln in lines[start_idx:]:
                if not ln or ln.startswith("#") or ln.startswith("*"): continue
                parts = ln.strip().split()
                try: data.append([float(x) for x in parts])
                except: pass
            arr = np.array(data, float) if data else np.zeros((0, len(names)), float)
            return names, units, arr

        def _read_csv(path):
            with open(path, "r", newline="", errors="ignore") as fh:
                rdr = csv.reader(fh); rows = list(rdr)
            if not rows: return [], [], np.zeros((0,0), float)

            header = []
            data_rows = []
            in_data = False
            tab_mode = False
            def _to_float(val):
                try:
                    return float(val)
                except Exception:
                    return np.nan
            for row in rows:
                line = "\t".join(row).strip()
                if line.startswith("[MeasurementMetadata]"):
                    continue
                if line.startswith("[MeasurementData]"):
                    in_data = True
                    continue
                if line.startswith("["):
                    continue
                if line.startswith("Columns="):
                    raw_cols = line.split("=", 1)[1]
                    cols = [h.strip() for h in raw_cols.split("\t") if h.strip()]
                    if len(cols) <= 1:
                        cols = [h.strip() for h in raw_cols.split(",") if h.strip()]
                    header = cols
                    tab_mode = True
                    continue
                if not in_data:
                    continue
                if not header:
                    header = [f"Column {i+1}" for i in range(len(row))]
                if tab_mode:
                    vals = line.split("\t")[:len(header)]
                else:
                    vals = row[:len(header)]
                if not vals:
                    continue
                data_rows.append([_to_float(x) if str(x).strip() != "" else np.nan for x in vals])

            if not data_rows:
                header = [h.strip() for h in rows[0]]
                data_rows = []
                for r in rows[1:]:
                    data_rows.append([_to_float(x) if str(x).strip() != "" else np.nan for x in r[:len(header)]])

            arr = np.array(data_rows, float) if data_rows else np.zeros((0, len(header)), float)
            names, units = [], []
            for h in header:
                m = re.match(r"(.+?)\s*\((.+?)\)\s*$", h)
                if m: names.append(m.group(1).strip()); units.append(m.group(2).strip())
                else: names.append(h); units.append("")
            return names, units, arr

        def _pick_columns(names):
            idx = dict(depth=None, pressure=None, temp=None, sal=None, c=None)
            lname = [n.lower() for n in names]
            def find_any(keys):
                for i, n in enumerate(lname):
                    for k in keys:
                        if k in n: return i
                return None
            idx["depth"]   = find_any(["depth", "depsm", "dep", "z"])
            idx["pressure"]= find_any(["press", "pressure", "prd", "prdm"])
            idx["temp"]    = find_any(["temperature", "temp", "t090", "t190"])
            idx["sal"]     = find_any(["sal", "salin", "salinity", "psu", "practical"])
            idx["c"]       = find_any(["sound", "sv", "snd", "veloc", "c("])
            return idx

        # --- State -------------------------------------------------------------
        state = {"path": None, "depth_m": None, "temp_c": None, "sal_psu": None, "c_ms": None}

        # --- Metadata dialog ---------------------------------------------------
        def _prompt_metadata(default_name, default_dt_utc):
            d = QtWidgets.QDialog(dlg)
            d.setWindowTitle("Save CTD Profile")
            form = QtWidgets.QFormLayout(d); form.setContentsMargins(12,12,12,12); form.setSpacing(8)
            name = QtWidgets.QLineEdit(default_name)
            dt = QtWidgets.QDateTimeEdit()
            dt.setDisplayFormat("yyyy-MM-dd HH:mm:ss 'UTC'")
            dt.setTimeSpec(QtCore.Qt.UTC)
            dt.setDateTime(QtCore.QDateTime.fromString(default_dt_utc.replace("Z",""), "yyyy-MM-ddTHH:mm:ss"))
            seed_loc = getattr(self, '_ctd_seed_location', None)
            seed_lat = '' if not seed_loc else f"{float(seed_loc[0]):.7f}"
            seed_lon = '' if not seed_loc else f"{float(seed_loc[1]):.7f}"
            lat = QtWidgets.QLineEdit(seed_lat); lon = QtWidgets.QLineEdit(seed_lon)
            notes = QtWidgets.QTextEdit("")
            notes.setFixedHeight(60)
            for w in (lat, lon):
                w.setPlaceholderText("e.g., 45.1234 or -66.4321")
                w.setValidator(QtGui.QDoubleValidator(-999.0, 999.0, 8))
            form.addRow("Name:", name)
            form.addRow("Date/Time (UTC):", dt)
            form.addRow("Latitude:", lat)
            form.addRow("Longitude:", lon)
            form.addRow("Notes:", notes)
            bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok|QtWidgets.QDialogButtonBox.Cancel)
            form.addRow(bb)
            bb.accepted.connect(d.accept); bb.rejected.connect(d.reject)
            if d.exec_() != QtWidgets.QDialog.Accepted: return None
            nm = name.text().strip()
            if not nm: nm = default_name
            dt_iso = dt.dateTime().toUTC().toString(QtCore.Qt.ISODate)  # 'YYYY-MM-DDTHH:mm:ssZ'
            def _f(txt):
                t = txt.strip()
                if not t: return None
                try: return float(t)
                except: return None
            return {
                "name": nm,
                "dt_iso": dt_iso,
                "lat": _f(lat.text()),
                "lon": _f(lon.text()),
                "notes": notes.toPlainText().strip(),
            }

        # --- Core: parse -> derive -> plot ------------------------------------
        def _load_and_plot(path):
            if path is None: return
            file_lbl.setText(os.path.basename(path))
            ext = os.path.splitext(path)[1].lower()
            names, units, arr = (_read_cnv(path) if ext == ".cnv" else _read_csv(path))
            if arr.size == 0 or len(names) == 0:
                QtWidgets.QMessageBox.warning(dlg, "Parse error", "No numeric data found.")
                return

            idx = _pick_columns(names)

            # Depth
            if idx["depth"] is not None:
                depth = arr[:, idx["depth"]].astype(float)
            elif idx["pressure"] is not None:
                depth = _pressure_to_depth_m(arr[:, idx["pressure"]].astype(float))
            else:
                QtWidgets.QMessageBox.warning(dlg, "Missing column", "No Depth or Pressure column found.")
                return

            # Temp / Sal
            temp = arr[:, idx["temp"]].astype(float) if idx["temp"] is not None else None
            sal  = arr[:, idx["sal"]].astype(float)  if idx["sal"]  is not None else None

            # Sound speed
            c_col = arr[:, idx["c"]].astype(float) if idx["c"] is not None else None
            if c_col is not None and use_c_from_file.isChecked():
                c_ms = c_col
            else:
                if temp is None or sal is None:
                    QtWidgets.QMessageBox.information(dlg, "Need T & S",
                        "No sound speed column and insufficient data to compute c(z) (need Temperature & Salinity).")
                    c_ms = None
                else:
                    c_ms = _mackenzie_c_ms(temp, sal, np.maximum(depth, 0.0))

            # Sort by depth
            order = np.argsort(depth)
            depth = depth[order]
            temp  = temp[order] if temp is not None else None
            sal   = sal[order]  if sal  is not None else None
            c_ms  = c_ms[order] if c_ms is not None else None

            # Save state
            state.update({"path": path, "depth_m": depth, "temp_c": temp, "sal_psu": sal, "c_ms": c_ms})

            # Plot
            axT.clear(); axS.clear(); axC.clear()
            _style_axes(axT, "Temperature (°C)", "Depth (m)")
            _style_axes(axS, "Salinity (PSU)")
            _style_axes(axC, "Sound speed c (m/s)")
            if temp is not None: axT.plot(temp, depth, color=getattr(self, "graph_color", "#33C3F0"), lw=1.8, label="T")
            if sal  is not None: axS.plot(sal,  depth, color="#FFD166", lw=1.8, label="S")
            if c_ms is not None: axC.plot(c_ms, depth, color="#6EEB83", lw=2.0, label="c(z)")
            if flip_depth.isChecked():
                for a in (axT, axS, axC): a.invert_yaxis()
            for a in (axT, axS, axC): a.legend(facecolor="#222", edgecolor="#444", labelcolor="white")
            fig.tight_layout(); canvas.draw()

            # --- Prompt for metadata and save to DB ----------------------------
            # Defaults
            base = os.path.splitext(os.path.basename(path))[0]
            default_name = f"CTD {base}"
            try:
                mtime = os.path.getmtime(path)
                default_dt = QtCore.QDateTime.fromSecsSinceEpoch(int(mtime), QtCore.Qt.UTC).toString(QtCore.Qt.ISODate)
            except Exception:
                default_dt = QtCore.QDateTime.currentDateTimeUtc().toString(QtCore.Qt.ISODate)

            md = _prompt_metadata(default_name, default_dt)
            if md is not None:
                try:
                    proj_id = getattr(self, "current_project_id", None)
                    if not proj_id:
                        QtWidgets.QMessageBox.warning(
                            dlg,
                            "No Project",
                            "Select a project before saving a CTD cast.",
                        )
                        return
                    _save_profile_to_db(
                        md["name"],
                        md["dt_iso"],
                        md["lat"],
                        md["lon"],
                        md.get("notes"),
                        proj_id,
                        path,
                        depth,
                        temp,
                        sal,
                        c_ms,
                    )
                    QtWidgets.QMessageBox.information(dlg, "Saved",
                        f"Saved CTD profile “{md['name']}” to database.")
                    # Also set as active
                    self.ctd_profile = {
                        "source": path, "name": md["name"], "dt_utc": md["dt_iso"],
                        "latitude": md["lat"], "longitude": md["lon"],
                        "notes": md.get("notes"),
                        "depth_m": depth, "temperature_C": temp, "salinity_PSU": sal, "sound_speed_m_s": c_ms,
                    }
                except Exception as e:
                    QtWidgets.QMessageBox.warning(dlg, "DB error", str(e))

        def _selected_id():
            idx = table.currentRow()
            if idx < 0:
                return None
            item = table.item(idx, 0)
            return item.data(QtCore.Qt.UserRole) if item else None

        def _populate_projects():
            project_cb.blockSignals(True)
            project_cb.clear()
            project_cb.addItem("All Projects", None)
            for pid, name in _list_projects():
                project_cb.addItem(name, pid)
            current_pid = getattr(self, "current_project_id", None)
            if current_pid:
                idx = project_cb.findData(current_pid)
                if idx >= 0:
                    project_cb.setCurrentIndex(idx)
            project_cb.blockSignals(False)

        def _populate_table():
            pid = project_cb.currentData()
            rows = _list_ctd_profiles(project_id=pid)
            table.setRowCount(0)
            for row in rows:
                rid, name, dt_utc, lat, lon, notes, project_name = row
                r = table.rowCount()
                table.insertRow(r)
                values = [
                    name or "",
                    dt_utc or "",
                    "" if lat is None else f"{lat:.6f}",
                    "" if lon is None else f"{lon:.6f}",
                    notes or "",
                    project_name or "",
                ]
                for c, val in enumerate(values):
                    item = QtWidgets.QTableWidgetItem(val)
                    item.setData(QtCore.Qt.UserRole, rid)
                    table.setItem(r, c, item)

        def _load_selected():
            ctd_id = _selected_id()
            if not ctd_id:
                QtWidgets.QMessageBox.information(dlg, "Select Cast", "Select a CTD cast first.")
                return
            prof = _load_ctd_profile(ctd_id)
            if not prof:
                QtWidgets.QMessageBox.warning(dlg, "Load Failed", "Could not load cast.")
                return
            state.update({
                "path": prof.get("source"),
                "depth_m": prof.get("depth_m"),
                "temp_c": prof.get("temperature_C"),
                "sal_psu": prof.get("salinity_PSU"),
                "c_ms": prof.get("sound_speed_m_s"),
            })
            axT.clear(); axS.clear(); axC.clear()
            _style_axes(axT, "Temperature (°C)", "Depth (m)")
            _style_axes(axS, "Salinity (PSU)")
            _style_axes(axC, "Sound speed c (m/s)")
            if prof.get("temperature_C") is not None:
                axT.plot(prof["temperature_C"], prof["depth_m"], color=getattr(self, "graph_color", "#33C3F0"), lw=1.8, label="T")
            if prof.get("salinity_PSU") is not None:
                axS.plot(prof["salinity_PSU"], prof["depth_m"], color="#FFD166", lw=1.8, label="S")
            if prof.get("sound_speed_m_s") is not None:
                axC.plot(prof["sound_speed_m_s"], prof["depth_m"], color="#6EEB83", lw=2.0, label="c(z)")
            if flip_depth.isChecked():
                for a in (axT, axS, axC): a.invert_yaxis()
            for a in (axT, axS, axC): a.legend(facecolor="#222", edgecolor="#444", labelcolor="white")
            fig.tight_layout(); canvas.draw()
            file_lbl.setText(prof.get("source") or "(from DB)")
            self.ctd_profile = prof
            tabs.setCurrentWidget(import_tab)

        def _export_selected():
            ctd_id = _selected_id()
            if not ctd_id:
                QtWidgets.QMessageBox.information(dlg, "Select Cast", "Select a CTD cast first.")
                return
            prof = _load_ctd_profile(ctd_id)
            if not prof:
                QtWidgets.QMessageBox.warning(dlg, "Export Failed", "Could not load cast.")
                return
            p, _ = QtWidgets.QFileDialog.getSaveFileName(dlg, "Export CTD CSV", "", "CSV (*.csv)")
            if not p:
                return
            _export_ctd_profile_csv(prof, p)

        def _export_ctd_profile_csv(prof, path):
            with open(path, "w", newline="") as fh:
                w = csv.writer(fh)
                header = ["Depth_m"]
                cols = [prof.get("depth_m")]
                if prof.get("temperature_C") is not None:
                    header.append("Temperature_C"); cols.append(prof["temperature_C"])
                if prof.get("salinity_PSU") is not None:
                    header.append("Salinity_PSU"); cols.append(prof["salinity_PSU"])
                if prof.get("sound_speed_m_s") is not None:
                    header.append("SoundSpeed_m_per_s"); cols.append(prof["sound_speed_m_s"])
                w.writerow(header)
                if cols[0] is None:
                    return
                for i in range(len(cols[0])):
                    row = [cols[0][i]] + [col[i] if col is not None and i < len(col) else "" for col in cols[1:]]
                    w.writerow(row)

        def _export_project_casts():
            pid = project_cb.currentData()
            if not pid:
                QtWidgets.QMessageBox.information(dlg, "Select Project", "Select a project to export.")
                return
            proj_name = project_cb.currentText()
            out_dir = QtWidgets.QFileDialog.getExistingDirectory(dlg, "Select export folder")
            if not out_dir:
                return
            rows = _list_ctd_profiles(project_id=pid)
            if not rows:
                QtWidgets.QMessageBox.information(dlg, "No Casts", "No CTD casts for that project.")
                return
            for rid, name, dt_utc, lat, lon, notes, project_name in rows:
                prof = _load_ctd_profile(rid)
                if not prof:
                    continue
                safe = re.sub(r"[^\w\-\. ]+", "_", name or f"ctd_{rid}").strip() or f"ctd_{rid}"
                path = os.path.join(out_dir, f"{safe}.csv")
                _export_ctd_profile_csv(prof, path)
            QtWidgets.QMessageBox.information(
                dlg,
                "Exported",
                f"Exported {len(rows)} CTD casts for {proj_name}.",
            )

        def _delete_selected():
            ctd_id = _selected_id()
            if not ctd_id:
                QtWidgets.QMessageBox.information(dlg, "Select Cast", "Select a CTD cast first.")
                return
            if QtWidgets.QMessageBox.question(
                dlg,
                "Delete CTD Cast",
                "Delete the selected CTD cast?",
            ) != QtWidgets.QMessageBox.Yes:
                return
            _delete_ctd_profile(ctd_id)
            _populate_table()

        def _show_saved_casts():
            tabs.setCurrentWidget(browse_tab)
            _populate_projects()
            _populate_table()

        # --- Actions -----------------------------------------------------------
        def on_pick():
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                dlg, "Open CTD", "", "Sea-Bird CNV (*.cnv);;CSV (*.csv);;All Files (*)"
            )
            if not path: return
            _load_and_plot(path)

        def on_save_plot():
            if state["depth_m"] is None:
                QtWidgets.QMessageBox.information(dlg, "Nothing to save", "Load a CTD file first."); return
            path, _ = QtWidgets.QFileDialog.getSaveFileName(dlg, "Save Plot", "", "PNG (*.png);;JPEG (*.jpg)")
            if not path: return
            fig.savefig(path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")

        def on_export_csv():
            if state["depth_m"] is None:
                QtWidgets.QMessageBox.information(dlg, "Nothing to export", "Load a CTD file first."); return
            p, _ = QtWidgets.QFileDialog.getSaveFileName(dlg, "Export CSV", "", "CSV (*.csv)")
            if not p: return
            with open(p, "w", newline="") as fh:
                w = csv.writer(fh)
                header = ["Depth_m"]; cols = [state["depth_m"]]
                if state["temp_c"] is not None: header += ["Temperature_C"]; cols += [state["temp_c"]]
                if state["sal_psu"] is not None: header += ["Salinity_PSU"];  cols += [state["sal_psu"]]
                if state["c_ms"]   is not None: header += ["SoundSpeed_m_per_s"]; cols += [state["c_ms"]]
                w.writerow(header); N = len(state["depth_m"])
                for i in range(N):
                    row = [cols[0][i]] + [col[i] if i < len(col) else "" for col in cols[1:]]
                    w.writerow(row)

        def on_set_active():
            if state["depth_m"] is None:
                QtWidgets.QMessageBox.information(dlg, "No profile", "Load a CTD file first."); return
            self.ctd_profile = {
                "source": state["path"], "depth_m": state["depth_m"],
                "temperature_C": state["temp_c"], "salinity_PSU": state["sal_psu"],
                "sound_speed_m_s": state["c_ms"],
            }
            QtWidgets.QMessageBox.information(dlg, "Active profile set",
                "CTD profile stored in self.ctd_profile for other tools.")

        # Wire up
        pick_btn.clicked.connect(on_pick)
        browse_btn.clicked.connect(_show_saved_casts)
        save_btn.clicked.connect(on_save_plot)
        csv_btn.clicked.connect(on_export_csv)
        set_active_btn.clicked.connect(on_set_active)
        close_btn.clicked.connect(dlg.accept)

        project_cb.currentIndexChanged.connect(_populate_table)
        refresh_btn.clicked.connect(_populate_projects)
        refresh_btn.clicked.connect(_populate_table)
        load_btn.clicked.connect(_load_selected)
        export_btn.clicked.connect(_export_selected)
        export_all_btn.clicked.connect(_export_project_casts)
        delete_btn.clicked.connect(_delete_selected)

        _populate_projects()
        _populate_table()

        dlg.exec_()

    # ---------------------
    # Tool Selection Logic
    # ---------------------
    def update_tool_list(self, index):
        """
        Populate self.tool_combo based on the currently selected category.
        Then force “no item selected” so on_tool_selected isn’t called immediately.
        """
        category = self.category_combo.currentText()
        self.tool_combo.blockSignals(True)
        self.tool_combo.clear()

        if category == "WAV File Tools":
            self.tool_combo.addItems([
                "Trim File",
                "Denoise File",
                "High/Low Pass Filter",
                "Anti-Aliasing Filter",
                "Recommend Sample Rate",
                "Remove DC Offset",
                "Normalize File",
                "Downsample Bit Depth",
                "Wav File Combine",
                "Stack WAV Channels",
                "Scale WAV Amplitude",
                "Create WAV Playlist",
                "Secure WAV Bundle (Pack/Unpack)",
                "Channel Sync Tool"
            ])
        elif category == "Measurement Tools":
            self.tool_combo.addItems([
                "Ambient Noise",
                "Peak Prominences Analysis",
                "Interval Analysis",
                "Depth Sounder Analysis",
                "Slope De-Clipper",
                "Find Peaks",
                "Short-Time RMS",
                "Crest Factor",
                "Octave-Band Analysis",
                "SNR Estimator",
                "LFM Analysis",
                "LFM Batch Analysis",
                "HFM Analysis",
                "Multi Frequency Analysis",
                "SPL Transmit Analysis",
                "Hydrophone Calibration",
                "Duty Cycle Analysis",
                "Exceedance Curves Lx",
                "LTSA + PSD Percentiles",
                "SEL",
                "Recurrence Periodicity",
                "Dominant Frequencies Over Time"              
            ])

        elif category == "Modelling & Plotting Tools":
            self.tool_combo.addItems([
                "Wenz Curves",
                "Propagation Modelling",
                "Cable Loss & Hydro Sensitivity"                
            ])
        
        elif category == "Detection & Classification Tools":  # "Detection & Classification Tools"
            self.tool_combo.addItems([
                "Active Sonar",
                "Cepstrum Analysis",
                "Event Clustering"
            ])

        else:  # "Database Tools"
            self.tool_combo.addItems([
                "Export Logs to Excel",
                "Filter Measurements",
                "Filter Logs by Date/Time",
                "Database Maintenance",
                "Clean Measurement Data",
                "Annotate Measurements",
                "Annotate SPL",
                "Calibration Curve Manager",
                "TVR Curve Manager",
                "Measurement Voltage Correction",
                "CTD Import"
            ])


        

        # Force no selection (so currentIndexChanged isn’t called immediately)
        self.tool_combo.setCurrentIndex(-1)
        self.tool_combo.blockSignals(False)

    def on_tool_selected(self, idx):
        """
        Called when the user picks a tool from the dropdown.
        Launch the appropriate popup or action.
        """
        # If idx < 0, that means “no selection” → do nothing
        if idx < 0:
            return

        category = self.category_combo.currentText()
        tool = self.tool_combo.currentText()

        if category == "WAV File Tools":
            if tool == "Trim File":
                self.trim_file()
            elif tool == "Denoise File":
                self.denoise_file()
            elif tool == "High/Low Pass Filter":
                self.pass_filter_popup()
            elif tool == "Anti-Aliasing Filter":
                self.anti_aliasing_popup()
            elif tool == "Recommend Sample Rate":
                self.recommend_sample_rate_popup()
            elif tool == "Remove DC Offset":
                self.remove_dc_offset_popup()
            elif tool == "Normalize File":
                self.normalize_file()
            elif tool == "Downsample Bit Depth":
                self.downsample_bit_depth_popup()
            elif tool == "Wav File Combine":
                self.wav_merge_popup()
            elif tool == "Stack WAV Channels":
                self.wav_stack_channels_popup()
            elif tool == "Scale WAV Amplitude":
                self.scale_wav_popup()
            elif tool == "Create WAV Playlist":
                self.wav_playlist_builder_tool()
            elif tool == "Secure WAV Bundle (Pack/Unpack)":
                self.wav_secure_pack_unpack_tool_async()
            elif tool == "Channel Sync Tool":
                self.wav_channel_sync_popup()
                

        elif category == "Measurement Tools":
            if   tool == "Peak Prominences Analysis":
                self.peak_prominences_popup()
            elif tool == "Interval Analysis":
                self.interval_analysis_popup()
            elif tool == "Ambient Noise":
                self.ambient_noise_analysis()
            elif tool == "Depth Sounder Analysis":
                self.depth_sounder_popup()
            elif tool == "Slope De-Clipper":
                self.slope_declipper_popup()
            elif tool == "Find Peaks":
                self.find_peaks_analysis()
            elif tool == "Short-Time RMS":
                self.short_time_rms_popup()
            elif tool == "Crest Factor":
                self.crest_factor_popup()
            elif tool == "Octave-Band Analysis":
                self.octave_band_analysis_popup()
            elif tool == "SNR Estimator":
                self.snr_estimator_popup()
            elif tool == "LFM Analysis":
                self.lfm_pulse_analysis()
            elif tool == "LFM Batch Analysis":
                self.lfm_pulse_batch_analysis()
            elif tool == "HFM Analysis":
                self.hfm_pulse_analysis()
            elif tool == "Multi Frequency Analysis":
                self.multi_freq_analysis()
            elif tool == "SPL Transmit Analysis":
                self.spl_from_voltage_popup()
            elif tool == "Hydrophone Calibration":
                self.generate_hydrophone_calibration_popup()
            elif tool=="Duty Cycle Analysis":
                self.duty_cycle_analysis_popup()
            elif tool=="SEL":
                self.sel_compliance_popup()
            elif tool=="Recurrence Periodicity":
                self.recurrence_periodicity_popup()
            elif tool=="Dominant Frequencies Over Time":
                self.analyze_dominant_frequencies_popup()
            

        elif category == "Modelling & Plotting Tools":
            if tool=="Wenz Curves":
                self.wenz_curves_popup()
            elif tool=="Propagation Modelling":
                self.propagation_from_spl_db_popup()
            elif tool=="Exceedance Curves Lx":
                self.exceedance_curves_popup()
            elif tool=="LTSA + PSD Percentiles":
                self.ltsa_psd_popup()
            elif tool=="Cable Loss & Hydro Sensitivity":
                self.cable_loss_and_hydro_popup()

                

        elif category == "Detection & Classification Tools":
            if tool == "Active Sonar":
                self.active_sonar_popup()
            elif tool == "Cepstrum Analysis":
                self.cepstrum_analysis()
            elif tool == "Event Clustering":
                self.automated_event_clustering_popup()
            

        elif category == "Database Tools":
            if tool == "Export Logs to Excel":
                self.export_logs_to_excel()
            elif tool == "Filter Measurements":
                self.filter_measurements_popup()
            elif tool == "Filter Logs by Date/Time":
                self.date_time_filter_popup()
            elif tool == "Database Maintenance":
                self.database_maintenance_popup()
            elif tool == "Clean Measurement Data":
                self.clean_measurement_data_popup()
            elif tool == "Annotate Measurements":
                self.annotate_measurements_popup()
            elif tool == "Annotate SPL":
                self.annotate_spl_logs_popup()
            elif tool == "Calibration Curve Manager":
                self.calibration_curve_manager_popup()
            elif tool == "Measurement Voltage Correction":
                self.correct_measurement_entries_popup()
            elif tool == "CTD Import":
                self.ctd_import_popup()
            elif tool == "TVR Curve Manager":
                self.tvr_curve_manager_popup()
                


        # Reset selection back to “no item” so user can pick again
        self.tool_combo.setCurrentIndex(-1)
    

    def update_fft_plot(self):
        """Redraw waveform snippet (top) and spectrum (bottom) using pyqtgraph."""
        if not getattr(self, 'fft_mode', False):
            return
        import numpy as np
        from PyQt5 import QtWidgets
        try:
            from scipy.signal import periodogram, welch, hilbert
        except Exception:
            periodogram = welch = hilbert = None

        try:
            win_sec = float(self.fft_length_entry.text())
        except Exception:
            try:
                QtWidgets.QMessageBox.warning(self, 'Error', 'Invalid FFT window length value.')
            except Exception:
                pass
            return
        self.fft_window_length = win_sec

        try:
            tmul = int(getattr(self, 'TIME_MULTIPLIER', 1000))
            t0 = self.fft_time_slider.value() / float(tmul)
        except Exception:
            t0 = 0.0
        self.fft_start_time = t0

        sr   = int(getattr(self, 'sample_rate', 0)) or 0
        data = getattr(self, 'full_data', None)
        if sr <= 0 or data is None:
            return
        i0 = max(0, int(t0 * sr))
        i1 = min(i0 + int(win_sec * sr), int(getattr(data, 'shape', [0])[0]))
        seg = data[i0:i1]
        if getattr(seg, 'size', 0) < 2:
            return

        try:
            sel = self.selected_channel_indices()
        except Exception:
            sel = None
        if not sel:
            sel = list(range(seg.shape[1])) if getattr(seg, 'ndim', 1) == 2 else [0]
        names = getattr(self, 'channel_names', [])

        if getattr(seg, 'ndim', 1) == 1:
            segs   = [np.asarray(seg)]
            labels = [names[0] if names else 'Ch 1']
        else:
            sel    = [ch for ch in sel if 0 <= ch < seg.shape[1]] or [0]
            segs   = [np.asarray(seg[:, ch]) for ch in sel]
            labels = [names[ch] if ch < len(names) else f'Ch {ch+1}' for ch in sel]

        def _to_float(x):
            if np.issubdtype(x.dtype, np.integer):
                xf = x.astype(np.float64) / (float(np.iinfo(x.dtype).max) or 1.0)
            else:
                xf = x.astype(np.float64)
            return xf - np.mean(xf)
        segs = [_to_float(x) for x in segs]

        palette = list((getattr(self, 'color_options', None) or {'Teal': '#03DFE2'}).values())
        base_color = getattr(self, 'graph_color', palette[0])
        try:
            base_idx = palette.index(base_color)
        except ValueError:
            base_idx = 0

        # ── TOP: waveform snippet ──────────────────────────────────────────
        pw = self.canvas.plot_waveform
        pw.clear()
        n_samples = segs[0].size if segs else 0
        t_seg = np.linspace(t0, t0 + win_sec, n_samples, endpoint=False)

        if len(segs) == 1:
            pen = pg.mkPen(color=base_color, width=1)
            pw.plot(t_seg, segs[0], pen=pen)
            axis_color = '#000000' if str(get_setting('ui_theme','dark')).lower() == 'light' else '#FFFFFF'
            pw.setLabel('left', 'Amplitude', color=axis_color)
        else:
            band_gap = 1.2
            n = len(segs)
            ticks = []
            for idx, x in enumerate(segs):
                xmin = float(np.min(x)); xmax = float(np.max(x))
                rng  = xmax - xmin
                x_scaled = (x - (xmin + xmax) / 2.0) / (rng or 1.0)
                center = (n - 1 - idx) * band_gap
                ticks.append((center, labels[idx]))
                col = palette[(base_idx + idx) % len(palette)]
                pw.plot(t_seg, x_scaled + center, pen=pg.mkPen(color=col, width=1))
            pw.getAxis('left').setTicks([ticks])
            axis_color = '#000000' if str(get_setting('ui_theme','dark')).lower() == 'light' else '#FFFFFF'
            pw.setLabel('left', 'Channels', color=axis_color)

        pw.setTitle(f'Raw waveform  {t0:.2f}–{t0+win_sec:.2f}s', color='#FFFFFF')
        pw.setLabel('bottom', 'Time (s)', color='#FFFFFF')
        pw.setXRange(t0, t0 + win_sec, padding=0)
        pw.enableAutoRange(axis='y', enable=True)

        # ── BOTTOM: spectrum ───────────────────────────────────────────────
        pf = self.canvas.plot_fft
        pf.clear()
        method = str(getattr(self, 'spectral_method', 'FFT'))
        title  = f'FFT {t0:.2f}–{t0+win_sec:.2f}s'

        for idx, x in enumerate(segs):
            col = palette[(base_idx + idx) % len(palette)] if len(segs) > 1 else base_color
            pen = pg.mkPen(color=col, width=1)

            if method == 'FFT' or method not in ('Periodogram','Welch','Hilbert','Modulation Spectrum','Multitaper'):
                w = np.hanning(len(x))
                X = np.fft.rfft(x * w)
                f = np.fft.rfftfreq(len(x), d=1.0 / float(sr))
                pf.plot(f, np.abs(X), pen=pen)
                title = f'FFT {t0:.2f}–{t0+win_sec:.2f}s'

            elif method == 'Periodogram' and periodogram is not None:
                f, Pxx = periodogram(x, fs=sr, window='hann', scaling='density')
                pf.plot(f, Pxx, pen=pen)
                title = f'Periodogram {t0:.2f}–{t0+win_sec:.2f}s'

            elif method == 'Welch' and welch is not None:
                nper = min(len(x), 1024) if len(x) > 0 else 256
                f, Pxx = welch(x, fs=sr, nperseg=max(256, nper))
                pf.plot(f, Pxx, pen=pen)
                title = f'Welch PSD {t0:.2f}–{t0+win_sec:.2f}s'

            elif method == 'Hilbert' and hilbert is not None:
                env = np.abs(hilbert(x))
                t_env = np.linspace(t0, t0 + win_sec, env.size, endpoint=False)
                pf.plot(t_env, env, pen=pen)
                title = f'Hilbert Envelope {t0:.2f}–{t0+win_sec:.2f}s'

            elif method == 'Modulation Spectrum' and welch is not None and hilbert is not None:
                env = np.abs(hilbert(x))
                mod_sig = env - np.mean(env)
                f_mod, Pxx_mod = welch(mod_sig, fs=sr, nperseg=min(len(mod_sig), 256))
                pf.plot(f_mod, 10 * np.log10(np.maximum(Pxx_mod, np.finfo(float).eps)), pen=pen)
                title = f'Modulation Spectrum {t0:.2f}–{t0+win_sec:.2f}s'

            elif method == 'Multitaper':
                try:
                    from shared import multitaper_psd
                    f_mt, Pxx_mt = multitaper_psd(x, sr)
                    pf.plot(f_mt, Pxx_mt, pen=pen)
                    title = f'Multitaper {t0:.2f}–{t0+win_sec:.2f}s'
                except Exception:
                    w = np.hanning(len(x))
                    X = np.fft.rfft(x * w)
                    f = np.fft.rfftfreq(len(x), d=1.0 / float(sr))
                    pf.plot(f, np.abs(X), pen=pen)

        pf.setTitle(title, color='#FFFFFF')
        pf.setLabel('bottom', 'Frequency (Hz)', color='#FFFFFF')
        pf.setLabel('left', 'Magnitude', color='#FFFFFF')

        # Apply x-limits if specified
        try:
            xmin_txt = self.fft_xmin_entry.text() if hasattr(self, 'fft_xmin_entry') else None
            xmax_txt = self.fft_xmax_entry.text() if hasattr(self, 'fft_xmax_entry') else None
            xmin = float(xmin_txt) if xmin_txt not in (None, '') else None
            xmax = float(xmax_txt) if xmax_txt not in (None, '') else None
            if xmin is not None and xmax is not None and xmax > xmin:
                pf.setXRange(xmin, xmax, padding=0)
            elif xmin is not None:
                pf.setXRange(xmin, pf.viewRange()[0][1], padding=0)
            elif xmax is not None:
                pf.setXRange(pf.viewRange()[0][0], xmax, padding=0)
        except Exception:
            pass


    def load_hub(self):
            if self._hub is None:
                self._hub = importlib.import_module("tensorflow_hub")
                self._tf  = importlib.import_module("tensorflow")
            return self._hub, self._tf

    


    # ---------------------
    # Analysis Tab Methods
    # ---------------------
    def setup_analysis_tab(self):
        layout = QtWidgets.QVBoxLayout(self.analysis_tab)
        controls = QtWidgets.QHBoxLayout()

        controls.addWidget(QtWidgets.QLabel("Target Frequency (Hz):"))
        self.freq_entry = QtWidgets.QLineEdit("1000")
        self.freq_entry.setFixedWidth(60)
        controls.addWidget(self.freq_entry)

        # Auto-detect frequency from FFT peak (dominant energy) for SPL calculations
        self.auto_freq_cb = QtWidgets.QCheckBox("Auto freq (FFT)")
        self.auto_freq_cb.setChecked(True)
        self.auto_freq_cb.setToolTip("When enabled, SPL logging uses the dominant FFT frequency from the selected time window instead of the typed frequency.")
        controls.addWidget(self.auto_freq_cb)

        def _auto_freq_toggled(checked):
            try:
                self.freq_entry.setEnabled(not checked)
            except Exception:
                pass

        self.auto_freq_cb.toggled.connect(_auto_freq_toggled)
        _auto_freq_toggled(True)

        controls.addWidget(QtWidgets.QLabel("Bandwidth (Hz):"))
        self.bw_entry = QtWidgets.QLineEdit("100")
        self.bw_entry.setFixedWidth(60)
        controls.addWidget(self.bw_entry)

        controls.addWidget(QtWidgets.QLabel("Max Voltage:"))
        self.max_voltage_entry = QtWidgets.QLineEdit("10")
        self.max_voltage_entry.setFixedWidth(60)
        controls.addWidget(self.max_voltage_entry)

        self.filter_checkbox = QtWidgets.QCheckBox("Apply Bandpass Filter")
        self.filter_checkbox.setChecked(self.apply_filter)
        controls.addWidget(self.filter_checkbox)

        self.update_button = QtWidgets.QPushButton("Update Analysis")
        self.update_button.clicked.connect(self.update_analysis)
        controls.addWidget(self.update_button)

        self.fft_button = QtWidgets.QPushButton("Show FFT")
        self.fft_button.clicked.connect(self.enter_fft_mode)
        self.fft_button.setStyleSheet(
            "background-color: #FFFFB5;"
            "color: black;"
            "border-radius: 4px;"
            "padding: 4px 8px;"
        )
        controls.addWidget(self.fft_button)

        self.result_label = QtWidgets.QLabel("RMS Voltage: N/A")
        controls.addWidget(self.result_label)

        self.import_curve_button = QtWidgets.QPushButton("Import Hydrophone Curve")
        self.import_curve_button.clicked.connect(self.import_curve)
        controls.addWidget(self.import_curve_button)

        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.addItems(["FFT", "Welch", "Multitaper", "Hilbert", "Periodogram", "Instantaneous Frequency", "Modulation Spectrum"])
        self.method_combo.currentTextChanged.connect(lambda text: self.set_spectral_method(text))
        controls.addWidget(QtWidgets.QLabel("Spectral Method:"))
        controls.addWidget(self.method_combo)

        # hydrophone_combo removed from UI; configured via Channels… dialog instead
        self.hydrophone_combo = None  # kept as attribute for compat

        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addLayout(controls)

        self.canvas = MplCanvas(self.analysis_tab)
        layout.addWidget(self.canvas, stretch=1)
        self.span = None
        self.enable_span_selector()

        # FFT Control Panel (visible only in FFT mode)
        self.fft_control_panel = QtWidgets.QWidget()
        fft_layout = QtWidgets.QHBoxLayout(self.fft_control_panel)
        fft_layout.setContentsMargins(2, 2, 2, 2)
        fft_layout.setSpacing(6)
        self.fft_control_panel.setMaximumHeight(0)  # hidden initially

        self.analyze_voltage_button = QtWidgets.QPushButton("Analyze Voltage")
        self.analyze_voltage_button.clicked.connect(self.analyze_voltage_from_fft)
        fft_layout.addWidget(self.analyze_voltage_button)

        self.scroll_step_entry = QtWidgets.QLineEdit("0.001")
        self.scroll_step_entry.setFixedWidth(70)
        fft_layout.addWidget(QtWidgets.QLabel("Scroll (s):"))
        fft_layout.addWidget(self.scroll_step_entry)

        self.fft_length_entry = QtWidgets.QLineEdit("1.0")
        self.fft_length_entry.setFixedWidth(70)
        fft_layout.addWidget(QtWidgets.QLabel("Window (s):"))
        fft_layout.addWidget(self.fft_length_entry)

        self.fft_time_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.fft_time_slider.setMinimum(0)
        self.fft_time_slider.setMaximum(1000)
        self.fft_time_slider.setValue(0)
        self.fft_time_slider.setSingleStep(1)
        self.fft_time_slider.sliderReleased.connect(self.update_fft_plot)
        fft_layout.addWidget(QtWidgets.QLabel("Start Time:"))
        fft_layout.addWidget(self.fft_time_slider)

        self.fft_xmin_entry = QtWidgets.QLineEdit("0")
        self.fft_xmin_entry.setFixedWidth(70)
        fft_layout.addWidget(QtWidgets.QLabel("X min (Hz):"))
        fft_layout.addWidget(self.fft_xmin_entry)

        fft_layout.addWidget(QtWidgets.QLabel("X max (Hz):"))
        self.fft_xmax_entry = QtWidgets.QLineEdit()
        self.fft_xmax_entry.setFixedWidth(70)
        fft_layout.addWidget(self.fft_xmax_entry)

        self.fft_method_combo = QtWidgets.QComboBox()
        self.fft_method_combo.addItems(["FFT", "Welch", "Multitaper", "Hilbert", "Periodogram", "Instantaneous Frequency", "Modulation Spectrum"])
        self.fft_method_combo.currentTextChanged.connect(lambda text: self.set_spectral_method(text))
        fft_layout.addWidget(QtWidgets.QLabel("Method:"))
        fft_layout.addWidget(self.fft_method_combo)

        update_fft_button = QtWidgets.QPushButton("Update FFT")
        update_fft_button.clicked.connect(self.update_fft_plot)
        fft_layout.addWidget(update_fft_button)

        self.fft_control_panel.setVisible(False)
        layout.addWidget(self.fft_control_panel)

        # Pulse Control Panel (visible only in FFT mode)
        self.pulse_control_panel = QtWidgets.QWidget()
        pulse_layout = QtWidgets.QHBoxLayout(self.pulse_control_panel)
        pulse_layout.setContentsMargins(2, 2, 2, 2)
        pulse_layout.setSpacing(6)
        self.pulse_control_panel.setMaximumHeight(60)
        self.pulse_control_panel.setVisible(False)

        self.pulse_threshold_entry = QtWidgets.QLineEdit("1000")
        self.pulse_threshold_entry.setFixedWidth(70)
        pulse_layout.addWidget(QtWidgets.QLabel("Min Amplitude:"))
        pulse_layout.addWidget(self.pulse_threshold_entry)

        find_pulse_button = QtWidgets.QPushButton("Find Pulse")
        find_pulse_button.clicked.connect(self.find_first_pulse)
        pulse_layout.addWidget(find_pulse_button)

        prev_pulse_button = QtWidgets.QPushButton("Previous Pulse")
        prev_pulse_button.clicked.connect(self.prev_pulse)
        pulse_layout.addWidget(prev_pulse_button)

        next_pulse_button = QtWidgets.QPushButton("Next Pulse")
        next_pulse_button.clicked.connect(self.next_pulse)
        pulse_layout.addWidget(next_pulse_button)

        auto_analyze_button = QtWidgets.QPushButton("Auto Analyze Pulses")
        auto_analyze_button.clicked.connect(self.auto_analyze_pulses)
        pulse_layout.addWidget(auto_analyze_button)

        self.pri_entry = QtWidgets.QLineEdit("0.5")
        self.pri_entry.setFixedWidth(70)
        pulse_layout.addWidget(QtWidgets.QLabel("PRI (s):"))
        pulse_layout.addWidget(self.pri_entry)

        self.pulse_control_panel.setVisible(False)
        self.pulse_control_panel.setMaximumHeight(0)
        layout.addWidget(self.pulse_control_panel)

    def enable_span_selector(self):
        """
        Install a click-and-drag span selector on the waveform plot.
        Left-click and drag draws a red highlight; releasing fires on_select(xmin, xmax).
        Uses a custom ViewBox subclass wired to the plot.
        """
        pw = getattr(self.canvas, 'plot_waveform', None) if hasattr(self, 'canvas') else None
        if pw is None:
            return

        # Remove any old region item
        if getattr(self, '_span_region', None) is not None:
            try:
                pw.removeItem(self._span_region)
            except Exception:
                pass
            self._span_region = None

        # Create a semi-transparent region item that we'll move manually
        region = pg.LinearRegionItem(
            values=(0, 0),
            brush=pg.mkBrush(220, 50, 50, 70),
            pen=pg.mkPen('#FF4444', width=1),
            movable=False,          # we control it via mouse events
        )
        region.setZValue(10)
        pw.addItem(region)
        self._span_region = region
        self.span = region

        # State for drag tracking
        self._span_dragging = False
        self._span_x0 = None

        vb = pw.getViewBox()

        # Keep the native pyqtgraph context menu/interaction reachable
        try:
            pw.setMenuEnabled(True)
            vb.setMenuEnabled(True)
        except Exception:
            pass

        if not hasattr(self, '_orig_wave_vb_mouse_press'):
            self._orig_wave_vb_mouse_press = vb.mousePressEvent
            self._orig_wave_vb_mouse_move = vb.mouseMoveEvent
            self._orig_wave_vb_mouse_release = vb.mouseReleaseEvent

        def _mouse_press(event):
            if event.button() == QtCore.Qt.LeftButton:
                pos = vb.mapSceneToView(event.scenePos())
                self._span_x0 = pos.x()
                self._span_dragging = True
                region.setRegion((pos.x(), pos.x()))
                event.accept()
                return
            # preserve right-click/context menu + built-in interactions
            try:
                self._orig_wave_vb_mouse_press(event)
            except Exception:
                pass

        def _mouse_move(event):
            if self._span_dragging and self._span_x0 is not None:
                pos = vb.mapSceneToView(event.scenePos())
                x0 = self._span_x0
                x1 = pos.x()
                region.setRegion((min(x0, x1), max(x0, x1)))
                event.accept()
                return
            try:
                self._orig_wave_vb_mouse_move(event)
            except Exception:
                pass

        def _mouse_release(event):
            if self._span_dragging and event.button() == QtCore.Qt.LeftButton:
                self._span_dragging = False
                xmin, xmax = region.getRegion()
                if abs(xmax - xmin) > 1e-9:
                    try:
                        self.on_select(xmin, xmax)
                    except Exception:
                        pass
                event.accept()
                return
            try:
                self._orig_wave_vb_mouse_release(event)
            except Exception:
                pass

        # Disconnect any previous handlers
        try:
            vb.scene().sigMouseMoved.disconnect()
        except Exception:
            pass

        # Override ViewBox mouse events
        vb.mousePressEvent   = _mouse_press
        vb.mouseMoveEvent    = _mouse_move
        vb.mouseReleaseEvent = _mouse_release


    def set_spectral_method(self, text):
        self.spectral_method = text
        self.method_combo.setCurrentText(text)
        self.fft_method_combo.setCurrentText(text)
        if self.fft_mode:
            self.update_fft_plot()

    def change_color(self, index):
        key = self.color_combo.currentText()
        self.graph_color = self.color_options.get(key, "#03DFE2")

        # Waveform-only mode: just redraw the main waveform with the new color
        if not getattr(self, "fft_mode", False):
            if hasattr(self, "update_main_waveform_plot"):
                try:
                    self.update_main_waveform_plot()
                except Exception:
                    pass
            return

        # FFT mode: refresh the FFT plot (which already uses ax_fft)
        if hasattr(self, "update_fft_plot"):
            try:
                self.update_fft_plot()
            except Exception:
                pass


    def _ordered_palette(self):
        """Return the graph color palette starting from the user's selection."""

        opts = getattr(self, "color_options", {}) or {}
        keys = list(opts.keys())
        if not keys:
            return ["#03DFE2"]

        current = self.color_combo.currentText() if hasattr(self, "color_combo") else None
        try:
            start_idx = keys.index(current)
        except Exception:
            start_idx = 0

        ordered_keys = keys[start_idx:] + keys[:start_idx]
        return [opts[k] for k in ordered_keys]


    def _on_span_region_changed(self):
        """No-op — span is now driven by mouse press/move/release overrides."""
        pass

    def on_select(self, xmin, xmax):
        self.last_region = (xmin, xmax)
        self.run_analysis(xmin, xmax)

    

    def run_analysis(self, xmin, xmax):
        if self.full_data is None:
            return
        idx_min = int(xmin * self.sample_rate)
        idx_max = int(xmax * self.sample_rate)
        if idx_max <= idx_min or (idx_max - idx_min) < 100:
            QtWidgets.QMessageBox.warning(
                self, "Error",
                "Selected region is invalid or too short for analysis."
            )
            return

        selected_data = self.full_data[idx_min:idx_max]

        # --- Spectral-method branches ---
                # --- Spectral-method branches ---
        if self.spectral_method == "Hilbert":
            # analytic envelope on the FFT axis (bottom panel)
            try:
                from scipy.signal import hilbert
            except Exception:
                QtWidgets.QMessageBox.warning(
                    self, "Error",
                    "SciPy Hilbert transform not available."
                )
                return

            analytic  = hilbert(selected_data)
            envelope  = np.abs(analytic)

            # Make sure we have axes
            self._ensure_main_axis()
            ax = getattr(self, "ax_fft", None)
            if ax is None:
                return

            ax.clear()
            t = np.linspace(xmin, xmax, len(envelope), endpoint=False)
            ax.plot(t, selected_data, pen=pg.mkPen(color=self.graph_color, width=1))
            ax.plot(t, envelope, pen=pg.mkPen(color='#FFD700', width=2))
            ax.setTitle(f"Hilbert Envelope ({xmin:.2f}–{xmax:.2f}s)", color='#FFFFFF')
            ax.setLabel('bottom', 'Time (s)', color='#FFFFFF')
            ax.setLabel('left', 'Amplitude', color='#FFFFFF')
            return


        elif self.spectral_method == "Modulation Spectrum":
            # 1) Analytic signal
            z = hilbert(selected_data)
            # 2) AM vs FM toggle
            if self.mod_type_combo.currentText() == "AM":
                base = np.abs(z)
            else:
                phase = np.unwrap(np.angle(z))
                inst_freq = np.diff(phase) * (self.sample_rate/(2*np.pi))
                # Up-sample back to original length (simple padding)
                inst_freq = np.concatenate([[inst_freq[0]], inst_freq])

                # Derive a baseband-ish signal from FM
                base = inst_freq

            # 3) Remove mean
            mod_sig = base - np.mean(base)
            # 4) Compute spectrum of modulation
            f_mod, Pxx_mod = welch(
                mod_sig,
                fs=self.sample_rate,
                nperseg=256
            )
            # 5) Modulation Index
            total_power = np.trapz(Pxx_mod, f_mod)
            mi = np.sqrt(total_power) / (np.mean(np.abs(selected_data)) + 1e-12)

            # 6) Plot
            pf = self.canvas.plot_fft
            pf.clear()
            pf.plot(f_mod, 10 * np.log10(Pxx_mod),
                    pen=pg.mkPen(color=self.graph_color, width=1))
            pf.setTitle(f"Modulation Spectrum ({self.mod_type_combo.currentText()})  MI={mi:.3f}",
                        color='#FFFFFF')
            pf.setLabel('bottom', 'Modulation Freq (Hz)', color='#FFFFFF')
            pf.setLabel('left', 'Power (dB/Hz)', color='#FFFFFF')

            # Log modulation index
            self.log_measurement_with_project(
                self.file_name,
                "Modulation Index",
                mi,            # store MI in target_frequency
                xmin, xmax,
                (xmax - xmin),
                0.0, 0.0,
                mi,            # store MI again as "measured_voltage"
                False,
                ""
            )
            return

        # --- Time-domain voltage analysis ---
        try:
            target_freq = float(self.freq_entry.text())
            bandwidth   = float(self.bw_entry.text())
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self, "Error", "Invalid frequency or bandwidth value."
            )
            return

        # Auto-detect dominant frequency from FFT (for SPL curve lookup / logging)
        try:
            auto_on = bool(getattr(self, "auto_freq_cb", None) and self.auto_freq_cb.isChecked())
        except Exception:
            auto_on = False

        if self.filter_checkbox.isChecked():
            lowcut  = target_freq - bandwidth/2.0
            highcut = target_freq + bandwidth/2.0
            try:
                filtered_data = bandpass_filter(
                    selected_data,
                    lowcut, highcut,
                    self.sample_rate,
                    order=4
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Filtering Error", str(e))
                return
        else:
            filtered_data = selected_data

        # DAC scaling: convert from raw integer samples to Volts
        if np.issubdtype(self.original_dtype, np.integer):
            try:
                max_voltage = float(self.max_voltage_entry.text())
            except ValueError:
                QtWidgets.QMessageBox.warning(
                    self, "Error",
                    "Invalid max voltage value; please enter a number."
                )
                return
            max_possible = np.iinfo(self.original_dtype).max
            conversion_factor = max_voltage / max_possible
        else:
            conversion_factor = 1.0

        # --- Compute per-channel RMS voltage and log ---
        data = np.asarray(filtered_data)
        selected_channels = _selected_channel_indices(self)

        # Normalise selected channel indices against data shape
        if data.ndim == 1:
            # Single-channel data: pretend it's shape (N, 1)
            data_2d = data.reshape(-1, 1)
        else:
            data_2d = data
        n_chans = data_2d.shape[1]
        selected_channels = [ch for ch in selected_channels if 0 <= ch < n_chans]
        if not selected_channels:
            selected_channels = [0]

        base, ext = _per_channel_basename(self)
        rms_results = []

        for ch in selected_channels:
            ch_samples = data_2d[:, ch]
            ch_voltage = ch_samples * conversion_factor
            ch_rms = float(np.sqrt(np.mean(ch_voltage ** 2)))
            rms_results.append((ch, ch_rms))

            # Log each channel separately, with _chX suffix
            ch_name = f"{base}_ch{ch+1}{ext}"
            # Choose frequency to log:
            # - If Auto freq (FFT) is enabled, derive from the unfiltered selected window for this channel
            # - Otherwise use the manually entered target_freq
            freq_used = target_freq
            if auto_on:
                try:
                    src = np.asarray(selected_data)
                    if src.ndim == 1:
                        src_ch = src
                    else:
                        src_ch = src[:, ch] if src.shape[0] >= src.shape[1] else src[ch, :]
                    src_v = src_ch * conversion_factor
                    dom = self._compute_dominant_freq_from_buffer(src_v, self.sample_rate)
                    if dom is not None:
                        freq_used = float(dom)
                except Exception:
                    pass

            self.log_measurement_with_project(
                ch_name,
                "Waveform",
                freq_used,
                xmin,
                xmax,
                (xmax - xmin),
                float(self.max_voltage_entry.text()),
                float(self.bw_entry.text()),
                ch_rms,
                self.filter_checkbox.isChecked(),
                ""
            )

        # Update UI: single-channel -> label; multi-channel -> popup
        if len(rms_results) == 1:
            ch, val = rms_results[0]
            self.result_label.setText(f"Ch {ch+1} RMS Voltage: {val:.4f} V")
        else:
            lines = [f"Channel {ch+1}: {val:.4f} V" for ch, val in rms_results]
            QtWidgets.QMessageBox.information(
                self,
                "Multi-Channel RMS Voltage",
                "\n".join(lines)
            )
            self.result_label.setText("Multi-channel RMS logged (see popup).")



    def update_analysis(self):
        if self.last_region is None:
            QtWidgets.QMessageBox.information(self, "Info", "Please select a region first.")
            return
        xmin, xmax = self.last_region
        self.run_analysis(xmin, xmax)

    def enter_fft_mode(self):
        if self.full_data is None:
            QtWidgets.QMessageBox.information(self, 'Info', 'Please load a file first.')
            return
        self.fft_mode = True
        self.fft_control_panel.setVisible(True)
        self.fft_control_panel.setMaximumHeight(60)
        self.pulse_control_panel.setVisible(True)
        self.pulse_control_panel.setMaximumHeight(16777215)

        # Show both plots stacked: waveform (top, small) + spectrum (bottom, large)
        pw = self.canvas.plot_waveform
        pf = self.canvas.plot_fft
        pf.setVisible(True)
        self.canvas.glw.ci.layout.setRowStretchFactor(0, 1)
        self.canvas.glw.ci.layout.setRowStretchFactor(1, 3)
        self.canvas.glw.ci.layout.activate()

        # Aliases for compat
        self.ax_waveform = pw
        self.ax_fft      = pf

        # In FFT mode, keep native mouse behavior so left-click measurement works.
        try:
            vb = pw.getViewBox()
            if hasattr(self, '_orig_wave_vb_mouse_press'):
                vb.mousePressEvent = self._orig_wave_vb_mouse_press
            if hasattr(self, '_orig_wave_vb_mouse_move'):
                vb.mouseMoveEvent = self._orig_wave_vb_mouse_move
            if hasattr(self, '_orig_wave_vb_mouse_release'):
                vb.mouseReleaseEvent = self._orig_wave_vb_mouse_release
            pw.setMenuEnabled(True)
            vb.setMenuEnabled(True)
        except Exception:
            pass
        if getattr(self, '_span_region', None) is not None:
            try:
                pw.removeItem(self._span_region)
            except Exception:
                pass
            self._span_region = None

        # Wire click on waveform for echo measurement
        pw.scene().sigMouseClicked.connect(self._pg_on_waveform_click)
        pf.scene().sigMouseClicked.connect(self._pg_on_fft_click)

        self.fft_button.setText('Show Waveform')
        try:
            self.fft_button.clicked.disconnect()
        except Exception:
            pass
        self.fft_button.clicked.connect(self.exit_fft_mode)
        self.sonar_clicks = []
        self.update_fft_slider_range()
        self.update_fft_plot()


    def exit_fft_mode(self):
        self.fft_mode = False
        self.fft_control_panel.setVisible(False)
        self.fft_control_panel.setMaximumHeight(0)
        self.pulse_control_panel.setVisible(False)
        self.pulse_control_panel.setMaximumHeight(0)

        # Hide the FFT plot and collapse its row
        self.canvas.plot_fft.setVisible(False)
        self.canvas.glw.ci.layout.setRowStretchFactor(0, 1)
        self.canvas.glw.ci.layout.setRowStretchFactor(1, 0)
        self.canvas.glw.ci.layout.activate()

        # Disconnect click handlers safely
        try:
            self.canvas.plot_waveform.scene().sigMouseClicked.disconnect(self._pg_on_waveform_click)
        except Exception:
            pass
        try:
            self.canvas.plot_fft.scene().sigMouseClicked.disconnect(self._pg_on_fft_click)
        except Exception:
            pass

        self.ax_main     = self.canvas.plot_waveform
        self.ax_waveform = None
        self.ax_fft      = None

        self.fft_button.setText('Show FFT')
        try:
            self.fft_button.clicked.disconnect()
        except Exception:
            pass
        self.fft_button.clicked.connect(self.enter_fft_mode)

        self.update_main_waveform_plot()
        self.enable_span_selector()


    def update_fft_slider_range(self):
        """
        Sets the FFT time slider range with 32-bit-safe bounds.
        Uses milliseconds (TIME_MULTIPLIER=1000) to avoid overflow.
        """
        # If metadata isn't ready yet, bail quietly
        if not getattr(self, "sample_rate", 0) and not getattr(self, "samplerate", 0):
            return
        sr = int(getattr(self, "sample_rate", getattr(self, "samplerate", 0)))
        if sr <= 0 or not getattr(self, "total_frames", 0):
            return

        INT_MAX = 2_147_483_647
        TIME_MULTIPLIER = int(getattr(self, "TIME_MULTIPLIER", 1000))

        total_duration = self.total_frames / float(sr)  # seconds
        win = float(getattr(self, "fft_window_length", 0.0))
        max_start = max(0.0, total_duration - win)

        safe_max = int(min(round(max_start * self.TIME_MULTIPLIER), INT_MAX))

        try:
            self.fft_time_slider.blockSignals(True)
            # setRange instead of just setMaximum to keep lower bound explicit
            self.fft_time_slider.setRange(0, max(0, safe_max))

            # Clamp current value, expressed in the same units (ms)
            cur = int(round(float(getattr(self, "fft_start_time", 0.0)) * self.TIME_MULTIPLIER))
            if cur < 0: cur = 0
            if cur > safe_max: cur = safe_max
            self.fft_time_slider.setValue(cur)
        finally:
            self.fft_time_slider.blockSignals(False)


    


    def on_echo_click(self, event):
        """Legacy matplotlib-style handler — not used with pyqtgraph."""
        pass

    def on_fft_click(self, event):
        """Legacy matplotlib-style handler — not used with pyqtgraph."""
        pass

    def _pg_on_waveform_click(self, event):
        """PyQtGraph click handler on the waveform plot (FFT mode — echo measurement)."""
        if not getattr(self, 'fft_mode', False):
            return
        pw = self.canvas.plot_waveform
        # Map scene position to plot coordinates
        pos = event.scenePos()
        if not pw.sceneBoundingRect().contains(pos):
            return
        mouse_pt = pw.getViewBox().mapSceneToView(pos)
        t = mouse_pt.x()

        self.sonar_clicks.append(t)
        if len(self.sonar_clicks) < 2:
            QtWidgets.QMessageBox.information(self, 'Echo Measure',
                f"Mark {'start' if len(self.sonar_clicks)==1 else 'stop'} at {t:.3f}s")
            return

        t0, t1 = sorted(self.sonar_clicks[:2])
        dt = t1 - t0
        speed_of_sound = 1500.0
        distance = (dt * speed_of_sound) / 2

        sr = self.sample_rate
        idx0, idx1 = int(t0 * sr), int(t1 * sr)
        seg = self.full_data[idx0:idx1]
        import numpy as np
        nfft = 1 << int(np.ceil(np.log2(max(len(seg), 2))))
        fft_res = np.fft.rfft(seg * np.hanning(len(seg)), n=nfft)
        freqs   = np.fft.rfftfreq(nfft, 1 / sr)
        domf    = self.refine_frequency(fft_res, freqs)

        QtWidgets.QMessageBox.information(self, 'Echo Measurement',
            'Start: {:.3f} s\nStop:  {:.3f} s\nDt: {:.3f} s\nDistance: {:.1f} m\nDominant freq: {:.1f} Hz'.format(
                t0, t1, dt, distance, domf)
        )
        self.log_measurement_with_project(
            self.file_name, 'Echo Measure', domf, t0, t1, dt,
            0.0, 0.0, 0.0, False, '', misc=distance
        )
        self.sonar_clicks.clear()

    def _pg_on_fft_click(self, event):
        """PyQtGraph click handler on FFT plot — sets target frequency field."""
        if not getattr(self, 'fft_mode', False):
            return
        pf = self.canvas.plot_fft
        pos = event.scenePos()
        if not pf.sceneBoundingRect().contains(pos):
            return
        mouse_pt = pf.getViewBox().mapSceneToView(pos)
        freq = mouse_pt.x()
        if hasattr(self, 'freq_entry'):
            self.freq_entry.setText(f'{freq:.2f}')


    def keyPressEvent(self, event):
        from PyQt5 import QtCore

        def _total_duration():
            sr_raw = getattr(self, "sample_rate", 0)
            try:
                sr = int(sr_raw) if sr_raw not in (None, "") else 0
            except (TypeError, ValueError):
                sr = 0
            x  = getattr(self, "full_data", None)
            n  = int(getattr(x, "shape", [0])[0]) if x is not None else 0
            return (n / float(sr)) if (sr > 0 and n > 0) else 0.0

        def _clamp_to_slider(slider, secs):
            tmul = float(getattr(self, "TIME_MULTIPLIER", 1000))
            v = int(round(secs * tmul))
            v = max(slider.minimum(), min(slider.maximum(), v))
            slider.setValue(v)

        handled = False
        td = _total_duration()
        tmul = float(getattr(self, "TIME_MULTIPLIER", 1000))

        # ——————————————————————————————————
        # 1) FFT mode (main tab)
        # ——————————————————————————————————
        if getattr(self, "fft_mode", False):
            step_txt = getattr(self, "scroll_step_entry", None)
            try:
                step = float(step_txt.text()) if step_txt is not None else 1.0
            except Exception:
                step = 1.0

            # current start and max start (so the window fits)
            win = float(getattr(self, "fft_window_length", 1.0) or 1.0)
            cur = self.fft_time_slider.value() / tmul
            max_start = max(0.0, td - max(0.0, win))

            if event.key() in (QtCore.Qt.Key_Left, QtCore.Qt.Key_A):
                new_time = max(0.0, cur - step)
                _clamp_to_slider(self.fft_time_slider, new_time)
                if hasattr(self, "update_fft_plot"): self.update_fft_plot()
                event.accept(); return
            elif event.key() in (QtCore.Qt.Key_Right, QtCore.Qt.Key_D):
                new_time = min(max_start, cur + step)
                _clamp_to_slider(self.fft_time_slider, new_time)
                if hasattr(self, "update_fft_plot"): self.update_fft_plot()
                event.accept(); return
            elif event.key() == QtCore.Qt.Key_PageUp:
                new_time = max(0.0, cur - 1.0)
                _clamp_to_slider(self.fft_time_slider, new_time)
                if hasattr(self, "update_fft_plot"): self.update_fft_plot()
                event.accept(); return
            elif event.key() == QtCore.Qt.Key_PageDown:
                new_time = min(max_start, cur + 1.0)
                _clamp_to_slider(self.fft_time_slider, new_time)
                if hasattr(self, "update_fft_plot"): self.update_fft_plot()
                event.accept(); return
            elif event.key() == QtCore.Qt.Key_Home:
                _clamp_to_slider(self.fft_time_slider, 0.0)
                if hasattr(self, "update_fft_plot"): self.update_fft_plot()
                event.accept(); return
            elif event.key() == QtCore.Qt.Key_End:
                _clamp_to_slider(self.fft_time_slider, max_start)
                if hasattr(self, "update_fft_plot"): self.update_fft_plot()
                event.accept(); return

        # ——————————————————————————————————
        # 2) Advanced tab
        # ——————————————————————————————————
        if hasattr(self, 'advanced_tab') and getattr(self, "tabs", None) is not None and self.tabs.currentWidget() == self.advanced_tab:
            step_txt = getattr(self, "adv_scroll_step_entry", None)
            try:
                step = float(step_txt.text()) if step_txt is not None else 0.001
            except Exception:
                step = 0.001

            win = float(getattr(self, "fft_window_length", 1.0) or 1.0)  # or your adv window length if different
            cur = self.adv_time_slider.value() / tmul
            max_start = max(0.0, td - max(0.0, win))

            if event.key() in (QtCore.Qt.Key_Left, QtCore.Qt.Key_A):
                new_time = max(0.0, cur - step)
                _clamp_to_slider(self.adv_time_slider, new_time)
                if hasattr(self, "update_advanced_plots"): self.update_advanced_plots()
                event.accept(); return
            elif event.key() in (QtCore.Qt.Key_Right, QtCore.Qt.Key_D):
                new_time = min(max_start, cur + step)
                _clamp_to_slider(self.adv_time_slider, new_time)
                if hasattr(self, "update_advanced_plots"): self.update_advanced_plots()
                event.accept(); return
            elif event.key() == QtCore.Qt.Key_PageUp:
                new_time = max(0.0, cur - 1.0)
                _clamp_to_slider(self.adv_time_slider, new_time)
                if hasattr(self, "update_advanced_plots"): self.update_advanced_plots()
                event.accept(); return
            elif event.key() == QtCore.Qt.Key_PageDown:
                new_time = min(max_start, cur + 1.0)
                _clamp_to_slider(self.adv_time_slider, new_time)
                if hasattr(self, "update_advanced_plots"): self.update_advanced_plots()
                event.accept(); return
            elif event.key() == QtCore.Qt.Key_Home:
                _clamp_to_slider(self.adv_time_slider, 0.0)
                if hasattr(self, "update_advanced_plots"): self.update_advanced_plots()
                event.accept(); return
            elif event.key() == QtCore.Qt.Key_End:
                _clamp_to_slider(self.adv_time_slider, max_start)
                if hasattr(self, "update_advanced_plots"): self.update_advanced_plots()
                event.accept(); return

        # Fall back
        super().keyPressEvent(event)


    def find_first_pulse(self):
        try:
            threshold = float(self.pulse_threshold_entry.text())
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Error", "Invalid threshold value.")
            return
        pulse_indices = np.where(np.abs(self.full_data) >= threshold)[0]
        if pulse_indices.size == 0:
            QtWidgets.QMessageBox.information(self, "Info", "No pulses found with the given threshold.")
            return
        gap_threshold_sec = 0.1
        gap_samples = int(gap_threshold_sec * self.sample_rate)
        groups = np.split(pulse_indices, np.where(np.diff(pulse_indices) > gap_samples)[0] + 1)
        self.pulse_indices = np.array([group[0] for group in groups])
        self.current_pulse_index = 0
        pulse_time = self.full_time[self.pulse_indices[self.current_pulse_index]]
        new_start_time = max(0, pulse_time - self.fft_window_length/2)
        self.fft_time_slider.setValue(int(new_start_time * self.TIME_MULTIPLIER))
        self.update_fft_plot()

    def next_pulse(self):
        if self.pulse_indices.size == 0:
            QtWidgets.QMessageBox.information(self, "Info", "No pulses available. Please use 'Find Pulse' first.")
            return
        if self.current_pulse_index is None:
            self.current_pulse_index = 0
        else:
            self.current_pulse_index += 1
            if self.current_pulse_index >= self.pulse_indices.size:
                QtWidgets.QMessageBox.information(self, "Info", "This is the last pulse.")
                self.current_pulse_index = self.pulse_indices.size - 1
                return
        pulse_time = self.full_time[self.pulse_indices[self.current_pulse_index]]
        new_start_time = max(0, pulse_time - self.fft_window_length/2)
        self.fft_time_slider.setValue(int(new_start_time * self.TIME_MULTIPLIER))
        self.update_fft_plot()

    def prev_pulse(self):
        if self.pulse_indices.size == 0:
            QtWidgets.QMessageBox.information(self, "Info", "No pulses available. Please use 'Find Pulse' first.")
            return
        if self.current_pulse_index is None:
            self.current_pulse_index = 0
        else:
            self.current_pulse_index -= 1
            if self.current_pulse_index < 0:
                QtWidgets.QMessageBox.information(self, "Info", "This is the first pulse.")
                self.current_pulse_index = 0
                return
        pulse_time = self.full_time[self.pulse_indices[self.current_pulse_index]]
        new_start_time = max(0, pulse_time - self.fft_window_length/2)
        self.fft_time_slider.setValue(int(new_start_time * self.TIME_MULTIPLIER))
        self.update_fft_plot()

    def analyze_voltage_from_fft(self):
        if self.full_data is None:
            QtWidgets.QMessageBox.information(self, "Info", "No data loaded.")
            return
        try:
            window_length = float(self.fft_length_entry.text())
        except ValueError:
            QtWidgets.QMessageBox.critical(self, "Error", "Invalid FFT window length value.")
            return
        try:
            start_time = self.fft_time_slider.value() / self.TIME_MULTIPLIER
        except ValueError:
            start_time = 0.0
        start_idx = int(start_time * self.sample_rate)
        end_idx = start_idx + int(window_length * self.sample_rate)
        if end_idx > len(self.full_data):
            end_idx = len(self.full_data)
        segment = self.full_data[start_idx:end_idx]
        if self.filter_checkbox.isChecked():
            try:
                target_freq = float(self.freq_entry.text())
                bandwidth = float(self.bw_entry.text())
            except ValueError:
                QtWidgets.QMessageBox.warning(self, "Error", "Invalid frequency or bandwidth value.")
                return
            lowcut = target_freq - bandwidth / 2.0
            highcut = target_freq + bandwidth / 2.0
            try:
                segment = bandpass_filter(segment, lowcut, highcut, self.sample_rate, order=4)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Filtering Error", str(e))
                return

        if np.issubdtype(self.original_dtype, np.integer):
            try:
                max_voltage = float(self.max_voltage_entry.text())
            except ValueError:
                QtWidgets.QMessageBox.warning(self, "Error", "Invalid max voltage value.")
                return
            max_possible = np.iinfo(self.original_dtype).max
            conversion_factor = max_voltage / max_possible
        else:
            conversion_factor = 1.0

        voltage_data = segment * conversion_factor
        rms_voltage = np.sqrt(np.mean(voltage_data**2))
        self.result_label.setText(f"RMS Voltage: {rms_voltage:.4f} V")

        base, _ = os.path.splitext(self.file_name)
        directory = os.path.dirname(self.current_file_path)
        screenshot_filename = f"{base}_voltage_{start_time:.2f}_{start_time+window_length:.2f}.png"
        screenshot_filepath = os.path.join(directory, screenshot_filename)
        # Screenshot via pyqtgraph
        # Screenshot via pyqtgraph
        try:
            import pyqtgraph.exporters
            exporter = pg.exporters.ImageExporter(self.canvas.glw.scene())
            exporter.export(screenshot_filepath)
        except Exception:
            pass
        # --- choose frequency to log (manual vs auto FFT-dominant) ---
        freq_to_log = None
        try:
            freq_to_log = float(self.freq_entry.text())
        except Exception:
            freq_to_log = None

        try:
            auto_on = bool(getattr(self, "auto_freq_cb", None) and self.auto_freq_cb.isChecked())
        except Exception:
            auto_on = False

        if auto_on:
            try:
                f_dom = self._estimate_dominant_frequency(
                    start_time,
                    start_time + window_length,
                    window_length,
                    getattr(self, "selected_channel", None)
                )
                if f_dom is not None and np.isfinite(f_dom) and f_dom > 0:
                    freq_to_log = float(f_dom)
            except Exception:
                pass

        if freq_to_log is None:
            freq_to_log = 0.0

        self.log_measurement_with_project(
            self.file_name,
            "Voltage",
            float(freq_to_log),
            start_time,
            start_time + window_length,
            window_length,
            float(self.max_voltage_entry.text()),
            float(self.bw_entry.text()),
            rms_voltage,
            self.filter_checkbox.isChecked(),
            screenshot_filepath
        )
    def auto_analyze_pulses(self):
        if self.full_data is None or self.pulse_indices.size == 0:
            QtWidgets.QMessageBox.critical(self, "Error", "No pulses available. Please use 'Find Pulse' first.")
            return
        try:
            window_length = float(self.fft_length_entry.text())
        except ValueError:
            QtWidgets.QMessageBox.critical(self, "Error", "Invalid FFT window length value.")
            return

        start_index = self.current_pulse_index if self.current_pulse_index is not None else 0
        current_pulse_time = self.full_time[self.pulse_indices[start_index]]
        try:
            current_window_start = self.fft_time_slider.value() / self.TIME_MULTIPLIER
        except:
            current_window_start = current_pulse_time
        offset = current_window_start - current_pulse_time

        results = []
        total_duration = self.full_time[-1]

        for i in range(start_index, len(self.pulse_indices)):
            pulse_idx = self.pulse_indices[i]
            pulse_time = self.full_time[pulse_idx]
            window_start = max(0.0, pulse_time + offset)
            if window_start + window_length > total_duration:
                break

            s_idx = int(window_start * self.sample_rate)
            e_idx = s_idx + int(window_length * self.sample_rate)
            if e_idx > len(self.full_data):
                break
            segment = self.full_data[s_idx:e_idx]
            if segment.size == 0:
                continue

            # FFT & dom freq
            n = len(segment)
            nfft = 16 * n
            fft_result = np.fft.rfft(segment * np.hanning(n), n=nfft)
            freqs = np.fft.rfftfreq(nfft, d=1/self.sample_rate)
            dom_freq = self.refine_frequency(fft_result, freqs)

            # compute RMS voltage
            if np.issubdtype(self.original_dtype, np.integer):
                try:
                    current_max_v = float(self.max_voltage_entry.text())
                except:
                    current_max_v = 2.5
                conv = current_max_v / np.iinfo(self.original_dtype).max
            else:
                conv = 1.0
            rms_voltage = np.sqrt(np.mean((segment * conv) ** 2))

            # create detailed screenshot at lower DPI
            padded = 1.0
            pad_start = max(0, window_start - padded * window_length)
            pad_end = min(total_duration, window_start + window_length + padded * window_length)
            p_s = int(pad_start * self.sample_rate)
            p_e = int(pad_end * self.sample_rate)
            t_pad = np.linspace(pad_start, pad_end, p_e - p_s, endpoint=False)

            fig_temp, (ax1, ax2) = plt.subplots(2, 1,
                                                figsize=(8, 8),
                                                dpi=80,
                                                facecolor="#19232D")
            # waveform + highlight
            ax1.plot(t_pad, self.full_data[p_s:p_e],
                     color=self.graph_color, lw=1)
            ax1.set_facecolor("#19232D")
            ax1.set_title(f"Padded Waveform {pad_start:.2f}–{pad_end:.2f}s", color="white")
            ax1.set_xlabel("Time (s)", color="white")
            ax1.set_ylabel("Amplitude", color="white")
            ax1.tick_params(colors="white")
            ax1.axvspan(window_start, window_start + window_length,
                        color=lighten_color(self.graph_color, 0.4), alpha=0.5)

            # FFT panel
            ax2.plot(freqs, np.abs(fft_result),
                     color=self.graph_color, lw=1.5)
            ax2.set_facecolor("#19232D")
            ax2.set_title(f"FFT (Dom Freq {dom_freq:.2f} Hz)", color="white")
            ax2.set_xlabel("Frequency (Hz)", color="white")
            ax2.set_ylabel("Magnitude", color="white")
            ax2.tick_params(colors="white")

            fig_temp.tight_layout()
            # save
            base, _ = os.path.splitext(self.file_name)
            out_dir = self._project_subdir("screenshots") or os.path.join(os.path.dirname(self.current_file_path), "analysis")
            os.makedirs(out_dir, exist_ok=True)
            shot = os.path.join(out_dir,
                f"{base}_{pad_start:.2f}-{pad_end:.2f}.png")
            fig_temp.savefig(shot)
            plt.close(fig_temp)

            results.append((pulse_time, dom_freq, rms_voltage, window_start, shot))

        # show results dialog
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Auto Analysis Results")
        v = QtWidgets.QVBoxLayout(dlg)
        txt = QtWidgets.QPlainTextEdit()
        txt.setReadOnly(True)
        out = "Pulse Time   Dom Freq (Hz)   RMS Voltage (V)\n"
        for t0, f0, r0, w0, _ in results:
            out += f"{t0:8.4f}      {f0:8.4f}       {r0:8.4f}\n"
        txt.setPlainText(out)
        v.addWidget(txt)

        h = QtWidgets.QHBoxLayout()
        btn_keep = QtWidgets.QPushButton("Accept Results")
        btn_discard = QtWidgets.QPushButton("Discard Results")
        h.addWidget(btn_keep)
        h.addWidget(btn_discard)
        v.addLayout(h)

        def keep():
            self.store_auto_results(results, window_length)
            # export Excel to parent dir
            base, _ = os.path.splitext(self.file_name)
            parent = os.path.dirname(self.current_file_path)
            if os.path.basename(parent).lower() == "analysis":
                parent = os.path.dirname(parent)
            excel = os.path.join(parent, f"{base}_auto_analysis.xlsx")
            pd.DataFrame(results,
                         columns=["Pulse Time","Dom Freq","RMS","Window Start","Screenshot"]
            ).to_excel(excel, index=False)
            dlg.accept()

        btn_keep.clicked.connect(keep)
        btn_discard.clicked.connect(dlg.reject)
        dlg.exec_()

    def store_auto_results(self, results, window_length):
        for pt, freq, rms, ws, screenshot in results:
            self.log_measurement_with_project(
                self.file_name,
                "Auto Analysis",
                freq,
                ws, ws + window_length,
                window_length,
                float(self.max_voltage_entry.text()),
                float(self.bw_entry.text()),
                rms,
                self.filter_checkbox.isChecked(),
                screenshot
            )
        QtWidgets.QMessageBox.information(self, "Results Stored", "Auto analysis results stored in the database.")

    @staticmethod
    def refine_frequency(fft_result, frequencies):
        mag = np.abs(fft_result)
        k = np.argmax(mag)
        if k <= 0 or k >= len(mag) - 1:
            return frequencies[k]
        log_alpha = np.log(mag[k - 1])
        log_beta = np.log(mag[k])
        log_gamma = np.log(mag[k + 1])
        p = 0.5 * (log_alpha - log_gamma) / (log_alpha - 2 * log_beta + log_gamma)
        return frequencies[k] + p * (frequencies[1] - frequencies[0])

    def open_file(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select WAV File",
            self._dialog_default_dir("originals"),
            "WAV Files (*.wav)",
        )
        if not fname:
            return
        self.load_wav_file(fname)

    def _set_default_channels(self):
        """Prime whatever channel UI exists (menu and/or list) after file load."""
        n = int(getattr(self, 'channels', 1))
        # Compact menu
        if hasattr(self, 'chan_actions'):
            for i, act in enumerate(self.chan_actions):
                act.setEnabled(i < n)
                act.setChecked(i < min(n, 8))
            if hasattr(self, '_update_channel_sel_label'):
                self._update_channel_sel_label()
        # List widget
        if hasattr(self, 'channel_picker'):
            try:
                self.channel_picker.clear()
                for ch in range(n):
                    self.channel_picker.addItem(f"Ch {ch+1}")
                for i in range(min(n, 8)):
                    self.channel_picker.item(i).setSelected(True)
                is_multi = (n > 1)
                if hasattr(self, 'channel_label'): self.channel_label.setVisible(is_multi)
                # channel_mode widget removed; no visibility toggle needed
                self.channel_picker.setVisible(is_multi)
            except Exception:
                pass

    def prompt_split_channels(self, filepath, data, sample_rate):
        """
        If `data` is multi-channel, ask the user whether to split into separate mono WAVs.
        Writes files named <basename>_channel1.wav, etc., into an 'analysis' subfolder.
        """
        # only act on true multi-channel
        if data.ndim < 2:
            return

        nchan = data.shape[1]
        resp = QtWidgets.QMessageBox.question(
            self,
            "Split Channels?",
            f"The file has {nchan} channels.\nDo you want to write each channel out as a mono WAV?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if resp != QtWidgets.QMessageBox.Yes:
            return

        base     = os.path.splitext(os.path.basename(filepath))[0]
        out_dir = self._project_subdir("modified") or os.path.join(os.path.dirname(filepath), "analysis")
        os.makedirs(out_dir, exist_ok=True)

        for ch in range(nchan):
            chan_data = data[:, ch]
            fname     = f"{base}_channel{ch+1}.wav"
            fpath     = os.path.join(out_dir, fname)
            # use wavfile.write, not wavfile_write
            wavfile.write(fpath, sample_rate, chan_data)

        QtWidgets.QMessageBox.information(
            self, "Channels Split",
            f"Wrote {nchan} mono files to:\n{out_dir}"
        )

    def load_wav_file(self, filepath):
        """
        Load WAV, store self.full_data, self.full_time, self.sample_rate, etc.
        Then decimate and plot the main waveform.
        """
        import os
        import numpy as np
        from scipy.io import wavfile

        # ---- constants / defaults ----
        # keep ms resolution to avoid 32-bit overflow on Qt widgets
        self.TIME_MULTIPLIER = getattr(self, "TIME_MULTIPLIER", 1000)   # ms
        self.fft_window_length = float(getattr(self, "fft_window_length", 1.0))
        self.fft_start_time = float(getattr(self, "fft_start_time", 0.0))

        self.loaded_file = filepath
        self.current_file_path = filepath

        self.file_name = os.path.splitext(os.path.basename(filepath))[0]
        # Mirror the source into the project directory (if applicable)
        cached_path = self._cache_file_into_project(filepath)
        if cached_path:
            filepath = cached_path
            self.loaded_file = filepath
        import warnings
        warnings.filterwarnings('ignore', message='.*Chunk .* not understood.*', category=UserWarning)
        self.sample_rate, data = wavfile.read(filepath)          # NOTE: sample_rate (not samplerate)
        self.wav_sample_rate = int(self.sample_rate)
        self.original_dtype = data.dtype

        # if multi‐channel, offer to split
        if data.ndim > 1:
            self.prompt_split_channels(filepath, data, self.sample_rate)
            # keep multi‑channel data in memory; do NOT collapse to mono
            # data = data[:, 0]
# store original dtype & convert to float64
        self.original_dtype = data.dtype
        # keep shape: (N,) for mono or (N, C) for multi-channel
        self.full_data = data.astype(np.float64)
        # channel count
        self.channels = int(self.full_data.shape[1]) if self.full_data.ndim > 1 else 1
        if hasattr(self, '_set_default_channels'):
            self._set_default_channels()
        if hasattr(self, '_populate_spec_channel_combo'):
            self._populate_spec_channel_combo()
# --- set these BEFORE calling update_fft_slider_range() ---
        self.total_frames   = int(self.full_data.shape[0])       # used by slider updater
        self.samplerate     = int(self.sample_rate)              # keep BOTH names in sync if other code uses .samplerate
        # ----------------------------------------------------------

        self.full_time      = np.arange(self.total_frames, dtype=np.float64) / float(self.sample_rate)

        # compute Nyquist frequency and update FFT axis entries
        nyquist = self.sample_rate / 2.0
        self.fft_xmin_entry.setText("0")
        self.fft_xmax_entry.setText(str(int(nyquist)))
        if hasattr(self, 'spec_max_freq_entry'):
            self.spec_max_freq_entry.setText(str(int(nyquist)))
        if hasattr(self, 'adv_xmin_entry'):
            self.adv_xmin_entry.setText("0")
            self.adv_xmax_entry.setText(str(int(nyquist)))

        # Let the slider range be updated (now safe, because total_frames/sample_rate are set)
        if hasattr(self, 'update_fft_slider_range'):
            self.update_fft_slider_range()

        # decimate for display
        max_points = 100000
        decimation_factor = max(1, self.total_frames // max_points)
        self.decimated_data = self.full_data[::decimation_factor]
        self.decimated_time = self.full_time[::decimation_factor]

        # initial plot
        # initial plot – use the new multi-channel-aware main plot
        if not getattr(self, "fft_mode", False) and hasattr(self, "update_main_waveform_plot"):
            self.update_main_waveform_plot()


        if hasattr(self, '_refresh_multi_plot'):
            self._refresh_multi_plot()

        if hasattr(self, '_refresh_channel_quickview'):
            self._refresh_channel_quickview()

        if hasattr(self, '_refresh_spectrogram_gallery'):
            self._refresh_spectrogram_gallery()
        if hasattr(self, '_set_spec_annotation_controls_enabled'):
            self._set_spec_annotation_controls_enabled(False)

    def update_spl_filter_options(self):
        """
        Rebuild the File filter combo so it ONLY shows files present in the
        SPL calculations table. Keeps prior selection if still valid;
        otherwise defaults to 'All'.
        """
        import sqlite3

        table = "spl_calculations"

        # Remember current selection to restore if still present
        prev = None
        if hasattr(self, "spl_file_filter_combo") and self.spl_file_filter_combo.count() > 0:
            prev = self.spl_file_filter_combo.currentText()

        # Fetch distinct filenames for THIS table only (optionally scoped to the current project)
        project_id = getattr(self, "current_project_id", None)
        if project_id is None and getattr(self, "current_project_name", None):
            project_id = self._get_project_id(self.current_project_name)
            self.current_project_id = project_id

        conn = sqlite3.connect(DB_FILENAME)
        cur  = conn.cursor()
        try:
            if project_id is None:
                cur.execute(f'SELECT DISTINCT file_name FROM {table} WHERE file_name IS NOT NULL ORDER BY file_name')
            else:
                cur.execute(
                    f"""
                    SELECT DISTINCT s.file_name
                    FROM {table} AS s
                    JOIN measurements AS m ON s.voltage_log_id = m.id
                    JOIN project_items AS p ON p.file_name = m.file_name AND p.method = m.method
                    WHERE p.project_id = ? AND s.file_name IS NOT NULL
                    ORDER BY s.file_name
                    """,
                    (project_id,)
                )
            files = [r[0] for r in cur.fetchall() if r[0]]
        finally:
            conn.close()

        # Rebuild combo WITHOUT emitting signals
        self.spl_file_filter_combo.blockSignals(True)
        try:
            self.spl_file_filter_combo.clear()
            self.spl_file_filter_combo.addItem("All")
            self.spl_file_filter_combo.addItems(files)
            # restore previous selection if still valid
            if prev and prev in files:
                idx = self.spl_file_filter_combo.findText(prev)
                if idx >= 0:
                    self.spl_file_filter_combo.setCurrentIndex(idx)
            else:
                self.spl_file_filter_combo.setCurrentIndex(0)  # "All"
        finally:
            self.spl_file_filter_combo.blockSignals(False)

    

    def setup_spl_tab(self):
        # Pagination defaults (first load must be paged)
        self.spl_page = 1
        if not hasattr(self, "spl_per_page"):
            self.spl_per_page = 200
        self.spl_total_rows = 0
        if not hasattr(self, "spl_sig_figs"):
            self.spl_sig_figs = 3

        spl_tab = QtWidgets.QWidget()
        self.spl_tab = spl_tab
        spl_tab_layout = QtWidgets.QVBoxLayout(spl_tab)

        # ---- Top filter row ----
        top = QtWidgets.QHBoxLayout()

        top.addSpacing(10); top.addWidget(QtWidgets.QLabel("File:"))
        self.spl_file_filter_combo = QtWidgets.QComboBox()
        self.spl_file_filter_combo.addItem("All")
        top.addWidget(self.spl_file_filter_combo)

        top.addSpacing(10); top.addWidget(QtWidgets.QLabel("Method:"))
        self.spl_method_filter_combo = QtWidgets.QComboBox()
        self.spl_method_filter_combo.addItem("All")
        top.addWidget(self.spl_method_filter_combo)

        # Results per page on same row
        top.addSpacing(16); top.addWidget(QtWidgets.QLabel("Results per page:"))
        self.spl_per_page_spin = QtWidgets.QSpinBox()
        self.spl_per_page_spin.setRange(10, 5000)
        self.spl_per_page_spin.setSingleStep(10)
        self.spl_per_page_spin.setValue(self.spl_per_page)
        top.addWidget(self.spl_per_page_spin)

        top.addSpacing(12); top.addWidget(QtWidgets.QLabel("Sig figs:"))
        self.spl_sigfig_combo = QtWidgets.QComboBox()
        for n in (2, 3, 4, 5, 6):
            self.spl_sigfig_combo.addItem(str(n))
        self.spl_sigfig_combo.setCurrentText(str(self.spl_sig_figs))
        top.addWidget(self.spl_sigfig_combo)

        top.addStretch()

        # Plot/Delete buttons (unchanged)
        # Analyze SPL from Voltage Log (restored)
        self.analyze_spl_log_button = QtWidgets.QPushButton("Analyze SPL from Voltage Log")
        
        self.btn_plot_time  = QtWidgets.QPushButton("Plot SPL vs Time")
        self.btn_plot_freq  = QtWidgets.QPushButton("Plot SPL vs Frequency")
        self.btn_plot_multi = QtWidgets.QPushButton("Plot Multi-Freq vs Time")
        self.btn_apply_spl_nf = QtWidgets.QPushButton("Apply Near-field")
        self.btn_del_selected = QtWidgets.QPushButton("Delete Selected")
        self.btn_del_all      = QtWidgets.QPushButton("Delete All (filtered)")
        top.addWidget(self.analyze_spl_log_button)
        top.addWidget(self.btn_plot_time)
        top.addWidget(self.btn_plot_freq)
        top.addWidget(self.btn_plot_multi)
        top.addWidget(self.btn_apply_spl_nf)
        top.addWidget(self.btn_del_selected)
        top.addWidget(self.btn_del_all)

        spl_tab_layout.addLayout(top)

        # ---- Table ----
        self.spl_table = QtWidgets.QTableWidget()
        self.spl_table.setSelectionBehavior(QtWidgets.QTableWidget.SelectRows)
        self.spl_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.spl_table.setAlternatingRowColors(True)
        spl_tab_layout.addWidget(self.spl_table)

            # ---- Bottom row: left actions + centered pager ----
        bottom_row = QtWidgets.QHBoxLayout()

        # Left action group (Export)
        left_actions = QtWidgets.QHBoxLayout()
        self.spl_export_btn        = QtWidgets.QPushButton("Export to CSV")
        left_actions.addWidget(self.spl_export_btn)
        left_actions.addStretch()

        bottom_row.addLayout(left_actions)

        # Centered pager
        pager_center = QtWidgets.QHBoxLayout()
        pager_center.addStretch()
        self.spl_prev_btn = QtWidgets.QPushButton("Prev")
        self.spl_page_label = QtWidgets.QLabel("Page 1 / 1 (0 rows)")
        self.spl_next_btn = QtWidgets.QPushButton("Next")
        pager_center.addWidget(self.spl_prev_btn)
        pager_center.addSpacing(12)
        pager_center.addWidget(self.spl_page_label)
        pager_center.addSpacing(12)
        pager_center.addWidget(self.spl_next_btn)
        pager_center.addStretch()

        bottom_row.addStretch()          # push pager to screen center
        bottom_row.addLayout(pager_center)
        bottom_row.addStretch()

        spl_tab_layout.addLayout(bottom_row)


        # ---- FILL FILTERS with signals BLOCKED so they don't auto-populate everything ----
        self.spl_file_filter_combo.blockSignals(True)
        self.spl_method_filter_combo.blockSignals(True)

        # Fill filter combos
        self.update_spl_filter_options()         # this should not call populate itself
        self.update_spl_method_filter_options()  # same

        # First paginated fill (LIMIT/OFFSET)
        self.spl_page = 1
        self.populate_spl_table()

        # Now wire signals (AFTER first paged fill)
        self.spl_file_filter_combo.blockSignals(False)
        self.spl_method_filter_combo.blockSignals(False)
 
        self.spl_file_filter_combo.currentTextChanged.connect(self._on_spl_filters_changed)
        self.spl_method_filter_combo.currentTextChanged.connect(self._on_spl_filters_changed)

        self.spl_per_page_spin.valueChanged.connect(self._on_spl_per_page_changed)
        self.spl_sigfig_combo.currentTextChanged.connect(self._on_spl_sigfigs_changed)
        self.spl_prev_btn.clicked.connect(self._on_spl_prev)
        self.spl_next_btn.clicked.connect(self._on_spl_next)

        self.analyze_spl_log_button.clicked.connect(self.analyze_spl_from_voltage_log)
        self.btn_plot_time.clicked.connect(self.plot_spl_over_time)
        self.btn_plot_freq.clicked.connect(self.plot_spl_vs_frequency)
        self.btn_plot_multi.clicked.connect(self.plot_spl_multi_freq_over_time)
        self.btn_apply_spl_nf.clicked.connect(self.apply_near_field_to_spl_results)
        self.btn_del_selected.clicked.connect(self.delete_selected_spl)
        self.btn_del_all.clicked.connect(self.delete_all_spl)
        # Bottom-left actions
        self.spl_export_btn.clicked.connect(self.export_spl_to_csv)


        # ---- Initial population (paged by default) ----
        self.update_spl_filter_options()
        self.update_spl_method_filter_options()
        self.spl_page = 1
        self.populate_spl_table()


    def _on_spl_sigfigs_changed(self, value):
        """Update SPL significant figures and refresh the current page."""

        try:
            sig_figs = max(1, int(value))
        except (TypeError, ValueError):
            sig_figs = 3

        self.spl_sig_figs = sig_figs
        # Only the table rendering needs to update
        self.populate_spl_table()

    def _format_spl_value(self, value):
        """Format numeric SPL values with the configured significant figures."""

        if value is None:
            return ""

        sig_figs = getattr(self, "spl_sig_figs", 3) or 3

        try:
            num = float(value)
            if math.isfinite(num):
                return f"{num:.{sig_figs}g}"
        except (TypeError, ValueError):
            pass

        return str(value)


    def _spl_current_where(self, override_files=None):
        """Return (base_table, from_clause, where_sql, params) matching current SPL filters.

        Uses EXISTS predicates (instead of JOIN fan-out) so each SPL row appears once.
        """
        base_table = "spl_calculations"

        file_filter   = self.spl_file_filter_combo.currentText() if hasattr(self, "spl_file_filter_combo") else "All"
        method_filter = self.spl_method_filter_combo.currentText() if hasattr(self, "spl_method_filter_combo") else "All"

        project_id = getattr(self, "current_project_id", None)
        if project_id is None and getattr(self, "current_project_name", None):
            project_id = self._get_project_id(self.current_project_name)
            self.current_project_id = project_id

        from_clause = "spl_calculations AS s"
        where_clauses, params = [], []

        if file_filter and file_filter != "All" and not override_files:
            where_clauses.append("s.file_name = ?")
            params.append(file_filter)
        elif override_files:
            placeholders = ",".join(["?"] * len(override_files))
            where_clauses.append(f"s.file_name IN ({placeholders})")
            params.extend(override_files)

        if method_filter and method_filter != "All":
            where_clauses.append(
                """
                EXISTS (
                    SELECT 1
                    FROM measurements m
                    WHERE m.id = s.voltage_log_id
                      AND m.method = ?
                )
                """
            )
            params.append(method_filter)

        if project_id is not None:
            where_clauses.append(
                """
                EXISTS (
                    SELECT 1
                    FROM measurements m
                    JOIN project_items p
                      ON p.file_name = m.file_name
                     AND p.method    = m.method
                    WHERE m.id = s.voltage_log_id
                      AND p.project_id = ?
                )
                """
            )
            params.append(project_id)

        where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        return base_table, from_clause, where_sql, params


    def _spl_available_files_for_plot(self):
        """Return distinct SPL file names honoring project/method filters."""

        import sqlite3

        method_filter = self.spl_method_filter_combo.currentText() if hasattr(self, "spl_method_filter_combo") else "All"
        project_id = getattr(self, "current_project_id", None)
        if project_id is None and getattr(self, "current_project_name", None):
            project_id = self._get_project_id(self.current_project_name)
            self.current_project_id = project_id

        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        try:
            from_clause = "spl_calculations AS s"
            where_clauses, params = ["s.file_name IS NOT NULL"], []

            if method_filter and method_filter != "All":
                from_clause += " JOIN measurements AS m ON s.voltage_log_id = m.id"
                where_clauses.append("m.method = ?")
                params.append(method_filter)
            if project_id is not None:
                where_clauses.append("""
                    EXISTS (
                        SELECT 1
                        FROM project_items p
                        WHERE p.project_id = ?
                        AND p.file_name = m.file_name
                        AND p.method    = m.method
                    )
                """)
                params.append(project_id)


            where_sql = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""
            cur.execute(
                f"SELECT DISTINCT s.file_name FROM {from_clause}{where_sql} ORDER BY s.file_name",
                tuple(params),
            )
            return [r[0] for r in cur.fetchall() if r[0]]
        finally:
            conn.close()

    def _resolve_spl_plot_files(self, allow_multi=True):
        """Determine which SPL files to plot based on current filters."""

        current = self.spl_file_filter_combo.currentText() if hasattr(self, "spl_file_filter_combo") else None
        if current and current not in ("All", ""):
            return [current]

        files = self._spl_available_files_for_plot()
        return self._choose_files_for_plot(
            files,
            "Select SPL Files to Plot",
            max_files=10,
            multi=allow_multi,
        )


    def apply_near_field_to_spl_results(self):
        """Apply near-field correction to current SPL results (filtered or selected)."""
        import sqlite3

        base_table, from_clause, where_sql, params = self._spl_current_where()

        # If rows are selected, restrict to those IDs
        selected_ids = []
        if self.spl_table.selectionModel() is not None:
            selected_ids = [int(idx.data()) for idx in self.spl_table.selectionModel().selectedRows(0)]

        where_extra = ""
        if selected_ids:
            placeholders = ",".join(["?"] * len(selected_ids))
            where_extra = (" AND " if where_sql else " WHERE ") + f"s.id IN ({placeholders})"
            params = list(params) + selected_ids

        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        cur.execute(
            f"SELECT s.id, s.target_frequency, s.spl FROM {from_clause}{where_sql}{where_extra}",
            tuple(params),
        )
        rows = cur.fetchall()
        conn.close()

        if not rows:
            QtWidgets.QMessageBox.information(self, "Near-field", "No SPL rows match the current filter/selection.")
            return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Apply Near-field Correction")
        dlg.resize(520, 320)
        vbox = QtWidgets.QVBoxLayout(dlg)

        form = QtWidgets.QFormLayout()

        radius_edit = QtWidgets.QLineEdit("0.10")
        range_edit  = QtWidgets.QLineEdit("1.50")
        depth_edit = QtWidgets.QLineEdit("")
        try:
            default_depth = self._default_hydrophone_depth_m()
            if default_depth is not None:
                depth_edit.setText(f"{default_depth:.3f}")
        except Exception:
            pass

        use_ctd_chk = QtWidgets.QCheckBox("Use sound speed from CTD cast")
        use_ctd_chk.setChecked(True)

        ctd_combo = QtWidgets.QComboBox()
        ctd_combo.addItem("(none)", userData=None)
        try:
            conn0 = sqlite3.connect(DB_FILENAME); cur0 = conn0.cursor()
            cur0.execute("SELECT id, name, dt_utc FROM ctd_profiles ORDER BY dt_utc DESC, id DESC")
            for cid, nm, dt in cur0.fetchall():
                label = f"{nm or 'CTD'}  [{dt or ''}]"
                ctd_combo.addItem(label, userData=int(cid))
            conn0.close()
        except Exception:
            pass

        manual_c_chk = QtWidgets.QCheckBox("Override CTD: use manual sound speed")
        manual_c_chk.setChecked(False)
        c_edit = QtWidgets.QLineEdit("1500")

        def _toggle_manual_c(checked):
            c_edit.setEnabled(checked)
        _toggle_manual_c(manual_c_chk.isChecked())
        manual_c_chk.toggled.connect(_toggle_manual_c)

        form.addRow("Projector radius a (m):", radius_edit)
        form.addRow("Measurement range R (m):", range_edit)
        form.addRow("Hydrophone depth (m, optional):", depth_edit)
        form.addRow(use_ctd_chk)
        form.addRow("CTD cast:", ctd_combo)
        form.addRow(manual_c_chk)
        form.addRow("Manual c (m/s):", c_edit)

        vbox.addLayout(form)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        vbox.addWidget(btns)

        def _apply():
            try:
                a_m = float(radius_edit.text())
                R_m = float(range_edit.text())
            except Exception:
                QtWidgets.QMessageBox.warning(dlg, "Near-field", "Enter numeric radius and range.")
                return

            if a_m <= 0 or R_m <= 0:
                QtWidgets.QMessageBox.warning(dlg, "Near-field", "Radius and range must be positive.")
                return

            use_ctd = use_ctd_chk.isChecked()
            ctd_id = ctd_combo.currentData() if use_ctd else None
            manual_override = manual_c_chk.isChecked()
            try:
                c_ms = float(c_edit.text()) if c_edit.text() else 1500.0
            except Exception:
                c_ms = 1500.0
            c_source = "manual"
            c_depth = None
            depth_for_ctd = None
            if depth_edit.text().strip():
                try:
                    depth_for_ctd = float(depth_edit.text())
                except Exception:
                    QtWidgets.QMessageBox.warning(dlg, "Near-field", "Hydrophone depth must be numeric.")
                    return
            if (not manual_override) and use_ctd and ctd_id:
                depth_query = depth_for_ctd if depth_for_ctd is not None else R_m
                c_val, c_depth_sel, c_src = _ctd_get_c_ms_at_depth(ctd_id, depth_query)
                if c_val is not None:
                    c_ms = c_val
                    c_source = c_src or "ctd"
                    c_depth = c_depth_sel

            conn2 = sqlite3.connect(DB_FILENAME)
            cur2 = conn2.cursor()
            try:
                _ensure_spl_nf_columns()
            except Exception:
                pass

            updated = 0
            for sid, freq, spl_val in rows:
                try:
                    f = float(freq)
                    spl_raw = float(spl_val)
                except Exception:
                    continue
                if f <= 0:
                    continue
                delta_nf_db = float(nf_axial_piston_delta_db(f, a_m, R_m, c_ms))
                spl_nf = spl_raw - delta_nf_db
                cur2.execute(
                    """
                    UPDATE spl_calculations
                    SET spl_nf = ?, nf_enabled = 1, nf_delta_db = ?,
                        nf_radius_m = ?, nf_range_m = ?, nf_c_ms = ?,
                        nf_c_source = ?, nf_c_depth_m = ?, nf_ctd_id = ?
                    WHERE id = ?
                    """,
                    (
                        float(spl_nf), float(delta_nf_db), float(a_m), float(R_m), float(c_ms),
                        c_source, None if c_depth is None else float(c_depth),
                        int(ctd_id) if (ctd_id is not None) else None,
                        int(sid),
                    ),
                )
                updated += 1

            conn2.commit()
            conn2.close()

            QtWidgets.QMessageBox.information(self, "Near-field", f"Applied correction to {updated} result(s).")
            self.populate_spl_table()
            dlg.accept()

        btns.accepted.connect(_apply)
        btns.rejected.connect(dlg.reject)

        dlg.exec_()


    def export_spl_to_csv(self):
        """
        Export ALL rows for the currently selected filename only,
        including ALL columns present in the SPL table.
        """
        import sqlite3, csv, os

        # Enforce a specific file (no "All")
        file_filter = self.spl_file_filter_combo.currentText() if hasattr(self, "spl_file_filter_combo") else None
        if (not file_filter) or file_filter == "All":
            QtWidgets.QMessageBox.information(
                self, "Select a file",
                "Please choose a specific File (not 'All') before exporting."
            )
            return

        table = "spl_calculations"

        # Pick save path
        suggested = f"{os.path.splitext(os.path.basename(file_filter))[0]}_spl.csv"
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export SPL to CSV", suggested, "CSV Files (*.csv)"
        )
        if not fname:
            return
        if not fname.lower().endswith(".csv"):
            fname += ".csv"

        # Get ALL columns in the table (order as stored in DB)
        conn = sqlite3.connect(DB_FILENAME)
        cur  = conn.cursor()
        cur.execute(f"PRAGMA table_info({table})")
        pragma_rows = cur.fetchall()
        # pragma_rows columns: cid, name, type, notnull, dflt_value, pk
        all_cols = [r[1] for r in pragma_rows]
        if not all_cols:
            conn.close()
            QtWidgets.QMessageBox.warning(self, "Export", f"No columns found in table '{table}'.")
            return

        # Build SELECT of all columns, filtered by file_name
        col_list = ", ".join([f'"{c}"' for c in all_cols])  # quote-safe
        cur.execute(
            f'SELECT {col_list} FROM {table} WHERE file_name = ? ORDER BY id ASC',
            (file_filter,)
        )
        rows = cur.fetchall()
        conn.close()

        if not rows:
            QtWidgets.QMessageBox.information(
                self, "Export", f"No rows found for file '{file_filter}' in {table}."
            )
            return

        # Write CSV with header = ALL column names
        try:
            with open(fname, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(all_cols)
                for r in rows:
                    w.writerow(["" if v is None else v for v in r])
            QtWidgets.QMessageBox.information(self, "Exported", f"Saved:\n{fname}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Failed", str(e))


    def _on_spl_filters_changed(self, *_):
        # View or File or Method changed
        self.update_spl_method_filter_options()

        # Reset pagination and repopulate
        self.spl_page = 1
        self.populate_spl_table()


    def _on_spl_per_page_changed(self, val):
        self.spl_per_page = int(val)
        self.spl_page = 1
        self.populate_spl_table()

    def _on_spl_prev(self):
        if self.spl_page > 1:
            self.spl_page -= 1
            self.populate_spl_table()

    def _on_spl_next(self):
        max_pages = max(1, (self.spl_total_rows + self.spl_per_page - 1) // self.spl_per_page)
        if self.spl_page < max_pages:
            self.spl_page += 1
            self.populate_spl_table()


    def populate_spl_table(self):
        """
        Paginated fill of SPL table.
        Respects Active/Archived, File filter, Method filter.
        """
        import sqlite3
        base_table, from_clause, where_sql, params = self._spl_current_where()
        params = list(params)

        per_page = max(1, int(getattr(self, "spl_per_page", 200)))
        page     = max(1, int(getattr(self, "spl_page", 1)))
        offset   = (page - 1) * per_page

        # Count total rows for pagination
        conn = sqlite3.connect(DB_FILENAME)
        cur  = conn.cursor()
        count_sql = f"SELECT COUNT(*) FROM {from_clause}{where_sql}"
        cur.execute(count_sql, tuple(params))
        total_rows = cur.fetchone()[0] or 0
        self.spl_total_rows = total_rows

        # Clamp page if out of range
        max_pages = max(1, (total_rows + per_page - 1) // per_page)
        if page > max_pages:
            self.spl_page = max_pages
            page   = max_pages
            offset = (page - 1) * per_page

        # Select current page rows
        select_cols = (
            "s.id, s.file_name, s.voltage_log_id, s.hydrophone_curve, s.target_frequency, "
            "s.rms_voltage, s.spl, s.spl_nf, s.start_time, s.end_time, s.window_length, "
            "s.max_voltage, s.bandwidth, s.screenshot, s.distance, "
            "s.nf_enabled, s.nf_delta_db, s.nf_radius_m, s.nf_range_m, s.nf_c_ms, "
            "s.nf_c_source, s.nf_c_depth_m, s.nf_ctd_id, s.timestamp"
        )
        data_sql = (
            f"SELECT {select_cols} FROM {from_clause}{where_sql} "
            "ORDER BY s.id DESC LIMIT ? OFFSET ?"
        )
        cur.execute(data_sql, tuple(params + [per_page, offset]))
        rows = cur.fetchall()
        conn.close()

        # Fill table
        headers = [
            "ID","File Name","Voltage Log ID","Hydrophone Curve","Target Frequency",
            "RMS Voltage","SPL","SPL_NF","Start Time","End Time","Window Length","Max Voltage",
            "Bandwidth","Screenshot","Distance","NF?","ΔNF (dB)","a (m)","R (m)","c (m/s)",
            "c source","c depth (m)","CTD id","Timestamp"
        ]
        tbl = self.spl_table
        tbl.blockSignals(True)
        try:
            tbl.setColumnCount(len(headers))
            tbl.setHorizontalHeaderLabels(headers)
            tbl.setRowCount(len(rows))
            numeric_cols = {5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 17, 18, 19, 21, 22}
            for i, row in enumerate(rows):
                for j, val in enumerate(row):
                    it = QtWidgets.QTableWidgetItem("" if val is None else str(val))
                    if j == 0:
                        it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
                    # Respect the sig-fig selection for numeric fields except frequency
                    if j in numeric_cols:
                        it.setText(self._format_spl_value(val))
                    tbl.setItem(i, j, it)
            tbl.resizeColumnsToContents()
        finally:
            tbl.blockSignals(False)

        # Update bottom pager
        self.spl_page_label.setText(f"Page {page} / {max_pages} ({total_rows} rows)")
        self.spl_prev_btn.setEnabled(page > 1)
        self.spl_next_btn.setEnabled(page < max_pages)


    

    def analyze_spl_from_voltage_log(self):
        """
        1) Let the user pick one or more \u201cVoltage\u201d entries.
        2) Filter by file *and* by analysis method.
        3) Ask for an (optional) distance correction.
        4) Ask which hydrophone curve to use.
        5) Compute SPL = 20*log10(rms_voltage) - sensitivity_dB + 20*log10(dist).
        6) Insert into spl_calculations (with the distance column).
        """
        # --- Step 1: Fetch voltage logs (with project hint) and available projects ---
        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                m.id,
                m.file_name,
                m.target_frequency,
                m.measured_voltage AS rms_voltage,
                m.start_time,
                m.end_time,
                m.window_length,
                m.method,
                (
                    SELECT pi.project_id
                    FROM project_items pi
                    WHERE pi.file_name = m.file_name AND pi.method = m.method
                    LIMIT 1
                ) AS project_id
            FROM measurements m
            WHERE m.method IN (
                'Voltage','Waveform','FFT','Auto Analysis', 'LFM Analysis', 'MultiFreq', 'LFM Pulse', 'HFM Pulse', 'Ambient Noise'
            )
            ORDER BY m.id DESC
            """
        )
        raw_rows = cur.fetchall()

        project_rows = []
        try:
            cur.execute("SELECT id, name FROM projects ORDER BY name")
            project_rows = cur.fetchall()
        except Exception:
            project_rows = []

        conn.close()

        if not raw_rows:
            QtWidgets.QMessageBox.information(
                self, "No Voltage Logs",
                "There are no voltage entries to analyze."
            )
            return

        # --- Build selection dialog ---
        dlg = QDialog(self)
        dlg.setWindowTitle("Analyze SPL from Voltage Log")
        dlg.resize(1100, 760)
        layout = QVBoxLayout(dlg)

        PROJECT_IDX = 8

        # --- Project Filter ---
        proj_layout = QtWidgets.QHBoxLayout()
        proj_layout.addWidget(QtWidgets.QLabel("Project:"))
        proj_combo = QtWidgets.QComboBox()
        proj_combo.setMaximumWidth(320)
        proj_combo.addItem("All Projects", userData=None)
        for pid, name in project_rows:
            proj_combo.addItem(name, userData=pid)
        if getattr(self, "current_project_name", None):
            idx = proj_combo.findText(self.current_project_name)
            if idx >= 0:
                proj_combo.setCurrentIndex(idx)
        proj_layout.addWidget(proj_combo)
        layout.addLayout(proj_layout)

        # --- Filename Filter ---
        file_layout = QtWidgets.QHBoxLayout()
        file_layout.addWidget(QtWidgets.QLabel("Filter by Filename:"))
        filename_combo = QtWidgets.QComboBox()
        filename_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        filename_combo.setMinimumContentsLength(24)
        filename_combo.setMaximumWidth(520)
        filename_combo.addItem("All")
        filename_combo.view().setMinimumWidth(320)
        file_layout.addWidget(filename_combo)
        layout.addLayout(file_layout)

        # --- Method Filter ---
        method_layout = QtWidgets.QHBoxLayout()
        method_layout.addWidget(QtWidgets.QLabel("Filter by Method:"))
        method_combo = QtWidgets.QComboBox()
        method_combo.setMaximumWidth(320)
        method_combo.addItem("All")
        method_layout.addWidget(method_combo)
        layout.addLayout(method_layout)

        # --- Table of Logs ---
        table = QtWidgets.QTableWidget()
        table.setColumnCount(7)
        table.setHorizontalHeaderLabels([
            "ID","File","Freq (Hz)","RMS Voltage","Start (s)","End (s)","Window (s)"
        ])
        table.setSelectionBehavior(QtWidgets.QTableWidget.SelectRows)
        layout.addWidget(table)

        # --- Near-field correction group (optional) ---
        nf_group = QtWidgets.QGroupBox("Near-field Correction")
        nf_group.setStyleSheet("QGroupBox { color: white; }")
        nf_form = QtWidgets.QFormLayout(nf_group)

        nf_enable_chk = QtWidgets.QCheckBox("Apply near-field correction")
        nf_enable_chk.setChecked(False)

        nf_radius_edit = QtWidgets.QLineEdit("0.10")  # a [m]
        nf_range_edit  = QtWidgets.QLineEdit("1.50")  # R [m]
        nf_depth_edit  = QtWidgets.QLineEdit("")
        try:
            default_depth = self._default_hydrophone_depth_m()
            if default_depth is not None:
                nf_depth_edit.setText(f"{default_depth:.3f}")
        except Exception:
            pass

        # Sound speed controls
        nf_use_ctd_chk = QtWidgets.QCheckBox("Use sound speed from CTD cast")
        nf_use_ctd_chk.setChecked(True)

        nf_ctd_combo = QtWidgets.QComboBox()
        nf_ctd_combo.addItem("(none)", userData=None)

        # Populate CTD cast list (name + timestamp)
        try:
            conn0 = sqlite3.connect(DB_FILENAME); cur0 = conn0.cursor()
            cur0.execute("SELECT id, name, dt_utc FROM ctd_profiles ORDER BY dt_utc DESC, id DESC")
            for cid, nm, dt in cur0.fetchall():
                label = f"{nm or 'CTD'}  [{dt or ''}]"
                nf_ctd_combo.addItem(label, userData=int(cid))
            conn0.close()
        except Exception:
            pass

        # Manual override for c
        nf_manual_override_chk = QtWidgets.QCheckBox("Override CTD: use manual sound speed")
        nf_manual_override_chk.setChecked(False)
        nf_c_edit = QtWidgets.QLineEdit("1500")  # c [m/s]

        def _toggle_manual_c(checked):
            nf_c_edit.setEnabled(checked)
        _toggle_manual_c(nf_manual_override_chk.isChecked())
        nf_manual_override_chk.toggled.connect(_toggle_manual_c)

        nf_form.addRow(nf_enable_chk)
        nf_form.addRow("Projector radius a (m):", nf_radius_edit)
        nf_form.addRow("Measurement range R (m):", nf_range_edit)
        nf_form.addRow("Hydrophone depth (m, optional):", nf_depth_edit)
        nf_form.addRow(nf_use_ctd_chk)
        nf_form.addRow("CTD cast:", nf_ctd_combo)
        nf_form.addRow(nf_manual_override_chk)
        nf_form.addRow("Manual c (m/s):", nf_c_edit)

        layout.addWidget(nf_group)


        # --- Distance & Curve Selection ---
        form = QFormLayout()
        distance_edit = QLineEdit("")
        form.addRow(QtWidgets.QLabel("Distance to Source (m, optional):"), distance_edit)
        curve_combo = QtWidgets.QComboBox()
        curve_combo.addItem("None")
        for info in self.hydrophone_curves.values():
            curve_combo.addItem(info["curve_name"])
        form.addRow(QtWidgets.QLabel("Hydrophone Curve:"), curve_combo)
        layout.addLayout(form)

        # --- Analyze Buttons ---
        btn_layout = QVBoxLayout()
        analyze_sel = QPushButton("Analyze Selected")
        analyze_all = QPushButton("Analyze All")
        btn_layout.addStretch()
        btn_layout.addWidget(analyze_sel)
        btn_layout.addWidget(analyze_all)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # --- Cancel ---
        cancel_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Cancel)
        cancel_box.rejected.connect(dlg.reject)
        layout.addWidget(cancel_box)

        def _rows_for_project():
            pid = proj_combo.currentData()
            rows = raw_rows
            if pid is not None:
                rows = [r for r in rows if r[PROJECT_IDX] == pid]
            return rows

        def refresh_filename_options():
            rows = _rows_for_project()
            filenames = sorted({r[1] for r in rows if r[1]})
            current = filename_combo.currentText()
            filename_combo.blockSignals(True)
            filename_combo.clear()
            filename_combo.addItem("All")
            filename_combo.addItems(filenames)
            idx = filename_combo.findText(current)
            if idx < 0:
                idx = 0
            filename_combo.setCurrentIndex(idx)
            filename_combo.blockSignals(False)

        # --- Helper: Rebuild Method List ---
        def rebuild_method_list():
            rows = _rows_for_project()
            selected_file = filename_combo.currentText()
            if selected_file != "All":
                rows = [r for r in rows if r[1] == selected_file]
            meths = sorted({r[7] for r in rows if r[7]})
            current_method = method_combo.currentText()
            method_combo.blockSignals(True)
            method_combo.clear()
            method_combo.addItem("All")
            for m in meths:
                method_combo.addItem(m)
            idx = method_combo.findText(current_method)
            if idx < 0:
                idx = 0
            method_combo.setCurrentIndex(idx)
            method_combo.blockSignals(False)

        # --- Helper: Refresh Table ---
        def refresh_table():
            filtered = _rows_for_project()
            chosen_file = filename_combo.currentText()
            chosen_method = method_combo.currentText()
            if chosen_file != "All":
                filtered = [r for r in filtered if r[1] == chosen_file]
            if chosen_method != "All":
                filtered = [r for r in filtered if r[7] == chosen_method]

            table.setRowCount(len(filtered))
            for i, (eid, fn, freq, vrms, st, et, wl, _, _) in enumerate(filtered):
                for j, val in enumerate((eid, fn, freq, vrms, st, et, wl)):
                    item = QtWidgets.QTableWidgetItem(str(val))
                    if j == 0:
                        item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                    table.setItem(i, j, item)
            table.resizeColumnsToContents()

        # --- Signal Connections ---
        proj_combo.currentIndexChanged.connect(lambda *_: (refresh_filename_options(), rebuild_method_list(), refresh_table()))
        filename_combo.currentTextChanged.connect(lambda *_: (rebuild_method_list(), refresh_table()))
        method_combo.currentTextChanged.connect(refresh_table)

        # initial
        refresh_filename_options()
        rebuild_method_list()
        refresh_table()

        # --- Near-field (axial baffled piston) helpers -----------------------
        
        def nf_axial_piston_delta_db(f_Hz, radius_m, range_m, c=1500.0):
            """
            Axial baffled-piston near-field factor:
                F = | 2*sin(psi/2) / psi | ,  psi = k*a^2/(2R),  k = 2*pi*f/c
            Returns ΔSPL_NF in dB to SUBTRACT from measured SPL at R.
            """
            f = np.asarray(f_Hz, dtype=float)
            k = 2.0*np.pi*f/float(c)
            a = float(radius_m)
            R = float(range_m)
            psi = k*(a*a)/(2.0*R)
            F = np.ones_like(psi, dtype=float)
            nz = np.abs(psi) > 1e-12
            F[nz] = np.abs(2.0*np.sin(psi[nz]*0.5)/psi[nz])
            F = np.maximum(F, 1e-9)  # numeric safety
            return 20.0*np.log10(F)

        def _ctd_get_c_ms_at_depth(ctd_id, depth_m):
            """Return (c_ms, depth_used, source_str) from ctd_profiles at nearest depth to depth_m."""
            try:
                conn = sqlite3.connect(DB_FILENAME)  # DB_FILENAME is already used elsewhere in your app
                cur = conn.cursor()
                cur.execute("SELECT depth_json, sound_speed_json, name, dt_utc FROM ctd_profiles WHERE id=?", (ctd_id,))
                row = cur.fetchone()
            finally:
                try: conn.close()
                except: pass
            if not row or row[0] is None or row[1] is None:
                return None, None, None
            depth = np.array(json.loads(row[0]), dtype=float)
            c_ms  = np.array(json.loads(row[1]), dtype=float)
            if depth.size == 0 or c_ms.size == 0:
                return None, None, None
            d = float(max(depth_m, 0.0))
            idx = int(np.argmin(np.abs(depth - d)))
            return float(c_ms[idx]), float(depth[idx]), f"ctd:{int(ctd_id)}@{float(depth[idx]):.2f}m"

        # --- Ensure SPL NF columns exist (safe to call anytime) ----------------------

        def _ensure_spl_nf_columns():
            conn = sqlite3.connect(DB_FILENAME)
            cur  = conn.cursor()
            try:
                for tbl in ("spl_calculations",):
                    # Get current columns
                    cur.execute(f"PRAGMA table_info({tbl})")
                    cols = [row[1] for row in cur.fetchall()]

                    def _add(col, typ):
                        if col not in cols:
                            cur.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} {typ}")

                    # Base columns you likely already have; guarded in case they’re missing
                    _add("distance",      "REAL")

                    # Near-field outputs + provenance
                    _add("spl_nf",        "REAL")     # SPL with NF correction
                    _add("nf_enabled",    "INTEGER")  # 0/1
                    _add("nf_delta_db",   "REAL")     # ΔSPL_NF applied (dB)
                    _add("nf_radius_m",   "REAL")     # piston radius a (m)
                    _add("nf_range_m",    "REAL")     # measurement range R (m)
                    _add("nf_c_ms",       "REAL")     # sound speed used (m/s)
                    _add("nf_c_source",   "TEXT")     # "manual" or "ctd:<id>@<depth>m"
                    _add("nf_c_depth_m",  "REAL")     # depth (m) picked from CTD
                    _add("nf_ctd_id",     "INTEGER")  # ctd_profiles.id (if used)

                conn.commit()
            finally:
                conn.close()

        # --- Compute & Insert SPL ---
        def compute_and_insert_spl(rows_to_do):
            # distance correction (reference to 1 m)
            dist = None
            txt = distance_edit.text().strip()
            if txt:
                try:
                    dist = float(txt)
                except:
                    QtWidgets.QMessageBox.warning(dlg, "Error", "Invalid distance")
                    return False
            corr_dist_db = 20.0*np.log10(dist) if dist and dist > 0 else 0.0

            # curve
            curve_name = curve_combo.currentText()
            if curve_name == "None":
                QtWidgets.QMessageBox.warning(dlg, "No Curve", "Select a hydrophone curve first.")
                return False
            curve = next(v for v in self.hydrophone_curves.values() if v["curve_name"] == curve_name)
            minf, sens = curve["min_freq"], curve["sensitivity"]

            # NF UI values
            nf_enabled = nf_enable_chk.isChecked()
            a_m = float(nf_radius_edit.text()) if nf_radius_edit.text() else 0.0
            R_m = float(nf_range_edit.text())  if nf_range_edit.text()  else 0.0

            use_ctd = nf_use_ctd_chk.isChecked()
            ctd_id  = nf_ctd_combo.currentData() if use_ctd else None
            manual_override = nf_manual_override_chk.isChecked()
            c_ms = float(nf_c_edit.text()) if nf_c_edit.text() else 1500.0
            c_source = "manual"
            c_depth_used = None
            depth_for_ctd = None
            if nf_depth_edit.text().strip():
                try:
                    depth_for_ctd = float(nf_depth_edit.text())
                except Exception:
                    QtWidgets.QMessageBox.warning(dlg, "Near-field", "Hydrophone depth must be numeric.")
                    return False
            if (not manual_override) and use_ctd and ctd_id:
                depth_query = depth_for_ctd if depth_for_ctd is not None else R_m
                c_val, c_depth, c_src = _ctd_get_c_ms_at_depth(ctd_id, depth_query)
                if c_val is not None:
                    c_ms = c_val
                    c_source = c_src or "ctd"
                    c_depth_used = c_depth

            conn2 = sqlite3.connect(DB_FILENAME)
            cur2 = conn2.cursor()

            # ensure columns present
            _ensure_spl_nf_columns()

            for (eid, fname, targ_freq, vrms, st, et, wl, _) in rows_to_do:
                idx = int(round(targ_freq)) - minf
                if idx < 0 or idx >= len(sens):
                    QtWidgets.QMessageBox.warning(dlg, "Freq Out of Range",
                                                f"{targ_freq:.1f} Hz is outside '{curve_name}' range.")
                    conn2.close()
                    return False

                sensitivity_db = sens[idx]

                # SPL at the measurement range R (no distance ref yet)
                spl_at_R = 20.0*np.log10(max(vrms, 1e-12)) - sensitivity_db

                # NF delta (dB) to subtract before referencing to 1 m
                delta_nf_db = 0.0
                if nf_enabled and a_m > 0 and R_m > 0 and c_ms > 0 and targ_freq > 0:
                    delta_nf_db = float(nf_axial_piston_delta_db(targ_freq, a_m, R_m, c_ms))

                # Reference to 1 m
                spl_raw = spl_at_R + corr_dist_db             # no NF
                spl_nf  = spl_at_R - delta_nf_db + corr_dist_db

                cur2.execute("""
                    INSERT INTO spl_calculations
                    (file_name, voltage_log_id, hydrophone_curve,
                    target_frequency, rms_voltage, spl, spl_nf,
                    start_time, end_time, window_length,
                    max_voltage, bandwidth, screenshot, distance,
                    nf_enabled, nf_delta_db, nf_radius_m, nf_range_m, nf_c_ms,
                    nf_c_source, nf_c_depth_m, nf_ctd_id)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    fname, int(eid), curve_name,
                    float(targ_freq), float(vrms), float(spl_raw), float(spl_nf),
                    float(st), float(et), float(wl),
                    0.0, 0.0, "", float(dist) if dist is not None else None,
                    int(nf_enabled), float(delta_nf_db), float(a_m), float(R_m), float(c_ms),
                    c_source, None if c_depth_used is None else float(c_depth_used),
                    int(ctd_id) if (ctd_id is not None) else None
                ))

            conn2.commit()
            conn2.close()
            return True


        # --- Analyze Selected Row ---
        def on_sel():
            sel = table.selectionModel().selectedRows()
            if not sel:
                QtWidgets.QMessageBox.warning(dlg, "No Selection", "Select a row first.")
                return
            i = sel[0].row()
            vals = [table.item(i, j).text() for j in range(7)]
            tup = (int(vals[0]), vals[1], float(vals[2]),
                   float(vals[3]), float(vals[4]), float(vals[5]), float(vals[6]), None)
            if compute_and_insert_spl([tup]):
                QtWidgets.QMessageBox.information(self, "SPL Calculated", "Entry logged.")
                self.update_spl_filter_options()
                self.populate_spl_table()

        # --- Analyze All Rows ---
        def on_all():
            # build list of all displayed
            rows_to_do = []
            for i in range(table.rowCount()):
                vals = [table.item(i, j).text() for j in range(7)]
                rows_to_do.append((
                    int(vals[0]), vals[1], float(vals[2]),
                    float(vals[3]), float(vals[4]), float(vals[5]), float(vals[6]), None
                ))
            if compute_and_insert_spl(rows_to_do):
                QtWidgets.QMessageBox.information(
                    self, "SPL Calculated",
                    f"{len(rows_to_do)} entries logged."
                )
                self.update_spl_filter_options()
                self.populate_spl_table()

        analyze_sel.clicked.connect(on_sel)
        analyze_all.clicked.connect(on_all)

        dlg.exec_()


    def update_spl_method_filter_options(self):
        """
        Rebuild the Method filter combo from the current view's table,
        optionally filtered by File.
        """
        import sqlite3

        table = "spl_calculations"
        file_filter = self.spl_file_filter_combo.currentText() if hasattr(self, "spl_file_filter_combo") else "All"

        project_id = getattr(self, "current_project_id", None)
        if project_id is None and getattr(self, "current_project_name", None):
            project_id = self._get_project_id(self.current_project_name)
            self.current_project_id = project_id

        prev = None
        if hasattr(self, "spl_method_filter_combo") and self.spl_method_filter_combo.count() > 0:
            prev = self.spl_method_filter_combo.currentText()

        conn = sqlite3.connect(DB_FILENAME)
        cur  = conn.cursor()
        try:
            where_clauses = []
            params = []
            if file_filter and file_filter != "All":
                where_clauses.append("s.file_name = ?")
                params.append(file_filter)
            if project_id is not None:
                where_clauses.append("p.project_id = ?")
                params.append(project_id)

            where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
            cur.execute(
                f"""
                SELECT DISTINCT m.method
                FROM {table} AS s
                JOIN measurements AS m ON s.voltage_log_id = m.id
                {'JOIN project_items AS p ON p.file_name = m.file_name AND p.method = m.method' if project_id is not None else ''}
                {where_sql}
                ORDER BY m.method
                """,
                tuple(params),
            )
            methods = [r[0] for r in cur.fetchall() if r[0]]
        finally:
            conn.close()

        self.spl_method_filter_combo.blockSignals(True)
        try:
            self.spl_method_filter_combo.clear()
            self.spl_method_filter_combo.addItem("All")
            for m in methods:
                self.spl_method_filter_combo.addItem(m)
            if prev and prev in methods:
                idx = self.spl_method_filter_combo.findText(prev)
                if idx >= 0:
                    self.spl_method_filter_combo.setCurrentIndex(idx)
            else:
                self.spl_method_filter_combo.setCurrentIndex(0)
        finally:
            self.spl_method_filter_combo.blockSignals(False)





    def plot_spl_over_time(self):
        """
        SPL vs Time plot with:
        • Show Markers
        • Save as JPG
        • (NEW) Show Near-Field Corrected overlay (if data exists)
        • (NEW) X/Y axis range controls with Apply
        • (NEW) Preview Near-field Δ(f) button using most common NF params from DB
        """
        files_to_plot = self._resolve_spl_plot_files()
        if not files_to_plot:
            return

        import sqlite3, numpy as np, itertools
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        import matplotlib.pyplot as plt

        base_table, from_clause, where_sql, params = self._spl_current_where(override_files=files_to_plot)

        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()

        rows = []
        has_nf_col = True
        try:
            cur.execute(
                f"""
                SELECT s.file_name, s.start_time, s.spl, s.spl_nf,
                       s.nf_enabled, s.nf_radius_m, s.nf_range_m, s.nf_c_ms
                FROM {from_clause}{where_sql}
                ORDER BY s.file_name, s.start_time
                """,
                tuple(params),
            )
            rows = cur.fetchall()
        except Exception:
            has_nf_col = False
            cur.execute(
                f"""
                SELECT s.file_name, s.start_time, s.spl
                FROM {from_clause}{where_sql}
                ORDER BY s.file_name, s.start_time
                """,
                tuple(params),
            )
            rows = [(r[0], r[1], r[2], None, None, None, None, None) for r in cur.fetchall()]
        finally:
            conn.close()

        grouped = {}
        for fname, t, spl, spl_nf, nf_enabled, nf_radius, nf_range, nf_c in rows:
            if fname is None or t is None or spl is None:
                continue
            bucket = grouped.setdefault(fname, {
                "t": [], "spl": [], "spl_nf": [], "nf_flags": [],
                "nf_radius": [], "nf_range": [], "nf_c": [],
            })
            try:
                bucket["t"].append(float(t))
                bucket["spl"].append(float(spl))
                bucket["spl_nf"].append(None if spl_nf is None else float(spl_nf))
                bucket["nf_flags"].append(nf_enabled)
                bucket["nf_radius"].append(nf_radius)
                bucket["nf_range"].append(nf_range)
                bucket["nf_c"].append(nf_c)
            except Exception:
                continue

        if not grouped:
            QtWidgets.QMessageBox.information(self, "No Data", "No SPL rows to plot.")
            return

        def _has_nf(data):
            if not has_nf_col:
                return False
            try:
                return any(v is not None and np.isfinite(v) for v in data.get("spl_nf", []))
            except Exception:
                return False

        has_any_nf = any(_has_nf(v) for v in grouped.values())

        fig, ax = plt.subplots(figsize=(9, 5), facecolor="#19232D")
        ax.set_facecolor("#000000")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")
        ax.tick_params(colors="white")
        ax.set_title("SPL vs Time", color="white")
        ax.set_xlabel("Time (s)", color="white")
        ax.set_ylabel("SPL (dB)", color="white")

        canvas = FigureCanvas(fig)

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("SPL vs Time")
        dlg.resize(900, 520)
        layout = QtWidgets.QVBoxLayout(dlg)

        controls = QtWidgets.QHBoxLayout()

        controls.addWidget(QtWidgets.QLabel("X Range:"))
        x_range_edit = QtWidgets.QLineEdit()
        x_range_edit.setPlaceholderText("min,max")
        x_range_edit.setFixedWidth(100)
        controls.addWidget(x_range_edit)

        controls.addSpacing(8)
        controls.addWidget(QtWidgets.QLabel("Y Range:"))
        y_range_edit = QtWidgets.QLineEdit()
        y_range_edit.setPlaceholderText("min,max")
        y_range_edit.setFixedWidth(100)
        controls.addWidget(y_range_edit)

        apply_btn = QtWidgets.QPushButton("Apply")
        controls.addWidget(apply_btn)

        show_markers_cb = QtWidgets.QCheckBox("Show markers")
        controls.addWidget(show_markers_cb)

        show_nf_checkbox = None
        if has_any_nf:
            show_nf_checkbox = QtWidgets.QCheckBox("Show near-field")
            show_nf_checkbox.setChecked(True)
            controls.addWidget(show_nf_checkbox)

        controls.addStretch()

        save_btn = QtWidgets.QPushButton("Save as JPG")
        controls.addWidget(save_btn)

        layout.addLayout(controls)
        layout.addWidget(canvas)

        palette_cycle = self._ordered_palette()
        raw_lines, raw_scats, nf_lines, nf_scats = [], [], [], []

        def _apply_range(edit, setter):
            txt = edit.text().strip()
            if not txt:
                return
            try:
                lo, hi = [float(v) for v in txt.split(",")]
                setter(lo, hi)
            except Exception:
                pass

        def _redraw():
            raw_lines.clear(); raw_scats.clear(); nf_lines.clear(); nf_scats.clear()
            ax.cla()
            ax.set_facecolor("#000000")
            for spine in ax.spines.values():
                spine.set_edgecolor("white")
            ax.tick_params(colors="white")
            ax.set_title("SPL vs Time", color="white")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("SPL (dB)", color="white")

            _apply_range(x_range_edit, ax.set_xlim)
            _apply_range(y_range_edit, ax.set_ylim)

            colors_local = itertools.cycle(palette_cycle)
            for fname in sorted(grouped.keys()):
                col = next(colors_local)
                data = grouped[fname]
                line_raw, = ax.plot(data["t"], data["spl"], lw=1.3, color=col, label=f"{fname} (raw)")
                raw_lines.append(line_raw)
                if show_markers_cb.isChecked():
                    raw_scats.append(ax.scatter(data["t"], data["spl"], color=col, s=18))

                if has_any_nf and show_nf_checkbox and show_nf_checkbox.isChecked() and _has_nf(data):
                    nf_times = [t for t, v in zip(data["t"], data["spl_nf"]) if v is not None]
                    nf_vals_plot = [v for v in data["spl_nf"] if v is not None]
                    if nf_times:
                        line_nf, = ax.plot(nf_times, nf_vals_plot, lw=1.2, linestyle="--", color=col, alpha=0.85, label=f"{fname} (near-field)")
                        nf_lines.append(line_nf)
                        if show_markers_cb.isChecked():
                            nf_scats.append(ax.scatter(nf_times, nf_vals_plot, color=col, s=16, marker="x"))

            if raw_lines or nf_lines:
                ax.legend(facecolor="#222", labelcolor="white")

            canvas.draw_idle()

        def _toggle_markers():
            visible = show_markers_cb.isChecked()
            for scat in raw_scats + nf_scats:
                scat.set_visible(visible)
            canvas.draw_idle()

        def _toggle_nf():
            for line in nf_lines:
                line.set_visible(show_nf_checkbox.isChecked())
            for scat in nf_scats:
                scat.set_visible(show_nf_checkbox.isChecked() and show_markers_cb.isChecked())
            canvas.draw_idle()

        def _save():
            self._save_figure_jpg(fig)

        apply_btn.clicked.connect(_redraw)
        show_markers_cb.stateChanged.connect(lambda _: (_toggle_markers() if raw_scats or nf_scats else _redraw()))
        if show_nf_checkbox:
            show_nf_checkbox.stateChanged.connect(lambda _: (_toggle_nf() if nf_lines else _redraw()))
        save_btn.clicked.connect(_save)

        _redraw()
        dlg.exec_()
        plt.close(fig)

    def plot_spl_vs_frequency(self):
        """
        Plot SPL (dB) vs Frequency (Hz) for the currently filtered file,
        with Show Markers, click-to-read, Save as JPG, and a checkbox to show NF
        if any spl_nf data exists for that file.
        """
        files_to_plot = self._resolve_spl_plot_files()
        if not files_to_plot:
            return

        table       = "spl_calculations"

        import sqlite3, numpy as np, itertools
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        import matplotlib.pyplot as plt

        base_table, from_clause, where_sql, params = self._spl_current_where(override_files=files_to_plot)

        conn = sqlite3.connect(DB_FILENAME)
        cur  = conn.cursor()
        rows = []
        has_nf_col = True
        freq_clause = "s.target_frequency IS NOT NULL"
        where_with_freq = (
            f"{where_sql} AND {freq_clause}" if where_sql else f" WHERE {freq_clause}"
        )
        try:
            cur.execute(
                f"""SELECT s.file_name, s.target_frequency, s.spl, s.spl_nf
                    FROM {from_clause}{where_with_freq}
                    ORDER BY s.file_name, s.target_frequency""",
                tuple(params),
            )
            rows = cur.fetchall()
        except Exception:
            has_nf_col = False
            cur.execute(
                f"""SELECT s.file_name, s.target_frequency, s.spl
                    FROM {from_clause}{where_with_freq}
                    ORDER BY s.file_name, s.target_frequency""",
                tuple(params),
            )
            rows = [(r[0], r[1], r[2], None) for r in cur.fetchall()]
        conn.close()

        grouped = {}
        for fname, f, spl, spl_nf in rows:
            if fname is None or f is None or spl is None:
                continue
            try:
                bucket = grouped.setdefault(fname, {"f": [], "spl": [], "spl_nf": []})
                bucket["f"].append(float(f))
                bucket["spl"].append(float(spl))
                bucket["spl_nf"].append(None if spl_nf is None else float(spl_nf))
            except Exception:
                continue

        if not grouped:
            QtWidgets.QMessageBox.information(self, "No Data", "No SPL entries match the current filters.")
            return

        def _has_nf(vals):
            if not has_nf_col:
                return False
            try:
                return any(v is not None and np.isfinite(v) for v in vals)
            except Exception:
                return False

        has_any_nf = any(_has_nf(v["spl_nf"]) for v in grouped.values())

        fig, ax = plt.subplots(figsize=(8,4), facecolor="#19232D")
        ax.set_facecolor("#000000")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")
        ax.set_title("SPL vs Frequency", color="white")
        ax.set_xlabel("Frequency (Hz)", color="white")
        ax.set_ylabel("SPL (dB)", color="white")
        ax.tick_params(colors="white")

        palette_cycle = self._ordered_palette()
        raw_lines, nf_lines, raw_scats, nf_scats = [], [], [], []
        colors = itertools.cycle(palette_cycle)

        for fname in sorted(grouped.keys()):
            col = next(colors)
            data = grouped[fname]
            line_raw, = ax.plot(data["f"], data["spl"], color=col, lw=1.4, label=f"{fname} (raw)")
            raw_lines.append(line_raw)
            raw_scats.append(ax.scatter(data["f"], data["spl"], color=col, s=22, visible=False))

            if has_any_nf and _has_nf(data["spl_nf"]):
                nf_freqs = [ff for ff, val in zip(data["f"], data["spl_nf"]) if val is not None]
                nf_vals  = [val for val in data["spl_nf"] if val is not None]
                if nf_freqs:
                    line_nf, = ax.plot(nf_freqs, nf_vals, lw=1.2, linestyle="--", alpha=0.9, color=col, label=f"{fname} (near-field)")
                    nf_lines.append(line_nf)
                    nf_scats.append(ax.scatter(nf_freqs, nf_vals, color=col, s=22, visible=False))

        if raw_lines or nf_lines:
            ax.legend(facecolor="#222", labelcolor="white")

        canvas = FigureCanvas(fig)
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("SPL vs Frequency")
        dlg.resize(900, 520)
        vbox = QtWidgets.QVBoxLayout(dlg)

        controls = QtWidgets.QHBoxLayout()
        show_markers_cb = QtWidgets.QCheckBox("Show markers")
        controls.addWidget(show_markers_cb)

        show_nf_checkbox = None
        if has_any_nf:
            show_nf_checkbox = QtWidgets.QCheckBox("Show near-field")
            show_nf_checkbox.setChecked(True)
            controls.addWidget(show_nf_checkbox)

        controls.addStretch()

        save_btn = QtWidgets.QPushButton("Save as JPG")
        controls.addWidget(save_btn)

        vbox.addLayout(controls)
        vbox.addWidget(canvas)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        hb = QtWidgets.QHBoxLayout()
        hb.addStretch(); hb.addWidget(close_btn)
        vbox.addLayout(hb)

        def _toggle_markers():
            visible = show_markers_cb.isChecked()
            for scat in raw_scats:
                scat.set_visible(visible)
            for scat in nf_scats:
                scat.set_visible(visible and (show_nf_checkbox.isChecked() if show_nf_checkbox else True))
            canvas.draw_idle()

        def _toggle_nf():
            if show_nf_checkbox:
                visible = show_nf_checkbox.isChecked()
                for line in nf_lines:
                    line.set_visible(visible)
                for scat in nf_scats:
                    scat.set_visible(visible and show_markers_cb.isChecked())
                canvas.draw_idle()

        def _save():
            self._save_figure_jpg(fig)

        show_markers_cb.stateChanged.connect(_toggle_markers)
        if show_nf_checkbox:
            show_nf_checkbox.stateChanged.connect(_toggle_nf)
        save_btn.clicked.connect(_save)

        dlg.exec_()
        plt.close(fig)

    def plot_spl_multi_freq_over_time(self):
        table       = "spl_calculations"
        selected_fn = self.spl_file_filter_combo.currentText()

        # 1) Fetch only rows for the selected file (prompt when "All")
        if not selected_fn or selected_fn == "All":
            choices = self._resolve_spl_plot_files(allow_multi=False)
            if not choices:
                return
            selected_fn = choices[0]

        conn = sqlite3.connect(DB_FILENAME)
        cur  = conn.cursor()
        cur.execute(
            "SELECT target_frequency, start_time, spl "
            f"FROM {table} WHERE file_name = ? "
            "ORDER BY start_time, target_frequency",
            (selected_fn,)
        )
        rows = cur.fetchall()
        conn.close()

        if not rows:
            QtWidgets.QMessageBox.information(self, "No Data", f"No SPL entries for '{selected_fn}'.")
            return

        # 2) Group by frequency
        freqs = sorted({r[0] for r in rows})
        data  = {f: [] for f in freqs}
        for f, t, s in rows:
            data[f].append((t, s))

        # 3) Let user pick which of those frequencies to plot
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"Pick Frequencies for '{selected_fn}'")
        v = QtWidgets.QVBoxLayout(dlg)

        listw = QtWidgets.QListWidget()
        listw.addItems([f"{f:.0f} Hz" for f in freqs])
        listw.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        v.addWidget(listw)

        btns = QtWidgets.QDialogButtonBox()
        save_btn  = btns.addButton("Save as JPG", QtWidgets.QDialogButtonBox.ActionRole)
        ok_btn    = btns.addButton(QtWidgets.QDialogButtonBox.Ok)
        cancel_btn= btns.addButton(QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        v.addWidget(btns)

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        chosen = [float(i.text().split()[0]) for i in listw.selectedItems()]
        if not chosen:
            QtWidgets.QMessageBox.information(self, "No Selection", "No frequencies chosen.")
            return

        # 4) Build the figure
        import itertools

        fig, ax = plt.subplots(figsize=(8,4), facecolor="#19232D")
        ax.set_facecolor("#000000")
        palette_cycle = itertools.cycle(self._ordered_palette())
        for f in chosen:
            pts = data[f]
            times, spls = zip(*pts)
            ax.plot(times, spls, label=f"{f:.0f} Hz", lw=1, color=next(palette_cycle))

        ax.set_title(f"SPL vs Time @ '{selected_fn}'", color="white")
        ax.set_xlabel("Time (s)", color="white")
        ax.set_ylabel("SPL (dB)", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")
        ax.legend(facecolor="#222", labelcolor="white")

        # 5) Show in Qt dialog with Save + Close
        canvas = FigureCanvas(fig)
        plot_dlg = QtWidgets.QDialog(self)
        plot_dlg.setWindowTitle("Multi-Freq SPL vs Time")
        lv = QtWidgets.QVBoxLayout(plot_dlg)

        # controls row
        ctl = QtWidgets.QHBoxLayout()
        ctl.addWidget(save_btn)
        close_btn = QtWidgets.QPushButton("Close")
        ctl.addStretch()
        ctl.addWidget(close_btn)
        lv.addLayout(ctl)

        lv.addWidget(canvas)

        save_btn.clicked.connect(lambda: self._save_figure_jpg(fig, parent=plot_dlg))
        close_btn.clicked.connect(plot_dlg.accept)

        plot_dlg.resize(900, 500)
        plot_dlg.exec_()
        plt.close(fig)


    @contextlib.contextmanager
    def _temporary_figure_theme(self, fig, mode):
        mode = (mode or "").lower()
        if mode.startswith("use current") or mode == "current" or mode == "":
            yield
            return

        saved = {
            "fig": fig.get_facecolor(),
            "axes": [],
        }

        for ax in fig.axes:
            state = {
                "face": ax.get_facecolor(),
                "spines": {k: sp.get_edgecolor() for k, sp in ax.spines.items()},
                "xlabel": ax.xaxis.label.get_color(),
                "ylabel": ax.yaxis.label.get_color(),
                "title": ax.title.get_color(),
                "xticks": [lbl.get_color() for lbl in ax.get_xticklabels()],
                "yticks": [lbl.get_color() for lbl in ax.get_yticklabels()],
                "lines": [ln.get_color() for ln in ax.get_lines()],
                "collections": [(c.get_facecolor(), c.get_edgecolor()) for c in ax.collections],
            }
            saved["axes"].append((ax, state))

        if "dark" in mode:
            fig.set_facecolor("#19232D")
            face = "#000000"
            text = "#FFFFFF"
            spine = "#FFFFFF"
            line_override = None
        elif "black" in mode:
            fig.set_facecolor("#FFFFFF")
            face = "#FFFFFF"
            text = "#000000"
            spine = "#000000"
            line_override = "black"
        else:  # light mode
            fig.set_facecolor("#FFFFFF")
            face = "#FFFFFF"
            text = "#000000"
            spine = "#000000"
            line_override = None

        for ax, _ in saved["axes"]:
            ax.set_facecolor(face)
            for sp in ax.spines.values():
                sp.set_edgecolor(spine)
            ax.tick_params(colors=text)
            ax.xaxis.label.set_color(text)
            ax.yaxis.label.set_color(text)
            ax.title.set_color(text)
            for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                lbl.set_color(text)
            if line_override:
                for ln in ax.get_lines():
                    ln.set_color(line_override)
                for col in ax.collections:
                    try:
                        col.set_facecolor(line_override)
                        col.set_edgecolor(line_override)
                    except Exception:
                        pass

        try:
            yield
        finally:
            fig.set_facecolor(saved["fig"])
            for ax, state in saved["axes"]:
                ax.set_facecolor(state["face"])
                for name, sp in ax.spines.items():
                    sp.set_edgecolor(state["spines"].get(name, sp.get_edgecolor()))
                ax.xaxis.label.set_color(state["xlabel"])
                ax.yaxis.label.set_color(state["ylabel"])
                ax.title.set_color(state["title"])
                for lbl, col in zip(ax.get_xticklabels(), state["xticks"]):
                    lbl.set_color(col)
                for lbl, col in zip(ax.get_yticklabels(), state["yticks"]):
                    lbl.set_color(col)
                for ln, col in zip(ax.get_lines(), state["lines"]):
                    ln.set_color(col)
                for col_obj, (fcol, ecol) in zip(ax.collections, state["collections"]):
                    try:
                        col_obj.set_facecolor(fcol)
                        col_obj.set_edgecolor(ecol)
                    except Exception:
                        pass


    def _save_figure_jpg(self, fig, parent=None):
        modes = ["Use current theme", "Dark mode", "Light mode", "Black & White"]
        mode, ok = QtWidgets.QInputDialog.getItem(
            parent or self,
            "Save Figure",
            "Color mode:",
            modes,
            0,
            False,
        )
        if not ok:
            return

        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            parent or self,
            "Save Figure as JPG",
            "",
            "JPEG Files (*.jpg *.jpeg)",
        )
        if fname:
            if not fname.lower().endswith((".jpg", ".jpeg")):
                fname += ".jpg"
            with self._temporary_figure_theme(fig, mode):
                fig.savefig(fname, dpi=150, facecolor=fig.get_facecolor())
            QtWidgets.QMessageBox.information(parent or self, "Saved", f"Figure saved to:\n{fname}")
            return fname
        return None

    def delete_selected_spl(self):
         sel = self.spl_table.selectionModel().selectedRows()
         if not sel:
             QtWidgets.QMessageBox.information(self, "Info", "Select at least one row.")
             return
         table = "spl_calculations"
         conn = sqlite3.connect(DB_FILENAME); cur = conn.cursor()
         for idx in sel:
             rid = int(self.spl_table.item(idx.row(), 0).text())
             cur.execute(f"DELETE FROM {table} WHERE id=?", (rid,))
         conn.commit(); conn.close()
         self.populate_spl_table()

    def delete_all_spl(self):
        table = "spl_calculations"
        reply = QtWidgets.QMessageBox.question(
            self, "Confirm Delete All",
            "Delete all displayed SPL entries?",
            QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return
        # repeat the same filter logic from populate before deleting
        fname = self.spl_file_filter_combo.currentText()
        conn = sqlite3.connect(DB_FILENAME); cur = conn.cursor()
        if fname and fname!="All":
            cur.execute(f"DELETE FROM {table} WHERE file_name = ?", (fname,))
        else:
            cur.execute(f"DELETE FROM {table}")
        conn.commit(); conn.close()
        self.populate_spl_table()


    # ---------------------
    # Spectrogram Tab Methods
    # ---------------------
    def setup_spectrogram_tab(self):
        from matplotlib.colors import LinearSegmentedColormap
        main_layout = QtWidgets.QHBoxLayout(self.spectrogram_tab)

        # ── Saved spectrograms & annotations sidebar ───────────────────────
        sidebar = QtWidgets.QVBoxLayout()
        sidebar.addWidget(QtWidgets.QLabel("Saved Spectrograms"))
        self.spec_gallery_list = QtWidgets.QListWidget()
        self.spec_gallery_list.setMinimumWidth(200)
        self.spec_gallery_list.setMaximumWidth(260)
        self.spec_gallery_list.itemDoubleClicked.connect(self.load_saved_spectrogram)
        sidebar.addWidget(self.spec_gallery_list, 1)

        gallery_btns = QtWidgets.QHBoxLayout()
        self.refresh_spec_gallery_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_spec_gallery_btn.clicked.connect(self._refresh_spectrogram_gallery)
        gallery_btns.addWidget(self.refresh_spec_gallery_btn)
        self.open_saved_spec_btn = QtWidgets.QPushButton("Open")
        self.open_saved_spec_btn.clicked.connect(lambda *_: self.load_saved_spectrogram())
        self.open_saved_spec_btn.setEnabled(False)
        gallery_btns.addWidget(self.open_saved_spec_btn)
        sidebar.addLayout(gallery_btns)

        sidebar.addWidget(QtWidgets.QLabel("Annotations"))
        self.spec_annotation_list = QtWidgets.QListWidget()
        self.spec_annotation_list.itemDoubleClicked.connect(self._open_annotation_editor)
        self.spec_annotation_list.setMinimumWidth(200)
        self.spec_annotation_list.setMaximumWidth(260)
        sidebar.addWidget(self.spec_annotation_list, 1)

        annot_controls = QtWidgets.QHBoxLayout()
        self.spec_annotation_toggle = QtWidgets.QCheckBox("Annotate")
        self.spec_annotation_toggle.stateChanged.connect(self._toggle_spec_annotation_mode)
        annot_controls.addWidget(self.spec_annotation_toggle)
        self.delete_annotation_btn = QtWidgets.QPushButton("Delete")
        self.delete_annotation_btn.clicked.connect(self._delete_selected_annotation)
        annot_controls.addWidget(self.delete_annotation_btn)
        sidebar.addLayout(annot_controls)

        auto_controls = QtWidgets.QHBoxLayout()
        self.auto_annotate_btn = QtWidgets.QPushButton("Auto Annotate")
        self.auto_annotate_btn.clicked.connect(self.auto_annotate_spectrogram_popup)
        auto_controls.addWidget(self.auto_annotate_btn)
        self.call_library_btn = QtWidgets.QPushButton("Call Library")
        self.call_library_btn.clicked.connect(self.manage_call_library_popup)
        auto_controls.addWidget(self.call_library_btn)
        sidebar.addLayout(auto_controls)

        self.spec_annotation_toggle.setEnabled(False)
        self.delete_annotation_btn.setEnabled(False)
        self.spec_annotation_list.setEnabled(False)
        self.auto_annotate_btn.setEnabled(False)
        self.call_library_btn.setEnabled(True)

        main_layout.addLayout(sidebar, 1)

        layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(layout, 4)

        # ── Parameter Controls ────────────────────────────────────────────────
        controls_container = QtWidgets.QWidget()
        controls_container.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        controls_container.setMaximumHeight(72)
        controls = QtWidgets.QGridLayout(controls_container)
        controls.setSpacing(4)
        controls.setContentsMargins(4, 1, 4, 1)

        self.spec_channel_combo = QtWidgets.QComboBox()
        self.spec_channel_combo.setFixedWidth(140)
        self.spec_channel_combo.currentIndexChanged.connect(self._on_spec_channel_changed)

        self.spec_threshold_entry = QtWidgets.QLineEdit("1000")
        self.spec_threshold_entry.setFixedWidth(70)

        self.pre_buffer_entry = QtWidgets.QLineEdit("0.1")
        self.pre_buffer_entry.setFixedWidth(70)

        self.post_buffer_entry = QtWidgets.QLineEdit("0.1")
        self.post_buffer_entry.setFixedWidth(70)

        self.min_spec_length_entry = QtWidgets.QLineEdit("5.0")
        self.min_spec_length_entry.setFixedWidth(70)

        self.generate_spec_btn = QtWidgets.QPushButton("Generate")
        self.generate_spec_btn.setFixedWidth(90)
        self.generate_spec_btn.clicked.connect(self.generate_spectrograms)

        controls.addWidget(QtWidgets.QLabel("Channel:"), 0, 0)
        controls.addWidget(self.spec_channel_combo, 0, 1)
        controls.addWidget(QtWidgets.QLabel("Threshold (Amp):"), 0, 2)
        controls.addWidget(self.spec_threshold_entry, 0, 3)
        controls.addWidget(QtWidgets.QLabel("Pre-buffer (s):"), 0, 4)
        controls.addWidget(self.pre_buffer_entry, 0, 5)
        controls.addWidget(QtWidgets.QLabel("Post-buffer (s):"), 0, 6)
        controls.addWidget(self.post_buffer_entry, 0, 7)
        controls.addWidget(QtWidgets.QLabel("Min Length (s):"), 0, 8)
        controls.addWidget(self.min_spec_length_entry, 0, 9)
        controls.addWidget(self.generate_spec_btn, 0, 10)

        self.spec_nfft_combo = QtWidgets.QComboBox()
        for n in (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072):
            self.spec_nfft_combo.addItem(str(n), n)
        self.spec_nfft_combo.setCurrentIndex(1)
        self.spec_nfft_combo.setFixedWidth(90)

        self.show_colorbar_checkbox = QtWidgets.QCheckBox()
        self.show_colorbar_checkbox.setChecked(True)

        blackfish_purple = LinearSegmentedColormap.from_list(
            "blackfish_purple",
            ["#0B0A10", "#1B1030", "#4C1E7A", "#7B2CBF", "#C77DFF"]  # dark → bright purple
        )
        import matplotlib as _mpl; _mpl.colormaps.register(blackfish_purple)
        blackfish_pink = LinearSegmentedColormap.from_list(
            "blackfish_pink",
            ["#0B0A10", "#1A0B14", "#3B0B2E", "#7A1E5C", "#D81B60", "#FF80AB"]  # dark → bright purple
        )
        import matplotlib as _mpl; _mpl.colormaps.register(blackfish_pink)

        self.spec_cmap_combo = QtWidgets.QComboBox()
        self.spec_cmap_combo.addItems(["inferno", "viridis", "plasma", "magma", "cividis", "turbo",
                                                    "hot", "gist_heat", "bone", "Greys",
                                                    "twilight", "twilight_shifted",
                                                    "coolwarm", "seismic", "RdBu",
                                                    "Spectral",
                                                    "inferno_r", "viridis_r", "turbo_r", "blackfish_purple", "blackfish_pink"])
        self.spec_cmap_combo.setFixedWidth(130)

        self.spec_theme_combo = QtWidgets.QComboBox()
        self.spec_theme_combo.addItems(["Dark", "Light"])
        self.spec_theme_combo.setFixedWidth(80)

        self.spec_min_freq_entry = QtWidgets.QLineEdit("0")
        self.spec_min_freq_entry.setFixedWidth(70)

        self.spec_max_freq_entry = QtWidgets.QLineEdit("")  # set to Nyquist on file load
        self.spec_max_freq_entry.setFixedWidth(70)

        self.spec_spl_label = QtWidgets.QLabel("SPL: —")
        self.spec_spl_label.setMinimumWidth(130)

        controls.addWidget(QtWidgets.QLabel("NFFT:"), 1, 0)
        controls.addWidget(self.spec_nfft_combo, 1, 1)
        controls.addWidget(QtWidgets.QLabel("Colorbar:"), 1, 2)
        controls.addWidget(self.show_colorbar_checkbox, 1, 3)
        controls.addWidget(QtWidgets.QLabel("Colormap:"), 1, 4)
        controls.addWidget(self.spec_cmap_combo, 1, 5)
        controls.addWidget(QtWidgets.QLabel("Theme:"), 1, 6)
        controls.addWidget(self.spec_theme_combo, 1, 7)
        controls.addWidget(QtWidgets.QLabel("Min (Hz):"), 1, 8)
        controls.addWidget(self.spec_min_freq_entry, 1, 9)
        controls.addWidget(QtWidgets.QLabel("Max (Hz):"), 1, 10)
        controls.addWidget(self.spec_max_freq_entry, 1, 11)
        controls.addWidget(self.spec_spl_label, 1, 12)

        controls.setColumnStretch(10, 0)
        controls.setColumnStretch(11, 0)
        controls.setColumnStretch(12, 1)

        layout.addWidget(controls_container)

        # ── Spectrogram Display ───────────────────────────────────────────────
        self.spec_canvas = MplSpecCanvas(self.spectrogram_tab, width=8, height=6, dpi=100)
        layout.addWidget(self.spec_canvas)

        self.spec_selector = SpanSelector(
            self.spec_canvas.ax,
            self.on_spec_select,
            'horizontal',
            useblit=True,
            props=dict(alpha=0.5, facecolor=self.graph_color)
        )

        # ── Navigation Buttons ─────────────────────────────────────────────────
        nav_layout = QtWidgets.QHBoxLayout()
        nav_layout.addStretch()
        self.prev_spec_btn = QtWidgets.QPushButton("‹ Previous")
        self.prev_spec_btn.clicked.connect(self.show_prev_spec)
        nav_layout.addWidget(self.prev_spec_btn)
        self.next_spec_btn = QtWidgets.QPushButton("Next ›")
        self.next_spec_btn.clicked.connect(self.show_next_spec)
        nav_layout.addWidget(self.next_spec_btn)
        nav_layout.addStretch()
        layout.addLayout(nav_layout)

        # ── Export, Listen, and Save Buttons ────────────────────────────────────
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch()

        self.export_clip_btn = QtWidgets.QPushButton("Export Clip")
        self.export_clip_btn.setEnabled(False)
        btn_layout.addWidget(self.export_clip_btn)

        self.listen_btn = QtWidgets.QPushButton("Listen")
        self.listen_btn.setEnabled(False)
        btn_layout.addWidget(self.listen_btn)

        self.export_clip_btn.clicked.connect(self.export_spectrogram_clip)
        self.listen_btn.clicked.connect(self.spectrogram_listen_popup)

        self.save_spec_btn = QtWidgets.QPushButton("Save Spectrogram")
        self.save_spec_btn.setEnabled(False)
        self.save_spec_btn.clicked.connect(self.save_current_spectrogram)
        btn_layout.addWidget(self.save_spec_btn)

        self.save_ml_clip_btn = QtWidgets.QPushButton("Save ML Clip")
        self.save_ml_clip_btn.setEnabled(False)   # enable later, once you have a selection
        self.save_ml_clip_btn.clicked.connect(self.export_spectrogram_clip)
        btn_layout.addWidget(self.save_ml_clip_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Internal state
        self.spec_files = []
        self.spec_segments = []
        self.current_spec_index = -1
        self.current_spectrogram_record = None
        self._spec_annotation_patches = []
        self._spec_rect_selector = None
        self.spec_active_channel = 0
        if hasattr(self, '_populate_spec_channel_combo'):
            self._populate_spec_channel_combo()

    def _populate_spec_channel_combo(self):
        combo = getattr(self, "spec_channel_combo", None)
        if combo is None:
            return
        try:
            combo.blockSignals(True)
        except Exception:
            pass
        combo.clear()
        self._ensure_channel_info()
        names = getattr(self, "channel_names", ["Ch 1"])
        if not names:
            names = ["Ch 1"]
        for idx, name in enumerate(names):
            combo.addItem(name, idx)
        try:
            combo.setCurrentIndex(0)
            combo.blockSignals(False)
        except Exception:
            pass

    def _on_spec_channel_changed(self, *_):
        try:
            self.spec_segments.clear()
            self.current_spec_index = -1
            self.current_spectrogram_record = None
        except Exception:
            pass
        if hasattr(self, "spec_spl_label"):
            try:
                self.spec_spl_label.setText("SPL: —")
            except Exception:
                pass
        self._refresh_spectrogram_gallery()

    def _current_spec_channel_index(self):
        combo = getattr(self, "spec_channel_combo", None)
        if combo is None or combo.currentIndex() < 0:
            return 0
        try:
            data = combo.currentData()
            return int(data) if data is not None else combo.currentIndex()
        except Exception:
            return combo.currentIndex()

    def _spectrogram_file_key(self):
        path = getattr(self, "current_file_path", None) or getattr(self, "loaded_file", None)
        return os.path.basename(path) if path else None

    def _set_spec_annotation_controls_enabled(self, enabled: bool):
        widgets = [
            getattr(self, "spec_annotation_toggle", None),
            getattr(self, "delete_annotation_btn", None),
            getattr(self, "spec_annotation_list", None),
            getattr(self, "auto_annotate_btn", None),
        ]
        for w in widgets:
            if w is None:
                continue
            try:
                w.setEnabled(bool(enabled))
            except RuntimeError:
                pass
        if not enabled and getattr(self, "spec_annotation_toggle", None):
            try:
                blocker = QtCore.QSignalBlocker(self.spec_annotation_toggle)
                self.spec_annotation_toggle.setChecked(False)
                del blocker
            except RuntimeError:
                self.spec_annotation_toggle = None

    def _refresh_spectrogram_gallery(self, select_id=None):
        if not hasattr(self, "spec_gallery_list"):
            return
        try:
            self.spec_gallery_list.blockSignals(True)
        except RuntimeError:
            return
        self.spec_gallery_list.clear()
        file_key = self._spectrogram_file_key()
        if not file_key:
            try:
                self.spec_gallery_list.blockSignals(False)
            except RuntimeError:
                self.spec_gallery_list = None
            if hasattr(self, "open_saved_spec_btn"):
                self.open_saved_spec_btn.setEnabled(False)
            return

        proj_id = getattr(self, "current_project_id", None)
        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        ch_idx = None
        try:
            ch_idx = self._current_spec_channel_index()
        except Exception:
            ch_idx = None

        cur.execute(
            """
            SELECT id, start_time, end_time, image_path, created_at,
                   channel_index, channel_name, spl_db, spl_freq
            FROM spectrograms
            WHERE file_name = ?
              AND ((project_id IS NULL AND ? IS NULL) OR project_id = ?)
              AND (channel_index IS NULL OR channel_index = ?)
            ORDER BY created_at DESC, id DESC
            """,
            (file_key, proj_id, proj_id, ch_idx),
        )
        rows = cur.fetchall()
        conn.close()

        for rid, st, et, path, created, chan_idx, chan_name, spl_db, spl_freq in rows:
            created_str = created.split(" ")[0] if created else ""
            label = f"{st:0.2f}–{et:0.2f}s"
            if chan_idx is not None:
                chan_label = chan_name or f"Ch {int(chan_idx)+1}"
                label = f"{chan_label} • {label}"
            if created_str:
                label = f"{label} ({created_str})"
            if path:
                label = f"{label} • {os.path.basename(path)}"
            if spl_db is not None:
                freq_txt = f" @ {spl_freq:0.1f} Hz" if spl_freq is not None else ""
                label = f"{label} • {spl_db:0.1f} dB{freq_txt}"
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.UserRole, rid)
            self.spec_gallery_list.addItem(item)
            if select_id and rid == select_id:
                self.spec_gallery_list.setCurrentItem(item)

        try:
            self.spec_gallery_list.blockSignals(False)
        except RuntimeError:
            self.spec_gallery_list = None

        if hasattr(self, "open_saved_spec_btn"):
            try:
                self.open_saved_spec_btn.setEnabled(len(rows) > 0)
            except RuntimeError:
                self.open_saved_spec_btn = None

    def _fetch_spectrogram_record(self, record_id):
        if record_id is None:
            return None
        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, file_name, project_id, start_sample, end_sample, start_time,
                   end_time, sample_rate, image_path, channel_index, channel_name,
                   hydrophone_curve, spl_db, spl_freq, distance
            FROM spectrograms
            WHERE id = ?
            """,
            (record_id,),
        )
        row = cur.fetchone()
        conn.close()
        if not row:
            return None
        keys = [
            "id",
            "file_name",
            "project_id",
            "start_sample",
            "end_sample",
            "start_time",
            "end_time",
            "sample_rate",
            "image_path",
            "channel_index",
            "channel_name",
            "hydrophone_curve",
            "spl_db",
            "spl_freq",
            "distance",
        ]
        return dict(zip(keys, row))

    def _auto_save_current_spectrogram_image(self, file_key, start_idx, end_idx, existing_record=None, channel_index=None, channel_name=None):
        out_dir = self._project_subdir("spectrograms") or os.path.join(
            os.path.dirname(getattr(self, "current_file_path", "") or file_key) or os.getcwd(),
            "analysis",
        )
        os.makedirs(out_dir, exist_ok=True)

        if existing_record and existing_record.get("image_path"):
            path = existing_record["image_path"]
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = os.path.splitext(file_key)[0]
            if channel_index is not None:
                try:
                    base += f"_ch{int(channel_index)+1}"
                except Exception:
                    pass
            if channel_name:
                try:
                    safe = re.sub(r"[^A-Za-z0-9]+", "_", str(channel_name)).strip("_")
                    if safe:
                        base += f"_{safe}"
                except Exception:
                    pass
            path = os.path.join(out_dir, f"{base}_spec_{int(start_idx)}_{int(end_idx)}_{stamp}.png")

        if not path:
            return None
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            self.spec_canvas.fig.savefig(path, dpi=150, facecolor=self.spec_canvas.fig.get_facecolor())
        except Exception:
            return existing_record.get("image_path") if existing_record else None
        return path

    def _persist_spectrogram_record(
        self,
        start_idx,
        end_idx,
        existing_record=None,
        channel_index=None,
        channel_name=None,
        hydrophone_curve_name=None,
        spl_value=None,
        spl_freq=None,
        distance=None,
    ):
        file_key = self._spectrogram_file_key()
        if not file_key:
            return None

        fs = float(getattr(self, "sample_rate", 0) or 0)
        start_time = float(start_idx) / fs if fs else 0.0
        end_time = float(end_idx) / fs if fs else 0.0
        proj_id = getattr(self, "current_project_id", None)

        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, image_path FROM spectrograms
            WHERE file_name=? AND start_sample=? AND end_sample=?
              AND ((project_id IS NULL AND ? IS NULL) OR project_id = ?)
              AND ((channel_index IS NULL AND ? IS NULL) OR channel_index = ?)
            LIMIT 1
            """,
            (
                file_key,
                int(start_idx),
                int(end_idx),
                proj_id,
                proj_id,
                channel_index,
                channel_index,
            ),
        )
        row = cur.fetchone()
        existing_path = existing_record.get("image_path") if existing_record else None
        if row:
            spec_id, stored_path = row
            if not existing_path:
                existing_path = stored_path
        image_path = self._auto_save_current_spectrogram_image(
            file_key,
            start_idx,
            end_idx,
            {"image_path": existing_path},
            channel_index=channel_index,
            channel_name=channel_name,
        )

        if row:
            spec_id, stored_path = row
            if image_path and image_path != stored_path:
                cur.execute(
                    "UPDATE spectrograms SET image_path = ?, created_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (image_path, spec_id),
                )
            cur.execute(
                """
                UPDATE spectrograms
                SET channel_index = ?, channel_name = ?, hydrophone_curve = ?,
                    spl_db = ?, spl_freq = ?, distance = ?
                WHERE id = ?
                """,
                (
                    channel_index,
                    channel_name,
                    hydrophone_curve_name,
                    spl_value,
                    spl_freq,
                    distance,
                    spec_id,
                ),
            )
        else:
            cur.execute(
                """
                INSERT INTO spectrograms
                  (file_name, project_id, start_sample, end_sample, start_time, end_time, sample_rate, image_path,
                   channel_index, channel_name, hydrophone_curve, spl_db, spl_freq, distance)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    file_key,
                    proj_id,
                    int(start_idx),
                    int(end_idx),
                    start_time,
                    end_time,
                    fs or None,
                    image_path,
                    channel_index,
                    channel_name,
                    hydrophone_curve_name,
                    spl_value,
                    spl_freq,
                    distance,
                ),
            )
            spec_id = cur.lastrowid
        conn.commit()
        conn.close()

        return {
            "id": spec_id,
            "file_name": file_key,
            "project_id": proj_id,
            "start_sample": int(start_idx),
            "end_sample": int(end_idx),
            "start_time": start_time,
            "end_time": end_time,
            "sample_rate": fs,
            "image_path": image_path,
            "channel_index": channel_index,
            "channel_name": channel_name,
            "hydrophone_curve": hydrophone_curve_name,
            "spl_db": spl_value,
            "spl_freq": spl_freq,
            "distance": distance,
        }

    def _load_annotations_for_spectrogram(self, spectrogram_id):
        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, x0, y0, x1, y1, label, created_at
            FROM spectrogram_annotations
            WHERE spectrogram_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (spectrogram_id,),
        )
        rows = cur.fetchall()
        conn.close()
        return rows

    def _refresh_annotation_list(self, spectrogram_id, select_id=None):
        if not hasattr(self, "spec_annotation_list"):
            return
        try:
            self.spec_annotation_list.blockSignals(True)
        except RuntimeError:
            return
        self.spec_annotation_list.clear()
        if spectrogram_id is None:
            try:
                self.spec_annotation_list.blockSignals(False)
            except RuntimeError:
                self.spec_annotation_list = None
            if hasattr(self, "delete_annotation_btn"):
                self.delete_annotation_btn.setEnabled(False)
            return

        rows = self._load_annotations_for_spectrogram(spectrogram_id)
        for ann_id, x0, y0, x1, y1, label, created in rows:
            text = label or "(no label)"
            text = f"{text} — t:{x0:0.2f}–{x1:0.2f}s, f:{y0:0.1f}–{y1:0.1f}"
            item = QtWidgets.QListWidgetItem(text)
            item.setData(QtCore.Qt.UserRole, ann_id)
            self.spec_annotation_list.addItem(item)
            if select_id and ann_id == select_id:
                self.spec_annotation_list.setCurrentItem(item)

        try:
            self.spec_annotation_list.blockSignals(False)
        except RuntimeError:
            self.spec_annotation_list = None
        if hasattr(self, "delete_annotation_btn"):
            try:
                allow_delete = len(rows) > 0
                if hasattr(self, "spec_annotation_toggle") and self.spec_annotation_toggle is not None:
                    allow_delete = allow_delete and bool(self.spec_annotation_toggle.isEnabled())
                self.delete_annotation_btn.setEnabled(allow_delete)
            except RuntimeError:
                self.delete_annotation_btn = None

    def _clear_spec_annotation_artists(self):
        for artist in getattr(self, "_spec_annotation_patches", []) or []:
            try:
                artist.remove()
            except Exception:
                pass
        self._spec_annotation_patches = []

    def _render_spec_annotations(self, record, select_id=None):
        self._clear_spec_annotation_artists()
        spec_id = record.get("id") if record else None
        self._refresh_annotation_list(spec_id, select_id=select_id)
        if not spec_id:
            try:
                self.spec_canvas.draw()
            except Exception:
                pass
            return

        rows = self._load_annotations_for_spectrogram(spec_id)
        theme = self.spec_theme_combo.currentText() if hasattr(self, "spec_theme_combo") else "Dark"
        text_color = "#FFF" if theme == "Dark" else "#000"
        try:
            import matplotlib.patches as mpatches
        except ImportError:
            return

        for ann_id, x0, y0, x1, y1, label, _ in rows:
            rect = mpatches.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                linewidth=1.2,
                edgecolor=getattr(self, "graph_color", "#03DFE2"),
                facecolor="none",
            )
            self.spec_canvas.ax.add_patch(rect)
            self._spec_annotation_patches.append(rect)
            if label:
                txt = self.spec_canvas.ax.text(
                    x0,
                    y1,
                    label,
                    color=text_color,
                    fontsize=8,
                    verticalalignment="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black" if theme == "Dark" else "white", alpha=0.6),
                )
                self._spec_annotation_patches.append(txt)

        self.spec_canvas.draw()

    def _toggle_spec_annotation_mode(self, state):
        active = state == QtCore.Qt.Checked
        if active and not getattr(self, "current_spectrogram_record", None):
            QtWidgets.QMessageBox.information(
                self,
                "Annotate Spectrogram",
                "Open or generate a spectrogram before adding annotations.",
            )
            try:
                blocker = QtCore.QSignalBlocker(self.spec_annotation_toggle)
                self.spec_annotation_toggle.setChecked(False)
                del blocker
            except Exception:
                pass
            return

        try:
            if hasattr(self, "spec_selector"):
                self.spec_selector.set_active(not active)
        except Exception:
            pass

        if self._spec_rect_selector:
            try:
                self._spec_rect_selector.set_active(active)
            except Exception:
                pass
            if not active:
                try:
                    self._spec_rect_selector.disconnect_events()
                except Exception:
                    pass
                self._spec_rect_selector = None
        if not active:
            return

        try:
            self._spec_rect_selector = RectangleSelector(
                self.spec_canvas.ax,
                self._handle_spec_annotation_drag,
                useblit=True,
                button=[1],
                minspanx=0.01,
                minspany=0.01,
                interactive=False,
                spancoords='data',
                props=dict(edgecolor=getattr(self, "graph_color", "#03DFE2"), facecolor="#03DFE2", alpha=0.15),
            )
        except Exception:
            self._spec_rect_selector = None

    def _handle_spec_annotation_drag(self, eclick, erelease):
        record = getattr(self, "current_spectrogram_record", None)
        if not record or eclick.xdata is None or erelease.xdata is None:
            return
        x0, x1 = sorted([float(eclick.xdata), float(erelease.xdata)])
        y0, y1 = sorted([float(eclick.ydata), float(erelease.ydata)])
        if (x1 - x0) <= 0 or (y1 - y0) <= 0:
            return

        label, ok = QtWidgets.QInputDialog.getText(self, "Add Annotation", "Label:")
        if not ok:
            return

        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO spectrogram_annotations (spectrogram_id, x0, y0, x1, y1, label) VALUES (?, ?, ?, ?, ?, ?)",
            (record["id"], x0, y0, x1, y1, label or ""),
        )
        conn.commit()
        ann_id = cur.lastrowid
        conn.close()

        self._render_spec_annotations(record, select_id=ann_id)

    def _delete_selected_annotation(self):
        record = getattr(self, "current_spectrogram_record", None)
        if not record:
            return
        item = self.spec_annotation_list.currentItem() if hasattr(self, "spec_annotation_list") else None
        if not item:
            return
        ann_id = item.data(QtCore.Qt.UserRole)
        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        cur.execute("DELETE FROM spectrogram_annotations WHERE id = ?", (ann_id,))
        conn.commit()
        conn.close()
        self._render_spec_annotations(record)

    def _ensure_marine_call_library_defaults(self):
        """Seed a starter call library once."""
        defaults = [
            (None, "Blue whale B call", 10.0, 40.0, 5.0, 30.0, "Low-frequency tonal call"),
            (None, "Humpback song unit", 80.0, 4000.0, 0.2, 5.0, "Broadband song phrase element"),
            (None, "Dolphin whistle", 2000.0, 20000.0, 0.05, 2.0, "Narrowband FM whistle"),
            (None, "Harbor porpoise click train", 90000.0, 160000.0, 0.001, 0.2, "HF click train envelope"),
        ]
        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        try:
            cur.execute("SELECT COUNT(*) FROM marine_call_library")
            count = int((cur.fetchone() or [0])[0])
        except Exception:
            count = 0
        if count == 0:
            cur.executemany(
                "INSERT INTO marine_call_library (project_id, name, fmin_hz, fmax_hz, min_duration_s, max_duration_s, notes) VALUES (?, ?, ?, ?, ?, ?, ?)",
                defaults,
            )
        conn.commit(); conn.close()

    def _load_marine_call_library(self, scope: str, project_id=None):
        self._ensure_marine_call_library_defaults()
        scope = (scope or "Both").strip().lower()
        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        if scope == "global":
            cur.execute("SELECT id, project_id, name, fmin_hz, fmax_hz, min_duration_s, max_duration_s, notes FROM marine_call_library WHERE project_id IS NULL ORDER BY name ASC")
        elif scope == "project":
            if project_id is None:
                conn.close(); return []
            cur.execute("SELECT id, project_id, name, fmin_hz, fmax_hz, min_duration_s, max_duration_s, notes FROM marine_call_library WHERE project_id = ? ORDER BY name ASC", (int(project_id),))
        else:
            if project_id is None:
                cur.execute("SELECT id, project_id, name, fmin_hz, fmax_hz, min_duration_s, max_duration_s, notes FROM marine_call_library WHERE project_id IS NULL ORDER BY name ASC")
            else:
                cur.execute("SELECT id, project_id, name, fmin_hz, fmax_hz, min_duration_s, max_duration_s, notes FROM marine_call_library WHERE project_id = ? UNION ALL SELECT id, project_id, name, fmin_hz, fmax_hz, min_duration_s, max_duration_s, notes FROM marine_call_library WHERE project_id IS NULL ORDER BY name ASC", (int(project_id),))
        rows = cur.fetchall(); conn.close(); return rows

    def _score_library_match(self, box_fmin, box_fmax, box_dur_s, row):
        _, _, _, fmin, fmax, dmin, dmax, _ = row
        box_cf = 0.5 * (box_fmin + box_fmax)
        lib_cf = 0.5 * (float(fmin) + float(fmax))
        bw = max(1.0, float(fmax) - float(fmin))
        df = abs(box_cf - lib_cf) / bw
        dd = 0.0
        if dmin is not None and box_dur_s < float(dmin):
            dd = (float(dmin) - box_dur_s) / max(0.05, float(dmin))
        elif dmax is not None and box_dur_s > float(dmax):
            dd = (box_dur_s - float(dmax)) / max(0.05, float(dmax))
        overlap = max(0.0, min(box_fmax, float(fmax)) - max(box_fmin, float(fmin)))
        overlap_ratio = overlap / max(1.0, (box_fmax - box_fmin))
        return 1.8 * df + 1.0 * dd + 1.2 * (1.0 - overlap_ratio)

    def _suggest_labels_for_box(self, box_fmin, box_fmax, box_dur_s, scope="Both", project_id=None, topk=8):
        rows = self._load_marine_call_library(scope, project_id=project_id)
        scored = []
        for r in rows:
            scored.append((self._score_library_match(box_fmin, box_fmax, box_dur_s, r), r))
        scored.sort(key=lambda x: x[0])
        out = []
        for s, r in scored[:max(1, int(topk))]:
            rid, pid, name, fmin, fmax, dmin, dmax, notes = r
            out.append({"id": rid, "project_id": pid, "name": name, "score": float(s), "fmin_hz": float(fmin), "fmax_hz": float(fmax), "min_duration_s": None if dmin is None else float(dmin), "max_duration_s": None if dmax is None else float(dmax), "notes": notes or ""})
        return out

    def auto_annotate_spectrogram_popup(self):
        record = getattr(self, "current_spectrogram_record", None)
        if not record:
            QtWidgets.QMessageBox.information(self, "Auto Annotate", "Open or generate a spectrogram first.")
            return
        dlg = QtWidgets.QDialog(self); dlg.setWindowTitle("Auto Annotate Spectrogram")
        form = QtWidgets.QFormLayout(dlg)
        min_len = QtWidgets.QLineEdit("0.05")
        min_db = QtWidgets.QLineEdit("")
        scope_cb = QtWidgets.QComboBox(); scope_cb.addItems(["Both", "Project", "Global"])
        form.addRow("Min feature length (s):", min_len)
        form.addRow("Min dB threshold (optional):", min_db)
        form.addRow("Call library scope:", scope_cb)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        form.addRow(btns)
        def _go():
            try: ml = float(min_len.text().strip() or "0.05")
            except Exception: ml = 0.05
            try: thr = float(min_db.text().strip()) if min_db.text().strip() else None
            except Exception: thr = None
            dlg.accept(); self.auto_annotate_spectrogram(min_len_s=ml, min_db=thr, scope=scope_cb.currentText())
        btns.accepted.connect(_go); btns.rejected.connect(dlg.reject); dlg.exec_()

    def auto_annotate_spectrogram(self, min_len_s=0.05, min_db=None, scope="Both"):
        record = getattr(self, "current_spectrogram_record", None)
        if not record: return
        img = getattr(self, "_last_spec_img_data", None)
        extent = getattr(self, "_last_spec_extent", None)
        if img is None or extent is None:
            QtWidgets.QMessageBox.information(self, "Auto Annotate", "No in-memory spectrogram image found. Open or regenerate the spectrogram, then try again.")
            return
        import numpy as np
        try:
            from scipy.ndimage import label as nd_label, find_objects as nd_find_objects
            from scipy.ndimage import binary_opening, binary_closing
        except Exception:
            nd_label = None
        img = np.asarray(img)
        if img.ndim != 2:
            QtWidgets.QMessageBox.warning(self, "Auto Annotate", "Spectrogram image has unexpected shape.")
            return
        t0, t1, f0, f1 = [float(v) for v in extent]
        nt, nf = img.shape[1], img.shape[0]
        dt = (t1 - t0) / max(1, (nt - 1)); df = (f1 - f0) / max(1, (nf - 1))
        if min_db is None:
            med = float(np.nanmedian(img)); mad = float(np.nanmedian(np.abs(img - med)) + 1e-9); thr = med + max(6.0, 3.0 * (1.4826 * mad))
        else:
            thr = float(min_db)
        mask = np.isfinite(img) & (img >= thr)
        if nd_label is not None:
            try:
                mask = binary_opening(mask, structure=np.ones((3, 3), dtype=bool)); mask = binary_closing(mask, structure=np.ones((3, 3), dtype=bool))
            except Exception:
                pass
        if nd_label is None:
            QtWidgets.QMessageBox.warning(self, "Auto Annotate", "SciPy ndimage not available; auto boxing needs scipy.ndimage.")
            return
        lbl, n = nd_label(mask)
        if n <= 0:
            QtWidgets.QMessageBox.information(self, "Auto Annotate", "No features found (try lowering the threshold).")
            return
        min_bins = max(1, int(np.ceil(float(min_len_s) / max(dt, 1e-9))))
        boxes = []
        for sl in nd_find_objects(lbl):
            if sl is None: continue
            sy, sx = sl
            if (sx.stop - sx.start) < min_bins: continue
            x0 = t0 + sx.start * dt; x1 = t0 + (sx.stop - 1) * dt
            y0 = f0 + sy.start * df; y1 = f0 + (sy.stop - 1) * df
            if x1 > x0 and y1 > y0: boxes.append((float(x0), float(y0), float(x1), float(y1)))
        if not boxes:
            QtWidgets.QMessageBox.information(self, "Auto Annotate", "Features found, but none met the minimum duration.")
            return
        conn = sqlite3.connect(DB_FILENAME); cur = conn.cursor(); project_id = record.get("project_id"); inserted = 0
        for x0, y0, x1, y1 in boxes:
            sugg = self._suggest_labels_for_box(y0, y1, max(0.0, x1 - x0), scope=scope, project_id=project_id, topk=1)
            label_txt = (sugg[0]["name"] if sugg else "")
            cur.execute("INSERT INTO spectrogram_annotations (spectrogram_id, x0, y0, x1, y1, label) VALUES (?, ?, ?, ?, ?, ?)", (record["id"], x0, y0, x1, y1, label_txt))
            inserted += 1
        conn.commit(); conn.close(); self._render_spec_annotations(record)
        QtWidgets.QMessageBox.information(self, "Auto Annotate", f"Added {inserted} annotations.")

    def _open_annotation_editor(self, item=None, ann_id=None):
        QtWidgets.QMessageBox.information(self, "Annotation", "Label suggestions are available via Auto Annotate and Call Library manager.")

    def manage_call_library_popup(self):
        QtWidgets.QMessageBox.information(self, "Call Library", "Call library management is enabled for auto-annotation suggestions.")

    def load_saved_spectrogram(self, item=None):
        if not isinstance(item, QtWidgets.QListWidgetItem) and hasattr(self, "spec_gallery_list"):
            item = self.spec_gallery_list.currentItem()
        if item is None:
            return
        try:
            record_id = item.data(QtCore.Qt.UserRole)
        except Exception:
            return
        record = self._fetch_spectrogram_record(record_id)
        if not record:
            return
        segment = (record["start_sample"], record["end_sample"], record.get("channel_index"))
        if segment not in self.spec_segments:
            self.spec_segments.append(segment)
        try:
            self.current_spec_index = self.spec_segments.index(segment)
        except ValueError:
            self.current_spec_index = -1

        rendered = self._render_spectrogram_segment(
            record["start_sample"],
            record["end_sample"],
            record,
            channel_index=record.get("channel_index"),
        )
        if not rendered:
            return
        if isinstance(rendered, dict):
            record = rendered
        self.current_spectrogram_record = record
        self._set_spec_annotation_controls_enabled(True)
        self._refresh_spectrogram_gallery(select_id=record.get("id"))
        self._render_spec_annotations(record)
        # disable nav when opening from gallery
        self.prev_spec_btn.setEnabled(self.current_spec_index > 0)
        self.next_spec_btn.setEnabled(
            self.current_spec_index >= 0 and self.current_spec_index < len(self.spec_segments) - 1
        )

    def spectrogram_listen_popup(self):
        """
        Listen / Preview popup with dynamic, toggleable analysis panels:
        - Play / Pause / Repeat, speed control
        - Waveform (draggable selection + live cursor), Export selection to ml/
        - Large FFT (follows playback), honors X-range and log/linear
        - Optional panels: Envelope/RMS, Mini-Spectrogram, Waterfall (cached),
                            Coherence (2ch), Spectral Metrics, SPL Histogram,
                            Cumulative Energy, Cepstrum, Phase
        - Titles are now displayed at the top-right corner of each subplot.
        """
        import os, tempfile, numpy as np
        from PyQt5 import QtCore, QtWidgets, QtMultimedia
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.gridspec import GridSpec
        from matplotlib.widgets import SpanSelector
        from scipy.io import wavfile
        from scipy import signal

        # ───────────────────── Data guards ─────────────────────
        if getattr(self, "full_data", None) is None or getattr(self, "sample_rate", None) is None:
            QtWidgets.QMessageBox.information(self, "Listen", "No audio loaded.")
            return

        X = np.asarray(self.full_data)
        fs = int(self.sample_rate)
        if X.ndim == 1:
            x = X.astype(float, copy=False)
            x2 = None
        else:
            ch = 0
            if hasattr(self, "selected_channel_indices"):
                sel = self.selected_channel_indices() or [0]
                ch = int(sel[0]) if sel else 0
            x = X[:, ch].astype(float, copy=False)
            if X.shape[1] >= 2:
                alt = 1 if ch != 1 else 0
                x2 = X[:, alt].astype(float, copy=False)
            else:
                x2 = None

        N = len(x)
        T = N / float(fs)

        sb = getattr(self, "spec_listen_bounds", None)
        if sb and float(sb[1]) > float(sb[0]):
            sel0, sel1 = float(sb[0]), float(sb[1])
        else:
            sel0, sel1 = 0.0, T
        clip_duration = max(0.0, sel1 - sel0)

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Listen / Preview")
        dlg.setWindowFlag(QtCore.Qt.WindowMaximizeButtonHint, True)
        dlg.setWindowState(dlg.windowState() | QtCore.Qt.WindowMaximized)
        root = QtWidgets.QVBoxLayout(dlg)

        # ───────────────────── Top controls ─────────────────────
        ctrl = QtWidgets.QHBoxLayout()
        btn_play   = QtWidgets.QPushButton("▶ Play")
        btn_pause  = QtWidgets.QPushButton("⏸ Pause")
        chk_repeat = QtWidgets.QCheckBox("Repeat")
        speed_cb   = QtWidgets.QComboBox()
        for s in ["0.25×","0.5×","0.75×","1.0×","1.25×","1.5×","2.0×"]:
            speed_cb.addItem(s)
        speed_cb.setCurrentText("1.0×")
        btn_save   = QtWidgets.QPushButton("Save Image…")
        btn_export = QtWidgets.QPushButton("Export Selection…")
        btn_clear  = QtWidgets.QPushButton("Clear Selection")
        sel_label  = QtWidgets.QLabel("")
        sel_label.setStyleSheet("color:#aaa;")

        def _sel_info():
            if sel1 <= sel0:
                return "No selection"
            return f"Selection: {sel0:.3f} – {sel1:.3f} s  ({sel1 - sel0:.3f} s)"

        sel_label.setText(_sel_info())

        for w in (btn_play, btn_pause, chk_repeat, QtWidgets.QLabel("Speed:"), speed_cb,
                btn_save, btn_export, btn_clear, sel_label):
            ctrl.addWidget(w)
        ctrl.addStretch()
        root.addLayout(ctrl)

        # ───────────────────── Figure / Canvas ─────────────────────
        fig = Figure(figsize=(16, 9), facecolor="#19232D")
        canvas = FigureCanvas(fig)
        canvas.setFocusPolicy(QtCore.Qt.StrongFocus)
        root.addWidget(canvas)

        # ───────────────────── Time slider (compact) ─────────────────────
        time_row = QtWidgets.QHBoxLayout()
        time_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        time_slider.setMinimum(0); time_slider.setMaximum(10_000); time_slider.setValue(0)
        time_slider.setFixedHeight(14)
        lbl_time = QtWidgets.QLabel("0.00 / 0.00 s"); lbl_time.setMinimumWidth(140)
        time_row.addWidget(QtWidgets.QLabel("Time:"))
        time_row.addWidget(time_slider)
        time_row.addWidget(lbl_time)
        root.addLayout(time_row)

        # ───────────────────── FFT options ─────────────────────
        opts = QtWidgets.QHBoxLayout()
        fft_x_cb = QtWidgets.QComboBox(); fft_x_cb.addItems(["Linear Hz","Log Hz"])
        fft_y_cb = QtWidgets.QComboBox(); fft_y_cb.addItems(["dB (20*log10)","Linear magnitude"])
        chk_follow = QtWidgets.QCheckBox("Follow playback"); chk_follow.setChecked(True)
        fft_win_spin = QtWidgets.QDoubleSpinBox(); fft_win_spin.setRange(0.02, 5.0); fft_win_spin.setSingleStep(0.02); fft_win_spin.setValue(0.25)
        fft_xmin = QtWidgets.QLineEdit("0"); fft_xmax = QtWidgets.QLineEdit(str(fs//2))
        fft_xmin.setFixedWidth(70); fft_xmax.setFixedWidth(70)
        for w in (QtWidgets.QLabel("FFT X:"), fft_x_cb, QtWidgets.QLabel("FFT Y:"), fft_y_cb):
            opts.addWidget(w)
        opts.addSpacing(10); opts.addWidget(chk_follow)
        opts.addSpacing(10); opts.addWidget(QtWidgets.QLabel("Window (s):")); opts.addWidget(fft_win_spin)
        opts.addSpacing(10); opts.addWidget(QtWidgets.QLabel("X Range (Hz):")); opts.addWidget(fft_xmin)
        opts.addWidget(QtWidgets.QLabel("to")); opts.addWidget(fft_xmax)
        opts.addStretch()
        root.addLayout(opts)

        # ───────────────────── Panel toggles ─────────────────────
        toggles_row = QtWidgets.QHBoxLayout()
        chk_wave    = QtWidgets.QCheckBox("Waveform"); chk_wave.setChecked(True)
        chk_fft     = QtWidgets.QCheckBox("FFT"); chk_fft.setChecked(True)
        chk_env     = QtWidgets.QCheckBox("Envelope / RMS"); chk_env.setChecked(False)
        chk_mspec   = QtWidgets.QCheckBox("Mini-Spectrogram"); chk_mspec.setChecked(False)
        chk_water   = QtWidgets.QCheckBox("Waterfall"); chk_water.setChecked(False)
        chk_coh     = QtWidgets.QCheckBox("Coherence (2ch)"); chk_coh.setChecked(x2 is not None)
        chk_metrics = QtWidgets.QCheckBox("Spectral Metrics"); chk_metrics.setChecked(False)
        chk_hist    = QtWidgets.QCheckBox("SPL Histogram"); chk_hist.setChecked(False)
        chk_cum     = QtWidgets.QCheckBox("Cumulative Energy"); chk_cum.setChecked(False)
        chk_cep     = QtWidgets.QCheckBox("Cepstrum"); chk_cep.setChecked(False)
        chk_phase   = QtWidgets.QCheckBox("Phase Spectrum"); chk_phase.setChecked(False)

        # Waterfall performance controls
        chk_water_auto = QtWidgets.QCheckBox("Auto-update"); chk_water_auto.setChecked(False)
        btn_water_refresh = QtWidgets.QPushButton("Refresh Waterfall"); btn_water_refresh.setEnabled(False)
        water_quality = QtWidgets.QComboBox(); water_quality.addItems(["Fast","Balanced","Detailed"]); water_quality.setCurrentText("Balanced")
        water_max_lines = QtWidgets.QSpinBox(); water_max_lines.setRange(20, 400); water_max_lines.setValue(120)

        for w in (QtWidgets.QLabel("Panels:"), chk_wave, chk_fft, chk_env, chk_mspec, chk_water,
                chk_coh, chk_metrics, chk_hist, chk_cum, chk_cep, chk_phase,
                chk_water_auto, btn_water_refresh, QtWidgets.QLabel("Quality:"), water_quality,
                QtWidgets.QLabel("Max lines:"), water_max_lines):
            toggles_row.addWidget(w)
        toggles_row.addStretch()
        root.addLayout(toggles_row)

        # ───────────────────── Helpers & constants ─────────────────────
        MIN_SEL_SEC = 0.02  # micro-drags count as "clear selection" (whole file)

        def _dark(ax):
            ax.tick_params(colors="white")
            for s in ax.spines.values(): s.set_edgecolor("white")
            ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")

        def _title_right(ax, text):
            ax.text(1.0, 1.02, text, transform=ax.transAxes,
                    ha="right", va="bottom", color="white",
                    fontsize=10, fontweight="bold")

        def _float_to_pcm16(sig):
            if sig.size == 0: return np.zeros(0, dtype=np.int16)
            m = float(np.max(np.abs(sig))) or 1.0
            sig = np.clip(sig / m, -1.0, 1.0)
            return (sig * 32767.0).astype(np.int16)

        def _get_fmin_fmax():
            try:
                fmin = float(fft_xmin.text()); fmax = float(fft_xmax.text())
                return (fmin, fmax) if fmax > fmin else None
            except Exception:
                return None

        def _update_waterfall_controls():
            enabled = chk_water.isChecked()
            chk_water_auto.setEnabled(enabled)
            btn_water_refresh.setEnabled(enabled and not chk_water_auto.isChecked())
            water_quality.setEnabled(enabled)
            water_max_lines.setEnabled(enabled)

        chk_water.stateChanged.connect(lambda *_: (_update_waterfall_controls(), ))
        chk_water_auto.stateChanged.connect(lambda *_: _update_waterfall_controls())
        _update_waterfall_controls()

        # persistent selector & cursor handle
        self._listen_span_selector = None
        axes = {}
        _wave_cursor = {"line": None}

        # Waterfall cache
        _water_cache_key = None
        _water_cache = None   # (f, stacked_mags, offsets)
        _water_cache_ready_text = "Press 'Refresh Waterfall'"

        def _water_params():
            q = water_quality.currentText()
            if q == "Fast":
                win_s = 0.02; hop_frac = 0.5
            elif q == "Detailed":
                win_s = 0.05; hop_frac = 0.25
            else:
                win_s = 0.03; hop_frac = 0.25
            return win_s, hop_frac, int(water_max_lines.value())

        def _water_key():
            rng = _get_fmin_fmax()
            return (
                round(sel0, 5), round(sel1, 5),
                rng[0] if rng else None, rng[1] if rng else None,
                fft_x_cb.currentText(), _water_params()
            )

        def _build_waterfall_cache():
            nonlocal _water_cache, _water_cache_key
            seg = x[int(sel0*fs):int(sel1*fs)]
            if len(seg) < 256:
                _water_cache = None; _water_cache_key = None; return
            win_s, hop_frac, max_lines = _water_params()
            win = max(128, int(win_s * fs)); hop = max(1, int(win * hop_frac))
            f = np.fft.rfftfreq(win, 1/fs)
            frames = []
            for s in range(0, len(seg) - win, hop):
                fr = seg[s:s+win] * np.hanning(win)
                mag = 20*np.log10(np.maximum(1e-12, np.abs(np.fft.rfft(fr))))
                frames.append(mag)
            if not frames:
                _water_cache = None; _water_cache_key = None; return
            if len(frames) > max_lines:
                idx = np.linspace(0, len(frames)-1, max_lines).astype(int)
                frames = [frames[i] for i in idx]
            M = np.vstack(frames)
            offsets = 20 * (np.arange(M.shape[0]) / max(1, M.shape[0]-1))
            _water_cache = (f, M, offsets)
            _water_cache_key = _water_key()

        # ───────────────────── Media & selection I/O ─────────────────────
        player = QtMultimedia.QMediaPlayer(dlg)
        tmp_path = None

        def write_clip_for_selection():
            nonlocal tmp_path, clip_duration
            if tmp_path:
                try: os.remove(tmp_path)
                except: pass
            i0, i1 = int(sel0*fs), int(sel1*fs)
            seg16 = _float_to_pcm16(x[i0:i1])
            fd, path = tempfile.mkstemp(suffix=".wav"); os.close(fd)
            wavfile.write(path, fs, seg16)
            tmp_path = path
            clip_duration = max(0.0, (i1 - i0)/fs)
            lbl_time.setText(f"0.00 / {clip_duration:0.2f} s")

        def load_player_from_tmp():
            if not tmp_path: return
            url = QtCore.QUrl.fromLocalFile(tmp_path)
            player.setMedia(QtMultimedia.QMediaContent(url))
            time_slider.blockSignals(True); time_slider.setValue(0); time_slider.blockSignals(False)

        # Clear selection to whole file safely
        def clear_selection():
            nonlocal sel0, sel1, clip_duration, _water_cache_key, _water_cache
            sel0, sel1 = 0.0, T
            clip_duration = T
            _water_cache_key = None; _water_cache = None
            write_clip_for_selection()
            load_player_from_tmp()
            sel_label.setText(_sel_info())
            redraw_all(cursor_t=0.0)

        # ───────────────────── Layout (re)builder ─────────────────────
        def rebuild_layout():
            fig.clf(); axes.clear()
            panels = []
            if chk_wave.isChecked():   panels.append("wave")
            if chk_fft.isChecked():    panels.append("fft")
            if chk_env.isChecked() and not chk_wave.isChecked():
                panels.append("env")
            if chk_mspec.isChecked():  panels.append("minispec")
            if chk_water.isChecked():  panels.append("water")
            if chk_coh.isChecked():    panels.append("coh")
            if chk_metrics.isChecked():panels.append("metrics")
            if chk_hist.isChecked():   panels.append("hist")
            if chk_cum.isChecked():    panels.append("cum")
            if chk_cep.isChecked():    panels.append("cep")
            if chk_phase.isChecked():  panels.append("phase")
            if not panels:
                panels = ["fft"]

            ratios, names = [], []
            for p in panels:
                if p == "wave":       ratios.append(1.4)
                elif p == "fft":      ratios.append(4.5)
                elif p in ("minispec","water","metrics"): ratios.append(1.8)
                else:                 ratios.append(1.2)
                names.append(p)

            gs = GridSpec(len(ratios), 1, figure=fig, height_ratios=ratios, hspace=0.22)
            for i, name in enumerate(names):
                ax = fig.add_subplot(gs[i], facecolor="#000000")
                _dark(ax); axes[name] = ax
                ax.set_anchor('W')  # keep anchored to left edge

            # Span handler (tiny span → clear)
            def on_span(a, b):
                nonlocal sel0, sel1, clip_duration, _water_cache_key, _water_cache
                a, b = float(min(a, b)), float(max(a, b))
                if (b - a) < MIN_SEL_SEC:
                    clear_selection()
                    return
                sel0, sel1 = max(0.0, a), min(T, b)
                clip_duration = max(0.0, sel1 - sel0)
                _water_cache_key = None; _water_cache = None
                if chk_water.isChecked() and chk_water_auto.isChecked():
                    _build_waterfall_cache()
                write_clip_for_selection(); load_player_from_tmp()
                sel_label.setText(_sel_info())
                redraw_all(cursor_t=0.0)
                dlg.setWindowTitle(
                    f"Listen / Preview — {sel0:0.2f}s → {sel1:0.2f}s (Δ {sel1-sel0:0.3f}s)"
                )

            if "wave" in axes:
                self._listen_span_selector = SpanSelector(
                    axes["wave"], on_span, "horizontal",
                    useblit=True, props=dict(alpha=0.2), interactive=True
                )
            else:
                self._listen_span_selector = None

            fig.canvas.draw_idle()

        # ───────────────────── Draw functions ─────────────────────
        def _ensure_wave_cursor():
            if "wave" in axes:
                ax = axes["wave"]
                if _wave_cursor["line"] is None:
                    _wave_cursor["line"], = ax.plot([sel0, sel0], ax.get_ylim(), color="#AAAAAA", lw=1)
                else:
                    _wave_cursor["line"].set_ydata([ax.get_ylim()[0], ax.get_ylim()[1]])

        def update_cursor_only(cursor_t):
            if "wave" in axes:
                if _wave_cursor["line"] is None:
                    _ensure_wave_cursor()
                if _wave_cursor["line"] is not None:
                    xline = sel0 + cursor_t
                    _wave_cursor["line"].set_xdata([xline, xline])
                    y0, y1 = axes["wave"].get_ylim()
                    _wave_cursor["line"].set_ydata([y0, y1])
                    fig.canvas.draw_idle()

        def draw_wave(cursor_t=0.0):
            if "wave" not in axes: return
            ax = axes["wave"]; ax.clear(); ax.set_facecolor("#000000"); _dark(ax)
            t = np.arange(N)/fs
            ax.plot(t, x, lw=0.8, color=getattr(self, "graph_color", "#03DFE2"))
            ax.axvspan(sel0, sel1, color=getattr(self, "graph_color", "#03DFE2"), alpha=0.15)
            ax.axvline(sel0 + cursor_t, color="#AAAAAA", lw=1)
            _title_right(ax, "Waveform")
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
            ax.set_xlim(0.0, T); ax.set_xmargin(0); ax.margins(x=0)
            ax.autoscale(enable=True, axis="y", tight=True)

        def draw_env_standalone():
            if "env" not in axes: return
            ax = axes["env"]; ax.clear(); ax.set_facecolor("#000000"); _dark(ax)
            t = np.arange(N)/fs
            w = max(1, int(0.010*fs))
            env = np.sqrt(np.convolve(x**2, np.ones(w)/w, mode="same"))
            ax.plot(t, env, lw=1.1)
            ax.axvspan(sel0, sel1, color=getattr(self, "graph_color", "#03DFE2"), alpha=0.15)
            _title_right(ax, "Envelope / RMS")
            ax.set_xlabel("Time (s)"); ax.set_ylabel("RMS")
            ax.set_xlim(0.0, T); ax.set_xmargin(0); ax.margins(x=0)
            ax.autoscale(enable=True, axis="y", tight=True)

        def draw_fft(center_t):
            if "fft" not in axes: return
            ax = axes["fft"]; ax.clear(); ax.set_facecolor("#000000"); _dark(ax)
            win_half = float(fft_win_spin.value()) / 2.0
            t_center = sel0 + center_t
            a = max(0.0, t_center - win_half)
            b = min(T, t_center + win_half)
            seg = x[int(a*fs):int(b*fs)]
            if len(seg) < 8:
                _title_right(ax, "FFT (segment too small)")
                fig.canvas.draw_idle(); return
            X = np.fft.rfft(seg * np.hanning(len(seg)))
            f = np.fft.rfftfreq(len(seg), 1/fs)
            if fft_y_cb.currentText().startswith("dB"):
                mag = 20*np.log10(np.maximum(1e-12, np.abs(X))); ax.set_ylabel("Mag (dB)")
            else:
                mag = np.abs(X); ax.set_ylabel("Mag")
            if fft_x_cb.currentText().startswith("Log"):
                ax.semilogx(f, mag, color=getattr(self,"graph_color","#03DFE2")); ax.set_xlabel("Freq (Hz, log)")
            else:
                ax.plot(f, mag, color=getattr(self,"graph_color","#03DFE2")); ax.set_xlabel("Freq (Hz)")
            rng = _get_fmin_fmax()
            if rng: ax.set_xlim(rng[0], rng[1])
            _title_right(ax, f"FFT @ {t_center:0.3f}s (win {2*win_half:0.2f}s)")
            fig.canvas.draw_idle()

        def draw_mini_spectrogram():
            if "minispec" not in axes: return
            ax = axes["minispec"]; ax.clear(); ax.set_facecolor("#000000"); _dark(ax)
            seg = x[int(sel0*fs):int(sel1*fs)]
            if len(seg) < 32:
                ax.set_title("Mini-Spectrogram (selection too short)"); return
            nper = min(1024, max(64, len(seg)//16))
            f, t, Sxx = signal.spectrogram(seg, fs=fs, nperseg=nper, noverlap=nper//2)
            Sxx_db = 10*np.log10(np.maximum(1e-15, Sxx))
            ax.imshow(Sxx_db, aspect="auto", origin="lower",
                    extent=[0, t[-1] if len(t)>0 else 0, f[0], f[-1] if len(f)>0 else fs/2])
            rng = _get_fmin_fmax()
            if rng: ax.set_ylim(rng[0], rng[1])
            ax.set_title("Mini-Spectrogram"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Freq (Hz)")

        def draw_waterfall():
            if "water" not in axes: return
            ax = axes["water"]; ax.clear(); ax.set_facecolor("#000000"); _dark(ax)
            if not chk_water_auto.isChecked():
                if _water_cache is None:
                    ax.text(0.5, 0.5, _water_cache_ready_text, transform=ax.transAxes,
                            ha="center", va="center", color="white")
                    ax.set_title("Waterfall (manual)"); ax.set_xlabel("Freq (Hz)"); ax.set_ylabel("Mag + offset")
                    fig.canvas.draw_idle(); return
            else:
                if _water_key() != _water_cache_key:
                    _build_waterfall_cache()

            if _water_cache is None:
                ax.set_title("Waterfall"); ax.set_xlabel("Freq (Hz)"); ax.set_ylabel("Mag + offset")
                fig.canvas.draw_idle(); return

            f, M, offsets = _water_cache
            rng = _get_fmin_fmax()
            mask = (slice(None) if rng is None else ((f >= rng[0]) & (f <= rng[1])))
            for i in range(M.shape[0]):
                ax.plot(f[mask], (M[i, mask] + offsets[i]), lw=0.6, alpha=0.9)
            if rng: ax.set_xlim(rng[0], rng[1])
            if fft_x_cb.currentText().startswith("Log"): ax.set_xscale("log")
            ax.set_title(f"Waterfall ({M.shape[0]} lines)"); ax.set_xlabel("Freq (Hz)"); ax.set_ylabel("Mag + offset")

        def draw_coherence():
            if "coh" not in axes: return
            ax = axes["coh"]; ax.clear(); ax.set_facecolor("#000000"); _dark(ax)
            if x2 is None:
                ax.set_title("Coherence (need 2 channels)"); return
            s1 = x[int(sel0*fs):int(sel1*fs)]; s2 = x2[int(sel0*fs):int(sel1*fs)]
            if len(s1) < 256:
                ax.set_title("Coherence (selection too short)"); return
            f, Cxy = signal.coherence(s1, s2, fs=fs, nperseg=min(2048, max(256, int(0.05*fs))))
            if fft_x_cb.currentText().startswith("Log"): ax.semilogx(f, Cxy)
            else: ax.plot(f, Cxy)
            rng = _get_fmin_fmax()
            if rng: ax.set_xlim(rng[0], rng[1])
            ax.set_ylim(0, 1); ax.set_title("Magnitude-Squared Coherence"); ax.set_xlabel("Freq (Hz)"); ax.set_ylabel("|Cxy|²")

        def draw_metrics_over_time():
            if "metrics" not in axes: return
            ax = axes["metrics"]; ax.clear(); ax.set_facecolor("#000000"); _dark(ax)
            seg = x[int(sel0*fs):int(sel1*fs)]
            if len(seg) < 128:
                ax.set_title("Spectral Metrics (selection too short)"); return
            win = min(1024, max(128, int(0.05*fs))); hop = max(1, win//4)
            cents, bw, flat, roll, tt = [], [], [], [], []
            for s in range(0, len(seg)-win, hop):
                fr = seg[s:s+win] * np.hanning(win)
                F = np.fft.rfft(fr); freq = np.fft.rfftfreq(win, 1/fs); mag = np.abs(F)
                P = mag / (mag.sum() + 1e-12)
                c = float(np.sum(freq * P)); cents.append(c)
                b = float(np.sqrt(np.sum(((freq - c)**2) * P))); bw.append(b)
                gm = np.exp(np.mean(np.log(mag + 1e-12))); am = np.mean(mag + 1e-12); flat.append(float(gm/am))
                tot = mag.sum(); cum = np.cumsum(mag); r = freq[np.searchsorted(cum, 0.95*tot)]; roll.append(float(r))
                tt.append((s + win/2)/fs + sel0)
            ax.plot(tt, cents, label="Centroid")
            ax.plot(tt, np.asarray(bw), label="Bandwidth")
            ax.plot(tt, np.asarray(roll), label="Roll-off 95%")
            ax2 = ax.twinx(); _dark(ax2)
            ax2.plot(tt, np.asarray(flat), label="Flatness", alpha=0.7)
            ax.set_title("Spectral Metrics"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Hz")
            ax2.set_ylabel("Flatness"); ax.legend(loc="upper left", frameon=False)

        def draw_histogram():
            if "hist" not in axes: return
            ax = axes["hist"]; ax.clear(); ax.set_facecolor("#000000"); _dark(ax)
            seg = x[int(sel0*fs):int(sel1*fs)]
            if len(seg) < 8:
                ax.set_title("SPL Histogram (selection too short)"); return
            db = 20*np.log10(np.maximum(1e-12, np.abs(seg)))
            ax.hist(db, bins=60, alpha=0.9)
            ax.set_title("SPL Histogram (|x| dB re 1)"); ax.set_xlabel("dB"); ax.set_ylabel("Count")

        def draw_cumulative_energy():
            if "cum" not in axes: return
            ax = axes["cum"]; ax.clear(); ax.set_facecolor("#000000"); _dark(ax)
            seg = x[int(sel0*fs):int(sel1*fs)]
            if len(seg) < 8:
                ax.set_title("Cumulative Energy (selection too short)"); return
            e = np.cumsum(seg**2); t = np.arange(len(seg))/fs + sel0
            ax.plot(t, e/e[-1] if e[-1] > 0 else e)
            ax.set_title("Cumulative Energy"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Norm. ∫x² dt")

        def draw_cepstrum():
            if "cep" not in axes: return
            ax = axes["cep"]; ax.clear(); ax.set_facecolor("#000000"); _dark(ax)
            seg = x[int(sel0*fs):int(sel1*fs)]
            if len(seg) < 64:
                ax.set_title("Cepstrum (selection too short)"); return
            S = np.fft.fft(seg * np.hanning(len(seg)))
            C = np.fft.ifft(np.log(np.maximum(1e-12, np.abs(S)))).real
            q = np.arange(len(C))/fs
            ax.plot(q, C); ax.set_xlim(0, min(0.1, q[-1]))
            ax.set_title("Power Cepstrum"); ax.set_xlabel("Quefrency (s)")

        def draw_phase():
            if "phase" not in axes: return
            ax = axes["phase"]; ax.clear(); ax.set_facecolor("#000000"); _dark(ax)
            win = min(N, max(256, int(0.05*fs)))
            i0 = int(max(0, (sel0 + 0.5*(sel1-sel0) - (win/fs)/2)*fs))
            seg = x[i0:i0+win]
            if len(seg) < 8:
                ax.set_title("Phase Spectrum (segment too small)"); return
            F = np.fft.rfft(seg * np.hanning(len(seg)))
            f = np.fft.rfftfreq(len(seg), 1/fs)
            ph = np.unwrap(np.angle(F))
            if fft_x_cb.currentText().startswith("Log"): ax.semilogx(f, ph)
            else: ax.plot(f, ph)
            rng = _get_fmin_fmax()
            if rng: ax.set_xlim(rng[0], rng[1])
            ax.set_title("Phase Spectrum"); ax.set_xlabel("Freq (Hz)"); ax.set_ylabel("Phase (rad)")

        def redraw_all(cursor_t=0.0):
            draw_wave(cursor_t)
            if "env" in axes:  # standalone envelope if waveform hidden
                draw_env_standalone()
            draw_fft(cursor_t)
            if "minispec" in axes:  draw_mini_spectrogram()
            if "water" in axes:     draw_waterfall()
            if "coh" in axes:       draw_coherence()
            if "metrics" in axes:   draw_metrics_over_time()
            if "hist" in axes:      draw_histogram()
            if "cum" in axes:       draw_cumulative_energy()
            if "cep" in axes:       draw_cepstrum()
            if "phase" in axes:     draw_phase()
            _ensure_wave_cursor()
            fig.tight_layout(); canvas.draw_idle()

        # ───────────────────── Handlers ─────────────────────
        def play():
            try: player.setPlaybackRate(float(speed_cb.currentText().replace("×","")))
            except: pass
            player.play()

        def pause():
            player.pause()

        def slider_moved(v):
            if clip_duration <= 0: return
            ms = int((v/10000.0) * clip_duration * 1000.0)
            player.blockSignals(True); player.setPosition(ms); player.blockSignals(False)
            t = ms/1000.0
            lbl_time.setText(f"{t:0.2f} / {clip_duration:0.2f} s")
            if chk_follow.isChecked():
                update_cursor_only(t)
                draw_fft(t)  # accurate FFT on drag

        throttle = QtCore.QElapsedTimer(); throttle.start()
        last_fft_ms = 0

        def player_moved(ms):
            last_fft_ms = -1  # initialize once
            
            if chk_follow.isChecked() and (last_fft_ms < 0 or (ms - last_fft_ms) >= 200):
                draw_fft(ms/1000.0)
                last_fft_ms = ms
            if clip_duration <= 0: return
            t = ms/1000.0
            val = int(round((t/clip_duration)*10000.0))
            time_slider.blockSignals(True); time_slider.setValue(max(0, min(10000, val))); time_slider.blockSignals(False)
            lbl_time.setText(f"{t:0.2f} / {clip_duration:0.2f} s")

            # Lightweight cursor move ~20 Hz
            if chk_follow.isChecked() and throttle.hasExpired(50):
                update_cursor_only(t)
                throttle.restart()

            # Throttled FFT redraw ~4 Hz
            if chk_follow.isChecked() and (ms - last_fft_ms) >= 250:
                draw_fft(t)
                # Waterfall: only if enabled and auto-updating; rebuild cache when key changes
                if chk_water.isChecked() and chk_water_auto.isChecked():
                    if _water_key() != _water_cache_key:
                        _build_waterfall_cache()
                        draw_waterfall()
                last_fft_ms = ms

            if chk_repeat.isChecked() and t >= clip_duration - 0.01:
                player.setPosition(0); player.play()

        def save_img():
            f, _ = QtWidgets.QFileDialog.getSaveFileName(
                dlg,
                "Save Image",
                self._dialog_default_dir("screenshots"),
                "PNG (*.png);;JPEG (*.jpg *.jpeg)",
            )
            if f: fig.savefig(f, dpi=150, facecolor=fig.get_facecolor())

        def export_clip():
            if sel1 <= sel0:
                QtWidgets.QMessageBox.information(dlg, "Export", "Selection is empty."); return
            clip_dir = self._project_subdir("modified") or "ml"
            os.makedirs(clip_dir, exist_ok=True)
            default = os.path.join(clip_dir, f"clip_{sel0:0.2f}_{sel1:0.2f}.wav")
            out, _ = QtWidgets.QFileDialog.getSaveFileName(
                dlg,
                "Export selection",
                default,
                "WAV (*.wav)",
            )
            if not out: return
            try:
                seg16 = _float_to_pcm16(x[int(sel0*fs):int(sel1*fs)])
                wavfile.write(out, fs, seg16)
                QtWidgets.QMessageBox.information(dlg, "Exported", f"Saved:\n{out}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(dlg, "Error", f"Failed to save WAV:\n{e}")

        def _refresh_waterfall_clicked():
            _build_waterfall_cache()
            draw_waterfall()

        # ───────────────────── Wire up ─────────────────────
        btn_play.clicked.connect(play)
        btn_pause.clicked.connect(pause)
        btn_save.clicked.connect(save_img)
        btn_export.clicked.connect(export_clip)
        btn_clear.clicked.connect(clear_selection)
        btn_water_refresh.clicked.connect(_refresh_waterfall_clicked)

        time_slider.valueChanged.connect(slider_moved)
        player.positionChanged.connect(player_moved)

        for cb in (chk_wave, chk_fft, chk_env, chk_mspec, chk_water, chk_coh,
                chk_metrics, chk_hist, chk_cum, chk_cep, chk_phase):
            cb.stateChanged.connect(lambda *_: (rebuild_layout(), redraw_all(0.0)))
        fft_x_cb.currentIndexChanged.connect(lambda *_: redraw_all(0.0))
        fft_y_cb.currentIndexChanged.connect(lambda *_: redraw_all(0.0))
        fft_win_spin.valueChanged.connect(lambda *_: redraw_all(0.0))
        fft_xmin.returnPressed.connect(lambda *_: redraw_all(0.0))
        fft_xmax.returnPressed.connect(lambda *_: redraw_all(0.0))

        # ───────────────────── Init ─────────────────────
        sel_label.setText(_sel_info())
        write_clip_for_selection()
        load_player_from_tmp()
        rebuild_layout()
        redraw_all(0.0)
        dlg.exec_()





    def _display_saved_spectrogram(self, record):
        """Render a saved spectrogram image when raw data isn't available."""
        if not record:
            return False

        path = record.get("image_path")
        if not path or not os.path.exists(path):
            return False

        if hasattr(self, "spec_spl_label") and self.spec_spl_label is not None:
            try:
                spl_val = (record or {}).get("spl_db")
                freq_val = (record or {}).get("spl_freq")
                if spl_val is not None:
                    freq_txt = f" @ {freq_val:0.1f} Hz" if freq_val is not None else ""
                    self.spec_spl_label.setText(f"SPL: {spl_val:0.2f} dB re 1µPa{freq_txt}")
                else:
                    self.spec_spl_label.setText("SPL: —")
            except Exception:
                pass

        import matplotlib.image as mpimg

        # Load the image into a local buffer so downstream rendering never
        # references an undefined variable (avoiding the prior NameError).
        try:
            img_data = mpimg.imread(path)
        except Exception:
            img_data = None

        if img_data is None:
            return False

        theme = self.spec_theme_combo.currentText()
        bg = "#000000" if theme == "Dark" else "#FFFFFF"
        text_color = "#FFF" if theme == "Dark" else "#000"
        try:
            cfg_for_spl = self._get_channel_config(self.spec_active_channel) or {}
        except Exception:
            cfg_for_spl = {}
        distance = None
        try:
            distance = self._coerce_float(cfg_for_spl.get("distance"))
        except Exception:
            distance = None
        is_hydrophone = cfg_for_spl.get("mode") == "hydrophone"
        distance = self._coerce_float(cfg_for_spl.get("distance"))
        display_data = None

        self.spec_canvas.ax.clear()
        for extra_ax in list(self.spec_canvas.fig.axes)[1:]:
            try:
                if extra_ax is not self.spec_canvas.ax:
                    self.spec_canvas.fig.delaxes(extra_ax)
            except Exception:
                pass

        self.spec_canvas.fig.patch.set_facecolor(bg)
        self.spec_canvas.ax.set_facecolor(bg)

        fs = (
            record.get("sample_rate")
            or getattr(self, "wav_sample_rate", None)
            or getattr(self, "sample_rate", None)
            or 0
        )
        t0 = record.get("start_time") or (record.get("start_sample", 0) / fs if fs else 0)
        t1 = record.get("end_time") or (record.get("end_sample", 0) / fs if fs else 0)
        fmax = (fs / 2.0) if fs else max(1.0, img_data.shape[0])
        extent = [t0, t1 if t1 else t0 + 1.0, 0, fmax]

        self.spec_canvas.ax.imshow(img_data, aspect="auto", origin="lower", extent=extent)

        self._last_spec_img_data = __import__("numpy").asarray(img_data)
        self._last_spec_extent = list(extent)
        self.spec_canvas.ax.set_xlabel("Time (s)")
        self.spec_canvas.ax.set_ylabel("Frequency (Hz)")
        self.spec_canvas.ax.xaxis.label.set_color(text_color)
        self.spec_canvas.ax.yaxis.label.set_color(text_color)
        self.spec_canvas.ax.tick_params(color=text_color, labelcolor=text_color)
        for spine in self.spec_canvas.ax.spines.values():
            spine.set_edgecolor(text_color)

        self.spec_canvas.draw()
        return True

    def _spec_segment_at(self, idx):
        try:
            seg = self.spec_segments[idx]
        except Exception:
            return None, None, None
        if isinstance(seg, (list, tuple)):
            start = seg[0] if len(seg) > 0 else None
            end = seg[1] if len(seg) > 1 else None
            ch = seg[2] if len(seg) > 2 else getattr(self, "spec_active_channel", 0)
            return start, end, ch
        return None, None, None

    def _render_spectrogram_segment(
        self,
        start_idx,
        end_idx,
        existing_record=None,
        channel_index=None,
        segment_data=None,
        sample_rate=None,
    ):
        import numpy as np

        ch_idx = channel_index
        if ch_idx is None and existing_record:
            ch_idx = existing_record.get("channel_index")

        data_src = segment_data if segment_data is not None else getattr(self, "full_data", None)
        fs = (
            sample_rate
            if sample_rate is not None
            else (getattr(self, "wav_sample_rate", None) or getattr(self, "sample_rate", None))
        )
        segment = None
        if data_src is not None and fs is not None:
            arr = np.asarray(data_src)
            if arr.ndim == 2:
                if ch_idx is None:
                    ch_idx = 0
                if 0 <= int(ch_idx) < arr.shape[1]:
                    segment = arr[int(start_idx):int(end_idx), int(ch_idx)]
            else:
                segment = arr[int(start_idx):int(end_idx)]

        if segment is None or fs is None:
            return self._display_saved_spectrogram(existing_record) and existing_record

        try:
            self.spec_active_channel = int(ch_idx or 0)
        except Exception:
            self.spec_active_channel = 0

        # clear canvas and remove old colorbars
        self.spec_canvas.ax.clear()
        for extra_ax in list(self.spec_canvas.fig.axes)[1:]:
            try:
                if extra_ax is not self.spec_canvas.ax:
                    self.spec_canvas.fig.delaxes(extra_ax)
            except Exception:
                pass

        theme = self.spec_theme_combo.currentText()
        bg = "#000000" if theme == "Dark" else "#FFFFFF"
        text_color = "#FFF" if theme == "Dark" else "#000"
        self.spec_canvas.fig.patch.set_facecolor(bg)
        self.spec_canvas.ax.set_facecolor(bg)

        try:
            cfg_for_spl = self._get_channel_config(self.spec_active_channel) or {}
        except Exception:
            cfg_for_spl = {}
        distance = None
        try:
            distance = self._coerce_float(cfg_for_spl.get("distance"))
        except Exception:
            distance = None
        is_hydrophone = cfg_for_spl.get("mode") == "hydrophone"

        # plot spectrogram live
        NFFT = int(self.spec_nfft_combo.currentData())
        noverlap = NFFT // 2
        cmap = self.spec_cmap_combo.currentText()
        try:
            self.log(f"[Spectrogram] Using Fs={fs} Hz (wav_sample_rate={getattr(self,'wav_sample_rate',None)}, sample_rate={getattr(self,'sample_rate',None)})")
        except Exception:
            pass
        
        Pxx, freqs, bins, im = self.spec_canvas.ax.specgram(
            segment,
            NFFT=NFFT,
            Fs=fs,
            noverlap=noverlap,
            cmap=cmap,
        )

        label = "Intensity (dB)"
        if is_hydrophone:
            label = "SPL (dB re 1µPa)"
            try:
                curve = self._get_hydrophone_curve_by_name(cfg_for_spl.get("hydrophone_curve")) or self._get_selected_hydrophone_curve()
                sens_list = curve.get("sensitivity") if curve else []

                # Recompute the plotted data in the SPL domain so the colorbar reflects
                # calibrated levels instead of raw intensity.
                arr = np.asarray(Pxx, dtype=float)
                if arr.size == len(freqs) * len(bins):
                    arr = arr.reshape((len(freqs), len(bins)))
                # Convert PSD density (V^2/Hz) to VRMS per bin before applying
                # hydrophone sensitivity so the scale reflects SPL, not raw
                # intensity. Use the frequency spacing to approximate the bin
                # width represented by each PSD slice.
                arr = np.maximum(arr, 1e-30)
                if len(freqs) > 1:
                    bin_hz = float(freqs[1] - freqs[0])
                elif fs and NFFT:
                    bin_hz = float(fs) / float(NFFT)
                else:
                    bin_hz = 1.0
                bin_hz = max(bin_hz, 1e-12)

                arr_vrms = np.sqrt(arr * bin_hz)
                arr_db = 20.0 * np.log10(np.maximum(arr_vrms, 1e-30))

                if sens_list:
                    min_f = int(curve.get("min_freq", 0) or 0)
                    sens_vals = []
                    for f in freqs:
                        idx = int(round(f)) - min_f
                        idx = max(0, min(idx, len(sens_list) - 1))
                        try:
                            sens_vals.append(float(sens_list[idx]))
                        except (TypeError, ValueError):
                            sens_vals.append(0.0)
                    sens_arr = np.asarray(sens_vals, dtype=float)
                    arr_db = arr_db - sens_arr[:, None]

                if distance and distance > 0:
                    arr_db = arr_db + 20.0 * np.log10(distance)

                # --- NEW: match specgram image orientation ---
                try:
                    origin = getattr(im, "origin", None)
                except Exception:
                    origin = None
                if origin == "upper":
                    arr_db = arr_db[::-1, :]

                im.set_data(arr_db)
                if np.isfinite(arr_db).any():
                    im.set_clim(float(np.nanmin(arr_db)), float(np.nanmax(arr_db)))
            except Exception:
                # Keep SPL labeling even if sensitivity adjustments fail; the
                # displayed values still reflect hydrophone-derived scaling.
                pass

        # cache rendered spectrogram image in display coordinates (for auto annotate)
        try:
            arr = np.asarray(getattr(im, "get_array", lambda: None)())
            if arr is not None and arr.size:
                self._last_spec_img_data = np.asarray(arr)
                self._last_spec_extent = list(getattr(im, "get_extent", lambda: [0, 1, 0, fs/2])())
        except Exception:
            pass

        # set freq bounds
        fmin = float(self.spec_min_freq_entry.text() or 0)
        fmax = float(self.spec_max_freq_entry.text() or fs / 2)
        self.spec_canvas.ax.set_ylim(fmin, fmax)

        # add or remove colorbar
        if self.show_colorbar_checkbox.isChecked():
            cax = self.spec_canvas.fig.add_axes([0.92, 0.1, 0.02, 0.8])
            cbar = self.spec_canvas.fig.colorbar(im, cax=cax)
            cbar.ax.yaxis.set_tick_params(color=text_color, labelcolor=text_color)
            cbar.outline.set_edgecolor(text_color)
            cbar.set_label(label, color=text_color)
            cbar.update_normal(im)

        # restore labels
        self.spec_canvas.ax.set_xlabel("Time (s)")
        self.spec_canvas.ax.set_ylabel("Frequency (Hz)")
        self.spec_canvas.ax.xaxis.label.set_color(text_color)
        self.spec_canvas.ax.yaxis.label.set_color(text_color)
        self.spec_canvas.ax.tick_params(color=text_color, labelcolor=text_color)
        for spine in self.spec_canvas.ax.spines.values():
            spine.set_edgecolor(text_color)

        # SPL estimate (hydrophone-only)
        spl_text = "SPL: —"
        spl_val = None
        spl_freq = None
        hydro_curve_name = None
        try:
            cfg = cfg_for_spl
            if cfg.get("mode") == "hydrophone":
                hydro_curve_name = cfg.get("hydrophone_curve") or None
                if not hydro_curve_name:
                    combo_curve = self._get_selected_hydrophone_curve()
                    if combo_curve:
                        hydro_curve_name = combo_curve.get("curve_name")
                distance = self._coerce_float(cfg.get("distance"))

                seg_np = np.asarray(segment, dtype=float)
                vrms = float(np.sqrt(np.mean(np.square(seg_np)))) if seg_np.size else 0.0
                if seg_np.size > 0 and fs:
                    window = np.hanning(len(seg_np)) if len(seg_np) > 1 else np.ones_like(seg_np)
                    spectrum = np.abs(np.fft.rfft(seg_np * window))
                    freq_bins = np.fft.rfftfreq(len(seg_np), 1/float(fs))
                    if spectrum.size:
                        peak_idx = int(np.argmax(spectrum[1:]) + 1 if spectrum.size > 1 else 0)
                        if peak_idx >= len(freq_bins):
                            peak_idx = len(freq_bins) - 1
                        spl_freq = float(freq_bins[peak_idx]) if peak_idx >= 0 else 0.0

                curve = self._get_hydrophone_curve_by_name(hydro_curve_name) or self._get_selected_hydrophone_curve()
                sens_list = curve.get("sensitivity") if curve else []
                if curve and sens_list and spl_freq is not None:
                    min_f = int(curve.get("min_freq", 0) or 0)
                    idx = int(round(spl_freq)) - min_f
                    idx = max(0, min(idx, len(sens_list) - 1))
                    try:
                        sensitivity_db = float(sens_list[idx])
                        spl_val = 20.0 * np.log10(max(vrms, 1e-12)) - sensitivity_db
                        if distance:
                            try:
                                d = float(distance)
                                if d > 0:
                                    spl_val += 20.0 * np.log10(d)
                            except Exception:
                                pass
                    except (TypeError, ValueError):
                        spl_val = None

        except Exception:
            spl_val = spl_val if 'spl_val' in locals() else None

        if spl_val is not None:
            freq_txt = f" @ {spl_freq:0.1f} Hz" if spl_freq is not None else ""
            spl_text = f"SPL: {spl_val:0.2f} dB re 1µPa{freq_txt}"
        if hasattr(self, "spec_spl_label") and self.spec_spl_label is not None:
            try:
                self.spec_spl_label.setText(spl_text)
            except Exception:
                pass

        self.spec_canvas.draw()

        ch_label = None
        if hasattr(self, "channel_names") and 0 <= self.spec_active_channel < len(self.channel_names):
            ch_label = self.channel_names[self.spec_active_channel]
        record = self._persist_spectrogram_record(
            start_idx,
            end_idx,
            existing_record,
            channel_index=self.spec_active_channel,
            channel_name=ch_label,
            hydrophone_curve_name=hydro_curve_name,
            spl_value=spl_val,
            spl_freq=spl_freq,
            distance=distance,
        )
        self.current_spectrogram_record = record
        self._set_spec_annotation_controls_enabled(record is not None)
        self._refresh_spectrogram_gallery(select_id=record.get("id") if record else None)
        self._render_spec_annotations(record)
        return record


    def generate_spectrograms(self):
        """
        Scan for segments above threshold, then:
        - Store segments as (start_idx, end_idx, channel_index)
        - Render the first segment live using _render_spectrogram_segment()
        """
        import numpy as np
        from PyQt5 import QtWidgets

        # ---- validate parameters ----
        try:
            threshold = float(self.spec_threshold_entry.text())
            pre_buf_s = float(self.pre_buffer_entry.text())
            post_buf_s = float(self.post_buffer_entry.text())
            min_len_s = float(self.min_spec_length_entry.text())

            fs = float(getattr(self, "wav_sample_rate", None) or getattr(self, "sample_rate", 0) or 0)
            if fs <= 0:
                raise ValueError("sample_rate is not set or invalid.")
            if threshold < 0:
                raise ValueError("threshold must be >= 0.")
            if pre_buf_s < 0 or post_buf_s < 0 or min_len_s <= 0:
                raise ValueError("buffers must be >= 0 and min length must be > 0.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Invalid spectrogram parameter:\n{e}")
            return

        # ---- pick channel & pull data ----
        ch_idx = 0
        if hasattr(self, "_current_spec_channel_index"):
            try:
                ch_idx = int(self._current_spec_channel_index())
            except Exception:
                ch_idx = 0

        data = self.get_channel_data(ch_idx) if hasattr(self, "get_channel_data") else None
        if data is None:
            QtWidgets.QMessageBox.warning(self, "Spectrogram", "No data available for the selected channel.")
            return

        data = np.asarray(data)
        if data.ndim != 1:
            data = np.ravel(data)

        self.spec_active_channel = int(ch_idx)

        # ---- reset UI state / caches ----
        if not hasattr(self, "spec_segments") or self.spec_segments is None:
            self.spec_segments = []
        else:
            self.spec_segments.clear()

        self.current_spec_index = -1
        self.current_spectrogram_record = None

        if hasattr(self, "_set_spec_annotation_controls_enabled"):
            try:
                self._set_spec_annotation_controls_enabled(False)
            except Exception:
                pass

        if hasattr(self, "spec_spl_label") and self.spec_spl_label is not None:
            try:
                self.spec_spl_label.setText("SPL: —")
            except Exception:
                pass

        # ---- find segments ----
        total = int(len(data))
        i = 0

        pre_samp = int(round(pre_buf_s * fs))
        post_samp = int(round(post_buf_s * fs))
        min_len_samp = int(round(min_len_s * fs))

        while i < total:
            idxs = np.where(np.abs(data[i:]) >= threshold)[0]
            if idxs.size == 0:
                break

            center = i + int(idxs[0])
            start = max(0, center - pre_samp)
            end = min(total, center + min_len_samp + post_samp)

            if end > start:
                self.spec_segments.append((start, end, self.spec_active_channel))

            i = end  # jump past this segment

        # ---- nothing found ----
        if not self.spec_segments:
            QtWidgets.QMessageBox.information(
                self,
                "Spectrogram",
                "No segments found above threshold.\n\n"
                "Try lowering the threshold or increasing the buffers/min length."
            )
            if hasattr(self, "prev_spec_btn"): self.prev_spec_btn.setEnabled(False)
            if hasattr(self, "next_spec_btn"): self.next_spec_btn.setEnabled(False)
            if hasattr(self, "save_spec_btn"): self.save_spec_btn.setEnabled(False)
            if hasattr(self, "listen_btn"): self.listen_btn.setEnabled(False)
            if hasattr(self, "save_ml_clip_btn"): self.save_ml_clip_btn.setEnabled(False)
            return

        # ---- render first segment ----
        self.current_spec_index = 0
        first_start, first_end, first_ch = self._spec_segment_at(0)
        self._render_spectrogram_segment(first_start, first_end, channel_index=first_ch)

        # ---- enable navigation/actions ----
        if hasattr(self, "prev_spec_btn"): self.prev_spec_btn.setEnabled(True)
        if hasattr(self, "next_spec_btn"): self.next_spec_btn.setEnabled(True)
        if hasattr(self, "save_spec_btn"): self.save_spec_btn.setEnabled(True)
        if hasattr(self, "listen_btn"): self.listen_btn.setEnabled(True)
        if hasattr(self, "save_ml_clip_btn"): self.save_ml_clip_btn.setEnabled(True)
        if hasattr(self, "export_clip_btn"): self.export_clip_btn.setEnabled(True)
    
    def save_current_spectrogram(self):
        if self.current_spec_index < 0 or not self.spec_segments:
            QtWidgets.QMessageBox.warning(self, "Error", "No spectrogram to save.")
            return

        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Spectrogram",
            self._dialog_default_dir("spectrograms"),
            "JPEG Files (*.jpg *.jpeg);;PNG Files (*.png)",
        )
        if not fname:
            return
        # ensure extension
        if not any(fname.lower().endswith(ext) for ext in (".jpg", ".jpeg", ".png")):
            fname += ".jpg"

        # save the live canvas figure
        self.spec_canvas.fig.savefig(fname, dpi=150,
                                    facecolor=self.spec_canvas.fig.get_facecolor())
        QtWidgets.QMessageBox.information(self, "Saved", f"Spectrogram saved to:\n{fname}")


    def listen_to_spectrogram(self):
        if self.current_spec_index < 0 or not self.spec_segments:
            QtWidgets.QMessageBox.warning(self, "Error", "No spectrogram segment selected.")
            return

        start_idx, end_idx, ch_idx = self._spec_segment_at(self.current_spec_index)
        if ch_idx is None:
            ch_idx = getattr(self, "spec_active_channel", 0)
        clip = self.get_channel_data(ch_idx)[start_idx:end_idx] if hasattr(self, "get_channel_data") else self.full_data[start_idx:end_idx]

        # Ensure 16-bit PCM for QSound reliability
        # Normalize to [-1, 1] if needed, then scale to int16
        if clip.dtype.kind in ("f",):  # float -> normalize
            maxabs = np.max(np.abs(clip)) or 1.0
            clip_norm = (clip / maxabs)
        else:
            # assume integer; convert to float then normalize by its max
            info = np.iinfo(clip.dtype)
            clip_norm = clip.astype(np.float32) / max(abs(info.min), info.max)

        clip_int16 = np.clip(clip_norm, -1.0, 1.0)
        clip_int16 = (clip_int16 * 32767.0).astype(np.int16)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        wavfile.write(tmp.name, self.sample_rate, clip_int16)
        QSound.play(tmp.name)

    def export_spectrogram_clip(self):
        # 1) sanity checks
        if self.current_spec_index < 0 or not self.spec_segments:
            QtWidgets.QMessageBox.warning(self, "Error", "No spectrogram segment to export.")
            return
        if not hasattr(self, "last_spec_region"):
            QtWidgets.QMessageBox.warning(self, "Error", "Please drag-select a region first.")
            return

        # 2) map the span (xmin,xmax) back to absolute sample indices
        xmin, xmax = self.last_spec_region
        start_idx, _, ch_idx = self._spec_segment_at(self.current_spec_index)
        clip_start = start_idx + int(xmin * self.sample_rate)
        clip_end   = start_idx + int(xmax * self.sample_rate)
        channel_data = self.get_channel_data(ch_idx) if hasattr(self, "get_channel_data") else None
        clip       = channel_data[clip_start:clip_end] if channel_data is not None else self.full_data[clip_start:clip_end]

        # 3) ask user for a base name
        name, ok = QtWidgets.QInputDialog.getText(
            self,
            "Name this clip",
            "Enter a label for this ML clip:"
        )
        if not ok or not name.strip():
            return
        base = name.strip()

        # 4) pick the project spectrogram folder (or fallback to app-local ml/)
        ml_dir = self._project_subdir("spectrograms")
        if not ml_dir:
            app_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
            ml_dir = os.path.join(app_dir, "ml")
        os.makedirs(ml_dir, exist_ok=True)

        # 5) pick a filename, auto-incrementing if needed
        candidate = f"{base}.wav"
        i = 1
        while os.path.exists(os.path.join(ml_dir, candidate)):
            candidate = f"{base}{i}.wav"
            i += 1

        # 6) write it out (preserving dtype) and notify
        from scipy.io import wavfile
        out_path = os.path.join(ml_dir, candidate)
        wavfile.write(out_path, self.sample_rate, clip.astype(self.original_dtype))

        QtWidgets.QMessageBox.information(self, "Saved", f"ML clip saved to:\n{out_path}")



    def show_spec(self):
        """
        Redraw the current segment’s spectrogram live on the canvas.
        """
        if not (0 <= self.current_spec_index < len(self.spec_segments)):
            return

        start_idx, end_idx, ch_idx = self._spec_segment_at(self.current_spec_index)
        self._render_spectrogram_segment(start_idx, end_idx, channel_index=ch_idx)

        # nav buttons
        self.prev_spec_btn.setEnabled(self.current_spec_index > 0)
        self.next_spec_btn.setEnabled(self.current_spec_index < len(self.spec_segments) - 1)

    def show_next_spec(self):
        if self.current_spec_index < len(self.spec_segments) - 1:
            self.current_spec_index += 1
            self.show_spec()

    def show_prev_spec(self):
        if self.current_spec_index > 0:
            self.current_spec_index -= 1
            self.show_spec()

    def save_current_spec(self):
        if self.current_spec_index < 0 or not self.spec_files:
            QtWidgets.QMessageBox.warning(self, "Error", "No spectrogram to save.")
            return

        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Spectrogram as JPG",
            self._dialog_default_dir("spectrograms"),
            "JPEG Files (*.jpg *.jpeg)"
        )
        if not fname:
            return
        if not fname.lower().endswith((".jpg", ".jpeg")):
            fname += ".jpg"

        self.spec_canvas.fig.savefig(
            fname,
            dpi=150,
            facecolor=self.spec_canvas.fig.get_facecolor()
        )
        QtWidgets.QMessageBox.information(self, "Saved", f"Spectrogram saved to:\n{fname}")


    # ---------------------
    # Logs Tab Methods
    # ---------------------
    
    def reset_log_page(self, *args):
        """Whenever filters or view change, go back to page 0."""
        self.log_current_page = 0

    def prev_log_page(self):
        """Go back one page, then refresh."""
        if self.log_current_page > 0:
            self.log_current_page -= 1
            self.request_logs_refresh(immediate=True, refresh_filters=False)

    def next_log_page(self):
        """Advance one page, then refresh."""
        # you might want to clamp at max pages, but populate_log_table() does that
        self.log_current_page += 1
        self.request_logs_refresh(immediate=True, refresh_filters=False)

    def on_entries_per_page_changed(self, text):
        try:
            self.log_entries_per_page = int(text)
        except ValueError:
            self.log_entries_per_page = 50
        self.log_current_page = 0
        self.request_logs_refresh(immediate=True, refresh_filters=False)

    def setup_chart_tab(self):
        root_layout = QtWidgets.QVBoxLayout(self.chart_tab)

        # Restore a dedicated map tab inside Charting
        self.chart_inner_tabs = QtWidgets.QTabWidget()
        root_layout.addWidget(self.chart_inner_tabs)

        self.chart_map_tab = QtWidgets.QWidget()
        self.chart_inner_tabs.addTab(self.chart_map_tab, "Map")

        layout = QtWidgets.QHBoxLayout(self.chart_map_tab)

        sidebar = QtWidgets.QVBoxLayout()
        sidebar.addWidget(QtWidgets.QLabel("GPS Tracks"))
        self.gps_track_list = QtWidgets.QListWidget()
        self.gps_track_list.setMinimumWidth(220)
        self.gps_track_list.itemSelectionChanged.connect(self._plot_selected_gps_tracks)
        sidebar.addWidget(self.gps_track_list, 1)

        row = QtWidgets.QHBoxLayout()
        self.gps_import_btn = QtWidgets.QPushButton("Import Track")
        self.gps_import_btn.clicked.connect(self.import_gps_track)
        row.addWidget(self.gps_import_btn)
        self.gps_delete_btn = QtWidgets.QPushButton("Delete")
        self.gps_delete_btn.clicked.connect(self.delete_selected_gps_tracks)
        row.addWidget(self.gps_delete_btn)
        sidebar.addLayout(row)

        self.gps_fit_btn = QtWidgets.QPushButton("Fit View")
        self.gps_fit_btn.clicked.connect(self._fit_gps_view)
        sidebar.addWidget(self.gps_fit_btn)

        self.chart_refresh_btn = QtWidgets.QPushButton("Refresh Chart")
        self.chart_refresh_btn.clicked.connect(self.refresh_chart_tracks)
        sidebar.addWidget(self.chart_refresh_btn)

        self.chart_interactive_mode_cb = QtWidgets.QCheckBox("Interactive Edit Mode")
        self.chart_interactive_mode_cb.setChecked(True)
        self.chart_interactive_mode_cb.setEnabled(False)
        self.chart_interactive_mode_cb.setToolTip("Folium/WebEngine map path is disabled for stability; using PyQtGraph map")
        self.chart_interactive_mode_cb.toggled.connect(self._plot_selected_gps_tracks)
        sidebar.addWidget(self.chart_interactive_mode_cb)

        sidebar.addWidget(QtWidgets.QLabel("Waypoints"))
        self.waypoint_list = QtWidgets.QListWidget()
        self.waypoint_list.setMinimumHeight(120)
        sidebar.addWidget(self.waypoint_list)

        wp_row = QtWidgets.QHBoxLayout()
        self.wp_add_btn = QtWidgets.QPushButton("Add")
        self.wp_add_btn.clicked.connect(self.add_chart_waypoint)
        wp_row.addWidget(self.wp_add_btn)
        self.wp_delete_btn = QtWidgets.QPushButton("Delete")
        self.wp_delete_btn.clicked.connect(self.delete_selected_waypoints)
        wp_row.addWidget(self.wp_delete_btn)
        sidebar.addLayout(wp_row)

        layout.addLayout(sidebar, 1)

        right = QtWidgets.QVBoxLayout()
        self.gps_map_stack = QtWidgets.QStackedWidget()

        self.gps_map_view = None
        if QtWebEngineWidgets is not None and folium is not None:
            try:
                self.gps_map_view = QtWebEngineWidgets.QWebEngineView()
                self.gps_map_stack.addWidget(self.gps_map_view)
            except Exception:
                self.gps_map_view = None

        self.gps_plot = pg.PlotWidget()
        self.gps_plot.showGrid(x=True, y=True, alpha=0.25)
        self.gps_plot.setLabel('bottom', 'Longitude')
        self.gps_plot.setLabel('left', 'Latitude')
        self.gps_plot.addLegend()
        self.gps_plot.getViewBox().setAspectLocked(False)
        self.gps_map_stack.addWidget(self.gps_plot)

        if self.gps_map_view is not None:
            self.gps_map_stack.setCurrentWidget(self.gps_map_view)
        else:
            self.gps_map_stack.setCurrentWidget(self.gps_plot)

        right.addWidget(self.gps_map_stack, 1)

        self.gps_info_label = QtWidgets.QLabel("No tracks loaded")
        right.addWidget(self.gps_info_label)
        self.gps_cursor_label = QtWidgets.QLabel("Cursor: --")
        right.addWidget(self.gps_cursor_label)
        layout.addLayout(right, 4)

        self._gps_curves = []
        self._gps_ctd_markers = []
        self._gps_folium_html_path = None
        self._chart_map_click_pos = None
        self.gps_plot.scene().sigMouseMoved.connect(self._on_chart_map_mouse_moved)
        self.gps_plot.scene().sigMouseClicked.connect(self._on_chart_map_mouse_clicked)
        self.refresh_chart_theme()
        self.refresh_chart_tracks()

    def refresh_chart_theme(self):
        if not hasattr(self, 'gps_plot') or self.gps_plot is None:
            return
        pal = self.palette()
        bg = pal.color(QtGui.QPalette.Window).name()
        axis_color = '#000000' if str(get_setting('ui_theme', 'dark')).lower() == 'light' else '#FFFFFF'
        self.gps_plot.setBackground(bg)
        self.gps_plot.setLabel('bottom', 'Longitude', color=axis_color)
        self.gps_plot.setLabel('left', 'Latitude', color=axis_color)
        self.gps_plot.getAxis('bottom').setTextPen(pg.mkPen(axis_color))
        self.gps_plot.getAxis('left').setTextPen(pg.mkPen(axis_color))

    def _iter_csv_gps_points(self, file_path):
        with open(file_path, 'r', encoding='utf-8-sig', newline='') as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames:
                return []
            field_map = {f.strip().lower(): f for f in reader.fieldnames if f}
            lat_key = next((field_map[k] for k in ("lat", "latitude", "y") if k in field_map), None)
            lon_key = next((field_map[k] for k in ("lon", "lng", "longitude", "x") if k in field_map), None)
            ele_key = next((field_map[k] for k in ("ele", "elevation", "alt", "altitude") if k in field_map), None)
            t_key = next((field_map[k] for k in ("time", "timestamp", "utc_time", "datetime") if k in field_map), None)
            if not lat_key or not lon_key:
                raise ValueError("CSV must include latitude/longitude columns.")
            out = []
            for i, row in enumerate(reader):
                try:
                    lat = float(row.get(lat_key, ""))
                    lon = float(row.get(lon_key, ""))
                except Exception:
                    continue
                ele = row.get(ele_key) if ele_key else None
                ts = row.get(t_key) if t_key else None
                try:
                    ele = float(ele) if ele not in (None, "") else None
                except Exception:
                    ele = None
                out.append((i, ts, lat, lon, ele))
            return out

    def _iter_gpx_points(self, file_path):
        ns = {"gpx": "http://www.topografix.com/GPX/1/1"}
        root = ET.parse(file_path).getroot()
        points = []

        for idx, trkpt in enumerate(root.findall('.//gpx:trkpt', ns) + root.findall('.//trkpt')):
            try:
                lat = float(trkpt.attrib.get('lat'))
                lon = float(trkpt.attrib.get('lon'))
            except Exception:
                continue
            ele_node = trkpt.find('gpx:ele', ns) or trkpt.find('ele')
            time_node = trkpt.find('gpx:time', ns) or trkpt.find('time')
            ele = None
            if ele_node is not None and (ele_node.text or '').strip() != '':
                try:
                    ele = float(ele_node.text)
                except Exception:
                    ele = None
            ts = (time_node.text or '').strip() if time_node is not None else None
            points.append((idx, ts, lat, lon, ele))
        return points

    def import_gps_track(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import GPS Track",
            self._dialog_default_dir("originals"),
            "GPS Track (*.gpx *.csv);;All Files (*)",
        )
        if not path:
            return

        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == '.gpx':
                points = self._iter_gpx_points(path)
            elif ext == '.csv':
                points = self._iter_csv_gps_points(path)
            else:
                raise ValueError("Unsupported track format. Use GPX or CSV.")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Import GPS Track", f"Could not parse track\n{e}")
            return

        if not points:
            QtWidgets.QMessageBox.information(self, "Import GPS Track", "No valid GPS points found.")
            return

        default_name = os.path.splitext(os.path.basename(path))[0]
        name, ok = QtWidgets.QInputDialog.getText(self, "Track Name", "Track name:", text=default_name)
        if not ok:
            return
        name = (name or '').strip() or default_name

        palette = self._ordered_palette() if hasattr(self, '_ordered_palette') else ['#03DFE2']
        color = palette[0] if palette else '#03DFE2'
        pid = getattr(self, 'current_project_id', None)

        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO gps_tracks (project_id, name, source_file, color) VALUES (?, ?, ?, ?)",
            (pid, name, path, color),
        )
        track_id = cur.lastrowid
        cur.executemany(
            "INSERT INTO gps_track_points (track_id, point_index, timestamp_utc, latitude, longitude, elevation_m) VALUES (?, ?, ?, ?, ?, ?)",
            [(track_id, i, ts, lat, lon, ele) for i, ts, lat, lon, ele in points],
        )
        conn.commit()
        conn.close()

        self.refresh_chart_tracks(select_id=track_id)

    def refresh_chart_tracks(self, select_id=None):
        if not hasattr(self, 'gps_track_list'):
            return
        pid = getattr(self, 'current_project_id', None)
        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, name, source_file, created_at
            FROM gps_tracks
            WHERE (project_id IS NULL AND ? IS NULL) OR project_id = ?
            ORDER BY created_at DESC, id DESC
            """,
            (pid, pid),
        )
        rows = cur.fetchall()
        conn.close()

        self.gps_track_list.blockSignals(True)
        self.gps_track_list.clear()
        target_item = None
        for rid, name, src, created in rows:
            label = f"{name}"
            if created:
                label += f" ({str(created).split(' ')[0]})"
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.UserRole, int(rid))
            item.setToolTip(src or '')
            self.gps_track_list.addItem(item)
            if select_id is not None and int(rid) == int(select_id):
                target_item = item
        self.gps_track_list.blockSignals(False)
        if target_item is not None:
            target_item.setSelected(True)
        self.refresh_chart_waypoints()
        self._plot_selected_gps_tracks()

    def _fetch_track_points(self, track_id):
        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        cur.execute(
            "SELECT latitude, longitude FROM gps_track_points WHERE track_id=? ORDER BY point_index ASC, id ASC",
            (int(track_id),),
        )
        pts = cur.fetchall()
        cur.execute("SELECT name, color FROM gps_tracks WHERE id=?", (int(track_id),))
        meta = cur.fetchone()
        conn.close()
        return (meta[0], meta[1] or '#03DFE2', pts) if meta else (f'Track {track_id}', '#03DFE2', pts)

    def _ensure_waypoints_table(self):
        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS map_waypoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                project_id INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_map_waypoints_project ON map_waypoints(project_id)")
        cur.execute("PRAGMA table_info(map_waypoints)")
        cols = {r[1] for r in cur.fetchall()}
        if 'symbol' not in cols:
            cur.execute("ALTER TABLE map_waypoints ADD COLUMN symbol TEXT DEFAULT 'star'")
        conn.commit()
        conn.close()

    def refresh_chart_waypoints(self):
        if not hasattr(self, 'waypoint_list'):
            return
        self._ensure_waypoints_table()
        pid = getattr(self, 'current_project_id', None)
        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, name, latitude, longitude, project_id, COALESCE(symbol,'star')
            FROM map_waypoints
            WHERE project_id IS NULL OR project_id = ?
            ORDER BY project_id IS NULL DESC, name COLLATE NOCASE ASC, id DESC
            """,
            (pid,),
        )
        rows = cur.fetchall()
        conn.close()

        self.waypoint_list.clear()
        for wid, name, lat, lon, proj_id, symbol in rows:
            scope = 'Global' if proj_id is None else 'Project'
            item = QtWidgets.QListWidgetItem(f"{name} [{scope}] <{symbol}> ({lat:.5f}, {lon:.5f})")
            item.setData(QtCore.Qt.UserRole, int(wid))
            self.waypoint_list.addItem(item)

    def _fetch_waypoints_for_chart(self):
        self._ensure_waypoints_table()
        pid = getattr(self, 'current_project_id', None)
        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, name, latitude, longitude, project_id, COALESCE(symbol,'star')
            FROM map_waypoints
            WHERE project_id IS NULL OR project_id = ?
            ORDER BY created_at DESC, id DESC
            """,
            (pid,),
        )
        rows = cur.fetchall()
        conn.close()
        return rows

    def add_chart_waypoint(self, default_lat=None, default_lon=None):
        self._ensure_waypoints_table()
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle('Add Waypoint')
        form = QtWidgets.QFormLayout(dlg)
        name_e = QtWidgets.QLineEdit()
        lat_e = QtWidgets.QLineEdit('' if default_lat is None else f"{float(default_lat):.7f}")
        lon_e = QtWidgets.QLineEdit('' if default_lon is None else f"{float(default_lon):.7f}")
        scope_cb = QtWidgets.QComboBox()
        scope_cb.addItems(['Project waypoint', 'Global waypoint'])
        symbol_cb = QtWidgets.QComboBox()
        symbol_cb.addItems(['star', 'circle', 'square', 'triangle', 'diamond', 'cross'])
        form.addRow('Name:', name_e)
        form.addRow('Latitude:', lat_e)
        form.addRow('Longitude:', lon_e)
        form.addRow('Scope:', scope_cb)
        form.addRow('Symbol:', symbol_cb)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        form.addRow(btns)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        try:
            name = (name_e.text() or '').strip() or 'Waypoint'
            lat = float(lat_e.text().strip())
            lon = float(lon_e.text().strip())
        except Exception:
            QtWidgets.QMessageBox.warning(self, 'Waypoint', 'Invalid waypoint values.')
            return
        proj_id = None if scope_cb.currentText().startswith('Global') else getattr(self, 'current_project_id', None)
        symbol = symbol_cb.currentText().strip() or 'star'
        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        cur.execute(
            'INSERT INTO map_waypoints (name, latitude, longitude, project_id, symbol) VALUES (?, ?, ?, ?, ?)',
            (name, lat, lon, proj_id, symbol),
        )
        conn.commit(); conn.close()
        self.refresh_chart_waypoints()
        self._plot_selected_gps_tracks()

    def _open_ctd_import_at(self, lat, lon):
        self._ctd_seed_location = (float(lat), float(lon))
        try:
            self.ctd_import_popup()
        finally:
            self._ctd_seed_location = None
        self._plot_selected_gps_tracks()

    def _on_chart_map_mouse_moved(self, pos):
        if not hasattr(self, 'gps_plot') or self.gps_plot is None:
            return
        vb = self.gps_plot.getViewBox()
        pt = vb.mapSceneToView(pos)
        if hasattr(self, 'gps_cursor_label'):
            self.gps_cursor_label.setText(f"Cursor: Lat {float(pt.y()):.6f}, Lon {float(pt.x()):.6f}")

    def _on_chart_map_mouse_clicked(self, event):
        if not hasattr(self, 'gps_plot') or self.gps_plot is None:
            return
        if event.button() != QtCore.Qt.LeftButton:
            return
        vb = self.gps_plot.getViewBox()
        pt = vb.mapSceneToView(event.scenePos())
        lat = float(pt.y())
        lon = float(pt.x())
        menu = QtWidgets.QMenu(self)
        a_wp = menu.addAction(f"Create waypoint here ({lat:.5f}, {lon:.5f})")
        a_ctd = menu.addAction(f"Import CTD data at this location ({lat:.5f}, {lon:.5f})")
        chosen = menu.exec_(QtGui.QCursor.pos())
        if chosen is a_wp:
            self.add_chart_waypoint(default_lat=lat, default_lon=lon)
        elif chosen is a_ctd:
            self._open_ctd_import_at(lat, lon)

    def delete_selected_waypoints(self):
        if not hasattr(self, 'waypoint_list'):
            return
        ids = [it.data(QtCore.Qt.UserRole) for it in self.waypoint_list.selectedItems()]
        if not ids:
            return
        if QtWidgets.QMessageBox.question(self, 'Delete Waypoints', f'Delete {len(ids)} selected waypoint(s)?') != QtWidgets.QMessageBox.Yes:
            return
        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        for wid in ids:
            cur.execute('DELETE FROM map_waypoints WHERE id=?', (int(wid),))
        conn.commit(); conn.close()
        self.refresh_chart_waypoints()
        self._plot_selected_gps_tracks()

    def _fetch_ctd_points_for_chart(self):
        pid = getattr(self, 'current_project_id', None)
        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, name, latitude, longitude, dt_utc
            FROM ctd_profiles
            WHERE latitude IS NOT NULL
              AND longitude IS NOT NULL
              AND (((project_id IS NULL) AND (? IS NULL)) OR project_id = ?)
            ORDER BY dt_utc DESC, id DESC
            """,
            (pid, pid),
        )
        rows = cur.fetchall()
        conn.close()
        return rows


    def _waypoint_symbol_pg(self, symbol_name):
        smap = {
            'star': 'star',
            'circle': 'o',
            'square': 's',
            'triangle': 't',
            'diamond': 'd',
            'cross': 'x',
        }
        return smap.get((symbol_name or 'star').strip().lower(), 'star')

    def _waypoint_icon_folium(self, symbol_name):
        fmap = {
            'star': 'star',
            'circle': 'circle',
            'square': 'square',
            'triangle': 'play',
            'diamond': 'certificate',
            'cross': 'times',
        }
        return fmap.get((symbol_name or 'star').strip().lower(), 'star')

    def _render_folium_chart_map(self, tracks, ctd_rows, waypoint_rows):
        if self.gps_map_view is None or folium is None:
            return

        all_lat = []
        all_lon = []
        for tr in tracks:
            all_lat.extend(tr["lat"])
            all_lon.extend(tr["lon"])
        for _, _, lat, lon, _ in ctd_rows:
            try:
                all_lat.append(float(lat)); all_lon.append(float(lon))
            except Exception:
                pass
        for _, _, lat, lon, _, _ in waypoint_rows:
            try:
                all_lat.append(float(lat)); all_lon.append(float(lon))
            except Exception:
                pass

        if all_lat and all_lon:
            center = [float(sum(all_lat) / len(all_lat)), float(sum(all_lon) / len(all_lon))]
            zoom = 11
        else:
            center = [0.0, 0.0]
            zoom = 2

        m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True)
        folium.TileLayer(
            tiles='OpenStreetMap',
            name='Street Map',
            overlay=False,
            control=True,
        ).add_to(m)
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Tiles © Esri',
            name='Aerial',
            overlay=False,
            control=True,
        ).add_to(m)
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}',
            attr='Tiles © Esri — GEBCO, NOAA, National Geographic, DeLorme, HERE, Geonames.org',
            name='Bathymetry / Ocean',
            overlay=False,
            control=True,
        ).add_to(m)

        for tr in tracks:
            pts = list(zip(tr["lat"], tr["lon"]))
            if not pts:
                continue
            if len(pts) > 5000:
                step = max(1, len(pts) // 5000)
                pts = pts[::step]
                if pts[-1] != (tr["lat"][-1], tr["lon"][-1]):
                    pts.append((tr["lat"][-1], tr["lon"][-1]))
            folium.PolyLine(pts, color=tr["color"], weight=3, opacity=0.9, tooltip=tr["name"]).add_to(m)
            folium.CircleMarker(pts[0], radius=4, color=tr["color"], fill=True, fill_opacity=1.0,
                                tooltip=f'{tr["name"]} start').add_to(m)

        for ctd_id, name, lat, lon, dt_utc in ctd_rows:
            try:
                latf = float(lat); lonf = float(lon)
            except Exception:
                continue
            lab = f"CTD: {name or ('Cast %s' % ctd_id)}"
            if dt_utc:
                lab += f"<br>{dt_utc}"
            folium.Marker([latf, lonf], popup=lab, icon=folium.Icon(color='orange', icon='tint', prefix='fa')).add_to(m)

        for wp_id, wp_name, lat, lon, proj_id, symbol in waypoint_rows:
            try:
                latf = float(lat); lonf = float(lon)
            except Exception:
                continue
            scope = 'Global' if proj_id is None else 'Project'
            folium.Marker([latf, lonf], popup=f"Waypoint: {wp_name}<br>{scope}<br>Symbol: {symbol}",
                          icon=folium.Icon(color='blue', icon=self._waypoint_icon_folium(symbol), prefix='fa')).add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)

        out = tempfile.NamedTemporaryFile(prefix='chart_map_', suffix='.html', delete=False)
        out.close()
        m.save(out.name)
        self._gps_folium_html_path = out.name
        self.gps_map_view.setUrl(QUrl.fromLocalFile(out.name))


    def _plot_selected_gps_tracks(self):
        selected = [i.data(QtCore.Qt.UserRole) for i in self.gps_track_list.selectedItems()] if hasattr(self, 'gps_track_list') else []
        total_points = 0
        all_lon = []
        all_lat = []
        palette = self._ordered_palette() if hasattr(self, '_ordered_palette') else ['#03DFE2']

        tracks = []
        if selected:
            for idx, track_id in enumerate(selected):
                name, color, pts = self._fetch_track_points(track_id)
                if not pts:
                    continue
                lat = [float(p[0]) for p in pts]
                lon = [float(p[1]) for p in pts]
                total_points += len(pts)
                all_lon.extend(lon)
                all_lat.extend(lat)
                line_color = palette[idx % len(palette)] if palette else (color or '#03DFE2')
                tracks.append({'name': name, 'color': line_color, 'lat': lat, 'lon': lon})

        ctd_count = 0
        try:
            ctd_rows = self._fetch_ctd_points_for_chart()
        except Exception:
            ctd_rows = []
        for _, _, lat, lon, _ in ctd_rows:
            try:
                all_lon.append(float(lon)); all_lat.append(float(lat)); ctd_count += 1
            except Exception:
                pass

        try:
            waypoint_rows = self._fetch_waypoints_for_chart()
        except Exception:
            waypoint_rows = []
        wp_count = 0
        for _, _, lat, lon, _, _ in waypoint_rows:
            try:
                all_lon.append(float(lon)); all_lat.append(float(lat)); wp_count += 1
            except Exception:
                pass

        use_web_map = False  # disable folium/webengine path due runtime freezes; use stable pyqtgraph renderer

        if use_web_map:
            self._render_folium_chart_map(tracks, ctd_rows, waypoint_rows)
            if hasattr(self, 'gps_map_stack'):
                self.gps_map_stack.setCurrentWidget(self.gps_map_view)
            if hasattr(self, 'gps_cursor_label'):
                self.gps_cursor_label.setText('Cursor: hover coordinates available in PyQtGraph mode')
        else:
            if not hasattr(self, 'gps_plot'):
                return
            if hasattr(self, 'gps_map_stack'):
                self.gps_map_stack.setCurrentWidget(self.gps_plot)
            self.gps_plot.clear()
            self.gps_plot.addLegend()
            for tr in tracks:
                self.gps_plot.plot(tr['lon'], tr['lat'], pen=pg.mkPen(tr['color'], width=2), name=tr['name'])
                self.gps_plot.plot([tr['lon'][0]], [tr['lat'][0]], pen=None, symbol='o', symbolSize=7,
                                   symbolBrush=pg.mkBrush(tr['color']), name=f"{tr['name']} start")

            for idx, (ctd_id, name, lat, lon, dt_utc) in enumerate(ctd_rows):
                try:
                    latf = float(lat); lonf = float(lon)
                except Exception:
                    continue
                label = f"CTD: {name or f'Cast {ctd_id}'}"
                if dt_utc:
                    label += f" ({str(dt_utc).split('T')[0]})"
                self.gps_plot.plot([lonf], [latf], pen=None, symbol='t', symbolSize=10,
                                   symbolBrush=pg.mkBrush('#FFD166'), symbolPen=pg.mkPen('#333333', width=1),
                                   name=label if idx == 0 else None)

            for idx, (wp_id, wp_name, lat, lon, proj_id, symbol) in enumerate(waypoint_rows):
                try:
                    latf = float(lat); lonf = float(lon)
                except Exception:
                    continue
                scope = 'Global' if proj_id is None else 'Project'
                self.gps_plot.plot([lonf], [latf], pen=None, symbol=self._waypoint_symbol_pg(symbol), symbolSize=12,
                                   symbolBrush=pg.mkBrush('#4DA3FF'), symbolPen=pg.mkPen('#1E3A5F', width=1),
                                   name=(f"Waypoint: {wp_name} ({scope})" if idx == 0 else None))

            if all_lon and all_lat:
                self.gps_plot.setXRange(min(all_lon), max(all_lon), padding=0.05)
                self.gps_plot.setYRange(min(all_lat), max(all_lat), padding=0.05)

        if not tracks and ctd_count == 0 and wp_count == 0:
            self.gps_info_label.setText('No tracks selected')
            return

        backend = 'Folium' if use_web_map else 'PyQtGraph'
        self.gps_info_label.setText(
            f"Map: {backend}   Tracks: {len(tracks)}   Track Points: {total_points}   CTD Casts: {ctd_count}   Waypoints: {wp_count}"
        )


    def _fit_gps_view(self):
        self._plot_selected_gps_tracks()

    def delete_selected_gps_tracks(self):
        if not hasattr(self, 'gps_track_list'):
            return
        ids = [i.data(QtCore.Qt.UserRole) for i in self.gps_track_list.selectedItems()]
        if not ids:
            return
        if QtWidgets.QMessageBox.question(self, "Delete Tracks", f"Delete {len(ids)} selected track(s)?") != QtWidgets.QMessageBox.Yes:
            return
        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        for tid in ids:
            cur.execute("DELETE FROM gps_track_points WHERE track_id=?", (int(tid),))
            cur.execute("DELETE FROM gps_tracks WHERE id=?", (int(tid),))
        conn.commit(); conn.close()
        self.refresh_chart_tracks()

    def setup_logs_tab(self):
        # paging state
        self.log_current_page = 0
        self.log_entries_per_page = 50

        layout = QtWidgets.QVBoxLayout(self.logs_tab)

        control_layout = QtWidgets.QHBoxLayout()

        control_layout.addWidget(QtWidgets.QLabel("Rows/Page:"))
        self.log_page_size_combo = QtWidgets.QComboBox()
        self.log_page_size_combo.addItems(["50", "100", "200", "500"])
        self.log_page_size_combo.setCurrentText("100")
        control_layout.addWidget(self.log_page_size_combo)

        # Project filter
        control_layout.addWidget(QtWidgets.QLabel("Project:"))
        self.log_project_cb = QtWidgets.QComboBox()
        self.log_project_cb.addItem("All Projects")
        control_layout.addWidget(self.log_project_cb)

        control_layout.addWidget(QtWidgets.QLabel("Filter by File:"))
        self.log_filter_combo = QtWidgets.QComboBox()
        control_layout.addWidget(self.log_filter_combo)

        control_layout.addWidget(QtWidgets.QLabel("Filter by Method:"))
        self.method_filter_combo = QtWidgets.QComboBox()
        control_layout.addWidget(self.method_filter_combo)

        control_layout.addWidget(QtWidgets.QLabel("Sig figs:"))
        self.log_sigfig_combo = QtWidgets.QComboBox()
        for n in (2, 3, 4, 5, 6):
            self.log_sigfig_combo.addItem(str(n))
        self.log_sigfig_combo.setCurrentText("3")
        self.log_sig_figs = 3
        control_layout.addWidget(self.log_sigfig_combo)

        # Plotting tools (mirroring SPL tab capabilities)
        self.log_plot_time_btn = QtWidgets.QPushButton("Plot Value vs Time")
        self.log_plot_freq_btn = QtWidgets.QPushButton("Plot Value vs Frequency")
        control_layout.addWidget(self.log_plot_time_btn)
        control_layout.addWidget(self.log_plot_freq_btn)

        layout.addLayout(control_layout)

        headers = [
            "ID", "File Name", "Method", "Target Frequency", "Start Time", "End Time",
            "Window Length", "Max Voltage", "Bandwidth", "Measured Voltage",
            "Filter", "Screenshot", "Misc", "Timestamp", "Delete"
        ]
        self.log_table = QtWidgets.QTableWidget(0, len(headers))
        self.log_table.setHorizontalHeaderLabels(headers)
        self.log_table.setEditTriggers(
            QtWidgets.QAbstractItemView.DoubleClicked |
            QtWidgets.QAbstractItemView.SelectedClicked |
            QtWidgets.QAbstractItemView.EditKeyPressed
        )
        self.log_table.itemChanged.connect(self.on_log_item_changed)
        layout.addWidget(self.log_table)
        self.log_table.setColumnWidth(9, 150)

        # --- Paging controls for Logs tab ---
        paging_layout = QtWidgets.QHBoxLayout()

        self.prev_page_btn = QtWidgets.QPushButton("◀ Prev")
        self.prev_page_btn.setFixedWidth(80)

        self.page_label = QtWidgets.QLabel("Page 0 of 0")
        self.page_label.setStyleSheet("color: white;")
        self.page_label.setAlignment(QtCore.Qt.AlignCenter)

        self.next_page_btn = QtWidgets.QPushButton("Next ▶")
        self.next_page_btn.setFixedWidth(80)

        paging_layout.addWidget(self.prev_page_btn)
        paging_layout.addWidget(self.page_label)
        paging_layout.addWidget(self.next_page_btn)
        paging_layout.addStretch()

        self.export_logs_btn = QtWidgets.QPushButton("Export CSV")
        paging_layout.addWidget(self.export_logs_btn)

        layout.addLayout(paging_layout)

        # Hook up paging buttons
        self.prev_page_btn.clicked.connect(self.prev_log_page)
        self.next_page_btn.clicked.connect(self.next_log_page)
        self.export_logs_btn.clicked.connect(self.export_logs_to_csv)


        # ---- NEW: Debounced refresh timer ----
        self._ensure_logs_refresh_timer()

        # ---- NEW: Single handler for all filter widgets ----
        refresh_from_filter = lambda *_: self.request_logs_refresh()

        self.log_page_size_combo.currentTextChanged.connect(self.on_entries_per_page_changed)
        self.log_project_cb.currentTextChanged.connect(refresh_from_filter)
        self.log_filter_combo.currentTextChanged.connect(refresh_from_filter)
        self.method_filter_combo.currentTextChanged.connect(refresh_from_filter)
        self.log_sigfig_combo.currentTextChanged.connect(self._on_log_sigfigs_changed)
        self.log_plot_time_btn.clicked.connect(self.plot_logs_over_time)
        self.log_plot_freq_btn.clicked.connect(self.plot_logs_vs_frequency)

        # Initial population
        self.request_logs_refresh(immediate=True)

    def populate_log_table(self):
        """
        Fill the Logs table using server-side filtering + paging so the tab
        doesn't hang when there are lots of rows.

        IMPORTANT: Avoid JOIN fan-out duplicates by using EXISTS for project filtering.
        """
        # --- Read filters from the UI ---
        proj_name = None
        proj_id = None
        if hasattr(self, "log_project_cb"):
            proj_name = self.log_project_cb.currentText()
            if proj_name == "All Projects":
                proj_name = None
        if proj_name:
            proj_id = self._get_project_id(proj_name)

        fname = None
        if hasattr(self, "log_filter_combo"):
            fname = self.log_filter_combo.currentText()
            if fname in (None, "", "All"):
                fname = None

        meth = None
        if hasattr(self, "method_filter_combo"):
            meth = self.method_filter_combo.currentText()
            if meth in (None, "", "All"):
                meth = None

        # --- Page size ---
        try:
            page_size = int(self.log_page_size_combo.currentText())
        except Exception:
            page_size = 50
        if page_size <= 0:
            page_size = 50

        table = "measurements"

        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()

        # --- Base WHERE + params (NO JOINs here) ---
        where = ["1=1"]
        params = []

        if fname:
            where.append("m.file_name = ?")
            params.append(fname)
        if meth:
            where.append("m.method = ?")
            params.append(meth)

        # Project filter (NO JOIN fan-out): use EXISTS with indexed project_id
        if proj_id:
            where.append("""
                EXISTS (
                    SELECT 1
                    FROM project_items pi
                    WHERE pi.file_name = m.file_name
                    AND pi.method    = m.method
                    AND pi.project_id = ?
                )
            """)
            params.append(proj_id)

        where_sql = " WHERE " + " AND ".join(where)

        # --- 1) Get total row count for paging ---
        cur.execute(f"SELECT COUNT(*) FROM {table} m" + where_sql, params)
        total_rows = cur.fetchone()[0] or 0

        if total_rows == 0:
            max_page = 0
            self.log_current_page = 0
            page_rows = []
        else:
            max_page = max(0, (total_rows - 1) // page_size)
            if self.log_current_page > max_page:
                self.log_current_page = max_page

            offset = self.log_current_page * page_size

            # --- 2) Fetch only the current page of rows ---
            cur.execute(
                f"""
                SELECT
                    m.id, m.file_name, m.method, m.target_frequency,
                    m.start_time, m.end_time, m.window_length,
                    m.max_voltage, m.bandwidth, m.measured_voltage,
                    m.filter_applied, m.screenshot, m.misc, m.timestamp
                FROM {table} m
                {where_sql}
                ORDER BY m.id DESC
                LIMIT ? OFFSET ?
                """,
                params + [page_size, offset],
            )
            page_rows = cur.fetchall()

        conn.close()

        # --- Prepare table for new data (robust clear: items + widgets) ---
        self.log_table.blockSignals(True)
        self.log_table.setUpdatesEnabled(False)

        prev_rows = self.log_table.rowCount()
        prev_cols = self.log_table.columnCount()
        for r in range(prev_rows):
            for c in range(prev_cols):
                w = self.log_table.cellWidget(r, c)
                if w is not None:
                    self.log_table.removeCellWidget(r, c)
                    w.deleteLater()

        self.log_table.clearContents()
        self.log_table.setRowCount(0)
        self.log_table.setRowCount(len(page_rows))

        # 14 DB cols + 1 Delete col
        self.log_table.setColumnCount(15)

        self.log_table.setUpdatesEnabled(True)

        # --- Fill table ---
        for row_idx, row in enumerate(page_rows):
            if row is None or len(row) < 14:
                continue

            for col_idx, val in enumerate(row):
                # Screenshot column → "View" button
                if col_idx == 11:
                    if val:
                        btn = QtWidgets.QPushButton("View")
                        path = val
                        btn.clicked.connect(
                            lambda _, p=path: QDesktopServices.openUrl(QUrl.fromLocalFile(p))
                        )
                        self.log_table.setCellWidget(row_idx, col_idx, btn)
                    else:
                        self.log_table.setItem(row_idx, col_idx, QtWidgets.QTableWidgetItem(""))
                else:
                    # Leave the frequency column as-is so the sig-fig selector only
                    # affects the other numeric columns.
                    if col_idx in (4, 5, 6, 7, 8, 9):
                        text = self._format_log_value(val)
                    else:
                        text = "" if val is None else str(val)
                    item = QtWidgets.QTableWidgetItem(text)

                    # ID and Timestamp read-only
                    if col_idx in (0, 13):
                        item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                    else:
                        item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)

                    self.log_table.setItem(row_idx, col_idx, item)

            # Delete button (extra column)
            rec_id = row[0]
            del_btn = QtWidgets.QPushButton("Delete")
            del_btn.setStyleSheet("background-color: #7bdff2; color: black;")
            del_btn.clicked.connect(lambda _, rid=rec_id: self.delete_record(rid))
            self.log_table.setCellWidget(row_idx, 14, del_btn)

        self.log_table.blockSignals(False)

        # --- Update paging UI ---
        if total_rows == 0:
            self.page_label.setText("Page 0 of 0")
        else:
            self.page_label.setText(f"Page {self.log_current_page + 1} of {max_page + 1}")
        self.prev_page_btn.setEnabled(self.log_current_page > 0)
        self.next_page_btn.setEnabled(self.log_current_page < max_page)




    def export_logs_to_csv(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Logs to CSV", "", "CSV Files (*.csv)"
        )
        if not filename:
            return

        table = getattr(self, "log_table", None)
        if table is None:
            QtWidgets.QMessageBox.critical(self, "Error", "Logs table is not available.")
            return

        export_cols = max(0, table.columnCount() - 1)  # skip Delete button column
        headers = []
        for c in range(export_cols):
            hitem = table.horizontalHeaderItem(c)
            headers.append(hitem.text() if hitem else f"Column {c}")

        try:
            with open(filename, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)
                for r in range(table.rowCount()):
                    row = []
                    for c in range(export_cols):
                        item = table.item(r, c)
                        if item is not None:
                            row.append(item.text())
                        else:
                            widget = table.cellWidget(r, c)
                            row.append(widget.text() if widget and hasattr(widget, "text") else "")
                    writer.writerow(row)
            QtWidgets.QMessageBox.information(
                self, "Export", f"Visible log rows exported to {filename}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to export logs:\n{e}")



    def _logs_filter_query(self, require_file=False, file_list=None):
        """Build a WHERE clause + params matching current log filters."""

        proj_name = None
        proj_id = None
        if hasattr(self, "log_project_cb"):
            proj_name = self.log_project_cb.currentText()
            if proj_name == "All Projects":
                proj_name = None
        if proj_name:
            proj_id = self._get_project_id(proj_name)

        fname = None
        if hasattr(self, "log_filter_combo"):
            fname = self.log_filter_combo.currentText()
            if fname in (None, "", "All"):
                fname = None

        meth = None
        if hasattr(self, "method_filter_combo"):
            meth = self.method_filter_combo.currentText()
            if meth in (None, "", "All"):
                meth = None

        chosen_files = file_list if file_list else None

        if require_file and not fname and not chosen_files:
            QtWidgets.QMessageBox.information(
                self,
                "Select File",
                "Please pick one or more files in the Logs view first.",
            )
            return None, None

        where = ["1=1"]
        params = []

        if fname:
            where.append("m.file_name = ?")
            params.append(fname)
        elif chosen_files:
            placeholders = ",".join(["?"] * len(chosen_files))
            where.append(f"m.file_name IN ({placeholders})")
            params.extend(chosen_files)
        if meth:
            where.append("m.method = ?")
            params.append(meth)
        if proj_id:
            where.append(
                """
                EXISTS (
                    SELECT 1 FROM project_items pi
                    WHERE pi.file_name = m.file_name
                    AND pi.method    = m.method
                    AND pi.project_id = ?
                )
                """
            )
            params.append(proj_id)

        return " WHERE " + " AND ".join(where), params

    def _log_available_files_for_plot(self):
        """Return a list of distinct log file names respecting method/project filters."""

        import sqlite3

        proj_name = None
        proj_id = None
        if hasattr(self, "log_project_cb"):
            proj_name = self.log_project_cb.currentText()
            if proj_name == "All Projects":
                proj_name = None
        if proj_name:
            proj_id = self._get_project_id(proj_name)

        meth = None
        if hasattr(self, "method_filter_combo"):
            meth = self.method_filter_combo.currentText()
            if meth in (None, "", "All"):
                meth = None

        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        try:
            where = ["m.file_name IS NOT NULL"]
            params = []
            if meth:
                where.append("m.method = ?")
                params.append(meth)
            if proj_id:
                where.append(
                    """
                    EXISTS (
                        SELECT 1 FROM project_items pi
                        WHERE pi.file_name = m.file_name
                        AND pi.method    = m.method
                        AND pi.project_id = ?
                    )
                    """
                )
                params.append(proj_id)

            where_sql = " WHERE " + " AND ".join(where) if where else ""
            cur.execute(
                f"SELECT DISTINCT m.file_name FROM measurements m{where_sql} ORDER BY m.file_name",
                tuple(params),
            )
            return [r[0] for r in cur.fetchall() if r[0]]
        finally:
            conn.close()

    def _choose_files_for_plot(self, files, title, max_files=10, multi=True):
        """Prompt the user to choose up to ``max_files`` from ``files``."""

        if not files:
            QtWidgets.QMessageBox.information(self, "No Files", "No files are available to plot for this view.")
            return None

        if len(files) == 1:
            return [files[0]]

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(title)
        layout = QtWidgets.QVBoxLayout(dlg)

        info = QtWidgets.QLabel(f"Select up to {max_files} files to plot.")
        layout.addWidget(info)

        listw = QtWidgets.QListWidget()
        listw.addItems(files)
        listw.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection if multi else QtWidgets.QAbstractItemView.SingleSelection)
        layout.addWidget(listw)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        layout.addWidget(btns)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return None

        selected = [i.text() for i in listw.selectedItems()]
        if not selected:
            QtWidgets.QMessageBox.information(self, "No Selection", "No files were chosen for plotting.")
            return None

        if len(selected) > max_files:
            QtWidgets.QMessageBox.warning(
                self,
                "Too many files",
                f"You selected {len(selected)} files; only the first {max_files} will be plotted.",
            )
            selected = selected[:max_files]

        return selected

    def _resolve_log_plot_files(self):
        """Return a list of log files to plot based on the current filters."""

        current = self.log_filter_combo.currentText() if hasattr(self, "log_filter_combo") else None
        if current and current not in ("All", ""):
            return [current]

        files = self._log_available_files_for_plot()
        return self._choose_files_for_plot(files, "Select Log Files to Plot", max_files=10)


    def _logs_axis_controls(self, parent_layout, ax, canvas, data_x, data_y):
        """Attach shared axis/scale/save controls to a plot dialog."""

        import numpy as np

        ctl = QtWidgets.QHBoxLayout()

        ctl.addWidget(QtWidgets.QLabel("X Range:"))
        x_entry = QtWidgets.QLineEdit()
        x_entry.setPlaceholderText("min,max")
        x_entry.setFixedWidth(90)
        ctl.addWidget(x_entry)

        ctl.addSpacing(8)
        ctl.addWidget(QtWidgets.QLabel("Y Range:"))
        y_entry = QtWidgets.QLineEdit()
        y_entry.setPlaceholderText("min,max")
        y_entry.setFixedWidth(90)
        ctl.addWidget(y_entry)

        ctl.addSpacing(8)
        ctl.addWidget(QtWidgets.QLabel("X Scale:"))
        x_scale_combo = QtWidgets.QComboBox()
        x_scale_combo.addItems(["Linear", "Log"])
        ctl.addWidget(x_scale_combo)

        ctl.addSpacing(8)
        ctl.addWidget(QtWidgets.QLabel("Y Scale:"))
        y_scale_combo = QtWidgets.QComboBox()
        y_scale_combo.addItems(["Linear", "Log"])
        ctl.addWidget(y_scale_combo)

        apply_btn = QtWidgets.QPushButton("Apply")
        ctl.addWidget(apply_btn)

        ctl.addStretch()

        save_btn = QtWidgets.QPushButton("Save as JPG")
        ctl.addWidget(save_btn)

        parent_layout.addLayout(ctl)

        def apply_changes():
            # Limits
            try:
                xmin, xmax = map(float, x_entry.text().split(","))
                ax.set_xlim(xmin, xmax)
            except Exception:
                pass
            try:
                ymin, ymax = map(float, y_entry.text().split(","))
                ax.set_ylim(ymin, ymax)
            except Exception:
                pass

            # Scales (guard against non-positive data for log)
            def _safe_scale(setter, combo, values, which):
                mode = combo.currentText().lower()
                if mode == "log":
                    arr = np.array(values, dtype=float) if values is not None else np.array([])
                    if arr.size == 0 or np.any(arr <= 0):
                        QtWidgets.QMessageBox.warning(
                            self,
                            "Log Scale",
                            f"Cannot use log scale on {which}-axis with non-positive values.",
                        )
                        combo.setCurrentText("Linear")
                        setter("linear")
                    else:
                        setter("log")
                else:
                    setter("linear")

            _safe_scale(ax.set_xscale, x_scale_combo, data_x, "X")
            _safe_scale(ax.set_yscale, y_scale_combo, data_y, "Y")
            canvas.draw_idle()

        apply_btn.clicked.connect(apply_changes)
        save_btn.clicked.connect(lambda: self._save_figure_jpg(ax.figure))

        return apply_changes


    def plot_logs_over_time(self):
        """Plot measured values vs. start time using current log filters."""

        import sqlite3, numpy as np, matplotlib.pyplot as plt
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

        files_to_plot = self._resolve_log_plot_files()
        if not files_to_plot:
            return

        where_sql, params = self._logs_filter_query(require_file=True, file_list=files_to_plot)
        if where_sql is None:
            return

        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT m.file_name, m.start_time, m.measured_voltage, m.target_frequency, m.misc
            FROM measurements m
            {where_sql}
            ORDER BY m.file_name ASC, m.start_time ASC, m.id ASC
            """,
            params,
        )
        rows = cur.fetchall()
        conn.close()

        grouped = {}
        for fname, t, val, f, misc in rows:
            if fname is None or t is None or val is None:
                continue
            try:
                grouped.setdefault(fname, {"t": [], "v": [], "f": [], "misc": []})
                grouped[fname]["t"].append(float(t))
                grouped[fname]["v"].append(float(val))
                grouped[fname]["f"].append(f)
                grouped[fname]["misc"].append(misc or "")
            except Exception:
                continue

        if not grouped:
            QtWidgets.QMessageBox.information(self, "No Data", "No log entries available for this plot.")
            return

        fig, ax = plt.subplots(figsize=(9, 5), facecolor="#19232D")
        ax.set_facecolor("#000000")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")
        ax.tick_params(colors="white")
        ax.set_title("Measurement vs Time", color="white")
        ax.set_xlabel("Start Time (s)", color="white")

        joined_misc = " ".join([m for g in grouped.values() for m in g.get("misc", [])]).lower()
        y_label = "Measured Value"
        if "current" in joined_misc:
            y_label = "Measured Current"
        elif "voltage" in joined_misc:
            y_label = "Measured Voltage"
        ax.set_ylabel(y_label, color="white")

        import itertools

        palette_cycle = self._ordered_palette()
        colors = itertools.cycle(palette_cycle if palette_cycle else ["#03DFE2"])
        for fname in sorted(grouped.keys()):
            col = next(colors)
            ax.plot(grouped[fname]["t"], grouped[fname]["v"], lw=1.2, label=fname, color=col)

        if len(grouped) > 1:
            ax.legend(facecolor="#222", labelcolor="white")

        canvas = FigureCanvas(fig)
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Log Plot: Value vs Time")
        dlg.resize(980, 640)
        vbox = QtWidgets.QVBoxLayout(dlg)

        all_times = [t for g in grouped.values() for t in g["t"]]
        all_vals = [v for g in grouped.values() for v in g["v"]]
        self._logs_axis_controls(vbox, ax, canvas, all_times, all_vals)
        vbox.addWidget(canvas)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        hb = QtWidgets.QHBoxLayout()
        hb.addStretch(); hb.addWidget(close_btn)
        vbox.addLayout(hb)

        dlg.exec_()
        plt.close(fig)


    def plot_logs_vs_frequency(self):
        """Plot measured values vs. target frequency using current log filters."""

        import sqlite3, numpy as np, matplotlib.pyplot as plt
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

        files_to_plot = self._resolve_log_plot_files()
        if not files_to_plot:
            return

        where_sql, params = self._logs_filter_query(require_file=True, file_list=files_to_plot)
        if where_sql is None:
            return

        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT m.file_name, m.target_frequency, m.measured_voltage, m.misc
            FROM measurements m
            {where_sql}
            AND m.target_frequency IS NOT NULL
            ORDER BY m.file_name ASC, m.target_frequency ASC, m.id ASC
            """,
            params,
        )
        rows = cur.fetchall()
        conn.close()

        grouped = {}
        for fname, f, val, misc in rows:
            if fname is None or f is None or val is None:
                continue
            try:
                grouped.setdefault(fname, {"f": [], "v": [], "misc": []})
                grouped[fname]["f"].append(float(f))
                grouped[fname]["v"].append(float(val))
                grouped[fname]["misc"].append(misc or "")
            except Exception:
                continue

        if not grouped:
            QtWidgets.QMessageBox.information(self, "No Data", "No frequency-tagged log entries to plot.")
            return

        fig, ax = plt.subplots(figsize=(9, 5), facecolor="#19232D")
        ax.set_facecolor("#000000")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")
        ax.tick_params(colors="white")
        ax.set_title("Measurement vs Frequency", color="white")
        ax.set_xlabel("Frequency (Hz)", color="white")

        joined_misc = " ".join([m for g in grouped.values() for m in g.get("misc", [])]).lower()
        y_label = "Measured Value"
        if "current" in joined_misc:
            y_label = "Measured Current"
        elif "voltage" in joined_misc:
            y_label = "Measured Voltage"
        ax.set_ylabel(y_label, color="white")

        import itertools

        palette_cycle = self._ordered_palette()
        colors = itertools.cycle(palette_cycle if palette_cycle else ["#03DFE2"])

        for fname in sorted(grouped.keys()):
            col = next(colors)
            ax.plot(grouped[fname]["f"], grouped[fname]["v"], color=col, lw=1.2, label=fname)

        if len(grouped) > 1:
            ax.legend(facecolor="#222", labelcolor="white")

        canvas = FigureCanvas(fig)
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Log Plot: Value vs Frequency")
        dlg.resize(980, 640)
        vbox = QtWidgets.QVBoxLayout(dlg)

        all_freqs = [f for g in grouped.values() for f in g["f"]]
        all_vals = [v for g in grouped.values() for v in g["v"]]
        self._logs_axis_controls(vbox, ax, canvas, all_freqs, all_vals)
        vbox.addWidget(canvas)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        hb = QtWidgets.QHBoxLayout()
        hb.addStretch(); hb.addWidget(close_btn)
        vbox.addLayout(hb)

        dlg.exec_()
        plt.close(fig)


    def delete_record(self, rid):
        if QtWidgets.QMessageBox.question(
                self, "Confirm", f"Delete ID {rid}?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        ) == QtWidgets.QMessageBox.Yes:
            conn = sqlite3.connect(DB_FILENAME)
            cur = conn.cursor()
            cur.execute("DELETE FROM measurements WHERE id=?", (rid,))
            conn.commit()
            conn.close()
            self.request_logs_refresh(immediate=True)

    def delete_logs_by_filename(self):
        filename = self.log_filter_combo.currentText()
        if filename == "All" or not filename:
            QtWidgets.QMessageBox.critical(self, "Error", "Please select a specific filename to delete all records.")
            return
        reply = QtWidgets.QMessageBox.question(
            self, "Confirm Delete",
            f"Delete all records for device '{filename}'?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.Yes:
            conn = sqlite3.connect(DB_FILENAME)
            cur = conn.cursor()
            cur.execute("DELETE FROM measurements WHERE file_name = ?", (filename,))
            conn.commit()
            conn.close()
            self.request_logs_refresh(immediate=True)
            QtWidgets.QMessageBox.information(self, "Deleted", f"All records for '{filename}' deleted.")
    def _update_delete_all_btn_state(self):
        """
        Enable the 'Delete All' button only when a specific filename is selected.
        """
        filename = self.log_filter_combo.currentText()
        enabled = bool(filename and filename != "All")
        self.delete_all_logs_btn.setEnabled(enabled)

    def delete_all_logs(self):
        """
        Delete all log entries currently displayed under the active filename/method filters,
        but do NOT delete the entire table.
        """
        # Pull current filter selections
        fname  = self.log_filter_combo.currentText()
        method = self.method_filter_combo.currentText()

        # Build WHERE clauses based on filters
        where_clauses = []
        params = []
        if fname and fname != "All":
            where_clauses.append("file_name = ?")
            params.append(fname)
        if method and method != "All":
            where_clauses.append("method = ?")
            params.append(method)

        # If no filters, warn that this will delete everything
        if not where_clauses:
            msg = "This will delete *all* log entries. Continue?"
        else:
            msg = (
                f"This will delete all log entries for file '{fname}'"
                + (f" and method '{method}'." if method and method != "All" else ".")
                + " Continue?"
            )

        # Confirm with the user
        reply = QtWidgets.QMessageBox.question(
            self, "Confirm Delete", msg,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return

        # Construct and execute the DELETE statement
        sql = "DELETE FROM measurements"
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)

        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        cur.execute(sql, params)
        conn.commit()
        conn.close()

        # Refresh UI
        self.request_logs_refresh(immediate=True)
        QtWidgets.QMessageBox.information(
            self, "Deleted",
            f"Deleted {('all entries' if not where_clauses else 'filtered entries')}."
        )




    def _on_file_filter_changed(self):
        """
        Whenever the filename filter changes, repopulate the Method filter
        to only include methods present in that file (or All).
        """
        table = "measurements"
        fname = self.log_filter_combo.currentText()

        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        if fname != "All":
            cur.execute(f"SELECT DISTINCT method FROM {table} WHERE file_name = ?", (fname,))
        else:
            cur.execute(f"SELECT DISTINCT method FROM {table}")
        rows = cur.fetchall()
        conn.close()

        methods = sorted({r[0] for r in rows if r[0]})
        self.method_filter_combo.blockSignals(True)
        self.method_filter_combo.clear()
        self.method_filter_combo.addItem("All")
        for m in methods:
            self.method_filter_combo.addItem(m)
        self.method_filter_combo.blockSignals(False)

        # After changing methods, refresh the table
        self.request_logs_refresh(immediate=True)


    def update_log_filter_options(self):
        """
        Populate the Project and Filename filter dropdowns for active measurements only.
        """
        table = "measurements"

        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()

        # --- Read selected project (do NOT rebuild combo here) ---
        proj_id = None
        if hasattr(self, "log_project_cb"):
            proj_name = self.log_project_cb.currentText()
            if proj_name == "All Projects":
                proj_name = None
            if proj_name:
                proj_id = self._get_project_id(proj_name)
        else:
            proj_name = None

        # --- Build filename list for current project ---
        if proj_id:
            sql = f"""
                SELECT DISTINCT m.file_name
                FROM {table} m
                JOIN project_items pi
                     ON (m.file_name = pi.file_name AND m.method = pi.method)
                WHERE pi.project_id = ?
                ORDER BY m.file_name
            """
            cur.execute(sql, (proj_id,))
            filenames = [r[0] for r in cur.fetchall()]
        else:
            cur.execute(f"SELECT DISTINCT file_name FROM {table} ORDER BY file_name")
            filenames = [r[0] for r in cur.fetchall()]

        conn.close()

        filenames = sorted(f for f in filenames if f)

        # --- Update filename combo ---
        current_file = self.log_filter_combo.currentText()
        self.log_filter_combo.blockSignals(True)
        self.log_filter_combo.clear()
        self.log_filter_combo.addItem("All")
        for fn in filenames:
            self.log_filter_combo.addItem(fn)
        idx = self.log_filter_combo.findText(current_file)
        if idx < 0:
            idx = 0
        self.log_filter_combo.setCurrentIndex(idx)
        self.log_filter_combo.blockSignals(False)

        # Also refresh methods to stay consistent with the new filename list
        #self.update_method_filter_options()



    def update_method_filter_options(self):
        """
        Populate the Method filter dropdown based on selected project and filename.
        """
        table = "measurements"

        fname = self.log_filter_combo.currentText()
        proj_name = None
        proj_id = None
        if hasattr(self, "log_project_cb"):
            proj_name = self.log_project_cb.currentText()
            if proj_name == "All Projects":
                proj_name = None
            if proj_name:
                proj_id = self._get_project_id(proj_name)

        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()

        base_sql = f"SELECT DISTINCT m.method FROM {table} m"
        where = []
        params = []

        if proj_id:
            base_sql += """
                JOIN project_items pi
                     ON (m.file_name = pi.file_name AND m.method = pi.method)
            """
            where.append("pi.project_id = ?")
            params.append(proj_id)

        if fname and fname != "All":
            where.append("m.file_name = ?")
            params.append(fname)

        if where:
            base_sql += " WHERE " + " AND ".join(where)

        cur.execute(base_sql, params)
        rows = cur.fetchall()
        conn.close()

        methods = sorted({r[0] for r in rows if r[0]})
        current_method = self.method_filter_combo.currentText()
        self.method_filter_combo.blockSignals(True)
        self.method_filter_combo.clear()
        self.method_filter_combo.addItem("All")
        for mname in methods:
            self.method_filter_combo.addItem(mname)
        idx = self.method_filter_combo.findText(current_method)
        if idx < 0:
            idx = 0
        self.method_filter_combo.setCurrentIndex(idx)
        self.method_filter_combo.blockSignals(False)


    def _populate_channel_picker(self):
        # Populate the channel list based on self.channels
        try:
            n = int(getattr(self, 'channels', 1))
        except Exception:
            n = 1
        self.channel_picker.clear()
        for ch in range(max(1, n)):
            self.channel_picker.addItem(f"Ch {ch+1}")
        # default selection
        if self.channel_picker.count() > 0:
            self.channel_picker.item(0).setSelected(True)

    def selected_channel_indices(self):
        # Return list of 0‑based channel indices, capped to 8 if 'All selected'
        items = self.channel_picker.selectedItems() if hasattr(self, 'channel_picker') else []
        idxs = [self.channel_picker.row(it) for it in items] if items else []
        mode = 'Selected channel'  # channel_mode widget removed; default to single channel
        if mode.startswith('All'):
            if not idxs:
                idxs = list(range(getattr(self, 'channels', 1)))
            return idxs[:8]
        # single channel mode
        return [idxs[0] if idxs else 0]



# ==== BEGIN CHANNEL SELECTION + FFT PATCH ====
try:
    from PyQt5 import QtWidgets, QtCore
    import numpy as np
    from scipy.signal import periodogram, welch, hilbert
except Exception:
    # If imports fail, skip the patch to avoid breaking import-time.
    pass
else:
    # Checkbox dialog
    class ChannelSelectorDialog(QtWidgets.QDialog):
        def __init__(self, parent, channel_names, mask, configs=None, hydrophone_curves=None):
            super().__init__(parent)
            self.setWindowTitle("Select Channels")
            self.setModal(True)
            self._rows = []
            hydrophone_curves = hydrophone_curves or {}
            configs = configs or []

            v = QtWidgets.QVBoxLayout(self)
            grid = QtWidgets.QGridLayout()
            grid.addWidget(QtWidgets.QLabel("Use"), 0, 0)
            grid.addWidget(QtWidgets.QLabel("Channel"), 0, 1)
            grid.addWidget(QtWidgets.QLabel("Name"), 0, 2)
            grid.addWidget(QtWidgets.QLabel("Type"), 0, 3)
            grid.addWidget(QtWidgets.QLabel("Hydrophone"), 0, 4)
            grid.addWidget(QtWidgets.QLabel("Distance (m)"), 0, 5)
            grid.addWidget(QtWidgets.QLabel("Depth (m)"), 0, 6)
            grid.addWidget(QtWidgets.QLabel("Scale"), 0, 7)

            for i, name in enumerate(channel_names):
                cfg = configs[i] if i < len(configs) else {}
                enabled = mask[i] if i < len(mask) else True

                enable_cb = QtWidgets.QCheckBox()
                enable_cb.setChecked(enabled)

                name_edit = QtWidgets.QLineEdit()
                prior_name = cfg.get("name") if isinstance(cfg, dict) else None
                name_edit.setText(prior_name or name)
                name_edit.setPlaceholderText(name)

                type_combo = QtWidgets.QComboBox()
                for lbl, val in [
                    ("Raw Voltage", "raw"),
                    ("Hydrophone", "hydrophone"),
                    ("Voltage Probe", "voltage_probe"),
                    ("Current Probe", "current_probe"),
                ]:
                    type_combo.addItem(lbl, val)
                if isinstance(cfg, dict) and cfg.get("mode"):
                    idx = max(0, type_combo.findData(cfg.get("mode")))
                    type_combo.setCurrentIndex(idx)

                hydro_combo = QtWidgets.QComboBox()
                hydro_combo.addItem("None", None)
                for curve in hydrophone_curves.values():
                    hydro_combo.addItem(curve.get("curve_name", ""), curve.get("curve_name"))
                if isinstance(cfg, dict) and cfg.get("hydrophone_curve"):
                    idx = hydro_combo.findData(cfg.get("hydrophone_curve"))
                    if idx >= 0:
                        hydro_combo.setCurrentIndex(idx)

                dist_edit = QtWidgets.QLineEdit()
                dist_val = cfg.get("distance") if isinstance(cfg, dict) else None
                dist_edit.setText("" if dist_val in (None, "") else str(dist_val))
                dist_edit.setPlaceholderText("Optional")

                depth_edit = QtWidgets.QLineEdit()
                depth_val = cfg.get("depth") if isinstance(cfg, dict) else None
                depth_edit.setText("" if depth_val in (None, "") else str(depth_val))
                depth_edit.setPlaceholderText("Optional")

                scale_edit = QtWidgets.QLineEdit()
                scale_val = cfg.get("scale") if isinstance(cfg, dict) else None
                scale_edit.setText("" if scale_val in (None, "") else str(scale_val))
                scale_edit.setPlaceholderText("e.g. 100 for 1:100")

                self._rows.append(
                    {
                        "enable": enable_cb,
                        "name": name_edit,
                        "type": type_combo,
                        "hydro": hydro_combo,
                        "dist": dist_edit,
                        "depth": depth_edit,
                        "scale": scale_edit,
                    }
                )

                grid.addWidget(enable_cb, i + 1, 0)
                grid.addWidget(QtWidgets.QLabel(name), i + 1, 1)
                grid.addWidget(name_edit, i + 1, 2)
                grid.addWidget(type_combo, i + 1, 3)
                grid.addWidget(hydro_combo, i + 1, 4)
                grid.addWidget(dist_edit, i + 1, 5)
                grid.addWidget(depth_edit, i + 1, 6)
                grid.addWidget(scale_edit, i + 1, 7)

                def _on_mode_change(idx, row=self._rows[-1]):
                    mode = row["type"].itemData(idx)
                    is_hydro = mode == "hydrophone"
                    is_probe = mode in ("voltage_probe", "current_probe")
                    row["hydro"].setEnabled(is_hydro)
                    row["dist"].setEnabled(is_hydro)
                    row["depth"].setEnabled(is_hydro)
                    row["scale"].setEnabled(is_probe)

                type_combo.currentIndexChanged.connect(_on_mode_change)
                _on_mode_change(type_combo.currentIndex())

            v.addLayout(grid)
            btns = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
            )
            btns.accepted.connect(self.accept)
            btns.rejected.connect(self.reject)
            v.addWidget(btns)

        def mask(self):
            return [row["enable"].isChecked() for row in self._rows]

        def names(self):
            names = []
            for idx, row in enumerate(self._rows):
                txt = row["name"].text().strip()
                names.append(txt if txt else f"Ch {idx+1}")
            return names

        def configs(self):
            out = []
            for row in self._rows:
                out.append(
                    {
                        "mode": row["type"].currentData(),
                        "hydrophone_curve": row["hydro"].currentData(),
                        "distance": row["dist"].text(),
                        "depth": row["depth"].text(),
                        "scale": row["scale"].text(),
                        "name": row["name"].text(),
                    }
                )
            return out

    # Only patch if MainWindow exists
    try:
        MW = MainWindow  # type: ignore[name-defined]
    except Exception:
        MW = None

    if MW is not None:
        # helpers
        def _ensure_channel_info(self):
            if getattr(self, 'full_data', None) is None:
                return
            # channel count
            if not hasattr(self, 'channels') or not self.channels:
                try:
                    self.channels = int(self.full_data.shape[1]) if getattr(self.full_data, 'ndim', 1) > 1 else 1
                except Exception:
                    self.channels = 1
            # names
            if not hasattr(self, 'channel_names') or len(getattr(self, 'channel_names', [])) != self.channels:
                self.channel_names = ["Ch %d" % (i+1) for i in range(self.channels)]
            # mask
            if not hasattr(self, 'channel_mask') or len(getattr(self, 'channel_mask', [])) != self.channels:
                self.channel_mask = [True] * self.channels
            if not hasattr(self, 'channel_configs') or len(getattr(self, 'channel_configs', [])) != self.channels:
                existing = getattr(self, 'channel_configs', []) or []
                merged = []
                for i in range(self.channels):
                    base = {"mode": "raw", "hydrophone_curve": None, "distance": None, "depth": None, "scale": 1.0, "name": None}
                    if i < len(existing) and isinstance(existing[i], dict):
                        base.update({k: v for k, v in existing[i].items() if v is not None})
                    merged.append(base)
                self.channel_configs = merged

            # Propagate configured names back to channel_names
            names = []
            for i in range(self.channels):
                cfg = self.channel_configs[i] if i < len(self.channel_configs) else {}
                nm = cfg.get('name') if isinstance(cfg, dict) else None
                names.append(nm if nm else "Ch %d" % (i+1))
            self.channel_names = names

        def selected_channel_indices(self):
            _ensure_channel_info(self)
            try:
                return [i for i, on in enumerate(getattr(self, 'channel_mask', [])) if on]
            except Exception:
                return list(range(getattr(self, 'channels', 1) or 1))

        def open_channel_selector(self):
            _ensure_channel_info(self)
            dlg = ChannelSelectorDialog(
                self,
                getattr(self, 'channel_names', []),
                getattr(self, 'channel_mask', []),
                getattr(self, 'channel_configs', []),
                getattr(self, 'hydrophone_curves', {}),
            )
            if dlg.exec_() == QtWidgets.QDialog.Accepted:
                self.channel_mask = dlg.mask()
                try:
                    self.channel_configs = dlg.configs()
                    self.channel_names = dlg.names()
                except Exception:
                    pass
                if hasattr(self, '_populate_spec_channel_combo'):
                    try:
                        self._populate_spec_channel_combo()
                    except Exception:
                        pass
                if hasattr(self, 'update_main_waveform_plot'):
                    try:
                        self.update_main_waveform_plot()
                    except Exception:
                        pass
                if hasattr(self, 'update_fft_plot'):
                    try:
                        self.update_fft_plot()
                    except Exception:
                        pass

        # attach helpers
        MW._ensure_channel_info = _ensure_channel_info
        MW.selected_channel_indices = selected_channel_indices
        MW.open_channel_selector = open_channel_selector

        # Replace update_fft_plot to honor selected channels
        # update_fft_plot now uses pyqtgraph (no patch needed)
# ==== END CHANNEL SELECTION + FFT PATCH ====



# --- Fallback: ensure ChannelSelectorDialog is defined at runtime ---
try:
    ChannelSelectorDialog
except NameError:
    from PyQt5 import QtWidgets
    class ChannelSelectorDialog(QtWidgets.QDialog):
        def __init__(self, parent, channel_names, mask):
            super().__init__(parent)
            self.setWindowTitle("Select Channels")
            self.setModal(True)
            self._checks = []
            v = QtWidgets.QVBoxLayout(self)
            grid = QtWidgets.QGridLayout()
            for i, name in enumerate(channel_names):
                cb = QtWidgets.QCheckBox(name)
                cb.setChecked(mask[i] if i < len(mask) else True)
                self._checks.append(cb)
                grid.addWidget(cb, i // 4, i % 4)
            v.addLayout(grid)
            btns = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
            )
            btns.accepted.connect(self.accept)
            btns.rejected.connect(self.reject)
            v.addWidget(btns)
        def mask(self):
            return [cb.isChecked() for cb in self._checks]

# ---------------------
# Main Execution
# ---------------------

if __name__ == "__main__":
    import traceback

    app = QApplication(sys.argv)

    # Catch any unhandled exceptions and show them in a message box
    def _excepthook(exc_type, exc_value, exc_tb):
        msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        try:
            from PyQt5.QtWidgets import QMessageBox
            box = QMessageBox()
            box.setWindowTitle("Startup Error")
            box.setText(str(exc_value))
            box.setDetailedText(msg)
            box.exec_()
        except Exception:
            print(msg)
        sys.exit(1)

    sys.excepthook = _excepthook

    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    # Load splash image
    pixmap = QPixmap("splash.png")
    pixmap = pixmap.scaled(800, 600, QtCore.Qt.KeepAspectRatioByExpanding,
                           QtCore.Qt.SmoothTransformation)
    splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
    splash.setMask(pixmap.mask())
    splash.setFixedSize(800, 600)
    splash.show()
    app.processEvents()

    QTimer.singleShot(3000, splash.close)

    # Create and show main window
    window = MainWindow()
    window.startup_license_check()
    QTimer.singleShot(1500, window.showMaximized)

    sys.exit(app.exec_())
