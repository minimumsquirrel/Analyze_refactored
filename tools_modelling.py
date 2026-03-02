#!/usr/bin/env python3
"""
Modelling & Plotting Tools — methods for MainWindow mixin
Auto-extracted from main application. Each method is a mixin for self.
"""
import os
import sys
import math
import csv
import json
import sqlite3
import tempfile
import shutil
import numpy as np
import numpy.fft as fft
import soundfile as sf
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import RectangleSelector, SpanSelector
from matplotlib.figure import Figure
import scipy.signal.windows as windows
from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, welch, wiener, hilbert, periodogram, find_peaks as sp_find_peaks
from scipy.fft import rfft, rfftfreq
import pandas as pd
import pywt
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QUrl, Qt, QTimer
from PyQt5.QtGui import QDesktopServices, QFont, QPixmap
from datetime import datetime, timezone


# ── Shared utilities (DB, helpers, base UI classes) ──────────────────────────
from shared import (
    DB_FILENAME, safe_filtfilt, safe_sosfiltfilt, multitaper_psd, bandpass_filter,
    lighten_color, load_help_text, load_hydrophone_curves, save_hydrophone_curve,
    import_hydrophone_curve_file, init_db, get_setting, set_setting,
    log_measurement, fetch_logs, fetch_archived_logs, archive_log_entry,
    unarchive_log_entry, log_spl_calculation, fetch_spl_calculations,
    fetch_spl_archived_calculations, update_spl_calculation, archive_spl_calculation,
    unarchive_spl_calculation, load_or_convert_model, TrimDialog, MplCanvas,
)
# ─────────────────────────────────────────────────────────────────────────────

class ModellingToolsMixin:
    """Mixin class providing all Modelling & Plotting Tools for self."""

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
            lat = QtWidgets.QLineEdit(""); lon = QtWidgets.QLineEdit("")
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
    

    def cable_loss_and_hydro_popup(self):
        """
        Cable Loss & Hydrophone Sensitivity popup.

        TX tab:
        - Cable preset + family + AWG → R_cable(f) with skin effect
        - Cable capacitance per meter (pF/m) → lumped C_total = length * C'/m
        - Transducer impedance from:
            * CSV (freq, Re, Im)
            * CSV (freq, |Z|, phase_deg)
            * CSV (freq, G, B) with selectable units (S/mS/µS), converted to Re/Im
            * Up to 5 user points of (f, R) for purely resistive Z
        - Cable model: series R_cable(f) + shunt C_total in parallel with Z_trans
            → plots |V_trans / V_amp| in dB vs frequency.

        RX tab:
        - Uses hydrophone sensitivity curves from DB (open-circuit sensitivity).
        - Cable preset + family + AWG + capacitance per meter → R_cable(f) + C_total.
        - DAQ/load resistance.
        - No hydrophone impedance modelling: treats hydro as ideal Thevenin source.
        - Cable model: series R_cable(f) + shunt C_total in parallel with R_load
            → plots system sensitivity at DAQ input.
        """
        from PyQt5 import QtWidgets
        import numpy as np
        import csv
        import math
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Cable Loss & Hydrophone Sensitivity")
        dlg.setStyleSheet("""
            QDialog {
                background-color: #19232D;
                color: white;
            }
            QLabel {
                color: white;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTableWidget {
                background-color: #101820;
                color: white;
                border: 1px solid #32414B;
            }
            QTabWidget::pane {
                border: 1px solid #32414B;
            }
            QTabBar::tab {
                background: #101820;
                color: white;
                padding: 4px 10px;
            }
            QTabBar::tab:selected {
                background: #32414B;
            }
            QPushButton {
                background-color: #1E88E5;
                color: white;
                border: 1px solid #90CAF9;
                border-radius: 3px;
                padding: 4px 12px;
            }
            QPushButton:hover {
                background-color: #42A5F5;
            }
            QPushButton:pressed {
                background-color: #1565C0;
            }
        """)
        dlg.resize(1000, 700)

        main_vbox = QtWidgets.QVBoxLayout(dlg)
        main_vbox.setContentsMargins(10, 10, 10, 10)
        main_vbox.setSpacing(8)

        tabs = QtWidgets.QTabWidget(dlg)
        main_vbox.addWidget(tabs, 1)

        # ------------------------------------------------------------------
        # Cable models & "library"
        # ------------------------------------------------------------------
        # DC resistance per meter (single conductor, copper) ~20°C
        awg_ohm_per_m = {
            10: 0.00328,
            12: 0.00521,
            14: 0.00829,
            16: 0.01317,
            18: 0.02095,
            20: 0.03328,
            22: 0.05296,
            24: 0.08406,
        }

        # Approximate capacitance per meter (pF/m) for different cable families.
        # These are ballpark values – tweak to match your real cable datasheets.
        cable_cap_pf_per_m = {
            "Generic coax": {
                10: 55.0,
                12: 60.0,
                14: 65.0,
                16: 75.0,
                18: 90.0,
                20: 95.0,
                22: 100.0,
                24: 110.0,
            },
            "Generic twisted pair": {
                10: 40.0,
                12: 45.0,
                14: 50.0,
                16: 55.0,
                18: 60.0,
                20: 65.0,
                22: 70.0,
                24: 75.0,
            },
            "Shielded twisted pair": {
                10: 50.0,
                12: 55.0,
                14: 60.0,
                16: 65.0,
                18: 70.0,
                20: 75.0,
                22: 80.0,
                24: 85.0,
            },
            "Generic multi-pair": {
                10: 60.0,
                12: 65.0,
                14: 70.0,
                16: 80.0,
                18: 90.0,
                20: 100.0,
                22: 110.0,
                24: 120.0,
            },
            # Example “real” families – customize these to your actual cable parts
            "M150 tow cable": {
                16: 85.0,
                18: 90.0,
                20: 95.0,
            },
            "Deck cable (multi-pair)": {
                16: 100.0,
                18: 110.0,
                20: 120.0,
            },
            "Oil-filled tow umbilical": {
                16: 70.0,
                18: 75.0,
                20: 80.0,
            },
        }

        def default_cap_pf_per_m(cable_family: str, awg: int):
            """
            Look up a default capacitance per meter (pF/m) for a given cable
            family and AWG. Returns None if unknown.
            """
            table = cable_cap_pf_per_m.get(cable_family, {})
            return table.get(awg, None)

        # Cable presets: tie together a cable "family", AWG, and typical pF/m
        cable_presets = {
            "Custom / manual": None,
            "M150 tow 18 AWG": {
                "family": "M150 tow cable",
                "awg": 18,
                "cap_pf_per_m": 90.0,
                "comment": "Typical M150 tow leg",
            },
            "Deck cable 18 AWG": {
                "family": "Deck cable (multi-pair)",
                "awg": 18,
                "cap_pf_per_m": 110.0,
                "comment": "Generic deck multi-pair",
            },
            "Oil-filled umbilical 18 AWG": {
                "family": "Oil-filled tow umbilical",
                "awg": 18,
                "cap_pf_per_m": 75.0,
                "comment": "Typical oil-filled tow umbilical",
            },
        }

        def cable_R_skin_vs_freq(awg_str, length_m, f_array,
                                rho=1.72e-8, mu=4 * math.pi * 1e-7):
            """
            Return (R_cable(f), R_dc, description_string).

            R_dc from AWG + length; R(f) flattens to R_dc up to f_skin,
            then scales as sqrt(f / f_skin) above that.
            """
            try:
                awg = int(str(awg_str).replace("AWG", "").strip())
                length = float(str(length_m))
            except Exception:
                return None, None, "Invalid AWG or length."

            if awg not in awg_ohm_per_m:
                return None, None, f"AWG {awg} not in lookup table."

            if length <= 0.0:
                return None, None, "Cable length must be > 0."

            r_per_m_dc = awg_ohm_per_m[awg]
            R_dc = 2.0 * length * r_per_m_dc  # round trip (out + return)

            # AWG diameter (inches) → radius (m)
            d_inch = 0.005 * (92.0 ** ((36.0 - awg) / 39.0))
            d_m = d_inch * 0.0254
            a = 0.5 * d_m

            # Frequency where skin depth ~ radius
            f_skin = rho / (math.pi * mu * a * a)

            f_array = np.asarray(f_array, dtype=float)
            R_f = np.full_like(f_array, R_dc, dtype=float)

            mask = f_array > f_skin
            if np.any(mask):
                R_f[mask] = R_dc * np.sqrt(f_array[mask] / f_skin)

            desc = (
                f"AWG {awg}, length {length:.1f} m → R_dc ≈ {R_dc:.3f} Ω, "
                f"f_skin ≈ {f_skin:.0f} Hz"
            )
            return R_f, R_dc, desc

        def cable_total_cap(length_text, cap_pF_per_m_text):
            """
            Compute total cable capacitance C_total in farads
            from length (m) and capacitance per meter (pF/m).
            """
            try:
                length = float(str(length_text))
                cap_pF_per_m = float(str(cap_pF_per_m_text))
            except Exception:
                return 0.0, ""
            if length <= 0.0 or cap_pF_per_m <= 0.0:
                return 0.0, ""
            C_total = length * cap_pF_per_m * 1e-12  # pF/m → F/m
            desc = f", C ≈ {C_total*1e9:.1f} nF total"
            return C_total, desc

        # ------------------------------------------------------------------
        # TX TAB
        # ------------------------------------------------------------------
        tx_tab = QtWidgets.QWidget()
        tabs.addTab(tx_tab, "Transmit / Projector")

        tx_vbox = QtWidgets.QVBoxLayout(tx_tab)
        tx_form = QtWidgets.QGridLayout()
        tx_vbox.setContentsMargins(6, 6, 6, 6)
        tx_vbox.setSpacing(6)
        tx_vbox.addLayout(tx_form)
        tx_form.setHorizontalSpacing(8)
        tx_form.setVerticalSpacing(4)

        r = 0
        # Cable preset row
        tx_form.addWidget(QtWidgets.QLabel("Cable preset:"), r, 0)
        tx_preset_cb = QtWidgets.QComboBox()
        tx_preset_cb.addItems(list(cable_presets.keys()))
        tx_preset_cb.setToolTip(
            "Select a predefined cable (family + AWG + typical capacitance),\n"
            "or 'Custom / manual' to configure manually."
        )
        tx_form.addWidget(tx_preset_cb, r, 1, 1, 3)

        # Cable family + AWG
        r += 1
        tx_form.addWidget(QtWidgets.QLabel("Cable family:"), r, 0)
        tx_type_cb = QtWidgets.QComboBox()
        tx_type_cb.addItems([
            "Generic coax",
            "Generic twisted pair",
            "Shielded twisted pair",
            "Generic multi-pair",
            "M150 tow cable",
            "Deck cable (multi-pair)",
            "Oil-filled tow umbilical",
        ])
        tx_type_cb.setToolTip(
            "Cable construction/family used to auto-fill typical capacitance per meter.\n"
            "Values are approximate – override with datasheet if available."
        )
        tx_form.addWidget(tx_type_cb, r, 1)

        tx_form.addWidget(QtWidgets.QLabel("Cable AWG:"), r, 2)
        tx_awg_cb = QtWidgets.QComboBox()
        for g in [10, 12, 14, 16, 18, 20, 22, 24]:
            tx_awg_cb.addItem(f"{g} AWG")
        tx_awg_cb.setCurrentText("18 AWG")
        tx_form.addWidget(tx_awg_cb, r, 3)

        # Length
        r += 1
        tx_form.addWidget(QtWidgets.QLabel("Cable length (m):"), r, 0)
        tx_len_edit = QtWidgets.QLineEdit("100")
        tx_len_edit.setFixedWidth(80)
        tx_form.addWidget(tx_len_edit, r, 1)

        # Capacitance per meter
        r += 1
        tx_form.addWidget(QtWidgets.QLabel("Cable capacitance (pF/m):"), r, 0)
        tx_c_per_m_edit = QtWidgets.QLineEdit("")
        tx_c_per_m_edit.setFixedWidth(80)
        tx_c_per_m_edit.setToolTip(
            "Capacitance per meter of the cable (pF/m).\n"
            "Auto-filled from cable family/AWG or preset; override with datasheet if known."
        )
        tx_form.addWidget(tx_c_per_m_edit, r, 1)

        # Manual cable R override
        r += 1
        tx_form.addWidget(QtWidgets.QLabel("Manual cable R override (Ω):"), r, 0)
        tx_R_override_edit = QtWidgets.QLineEdit("")
        tx_R_override_edit.setPlaceholderText("Leave blank to use AWG + skin effect")
        tx_form.addWidget(tx_R_override_edit, r, 1, 1, 3)

        # Frequency grid
        r += 1
        tx_form.addWidget(QtWidgets.QLabel("fmin (Hz):"), r, 0)
        tx_fmin_edit = QtWidgets.QLineEdit("100")
        tx_fmin_edit.setFixedWidth(80)
        tx_form.addWidget(tx_fmin_edit, r, 1)

        tx_form.addWidget(QtWidgets.QLabel("fmax (Hz):"), r, 2)
        tx_fmax_edit = QtWidgets.QLineEdit("5000")
        tx_fmax_edit.setFixedWidth(80)
        tx_form.addWidget(tx_fmax_edit, r, 3)

        r += 1
        tx_form.addWidget(QtWidgets.QLabel("N points:"), r, 0)
        tx_npts_edit = QtWidgets.QLineEdit("200")
        tx_npts_edit.setFixedWidth(80)
        tx_form.addWidget(tx_npts_edit, r, 1)

        # Impedance model selection
        r += 1
        tx_form.addWidget(QtWidgets.QLabel("Transducer impedance model:"), r, 0, 1, 4)

        r += 1
        tx_imp_csv_reim_rb = QtWidgets.QRadioButton("Impedance CSV (freq, Re, Im)")
        tx_form.addWidget(tx_imp_csv_reim_rb, r, 0, 1, 4)

        r += 1
        tx_imp_csv_magphase_rb = QtWidgets.QRadioButton("Impedance CSV (freq, |Z|, phase_deg)")
        tx_imp_csv_magphase_rb.setChecked(True)
        tx_form.addWidget(tx_imp_csv_magphase_rb, r, 0, 1, 4)

        r += 1
        tx_imp_csv_gb_rb = QtWidgets.QRadioButton("Admittance CSV (freq, G, B)")
        tx_form.addWidget(tx_imp_csv_gb_rb, r, 0, 1, 2)
        tx_form.addWidget(QtWidgets.QLabel("Units:"), r, 2)
        tx_gb_unit_cb = QtWidgets.QComboBox()
        tx_gb_unit_cb.addItems(["S", "mS", "µS"])
        tx_gb_unit_cb.setCurrentText("mS")
        tx_gb_unit_cb.setFixedWidth(70)
        tx_form.addWidget(tx_gb_unit_cb, r, 3)

        # CSV path
        r += 1
        tx_csv_path_edit = QtWidgets.QLineEdit()
        tx_csv_browse_btn = QtWidgets.QPushButton("Browse CSV")
        tx_csv_browse_btn.setFixedWidth(110)
        tx_form.addWidget(tx_csv_path_edit, r, 0, 1, 3)
        tx_form.addWidget(tx_csv_browse_btn, r, 3)

        # Simple R(f) model
        r += 1
        tx_simple_rb = QtWidgets.QRadioButton(
            "Use up to 5 (frequency, resistance) points (purely resistive)"
        )
        tx_form.addWidget(tx_simple_rb, r, 0, 1, 4)

        r += 1
        tx_form.addWidget(QtWidgets.QLabel("f (Hz)"), r, 1)
        tx_form.addWidget(QtWidgets.QLabel("R (Ω)"), r, 2)
        r += 1
        tx_f_edits = []
        tx_R_edits = []
        for i in range(5):
            tx_form.addWidget(QtWidgets.QLabel(f"Point {i+1}:"), r, 0)
            f_edit = QtWidgets.QLineEdit("")
            f_edit.setFixedWidth(80)
            R_edit = QtWidgets.QLineEdit("")
            R_edit.setFixedWidth(80)
            tx_form.addWidget(f_edit, r, 1)
            tx_form.addWidget(R_edit, r, 2)
            tx_f_edits.append(f_edit)
            tx_R_edits.append(R_edit)
            r += 1

        # TX figure
        tx_fig = Figure(facecolor="#19232D")
        tx_ax = tx_fig.add_subplot(111)
        tx_ax.set_facecolor("#19232D")
        tx_ax.tick_params(colors="white")
        for sp in tx_ax.spines.values():
            sp.set_edgecolor("white")
        tx_ax.set_xlabel("Frequency (Hz)", color="white")
        tx_ax.set_ylabel("Voltage loss (dB)", color="white")
        tx_ax.grid(True, ls="--", alpha=0.3, color="gray")
        tx_ax.set_title("Transducer Voltage Loss vs Frequency", color="white")

        tx_canvas = FigureCanvas(tx_fig)
        tx_vbox.addWidget(tx_canvas, 1)

        # ------------------------------------------------------------------
        # RX TAB
        # ------------------------------------------------------------------
        rx_tab = QtWidgets.QWidget()
        tabs.addTab(rx_tab, "Hydrophone / Receive")

        rx_vbox = QtWidgets.QVBoxLayout(rx_tab)
        rx_form = QtWidgets.QGridLayout()
        rx_vbox.setContentsMargins(6, 6, 6, 6)
        rx_vbox.setSpacing(6)
        rx_vbox.addLayout(rx_form)
        rx_form.setHorizontalSpacing(8)
        rx_form.setVerticalSpacing(4)

        r = 0
        # Cable preset row
        rx_form.addWidget(QtWidgets.QLabel("Cable preset:"), r, 0)
        rx_preset_cb = QtWidgets.QComboBox()
        rx_preset_cb.addItems(list(cable_presets.keys()))
        rx_preset_cb.setToolTip(
            "Select a predefined cable (family + AWG + typical capacitance),\n"
            "or 'Custom / manual' to configure manually."
        )
        rx_form.addWidget(rx_preset_cb, r, 1, 1, 3)

        # Cable family + AWG
        r += 1
        rx_form.addWidget(QtWidgets.QLabel("Cable family:"), r, 0)
        rx_type_cb = QtWidgets.QComboBox()
        rx_type_cb.addItems([
            "Generic coax",
            "Generic twisted pair",
            "Shielded twisted pair",
            "Generic multi-pair",
            "M150 tow cable",
            "Deck cable (multi-pair)",
            "Oil-filled tow umbilical",
        ])
        rx_type_cb.setToolTip(
            "Cable construction/family used to auto-fill typical capacitance per meter.\n"
            "Values are approximate – override with datasheet if available."
        )
        rx_form.addWidget(rx_type_cb, r, 1)

        rx_form.addWidget(QtWidgets.QLabel("Cable AWG:"), r, 2)
        rx_awg_cb = QtWidgets.QComboBox()
        for g in [10, 12, 14, 16, 18, 20, 22, 24]:
            rx_awg_cb.addItem(f"{g} AWG")
        rx_awg_cb.setCurrentText("18 AWG")
        rx_form.addWidget(rx_awg_cb, r, 3)

        # Length
        r += 1
        rx_form.addWidget(QtWidgets.QLabel("Cable length (m):"), r, 0)
        rx_len_edit = QtWidgets.QLineEdit("100")
        rx_len_edit.setFixedWidth(80)
        rx_form.addWidget(rx_len_edit, r, 1)

        # Capacitance per meter
        r += 1
        rx_form.addWidget(QtWidgets.QLabel("Cable capacitance (pF/m):"), r, 0)
        rx_c_per_m_edit = QtWidgets.QLineEdit("")
        rx_c_per_m_edit.setFixedWidth(80)
        rx_c_per_m_edit.setToolTip(
            "Capacitance per meter of the cable (pF/m).\n"
            "Auto-filled from cable family/AWG or preset; override with datasheet if known."
        )
        rx_form.addWidget(rx_c_per_m_edit, r, 1)

        # Manual cable R override
        r += 1
        rx_form.addWidget(QtWidgets.QLabel("Manual cable R override (Ω):"), r, 0)
        rx_R_override_edit = QtWidgets.QLineEdit("")
        rx_R_override_edit.setPlaceholderText("Leave blank to use AWG + skin effect")
        rx_form.addWidget(rx_R_override_edit, r, 1, 1, 3)

        # Load resistance
        r += 1
        rx_form.addWidget(QtWidgets.QLabel("DAQ / load resistance (Ω):"), r, 0)
        rx_load_edit = QtWidgets.QLineEdit("1000000")
        rx_load_edit.setFixedWidth(100)
        rx_form.addWidget(rx_load_edit, r, 1)

        # Hydrophone sensitivity curve from DB
        r += 1
        rx_form.addWidget(QtWidgets.QLabel("Hydrophone curve (sensitivity):"), r, 0)
        rx_curve_cb = QtWidgets.QComboBox()
        rx_form.addWidget(rx_curve_cb, r, 1, 1, 3)

        # RX figure
        rx_fig = Figure(facecolor="#19232D")
        rx_ax = rx_fig.add_subplot(111)
        rx_ax.set_facecolor("#19232D")
        rx_ax.tick_params(colors="white")
        for sp in rx_ax.spines.values():
            sp.set_edgecolor("white")
        rx_ax.set_xlabel("Frequency (Hz)", color="white")
        rx_ax.set_ylabel("Sensitivity (dB re 1 V/µPa)", color="white")
        rx_ax.grid(True, ls="--", alpha=0.3, color="gray")
        rx_ax.set_title("Hydrophone Sensitivity with Cable + Load", color="white")

        rx_canvas = FigureCanvas(rx_fig)
        rx_vbox.addWidget(rx_canvas, 1)

        # ------------------------------------------------------------------
        # Bottom buttons
        # ------------------------------------------------------------------
        bottom_hbox = QtWidgets.QHBoxLayout()
        bottom_hbox.addStretch(1)
        btn_calc_tx = QtWidgets.QPushButton("Calculate Projector")
        btn_calc_rx = QtWidgets.QPushButton("Calculate Hydrophone")
        btn_export = QtWidgets.QPushButton("Export CSV")
        btn_save = QtWidgets.QPushButton("Save Plot JPG")
        btn_close = QtWidgets.QPushButton("Close")
        bottom_hbox.addWidget(btn_calc_tx)
        bottom_hbox.addWidget(btn_calc_rx)
        bottom_hbox.addWidget(btn_export)
        bottom_hbox.addWidget(btn_save)
        bottom_hbox.addWidget(btn_close)
        main_vbox.addLayout(bottom_hbox)

        # ------------------------------------------------------------------
        # Load hydro curves from DB (for RX tab)
        # ------------------------------------------------------------------
        try:
            curves_dict = load_hydrophone_curves()
        except Exception:
            curves_dict = {}

        rx_curve_name_to_data = {}
        for cid, info in curves_dict.items():
            name = info.get("curve_name", f"Curve {cid}")
            min_f = float(info.get("min_freq", 0.0))
            max_f = float(info.get("max_freq", 0.0))
            sens_list = info.get("sensitivity", [])
            if sens_list and max_f > min_f:
                rx_curve_name_to_data[name] = (
                    min_f,
                    max_f,
                    np.array(sens_list, dtype=float),
                )

        rx_curve_cb.clear()
        if rx_curve_name_to_data:
            rx_curve_cb.addItems(sorted(rx_curve_name_to_data.keys()))
        else:
            rx_curve_cb.addItem("No curves found in DB")

        # Storage for CSV export
        last_tx_results = None
        last_rx_results = None

        # ------------------------------------------------------------------
        # File browsing for TX impedance CSV
        # ------------------------------------------------------------------
        def browse_tx_csv():
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                dlg, "Select Transducer Impedance CSV",
                "", "CSV Files (*.csv);;All Files (*)"
            )
            if path:
                tx_csv_path_edit.setText(path)

        tx_csv_browse_btn.clicked.connect(browse_tx_csv)

        # ------------------------------------------------------------------
        # Helper: parse up to 5 (f,R) points for TX simple model
        # ------------------------------------------------------------------
        def parse_tx_simple_points():
            freqs = []
            Rs = []
            for f_edit, R_edit in zip(tx_f_edits, tx_R_edits):
                f_txt = f_edit.text().strip()
                R_txt = R_edit.text().strip()
                if not f_txt and not R_txt:
                    continue
                try:
                    f_val = float(f_txt)
                    R_val = float(R_txt)
                    if f_val > 0.0 and R_val > 0.0:
                        freqs.append(f_val)
                        Rs.append(R_val)
                except Exception:
                    pass
            if not freqs:
                return None, None
            freqs = np.array(freqs, dtype=float)
            Rs = np.array(Rs, dtype=float)
            idx = np.argsort(freqs)
            return freqs[idx], Rs[idx]

        # ------------------------------------------------------------------
        # CSV readers for TX impedance
        # ------------------------------------------------------------------
        def read_impedance_csv_reim(path):
            f_data, Re_data, Im_data = [], [], []
            with open(path, "r", newline="") as fcsv:
                reader = csv.reader(fcsv)
                header = next(reader, None)  # ignore header
                for row in reader:
                    if len(row) < 3:
                        continue
                    try:
                        fi = float(row[0])
                        Rei = float(row[1])
                        Imi = float(row[2])
                    except Exception:
                        continue
                    f_data.append(fi)
                    Re_data.append(Rei)
                    Im_data.append(Imi)
            if not f_data:
                raise ValueError("No valid rows in Re/Im CSV.")
            return (
                np.array(f_data, dtype=float),
                np.array(Re_data, dtype=float),
                np.array(Im_data, dtype=float),
            )

        def read_impedance_csv_magphase(path):
            f_data, Zmag_data, phase_data = [], [], []
            with open(path, "r", newline="") as fcsv:
                reader = csv.reader(fcsv)
                header = next(reader, None)  # ignore header
                for row in reader:
                    if len(row) < 3:
                        continue
                    try:
                        fi = float(row[0])
                        Zi = float(row[1])      # |Z|
                        ph_deg = float(row[2])  # phase in degrees
                    except Exception:
                        continue
                    f_data.append(fi)
                    Zmag_data.append(Zi)
                    phase_data.append(ph_deg)
            if not f_data:
                raise ValueError("No valid rows in |Z|+phase CSV.")
            return (
                np.array(f_data, dtype=float),
                np.array(Zmag_data, dtype=float),
                np.array(phase_data, dtype=float),
            )

        def read_admittance_csv_gb(path, unit_scale):
            """
            Read admittance CSV: freq, G, B (in given unit),
            convert to impedance components R, X in ohms.

            unit_scale:
                1.0   → values already in siemens (S)
                1e-3  → values in mS
                1e-6  → values in µS
            """
            f_data, G_data, B_data = [], [], []
            with open(path, "r", newline="") as fcsv:
                reader = csv.reader(fcsv)
                header = next(reader, None)  # ignore header
                for row in reader:
                    if len(row) < 3:
                        continue
                    try:
                        fi = float(row[0])
                        Gi = float(row[1]) * unit_scale   # → S
                        Bi = float(row[2]) * unit_scale   # → S
                    except Exception:
                        continue
                    f_data.append(fi)
                    G_data.append(Gi)
                    B_data.append(Bi)

            if not f_data:
                raise ValueError("No valid rows in G/B CSV.")

            f_data = np.array(f_data, dtype=float)
            G_data = np.array(G_data, dtype=float)
            B_data = np.array(B_data, dtype=float)

            # Z = 1 / (G + jB) = (G - jB) / (G^2 + B^2)
            denom = G_data * G_data + B_data * B_data
            denom = np.where(denom <= 0.0, np.nan, denom)  # avoid div by zero

            R = G_data / denom
            X = -B_data / denom

            return f_data, R, X

        # ------------------------------------------------------------------
        # Auto-fill capacitance from family+AWG or presets
        # ------------------------------------------------------------------
        def update_tx_cap_default(force: bool = False):
            """
            Auto-fill TX capacitance (pF/m) based on cable family + AWG.

            Currently we always overwrite the field when called, regardless of
            the 'force' flag, so changing cable type or gauge always updates pF/m.
            """
            try:
                awg = int(tx_awg_cb.currentText().replace("AWG", "").strip())
            except Exception:
                return

            cable_family = tx_type_cb.currentText()
            cap_default = default_cap_pf_per_m(cable_family, awg)

            if cap_default is not None:
                tx_c_per_m_edit.setText(f"{cap_default:.1f}")



        def update_rx_cap_default(force: bool = False):
            """
            Auto-fill RX capacitance (pF/m) based on cable family + AWG.

            Always overwrites the field so that changing cable type or gauge
            updates pF/m every time.
            """
            try:
                awg = int(rx_awg_cb.currentText().replace("AWG", "").strip())
            except Exception:
                return

            cable_family = rx_type_cb.currentText()
            cap_default = default_cap_pf_per_m(cable_family, awg)

            if cap_default is not None:
                rx_c_per_m_edit.setText(f"{cap_default:.1f}")



        def apply_tx_preset(name: str):
            info = cable_presets.get(name)
            if not info:
                # "Custom / manual" – only fill from family+AWG if empty
                update_tx_cap_default(force=False)
                return
            family = info.get("family")
            awg = info.get("awg")
            cap = info.get("cap_pf_per_m")
            if family and tx_type_cb.findText(family) >= 0:
                tx_type_cb.setCurrentText(family)
            if awg is not None:
                tx_awg_cb.setCurrentText(f"{awg} AWG")
            if cap is not None:
                tx_c_per_m_edit.setText(f"{cap:.1f}")

        def apply_rx_preset(name: str):
            info = cable_presets.get(name)
            if not info:
                update_rx_cap_default(force=False)
                return
            family = info.get("family")
            awg = info.get("awg")
            cap = info.get("cap_pf_per_m")
            if family and rx_type_cb.findText(family) >= 0:
                rx_type_cb.setCurrentText(family)
            if awg is not None:
                rx_awg_cb.setCurrentText(f"{awg} AWG")
            if cap is not None:
                rx_c_per_m_edit.setText(f"{cap:.1f}")

        # Connect signals for auto-fill and presets
        # AWG changes: only fill if empty (don't stomp manual)
        tx_awg_cb.currentTextChanged.connect(lambda _: update_tx_cap_default(False))
        rx_awg_cb.currentTextChanged.connect(lambda _: update_rx_cap_default(False))

        # Cable family changes: always overwrite with that family's default
        tx_type_cb.currentTextChanged.connect(lambda _: update_tx_cap_default(True))
        rx_type_cb.currentTextChanged.connect(lambda _: update_rx_cap_default(True))

        tx_preset_cb.currentTextChanged.connect(apply_tx_preset)
        rx_preset_cb.currentTextChanged.connect(apply_rx_preset)

        # Initial pre-fill when dialog opens (use presets default, which is "Custom / manual")
        apply_tx_preset(tx_preset_cb.currentText())
        apply_rx_preset(rx_preset_cb.currentText())

        # ------------------------------------------------------------------
        # Calculate TX
        # ------------------------------------------------------------------
        def do_calc_tx():
            nonlocal last_tx_results

            # Frequency grid
            try:
                fmin = float(tx_fmin_edit.text())
                fmax = float(tx_fmax_edit.text())
                npts = int(tx_npts_edit.text())
            except Exception:
                QtWidgets.QMessageBox.critical(dlg, "Error", "Invalid frequency grid.")
                return

            if fmax <= fmin or npts < 2:
                QtWidgets.QMessageBox.critical(
                    dlg, "Error", "Need fmax>fmin and N>1."
                )
                return

            f = np.linspace(fmin, fmax, npts)

            # Total cable capacitance
            C_total, desc_c = cable_total_cap(tx_len_edit.text(), tx_c_per_m_edit.text())

            # Cable R(f): manual override or skin model
            manual_R_txt = tx_R_override_edit.text().strip()
            if manual_R_txt:
                try:
                    R_const = float(manual_R_txt)
                    if R_const < 0:
                        raise ValueError
                except Exception:
                    QtWidgets.QMessageBox.critical(
                        dlg, "Error", "Invalid manual cable R override."
                    )
                    return
                R_cable_f = np.full_like(f, R_const, dtype=float)
                desc_cable = f"Manual cable R = {R_const:.3f} Ω" + desc_c
            else:
                R_cable_f, R_dc, desc_base = cable_R_skin_vs_freq(
                    tx_awg_cb.currentText(),
                    tx_len_edit.text(),
                    f,
                )
                if R_cable_f is None:
                    QtWidgets.QMessageBox.critical(dlg, "Error", R_dc)
                    return
                desc_cable = desc_base + desc_c

            # Transducer impedance model
            if tx_simple_rb.isChecked():
                # Simple R(f) points
                f_pts, R_pts = parse_tx_simple_points()
                if f_pts is None:
                    QtWidgets.QMessageBox.critical(
                        dlg, "Error",
                        "No valid (f,R) points entered for simple impedance model."
                    )
                    return
                Re = np.interp(f, f_pts, R_pts,
                            left=R_pts[0], right=R_pts[-1])
                Im = np.zeros_like(Re)
            else:
                path = tx_csv_path_edit.text().strip()
                if not path:
                    QtWidgets.QMessageBox.critical(
                        dlg, "Error", "Please select an impedance CSV file."
                    )
                    return
                try:
                    if tx_imp_csv_reim_rb.isChecked():
                        f_data, Re_data, Im_data = read_impedance_csv_reim(path)
                        Re = np.interp(f, f_data, Re_data,
                                    left=Re_data[0], right=Re_data[-1])
                        Im = np.interp(f, f_data, Im_data,
                                    left=Im_data[0], right=Im_data[-1])

                    elif tx_imp_csv_magphase_rb.isChecked():
                        f_data, Zmag_data, phase_data = read_impedance_csv_magphase(path)
                        Zmag = np.interp(f, f_data, Zmag_data,
                                        left=Zmag_data[0], right=Zmag_data[-1])
                        phase_deg = np.interp(f, f_data, phase_data,
                                            left=phase_data[0], right=phase_data[-1])
                        phase_rad = np.deg2rad(phase_deg)
                        Re = Zmag * np.cos(phase_rad)
                        Im = Zmag * np.sin(phase_rad)

                    elif tx_imp_csv_gb_rb.isChecked():
                        unit_txt = tx_gb_unit_cb.currentText()
                        if unit_txt == "S":
                            scale = 1.0
                        elif unit_txt == "mS":
                            scale = 1e-3
                        else:
                            scale = 1e-6
                        f_data, R_data, X_data = read_admittance_csv_gb(path, scale)
                        Re = np.interp(f, f_data, R_data,
                                    left=R_data[0], right=R_data[-1])
                        Im = np.interp(f, f_data, X_data,
                                    left=X_data[0], right=X_data[-1])

                    else:
                        # fallback = treat as |Z|+phase CSV
                        f_data, Zmag_data, phase_data = read_impedance_csv_magphase(path)
                        Zmag = np.interp(f, f_data, Zmag_data,
                                        left=Zmag_data[0], right=Zmag_data[-1])
                        phase_deg = np.interp(f, f_data, phase_data,
                                            left=phase_data[0], right=phase_data[-1])
                        phase_rad = np.deg2rad(phase_deg)
                        Re = Zmag * np.cos(phase_rad)
                        Im = np.sin(phase_rad) * Zmag
                except Exception as e:
                    QtWidgets.QMessageBox.critical(
                        dlg, "Error", f"Failed to read impedance CSV:\n{e}"
                    )
                    return

            Zt = Re + 1j * Im

            # Cable capacitance model: Amp → R_cable → node → (Zt ∥ Zc) → ground
            omega = 2.0 * math.pi * f
            if C_total > 0.0:
                Zc = 1.0 / (1j * omega * C_total)
                Z_parallel = (Zt * Zc) / (Zt + Zc)
            else:
                Z_parallel = Zt

            Ztotal = R_cable_f + Z_parallel
            Vratio = np.abs(Z_parallel) / np.abs(Ztotal)
            loss_dB = 20.0 * np.log10(np.maximum(Vratio, 1e-12))

            tx_ax.clear()
            tx_ax.set_facecolor("#19232D")
            tx_ax.tick_params(colors="white")
            for sp in tx_ax.spines.values():
                sp.set_edgecolor("white")
            tx_ax.grid(True, ls="--", alpha=0.3, color="gray")
            tx_ax.plot(f, loss_dB, lw=1.8, color="#03DFE2", label="Voltage loss")
            tx_ax.set_xlabel("Frequency (Hz)", color="white")
            tx_ax.set_ylabel("Voltage loss (dB)", color="white")
            tx_ax.set_xscale("log")
            tx_ax.set_title(
                "Transducer Voltage Loss vs Frequency\n" + desc_cable,
                color="white",
            )
            tx_ax.legend(facecolor="#222222", edgecolor="#444444", labelcolor="white")
            tx_fig.tight_layout()
            tx_canvas.draw()

            last_tx_results = {
                "freq_Hz": f,
                "loss_dB": loss_dB,
                "R_cable_f": R_cable_f,
                "C_total": C_total,
                "desc_cable": desc_cable,
            }

        btn_calc_tx.clicked.connect(do_calc_tx)

        # ------------------------------------------------------------------
        # Calculate RX (no hydro impedance, just cable RC + load)
        # ------------------------------------------------------------------
        def do_calc_rx():
            nonlocal last_rx_results

            if not rx_curve_name_to_data:
                QtWidgets.QMessageBox.critical(
                    dlg, "Error", "No hydrophone curves found in the database."
                )
                return

            # Load resistance
            try:
                R_load = float(rx_load_edit.text())
            except Exception:
                QtWidgets.QMessageBox.critical(
                    dlg, "Error", "Invalid DAQ/load resistance."
                )
                return
            if R_load <= 0.0:
                QtWidgets.QMessageBox.critical(
                    dlg, "Error", "Load resistance must be > 0."
                )
                return

            # Hydrophone sensitivity curve
            name = rx_curve_cb.currentText()
            if name not in rx_curve_name_to_data:
                QtWidgets.QMessageBox.critical(
                    dlg, "Error", "Selected hydrophone curve not found in DB."
                )
                return

            fmin, fmax, sens_arr = rx_curve_name_to_data[name]
            N = sens_arr.size
            if N < 2 or fmax <= fmin:
                QtWidgets.QMessageBox.critical(
                    dlg, "Error",
                    "Hydrophone curve has invalid min/max or too few points."
                )
                return

            f = np.linspace(fmin, fmax, N)

            # Cable capacitance (total)
            C_total, desc_c = cable_total_cap(rx_len_edit.text(), rx_c_per_m_edit.text())

            # Cable R(f)
            manual_R_txt = rx_R_override_edit.text().strip()
            if manual_R_txt:
                try:
                    R_const = float(manual_R_txt)
                    if R_const < 0:
                        raise ValueError
                except Exception:
                    QtWidgets.QMessageBox.critical(
                        dlg, "Error", "Invalid manual cable R override."
                    )
                    return
                R_cable_f = np.full_like(f, R_const, dtype=float)
                desc_cable = f"Manual cable R = {R_const:.3f} Ω" + desc_c
            else:
                R_cable_f, R_dc, desc_base = cable_R_skin_vs_freq(
                    rx_awg_cb.currentText(),
                    rx_len_edit.text(),
                    f,
                )
                if R_cable_f is None:
                    QtWidgets.QMessageBox.critical(dlg, "Error", R_dc)
                    return
                desc_cable = desc_base + desc_c

            # RC model: Vs → series R_cable(f) → node → parallel(Z_load, Zc_total)
            omega = 2.0 * math.pi * f
            if C_total > 0.0:
                Zc = 1.0 / (1j * omega * C_total)
                Z_load = R_load + 0j
                Z_par = (Z_load * Zc) / (Z_load + Zc)
            else:
                Z_par = R_load + 0j

            Ztotal = R_cable_f + Z_par
            Vratio = np.abs(Z_par) / np.abs(Ztotal)

            sens_open_dB = sens_arr
            delta_dB = 20.0 * np.log10(np.maximum(Vratio, 1e-12))
            sens_system_dB = sens_open_dB + delta_dB

            rx_ax.clear()
            rx_ax.set_facecolor("#19232D")
            rx_ax.tick_params(colors="white")
            for sp in rx_ax.spines.values():
                sp.set_edgecolor("white")
            rx_ax.grid(True, ls="--", alpha=0.3, color="gray")
            rx_ax.plot(
                f, sens_open_dB, lw=1.5, color="#C8B6FF",
                label="Hydrophone (open-circuit)",
            )
            rx_ax.plot(
                f, sens_system_dB, lw=1.8, color="#03DFE2",
                label=f"System (cable + {R_load:.1f} Ω load)",
            )
            rx_ax.set_xlabel("Frequency (Hz)", color="white")
            rx_ax.set_ylabel("Sensitivity (dB re 1 V/µPa)", color="white")
            rx_ax.set_xscale("log")
            rx_ax.set_title(
                f"{name} – Predicted Sensitivity with Cable + Load\n"
                f"{desc_cable}, R_load = {R_load:.1f} Ω",
                color="white",
            )
            rx_ax.legend(facecolor="#222222", edgecolor="#444444", labelcolor="white")
            rx_fig.tight_layout()
            rx_canvas.draw()

            last_rx_results = {
                "freq_Hz": f,
                "sens_open_dB": sens_open_dB,
                "sens_system_dB": sens_system_dB,
                "R_cable_f": R_cable_f,
                "C_total": C_total,
                "R_load": R_load,
                "curve_name": name,
                "cable_desc": desc_cable,
            }

        btn_calc_rx.clicked.connect(do_calc_rx)

        # ------------------------------------------------------------------
        # Export CSV (active tab)
        # ------------------------------------------------------------------
        def do_export_csv():
            idx = tabs.currentIndex()
            if idx == 0:
                if not last_tx_results:
                    QtWidgets.QMessageBox.information(
                        dlg, "No data",
                        "Please run a projector calculation first."
                    )
                    return
                path, _ = QtWidgets.QFileDialog.getSaveFileName(
                    dlg, "Export Projector CSV", "", "CSV Files (*.csv)"
                )
                if not path:
                    return
                try:
                    with open(path, "w", newline="") as fcsv:
                        writer = csv.writer(fcsv)
                        writer.writerow(
                            ["freq_Hz", "loss_dB", "R_cable_ohm", "C_total_F", "cable_desc"]
                        )
                        f = last_tx_results["freq_Hz"]
                        loss = last_tx_results["loss_dB"]
                        Rf = last_tx_results["R_cable_f"]
                        C_total = last_tx_results["C_total"]
                        desc = last_tx_results["desc_cable"]
                        for fi, li, Ri in zip(f, loss, Rf):
                            writer.writerow([fi, li, Ri, C_total, desc])
                except Exception as e:
                    QtWidgets.QMessageBox.critical(
                        dlg, "Error", f"Failed to export CSV:\n{e}"
                    )
                    return
                QtWidgets.QMessageBox.information(
                    dlg, "Exported", f"Projector data saved to:\n{path}"
                )
            else:
                if not last_rx_results:
                    QtWidgets.QMessageBox.information(
                        dlg, "No data",
                        "Please run a hydrophone calculation first."
                    )
                    return
                path, _ = QtWidgets.QFileDialog.getSaveFileName(
                    dlg, "Export Hydrophone CSV", "", "CSV Files (*.csv)"
                )
                if not path:
                    return
                try:
                    with open(path, "w", newline="") as fcsv:
                        writer = csv.writer(fcsv)
                        writer.writerow([
                            "freq_Hz",
                            "sens_open_dB",
                            "sens_system_dB",
                            "R_cable_ohm",
                            "C_total_F",
                            "R_load_ohm",
                            "curve_name",
                            "cable_desc",
                        ])
                        f = last_rx_results["freq_Hz"]
                        so = last_rx_results["sens_open_dB"]
                        ss = last_rx_results["sens_system_dB"]
                        Rf = last_rx_results["R_cable_f"]
                        C_total = last_rx_results["C_total"]
                        Rl = last_rx_results["R_load"]
                        name = last_rx_results["curve_name"]
                        cdesc = last_rx_results["cable_desc"]
                        for fi, soi, ssi, Ri in zip(f, so, ss, Rf):
                            writer.writerow([fi, soi, ssi, Ri, C_total, Rl, name, cdesc])
                except Exception as e:
                    QtWidgets.QMessageBox.critical(
                        dlg, "Error", f"Failed to export CSV:\n{e}"
                    )
                    return
                QtWidgets.QMessageBox.information(
                    dlg, "Exported", f"Hydrophone data saved to:\n{path}"
                )

        btn_export.clicked.connect(do_export_csv)

        # ------------------------------------------------------------------
        # Save Plot JPG (active tab)
        # ------------------------------------------------------------------
        def do_save_plot():
            idx = tabs.currentIndex()
            fig = tx_fig if idx == 0 else rx_fig
            label = "Projector" if idx == 0 else "Hydrophone"
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                dlg, f"Save {label} Plot", "", "JPEG Files (*.jpg *.jpeg)"
            )
            if not path:
                return
            try:
                fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    dlg, "Error", f"Failed to save plot:\n{e}"
                )
                return
            QtWidgets.QMessageBox.information(
                dlg, "Saved", f"{label} plot saved to:\n{path}"
            )

        btn_save.clicked.connect(do_save_plot)
        btn_close.clicked.connect(dlg.accept)
        dlg.exec_()






    def propagation_from_spl_db_popup(self):
        """
        Propagation modeling from SPL (DB) with optional CTD.

        Includes:
        - RL vs Frequency plot fix: if only 1 unique frequency, plot #2 becomes RL vs Range (single-freq mode)
        - Echo detect threshold (dB) in Config: marks where echo drops below threshold on echo curve
        - CTD profile viewer:
            * colors follow user's graph-color palette
            * thermocline band start/end (optional shade + dashed lines)
            * CTD JPG export (dark / light / B&W) with distinguishable BW legend/markers
        - Main graphs JPG export (dark / light / B&W)
        - Plain-number tick labels on log x-axes (1000 not 10^3)
        - Clickable lines/markers show x/y value (pick_event)

        CTD schema expected (yours):
        ctd_profiles(depth_json, temp_json, sal_json, sound_speed_json)
        """
        from PyQt5 import QtWidgets, QtCore
        import os, json, ast, sqlite3, numpy as np
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.ticker import LogLocator, FuncFormatter, ScalarFormatter

        # ------------------------------------------------------------
        # DB path
        # ------------------------------------------------------------
        def _db_path():
            for name in ("DB_FILENAME", "DB_PATH", "DATABASE_FILENAME", "DATABASE_PATH"):
                if hasattr(self, name):
                    p = getattr(self, name)
                    if isinstance(p, str) and p.strip():
                        return p.strip()
            try:
                return DB_FILENAME  # noqa: F821
            except Exception:
                pass
            try:
                from analyze_qt import DB_FILENAME as OLD
                return OLD
            except Exception:
                return os.path.join(os.path.abspath(os.getcwd()), "analyze_qt.db")

        # ------------------------------------------------------------
        # Colors / palette
        # ------------------------------------------------------------
        def _sel_color():
            try:
                c = getattr(self, "graph_color", None)
                if isinstance(c, str) and c.strip():
                    return c.strip()
            except Exception:
                pass
            return "#03DFE2"

        def _get_user_color_list():
            colors = []
            cb = None
            for name in ("graph_color_cb", "graph_color_combo", "graph_color_dropdown", "graph_color_combo_box"):
                if hasattr(self, name):
                    cb = getattr(self, name)
                    break
            if cb is not None and hasattr(cb, "count") and hasattr(cb, "itemText"):
                try:
                    for i in range(cb.count()):
                        t = str(cb.itemText(i)).strip()
                        if hasattr(cb, "itemData"):
                            d = cb.itemData(i)
                            if isinstance(d, str) and d.strip().startswith("#"):
                                colors.append(d.strip())
                                continue
                        if t.startswith("#") and len(t) >= 7:
                            colors.append(t)
                except Exception:
                    pass
            if not colors and hasattr(self, "graph_colors") and isinstance(self.graph_colors, (list, tuple)):
                for c in self.graph_colors:
                    if isinstance(c, str) and c.strip().startswith("#"):
                        colors.append(c.strip())
            if not colors:
                colors = ["#03DFE2", "#A855F7", "#22C55E", "#F97316", "#EF4444", "#60A5FA", "#EAB308"]
            out, seen = [], set()
            for c in colors:
                if c not in seen:
                    out.append(c); seen.add(c)
            return out

        def _palette_from_user_selected(n_needed=12):
            allc = _get_user_color_list()
            sel = _sel_color()
            start = allc.index(sel) if sel in allc else 0
            return [allc[(start + k) % len(allc)] for k in range(max(1, n_needed))]

        # ------------------------------------------------------------
        # Axes styling + tick formatting
        # ------------------------------------------------------------
        def _style_axes(ax, dark=True):
            if dark:
                ax.set_facecolor("#1e1e1e")
                ax.tick_params(colors="white")
                for sp in ax.spines.values():
                    sp.set_color("#777")
                ax.title.set_color("white")
                ax.xaxis.label.set_color("white")
                ax.yaxis.label.set_color("white")
            else:
                ax.set_facecolor("white")
                ax.tick_params(colors="black")
                for sp in ax.spines.values():
                    sp.set_color("black")
                ax.title.set_color("black")
                ax.xaxis.label.set_color("black")
                ax.yaxis.label.set_color("black")

        def _force_plain_x(ax, log=False):
            if log:
                ax.set_xscale("log")
                ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
                ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1, 10) * 0.1))

                def fmt(x, pos=None):
                    if x is None or not np.isfinite(x) or x <= 0:
                        return ""
                    if x >= 1:
                        return f"{int(round(x))}"
                    return f"{x:g}"

                ax.xaxis.set_major_formatter(FuncFormatter(fmt))
            else:
                sf = ScalarFormatter(useOffset=False)
                sf.set_scientific(False)
                ax.xaxis.set_major_formatter(sf)

            try:
                ax.xaxis.get_offset_text().set_visible(False)
            except Exception:
                pass

        # ------------------------------------------------------------
        # Data access
        # ------------------------------------------------------------
        def _list_methods():
            q = "SELECT DISTINCT method FROM measurements WHERE method IS NOT NULL AND TRIM(method)!='' ORDER BY method"
            try:
                conn = sqlite3.connect(_db_path()); cur = conn.cursor()
                cur.execute(q)
                out = [r[0] for r in cur.fetchall()]
                conn.close()
                return out
            except Exception:
                return []

        def _list_files(method_filter=None):
            try:
                conn = sqlite3.connect(_db_path()); cur = conn.cursor()
                if method_filter and method_filter != "<All>":
                    cur.execute("""
                        SELECT sc.file_name, COUNT(*), MAX(COALESCE(sc.timestamp, sc.id))
                        FROM spl_calculations sc
                        LEFT JOIN measurements m ON sc.voltage_log_id = m.id
                        WHERE m.method = ?
                        GROUP BY sc.file_name
                        ORDER BY MAX(COALESCE(sc.timestamp, sc.id)) DESC
                    """, (method_filter,))
                else:
                    cur.execute("""
                        SELECT sc.file_name, COUNT(*), MAX(COALESCE(sc.timestamp, sc.id))
                        FROM spl_calculations sc
                        LEFT JOIN measurements m ON sc.voltage_log_id = m.id
                        GROUP BY sc.file_name
                        ORDER BY MAX(COALESCE(sc.timestamp, sc.id)) DESC
                    """)
                rows = cur.fetchall()
                conn.close()
                return rows
            except Exception:
                return []

        def _load_points_for_file(file_name, method_filter=None):
            pts = []
            if not file_name:
                return pts
            try:
                conn = sqlite3.connect(_db_path()); cur = conn.cursor()
                if method_filter and method_filter != "<All>":
                    cur.execute("""
                        SELECT sc.id, sc.target_frequency, sc.spl, sc.distance, m.method,
                            sc.start_time, sc.end_time
                        FROM spl_calculations sc
                        LEFT JOIN measurements m ON sc.voltage_log_id = m.id
                        WHERE sc.file_name=? AND m.method=?
                        ORDER BY sc.start_time IS NULL, sc.start_time, sc.id
                    """, (file_name, method_filter))
                else:
                    cur.execute("""
                        SELECT sc.id, sc.target_frequency, sc.spl, sc.distance, m.method,
                            sc.start_time, sc.end_time
                        FROM spl_calculations sc
                        LEFT JOIN measurements m ON sc.voltage_log_id = m.id
                        WHERE sc.file_name=?
                        ORDER BY sc.start_time IS NULL, sc.start_time, sc.id
                    """, (file_name,))
                rows = cur.fetchall()
                conn.close()
            except Exception:
                rows = []

            for (rid, f, spl, dist, method, t0, t1) in rows:
                if f is None or spl is None:
                    continue
                if t0 is not None and t1 is not None:
                    try:
                        t_mid = 0.5*(float(t0) + float(t1))
                    except Exception:
                        t_mid = None
                else:
                    t_mid = None

                pts.append({
                    "row_id": int(rid),
                    "f_Hz": float(f),
                    "SPL_dB": float(spl),
                    "r_meas_m": None if dist is None else float(dist),
                    "method": (method or ""),
                    "t_mid_s": float(t_mid) if t_mid is not None else 0.0,
                })

            if pts and all(p["t_mid_s"] == 0.0 for p in pts):
                for i, p in enumerate(pts):
                    p["t_mid_s"] = float(i)
            return pts

        # ------------------------------------------------------------
        # CTD schema: separate *_json columns (your table)
        # ------------------------------------------------------------
        def _ctd_list():
            rows = []
            try:
                conn = sqlite3.connect(_db_path()); cur = conn.cursor()
                cur.execute("""
                    SELECT id, name, dt_utc, latitude, longitude
                    FROM ctd_profiles
                    ORDER BY dt_utc DESC, id DESC
                """)
                for r in cur.fetchall():
                    rows.append({
                        "id": r[0],
                        "name": r[1] or f"CTD #{r[0]}",
                        "dt": r[2] or "",
                        "lat": r[3],
                        "lon": r[4],
                    })
                conn.close()
            except Exception:
                pass
            return rows

        def _load_ctd_profile(ctd_id):
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
                    if a.ndim == 2 and a.shape[1] >= 1:
                        try:
                            a = np.asarray(a[:, 0], dtype=float)
                        except Exception:
                            return None
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

            if T is not None:
                n = min(D.size, T.size)
                D = D[:n]; T = T[:n]
                if S is not None and S.size >= n:
                    S = S[:n]
                else:
                    S = None
                if C is not None and C.size >= n:
                    C = C[:n]
                else:
                    C = None

                idx = np.argsort(D)
                D = D[idx]; T = T[idx]
                if S is not None: S = S[idx]
                if C is not None: C = C[idx]

            return {"depth_m": D, "temp_C": T, "sal_ppt": S, "pH": None, "c_ms": C}

        # ------------------------------------------------------------
        # Physics / models
        # ------------------------------------------------------------
        def _mackenzie_c_ms(T_C, S_ppt, D_m):
            T = np.asarray(T_C, float)
            S = np.asarray(S_ppt, float)
            D = np.asarray(D_m, float)
            return (1448.96 + 4.591*T - 5.304e-2*T**2 + 2.374e-4*T**3
                    + 1.340*(S-35) + 1.630e-2*D + 1.675e-7*D**2
                    - 1.025e-2*T*(S-35) - 7.139e-13*T*D**3)

        def thorp_alpha_dB_per_km(f_Hz):
            fk = np.asarray(f_Hz, float)/1000.0
            fk2 = fk*fk
            return 0.11*fk2/(1.0+fk2) + 44.0*fk2/(4100.0+fk2) + 2.75e-4*fk2 + 0.003

        def fg_alpha_dB_per_km(f_Hz, depth_m, T_C, S_ppt, pH_val=8.0):
            f = np.asarray(f_Hz, float)/1000.0
            T = float(T_C); S = float(S_ppt); D = float(depth_m); pH = float(pH_val)
            c = float(_mackenzie_c_ms(T, S, D))
            A1 = (8.86/c) * 10.0**(0.78*pH - 5.0)
            f1 = 2.8 * np.sqrt(S/35.0) * 10.0**(4.0 - 1245.0/(T+273.0))
            A2 = 21.44 * (S/c) * (1.0 + 0.025*T)
            f2 = (8.17 * 10.0**(8.0 - 1990.0/(T+273.0))) / (1.0 + 0.0018*(S-35.0))
            P2 = 1.0 - 1.37e-4*D + 6.2e-9*D*D
            A3 = 4.937e-4 - 2.59e-5*T + 9.11e-7*T*T - 1.5e-8*T*T*T
            P3 = 1.0 - 3.83e-5*D + 4.9e-10*D*D
            term1 = A1 * (f1 * f*f) / (f1*f1 + f*f)
            term2 = A2 * (f2 * f*f) / (f2*f2 + f*f) * P2
            term3 = A3 * f*f * P3
            return term1 + term2 + term3

        def spreading_TL_dB(r_ratio, mode, r0_ratio=1000.0):
            r = np.maximum(1e-12, np.asarray(r_ratio, float))
            if mode.startswith("Spherical"):
                return 20.0*np.log10(r)
            if mode.startswith("Cylindrical"):
                return 10.0*np.log10(r)
            r0 = max(1.0, float(r0_ratio))
            return np.where(r <= r0,
                            20.0*np.log10(r),
                            20.0*np.log10(r0) + 10.0*np.log10(r/r0))

        SEABED_DBPKM = {
            "None": 0.0, "Very soft mud": 0.7, "Mud (soft silt)": 0.5, "Silt": 0.6,
            "Clay": 0.55, "Fine sand": 0.8, "Medium sand": 1.0, "Coarse sand": 1.2,
            "Shell hash / bioclastic sand": 1.3, "Sandy gravel": 1.5, "Gravel / pebble": 2.0,
            "Cobbles / boulder field": 2.2, "Bedrock (smooth)": 0.2, "Bedrock (rough/karst)": 0.4,
            "User (custom)": 0.0,
        }
        SURFACE_DBPKM = {
            "None": 0.0, "Calm (SS 0–1)": 0.05, "Moderate (SS 2–3)": 0.10,
            "Rough (SS 4–5)": 0.30, "Very rough (SS ≥6)": 0.50,
            "User (custom)": 0.0
        }

        def _recommend_depths_from_ctd(ctd_prof):
            if not ctd_prof:
                return None

            D = np.asarray(ctd_prof.get("depth_m", None), float)
            T = ctd_prof.get("temp_C", None)
            if T is None:
                return None
            T = np.asarray(T, float)

            S = ctd_prof.get("sal_ppt", None)
            if S is None:
                S = 35.0*np.ones_like(D)
            else:
                S = np.asarray(S, float)

            ok = np.isfinite(D) & np.isfinite(T) & np.isfinite(S)
            D = D[ok]; T = T[ok]; S = S[ok]
            if D.size < 3:
                return None

            idx = np.argsort(D)
            D = D[idx]; T = T[idx]; S = S[idx]

            c_in = ctd_prof.get("c_ms", None)
            if c_in is not None:
                c_in = np.asarray(c_in, float)
                if c_in.size == ok.size:
                    c_in = c_in[ok]; c_in = c_in[idx]
                elif c_in.size == D.size:
                    c_in = c_in[idx]
                else:
                    c_in = None

            if c_in is not None and np.all(np.isfinite(c_in)) and c_in.size == D.size:
                c = c_in
            else:
                c = _mackenzie_c_ms(T, S, D)

            dc_dz = np.gradient(c, D)
            abs_grad = np.abs(dc_dz)

            i_min_c = int(np.argmin(c))
            i_min_g = int(np.argmin(abs_grad))
            depth_min_c = float(D[i_min_c])
            depth_min_grad = float(D[i_min_g])

            g_span = float(np.ptp(abs_grad) + 1e-12)
            d_span = float(np.ptp(D) + 1e-12)
            g_norm = (abs_grad - np.min(abs_grad)) / g_span
            dist_norm = np.abs(D - depth_min_c) / d_span

            score = g_norm + 0.35*dist_norm

            shallow_cut = float(np.min(D) + 0.02*d_span)
            score = np.where(D < shallow_cut, score + 1.0, score)

            i_best = int(np.argmin(score))
            return {
                "depth_ideal": float(D[i_best]),
                "depth_min_c": depth_min_c,
                "depth_min_grad": depth_min_grad
            }

        def _thermocline_band(D, T):
            D = np.asarray(D, float)
            T = np.asarray(T, float)
            ok = np.isfinite(D) & np.isfinite(T)
            D = D[ok]; T = T[ok]
            if D.size < 5:
                return None

            idx = np.argsort(D)
            D = D[idx]; T = T[idx]

            dTdz = np.gradient(T, D)  # degC/m
            g = np.abs(dTdz)

            p90 = float(np.percentile(g, 90))
            thr = max(0.02, 0.5*p90)

            mask = g >= thr
            if not np.any(mask):
                return None

            inds = np.where(mask)[0]
            splits = np.where(np.diff(inds) > 1)[0]
            groups = np.split(inds, splits + 1)
            best = max(groups, key=len)
            z0 = float(D[best[0]])
            z1 = float(D[best[-1]])
            if (z1 - z0) < 1.0:
                return None
            return z0, z1, thr

        # ------------------------------------------------------------
        # Config state
        # ------------------------------------------------------------
        cfg = {
            "r_min": 1.0,
            "r_max": 50000.0,
            "r_N": 400,
            "spreading": "Spherical (20logR)",
            "r0_abs_m": 1000.0,
            "absorption": "Thorp",
            "depth_for_absorp_m": 50.0,
            "surface_key": "None",
            "surface_user_dbpkm": 0.0,
            "seabed_key": "None",
            "seabed_user_dbpkm": 0.0,

            "echo_enabled": False,
            "echo_loss_db": 6.0,
            "echo_detect_threshold_enabled": True,
            "echo_detect_threshold_db": 75.0,

            "selected_row_ids": [],
            "thermocline_shade": True,
            "thermocline_lines": True,
        }

        current_points = []
        last_results = None

        # ------------------------------------------------------------
        # Config dialog
        # ------------------------------------------------------------
        class PropagationConfigDialog(QtWidgets.QDialog):
            def __init__(self, parent):
                super().__init__(parent)
                self.setWindowTitle("Propagation Configuration")
                self.setModal(True)
                self.resize(980, 680)

                v = QtWidgets.QVBoxLayout(self)
                tabs = QtWidgets.QTabWidget()
                v.addWidget(tabs)

                # Propagation tab
                t1 = QtWidgets.QWidget()
                f1 = QtWidgets.QFormLayout(t1)
                self.rmin = QtWidgets.QLineEdit(str(cfg["r_min"]))
                self.rmax = QtWidgets.QLineEdit(str(cfg["r_max"]))
                self.rN = QtWidgets.QLineEdit(str(cfg["r_N"]))

                self.spread = QtWidgets.QComboBox()
                self.spread.addItems(["Spherical (20logR)", "Cylindrical (10logR)", "Hybrid (sph->cyl)"])
                self.spread.setCurrentText(cfg["spreading"])

                self.r0 = QtWidgets.QLineEdit(str(cfg["r0_abs_m"]))

                self.absorp = QtWidgets.QComboBox()
                self.absorp.addItems(["None", "Thorp", "Francois-Garrison (CTD-aware)"])
                self.absorp.setCurrentText(cfg["absorption"])

                self.depth = QtWidgets.QLineEdit(str(cfg["depth_for_absorp_m"]))

                f1.addRow("Range min (m)", self.rmin)
                f1.addRow("Range max (m)", self.rmax)
                f1.addRow("Range points (N)", self.rN)
                f1.addRow("Spreading model", self.spread)
                f1.addRow("Hybrid r0 (m)", self.r0)
                f1.addRow("Absorption model", self.absorp)
                f1.addRow("Depth for absorption (m)", self.depth)
                tabs.addTab(t1, "Propagation")

                # Environment tab
                t2 = QtWidgets.QWidget()
                f2 = QtWidgets.QFormLayout(t2)
                self.surface = QtWidgets.QComboBox()
                self.surface.addItems(list(SURFACE_DBPKM.keys()))
                self.surface.setCurrentText(cfg["surface_key"])
                self.surface_user = QtWidgets.QLineEdit(str(cfg["surface_user_dbpkm"]))

                self.seabed = QtWidgets.QComboBox()
                self.seabed.addItems(list(SEABED_DBPKM.keys()))
                self.seabed.setCurrentText(cfg["seabed_key"])
                self.seabed_user = QtWidgets.QLineEdit(str(cfg["seabed_user_dbpkm"]))

                f2.addRow("Surface extra loss preset", self.surface)
                f2.addRow("Surface user (dB/km)", self.surface_user)
                f2.addRow("Seabed extra loss preset", self.seabed)
                f2.addRow("Seabed user (dB/km)", self.seabed_user)
                tabs.addTab(t2, "Environment")

                # Echo tab
                t3 = QtWidgets.QWidget()
                f3 = QtWidgets.QFormLayout(t3)
                self.echo = QtWidgets.QCheckBox("Enable echo (simple single-bounce)")
                self.echo.setChecked(bool(cfg["echo_enabled"]))
                self.echo_loss = QtWidgets.QLineEdit(str(cfg["echo_loss_db"]))

                self.echo_thresh_enable = QtWidgets.QCheckBox("Show detect threshold marker on echo")
                self.echo_thresh_enable.setChecked(bool(cfg.get("echo_detect_threshold_enabled", True)))
                self.echo_thresh_db = QtWidgets.QLineEdit(str(cfg.get("echo_detect_threshold_db", 75.0)))

                f3.addRow(self.echo)
                f3.addRow("Echo loss (dB)", self.echo_loss)
                f3.addRow(self.echo_thresh_enable)
                f3.addRow("Echo detect threshold (dB re 1 µPa)", self.echo_thresh_db)
                tabs.addTab(t3, "Echo")

                # CTD tab
                t5 = QtWidgets.QWidget()
                f5 = QtWidgets.QFormLayout(t5)
                self.tc_shade = QtWidgets.QCheckBox("Shade thermocline band (CTD viewer)")
                self.tc_shade.setChecked(bool(cfg["thermocline_shade"]))
                self.tc_lines = QtWidgets.QCheckBox("Show thermocline start/end lines (CTD viewer)")
                self.tc_lines.setChecked(bool(cfg["thermocline_lines"]))
                f5.addRow(self.tc_shade)
                f5.addRow(self.tc_lines)
                tabs.addTab(t5, "CTD Viewer")

                # Points tab
                t4 = QtWidgets.QWidget()
                v4 = QtWidgets.QVBoxLayout(t4)
                v4.addWidget(QtWidgets.QLabel("Select SPL rows to use. If none selected, all rows are used."))
                self.ptable = QtWidgets.QTableWidget()
                self.ptable.setColumnCount(6)
                self.ptable.setHorizontalHeaderLabels(["ID", "Freq (Hz)", "SPL (dB)", "Range (m)", "Method", "t_mid (s)"])
                self.ptable.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
                self.ptable.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
                self.ptable.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
                v4.addWidget(self.ptable, 1)

                rowb = QtWidgets.QHBoxLayout()
                b_all = QtWidgets.QPushButton("Select All")
                b_none = QtWidgets.QPushButton("Select None")
                rowb.addWidget(b_all)
                rowb.addWidget(b_none)
                rowb.addStretch(1)
                v4.addLayout(rowb)

                b_all.clicked.connect(self.ptable.selectAll)
                b_none.clicked.connect(self.ptable.clearSelection)

                tabs.addTab(t4, "Points")

                bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
                bb.accepted.connect(self._apply_and_accept)
                bb.rejected.connect(self.reject)
                v.addWidget(bb)

                self._fill_points()

            def _fill_points(self):
                pts = current_points
                self.ptable.setRowCount(len(pts))
                for r, p in enumerate(pts):
                    self.ptable.setItem(r, 0, QtWidgets.QTableWidgetItem(str(p["row_id"])))
                    self.ptable.setItem(r, 1, QtWidgets.QTableWidgetItem(f'{p["f_Hz"]:.3f}'))
                    self.ptable.setItem(r, 2, QtWidgets.QTableWidgetItem(f'{p["SPL_dB"]:.2f}'))
                    rm = p["r_meas_m"]
                    self.ptable.setItem(r, 3, QtWidgets.QTableWidgetItem("" if rm is None else f"{rm:.2f}"))
                    self.ptable.setItem(r, 4, QtWidgets.QTableWidgetItem(p["method"]))
                    self.ptable.setItem(r, 5, QtWidgets.QTableWidgetItem(f'{p["t_mid_s"]:.3f}'))
                self.ptable.resizeColumnsToContents()

                want = set(int(x) for x in (cfg.get("selected_row_ids") or []))
                if want:
                    sel = self.ptable.selectionModel()
                    for r, p in enumerate(pts):
                        if int(p["row_id"]) in want:
                            sel.select(self.ptable.model().index(r, 0),
                                    QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)

            def _apply_and_accept(self):
                def f(le, default):
                    try: return float(le.text().strip())
                    except Exception: return float(default)

                def i(le, default):
                    try: return int(float(le.text().strip()))
                    except Exception: return int(default)

                cfg["r_min"] = max(1.0, f(self.rmin, cfg["r_min"]))
                cfg["r_max"] = max(cfg["r_min"] + 1.0, f(self.rmax, cfg["r_max"]))
                cfg["r_N"] = max(10, i(self.rN, cfg["r_N"]))

                cfg["spreading"] = self.spread.currentText()
                cfg["r0_abs_m"] = max(1.0, f(self.r0, cfg["r0_abs_m"]))

                cfg["absorption"] = self.absorp.currentText()
                cfg["depth_for_absorp_m"] = max(0.0, f(self.depth, cfg["depth_for_absorp_m"]))

                cfg["surface_key"] = self.surface.currentText()
                cfg["surface_user_dbpkm"] = f(self.surface_user, cfg["surface_user_dbpkm"])
                cfg["seabed_key"] = self.seabed.currentText()
                cfg["seabed_user_dbpkm"] = f(self.seabed_user, cfg["seabed_user_dbpkm"])

                cfg["echo_enabled"] = self.echo.isChecked()
                cfg["echo_loss_db"] = f(self.echo_loss, cfg["echo_loss_db"])
                cfg["echo_detect_threshold_enabled"] = self.echo_thresh_enable.isChecked()
                cfg["echo_detect_threshold_db"] = f(self.echo_thresh_db, cfg.get("echo_detect_threshold_db", 75.0))

                cfg["thermocline_shade"] = self.tc_shade.isChecked()
                cfg["thermocline_lines"] = self.tc_lines.isChecked()

                sel_rows = self.ptable.selectionModel().selectedRows()
                if not sel_rows:
                    cfg["selected_row_ids"] = []
                else:
                    ids = []
                    for s in sel_rows:
                        r = s.row()
                        try:
                            rid = int(self.ptable.item(r, 0).text())
                            ids.append(rid)
                        except Exception:
                            pass
                    cfg["selected_row_ids"] = sorted(set(ids))

                self.accept()

        # ------------------------------------------------------------
        # Main dialog UI
        # ------------------------------------------------------------
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Propagation Modeling (SPL DB + CTD)")
        dlg.setWindowState(QtCore.Qt.WindowMaximized)

        outer = QtWidgets.QVBoxLayout(dlg)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        top = QtWidgets.QHBoxLayout()
        outer.addLayout(top)

        method_cb = QtWidgets.QComboBox()
        method_cb.addItem("<All>")
        for m in _list_methods():
            method_cb.addItem(m)

        file_cb = QtWidgets.QComboBox()

        ctd_cb = QtWidgets.QComboBox()
        ctd_cb.addItem("<None>")
        for r in _ctd_list():
            ctd_cb.addItem(f'{r["name"]} ({r["dt"]})', userData=r["id"])

        focus_freq_cb = QtWidgets.QComboBox()
        focus_freq_cb.addItem("<Median>")
        focus_freq_cb.addItem("<Custom…>")
        focus_freq_edit = QtWidgets.QLineEdit("3000")
        focus_freq_edit.setFixedWidth(90)
        focus_freq_edit.setEnabled(False)

        btn_cfg = QtWidgets.QPushButton("Config…")
        btn_compute = QtWidgets.QPushButton("Compute")
        btn_export_img = QtWidgets.QPushButton("Export JPG…")
        btn_close = QtWidgets.QPushButton("Close")

        top.addWidget(QtWidgets.QLabel("Method:"))
        top.addWidget(method_cb)
        top.addSpacing(10)
        top.addWidget(QtWidgets.QLabel("File:"))
        top.addWidget(file_cb, 1)
        top.addSpacing(10)
        top.addWidget(QtWidgets.QLabel("CTD:"))
        top.addWidget(ctd_cb)
        top.addSpacing(10)
        top.addWidget(QtWidgets.QLabel("Focus Freq:"))
        top.addWidget(focus_freq_cb)
        top.addWidget(focus_freq_edit)

        dist_unit_cb = QtWidgets.QComboBox()
        dist_unit_cb.addItems(["m", "km", "ft", "nm", "miles"])
        top.addSpacing(10)
        top.addWidget(QtWidgets.QLabel("Distance Unit:"))
        top.addWidget(dist_unit_cb)

        top.addStretch(1)
        top.addWidget(btn_cfg)
        top.addWidget(btn_compute)
        top.addWidget(btn_export_img)
        top.addWidget(btn_close)

        depth_row = QtWidgets.QHBoxLayout()
        depth_hint_lbl = QtWidgets.QLabel("")
        depth_hint_lbl.setStyleSheet("color: white;")
        btn_ctd_profile = QtWidgets.QPushButton("CTD Profile…")
        btn_ctd_profile.setEnabled(False)
        depth_row.addWidget(depth_hint_lbl, 1)
        depth_row.addWidget(btn_ctd_profile, 0)
        outer.addLayout(depth_row)

        # Primary plotting surface uses pyqtgraph (theme-matching app background)
        pg_glw = pg.GraphicsLayoutWidget()
        try:
            bg = dlg.palette().color(QtGui.QPalette.Window).name()
            pg_glw.setBackground(bg)
        except Exception:
            pg_glw.setBackground('#19232D')
        pg_ax1 = pg_glw.addPlot(row=0, col=0, title='RL vs Range (all frequencies)')
        pg_ax2 = pg_glw.addPlot(row=1, col=0, title='RL vs Frequency at sample ranges')
        pg_ax3 = pg_glw.addPlot(row=2, col=0, title='RL vs Range (representative frequencies)')
        outer.addWidget(pg_glw, 1)

        # Keep matplotlib figure for existing export pipeline
        fig = Figure(facecolor="#1e1e1e")
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        for ax in (ax1, ax2, ax3):
            _style_axes(ax, dark=True)

        canvas = FigureCanvas(fig)
        canvas.setVisible(False)
        outer.addWidget(canvas, 0)

        # ------------------------------------------------------------
        # Pick readout (main plots)
        # ------------------------------------------------------------
        picked_marker = None
        picked_annot = None

        def _clear_pick():
            nonlocal picked_marker, picked_annot
            if picked_marker is not None:
                try: picked_marker.remove()
                except Exception: pass
                picked_marker = None
            if picked_annot is not None:
                try: picked_annot.remove()
                except Exception: pass
                picked_annot = None

        def _on_pick(event):
            nonlocal picked_marker, picked_annot
            artist = event.artist
            if not hasattr(artist, "get_xdata"):
                return
            xdata = np.asarray(artist.get_xdata(), float)
            ydata = np.asarray(artist.get_ydata(), float)
            if xdata.size == 0 or ydata.size == 0:
                return
            try:
                idx = int(event.ind[0])
            except Exception:
                return
            x = float(xdata[idx]); y = float(ydata[idx])
            ax = artist.axes

            _clear_pick()
            picked_marker = ax.plot([x], [y], marker="o", markersize=7, linestyle="")[0]
            picked_annot = ax.annotate(
                f"x={x:g}\ny={y:.2f}",
                xy=(x, y),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=10,
                color="white",
                bbox=dict(boxstyle="round,pad=0.3", fc="#2b2b2b", ec="#777", alpha=0.95),
                arrowprops=dict(arrowstyle="->", color="#888"),
            )
            canvas.draw_idle()

        canvas.mpl_connect("pick_event", _on_pick)

        # ------------------------------------------------------------
        # Populate / refresh
        # ------------------------------------------------------------
        def _populate_files():
            file_cb.blockSignals(True)
            file_cb.clear()
            mf = method_cb.currentText().strip()
            mf = None if (not mf or mf == "<All>") else mf
            rows = _list_files(mf)
            for (fname, n, ts) in rows:
                file_cb.addItem(f"{fname} ({n} pts)", userData=fname)
            file_cb.blockSignals(False)

        def _populate_points():
            nonlocal current_points
            fname = file_cb.currentData()
            mf = method_cb.currentText().strip()
            mf = None if (not mf or mf == "<All>") else mf
            current_points = _load_points_for_file(fname, mf) if fname else []

            while focus_freq_cb.count() > 2:
                focus_freq_cb.removeItem(2)
            freqs = sorted({round(float(p["f_Hz"]), 3) for p in current_points if p.get("f_Hz") is not None})
            for f in freqs:
                focus_freq_cb.addItem(f"{f:g} Hz", userData=float(f))

        def _update_depth_hint():
            if ctd_cb.currentIndex() <= 0:
                depth_hint_lbl.setText("")
                btn_ctd_profile.setEnabled(False)
                return

            ctd_id = ctd_cb.currentData()
            prof = _load_ctd_profile(ctd_id)
            if not prof or prof.get("depth_m") is None or len(prof["depth_m"]) < 3:
                depth_hint_lbl.setText("CTD selected: could not load CTD arrays (depth_json missing/invalid).")
                btn_ctd_profile.setEnabled(False)
                return
            if prof.get("temp_C") is None or len(prof["temp_C"]) < 3:
                depth_hint_lbl.setText("CTD selected: temp_json missing/invalid (need depth_json + temp_json).")
                btn_ctd_profile.setEnabled(False)
                return

            rec = _recommend_depths_from_ctd(prof)
            if not rec:
                depth_hint_lbl.setText("CTD selected: not enough valid points (need ≥ 3).")
                btn_ctd_profile.setEnabled(False)
                return

            depth_hint_lbl.setText(
                f"CTD depth guidance: ideal ≈ {rec['depth_ideal']:.1f} m | "
                f"min c(z) @ {rec['depth_min_c']:.1f} m | "
                f"min |dc/dz| @ {rec['depth_min_grad']:.1f} m"
            )
            btn_ctd_profile.setEnabled(True)

        def _refresh_all():
            _populate_files()
            _populate_points()
            _update_depth_hint()

        def _selected_points_or_all():
            if not current_points:
                return []
            sel_ids = cfg.get("selected_row_ids") or []
            if not sel_ids:
                return list(current_points)
            want = set(int(x) for x in sel_ids)
            out = [p for p in current_points if int(p["row_id"]) in want]
            return out if out else list(current_points)

        def _on_focus_freq_changed():
            focus_freq_edit.setEnabled(focus_freq_cb.currentText() == "<Custom…>")

        focus_freq_cb.currentIndexChanged.connect(_on_focus_freq_changed)

        # ------------------------------------------------------------
        # CTD profile popup
        # ------------------------------------------------------------
        def _show_ctd_sound_speed_popup():
            if ctd_cb.currentIndex() <= 0:
                QtWidgets.QMessageBox.information(dlg, "No CTD", "Select a CTD cast first.")
                return

            ctd_id = ctd_cb.currentData()
            prof = _load_ctd_profile(ctd_id)
            if not prof or prof.get("depth_m") is None or len(prof["depth_m"]) < 3:
                QtWidgets.QMessageBox.information(dlg, "CTD missing", "Could not load CTD depth array.")
                return
            if prof.get("temp_C") is None or len(prof["temp_C"]) < 3:
                QtWidgets.QMessageBox.information(dlg, "CTD missing", "Need temp_json to compute/validate sound speed.")
                return

            D = np.asarray(prof["depth_m"], float)
            T = np.asarray(prof["temp_C"], float)
            S = prof.get("sal_ppt", None)
            if S is None:
                S = 35.0*np.ones_like(D)
            else:
                S = np.asarray(S, float)

            C = prof.get("c_ms", None)
            if C is None or len(C) != len(D) or not np.all(np.isfinite(C)):
                C = _mackenzie_c_ms(T, S, D)
            else:
                C = np.asarray(C, float)

            rec = _recommend_depths_from_ctd(prof)
            if not rec:
                QtWidgets.QMessageBox.information(dlg, "CTD", "Not enough data to compute depth recommendations.")
                return

            band = _thermocline_band(D, T)

            w = QtWidgets.QDialog(dlg)
            w.setWindowTitle("CTD Sound Speed Profile")
            w.resize(920, 680)
            lay = QtWidgets.QVBoxLayout(w)

            fig2 = Figure(facecolor="#1e1e1e")
            ax = fig2.add_subplot(111)
            canvas2 = FigureCanvas(fig2)
            lay.addWidget(canvas2, 1)

            _style_axes(ax, dark=True)
            ax.grid(True, alpha=0.25)

            pal = _palette_from_user_selected(10)
            c_line = pal[0]
            c_ideal = pal[1]
            c_minc = pal[2]
            c_ming = pal[3]
            c_tc = pal[4]

            ax.plot(C, D, linewidth=2.2, color=c_line, label="c(z)", picker=5)

            def _mark(depth_m, label, color, marker):
                try:
                    x = float(np.interp(depth_m, D, C))
                    y = float(depth_m)
                except Exception:
                    return
                ax.plot([x], [y],
                        marker=marker,
                        markersize=12 if marker == "*" else 10,
                        markerfacecolor=color,
                        markeredgecolor="white",
                        markeredgewidth=1.2,
                        linestyle="",
                        color=color,
                        label=label,
                        picker=5)

            _mark(rec["depth_ideal"], "Ideal depth", c_ideal, marker="*")
            _mark(rec["depth_min_c"], "Min c(z)", c_minc, marker="o")
            _mark(rec["depth_min_grad"], "Min |dc/dz|", c_ming, marker="s")

            if band is not None:
                z0, z1, thr = band
                if cfg.get("thermocline_shade", True):
                    ax.axhspan(z0, z1, color=c_tc, alpha=0.18, label="Thermocline band")
                if cfg.get("thermocline_lines", True):
                    ax.axhline(z0, color=c_tc, linestyle="--", linewidth=1.6, label="Thermocline start")
                    ax.axhline(z1, color=c_tc, linestyle="--", linewidth=1.6, label="Thermocline end")
                ax.text(0.02, 0.98,
                        f"Thermocline ~ {z0:.1f}–{z1:.1f} m",
                        transform=ax.transAxes,
                        va="top", ha="left",
                        fontsize=10,
                        color="white",
                        bbox=dict(boxstyle="round,pad=0.25", fc="#2b2b2b", ec="#777", alpha=0.85))

            ax.set_xlabel("Sound speed (m/s)")
            ax.set_ylabel("Depth (m)")
            ax.set_title("Sound speed profile with recommended depths")
            ax.invert_yaxis()
            ax.legend(loc="best", facecolor="#1e1e1e", edgecolor="#555", labelcolor="white")

            picked_marker2 = None
            picked_annot2 = None

            def _clear_pick2():
                nonlocal picked_marker2, picked_annot2
                if picked_marker2 is not None:
                    try: picked_marker2.remove()
                    except Exception: pass
                    picked_marker2 = None
                if picked_annot2 is not None:
                    try: picked_annot2.remove()
                    except Exception: pass
                    picked_annot2 = None

            def _on_pick2(event):
                nonlocal picked_marker2, picked_annot2
                artist = event.artist
                if not hasattr(artist, "get_xdata"):
                    return
                xdata = np.asarray(artist.get_xdata(), float)
                ydata = np.asarray(artist.get_ydata(), float)
                if xdata.size == 0 or ydata.size == 0:
                    return
                try:
                    idx = int(event.ind[0])
                except Exception:
                    return
                x = float(xdata[idx]); y = float(ydata[idx])
                _clear_pick2()
                picked_marker2 = ax.plot([x], [y], marker="o", markersize=8, linestyle="")[0]
                picked_annot2 = ax.annotate(
                    f"c={x:.2f} m/s\nd={y:.2f} m",
                    xy=(x, y), xytext=(10, 10), textcoords="offset points",
                    fontsize=10, color="white",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#2b2b2b", ec="#777", alpha=0.95),
                    arrowprops=dict(arrowstyle="->", color="#888"),
                )
                canvas2.draw_idle()

            canvas2.mpl_connect("pick_event", _on_pick2)

            # -------- CTD export helpers --------
            def _apply_fig2_theme(mode):
                st = {
                    "fig_face": fig2.get_facecolor(),
                    "ax_face": ax.get_facecolor(),
                    "title": ax.title.get_color(),
                    "xl": ax.xaxis.label.get_color(),
                    "yl": ax.yaxis.label.get_color(),
                    "spines": [sp.get_edgecolor() for sp in ax.spines.values()],
                    "lines": [],
                    "leg": ax.get_legend(),
                }
                for ln in ax.lines:
                    st["lines"].append({
                        "ln": ln,
                        "color": ln.get_color(),
                        "lw": ln.get_linewidth(),
                        "ls": ln.get_linestyle(),
                        "marker": ln.get_marker(),
                        "ms": ln.get_markersize(),
                        "mfc": ln.get_markerfacecolor(),
                        "mec": ln.get_markeredgecolor(),
                        "mew": ln.get_markeredgewidth(),
                    })

                def _legend_colors(text_color, face, edge):
                    leg = ax.get_legend()
                    if leg is None:
                        return
                    try:
                        for t in leg.get_texts():
                            t.set_color(text_color)
                        leg.get_frame().set_facecolor(face)
                        leg.get_frame().set_edgecolor(edge)
                    except Exception:
                        pass

                if mode == "dark":
                    fig2.set_facecolor("#1e1e1e")
                    _style_axes(ax, dark=True)
                    ax.grid(True, alpha=0.25)
                    _legend_colors("white", "#1e1e1e", "#555")
                    for txt in ax.texts:
                        if txt.get_bbox_patch():
                            txt.set_color("white")
                            txt.get_bbox_patch().set_facecolor("#2b2b2b")
                            txt.get_bbox_patch().set_edgecolor("#777")

                elif mode == "light":
                    fig2.set_facecolor("white")
                    _style_axes(ax, dark=False)
                    ax.grid(True, alpha=0.25)
                    _legend_colors("black", "white", "black")
                    for txt in ax.texts:
                        if txt.get_bbox_patch():
                            txt.set_color("black")
                            txt.get_bbox_patch().set_facecolor("white")
                            txt.get_bbox_patch().set_edgecolor("black")

                else:  # bw
                    fig2.set_facecolor("white")
                    _style_axes(ax, dark=False)
                    ax.grid(True, alpha=0.25)

                    for ln in ax.lines:
                        lab = (ln.get_label() or "").lower()
                        if lab == "c(z)":
                            ln.set_color("black")
                            ln.set_linewidth(max(ln.get_linewidth(), 2.2))
                            ln.set_marker("None")
                        elif "ideal" in lab:
                            ln.set_color("black")
                            ln.set_marker("*")
                            ln.set_markersize(13)
                            ln.set_markerfacecolor("black")
                            ln.set_markeredgecolor("black")
                            ln.set_markeredgewidth(1.0)
                        elif "min c" in lab:
                            ln.set_color("black")
                            ln.set_marker("o")
                            ln.set_markersize(11)
                            ln.set_markerfacecolor("white")
                            ln.set_markeredgecolor("black")
                            ln.set_markeredgewidth(1.5)
                        elif "min |dc/dz|" in lab or ("min" in lab and "grad" in lab):
                            ln.set_color("black")
                            ln.set_marker("s")
                            ln.set_markersize(11)
                            ln.set_markerfacecolor("0.65")
                            ln.set_markeredgecolor("black")
                            ln.set_markeredgewidth(1.2)
                        elif "thermocline" in lab:
                            ln.set_color("0.25")
                            ln.set_linewidth(max(ln.get_linewidth(), 1.6))
                        else:
                            ln.set_color("0.2")

                    for txt in ax.texts:
                        if txt.get_bbox_patch():
                            txt.set_color("black")
                            txt.get_bbox_patch().set_facecolor("white")
                            txt.get_bbox_patch().set_edgecolor("black")

                    _legend_colors("black", "white", "black")

                return st

            def _restore_fig2_theme(st):
                fig2.set_facecolor(st["fig_face"])
                ax.set_facecolor(st["ax_face"])
                ax.title.set_color(st["title"])
                ax.xaxis.label.set_color(st["xl"])
                ax.yaxis.label.set_color(st["yl"])
                for sp, c in zip(ax.spines.values(), st["spines"]):
                    try: sp.set_color(c)
                    except Exception: pass

                for d in st["lines"]:
                    ln = d["ln"]
                    try:
                        ln.set_color(d["color"])
                        ln.set_linewidth(d["lw"])
                        ln.set_linestyle(d["ls"])
                        ln.set_marker(d["marker"])
                        ln.set_markersize(d["ms"])
                        ln.set_markerfacecolor(d["mfc"])
                        ln.set_markeredgecolor(d["mec"])
                        ln.set_markeredgewidth(d["mew"])
                    except Exception:
                        pass

            def _export_ctd_jpg():
                base, _ = QtWidgets.QFileDialog.getSaveFileName(
                    w, "Export CTD Profile (Base Filename)", "", "JPG Files (*.jpg)"
                )
                if not base:
                    return
                if not base.lower().endswith(".jpg"):
                    base += ".jpg"
                base_noext = os.path.splitext(base)[0]

                msg = QtWidgets.QMessageBox(w)
                msg.setWindowTitle("Export CTD Modes")
                msg.setText("Choose which CTD styles to export:")
                dark_cb = QtWidgets.QCheckBox("Dark mode (dark background)")
                light_cb = QtWidgets.QCheckBox("Color on white background")
                bw_cb = QtWidgets.QCheckBox("Black & white")
                dark_cb.setChecked(True); light_cb.setChecked(True); bw_cb.setChecked(True)

                box = QtWidgets.QWidget()
                mlay = QtWidgets.QVBoxLayout(box)
                mlay.addWidget(dark_cb); mlay.addWidget(light_cb); mlay.addWidget(bw_cb)
                msg.layout().addWidget(box, 1, 0, 1, msg.layout().columnCount())

                msg.addButton("Export", QtWidgets.QMessageBox.AcceptRole)
                msg.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
                if msg.exec_() != 0:
                    return

                exports = []
                if dark_cb.isChecked():  exports.append(("dark",  f"{base_noext}_dark.jpg"))
                if light_cb.isChecked(): exports.append(("light", f"{base_noext}_light.jpg"))
                if bw_cb.isChecked():    exports.append(("bw",    f"{base_noext}_bw.jpg"))
                if not exports:
                    return

                for mode, path in exports:
                    st = _apply_fig2_theme(mode)
                    try:
                        try:
                            fig2.tight_layout()
                        except Exception:
                            pass
                        fig2.savefig(path, dpi=200, format="jpg", bbox_inches="tight")
                    finally:
                        _restore_fig2_theme(st)

                canvas2.draw_idle()
                QtWidgets.QMessageBox.information(w, "Exported", f"Saved {len(exports)} file(s).")

            try:
                fig2.tight_layout()
            except Exception:
                pass
            canvas2.draw_idle()

            row = QtWidgets.QHBoxLayout()
            btn_export_ctd = QtWidgets.QPushButton("Export JPG…")
            btn_export_ctd.setToolTip("Export this CTD profile plot (dark / light / black&white)")
            btn_close_ctd = QtWidgets.QPushButton("Close")
            row.addStretch(1)
            row.addWidget(btn_export_ctd)
            row.addWidget(btn_close_ctd)
            lay.addLayout(row)

            btn_export_ctd.clicked.connect(_export_ctd_jpg)
            btn_close_ctd.clicked.connect(w.accept)

            w.exec_()

        # ------------------------------------------------------------
        # Export JPG (MAIN)
        # ------------------------------------------------------------
        def _apply_fig_theme(mode):
            st = {"fig_face": fig.get_facecolor(), "axes": []}
            axes = [ax1, ax2, ax3]

            for ax in axes:
                ax_state = {
                    "ax_face": ax.get_facecolor(),
                    "title": ax.title.get_color(),
                    "xl": ax.xaxis.label.get_color(),
                    "yl": ax.yaxis.label.get_color(),
                    "spine": [sp.get_edgecolor() for sp in ax.spines.values()],
                    # store ONLY style tuples (no ln object)
                    "lines": [(
                        ln.get_color(),
                        ln.get_linestyle(),
                        ln.get_linewidth(),
                        ln.get_marker(),
                        ln.get_markersize(),
                        ln.get_markerfacecolor(),
                        ln.get_markeredgecolor(),
                        ln.get_markeredgewidth()
                    ) for ln in ax.lines],
                }
                st["axes"].append((ax, ax_state))

            if mode == "dark":
                fig.set_facecolor("#1e1e1e")
                for ax in axes:
                    _style_axes(ax, dark=True)
                    ax.grid(True, alpha=0.25)
                    leg = ax.get_legend()
                    if leg is not None:
                        try:
                            for txt in leg.get_texts():
                                txt.set_color("white")
                            leg.get_frame().set_facecolor("#1e1e1e")
                            leg.get_frame().set_edgecolor("#555")
                        except Exception:
                            pass

            elif mode == "light":
                fig.set_facecolor("white")
                for ax in axes:
                    _style_axes(ax, dark=False)
                    ax.grid(True, alpha=0.25)
                    leg = ax.get_legend()
                    if leg is not None:
                        try:
                            for txt in leg.get_texts():
                                txt.set_color("black")
                            leg.get_frame().set_facecolor("white")
                            leg.get_frame().set_edgecolor("black")
                        except Exception:
                            pass

            else:  # bw
                fig.set_facecolor("white")
                mono = ["black", "dimgray", "gray", "darkgray"]
                for ax in axes:
                    _style_axes(ax, dark=False)
                    ax.grid(True, alpha=0.25)
                    for i, ln in enumerate(ax.lines):
                        try:
                            ln.set_color(mono[i % len(mono)])
                        except Exception:
                            pass
                    leg = ax.get_legend()
                    if leg is not None:
                        try:
                            for txt in leg.get_texts():
                                txt.set_color("black")
                            leg.get_frame().set_facecolor("white")
                            leg.get_frame().set_edgecolor("black")
                        except Exception:
                            pass

            return st


        def _restore_fig_theme(st):
            fig.set_facecolor(st["fig_face"])
            for ax, ax_state in st["axes"]:
                ax.set_facecolor(ax_state["ax_face"])
                ax.title.set_color(ax_state["title"])
                ax.xaxis.label.set_color(ax_state["xl"])
                ax.yaxis.label.set_color(ax_state["yl"])
                for sp, c in zip(ax.spines.values(), ax_state["spine"]):
                    try:
                        sp.set_color(c)
                    except Exception:
                        pass

                # restore by index; handle line count changes safely
                saved = ax_state["lines"]
                for i, ln in enumerate(ax.lines):
                    if i >= len(saved):
                        break
                    (c, ls, lw, mk, ms, mfc, mec, mew) = saved[i]
                    try:
                        ln.set_color(c)
                        ln.set_linestyle(ls)
                        ln.set_linewidth(lw)
                        ln.set_marker(mk)
                        ln.set_markersize(ms)
                        ln.set_markerfacecolor(mfc)
                        ln.set_markeredgecolor(mec)
                        ln.set_markeredgewidth(mew)
                    except Exception:
                        pass

        def _export_graphs_jpg():
            base, _ = QtWidgets.QFileDialog.getSaveFileName(dlg, "Export Graphs (Base Filename)", "", "JPG Files (*.jpg)")
            if not base:
                return
            if not base.lower().endswith(".jpg"):
                base += ".jpg"
            base_noext = os.path.splitext(base)[0]

            msg = QtWidgets.QMessageBox(dlg)
            msg.setWindowTitle("Export Modes")
            msg.setText("Choose which JPG styles to export:")
            dark_cb = QtWidgets.QCheckBox("Dark mode (dark background)")
            light_cb = QtWidgets.QCheckBox("Color on white background")
            bw_cb = QtWidgets.QCheckBox("Black & white")
            dark_cb.setChecked(True); light_cb.setChecked(True); bw_cb.setChecked(True)

            box = QtWidgets.QWidget()
            lay2 = QtWidgets.QVBoxLayout(box)
            lay2.addWidget(dark_cb); lay2.addWidget(light_cb); lay2.addWidget(bw_cb)
            msg.layout().addWidget(box, 1, 0, 1, msg.layout().columnCount())

            msg.addButton("Export", QtWidgets.QMessageBox.AcceptRole)
            msg.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
            if msg.exec_() != 0:
                return

            exports = []
            if dark_cb.isChecked():  exports.append(("dark",  f"{base_noext}_dark.jpg"))
            if light_cb.isChecked(): exports.append(("light", f"{base_noext}_light.jpg"))
            if bw_cb.isChecked():    exports.append(("bw",    f"{base_noext}_bw.jpg"))
            if not exports:
                return

            for mode, path in exports:
                st = _apply_fig_theme(mode)
                try:
                    try:
                        fig.tight_layout()
                    except Exception:
                        pass
                    fig.savefig(path, dpi=200, format="jpg", bbox_inches="tight")
                finally:
                    _restore_fig_theme(st)

            canvas.draw_idle()
            QtWidgets.QMessageBox.information(dlg, "Exported", f"Saved {len(exports)} file(s).")

        # ------------------------------------------------------------
        # Compute / plot
        # ------------------------------------------------------------
        def _style_pg_plot(ax, left_label, bottom_label):
            axis_color = '#000000' if str(get_setting('ui_theme', 'dark')).lower() == 'light' else '#FFFFFF'
            ax.showGrid(x=True, y=True, alpha=0.25)
            ax.setLabel('left', left_label, color=axis_color)
            ax.setLabel('bottom', bottom_label, color=axis_color)
            ax.getAxis('left').setTextPen(pg.mkPen(axis_color))
            ax.getAxis('bottom').setTextPen(pg.mkPen(axis_color))
            try:
                ax.getAxis('bottom').enableAutoSIPrefix(False)
            except Exception:
                pass

        def _distance_unit_spec():
            unit = dist_unit_cb.currentText() if 'dist_unit_cb' in locals() else 'm'
            table = {
                'm': (1.0, 'm'),
                'km': (1.0 / 1000.0, 'km'),
                'ft': (3.280839895, 'ft'),
                'nm': (1.0 / 1852.0, 'nm'),
                'miles': (1.0 / 1609.344, 'miles'),
            }
            return table.get(unit, (1.0, 'm'))

        def _plot_pyqtgraph(rr, RL_med, RL_min, RL_max, echo_RL, echo_thresh_pt, f_u, RL_mat, single_freq_mode):
            for ax in (pg_ax1, pg_ax2, pg_ax3):
                ax.clear()
                ax.setLogMode(x=False, y=False)

            # Plot 1
            sf, unit_lbl = _distance_unit_spec()
            rr_disp = np.asarray(rr, float) * sf
            pg_ax1.setTitle('RL vs Range (all frequencies)')
            pg_ax1.plot(rr_disp, RL_med, pen=pg.mkPen('#03DFE2', width=2), name='Median RL')
            pg_ax1.plot(rr_disp, RL_min, pen=pg.mkPen('#77DD77', width=1), name='Min RL')
            pg_ax1.plot(rr_disp, RL_max, pen=pg.mkPen('#FD8A8A', width=1), name='Max RL')
            if echo_RL is not None:
                pg_ax1.plot(rr_disp, echo_RL, pen=pg.mkPen('#FFC8A2', width=2, style=QtCore.Qt.DashLine), name='Echo (focus)')
                if echo_thresh_pt is not None:
                    pg_ax1.plot([echo_thresh_pt[0] * sf], [echo_thresh_pt[1]], pen=None, symbol='x', symbolSize=10,
                                symbolPen=pg.mkPen('#FFC8A2', width=2))
            _style_pg_plot(pg_ax1, 'Level (dB re 1 µPa)', f'Range ({unit_lbl})')

            # Plot 2
            if not single_freq_mode:
                pg_ax2.setTitle('RL vs Frequency at sample ranges')
                sample_ranges = [rr[0], rr[len(rr)//2], rr[-1]]
                cols = ['#C8B6FF', '#FFFFB5', '#03DFE2']
                f_khz = np.asarray(f_u, float) / 1000.0
                for i, r_s in enumerate(sample_ranges):
                    j = int(np.argmin(np.abs(rr - r_s)))
                    pg_ax2.plot(f_khz, RL_mat[:, j], pen=pg.mkPen(cols[i % len(cols)], width=2),
                                symbol='o', symbolSize=4)
                _style_pg_plot(pg_ax2, 'RL (dB re 1 µPa)', 'Frequency (kHz)')
            else:
                f0 = float(f_u[0])
                pg_ax2.setTitle(f'RL vs Range (only one frequency in dataset: {f0:.0f} Hz)')
                pg_ax2.plot(rr_disp, RL_mat[0, :], pen=pg.mkPen('#C8B6FF', width=2))
                if echo_RL is not None:
                    pg_ax2.plot(rr_disp, echo_RL, pen=pg.mkPen('#FFC8A2', width=2, style=QtCore.Qt.DashLine))
                _style_pg_plot(pg_ax2, 'RL (dB re 1 µPa)', f'Range ({unit_lbl})')

            # Plot 3
            pg_ax3.setTitle('RL vs Range (representative frequencies)')
            order = np.argsort(f_u)
            f_sorted = f_u[order]
            RL_sorted = RL_mat[order, :]
            idxs = []
            if f_sorted.size >= 1:
                idxs.append(0)
            if f_sorted.size >= 3:
                idxs.append(f_sorted.size // 2); idxs.append(f_sorted.size - 1)
            elif f_sorted.size == 2:
                idxs.append(1)
            idxs = sorted(set(idxs))
            cols3 = ['#77DD77', '#03DFE2', '#FD8A8A', '#C8B6FF']
            for i, k in enumerate(idxs):
                pg_ax3.plot(rr_disp, RL_sorted[k, :], pen=pg.mkPen(cols3[i % len(cols3)], width=2))
            _style_pg_plot(pg_ax3, 'RL (dB re 1 µPa)', f'Range ({unit_lbl})')

        def _compute_and_plot():
            nonlocal last_results

            pts = _selected_points_or_all()
            if not pts:
                QtWidgets.QMessageBox.information(dlg, "No data", "No SPL points found for the selected file/method.")
                return

            r_min = max(1.0, float(cfg["r_min"]))
            r_max = max(r_min + 1.0, float(cfg["r_max"]))
            r_N = max(10, int(cfg["r_N"]))
            rr = np.geomspace(r_min, r_max, r_N)
            rr = rr[np.isfinite(rr) & (rr > 0)]
            if rr.size < 2:
                QtWidgets.QMessageBox.information(dlg, "Bad ranges", "Range grid invalid (needs >0 values).")
                return

            f_list = np.asarray([p["f_Hz"] for p in pts], float)
            spl_list = np.asarray([p["SPL_dB"] for p in pts], float)

            f_key = np.round(f_list, 6)
            uniq = np.unique(f_key)

            f_u, spl_u, r_meas_u = [], [], []
            for fk in uniq:
                mask = (f_key == fk)
                f_u.append(np.median(f_list[mask]))
                spl_u.append(np.median(spl_list[mask]))
                dists = [p["r_meas_m"] for p in pts
                        if p["r_meas_m"] is not None and abs(np.round(p["f_Hz"], 6) - fk) < 1e-9]
                r_meas_u.append(np.median(dists) if dists else None)

            f_u = np.asarray(f_u, float)
            spl_u = np.asarray(spl_u, float)

            keep_f = np.isfinite(f_u) & (f_u > 0)
            if not np.any(keep_f):
                QtWidgets.QMessageBox.information(
                    dlg, "No valid frequencies",
                    "All selected rows have non-positive frequencies.\nFix target_frequency in spl_calculations."
                )
                return

            f_u = f_u[keep_f]
            spl_u = spl_u[keep_f]
            r_meas_u = [r_meas_u[i] for i in range(len(r_meas_u)) if bool(keep_f[i])]

            single_freq_mode = (len(f_u) < 2)

            dvals = [d for d in r_meas_u if d is not None and np.isfinite(d) and d > 0]
            r_ref = float(np.median(dvals)) if dvals else 1.0
            r_ref = max(1.0, r_ref)

            spread_mode = cfg["spreading"]
            r0_abs = max(1.0, float(cfg["r0_abs_m"]))
            r0_ratio = r0_abs / r_ref
            TL_spread = spreading_TL_dB(rr / r_ref, spread_mode, r0_ratio=r0_ratio)

            absorp_mode = cfg["absorption"]
            depth_use = float(cfg["depth_for_absorp_m"])

            ctd_prof = None
            if ctd_cb.currentIndex() > 0:
                ctd_id = ctd_cb.currentData()
                ctd_prof = _load_ctd_profile(ctd_id)

            def _rep_ctd_at_depth(depth_m):
                if not ctd_prof or ctd_prof.get("temp_C") is None:
                    return (10.0, 35.0, 8.0, depth_m)
                Dp = np.asarray(ctd_prof["depth_m"], float)
                Tp = np.asarray(ctd_prof["temp_C"], float)
                Sp = ctd_prof.get("sal_ppt", None)
                if Sp is None:
                    Sp = 35.0*np.ones_like(Dp)
                else:
                    Sp = np.asarray(Sp, float)
                pH = 8.0
                Ti = float(np.interp(depth_m, Dp, Tp))
                Si = float(np.interp(depth_m, Dp, Sp))
                return (Ti, Si, pH, depth_m)

            surf_key = cfg["surface_key"]
            bed_key = cfg["seabed_key"]
            surf_extra = SURFACE_DBPKM.get(surf_key, 0.0)
            bed_extra = SEABED_DBPKM.get(bed_key, 0.0)
            if surf_key.startswith("User"):
                surf_extra = float(cfg["surface_user_dbpkm"])
            if bed_key.startswith("User"):
                bed_extra = float(cfg["seabed_user_dbpkm"])
            extra_dbpkm = float(surf_extra) + float(bed_extra)

            RL_mat = np.zeros((len(f_u), len(rr)), float)
            for i, f in enumerate(f_u):
                if absorp_mode.startswith("None"):
                    alpha = 0.0
                elif absorp_mode.startswith("Thorp"):
                    alpha = float(thorp_alpha_dB_per_km(f))
                else:
                    Ti, Si, pHi, Di = _rep_ctd_at_depth(depth_use)
                    alpha = float(fg_alpha_dB_per_km(f, Di, Ti, Si, pH_val=pHi))
                alpha_total = alpha + extra_dbpkm
                dr_km = np.maximum(0.0, (rr - r_ref) / 1000.0)
                TL_abs = alpha_total * dr_km
                RL_mat[i, :] = spl_u[i] - TL_spread - TL_abs

            RL_med = np.median(RL_mat, axis=0)
            RL_min = np.min(RL_mat, axis=0)
            RL_max = np.max(RL_mat, axis=0)

            focus_mode = focus_freq_cb.currentText()
            if focus_mode == "<Custom…>":
                try:
                    f_focus = float(focus_freq_edit.text().strip())
                except Exception:
                    f_focus = float(np.median(f_u))
            elif focus_mode == "<Median>":
                f_focus = float(np.median(f_u))
            else:
                ud = focus_freq_cb.currentData()
                f_focus = float(ud) if ud is not None else float(np.median(f_u))

            i_focus = int(np.argmin(np.abs(f_u - f_focus)))
            f_focus = float(f_u[i_focus])
            RL_focus = RL_mat[i_focus, :]

            echo_RL = None
            echo_thresh_pt = None  # (range_m, level_dB)
            if cfg["echo_enabled"]:
                echo_loss = float(cfg["echo_loss_db"])
                f_mid = f_focus
                if absorp_mode.startswith("None"):
                    alpha_mid = 0.0
                elif absorp_mode.startswith("Thorp"):
                    alpha_mid = float(thorp_alpha_dB_per_km(f_mid))
                else:
                    Ti, Si, pHi, Di = _rep_ctd_at_depth(depth_use)
                    alpha_mid = float(fg_alpha_dB_per_km(f_mid, Di, Ti, Si, pH_val=pHi))
                alpha_mid += extra_dbpkm
                dr_km = np.maximum(0.0, (rr - r_ref) / 1000.0)
                TL_abs_mid = alpha_mid * dr_km

                echo_RL = (float(np.median(spl_u))
                        - 2.0*TL_spread
                        - 2.0*TL_abs_mid
                        - echo_loss)

                # threshold marker
                if cfg.get("echo_detect_threshold_enabled", True) and echo_RL is not None:
                    thr = float(cfg.get("echo_detect_threshold_db", 75.0))
                    y = np.asarray(echo_RL, float)
                    x = np.asarray(rr, float)
                    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
                    x = x[ok]; y = y[ok]
                    if x.size >= 2:
                        below = (y < thr)
                        if np.any(below):
                            k = int(np.argmax(below))  # first True index
                            if k == 0:
                                echo_thresh_pt = (float(x[0]), float(y[0]))
                            else:
                                x0, x1 = float(x[k-1]), float(x[k])
                                y0, y1 = float(y[k-1]), float(y[k])
                                if y1 != y0:
                                    t = (thr - y0) / (y1 - y0)
                                    t = float(np.clip(t, 0.0, 1.0))
                                    xr = np.exp(np.log(x0) + t*(np.log(x1) - np.log(x0)))
                                    echo_thresh_pt = (float(xr), float(thr))

            last_results = {
                "range_m": rr,
                "freqs_Hz": f_u,
                "RL_mat_dB": RL_mat,
                "RL_median_dB": RL_med,
                "RL_min_dB": RL_min,
                "RL_max_dB": RL_max,
                "r_ref_m": r_ref,
                "focus_freq_Hz": f_focus,
                "RL_focus_dB": RL_focus,
                "echo_focus_dB": echo_RL,
                "echo_thresh_pt": echo_thresh_pt,
            }

            pal = _palette_from_user_selected(16)

            ax1.clear(); ax2.clear(); ax3.clear()
            _clear_pick()
            for ax in (ax1, ax2, ax3):
                _style_axes(ax, dark=True)

            # --- Plot 1: RL vs Range ---
            ax1.set_title("Received Level vs Range")
            ax1.semilogx(rr, RL_med, color=pal[0], linewidth=2, label="RL median", picker=5)
            ax1.fill_between(rr, RL_min, RL_max, color=pal[0], alpha=0.15, label="min..max (freq)")
            ax1.semilogx(rr, RL_focus, color=pal[1], linewidth=2, linestyle=":", label=f"RL @ {f_focus:.0f} Hz", picker=5)

            if echo_RL is not None:
                ax1.semilogx(rr, echo_RL, color=pal[2], linewidth=2, linestyle="--",
                            label=f"Echo @ {f_focus:.0f} Hz", picker=5)

                if echo_thresh_pt is not None:
                    xr, yr = echo_thresh_pt
                    ax1.plot([xr], [yr],
                            marker="X", markersize=10,
                            markerfacecolor=pal[2],
                            markeredgecolor="white",
                            markeredgewidth=1.2,
                            linestyle="",
                            label=f"Echo < {cfg.get('echo_detect_threshold_db', 75.0):.0f} dB @ {xr:.0f} m",
                            picker=5)
                    ax1.annotate(
                        f"{xr:.0f} m",
                        xy=(xr, yr),
                        xytext=(10, -12),
                        textcoords="offset points",
                        color="white",
                        fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.2", fc="#2b2b2b", ec="#777", alpha=0.9),
                        arrowprops=dict(arrowstyle="->", color="#888"),
                    )

            ax1.set_xlabel("Range (m)")
            ax1.set_ylabel("Level (dB re 1 µPa)")
            ax1.grid(True, alpha=0.25)
            ax1.legend(loc="best", facecolor="#1e1e1e", edgecolor="#555", labelcolor="white")
            _force_plain_x(ax1, log=True)

            # --- Plot 2: RL vs Frequency OR fallback ---
            if not single_freq_mode:
                ax2.set_title("RL vs Frequency at sample ranges")
                sample_ranges = [rr[0], rr[len(rr)//2], rr[-1]]
                for i, r_s in enumerate(sample_ranges):
                    j = int(np.argmin(np.abs(rr - r_s)))
                    ax2.semilogx(
                        f_u, RL_mat[:, j],
                        linewidth=2, color=pal[3+i],
                        marker="o", markersize=4,
                        label=f"r={rr[j]:.0f} m",
                        picker=5
                    )
                ax2.set_xlabel("Frequency (Hz)")
                ax2.set_ylabel("RL (dB re 1 µPa)")
                ax2.grid(True, alpha=0.25)
                ax2.legend(loc="best", facecolor="#1e1e1e", edgecolor="#555", labelcolor="white")
                _force_plain_x(ax2, log=True)
            else:
                f0 = float(f_u[0])
                ax2.set_title(f"RL vs Range (only one frequency in dataset: {f0:.0f} Hz)")
                ax2.semilogx(rr, RL_mat[0, :], linewidth=2, color=pal[3],
                            label=f"RL @ {f0:.0f} Hz", picker=5)
                if echo_RL is not None:
                    ax2.semilogx(rr, echo_RL, linewidth=2, color=pal[4], linestyle="--",
                                label="Echo (focus)", picker=5)
                    if echo_thresh_pt is not None:
                        xr, yr = echo_thresh_pt
                        ax2.plot([xr], [yr],
                                marker="X", markersize=10,
                                markerfacecolor=pal[4],
                                markeredgecolor="white",
                                markeredgewidth=1.2,
                                linestyle="",
                                label=f"Below {cfg.get('echo_detect_threshold_db', 75.0):.0f} dB @ {xr:.0f} m",
                                picker=5)
                ax2.set_xlabel("Range (m)")
                ax2.set_ylabel("RL (dB re 1 µPa)")
                ax2.grid(True, alpha=0.25)
                ax2.legend(loc="best", facecolor="#1e1e1e", edgecolor="#555", labelcolor="white")
                _force_plain_x(ax2, log=True)

            # --- Plot 3: RL vs Range for representative frequencies ---
            ax3.set_title("RL vs Range (representative frequencies)")
            order = np.argsort(f_u)
            f_sorted = f_u[order]
            RL_sorted = RL_mat[order, :]

            idxs = []
            if f_sorted.size >= 1:
                idxs.append(0)
            if f_sorted.size >= 3:
                idxs.append(f_sorted.size // 2)
                idxs.append(f_sorted.size - 1)
            elif f_sorted.size == 2:
                idxs.append(1)
            idxs = sorted(set(idxs))

            for i, k in enumerate(idxs):
                ax3.semilogx(rr, RL_sorted[k, :], linewidth=2,
                            color=pal[6+i], label=f"{f_sorted[k]:.0f} Hz", picker=5)

            ax3.set_xlabel("Range (m)")
            ax3.set_ylabel("RL (dB re 1 µPa)")
            ax3.grid(True, alpha=0.25)
            ax3.legend(loc="best", facecolor="#1e1e1e", edgecolor="#555", labelcolor="white")
            _force_plain_x(ax3, log=True)

            _plot_pyqtgraph(rr, RL_med, RL_min, RL_max, echo_RL, echo_thresh_pt, f_u, RL_mat, single_freq_mode)

            try:
                fig.tight_layout()
            except Exception:
                pass
            canvas.draw_idle()

        # ------------------------------------------------------------
        # Wiring
        # ------------------------------------------------------------
        def _open_config():
            _populate_points()
            d = PropagationConfigDialog(dlg)
            d.exec_()

        def _on_method_changed():
            _refresh_all()

        def _on_file_changed():
            _populate_points()

        method_cb.currentIndexChanged.connect(_on_method_changed)
        file_cb.currentIndexChanged.connect(_on_file_changed)
        ctd_cb.currentIndexChanged.connect(_update_depth_hint)
        dist_unit_cb.currentIndexChanged.connect(lambda *_: _compute_and_plot())

        btn_ctd_profile.clicked.connect(_show_ctd_sound_speed_popup)
        btn_cfg.clicked.connect(_open_config)
        btn_compute.clicked.connect(_compute_and_plot)
        btn_export_img.clicked.connect(_export_graphs_jpg)
        btn_close.clicked.connect(dlg.accept)

        _refresh_all()
        dlg.exec_()












    def wenz_curves_popup(self):
        """
        Wenz Curves + Measured PSD + CTD integration (active or saved)
        - Theme colors, measured PSD (Welch + calibration), smoothing
        - Save: Color/B&W with legend to the RIGHT in saved images
        - CTD panel: use active CTD or load any saved CTD from sqlite
        """
        from PyQt5 import QtWidgets, QtCore
        import numpy as np, os, json, ast, sqlite3, csv

        # -------- SciPy Welch (fallback) ----------
        try:
            from scipy.signal import welch
        except Exception:
            def welch(x, fs, window='hann', nperseg=8192, noverlap=None, detrend=False, scaling='density'):
                import numpy as _np
                if noverlap is None: noverlap = nperseg//2
                step = max(1, nperseg - noverlap)
                n = len(x)
                if n < nperseg:
                    f = _np.fft.rfftfreq(n, 1.0/fs)
                    win = _np.hanning(n); X = _np.fft.rfft(x * win)
                    Pxx = (np.abs(X)**2) / (fs * (win**2).sum()); return f, Pxx
                outP = []; win = _np.hanning(nperseg) if window == 'hann' else _np.ones(nperseg)
                wnorm = (win**2).sum()
                for s in range(0, n - nperseg + 1, step):
                    seg = x[s:s+nperseg]; X = _np.fft.rfft(seg * win)
                    P = (np.abs(X)**2) / (fs * wnorm); outP.append(P)
                f = _np.fft.rfftfreq(nperseg, 1.0/fs)
                Pxx = _np.mean(outP, axis=0) if outP else _np.zeros_like(f)
                return f, Pxx

        # -------- DB helpers (load saved CTDs) ----------
        def _db_path():
            try:
                from analyze_qt import DB_FILENAME
                return DB_FILENAME
            except Exception:
                return os.path.join(os.path.abspath(os.getcwd()), "analyze_qt.db")

        def _list_ctd_rows():
            try:
                conn = sqlite3.connect(_db_path()); cur = conn.cursor()
                cur.execute("SELECT id, name, dt_utc, latitude, longitude FROM ctd_profiles ORDER BY dt_utc DESC")
                rows = cur.fetchall(); conn.close()
                return rows
            except Exception:
                return []

        def _load_ctd_by_id(pk):
            conn = sqlite3.connect(_db_path()); cur = conn.cursor()
            cur.execute("""SELECT id, name, dt_utc, latitude, longitude, source,
                                depth_json, temp_json, sal_json, sound_speed_json
                        FROM ctd_profiles WHERE id=?""", (pk,))
            row = cur.fetchone(); conn.close()
            if not row: return None
            (_id, name, dt_utc, lat, lon, source, dj, tj, sj, cj) = row
            def _arr(js):
                if js is None: return None
                try:
                    a = json.loads(js)
                    return None if a is None else np.asarray(a, float)
                except Exception:
                    try:
                        a = ast.literal_eval(js); return np.asarray(a, float)
                    except Exception:
                        return None
            depth = _arr(dj); temp = _arr(tj); sal = _arr(sj); c_ms = _arr(cj)
            return {"id": _id, "name": name, "dt_utc": dt_utc, "latitude": lat, "longitude": lon,
                    "source": source, "depth_m": depth, "temperature_C": temp,
                    "salinity_PSU": sal, "sound_speed_m_s": c_ms}

        def _list_spl_files():
            try:
                conn = sqlite3.connect(_db_path()); cur = conn.cursor()
                cur.execute("SELECT DISTINCT file_name FROM spl_calculations WHERE file_name IS NOT NULL AND TRIM(file_name)!='' ORDER BY file_name")
                rows = [r[0] for r in cur.fetchall() if r and r[0]]
                conn.close()
                return rows
            except Exception:
                return []

        def _load_spl_curve_for_file(file_name):
            if not file_name:
                return None
            try:
                conn = sqlite3.connect(_db_path()); cur = conn.cursor()
                cur.execute(
                    "SELECT target_frequency, spl FROM spl_calculations WHERE file_name=? ORDER BY id DESC",
                    (file_name,),
                )
                rows = cur.fetchall(); conn.close()
            except Exception:
                return None
            if not rows:
                return None
            f = np.asarray([r[0] for r in rows], float)
            L = np.asarray([r[1] for r in rows], float)
            m = np.isfinite(f) & np.isfinite(L) & (f > 0)
            if not np.any(m):
                return None
            f = f[m]; L = L[m]
            fk = np.round(f, 3)
            uniq = np.unique(fk)
            fu, Lu = [], []
            for u in uniq:
                mm = fk == u
                fu.append(float(np.median(f[mm])))
                Lu.append(float(np.median(L[mm])))
            return np.asarray(fu, float), np.asarray(Lu, float)

        # -------- Hydrophone curves ----------
        curve_rows = []
        conn_h = None
        try:
            from analyze_qt import DB_FILENAME
            conn_h = sqlite3.connect(DB_FILENAME)
            cur = conn_h.cursor()
            cur.execute("SELECT curve_name, min_frequency, max_frequency, sensitivity_json FROM hydrophone_curves ORDER BY curve_name")
            curve_rows = cur.fetchall()
        except Exception:
            curve_rows = []

        def build_curve_interp(name):
            for nm, fmin, fmax, sj in curve_rows:
                if nm == name and sj:
                    arr = np.array(ast.literal_eval(sj) if str(sj).strip().startswith('[') else json.loads(sj), float)
                    freqs = np.linspace(float(fmin), float(fmax), arr.size) if arr.size > 1 else np.array([float(fmin)])
                    def interp(f_Hz):
                        f = np.atleast_1d(f_Hz).astype(float)
                        if freqs.size == 1:
                            return np.full_like(f, arr[0], dtype=float)
                        return np.interp(f, freqs, arr, left=arr[0], right=arr[-1])
                    return interp, nm
            return None, None

        # -------- UI ----------
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Wenz Curves + Measured PSD + CTD")
        dlg.setStyleSheet("background:#19232D; color:white;")
        dlg.resize(1120, 800)
        vbox = QtWidgets.QVBoxLayout(dlg); vbox.setContentsMargins(10,10,10,10); vbox.setSpacing(8)

        grid = QtWidgets.QGridLayout(); r = 0
        def _num(txt, w=90, ph=None):
            e = QtWidgets.QLineEdit(txt); e.setFixedWidth(w); 
            if ph: e.setPlaceholderText(ph); 
            return e

        # Wenz params
        grid.addWidget(QtWidgets.QLabel("Wind speed (m/s):"), r, 0)
        wind_edit = _num("8.0"); grid.addWidget(wind_edit, r, 1)
        grid.addWidget(QtWidgets.QLabel("Shipping b [0..1]:"), r, 2)
        ship_edit = _num("0.5"); grid.addWidget(ship_edit, r, 3)
        grid.addWidget(QtWidgets.QLabel("fmin (Hz):"), r, 4)
        fmin_edit = _num("10", 80); grid.addWidget(fmin_edit, r, 5)
        grid.addWidget(QtWidgets.QLabel("fmax (Hz):"), r, 6)
        fmax_edit = _num("100000", 100); grid.addWidget(fmax_edit, r, 7)

        r += 1
        turb_cb = QtWidgets.QCheckBox("Turbulence"); turb_cb.setChecked(True); grid.addWidget(turb_cb, r, 0, 1, 2)
        ship_cb = QtWidgets.QCheckBox("Shipping");   ship_cb.setChecked(True); grid.addWidget(ship_cb, r, 2, 1, 2)
        wind_cb = QtWidgets.QCheckBox("Wind/Weather"); wind_cb.setChecked(True); grid.addWidget(wind_cb, r, 4, 1, 2)
        therm_cb= QtWidgets.QCheckBox("Thermal");    therm_cb.setChecked(True); grid.addWidget(therm_cb, r, 6, 1, 2)

        r += 1
        grid.addWidget(QtWidgets.QLabel("Env. offset (dB):"), r, 0)
        env_off = _num("0", 80); grid.addWidget(env_off, r, 1)
        note = QtWidgets.QLabel("Tip: shallow water often ~+5 dB vs deep (environment offset).")
        note.setStyleSheet("color:#A0C4FF;")
        grid.addWidget(note, r, 2, 1, 6)

        # Measured PSD controls
        r += 1
        grid.addWidget(QtWidgets.QLabel("Measured PSD (WAV):"), r, 0)
        use_current_btn = QtWidgets.QPushButton("Use current file"); use_current_btn.setFixedWidth(140)
        pick_wav_btn = QtWidgets.QPushButton("Browse WAV…"); pick_wav_btn.setFixedWidth(120)
        grid.addWidget(use_current_btn, r, 1); grid.addWidget(pick_wav_btn, r, 2)
        meas_lbl = QtWidgets.QLabel("(none)"); meas_lbl.setStyleSheet("color:#ddd;")
        grid.addWidget(meas_lbl, r, 3, 1, 5)

        r += 1
        grid.addWidget(QtWidgets.QLabel("Measured Source:"), r, 0)
        meas_source_cb = QtWidgets.QComboBox()
        meas_source_cb.addItems(["WAV PSD", "SPL DB (Analyzed)", "WAV + SPL DB"])
        grid.addWidget(meas_source_cb, r, 1, 1, 2)
        grid.addWidget(QtWidgets.QLabel("Analyzed File:"), r, 3)
        spl_file_cb = QtWidgets.QComboBox(); spl_file_cb.setMinimumWidth(220)
        spl_file_cb.addItem("<Current file>", userData=None)
        for _fn in _list_spl_files():
            spl_file_cb.addItem(_fn, userData=_fn)
        grid.addWidget(spl_file_cb, r, 4, 1, 4)

        r += 1
        grid.addWidget(QtWidgets.QLabel("Start (s):"), r, 0); start_edit = _num("0", 80); grid.addWidget(start_edit, r, 1)
        grid.addWidget(QtWidgets.QLabel("Duration (s, 0=all):"), r, 2); dur_edit = _num("0", 90); grid.addWidget(dur_edit, r, 3)
        grid.addWidget(QtWidgets.QLabel("Welch seglen (s):"), r, 4); seglen_edit = _num("2.0", 80); grid.addWidget(seglen_edit, r, 5)
        grid.addWidget(QtWidgets.QLabel("Overlap (%):"), r, 6); ovlp_edit = _num("50", 80); grid.addWidget(ovlp_edit, r, 7)

        r += 1
        smooth_cb = QtWidgets.QCheckBox("1/3-octave smooth (measured)"); smooth_cb.setChecked(True)
        grid.addWidget(smooth_cb, r, 0, 1, 2)
        show_meas_cb = QtWidgets.QCheckBox("Show measured PSD"); show_meas_cb.setChecked(True)
        grid.addWidget(show_meas_cb, r, 2, 1, 2)

        # Sea state overlays
        r += 1
        sea_state_cb = QtWidgets.QCheckBox("Plot sea states (total)")
        sea_state_cb.setChecked(False)
        grid.addWidget(sea_state_cb, r, 0, 1, 2)
        grid.addWidget(QtWidgets.QLabel("# Sea states:"), r, 2)
        sea_state_count = QtWidgets.QSpinBox()
        sea_state_count.setRange(1, 8)
        sea_state_count.setValue(4)
        grid.addWidget(sea_state_count, r, 3)
        grid.addWidget(QtWidgets.QLabel("Values (0–6):"), r, 4)
        sea_state_edit = _num("0,1,2,3,4,5,6", 140)
        grid.addWidget(sea_state_edit, r, 5, 1, 2)

        # ---- CTD controls -----------------------------------------------------
        r += 1
        grid.addWidget(QtWidgets.QLabel("CTD:"), r, 0)
        use_ctd_cb = QtWidgets.QCheckBox("Use CTD")
        use_ctd_cb.setChecked(bool(getattr(self, "ctd_profile", None)))
        grid.addWidget(use_ctd_cb, r, 1)

        ctd_mode_cb = QtWidgets.QComboBox()
        ctd_mode_cb.addItems(["Active (in-memory)", "Pick saved from DB…"])
        grid.addWidget(ctd_mode_cb, r, 2, 1, 2)

        ctd_db_cb = QtWidgets.QComboBox(); ctd_db_cb.setEnabled(False)
        grid.addWidget(ctd_db_cb, r, 4, 1, 2)
        ctd_refresh_btn = QtWidgets.QPushButton("Refresh"); ctd_refresh_btn.setFixedWidth(90); ctd_refresh_btn.setEnabled(False)
        grid.addWidget(ctd_refresh_btn, r, 6)
        ctd_load_btn = QtWidgets.QPushButton("Load"); ctd_load_btn.setFixedWidth(70); ctd_load_btn.setEnabled(False)
        grid.addWidget(ctd_load_btn, r, 7)

        r += 1
        view_ctd_btn = QtWidgets.QPushButton("Open CTD Tool…"); view_ctd_btn.setFixedWidth(140)
        grid.addWidget(view_ctd_btn, r, 0)
        ctd_status = QtWidgets.QLabel(""); ctd_status.setStyleSheet("color:#A0C4FF;")
        grid.addWidget(ctd_status, r, 1, 1, 7)

        vbox.addLayout(grid)

        # Figure & axes
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        fig = Figure(facecolor="#19232D")
        canvas = FigureCanvas(fig)
        vbox.addWidget(canvas, 1)
        axes = {"main": None, "ctd": None}

        # Buttons
        btnrow = QtWidgets.QHBoxLayout()
        plot_btn = QtWidgets.QPushButton("Update"); plot_btn.setStyleSheet("background:#6EEB83;color:#111;padding:6px 12px;border-radius:6px;font-weight:bold;")
        save_btn = QtWidgets.QPushButton("Save Plot"); save_btn.setStyleSheet("background:#3E6C8A;color:white;padding:6px 12px;border-radius:4px;")
        csv_btn  = QtWidgets.QPushButton("Export CSV"); csv_btn.setStyleSheet("background:#3E6C8A;color:white;padding:6px 12px;border-radius:4px;")
        close_btn= QtWidgets.QPushButton("Close"); close_btn.setStyleSheet("background:#3E6C8A;color:white;padding:6px 12px;border-radius:4px;")
        btnrow.addStretch(); btnrow.addWidget(plot_btn); btnrow.addWidget(save_btn); btnrow.addWidget(csv_btn); btnrow.addWidget(close_btn)
        vbox.addLayout(btnrow)

        # Theme colors
        def _theme_palette():
            if hasattr(self, "theme_colors") and isinstance(self.theme_colors, (list, tuple)) and self.theme_colors:
                return list(self.theme_colors)
            if hasattr(self, "palette_colors") and isinstance(self.palette_colors, (list, tuple)) and self.palette_colors:
                return list(self.palette_colors)
            return ["#33C3F0", "#6EEB83", "#FF5964", "#FFD166", "#C792EA", "#4DD0E1"]
        def _component_colors():
            pals = _theme_palette()
            def pick(i, default="#33C3F0"): return pals[i] if i < len(pals) else default
            return {
                "Turbulence": pick(4), "Shipping": pick(2), "Wind/Weather": pick(0),
                "Thermal": pick(5), "Total (log-sum)": pick(1),
                "Measured PSD": getattr(self, "graph_color", pick(0)), "c(z)": pick(1),
            }

        # Wenz math
        def _wenz_components(fHz, v_w, b):
            fkHz = fHz / 1e3; eps = 1e-12
            Nt = 17.0 - 30.0*np.log10(np.maximum(fkHz, eps))
            Ns = 40.0 + 20.0*(b - 0.5) + 26.0*np.log10(np.maximum(fkHz, eps)) - 60.0*np.log10(np.maximum(fkHz + 0.03, eps))
            Nw = 50.0 + 7.5*np.sqrt(max(0.0, v_w)) + 20.0*np.log10(np.maximum(fkHz, eps)) - 40.0*np.log10(np.maximum(fkHz + 0.4, eps))
            Nth = -15.0 + 20.0*np.log10(np.maximum(fkHz, eps))
            return Nt, Ns, Nw, Nth
        def _logsum_db(*arrs):
            lin = np.zeros_like(arrs[0], dtype=float)
            for A in arrs: lin += 10.0**(A/10.0)
            return 10.0*np.log10(np.maximum(lin, 1e-30))
        def _sea_state_to_wind_ms(ss):
            table = [0.0, 2.0, 5.0, 8.0, 11.0, 14.0, 17.0]
            ss = int(np.clip(int(ss), 0, 6))
            return table[ss]

        # Measured helpers
        def _counts_to_volts_scale():
            x = getattr(self, "full_data", None)
            try: orig_dtype = self.original_dtype
            except Exception: orig_dtype = None if x is None else x.dtype
            if orig_dtype is not None and np.issubdtype(orig_dtype, np.integer):
                try: v_fullscale = float(self.max_voltage_entry.text())
                except Exception: v_fullscale = 1.0
                import numpy as _np
                return v_fullscale / _np.iinfo(orig_dtype).max
            return 1.0
        def _load_chunk(path):
            import soundfile as sf, numpy as _np
            if path is None and getattr(self, "full_data", None) is not None:
                x = self.full_data; fs = float(self.sample_rate)
                if x.ndim == 1:
                    cfg = self._get_channel_config(0) or {}
                    if cfg.get("mode") != "hydrophone" or not getattr(self, "channel_mask", [True])[0]:
                        return [], [], [], fs
                    data = [x.astype(_np.float64, copy=False) * _counts_to_volts_scale()]
                    labels = [getattr(self, "channel_names", ["Ch 1"])[0]]
                    idxs = [0]
                else:
                    mask = getattr(self, "channel_mask", []) or [True] * x.shape[1]
                    idxs = [
                        ch for ch in range(x.shape[1])
                        if ch < len(mask)
                        and mask[ch]
                        and (self._get_channel_config(ch) or {}).get("mode") == "hydrophone"
                    ]
                    names = getattr(self, "channel_names", [])
                    data = []
                    labels = []
                    for ch in idxs:
                        data.append(x[:, ch].astype(_np.float64, copy=False) * _counts_to_volts_scale())
                        labels.append(names[ch] if ch < len(names) else f"Ch {ch+1}")
                return data, labels, idxs, fs
            with sf.SoundFile(path) as f:
                fs = float(f.samplerate)
                f0 = float(start_edit.text() or "0"); dur = float(dur_edit.text() or "0")
                f.seek(int(max(0.0, f0)*fs)); frames = None if dur <= 0 else int(dur*fs)
                data = f.read(frames, dtype='float64', always_2d=True)
            if data.ndim == 2 and data.shape[1] > 1:
                channels = [data[:, ch].ravel() for ch in range(data.shape[1])]
                labels = [f"Ch {ch+1}" for ch in range(data.shape[1])]
            else:
                channels = [data.ravel()]
                labels = ["Ch 1"]
            idxs = [None] * len(labels)
            return channels, labels, idxs, fs
        def _welch_psd_volts(data, fs):
            try: seglen = max(0.02, float(seglen_edit.text()))
            except Exception: seglen = 2.0
            nperseg = max(256, int(seglen * fs))
            try: ovlp = float(ovlp_edit.text()); ovlp = min(95.0, max(0.0, ovlp))
            except Exception: ovlp = 50.0
            noverlap = int(nperseg * (ovlp/100.0))
            f, Pxx = welch(data, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap,
                        detrend=False, scaling='density')
            return f, Pxx
        def _apply_curve_cal(Pxx_V2Hz, fHz, RS_fn):
            L = 10.0*np.log10(np.maximum(Pxx_V2Hz, 1e-30))
            return L - RS_fn(fHz) if RS_fn is not None else L
        def _smooth_third_oct(fHz, L_dB):
            if not smooth_cb.isChecked(): return fHz, L_dB
            logf = np.log10(fHz)
            logf_u = np.linspace(logf.min(), logf.max(), len(logf))
            L_u = np.interp(logf_u, logf, L_dB)
            N = len(logf_u); width = max(5, N//60)
            k = np.arange(-width, width+1); sigma = width/2.355
            g = np.exp(-(k**2)/(2*sigma**2)); g /= g.sum()
            L_s = np.convolve(L_u, g, mode='same'); f_out = 10**logf_u
            return f_out, L_s

        # Compute everything
        def _compute_all():
            try:
                v_w = max(0.0, float(wind_edit.text()))
                b   = min(1.0, max(0.0, float(ship_edit.text())))
                fmin= max(1.0, float(fmin_edit.text()))
                fmax= max(fmin*1.01, float(fmax_edit.text()))
                off = float(env_off.text())
            except Exception as e:
                QtWidgets.QMessageBox.critical(dlg, "Input error", str(e)); return None
            fHz = np.logspace(np.log10(fmin), np.log10(fmax), 800)
            Nt, Ns, Nw, Nth = _wenz_components(fHz, v_w, b)
            comps, labels = [], []
            if turb_cb.isChecked(): comps.append(Nt); labels.append("Turbulence")
            if ship_cb.isChecked(): comps.append(Ns); labels.append("Shipping")
            if wind_cb.isChecked(): comps.append(Nw); labels.append("Wind/Weather")
            if therm_cb.isChecked(): comps.append(Nth); labels.append("Thermal")
            if not comps: comps, labels = [Nt, Ns, Nw, Nth], ["Turbulence","Shipping","Wind/Weather","Thermal"]
            Ntot = _logsum_db(*comps) + off

            meas = []
            source_mode = meas_source_cb.currentText() if 'meas_source_cb' in locals() else 'WAV PSD'
            use_wav = source_mode in ('WAV PSD', 'WAV + SPL DB')
            use_spl_db = source_mode in ('SPL DB (Analyzed)', 'WAV + SPL DB')

            if show_meas_cb.isChecked() and use_wav and (getattr(self, "full_data", None) is not None or getattr(self, "file_name", None)):
                try:
                    data_list, meas_labels, meas_idxs, fs = _load_chunk(meas_path["path"])
                except Exception as e:
                    QtWidgets.QMessageBox.critical(dlg, "Read error", str(e)); return (fHz, labels, comps, Ntot, None, {})
                if not data_list:
                    QtWidgets.QMessageBox.warning(
                        dlg,
                        "No hydrophone channels",
                        "No channels are configured as hydrophones for the current file.",
                    )
                for data, ch_label, ch_idx in zip(data_list, meas_labels, meas_idxs):
                    RS_fn = None
                    if ch_idx is not None:
                        cfg = self._get_channel_config(ch_idx) or {}
                        curve_name = cfg.get("hydrophone_curve")
                        if curve_name:
                            RS_fn, _ = build_curve_interp(curve_name)
                    if RS_fn is None:
                        QtWidgets.QMessageBox.warning(
                            dlg,
                            "Calibration required",
                            f"No hydrophone curve configured for {ch_label}.",
                        )
                        continue
                    if data is None or data.size < 256:
                        QtWidgets.QMessageBox.warning(dlg, "Not enough data", f"Audio segment too short for PSD: {ch_label}.")
                        continue
                    fM, Pxx = _welch_psd_volts(data, fs)
                    m = (fM >= fmin) & (fM <= fmax)
                    if np.any(m):
                        fM = fM[m]; Pxx = Pxx[m]
                        L_meas = _apply_curve_cal(Pxx, fM, RS_fn)
                        fS, Ls = _smooth_third_oct(fM, L_meas)
                        meas.append({"label": ch_label, "f": fHz, "L": np.interp(fHz, fS, Ls)})

            if show_meas_cb.isChecked() and use_spl_db:
                spl_file = spl_file_cb.currentData() if 'spl_file_cb' in locals() else None
                if not spl_file:
                    spl_file = getattr(self, 'file_name', None)
                spl_curve = _load_spl_curve_for_file(spl_file)
                if spl_curve is not None:
                    f_db, L_db = spl_curve
                    m = (f_db >= fmin) & (f_db <= fmax)
                    if np.any(m):
                        f_db = f_db[m]; L_db = L_db[m]
                        meas.append({
                            "label": f"Analyzed SPL: {os.path.basename(spl_file) if spl_file else 'current'}",
                            "f": fHz,
                            "L": np.interp(fHz, f_db, L_db),
                        })

            sea_state_totals = {}
            if sea_state_cb.isChecked():
                try:
                    raw_states = [int(s.strip()) for s in sea_state_edit.text().split(",") if s.strip() != ""]
                except Exception:
                    raw_states = []
                raw_states = [int(np.clip(s, 0, 6)) for s in raw_states]
                if not raw_states:
                    raw_states = [0, 1, 2, 3, 4, 5, 6]
                count = int(sea_state_count.value())
                for ss in raw_states[:count]:
                    v_ss = _sea_state_to_wind_ms(ss)
                    Nt, Ns, Nw, Nth = _wenz_components(fHz, v_ss, b)
                    ss_comps = []
                    if turb_cb.isChecked(): ss_comps.append(Nt)
                    if ship_cb.isChecked(): ss_comps.append(Ns)
                    if wind_cb.isChecked(): ss_comps.append(Nw)
                    if therm_cb.isChecked(): ss_comps.append(Nth)
                    if not ss_comps:
                        ss_comps = [Nt, Ns, Nw, Nth]
                    sea_state_totals[ss] = _logsum_db(*ss_comps) + off

            return fHz, labels, comps, Ntot, meas if meas else None, sea_state_totals

        # Axes builder
        def _build_axes():
            fig.clear()
            have_ctd = bool(_get_ctd_profile()) and use_ctd_cb.isChecked()
            if have_ctd:
                gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3, 1], hspace=0.28)
                ax_main = fig.add_subplot(gs[0, 0]); ax_ctd = fig.add_subplot(gs[1, 0])
            else:
                ax_main = fig.add_subplot(111); ax_ctd = None
            for axx in [ax_main] + ([ax_ctd] if ax_ctd is not None else []):
                axx.set_facecolor("#19232D")
                for s in axx.spines.values(): s.set_color("white")
                axx.tick_params(colors="white"); axx.grid(True, ls="--", alpha=0.35, color="gray")
            ax_main.set_xscale("log")
            ax_main.set_xlabel("Frequency (Hz)", color="white")
            ax_main.set_ylabel("PSD level (dB re 1 µPa²/Hz)", color="white")
            axes["main"], axes["ctd"] = ax_main, ax_ctd
            fig.subplots_adjust(right=0.95)

        # Legend helper (outside for saved)
        def _make_legend_outside(ax_in, fig_in, on_white=False):
            fig_in.subplots_adjust(right=0.78)
            lab_color = "black" if on_white else "white"
            face = "#FFFFFF" if on_white else "#222"; edge = "#444444"
            return ax_in.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0,
                                facecolor=face, edgecolor=edge, labelcolor=lab_color)

        # CTD source handling
        ctd_rows_cache = []  # [(id, name, dt, lat, lon), ...]
        def _refresh_ctd_db_list():
            nonlocal ctd_rows_cache
            ctd_rows_cache = _list_ctd_rows()
            ctd_db_cb.clear()
            for (pk, name, dt, lat, lon) in ctd_rows_cache:
                loc = "" if (lat is None or lon is None) else f" ({lat:.3f},{lon:.3f})"
                lab = f"{name} — {dt}{loc}"
                ctd_db_cb.addItem(lab, pk)

        def _get_ctd_profile():
            # Active (in-memory)
            if ctd_mode_cb.currentText().startswith("Active"):
                return getattr(self, "ctd_profile", None)
            # Or currently loaded from DB (we stash it on self for reuse)
            return getattr(self, "_ctd_profile_from_db", None)

        def _load_selected_ctd_from_db():
            if ctd_db_cb.count() == 0: return
            pk = ctd_db_cb.currentData()
            prof = _load_ctd_by_id(pk)
            if prof is None:
                QtWidgets.QMessageBox.warning(dlg, "Load CTD", "Could not load selected profile."); return
            # store and also set as active so other tools can see it
            self._ctd_profile_from_db = prof
            self.ctd_profile = prof
            _update_ctd_status()
            _plot()

        # Plot
        def _plot():
            _update_ctd_status()
            _build_axes()
            res = _compute_all()
            if res is None: canvas.draw(); return
            fHz, labels, comps, Ntot, meas, sea_state_totals = res
            ax = axes["main"]; cols = _component_colors()
            for L, lab in zip(comps, labels):
                ax.plot(fHz, L, lw=1.6, label=lab, color=cols.get(lab))
            ax.plot(fHz, Ntot, lw=2.2, label="Total (log-sum)", color=cols.get("Total (log-sum)"))
            if meas is not None:
                for idx, entry in enumerate(meas):
                    color = cols.get("Measured PSD")
                    if isinstance(color, str):
                        pass
                    else:
                        color = _theme_palette()[idx % len(_theme_palette())]
                    ax.plot(entry["f"], entry["L"], lw=2.0, label=f"Measured PSD ({entry['label']})", color=color)
            if sea_state_totals:
                for idx, ss in enumerate(sorted(sea_state_totals.keys())):
                    ax.plot(fHz, sea_state_totals[ss], lw=1.8, linestyle="--", label=f"Total (SS {ss})")
            ax.legend(facecolor="#222", edgecolor="#444", labelcolor="white")
            ax.set_title("Wenz curves + measured PSD (dB re 1 µPa²/Hz)", color="white")

            ax_ctd = axes["ctd"]
            if ax_ctd is not None:
                prof = _get_ctd_profile()
                depth = prof.get("depth_m") if prof else None
                c_ms  = prof.get("sound_speed_m_s") if prof else None
                if depth is not None and c_ms is not None and len(depth) > 1:
                    ax_ctd.clear()
                    ax_ctd.set_facecolor("#19232D")
                    for s in ax_ctd.spines.values(): s.set_color("white")
                    ax_ctd.tick_params(colors="white"); ax_ctd.grid(True, ls="--", alpha=0.35, color="gray")
                    ax_ctd.plot(c_ms, depth, color=cols.get("c(z)"), lw=1.8, label="c(z)")
                    ax_ctd.set_xlabel("Sound speed c (m/s)", color="white")
                    ax_ctd.set_ylabel("Depth (m)", color="white"); ax_ctd.invert_yaxis()
                    ax_ctd.legend(facecolor="#222", edgecolor="#444", labelcolor="white")
                else:
                    ax_ctd.text(0.5,0.5,"CTD missing c(z) or depth", color="white", ha="center", va="center",
                                transform=ax_ctd.transAxes)
            canvas.draw()

        # Save (Color/B&W; legend to the right)
        def _choose_save_style(parent):
            choice, ok = QtWidgets.QInputDialog.getItem(
                parent, "Save Plot", "Style:", ["Color (theme)", "Black & White (print)"], 0, False
            )
            if not ok: return None
            return "color" if choice.startswith("Color") else "bw"

        def _render_bw_and_save(path):
            from matplotlib.figure import Figure
            res = _compute_all()
            if res is None: return
            fHz, labels, comps, Ntot, meas, sea_state_totals = res
            prof = _get_ctd_profile()
            have_ctd = bool(prof) and use_ctd_cb.isChecked()
            depth = prof.get("depth_m") if have_ctd else None
            c_ms  = prof.get("sound_speed_m_s") if have_ctd else None
            # match current limits
            try: xlim = axes["main"].get_xlim(); ylim = axes["main"].get_ylim()
            except Exception: xlim = ylim = None
            if have_ctd and depth is not None and c_ms is not None and len(depth)>1:
                fig_bw = Figure(facecolor="#FFFFFF")
                gs = fig_bw.add_gridspec(nrows=2, ncols=1, height_ratios=[3, 1], hspace=0.28)
                ax_bw = fig_bw.add_subplot(gs[0, 0]); ax_ctd = fig_bw.add_subplot(gs[1, 0])
            else:
                fig_bw = Figure(facecolor="#FFFFFF"); ax_bw = fig_bw.add_subplot(111); ax_ctd = None
            for axx in [ax_bw] + ([ax_ctd] if ax_ctd is not None else []):
                axx.set_facecolor("#FFFFFF")
                for s in axx.spines.values(): s.set_color("black")
                axx.tick_params(colors="black"); axx.grid(True, ls="--", alpha=0.4, color="#999999")
            ax_bw.set_xscale("log")
            ax_bw.set_xlabel("Frequency (Hz)", color="black")
            ax_bw.set_ylabel("PSD level (dB re 1 µPa²/Hz)", color="black")
            if xlim: ax_bw.set_xlim(xlim)
            if ylim: ax_bw.set_ylim(ylim)
            styles = ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]
            for i, (L, lab) in enumerate(zip(comps, labels)):
                ax_bw.plot(fHz, L, linestyle=styles[i % len(styles)], color="black", lw=1.8, label=lab)
            ax_bw.plot(fHz, Ntot, linestyle="-", color="black", lw=2.4, label="Total (log-sum)")
            if meas is not None:
                for entry in meas:
                    ax_bw.plot(entry["f"], entry["L"], linestyle="-", color="black", lw=2.0,
                            label=f"Measured PSD ({entry['label']})", marker="o", markevery=40, ms=3)
            if sea_state_totals:
                for idx, ss in enumerate(sorted(sea_state_totals.keys())):
                    ax_bw.plot(fHz, sea_state_totals[ss], linestyle=styles[(idx + 1) % len(styles)],
                            color="black", lw=2.0, label=f"Total (SS {ss})")
            _make_legend_outside(ax_bw, fig_bw, on_white=True)
            ax_bw.set_title("Wenz curves + measured PSD (B/W export)", color="black")
            if ax_ctd is not None:
                ax_ctd.plot(c_ms, depth, color="black", lw=1.8, label="c(z)")
                ax_ctd.invert_yaxis()
                ax_ctd.set_xlabel("Sound speed c (m/s)", color="black")
                ax_ctd.set_ylabel("Depth (m)", color="black")
                ax_ctd.legend(facecolor="#FFFFFF", edgecolor="#444444", labelcolor="black")
            fig_bw.savefig(path, dpi=220, facecolor=fig_bw.get_facecolor(), bbox_inches="tight")

        def _save_plot():
            style = _choose_save_style(dlg)
            if style is None: return
            p,_ = QtWidgets.QFileDialog.getSaveFileName(dlg, "Save Plot", "", "PNG (*.png);;JPEG (*.jpg)")
            if not p: return
            if style == "color":
                _plot(); _make_legend_outside(axes["main"], fig, on_white=False)
                fig.savefig(p, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
                _plot()
            else:
                _render_bw_and_save(p)

        def _export_csv():
            res = _compute_all()
            if res is None: return
            fHz, labels, comps, Ntot, meas, sea_state_totals = res
            p,_ = QtWidgets.QFileDialog.getSaveFileName(dlg, "Export CSV", "", "CSV (*.csv)")
            if not p: return
            with open(p, "w", newline="") as fh:
                w = csv.writer(fh)
                header = ["f_Hz"] + [f"{lab}_dB" for lab in labels] + ["Total_dB"]
                if meas is not None:
                    header += [f"Measured_{entry['label']}_dB" for entry in meas]
                if sea_state_totals:
                    header += [f"Total_SS{ss}_dB" for ss in sorted(sea_state_totals.keys())]
                w.writerow(header)
                for i in range(len(fHz)):
                    row = [fHz[i]] + [col[i] for col in comps] + [Ntot[i]]
                    if meas is not None:
                        row.extend([entry["L"][i] for entry in meas])
                    if sea_state_totals:
                        row.extend([sea_state_totals[ss][i] for ss in sorted(sea_state_totals.keys())])
                    w.writerow(row)
                # Append CTD if present
                prof = _get_ctd_profile()
                if use_ctd_cb.isChecked() and prof and prof.get("depth_m") is not None and prof.get("sound_speed_m_s") is not None:
                    w.writerow([]); w.writerow(["Depth_m", "SoundSpeed_m_s"])
                    for d, c in zip(prof["depth_m"], prof["sound_speed_m_s"]):
                        w.writerow([d, c])

        # Status text
        def _update_ctd_status():
            if not use_ctd_cb.isChecked():
                ctd_status.setText("CTD disabled."); return
            prof = _get_ctd_profile()
            if prof:
                src = prof.get("name") or os.path.basename(prof.get("source","(in-memory)"))
                dt = prof.get("dt_utc",""); lat = prof.get("latitude", None); lon = prof.get("longitude", None)
                loc = "" if (lat is None or lon is None) else f"  ({lat:.3f},{lon:.3f})"
                npts = len(prof.get("depth_m", [])) if prof.get("depth_m") is not None else 0
                ctd_status.setText(f"Using CTD: {src}  {dt}{loc}  —  {npts} pts")
            else:
                ctd_status.setText("No CTD selected.")

        # Handlers
        meas_path = {"path": None}
        def on_use_current():
            meas_path["path"] = None
            nm = getattr(self, "file_name", None)
            meas_lbl.setText(nm if nm else "(current buffer)"); _plot()
        def on_pick_wav():
            path,_ = QtWidgets.QFileDialog.getOpenFileName(dlg, "Select WAV", "", "WAV Files (*.wav *.wave)")
            if not path: return
            meas_path["path"] = path; meas_lbl.setText(os.path.basename(path)); _plot()
        def on_open_ctd():
            try:
                if hasattr(self, "ctd_import_popup"): self.ctd_import_popup()
            except Exception as e:
                QtWidgets.QMessageBox.warning(dlg, "CTD tool", str(e))
            if ctd_mode_cb.currentText().startswith("Pick"):
                _refresh_ctd_db_list()
            _update_ctd_status(); _plot()
        def on_ctd_mode_changed():
            pick = ctd_mode_cb.currentText().startswith("Pick")
            ctd_db_cb.setEnabled(pick); ctd_refresh_btn.setEnabled(pick); ctd_load_btn.setEnabled(pick)
            if pick: _refresh_ctd_db_list()
            _update_ctd_status(); _plot()

        # Wire up
        use_current_btn.clicked.connect(on_use_current)
        pick_wav_btn.clicked.connect(on_pick_wav)
        view_ctd_btn.clicked.connect(on_open_ctd)
        ctd_mode_cb.currentIndexChanged.connect(on_ctd_mode_changed)
        ctd_refresh_btn.clicked.connect(_refresh_ctd_db_list)
        ctd_load_btn.clicked.connect(_load_selected_ctd_from_db)

        plot_btn.clicked.connect(_plot)
        save_btn.clicked.connect(_save_plot)
        csv_btn.clicked.connect(_export_csv)
        close_btn.clicked.connect(dlg.accept)

        for w in (wind_edit, ship_edit, fmin_edit, fmax_edit, env_off, start_edit, dur_edit, seglen_edit, ovlp_edit):
            w.editingFinished.connect(_plot)
        for cb in (turb_cb, ship_cb, wind_cb, therm_cb, smooth_cb, show_meas_cb, use_ctd_cb, sea_state_cb):
            cb.stateChanged.connect(_plot)
        sea_state_count.valueChanged.connect(_plot)
        sea_state_edit.editingFinished.connect(_plot)
        meas_source_cb.currentIndexChanged.connect(_plot)
        spl_file_cb.currentIndexChanged.connect(_plot)

        # init
        on_ctd_mode_changed()
        _plot()
        dlg.exec_()
        if conn_h: conn_h.close()








