#!/usr/bin/env python3
"""DIFAR tools mixin for MainWindow integration."""

import os
import json
import sqlite3
import inspect
from datetime import timezone, datetime

from PyQt5 import QtWidgets, QtCore, QtGui

from shared import DB_FILENAME
from difar_core import (
    DifarConfig,
    import_difar_calibration_csv_to_db,
    load_difar_calibration_from_db,
    load_compass_csv,
    process_wav_to_bearing_time_series,
    process_wav_to_bearing_time_series_chunked,
    bearing_series_static_map_vectors,
)


class DifarToolsMixin:
    """Mixin class providing DIFAR tools for MainWindow."""

    def _resolve_active_project_id(self):
        """Best-effort resolve of currently selected project id."""
        pid = getattr(self, "current_project_id", None)
        try:
            if pid is not None:
                return int(pid)
        except Exception:
            pass

        pname = (getattr(self, "current_project_name", None) or "").strip()
        if not pname and hasattr(self, "project_combo") and self.project_combo is not None:
            try:
                pname = (self.project_combo.currentText() or "").strip()
            except Exception:
                pname = ""

        if pname in ("", "(No project)", "➕ Add project…"):
            return None

        getter = getattr(self, "_get_project_id", None)
        if callable(getter):
            try:
                resolved = getter(pname)
                if resolved is not None:
                    self.current_project_id = int(resolved)
                    self.current_project_name = pname
                    return int(resolved)
            except Exception:
                pass
        return None


    def _ensure_difar_rays_table_compat(self, conn):
        """Call `_ensure_difar_rays_table` across legacy/new signatures."""
        ensure = getattr(self, "_ensure_difar_rays_table", None)
        if not callable(ensure):
            return
        try:
            n_params = len(inspect.signature(ensure).parameters)
        except Exception:
            n_params = 1
        if n_params == 0:
            ensure()
        else:
            ensure(conn)

    def _resolve_active_project_id(self):
        """Best-effort resolve of currently selected project id."""
        pid = getattr(self, "current_project_id", None)
        try:
            if pid is not None:
                return int(pid)
        except Exception:
            pass

        pname = (getattr(self, "current_project_name", None) or "").strip()
        if not pname and hasattr(self, "project_combo") and self.project_combo is not None:
            try:
                pname = (self.project_combo.currentText() or "").strip()
            except Exception:
                pname = ""

        if pname in ("", "(No project)", "➕ Add project…"):
            return None

        getter = getattr(self, "_get_project_id", None)
        if callable(getter):
            try:
                resolved = getter(pname)
                if resolved is not None:
                    self.current_project_id = int(resolved)
                    self.current_project_name = pname
                    return int(resolved)
            except Exception:
                pass
        return None


    def _ensure_difar_rays_table_compat(self, conn):
        """Call `_ensure_difar_rays_table` across legacy/new signatures."""
        ensure = getattr(self, "_ensure_difar_rays_table", None)
        if not callable(ensure):
            return
        try:
            n_params = len(inspect.signature(ensure).parameters)
        except Exception:
            n_params = 1
        if n_params == 0:
            ensure()
        else:
            ensure(conn)

    def _resolve_active_project_id(self):
        """Best-effort resolve of currently selected project id."""
        pid = getattr(self, "current_project_id", None)
        try:
            if pid is not None:
                return int(pid)
        except Exception:
            pass

        pname = (getattr(self, "current_project_name", None) or "").strip()
        if not pname and hasattr(self, "project_combo") and self.project_combo is not None:
            try:
                pname = (self.project_combo.currentText() or "").strip()
            except Exception:
                pname = ""

        if pname in ("", "(No project)", "➕ Add project…"):
            return None

        getter = getattr(self, "_get_project_id", None)
        if callable(getter):
            try:
                resolved = getter(pname)
                if resolved is not None:
                    self.current_project_id = int(resolved)
                    self.current_project_name = pname
                    return int(resolved)
            except Exception:
                pass
        return None


    def _ensure_difar_rays_table_compat(self, conn):
        """Call `_ensure_difar_rays_table` across legacy/new signatures."""
        ensure = getattr(self, "_ensure_difar_rays_table", None)
        if not callable(ensure):
            return
        try:
            n_params = len(inspect.signature(ensure).parameters)
        except Exception:
            n_params = 1
        if n_params == 0:
            ensure()
        else:
            ensure(conn)

    def _resolve_active_project_id(self):
        """Best-effort resolve of currently selected project id."""
        pid = getattr(self, "current_project_id", None)
        try:
            if pid is not None:
                return int(pid)
        except Exception:
            pass

        pname = (getattr(self, "current_project_name", None) or "").strip()
        if not pname and hasattr(self, "project_combo") and self.project_combo is not None:
            try:
                pname = (self.project_combo.currentText() or "").strip()
            except Exception:
                pname = ""

        if pname in ("", "(No project)", "➕ Add project…"):
            return None

        getter = getattr(self, "_get_project_id", None)
        if callable(getter):
            try:
                resolved = getter(pname)
                if resolved is not None:
                    self.current_project_id = int(resolved)
                    self.current_project_name = pname
                    return int(resolved)
            except Exception:
                pass
        return None


    def _ensure_difar_rays_table_compat(self, conn):
        """Call `_ensure_difar_rays_table` across legacy/new signatures."""
        ensure = getattr(self, "_ensure_difar_rays_table", None)
        if not callable(ensure):
            return
        try:
            n_params = len(inspect.signature(ensure).parameters)
        except Exception:
            n_params = 1
        if n_params == 0:
            ensure()
        else:
            ensure(conn)

    def _resolve_active_project_id(self):
        """Best-effort resolve of currently selected project id."""
        pid = getattr(self, "current_project_id", None)
        try:
            if pid is not None:
                return int(pid)
        except Exception:
            pass

        pname = (getattr(self, "current_project_name", None) or "").strip()
        if not pname and hasattr(self, "project_combo") and self.project_combo is not None:
            try:
                pname = (self.project_combo.currentText() or "").strip()
            except Exception:
                pname = ""

        if pname in ("", "(No project)", "➕ Add project…"):
            return None

        getter = getattr(self, "_get_project_id", None)
        if callable(getter):
            try:
                resolved = getter(pname)
                if resolved is not None:
                    self.current_project_id = int(resolved)
                    self.current_project_name = pname
                    return int(resolved)
            except Exception:
                pass
        return None


    def _ensure_difar_rays_table_compat(self, conn):
        """Call `_ensure_difar_rays_table` across legacy/new signatures."""
        ensure = getattr(self, "_ensure_difar_rays_table", None)
        if not callable(ensure):
            return
        try:
            n_params = len(inspect.signature(ensure).parameters)
        except Exception:
            n_params = 1
        if n_params == 0:
            ensure()
        else:
            ensure(conn)

    def _resolve_active_project_id(self):
        """Best-effort resolve of currently selected project id."""
        pid = getattr(self, "current_project_id", None)
        try:
            if pid is not None:
                return int(pid)
        except Exception:
            pass

        pname = (getattr(self, "current_project_name", None) or "").strip()
        if not pname and hasattr(self, "project_combo") and self.project_combo is not None:
            try:
                pname = (self.project_combo.currentText() or "").strip()
            except Exception:
                pname = ""

        if pname in ("", "(No project)", "➕ Add project…"):
            return None

        getter = getattr(self, "_get_project_id", None)
        if callable(getter):
            try:
                resolved = getter(pname)
                if resolved is not None:
                    self.current_project_id = int(resolved)
                    self.current_project_name = pname
                    return int(resolved)
            except Exception:
                pass
        return None


    def _ensure_difar_rays_table_compat(self, conn):
        """Call `_ensure_difar_rays_table` across legacy/new signatures."""
        ensure = getattr(self, "_ensure_difar_rays_table", None)
        if not callable(ensure):
            return
        try:
            n_params = len(inspect.signature(ensure).parameters)
        except Exception:
            n_params = 1
        if n_params == 0:
            ensure()
        else:
            ensure(conn)

    def _resolve_active_project_id(self):
        """Best-effort resolve of currently selected project id."""
        pid = getattr(self, "current_project_id", None)
        try:
            if pid is not None:
                return int(pid)
        except Exception:
            pass

        pname = (getattr(self, "current_project_name", None) or "").strip()
        if not pname and hasattr(self, "project_combo") and self.project_combo is not None:
            try:
                pname = (self.project_combo.currentText() or "").strip()
            except Exception:
                pname = ""

        if pname in ("", "(No project)", "➕ Add project…"):
            return None

        getter = getattr(self, "_get_project_id", None)
        if callable(getter):
            try:
                resolved = getter(pname)
                if resolved is not None:
                    self.current_project_id = int(resolved)
                    self.current_project_name = pname
                    return int(resolved)
            except Exception:
                pass
        return None


    def _ensure_difar_rays_table_compat(self, conn):
        """Call `_ensure_difar_rays_table` across legacy/new signatures."""
        ensure = getattr(self, "_ensure_difar_rays_table", None)
        if not callable(ensure):
            return
        try:
            n_params = len(inspect.signature(ensure).parameters)
        except Exception:
            n_params = 1
        if n_params == 0:
            ensure()
        else:
            ensure(conn)

    def _resolve_active_project_id(self):
        """Best-effort resolve of currently selected project id."""
        pid = getattr(self, "current_project_id", None)
        try:
            if pid is not None:
                return int(pid)
        except Exception:
            pass

        pname = (getattr(self, "current_project_name", None) or "").strip()
        if not pname and hasattr(self, "project_combo") and self.project_combo is not None:
            try:
                pname = (self.project_combo.currentText() or "").strip()
            except Exception:
                pname = ""

        if pname in ("", "(No project)", "➕ Add project…"):
            return None

        getter = getattr(self, "_get_project_id", None)
        if callable(getter):
            try:
                resolved = getter(pname)
                if resolved is not None:
                    self.current_project_id = int(resolved)
                    self.current_project_name = pname
                    return int(resolved)
            except Exception:
                pass
        return None


    def _ensure_difar_rays_table_compat(self, conn):
        """Call `_ensure_difar_rays_table` across legacy/new signatures."""
        ensure = getattr(self, "_ensure_difar_rays_table", None)
        if not callable(ensure):
            return
        try:
            n_params = len(inspect.signature(ensure).parameters)
        except Exception:
            n_params = 1
        if n_params == 0:
            ensure()
        else:
            ensure(conn)

    def _resolve_active_project_id(self):
        """Best-effort resolve of currently selected project id."""
        pid = getattr(self, "current_project_id", None)
        try:
            if pid is not None:
                return int(pid)
        except Exception:
            pass

        pname = (getattr(self, "current_project_name", None) or "").strip()
        if not pname and hasattr(self, "project_combo") and self.project_combo is not None:
            try:
                pname = (self.project_combo.currentText() or "").strip()
            except Exception:
                pname = ""

        if pname in ("", "(No project)", "➕ Add project…"):
            return None

        getter = getattr(self, "_get_project_id", None)
        if callable(getter):
            try:
                resolved = getter(pname)
                if resolved is not None:
                    self.current_project_id = int(resolved)
                    self.current_project_name = pname
                    return int(resolved)
            except Exception:
                pass
        return None


    def _ensure_difar_rays_table_compat(self, conn):
        """Call `_ensure_difar_rays_table` across legacy/new signatures."""
        ensure = getattr(self, "_ensure_difar_rays_table", None)
        if not callable(ensure):
            return
        try:
            n_params = len(inspect.signature(ensure).parameters)
        except Exception:
            n_params = 1
        if n_params == 0:
            ensure()
        else:
            ensure(conn)

    def _resolve_active_project_id(self):
        """Best-effort resolve of currently selected project id."""
        pid = getattr(self, "current_project_id", None)
        try:
            if pid is not None:
                return int(pid)
        except Exception:
            pass

        pname = (getattr(self, "current_project_name", None) or "").strip()
        if not pname and hasattr(self, "project_combo") and self.project_combo is not None:
            try:
                pname = (self.project_combo.currentText() or "").strip()
            except Exception:
                pname = ""

        if pname in ("", "(No project)", "➕ Add project…"):
            return None

        getter = getattr(self, "_get_project_id", None)
        if callable(getter):
            try:
                resolved = getter(pname)
                if resolved is not None:
                    self.current_project_id = int(resolved)
                    self.current_project_name = pname
                    return int(resolved)
            except Exception:
                pass
        return None


    def _ensure_difar_rays_table_compat(self, conn):
        """Call `_ensure_difar_rays_table` across legacy/new signatures."""
        ensure = getattr(self, "_ensure_difar_rays_table", None)
        if not callable(ensure):
            return
        try:
            n_params = len(inspect.signature(ensure).parameters)
        except Exception:
            n_params = 1
        if n_params == 0:
            ensure()
        else:
            ensure(conn)

    def _resolve_active_project_id(self):
        """Best-effort resolve of currently selected project id."""
        pid = getattr(self, "current_project_id", None)
        try:
            if pid is not None:
                return int(pid)
        except Exception:
            pass

        pname = (getattr(self, "current_project_name", None) or "").strip()
        if not pname and hasattr(self, "project_combo") and self.project_combo is not None:
            try:
                pname = (self.project_combo.currentText() or "").strip()
            except Exception:
                pname = ""

        if pname in ("", "(No project)", "➕ Add project…"):
            return None

        getter = getattr(self, "_get_project_id", None)
        if callable(getter):
            try:
                resolved = getter(pname)
                if resolved is not None:
                    self.current_project_id = int(resolved)
                    self.current_project_name = pname
                    return int(resolved)
            except Exception:
                pass
        return None


    def _ensure_difar_rays_table_compat(self, conn):
        """Call `_ensure_difar_rays_table` across legacy/new signatures."""
        ensure = getattr(self, "_ensure_difar_rays_table", None)
        if not callable(ensure):
            return
        try:
            n_params = len(inspect.signature(ensure).parameters)
        except Exception:
            n_params = 1
        if n_params == 0:
            ensure()
        else:
            ensure(conn)

    def _resolve_active_project_id(self):
        """Best-effort resolve of currently selected project id."""
        pid = getattr(self, "current_project_id", None)
        try:
            if pid is not None:
                return int(pid)
        except Exception:
            pass

        pname = (getattr(self, "current_project_name", None) or "").strip()
        if not pname and hasattr(self, "project_combo") and self.project_combo is not None:
            try:
                pname = (self.project_combo.currentText() or "").strip()
            except Exception:
                pname = ""

        if pname in ("", "(No project)", "➕ Add project…"):
            return None

        getter = getattr(self, "_get_project_id", None)
        if callable(getter):
            try:
                resolved = getter(pname)
                if resolved is not None:
                    self.current_project_id = int(resolved)
                    self.current_project_name = pname
                    return int(resolved)
            except Exception:
                pass
        return None


    def _ensure_difar_rays_table_compat(self, conn):
        """Call `_ensure_difar_rays_table` across legacy/new signatures."""
        ensure = getattr(self, "_ensure_difar_rays_table", None)
        if not callable(ensure):
            return
        try:
            n_params = len(inspect.signature(ensure).parameters)
        except Exception:
            n_params = 1
        if n_params == 0:
            ensure()
        else:
            ensure(conn)

    def _resolve_active_project_id(self):
        """Best-effort resolve of currently selected project id."""
        pid = getattr(self, "current_project_id", None)
        try:
            if pid is not None:
                return int(pid)
        except Exception:
            pass

        pname = (getattr(self, "current_project_name", None) or "").strip()
        if not pname and hasattr(self, "project_combo") and self.project_combo is not None:
            try:
                pname = (self.project_combo.currentText() or "").strip()
            except Exception:
                pname = ""

        if pname in ("", "(No project)", "➕ Add project…"):
            return None

        getter = getattr(self, "_get_project_id", None)
        if callable(getter):
            try:
                resolved = getter(pname)
                if resolved is not None:
                    self.current_project_id = int(resolved)
                    self.current_project_name = pname
                    return int(resolved)
            except Exception:
                pass
        return None


    @staticmethod
    def _make_difar_config_compat(**kwargs):
        """Build DifarConfig while tolerating older/newer parameter names."""
        params = set(inspect.signature(DifarConfig).parameters.keys())

        # Name compatibility across evolving difar_core versions.
        aliases = {
            "bearing_offset_deg": ["bearing_offset"],
            "min_directional_percentile": ["directional_gate_percentile", "directional_percentile_gate"],
            "bearing_smooth_frames": ["smooth_frames", "bearing_smoothing_frames"],
            "resolve_180_ambiguity": ["resolve_180", "resolve_left_right_ambiguity"],
            "swap_xy": ["swap_channels_xy"],
            "invert_x": ["flip_x"],
            "invert_y": ["flip_y"],
        }

        selected = {}

        # Direct-match keys first.
        for k, v in kwargs.items():
            if k in params:
                selected[k] = v

        # If canonical key isn't supported, try aliases accepted by this DifarConfig.
        for canonical, alts in aliases.items():
            if canonical in kwargs:
                if canonical in params:
                    selected[canonical] = kwargs[canonical]
                else:
                    for alt in alts:
                        if alt in params:
                            selected[alt] = kwargs[canonical]
                            break

        return DifarConfig(**selected)

    @staticmethod
    def _ensure_difar_results_table(conn):
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS difar_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_utc TEXT,
                wav_path TEXT,
                calibration_name TEXT,
                start_time_utc TEXT,
                omni_channel INTEGER,
                x_channel INTEGER,
                y_channel INTEGER,
                z_channel INTEGER,
                compass_path TEXT,
                sensor_lat REAL,
                sensor_lon REAL,
                target_profile_id INTEGER,
                target_classification TEXT,
                result_json TEXT
            )
            """
        )
        cur.execute("PRAGMA table_info(difar_results)")
        cols = {r[1] for r in cur.fetchall()}
        if "target_profile_id" not in cols:
            cur.execute("ALTER TABLE difar_results ADD COLUMN target_profile_id INTEGER")
        if "target_classification" not in cols:
            cur.execute("ALTER TABLE difar_results ADD COLUMN target_classification TEXT")
        conn.commit()

    @staticmethod
    def _ensure_difar_target_profiles_table(conn):
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS difar_target_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_utc TEXT,
                updated_utc TEXT,
                last_used_utc TEXT,
                use_count INTEGER DEFAULT 0,
                project_id INTEGER,
                profile_name TEXT NOT NULL,
                classification TEXT,
                band_lo_hz REAL,
                band_hi_hz REAL,
                target_bands_text TEXT,
                notes TEXT
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_difar_target_profiles_project ON difar_target_profiles(project_id, profile_name)")
        cur.execute("PRAGMA table_info(difar_target_profiles)")
        cols = {r[1] for r in cur.fetchall()}
        if "last_used_utc" not in cols:
            cur.execute("ALTER TABLE difar_target_profiles ADD COLUMN last_used_utc TEXT")
        if "use_count" not in cols:
            cur.execute("ALTER TABLE difar_target_profiles ADD COLUMN use_count INTEGER DEFAULT 0")
        cur.execute("SELECT COUNT(*) FROM difar_target_profiles")
        if int(cur.fetchone()[0]) == 0:
            now = datetime.now(timezone.utc).isoformat()
            seeds = [
                (now, now, None, 0, None, "Low Tonal Vessel", "vessel", 20.0, 80.0, "20-80", "Preloaded"),
                (now, now, None, 0, None, "Mid Broadband Machinery", "vessel", 80.0, 250.0, "80-250", "Preloaded"),
                (now, now, None, 0, None, "High Tonal / Transient", "unknown", 250.0, 600.0, "250-600", "Preloaded"),
            ]
            cur.executemany(
                """
                INSERT INTO difar_target_profiles
                (created_utc, updated_utc, last_used_utc, use_count, project_id, profile_name, classification, band_lo_hz, band_hi_hz, target_bands_text, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                seeds,
            )
        conn.commit()

    @staticmethod
    def _ensure_difar_rays_table(conn):
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS difar_map_rays (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_utc TEXT,
                project_id INTEGER,
                run_id INTEGER,
                label TEXT,
                sensor_lat REAL,
                sensor_lon REAL,
                lat2_json TEXT,
                lon2_json TEXT,
                time_s_json TEXT,
                bearing_true_deg_json TEXT
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_difar_map_rays_project ON difar_map_rays(project_id, created_utc)")
        conn.commit()

    @staticmethod
    def _ensure_difar_heatmap_table(conn):
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS difar_heatmap_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_utc TEXT,
                project_id INTEGER,
                run_id INTEGER,
                label TEXT,
                time_s_json TEXT,
                bearing_true_deg_json TEXT,
                confidence_json TEXT
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_difar_heatmap_project ON difar_heatmap_data(project_id, created_utc)")
        conn.commit()

    def _difar_calibration_import_dialog(self, parent=None, on_imported=None):
        dlg = QtWidgets.QDialog(parent or self)
        dlg.setWindowTitle("DIFAR Calibration Import")
        dlg.resize(620, 220)
        lay = QtWidgets.QVBoxLayout(dlg)

        form = QtWidgets.QFormLayout()
        cal_name_edit = QtWidgets.QLineEdit()
        cal_name_edit.setPlaceholderText("e.g., DIFAR_SN123_2026-01")
        form.addRow("Calibration Name:", cal_name_edit)

        cal_file_edit = QtWidgets.QLineEdit()
        cal_browse_btn = QtWidgets.QPushButton("Browse CSV/TSV")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(cal_file_edit)
        row.addWidget(cal_browse_btn)
        row_w = QtWidgets.QWidget()
        row_w.setLayout(row)
        form.addRow("Calibration File:", row_w)

        lay.addLayout(form)

        out = QtWidgets.QPlainTextEdit()
        out.setReadOnly(True)
        out.setMaximumHeight(70)
        lay.addWidget(out)

        btn_row = QtWidgets.QHBoxLayout()
        import_btn = QtWidgets.QPushButton("Import")
        close_btn = QtWidgets.QPushButton("Close")
        btn_row.addWidget(import_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(close_btn)
        lay.addLayout(btn_row)

        def _browse():
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                dlg,
                "Select DIFAR Calibration CSV",
                "",
                "CSV/TSV Files (*.csv *.tsv *.txt);;All Files (*)",
            )
            if path:
                cal_file_edit.setText(path)

        def _import():
            name = cal_name_edit.text().strip()
            path = cal_file_edit.text().strip()
            if not name:
                QtWidgets.QMessageBox.warning(dlg, "Missing Name", "Enter a calibration name.")
                return
            if not path or not os.path.isfile(path):
                QtWidgets.QMessageBox.warning(dlg, "Missing File", "Select a valid calibration CSV/TSV file.")
                return
            try:
                n = import_difar_calibration_csv_to_db(DB_FILENAME, path, name)
                out.appendPlainText(f"Imported calibration '{name}' with {n} rows.")
                if callable(on_imported):
                    on_imported(name)
            except Exception as e:
                out.appendPlainText(f"Import failed: {e}")
                QtWidgets.QMessageBox.critical(dlg, "Import Failed", str(e))

        cal_browse_btn.clicked.connect(_browse)
        import_btn.clicked.connect(_import)
        close_btn.clicked.connect(dlg.accept)
        dlg.exec_()

    def difar_processing_popup(self):
        """Popup for calibration import and DIFAR processing run."""
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("DIFAR Processing")
        dlg.resize(1500, 900)
        dlg.setWindowState(dlg.windowState() | QtCore.Qt.WindowMaximized)

        layout = QtWidgets.QVBoxLayout(dlg)

        gui_bg = "#19232d"
        gui_panel_bg = "#19232d"
        gui_fg = "#DDDDDD"
        gui_grid = "#666666"
        gui_border = "#8FA3B5"

        content_row = QtWidgets.QHBoxLayout()
        layout.addLayout(content_row, stretch=1)

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        content_row.addWidget(left_panel, stretch=3)

        right_panel = QtWidgets.QGroupBox("Calibration Curves")
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_panel.setMinimumWidth(520)
        content_row.addWidget(right_panel, stretch=2)

        import_row = QtWidgets.QHBoxLayout()
        import_btn = QtWidgets.QPushButton("Calibration Import...")
        sim_btn = QtWidgets.QPushButton("Open DIFAR Simulator...")
        import_row.addWidget(import_btn)
        import_row.addWidget(sim_btn)
        import_row.addStretch(1)
        left_layout.addLayout(import_row)

        proc_box = QtWidgets.QGroupBox("Processing Run")
        proc_form = QtWidgets.QFormLayout(proc_box)

        loaded_wav = getattr(self, "current_file_path", None)
        loaded_wav = loaded_wav if (loaded_wav and os.path.isfile(loaded_wav)) else None

        wav_edit = QtWidgets.QLineEdit()
        wav_browse = QtWidgets.QPushButton("Browse WAV")
        if loaded_wav:
            wav_edit.setText(loaded_wav)
            wav_info = QtWidgets.QLabel(f"Using loaded WAV: {os.path.basename(loaded_wav)}")
            wav_info.setToolTip(loaded_wav)
            proc_form.addRow("Input WAV:", wav_info)
        else:
            wav_row = QtWidgets.QHBoxLayout()
            wav_row.addWidget(wav_edit)
            wav_row.addWidget(wav_browse)
            wav_w = QtWidgets.QWidget()
            wav_w.setLayout(wav_row)
            proc_form.addRow("Input WAV:", wav_w)

        cal_combo = QtWidgets.QComboBox()
        proc_form.addRow("Calibration Set:", cal_combo)

        ch_row = QtWidgets.QHBoxLayout()
        omni_spin = QtWidgets.QSpinBox(); omni_spin.setMinimum(1); omni_spin.setMaximum(256); omni_spin.setValue(1); omni_spin.setPrefix("OMNI ")
        x_spin = QtWidgets.QSpinBox(); x_spin.setMinimum(1); x_spin.setMaximum(256); x_spin.setValue(2); x_spin.setPrefix("X ")
        y_spin = QtWidgets.QSpinBox(); y_spin.setMinimum(1); y_spin.setMaximum(256); y_spin.setValue(3); y_spin.setPrefix("Y ")
        z_spin = QtWidgets.QSpinBox(); z_spin.setMinimum(0); z_spin.setMaximum(256); z_spin.setValue(4); z_spin.setSpecialValueText("Z unused")
        ch_row.addWidget(omni_spin); ch_row.addWidget(x_spin); ch_row.addWidget(y_spin); ch_row.addWidget(z_spin)
        ch_w = QtWidgets.QWidget(); ch_w.setLayout(ch_row)
        proc_form.addRow("Channel Mapping:", ch_w)

        tune_row = QtWidgets.QHBoxLayout()
        swap_xy_chk = QtWidgets.QCheckBox("Swap X/Y")
        invert_x_chk = QtWidgets.QCheckBox("Invert X")
        invert_y_chk = QtWidgets.QCheckBox("Invert Y")
        tune_row.addWidget(swap_xy_chk); tune_row.addWidget(invert_x_chk); tune_row.addWidget(invert_y_chk)
        tune_w = QtWidgets.QWidget(); tune_w.setLayout(tune_row)
        proc_form.addRow("Bearing Convention:", tune_w)

        offs_spin = QtWidgets.QDoubleSpinBox(); offs_spin.setRange(-360.0, 360.0); offs_spin.setDecimals(2); offs_spin.setSingleStep(1.0)
        proc_form.addRow("Bearing Offset (deg):", offs_spin)

        gate_spin = QtWidgets.QDoubleSpinBox(); gate_spin.setRange(0.0, 99.9); gate_spin.setDecimals(1); gate_spin.setValue(20.0)
        proc_form.addRow("Directional Gate Percentile:", gate_spin)

        smooth_spin = QtWidgets.QSpinBox(); smooth_spin.setRange(1, 99); smooth_spin.setValue(5)
        proc_form.addRow("Bearing Smooth Frames:", smooth_spin)

        ambig_chk = QtWidgets.QCheckBox("Resolve ±180° ambiguity")
        ambig_chk.setChecked(True)
        proc_form.addRow("", ambig_chk)

        omni_ambig_chk = QtWidgets.QCheckBox("Use OMNI + phase for ±180 disambiguation")
        omni_ambig_chk.setChecked(True)
        proc_form.addRow("", omni_ambig_chk)

        start_dt = QtWidgets.QDateTimeEdit()
        start_dt.setCalendarPopup(True)
        start_dt.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        start_dt.setDateTime(QtCore.QDateTime.currentDateTimeUtc())
        proc_form.addRow("WAV Start UTC:", start_dt)

        compass_edit = QtWidgets.QLineEdit()
        compass_browse = QtWidgets.QPushButton("Browse Compass CSV")
        compass_row = QtWidgets.QHBoxLayout(); compass_row.addWidget(compass_edit); compass_row.addWidget(compass_browse)
        compass_w = QtWidgets.QWidget(); compass_w.setLayout(compass_row)
        proc_form.addRow("Compass CSV (opt):", compass_w)

        export_edit = QtWidgets.QLineEdit()
        export_browse = QtWidgets.QPushButton("Export CSV")
        export_row = QtWidgets.QHBoxLayout(); export_row.addWidget(export_edit); export_row.addWidget(export_browse)
        export_w = QtWidgets.QWidget(); export_w.setLayout(export_row)
        proc_form.addRow("Output CSV (opt):", export_w)

        chunk_chk = QtWidgets.QCheckBox("Chunk long files (recommended)")
        chunk_chk.setChecked(True)
        proc_form.addRow("Long-file mode:", chunk_chk)

        chunk_row = QtWidgets.QHBoxLayout()
        chunk_minutes = QtWidgets.QDoubleSpinBox(); chunk_minutes.setRange(1.0, 180.0); chunk_minutes.setDecimals(1); chunk_minutes.setValue(15.0); chunk_minutes.setSuffix(" min")
        overlap_seconds = QtWidgets.QDoubleSpinBox(); overlap_seconds.setRange(0.0, 120.0); overlap_seconds.setDecimals(1); overlap_seconds.setValue(2.0); overlap_seconds.setSuffix(" s overlap")
        chunk_row.addWidget(chunk_minutes); chunk_row.addWidget(overlap_seconds)
        chunk_w = QtWidgets.QWidget(); chunk_w.setLayout(chunk_row)
        proc_form.addRow("Chunk settings:", chunk_w)

        band_row = QtWidgets.QHBoxLayout()
        band_lo = QtWidgets.QDoubleSpinBox(); band_lo.setRange(1.0, 200000.0); band_lo.setDecimals(1); band_lo.setValue(20.0)
        band_hi = QtWidgets.QDoubleSpinBox(); band_hi.setRange(1.0, 200000.0); band_hi.setDecimals(1); band_hi.setValue(500.0)
        band_row.addWidget(QtWidgets.QLabel("Lo")); band_row.addWidget(band_lo)
        band_row.addWidget(QtWidgets.QLabel("Hi")); band_row.addWidget(band_hi)
        band_w = QtWidgets.QWidget(); band_w.setLayout(band_row)
        proc_form.addRow("Bandpass (Hz):", band_w)

        adc_fs_spin = QtWidgets.QDoubleSpinBox()
        adc_fs_spin.setRange(0.001, 1000.0)
        adc_fs_spin.setDecimals(3)
        adc_fs_spin.setSingleStep(0.1)
        adc_fs_spin.setValue(1.0)
        adc_fs_spin.setSuffix(" V FS")
        adc_fs_spin.setToolTip("Normalized sample scaling to volts before calibration (set to DAQ full-scale volts if WAV is normalized).")
        proc_form.addRow("DAQ full-scale:", adc_fs_spin)

        target_profile_summary = QtWidgets.QLabel("No target profile selected (using Bandpass above).")
        target_profile_summary.setWordWrap(True)
        manage_profiles_btn = QtWidgets.QPushButton("Target Profiles...")
        prof_row = QtWidgets.QHBoxLayout(); prof_row.addWidget(target_profile_summary, 1); prof_row.addWidget(manage_profiles_btn)
        prof_w = QtWidgets.QWidget(); prof_w.setLayout(prof_row)
        proc_form.addRow("Target profile:", prof_w)

        lat_edit = QtWidgets.QLineEdit(); lat_edit.setPlaceholderText("sensor latitude")
        lon_edit = QtWidgets.QLineEdit(); lon_edit.setPlaceholderText("sensor longitude")
        latlon_row = QtWidgets.QHBoxLayout(); latlon_row.addWidget(lat_edit); latlon_row.addWidget(lon_edit)
        latlon_w = QtWidgets.QWidget(); latlon_w.setLayout(latlon_row)
        proc_form.addRow("Static map rays (opt):", latlon_w)

        show_on_chart_chk = QtWidgets.QCheckBox("Display DIFAR on Chart map (sensor + rays)")
        show_on_chart_chk.setChecked(True)
        proc_form.addRow("", show_on_chart_chk)

        animate_map_chk = QtWidgets.QCheckBox("Animate DIFAR detections on Chart map")
        animate_map_chk.setChecked(False)
        proc_form.addRow("", animate_map_chk)

        anim_step_spin = QtWidgets.QSpinBox(); anim_step_spin.setRange(1, 200); anim_step_spin.setValue(8); anim_step_spin.setSuffix(" rays/step")
        anim_ms_spin = QtWidgets.QSpinBox(); anim_ms_spin.setRange(20, 5000); anim_ms_spin.setValue(250); anim_ms_spin.setSuffix(" ms")
        anim_row = QtWidgets.QHBoxLayout(); anim_row.addWidget(anim_step_spin); anim_row.addWidget(anim_ms_spin)
        anim_w = QtWidgets.QWidget(); anim_w.setLayout(anim_row)
        proc_form.addRow("Animation settings:", anim_w)

        save_db_chk = QtWidgets.QCheckBox("Save analyzed DIFAR output to database")
        save_db_chk.setChecked(True)
        proc_form.addRow("", save_db_chk)

        run_btn = QtWidgets.QPushButton("Run DIFAR Processing")
        proc_form.addRow("", run_btn)

        left_layout.addWidget(proc_box)

        out = QtWidgets.QPlainTextEdit()
        out.setReadOnly(True)
        out.setPlaceholderText("Status output...")
        left_layout.addWidget(out, stretch=1)

        heat_btn_row = QtWidgets.QHBoxLayout()
        open_heatmap_btn = QtWidgets.QPushButton("Open Time vs Bearing Heatmap...")
        open_difargram_btn = QtWidgets.QPushButton("Open DIFARGram Display...")
        heat_btn_row.addWidget(open_heatmap_btn)
        heat_btn_row.addWidget(open_difargram_btn)
        heat_btn_row.addStretch(1)
        left_layout.addLayout(heat_btn_row)

        close_btn = QtWidgets.QPushButton("Close")
        close_row = QtWidgets.QHBoxLayout(); close_row.addStretch(1); close_row.addWidget(close_btn)
        left_layout.addLayout(close_row)

        cal_canvas = None
        cal_fig = None
        cal_ax_motion = None
        cal_ax_omni = None
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            cal_fig = Figure(figsize=(5.4, 7.0), facecolor="#19232d")
            cal_canvas = FigureCanvas(cal_fig)
            try:
                cal_canvas.setStyleSheet("background-color: #19232d;")
            except Exception:
                pass
            cal_ax_motion = cal_fig.add_subplot(2, 1, 1)
            cal_ax_omni = cal_fig.add_subplot(2, 1, 2)
            right_layout.addWidget(cal_canvas, stretch=1)
        except Exception:
            out.appendPlainText("Calibration plots unavailable (matplotlib Qt backend not available).")

        def _palette_for_plots(count: int):
            try:
                if hasattr(self, "_ordered_palette"):
                    pal = [str(c) for c in (self._ordered_palette() or []) if c]
                else:
                    pal = []
            except Exception:
                pal = []
            if not pal:
                base = str(getattr(self, "graph_color", "#03DFE2"))
                pal = [base, "#80E27E", "#FFB347", "#C77DFF", "#FFD166"]
            out_cols = []
            for i in range(max(1, int(count))):
                out_cols.append(pal[i % len(pal)])
            return out_cols

        def _style_cal_axes(ax):
            ax.set_facecolor("#19232d")
            ax.grid(True, alpha=0.25)
            ax.tick_params(colors="#DDDDDD")
            for sp in ax.spines.values():
                sp.set_color("#666666")
            ax.xaxis.label.set_color("#DDDDDD")
            ax.yaxis.label.set_color("#DDDDDD")
            ax.title.set_color("#DDDDDD")

        def _update_calibration_plots(*_args):
            if cal_canvas is None or cal_fig is None:
                return
            cal_ax_motion.clear(); cal_ax_omni.clear()
            _style_cal_axes(cal_ax_motion); _style_cal_axes(cal_ax_omni)
            cal_ax_motion.set_title("Particle Velocity Calibration (X/Y/Z)")
            cal_ax_omni.set_title("OMNI Pressure Calibration")
            cal_ax_motion.set_xlabel("Frequency (Hz)")
            cal_ax_motion.set_ylabel("Sensitivity (dB)")
            cal_ax_omni.set_xlabel("Frequency (Hz)")
            cal_ax_omni.set_ylabel("Sensitivity (dB)")

            cal_name = cal_combo.currentData() or cal_combo.currentText().strip()
            if not cal_name:
                cal_canvas.draw_idle()
                return

            try:
                cal = load_difar_calibration_from_db(DB_FILENAME, cal_name)
            except Exception as e:
                out.appendPlainText(f"Could not load calibration '{cal_name}': {e}")
                cal_canvas.draw_idle()
                return

            cols = _palette_for_plots(4)
            plotted_motion = False
            if getattr(cal, "x", None) is not None:
                cal_ax_motion.plot(cal.x.freq_hz, cal.x.sensitivity_db, color=cols[0], linewidth=2.0, label="X")
                plotted_motion = True
            if getattr(cal, "y", None) is not None:
                cal_ax_motion.plot(cal.y.freq_hz, cal.y.sensitivity_db, color=cols[1], linewidth=2.0, label="Y")
                plotted_motion = True
            if getattr(cal, "z", None) is not None:
                cal_ax_motion.plot(cal.z.freq_hz, cal.z.sensitivity_db, color=cols[2], linewidth=2.0, label="Z")
                plotted_motion = True
            if plotted_motion:
                cal_ax_motion.legend(loc="best", framealpha=0.35)
            else:
                cal_ax_motion.text(0.5, 0.5, "No X/Y/Z data", ha="center", va="center", transform=cal_ax_motion.transAxes, color="#DDDDDD")

            if getattr(cal, "omni", None) is not None:
                cal_ax_omni.plot(cal.omni.freq_hz, cal.omni.sensitivity_db, color=cols[3], linewidth=2.2, label="OMNI")
                cal_ax_omni.legend(loc="best", framealpha=0.35)
            else:
                cal_ax_omni.text(0.5, 0.5, "No OMNI data", ha="center", va="center", transform=cal_ax_omni.transAxes, color="#DDDDDD")

            cal_fig.tight_layout(pad=1.2)
            cal_canvas.draw_idle()

        def _safe_seq(value):
            if value is None:
                return []
            try:
                return list(value)
            except Exception:
                return []

        def _result_to_heatmap_payload(result: dict):
            time_raw = result.get("time_s", [])
            bearing_raw = result.get("bearing_true_deg", [])
            conf_raw = result.get("confidence", [])

            if time_raw is None:
                time_raw = []
            if bearing_raw is None:
                bearing_raw = []
            if conf_raw is None:
                conf_raw = []

            time_s = [float(v) for v in list(time_raw)]
            bearing = [float(v) % 360.0 for v in list(bearing_raw)]
            conf_raw = list(conf_raw)
            n = min(len(time_s), len(bearing))
            if n <= 1:
                return None
            conf = []
            for i in range(n):
                try:
                    conf.append(float(conf_raw[i]))
                except Exception:
                    conf.append(1.0)
            return {
                "time_s": time_s[:n],
                "bearing_true_deg": bearing[:n],
                "confidence": conf,
            }

        def _save_heatmap_payload(payload: dict, run_id: int, label: str):
            if payload is None:
                return
            try:
                project_id = self._resolve_active_project_id()
                conn = sqlite3.connect(DB_FILENAME)
                self._ensure_difar_heatmap_table(conn)
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO difar_heatmap_data (
                        created_utc, project_id, run_id, label,
                        time_s_json, bearing_true_deg_json, confidence_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        datetime.now(timezone.utc).isoformat(),
                        project_id,
                        run_id,
                        label,
                        json.dumps([float(v) for v in payload.get("time_s", [])]),
                        json.dumps([float(v) for v in payload.get("bearing_true_deg", [])]),
                        json.dumps([float(v) for v in payload.get("confidence", [])]),
                    ),
                )
                conn.commit()
                conn.close()
            except Exception as e:
                out.appendPlainText(f"Warning: could not save heatmap data to DB: {e}")

        def _open_heatmap_popup(initial_payload: dict = None, initial_label: str = ""):
            pop = QtWidgets.QDialog(dlg)
            pop.setWindowTitle("DIFAR Time vs Bearing Heatmap")
            pop.resize(1100, 720)
            lay = QtWidgets.QVBoxLayout(pop)

            ctl = QtWidgets.QHBoxLayout()
            series_combo = QtWidgets.QComboBox()
            refresh_series_btn = QtWidgets.QPushButton("Refresh")
            load_btn = QtWidgets.QPushButton("Load Selected")
            save_jpg_btn = QtWidgets.QPushButton("Save JPG...")
            play_btn = QtWidgets.QPushButton("▶ Play")
            stop_btn = QtWidgets.QPushButton("■ Stop")
            play_step_spin = QtWidgets.QSpinBox(); play_step_spin.setRange(1, 200); play_step_spin.setValue(8); play_step_spin.setSuffix(" bins/step")
            play_ms_spin = QtWidgets.QSpinBox(); play_ms_spin.setRange(20, 3000); play_ms_spin.setValue(200); play_ms_spin.setSuffix(" ms")
            t_bins = QtWidgets.QSpinBox(); t_bins.setRange(10, 600); t_bins.setValue(120)
            b_bins = QtWidgets.QSpinBox(); b_bins.setRange(12, 360); b_bins.setValue(72)
            metric_combo = QtWidgets.QComboBox(); metric_combo.addItems(["Count", "Confidence-weighted"])
            ctl.addWidget(QtWidgets.QLabel("Saved runs")); ctl.addWidget(series_combo, 1)
            ctl.addWidget(refresh_series_btn); ctl.addWidget(load_btn); ctl.addWidget(save_jpg_btn)
            ctl.addWidget(play_btn); ctl.addWidget(stop_btn)
            ctl.addWidget(play_step_spin); ctl.addWidget(play_ms_spin)
            ctl.addWidget(QtWidgets.QLabel("Time bins")); ctl.addWidget(t_bins)
            ctl.addWidget(QtWidgets.QLabel("Bearing bins")); ctl.addWidget(b_bins)
            ctl.addWidget(QtWidgets.QLabel("Color metric")); ctl.addWidget(metric_combo)
            lay.addLayout(ctl)

            status_lbl = QtWidgets.QLabel("Select a saved run (filtered by active project) or process new data.")
            status_lbl.setWordWrap(True)
            lay.addWidget(status_lbl)

            canvas = None
            fig = None
            ax = None
            cax = None
            try:
                from matplotlib.figure import Figure
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
                fig = Figure(facecolor=gui_panel_bg)
                canvas = FigureCanvas(fig)
                gs = fig.add_gridspec(1, 2, width_ratios=[30, 1], wspace=0.08)
                ax = fig.add_subplot(gs[0, 0])
                cax = fig.add_subplot(gs[0, 1])
                lay.addWidget(canvas, 1)
            except Exception:
                status_lbl.setText("Heatmap unavailable (matplotlib Qt backend not available).")

            current_payload = initial_payload if initial_payload is not None else getattr(self, "_difar_last_heatmap_payload", None)
            current_label = initial_label if initial_label else getattr(self, "_difar_last_heatmap_label", "latest run")
            play_timer = QtCore.QTimer(pop)
            play_state = {"idx": 0}

            def _render(payload: dict, label: str, max_time_bin: int = None):
                nonlocal current_payload, current_label
                if canvas is None or ax is None or fig is None or payload is None:
                    return
                time_s = [float(v) for v in payload.get("time_s", [])]
                bearing = [float(v) % 360.0 for v in payload.get("bearing_true_deg", [])]
                conf = [float(v) for v in payload.get("confidence", [])]
                n = min(len(time_s), len(bearing), len(conf))
                if n <= 1:
                    status_lbl.setText("No data available for heatmap.")
                    return
                time_s = time_s[:n]; bearing = bearing[:n]; conf = conf[:n]
                t_min, t_max = min(time_s), max(time_s)
                if t_max <= t_min:
                    t_max = t_min + 1.0
                tb = int(t_bins.value()); bb = int(b_bins.value())
                weighted = (metric_combo.currentText() == "Confidence-weighted")
                grid = [[0.0 for _ in range(tb)] for _ in range(bb)]
                span = (t_max - t_min)
                used = 0
                for i in range(n):
                    ti = int((time_s[i] - t_min) / span * tb)
                    bi = int((bearing[i] / 360.0) * bb)
                    ti = 0 if ti < 0 else (tb - 1 if ti >= tb else ti)
                    bi = 0 if bi < 0 else (bb - 1 if bi >= bb else bi)
                    if max_time_bin is not None and ti > int(max_time_bin):
                        continue
                    grid[bi][ti] += (conf[i] if weighted else 1.0)
                    used += 1

                ax.clear()
                ax.set_facecolor(gui_bg)
                im = ax.imshow(grid, origin="lower", aspect="auto", extent=[t_min, t_max, 0.0, 360.0], interpolation="nearest", cmap="plasma")
                ax.set_xlabel("Time (s)", color=gui_fg)
                ax.set_ylabel("Bearing True (deg)", color=gui_fg)
                ax.set_title(f"DIFAR Heatmap [{label}]", color=gui_fg)
                ax.tick_params(colors=gui_fg)
                for sp in ax.spines.values():
                    sp.set_color(gui_border)
                    sp.set_linewidth(1.4)

                if cax is not None:
                    cax.clear()
                    cb = fig.colorbar(im, cax=cax)
                else:
                    cb = fig.colorbar(im, ax=ax)
                cb.set_label("Weighted density" if weighted else "Sample count", color=gui_fg)
                cb.ax.yaxis.set_tick_params(color=gui_fg)
                for tick in cb.ax.get_yticklabels():
                    tick.set_color(gui_fg)
                canvas.draw_idle()
                metric = "confidence-weighted" if weighted else "count"
                status_lbl.setText(f"Showing: {label} | {used}/{n} samples | {tb}x{bb} bins | {metric}")
                current_payload = payload
                current_label = label

            def _load_saved_series():
                series_combo.clear()
                try:
                    pid = self._resolve_active_project_id()
                    conn = sqlite3.connect(DB_FILENAME)
                    self._ensure_difar_heatmap_table(conn)
                    cur = conn.cursor()
                    cur.execute(
                        """
                        SELECT id, created_utc, COALESCE(label, ''), COALESCE(run_id, -1)
                        FROM difar_heatmap_data
                        WHERE (project_id IS NULL AND ? IS NULL) OR project_id = ?
                        ORDER BY id DESC
                        LIMIT 300
                        """,
                        (pid, pid),
                    )
                    rows = cur.fetchall()
                    conn.close()
                except Exception as e:
                    rows = []
                    status_lbl.setText(f"Could not load saved heatmaps: {e}")
                for hid, created, label, run_id in rows:
                    text = f"#{hid} | run {run_id if int(run_id) >= 0 else '-'} | {created} | {label}"
                    series_combo.addItem(text, int(hid))
                if series_combo.count() == 0:
                    series_combo.addItem("(No saved heatmaps for current project)", None)

            def _load_selected():
                hid = series_combo.currentData()
                if hid is None:
                    return
                try:
                    conn = sqlite3.connect(DB_FILENAME)
                    cur = conn.cursor()
                    cur.execute(
                        """
                        SELECT label, time_s_json, bearing_true_deg_json, confidence_json
                        FROM difar_heatmap_data WHERE id = ?
                        """,
                        (int(hid),),
                    )
                    row = cur.fetchone()
                    conn.close()
                    if not row:
                        status_lbl.setText("Saved heatmap row not found.")
                        return
                    label, t_js, b_js, c_js = row
                    payload = {
                        "time_s": json.loads(t_js or "[]"),
                        "bearing_true_deg": json.loads(b_js or "[]"),
                        "confidence": json.loads(c_js or "[]"),
                    }
                    _render(payload, label or f"saved #{hid}")
                except Exception as e:
                    status_lbl.setText(f"Failed loading saved heatmap: {e}")

            def _stop_heatmap_playback():
                if play_timer.isActive():
                    play_timer.stop()
                play_state["idx"] = 0

            def _play_heatmap():
                if current_payload is None:
                    status_lbl.setText("Nothing to animate. Load or process a heatmap first.")
                    return
                tb = int(t_bins.value())
                step = max(1, int(play_step_spin.value()))
                interval = max(20, int(play_ms_spin.value()))
                _stop_heatmap_playback()
                play_state["idx"] = 0

                def _tick():
                    i = int(play_state["idx"])
                    if i >= tb:
                        play_timer.stop()
                        status_lbl.setText("Heatmap animation complete.")
                        return
                    _render(current_payload, current_label, max_time_bin=i)
                    play_state["idx"] = i + step

                try:
                    play_timer.timeout.disconnect()
                except Exception:
                    pass
                play_timer.timeout.connect(_tick)
                _tick()
                play_timer.start(interval)

            def _save_current_heatmap_jpg():
                if canvas is None or fig is None or current_payload is None:
                    status_lbl.setText("Nothing to save yet. Load or render a heatmap first.")
                    return
                default_dir = ""
                if hasattr(self, "_project_subdir"):
                    try:
                        default_dir = self._project_subdir("difar_heatmaps") or ""
                    except Exception:
                        default_dir = ""
                if not default_dir:
                    default_dir = os.getcwd()

                raw = (current_label or "difar_heatmap").strip() or "difar_heatmap"
                safe = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in raw).strip("_") or "difar_heatmap"
                default_path = os.path.join(default_dir, f"{safe}.jpg")
                path, _ = QtWidgets.QFileDialog.getSaveFileName(
                    pop,
                    "Save DIFAR Heatmap as JPG",
                    default_path,
                    "JPEG Files (*.jpg *.jpeg)",
                )
                if not path:
                    return
                if not path.lower().endswith((".jpg", ".jpeg")):
                    path = f"{path}.jpg"
                try:
                    fig.savefig(path, format="jpg", dpi=180, bbox_inches="tight")
                    status_lbl.setText(f"Saved heatmap JPG: {path}")
                except Exception as e:
                    status_lbl.setText(f"Failed to save heatmap JPG: {e}")

            refresh_series_btn.clicked.connect(_load_saved_series)
            load_btn.clicked.connect(_load_selected)
            save_jpg_btn.clicked.connect(_save_current_heatmap_jpg)
            play_btn.clicked.connect(_play_heatmap)
            stop_btn.clicked.connect(lambda *_: (_stop_heatmap_playback(), _render(current_payload, current_label) if current_payload is not None else None))
            t_bins.valueChanged.connect(lambda *_: (_stop_heatmap_playback(), _render(current_payload, current_label) if current_payload is not None else None))
            b_bins.valueChanged.connect(lambda *_: (_stop_heatmap_playback(), _render(current_payload, current_label) if current_payload is not None else None))
            metric_combo.currentIndexChanged.connect(lambda *_: (_stop_heatmap_playback(), _render(current_payload, current_label) if current_payload is not None else None))

            _load_saved_series()
            if current_payload is not None:
                _render(current_payload, current_label)
            pop.finished.connect(lambda *_: _stop_heatmap_playback())
            pop.exec_()

        def _open_difargram_popup():
            pop = QtWidgets.QDialog(dlg)
            pop.setWindowTitle("DIFARGram-style Display")
            pop.resize(1250, 820)
            pop.setWindowState(pop.windowState() | QtCore.Qt.WindowMaximized)
            lay = QtWidgets.QVBoxLayout(pop)

            ctl_top = QtWidgets.QHBoxLayout()
            ctl_bottom = QtWidgets.QHBoxLayout()
            start_sec = QtWidgets.QDoubleSpinBox(); start_sec.setRange(0.0, 86400.0); start_sec.setDecimals(2); start_sec.setValue(0.0); start_sec.setSuffix(" s start")
            win_sec = QtWidgets.QDoubleSpinBox(); win_sec.setRange(2.0, 600.0); win_sec.setDecimals(1); win_sec.setValue(60.0); win_sec.setSuffix(" s window")
            max_freq = QtWidgets.QDoubleSpinBox(); max_freq.setRange(100.0, 100000.0); max_freq.setDecimals(1); max_freq.setValue(1500.0); max_freq.setSuffix(" Hz max")
            nfft_combo = QtWidgets.QComboBox(); nfft_combo.addItems(["512", "1024", "2048", "4096", "8192"]); nfft_combo.setCurrentText("1024")
            spec_cmap_combo = QtWidgets.QComboBox(); spec_cmap_combo.addItems(["inferno", "magma", "plasma", "viridis", "cividis", "turbo", "gray", "gray_r", "inferno_r", "viridis_r", "turbo_r"]); spec_cmap_combo.setCurrentText("magma")
            smooth_mode_combo = QtWidgets.QComboBox(); smooth_mode_combo.addItems(["None", "Moving average"]); smooth_mode_combo.setCurrentText("Moving average")
            smooth_win_spin = QtWidgets.QSpinBox(); smooth_win_spin.setRange(1, 101); smooth_win_spin.setValue(40); smooth_win_spin.setSuffix(" pts")
            spec_ymin_spin = QtWidgets.QDoubleSpinBox(); spec_ymin_spin.setRange(0.0, 100000.0); spec_ymin_spin.setDecimals(1); spec_ymin_spin.setValue(0.0); spec_ymin_spin.setSuffix(" Hz")
            spec_ymax_spin = QtWidgets.QDoubleSpinBox(); spec_ymax_spin.setRange(1.0, 100000.0); spec_ymax_spin.setDecimals(1); spec_ymax_spin.setValue(1500.0); spec_ymax_spin.setSuffix(" Hz")
            spec_axes_combo = QtWidgets.QComboBox(); spec_axes_combo.addItems(["Time→X | Freq→Y", "Freq→X | Time→Y"]); spec_axes_combo.setCurrentIndex(0)
            track_mode_combo = QtWidgets.QComboBox(); track_mode_combo.addItems(["Full-band track", "Processed-band track", "Custom-band track"]); track_mode_combo.setCurrentText("Processed-band track")
            track_lo_hz_spin = QtWidgets.QDoubleSpinBox(); track_lo_hz_spin.setRange(0.0, 100000.0); track_lo_hz_spin.setDecimals(1); track_lo_hz_spin.setValue(0.0); track_lo_hz_spin.setSuffix(" Hz")
            track_hi_hz_spin = QtWidgets.QDoubleSpinBox(); track_hi_hz_spin.setRange(1.0, 100000.0); track_hi_hz_spin.setDecimals(1); track_hi_hz_spin.setValue(1500.0); track_hi_hz_spin.setSuffix(" Hz")
            amp_min_db_spin = QtWidgets.QDoubleSpinBox(); amp_min_db_spin.setRange(-240.0, 120.0); amp_min_db_spin.setDecimals(1); amp_min_db_spin.setSingleStep(1.0); amp_min_db_spin.setValue(-120.0); amp_min_db_spin.setSuffix(" dB min")
            suggest_amp_btn = QtWidgets.QPushButton("Suggest Amp")
            color_band_halfwidth_hz = QtWidgets.QDoubleSpinBox(); color_band_halfwidth_hz.setRange(0.0, 5000.0); color_band_halfwidth_hz.setDecimals(1); color_band_halfwidth_hz.setValue(50.0); color_band_halfwidth_hz.setSuffix(" Hz")
            render_btn = QtWidgets.QPushButton("Render")
            save_btn = QtWidgets.QPushButton("Save JPG...")
            ctl_top.addWidget(QtWidgets.QLabel("Segment:")); ctl_top.addWidget(start_sec); ctl_top.addWidget(win_sec)
            ctl_top.addWidget(max_freq); ctl_top.addWidget(QtWidgets.QLabel("NFFT")); ctl_top.addWidget(nfft_combo)
            ctl_top.addWidget(QtWidgets.QLabel("Spec axes")); ctl_top.addWidget(spec_axes_combo)
            ctl_top.addWidget(QtWidgets.QLabel("Theme")); ctl_top.addWidget(spec_cmap_combo)
            ctl_top.addWidget(QtWidgets.QLabel("Spec Y")); ctl_top.addWidget(spec_ymin_spin); ctl_top.addWidget(QtWidgets.QLabel("to")); ctl_top.addWidget(spec_ymax_spin)
            ctl_top.addStretch(1)

            ctl_bottom.addWidget(QtWidgets.QLabel("Bearing smooth")); ctl_bottom.addWidget(smooth_mode_combo); ctl_bottom.addWidget(smooth_win_spin)
            ctl_bottom.addWidget(QtWidgets.QLabel("Track")); ctl_bottom.addWidget(track_mode_combo)
            ctl_bottom.addWidget(QtWidgets.QLabel("Track Hz")); ctl_bottom.addWidget(track_lo_hz_spin); ctl_bottom.addWidget(QtWidgets.QLabel("to")); ctl_bottom.addWidget(track_hi_hz_spin)
            ctl_bottom.addWidget(QtWidgets.QLabel("Amp")); ctl_bottom.addWidget(amp_min_db_spin); ctl_bottom.addWidget(suggest_amp_btn)
            ctl_bottom.addWidget(QtWidgets.QLabel("Color ±Hz")); ctl_bottom.addWidget(color_band_halfwidth_hz)
            ctl_bottom.addWidget(render_btn); ctl_bottom.addWidget(save_btn)
            ctl_bottom.addStretch(1)

            lay.addLayout(ctl_top)
            lay.addLayout(ctl_bottom)

            status_lbl = QtWidgets.QLabel("DIFARGram view of latest DIFAR run: spectrogram + bearing track.")
            status_lbl.setWordWrap(True)
            lay.addWidget(status_lbl)

            tabs = QtWidgets.QTabWidget()
            plot_tab = QtWidgets.QWidget()
            plot_lay = QtWidgets.QVBoxLayout(plot_tab)
            plot_lay.setContentsMargins(0, 0, 0, 0)
            table_tab = QtWidgets.QWidget()
            table_lay = QtWidgets.QVBoxLayout(table_tab)
            table_lay.setContentsMargins(0, 0, 0, 0)

            detection_table = QtWidgets.QTableWidget()
            detection_table.setColumnCount(12)
            detection_table.setHorizontalHeaderLabels([
                "Time Offset (s)",
                "Bearing (deg)",
                "Detected Freq (Hz)",
                "Peak Amp (dB)",
                "OMNI SPL (dB re 1µPa)",
                "X RMS (m/s)",
                "X RMS (µm/s)",
                "Y RMS (m/s)",
                "Y RMS (µm/s)",
                "Z RMS (m/s)",
                "Z RMS (µm/s)",
                "Mode",
            ])
            detection_table.horizontalHeader().setStretchLastSection(True)
            detection_table.setAlternatingRowColors(True)
            detection_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            export_csv_btn = QtWidgets.QPushButton("Export CSV...")
            table_btn_row = QtWidgets.QHBoxLayout()
            table_btn_row.addStretch(1)
            table_btn_row.addWidget(export_csv_btn)
            table_lay.addLayout(table_btn_row)
            table_lay.addWidget(detection_table, 1)

            tabs.addTab(plot_tab, "Plots")
            tabs.addTab(table_tab, "Data Table")
            lay.addWidget(tabs, 1)

            def _sync_track_controls():
                is_custom = (track_mode_combo.currentText() == "Custom-band track")
                track_lo_hz_spin.setEnabled(is_custom)
                track_hi_hz_spin.setEnabled(is_custom)

            track_mode_combo.currentIndexChanged.connect(lambda *_: _sync_track_controls())
            _sync_track_controls()

            fig = None
            canvas = None
            ax_spec = None
            ax_bear = None
            ax_polar = None
            cax_bearing = None
            bearing_cbar = {"obj": None}
            hover_state = {"anno": None, "rows": []}
            try:
                from matplotlib.figure import Figure
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
                fig = Figure(facecolor=gui_panel_bg)
                canvas = FigureCanvas(fig)
                gs = fig.add_gridspec(2, 3, width_ratios=[1.6, 2.5, 0.12], height_ratios=[1.9, 1.25], wspace=0.30, hspace=0.22)
                ax_polar = fig.add_subplot(gs[0, 0], projection="polar")
                ax_spec = fig.add_subplot(gs[0, 1])
                ax_bear = fig.add_subplot(gs[1, :2])
                cax_bearing = fig.add_subplot(gs[:, 2])
                plot_lay.addWidget(canvas, 1)
            except Exception:
                status_lbl.setText("DIFARGram display unavailable (matplotlib Qt backend not available).")

            last_detection_rows = {"rows": []}

            def _set_detection_table(rows):
                rows = rows if isinstance(rows, list) else []
                last_detection_rows["rows"] = rows
                hover_state["rows"] = rows
                detection_table.setRowCount(len(rows))
                for r, row in enumerate(rows):
                    vals = [
                        f"{float(row.get('time_offset_s', 0.0)):.3f}",
                        f"{float(row.get('bearing_deg', 0.0)):.2f}",
                        ("" if row.get("detected_freq_hz") is None else f"{float(row.get('detected_freq_hz')):.2f}"),
                        ("" if row.get("peak_amp_db") is None else f"{float(row.get('peak_amp_db')):.2f}"),
                        ("" if row.get("omni_spl_db_re_1uPa") is None else f"{float(row.get('omni_spl_db_re_1uPa')):.2f}"),
                        ("" if row.get("x_rms_mps") is None else f"{float(row.get('x_rms_mps')):.6g}"),
                        ("" if row.get("x_rms_umps") is None else f"{float(row.get('x_rms_umps')):.3f}"),
                        ("" if row.get("y_rms_mps") is None else f"{float(row.get('y_rms_mps')):.6g}"),
                        ("" if row.get("y_rms_umps") is None else f"{float(row.get('y_rms_umps')):.3f}"),
                        ("" if row.get("z_rms_mps") is None else f"{float(row.get('z_rms_mps')):.6g}"),
                        ("" if row.get("z_rms_umps") is None else f"{float(row.get('z_rms_umps')):.3f}"),
                        str(row.get("mode", "")),
                    ]
                    for c_idx, txt in enumerate(vals):
                        detection_table.setItem(r, c_idx, QtWidgets.QTableWidgetItem(txt))
                detection_table.resizeColumnsToContents()

            def _on_bearing_hover(event):
                if fig is None or canvas is None or ax_bear is None:
                    return
                if event is None or event.inaxes != ax_bear:
                    anno = hover_state.get("anno")
                    if anno is not None:
                        anno.set_visible(False)
                        canvas.draw_idle()
                    return
                rows = hover_state.get("rows") if isinstance(hover_state, dict) else []
                if not isinstance(rows, list) or len(rows) <= 0:
                    return
                try:
                    import numpy as np
                    xvals = np.asarray([float(r.get("time_offset_s", 0.0)) for r in rows], dtype=float)
                    yvals = np.asarray([float(r.get("bearing_deg", 0.0)) for r in rows], dtype=float)
                    if xvals.size <= 0 or yvals.size <= 0:
                        return
                    if event.xdata is None or event.ydata is None:
                        return
                    x0 = float(event.xdata); y0 = float(event.ydata)
                    dx = xvals - x0
                    dy = yvals - y0
                    d2 = dx * dx + dy * dy
                    k = int(np.argmin(d2))
                    xk = float(xvals[k]); yk = float(yvals[k])
                    xspan = max(1e-6, abs(float(ax_bear.get_xlim()[1] - ax_bear.get_xlim()[0])))
                    yspan = max(1e-6, abs(float(ax_bear.get_ylim()[1] - ax_bear.get_ylim()[0])))
                    nd = ((xk - x0) / xspan) ** 2 + ((yk - y0) / yspan) ** 2
                    if nd > 0.0025:
                        anno = hover_state.get("anno")
                        if anno is not None:
                            anno.set_visible(False)
                            canvas.draw_idle()
                        return
                    row = rows[k]
                    spl = row.get("omni_spl_db_re_1uPa")
                    xu = row.get("x_rms_umps")
                    yu = row.get("y_rms_umps")
                    zu = row.get("z_rms_umps")
                    txt = (
                        f"t={xk:.3f}s\n"
                        f"bearing={yk:.2f}°\n"
                        f"OMNI SPL={'' if spl is None else f'{float(spl):.2f}'} dB re 1µPa\n"
                        f"X={'' if xu is None else f'{float(xu):.3f}'} µm/s\n"
                        f"Y={'' if yu is None else f'{float(yu):.3f}'} µm/s\n"
                        f"Z={'' if zu is None else f'{float(zu):.3f}'} µm/s"
                    )
                    anno = hover_state.get("anno")
                    if anno is None:
                        anno = ax_bear.annotate(
                            txt,
                            xy=(xk, yk),
                            xytext=(10, 10),
                            textcoords="offset points",
                            bbox=dict(boxstyle="round,pad=0.25", fc="#111111", ec="#03DFE2", alpha=0.92),
                            color="#E6F1FF",
                            fontsize=8.5,
                            zorder=20,
                        )
                        hover_state["anno"] = anno
                    else:
                        anno.xy = (xk, yk)
                        anno.set_text(txt)
                    anno.set_visible(True)
                    canvas.draw_idle()
                except Exception:
                    return

            def _default_seconds_from_detection():
                meta = getattr(self, "_difar_last_run_meta", None)
                if not isinstance(meta, dict):
                    return
                try:
                    t_vals = [float(v) for v in _safe_seq(meta.get("time_s"))]
                except Exception:
                    t_vals = []
                if len(t_vals) < 2:
                    return
                t0 = float(min(t_vals))
                t1 = float(max(t_vals))
                span = max(2.0, t1 - t0)
                start_sec.setValue(max(0.0, t0))
                win_sec.setValue(span)
                try:
                    bp = meta.get("bandpass_hz")
                    if isinstance(bp, (list, tuple)) and len(bp) >= 2:
                        lo = max(0.0, float(bp[0]))
                        hi = max(lo + 0.1, float(bp[1]))
                        spec_ymin_spin.setValue(lo)
                        spec_ymax_spin.setValue(hi)
                        track_lo_hz_spin.setValue(lo)
                        track_hi_hz_spin.setValue(hi)
                except Exception:
                    pass

            def _style_ax(ax):
                ax.set_facecolor(gui_bg)
                ax.tick_params(colors=gui_fg)
                for sp in ax.spines.values():
                    sp.set_color(gui_border)
                    sp.set_linewidth(1.4)
                ax.xaxis.label.set_color(gui_fg)
                ax.yaxis.label.set_color(gui_fg)
                ax.title.set_color(gui_fg)

            def _style_polar_ax(ax):
                ax.set_facecolor(gui_bg)
                ax.grid(True, alpha=0.25)
                ax.set_theta_zero_location("N")
                ax.set_theta_direction(-1)
                ax.tick_params(colors=gui_fg)
                for sp in ax.spines.values():
                    sp.set_color(gui_border)
                    sp.set_linewidth(1.4)
                ax.set_title("Bearing Detections (Polar)", color=gui_fg)
                ax.set_rlabel_position(135)

            def _read_wav_segment(path, channel_idx, start_s, duration_s):
                import wave
                import numpy as np

                def _slice_channel(arr, ch_idx):
                    if getattr(arr, "ndim", 1) <= 1:
                        return np.asarray(arr)
                    n_channels = int(arr.shape[1])
                    ch = 0 if ch_idx < 0 else (n_channels - 1 if ch_idx >= n_channels else int(ch_idx))
                    return np.asarray(arr[:, ch])

                try:
                    with wave.open(path, "rb") as wf:
                        n_channels = int(wf.getnchannels())
                        sampwidth = int(wf.getsampwidth())
                        rate = int(wf.getframerate())
                        n_frames = int(wf.getnframes())
                        ch = 0 if channel_idx < 0 else (n_channels - 1 if channel_idx >= n_channels else int(channel_idx))
                        start_frame = max(0, int(float(start_s) * rate))
                        end_frame = min(n_frames, start_frame + int(float(duration_s) * rate))
                        count = max(0, end_frame - start_frame)
                        wf.setpos(start_frame)
                        raw = wf.readframes(count)
                    if count <= 0:
                        return np.array([], dtype=float), float(rate)
                    if sampwidth == 1:
                        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
                        data = (data - 128.0) / 128.0
                    elif sampwidth == 2:
                        data = np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768.0
                    elif sampwidth == 3:
                        b = np.frombuffer(raw, dtype=np.uint8)
                        b = b.reshape(-1, 3)
                        v = (b[:, 0].astype(np.int32)
                             | (b[:, 1].astype(np.int32) << 8)
                             | (b[:, 2].astype(np.int32) << 16))
                        sign = (v & 0x800000) != 0
                        v[sign] -= (1 << 24)
                        data = v.astype(np.float32) / 8388608.0
                    elif sampwidth == 4:
                        data = np.frombuffer(raw, dtype='<i4').astype(np.float32) / 2147483648.0
                    else:
                        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")
                    if n_channels > 1:
                        data = data.reshape(-1, n_channels)[:, ch]
                    return data, float(rate)
                except Exception as wave_err:
                    try:
                        from scipy.io import wavfile as scipy_wavfile
                        rate, data_all = scipy_wavfile.read(path)
                        data_all = np.asarray(data_all)
                        if data_all.dtype.kind in ("i", "u"):
                            bits = int(data_all.dtype.itemsize) * 8
                            scale = float(2 ** (bits - 1)) if bits > 1 else 1.0
                            data_all = data_all.astype(np.float32) / scale
                        else:
                            data_all = data_all.astype(np.float32)
                        mono = _slice_channel(data_all, int(channel_idx))
                        start_frame = max(0, int(float(start_s) * float(rate)))
                        end_frame = min(int(mono.shape[0]), start_frame + int(float(duration_s) * float(rate)))
                        if end_frame <= start_frame:
                            return np.array([], dtype=float), float(rate)
                        return mono[start_frame:end_frame], float(rate)
                    except Exception as scipy_err:
                        raise ValueError(
                            f"Unsupported WAV encoding for DIFARGram reader ({wave_err}). "
                            "Convert WAV to PCM or install scipy for fallback decoding."
                        ) from scipy_err

            def _smooth_bearing_series_deg(values, window_n):
                import numpy as np
                arr = np.asarray(values, dtype=float)
                if arr.size <= 2:
                    return arr
                w = max(1, int(window_n))
                if w <= 1:
                    return arr
                if w % 2 == 0:
                    w += 1
                rad = np.deg2rad(arr)
                unwrapped = np.unwrap(rad)
                kernel = np.ones(w, dtype=float) / float(w)
                padded = np.pad(unwrapped, (w // 2, w // 2), mode="edge")
                sm = np.convolve(padded, kernel, mode="valid")
                return (np.rad2deg(sm) + 360.0) % 360.0

            def _render_difargram():
                if fig is None or canvas is None or ax_spec is None or ax_bear is None or ax_polar is None:
                    return
                meta = getattr(self, "_difar_last_run_meta", None)
                if not isinstance(meta, dict):
                    status_lbl.setText("No recent DIFAR run context available. Run DIFAR processing first.")
                    return
                wav_path = str(meta.get("wav_path") or "")
                if not wav_path or not os.path.isfile(wav_path):
                    status_lbl.setText("Recent run WAV path is unavailable. Re-run DIFAR processing or choose a valid WAV in the run.")
                    return
                try:
                    ch_idx = int(meta.get("omni_channel", 0))
                    samples, fs = _read_wav_segment(wav_path, ch_idx, float(start_sec.value()), float(win_sec.value()))
                    if len(samples) <= 8:
                        status_lbl.setText("Selected segment has no audio samples.")
                        return

                    if cax_bearing is not None:
                        cax_bearing.clear()
                        cax_bearing.set_facecolor(gui_bg)
                        cax_bearing.tick_params(colors=gui_fg)
                        for sp in cax_bearing.spines.values():
                            sp.set_color(gui_grid)
                    bearing_cbar["obj"] = None

                    ax_spec.clear(); ax_bear.clear(); ax_polar.clear()
                    hover_state["anno"] = None
                    _style_ax(ax_spec); _style_ax(ax_bear); _style_polar_ax(ax_polar)

                    nfft = int(nfft_combo.currentText())
                    nfft = max(64, min(nfft, max(64, int(len(samples) // 4))))
                    noverlap = max(0, int(nfft * 0.75))
                    spec_cmap = (spec_cmap_combo.currentText().strip() or "magma")
                    pxx, freqs, bins, im = ax_spec.specgram(samples, NFFT=nfft, Fs=fs, noverlap=noverlap, cmap=spec_cmap)
                    im.remove()
                    spec_y0 = float(spec_ymin_spin.value())
                    spec_y1 = float(spec_ymax_spin.value())
                    if spec_y1 <= spec_y0:
                        spec_y0, spec_y1 = 0.0, float(max_freq.value())
                    spec_swapped = (spec_axes_combo.currentIndex() == 1)
                    spec_note = ""
                    try:
                        import numpy as np
                        pxx_db = 10.0 * np.log10(np.maximum(np.asarray(pxx, dtype=float), 1e-30))
                    except Exception:
                        pxx_db = None
                    if pxx_db is not None:
                        if spec_swapped:
                            fmin = float(np.min(freqs)); fmax = float(np.max(freqs))
                            x0 = max(fmin, spec_y0)
                            x1 = min(fmax, spec_y1)
                            if x1 <= x0:
                                x0, x1 = fmin, fmax
                            if spec_y1 > fmax + 1e-9:
                                spec_note = f" (freq limited by Nyquist to {fmax:.1f} Hz)"
                            ax_spec.imshow(
                                pxx_db.T,
                                origin="lower",
                                aspect="auto",
                                extent=[float(np.min(freqs)), float(np.max(freqs)), 0.0, float(win_sec.value())],
                                cmap=spec_cmap,
                            )
                            ax_spec.set_xlim(x0, x1)
                            ax_spec.set_ylim(0.0, float(win_sec.value()))
                            ax_spec.set_xlabel("Frequency (Hz)")
                            ax_spec.set_ylabel("Time within selected segment (s)")
                        else:
                            ax_spec.imshow(
                                pxx_db,
                                origin="lower",
                                aspect="auto",
                                extent=[0.0, float(win_sec.value()), float(np.min(freqs)), float(np.max(freqs))],
                                cmap=spec_cmap,
                            )
                            ax_spec.set_xlim(0.0, float(win_sec.value()))
                            ax_spec.set_ylim(spec_y0, spec_y1)
                            ax_spec.set_xlabel("Time within selected segment (s)")
                            ax_spec.set_ylabel("Frequency (Hz)")
                    else:
                        ax_spec.set_ylim(spec_y0, spec_y1)
                        ax_spec.set_ylabel("Frequency (Hz)")
                    ax_spec.set_title(f"DIFARGram Spectrogram | {os.path.basename(wav_path)} | OMNI ch {ch_idx + 1}")

                    t_raw = [float(v) for v in _safe_seq(meta.get("time_s"))]
                    b_raw = [float(v) % 360.0 for v in _safe_seq(meta.get("bearing_true_deg"))]
                    c_raw = [float(v) for v in _safe_seq(meta.get("confidence"))]
                    omni_spl_raw = [float(v) for v in _safe_seq(meta.get("omni_spl_db_re_1uPa"))]
                    x_rms_raw = [float(v) for v in _safe_seq(meta.get("x_rms_mps"))]
                    y_rms_raw = [float(v) for v in _safe_seq(meta.get("y_rms_mps"))]
                    z_rms_raw = [float(v) for v in _safe_seq(meta.get("z_rms_mps"))]
                    n = min(len(t_raw), len(b_raw))
                    if n > 1:
                        t0 = float(start_sec.value())
                        t1 = t0 + float(win_sec.value())
                        t = []
                        b = []
                        c = []
                        for i in range(n):
                            if t_raw[i] < t0 or t_raw[i] > t1:
                                continue
                            t.append(t_raw[i] - t0)
                            b.append(b_raw[i])
                            c.append(c_raw[i] if i < len(c_raw) else 1.0)
                        if len(t) > 1:
                            b_plot = list(b)
                            if smooth_mode_combo.currentText() == "Moving average":
                                b_plot = _smooth_bearing_series_deg(b_plot, int(smooth_win_spin.value())).tolist()
                            detected_t = list(t)
                            detected_b = list(b_plot)
                            detected_f = [None for _ in detected_t]
                            detected_db = [None for _ in detected_t]
                            detected_mode = "fallback_unfiltered"

                            try:
                                from matplotlib import cm, colors
                                import numpy as np
                                cmap = cm.get_cmap("hsv")
                                norm = colors.Normalize(vmin=0.0, vmax=360.0)

                                # Detect dominant frequency (main band) near each bearing timestamp
                                # and color only that detected track by bearing.
                                bins_arr = np.asarray(bins, dtype=float).reshape(-1)
                                freqs_arr = np.asarray(freqs, dtype=float).reshape(-1)
                                pxx_arr = np.asarray(pxx, dtype=float)
                                if pxx_arr.ndim == 2 and bins_arr.size > 0 and freqs_arr.size > 0:
                                    track_lo = None
                                    track_hi = None
                                    mode = track_mode_combo.currentText()
                                    if mode == "Processed-band track":
                                        bp = meta.get("bandpass_hz")
                                        if isinstance(bp, (list, tuple)) and len(bp) >= 2:
                                            track_lo = float(bp[0])
                                            track_hi = float(bp[1])
                                    elif mode == "Custom-band track":
                                        track_lo = float(track_lo_hz_spin.value())
                                        track_hi = float(track_hi_hz_spin.value())
                                    if track_lo is not None and track_hi is not None:
                                        if track_hi <= track_lo:
                                            track_lo, track_hi = None, None
                                    amp_min_db = float(amp_min_db_spin.value())

                                    ridge_t = []
                                    ridge_f = []
                                    ridge_b = []
                                    ridge_db = []
                                    for ti, bi in zip(t, b_plot):
                                        j = int(np.argmin(np.abs(bins_arr - float(ti))))
                                        if j < 0 or j >= pxx_arr.shape[1]:
                                            continue
                                        col_pow = pxx_arr[:, j]
                                        if col_pow.size == 0:
                                            continue
                                        if track_lo is not None and track_hi is not None:
                                            band_mask = (freqs_arr >= float(track_lo)) & (freqs_arr <= float(track_hi))
                                            idx = np.where(band_mask)[0]
                                            if idx.size <= 0:
                                                continue
                                            band_pow = col_pow[idx]
                                            if band_pow.size <= 0:
                                                continue
                                            fi = int(idx[int(np.nanargmax(band_pow))])
                                        else:
                                            fi = int(np.nanargmax(col_pow))
                                        if fi < 0 or fi >= freqs_arr.size:
                                            continue
                                        peak_pow = float(col_pow[fi])
                                        if not np.isfinite(peak_pow):
                                            continue
                                        peak_db = float(10.0 * np.log10(max(peak_pow, 1e-30)))
                                        if peak_db < amp_min_db:
                                            continue
                                        ff = float(freqs_arr[fi])
                                        if ff < spec_y0 or ff > spec_y1:
                                            continue
                                        ridge_t.append(float(ti))
                                        ridge_f.append(ff)
                                        ridge_b.append(float(bi) % 360.0)
                                        ridge_db.append(float(peak_db))

                                    if len(ridge_t) > 1:
                                        detected_t = [float(v) for v in ridge_t]
                                        detected_b = [float(v) for v in ridge_b]
                                        detected_f = [float(v) for v in ridge_f]
                                        detected_db = [float(v) for v in ridge_db]
                                        detected_mode = "ridge_filtered"
                                        half_hz = max(0.0, float(color_band_halfwidth_hz.value()))
                                        if freqs_arr.size > 1:
                                            f_step = float(np.median(np.diff(freqs_arr)))
                                            f_step = abs(f_step) if np.isfinite(f_step) and abs(f_step) > 1e-9 else max(1.0, half_hz / 8.0)
                                        else:
                                            f_step = max(1.0, half_hz / 8.0)

                                        band_t = []
                                        band_f = []
                                        band_rgba = []

                                        for ti, ff, bb, p_db in zip(ridge_t, ridge_f, ridge_b, ridge_db):
                                            base_rgba = list(cmap(norm(float(bb) % 360.0)))
                                            if half_hz <= 1e-9:
                                                alpha_peak = 0.35 + 0.60 * min(1.0, max(0.0, (float(p_db) - amp_min_db) / 24.0))
                                                rgba = tuple(base_rgba[:3] + [alpha_peak])
                                                band_t.append(float(ti)); band_f.append(float(ff)); band_rgba.append(rgba)
                                                continue

                                            n_side = max(1, int(np.ceil(half_hz / max(1e-9, f_step))))
                                            offsets = np.linspace(-half_hz, half_hz, 2 * n_side + 1)
                                            for off in offsets:
                                                fcur = float(ff + off)
                                                if fcur < spec_y0 or fcur > spec_y1:
                                                    continue
                                                w = max(0.0, 1.0 - (abs(float(off)) / half_hz))
                                                alpha = 0.12 + 0.83 * w
                                                rgba = tuple(base_rgba[:3] + [alpha])
                                                band_t.append(float(ti))
                                                band_f.append(fcur)
                                                band_rgba.append(rgba)

                                        if band_t:
                                            if spec_swapped:
                                                ax_spec.scatter(
                                                    band_f,
                                                    band_t,
                                                    c=band_rgba,
                                                    s=14,
                                                    linewidths=0.0,
                                                    marker="s",
                                                )
                                            else:
                                                ax_spec.scatter(
                                                    band_t,
                                                    band_f,
                                                    c=band_rgba,
                                                    s=14,
                                                    linewidths=0.0,
                                                    marker="s",
                                                )

                                        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
                                        sm.set_array([])
                                        cb = fig.colorbar(sm, cax=cax_bearing) if cax_bearing is not None else fig.colorbar(sm, ax=ax_spec, fraction=0.046, pad=0.02)
                                        bearing_cbar["obj"] = cb
                                        cb.set_label("Bearing (deg)", color=gui_fg)
                                        cb.ax.yaxis.set_tick_params(color=gui_fg)
                                        for tick in cb.ax.get_yticklabels():
                                            tick.set_color(gui_fg)
                            except Exception:
                                pass
                            if len(detected_t) > 1:
                                ax_bear.plot(detected_t, detected_b, color="#03DFE2", linewidth=1.5, alpha=0.95, label="Detected bearing")
                                ax_bear.scatter(detected_t, detected_b, color="#03DFE2", s=12, alpha=0.8)
                            else:
                                ax_bear.plot(t, b_plot, color="#03DFE2", linewidth=1.2, alpha=0.7, label="Bearing (unfiltered fallback)")
                                ax_bear.scatter(t, b_plot, color="#03DFE2", s=9, alpha=0.55)
                            ax_bear.legend(loc="upper right", framealpha=0.3)

                            table_rows = []
                            import numpy as np
                            t_ref = np.asarray(t_raw, dtype=float)
                            def _nearest_metric(tt, arr):
                                try:
                                    if t_ref.size <= 0 or len(arr) <= 0:
                                        return None
                                    k = int(np.argmin(np.abs(t_ref - float(t0 + tt))))
                                    if k < 0 or k >= len(arr):
                                        return None
                                    return float(arr[k])
                                except Exception:
                                    return None
                            for ti, bi, fi, di in zip(detected_t, detected_b, detected_f, detected_db):
                                x_mps = _nearest_metric(float(ti), x_rms_raw)
                                y_mps = _nearest_metric(float(ti), y_rms_raw)
                                z_mps = _nearest_metric(float(ti), z_rms_raw)
                                table_rows.append(
                                    {
                                        "time_offset_s": float(ti),
                                        "bearing_deg": float(bi),
                                        "detected_freq_hz": (None if fi is None else float(fi)),
                                        "peak_amp_db": (None if di is None else float(di)),
                                        "omni_spl_db_re_1uPa": _nearest_metric(float(ti), omni_spl_raw),
                                        "x_rms_mps": x_mps,
                                        "x_rms_umps": (None if x_mps is None else float(x_mps) * 1e6),
                                        "y_rms_mps": y_mps,
                                        "y_rms_umps": (None if y_mps is None else float(y_mps) * 1e6),
                                        "z_rms_mps": z_mps,
                                        "z_rms_umps": (None if z_mps is None else float(z_mps) * 1e6),
                                        "mode": detected_mode,
                                    }
                                )
                            _set_detection_table(table_rows)

                            # Third panel: polar plot of bearing detections (radius = relative time in window)
                            try:
                                import numpy as np
                                theta = np.deg2rad(np.asarray(detected_b, dtype=float) % 360.0)
                                t_arr = np.asarray(detected_t, dtype=float)
                                if t_arr.size > 1:
                                    t0p = float(np.min(t_arr)); t1p = float(np.max(t_arr))
                                    denom = max(1e-9, (t1p - t0p))
                                    r = (t_arr - t0p) / denom
                                else:
                                    r = np.ones_like(theta) * 0.5
                                ax_polar.scatter(theta, r, c=np.asarray(detected_b, dtype=float) % 360.0, cmap="hsv", vmin=0.0, vmax=360.0, s=14, alpha=0.9)
                                ax_polar.set_ylim(0.0, 1.0)
                                ax_polar.set_yticks([0.25, 0.5, 0.75, 1.0])
                                ax_polar.set_yticklabels(["25%", "50%", "75%", "100%"], color=gui_fg)
                            except Exception:
                                pass
                        else:
                            _set_detection_table([])
                    else:
                        _set_detection_table([])
                    ax_bear.set_xlim(0.0, float(win_sec.value()))
                    ax_bear.set_ylim(0.0, 360.0)
                    ax_bear.set_xlabel("Time within selected segment (s)")
                    ax_bear.set_ylabel("Bearing True (deg)")
                    ax_bear.grid(True, alpha=0.25)

                    _style_ax(ax_spec)
                    _style_ax(ax_bear)
                    canvas.draw_idle()
                    mode_txt = track_mode_combo.currentText()
                    if mode_txt == "Processed-band track":
                        bp = meta.get("bandpass_hz")
                        if isinstance(bp, (list, tuple)) and len(bp) >= 2:
                            mode_txt = f"Processed-band track ({float(bp[0]):.1f}-{float(bp[1]):.1f} Hz)"
                    elif mode_txt == "Custom-band track":
                        mode_txt = f"Custom-band track ({float(track_lo_hz_spin.value()):.1f}-{float(track_hi_hz_spin.value()):.1f} Hz)"
                    status_lbl.setText(f"Rendered DIFARGram-style display from latest run. Tracking mode: {mode_txt}.{spec_note}")
                except Exception as e:
                    _set_detection_table([])
                    status_lbl.setText(f"DIFARGram render failed: {e}")

            def _save_jpg():
                if fig is None:
                    return
                meta = getattr(self, "_difar_last_run_meta", {}) or {}
                wav_path = str(meta.get("wav_path") or "")
                wav_name = os.path.splitext(os.path.basename(wav_path))[0] if wav_path else "unknown_wav"
                if hasattr(self, "_project_subdir"):
                    try:
                        base_dir = self._project_subdir("difargram")
                    except Exception:
                        base_dir = os.path.join(os.getcwd(), "difargram")
                else:
                    base_dir = os.path.join(os.getcwd(), "difargram")
                out_dir = os.path.join(base_dir, wav_name)
                os.makedirs(out_dir, exist_ok=True)
                default_path = os.path.join(out_dir, "difargram_spectrogram.jpg")
                path, _ = QtWidgets.QFileDialog.getSaveFileName(pop, "Save DIFARGram JPG", default_path, "JPEG Files (*.jpg *.jpeg)")
                if not path:
                    return
                if not path.lower().endswith((".jpg", ".jpeg")):
                    path = f"{path}.jpg"
                try:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    fig.savefig(path, format="jpg", dpi=180, bbox_inches="tight")
                    status_lbl.setText(f"Saved DIFARGram JPG: {path}")
                except Exception as e:
                    status_lbl.setText(f"Failed saving JPG: {e}")

            def _suggest_target_amplitude():
                meta = getattr(self, "_difar_last_run_meta", None)
                if not isinstance(meta, dict):
                    status_lbl.setText("No recent DIFAR run context available for amplitude suggestion.")
                    return
                wav_path = str(meta.get("wav_path") or "")
                if not wav_path or not os.path.isfile(wav_path):
                    status_lbl.setText("Recent run WAV path is unavailable for amplitude suggestion.")
                    return
                try:
                    import numpy as np
                    ch_idx = int(meta.get("omni_channel", 0))
                    samples, fs = _read_wav_segment(wav_path, ch_idx, float(start_sec.value()), float(win_sec.value()))
                    if len(samples) <= 8:
                        status_lbl.setText("Selected segment has no audio samples for amplitude suggestion.")
                        return

                    nfft = int(nfft_combo.currentText())
                    nfft = max(64, min(nfft, max(64, int(len(samples) // 4))))
                    noverlap = max(0, int(nfft * 0.75))
                    spec_cmap = (spec_cmap_combo.currentText().strip() or "magma")
                    pxx, freqs, bins, _im = ax_spec.specgram(samples, NFFT=nfft, Fs=fs, noverlap=noverlap, cmap=spec_cmap)
                    pxx_arr = np.asarray(pxx, dtype=float)
                    freqs_arr = np.asarray(freqs, dtype=float).reshape(-1)
                    if pxx_arr.ndim != 2 or pxx_arr.size <= 0 or freqs_arr.size <= 0:
                        status_lbl.setText("Could not compute amplitude suggestion from spectrogram.")
                        return

                    mode = track_mode_combo.currentText()
                    track_lo = float(spec_ymin_spin.value())
                    track_hi = float(spec_ymax_spin.value())
                    if mode == "Processed-band track":
                        bp = meta.get("bandpass_hz")
                        if isinstance(bp, (list, tuple)) and len(bp) >= 2:
                            track_lo = float(bp[0]); track_hi = float(bp[1])
                    elif mode == "Custom-band track":
                        track_lo = float(track_lo_hz_spin.value()); track_hi = float(track_hi_hz_spin.value())
                    if track_hi <= track_lo:
                        track_lo, track_hi = float(spec_ymin_spin.value()), float(spec_ymax_spin.value())
                    mask = (freqs_arr >= track_lo) & (freqs_arr <= track_hi)
                    idx = np.where(mask)[0]
                    if idx.size <= 0:
                        status_lbl.setText(f"No frequency bins in {track_lo:.1f}-{track_hi:.1f} Hz for amplitude suggestion.")
                        return
                    band = pxx_arr[idx, :]
                    if band.size <= 0:
                        status_lbl.setText("No band power data available for amplitude suggestion.")
                        return
                    peak_per_t = np.nanmax(band, axis=0)
                    peak_db = 10.0 * np.log10(np.maximum(peak_per_t, 1e-30))
                    peak_db = peak_db[np.isfinite(peak_db)]
                    if peak_db.size <= 4:
                        status_lbl.setText("Not enough valid amplitude samples to suggest a threshold.")
                        return
                    p50 = float(np.percentile(peak_db, 50.0))
                    p75 = float(np.percentile(peak_db, 75.0))
                    p90 = float(np.percentile(peak_db, 90.0))
                    suggested = p75
                    amp_min_db_spin.setValue(suggested)
                    status_lbl.setText(
                        f"Suggested amp min set to {suggested:.1f} dB (p50={p50:.1f}, p75={p75:.1f}, p90={p90:.1f}) in {track_lo:.1f}-{track_hi:.1f} Hz."
                    )
                except Exception as e:
                    status_lbl.setText(f"Amplitude suggestion failed: {e}")

            def _export_detection_csv():
                rows = last_detection_rows.get("rows") if isinstance(last_detection_rows, dict) else None
                rows = rows if isinstance(rows, list) else []
                if len(rows) <= 0:
                    status_lbl.setText("No detection rows to export. Render DIFARGram first.")
                    return
                meta = getattr(self, "_difar_last_run_meta", {}) or {}
                wav_path = str(meta.get("wav_path") or "")
                wav_name = os.path.splitext(os.path.basename(wav_path))[0] if wav_path else "unknown_wav"
                if hasattr(self, "_project_subdir"):
                    try:
                        out_dir = self._project_subdir("difargram_tables")
                    except Exception:
                        out_dir = os.path.join(os.getcwd(), "difargram_tables")
                else:
                    out_dir = os.path.join(os.getcwd(), "difargram_tables")
                os.makedirs(out_dir, exist_ok=True)
                default_path = os.path.join(out_dir, f"{wav_name}_detections.csv")
                path, _ = QtWidgets.QFileDialog.getSaveFileName(pop, "Export Detection Table CSV", default_path, "CSV Files (*.csv)")
                if not path:
                    return
                if not path.lower().endswith(".csv"):
                    path = f"{path}.csv"
                try:
                    import csv
                    with open(path, "w", newline="", encoding="utf-8") as f:
                        w = csv.writer(f)
                        w.writerow([
                            "time_offset_s",
                            "bearing_deg",
                            "detected_freq_hz",
                            "peak_amp_db",
                            "omni_spl_db_re_1uPa",
                            "x_rms_mps",
                            "x_rms_um_per_s",
                            "y_rms_mps",
                            "y_rms_um_per_s",
                            "z_rms_mps",
                            "z_rms_um_per_s",
                            "mode",
                        ])
                        for row in rows:
                            w.writerow([
                                row.get("time_offset_s"),
                                row.get("bearing_deg"),
                                row.get("detected_freq_hz"),
                                row.get("peak_amp_db"),
                                row.get("omni_spl_db_re_1uPa"),
                                row.get("x_rms_mps"),
                                row.get("x_rms_umps"),
                                row.get("y_rms_mps"),
                                row.get("y_rms_umps"),
                                row.get("z_rms_mps"),
                                row.get("z_rms_umps"),
                                row.get("mode"),
                            ])
                    status_lbl.setText(f"Exported detection table CSV: {path}")
                except Exception as e:
                    status_lbl.setText(f"Failed exporting detection table CSV: {e}")

            if canvas is not None:
                try:
                    canvas.mpl_connect("motion_notify_event", _on_bearing_hover)
                except Exception:
                    pass

            render_btn.clicked.connect(_render_difargram)
            save_btn.clicked.connect(_save_jpg)
            suggest_amp_btn.clicked.connect(_suggest_target_amplitude)
            export_csv_btn.clicked.connect(_export_detection_csv)
            _default_seconds_from_detection()
            status_lbl.setText("Ready. Click Render to build DIFARGram for the selected segment.")
            pop.exec_()

        def _refresh_calibration_list(select_name: str = ""):
            cal_combo.clear()
            cal_combo.addItem("", "")
            try:
                conn = sqlite3.connect(DB_FILENAME)
                cur = conn.cursor()
                cur.execute("SELECT calibration_name FROM difar_calibration_sets ORDER BY calibration_name")
                names = [r[0] for r in cur.fetchall()]
                conn.close()
            except Exception:
                names = []
            for n in names:
                cal_combo.addItem(n, n)
            if select_name:
                idx = cal_combo.findData(select_name)
                if idx >= 0:
                    cal_combo.setCurrentIndex(idx)
            _update_calibration_plots()

        def _browse_wav():
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                dlg,
                "Select DIFAR WAV",
                "",
                "WAV Files (*.wav);;All Files (*)",
            )
            if path:
                wav_edit.setText(path)

        def _browse_compass():
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                dlg,
                "Select Compass CSV",
                "",
                "CSV/TSV Files (*.csv *.tsv *.txt);;All Files (*)",
            )
            if path:
                compass_edit.setText(path)

        def _browse_export():
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                dlg,
                "Save DIFAR Output CSV",
                "difar_bearing_timeseries.csv",
                "CSV Files (*.csv)",
            )
            if path:
                export_edit.setText(path)

        def _open_cal_import():
            self._difar_calibration_import_dialog(
                dlg,
                on_imported=lambda nm: _refresh_calibration_list(select_name=nm),
            )

        def _parse_target_bands(text: str):
            text = (text or "").strip()
            if not text:
                return []
            bands = []
            for tok in text.split(";"):
                part = tok.strip()
                if not part:
                    continue
                if "-" not in part:
                    raise ValueError(f"Invalid band token '{part}'. Use lo-hi format.")
                lo_txt, hi_txt = [x.strip() for x in part.split("-", 1)]
                lo = float(lo_txt); hi = float(hi_txt)
                if lo <= 0 or hi <= lo:
                    raise ValueError(f"Invalid band '{part}'. Need hi > lo > 0.")
                bands.append((lo, hi))
            return bands

        selected_profile_meta = None
        selected_target_classification = ""
        selected_target_bands_text = ""

        def _update_target_profile_summary():
            nonlocal selected_profile_meta, selected_target_classification, selected_target_bands_text
            if selected_profile_meta:
                _, name, cls, _, _, btxt = selected_profile_meta
                cls_txt = selected_target_classification or cls or "unclassified"
                bands_txt = selected_target_bands_text or btxt or f"{band_lo.value():.1f}-{band_hi.value():.1f}"
                target_profile_summary.setText(f"{name} | {cls_txt} | {bands_txt}")
            else:
                target_profile_summary.setText("No target profile selected (using Bandpass above).")

        def _open_target_profiles_popup():
            nonlocal selected_profile_meta, selected_target_classification, selected_target_bands_text
            pop = QtWidgets.QDialog(dlg)
            pop.setWindowTitle("DIFAR Target Profiles")
            pop.resize(700, 320)
            lay = QtWidgets.QVBoxLayout(pop)
            form = QtWidgets.QFormLayout()
            lay.addLayout(form)

            search_edit = QtWidgets.QLineEdit(); search_edit.setPlaceholderText("Search saved target profiles...")
            profile_combo = QtWidgets.QComboBox()
            profile_name_edit = QtWidgets.QLineEdit(); profile_name_edit.setPlaceholderText("Profile name")
            class_edit = QtWidgets.QLineEdit(); class_edit.setPlaceholderText("e.g., vessel, cetacean, unknown")
            bands_combo = QtWidgets.QComboBox(); bands_combo.setEditable(True)
            bands_combo.addItems(["20-80", "80-250", "250-600"])
            add_preset_btn = QtWidgets.QPushButton("Add Band Preset")
            save_profile_btn = QtWidgets.QPushButton("Save New Profile")
            update_profile_btn = QtWidgets.QPushButton("Update Selected")
            delete_profile_btn = QtWidgets.QPushButton("Delete Selected")
            apply_btn = QtWidgets.QPushButton("Use Selected Profile")
            close_btn2 = QtWidgets.QPushButton("Close")

            b_row = QtWidgets.QHBoxLayout(); b_row.addWidget(bands_combo); b_row.addWidget(add_preset_btn)
            b_w = QtWidgets.QWidget(); b_w.setLayout(b_row)
            form.addRow("Search:", search_edit)
            form.addRow("Profile:", profile_combo)
            form.addRow("Profile name:", profile_name_edit)
            form.addRow("Classification:", class_edit)
            form.addRow("Target bands:", b_w)
            act_row = QtWidgets.QHBoxLayout(); act_row.addWidget(save_profile_btn); act_row.addWidget(update_profile_btn); act_row.addWidget(delete_profile_btn)
            act_w = QtWidgets.QWidget(); act_w.setLayout(act_row)
            form.addRow("", act_w)

            btns = QtWidgets.QHBoxLayout(); btns.addWidget(apply_btn); btns.addStretch(1); btns.addWidget(close_btn2)
            lay.addLayout(btns)

            def _refresh_profiles(qtxt=""):
                conn = sqlite3.connect(DB_FILENAME)
                self._ensure_difar_target_profiles_table(conn)
                cur = conn.cursor()
                pid = self._resolve_active_project_id()
                q = (qtxt or "").strip()
                sql = (
                    "SELECT id, profile_name, classification, band_lo_hz, band_hi_hz, target_bands_text, COALESCE(use_count,0), last_used_utc "
                    "FROM difar_target_profiles WHERE (project_id IS NULL OR project_id = ?) "
                )
                params = [pid]
                if q:
                    like = f"%{q}%"
                    sql += "AND (profile_name LIKE ? OR classification LIKE ? OR target_bands_text LIKE ?) "
                    params.extend([like, like, like])
                sql += "ORDER BY COALESCE(last_used_utc,'') DESC, profile_name COLLATE NOCASE"
                cur.execute(sql, tuple(params))
                rows = cur.fetchall(); conn.close()
                profile_combo.blockSignals(True)
                profile_combo.clear(); profile_combo.addItem("(No profile)", None)
                for rid, name, cls, lo, hi, btxt, used_n, last_used in rows:
                    used_note = f"used {int(used_n)}x"
                    label = f"{name} | {cls or 'unclassified'} | {btxt or f'{lo:.0f}-{hi:.0f}'} | {used_note}"
                    profile_combo.addItem(label, (int(rid), name or "", cls or "", float(lo or 0.0), float(hi or 0.0), btxt or "", int(used_n), last_used or ""))
                    if btxt and bands_combo.findText(btxt) < 0:
                        bands_combo.addItem(btxt)
                profile_combo.blockSignals(False)

            def _on_selected(_idx=None):
                meta = profile_combo.currentData()
                if not meta:
                    return
                _, name, cls, lo, hi, btxt, _, _ = meta
                profile_name_edit.setText(name)
                class_edit.setText(cls)
                if lo > 0 and hi > lo:
                    band_lo.setValue(lo); band_hi.setValue(hi)
                if btxt:
                    bands_combo.setEditText(btxt)

            def _save_profile():
                name = profile_name_edit.text().strip()
                if not name:
                    QtWidgets.QMessageBox.warning(pop, "Missing Profile Name", "Enter a target profile name.")
                    return
                bands_txt = bands_combo.currentText().strip() or f"{band_lo.value():.1f}-{band_hi.value():.1f}"
                _parse_target_bands(bands_txt)
                cls = class_edit.text().strip() or None
                now = datetime.now(timezone.utc).isoformat()
                pid = self._resolve_active_project_id()
                conn = sqlite3.connect(DB_FILENAME)
                self._ensure_difar_target_profiles_table(conn)
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO difar_target_profiles
                    (created_utc, updated_utc, project_id, profile_name, classification, band_lo_hz, band_hi_hz, target_bands_text, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (now, now, pid, name, cls, float(band_lo.value()), float(band_hi.value()), bands_txt, "Saved from DIFAR tool"),
                )
                conn.commit(); conn.close()
                _refresh_profiles(search_edit.text())
                out.appendPlainText(f"Saved target profile: {name} ({bands_txt})")

            def _update_profile():
                meta = profile_combo.currentData()
                if not meta:
                    QtWidgets.QMessageBox.warning(pop, "No Profile Selected", "Select a profile to update.")
                    return
                rid = int(meta[0])
                name = profile_name_edit.text().strip()
                if not name:
                    QtWidgets.QMessageBox.warning(pop, "Missing Profile Name", "Enter a target profile name.")
                    return
                bands_txt = bands_combo.currentText().strip() or f"{band_lo.value():.1f}-{band_hi.value():.1f}"
                _parse_target_bands(bands_txt)
                cls = class_edit.text().strip() or None
                now = datetime.now(timezone.utc).isoformat()
                conn = sqlite3.connect(DB_FILENAME)
                self._ensure_difar_target_profiles_table(conn)
                cur = conn.cursor()
                cur.execute(
                    """
                    UPDATE difar_target_profiles
                    SET updated_utc=?, profile_name=?, classification=?, band_lo_hz=?, band_hi_hz=?, target_bands_text=?
                    WHERE id=?
                    """,
                    (now, name, cls, float(band_lo.value()), float(band_hi.value()), bands_txt, int(rid)),
                )
                conn.commit(); conn.close()
                _refresh_profiles(search_edit.text())
                out.appendPlainText(f"Updated target profile id={rid}: {name} ({bands_txt})")

            def _delete_profile():
                meta = profile_combo.currentData()
                if not meta:
                    QtWidgets.QMessageBox.warning(pop, "No Profile Selected", "Select a profile to delete.")
                    return
                rid = int(meta[0]); name = str(meta[1])
                ans = QtWidgets.QMessageBox.question(pop, "Delete Profile", f"Delete target profile '{name}'?")
                if ans != QtWidgets.QMessageBox.Yes:
                    return
                conn = sqlite3.connect(DB_FILENAME)
                self._ensure_difar_target_profiles_table(conn)
                cur = conn.cursor()
                cur.execute("DELETE FROM difar_target_profiles WHERE id=?", (int(rid),))
                conn.commit(); conn.close()
                _refresh_profiles(search_edit.text())
                out.appendPlainText(f"Deleted target profile id={rid}: {name}")

            def _add_preset():
                txt = bands_combo.currentText().strip()
                if not txt:
                    return
                try:
                    _parse_target_bands(txt)
                except Exception as e:
                    QtWidgets.QMessageBox.warning(pop, "Invalid Band Preset", str(e))
                    return
                if bands_combo.findText(txt) < 0:
                    bands_combo.addItem(txt)

            def _apply_profile():
                nonlocal selected_profile_meta, selected_target_classification, selected_target_bands_text
                selected_profile_meta = profile_combo.currentData()
                selected_target_classification = class_edit.text().strip()
                selected_target_bands_text = bands_combo.currentText().strip()
                _update_target_profile_summary()
                pop.accept()

            search_edit.textChanged.connect(_refresh_profiles)
            profile_combo.currentIndexChanged.connect(_on_selected)
            save_profile_btn.clicked.connect(_save_profile)
            update_profile_btn.clicked.connect(_update_profile)
            delete_profile_btn.clicked.connect(_delete_profile)
            add_preset_btn.clicked.connect(_add_preset)
            apply_btn.clicked.connect(_apply_profile)
            close_btn2.clicked.connect(pop.reject)
            _refresh_profiles()
            if selected_target_bands_text:
                bands_combo.setEditText(selected_target_bands_text)
            pop.finished.connect(lambda *_: _stop_heatmap_playback())
            pop.exec_()

        def _stop_difar_map_animation():
            tmr = getattr(self, "_difar_overlay_timer", None)
            if tmr is not None:
                try:
                    tmr.stop()
                except Exception:
                    pass
                try:
                    tmr.deleteLater()
                except Exception:
                    pass
            self._difar_overlay_timer = None

        def _push_overlay_to_chart(overlay: dict):
            self._difar_chart_overlay = overlay
            if hasattr(self, "refresh_chart_tracks"):
                try:
                    self.refresh_chart_tracks()
                except Exception:
                    pass

        def _start_difar_map_animation(base_overlay: dict, out_box):
            if base_overlay is None:
                return
            lat2_raw = base_overlay.get("lat2")
            lon2_raw = base_overlay.get("lon2")
            time_raw = base_overlay.get("time_s")
            bearing_raw = base_overlay.get("bearing_true_deg")
            lat2 = list([] if lat2_raw is None else lat2_raw)
            lon2 = list([] if lon2_raw is None else lon2_raw)
            time_s = list([] if time_raw is None else time_raw)
            bearing = list([] if bearing_raw is None else bearing_raw)
            n = min(len(lat2), len(lon2), len(time_s), len(bearing))
            if n <= 0:
                out_box.appendPlainText("Animation skipped: no rays available.")
                return

            _stop_difar_map_animation()
            idx = {"i": 0}
            step = max(1, int(anim_step_spin.value()))
            interval_ms = max(20, int(anim_ms_spin.value()))

            def _tick():
                i = idx["i"]
                if i >= n:
                    _stop_difar_map_animation()
                    out_box.appendPlainText("DIFAR map animation completed.")
                    return
                j = min(n, i + step)
                ov = dict(base_overlay)
                ov["lat2"] = lat2[:j]
                ov["lon2"] = lon2[:j]
                ov["time_s"] = time_s[:j]
                ov["bearing_true_deg"] = bearing[:j]
                _push_overlay_to_chart(ov)
                idx["i"] = j

            tmr = QtCore.QTimer(dlg)
            tmr.timeout.connect(_tick)
            self._difar_overlay_timer = tmr
            _tick()
            tmr.start(interval_ms)
            out_box.appendPlainText(f"Animating DIFAR rays on Chart map ({n} rays, step={step}, interval={interval_ms} ms).")

        def _run_processing():
            wav_path = (loaded_wav or "").strip() or wav_edit.text().strip()
            cal_name = cal_combo.currentData() or cal_combo.currentText().strip()
            compass_path = compass_edit.text().strip()
            export_path = export_edit.text().strip() or None

            if not wav_path or not os.path.isfile(wav_path):
                QtWidgets.QMessageBox.warning(dlg, "Missing WAV", "Select a valid WAV file.")
                return
            if not cal_name:
                QtWidgets.QMessageBox.warning(dlg, "Missing Calibration", "Select a calibration set.")
                return

            try:
                cal = load_difar_calibration_from_db(DB_FILENAME, cal_name)

                compass = None
                if compass_path:
                    if not os.path.isfile(compass_path):
                        raise ValueError("Compass CSV path is invalid.")
                    compass = load_compass_csv(compass_path)

                dt = start_dt.dateTime().toPyDateTime()
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)

                z_val = int(z_spin.value())
                z_idx = None if z_val <= 0 else (z_val - 1)

                base_cfg_kwargs = dict(
                    omni_channel=int(omni_spin.value()) - 1,
                    x_channel=int(x_spin.value()) - 1,
                    y_channel=int(y_spin.value()) - 1,
                    z_channel=z_idx,
                    calibration=cal,
                    compass=compass,
                    start_time_utc=dt,
                    swap_xy=bool(swap_xy_chk.isChecked()),
                    invert_x=bool(invert_x_chk.isChecked()),
                    invert_y=bool(invert_y_chk.isChecked()),
                    bearing_offset_deg=float(offs_spin.value()),
                    min_directional_percentile=float(gate_spin.value()),
                    bearing_smooth_frames=int(smooth_spin.value()),
                    resolve_180_ambiguity=bool(ambig_chk.isChecked()),
                    use_omni_for_ambiguity=bool(omni_ambig_chk.isChecked()),
                    adc_full_scale_volts=float(adc_fs_spin.value()),
                )
                bands_text = (selected_target_bands_text or "").strip()
                bands = _parse_target_bands(bands_text)
                if not bands:
                    bands = [(float(band_lo.value()), float(band_hi.value()))]

                selected_profile_id = selected_profile_meta[0] if selected_profile_meta else None
                target_class = selected_target_classification or (selected_profile_meta[2] if selected_profile_meta else None)

                for band_idx, (bp_lo, bp_hi) in enumerate(bands, start=1):
                    cfg = self._make_difar_config_compat(**base_cfg_kwargs, bandpass_hz=(bp_lo, bp_hi))
                    run_tag = f"band {bp_lo:.1f}-{bp_hi:.1f} Hz"

                    run_export_path = None
                    if export_path:
                        if len(bands) == 1:
                            run_export_path = export_path
                        else:
                            base, ext = os.path.splitext(export_path)
                            ext = ext or ".csv"
                            run_export_path = f"{base}_b{int(round(bp_lo))}-{int(round(bp_hi))}{ext}"

                    if chunk_chk.isChecked():
                        result = process_wav_to_bearing_time_series_chunked(
                            wav_path,
                            cfg=cfg,
                            export_csv_path=run_export_path,
                            chunk_seconds=float(chunk_minutes.value()) * 60.0,
                            overlap_seconds=float(overlap_seconds.value()),
                        )
                    else:
                        result = process_wav_to_bearing_time_series(
                            wav_path,
                            cfg=cfg,
                            export_csv_path=run_export_path,
                        )

                    if selected_profile_id is not None:
                        try:
                            conn = sqlite3.connect(DB_FILENAME)
                            self._ensure_difar_target_profiles_table(conn)
                            cur = conn.cursor()
                            cur.execute(
                                """
                                UPDATE difar_target_profiles
                                SET last_used_utc=?, use_count=COALESCE(use_count,0)+1
                                WHERE id=?
                                """,
                                (datetime.now(timezone.utc).isoformat(), int(selected_profile_id)),
                            )
                            conn.commit(); conn.close()
                        except Exception:
                            pass

                    n_frames = len(result.get("time_s", []))
                    out.appendPlainText(f"Processed WAV: {os.path.basename(wav_path)} [{band_idx}/{len(bands)} {run_tag}]")
                    out.appendPlainText(f"Frames: {n_frames}")
                    out.appendPlainText(
                        f"Channel map (1-based): OMNI={omni_spin.value()}, X={x_spin.value()}, Y={y_spin.value()}, Z={(z_spin.value() if z_spin.value() > 0 else 'unused')}"
                    )
                    out.appendPlainText(
                        f"Convention: swap_xy={swap_xy_chk.isChecked()} invert_x={invert_x_chk.isChecked()} invert_y={invert_y_chk.isChecked()} offset={offs_spin.value():.2f}° gate={gate_spin.value():.1f}% smooth={smooth_spin.value()} resolve180={ambig_chk.isChecked()} omni_disambig={omni_ambig_chk.isChecked()}"
                    )
                    out.appendPlainText(f"Amplitude scaling: DAQ full-scale={adc_fs_spin.value():.3f} V before calibration.")
                    out.appendPlainText(f"Output keys: {', '.join(result.keys())}")
                    if run_export_path:
                        out.appendPlainText(f"Saved CSV: {run_export_path}")

                    heatmap_payload = _result_to_heatmap_payload(result)
                    heatmap_label = f"DIFAR: {os.path.basename(wav_path)} [{run_tag}]"
                    if heatmap_payload is not None:
                        self._difar_last_heatmap_payload = heatmap_payload
                        self._difar_last_heatmap_label = heatmap_label

                    self._difar_last_run_meta = {
                        "wav_path": wav_path,
                        "label": heatmap_label,
                        "omni_channel": int(omni_spin.value()) - 1,
                        "bandpass_hz": [float(bp_lo), float(bp_hi)],
                        "adc_full_scale_volts": float(adc_fs_spin.value()),
                        "time_s": _safe_seq(result.get("time_s")),
                        "bearing_true_deg": _safe_seq(result.get("bearing_true_deg")),
                        "confidence": _safe_seq(result.get("confidence")),
                        "omni_spl_db_re_1uPa": _safe_seq(result.get("omni_spl_db_re_1uPa")),
                        "x_rms_mps": _safe_seq(result.get("x_rms_mps")),
                        "y_rms_mps": _safe_seq(result.get("y_rms_mps")),
                        "z_rms_mps": _safe_seq(result.get("z_rms_mps")),
                    }

                    run_id = None
                    if save_db_chk.isChecked():
                        serializable = {}
                        for k, v in result.items():
                            try:
                                serializable[k] = [float(x) if hasattr(x, "__float__") else str(x) for x in list(v)]
                            except Exception:
                                serializable[k] = str(v)

                        conn = sqlite3.connect(DB_FILENAME)
                        self._ensure_difar_results_table(conn)
                        cur = conn.cursor()
                        cur.execute(
                            """
                            INSERT INTO difar_results (
                                created_utc, wav_path, calibration_name, start_time_utc,
                                omni_channel, x_channel, y_channel, z_channel,
                                compass_path, sensor_lat, sensor_lon, target_profile_id, target_classification, result_json
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                datetime.now(timezone.utc).isoformat(),
                                wav_path,
                                f"{cal_name} [{run_tag}]",
                                dt.isoformat(),
                                int(omni_spin.value()) - 1,
                                int(x_spin.value()) - 1,
                                int(y_spin.value()) - 1,
                                (None if z_spin.value() <= 0 else int(z_spin.value()) - 1),
                                (compass_path or None),
                                (float(lat_edit.text()) if lat_edit.text().strip() else None),
                                (float(lon_edit.text()) if lon_edit.text().strip() else None),
                                selected_profile_id,
                                target_class,
                                json.dumps(serializable),
                            ),
                        )
                        run_id = cur.lastrowid
                        conn.commit()
                        conn.close()
                        out.appendPlainText(f"Saved analyzed DIFAR run to DB (difar_results.id={run_id}).")

                    if heatmap_payload is not None:
                        _save_heatmap_payload(heatmap_payload, run_id=run_id, label=heatmap_label)
                        out.appendPlainText("Saved heatmap data to DB for project-scoped recall.")

                    lat_txt, lon_txt = lat_edit.text().strip(), lon_edit.text().strip()
                    if not (lat_txt and lon_txt and "bearing_true_deg" in result and "time_s" in result):
                        continue

                    lat = float(lat_txt)
                    lon = float(lon_txt)
                    rays = bearing_series_static_map_vectors(
                        sensor_lat=lat,
                        sensor_lon=lon,
                        bearing_true_deg=result["bearing_true_deg"],
                        time_s=result["time_s"],
                    )
                    out.appendPlainText(f"Static map rays prepared: {len(rays['time_s'])}")
                    out.appendPlainText(
                        f"Display on Chart map: {'ON' if show_on_chart_chk.isChecked() else 'OFF'}"
                    )

                    def _as_float_list(raw):
                        if raw is None:
                            return []
                        return [float(v) for v in list(raw)]

                    project_id = self._resolve_active_project_id()
                    try:
                        conn = sqlite3.connect(DB_FILENAME)
                        self._ensure_difar_rays_table_compat(conn)
                        cur = conn.cursor()
                        cur.execute(
                            """
                            INSERT INTO difar_map_rays (
                                created_utc, project_id, run_id, label,
                                sensor_lat, sensor_lon,
                                lat2_json, lon2_json, time_s_json, bearing_true_deg_json
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                datetime.now(timezone.utc).isoformat(),
                                project_id,
                                run_id,
                                f"DIFAR: {os.path.basename(wav_path)} [{run_tag}]",
                                float(lat),
                                float(lon),
                                json.dumps(_as_float_list(rays.get("lat2"))),
                                json.dumps(_as_float_list(rays.get("lon2"))),
                                json.dumps(_as_float_list(rays.get("time_s"))),
                                json.dumps(_as_float_list(rays.get("bearing_true_deg"))),
                            ),
                        )
                        ray_id = cur.lastrowid
                        conn.commit(); conn.close()
                        out.appendPlainText(f"Saved DIFAR rays to DB (difar_map_rays.id={ray_id}, project_id={project_id}).")
                    except Exception as e:
                        out.appendPlainText(f"Warning: could not save rays to DB: {e}")

                    if show_on_chart_chk.isChecked():
                        base_overlay = {
                            "sensor_lat": lat,
                            "sensor_lon": lon,
                            "lat2": rays.get("lat2"),
                            "lon2": rays.get("lon2"),
                            "time_s": rays.get("time_s"),
                            "bearing_true_deg": rays.get("bearing_true_deg"),
                            "label": f"DIFAR: {os.path.basename(wav_path)} [{run_tag}]",
                        }
                        self._difar_chart_overlay = base_overlay
                        if hasattr(self, "refresh_chart_tracks"):
                            try:
                                self.refresh_chart_tracks()
                            except Exception:
                                pass
                        out.appendPlainText("DIFAR rays pushed to Chart tab overlay.")

            except Exception as e:
                out.appendPlainText(f"Run failed: {e}")
                QtWidgets.QMessageBox.critical(dlg, "Run Failed", str(e))

        def _open_simulator():
            try:
                from tools_difar_simulator import launch_difar_simulator
            except Exception as e:
                QtWidgets.QMessageBox.critical(dlg, "Simulator Unavailable", f"Failed to load DIFAR simulator tool:\n{e}")
                return
            try:
                project_id = self._resolve_active_project_id()
                output_dir = None
                if hasattr(self, "_project_subdir"):
                    try:
                        output_dir = self._project_subdir("difar_simulator")
                    except Exception:
                        output_dir = None
                self._difar_sim_window = launch_difar_simulator(
                    parent=dlg,
                    project_id=project_id,
                    output_dir=output_dir,
                    db_path=DB_FILENAME,
                    host_window=self,
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(dlg, "Simulator Error", str(e))

        import_btn.clicked.connect(_open_cal_import)
        sim_btn.clicked.connect(_open_simulator)
        wav_browse.clicked.connect(_browse_wav)
        compass_browse.clicked.connect(_browse_compass)
        export_browse.clicked.connect(_browse_export)
        manage_profiles_btn.clicked.connect(_open_target_profiles_popup)
        open_heatmap_btn.clicked.connect(lambda: _open_heatmap_popup())
        open_difargram_btn.clicked.connect(_open_difargram_popup)
        run_btn.clicked.connect(_run_processing)
        close_btn.clicked.connect(dlg.accept)
        dlg.finished.connect(lambda *_: _stop_difar_map_animation())
        cal_combo.currentIndexChanged.connect(_update_calibration_plots)
        if hasattr(self, "color_combo"):
            try:
                self.color_combo.currentIndexChanged.connect(_update_calibration_plots)
            except Exception:
                pass

        _refresh_calibration_list()
        _update_target_profile_summary()
        dlg.exec_()
