#!/usr/bin/env python3
"""DIFAR tools mixin for MainWindow integration."""

import os
import json
import sqlite3
from datetime import timezone, datetime

from PyQt5 import QtWidgets, QtCore

from shared import DB_FILENAME
from difar_core import (
    DifarConfig,
    import_difar_calibration_csv_to_db,
    load_difar_calibration_from_db,
    load_compass_csv,
    process_wav_to_bearing_time_series,
    bearing_series_static_map_vectors,
)


class DifarToolsMixin:
    """Mixin class providing DIFAR tools for MainWindow."""

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
                result_json TEXT
            )
            """
        )
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
        dlg.resize(980, 760)

        layout = QtWidgets.QVBoxLayout(dlg)

        help_lbl = QtWidgets.QLabel(
            "DIFAR workflow:\n"
            "1) Use Calibration Import button to add/update calibration sets.\n"
            "2) Use loaded WAV automatically (or choose one when none loaded), set mapping/start UTC/compass.\n"
            "3) Run bearing extraction, save analyzed output to DB, optional CSV export, and optional Chart overlay."
        )
        help_lbl.setWordWrap(True)
        layout.addWidget(help_lbl)

        import_row = QtWidgets.QHBoxLayout()
        import_btn = QtWidgets.QPushButton("Calibration Import...")
        import_row.addWidget(import_btn)
        import_row.addStretch(1)
        layout.addLayout(import_row)

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

        ambig_chk = QtWidgets.QCheckBox("Resolve ±180° ambiguity by continuity")
        ambig_chk.setChecked(True)
        proc_form.addRow("", ambig_chk)

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

        lat_edit = QtWidgets.QLineEdit(); lat_edit.setPlaceholderText("sensor latitude")
        lon_edit = QtWidgets.QLineEdit(); lon_edit.setPlaceholderText("sensor longitude")
        latlon_row = QtWidgets.QHBoxLayout(); latlon_row.addWidget(lat_edit); latlon_row.addWidget(lon_edit)
        latlon_w = QtWidgets.QWidget(); latlon_w.setLayout(latlon_row)
        proc_form.addRow("Static map rays (opt):", latlon_w)

        show_on_chart_chk = QtWidgets.QCheckBox("Display DIFAR on Chart map (sensor + rays)")
        show_on_chart_chk.setChecked(True)
        proc_form.addRow("", show_on_chart_chk)

        save_db_chk = QtWidgets.QCheckBox("Save analyzed DIFAR output to database")
        save_db_chk.setChecked(True)
        proc_form.addRow("", save_db_chk)

        run_btn = QtWidgets.QPushButton("Run DIFAR Processing")
        proc_form.addRow("", run_btn)

        layout.addWidget(proc_box)

        out = QtWidgets.QPlainTextEdit()
        out.setReadOnly(True)
        out.setPlaceholderText("Status output...")
        layout.addWidget(out, stretch=1)

        close_btn = QtWidgets.QPushButton("Close")
        close_row = QtWidgets.QHBoxLayout(); close_row.addStretch(1); close_row.addWidget(close_btn)
        layout.addLayout(close_row)

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

                cfg = DifarConfig(
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
                )

                result = process_wav_to_bearing_time_series(
                    wav_path,
                    cfg=cfg,
                    export_csv_path=export_path,
                )

                n_frames = len(result.get("time_s", []))
                out.appendPlainText(f"Processed WAV: {os.path.basename(wav_path)}")
                out.appendPlainText(f"Frames: {n_frames}")
                out.appendPlainText(
                    f"Channel map (1-based): OMNI={omni_spin.value()}, X={x_spin.value()}, Y={y_spin.value()}, Z={(z_spin.value() if z_spin.value() > 0 else 'unused')}"
                )
                out.appendPlainText(
                    f"Convention: swap_xy={swap_xy_chk.isChecked()} invert_x={invert_x_chk.isChecked()} invert_y={invert_y_chk.isChecked()} offset={offs_spin.value():.2f}° gate={gate_spin.value():.1f}% smooth={smooth_spin.value()} resolve180={ambig_chk.isChecked()}"
                )
                out.appendPlainText(f"Output keys: {', '.join(result.keys())}")
                if export_path:
                    out.appendPlainText(f"Saved CSV: {export_path}")

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
                            compass_path, sensor_lat, sensor_lon, result_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            datetime.now(timezone.utc).isoformat(),
                            wav_path,
                            cal_name,
                            dt.isoformat(),
                            int(omni_spin.value()) - 1,
                            int(x_spin.value()) - 1,
                            int(y_spin.value()) - 1,
                            (None if z_spin.value() <= 0 else int(z_spin.value()) - 1),
                            (compass_path or None),
                            (float(lat_edit.text()) if lat_edit.text().strip() else None),
                            (float(lon_edit.text()) if lon_edit.text().strip() else None),
                            json.dumps(serializable),
                        ),
                    )
                    run_id = cur.lastrowid
                    conn.commit()
                    conn.close()
                    out.appendPlainText(f"Saved analyzed DIFAR run to DB (difar_results.id={run_id}).")

                lat_txt, lon_txt = lat_edit.text().strip(), lon_edit.text().strip()
                if lat_txt and lon_txt and "bearing_true_deg" in result and "time_s" in result:
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

                    if show_on_chart_chk.isChecked():
                        self._difar_chart_overlay = {
                            "sensor_lat": lat,
                            "sensor_lon": lon,
                            "lat2": rays.get("lat2"),
                            "lon2": rays.get("lon2"),
                            "time_s": rays.get("time_s"),
                            "bearing_true_deg": rays.get("bearing_true_deg"),
                            "label": f"DIFAR: {os.path.basename(wav_path)}",
                        }
                        if hasattr(self, "refresh_chart_tracks"):
                            try:
                                self.refresh_chart_tracks()
                            except Exception:
                                pass
                        out.appendPlainText("DIFAR rays pushed to Chart tab overlay.")

            except Exception as e:
                out.appendPlainText(f"Run failed: {e}")
                QtWidgets.QMessageBox.critical(dlg, "Run Failed", str(e))

        import_btn.clicked.connect(_open_cal_import)
        wav_browse.clicked.connect(_browse_wav)
        compass_browse.clicked.connect(_browse_compass)
        export_browse.clicked.connect(_browse_export)
        run_btn.clicked.connect(_run_processing)
        close_btn.clicked.connect(dlg.accept)

        _refresh_calibration_list()
        dlg.exec_()
