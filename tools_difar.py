#!/usr/bin/env python3
"""DIFAR tools mixin for MainWindow integration."""

import os
import sqlite3
from datetime import timezone

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

    def difar_processing_popup(self):
        """Popup for calibration import and DIFAR processing run."""
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("DIFAR Processing")
        dlg.resize(920, 640)

        layout = QtWidgets.QVBoxLayout(dlg)

        help_lbl = QtWidgets.QLabel(
            "DIFAR workflow:\n"
            "1) Import calibration CSV to DB (frequency,x,y,z,omni + phase columns).\n"
            "2) Select WAV + calibration + channel mapping + start UTC + optional compass CSV.\n"
            "3) Run bearing extraction and optional CSV export."
        )
        help_lbl.setWordWrap(True)
        layout.addWidget(help_lbl)

        # --- Calibration import panel ---
        import_box = QtWidgets.QGroupBox("Calibration Import")
        import_form = QtWidgets.QFormLayout(import_box)

        cal_name_edit = QtWidgets.QLineEdit()
        cal_name_edit.setPlaceholderText("e.g., DIFAR_SN123_2026-01")
        import_form.addRow("Calibration Name:", cal_name_edit)

        cal_file_edit = QtWidgets.QLineEdit()
        cal_browse_btn = QtWidgets.QPushButton("Browse CSV")
        cal_file_row = QtWidgets.QHBoxLayout()
        cal_file_row.addWidget(cal_file_edit)
        cal_file_row.addWidget(cal_browse_btn)
        cal_file_w = QtWidgets.QWidget(); cal_file_w.setLayout(cal_file_row)
        import_form.addRow("Calibration CSV:", cal_file_w)

        import_btn = QtWidgets.QPushButton("Import Calibration to DB")
        import_form.addRow("", import_btn)

        layout.addWidget(import_box)

        # --- Processing panel ---
        proc_box = QtWidgets.QGroupBox("Processing Run")
        proc_form = QtWidgets.QFormLayout(proc_box)

        wav_edit = QtWidgets.QLineEdit()
        wav_browse = QtWidgets.QPushButton("Browse WAV")
        wav_row = QtWidgets.QHBoxLayout(); wav_row.addWidget(wav_edit); wav_row.addWidget(wav_browse)
        wav_w = QtWidgets.QWidget(); wav_w.setLayout(wav_row)
        proc_form.addRow("Input WAV:", wav_w)

        cal_combo = QtWidgets.QComboBox()
        proc_form.addRow("Calibration Set:", cal_combo)

        # User channel mapping (1-based indices in UI)
        ch_row = QtWidgets.QHBoxLayout()
        omni_spin = QtWidgets.QSpinBox(); omni_spin.setMinimum(1); omni_spin.setMaximum(256); omni_spin.setValue(1); omni_spin.setPrefix("OMNI ")
        x_spin = QtWidgets.QSpinBox(); x_spin.setMinimum(1); x_spin.setMaximum(256); x_spin.setValue(2); x_spin.setPrefix("X ")
        y_spin = QtWidgets.QSpinBox(); y_spin.setMinimum(1); y_spin.setMaximum(256); y_spin.setValue(3); y_spin.setPrefix("Y ")
        z_spin = QtWidgets.QSpinBox(); z_spin.setMinimum(0); z_spin.setMaximum(256); z_spin.setValue(4); z_spin.setSpecialValueText("Z unused")
        ch_row.addWidget(omni_spin); ch_row.addWidget(x_spin); ch_row.addWidget(y_spin); ch_row.addWidget(z_spin)
        ch_w = QtWidgets.QWidget(); ch_w.setLayout(ch_row)
        proc_form.addRow("Channel Mapping:", ch_w)

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

        def _browse_cal_csv():
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                dlg,
                "Select DIFAR Calibration CSV",
                "",
                "CSV/TSV Files (*.csv *.tsv *.txt);;All Files (*)"
            )
            if path:
                cal_file_edit.setText(path)

        def _browse_wav():
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                dlg,
                "Select DIFAR WAV",
                "",
                "WAV Files (*.wav);;All Files (*)"
            )
            if path:
                wav_edit.setText(path)

        def _browse_compass():
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                dlg,
                "Select Compass CSV",
                "",
                "CSV/TSV Files (*.csv *.tsv *.txt);;All Files (*)"
            )
            if path:
                compass_edit.setText(path)

        def _browse_export():
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                dlg,
                "Save DIFAR Output CSV",
                "difar_bearing_timeseries.csv",
                "CSV Files (*.csv)"
            )
            if path:
                export_edit.setText(path)

        def _import_calibration():
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
                _refresh_calibration_list(select_name=name)
            except Exception as e:
                out.appendPlainText(f"Import failed: {e}")
                QtWidgets.QMessageBox.critical(dlg, "Import Failed", str(e))

        def _run_processing():
            wav_path = wav_edit.text().strip()
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
                    f"Channel map (1-based): OMNI={omni_spin.value()}, X={x_spin.value()}, Y={y_spin.value()}, Z={(z_spin.value() if z_spin.value()>0 else 'unused')}"
                )
                out.appendPlainText(f"Output keys: {', '.join(result.keys())}")
                if export_path:
                    out.appendPlainText(f"Saved CSV: {export_path}")

                lat_txt, lon_txt = lat_edit.text().strip(), lon_edit.text().strip()
                if lat_txt and lon_txt and "bearing_true_deg" in result and "time_s" in result:
                    lat = float(lat_txt); lon = float(lon_txt)
                    rays = bearing_series_static_map_vectors(
                        sensor_lat=lat,
                        sensor_lon=lon,
                        bearing_true_deg=result["bearing_true_deg"],
                        time_s=result["time_s"],
                    )
                    out.appendPlainText(f"Static map rays prepared: {len(rays['time_s'])}")

            except Exception as e:
                out.appendPlainText(f"Run failed: {e}")
                QtWidgets.QMessageBox.critical(dlg, "Run Failed", str(e))

        cal_browse_btn.clicked.connect(_browse_cal_csv)
        wav_browse.clicked.connect(_browse_wav)
        compass_browse.clicked.connect(_browse_compass)
        export_browse.clicked.connect(_browse_export)
        import_btn.clicked.connect(_import_calibration)
        run_btn.clicked.connect(_run_processing)
        close_btn.clicked.connect(dlg.accept)

        _refresh_calibration_list()
        dlg.exec_()
