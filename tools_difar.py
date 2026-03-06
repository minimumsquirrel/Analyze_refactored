#!/usr/bin/env python3
"""DIFAR tools mixin for MainWindow integration."""

import os
from PyQt5 import QtWidgets

from shared import DB_FILENAME
from difar_core import import_difar_calibration_csv_to_db


class DifarToolsMixin:
    """Mixin class providing DIFAR tools for MainWindow."""

    def difar_processing_popup(self):
        """Entry popup for DIFAR workflow and calibration import."""
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("DIFAR Processing")
        dlg.resize(760, 420)

        layout = QtWidgets.QVBoxLayout(dlg)

        help_lbl = QtWidgets.QLabel(
            "DIFAR workflow:\n"
            "1) Import calibration CSV to DB (supports frequency,x,y,z,omni,x/y phase,z phase).\n"
            "2) Set WAV start UTC datetime + compass heading/time-series.\n"
            "3) Compute bearing time-series and render static-map rays from sensor marker."
        )
        help_lbl.setWordWrap(True)
        layout.addWidget(help_lbl)

        form = QtWidgets.QFormLayout()
        self._difar_cal_name_edit = QtWidgets.QLineEdit()
        self._difar_cal_name_edit.setPlaceholderText("e.g., DIFAR_SN123_2026-01")
        form.addRow("Calibration Name:", self._difar_cal_name_edit)

        self._difar_cal_file_edit = QtWidgets.QLineEdit()
        browse_btn = QtWidgets.QPushButton("Browse CSV")
        file_row = QtWidgets.QHBoxLayout()
        file_row.addWidget(self._difar_cal_file_edit)
        file_row.addWidget(browse_btn)
        file_row_w = QtWidgets.QWidget(); file_row_w.setLayout(file_row)
        form.addRow("Calibration CSV:", file_row_w)

        layout.addLayout(form)

        out = QtWidgets.QPlainTextEdit()
        out.setReadOnly(True)
        out.setPlaceholderText("Status output...")
        layout.addWidget(out, stretch=1)

        btn_row = QtWidgets.QHBoxLayout()
        import_btn = QtWidgets.QPushButton("Import Calibration to DB")
        close_btn = QtWidgets.QPushButton("Close")
        btn_row.addWidget(import_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        def _browse_csv():
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                dlg,
                "Select DIFAR Calibration CSV",
                "",
                "CSV Files (*.csv);;All Files (*)"
            )
            if path:
                self._difar_cal_file_edit.setText(path)

        def _import_cal():
            name = self._difar_cal_name_edit.text().strip()
            path = self._difar_cal_file_edit.text().strip()
            if not name:
                QtWidgets.QMessageBox.warning(dlg, "Missing Name", "Enter a calibration name.")
                return
            if not path or not os.path.isfile(path):
                QtWidgets.QMessageBox.warning(dlg, "Missing CSV", "Select a valid calibration CSV.")
                return
            try:
                n = import_difar_calibration_csv_to_db(DB_FILENAME, path, name)
                out.appendPlainText(f"Imported calibration '{name}' with {n} rows to DB: {DB_FILENAME}")
                QtWidgets.QMessageBox.information(dlg, "Imported", f"Calibration '{name}' imported ({n} rows).")
            except Exception as e:
                out.appendPlainText(f"Import failed: {e}")
                QtWidgets.QMessageBox.critical(dlg, "Import Failed", str(e))

        browse_btn.clicked.connect(_browse_csv)
        import_btn.clicked.connect(_import_cal)
        close_btn.clicked.connect(dlg.accept)

        dlg.exec_()
