#!/usr/bin/env python3
"""
Database Tools — methods for MainWindow mixin
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

class DatabaseToolsMixin:
    """Mixin class providing all Database Tools for self."""

    def export_logs_to_excel(self):
        """
        Export the entire measurements log to CSV or Excel.
        """

        # 1) Prompt for save-as path
        path, filt = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Logs",
            "",
            "CSV Files (*.csv);;Excel Files (*.xlsx *.xls)"
        )
        if not path:
            return  # user cancelled

        # 2) Ensure extension
        root, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext not in ('.csv', '.xls', '.xlsx'):
            ext = '.csv'
            path = root + ext

        # 3) Load data
        try:
            conn = sqlite3.connect(DB_FILENAME)
            df = pd.read_sql_query("SELECT * FROM measurements", conn)
            conn.close()
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Failed to read database",
                f"Could not load measurements table:\n{e}"
            )
            return

        # 4) Write out
        try:
            if ext == '.csv':
                df.to_csv(path, index=False)
            else:
                # for .xls/.xlsx, we need openpyxl or xlwt
                df.to_excel(path, index=False, engine='openpyxl')
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Failed to export log entries",
                str(e)
            )
            return

        QtWidgets.QMessageBox.information(
            self,
            "Export Complete",
            f"Wrote {len(df)} rows to:\n{path}"
        )


    def date_time_filter_popup(self):
        """
        Popup to let the user specify a start/end DATE (YYYY-MM-DD),
        preview matching rows, and optionally delete them.
        """
        from PyQt5 import QtWidgets, QtCore
        import sqlite3, datetime
        from analyze_qt import DB_FILENAME

        # 1) Open DB
        conn = sqlite3.connect(DB_FILENAME)
        cur  = conn.cursor()

        # 2) Build dialog
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Filter Logs by Date")
        dlg.resize(700, 400)
        vbox = QtWidgets.QVBoxLayout(dlg)

        form = QtWidgets.QFormLayout()
        start_date = QtWidgets.QDateEdit(QtCore.QDate.currentDate())
        start_date.setDisplayFormat("yyyy-MM-dd")
        start_date.setCalendarPopup(True)
        form.addRow("Start Date:", start_date)

        end_date = QtWidgets.QDateEdit(QtCore.QDate.currentDate())
        end_date.setDisplayFormat("yyyy-MM-dd")
        end_date.setCalendarPopup(True)
        form.addRow("End Date:", end_date)

        vbox.addLayout(form)

        # 3) Table for preview
        table = QtWidgets.QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels([
            "ID","File","Method","Timestamp","Measured V"
        ])
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        vbox.addWidget(table)

        # 4) Preview & Delete buttons
        btn_preview = QtWidgets.QPushButton("Preview Matches")
        btn_delete  = QtWidgets.QPushButton("Delete Matches")
        btn_box = QtWidgets.QHBoxLayout()
        btn_box.addWidget(btn_preview)
        btn_box.addStretch()
        btn_box.addWidget(btn_delete)
        vbox.addLayout(btn_box)

        def load_preview():
            # build full-day datetime range
            d0 = start_date.date()
            d1 = end_date.date()
            start_dt = datetime.datetime(d0.year(), d0.month(), d0.day(),   0,  0,  0)
            end_dt   = datetime.datetime(d1.year(), d1.month(), d1.day(),  23, 59, 59)

            cur.execute(
                "SELECT id, file_name, method, timestamp, measured_voltage "
                "FROM measurements WHERE timestamp BETWEEN ? AND ?",
                (start_dt, end_dt)
            )
            rows = cur.fetchall()
            table.setRowCount(len(rows))
            for r, row in enumerate(rows):
                for c, val in enumerate(row):
                    table.setItem(r, c, QtWidgets.QTableWidgetItem(str(val)))
            table.resizeColumnsToContents()

        def delete_matches():
            reply = QtWidgets.QMessageBox.question(
                self, "Confirm Delete",
                "Delete all previewed rows?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return
            d0 = start_date.date()
            d1 = end_date.date()
            start_dt = datetime.datetime(d0.year(), d0.month(), d0.day(),   0,  0,  0)
            end_dt   = datetime.datetime(d1.year(), d1.month(), d1.day(),  23, 59, 59)

            cur.execute(
                "DELETE FROM measurements WHERE timestamp BETWEEN ? AND ?",
                (start_dt, end_dt)
            )
            conn.commit()
            QtWidgets.QMessageBox.information(self, "Deleted", "Rows removed.")
            table.setRowCount(0)

        btn_preview.clicked.connect(load_preview)
        btn_delete.clicked.connect(delete_matches)

        dlg.exec_()
        conn.close()



    def filter_measurements_popup(self):
        """
        Popup to filter (and optionally delete) rows from the measurements
        table by filename and analysis method, removing outliers such as
        zero frequencies or frequencies outside a user-specified range.
        Shows matching rows in a table (styled like the Auto-Analyse popup)
        before asking for confirmation.
        """

        # 1) Open DB & build initial dialog
        conn = sqlite3.connect(DB_FILENAME)
        cur  = conn.cursor()

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Filter Measurement Log")
        dlg.resize(600, 300)                     # ← make initial dialog bigger
        dlg.setMinimumSize(600, 300)
        vbox = QtWidgets.QVBoxLayout(dlg)
        form = QtWidgets.QFormLayout()

        # File selector
        file_cb = QtWidgets.QComboBox()
        cur.execute("SELECT DISTINCT file_name FROM measurements;")
        files = [r[0] for r in cur.fetchall()]
        file_cb.addItems(files)
        form.addRow("File:", file_cb)

        # Method selector
        method_cb = QtWidgets.QComboBox()
        form.addRow("Analysis Method:", method_cb)
        def update_methods(idx):
            method_cb.clear()
            fname = files[idx]
            cur.execute(
                "SELECT DISTINCT method FROM measurements WHERE file_name = ?;",
                (fname,)
            )
            method_cb.addItems([r[0] for r in cur.fetchall()])
        file_cb.currentIndexChanged.connect(update_methods)
        update_methods(0)

        # Outlier options
        zero_chk = QtWidgets.QCheckBox("Remove zero frequencies")
        zero_chk.setChecked(True)
        form.addRow(zero_chk)

        min_edit = QtWidgets.QLineEdit()
        min_edit.setPlaceholderText("e.g. 20.0")
        form.addRow("Minimum Frequency (Hz):", min_edit)

        max_edit = QtWidgets.QLineEdit()
        max_edit.setPlaceholderText("e.g. 20000.0")
        form.addRow("Maximum Frequency (Hz):", max_edit)

        vbox.addLayout(form)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal, dlg
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        vbox.addWidget(btns)

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            conn.close()
            return

        # 2) Gather inputs and build WHERE clause
        sel_file   = files[file_cb.currentIndex()]
        sel_method = method_cb.currentText()
        remove_zero = zero_chk.isChecked()
        try:
            min_f = float(min_edit.text()) if min_edit.text().strip() else None
            max_f = float(max_edit.text()) if max_edit.text().strip() else None
        except ValueError:
            QtWidgets.QMessageBox.critical(self, "Error", "Invalid min/max frequency.")
            conn.close()
            return

        criteria = ["file_name = ?", "method = ?"]
        params    = [sel_file, sel_method]
        or_clauses = []
        if remove_zero:
            or_clauses.append("target_frequency = 0")
        if min_f is not None:
            or_clauses.append("target_frequency < ?")
            params.append(min_f)
        if max_f is not None:
            or_clauses.append("target_frequency > ?")
            params.append(max_f)
        if not or_clauses:
            QtWidgets.QMessageBox.information(self, "Nothing to do",
                "No outlier criteria selected.")
            conn.close()
            return

        where_sql = " AND ".join(criteria) + " AND (" + " OR ".join(or_clauses) + ")"

        # 3) Fetch matching rows
        cur.execute(f"""
            SELECT id, target_frequency, measured_voltage, start_time, end_time
            FROM measurements
            WHERE {where_sql}
            ORDER BY id
        """, params)
        rows_to_remove = cur.fetchall()
        conn.close()

        if not rows_to_remove:
            QtWidgets.QMessageBox.information(self, "No Matches",
                "No rows match the specified criteria.")
            return

        # 4) Show confirmation dialog with a table view
        review = QtWidgets.QDialog(self)
        review.setWindowTitle(f"Remove {len(rows_to_remove)} Rows?")
        review.resize(800, 400)                  # ← make review dialog larger
        review.setMinimumSize(800, 400)
        rvbox = QtWidgets.QVBoxLayout(review)

        info = QtWidgets.QLabel(f"The following {len(rows_to_remove)} rows will be removed:")
        rvbox.addWidget(info)

        # Table view
        table = QtWidgets.QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels([
            "ID", "Frequency (Hz)", "Vrms (V)", "Start (s)", "End (s)"
        ])
        table.setRowCount(len(rows_to_remove))
        for i, (rid, freq, vrms, t0, t1) in enumerate(rows_to_remove):
            table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(rid)))
            table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{freq:.2f}"))
            table.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{vrms:.4f}"))
            table.setItem(i, 3, QtWidgets.QTableWidgetItem(f"{t0:.3f}"))
            table.setItem(i, 4, QtWidgets.QTableWidgetItem(f"{t1:.3f}"))
        table.resizeColumnsToContents()
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        rvbox.addWidget(table)

        rbtns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Yes | QtWidgets.QDialogButtonBox.No,
            QtCore.Qt.Horizontal, review
        )
        rbtns.accepted.connect(review.accept)
        rbtns.rejected.connect(review.reject)
        rvbox.addWidget(rbtns)

        if review.exec_() != QtWidgets.QDialog.Accepted:
            return  # user cancelled

        # 5) Delete and commit
        conn = sqlite3.connect(DB_FILENAME)
        cur  = conn.cursor()
        cur.execute(f"DELETE FROM measurements WHERE {where_sql}", params)
        deleted = cur.rowcount
        conn.commit()
        conn.close()

        QtWidgets.QMessageBox.information(
            self,
            "Done",
            f"Removed {deleted} rows from measurements."
        )


    def database_maintenance_popup(self):
        """
        Popup offering VACUUM (rebuild) and Backup (copy DB file).
        """
        from PyQt5 import QtWidgets
        import sqlite3, shutil, os
        from analyze_qt import DB_FILENAME

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Database Maintenance")
        dlg.resize(400, 200)
        vbox = QtWidgets.QVBoxLayout(dlg)

        btn_vacuum = QtWidgets.QPushButton("Rebuild Database (VACUUM)")
        btn_backup = QtWidgets.QPushButton("Backup Database…")
        vbox.addWidget(btn_vacuum)
        vbox.addWidget(btn_backup)

        def do_vacuum():
            conn = sqlite3.connect(DB_FILENAME)
            conn.execute("VACUUM;")
            conn.close()
            QtWidgets.QMessageBox.information(self, "Vacuumed", "Database rebuilt successfully.")

        def do_backup():
            fname, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Backup DB To", "", "SQLite DB (*.db *.sqlite)"
            )
            if not fname:
                return
            shutil.copy2(DB_FILENAME, fname)
            QtWidgets.QMessageBox.information(self, "Backed Up", f"Copied to {fname}.")

        btn_vacuum.clicked.connect(do_vacuum)
        btn_backup.clicked.connect(do_backup)

        dlg.exec_()


    def clean_measurement_data_popup(self):
        """
        Data Cleaning Helpers:
        • Identify & merge duplicate methods per file within a time window
        • Detect & remove exact duplicate rows
        • Highlight misordered measurements based on frequency order toggle
        • Allow inline editing and deletion of measurements
        """
        from PyQt5 import QtWidgets, QtCore
        import sqlite3, datetime as dt, itertools
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        from analyze_qt import DB_FILENAME

        # Setup dialog and layout
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Clean Measurement Data")
        dlg.resize(1000, 800)
        dlg.setStyleSheet("background-color:#19232D;color:white;")
        layout = QtWidgets.QVBoxLayout(dlg)
        tabs = QtWidgets.QTabWidget()
        tabs.setStyleSheet("QTabBar::tab { background: #3E6C8A; color:white; padding:8px; }")
        layout.addWidget(tabs)

        # Database
        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()

        # --- Tab 1: Method Duplicates ---
        tab1 = QtWidgets.QWidget(); v1 = QtWidgets.QVBoxLayout(tab1)
        tabs.addTab(tab1, "Method Duplicates")
        # File selector & tolerance
        hb1 = QtWidgets.QHBoxLayout();
        hb1.addWidget(QtWidgets.QLabel("File:")); file_cb1 = QtWidgets.QComboBox(); hb1.addWidget(file_cb1);
        hb1.addWidget(QtWidgets.QLabel("Time tolerance (s):")); time_spin = QtWidgets.QSpinBox(); time_spin.setRange(0,3600); hb1.addWidget(time_spin)
        hb1.addStretch(); v1.addLayout(hb1)
        # Table & merge button
        table1 = QtWidgets.QTableWidget(); table1.setColumnCount(4);
        table1.setHorizontalHeaderLabels(["File","Method","Timestamp","Count"])
        table1.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        table1.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        v1.addWidget(table1)
        btn_merge = QtWidgets.QPushButton("Merge Selected Methods"); v1.addWidget(btn_merge, alignment=QtCore.Qt.AlignRight)

        # --- Tab 2: Exact Duplicates ---
        tab2 = QtWidgets.QWidget(); v2 = QtWidgets.QVBoxLayout(tab2)
        tabs.addTab(tab2, "Exact Duplicates")
        cols = ["file_name","method","target_frequency","start_time","end_time","window_length",
                "max_voltage","bandwidth","measured_voltage","filter_applied","screenshot","misc"]
        table2 = QtWidgets.QTableWidget(); table2.setColumnCount(len(cols)+1)
        table2.setHorizontalHeaderLabels(cols+['Count'])
        table2.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        table2.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        v2.addWidget(table2)
        btn_remove = QtWidgets.QPushButton("Remove Exact Duplicates"); v2.addWidget(btn_remove, alignment=QtCore.Qt.AlignRight)

        # --- Tab 3: Order Check ---
        tab3 = QtWidgets.QWidget(); v3 = QtWidgets.QVBoxLayout(tab3)
        tabs.addTab(tab3, "Order Check")
        hb3 = QtWidgets.QHBoxLayout()
        hb3.addWidget(QtWidgets.QLabel("File:")); file_cb3 = QtWidgets.QComboBox(); hb3.addWidget(file_cb3)
        hb3.addWidget(QtWidgets.QLabel("Method:")); method_cb3 = QtWidgets.QComboBox(); hb3.addWidget(method_cb3)
        hb3.addWidget(QtWidgets.QLabel("Frequency Order:")); order_cb = QtWidgets.QComboBox(); order_cb.addItems(["Descending", "Ascending"]); hb3.addWidget(order_cb)
        check_btn = QtWidgets.QPushButton("Check Misordered Points"); check_btn.setStyleSheet("background-color:#3E6C8A;color:white;"); hb3.addWidget(check_btn)
        save_btn = QtWidgets.QPushButton("Save Edits"); save_btn.setStyleSheet("background-color:#3E6C8A;color:white;"); hb3.addWidget(save_btn)
        hb3.addStretch(); v3.addLayout(hb3)
        splitter3 = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        table3 = QtWidgets.QTableWidget(); table3.setColumnCount(4)
        table3.setHorizontalHeaderLabels(["ID","Frequency (Hz)","Voltage","Delete"])
        table3.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        table3.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked|QtWidgets.QAbstractItemView.SelectedClicked)
        splitter3.addWidget(table3)
        fig3 = Figure(facecolor="#19232D"); ax3 = fig3.add_subplot(111)
        ax3.set_facecolor("#19232D"); ax3.tick_params(colors="white")
        canvas3 = FigureCanvas(fig3); splitter3.addWidget(canvas3)
        v3.addWidget(splitter3)

        # Populate file dropdowns
        cur.execute("SELECT DISTINCT file_name FROM measurements ORDER BY file_name")
        files = [r[0] for r in cur.fetchall()]
        file_cb1.addItems(files); file_cb3.addItems(files)

        # Update methods
        def update_methods(source, target):
            target.clear()
            methods = [m[0] for m in cur.execute(
                "SELECT DISTINCT method FROM measurements WHERE file_name=? ORDER BY method",
                (source.currentText(),)
            )]
            target.addItems(methods)
        file_cb1.currentTextChanged.connect(lambda: update_methods(file_cb1,file_cb1))
        file_cb3.currentTextChanged.connect(lambda: update_methods(file_cb3,method_cb3))
        update_methods(file_cb1,file_cb1); update_methods(file_cb3,method_cb3)

        # Tab1 loader
        method_clusters = []
        def load_method_dups():
            nonlocal method_clusters
            method_clusters=[]; rows=[]
            cur.execute("SELECT rowid,file_name,method,timestamp FROM measurements WHERE file_name=? ORDER BY method,timestamp",(file_cb1.currentText(),))
            tol = dt.timedelta(seconds=time_spin.value()); data=cur.fetchall()
            for (fn,m),grp in itertools.groupby(data,key=lambda x:(x[1],x[2])):
                buf=[]
                for rid,_,_,ts in grp:
                    t0=dt.datetime.fromisoformat(ts)
                    if not buf or t0-buf[-1][1]<=tol: buf.append((rid,t0))
                    else:
                        if len(buf)>1: rows.append((fn,m,buf[0][1],len(buf))); method_clusters.append([r for r,_ in buf])
                        buf=[(rid,t0)]
                if len(buf)>1: rows.append((fn,m,buf[0][1],len(buf))); method_clusters.append([r for r,_ in buf])
            table1.setRowCount(len(rows))
            for i,(fn,m,t0,c) in enumerate(rows):
                table1.setItem(i,0,QtWidgets.QTableWidgetItem(fn))
                table1.setItem(i,1,QtWidgets.QTableWidgetItem(m))
                table1.setItem(i,2,QtWidgets.QTableWidgetItem(t0.isoformat(sep=' ')))
                table1.setItem(i,3,QtWidgets.QTableWidgetItem(str(c)))
            table1.resizeColumnsToContents()

        # Tab2 loader
        def load_exact_dups():
            grp=",".join(cols)
            cur.execute(f"SELECT {grp},COUNT(*) FROM measurements GROUP BY {grp} HAVING COUNT(*)>1")
            dups=cur.fetchall()
            table2.setRowCount(len(dups))
            for i,row in enumerate(dups):
                for j,val in enumerate(row): table2.setItem(i,j,QtWidgets.QTableWidgetItem(str(val)))
            table2.resizeColumnsToContents()

        # Tab3 actions
        def check_order():
            order_sql = "ASC" if order_cb.currentText()=="Ascending" else "DESC"
            data=list(cur.execute(
                f"SELECT id,target_frequency,measured_voltage FROM measurements WHERE file_name=? AND method=? ORDER BY target_frequency {order_sql}",
                (file_cb3.currentText(),method_cb3.currentText())
            ))
            if not data: return
            ids,freqs,volts=zip(*data)
            bad_idx=[i for i in range(1,len(ids)) if ids[i]<=ids[i-1]]
            bad_set=set(bad_idx+[i-1 for i in bad_idx])
            table3.setRowCount(len(data))
            for i,(mid,fq,vt) in enumerate(data):
                itm_id=QtWidgets.QTableWidgetItem(str(mid)); itm_f=QtWidgets.QTableWidgetItem(f"{fq:.2f}"); itm_v=QtWidgets.QTableWidgetItem(f"{vt:.3f}")
                if i in bad_set: [itm.setBackground(QtCore.Qt.red) for itm in (itm_id,itm_f,itm_v)]
                table3.setItem(i,0,itm_id); table3.setItem(i,1,itm_f); table3.setItem(i,2,itm_v)
                btn=QtWidgets.QPushButton("Delete"); btn.setStyleSheet("background-color:#A33;color:white;"); btn.clicked.connect(lambda _,r=i: delete_row(r))
                table3.setCellWidget(i,3,btn)
            table3.resizeColumnsToContents()
            ax3.clear(); ax3.set_facecolor("#19232D"); ax3.tick_params(colors="white")
            ax3.plot(freqs,volts,marker='o')
            for i in bad_idx: ax3.plot(freqs[i],volts[i],marker='o',markersize=10,color='red')
            ax3.set_xlabel('Frequency (Hz)',color='white'); ax3.set_ylabel('Voltage',color='white'); ax3.grid(True,alpha=0.5,color='gray'); canvas3.draw()

        def merge_methods():
            sel=[idx.row() for idx in table1.selectionModel().selectedRows()]
            if not sel: QtWidgets.QMessageBox.information(dlg,"No Selection","Please select rows to merge."); return
            for ix in sel:
                ids=method_clusters[ix]
                cur.execute("SELECT file_name,method FROM measurements WHERE rowid=?",(ids[0],)); fn,orig=cur.fetchone()
                newm,ok=QtWidgets.QInputDialog.getText(dlg,"Merge Method",f"Rename '{orig}' in '{fn}' to:",text=orig)
                if ok and newm.strip()!=orig: ph=",".join("?" for _ in ids); cur.execute(f"UPDATE measurements SET method=? WHERE rowid IN ({ph})",(newm,*ids))
            conn.commit(); load_method_dups(); load_exact_dups(); QtWidgets.QMessageBox.information(dlg,"Merged","Methods merged.")

        def remove_exact():
            grp=",".join(cols)
            cur.execute(f"DELETE FROM measurements WHERE rowid NOT IN(SELECT MIN(rowid) FROM measurements GROUP BY {grp})")
            conn.commit(); load_exact_dups(); load_method_dups(); QtWidgets.QMessageBox.information(dlg,"Removed","Exact duplicates removed.")

        def delete_row(row):
            mid=int(table3.item(row,0).text())
            cur.execute("DELETE FROM measurements WHERE id=?",(mid,)); conn.commit(); check_order()

        def save_edits():
            for r in range(table3.rowCount()):
                try:
                    mid=int(table3.item(r,0).text()); fq=float(table3.item(r,1).text()); vt=float(table3.item(r,2).text())
                    cur.execute("UPDATE measurements SET target_frequency=?,measured_voltage=? WHERE id=?",(fq,vt,mid))
                except: continue
            conn.commit(); QtWidgets.QMessageBox.information(dlg,"Saved","Edits saved."); check_order()

        # Connect signals
        time_spin.valueChanged.connect(load_method_dups)
        file_cb1.currentTextChanged.connect(load_method_dups)
        btn_merge.clicked.connect(merge_methods)
        btn_remove.clicked.connect(remove_exact)
        file_cb3.currentTextChanged.connect(lambda: (update_methods(file_cb3,method_cb3),table3.clearContents()))
        method_cb3.currentTextChanged.connect(lambda: table3.clearContents())
        check_btn.clicked.connect(check_order)
        save_btn.clicked.connect(save_edits)

        # Initial load & show
        load_method_dups(); load_exact_dups(); check_order()
        dlg.showMaximized()
        dlg.exec_(); conn.close()



    def annotate_measurements_popup(self):
        """
        Annotation & Notes:
         • Select file + method
         • Display matching rows with all columns including an editable 'misc' column
         • Annotate All, Show Annotated, Delete All Annotations
         • Search by keyword in Notes
         • On Save, UPDATE measurements SET misc=? WHERE id=?
        """
        from PyQt5 import QtWidgets, QtCore
        import sqlite3
        from analyze_qt import DB_FILENAME

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Annotate Measurements")
        dlg.resize(900, 600)
        layout = QtWidgets.QVBoxLayout(dlg)

        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(measurements)")
        cols = [row[1] for row in cur.fetchall()]
        misc_idx = cols.index('misc') if 'misc' in cols else None

        form = QtWidgets.QFormLayout()
        file_cb = QtWidgets.QComboBox()
        cur.execute("SELECT DISTINCT file_name FROM measurements;")
        files = [r[0] for r in cur.fetchall()]
        file_cb.addItems(files)
        form.addRow("File:", file_cb)
        method_cb = QtWidgets.QComboBox()
        form.addRow("Method:", method_cb)
        def update_methods(idx):
            method_cb.clear()
            cur.execute(
                "SELECT DISTINCT method FROM measurements WHERE file_name=?;",
                (files[idx],)
            )
            method_cb.addItems([r[0] for r in cur.fetchall()])
        file_cb.currentIndexChanged.connect(update_methods)
        update_methods(0)
        layout.addLayout(form)

        # Buttons
        btns_layout = QtWidgets.QHBoxLayout()
        btn_annotate_all = QtWidgets.QPushButton("Annotate All")
        btn_show_annotated = QtWidgets.QPushButton("Show Annotated")
        btn_delete_all = QtWidgets.QPushButton("Delete All Annotations")
        btns_layout.addWidget(btn_annotate_all)
        btns_layout.addWidget(btn_show_annotated)
        btns_layout.addWidget(btn_delete_all)
        btns_layout.addStretch()
        layout.addLayout(btns_layout)

        # Search field
        search_layout = QtWidgets.QHBoxLayout()
        search_layout.addWidget(QtWidgets.QLabel("Search Notes:"))
        search_le = QtWidgets.QLineEdit()
        search_le.setPlaceholderText("Enter keyword...")
        search_layout.addWidget(search_le)
        search_layout.addStretch()
        layout.addLayout(search_layout)

        # Table
        table = QtWidgets.QTableWidget()
        table.setColumnCount(len(cols))
        table.setHorizontalHeaderLabels(cols)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(
            QtWidgets.QAbstractItemView.DoubleClicked |
            QtWidgets.QAbstractItemView.EditKeyPressed
        )
        layout.addWidget(table)

        def load_data(filter_annotated=False):
            fname = file_cb.currentText()
            meth = method_cb.currentText()
            sel_cols = ", ".join(cols)
            cur.execute(
                f"SELECT {sel_cols} FROM measurements WHERE file_name=? AND method=? ORDER BY timestamp",
                (fname, meth)
            )
            rows = cur.fetchall()
            if filter_annotated and misc_idx is not None:
                rows = [r for r in rows if r[misc_idx] and r[misc_idx].strip()]
            kw = search_le.text().strip().lower()
            if kw and misc_idx is not None:
                rows = [r for r in rows if r[misc_idx] and kw in r[misc_idx].lower()]
            table.setRowCount(len(rows))
            for i, row in enumerate(rows):
                for j, val in enumerate(row):
                    item = QtWidgets.QTableWidgetItem(str(val))
                    if j == misc_idx:
                        item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
                    else:
                        item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                    table.setItem(i, j, item)
            table.resizeColumnsToContents()

        file_cb.currentIndexChanged.connect(lambda: load_data(False))
        method_cb.currentIndexChanged.connect(lambda: load_data(False))
        search_le.textChanged.connect(lambda: load_data(False))
        load_data(False)

        btn_annotate_all.clicked.connect(lambda: [table.item(r, misc_idx).setText(
            QtWidgets.QInputDialog.getText(dlg, "Annotate All", "Enter annotation for all rows:")[0]
        ) for r in range(table.rowCount())])
        btn_show_annotated.clicked.connect(lambda: load_data(True))
        btn_delete_all.clicked.connect(lambda: [table.item(r, misc_idx).setText("") for r in range(table.rowCount())])

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel
        )
        layout.addWidget(btns)
        def save_and_close():
            updates = [(table.item(r, misc_idx).text(), int(table.item(r, 0).text()))
                       for r in range(table.rowCount())]
            cur.executemany("UPDATE measurements SET misc=? WHERE id=?", updates)
            conn.commit()
            dlg.accept()
        btns.accepted.connect(save_and_close)
        btns.rejected.connect(dlg.reject)

        dlg.exec_()
        conn.close()


    def annotate_spl_logs_popup(self):
        """
        Annotation & Notes for SPL Calculations:
         • Select file + method
         • Display matching rows with all columns including an editable 'notes' column
         • Annotate All, Show Annotated, Delete All Annotations
         • Search by keyword in Notes
         • On Save, UPDATE SPL_calculations SET notes=? WHERE id=?
        """
        from PyQt5 import QtWidgets, QtCore
        import sqlite3
        from analyze_qt import DB_FILENAME

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Annotate SPL Calculations")
        dlg.resize(900, 600)
        layout = QtWidgets.QVBoxLayout(dlg)

        # Connect DB and ensure notes column exists
        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        try:
            cur.execute("ALTER TABLE SPL_calculations ADD COLUMN notes TEXT;")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # already exists

        # Fetch full schema
        cur.execute("PRAGMA table_info(SPL_calculations)")
        cols = [row[1] for row in cur.fetchall()]
        notes_idx = cols.index('notes')
        # Identify method-like column if present
        method_cols = [c for c in cols if c.lower()=='method']
        method_col = method_cols[0] if method_cols else None

        # File + Method selectors
        form = QtWidgets.QFormLayout()
        file_cb = QtWidgets.QComboBox()
        cur.execute("SELECT DISTINCT file_name FROM SPL_calculations;")
        files = [r[0] for r in cur.fetchall()]
        file_cb.addItems(files)
        form.addRow("File:", file_cb)
        method_cb = None
        if method_col:
            method_cb = QtWidgets.QComboBox()
            form.addRow("Method:", method_cb)
            def update_methods(idx):
                method_cb.clear()
                cur.execute(
                    f"SELECT DISTINCT {method_col} FROM SPL_calculations WHERE file_name=?;",
                    (files[idx],)
                )
                method_cb.addItems([str(r[0]) for r in cur.fetchall()])
            file_cb.currentIndexChanged.connect(update_methods)
            update_methods(0)
        layout.addLayout(form)

        # Action buttons
        btn_layout = QtWidgets.QHBoxLayout()
        btn_all = QtWidgets.QPushButton("Annotate All")
        btn_show = QtWidgets.QPushButton("Show Annotated")
        btn_delete = QtWidgets.QPushButton("Delete All Annotations")
        btn_layout.addWidget(btn_all)
        btn_layout.addWidget(btn_show)
        btn_layout.addWidget(btn_delete)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Search field
        search_layout = QtWidgets.QHBoxLayout()
        search_layout.addWidget(QtWidgets.QLabel("Search Notes:"))
        search_le = QtWidgets.QLineEdit()
        search_le.setPlaceholderText("Enter keyword...")
        search_layout.addWidget(search_le)
        search_layout.addStretch()
        layout.addLayout(search_layout)

        # Table
        table = QtWidgets.QTableWidget()
        table.setColumnCount(len(cols))
        table.setHorizontalHeaderLabels(cols)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked | QtWidgets.QAbstractItemView.EditKeyPressed)
        layout.addWidget(table)

        def load_data(filter_ann=False):
            fname = file_cb.currentText()
            args = [fname]
            where = "file_name=?"
            if method_cb:
                where += f" AND {method_col}=?"
                args.append(method_cb.currentText())
            query = f"SELECT {', '.join(cols)} FROM SPL_calculations WHERE {where} ORDER BY timestamp"
            cur.execute(query, args)
            rows = cur.fetchall()
            if filter_ann:
                rows = [r for r in rows if r[notes_idx] and r[notes_idx].strip()]
            kw = search_le.text().strip().lower()
            if kw:
                rows = [r for r in rows if r[notes_idx] and kw in r[notes_idx].lower()]
            table.setRowCount(len(rows))
            for i, row in enumerate(rows):
                for j, val in enumerate(row):
                    item = QtWidgets.QTableWidgetItem(str(val))
                    if j == notes_idx:
                        item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
                    else:
                        item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                    table.setItem(i, j, item)
            table.resizeColumnsToContents()

        file_cb.currentIndexChanged.connect(lambda: load_data(False))
        if method_cb:
            method_cb.currentIndexChanged.connect(lambda: load_data(False))
        search_le.textChanged.connect(lambda: load_data(False))
        load_data(False)

        btn_all.clicked.connect(lambda: [table.item(r, notes_idx).setText(
            QtWidgets.QInputDialog.getText(dlg, "Annotate All", "Enter annotation for all rows:")[0]
        ) for r in range(table.rowCount())])
        btn_show.clicked.connect(lambda: load_data(True))
        btn_delete.clicked.connect(lambda: [table.item(r, notes_idx).setText("") for r in range(table.rowCount())])

        # Save/Cancel
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
        layout.addWidget(btns)
        def save_and_close():
            updates = [(table.item(r, notes_idx).text(), int(table.item(r, 0).text()))
                       for r in range(table.rowCount())]
            cur.executemany("UPDATE SPL_calculations SET notes=? WHERE id=?", updates)
            conn.commit()
            dlg.accept()
        btns.accepted.connect(save_and_close)
        btns.rejected.connect(dlg.reject)

        dlg.exec_()
        conn.close()


    

    def tvr_curve_manager_popup(self):
        """
        TVR Curve Manager (maximized):
        - Dropdown lists curves (by curve_name)
        - Import CSV (Frequency Hz, TVR_dB), name curve, resamples to 1 Hz grid
        - Editable table: Frequency (Hz) vs TVR_dB
        - Overlay different curves (auto-colored from self.color_options)
        - Axis limits: set X min/max (Hz) and Y min/max (dB). Apply / Fit Data / Reset.
        - Export CSV/Excel, Save to DB, Delete curve, Save graph (dark or light)
        """
        import os, sqlite3, json, numpy as np, pandas as pd
        from PyQt5 import QtWidgets, QtCore
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("TVR Curve Manager")
        dlg.setWindowState(QtCore.Qt.WindowMaximized)
        dlg.setStyleSheet("background-color: #19232D; color: white;")
        layout = QtWidgets.QVBoxLayout(dlg)

        # --- DB bootstrap + names
        conn = sqlite3.connect(DB_FILENAME)
        cur  = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS tvr_curves (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                curve_name TEXT UNIQUE,
                min_frequency INTEGER,
                max_frequency INTEGER,
                tvr_json TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        cur.execute("SELECT curve_name FROM tvr_curves ORDER BY curve_name")
        names = [r[0] for r in cur.fetchall()]

        # --- State
        overlay_cache  = {}  # name -> (freqs, tvr_vals)
        overlay_colors = {}  # name -> hex
        current_name   = None

        # ── Top bar: curve selector + import + save/delete/export ───────────────
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Curve:"))
        cb = QtWidgets.QComboBox()
        cb.addItems(names)
        cb.setEditable(False)
        cb.setMinimumWidth(260)
        top.addWidget(cb)

        btn_import = QtWidgets.QPushButton("Import CSV…")
        btn_save   = QtWidgets.QPushButton("Save to DB")
        btn_delete = QtWidgets.QPushButton("Delete Curve")
        btn_export = QtWidgets.QPushButton("Export…")
        top.addStretch()
        for b in (btn_import, btn_save, btn_delete, btn_export):
            top.addWidget(b)
        layout.addLayout(top)

        # ── Middle: table + plot + overlay tools ────────────────────────────────
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        left  = QtWidgets.QWidget()
        lyt   = QtWidgets.QVBoxLayout(left)

        # Table
        table = QtWidgets.QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Frequency (Hz)", "TVR (dB re 1 μPa/V @1 m)"])
        table.verticalHeader().setVisible(False)
        lyt.addWidget(table)

        # Overlay controls
        ov_row = QtWidgets.QHBoxLayout()
        ov_row.addWidget(QtWidgets.QLabel("Overlays:"))
        ov_cb = QtWidgets.QComboBox(); ov_cb.setMinimumWidth(220)
        ov_add = QtWidgets.QPushButton("Add Overlay")
        ov_clear = QtWidgets.QPushButton("Clear Overlays")
        ov_row.addWidget(ov_cb); ov_row.addWidget(ov_add); ov_row.addWidget(ov_clear); ov_row.addStretch()
        lyt.addLayout(ov_row)

        split.addWidget(left)

        # Plot
        fig = Figure(facecolor="#19232D")
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.set_facecolor("#19232D")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_color("white")

        # Plot controls: axis limits + color + save
        pc = QtWidgets.QHBoxLayout()

        color_options = getattr(self, "color_options", {
            "Aqua": "#03DFE2", "Purple": "#A78BFA", "Green": "#10B981", "Orange": "#F59E0B",
            "Blue": "#60A5FA", "Pink": "#F472B6", "Lime": "#84CC16", "Amber": "#F59E0B"
        })

        # Axis limit inputs
        x_min = QtWidgets.QDoubleSpinBox(); x_min.setDecimals(2); x_min.setRange(0.01, 1e9); x_min.setValue(100.0)
        x_max = QtWidgets.QDoubleSpinBox(); x_max.setDecimals(2); x_max.setRange(0.02, 1e9); x_max.setValue(10000.0)
        y_min = QtWidgets.QDoubleSpinBox(); y_min.setDecimals(2); y_min.setRange(-300.0, 300.0); y_min.setValue(-200.0)
        y_max = QtWidgets.QDoubleSpinBox(); y_max.setDecimals(2); y_max.setRange(-300.0, 300.0); y_max.setValue(200.0)

        # Buttons
        btn_apply  = QtWidgets.QPushButton("Apply Limits")
        btn_fit    = QtWidgets.QPushButton("Fit Data")
        btn_reset  = QtWidgets.QPushButton("Reset")

        # Color + Save
        color_cb = QtWidgets.QComboBox(); color_cb.addItems(list(color_options.keys()))
        save_png = QtWidgets.QPushButton("Save Graph…")

        pc.addWidget(QtWidgets.QLabel("X min (Hz):")); pc.addWidget(x_min)
        pc.addWidget(QtWidgets.QLabel("X max (Hz):")); pc.addWidget(x_max)
        pc.addSpacing(12)
        pc.addWidget(QtWidgets.QLabel("Y min (dB):")); pc.addWidget(y_min)
        pc.addWidget(QtWidgets.QLabel("Y max (dB):")); pc.addWidget(y_max)
        pc.addSpacing(12)
        pc.addWidget(btn_apply); pc.addWidget(btn_fit); pc.addWidget(btn_reset)
        pc.addStretch()
        pc.addWidget(QtWidgets.QLabel("Line Color:")); pc.addWidget(color_cb)
        pc.addWidget(save_png)

        right = QtWidgets.QWidget(); rlyt = QtWidgets.QVBoxLayout(right)
        rlyt.addLayout(pc)
        rlyt.addWidget(canvas, 1)
        split.addWidget(right)
        layout.addWidget(split, 1)

        # ── Bottom buttons ──────────────────────────────────────────────────────
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        layout.addWidget(btns)

        # ── BLUE BUTTON STYLES (scoped) ─────────────────────────────────────────
        BLUE_BTN = """
        QPushButton {
            background: #3E6C8A;
            color: white;
            border: 1px solid #4A7DA1;
            border-radius: 6px;
            padding: 6px 12px;
        }
        QPushButton:hover { background: #4A7DA1; }
        QPushButton:pressed { background: #345A73; }
        QPushButton:disabled {
            background: #263946; color: #9aa7b4; border: 1px solid #2F4A5F;
        }
        """
        for b in (btn_import, btn_save, btn_delete, btn_export, ov_add, ov_clear, save_png, btn_apply, btn_fit, btn_reset):
            b.setStyleSheet(BLUE_BTN)
        for b in btns.buttons():
            b.setStyleSheet(BLUE_BTN)

        # ── Overlay color allocator ──────────────────────────────────────────────
        overlay_palette = list(color_options.values())
        _overlay_idx = 0
        def _next_overlay_color():
            nonlocal _overlay_idx
            if not overlay_palette:
                return "#AAAAAA"
            # try to avoid main color
            main_col = color_options.get(color_cb.currentText(), "#03DFE2")
            tried = 0
            while tried < len(overlay_palette):
                c = overlay_palette[_overlay_idx % len(overlay_palette)]
                _overlay_idx += 1; tried += 1
                if c.lower() != main_col.lower():
                    return c
            return overlay_palette[(_overlay_idx - 1) % len(overlay_palette)]

        def _reseed_overlay_colors():
            nonlocal _overlay_idx
            _overlay_idx = 0
            for k in list(overlay_colors.keys()):
                overlay_colors[k] = _next_overlay_color()

        # --- Axis limits state
        auto_fit = True          # when True, axes follow data bounds
        xlim_user = None         # tuple (xmin, xmax) or None
        ylim_user = None         # tuple (ymin, ymax) or None

        # ---- Helpers -----------------------------------------------------------
        def save_tvr_curve(curve_name, min_freq, max_freq, tvr_list, file_name=None):
            conn2 = sqlite3.connect(DB_FILENAME)
            cur2 = conn2.cursor()
            cur2.execute(
                "INSERT OR REPLACE INTO tvr_curves (curve_name, min_frequency, max_frequency, tvr_json) VALUES (?, ?, ?, ?)",
                (curve_name, int(min_freq), int(max_freq), json.dumps(tvr_list))
            )
            conn2.commit()
            conn2.close()

        def import_tvr_csv(file_path):
            """Read flexible TVR CSV → (min_freq, max_freq, 1Hz TVR list)."""
            import csv, math
            with open(file_path, "r", newline="", encoding="utf-8") as fh:
                rdr = csv.reader(fh)
                rows = [r for r in rdr]
            if not rows:
                raise ValueError("CSV is empty.")

            def _clean(s):
                return "".join(ch.lower() for ch in s if ch.isalnum() or ch in (" ", "_", "-", "/", "(", ")", "[", "]"))
            freq_idx = tvr_idx = None
            if rows and any(c.strip() for c in rows[0]):
                hdr = [_clean(c) for c in rows[0]]
                def find_idx(h, needles):
                    for i, val in enumerate(h):
                        for n in needles:
                            if n in val: return i
                    return None
                freq_idx = find_idx(hdr, ("f","freq","frequency","hz"))
                tvr_idx  = find_idx(hdr, ("tvr","tvr_db","tvr (db","tvr dB","db re 1","μpa/v","upa/v"))
                data = rows[1:] if (freq_idx is not None or tvr_idx is not None) else rows
            else:
                data = rows

            if freq_idx is None or tvr_idx is None:
                # fallback: infer numeric columns and choose likely frequency vs TVR
                max_cols = max(len(r) for r in data) if data else 0
                numeric_cols = []
                for ci in range(max_cols):
                    vals = []
                    for r in data:
                        if ci < len(r):
                            try:
                                vals.append(float(r[ci]))
                            except Exception:
                                pass
                    if vals:
                        arr = np.asarray(vals, dtype=float)
                        # Frequency is usually positive, mostly monotonic in file order,
                        # and tends to span a wider range than TVR values.
                        pos_ratio = float(np.mean(arr > 0))
                        spread = float((np.nanmax(arr) - np.nanmin(arr)) if arr.size else 0.0)
                        if arr.size > 1:
                            monotonic_ratio = float(np.mean(np.diff(arr) >= 0))
                        else:
                            monotonic_ratio = 0.0
                        freq_score = (pos_ratio * 100.0) + (monotonic_ratio * 50.0) + spread
                        numeric_cols.append((ci, len(vals), freq_score))

                if len(numeric_cols) < 2:
                    raise ValueError("Could not identify Frequency/TVR columns in CSV.")

                # Prefer columns with most numeric values, then best frequency score.
                numeric_cols.sort(key=lambda t: (t[1], t[2]), reverse=True)
                if freq_idx is None:
                    freq_idx = max(numeric_cols, key=lambda t: (t[1], t[2]))[0]

                if tvr_idx is None:
                    remaining = [t for t in numeric_cols if t[0] != freq_idx]
                    if not remaining:
                        raise ValueError("Could not identify Frequency/TVR columns in CSV.")
                    # TVR column: still numeric-rich, but does not need freq-like score.
                    tvr_idx = max(remaining, key=lambda t: t[1])[0]

            freqs, tvals = [], []
            for r in data:
                if max(freq_idx, tvr_idx) >= len(r): continue
                try:
                    f = float(r[freq_idx]); v = float(r[tvr_idx])
                except Exception:
                    continue
                if f > 0 and np.isfinite(f) and np.isfinite(v):
                    freqs.append(f); tvals.append(v)
            if not freqs:
                raise ValueError("No valid frequency/TVR rows found.")
            freqs = np.asarray(freqs, dtype=float); tvals = np.asarray(tvals, dtype=float)
            order = np.argsort(freqs); freqs = freqs[order]; tvals = tvals[order]

            # de-dup identical freqs by averaging
            uniq_f, uniq_v = [], []
            i = 0
            while i < len(freqs):
                f0 = freqs[i]; j = i+1; s = tvals[i]; c = 1
                while j < len(freqs) and freqs[j] == f0:
                    s += tvals[j]; c += 1; j += 1
                uniq_f.append(f0); uniq_v.append(s/c); i = j
            x = np.asarray(uniq_f, dtype=float)
            y = np.asarray(uniq_v, dtype=float)
            if x.size < 2:
                raise ValueError("Need at least two frequency points.")

            fmin = int(np.ceil(x.min())); fmax = int(np.floor(x.max()))
            if fmax <= fmin:
                raise ValueError("Frequency range too narrow.")

            grid = np.arange(fmin, fmax+1, dtype=int)
            try:
                y_grid = np.interp(np.log10(grid), np.log10(x), y)
            except Exception:
                y_grid = np.interp(grid.astype(float), x, y)
            return fmin, fmax, y_grid.tolist()

        def load_curve_by_name(name):
            nonlocal current_name
            current_name = name
            table.blockSignals(True)
            table.setRowCount(0)
            if not name:
                table.blockSignals(False); return
            cur.execute("SELECT min_frequency, max_frequency, tvr_json FROM tvr_curves WHERE curve_name=?", (name,))
            row = cur.fetchone()
            if not row:
                table.blockSignals(False); return
            fmin, fmax, js = row
            arr = np.array(json.loads(js), dtype=float)
            freqs = np.arange(int(fmin), int(fmax)+1, dtype=int)
            table.setRowCount(freqs.size)
            for i,(f,v) in enumerate(zip(freqs, arr)):
                table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(int(f))))
                table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{float(v):.6f}"))
            table.resizeColumnsToContents()
            fit_data_limits()  # sync limits to data (if auto)
            redraw()

        def collect_table():
            freqs, vals = [], []
            for r in range(table.rowCount()):
                fi = table.item(r,0); vi = table.item(r,1)
                if not fi or not vi: continue
                try:
                    f = float(fi.text()); v = float(vi.text())
                    if f>0: freqs.append(f); vals.append(v)
                except ValueError:
                    continue
            if not freqs: return None
            freqs = np.array(freqs, dtype=float)
            vals  = np.array(vals,  dtype=float)
            idx = np.argsort(freqs)
            freqs = freqs[idx]; vals = vals[idx]
            fmin, fmax = int(np.ceil(freqs.min())), int(np.floor(freqs.max()))
            grid = np.arange(fmin, fmax+1, dtype=int)
            vals_g = np.interp(np.log10(grid), np.log10(freqs), vals)
            return fmin, fmax, grid, vals_g

        # ---- Axis limit helpers
        def current_data_bounds():
            """Return (xmin, xmax, ymin, ymax) from main + overlays."""
            xs = []; ys = []
            main = collect_table()
            if main:
                _,_,fx,vy = main
                if len(fx): xs.extend([np.min(fx), np.max(fx)])
                if len(vy): ys.extend([np.min(vy), np.max(vy)])
            for _, (fx, vy) in overlay_cache.items():
                if len(fx): xs.extend([np.min(fx), np.max(fx)])
                if len(vy): ys.extend([np.min(vy), np.max(vy)])
            if not xs or not ys:
                # fallbacks
                return 100.0, 10000.0, -100.0, 100.0
            # Add a small 5% pad
            xmin, xmax = float(np.min(xs)), float(np.max(xs))
            ymin, ymax = float(np.min(ys)), float(np.max(ys))
            if xmax > xmin:
                pad = 0.05 * (xmax - xmin)
                xmin -= pad; xmax += pad
            if ymax > ymin:
                pad = 0.05 * (ymax - ymin)
                ymin -= pad; ymax += pad
            # clamp sensible ranges
            xmin = max(xmin, 0.01)
            return xmin, xmax, ymin, ymax

        def fit_data_limits():
            nonlocal auto_fit, xlim_user, ylim_user
            xmin, xmax, ymin, ymax = current_data_bounds()
            auto_fit = True
            xlim_user = (xmin, xmax)
            ylim_user = (ymin, ymax)
            # populate widgets
            x_min.blockSignals(True); x_max.blockSignals(True); y_min.blockSignals(True); y_max.blockSignals(True)
            x_min.setValue(max(0.01, xmin)); x_max.setValue(max(x_min.value()+0.01, xmax))
            y_min.setValue(ymin); y_max.setValue(max(ymin + 0.01, ymax))
            x_min.blockSignals(False); x_max.blockSignals(False); y_min.blockSignals(False); y_max.blockSignals(False)

        def reset_limits():
            nonlocal auto_fit, xlim_user, ylim_user
            auto_fit = True
            xlim_user = None
            ylim_user = None
            fit_data_limits()
            redraw()

        def apply_limits():
            nonlocal auto_fit, xlim_user, ylim_user
            xmin = float(x_min.value()); xmax = float(x_max.value())
            ymin = float(y_min.value()); ymax = float(y_max.value())
            if xmax <= xmin:
                QtWidgets.QMessageBox.warning(dlg, "Axis Limits", "X max must be greater than X min.")
                return
            if ymax <= ymin:
                QtWidgets.QMessageBox.warning(dlg, "Axis Limits", "Y max must be greater than Y min.")
                return
            auto_fit = False
            xlim_user = (xmin, xmax)
            ylim_user = (ymin, ymax)
            redraw()

        # initial limits seed
        fit_data_limits()

        def redraw():
            ax.clear()
            ax.set_facecolor("#19232D")
            ax.tick_params(colors="white")
            for sp in ax.spines.values():
                sp.set_color("white")

            # main curve from table
            main = collect_table()
            if main:
                _,_,fx,vy = main
                color_hex = color_options.get(color_cb.currentText(), "#03DFE2")
                ax.plot(fx, vy, "-", lw=2, color=color_hex, label=current_name or "Current")

            # overlays
            for name,(fx,vy) in overlay_cache.items():
                col = overlay_colors.get(name)
                if col is None:
                    col = _next_overlay_color()
                    overlay_colors[name] = col
                ax.plot(fx, vy, "--", color=col, alpha=0.95, label=name)

            # Scales: standard TVR view: log X, linear Y
            ax.set_xscale("log")
            ax.set_yscale("linear")

            # Axis limits
            if auto_fit:
                xmin, xmax, ymin, ymax = current_data_bounds()
            else:
                xmin, xmax = xlim_user if xlim_user else current_data_bounds()[:2]
                ymin, ymax = ylim_user if ylim_user else current_data_bounds()[2:]
            # Apply and sanity
            xmin = max(xmin, 0.01)
            if xmax <= xmin: xmax = xmin * 10.0
            if ymax <= ymin: ymax = ymin + 1.0

            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

            ax.set_xlabel("Frequency (Hz)", color="white")
            ax.set_ylabel("TVR (dB re 1 μPa/V @1 m)", color="white")
            ax.grid(True, ls="--", alpha=0.4, color="gray")
            ax.legend(facecolor="#19232D", edgecolor="white", labelcolor="white")
            canvas.draw_idle()

        # Wire
        cb.currentTextChanged.connect(lambda n: (load_curve_by_name(n), ov_cb.setCurrentIndex(-1)))
        table.itemChanged.connect(lambda *_: (fit_data_limits() if auto_fit else None, redraw()))
        color_cb.currentIndexChanged.connect(lambda *_: (_reseed_overlay_colors(), redraw()))

        # Axis buttons
        btn_fit.clicked.connect(lambda: (fit_data_limits(), redraw()))
        btn_reset.clicked.connect(reset_limits)
        btn_apply.clicked.connect(apply_limits)

        # overlays
        ov_cb.clear()
        ov_cb.addItems([n for n in names if n])

        def add_overlay():
            name = ov_cb.currentText().strip()
            if not name: return
            if name in overlay_cache: return
            cur.execute("SELECT min_frequency, max_frequency, tvr_json FROM tvr_curves WHERE curve_name=?", (name,))
            row = cur.fetchone()
            if not row: return
            fmin,fmax,js = row
            arr = np.array(json.loads(js), dtype=float)
            freqs = np.arange(int(fmin), int(fmax)+1, dtype=int)
            overlay_cache[name] = (freqs, arr)
            if name not in overlay_colors:
                overlay_colors[name] = _next_overlay_color()
            if auto_fit:
                fit_data_limits()
            redraw()
        ov_add.clicked.connect(add_overlay)
        ov_clear.clicked.connect(lambda: (overlay_cache.clear(), overlay_colors.clear(), fit_data_limits(), redraw()))

        # Import
        def do_import():
            path, _ = QtWidgets.QFileDialog.getOpenFileName(dlg, "Import TVR CSV", "", "CSV Files (*.csv);;All Files (*)")
            if not path: return
            cname, ok = QtWidgets.QInputDialog.getText(dlg, "Curve Name", "Enter a unique TVR curve name:")
            if not ok or not cname.strip(): return
            try:
                fmin, fmax, arr = import_tvr_csv(path)
            except Exception as e:
                QtWidgets.QMessageBox.critical(dlg, "Import Failed", str(e)); return
            save_tvr_curve(cname.strip(), fmin, fmax, arr, os.path.basename(path))
            # refresh lists
            cur.execute("SELECT curve_name FROM tvr_curves ORDER BY curve_name")
            new_names = [r[0] for r in cur.fetchall()]
            cb.blockSignals(True); cb.clear(); cb.addItems(new_names); cb.blockSignals(False)
            ov_cb.clear(); ov_cb.addItems(new_names)
            cb.setCurrentText(cname.strip())
        btn_import.clicked.connect(do_import)

        # Save
        def do_save():
            if table.rowCount() == 0:
                QtWidgets.QMessageBox.information(dlg, "Nothing to save", "No rows in table.")
                return
            res = collect_table()
            if not res:
                QtWidgets.QMessageBox.warning(dlg, "Invalid", "No valid numeric rows.")
                return
            fmin, fmax, _, vals = res
            if not current_name:
                cname, ok = QtWidgets.QInputDialog.getText(dlg, "Curve Name", "Enter curve name:")
                if not ok or not cname.strip(): return
                name = cname.strip()
            else:
                name = current_name
            save_tvr_curve(name, fmin, fmax, vals)
            if name not in [cb.itemText(i) for i in range(cb.count())]:
                cb.addItem(name)
            cb.setCurrentText(name)
            QtWidgets.QMessageBox.information(dlg, "Saved", f"Saved TVR curve: {name}")
        btn_save.clicked.connect(do_save)

        # Delete
        def do_delete():
            name = cb.currentText().strip()
            if not name: return
            if QtWidgets.QMessageBox.question(dlg, "Delete", f"Delete TVR curve '{name}'?") != QtWidgets.QMessageBox.Yes:
                return
            cur.execute("DELETE FROM tvr_curves WHERE curve_name=?", (name,))
            conn.commit()
            # refresh
            cur.execute("SELECT curve_name FROM tvr_curves ORDER BY curve_name")
            rem = [r[0] for r in cur.fetchall()]
            cb.blockSignals(True); cb.clear(); cb.addItems(rem); cb.blockSignals(False)
            ov_cb.clear(); ov_cb.addItems(rem)
            table.setRowCount(0)
            overlay_cache.pop(name, None)
            overlay_colors.pop(name, None)
            fit_data_limits()
            redraw()
        btn_delete.clicked.connect(do_delete)

        # Export (current + overlays)
        def do_export():
            # Build dict of curves: name -> (fx, vy)
            res = collect_table()
            datasets = {}
            if res:
                _, _, fx, vy = res
                datasets[current_name or "Current"] = (fx, vy)
            for name, (fx, vy) in overlay_cache.items():
                datasets[name] = (fx, vy)
            if not datasets:
                QtWidgets.QMessageBox.information(dlg, "No Data", "Nothing to export.")
                return

            choices = [
                "Current curve -> CSV",
                "All curves -> multiple CSVs",
                "All curves -> Excel workbook (.xlsx)"
            ]
            selection_text, ok = QtWidgets.QInputDialog.getItem(
                dlg, "Export", "Choose export format:", choices, 0, False
            )
            if not ok: return
            sel = choices.index(selection_text)

            import re
            def sanitize_filename(s):
                s = re.sub(r"[^\w\-. ]+", "_", s.strip())
                return s or "curve"

            def to_dataframe(freqs, vals):
                return pd.DataFrame({
                    "Frequency_Hz": np.asarray(freqs, dtype=int),
                    "TVR_dB": np.asarray(vals, dtype=float)
                })

            if sel == 0:
                if not res:
                    QtWidgets.QMessageBox.warning(dlg, "No Current Curve", "No editable curve to export.")
                    return
                path, _ = QtWidgets.QFileDialog.getSaveFileName(dlg, "Export CSV", "", "CSV Files (*.csv)")
                if not path: return
                _, _, fx, vy = res
                try:
                    to_dataframe(fx, vy).to_csv(path, index=False)
                    QtWidgets.QMessageBox.information(dlg, "Exported", f"CSV saved to:\n{path}")
                except Exception as e:
                    QtWidgets.QMessageBox.critical(dlg, "Export Failed", str(e))
                return

            if sel == 1:
                out_dir = QtWidgets.QFileDialog.getExistingDirectory(dlg, "Choose output folder")
                if not out_dir: return
                failed = []
                for name, (fx, vy) in datasets.items():
                    fpath = os.path.join(out_dir, sanitize_filename(name) + ".csv")
                    try:
                        to_dataframe(fx, vy).to_csv(fpath, index=False)
                    except Exception as e:
                        failed.append((name, str(e)))
                if failed:
                    msg = "Some files failed:\n" + "\n".join(f"- {n}: {err}" for n, err in failed)
                    QtWidgets.QMessageBox.warning(dlg, "Partial Export", msg)
                else:
                    QtWidgets.QMessageBox.information(dlg, "Exported", f"Saved {len(datasets)} CSVs to:\n{out_dir}")
                return

            if sel == 2:
                path, _ = QtWidgets.QFileDialog.getSaveFileName(dlg, "Export Excel", "", "Excel Workbook (*.xlsx)")
                if not path: return
                if not path.lower().endswith(".xlsx"):
                    path += ".xlsx"
                try:
                    with pd.ExcelWriter(path, engine="openpyxl") as xw:
                        used = set()
                        for name, (fx, vy) in datasets.items():
                            sheet = (name or "Sheet").strip()[:31] or "Sheet"
                            base = sheet; i = 2
                            while sheet in used:
                                sheet = (base[:28] + f"_{i}")[:31]; i += 1
                            used.add(sheet)
                            to_dataframe(fx, vy).to_excel(xw, index=False, sheet_name=sheet)
                    QtWidgets.QMessageBox.information(dlg, "Exported", f"Workbook saved to:\n{path}")
                except Exception as e:
                    QtWidgets.QMessageBox.critical(
                        dlg, "Export Failed",
                        "Could not write Excel workbook.\n"
                        "Install an engine (e.g. `pip install openpyxl` or `pip install xlsxwriter`).\n\n"
                        f"Details:\n{e}"
                    )
                return
        btn_export.clicked.connect(do_export)

        # Save graph (respect axis limits)
        def save_graph():
            path, _ = QtWidgets.QFileDialog.getSaveFileName(dlg, "Save Graph", "", "PNG Files (*.png)")
            if not path: return
            mode, ok = QtWidgets.QInputDialog.getItem(dlg, "Theme", "Choose background:", ["Dark","Light"], 0, False)
            if not ok: return

            # Gather series
            series = []
            main = collect_table()
            if main:
                _,_,fx_main,vy_main = main
                series.append(("__main__", fx_main, vy_main, color_options.get(color_cb.currentText(), "#03DFE2")))
            for name,(fx,vy) in overlay_cache.items():
                series.append((name, fx, vy, overlay_colors.get(name, "#888888")))

            # Limits
            if auto_fit:
                xmin, xmax, ymin, ymax = current_data_bounds()
            else:
                cxmin, cxmax, cymin, cymax = current_data_bounds()
                xmin, xmax = xlim_user if xlim_user else (cxmin, cxmax)
                ymin, ymax = ylim_user if ylim_user else (cymin, cymax)

            if mode == "Light":
                fig2 = Figure(facecolor="white")
                ax2 = fig2.add_subplot(111)
                for name, fx, vy, col in series:
                    style = "-" if name == "__main__" else "--"
                    lw = 2 if name == "__main__" else 1.8
                    ax2.plot(fx, vy, style, lw=lw, color=col if name != "__main__" else None)
                ax2.set_xscale("log"); ax2.set_yscale("linear")
                ax2.set_xlim(max(0.01, xmin), xmax)
                ax2.set_ylim(ymin, ymax)
                ax2.set_xlabel("Frequency (Hz)")
                ax2.set_ylabel("TVR (dB re 1 μPa/V @1 m)")
                ax2.grid(True, ls="--", alpha=0.4, color="gray")
                fig2.savefig(path, dpi=150, facecolor="white")
            else:
                # update on-screen axes to ensure saved fig has the same limits
                ax.set_xlim(max(0.01, xmin), xmax)
                ax.set_ylim(ymin, ymax)
                fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())

            QtWidgets.QMessageBox.information(dlg, "Saved", f"Graph saved to:\n{path}")
        save_png.clicked.connect(save_graph)

        btns.rejected.connect(dlg.reject)

        # initial load
        ov_cb.clear(); ov_cb.addItems([n for n in names if n])
        if names:
            cb.setCurrentIndex(0)
            load_curve_by_name(cb.currentText())
        else:
            redraw()  # ensure empty plot still initializes

        # Open maximized so it fills the screen but keeps window controls.
        dlg.showMaximized()
        dlg.exec_()
        conn.close()







    def calibration_curve_manager_popup(self):
        """
        Hydrophone Calibration Curve Manager (maximized):
        - Dropdown lists curves by curve_name
        - Table: Frequency vs Sensitivity with per-row delete
        - Plot: dark theme, line color selectable, X-axis log toggle
        - Overlay other curves, label/clear intervals
        - Save Graph JPG: dark or light (B/W) modes
        - Export CSV (multi-tab), Save Changes to DB, Close
        """
        # --- heavy imports (ideally moved to module top) ---
        import sqlite3, ast, json
        import numpy as np
        import pandas as pd
        from PyQt5 import QtWidgets, QtCore, QtGui
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

        # --- dialog & layout setup ---
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Hydrophone Calibration Manager")
        dlg.setWindowState(QtCore.Qt.WindowMaximized)
        dlg.setStyleSheet("background-color: #19232D;")
        layout = QtWidgets.QVBoxLayout(dlg)

        # --- database init ---
        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS hydrophone_curves (
                id INTEGER PRIMARY KEY,
                curve_name TEXT UNIQUE,
                min_frequency REAL,
                max_frequency REAL,
                sensitivity_json TEXT
            )
        """)
        conn.commit()

        # --- load curve names ---
        cur.execute("SELECT curve_name FROM hydrophone_curves ORDER BY curve_name")
        names = [r[0] for r in cur.fetchall()]

        # --- parsed-curve cache (avoids repeated DB fetch + JSON parsing) ---
        curve_data_cache = {}

        def get_curve_data(name):
            if not name:
                return None
            cached = curve_data_cache.get(name)
            if cached is not None:
                return cached

            row = cur.execute(
                "SELECT min_frequency, max_frequency, sensitivity_json FROM hydrophone_curves WHERE curve_name=?",
                (name,)
            ).fetchone()
            if not row or not row[2]:
                return None

            minf, maxf, sj = row
            arr = np.array(
                ast.literal_eval(sj) if sj.strip().startswith('[') else json.loads(sj),
                float
            )
            data = (minf, maxf, arr)
            curve_data_cache[name] = data
            return data

        # --- state & caches ---
        overlays = []
        overlay_cache = {}        # curve_name → (freqs, arr)
        interval_texts = []
        interval_dots = []

        # --- debounced redraw timer ---
        def update_graph_from_table():
            ax.clear()
            ax.set_facecolor('#19232D')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')

            # read table data
            freqs, vals = [], []
            for r in range(table.rowCount()):
                fi = table.item(r, 0)
                vi = table.item(r, 1)
                if fi and vi:
                    try:
                        freqs.append(float(fi.text()))
                        vals.append(float(vi.text()))
                    except ValueError:
                        pass

            # main curve
            if freqs:
                draw_curve(np.array(freqs), np.array(vals), '-')

            # overlays from cache
            for f2, arr in overlay_cache.values():
                ax.plot(f2, arr, '--', color='gray')

            ax.set_xscale('log' if log_cb.isChecked() else 'linear')
            ax.set_xlabel('Frequency (Hz)', color='white')
            ax.set_ylabel('Sensitivity (dB re 1V/μPa @1m)', color='white')
            ax.grid(True, linestyle='--', alpha=0.5, color='gray')
            canvas.draw_idle()

        update_timer = QtCore.QTimer(dlg)
        update_timer.setSingleShot(True)
        update_timer.setInterval(200)
        update_timer.timeout.connect(update_graph_from_table)

        # --- helpers ---
        def safe_freqs(minf, maxf, n):
            return np.linspace(minf, maxf, n) if n > 1 and minf is not None and maxf is not None else np.arange(n)

        def draw_curve(freqs, arr, style='-'):
            color = getattr(self, 'graphColorDropdown', None).currentText() if hasattr(self, 'graphColorDropdown') else 'cyan'
            kw = {'linestyle': style, 'color': color}
            if markers_cb.isChecked():
                kw['marker'] = 'o'
            ax.plot(freqs, arr, **kw)

        # --- UI: top bar ---
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Select Curve:"))
        curve_cb = QtWidgets.QComboBox()
        curve_cb.addItems(names)
        top.addWidget(curve_cb)

        log_cb = QtWidgets.QCheckBox("Log X-axis")
        log_cb.setStyleSheet("color:white;")
        top.addWidget(log_cb)

        markers_cb = QtWidgets.QCheckBox("Show Markers")
        markers_cb.setChecked(False)
        markers_cb.setStyleSheet("color:white;")
        top.addWidget(markers_cb)

        btn_overlay = QtWidgets.QPushButton("Overlay Curves...")
        btn_clear_ov = QtWidgets.QPushButton("Clear Overlays")
        btn_label = QtWidgets.QPushButton("Label Intervals...")
        btn_clear_int = QtWidgets.QPushButton("Clear Intervals")
        for b in (btn_overlay, btn_clear_ov, btn_label, btn_clear_int):
            b.setStyleSheet("background-color:#3E6C8A;color:white;padding:6px;border-radius:4px;")
            top.addWidget(b)
        top.addStretch()
        layout.addLayout(top)

        # --- UI: splitter with table & plot ---
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        table = QtWidgets.QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Frequency (Hz)", "Sensitivity (dB)", "del"])
        table.setStyleSheet("background-color:#19232D;color:white;")
        table.setColumnWidth(2, 40)
        table.setEditTriggers(QtWidgets.QAbstractItemView.AllEditTriggers)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        splitter.addWidget(table)

        fig = Figure(facecolor='#19232D')
        ax = fig.add_subplot(111)
        canvas = FigureCanvas(fig)
        splitter.addWidget(canvas)
        layout.addWidget(splitter)

        # --- load a curve into table & plot ---
        def load_curve(name):
            table.blockSignals(True)
            try:
                table.clearContents()
                table.setRowCount(0)
                overlays.clear()
                overlay_cache.clear()
                interval_texts.clear()
                interval_dots.clear()

                curve_data = get_curve_data(name)
                if not curve_data:
                    return

                minf, maxf, arr = curve_data
                freqs = safe_freqs(minf, maxf, arr.size)
                table.setUpdatesEnabled(False)
                table.setRowCount(len(freqs))

                for i, (f, v) in enumerate(zip(freqs, arr)):
                    it = QtWidgets.QTableWidgetItem(f"{f:.2f}")
                    it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
                    it.setForeground(QtCore.Qt.white)
                    table.setItem(i, 0, it)

                    it2 = QtWidgets.QTableWidgetItem(f"{v:.3f}")
                    it2.setForeground(QtCore.Qt.white)
                    table.setItem(i, 1, it2)

                    del_item = QtWidgets.QTableWidgetItem("✕")
                    del_item.setFlags((del_item.flags() & ~QtCore.Qt.ItemIsEditable) | QtCore.Qt.ItemIsEnabled)
                    del_item.setForeground(QtGui.QColor("#FF6B6B"))
                    del_item.setTextAlignment(QtCore.Qt.AlignCenter)
                    table.setItem(i, 2, del_item)

                table.setUpdatesEnabled(True)
                table.resizeColumnsToContents()
            finally:
                table.blockSignals(False)
            update_timer.start()

        # --- overlay selection & caching ---
        def choose_overlays():
            dlg2 = QtWidgets.QDialog(dlg)
            dlg2.setWindowTitle('Select Overlays')
            lay2 = QtWidgets.QVBoxLayout(dlg2)
            lw = QtWidgets.QListWidget()
            lw.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
            for nm in names:
                it = QtWidgets.QListWidgetItem(nm)
                it.setSelected(nm in overlays)
                lw.addItem(it)
            lay2.addWidget(lw)
            bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
            lay2.addWidget(bb)
            bb.accepted.connect(dlg2.accept)
            bb.rejected.connect(dlg2.reject)

            if dlg2.exec_() == QtWidgets.QDialog.Accepted:
                overlays[:] = [it.text() for it in lw.selectedItems()]
                overlay_cache.clear()
                for nm in overlays:
                    curve_data = get_curve_data(nm)
                    if not curve_data:
                        continue
                    minf, maxf, arr = curve_data
                    overlay_cache[nm] = (safe_freqs(minf, maxf, arr.size), arr)
                update_timer.start()

        # --- interval labeling ---
        def label_intervals():
            step, ok = QtWidgets.QInputDialog.getDouble(dlg, 'Label Intervals', 'Interval (Hz):', decimals=1, min=0.1)
            if not ok or step <= 0:
                return
            for t in interval_texts:
                t.remove()
            for d in interval_dots:
                d.remove()
            interval_texts.clear()
            interval_dots.clear()

            freqs = [float(table.item(r, 0).text()) for r in range(table.rowCount())]
            sens = [float(table.item(r, 1).text()) for r in range(table.rowCount())]
            for f0 in np.arange(freqs[0], freqs[-1] + step, step):
                idx = np.abs(np.array(freqs) - f0).argmin()
                x, y = freqs[idx], sens[idx]
                txt = ax.text(x, y, f"{y:.1f}", color='white', fontsize=8, rotation=45, zorder=11)
                d, = ax.plot([x], [y], 'o', markersize=4, color='magenta', zorder=12)
                interval_texts.append(txt)
                interval_dots.append(d)
            canvas.draw_idle()

        # --- commit, delete, export & save graph ---
        def commit_changes():
            name = curve_cb.currentText()
            freqs, sens = [], []
            for r in range(table.rowCount()):
                fi = table.item(r, 0)
                vi = table.item(r, 1)
                if fi and vi:
                    freqs.append(float(fi.text()))
                    sens.append(float(vi.text()))
            if not freqs:
                return
            cur.execute(
                "UPDATE hydrophone_curves SET min_frequency=?, max_frequency=?, sensitivity_json=? WHERE curve_name=?",
                (min(freqs), max(freqs), json.dumps(sens), name)
            )
            conn.commit()
            curve_data_cache.pop(name, None)
            QtWidgets.QMessageBox.information(dlg, 'Saved', 'Changes saved.')

        def delete_db_curve():
            name = curve_cb.currentText()
            resp = QtWidgets.QMessageBox.question(
                dlg, 'Delete Curve?', f"Delete '{name}'?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            )
            if resp != QtWidgets.QMessageBox.Yes:
                return
            cur.execute("DELETE FROM hydrophone_curves WHERE curve_name=?", (name,))
            conn.commit()
            curve_data_cache.pop(name, None)
            idx = curve_cb.findText(name)
            if idx >= 0:
                curve_cb.removeItem(idx)
            dlg.accept()

        def export_csv():
            path, _ = QtWidgets.QFileDialog.getSaveFileName(dlg, 'Export Curves', 'curves.xlsx', 'Excel Files (*.xlsx)')
            if not path:
                return
            with pd.ExcelWriter(path, engine='openpyxl') as writer:
                for nm in [curve_cb.currentText()] + overlays:
                    curve_data = get_curve_data(nm)
                    if not curve_data:
                        continue
                    minf, maxf, arr = curve_data
                    freqs = safe_freqs(minf, maxf, arr.size)
                    df = pd.DataFrame({
                        'Frequency (Hz)': freqs,
                        'Sensitivity (dB re 1V/μPa @1m)': arr
                    })
                    df.to_excel(writer, sheet_name=nm[:31], index=False)
            QtWidgets.QMessageBox.information(dlg, 'Exported', f'Curves exported to {os.path.basename(path)}')

        def save_graph_png():
            fn, _ = QtWidgets.QFileDialog.getSaveFileName(dlg, 'Save Graph as JPG', 'curve.jpg', 'JPEG (*.jpg)')
            if not fn:
                return
            mode = mode_cb.currentText()
            tmp_fig = Figure(figsize=(12, 8), dpi=300)
            tmp_ax = tmp_fig.add_subplot(111)
            if mode.startswith('Light'):
                bg, fg, plot_c = 'white', 'black', 'black'
            else:
                bg, fg = '#19232D', 'white'
                plot_c = getattr(self, 'graphColorDropdown', None).currentText() if hasattr(self, 'graphColorDropdown') else 'cyan'
            tmp_ax.set_facecolor(bg)
            tmp_fig.patch.set_facecolor(bg)

            for idx, nm in enumerate([curve_cb.currentText()] + overlays):
                curve_data = get_curve_data(nm)
                if not curve_data:
                    continue
                minf, maxf, arr = curve_data
                freqs = safe_freqs(minf, maxf, arr.size)
                style = '-' if idx == 0 else '--'
                tmp_ax.plot(freqs, arr, style, color=plot_c)

            tmp_ax.set_xscale('log' if log_cb.isChecked() else 'linear')
            tmp_ax.set_xlabel('Frequency (Hz)', color=fg)
            tmp_ax.set_ylabel('Sensitivity (dB re 1V/μPa @1m)', color=fg)
            tmp_ax.tick_params(colors=fg)
            tmp_ax.grid(True, linestyle='--', alpha=0.5, color=('gray' if mode.startswith('Dark') else 'black'))
            tmp_fig.savefig(fn, facecolor=tmp_fig.get_facecolor(), bbox_inches='tight')

        # --- signal connections ---
        curve_cb.currentTextChanged.connect(load_curve)
        log_cb.stateChanged.connect(lambda _: update_timer.start())
        markers_cb.stateChanged.connect(lambda _: update_timer.start())
        btn_overlay.clicked.connect(choose_overlays)
        btn_clear_ov.clicked.connect(lambda: (overlays.clear(), overlay_cache.clear(), update_timer.start()))
        btn_label.clicked.connect(label_intervals)
        btn_clear_int.clicked.connect(lambda: (interval_texts.clear(), interval_dots.clear(), update_timer.start()))
        table.itemChanged.connect(lambda _: update_timer.start())

        def on_table_cell_clicked(row, column):
            if column == 2 and row >= 0:
                table.removeRow(row)
                update_timer.start()

        table.cellClicked.connect(on_table_cell_clicked)

        # --- bottom buttons ---
        bot = QtWidgets.QHBoxLayout()
        mode_cb = QtWidgets.QComboBox()
        mode_cb.addItems(["Dark Mode", "Light Mode (B/W)"])
        bot.addWidget(mode_cb)

        btn_save = QtWidgets.QPushButton("Save Graph as JPG")
        btn_save.setStyleSheet("background-color:#3E6C8A;color:white;padding:6px;border-radius:4px;")
        btn_save.clicked.connect(save_graph_png)
        bot.addWidget(btn_save)

        btn_export = QtWidgets.QPushButton("Export CSV")
        btn_export.setStyleSheet("background-color:#3E6C8A;color:white;padding:6px;border-radius:4px;")
        btn_export.clicked.connect(export_csv)
        bot.addWidget(btn_export)

        btn_commit = QtWidgets.QPushButton("Save Changes")
        btn_commit.setStyleSheet("background-color:#3E6C8A;color:white;padding:6px;border-radius:4px;")
        btn_commit.clicked.connect(commit_changes)
        bot.addWidget(btn_commit)

        btn_del = QtWidgets.QPushButton("Delete Curve")
        btn_del.setStyleSheet("background-color:#3E6C8A;color:white;padding:6px;border-radius:4px;")
        btn_del.clicked.connect(delete_db_curve)
        bot.addWidget(btn_del)

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.setStyleSheet("background-color:#3E6C8A;color:white;padding:6px;border-radius:4px;")
        btn_close.clicked.connect(dlg.close)
        bot.addWidget(btn_close)

        bot.addStretch()
        layout.addLayout(bot)

        # --- initial load & exec ---
        if names:
            load_curve(names[0])
        dlg.exec_()
        conn.close()



    def correct_measurement_entries_popup(self):
        """
        Let the user supply a corrected max_voltage for one file/method,
        then preview how every measured_voltage will scale, and commit.
        """
        from PyQt5 import QtWidgets, QtCore
        import sqlite3

        # 1) Ask for file, method, new max voltage
        dlg1 = QtWidgets.QDialog(self)
        dlg1.setWindowTitle("Correct Max Voltage")
        form = QtWidgets.QFormLayout(dlg1)

        # File selector
        file_cb = QtWidgets.QComboBox()
        conn = sqlite3.connect(DB_FILENAME)
        cur  = conn.cursor()
        cur.execute("SELECT DISTINCT file_name FROM measurements ORDER BY file_name")
        files = [r[0] for r in cur.fetchall()]
        file_cb.addItems(files)
        form.addRow("File:", file_cb)

        # Method selector
        method_cb = QtWidgets.QComboBox()
        form.addRow("Method:", method_cb)
        def update_methods(idx):
            method_cb.clear()
            fname = files[idx]
            cur.execute(
                "SELECT DISTINCT method FROM measurements WHERE file_name=?", (fname,)
            )
            method_cb.addItems([r[0] for r in cur.fetchall()])
        file_cb.currentIndexChanged.connect(update_methods)
        if files:
            update_methods(0)

        # New max voltage entry
        new_max_edit = QtWidgets.QLineEdit()
        new_max_edit.setPlaceholderText("e.g. 1.23")
        form.addRow("Corrected max_voltage:", new_max_edit)

        # OK / Cancel
        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal, dlg1
        )
        form.addRow(bb)
        bb.accepted.connect(dlg1.accept)
        bb.rejected.connect(dlg1.reject)

        if dlg1.exec_() != QtWidgets.QDialog.Accepted:
            conn.close()
            return

        # grab inputs
        sel_file   = file_cb.currentText()
        sel_method = method_cb.currentText()
        try:
            new_max = float(new_max_edit.text())
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid", "Please enter a numeric max_voltage.")
            conn.close()
            return

        # 2) Load all rows and compute preview
        cur.execute(
            "SELECT id, target_frequency, measured_voltage, max_voltage "
            "FROM measurements WHERE file_name=? AND method=?",
            (sel_file, sel_method)
        )
        rows = cur.fetchall()
        if not rows:
            QtWidgets.QMessageBox.information(self, "No Data",
                                              f"No entries for {sel_file} / {sel_method}.")
            conn.close()
            return

        # assume all max_voltage are same; take first
        old_max = rows[0][3]
        factor  = new_max / old_max if old_max != 0 else 1.0

        preview = []
        for row in rows:
            _id, freq, old_v, _m = row
            new_v = old_v * factor
            preview.append((row[0], freq, old_v, new_v))

        # 3) Show preview table
        dlg2 = QtWidgets.QDialog(self)
        dlg2.setWindowTitle("Preview Corrections")
        vbox = QtWidgets.QVBoxLayout(dlg2)

        table = QtWidgets.QTableWidget(len(preview), 3)
        table.setHorizontalHeaderLabels([
            "Frequency (Hz)",
            "Old measured_voltage",
            "New measured_voltage"
        ])
        for i, (_id, freq, old_v, new_v) in enumerate(preview):
    # freq: show two decimals, voltages: three decimals
            items = [
                QtWidgets.QTableWidgetItem(f"{freq:.2f}"),
                QtWidgets.QTableWidgetItem(f"{old_v:.3f}"),
                QtWidgets.QTableWidgetItem(f"{new_v:.3f}")
            ]
            for j, it in enumerate(items):
                it.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                table.setItem(i, j, it)
        table.resizeColumnsToContents()
        vbox.addWidget(table)

        # Accept or Cancel
        bb2 = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        bb2.accepted.connect(dlg2.accept)
        bb2.rejected.connect(dlg2.reject)
        vbox.addWidget(bb2)

        if dlg2.exec_() != QtWidgets.QDialog.Accepted:
            conn.close()
            return

        # 4) Commit updates
        for _id, _freq, _old, new_v in preview:
            cur.execute(
                "UPDATE measurements SET measured_voltage=?, max_voltage=? WHERE id=?",
                (new_v, new_max, _id)
            )
        conn.commit()
        conn.close()

        QtWidgets.QMessageBox.information(
            self, "Done",
            f"Updated {len(preview)} rows for {sel_file}/{sel_method}.\n"
            f"max_voltage set to {new_max:.4g}."
        )


    # ---------------------
    # End of MainWindow
    # ---------------------

# ---------------------
# Main Execution
# ---------------------

if __name__ == "__main__":

    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    # 1) Load splash image
    pixmap = QPixmap("splash.png")            # or path to splash PNG
    pixmap = pixmap.scaled(800, 600, QtCore.Qt.KeepAspectRatioByExpanding,
                           QtCore.Qt.SmoothTransformation)
    splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
    splash.setMask(pixmap.mask())
    splash.setFixedSize(800, 600)
    splash.show()
    app.processEvents()                                 

    QTimer.singleShot(3000, splash.close)               # hide after 3.0 s

    # 3) Create and show main window
    window = MainWindow()
    window.startup_license_check()
    QTimer.singleShot(1500, window.showMaximized)            # show it _after_ the splash

    sys.exit(app.exec_())

