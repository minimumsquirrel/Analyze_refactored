#!/usr/bin/env python3
"""
Measurement Tools — methods for MainWindow mixin
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

class MeasurementToolsMixin:
    """Mixin class providing all Measurement Tools for self."""

    def depth_sounder_popup(self):
        """
        Prompts for mode (single vs. batch), ping frequency, speed of sound,
        then either measures one depth or scans and logs all depths.
        """
        from PyQt5 import QtWidgets
        import numpy as np
        from scipy.signal import butter, sosfiltfilt
        from scipy.signal import hilbert, find_peaks

        if self.full_data is None:
            QtWidgets.QMessageBox.critical(self, "Error", "No file loaded.")
            return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Depth Sounder")
        layout = QtWidgets.QVBoxLayout(dlg)

        # 1) Mode
        mode_grp = QtWidgets.QGroupBox("Mode")
        mlay = QtWidgets.QHBoxLayout(mode_grp)
        single_rb = QtWidgets.QRadioButton("Single Ping")
        batch_rb = QtWidgets.QRadioButton("Scan Entire File")
        single_rb.setChecked(True)
        for w in (single_rb, batch_rb):
            mlay.addWidget(w)
        layout.addWidget(mode_grp)

        # 2) Params
        form = QtWidgets.QFormLayout()
        freq_edit = QtWidgets.QLineEdit("172000")
        speed_edit = QtWidgets.QLineEdit("1480")
        depth_edit = QtWidgets.QLineEdit("")
        for w in (freq_edit, speed_edit, depth_edit):
            w.setFixedWidth(80)
        form.addRow("Ping Centre Frequency (Hz):", freq_edit)
        form.addRow("Speed of Sound (m/s):", speed_edit)
        form.addRow("Hydrophone Depth (m):", depth_edit)
        layout.addLayout(form)

        # OK/Cancel
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        # parse
        try:
            center_hz = float(freq_edit.text())
            sound_c = float(speed_edit.text())
            hydro_depth = float(depth_edit.text() or 0.0)
        except ValueError:
            QtWidgets.QMessageBox.critical(self, "Error", "Invalid numeric values.")
            return

        # select data segment
        if single_rb.isChecked():
            if not getattr(self, "fft_mode", False) or self.last_region is None:
                QtWidgets.QMessageBox.information(
                    self,
                    "Single Ping",
                    "Switch to FFT mode and select a region first.",
                )
                return
            xmin, xmax = self.last_region
            s0 = int(xmin * self.sample_rate)
            s1 = int(xmax * self.sample_rate)
            data_to_filter = self.full_data[s0:s1]
            filter_offset = s0
        else:
            data_to_filter = self.full_data
            filter_offset = 0

        # design bandpass ±5 kHz
        bw = 5000.0
        nyq = self.sample_rate / 2.0
        low_norm = (center_hz - bw) / nyq
        high_norm = (center_hz + bw) / nyq

        # clamp into (0,1) and ensure low < high
        low_norm = max(low_norm, 1e-6)
        high_norm = min(high_norm, 1.0 - 1e-6)
        if low_norm >= high_norm:
            QtWidgets.QMessageBox.critical(
                self,
                "Filter Error",
                "Invalid bandpass settings: low ≥ high.\nAdjust ping centre or bandwidth.",
            )
            return

        sos = butter(8, [low_norm, high_norm], btype="band", output="sos")
        try:
            filt = sosfiltfilt(sos, data_to_filter)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Filter Error", str(e))
            return

        # helper to detect ping+echo and compute depth
        def _detect_depth(seg):
            env = np.abs(hilbert(seg))
            peaks, props = find_peaks(env, height=env.max() * 0.3)
            if len(peaks) < 2:
                return None
            t0_rel = peaks[0] / self.sample_rate
            echoes = [p for p in peaks[1:] if (p / self.sample_rate - t0_rel) > 0.05]
            if not echoes:
                return None
            t1_rel = echoes[0] / self.sample_rate
            raw_d = (t1_rel - t0_rel) * sound_c / 2.0
            return t0_rel, t1_rel, raw_d

        # single-ping case
        if single_rb.isChecked():
            out = _detect_depth(filt)
            if not out:
                QtWidgets.QMessageBox.information(self, "No Echo", "Could not find ping+echo.")
                return
            t0_rel, t1_rel, depth_raw = out
            t0 = filter_offset / self.sample_rate + t0_rel
            t1 = filter_offset / self.sample_rate + t1_rel
            depth_adj = depth_raw + hydro_depth

            QtWidgets.QMessageBox.information(
                self,
                "Depth Result",
                f"Ping @ {t0:.3f}s, Echo @ {t1:.3f}s\n"
                f"Raw depth: {depth_raw:.2f} m\n"
                f"+ hydrophone offset: {depth_adj:.2f} m",
            )

            log_measurement(
                self.file_name,
                "Depth Sounder",
                center_hz,
                t0,
                t1,
                (t1 - t0),
                0.0,
                0.0,
                0.0,
                False,
                "",
                misc=str(depth_adj),
            )

        # batch scanning
        else:
            win = int(self.sample_rate * 2.0)
            hop = int(self.sample_rate * 1.0)
            results = []
            total = len(data_to_filter)
            for start in range(0, total - win, hop):
                seg = filt[start:start + win]
                out = _detect_depth(seg)
                if out:
                    t0_rel, t1_rel, depth_raw = out
                    t0 = (filter_offset + start) / self.sample_rate + t0_rel
                    t1 = (filter_offset + start) / self.sample_rate + t1_rel
                    results.append((t0, t1, depth_raw + hydro_depth))

            if not results:
                QtWidgets.QMessageBox.information(self, "No Depths", "No valid echoes found.")
                return

            # show batch results
            dlg2 = QtWidgets.QDialog(self)
            dlg2.setWindowTitle("Batch Depth Results")
            v2 = QtWidgets.QVBoxLayout(dlg2)
            lines = "\n".join(f"{t0:.2f}s→{t1:.2f}s: {d:.2f} m" for t0, t1, d in results)
            txt = QtWidgets.QPlainTextEdit(f"Found {len(results)} depths:\n\n{lines}")
            txt.setReadOnly(True)
            v2.addWidget(txt)

            btns2 = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
            )
            btns2.accepted.connect(dlg2.accept)
            btns2.rejected.connect(dlg2.reject)
            v2.addWidget(btns2)

            if dlg2.exec_() == QtWidgets.QDialog.Accepted:
                for t0, t1, d in results:
                    log_measurement(
                        self.file_name,
                        "Depth Sounder",
                        center_hz,
                        t0,
                        t1,
                        (t1 - t0),
                        0.0,
                        0.0,
                        0.0,
                        False,
                        "",
                        misc=str(d),
                    )
                QtWidgets.QMessageBox.information(
                    self,
                    "Logged",
                    f"Logged {len(results)} entr{'y' if len(results) == 1 else 'ies'}.",
                )

    def find_peaks_analysis(self):
        if self.full_data is None:
            QtWidgets.QMessageBox.critical(self, "Error", "Load a WAV file first.")
            return

        # ── 1) Parameter Dialog ────────────────────────────────────────────────
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Find & Measure Peaks")
        layout = QtWidgets.QVBoxLayout(dlg)

        form = QtWidgets.QFormLayout()
        height_edit = QtWidgets.QLineEdit("0.0")
        dist_edit   = QtWidgets.QLineEdit("0.01")  # secs
        win_edit    = QtWidgets.QLineEdit("0.1")   # secs
        for w in (height_edit, dist_edit, win_edit):
            w.setFixedWidth(80)
        form.addRow("Min Peak Height:",    height_edit)
        form.addRow("Min Distance (s):",  dist_edit)
        form.addRow("Window Length (s):", win_edit)
        layout.addLayout(form)

        btns = QtWidgets.QHBoxLayout()
        find_btn   = QtWidgets.QPushButton("Find Peaks")
        cancel_btn = QtWidgets.QPushButton("Cancel")
        btns.addStretch(); btns.addWidget(find_btn); btns.addWidget(cancel_btn)
        layout.addLayout(btns)

        params = {}
        def on_find():
            try:
                params['h'] = float(height_edit.text())
                params['d'] = float(dist_edit.text())
                params['w'] = float(win_edit.text())
            except ValueError:
                QtWidgets.QMessageBox.critical(self, "Error", "Invalid numbers.")
                return
            dlg.accept()

        find_btn.clicked.connect(on_find)
        cancel_btn.clicked.connect(dlg.reject)

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        # ── 2) Segment & Measure ──────────────────────────────────────────────
        data = self.full_data
        sr   = self.sample_rate
        h    = params['h']
        d_s  = params['d']
        w_s  = params['w']

        # continuous‐above-threshold mask
        mask = np.abs(data) >= h
        change = np.diff(mask.astype(int))
        starts = np.where(change == 1)[0] + 1
        ends   = np.where(change == -1)[0] + 1
        if mask[0]:         starts = np.insert(starts, 0, 0)
        if mask[-1]:        ends   = np.append(ends, len(mask))

        half_win = int(w_s * sr / 2)
        min_dist = int(d_s * sr)

        results = []
        for st, en in zip(starts, ends):
            dur = (en - st) / sr
            if dur >= w_s:
                # one event per long tone
                idxs = [st + (en-st)//2]
            else:
                # find multiple peaks in this short region
                sub = data[st:en]
                pks, _ = find_peaks(sub, height=h, distance=min_dist)
                idxs = [st + p for p in pks]

            for idx in idxs:
                # define measurement window
                s0 = max(0, idx - half_win)
                e0 = min(len(data), idx + half_win)
                seg = data[s0:e0]

                # convert to voltage
                if np.issubdtype(self.original_dtype, np.integer):
                    vmax = float(self.max_voltage_entry.text())
                    conv = vmax / np.iinfo(self.original_dtype).max
                else:
                    conv = 1.0

                vrms = np.sqrt(np.mean((seg * conv)**2))

                # dominant frequency
                nfft = 1 << int(np.ceil(np.log2(len(seg))))
                fftr = np.fft.rfft(seg * np.hanning(len(seg)), n=nfft)
                freqs= np.fft.rfftfreq(nfft, 1/sr)
                domf = self.refine_frequency(fftr, freqs)

                t_peak = idx / sr
                results.append((t_peak, vrms, domf, s0/sr, e0/sr))

        if not results:
            QtWidgets.QMessageBox.information(self, "No Peaks", "No events found.")
            return

        # ── 3) Summary & Confirm ───────────────────────────────────────────────
        text = "Time (s)    VRMS (V)    DomFreq (Hz)\n"
        text += "\n".join(f"{t:8.4f}    {v:8.4f}    {f:8.2f}" for t,v,f,_,_ in results)

        dlg2 = QtWidgets.QDialog(self)
        dlg2.setWindowTitle("Peak Measurements")
        v2 = QtWidgets.QVBoxLayout(dlg2)
        txt = QtWidgets.QPlainTextEdit(text)
        txt.setReadOnly(True)
        v2.addWidget(txt)

        h2 = QtWidgets.QHBoxLayout()
        ok = QtWidgets.QPushButton("Accept")
        no = QtWidgets.QPushButton("Cancel")
        h2.addStretch(); h2.addWidget(ok); h2.addWidget(no)
        v2.addLayout(h2)

        ok.clicked.connect(dlg2.accept)
        no.clicked.connect(dlg2.reject)

        if dlg2.exec_() != QtWidgets.QDialog.Accepted:
            return

        # ── 4) Store in DB ────────────────────────────────────────────────────
        for t, vr, df, st, et in results:
            self.log_measurement_with_project(
                self.file_name,
                "Find Peaks",
                df,
                st, et,
                (et - st),
                float(self.max_voltage_entry.text()),
                0.0,
                vr,
                False,
                ""
            )
        QtWidgets.QMessageBox.information(
            self, "Stored",
            f"{len(results)} measurements saved."
        )



    def duty_cycle_analysis_popup(self):
        """
        Duty Cycle Analysis (multi-channel, project-aware).

        • Splits your loaded WAV into windows at a given interval & duration
        • For each selected channel:
            - Counts samples above a given threshold in each window
            - Computes high/low ratio and duty% = (ratio / (1+ratio)) * 100
        • Plots duty% vs time in dark mode, all selected channels overlaid
        • Shows a table with time and per-channel duty%
        • "Store All" logs each window’s duty% for every channel
        • "Store Average" logs a per-channel average duty%
        • File names use _chN suffix for multi-channel files
        """
        import numpy as np
        import math
        import os
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        import matplotlib.pyplot as plt

        # 1) Ensure WAV loaded
        if getattr(self, "full_data", None) is None:
            QtWidgets.QMessageBox.critical(
                self, "Duty Cycle Analysis", "Load a WAV file first."
            )
            return
        if not hasattr(self, "sample_rate") or self.sample_rate <= 0:
            QtWidgets.QMessageBox.critical(
                self, "Duty Cycle Analysis", "Invalid sample rate."
            )
            return

        # 2) Params dialog
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Duty Cycle Analysis Settings")
        form = QtWidgets.QFormLayout(dlg)
        interval_le = QtWidgets.QLineEdit("1.0")
        window_le   = QtWidgets.QLineEdit("0.5")
        threshold_le= QtWidgets.QLineEdit("0.1")
        for w in (interval_le, window_le, threshold_le):
            w.setFixedWidth(80)
        form.addRow("Sampling interval (s):", interval_le)
        form.addRow("Window duration  (s):", window_le)
        form.addRow("Amplitude threshold:", threshold_le)
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        form.addRow(btns)
        dlg.setLayout(form)

        def on_accept():
            dlg.accept()

        def on_reject():
            dlg.reject()

        btns.accepted.connect(on_accept)
        btns.rejected.connect(on_reject)

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        try:
            interval = float(interval_le.text())
            duration = float(window_le.text())
            thresh   = float(threshold_le.text())
        except ValueError:
            QtWidgets.QMessageBox.critical(self, "Error", "Invalid numeric values.")
            return
        if interval <= 0 or duration <= 0:
            QtWidgets.QMessageBox.critical(
                self, "Error", "Interval and duration must be positive."
            )
            return

        # 3) Prepare data as (N, C), scaled to float
        data = np.asarray(self.full_data)
        sr   = float(self.sample_rate)

        if data.ndim == 1:
            data_2d = data.reshape(-1, 1)
        else:
            data_2d = data
        n_samples, n_ch = data_2d.shape

        if np.issubdtype(self.original_dtype, np.integer):
            max_val = float(np.iinfo(self.original_dtype).max)
            data_2d = data_2d.astype(np.float64) / max_val
        else:
            data_2d = data_2d.astype(np.float64)

        # 4) Determine selected channels
        if hasattr(self, "selected_channel_indices") and callable(self.selected_channel_indices):
            sel_channels = self.selected_channel_indices()
        else:
            sel_channels = list(range(n_ch))
        sel_channels = [ch for ch in sel_channels if 0 <= ch < n_ch]
        if not sel_channels:
            sel_channels = [0]

        # 5) Compute windowed duty% per channel
        times = []  # window start times
        duty_by_ch  = {ch: [] for ch in sel_channels}
        ratio_by_ch = {ch: [] for ch in sel_channels}

        total_len = n_samples / sr
        t = 0.0
        while t + duration <= total_len:
            i0 = int(round(t * sr))
            i1 = int(round((t + duration) * sr))
            if i1 <= i0:
                break

            times.append(t)
            for ch in sel_channels:
                seg = np.abs(data_2d[i0:i1, ch])
                above = int(np.count_nonzero(seg >= thresh))
                total = int(seg.size)
                below = total - above
                if total <= 0:
                    ratio = float("nan")
                    duty  = float("nan")
                else:
                    ratio = above / below if below > 0 else float("nan")
                    duty  = (ratio / (1.0 + ratio)) * 100.0 if not math.isnan(ratio) else float("nan")
                ratio_by_ch[ch].append(ratio)
                duty_by_ch[ch].append(duty)

            t += interval

        if not times:
            QtWidgets.QMessageBox.information(
                self, "Duty Cycle Analysis", "No windows fit the parameters."
            )
            return

        # 6) Build results dialog: plot + table
        result_dlg = QtWidgets.QDialog(self)
        result_dlg.setWindowTitle("Duty Cycle Results")
        vlay = QtWidgets.QVBoxLayout(result_dlg)

        # Dark-mode figure with all selected channels
        fig, ax = plt.subplots(facecolor="#19232D")
        ax.set_facecolor("#000000")

        # Use your color palette for channels
        color_keys  = list(self.color_options.keys())
        color_vals  = [self.color_options[k] for k in color_keys]
        num_colors  = len(color_vals) if color_vals else 1

        for idx, ch in enumerate(sel_channels):
            col = color_vals[idx % num_colors] if color_vals else self.graph_color
            ax.plot(
                times,
                duty_by_ch[ch],
                lw=1.5,
                label=f"Ch {ch+1}",
                color=col,
            )

        ax.set_title("Duty Cycle vs Time", color="white")
        ax.set_xlabel("Window Start Time (s)", color="white")
        ax.set_ylabel("Duty Cycle (%)", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")
        if len(sel_channels) > 1:
            leg = ax.legend()
            for txt in leg.get_texts():
                txt.set_color("white")

        canvas = FigureCanvas(fig)
        vlay.addWidget(canvas)

        # Table: Time + per-channel duty%
        table = QtWidgets.QTableWidget(len(times), 1 + len(sel_channels))
        headers = ["Time (s)"] + [f"Ch {ch+1} Duty (%)" for ch in sel_channels]
        table.setHorizontalHeaderLabels(headers)
        for row_idx, t0 in enumerate(times):
            table.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(f"{t0:.3f}"))
            for col_idx, ch in enumerate(sel_channels, start=1):
                vals = duty_by_ch.get(ch, [])
                if row_idx < len(vals) and not math.isnan(vals[row_idx]):
                    txt = f"{vals[row_idx]:.1f}"
                else:
                    txt = "NaN"
                table.setItem(row_idx, col_idx, QtWidgets.QTableWidgetItem(txt))
        table.resizeColumnsToContents()
        vlay.addWidget(table)

        # Average label (per channel)
        avg_lines = []
        for ch in sel_channels:
            arr = np.array(duty_by_ch[ch], dtype=float)
            avg_ch = float(np.nanmean(arr)) if arr.size else float("nan")
            if math.isnan(avg_ch):
                avg_lines.append(f"Ch {ch+1}: NaN")
            else:
                avg_lines.append(f"Ch {ch+1}: {avg_ch:.2f}%")
        avg_lbl = QtWidgets.QLabel("Average Duty Cycle:\n" + "\n".join(avg_lines))
        avg_lbl.setStyleSheet("font-weight:bold; color:white;")
        vlay.addWidget(avg_lbl)

        # Buttons: Save Graph, Store All, Store Avg, Close
        screenshot_path = ""
        hbtn = QtWidgets.QHBoxLayout()
        save_plot = QtWidgets.QPushButton("Save Graph…")
        store_all = QtWidgets.QPushButton("Store All")
        store_avg = QtWidgets.QPushButton("Store Average")
        close_btn = QtWidgets.QPushButton("Close")
        for btn in (save_plot, store_all, store_avg, close_btn):
            hbtn.addWidget(btn)
        vlay.addLayout(hbtn)

        # Save Graph handler
        def on_save():
            nonlocal screenshot_path
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                result_dlg,
                "Save Graph as JPG",
                "",
                "JPEG Files (*.jpg *.jpeg)",
            )
            if path:
                fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
                screenshot_path = path
                QtWidgets.QMessageBox.information(
                    result_dlg, "Saved", f"Graph saved to:\n{path}"
                )

        save_plot.clicked.connect(on_save)

        # Store All handler: one DB row per channel per window
        def on_store_all():
            base, ext = os.path.splitext(self.file_name or "")
            total_rows = 0
            for row_idx, t0 in enumerate(times):
                t1 = t0 + duration
                for ch in sel_channels:
                    vals = duty_by_ch[ch]
                    ratios = ratio_by_ch[ch]
                    if row_idx >= len(vals):
                        continue
                    duty_val = vals[row_idx]
                    ratio_val = ratios[row_idx] if row_idx < len(ratios) else 0.0

                    if base:
                        fname = f"{base}_ch{ch+1}{ext}"
                    else:
                        fname = f"ch{ch+1}"

                    self.log_measurement_with_project(
                        fname,
                        "Duty Cycle",
                        0.0,              # target_frequency
                        float(t0),        # start_time
                        float(t1),        # end_time
                        float(duration),  # window_length
                        0.0,              # max_voltage
                        0.0,              # bandwidth
                        float(duty_val),  # measured_voltage = duty%
                        False,            # filter_applied
                        screenshot_path,
                        misc=float(ratio_val),  # store high/low ratio in misc
                    )
                    total_rows += 1

            QtWidgets.QMessageBox.information(
                result_dlg,
                "Stored",
                f"Stored {total_rows} duty-cycle windows across {len(sel_channels)} channel(s).",
            )
            result_dlg.accept()

        store_all.clicked.connect(on_store_all)

        # Store Average handler: one row per channel with average duty%
        def on_store_avg():
            base, ext = os.path.splitext(self.file_name or "")
            total_len_s = n_samples / sr
            stored = 0

            for ch in sel_channels:
                arr = np.array(duty_by_ch[ch], dtype=float)
                if not arr.size:
                    continue
                avg_ch = float(np.nanmean(arr))
                if base:
                    fname = f"{base}_ch{ch+1}{ext}"
                else:
                    fname = f"ch{ch+1}"

                self.log_measurement_with_project(
                    fname,
                    "Duty Cycle (Avg)",
                    0.0,             # target_frequency
                    0.0,             # start_time
                    float(total_len_s),
                    float(total_len_s),
                    0.0,             # max_voltage
                    0.0,             # bandwidth
                    avg_ch,          # measured_voltage = avg duty%
                    False,
                    screenshot_path,
                    misc=0.0,
                )
                stored += 1

            QtWidgets.QMessageBox.information(
                result_dlg,
                "Stored",
                f"Stored average duty for {stored} channel(s).",
            )
            result_dlg.accept()

        store_avg.clicked.connect(on_store_avg)
        close_btn.clicked.connect(result_dlg.reject)

        result_dlg.exec_()
        plt.close(fig)



    # ──────────────────────────────────────────────────────────────────────────────
    # 1) Short-Time RMS / Leq
    # ──────────────────────────────────────────────────────────────────────────────

        # ──────────────────────────────────────────────────────────────────────────────
    # 1) Short-Time RMS / Leq (multi-channel, with channel checkboxes)
    # ──────────────────────────────────────────────────────────────────────────────


    def short_time_rms_popup(self):
        """
        Short-Time RMS / Leq Analysis (multi-channel):
        - Converts raw audio to volts using max_voltage_entry
        - Computes RMS over sliding windows (window length, hop size)
        - Lets you select which channels to include (checkboxes)
        - Plots all selected channels on one graph (with legend)
        - Allows saving the plot as JPEG
        - Logs each window’s RMS per channel, with filename suffix _chN
        - Logs per-channel average RMS entries covering the full span
        """
        import numpy as np
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        import matplotlib.pyplot as plt

        # 1) Ensure WAV loaded
        if self.full_data is None:
            QtWidgets.QMessageBox.critical(self, "Error", "Load a WAV file first.")
            return
        if self.sample_rate <= 0:
            QtWidgets.QMessageBox.critical(self, "Error", "Sample rate is invalid.")
            return

        # Prepare data shape info early (for channel count)
        data = np.asarray(self.full_data)
        if data.ndim == 1:
            n_samples = data.shape[0]
            n_channels = 1
        else:
            n_samples, n_channels = data.shape

        # 2) Parameter dialog for window & hop sizes + channel checkboxes
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Short-Time RMS / Leq Settings")
        layout = QtWidgets.QVBoxLayout(dlg)

        # Window / hop form
        form = QtWidgets.QFormLayout()
        win_edit = QtWidgets.QLineEdit("0.1")
        hop_edit = QtWidgets.QLineEdit("0.05")
        for w in (win_edit, hop_edit):
            w.setFixedWidth(80)
        form.addRow("Window Length (s):", win_edit)
        form.addRow("Hop Size      (s):", hop_edit)
        layout.addLayout(form)

        # Channel selection group
        ch_group = QtWidgets.QGroupBox("Channels")
        ch_layout = QtWidgets.QHBoxLayout(ch_group)
        ch_boxes = []

        # Try to respect current channel selection from the main window
        default_selected = set()
        try:
            if hasattr(self, "selected_channel_indices"):
                default_selected = set(self.selected_channel_indices())
        except Exception:
            default_selected = set()

        if not default_selected:
            # Fallback: select all channels by default
            default_selected = set(range(n_channels))

        for ch in range(n_channels):
            cb = QtWidgets.QCheckBox(f"Ch {ch+1}")
            cb.setChecked(ch in default_selected)
            ch_layout.addWidget(cb)
            ch_boxes.append(cb)

        layout.addWidget(ch_group)

        # Buttons (Compute / Cancel)
        btn_h = QtWidgets.QHBoxLayout()
        calc_btn = QtWidgets.QPushButton("Compute")
        cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_h.addStretch()
        btn_h.addWidget(calc_btn)
        btn_h.addWidget(cancel_btn)
        layout.addLayout(btn_h)

        params = {}

        def on_compute():
            # Parse window & hop
            try:
                w_s = float(win_edit.text())
                h_s = float(hop_edit.text())
            except ValueError:
                QtWidgets.QMessageBox.critical(self, "Error", "Invalid numeric values.")
                return

            if w_s <= 0 or h_s <= 0:
                QtWidgets.QMessageBox.critical(
                    self, "Error", "Window and hop must be positive."
                )
                return

            # Collect selected channels
            selected = [i for i, cb in enumerate(ch_boxes) if cb.isChecked()]
            if not selected:
                QtWidgets.QMessageBox.critical(
                    self, "Error", "Select at least one channel."
                )
                return

            params["w_s"] = w_s
            params["h_s"] = h_s
            params["channels"] = selected
            dlg.accept()

        calc_btn.clicked.connect(on_compute)
        cancel_btn.clicked.connect(dlg.reject)

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        w_s = params["w_s"]
        h_s = params["h_s"]
        sel_channels = params["channels"]
        sr = float(self.sample_rate)

        # 3) Convert data to volts (all channels)
        if np.issubdtype(self.original_dtype, np.integer):
            try:
                vmax = float(self.max_voltage_entry.text())
            except Exception:
                vmax = 1.0
            conv = vmax / np.iinfo(self.original_dtype).max
        else:
            conv = 1.0
            vmax = 0.0  # For logging if needed

        data_volts = data.astype(np.float64) * conv
        if data_volts.ndim == 1:
            data_volts = data_volts[:, None]  # shape (N, 1)

        n_samples, n_channels = data_volts.shape

        # 4) Compute short-time RMS and times for all selected channels
        w_n = max(1, int(round(w_s * sr)))
        h_n = max(1, int(round(h_s * sr)))

        if n_samples < w_n:
            QtWidgets.QMessageBox.information(
                self, "Short-Time RMS", "No windows fit the parameters."
            )
            return

        times = []
        rms_by_ch = {ch: [] for ch in sel_channels}

        for start in range(0, n_samples - w_n + 1, h_n):
            end = start + w_n
            seg_all = data_volts[start:end, :]  # (w_n, n_channels)
            t_mid = (start + w_n / 2.0) / sr
            times.append(t_mid)

            for ch in sel_channels:
                seg = seg_all[:, ch]
                rms = float(np.sqrt(np.mean(seg ** 2)))
                rms_by_ch[ch].append(rms)

        if not times or all(len(rms_by_ch[ch]) == 0 for ch in sel_channels):
            QtWidgets.QMessageBox.information(
                self, "Short-Time RMS", "No windows fit the parameters."
            )
            return

        # 5) Plot in dark mode (all selected channels)
        fig, ax = plt.subplots(facecolor="#19232D")
        ax.set_facecolor("#000000")

        # --- assign colours per channel using the app’s defined palette ---
        color_keys = list(self.color_options.keys())
        color_vals = [self.color_options[k] for k in color_keys]
        num_colors = len(color_vals)

        for idx, ch in enumerate(sel_channels):
            ch_label = f"Ch {ch+1}"
            col = color_vals[idx % num_colors]  # wrap colours if > 9 channels
            ax.plot(times, rms_by_ch[ch], lw=1.5, label=ch_label, color=col)


        ax.set_title(
            f"Short-Time RMS (win={w_s:.3f}s, hop={h_s:.3f}s)", color="white"
        )
        ax.set_xlabel("Time (s)", color="white")
        ax.set_ylabel("RMS Voltage (V)", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")

        if len(sel_channels) > 1:
            leg = ax.legend()
            for text in leg.get_texts():
                text.set_color("white")

        canvas = FigureCanvas(fig)

        # 6) Results dialog with Save, Store, Close
        win = QtWidgets.QDialog(self)
        win.setWindowTitle("Short-Time RMS Results")
        vlay = QtWidgets.QVBoxLayout(win)
        vlay.addWidget(canvas)

        btn_bar = QtWidgets.QHBoxLayout()
        btn_bar.addStretch()
        save_btn = QtWidgets.QPushButton("Save Graph…")
        store_all = QtWidgets.QPushButton("Store All Windows")
        store_avg = QtWidgets.QPushButton("Store Average")
        close_btn = QtWidgets.QPushButton("Close")
        for b in (save_btn, store_all, store_avg, close_btn):
            btn_bar.addWidget(b)
        vlay.addLayout(btn_bar)

        screenshot_path = ""

        def on_save_graph():
            nonlocal screenshot_path
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                win, "Save Graph as JPEG", "", "JPEG Files (*.jpg *.jpeg)"
            )
            if path:
                fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
                screenshot_path = path
                QtWidgets.QMessageBox.information(
                    win, "Saved", f"Graph saved to:\n{path}"
                )

        save_btn.clicked.connect(on_save_graph)

        # 7) Store all windows (per channel, per window, with _chN suffix)
        def on_store_all():
            try:
                vmax_local = float(self.max_voltage_entry.text())
            except Exception:
                vmax_local = vmax

            total_count = 0
            for idx, t_mid in enumerate(times):
                t0 = t_mid - w_s / 2.0
                t1 = t_mid + w_s / 2.0

                for ch in sel_channels:
                    # Safeguard index
                    if idx >= len(rms_by_ch[ch]):
                        continue
                    rms_val = rms_by_ch[ch][idx]

                    # File name with _chN for multichannel
                    if n_channels > 1:
                        fname = f"{self.file_name}_ch{ch+1}"
                    else:
                        fname = self.file_name

                    self.log_measurement_with_project(
                        fname,
                        "Short-Time RMS",
                        0.0,                # target_frequency
                        float(t0),          # start_time
                        float(t1),          # end_time
                        float(w_s),         # window_length
                        vmax_local,         # max_voltage
                        0.0,                # bandwidth
                        rms_val,            # measured_voltage
                        False,              # filter_applied
                        screenshot_path,
                        misc=0.0
                    )
                    total_count += 1

            QtWidgets.QMessageBox.information(
                win, "Stored", f"Stored {total_count} window(s) across channels."
            )
            win.accept()

        store_all.clicked.connect(on_store_all)

        # 8) Store per-channel averages
        def on_store_avg():
            try:
                vmax_local = float(self.max_voltage_entry.text())
            except Exception:
                vmax_local = vmax

            if not times:
                QtWidgets.QMessageBox.information(
                    win, "Short-Time RMS", "No data to average."
                )
                return

            t0 = times[0] - w_s / 2.0
            t1 = times[-1] + w_s / 2.0
            ch_count = 0

            for ch in sel_channels:
                if not rms_by_ch[ch]:
                    continue
                avg = float(np.mean(rms_by_ch[ch]))

                if n_channels > 1:
                    fname = f"{self.file_name}_ch{ch+1}"
                    method_name = "Short-Time RMS (Avg)_ch{}".format(ch+1)
                else:
                    fname = self.file_name
                    method_name = "Short-Time RMS (Avg)"

                self.log_measurement_with_project(
                    fname,
                    method_name,
                    0.0,                # target_frequency
                    float(t0),          # start_time
                    float(t1),          # end_time
                    float(w_s),         # window_length
                    vmax_local,
                    0.0,                # bandwidth
                    avg,                # measured_voltage
                    False,              # filter_applied
                    screenshot_path,
                    misc=0.0
                )
                ch_count += 1

            QtWidgets.QMessageBox.information(
                win, "Stored", f"Stored averages for {ch_count} channel(s)."
            )
            win.accept()

        store_avg.clicked.connect(on_store_avg)
        close_btn.clicked.connect(win.reject)

        win.exec_()
        plt.close(fig)



    # ──────────────────────────────────────────────────────────────────────────────
    # 3) Octave-Band Analysis
    # ──────────────────────────────────────────────────────────────────────────────
        # ──────────────────────────────────────────────────────────────────────────────
    # 3) Octave-Band Analysis
    # ──────────────────────────────────────────────────────────────────────────────


    def crest_factor_popup(self):
        """Compute crest factor (peak/RMS) for selected channels."""
        import numpy as np

        if self.full_data is None:
            QtWidgets.QMessageBox.critical(self, "Error", "Load a WAV file first.")
            return
        if not getattr(self, "sample_rate", 0):
            QtWidgets.QMessageBox.critical(self, "Error", "Sample rate is invalid.")
            return

        # Choose analysis scope
        use_region = bool(getattr(self, "fft_mode", False) and getattr(self, "last_region", None) is not None)
        scope = "Selected FFT Region" if use_region else "Entire File"

        data = np.asarray(self.full_data)
        if data.ndim == 1:
            data2d = data[:, None]
        else:
            data2d = data

        if use_region:
            xmin, xmax = self.last_region
            s0 = max(0, int(xmin * self.sample_rate))
            s1 = min(data2d.shape[0], int(xmax * self.sample_rate))
            if s1 <= s0:
                QtWidgets.QMessageBox.warning(self, "Crest Factor", "Selected FFT region is empty.")
                return
            seg = data2d[s0:s1, :]
            t0 = s0 / float(self.sample_rate)
            t1 = s1 / float(self.sample_rate)
        else:
            seg = data2d
            t0 = 0.0
            t1 = data2d.shape[0] / float(self.sample_rate)

        # Determine channels (prefer selected channels when available)
        try:
            selected = list(self.selected_channel_indices()) if hasattr(self, "selected_channel_indices") else []
        except Exception:
            selected = []
        if not selected:
            selected = list(range(seg.shape[1]))

        rows = []
        for ch in selected:
            if ch < 0 or ch >= seg.shape[1]:
                continue
            x = seg[:, ch].astype(float)
            if x.size == 0:
                continue
            peak = float(np.max(np.abs(x)))
            rms = float(np.sqrt(np.mean(np.square(x))))
            if rms <= 0:
                crest = float("nan")
                crest_db = float("nan")
            else:
                crest = peak / rms
                crest_db = 20.0 * np.log10(max(crest, 1e-12))
            rows.append((ch + 1, peak, rms, crest, crest_db))

        if not rows:
            QtWidgets.QMessageBox.information(self, "Crest Factor", "No valid channel data to analyze.")
            return

        text_lines = [f"Scope: {scope} ({t0:.3f}s - {t1:.3f}s)", ""]
        for ch, peak, rms, crest, crest_db in rows:
            text_lines.append(
                f"Ch {ch}: Peak={peak:.6g}, RMS={rms:.6g}, Crest={crest:.4f}, Crest(dB)={crest_db:.3f}"
            )

        QtWidgets.QMessageBox.information(self, "Crest Factor Results", "\n".join(text_lines))

        # Log per-channel crest factor as measured_voltage
        base_name = self.file_name or "(unknown)"
        for ch, peak, rms, crest, crest_db in rows:
            fname = base_name if seg.shape[1] == 1 else f"{base_name}_ch{ch}"
            try:
                log_measurement(
                    fname,
                    "Crest Factor",
                    0.0,
                    t0,
                    t1,
                    (t1 - t0),
                    peak,
                    0.0,
                    float(crest),
                    False,
                    "",
                    misc=f"crest_db={crest_db:.3f},rms={rms:.6g}",
                )
            except Exception:
                pass

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







    # ──────────────────────────────────────────────────────────────────────────────
    # 4) Signal-to-Noise Ratio (SNR) Estimator
    # ──────────────────────────────────────────────────────────────────────────────

    def snr_estimator_popup(self):
        """
        SNR Estimator with graphical region selection.

        - Shows a popup with a waveform plot (first selected channel).
        - User clicks "Select Signal Region" and drags on the plot.
        - User clicks "Select Noise Region" and drags on the plot.
        - Computes SNR for *each* selected channel using those time ranges.
        - Logs one row per channel in the measurements table with _chN in file_name.
        """
        if self.full_data is None:
            QtWidgets.QMessageBox.critical(self, "Error", "Load a WAV file first.")
            return

        # --- Prepare data as (N, C) ---
        data = np.asarray(self.full_data)
        sr = self.sample_rate
        if sr is None or sr <= 0:
            QtWidgets.QMessageBox.critical(self, "Error", "Invalid sample rate.")
            return

        if data.ndim == 1:
            data_2d = data.reshape(-1, 1)
        else:
            data_2d = data
        n_samples, n_ch = data_2d.shape

        # Selected channels
        if hasattr(self, "selected_channel_indices") and callable(self.selected_channel_indices):
            sel_channels = self.selected_channel_indices()
        else:
            sel_channels = list(range(n_ch))

        sel_channels = [ch for ch in sel_channels if 0 <= ch < n_ch]
        if not sel_channels:
            sel_channels = [0]

        display_ch = sel_channels[0]  # channel we show in the popup

        # --- Dialog + plot ---
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("SNR Estimator (Graph Selection)")
        dlg.resize(900, 600)
        layout = QtWidgets.QVBoxLayout(dlg)

        canvas = MplCanvas(dlg, width=8, height=3, dpi=100)
        layout.addWidget(canvas)

        t = np.arange(n_samples) / sr
        canvas.ax.plot(t, data_2d[:, display_ch], color=self.graph_color)
        canvas.ax.set_xlabel("Time (s)", color="white")
        canvas.ax.set_ylabel("Amplitude", color="white")
        canvas.ax.set_title(
            f"SNR selection (showing channel {display_ch+1})",
            color="white",
        )
        canvas.fig.tight_layout()
        canvas.draw()

        # Labels showing current selections
        sig_label = QtWidgets.QLabel("Signal region: not selected")
        noise_label = QtWidgets.QLabel("Noise region: not selected")
        sig_label.setStyleSheet("color: white;")
        noise_label.setStyleSheet("color: white;")
        layout.addWidget(sig_label)
        layout.addWidget(noise_label)

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        select_sig_btn = QtWidgets.QPushButton("Select Signal Region")
        select_noise_btn = QtWidgets.QPushButton("Select Noise Region")
        btn_row.addWidget(select_sig_btn)
        btn_row.addWidget(select_noise_btn)
        btn_row.addStretch()
        compute_btn = QtWidgets.QPushButton("Compute SNR")
        cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_row.addWidget(compute_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        # --- Span selection state ---
        sig_range = [None, None]    # [start_time, end_time]
        noise_range = [None, None]  # [start_time, end_time]
        span_selector = {"obj": None}
        current_mode = {"mode": None}   # "signal" or "noise"

        def on_span_selected(xmin, xmax):
            """Callback from SpanSelector."""
            if xmin == xmax:
                return
            s = float(min(xmin, xmax))
            e = float(max(xmin, xmax))
            mode = current_mode["mode"]

            if mode == "signal":
                sig_range[0], sig_range[1] = s, e
                sig_label.setText(f"Signal region: {s:.3f} – {e:.3f} s")
            elif mode == "noise":
                noise_range[0], noise_range[1] = s, e
                noise_label.setText(f"Noise region: {s:.3f} – {e:.3f} s")
            else:
                return

            # Disable selector after each selection; user can re-enable via button
            if span_selector["obj"] is not None:
                span_selector["obj"].disconnect_events()
                span_selector["obj"] = None
                current_mode["mode"] = None

        def start_select(mode: str):
            """Enable a new SpanSelector for the given mode."""
            current_mode["mode"] = mode
            # Remove any existing selector
            if span_selector["obj"] is not None:
                span_selector["obj"].disconnect_events()
                span_selector["obj"] = None

            span_selector["obj"] = SpanSelector(
                canvas.ax,
                on_span_selected,
                'horizontal',
                useblit=True,
                props=dict(alpha=0.3, facecolor="#7A9E9F"),  # soft teal overlay
            )

        select_sig_btn.clicked.connect(lambda: start_select("signal"))
        select_noise_btn.clicked.connect(lambda: start_select("noise"))

        def do_compute():
            # Ensure both regions selected
            if sig_range[0] is None or noise_range[0] is None:
                QtWidgets.QMessageBox.warning(
                    dlg,
                    "Missing selection",
                    "Please select BOTH a signal region and a noise region using the buttons above."
                )
                return

            s0, s1 = sig_range
            n0, n1 = noise_range

            if s1 <= s0 or n1 <= n0:
                QtWidgets.QMessageBox.critical(
                    dlg,
                    "Error",
                    "End times must be greater than start times."
                )
                return

            # Convert to indices
            sig_start_idx   = max(0, int(round(s0 * sr)))
            sig_end_idx     = min(n_samples, int(round(s1 * sr)))
            noise_start_idx = max(0, int(round(n0 * sr)))
            noise_end_idx   = min(n_samples, int(round(n1 * sr)))

            if sig_end_idx <= sig_start_idx or noise_end_idx <= noise_start_idx:
                QtWidgets.QMessageBox.critical(
                    dlg,
                    "Error",
                    "Selected regions are too short."
                )
                return

            # Compute SNR for each selected channel
            results = []  # (ch, rms_sig, rms_bg, snr_db)
            for ch in sel_channels:
                ch_data = data_2d[:, ch]
                sig_seg = ch_data[sig_start_idx:sig_end_idx]
                noise_seg = ch_data[noise_start_idx:noise_end_idx]

                if sig_seg.size == 0 or noise_seg.size == 0:
                    continue

                rms_sig = float(np.sqrt(np.mean(sig_seg ** 2)))
                rms_bg  = float(np.sqrt(np.mean(noise_seg ** 2)))
                snr_db  = float(20 * np.log10(rms_sig / rms_bg)) if rms_bg > 0 else float("inf")
                results.append((ch, rms_sig, rms_bg, snr_db))

            if not results:
                QtWidgets.QMessageBox.warning(
                    dlg,
                    "No Data",
                    "No valid signal/noise segments for the selected channels."
                )
                return

            # Show results summary
            lines = []
            for ch, rms_sig, rms_bg, snr_db in results:
                lines.append(
                    f"Channel {ch+1}:\n"
                    f"  Signal RMS ({s0:.3f}-{s1:.3f}s): {rms_sig:.4f}\n"
                    f"  Noise  RMS ({n0:.3f}-{n1:.3f}s): {rms_bg:.4f}\n"
                    f"  SNR = {snr_db:.2f} dB"
                )
            text = "\n\n".join(lines)

            box = QtWidgets.QMessageBox(dlg)
            box.setWindowTitle("SNR Result")
            box.setText(text)
            box.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)

            if box.exec_() == QtWidgets.QMessageBox.Ok:
                base, ext = os.path.splitext(self.file_name or "")
                for ch, _, _, snr_db in results:
                    if base:
                        fname = f"{base}_ch{ch+1}{ext}"
                    else:
                        fname = f"ch{ch+1}"

                    # Log one measurement per channel; SNR in misc and attach to the active project
                    self.log_measurement_with_project(
                        fname,
                        "SNR Estimator",
                        0.0,                 # target_frequency
                        float(s0),           # start_time
                        float(s1),           # end_time
                        float(s1 - s0),      # window_length
                        0.0,                 # max_voltage (not used)
                        0.0,                 # bandwidth
                        0.0,                 # measured_voltage (unused here)
                        False,               # filter_applied
                        "",                  # screenshot
                        misc=snr_db,         # SNR (dB)
                    )

                QtWidgets.QMessageBox.information(
                    dlg, "Stored", "SNR stored for all selected channels."
                )
                dlg.accept()

        compute_btn.clicked.connect(do_compute)
        cancel_btn.clicked.connect(dlg.reject)

        dlg.exec_()





    def lfm_pulse_analysis(self):
        """
        Sweep through the loaded WAV, detect an LFM pulse, and analyze it in fixed windows.
        Adds a “Pre-analyze” step that:
        • Finds the pulse start/end by threshold (in amplitude)
        • Estimates start & end frequencies
        • Computes the linear sweep rate
        • Suggests a measurement window ≈ 5% of the total pulse length
        """
        from PyQt5 import QtWidgets, QtCore
        import numpy as np
        from scipy.signal import butter, sosfiltfilt, get_window
        # TIME_MULTIPLIER is your slider→seconds factor
        # refine_frequency is the @staticmethod on MainWindow

        # ── 1) Settings + Pre-analyze dialog ────────────────────────────────
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("LFM Pulse Analysis Settings")
        form = QtWidgets.QFormLayout(dlg)

        interval_le = QtWidgets.QLineEdit(self.scroll_step_entry.text())
        interval_le.setFixedWidth(80)
        form.addRow("Sampling Interval (s):", interval_le)

        window_le = QtWidgets.QLineEdit(self.fft_length_entry.text())
        window_le.setFixedWidth(80)
        form.addRow("Window Length (s):", window_le)

        start_le = QtWidgets.QLineEdit(f"{self.fft_time_slider.value()/self.TIME_MULTIPLIER:.3f}")
        start_le.setFixedWidth(80)
        form.addRow("Start Time (s):", start_le)

        noise_cb = QtWidgets.QComboBox()
        noise_cb.addItems(["None", "50 Hz Notch", "60 Hz Notch"])
        form.addRow("Remove Mains Noise:", noise_cb)

        threshold_le = QtWidgets.QLineEdit(self.pulse_threshold_entry.text())
        threshold_le.setFixedWidth(80)
        form.addRow("Pulse Threshold (Amplitude):", threshold_le)

        drop_zero_cb = QtWidgets.QCheckBox("Discard 0 Hz results")
        drop_zero_cb.setChecked(True)
        form.addRow(drop_zero_cb)

        detect_lbl = QtWidgets.QLabel("")
        form.addRow("Pre-analyze Info:", detect_lbl)

        pre_btn = QtWidgets.QPushButton("Pre‑analyze Pulse")
        form.addRow(pre_btn)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        form.addRow(btns)

        def do_pre():
            fs  = self.sample_rate
            raw = self.full_data
            # Work in the raw amplitude domain (PCM counts for integer data)
            amps = raw.astype(np.float64)

            # 2) threshold detection
            try:
                thresh = float(threshold_le.text())
            except:
                QtWidgets.QMessageBox.warning(dlg, "Error", "Invalid pulse threshold.")
                return
            idxs = np.where(np.abs(amps) >= thresh)[0]
            if idxs.size == 0:
                QtWidgets.QMessageBox.information(dlg, "Pre‑analyze", "No pulse above threshold.")
                detect_lbl.setText("No pulse detected")
                return

            # 3) compute pulse bounds & length
            i0, i1 = idxs[0], idxs[-1]
            t0, t1 = i0/fs, i1/fs
            plen    = t1 - t0

            # 4) estimate start/end freq via short-FFT
            win_len = min(int(fs*0.01), i1-i0)
            w       = get_window("hann", win_len)
            def dom_freq_at(idx):
                seg = amps[idx:idx+win_len] * w
                nfft = 1 << int(np.ceil(np.log2(win_len)))
                FFT  = np.fft.rfft(seg, n=nfft)
                f    = np.fft.rfftfreq(nfft, 1/fs)
                return self.refine_frequency(FFT, f)
            start_f = dom_freq_at(i0)
            end_f   = dom_freq_at(i1-win_len)

            # 5) compute sweep rate & suggest window ≈ 5% of pulse
            sweep_rate = (end_f - start_f) / plen if plen>0 else 0.0
            suggestion = plen * 0.05

            # 6) populate fields and info label
            window_le.setText(f"{suggestion:.3f}")
            start_le.setText(f"{t0:.3f}")
            detect_lbl.setText(
                f"Pulse {t0:.3f}–{t1:.3f}s;  "
                f"Start {start_f:.1f} Hz → End {end_f:.1f} Hz;  "
                f"Rate {sweep_rate:.1f} Hz/s;  "
                f"Suggest win {suggestion:.3f}s"
            )

        pre_btn.clicked.connect(do_pre)

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        # ── 2) Read & validate settings ───────────────────────────────────────
        try:
            scroll_step   = float(interval_le.text())
            window_length = float(window_le.text())
            start_time    = float(start_le.text())
            threshold     = float(threshold_le.text())
        except ValueError:
            QtWidgets.QMessageBox.critical(self, "Error", "Invalid settings.")
            return
        if scroll_step <= 0 or window_length <= 0:
            QtWidgets.QMessageBox.critical(
                self,
                "Error",
                "Window length and sampling interval must be greater than zero.",
            )
            return
        notch_choice = noise_cb.currentText()
        drop_zero    = drop_zero_cb.isChecked()

        channels = self.selected_channel_indices() if hasattr(self, "selected_channel_indices") else [0]
        results_by_channel = []

        # Pre-compute an iteration budget to keep the UI responsive with progress feedback
        duration_s = float(self.full_time[-1]) if len(getattr(self, "full_time", [])) else 0.0
        total_windows = 0
        if scroll_step > 0:
            for ch in channels:
                span = max(0.0, duration_s - start_time - window_length)
                total_windows += int(span // scroll_step) + 1 if span >= 0 else 0

        progress = QtWidgets.QProgressDialog(
            "Analyzing LFM pulse across channels…",
            "Cancel",
            0,
            max(total_windows, 1),
            self,
        )
        progress.setWindowModality(QtCore.Qt.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress_step = 0
        cancelled = False

        # ── 3) Prepare data (with optional notch) per channel ────────────────
        fs   = self.sample_rate
        for ch in channels:
            data = self.get_channel_data(ch)
            if data is None:
                continue

            data = np.asarray(data, dtype=np.float64)
            if notch_choice != "None":
                f0   = 50.0 if "50" in notch_choice else 60.0
                bw   = 2.0
                low  = (f0 - bw/2) / (fs/2)
                high = (f0 + bw/2) / (fs/2)
                sos  = butter(2, [low, high], btype="bandstop", output="sos")
                data = safe_sosfiltfilt(sos, data)

            if np.issubdtype(self.original_dtype, np.integer):
                max_count = max(1, float(np.iinfo(self.original_dtype).max))
            else:
                max_count = 1.0

            if np.issubdtype(self.original_dtype, np.integer):
                try:
                    vmax = float(self.max_voltage_entry.text())
                except Exception:
                    vmax = 1.0
                conv = vmax / max_count
            else:
                conv = 1.0

            # ── 4) Slide through measurement windows ─────────────────────────
            window_samples = max(1, int(window_length * fs))
            nfft = 1 << int(np.ceil(np.log2(window_samples)))
            hann_window = get_window("hann", window_samples)
            freq_bins = np.fft.rfftfreq(nfft, 1/fs)

            # Precompute helpers to avoid redundant per-window work
            volts_data = data * conv
            sq_cumsum = np.concatenate(([0.0], np.cumsum(volts_data * volts_data)))
            abs_data = np.abs(data)
            results = []
            t = start_time
            while t + window_length <= self.full_time[-1]:
                if progress.wasCanceled():
                    cancelled = True
                    break
                i0 = int(t * fs)
                i1 = i0 + window_samples
                if i1 > len(data):
                    break

                # Apply the threshold directly in the amplitude domain (PCM counts
                # for integer data or raw float amplitude for floating-point WAVs).
                peak_amp = float(abs_data[i0:i1].max()) if i1 > i0 else 0.0
                if peak_amp < threshold:
                    t += scroll_step
                    progress_step += 1
                    progress.setValue(min(progress_step, progress.maximum()))
                    QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, 50)
                    continue

                seg_volts = volts_data[i0:i1]
                sum_sq = sq_cumsum[i1] - sq_cumsum[i0]
                rms = float(np.sqrt(sum_sq / window_samples)) if window_samples else 0.0
                # FFT → dominant frequency
                win_seg = seg_volts * hann_window
                FFT     = np.fft.rfft(win_seg, n=nfft)
                dom     = self.refine_frequency(FFT, freq_bins)

                if drop_zero and (not np.isfinite(dom) or abs(dom) < 1e-9):
                    t += scroll_step
                    progress_step += 1
                    progress.setValue(min(progress_step, progress.maximum()))
                    QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, 50)
                    continue

                results.append((t, rms, dom))
                t += scroll_step

                # Keep the UI responsive during long analyses
                progress_step += 1
                progress.setValue(min(progress_step, progress.maximum()))
                QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, 50)

            if results:
                results_by_channel.append((ch, results))

            if cancelled:
                break

        progress.setValue(progress.maximum())

        if cancelled:
            return

        if not results_by_channel:
            QtWidgets.QMessageBox.information(self, "LFM Analysis", "No pulses found above threshold.")
            return

        # ── 5) Show & optionally store ────────────────────────────────────────
        dlg2 = QtWidgets.QDialog(self)
        dlg2.setWindowTitle("LFM Pulse Analysis Results")
        vlay2 = QtWidgets.QVBoxLayout(dlg2)

        lines = []
        for ch, results in results_by_channel:
            lines.append(f"Channel {ch+1} ({self.channel_file_label(ch)}):")
            lines.append("Time (s)   RMS Voltage (V)   Dom Freq (Hz)")
            lines.extend(
                f"{t0:8.3f}    {r:8.4f}    {f:8.2f}" for t0, r, f in results
            )
            lines.append("")

        viewer = QtWidgets.QPlainTextEdit("\n".join(lines).rstrip())
        viewer.setReadOnly(True)
        vlay2.addWidget(viewer)

        btns2 = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        vlay2.addWidget(btns2)
        btns2.accepted.connect(lambda: self._store_lfm_results(results_by_channel, window_length))
        btns2.rejected.connect(dlg2.reject)

        dlg2.exec_()






    def interval_analysis_single(self):
        """
        Single‑file Interval Analysis:
        • Ask for interval, window & total period
        • Convert raw samples → volts exactly like LFM analysis
        • Compute VRMS, dominant frequency, and max_voltage (user’s entry) in each window
        • Show results and let you Accept/Discard to save
        """
        from PyQt5 import QtWidgets
        import numpy as np
        from scipy.signal import get_window, butter, sosfiltfilt

        # 0) Ensure data is loaded
        if self.full_data is None:
            QtWidgets.QMessageBox.critical(self, "Interval Analysis", "Load a WAV file first.")
            return

        # 1) Parameter dialog
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Interval Analysis Settings")
        form = QtWidgets.QFormLayout(dlg)

        interval_le = QtWidgets.QLineEdit("1.0");  interval_le.setFixedWidth(80)
        duration_le = QtWidgets.QLineEdit("0.5");  duration_le.setFixedWidth(80)
        total_le    = QtWidgets.QLineEdit(f"{self.full_time[-1]:.1f}"); total_le.setFixedWidth(80)

        form.addRow("Sampling Interval (s):", interval_le)
        form.addRow("Window Duration  (s):", duration_le)
        form.addRow("Total Period      (s):", total_le)

        mains_cb = QtWidgets.QComboBox()
        mains_cb.addItems(["None", "50 Hz Notch", "60 Hz Notch"])
        form.addRow("Remove Mains Noise:", mains_cb)

        drop_zero_cb = QtWidgets.QCheckBox("Discard 0 Hz results")
        drop_zero_cb.setChecked(True)
        form.addRow(drop_zero_cb)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        form.addRow(btns)

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        # 2) Parse inputs
        try:
            interval = float(interval_le.text())
            duration = float(duration_le.text())
            total    = float(total_le.text())
        except ValueError:
            QtWidgets.QMessageBox.critical(self, "Error", "Invalid numeric values.")
            return
        if interval <= 0 or duration <= 0 or total <= 0:
            QtWidgets.QMessageBox.critical(self, "Error", "Values must be positive.")
            return

        sr   = self.sample_rate
        channels = self.selected_channel_indices() if hasattr(self, "selected_channel_indices") else [0]
        notch_choice = mains_cb.currentText()
        drop_zero   = drop_zero_cb.isChecked()

        # 3) Volts‐conversion **exactly** as in LFM analysis** per channel
        try:
            vmax = float(self.max_voltage_entry.text())
        except:
            vmax = 1.0

        results_by_channel = []
        for ch in channels:
            data_ch = self.get_channel_data(ch)
            if data_ch is None:
                continue

            if notch_choice != "None":
                f0   = 50.0 if "50" in notch_choice else 60.0
                bw   = 2.0
                low  = (f0 - bw/2) / (sr/2)
                high = (f0 + bw/2) / (sr/2)
                sos  = butter(2, [low, high], btype="bandstop", output="sos")
                data_ch = safe_sosfiltfilt(sos, np.asarray(data_ch, dtype=np.float64))

            if np.issubdtype(self.original_dtype, np.integer):
                conv = vmax / np.iinfo(self.original_dtype).max
            else:
                conv = 1.0

            volts_data = data_ch.astype(np.float64) * conv

            # 5) Slide through windows
            results = []
            t = 0.0
            while t + duration <= total:
                i0, i1 = int(t * sr), int((t + duration) * sr)
                seg = volts_data[i0:i1]
                if seg.size == 0:
                    break

                # VRMS and max_v (we use the user’s vmax here)
                vr    = float(np.sqrt(np.mean(seg**2)))
                max_v = float(vmax)

                # Dominant frequency via FFT+refine
                nfft    = 1 << int(np.ceil(np.log2(len(seg))))
                win_seg = seg * get_window("hann", len(seg))
                fft_res = np.fft.rfft(win_seg, n=nfft)
                freqs   = np.fft.rfftfreq(nfft, 1 / sr)
                dom     = self.refine_frequency(fft_res, freqs)

                if drop_zero and (not np.isfinite(dom) or abs(dom) < 1e-9):
                    t += interval
                    continue

                results.append((t, vr, dom, max_v))
                t += interval

            if results:
                results_by_channel.append((ch, results))

        if not results_by_channel:
            QtWidgets.QMessageBox.information(
                self, "Interval Analysis", "No segments fit within the period."
            )
            return

        # 6) Results popup
        res_dlg = QtWidgets.QDialog(self)
        res_dlg.setWindowTitle("Interval Analysis Results")
        vlay = QtWidgets.QVBoxLayout(res_dlg)

        total_rows = sum(len(r) for _, r in results_by_channel)
        table = QtWidgets.QTableWidget(total_rows, 5)
        table.setHorizontalHeaderLabels(["Channel", "Time (s)", "VRMS (V)", "DomFreq (Hz)", "MaxV (V)"])
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        row = 0
        for ch, results in results_by_channel:
            for tt, vr, dom, mv in results:
                table.setItem(row, 0, QtWidgets.QTableWidgetItem(f"Ch {ch+1}"))
                table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{tt:.3f}"))
                table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{vr:.6f}"))
                table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{dom:.2f}"))
                table.setItem(row, 4, QtWidgets.QTableWidgetItem(f"{mv:.4f}"))
                row += 1
        table.resizeColumnsToContents()
        vlay.addWidget(table)

        btns2 = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        vlay.addWidget(btns2)

        def on_accept():
            total_saved = 0
            for ch, results in results_by_channel:
                fname = self.channel_file_label(ch)
                for tt, vr, dom, mv in results:
                    self.log_measurement_with_project(
                        fname,
                        "Interval Analysis",
                        float(dom),
                        float(tt),
                        float(tt + duration),
                        float(duration),
                        float(mv),
                        0.0,
                        float(vr),
                        False,
                        "",
                    )
                    total_saved += 1
            QtWidgets.QMessageBox.information(
                self, "Stored", f"{total_saved} measurements saved."
            )
            res_dlg.accept()

        btns2.accepted.connect(on_accept)
        btns2.rejected.connect(res_dlg.reject)

        res_dlg.exec_()

    def interval_analysis_popup(self):
        """
        Interval Analysis with two modes:
        1) Single‐file (dispatches to interval_analysis_single)
        2) Batch‐files
        """
        from PyQt5 import QtWidgets, QtCore
        import numpy as np
        from scipy.signal import get_window, butter, sosfiltfilt
        from scipy.io import wavfile
        import os

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Interval Analysis")
        dlg.resize(800, 600)
        vlay = QtWidgets.QVBoxLayout(dlg)

        tabs = QtWidgets.QTabWidget()
        vlay.addWidget(tabs)

        # Tab 1: Single‐file
        tab1 = QtWidgets.QWidget()
        t1lay = QtWidgets.QVBoxLayout(tab1)
        run_single_btn = QtWidgets.QPushButton("Single‑File Analysis…")
        run_single_btn.clicked.connect(self.interval_analysis_single)
        t1lay.addStretch()
        t1lay.addWidget(run_single_btn, alignment=QtCore.Qt.AlignCenter)
        t1lay.addStretch()
        tabs.addTab(tab1, "Single File")

        # Tab 2: Batch‐files
        tab2 = QtWidgets.QWidget()
        t2lay = QtWidgets.QVBoxLayout(tab2)

        form2 = QtWidgets.QFormLayout()
        interval_le = QtWidgets.QLineEdit("1.0");  interval_le.setFixedWidth(80)
        duration_le = QtWidgets.QLineEdit("0.5");  duration_le.setFixedWidth(80)
        form2.addRow("Interval (s):", interval_le)
        form2.addRow("Window Duration (s):", duration_le)
        notch_cb = QtWidgets.QComboBox()
        notch_cb.addItems(["None", "50 Hz Notch", "60 Hz Notch"])
        form2.addRow("Remove Mains Noise:", notch_cb)
        drop_zero_cb = QtWidgets.QCheckBox("Discard 0 Hz results")
        drop_zero_cb.setChecked(True)
        form2.addRow(drop_zero_cb)
        t2lay.addLayout(form2)

        file_h = QtWidgets.QHBoxLayout()
        select_btn = QtWidgets.QPushButton("Select WAV Files…")
        file_h.addWidget(select_btn); file_h.addStretch()
        t2lay.addLayout(file_h)

        file_list = QtWidgets.QListWidget()
        t2lay.addWidget(file_list)

        analyze_btn = QtWidgets.QPushButton("Analyze Batch")
        t2lay.addWidget(analyze_btn, alignment=QtCore.Qt.AlignRight)
        tabs.addTab(tab2, "Batch Files")

        batch_files = []

        def choose_files():
            nonlocal batch_files
            files, _ = QtWidgets.QFileDialog.getOpenFileNames(
                dlg, "Select WAV Files", "", "WAV Files (*.wav)"
            )
            if not files:
                return
            batch_files = files
            file_list.clear()
            file_list.addItems([os.path.basename(f) for f in batch_files])

        select_btn.clicked.connect(choose_files)



        def run_batch():
            if not batch_files:
                QtWidgets.QMessageBox.warning(dlg, "No Files", "Select at least one WAV.")
                return
            try:
                interval = float(interval_le.text())
                duration = float(duration_le.text())
            except ValueError:
                QtWidgets.QMessageBox.critical(dlg, "Error", "Invalid numeric parameters.")
                return
            if interval <= 0 or duration <= 0:
                QtWidgets.QMessageBox.critical(dlg, "Error", "Values must be positive.")
                return

            try:
                vmax = float(self.max_voltage_entry.text())
            except:
                vmax = 1.0

            notch_choice = notch_cb.currentText()
            drop_zero = drop_zero_cb.isChecked()

            channels = self.selected_channel_indices() if hasattr(self, "selected_channel_indices") else [0]

            all_results = {}
            for wav_path in batch_files:
                fname = os.path.basename(wav_path)
                try:
                    fs, data = wavfile.read(wav_path)
                except Exception as e:
                    all_results[fname] = f"Load error: {e}"
                    continue

                if data.ndim == 1:
                    data = data[:, None]

                per_file_results = []
                base, ext = os.path.splitext(fname)
                for ch in channels:
                    if ch >= data.shape[1]:
                        continue
                    chan_name = f"{base}_ch{ch+1}{ext}"

                    chan_data = data[:, ch].astype(np.float64)
                    dt = data.dtype if hasattr(data, "dtype") else None
                    if notch_choice != "None":
                        f0   = 50.0 if "50" in notch_choice else 60.0
                        bw   = 2.0
                        low  = (f0 - bw/2) / (fs/2)
                        high = (f0 + bw/2) / (fs/2)
                        sos  = butter(2, [low, high], btype="bandstop", output="sos")
                        chan_data = safe_sosfiltfilt(sos, chan_data)
                    if dt is not None and np.issubdtype(dt, np.integer):
                        conv = vmax / np.iinfo(dt).max
                    else:
                        conv = vmax
                    volts_data = chan_data.astype(np.float64) * conv

                    t = 0.0
                    ch_results = []
                    while t + duration <= len(volts_data) / fs:
                        i0 = int(t * fs)
                        i1 = int((t + duration) * fs)
                        seg = volts_data[i0:i1]
                        if seg.size == 0:
                            break

                        vr = float(np.sqrt(np.mean(seg**2)))
                        max_v = vmax

                        nfft    = 1 << int(np.ceil(np.log2(len(seg))))
                        win_seg = seg * get_window("hann", len(seg))
                        fft_res = np.fft.rfft(win_seg, n=nfft)
                        freqs   = np.fft.rfftfreq(nfft, 1/fs)
                        dom     = self.refine_frequency(fft_res, freqs)

                        if drop_zero and (not np.isfinite(dom) or abs(dom) < 1e-9):
                            t += interval
                            continue

                        ch_results.append((t, vr, dom, max_v))
                        t += interval

                    if ch_results:
                        per_file_results.append((chan_name, ch, ch_results))

                all_results[fname] = per_file_results if per_file_results else "No segments found"

            dlg2 = QtWidgets.QDialog(dlg)
            dlg2.setWindowTitle("Batch Interval Analysis Results")
            v2 = QtWidgets.QVBoxLayout(dlg2)

            text = ""
            for fname, res in all_results.items():
                text += f"=== {fname} ===\n"
                if isinstance(res, str):
                    text += res + "\n\n"
                else:
                    for chan_name, ch, ch_res in res:
                        text += f"Channel {ch+1} ({chan_name})\n"
                        text += "Time(s)   VRMS(V)   Freq(Hz)   MaxV(V)\n"
                        for t0, vr, freq, mv in ch_res:
                            text += f"{t0:6.3f}   {vr:7.4f}   {freq:8.2f}   {mv:7.4f}\n"
                        text += "\n"

            viewer = QtWidgets.QPlainTextEdit(text)
            viewer.setReadOnly(True)
            v2.addWidget(viewer)

            btns = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
            )
            v2.addWidget(btns)

            def on_accept():
                for fname, res in all_results.items():
                    if isinstance(res, list):
                        for chan_name, ch, ch_res in res:
                            for t0, vr, dom, mv in ch_res:
                                self.log_measurement_with_project(
                                    chan_name,
                                    "Interval Batch",
                                    float(dom),
                                    float(t0),
                                    float(t0 + duration),
                                    float(duration),
                                    float(mv),
                                    0.0,
                                    float(vr),
                                    False,
                                    "",
                                )
                QtWidgets.QMessageBox.information(
                    dlg2, "Stored", f"Stored measurements from {len(batch_files)} files."
                )
                dlg2.accept()

            btns.accepted.connect(on_accept)
            btns.rejected.connect(dlg2.reject)
            dlg2.exec_()
        analyze_btn.clicked.connect(run_batch)
        dlg.exec_()







        


    def lfm_pulse_batch_analysis(self):
        """
        Batch‐mode LFM pulse analysis over multiple WAV files.
        Settings:
        • Sampling interval (s)
        • Window length (s)
        • Pulse threshold (Amplitude)
        • Start time (s)
        • Mains‐noise removal (None/50 Hz/60 Hz)
        • Discard 0 Hz results
        • [NEW] Apply bandpass around Dom Freq ± tolerance (Hz)
        Select multiple WAVs, processes each, and shows a consolidated report.
        """
        from PyQt5 import QtWidgets
        import numpy as np
        import os
        from scipy.signal import butter, sosfiltfilt
        from scipy.io import wavfile

        # 1) Settings dialog
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Batch LFM Analysis Settings")
        form = QtWidgets.QFormLayout(dlg)

        interval_le = QtWidgets.QLineEdit(self.scroll_step_entry.text())
        interval_le.setFixedWidth(80)
        form.addRow("Sampling Interval (s):", interval_le)

        window_le = QtWidgets.QLineEdit(self.fft_length_entry.text())
        window_le.setFixedWidth(80)
        form.addRow("Window Length (s):", window_le)

        thresh_le = QtWidgets.QLineEdit(self.pulse_threshold_entry.text())
        thresh_le.setFixedWidth(80)
        form.addRow("Pulse Threshold (Amplitude):", thresh_le)

        start_le = QtWidgets.QLineEdit(f"{self.fft_time_slider.value()/self.TIME_MULTIPLIER:.3f}")
        start_le.setFixedWidth(80)
        form.addRow("Start Time (s):", start_le)

        noise_cb = QtWidgets.QComboBox()
        noise_cb.addItems(["None", "50 Hz Notch", "60 Hz Notch"])
        form.addRow("Remove Mains Noise:", noise_cb)

        drop_cb = QtWidgets.QCheckBox("Discard 0 Hz results")
        drop_cb.setChecked(True)
        form.addRow(drop_cb)

        # ── New bandpass option ──────────────────────────────────────────────
        filter_cb = QtWidgets.QCheckBox("Apply bandpass around Dom Freq")
        filter_cb.setChecked(False)
        form.addRow(filter_cb)
        tol_le = QtWidgets.QLineEdit("10")
        tol_le.setFixedWidth(80)
        form.addRow("Tolerance (Hz):", tol_le)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        form.addRow(btns)

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        # 2) Parse settings
        try:
            scroll_step   = float(interval_le.text())
            window_length = float(window_le.text())
            threshold     = float(thresh_le.text())
            start_time    = float(start_le.text())
            tol           = float(tol_le.text())
        except ValueError:
            QtWidgets.QMessageBox.critical(self, "Error", "Please enter valid numeric settings.")
            return

        notch_choice   = noise_cb.currentText()
        drop_zero      = drop_cb.isChecked()
        bandpass       = filter_cb.isChecked()

        # 3) Select WAV files
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select WAV Files", "", "WAV Files (*.wav)"
        )
        if not files:
            return

        # 4) Process each file
        all_results = {}
        for wav_path in files:
            fname = os.path.basename(wav_path)
            try:
                fs, data = wavfile.read(wav_path)
            except Exception as e:
                all_results[fname] = f"Load error: {e}"
                continue

            # normalize & mono
            if np.issubdtype(data.dtype, np.integer):
                data = data.astype(np.float32) / np.iinfo(data.dtype).max
            if data.ndim > 1:
                    # use first selected channel
                    ch0 = self.selected_channel_indices()[0] if hasattr(self, 'selected_channel_indices') else 0
                    data = data[:, ch0]

            # optional mains notch
            if notch_choice != "None":
                f0  = 50.0 if "50" in notch_choice else 60.0
                bw0 = 2.0
                low  = (f0 - bw0/2)/(fs/2)
                high = (f0 + bw0/2)/(fs/2)
                sos = butter(2, [low, high], btype="bandstop", output="sos")
                data = safe_sosfiltfilt(sos, data)

            # FFT size per file
            nfft = 1 << int(np.ceil(np.log2(int(window_length * fs))))

            t = start_time
            results = []
            while t + window_length <= len(data)/fs:
                i0, i1 = int(t*fs), int((t+window_length)*fs)
                seg = data[i0:i1]
                if np.max(np.abs(seg)) < threshold:
                    t += scroll_step
                    continue

                # FFT for dominant frequency (always from full segment)
                win_data = seg * np.hanning(len(seg))
                fft_res  = np.fft.rfft(win_data, n=nfft)
                freqs    = np.fft.rfftfreq(nfft, d=1.0/fs)
                dom      = self.refine_frequency(fft_res, freqs)

                # optionally band‑limit seg to [dom ± tol]
                if bandpass and dom > 0:
                    lowb = max((dom - tol)/(fs/2), 0.0)
                    highb= min((dom + tol)/(fs/2), 1.0)
                    if lowb < highb:
                        sos2 = butter(4, [lowb, highb], btype="bandpass", output="sos")
                        meas_seg = safe_sosfiltfilt(sos2, seg)
                    else:
                        meas_seg = seg
                else:
                    meas_seg = seg

                # volts conversion
                if np.issubdtype(self.original_dtype, np.integer):
                    vmax = float(self.max_voltage_entry.text())
                    conv = vmax / np.iinfo(self.original_dtype).max
                else:
                    conv = 1.0
                volts = meas_seg * conv
                rms   = float(np.sqrt(np.mean(volts**2)))
                max_v = float(np.max(np.abs(volts)))

                # approximate –3 dB bandwidth from full FFT results
                mag = np.abs(fft_res)
                if mag.size:
                    half = np.max(mag)/2.0
                    idxs = np.where(mag >= half)[0]
                    bw   = float(freqs[idxs[-1]] - freqs[idxs[0]]) if idxs.size else 0.0
                else:
                    bw = 0.0

                if drop_zero and (not np.isfinite(dom) or abs(dom) < 1e-9):
                    t += scroll_step
                    continue

                results.append((t, rms, dom, max_v, bw))
                t += scroll_step

            all_results[fname] = results or ["No pulses found above threshold"]

        # 5) Consolidated report
        dlg2 = QtWidgets.QDialog(self)
        dlg2.setWindowTitle("Batch LFM Analysis Results")
        lay2 = QtWidgets.QVBoxLayout(dlg2)

        text = ""
        for fname, res in all_results.items():
            text += f"=== {fname} ===\n"
            if isinstance(res, str) or (isinstance(res, list) and res and isinstance(res[0], str)):
                text += f"{res}\n\n"
            else:
                text += "Time(s)   RMS(V)   Freq(Hz)   MaxV(V)   BW(Hz)\n"
                for t0, rms, dom, mv, bw in res:
                    text += f"{t0:6.3f}   {rms:7.4f}   {dom:8.2f}   {mv:7.4f}   {bw:8.2f}\n"
                text += "\n"

        viewer = QtWidgets.QPlainTextEdit(text)
        viewer.setReadOnly(True)
        lay2.addWidget(viewer)

        btns2 = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btns2.accepted.connect(lambda: self._store_batch_lfm_results(all_results, window_length))
        btns2.rejected.connect(dlg2.reject)
        lay2.addWidget(btns2)

        dlg2.exec_()



    def _store_batch_lfm_results(self, results_dict, window_length):
        """
        Write batch LFM results into measurements DB.
        Expects each entry to be (time, rms, freq, max_voltage, bandwidth).
        """
        import sqlite3
        from PyQt5 import QtWidgets

        conn = sqlite3.connect(DB_FILENAME)
        cur  = conn.cursor()
        count = 0

        for fname, res in results_dict.items():
            if not isinstance(res, list):
                continue
            for entry in res:
                if not (isinstance(entry, tuple) and len(entry) == 5):
                    continue
                t0, rms, freq, max_v, bw = entry
                project_id = getattr(self, "current_project_id", None)
                cur.execute(
                    "INSERT INTO measurements "
                    "(file_name, method, target_frequency, start_time, end_time, "
                    " window_length, max_voltage, bandwidth, measured_voltage, "
                    " filter_applied, screenshot, misc, project_id) VALUES "
                    "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        fname,
                        "LFM Batch",
                        float(freq),
                        float(t0),
                        float(t0 + window_length),
                        float(window_length),
                        float(max_v),
                        float(bw),
                        float(rms),
                        0,
                        "",
                        None,
                        None if project_id is None else int(project_id),
                    )
                )
                count += 1

        conn.commit()
        conn.close()
        QtWidgets.QMessageBox.information(
            self,
            "Batch LFM Analysis",
            f"Stored {count} measurements from {len(results_dict)} files."
        )





    def ambient_noise_analysis(self):
        """
        Multi-channel Ambient Noise Analysis.

        - Measures ambient noise over time by sampling VRMS and dominant frequency
        in each sample window.
        - Supports multi-channel data:
            * Uses currently selected channels (_selected_channel_indices).
            * Computes VRMS & dominant frequency per channel.
            * Plots one VRMS-vs-time curve per channel using the colour palette.
            * Logs one measurement row per (channel, window) if requested.
        """
        from PyQt5 import QtWidgets
        import numpy as np
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        import matplotlib.pyplot as plt

        # 1) Ensure data is loaded
        if getattr(self, "full_data", None) is None:
            QtWidgets.QMessageBox.critical(self, "Error", "Load a WAV file first.")
            return

        sr = int(getattr(self, "sample_rate", 0)) or 0
        if sr <= 0:
            QtWidgets.QMessageBox.critical(self, "Error", "Sample rate is not set.")
            return

        # 2) Parameter dialog
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Ambient Noise Analysis")
        dlg.setStyleSheet("background-color: #19232D; color: white;")
        layout = QtWidgets.QVBoxLayout(dlg)

        form = QtWidgets.QFormLayout()
        interval_edit = QtWidgets.QLineEdit("10.0")
        duration_edit = QtWidgets.QLineEdit("1.0")
        total_edit    = QtWidgets.QLineEdit("600.0")
        for w in (interval_edit, duration_edit, total_edit):
            w.setFixedWidth(80)
        form.addRow("Sampling Interval (s):", interval_edit)
        form.addRow("Sample Duration   (s):", duration_edit)
        form.addRow("Total Period      (s):", total_edit)
        layout.addLayout(form)

        info_lbl = QtWidgets.QLabel(
            "Ambient noise is measured as short VRMS windows over time.\n"
            "Interval = spacing between windows; Duration = length of each window."
        )
        info_lbl.setWordWrap(True)
        layout.addWidget(info_lbl)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btns.setStyleSheet(
            """
            QDialogButtonBox { background: transparent; }
            QDialogButtonBox QPushButton {
                color: white;
                padding: 6px 12px;
                border-radius: 4px;
                background-color: #333;
            }
            QDialogButtonBox QPushButton:hover {
                background-color: #444;
            }
            """
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        # 3) Read & validate parameters
        try:
            interval = float(interval_edit.text())
            duration = float(duration_edit.text())
            total    = float(total_edit.text())
        except ValueError:
            QtWidgets.QMessageBox.critical(self, "Error", "Invalid numeric values.")
            return

        if interval <= 0 or duration <= 0 or total <= 0:
            QtWidgets.QMessageBox.critical(
                self, "Error", "Interval, duration, and total period must be positive."
            )
            return

        # 4) Prepare data and channel selection
        data = getattr(self, "full_data", None)
        if data is None:
            QtWidgets.QMessageBox.critical(self, "Error", "No data loaded.")
            return

        data = np.asarray(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        n_samples, n_ch = data.shape

        # Clamp total to the actual file length
        max_total = min(total, n_samples / float(sr))

        # Determine channels from UI selection
        try:
            sel_channels = _selected_channel_indices(self)
        except Exception:
            sel_channels = list(range(n_ch))

        sel_channels = [ch for ch in sel_channels if 0 <= ch < n_ch]
        if not sel_channels:
            sel_channels = [0]

        # Counts -> volts conversion
        max_v = 1.0
        conv = 1.0
        try:
            orig_dtype = getattr(self, "original_dtype", None)
            if orig_dtype is not None and np.issubdtype(orig_dtype, np.integer):
                try:
                    max_v = float(self.max_voltage_entry.text())
                except Exception:
                    max_v = 1.0
                conv = max_v / np.iinfo(orig_dtype).max
            else:
                conv = 1.0  # assume already in volts
        except Exception:
            conv = 1.0

        # 5) Compute VRMS + dominant frequency per channel
        times = []
        vrms_by_ch = {ch: [] for ch in sel_channels}
        df_by_ch   = {ch: [] for ch in sel_channels}

        t = 0.0
        while t + duration <= max_total + 1e-9:
            s0 = int(round(t * sr))
            s1 = int(round((t + duration) * sr))
            if s1 > n_samples:
                break
            seg = data[s0:s1, :] * conv

            for ch in sel_channels:
                x = np.asarray(seg[:, ch], dtype=np.float64)
                if x.size < 2:
                    vr = 0.0
                    df = 0.0
                else:
                    vr = float(np.sqrt(np.mean(x ** 2)))
                    # Dominant frequency
                    w = np.hanning(len(x))
                    X = np.fft.rfft(x * w)
                    freqs = np.fft.rfftfreq(len(x), d=1.0 / float(sr))
                    if X.size:
                        k = int(np.argmax(np.abs(X)))
                        df = float(freqs[k]) if k < freqs.size else 0.0
                    else:
                        df = 0.0
                vrms_by_ch[ch].append(vr)
                df_by_ch[ch].append(df)

            times.append(t)
            t += interval

        if not times:
            QtWidgets.QMessageBox.information(self, "No Data", "No segments fit within the period.")
            return

        # 6) Plot VRMS vs Time (one curve per channel)
        fig, ax = plt.subplots(facecolor="#19232D")
        ax.set_facecolor("#000000")

        # Build palette starting at current graph_color
        if hasattr(self, "color_options") and isinstance(self.color_options, dict):
            palette = list(self.color_options.values())
        else:
            palette = [getattr(self, "graph_color", "#03DFE2")]

        base_color = getattr(self, "graph_color", palette[0])
        try:
            base_idx = palette.index(base_color)
        except ValueError:
            base_idx = 0

        channel_names = getattr(self, "channel_names", None)

        for idx, ch in enumerate(sel_channels):
            col = palette[(base_idx + idx) % len(palette)]
            label = (
                channel_names[ch]
                if channel_names and ch < len(channel_names)
                else f"Ch {ch+1}"
            )
            ax.plot(times, vrms_by_ch[ch], lw=1.5, color=col, label=label)

        ax.set_title("Ambient Noise (VRMS) vs Time", color="white")
        ax.set_xlabel("Time (s)", color="white")
        ax.set_ylabel("VRMS (V)", color="white")
        ax.tick_params(axis="x", colors="white")
        ax.tick_params(axis="y", colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")
        if len(sel_channels) > 1:
            leg = ax.legend()
            for txt in leg.get_texts():
                txt.set_color("white")

        canvas = FigureCanvas(fig)

        # 7) Results table: one row per (channel, time) window
        rows = len(times) * len(sel_channels)
        table = QtWidgets.QTableWidget(rows, 4)
        table.setHorizontalHeaderLabels(["Channel", "Time (s)", "VRMS (V)", "DomFreq (Hz)"])
        table.setStyleSheet(
            """
            QTableWidget {
                background-color: #111;
                color: white;
                gridline-color: #333;
            }
            QHeaderView::section {
                background-color: #19232D;
                color: white;
                border: none;
            }
            """
        )
        r = 0
        for ch in sel_channels:
            label = (
                channel_names[ch]
                if channel_names and ch < len(channel_names)
                else f"Ch {ch+1}"
            )
            for t0, vr, df in zip(times, vrms_by_ch[ch], df_by_ch[ch]):
                table.setItem(r, 0, QtWidgets.QTableWidgetItem(label))
                table.setItem(r, 1, QtWidgets.QTableWidgetItem(f"{t0:.3f}"))
                table.setItem(r, 2, QtWidgets.QTableWidgetItem(f"{vr:.6f}"))
                table.setItem(r, 3, QtWidgets.QTableWidgetItem(f"{df:.2f}"))
                r += 1
        table.resizeColumnsToContents()

        # 8) Results dialog with Save, Store, Close
        result_dlg = QtWidgets.QDialog(self)
        result_dlg.setWindowTitle("Ambient Noise Results")
        result_dlg.setStyleSheet("background:#19232D; color:white;")
        v = QtWidgets.QVBoxLayout(result_dlg)
        v.addWidget(canvas)
        v.addWidget(table)

        btn_bar = QtWidgets.QHBoxLayout()
        btn_bar.addStretch()
        save_btn = QtWidgets.QPushButton("Save Graph…")
        store_btn = QtWidgets.QPushButton("Store to DB")
        close_btn = QtWidgets.QPushButton("Close")
        for b in (save_btn, store_btn, close_btn):
            b.setStyleSheet(
                "QPushButton { background:#333; color:white; padding:4px 10px; border-radius:4px; }"
                "QPushButton:hover { background:#444; }"
            )
        btn_bar.addWidget(save_btn)
        btn_bar.addWidget(store_btn)
        btn_bar.addWidget(close_btn)
        v.addLayout(btn_bar)

        def on_save():
            import os
            base_path = getattr(self, "current_file_path", "") or getattr(
                self, "wav_file_path", ""
            ) or ""
            if not base_path:
                base_path = getattr(self, "file_name", "waveform.wav")

            base_dir = os.path.dirname(base_path)
            if not base_dir:
                base_dir = os.getcwd()

            root, _ext = os.path.splitext(os.path.basename(base_path))
            default_path = os.path.join(base_dir, f"{root}_ambient_noise.png")

            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                result_dlg,
                "Save Graph",
                default_path,
                "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)",
            )
            if not path:
                return
            try:
                fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
                QtWidgets.QMessageBox.information(result_dlg, "Saved", f"Graph saved to:\n{path}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(result_dlg, "Save Graph", f"Could not save graph:\n{e}")

        
        def on_store():
            import numpy as np

            # --- NEW: check project once so we don't spam the warning ---
            if not getattr(self, "current_project_name", None):
                QtWidgets.QMessageBox.warning(
                    result_dlg,
                    "No Project Selected",
                    "Please select a project in the 'Project' dropdown next to "
                    "'Select File' before storing measurements."
                )
                return  # don't attempt any DB writes

            base, ext = _per_channel_basename(self)
            sr   = int(getattr(self, 'sample_rate', 0)) or 0
            data = getattr(self, 'full_data', None)
            chs  = _selected_channel_indices(self)
            if data is None or sr <= 0 or not chs:
                return

            win_samples = max(1, int(duration * sr))
            for ch in chs:
                ch_name = f"{base}_ch{ch+1}{ext}"
                for t0 in times:
                    i0 = max(0, int(t0 * sr))
                    i1 = min(i0 + win_samples, int(getattr(data, 'shape', [0])[0]))
                    arr = np.asarray(data[i0:i1, ch], dtype=float)
                    if arr.size == 0:
                        continue

                    # convert to volts if needed
                    orig_dtype = getattr(self, "original_dtype", None)
                    if orig_dtype is not None and np.issubdtype(orig_dtype, np.integer):
                        try:
                            max_v = float(self.max_voltage_entry.text())
                        except Exception:
                            max_v = 1.0
                        conv = max_v / np.iinfo(orig_dtype).max
                    else:
                        max_v = 1.0
                        conv = 1.0

                    x = arr * conv
                    vr = float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0

                    if x.size > 1 and sr > 0:
                        w = np.hanning(len(x))
                        X = np.fft.rfft(x * w)
                        f = np.fft.rfftfreq(len(x), d=1.0 / float(sr))
                        k = int(np.argmax(np.abs(X))) if X.size else 0
                        df = float(f[k]) if k < f.size else 0.0
                    else:
                        df = 0.0

                    self.log_measurement_with_project(
                        ch_name,
                        "Ambient Noise",
                        df,
                        t0,
                        t0 + duration,
                        duration,
                        max_v,
                        0.0,
                        vr,
                        False,
                        "",
                    )

        save_btn.clicked.connect(on_save)
        store_btn.clicked.connect(on_store)
        close_btn.clicked.connect(result_dlg.accept)

        result_dlg.exec_()



    def analyze_dominant_frequencies_popup(self):
        """
        Popup analysis: Dominant frequencies over time (Top-N; rFFT or Welch), multi-channel.
        - Own QDialog + Matplotlib canvas (does NOT draw in main window)
        - Plot style dropdown: Lines or Points
        - Export CSV button
        - Save Graph button (Dark/Light)
        - Analyze Patterns... to highlight whistle-like regions inside a frequency band
        - Highlight color matches the plotted series color (rank-0 per channel)
        """
        from PyQt5 import QtWidgets, QtCore
        import numpy as np
        from scipy.signal import welch
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib import patches as _mpatches

        # ---- robust SR + data ----
        _sr_candidates = (
            getattr(self, 'sample_rate', None),
            getattr(self, 'samplerate', None),
            getattr(self, 'sr', None),
        )
        sr_val = next((v for v in _sr_candidates if isinstance(v, (int, float)) and v > 0), None)
        sr = int(sr_val) if sr_val is not None else 0
        data = getattr(self, 'full_data', None)
        if sr <= 0 or data is None or (getattr(data, 'size', 0) == 0):
            QtWidgets.QMessageBox.warning(self, "No data", "Load a WAV file first (sample rate and data required).")
            return

        # ---- parameter dialog ----
        class _ParamDlg(QtWidgets.QDialog):
            def __init__(self, parent=None):
                super().__init__(parent)
                self.setWindowTitle("Dominant Frequencies Over Time — Parameters")
                self.resize(540, 340)
                v = QtWidgets.QVBoxLayout(self)
                form = QtWidgets.QFormLayout()

                self.win_edit  = QtWidgets.QLineEdit("0.250")   # seconds
                self.hop_edit  = QtWidgets.QLineEdit("0.050")   # seconds
                self.fmin_edit = QtWidgets.QLineEdit("1.0")     # Hz
                self.fmax_edit = QtWidgets.QLineEdit(str(max(10.0, sr/2.0)))

                self.method_combo = QtWidgets.QComboBox()
                self.method_combo.addItems(["rFFT", "Welch"])

                self.topn_spin = QtWidgets.QSpinBox()
                self.topn_spin.setRange(1, 10)
                self.topn_spin.setValue(3)

                # Plot style
                self.style_combo = QtWidgets.QComboBox()
                self.style_combo.addItems(["Lines", "Points"])

                form.addRow("Window length (s):", self.win_edit)
                form.addRow("Hop (s):",          self.hop_edit)
                form.addRow("Min freq (Hz):",    self.fmin_edit)
                form.addRow("Max freq (Hz):",    self.fmax_edit)
                form.addRow("Method:",           self.method_combo)
                form.addRow("Top-N peaks:",      self.topn_spin)
                form.addRow("Plot style:",       self.style_combo)

                v.addLayout(form)
                btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok|QtWidgets.QDialogButtonBox.Cancel)
                v.addWidget(btns)
                btns.accepted.connect(self.accept)
                btns.rejected.connect(self.reject)

            def params(self):
                return (float(self.win_edit.text()),
                        float(self.hop_edit.text()),
                        float(self.fmin_edit.text()),
                        float(self.fmax_edit.text()),
                        self.method_combo.currentText(),
                        int(self.topn_spin.value()),
                        self.style_combo.currentText())

        pdlg = _ParamDlg(self)
        if pdlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        try:
            win_sec, hop_sec, fmin, fmax, method, topn, plot_style = pdlg.params()
            if win_sec <= 0 or hop_sec <= 0 or fmin < 0 or fmax <= 0 or fmax <= fmin:
                raise ValueError
        except Exception:
            QtWidgets.QMessageBox.warning(self, "Invalid parameters", "Please enter valid positive numbers.")
            return

        # ---- channels ----
        def _selected_channels():
            if hasattr(self, 'selected_channel_indices') and callable(self.selected_channel_indices):
                sel = self.selected_channel_indices()
                if sel:
                    return [int(ch) for ch in sel]
            if getattr(data, 'ndim', 1) > 1:
                return list(range(data.shape[1]))
            return [0]

        ch_idx = _selected_channels()

        # Standardize (T x C)
        if getattr(data, 'ndim', 1) == 1:
            X = np.asarray(data).reshape(-1, 1)
        else:
            X = np.asarray(data)
            if X.ndim != 2:
                QtWidgets.QMessageBox.warning(self, "Data error", "Unsupported data shape.")
                return

        N = X.shape[0]
        C = X.shape[1]
        ch_idx = [ch for ch in ch_idx if 0 <= ch < C]
        if not ch_idx:
            QtWidgets.QMessageBox.warning(self, "Channels", "No valid channels selected.")
            return

        # ---- framing ----
        win_len = int(round(win_sec * sr))
        hop_len = int(round(hop_sec * sr))
        if win_len < 8 or hop_len < 1:
            QtWidgets.QMessageBox.warning(self, "Too small", "Window or hop too small for analysis.")
            return
        if win_len > N:
            QtWidgets.QMessageBox.warning(self, "Too large", "Window larger than the file.")
            return
        starts = np.arange(0, N - win_len + 1, hop_len, dtype=int)
        if starts.size == 0:
            QtWidgets.QMessageBox.warning(self, "Short file", "No frames fit with the current parameters.")
            return
        time_stamps = starts / float(sr)

        # ---- frequency axis per method ----
        if method == "rFFT":
            nfft = 1 << int(np.ceil(np.log2(win_len)))
            freqs = np.fft.rfftfreq(nfft, 1.0/sr)
            band = (freqs >= fmin) & (freqs <= fmax)
            if not np.any(band):
                QtWidgets.QMessageBox.warning(self, "Band", "Min/Max frequency band has no bins. Adjust parameters.")
                return
            band_idx = np.where(band)[0]
            w = np.hanning(win_len).astype(np.float32)
            buf = np.zeros(nfft, dtype=np.float32)
        else:
            f_dummy, _ = welch(np.zeros(win_len, dtype=np.float32), fs=sr, nperseg=win_len,
                            noverlap=0, nfft=None, detrend='constant', return_onesided=True)
            freqs = f_dummy
            band = (freqs >= fmin) & (freqs <= fmax)
            if not np.any(band):
                QtWidgets.QMessageBox.warning(self, "Band", "Min/Max frequency band has no bins. Adjust parameters.")
                return
            band_idx = np.where(band)[0]

        # ---- compute Top-N per frame/channel ----
        tracks = {(ch, r): np.full(starts.size, np.nan, dtype=np.float32)
                for ch in ch_idx for r in range(topn)}

        prog = QtWidgets.QProgressDialog(f"Analyzing ({method})…", "Cancel", 0, int(starts.size), self)
        prog.setWindowTitle("Dominant Frequencies")
        prog.setWindowModality(QtCore.Qt.ApplicationModal)
        prog.setAutoClose(True); prog.setAutoReset(True)
        prog.show()

        canceled = False
        for i, s0 in enumerate(starts):
            s1 = s0 + win_len
            seg = X[s0:s1, :]  # (win_len, C)

            for ch in ch_idx:
                x = seg[:, ch].astype(np.float32, copy=False)
                if method == "rFFT":
                    xw = x * w
                    buf[:win_len] = xw
                    if len(buf) > win_len:
                        buf[win_len:] = 0.0
                    spec = np.fft.rfft(buf, n=nfft)
                    mag = np.abs(spec)
                else:
                    f, Pxx = welch(x, fs=sr, nperseg=win_len, noverlap=0, nfft=None,
                                detrend='constant', return_onesided=True, scaling='density')
                    mag = Pxx

                mb = mag[band_idx]
                if mb.size == 0:
                    continue

                k = min(topn, mb.size)
                idx_part = np.argpartition(-mb, k-1)[:k]
                idx_sorted = idx_part[np.argsort(-mb[idx_part])]

                for rank, local_idx in enumerate(idx_sorted):
                    global_bin = band_idx[local_idx]
                    tracks[(ch, rank)][i] = freqs[global_bin]

            if (i & 7) == 0:
                prog.setValue(i)
                QtWidgets.QApplication.processEvents()
                if prog.wasCanceled():
                    canceled = True
                    break

        prog.setValue(int(starts.size))
        if canceled:
            QtWidgets.QMessageBox.information(self, "Canceled", "Dominant-frequency analysis canceled.")
            return

        # ---- popup plot dialog (with Export CSV + Save Graph + Analyze Patterns) ----
        class _PlotDlg(QtWidgets.QDialog):
            def __init__(self, parent=None):
                super().__init__(parent)
                self.setWindowTitle("Dominant Frequencies Over Time — Plot")
                self.resize(1000, 600)
                v = QtWidgets.QVBoxLayout(self)

                self.fig = Figure(facecolor='black')
                self.canvas = FigureCanvas(self.fig)
                v.addWidget(self.canvas)

                self.ax = self.fig.add_subplot(111, facecolor='black')
                self.ax.tick_params(colors='white')
                for sp in self.ax.spines.values():
                    sp.set_edgecolor('white')

                # control row
                h = QtWidgets.QHBoxLayout()
                self.btn_export  = QtWidgets.QPushButton("Export CSV…")
                self.btn_save    = QtWidgets.QPushButton("Save Graph…")
                self.btn_pattern = QtWidgets.QPushButton("Analyze Patterns…")
                h.addWidget(self.btn_export)
                h.addWidget(self.btn_save)
                h.addWidget(self.btn_pattern)
                h.addStretch()
                v.addLayout(h)

                btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
                btns.rejected.connect(self.reject)
                btns.accepted.connect(self.accept)
                v.addWidget(btns)

        plotdlg = _PlotDlg(self)

        # ---- draw into popup canvas ----
        ax = plotdlg.ax
        names = getattr(self, 'channel_names', None)
        multi_ch = len(ch_idx) > 1
        multi_rank = (topn > 1)

        # NEW: capture actual colors used for each (channel, rank)
        plot_color = {}  # (ch, r) -> color used by Matplotlib

        finite_any = False
        for ch in ch_idx:
            for r in range(topn):
                y = tracks[(ch, r)]
                if np.isfinite(y).any():
                    finite_any = True
                lbl_ch = (names[ch] if names and ch < len(names) else f"Ch {ch+1}")
                lbl = f"{lbl_ch} #{r+1}" if multi_rank else lbl_ch

                # If only one series overall we try to use self.graph_color; otherwise let MPL cycle.
                explicit_color = getattr(self, 'graph_color', None) if (not multi_ch and not multi_rank) else None

                if plot_style == "Lines":
                    line, = ax.plot(time_stamps, y, lw=1.6, label=lbl, color=explicit_color)
                    used = line.get_color()
                else:
                    sc = ax.scatter(time_stamps, y, s=9.0, label=lbl, color=explicit_color)
                    # scatter may return an array of RGBA; take first
                    fcols = sc.get_facecolors()
                    used = fcols[0] if fcols is not None and len(fcols) else explicit_color

                plot_color[(ch, r)] = used

        ax.set_title(f"Top-{topn} Dominant Frequency Tracks ({method}, {plot_style})  "
                    f"(win={win_sec:.3f}s, hop={hop_sec:.3f}s)", color='white')
        ax.set_xlabel("Time (s)", color='white')
        ax.set_ylabel("Frequency (Hz)", color='white')
        ax.tick_params(colors='white')
        for sp in ax.spines.values():
            sp.set_edgecolor('white')

        if multi_ch or multi_rank:
            leg = ax.legend(frameon=False)
            if leg:
                for t in leg.get_texts():
                    t.set_color('white')

        ymin = max(0.0, fmin * 0.95)
        ymax = fmax * 1.05
        if ymax > ymin:
            ax.set_ylim([ymin, ymax])

        plotdlg.canvas.draw()

        # -------------- Export CSV handler --------------
        def on_export_csv():
            import csv
            path, _ = QtWidgets.QFileDialog.getSaveFileName(plotdlg, "Export CSV", "", "CSV Files (*.csv);;All Files (*)")
            if not path:
                return
            if not path.lower().endswith(".csv"):
                path += ".csv"

            headers = ["time_s"]
            for ch in ch_idx:
                for r in range(topn):
                    lbl_ch = (names[ch] if names and ch < len(names) else f"Ch {ch+1}")
                    headers.append(f"{lbl_ch} #{r+1} (Hz)" if multi_rank else f"{lbl_ch} (Hz)")

            try:
                with open(path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(headers)
                    for i in range(len(time_stamps)):
                        row = [f"{time_stamps[i]:.9f}"]
                        for ch in ch_idx:
                            for r in range(topn):
                                val = tracks[(ch, r)][i]
                                row.append(f"{float(val):.9f}" if np.isfinite(val) else "")
                        w.writerow(row)
                QtWidgets.QMessageBox.information(plotdlg, "Export CSV", f"Saved:\n{path}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(plotdlg, "Export CSV", f"Failed to save CSV:\n{e}")

        # -------------- Save Graph handler (Dark/Light) --------------
        def _apply_theme(ax, mode):
            fig = ax.figure
            if mode == "Dark":
                fig.set_facecolor("black"); ax.set_facecolor("black")
                ax.tick_params(colors='white')
                for sp in ax.spines.values(): sp.set_edgecolor('white')
                if ax.title: ax.title.set_color('white')
                ax.xaxis.label.set_color('white'); ax.yaxis.label.set_color('white')
                leg = ax.get_legend()
                if leg:
                    for t in leg.get_texts(): t.set_color('white')
            else:
                fig.set_facecolor("white"); ax.set_facecolor("white")
                ax.tick_params(colors='black')
                for sp in ax.spines.values(): sp.set_edgecolor('black')
                if ax.title: ax.title.set_color('black')
                ax.xaxis.label.set_color('black'); ax.yaxis.label.set_color('black')
                leg = ax.get_legend()
                if leg:
                    for t in leg.get_texts(): t.set_color('black')

        def on_save_graph():
            mode, ok = QtWidgets.QInputDialog.getItem(
                plotdlg, "Save Graph", "Color theme:", ["Dark", "Light"], 0, False
            )
            if not ok: return
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                plotdlg, "Save Graph", "", "PNG Image (*.png);;PDF (*.pdf);;SVG (*.svg);;All Files (*)"
            )
            if not path: return

            fig = ax.figure
            old_fc_fig = fig.get_facecolor()
            old_fc_ax  = ax.get_facecolor()
            old_tick   = ax.xaxis.label.get_color()
            old_spines = {k: v.get_edgecolor() for k, v in ax.spines.items()}
            leg = ax.get_legend()
            leg_colors = [t.get_color() for t in leg.get_texts()] if leg else None
            title_color = ax.title.get_color() if ax.title else None

            try:
                _apply_theme(ax, mode)
                plotdlg.canvas.draw()
                fig.savefig(path, dpi=160, bbox_inches="tight")
                QtWidgets.QMessageBox.information(plotdlg, "Save Graph", f"Saved:\n{path}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(plotdlg, "Save Graph", f"Failed to save graph:\n{e}")
            finally:
                fig.set_facecolor(old_fc_fig); ax.set_facecolor(old_fc_ax)
                ax.tick_params(colors=old_tick)
                for k, v in ax.spines.items():
                    v.set_edgecolor(old_spines.get(k, 'white'))
                if ax.title and title_color is not None:
                    ax.title.set_color(title_color)
                if leg and leg_colors is not None:
                    for t, c in zip(leg.get_texts(), leg_colors): t.set_color(c)
                plotdlg.canvas.draw()

        # -------------- Analyze Patterns (whistle-like) --------------
        def on_analyze_patterns():
            # small dialog for band + min duration + gaps + ranks + (optional) slope
            class _PD(QtWidgets.QDialog):
                def __init__(self, parent=None):
                    super().__init__(parent)
                    self.setWindowTitle("Analyze Patterns (Highlight Regions)")
                    self.resize(420, 260)
                    v = QtWidgets.QVBoxLayout(self)
                    form = QtWidgets.QFormLayout()

                    self.band_lo = QtWidgets.QLineEdit("10000")    # Hz
                    self.band_hi = QtWidgets.QLineEdit("14000")    # Hz
                    self.min_dur = QtWidgets.QLineEdit("0.25")     # seconds
                    self.allow_gap = QtWidgets.QSpinBox(); self.allow_gap.setRange(0, 50); self.allow_gap.setValue(1)
                    self.use_ranks = QtWidgets.QSpinBox(); self.use_ranks.setRange(1, max(1, topn)); self.use_ranks.setValue(min(2, topn))
                    self.max_slope = QtWidgets.QLineEdit("")       # Hz/sec empty = ignore

                    form.addRow("Band low (Hz):",  self.band_lo)
                    form.addRow("Band high (Hz):", self.band_hi)
                    form.addRow("Min duration (s):", self.min_dur)
                    form.addRow("Allow gap (frames):", self.allow_gap)
                    form.addRow("Use top ranks up to:", self.use_ranks)
                    form.addRow("Max slope (Hz/s, optional):", self.max_slope)

                    v.addLayout(form)
                    btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok|QtWidgets.QDialogButtonBox.Cancel)
                    v.addWidget(btns)
                    btns.accepted.connect(self.accept)
                    btns.rejected.connect(self.reject)

                def params(self):
                    lo = float(self.band_lo.text()); hi = float(self.band_hi.text())
                    md = float(self.min_dur.text()); gap = int(self.allow_gap.value())
                    ur = int(self.use_ranks.value())
                    ms = self.max_slope.text().strip()
                    ms_val = float(ms) if ms else None
                    return lo, hi, md, gap, ur, ms_val

            dlg2 = _PD(plotdlg)
            if dlg2.exec_() != QtWidgets.QDialog.Accepted:
                return

            try:
                blo, bhi, min_dur, allow_gap, use_ranks, max_slope = dlg2.params()
                if not (blo > 0 and bhi > blo and min_dur > 0 and use_ranks >= 1):
                    raise ValueError
            except Exception:
                QtWidgets.QMessageBox.warning(plotdlg, "Invalid", "Please enter valid pattern parameters.")
                return

            frames = len(time_stamps)
            hop = hop_sec  # seconds per step
            total_boxes = 0

            for ci, ch in enumerate(ch_idx):
                mask = np.zeros(frames, dtype=bool)
                for r in range(min(use_ranks, topn)):
                    y = tracks[(ch, r)]
                    mask |= (np.isfinite(y) & (y >= blo) & (y <= bhi))

                if not mask.any():
                    continue

                if max_slope is not None and max_slope > 0:
                    y0 = tracks[(ch, 0)]
                    ok = np.ones(frames, dtype=bool)
                    dy = np.diff(y0)
                    good = np.isfinite(y0[:-1]) & np.isfinite(y0[1:])
                    slope = np.zeros_like(y0, dtype=float)
                    slope[1:][good] = np.abs(dy[good] / hop)
                    ok &= (slope <= max_slope)
                    mask &= ok

                # Merge contiguous segments allowing short gaps
                i = 0
                while i < frames:
                    while i < frames and not mask[i]:
                        i += 1
                    if i >= frames: break
                    start = i
                    gaps_left = allow_gap
                    j = i + 1
                    while j < frames:
                        if mask[j]:
                            j += 1
                            continue
                        if gaps_left > 0:
                            gaps_left -= 1
                            j += 1
                            continue
                        else:
                            break
                    end = j - 1  # inclusive

                    t0 = time_stamps[start]
                    t1 = time_stamps[end] + win_sec
                    dur = max(0.0, t1 - t0)

                    if dur >= min_dur:
                        width = max(1e-9, t1 - t0)
                        height = max(1e-6, bhi - blo)
                        # >>> Use the plotted series color (rank-0) for this channel
                        rect_color = plot_color.get((ch, 0), None)  # fallback to MPL default if None
                        rect = _mpatches.Rectangle(
                            (t0, blo), width, height,
                            fill=True, alpha=0.15,
                            edgecolor=rect_color, linewidth=2.0,
                            facecolor=rect_color
                        )
                        ax.add_patch(rect)
                        total_boxes += 1

                    i = j

            plotdlg.canvas.draw()
            QtWidgets.QMessageBox.information(
                plotdlg, "Analyze Patterns",
                f"Highlighted {total_boxes} region(s) within {blo:.0f}–{bhi:.0f} Hz."
            )

        # connect buttons
        plotdlg.btn_export.clicked.connect(on_export_csv)
        plotdlg.btn_save.clicked.connect(on_save_graph)
        plotdlg.btn_pattern.clicked.connect(on_analyze_patterns)

        if not finite_any:
            QtWidgets.QMessageBox.information(
                plotdlg, "No peaks",
                "No dominant frequencies found within the selected band for the given parameters."
            )

        plotdlg.exec_()








    def recurrence_periodicity_popup(self):
        """
        Recurrence / Periodicity analysis:
        • Extract amplitude envelope (Hilbert) and detect events via threshold+refractory.
        • Inter-Event Intervals (IEI) histogram.
        • Event-train Autocorrelation (ACF) on resampled binary train.
        • Envelope periodogram (FFT) → candidate periods (T = 1/f).
        • Rank & report candidate periods via phase-locking (Rayleigh R-stat).
        • Dark theme, maximized. Export plots/CSV and optional DB log of top result.
        """
        from PyQt5 import QtWidgets, QtCore
        import numpy as np
        from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks, get_window
        from scipy import signal
        import sqlite3, csv, os, math
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

        if getattr(self, "full_data", None) is None:
            QtWidgets.QMessageBox.warning(self, "No file", "Load a WAV file first.")
            return

        fs = float(self.sample_rate)
        x  = self.full_data.astype(np.float64)
        if x.ndim > 1:
            x = x.mean(axis=1)

        # ---- UI shell ---------------------------------------------------------
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Recurrence / Periodicity")
        dlg.setWindowState(dlg.windowState() | QtCore.Qt.WindowMaximized)
        dlg.setStyleSheet("background:#19232D; color:white;")
        outer = QtWidgets.QVBoxLayout(dlg); outer.setContentsMargins(10,10,10,10); outer.setSpacing(8)

        # Controls (compact grid)
        grid = QtWidgets.QGridLayout(); outer.addLayout(grid)
        r = 0
        def _edit(txt, w=80):
            e = QtWidgets.QLineEdit(txt); e.setFixedWidth(w); return e

        grid.addWidget(QtWidgets.QLabel("Threshold (amp):"), r, 0)
        thr_edit = _edit("1000"); grid.addWidget(thr_edit, r, 1)
        grid.addWidget(QtWidgets.QLabel("Min gap (s):"), r, 2)
        gap_edit = _edit("0.20", 70); grid.addWidget(gap_edit, r, 3)
        grid.addWidget(QtWidgets.QLabel("Min event len (s):"), r, 4)
        evlen_edit = _edit("0.02", 70); grid.addWidget(evlen_edit, r, 5)

        r += 1
        grid.addWidget(QtWidgets.QLabel("Band (Hz):"), r, 0)
        fmin_edit = _edit("", 80); grid.addWidget(fmin_edit, r, 1)
        grid.addWidget(QtWidgets.QLabel("to"), r, 2)
        fmax_edit = _edit("", 80); grid.addWidget(fmax_edit, r, 3)
        grid.addWidget(QtWidgets.QLabel("BP order:"), r, 4)
        order_edit = _edit("6", 60); grid.addWidget(order_edit, r, 5)

        r += 1
        grid.addWidget(QtWidgets.QLabel("ACF window (s):"), r, 0)
        acf_win_edit = _edit("10.0", 80); grid.addWidget(acf_win_edit, r, 1)
        grid.addWidget(QtWidgets.QLabel("ACF dt (s):"), r, 2)
        acf_dt_edit  = _edit("0.01", 80); grid.addWidget(acf_dt_edit, r, 3)
        grid.addWidget(QtWidgets.QLabel("Env NFFT:"), r, 4)
        nfft_cb = QtWidgets.QComboBox(); 
        for n in (1024, 2048, 4096, 8192, 16384, 32768):
            nfft_cb.addItem(str(n), n)
        nfft_cb.setCurrentIndex(2)  # 4096
        nfft_cb.setFixedWidth(90)
        grid.addWidget(nfft_cb, r, 5)

        r += 1
        grid.addWidget(QtWidgets.QLabel("Period search range (s):"), r, 0)
        pmin_edit = _edit("0.05", 80); grid.addWidget(pmin_edit, r, 1)
        grid.addWidget(QtWidgets.QLabel("to"), r, 2)
        pmax_edit = _edit("5.0", 80); grid.addWidget(pmax_edit, r, 3)

        compute_btn = QtWidgets.QPushButton("Analyze")
        compute_btn.setStyleSheet("background:#3E6C8A;color:white;padding:6px 12px;border-radius:4px;")
        grid.addWidget(compute_btn, r, 5, 1, 1)
        grid.setColumnStretch(6, 1)

        # Splitter: table left, plots right
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        outer.addWidget(split, 1)

        # Candidate periods table
        table = QtWidgets.QTableWidget(0, 6)
        table.setHorizontalHeaderLabels(["Rank","Period (s)","Hz","Score (R)","From","Events"])
        table.setStyleSheet("QTableWidget{background:#19232D;color:white;gridline-color:#444;}")
        split.addWidget(table)

        # Plots (2x2)
        fig = Figure(facecolor="#19232D")
        ax_env  = fig.add_subplot(221)  # envelope + events
        ax_acf  = fig.add_subplot(222)  # ACF
        ax_fft  = fig.add_subplot(224)  # Envelope periodogram (freq→period)
        ax_iei  = fig.add_subplot(223)  # IEI histogram
        for a in (ax_env, ax_acf, ax_fft, ax_iei):
            a.set_facecolor("#19232D"); a.tick_params(colors="white")
            for s in a.spines.values(): s.set_color("white")
            a.grid(True, ls="--", alpha=0.35, color="gray")
        can = FigureCanvas(fig)
        right = QtWidgets.QWidget(); rv = QtWidgets.QVBoxLayout(right); rv.addWidget(can)
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 7)

        # Bottom buttons
        bot = QtWidgets.QHBoxLayout(); outer.addLayout(bot)
        save_img = QtWidgets.QPushButton("Save Plot"); save_csv = QtWidgets.QPushButton("Export CSV"); log_btn = QtWidgets.QPushButton("Log Top to DB"); close_btn = QtWidgets.QPushButton("Close")
        for b in (save_img, save_csv, log_btn, close_btn):
            b.setStyleSheet("background:#3E6C8A;color:white;padding:6px 12px;border-radius:4px;")
            bot.addWidget(b)
        bot.addStretch()

        # Working storage
        results = []         # candidate list dicts
        event_times = None   # seconds
        top_period = None    # best period (s)
        R_for_top  = None

        # ---- helpers ----------------------------------------------------------
        def _bandpass(sig):
            fmin_txt, fmax_txt = fmin_edit.text().strip(), fmax_edit.text().strip()
            if not fmin_txt and not fmax_txt:
                return sig
            try:
                lo_hz = float(fmin_txt) if fmin_txt else None
                hi_hz = float(fmax_txt) if fmax_txt else None
            except ValueError:
                return sig
            lo = 0.0 if lo_hz is None else max(0.0, lo_hz) / (fs/2)
            hi = 1.0 if hi_hz is None else min(hi_hz, fs/2) / (fs/2)
            if not (0.0 <= lo < hi <= 1.0):
                return sig
            try:
                ordN = max(2, int(float(order_edit.text())))
            except Exception:
                ordN = 6
            sos = butter(ordN, [lo, hi], btype="band", output="sos")
            try:
                return safe_sosfiltfilt(sos, sig)
            except Exception:
                return sig

        def _detect_events(env, thr, min_gap_s, min_len_s):
            # peaks on envelope; enforce refractory via distance
            dist = max(1, int(round(min_gap_s * fs)))
            peaks, props = find_peaks(env, height=thr, distance=dist)
            # optional min length using half-height width
            if "widths" not in props or props.get("widths", None) is None:
                # estimate widths at half-height
                try:
                    widths, _, _, _ = signal.peak_widths(env, peaks, rel_height=0.5)
                except Exception:
                    widths = np.zeros_like(peaks, dtype=float)
            else:
                widths = props["widths"]
            keep = []
            Lmin = max(1, int(round(min_len_s * fs)))
            for p, w in zip(peaks, widths):
                if w >= Lmin:
                    keep.append(p)
            pts = np.array(keep, dtype=int)
            return pts / fs, pts  # times, samples

        def _event_train(times, dt, dur):
            # binary train sampled at dt with ones at nearest bins to event times
            N = int(math.ceil(dur / dt))
            t = np.arange(N) * dt
            s = np.zeros(N, dtype=np.float64)
            if times.size:
                idx = np.clip(np.round(times / dt).astype(int), 0, N-1)
                s[idx] = 1.0
            return t, s

        def _acf(sig, max_lag_s, dt):
            # zero-mean normalized ACF (one-sided)
            Nlag = int(max_lag_s / dt)
            z = sig - sig.mean()
            if z.std() == 0:
                return np.linspace(0, max_lag_s, Nlag+1), np.zeros(Nlag+1)
            ac = np.correlate(z, z, mode="full")
            mid = ac.size // 2
            ac = ac[mid: mid + Nlag + 1]
            ac = ac / ac[0]  # normalize
            lags = np.arange(ac.size) * dt
            return lags, ac

        def _envelope_periodogram(env, nfft, pmin, pmax):
            # FFT of (demeaned) envelope → get frequency peaks within [1/pmax, 1/pmin]
            y = env - np.mean(env)
            nfft = int(nfft)
            if nfft < len(y):
                # use center segment to reduce leakage
                start = (len(y) - nfft) // 2
                y = y[start:start+nfft]
            win = get_window("hann", len(y))
            Y = np.fft.rfft(y * win, n=nfft)
            f = np.fft.rfftfreq(nfft, d=1.0/fs)  # Hz
            P = (np.abs(Y) ** 2)
            # mask frequency band
            fmin = 1.0 / max(pmax, 1e-6)
            fmax = 1.0 / max(pmin, 1e-6)
            m = (f >= fmin) & (f <= fmax)
            return f[m], P[m]

        def _rayleigh_R(times, T):
            # phase-lock score in [0,1]; higher means stronger periodicity
            if times.size == 0 or T <= 0:
                return 0.0
            ang = 2*np.pi * (times % T) / T
            R = np.abs(np.mean(np.exp(1j*ang)))
            return float(R)

        def _extract_candidates(lags, ac, f_env, P_env, iei):
            cand = {}
            # From ACF: local maxima (skip lag=0)
            if ac.size > 4:
                pk, _ = find_peaks(ac, height=0.05, distance=max(1, int(0.05 / max(1e-6, (lags[1]-lags[0])))))
                for p in pk:
                    if p == 0: 
                        continue
                    T = lags[p]
                    if T > 0:
                        cand.setdefault(round(T, 4), {"sources": set(), "score": 0.0})
                        cand[round(T,4)]["sources"].add("ACF")
            # From env FFT: take top K peaks in power
            if P_env.size > 4:
                # simple peak detect on spectrum
                spk, _ = find_peaks(P_env, distance=max(1, int(len(P_env)/50)))
                top = spk[np.argsort(P_env[spk])[-5:]] if spk.size else np.array([], dtype=int)
                for idx in top:
                    f0 = f_env[idx]
                    if f0 > 0:
                        T = 1.0 / f0
                        cand.setdefault(round(T, 4), {"sources": set(), "score": 0.0})
                        cand[round(T,4)]["sources"].add("FFT-env")
            # From IEI histogram: mode (and maybe 2×mode if harmonic)
            if iei.size >= 2:
                hist, edges = np.histogram(iei, bins='auto')
                if hist.size:
                    i_max = int(np.argmax(hist))
                    Tmode = 0.5*(edges[i_max]+edges[i_max+1])
                    if Tmode > 0:
                        for T in (Tmode, 2*Tmode/1.0):  # include harmonic
                            cand.setdefault(round(T, 4), {"sources": set(), "score": 0.0})
                            cand[round(T,4)]["sources"].add("IEI")
            return cand

        # ---- main compute -----------------------------------------------------
        def do_compute():
            nonlocal results, event_times, top_period, R_for_top
            results = []; event_times = None; top_period = None; R_for_top = None

            # read params
            try:
                thr     = float(thr_edit.text())
                min_gap = float(gap_edit.text())
                min_len = float(evlen_edit.text())
                dt      = max(1e-4, float(acf_dt_edit.text()))
                win_s   = max(0.1, float(acf_win_edit.text()))
                pmin    = max(1e-3, float(pmin_edit.text()))
                pmax    = max(pmin+1e-3, float(pmax_edit.text()))
                nfft    = int(nfft_cb.currentData())
            except ValueError as e:
                QtWidgets.QMessageBox.critical(dlg, "Error", str(e)); return

            # 1) Filter → envelope
            y = _bandpass(x)
            env = np.abs(hilbert(y))

            # 2) Detect events
            etimes, _ = _detect_events(env, thr, min_gap, min_len)
            event_times = etimes

            # 3) IEI
            if etimes.size >= 2:
                iei = np.diff(etimes)
            else:
                iei = np.array([], dtype=float)

            # 4) ACF on event-train
            dur = len(x) / fs
            t_train, s_train = _event_train(etimes, dt, dur)
            lags, ac = _acf(s_train, max_lag_s=min(win_s, dur), dt=dt)

            # 5) Envelope periodogram (limit to period window)
            f_env, P_env = _envelope_periodogram(env, nfft, pmin, pmax)

            # 6) Candidate periods & Rayleigh score
            cand = _extract_candidates(lags, ac, f_env, P_env, iei)
            # score each by Rayleigh R
            scored = []
            for T, meta in cand.items():
                R = _rayleigh_R(etimes, float(T))
                scored.append((float(T), 1.0/float(T), R, "+".join(sorted(meta["sources"])), etimes.size))
            # sort by score desc, then by how many cycles exist (favor supported periods)
            scored.sort(key=lambda z: (z[2], z[4]), reverse=True)

            # 7) Populate table
            table.setRowCount(len(scored))
            for i,(T, Hz, Rv, src, ne) in enumerate(scored, start=1):
                def _it(txt):
                    it = QtWidgets.QTableWidgetItem(txt); it.setForeground(QtCore.Qt.white); return it
                table.setItem(i-1, 0, _it(str(i)))
                table.setItem(i-1, 1, _it(f"{T:.6f}"))
                table.setItem(i-1, 2, _it(f"{Hz:.6f}"))
                table.setItem(i-1, 3, _it(f"{Rv:.3f}"))
                table.setItem(i-1, 4, _it(src))
                table.setItem(i-1, 5, _it(str(ne)))
            table.resizeColumnsToContents()

            # 8) Plots
            # Envelope + threshold + marks
            ax_env.clear(); ax_env.set_facecolor("#19232D"); ax_env.tick_params(colors="white")
            for s in ax_env.spines.values(): s.set_color("white")
            t = np.arange(len(env))/fs
            ax_env.plot(t, env, lw=0.8, color=getattr(self, "graph_color", "#33C3F0"))
            ax_env.axhline(thr, color="#FF5964", ls="--", lw=1.0)
            if etimes.size:
                ax_env.plot(etimes, np.interp(etimes, t, env), 'o', ms=4, color="#6EEB83")
            ax_env.set_title("Envelope & detected events", color="white")
            ax_env.set_xlabel("Time (s)", color="white"); ax_env.set_ylabel("Amplitude", color="white")

            # ACF
            ax_acf.clear(); ax_acf.set_facecolor("#19232D"); ax_acf.tick_params(colors="white")
            for s in ax_acf.spines.values(): s.set_color("white")
            ax_acf.plot(lags, ac, lw=1.4, color="#FFD166")
            ax_acf.set_title("Event-train autocorrelation", color="white")
            ax_acf.set_xlabel("Lag (s)", color="white"); ax_acf.set_ylabel("ACF (norm)", color="white")

            # IEI histogram
            ax_iei.clear(); ax_iei.set_facecolor("#19232D"); ax_iei.tick_params(colors="white")
            for s in ax_iei.spines.values(): s.set_color("white")
            if iei.size:
                bins = max(10, min(60, int(np.sqrt(iei.size))))
                ax_iei.hist(iei, bins=bins, color="#6EEB83", alpha=0.8)
            ax_iei.set_title("Inter-event intervals", color="white")
            ax_iei.set_xlabel("Interval (s)", color="white"); ax_iei.set_ylabel("Count", color="white")

            # Envelope periodogram (freq on top axis + period ticks below)
            ax_fft.clear(); ax_fft.set_facecolor("#19232D"); ax_fft.tick_params(colors="white")
            for s in ax_fft.spines.values(): s.set_color("white")
            if f_env.size:
                ax_fft.plot(f_env, 10*np.log10(np.maximum(P_env, 1e-24)), lw=1.4, color="#C792EA")
                ax_fft.set_xlabel("Recurrence frequency (Hz)", color="white")
                ax_fft.set_ylabel("Power (dB)", color="white")
                # secondary x-axis showing period seconds (few ticks)
                def f2p(x): return 1.0/np.maximum(x, 1e-9)
                ticks = np.linspace(f_env.min(), f_env.max(), 6)
                ax2 = ax_fft.secondary_xaxis('top', functions=(lambda f: f2p(f), lambda p: 1.0/np.maximum(p,1e-9)))
                ax2.set_xlabel("Period (s)", color="white"); ax2.tick_params(colors="white")
            ax_fft.set_title("Envelope periodogram", color="white")

            can.draw()

            # remember best
            if scored:
                top_period = float(scored[0][0])
                R_for_top  = float(scored[0][2])
                QtWidgets.QMessageBox.information(dlg, "Periodicity",
                    f"Top period ≈ {top_period:.6f} s (R = {R_for_top:.3f})\n"
                    f"Events detected: {etimes.size}"
                )
            else:
                QtWidgets.QMessageBox.information(dlg, "Periodicity", "No reliable periodicity found.")

        # ---- actions ----------------------------------------------------------
        def on_save_img():
            if fig is None:
                return
            p,_ = QtWidgets.QFileDialog.getSaveFileName(dlg, "Save Plot", "", "PNG (*.png);;JPEG (*.jpg)")
            if not p: return
            fig.savefig(p, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")

        def on_save_csv():
            if table.rowCount() == 0:
                QtWidgets.QMessageBox.information(dlg, "Nothing to export", "Run analysis first.")
                return
            p,_ = QtWidgets.QFileDialog.getSaveFileName(dlg, "Export CSV", "", "CSV (*.csv)")
            if not p: return
            with open(p, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["rank","period_s","freq_Hz","score_R","sources","num_events"])
                for r in range(table.rowCount()):
                    w.writerow([
                        table.item(r,0).text(),
                        table.item(r,1).text(),
                        table.item(r,2).text(),
                        table.item(r,3).text(),
                        table.item(r,4).text(),
                        table.item(r,5).text()
                    ])

        def on_log_db():
            if DB_FILENAME is None:
                QtWidgets.QMessageBox.information(dlg, "DB", "DB not available in this context.")
                return
            if top_period is None:
                QtWidgets.QMessageBox.information(dlg, "DB", "No top period to log. Analyze first.")
                return
            try:
                conn = sqlite3.connect(DB_FILENAME); cur = conn.cursor()
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS periodicity_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_name TEXT,
                        top_period_s REAL,
                        rayleigh_R REAL,
                        num_events INTEGER,
                        pmin REAL, pmax REAL,
                        threshold REAL,
                        min_gap REAL,
                        band_low REAL,
                        band_high REAL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                band_lo = float(fmin_edit.text()) if fmin_edit.text().strip() else None
                band_hi = float(fmax_edit.text()) if fmax_edit.text().strip() else None
                cur.execute("""
                    INSERT INTO periodicity_results
                    (file_name, top_period_s, rayleigh_R, num_events, pmin, pmax, threshold, min_gap, band_low, band_high)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                """, (
                    self.file_name, float(top_period), float(R_for_top), int(event_times.size if event_times is not None else 0),
                    float(pmin_edit.text()), float(pmax_edit.text()),
                    float(thr_edit.text()), float(gap_edit.text()),
                    band_lo, band_hi
                ))
                conn.commit(); conn.close()
                QtWidgets.QMessageBox.information(dlg, "Logged", "Top periodicity saved to database.")
            except Exception as e:
                QtWidgets.QMessageBox.critical(dlg, "DB Error", str(e))

        compute_btn.clicked.connect(do_compute)
        save_img.clicked.connect(on_save_img)
        save_csv.clicked.connect(on_save_csv)
        log_btn.clicked.connect(on_log_db)
        close_btn.clicked.connect(dlg.accept)

        dlg.exec_()


    def exceedance_curves_popup(self):
        """
        Exceedance (Lx) Curves:
        - Source = DB (spl_calculations) or WAV (windowed VRMS)
        - WAV mode UI appears only when selected; prompts to load WAV if needed
        - Optional band-pass (WAV mode)
        - Weighted by duration (DB: window_length; WAV: hop seconds)
        - Plot exceedance CDF and mark L10/L50/L90
        - User limits (dB): show % time and total exceeded time
        - Export CSV/PNG; optional DB log of summary
        """
        from PyQt5 import QtWidgets, QtCore
        import numpy as np, sqlite3, csv, os
        from scipy.signal import butter, sosfiltfilt, get_window
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Exceedance Curves (Lx)")
        dlg.setWindowState(dlg.windowState() | QtCore.Qt.WindowMaximized)
        dlg.setStyleSheet("background:#19232D; color:white;")
        vbox = QtWidgets.QVBoxLayout(dlg)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(6)

        # ── Top filter bar (compact) ───────────────────────────────────────────
        bar = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(bar)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)

        # Source
        src_lbl = QtWidgets.QLabel("Source:")
        db_rb = QtWidgets.QRadioButton("DB")
        wav_rb = QtWidgets.QRadioButton("WAV")
        db_rb.setChecked(True)
        src_box = QtWidgets.QHBoxLayout(); src_box.setSpacing(8)
        src_w = QtWidgets.QWidget(); src_w.setLayout(src_box)
        src_box.addWidget(db_rb); src_box.addWidget(wav_rb)
        grid.addWidget(src_lbl, 0, 0)
        grid.addWidget(src_w, 0, 1)

        # DB controls (same row)
        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        try:
            cur.execute("SELECT DISTINCT file_name FROM spl_calculations ORDER BY file_name")
            files = [r[0] for r in cur.fetchall()]
        except sqlite3.Error:
            files = []
        db_file_lbl = QtWidgets.QLabel("File:")
        db_file_cb = QtWidgets.QComboBox(); db_file_cb.addItems(files); db_file_cb.setMinimumWidth(200)

        db_curve_lbl = QtWidgets.QLabel("Hydrophone Curve:")
        db_curve_cb = QtWidgets.QComboBox(); db_curve_cb.addItem(""); db_curve_cb.setMinimumWidth(180)

        def upd_methods():
            db_curve_cb.blockSignals(True)
            db_curve_cb.clear(); db_curve_cb.addItem("")
            fn = db_file_cb.currentText()
            if fn:
                try:
                    cur.execute("""SELECT DISTINCT hydrophone_curve
                                FROM spl_calculations
                                WHERE file_name=? ORDER BY hydrophone_curve""", (fn,))
                    rows = [r[0] or "" for r in cur.fetchall()]
                except sqlite3.Error:
                    rows = []
                for r in rows:
                    if r and db_curve_cb.findText(r) < 0:
                        db_curve_cb.addItem(r)
            db_curve_cb.blockSignals(False)
        upd_methods()
        db_file_cb.currentTextChanged.connect(upd_methods)

        grid.addWidget(db_file_lbl, 0, 2)
        grid.addWidget(db_file_cb, 0, 3)
        grid.addWidget(db_curve_lbl, 0, 4)
        grid.addWidget(db_curve_cb, 0, 5)

        # Frequency range (DB mode)
        db_fr_lbl = QtWidgets.QLabel("Freq Range (Hz):")
        db_fmin = QtWidgets.QLineEdit("")
        db_fmax = QtWidgets.QLineEdit("")
        for w in (db_fmin, db_fmax):
            w.setFixedWidth(90)
            w.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

        db_fr = QtWidgets.QHBoxLayout()
        db_fr.setContentsMargins(0, 0, 0, 0)
        db_fr.setSpacing(6)

        db_frw = QtWidgets.QWidget()
        db_frw.setLayout(db_fr)
        db_frw.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

        db_fr.addWidget(db_fmin)
        db_fr.addWidget(QtWidgets.QLabel("to"))
        db_fr.addWidget(db_fmax)

        grid.addWidget(db_fr_lbl, 0, 6)
        grid.addWidget(db_frw, 0, 7, alignment=QtCore.Qt.AlignLeft)

        # Limits + Weighting (same line)
        lim_lbl = QtWidgets.QLabel("Limits (dB):")
        limit_edit = QtWidgets.QLineEdit("120"); limit_edit.setPlaceholderText("e.g., 120, 160"); limit_edit.setFixedWidth(180)
        weight_cb = QtWidgets.QCheckBox("Weight by duration"); weight_cb.setChecked(True)
        grid.addWidget(lim_lbl, 0, 8)
        grid.addWidget(limit_edit, 0, 9)
        grid.addWidget(weight_cb, 0, 10)

        # ── WAV controls (row 2) — hidden in DB mode ───────────────────────────
        wav_file_lbl = QtWidgets.QLabel("WAV File: <none>"); wav_file_lbl.setStyleSheet("color:#bbb;")
        wav_pick_btn = QtWidgets.QPushButton("Change…"); wav_pick_btn.setStyleSheet("background:#3E6C8A; color:white; padding:4px 8px; border-radius:4px;")

        win_lbl = QtWidgets.QLabel("Window (s):")
        win_edit = QtWidgets.QLineEdit("1.0"); win_edit.setFixedWidth(80)
        hop_lbl = QtWidgets.QLabel("Hop (s):")
        hop_edit = QtWidgets.QLineEdit("0.5"); hop_edit.setFixedWidth(80)

        mode_lbl = QtWidgets.QLabel("Mode:")
        mode_cb = QtWidgets.QComboBox(); mode_cb.addItems(["Full-band VRMS", "Band-limited (Butter)"])

        band_lbl = QtWidgets.QLabel("Band (Hz):")
        bl_f1 = QtWidgets.QLineEdit("20"); bl_f1.setFixedWidth(90)
        bl_f2 = QtWidgets.QLineEdit("20000"); bl_f2.setFixedWidth(90)
        band_row = QtWidgets.QHBoxLayout(); band_row.setSpacing(6)
        band_w = QtWidgets.QWidget(); band_w.setLayout(band_row)
        band_row.addWidget(bl_f1); band_row.addWidget(QtWidgets.QLabel("to")); band_row.addWidget(bl_f2)

        # place in grid row 1 (second row visually)
        grid.addWidget(wav_file_lbl, 1, 0, 1, 2)
        grid.addWidget(wav_pick_btn, 1, 2)
        grid.addWidget(win_lbl, 1, 3); grid.addWidget(win_edit, 1, 4)
        grid.addWidget(hop_lbl, 1, 5); grid.addWidget(hop_edit, 1, 6)
        grid.addWidget(mode_lbl, 1, 7); grid.addWidget(mode_cb, 1, 8)
        grid.addWidget(band_lbl, 1, 9); grid.addWidget(band_w, 1, 10)

        # Compute on far right
        compute_btn = QtWidgets.QPushButton("Compute")
        compute_btn.setStyleSheet("background:#3E6C8A; color:white; padding:6px 10px; border-radius:4px;")
        grid.addWidget(compute_btn, 1, 11, alignment=QtCore.Qt.AlignRight)

        vbox.addWidget(bar)

        db_widgets = [db_file_lbl, db_file_cb, db_curve_lbl, db_curve_cb, db_fr_lbl, db_frw]
        wav_widgets = [wav_file_lbl, wav_pick_btn, win_lbl, win_edit, hop_lbl, hop_edit, mode_lbl, mode_cb, band_lbl, band_w]

        def set_visible(widgets, vis: bool):
            for w in widgets:
                w.setVisible(vis)

        def ensure_wav_loaded():
            cur_name = getattr(self, "file_name", None)
            if self.full_data is not None and cur_name:
                wav_file_lbl.setText(f"WAV File: {cur_name}")
                return True
            path, _ = QtWidgets.QFileDialog.getOpenFileName(dlg, "Select WAV File", "", "WAV Files (*.wav)")
            if not path:
                QtWidgets.QMessageBox.information(dlg, "Cancelled", "No WAV selected; staying in DB mode.")
                db_rb.setChecked(True)
                return False
            try:
                if hasattr(self, "load_wav_file"):
                    self.load_wav_file(path)
                else:
                    from scipy.io import wavfile
                    fs, data = wavfile.read(path)
                    self.sample_rate = fs
                    self.full_data = data
                    self.original_dtype = data.dtype
                    self.current_file_path = path
                    self.file_name = os.path.basename(path)
            except Exception as e:
                QtWidgets.QMessageBox.critical(dlg, "Load Error", f"Failed to load WAV:\n{e}")
                db_rb.setChecked(True)
                return False
            wav_file_lbl.setText(f"WAV File: {self.file_name}")
            return True

        def apply_src_state():
            db_mode = db_rb.isChecked()
            set_visible(db_widgets, db_mode)
            set_visible(wav_widgets, not db_mode)
            if not db_mode:
                ensure_wav_loaded()

        db_rb.toggled.connect(apply_src_state)
        wav_rb.toggled.connect(apply_src_state)
        apply_src_state()

        wav_pick_btn.clicked.connect(lambda: ensure_wav_loaded())

        # ── Splitter: small table + large plot ─────────────────────────────────
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.setHandleWidth(6)

        table = QtWidgets.QTableWidget(4, 2)
        table.setHorizontalHeaderLabels(["Metric", "Value"])
        table.setVerticalHeaderLabels(["L10 (dB)", "L50 (dB)", "L90 (dB)", "Aggregate"])
        table.setStyleSheet(
            "QTableWidget{background:#19232D;color:white;gridline-color:#444;}"
            "QHeaderView::section{background:#19232D;color:white;border:none;}"
        )
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setMaximumWidth(300)
        split.addWidget(table)

        fig = Figure(facecolor="#19232D")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#19232D")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_color("white")
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        split.addWidget(canvas)

        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 7)
        vbox.addWidget(split, 1)

        # ── Bottom buttons ─────────────────────────────────────────────────────
        btns = QtWidgets.QHBoxLayout()
        export_csv_btn = QtWidgets.QPushButton("Export CSV")
        save_png_btn = QtWidgets.QPushButton("Save Plot")
        log_btn = QtWidgets.QPushButton("Log Summary")
        close_btn = QtWidgets.QPushButton("Close")
        for b in (export_csv_btn, save_png_btn, log_btn, close_btn):
            b.setStyleSheet("background:#3E6C8A; color:white; padding:6px 10px; border-radius:4px;")
        btns.addStretch(); btns.addWidget(export_csv_btn); btns.addWidget(save_png_btn)
        btns.addWidget(log_btn); btns.addWidget(close_btn)
        vbox.addLayout(btns)

        # ── Helpers ────────────────────────────────────────────────────────────
        def weighted_ecdf_sorted(x, w):
            x = np.asarray(x, float); w = np.asarray(w, float)
            m = np.isfinite(x) & np.isfinite(w) & (w >= 0)
            x = x[m]; w = w[m]
            if x.size == 0:
                return None
            order = np.argsort(x, kind="mergesort")
            xs = x[order]; ws = w[order]
            cw = np.cumsum(ws)
            tot = cw[-1]
            if tot <= 0:
                return None
            ecdf_w = cw / tot
            exceed = 100.0 * (1.0 - ecdf_w)
            return xs, exceed, ws, cw

        def weighted_quantile(xs, ws_sorted, cw_sorted, q):
            target = q * cw_sorted[-1]
            idx = int(np.searchsorted(cw_sorted, target, side="left"))
            idx = max(0, min(idx, len(xs) - 1))
            return float(xs[idx])

        def plot_exceedance(x_sorted, exceed_pct, Ls, units):
            ax.clear(); ax.set_facecolor("#19232D"); ax.tick_params(colors="white")
            for sp in ax.spines.values():
                sp.set_color("white")
            color = getattr(self, 'graph_color', '#33C3F0')
            ax.plot(x_sorted, exceed_pct, lw=2.2, color=color)
            L10, L50, L90 = Ls
            for val, lab, ypos in [(L10, "L10", 70), (L50, "L50", 50), (L90, "L90", 30)]:
                ax.axvline(val, color="gray", ls="--", lw=1)
                ax.text(val, ypos, f"{lab}={val:.1f} {units}", color="white",
                        rotation=90, va="center", ha="right", fontsize=9,
                        bbox=dict(facecolor="#00000066", edgecolor="none", pad=2))
            ax.set_xlabel(f"Level ({units})", color="white")
            ax.set_ylabel("Exceedance (%)", color="white")
            ax.grid(True, ls="--", alpha=0.35, color="gray")
            canvas.draw()

        def update_table(L10, L50, L90, aggregate_value, weighted=True):
            labels = ["L10 (dB)", "L50 (dB)", "L90 (dB)", "Total duration (s)" if weighted else "N windows"]
            values = [f"{L10:.2f}", f"{L50:.2f}", f"{L90:.2f}", f"{aggregate_value:.2f}" if weighted else str(int(aggregate_value))]
            for r, (lab, val) in enumerate(zip(labels, values)):
                table.setItem(r, 0, QtWidgets.QTableWidgetItem(lab))
                table.setItem(r, 1, QtWidgets.QTableWidgetItem(val))
            table.resizeColumnsToContents()

        cur_units = "dB"
        cur_points = None
        cur_summary = None
        raw_levels = None
        raw_weights = None

        # ── Compute ────────────────────────────────────────────────────────────
        def compute():
            nonlocal cur_units, cur_points, cur_summary, raw_levels, raw_weights
            try:
                if db_rb.isChecked():
                    q = ("SELECT target_frequency, spl, start_time, end_time, window_length "
                         "FROM spl_calculations")
                    where = []; args = []
                    fn = db_file_cb.currentText().strip()
                    hc = db_curve_cb.currentText().strip()
                    if fn:
                        where.append("file_name=?"); args.append(fn)
                    if hc:
                        where.append("hydrophone_curve=?"); args.append(hc)
                    if db_fmin.text().strip():
                        where.append("target_frequency>=?"); args.append(float(db_fmin.text()))
                    if db_fmax.text().strip():
                        where.append("target_frequency<=?"); args.append(float(db_fmax.text()))
                    if where:
                        q += " WHERE " + " AND ".join(where)
                    cur.execute(q, tuple(args))
                    rows = cur.fetchall()
                    if not rows:
                        QtWidgets.QMessageBox.information(dlg, "No Data", "No rows matched in spl_calculations.")
                        return
                    levels, weights = [], []
                    for _, spl, st, et, wl in rows:
                        if spl is None:
                            continue
                        levels.append(float(spl))
                        if weight_cb.isChecked():
                            if wl is not None and wl > 0:
                                weights.append(float(wl))
                            elif (st is not None) and (et is not None):
                                weights.append(float(et) - float(st))
                            else:
                                weights.append(1.0)
                        else:
                            weights.append(1.0)
                    raw_levels = np.array(levels, float)
                    raw_weights = np.array(weights, float)
                    cur_units = "dB re 1 µPa"

                else:
                    if not ensure_wav_loaded():
                        return
                    fs = float(self.sample_rate)
                    win_s = float(win_edit.text()); hop_s = float(hop_edit.text())
                    w = max(1, int(win_s * fs)); h = max(1, int(hop_s * fs))
                    x = self.full_data.astype(np.float64)
                    if x.ndim > 1:
                        x = x.mean(axis=1)
                    if mode_cb.currentText().startswith("Band"):
                        f1 = float(bl_f1.text()); f2 = float(bl_f2.text())
                        if not (0 < f1 < f2 < fs / 2):
                            QtWidgets.QMessageBox.warning(dlg, "Band Error", "Check band limits and Nyquist.")
                            return
                        sos = butter(6, [f1 / (fs / 2), f2 / (fs / 2)], btype="band", output="sos")
                        x = sosfiltfilt(sos, x, axis=0)
                    N = len(x)
                    vals = []
                    win = get_window("hann", w, fftbins=True)
                    norm = np.sqrt((win ** 2).mean())
                    for start in range(0, N - w + 1, h):
                        seg = x[start:start + w]
                        vrms = np.sqrt(np.mean((seg * win) ** 2)) / (norm if norm > 0 else 1.0)
                        vals.append(20 * np.log10(max(vrms, 1e-12)))
                    if not vals:
                        QtWidgets.QMessageBox.information(dlg, "No Windows", "Increase duration or reduce window.")
                        return
                    raw_levels = np.array(vals, float)
                    raw_weights = (np.full_like(raw_levels, hop_s) if weight_cb.isChecked()
                                   else np.ones_like(raw_levels))
                    cur_units = "dB (relative)"

                ecdf = weighted_ecdf_sorted(raw_levels, raw_weights)
                if ecdf is None:
                    QtWidgets.QMessageBox.information(dlg, "No Data", "No usable samples.")
                    return
                x_sorted, exceed_pct, ws_sorted, cw_sorted = ecdf

                L10 = weighted_quantile(x_sorted, ws_sorted, cw_sorted, q=0.90)
                L50 = weighted_quantile(x_sorted, ws_sorted, cw_sorted, q=0.50)
                L90 = weighted_quantile(x_sorted, ws_sorted, cw_sorted, q=0.10)
                aggregate = float(np.sum(raw_weights)) if weight_cb.isChecked() else int(raw_levels.size)

                plot_exceedance(x_sorted, exceed_pct, (L10, L50, L90), cur_units)
                update_table(L10, L50, L90, aggregate, weighted=weight_cb.isChecked())

                limits_txt = limit_edit.text().strip()
                if limits_txt:
                    try:
                        limits = [float(ss) for ss in limits_txt.split(",") if ss.strip()]
                    except Exception:
                        limits = []
                    for L in limits:
                        ax.axvline(L, color="#AAAAAA", ls=":", lw=1)
                        ex_at_L = float(np.interp(L, x_sorted, exceed_pct))
                        if weight_cb.isChecked():
                            exceeded_time = float(np.sum(raw_weights[raw_levels > L]))
                            lbl = f">{L:.1f} dB for {ex_at_L:.1f}% (~{exceeded_time:.1f} s)"
                        else:
                            exceeded_n = int(np.sum(raw_levels > L))
                            lbl = f">{L:.1f} dB for {ex_at_L:.1f}% ({exceeded_n} windows)"
                        ax.text(L, 5, lbl, color="white", rotation=90, va="bottom", ha="right",
                                fontsize=9, bbox=dict(facecolor="#00000066", edgecolor="none", pad=2))
                    canvas.draw()

                cur_points = (x_sorted, exceed_pct)
                cur_summary = (L10, L50, L90, aggregate)

            except Exception as e:
                QtWidgets.QMessageBox.critical(dlg, "Error", str(e))

        def export_csv():
            if cur_points is None:
                QtWidgets.QMessageBox.information(dlg, "Nothing to Export", "Compute first.")
                return
            path, _ = QtWidgets.QFileDialog.getSaveFileName(dlg, "Save Exceedance CSV", "", "CSV Files (*.csv)")
            if not path:
                return
            xvals, ex = cur_points
            with open(path, "w", newline="") as fh:
                wr = csv.writer(fh); wr.writerow(["Level_dB", "Exceedance_%"])
                for xi, ei in zip(xvals, ex):
                    wr.writerow([xi, ei])
            QtWidgets.QMessageBox.information(dlg, "Saved", f"CSV saved:\n{path}")
        def save_png():
            if cur_points is None:
                QtWidgets.QMessageBox.information(dlg, "Nothing to Save", "Compute first.")
                return
            path, _ = QtWidgets.QFileDialog.getSaveFileName(dlg, "Save Plot", "", "PNG Files (*.png);;JPEG Files (*.jpg)")
            if not path:
                return
            fig.savefig(path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
            QtWidgets.QMessageBox.information(dlg, "Saved", f"Plot saved:\n{path}")
        def log_summary():
            if cur_summary is None:
                QtWidgets.QMessageBox.information(dlg, "Nothing to Log", "Compute first.")
                return
            L10, L50, L90, aggregate = cur_summary
            try:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS exceedance_summaries(
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source TEXT,
                        file_name TEXT,
                        band_low REAL,
                        band_high REAL,
                        unit TEXT,
                        L10 REAL, L50 REAL, L90 REAL,
                        aggregate REAL,
                        weighted INTEGER,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                if db_rb.isChecked():
                    src = "DB"
                    fn = db_file_cb.currentText() or ""
                    b1 = float(db_fmin.text()) if db_fmin.text().strip() else None
                    b2 = float(db_fmax.text()) if db_fmax.text().strip() else None
                else:
                    src = "WAV"
                    fn = getattr(self, "file_name", "")
                    if mode_cb.currentText().startswith("Band"):
                        b1 = float(bl_f1.text()); b2 = float(bl_f2.text())
                    else:
                        b1 = b2 = None
                cur.execute(
                    "INSERT INTO exceedance_summaries(source,file_name,band_low,band_high,unit,L10,L50,L90,aggregate,weighted) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (src, fn, b1, b2, cur_units, float(L10), float(L50), float(L90),
                     float(aggregate), int(weight_cb.isChecked()))
                )
                conn.commit()
                QtWidgets.QMessageBox.information(dlg, "Logged", "Exceedance summary saved.")
            except Exception as e:
                QtWidgets.QMessageBox.critical(dlg, "DB Error", str(e))

        compute_btn.clicked.connect(compute)
        export_csv_btn.clicked.connect(export_csv)
        save_png_btn.clicked.connect(save_png)
        log_btn.clicked.connect(log_summary)
        close_btn.clicked.connect(dlg.accept)

        dlg.exec_()
        conn.close()



    def ltsa_psd_popup(self):
        """
        LTSA (time–frequency heatmap) + PSD Percentile Spectra.
        - Source: current loaded WAV (self.full_data / self.sample_rate)
        - LTSA uses scipy.signal.spectrogram with scaling='density' (V^2/Hz)
        - Percentiles computed on linear PSD, then converted to dB
        - Dark theme, maximized, export PNG/CSV
        """
        from PyQt5 import QtWidgets, QtCore
        import numpy as np
        from scipy import signal
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

        if getattr(self, "full_data", None) is None:
            QtWidgets.QMessageBox.warning(self, "No file", "Load a WAV file first.")
            return

        fs = float(self.sample_rate)
        x  = self.full_data.astype(np.float64)
        if x.ndim > 1:  # stereo -> mono
            x = x.mean(axis=1)

        # ---- Dialog shell -----------------------------------------------------
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("LTSA + PSD Percentiles")
        dlg.setWindowState(dlg.windowState() | QtCore.Qt.WindowMaximized)
        dlg.setStyleSheet("background:#19232D; color:white;")
        v = QtWidgets.QVBoxLayout(dlg)
        v.setContentsMargins(10,10,10,10)
        v.setSpacing(8)

        tabs = QtWidgets.QTabWidget()
        v.addWidget(tabs)

        # ======================================================================
        # TAB 1: LTSA
        # ======================================================================
        t1 = QtWidgets.QWidget(); tabs.addTab(t1, "LTSA")
        t1v = QtWidgets.QVBoxLayout(t1)

        # Controls (compact top bar)
        ctl1 = QtWidgets.QHBoxLayout(); t1v.addLayout(ctl1)
        def _numedit(txt, w=80):
            e = QtWidgets.QLineEdit(txt); e.setFixedWidth(w); return e

        ctl1.addWidget(QtWidgets.QLabel("Win (s):"))
        ltsa_win = _numedit("2.0")
        ctl1.addWidget(ltsa_win)

        ctl1.addWidget(QtWidgets.QLabel("Overlap (%):"))
        ltsa_ovp = _numedit("50")
        ctl1.addWidget(ltsa_ovp)

        ctl1.addWidget(QtWidgets.QLabel("NFFT:"))
        ltsa_nfft = QtWidgets.QComboBox()
        for n in (512, 1024, 2048, 4096, 8192, 16384, 32768):
            ltsa_nfft.addItem(str(n), n)
        ltsa_nfft.setCurrentIndex(3)  # 4096
        ltsa_nfft.setFixedWidth(90)
        ctl1.addWidget(ltsa_nfft)

        ctl1.addWidget(QtWidgets.QLabel("Freq (Hz):"))
        ltsa_fmin = _numedit("0", 90)
        ctl1.addWidget(ltsa_fmin)
        ctl1.addWidget(QtWidgets.QLabel("to"))
        ltsa_fmax = _numedit(str(int(fs/2)), 90)
        ctl1.addWidget(ltsa_fmax)

        ctl1.addWidget(QtWidgets.QLabel("Colormap:"))
        ltsa_cmap = QtWidgets.QComboBox()
        ltsa_cmap.addItems(["inferno", "magma", "plasma", "viridis", "cividis"])
        ctl1.addWidget(ltsa_cmap)

        ltsa_btn = QtWidgets.QPushButton("Compute LTSA")
        ltsa_btn.setStyleSheet("background:#3E6C8A;color:white;padding:6px 10px;border-radius:4px;")
        ctl1.addWidget(ltsa_btn)
        ctl1.addStretch()

        # Plot
        fig1 = Figure(facecolor="#19232D")
        ax1  = fig1.add_subplot(111)
        ax1.set_facecolor("#19232D"); ax1.tick_params(colors="white")
        for s in ax1.spines.values(): s.set_color("white")
        can1 = FigureCanvas(fig1)
        t1v.addWidget(can1, 1)

        # Bottom (exports)
        b1 = QtWidgets.QHBoxLayout(); t1v.addLayout(b1)
        save_img1 = QtWidgets.QPushButton("Save Image")
        save_img1.setStyleSheet("background:#3E6C8A;color:white;padding:6px 10px;border-radius:4px;")
        save_csv1 = QtWidgets.QPushButton("Export CSV (matrix)")
        save_csv1.setStyleSheet("background:#3E6C8A;color:white;padding:6px 10px;border-radius:4px;")
        b1.addStretch(); b1.addWidget(save_img1); b1.addWidget(save_csv1)

        # Storage for reuse
        _ltsa_F = None; _ltsa_T = None; _ltsa_S = None

        def do_ltsa():
            nonlocal _ltsa_F, _ltsa_T, _ltsa_S
            try:
                win_s   = float(ltsa_win.text())
                ovp_pct = float(ltsa_ovp.text())
                nfft_ui = int(ltsa_nfft.currentData())
                fmin    = float(ltsa_fmin.text() or 0.0)
                fmax    = float(ltsa_fmax.text() or fs/2)
            except ValueError as e:
                QtWidgets.QMessageBox.critical(dlg, "Param error", str(e)); return

            if win_s <= 0:
                QtWidgets.QMessageBox.warning(dlg, "Param", "Window must be > 0")
                return

            # samples per window
            nperseg = max(8, int(round(win_s * fs)))

            # ensure NFFT >= nperseg
            if nfft_ui < nperseg:
                nfft = 1 << int(np.ceil(np.log2(nperseg)))  # next power of 2
                # reflect in UI (optional but nice)
                idx = ltsa_nfft.findText(str(nfft))
                if idx < 0:
                    ltsa_nfft.addItem(str(nfft), nfft)
                    idx = ltsa_nfft.findText(str(nfft))
                ltsa_nfft.setCurrentIndex(idx)
            else:
                nfft = nfft_ui

            # ensure 0 <= noverlap < nperseg
            noverlap = int(np.clip(ovp_pct, 0, 95) / 100.0 * nperseg)
            if noverlap >= nperseg:
                noverlap = nperseg - 1

            try:
                F, T, Sxx = signal.spectrogram(
                    x, fs=fs, window="hann",
                    nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                    detrend=False, scaling="density", mode="psd"
                )  # Sxx: V^2/Hz
            except Exception as e:
                QtWidgets.QMessageBox.critical(dlg, "Spectrogram error", str(e)); return

            # crop frequency
            m = (F >= fmin) & (F <= fmax)
            F2, S2 = F[m], Sxx[m, :]
            S2_dB  = 10.0 * np.log10(np.maximum(S2, 1e-24))

            # draw
            ax1.clear(); ax1.set_facecolor("#19232D"); ax1.tick_params(colors="white")
            for s in ax1.spines.values(): s.set_color("white")
            im = ax1.pcolormesh(T, F2, S2_dB, shading="auto", cmap=ltsa_cmap.currentText())
            ax1.set_xlabel("Time (s)", color="white")
            ax1.set_ylabel("Frequency (Hz)", color="white")
            ax1.grid(True, ls="--", alpha=0.35, color="gray")

            # colorbar (right margin)
            for cax in fig1.axes[1:]:
                fig1.delaxes(cax)
            cax = fig1.add_axes([0.92, 0.12, 0.02, 0.76])
            cb  = fig1.colorbar(im, cax=cax)
            cb.ax.yaxis.set_tick_params(color="white", labelcolor="white")
            cb.outline.set_edgecolor("white")
            cb.set_label("PSD (dB re V²/Hz)", color="white")

            can1.draw()

            # keep for exports / percentiles reuse
            _ltsa_F, _ltsa_T, _ltsa_S = F2, T, S2

        def export_ltsa_img():
            if _ltsa_S is None:
                QtWidgets.QMessageBox.information(dlg, "Nothing to save", "Compute LTSA first."); return
            path,_ = QtWidgets.QFileDialog.getSaveFileName(dlg, "Save LTSA", "", "PNG (*.png);;JPEG (*.jpg)")
            if not path: return
            fig1.savefig(path, dpi=220, facecolor=fig1.get_facecolor(), bbox_inches="tight")

        def export_ltsa_csv():
            if _ltsa_S is None:
                QtWidgets.QMessageBox.information(dlg, "Nothing to export", "Compute LTSA first."); return
            path,_ = QtWidgets.QFileDialog.getSaveFileName(dlg, "Export LTSA CSV", "", "CSV (*.csv)")
            if not path: return
            import csv
            with open(path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["Freq_Hz\\Time_s"] + [f"{t:.6f}" for t in _ltsa_T])
                for f, row in zip(_ltsa_F, _ltsa_S):
                    # export linear PSD; spreadsheet can dB it
                    w.writerow([f"{f:.3f}"] + [f"{val:.6e}" for val in row])

        ltsa_btn.clicked.connect(do_ltsa)
        save_img1.clicked.connect(export_ltsa_img)
        save_csv1.clicked.connect(export_ltsa_csv)

        # ======================================================================
        # TAB 2: PSD Percentile Spectra
        # ======================================================================
        t2 = QtWidgets.QWidget(); tabs.addTab(t2, "PSD Percentiles")
        t2v = QtWidgets.QVBoxLayout(t2)

        ctl2 = QtWidgets.QHBoxLayout(); t2v.addLayout(ctl2)
        ctl2.addWidget(QtWidgets.QLabel("Use LTSA settings (recommended). Percentiles:"))
        perc_edit = QtWidgets.QLineEdit("5,50,95")
        perc_edit.setFixedWidth(100)
        ctl2.addWidget(perc_edit)

        ctl2.addWidget(QtWidgets.QLabel("Smooth bins:"))
        smooth_spin = QtWidgets.QSpinBox(); smooth_spin.setRange(1, 51); smooth_spin.setValue(1)
        ctl2.addWidget(smooth_spin)

        clr_btn = QtWidgets.QPushButton("Compute Percentiles")
        clr_btn.setStyleSheet("background:#3E6C8A;color:white;padding:6px 10px;border-radius:4px;")
        ctl2.addWidget(clr_btn)
        ctl2.addStretch()

        fig2 = Figure(facecolor="#19232D")
        ax2  = fig2.add_subplot(111)
        ax2.set_facecolor("#19232D"); ax2.tick_params(colors="white")
        for s in ax2.spines.values(): s.set_color("white")
        can2 = FigureCanvas(fig2)
        t2v.addWidget(can2, 1)

        b2 = QtWidgets.QHBoxLayout(); t2v.addLayout(b2)
        save_img2 = QtWidgets.QPushButton("Save Image")
        save_img2.setStyleSheet("background:#3E6C8A;color:white;padding:6px 10px;border-radius:4px;")
        save_csv2 = QtWidgets.QPushButton("Export CSV (curves)")
        save_csv2.setStyleSheet("background:#3E6C8A;color:white;padding:6px 10px;border-radius:4px;")
        b2.addStretch(); b2.addWidget(save_img2); b2.addWidget(save_csv2)

        _pct_curves = None  # (F, {p: curve_dB})

        def _smooth(y, k):
            if k <= 1: return y
            k = int(k) | 1
            pad = k//2
            ypad = np.pad(y, (pad,pad), mode="edge")
            ker = np.ones(k)/k
            return np.convolve(ypad, ker, mode="valid")

        def do_percentiles():
            nonlocal _pct_curves
            if _ltsa_S is None:
                # compute LTSA once with current settings
                do_ltsa()
                if _ltsa_S is None:
                    return

            # percentiles (linear → dB)
            try:
                percs = [float(p.strip()) for p in perc_edit.text().split(",") if p.strip()]
                percs = [p for p in percs if 0 <= p <= 100]
                if not percs:
                    raise ValueError
            except Exception:
                QtWidgets.QMessageBox.warning(dlg, "Percents", "Enter comma-separated values in [0..100].")
                return

            F = _ltsa_F; S = _ltsa_S  # shape: (freq, time)
            eps = 1e-24
            curves = {}
            for p in percs:
                q_lin = np.percentile(S, p, axis=1)      # linear PSD percentile per frequency
                q_db  = 10*np.log10(np.maximum(q_lin, eps))
                q_db  = _smooth(q_db, smooth_spin.value())
                curves[p] = q_db

            # draw
            ax2.clear(); ax2.set_facecolor("#19232D"); ax2.tick_params(colors="white")
            for s in ax2.spines.values(): s.set_color("white")
            def _col(i):
                pals = ["#33C3F0","#6EEB83","#FF5964","#FFD166","#C792EA","#4DD0E1"]
                return pals[i % len(pals)]
            for i, p in enumerate(sorted(curves.keys())):
                ax2.plot(F, curves[p], lw=2, color=_col(i), label=f"P{int(p)}")
            ax2.set_xlabel("Frequency (Hz)", color="white")
            ax2.set_ylabel("PSD (dB re V²/Hz)", color="white")
            ax2.grid(True, ls="--", alpha=0.35, color="gray")
            ax2.legend(facecolor="#222", edgecolor="#444", labelcolor="white")
            can2.draw()

            _pct_curves = (F, curves)

        def export_pct_img():
            if _pct_curves is None:
                QtWidgets.QMessageBox.information(dlg, "Nothing to save", "Compute percentiles first."); return
            path,_ = QtWidgets.QFileDialog.getSaveFileName(dlg, "Save PSD Percentiles", "", "PNG (*.png);;JPEG (*.jpg)")
            if not path: return
            fig2.savefig(path, dpi=220, facecolor=fig2.get_facecolor(), bbox_inches="tight")

        def export_pct_csv():
            if _pct_curves is None:
                QtWidgets.QMessageBox.information(dlg, "Nothing to export", "Compute percentiles first."); return
            F, curves = _pct_curves
            path,_ = QtWidgets.QFileDialog.getSaveFileName(dlg, "Export Percentiles CSV", "", "CSV (*.csv)")
            if not path: return
            import csv
            with open(path, "w", newline="") as fh:
                w = csv.writer(fh)
                hdr = ["Freq_Hz"] + [f"P{int(p)}_dB" for p in sorted(curves.keys())]
                w.writerow(hdr)
                rows = zip(F, *[curves[p] for p in sorted(curves.keys())])
                for r in rows:
                    w.writerow([f"{r[0]:.3f}"] + [f"{val:.6f}" for val in r[1:]])

        clr_btn.clicked.connect(do_percentiles)
        save_img2.clicked.connect(export_pct_img)
        save_csv2.clicked.connect(export_pct_csv)

        # ---- Show dialog
        dlg.exec_()




    def generate_hydrophone_calibration_popup(self):
        import sqlite3, csv, json
        import numpy as np
        from PyQt5 import QtWidgets, QtCore
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        from scipy.signal import savgol_filter

        # Dialog setup
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Generate Hydrophone Calibration")
        dlg.setWindowState(dlg.windowState() | QtCore.Qt.WindowMaximized)
        dlg.setStyleSheet("background-color:#19232D;color:white;")
        layout = QtWidgets.QVBoxLayout(dlg)

        # DB init & buffer
        conn = sqlite3.connect(DB_FILENAME)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS hydrophone_curves (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                curve_name TEXT UNIQUE,
                file_name TEXT,
                min_frequency REAL,
                max_frequency REAL,
                sensitivity_json TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )""")
        conn.commit()
        temp_curve = []  # list of {'id','freq','sens'}

        # ── Top controls: File, Method, CSV, Distance, Generate, Order, Markers, LogScale, Smoothing ──
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("File:"))
        file_cb = QtWidgets.QComboBox(); top.addWidget(file_cb)
        top.addWidget(QtWidgets.QLabel("Method:"))
        method_cb = QtWidgets.QComboBox(); top.addWidget(method_cb)

        load_csv_btn = QtWidgets.QPushButton("Load SPL CSV…")
        load_csv_btn.setStyleSheet("background-color:#3E6C8A;color:white;")
        top.addWidget(load_csv_btn)
        csv_label = QtWidgets.QLabel("<no file>"); top.addWidget(csv_label)

        top.addWidget(QtWidgets.QLabel("Distance (m):"))
        distance_edit = QtWidgets.QLineEdit("1.0")
        distance_edit.setFixedWidth(80)
        distance_edit.setStyleSheet("color:white;")
        top.addWidget(distance_edit)

        generate_btn = QtWidgets.QPushButton("Generate Curve")
        generate_btn.setEnabled(False)
        generate_btn.setStyleSheet("background-color:#3E6C8A;color:white;")
        top.addWidget(generate_btn)

        top.addWidget(QtWidgets.QLabel("Order:"))
        order_cb = QtWidgets.QComboBox(); order_cb.addItems(["Ascending","Descending"])
        order_cb.setStyleSheet("color:white;")
        top.addWidget(order_cb)

        markers_cb = QtWidgets.QCheckBox("Show Markers")
        markers_cb.setChecked(False); markers_cb.setStyleSheet("color:white;")
        top.addWidget(markers_cb)

        log_cb = QtWidgets.QCheckBox("Log Scale (X)")
        log_cb.setChecked(False); log_cb.setStyleSheet("color:white;")
        top.addWidget(log_cb)

        # Inline smoothing controls
        smooth_cb = QtWidgets.QCheckBox("Enable Smoothing")
        smooth_cb.setStyleSheet("color:white;")
        top.addWidget(smooth_cb)
        top.addWidget(QtWidgets.QLabel("S-G Window:"))
        win_spin = QtWidgets.QSpinBox()
        win_spin.setRange(3, 51)
        win_spin.setSingleStep(2)
        win_spin.setValue(5)
        win_spin.setStyleSheet("color:white;")
        top.addWidget(win_spin)

        top.addStretch()
        layout.addLayout(top)

        # ── Level-shift controls ───────────────────────────────────────────
        shift_layout = QtWidgets.QHBoxLayout()
        shift_layout.addWidget(QtWidgets.QLabel("Pivot Frequency (Hz):"))
        pivot_edit = QtWidgets.QLineEdit(); pivot_edit.setFixedWidth(80)
        shift_layout.addWidget(pivot_edit)
        shift_layout.addWidget(QtWidgets.QLabel("Set Sensitivity at Pivot (dB):"))
        low_edit = QtWidgets.QLineEdit(); low_edit.setFixedWidth(80)
        shift_layout.addWidget(low_edit)
        shift_btn = QtWidgets.QPushButton("Apply Level Shift")
        shift_btn.setStyleSheet("background-color:#3E6C8A;color:white;")
        shift_layout.addWidget(shift_btn)
        shift_layout.addStretch()
        layout.addLayout(shift_layout)

        # ── Table + Plot ───────────────────────────────────────────────────
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        table = QtWidgets.QTableWidget(); table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Frequency (Hz)", "Sensitivity (dB)", "Delete"])
        table.setStyleSheet("background-color:#19232D;color:white;gridline-color:#444;")
        table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        table.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked |
                        QtWidgets.QAbstractItemView.SelectedClicked)
        splitter.addWidget(table)

        fig = Figure(facecolor="#19232D")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#19232D"); ax.tick_params(colors="white")
        canvas = FigureCanvas(fig)
        splitter.addWidget(canvas)
        layout.addWidget(splitter)

        # ── Bottom buttons ──────────────────────────────────────────────────
        btn_layout = QtWidgets.QHBoxLayout()
        save_changes_btn = QtWidgets.QPushButton("Save Changes")
        save_changes_btn.setStyleSheet("background-color:#3E6C8A;color:white;")
        btn_layout.addWidget(save_changes_btn)

        save_curve_btn = QtWidgets.QPushButton("Save Curve…")
        save_curve_btn.setStyleSheet("background-color:#3E6C8A;color:white;")
        btn_layout.addWidget(save_curve_btn)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.setStyleSheet("background-color:#3E6C8A;color:white;")
        btn_layout.addWidget(close_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # ── Populate File & Method (project-scoped when available) ────────
        project_id = getattr(self, "current_project_id", None)
        project_name = (getattr(self, "current_project_name", None) or "").strip()

        if project_id is not None:
            files = [
                r[0]
                for r in cur.execute(
                    "SELECT DISTINCT file_name FROM measurements WHERE project_id=? ORDER BY file_name",
                    (int(project_id),),
                )
            ]
        elif project_name:
            # Fallback for legacy rows without measurements.project_id
            files = [
                r[0]
                for r in cur.execute(
                    """
                    SELECT DISTINCT m.file_name
                    FROM measurements m
                    JOIN project_items pi
                      ON pi.file_name = m.file_name
                     AND pi.method = m.method
                    JOIN projects p
                      ON p.id = pi.project_id
                    WHERE p.name = ?
                    ORDER BY m.file_name
                    """,
                    (project_name,),
                )
            ]
        else:
            files = [
                r[0]
                for r in cur.execute(
                    "SELECT DISTINCT file_name FROM measurements ORDER BY file_name"
                )
            ]

        file_cb.addItems(files)

        def update_methods(fn):
            method_cb.clear()
            if not fn:
                generate_btn.setEnabled(False)
                return

            if project_id is not None:
                methods = [
                    m[0]
                    for m in cur.execute(
                        "SELECT DISTINCT method FROM measurements WHERE project_id=? AND file_name=? ORDER BY method",
                        (int(project_id), fn),
                    )
                ]
            elif project_name:
                # Fallback for legacy rows without measurements.project_id
                methods = [
                    m[0]
                    for m in cur.execute(
                        """
                        SELECT DISTINCT m.method
                        FROM measurements m
                        JOIN project_items pi
                          ON pi.file_name = m.file_name
                         AND pi.method = m.method
                        JOIN projects p
                          ON p.id = pi.project_id
                        WHERE p.name = ?
                          AND m.file_name = ?
                        ORDER BY m.method
                        """,
                        (project_name, fn),
                    )
                ]
            else:
                methods = [
                    m[0]
                    for m in cur.execute(
                        "SELECT DISTINCT method FROM measurements WHERE file_name=? ORDER BY id",
                        (fn,),
                    )
                ]

            method_cb.addItems(methods)
            generate_btn.setEnabled(False)

        file_cb.currentTextChanged.connect(update_methods)
        if files:
            update_methods(files[0])

        # ── Load SPL CSV ───────────────────────────────────────────────────
        spl_f, spl_v = np.array([]), np.array([])
        def load_csv():
            path,_ = QtWidgets.QFileDialog.getOpenFileName(
                dlg, "Open SPL CSV", "", "CSV Files (*.csv)")
            if not path: return
            csv_label.setText(path.split("/")[-1])
            f_list,v_list = [],[]
            with open(path,newline='') as fh:
                reader = csv.reader(fh)
                for row in reader:
                    try:
                        f_list.append(float(row[0]))
                        v_list.append(float(row[1]))
                    except:
                        pass
            nonlocal spl_f, spl_v
            spl_f, spl_v = np.array(f_list), np.array(v_list)
            generate_btn.setEnabled(True)
        load_csv_btn.clicked.connect(load_csv)

        # ── Update table & plot ────────────────────────────────────────────
        def update_table_and_plot():
            asc = (order_cb.currentText() == "Ascending")
            sorted_curve = sorted(temp_curve,
                                key=lambda e: e['freq'],
                                reverse=not asc)
            freqs = [e['freq'] for e in sorted_curve]
            sens_vals = [e['sens'] for e in sorted_curve]

            # apply smoothing if checked
            if smooth_cb.isChecked() and len(sens_vals) >= win_spin.value():
                wl = win_spin.value()
                sens_vals = savgol_filter(sens_vals, wl, polyorder=2, mode='interp').tolist()

            # plot
            ax.clear()
            ax.set_facecolor("#19232D"); ax.tick_params(colors="white")
            if log_cb.isChecked(): ax.set_xscale('log')
            marker = 'o' if markers_cb.isChecked() else ''
            ax.plot(freqs, sens_vals, linestyle='-', marker=marker, color=self.graph_color)
            ax.set_xlabel('Frequency (Hz)', color='white')
            ax.set_ylabel('Sensitivity (dB)', color='white')
            ax.grid(True, linestyle='--', alpha=0.5, color='gray')

            # table
            table.setRowCount(len(freqs))
            for i,(fq,sv) in enumerate(zip(freqs,sens_vals)):
                table.setItem(i,0, QtWidgets.QTableWidgetItem(f"{fq:.2f}"))
                table.setItem(i,1, QtWidgets.QTableWidgetItem(f"{sv:.3f}"))
                btn = QtWidgets.QPushButton("Delete")
                btn.setFixedWidth(60)
                btn.setStyleSheet("background-color:#A33;color:white;")
                btn.clicked.connect(lambda _, r=i: delete_row(r))
                table.setCellWidget(i,2,btn)

            canvas.draw()

        # ── Generate curve ──────────────────────────────────────────────────
        def do_generate():
            temp_curve.clear()
            try:
                distance = float(distance_edit.text())
            except:
                distance = 1.0
            for mid, freq, volt in cur.execute(
                "SELECT id, target_frequency, measured_voltage "
                "FROM measurements WHERE file_name=? AND method=? ORDER BY id",
                (file_cb.currentText(), method_cb.currentText())
            ):
                sens_val = 20*np.log10(volt) + 20*np.log10(distance) - np.interp(freq, spl_f, spl_v)
                temp_curve.append({'id': mid, 'freq': freq, 'sens': sens_val})
            update_table_and_plot()
        generate_btn.clicked.connect(do_generate)

        # ── Level shift ────────────────────────────────────────────────────
        def apply_shift():
            try:
                pivot = float(pivot_edit.text())
                target = float(low_edit.text())
            except:
                QtWidgets.QMessageBox.warning(dlg, 'Error', 'Invalid pivot or sensitivity')
                return
            idx = min(range(len(temp_curve)), key=lambda i: abs(temp_curve[i]['freq']-pivot))
            base = temp_curve[idx]
            shift_val = target - base['sens']
            for e in temp_curve:
                e['sens'] = (target if e['freq'] <= base['freq']
                            else e['sens'] + shift_val)
            if not any(abs(e['freq'] - 10.0) < 1e-6 for e in temp_curve):
                temp_curve.append({'id': -1, 'freq': 10.0, 'sens': target})
            update_table_and_plot()
        shift_btn.clicked.connect(apply_shift)

        # ── Delete row ──────────────────────────────────────────────────────
        def delete_row(row):
            temp_curve.pop(row)
            update_table_and_plot()

        # ── Save changes ────────────────────────────────────────────────────
        def save_changes():
            new_curve = []
            for i in range(table.rowCount()):
                try:
                    fq = float(table.item(i,0).text())
                    sv = float(table.item(i,1).text())
                    new_curve.append({'id': temp_curve[i]['id'], 'freq': fq, 'sens': sv})
                except:
                    continue
            temp_curve.clear()
            temp_curve.extend(new_curve)
            update_table_and_plot()
            QtWidgets.QMessageBox.information(dlg,'Updated','Changes applied.')
        save_changes_btn.clicked.connect(save_changes)

        # ── Save curve to DB & CSV ─────────────────────────────────────────
        def save_curve():
            name, ok = QtWidgets.QInputDialog.getText(dlg, 'Save Curve', 'Curve Name:')
            if not ok or not name.strip():
                return
            asc = (order_cb.currentText() == "Ascending")
            freqs = [e['freq'] for e in sorted(temp_curve, key=lambda e:e['freq'],
                                                reverse=not asc)]
            sens_vals = [e['sens'] for e in sorted(temp_curve, key=lambda e:e['freq'],
                                                reverse=not asc)]
            cur.execute(
                "INSERT OR REPLACE INTO hydrophone_curves("
                "curve_name,file_name,min_frequency,max_frequency,sensitivity_json"
                ") VALUES (?,?,?,?,?)",
                (name.strip(),
                file_cb.currentText(),
                min(freqs),
                max(freqs),
                json.dumps(sens_vals))
            )
            conn.commit()

            fn = f"{name.strip()}.csv"
            with open(fn, 'w', newline='') as fh:
                w = csv.writer(fh)
                w.writerow(['Frequency (Hz)', 'Sensitivity (dB)'])
                for fq, sv in zip(freqs, sens_vals):
                    w.writerow([fq, sv])
            QtWidgets.QMessageBox.information(dlg, 'Saved', f'Curve saved & exported to {fn}')
        save_curve_btn.clicked.connect(save_curve)

        # ── Close ────────────────────────────────────────────────────────────
        close_btn.clicked.connect(dlg.accept)

        dlg.exec_()
        conn.close()








    def hfm_pulse_analysis(self):
        """
        Hyperbolic FM (HFM) Pulse Analysis:
        - Optionally remove mains hum (50 Hz or 60 Hz notch)
        - Slide a window through the FFT‐panel region, compute instantaneous frequency via Hilbert,
          fit f(t) = a*(1/t) + b, and report a, b.
        - Logs 'a' in target_frequency and 'b' in measured_voltage.
        """
        if not self.fft_mode:
            QtWidgets.QMessageBox.information(
                self, "HFM Analysis",
                "Switch to FFT mode first and set your window & start time."
            )
            return

        # ─── 1) Settings dialog ─────────────────────────────────────────────
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("HFM Analysis Settings")
        form = QtWidgets.QFormLayout(dlg)

        # Remove mains noise
        noise_combo = QtWidgets.QComboBox()
        noise_combo.addItems(["None", "50 Hz Notch", "60 Hz Notch"])
        form.addRow("Remove Mains Noise:", noise_combo)

        # Discard zero‐Freq option
        drop_zero_cb = QtWidgets.QCheckBox("Discard 0 Hz results")
        drop_zero_cb.setChecked(True)
        form.addRow(drop_zero_cb)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        form.addRow(btns)

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        notch_choice = noise_combo.currentText()
        drop_zero    = drop_zero_cb.isChecked()

        # ─── 2) Prepare data ─────────────────────────────────────────────────
        data = self.full_data.copy()
        fs   = self.sample_rate
        if notch_choice != "None":
            f0   = 50.0 if "50" in notch_choice else 60.0
            bw   = 2.0
            low  = (f0 - bw/2) / (fs/2)
            high = (f0 + bw/2) / (fs/2)
            sos  = butter(2, [low, high], btype="bandstop", output="sos")
            data = safe_sosfiltfilt(sos, data)

        # ─── 3) Read FFT window settings ───────────────────────────────────────
        window_length = float(self.fft_length_entry.text())
        start_time    = self.fft_time_slider.value() / self.TIME_MULTIPLIER
        idx0 = int(start_time * fs)
        idx1 = idx0 + int(window_length * fs)
        segment = data[idx0:idx1]
        if len(segment) == 0:
            return

        # ─── 4) Instantaneous frequency via Hilbert
        analytic = hilbert(segment)
        phase    = np.unwrap(np.angle(analytic))
        inst_freq = np.diff(phase) * (fs/(2*np.pi))
        times_if  = np.linspace(start_time, start_time + window_length,
                                len(inst_freq), endpoint=False)

        # drop zeros if requested
        if drop_zero:
            mask = inst_freq != 0.0
            inst_freq = inst_freq[mask]
            times_if  = times_if[mask]
            if inst_freq.size == 0:
                QtWidgets.QMessageBox.information(
                    self, "HFM Analysis",
                    "All detected frequencies were 0 Hz; nothing to show."
                )
                return

        # ─── 5) Fit hyperbolic model f = a*(1/t) + b
        # avoid t=0
        eps = 1e-6
        t_fit = times_if + eps
        X = np.vstack([1.0/t_fit, np.ones_like(t_fit)]).T
        a, b = np.linalg.lstsq(X, inst_freq, rcond=None)[0]

        # ─── 6) Results dialog
        dlg2 = QtWidgets.QDialog(self)
        dlg2.setWindowTitle("HFM Pulse Analysis Results")
        vbox = QtWidgets.QVBoxLayout(dlg2)
        txt = QtWidgets.QPlainTextEdit()
        txt.setReadOnly(True)
        txt.setPlainText(
            f"Fit parameters:\na (s·Hz): {a:.4f}\nb (Hz): {b:.4f}\n"
        )
        vbox.addWidget(txt)

        btns2 = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        vbox.addWidget(btns2)
        def on_accept():
            # log: a → target_frequency, b → measured_voltage
            self.log_measurement_with_project(
                self.file_name,
                "HFM Pulse",
                a,
                start_time, start_time + window_length,
                window_length,
                0.0, 0.0,
                b,
                False,
                ""
            )
            dlg2.accept()
        btns2.accepted.connect(on_accept)
        btns2.rejected.connect(dlg2.reject)

        dlg2.exec_()




    def electrical_noise_popup(self):
        import os
        import re
        import csv
        import json
        import sqlite3
        import numpy as np
        from datetime import datetime
        from PyQt5 import QtWidgets
        import pyqtgraph as pg
        from pyqtgraph.exporters import ImageExporter
        from scipy.signal import welch

        raw = getattr(self, "full_data", None)
        fs = float(getattr(self, "sample_rate", 0.0) or 0.0)
        if raw is None or fs <= 0:
            QtWidgets.QMessageBox.information(self, "Electrical Noise", "No waveform loaded.")
            return

        X = np.asarray(raw)
        if X.ndim == 1:
            X = X[:, None]
        if X.shape[0] < X.shape[1]:
            X = X.T
        if X.shape[0] < 64:
            QtWidgets.QMessageBox.information(self, "Electrical Noise", "Waveform too short.")
            return

        mask = getattr(self, "channel_mask", None) or [True] * X.shape[1]
        names = getattr(self, "channel_names", None) or []
        channels, labels = [], []
        for i in range(X.shape[1]):
            if i < len(mask) and not mask[i]:
                continue
            channels.append(X[:, i])
            labels.append(names[i] if i < len(names) else f"Ch {i+1}")
        if not channels:
            QtWidgets.QMessageBox.information(self, "Electrical Noise", "No active channels.")
            return

        def _sel_color():
            c = getattr(self, "graph_color", None)
            return c.strip() if isinstance(c, str) and c.strip() else "#03DFE2"

        def _palette(n_needed=12):
            colors = []
            cb = getattr(self, "graph_color_cb", None)
            if cb is not None and hasattr(cb, "count") and hasattr(cb, "itemText"):
                for i in range(cb.count()):
                    t = str(cb.itemText(i)).strip()
                    if t.startswith("#") and len(t) >= 7:
                        colors.append(t)
            if not colors:
                colors = ["#33C3F0", "#6EEB83", "#FF5964", "#FFD166", "#C792EA", "#4DD0E1", "#03DFE2"]
            sel = _sel_color()
            start = colors.index(sel) if sel in colors else 0
            return [colors[(start + k) % len(colors)] for k in range(max(1, n_needed))]

        pal = _palette(256)

        def _to_volts(sig):
            v = np.asarray(sig)
            range_v = None
            for attr in ("input_range_v", "ai_range_v", "daq_range_v", "voltage_range_v"):
                if hasattr(self, attr):
                    try:
                        range_v = float(getattr(self, attr))
                        break
                    except Exception:
                        pass
            if range_v is None or not np.isfinite(range_v) or range_v <= 0:
                range_v = 10.0
            maxabs = float(np.nanmax(np.abs(v))) if v.size else 0.0
            if maxabs > 50.0:
                return v.astype(np.float64, copy=False) * (range_v / 32767.0)
            return v.astype(np.float64, copy=False)

        def _safe(name):
            s = str(name or "").strip()
            s = re.sub(r'[<>:"/\\|?*\n\r\t]+', "_", s)
            s = re.sub(r"\s+", " ", s).strip()
            return s or "electrical_noise"

        def _current_file_stem():
            pth = getattr(self, "current_file_path", None) or getattr(self, "file_name", None) or "current_file.wav"
            return os.path.splitext(os.path.basename(str(pth)))[0]

        def _export_dir():
            base = self._project_subdir("electrical_noise") if hasattr(self, "_project_subdir") else None
            if not base:
                base = os.path.join(os.getcwd(), "electrical_noise")
            out = os.path.join(base, _safe(_current_file_stem()))
            os.makedirs(out, exist_ok=True)
            return out

        def _project_id():
            pid = getattr(self, "current_project_id", None)
            if isinstance(pid, int):
                return pid
            pname = (getattr(self, "current_project_name", None) or "").strip()
            if not pname:
                return None
            try:
                conn = sqlite3.connect(DB_FILENAME)
                cur = conn.cursor()
                cur.execute("SELECT id FROM projects WHERE name=?", (pname,))
                row = cur.fetchone()
                conn.close()
                return int(row[0]) if row else None
            except Exception:
                return None

        def _ensure_noise_tables(conn):
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS electrical_noise_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER,
                    project_name TEXT,
                    file_name TEXT,
                    units TEXT,
                    fmin REAL,
                    fmax REAL,
                    vrms_json TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS electrical_noise_points (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    channel_label TEXT,
                    freq_json TEXT,
                    value_json TEXT,
                    FOREIGN KEY(run_id) REFERENCES electrical_noise_runs(id) ON DELETE CASCADE
                )
            """)
            conn.commit()

        def _save_run_to_db():
            if plot_state["f"] is None or not plot_state["curves"]:
                QtWidgets.QMessageBox.information(dlg, "Save Run", "Compute a run first.")
                return
            pid = _project_id()
            pname = (getattr(self, "current_project_name", None) or "").strip()
            if pid is None:
                QtWidgets.QMessageBox.warning(dlg, "Project Required", "Please select a project before saving.")
                return

            try:
                fmin = float(fmin_edit.text())
            except Exception:
                fmin = 0.0
            try:
                fmax = float(fmax_edit.text())
            except Exception:
                fmax = fs / 2.0

            fname = _current_file_stem()
            vrms_map = {lab: v for lab, v in zip(plot_state["labels"], plot_state["vrms"])}
            conn = sqlite3.connect(DB_FILENAME)
            _ensure_noise_tables(conn)
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO electrical_noise_runs(project_id, project_name, file_name, units, fmin, fmax, vrms_json)
                   VALUES(?,?,?,?,?,?,?)""",
                (pid, pname, fname, units_cb.currentText().strip(), fmin, fmax, json.dumps(vrms_map)),
            )
            run_id = int(cur.lastrowid)
            freq = plot_state["f"].tolist()
            for lab, y in zip(plot_state["labels"], plot_state["curves"]):
                cur.execute(
                    "INSERT INTO electrical_noise_points(run_id, channel_label, freq_json, value_json) VALUES(?,?,?,?)",
                    (run_id, lab, json.dumps(freq), json.dumps(np.asarray(y, float).tolist())),
                )
            conn.commit(); conn.close()
            QtWidgets.QMessageBox.information(dlg, "Saved", "Electrical noise run saved to database.")

        def _history_compare_dialog():
            pid = _project_id()
            if pid is None:
                QtWidgets.QMessageBox.warning(dlg, "Project Required", "Please select a project first.")
                return

            conn = sqlite3.connect(DB_FILENAME)
            _ensure_noise_tables(conn)
            cur = conn.cursor()
            cur.execute(
                "SELECT id, file_name, units, created_at FROM electrical_noise_runs WHERE project_id=? ORDER BY id DESC",
                (pid,),
            )
            runs = cur.fetchall()
            if not runs:
                conn.close()
                QtWidgets.QMessageBox.information(dlg, "History", "No saved runs for this project.")
                return

            hd = QtWidgets.QDialog(dlg)
            hd.setWindowTitle("Electrical Noise History / Compare")
            hd.resize(1000, 700)
            hv = QtWidgets.QVBoxLayout(hd)
            hl = QtWidgets.QListWidget()
            hl.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
            for rid, fn, units, ts in runs:
                it = QtWidgets.QListWidgetItem(f"#{rid} | {fn} | {units} | {ts}")
                it.setData(QtCore.Qt.UserRole, int(rid))
                hl.addItem(it)
            hv.addWidget(hl)

            import pyqtgraph as pg
            pw = pg.PlotWidget()
            pw.setBackground("#19232D")
            pi = pw.getPlotItem()
            pi.showGrid(x=True, y=True, alpha=0.25)
            pi.setLogMode(x=True, y=False)
            pi.setLabel("bottom", "Frequency (Hz)", color="white")
            pi.setLabel("left", "Noise density", color="white")
            pi.getAxis("left").setPen(pg.mkPen("white"))
            pi.getAxis("left").setTextPen(pg.mkPen("white"))
            pi.getAxis("bottom").setPen(pg.mkPen("white"))
            pi.getAxis("bottom").setTextPen(pg.mkPen("white"))
            hv.addWidget(pw, 1)

            btn = QtWidgets.QPushButton("Overlay Selected")
            hv.addWidget(btn)

            def _draw_selected():
                pi.clear()
                leg = pi.addLegend()
                try:
                    leg.setBrush(pg.mkBrush(25, 35, 45, 220)); leg.setPen(pg.mkPen("#555"))
                except Exception:
                    pass
                selected = hl.selectedItems()
                if not selected:
                    return
                colors = _palette(512)
                cidx = 0
                for it in selected:
                    rid = int(it.data(QtCore.Qt.UserRole))
                    cur.execute("SELECT channel_label, freq_json, value_json FROM electrical_noise_points WHERE run_id=?", (rid,))
                    for ch, fjs, vjs in cur.fetchall():
                        try:
                            f = np.asarray(json.loads(fjs), float)
                            y = np.asarray(json.loads(vjs), float)
                        except Exception:
                            continue
                        m = np.isfinite(f) & np.isfinite(y) & (f > 0)
                        if np.any(m):
                            pi.plot(f[m], y[m], pen=pg.mkPen(colors[cidx % len(colors)], width=1.8), name=f"run{rid}:{ch}")
                            cidx += 1

            btn.clicked.connect(_draw_selected)
            hd.exec_()
            conn.close()

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Electrical Noise")
        dlg.resize(1100, 760)
        dlg.setStyleSheet("background:#19232D;color:white;")
        lay = QtWidgets.QVBoxLayout(dlg)

        ctrl = QtWidgets.QHBoxLayout()
        fmin_edit = QtWidgets.QLineEdit("10")
        fmax_edit = QtWidgets.QLineEdit(str(int(fs / 2)))
        for w in (fmin_edit, fmax_edit):
            w.setMaximumWidth(90)
            w.setStyleSheet("background:#2b2b2b;color:white;border:1px solid #444;padding:4px;")
        ctrl.addWidget(QtWidgets.QLabel("fmin (Hz):")); ctrl.addWidget(fmin_edit)
        ctrl.addSpacing(12); ctrl.addWidget(QtWidgets.QLabel("fmax (Hz):")); ctrl.addWidget(fmax_edit)

        ctrl.addSpacing(18); ctrl.addWidget(QtWidgets.QLabel("Units:"))
        units_cb = QtWidgets.QComboBox()
        units_cb.addItems(["dBV/√Hz", "dB", "V/√Hz", "nV/√Hz", "V²/Hz"])
        units_cb.setCurrentIndex(0)
        units_cb.setStyleSheet("background:#2b2b2b;color:white;border:1px solid #444;padding:3px;")
        units_cb.setMaximumWidth(140)
        ctrl.addWidget(units_cb)

        ctrl.addStretch(1)
        btn_recalc = QtWidgets.QPushButton("Recompute")
        btn_export_img = QtWidgets.QPushButton("Export JPG")
        btn_export_csv = QtWidgets.QPushButton("Export CSV")
        btn_save_db = QtWidgets.QPushButton("Save Run")
        btn_history = QtWidgets.QPushButton("History / Compare")
        for b in (btn_recalc, btn_export_img, btn_export_csv, btn_save_db, btn_history):
            b.setStyleSheet("background:#2b2b2b;color:white;border:1px solid #555;padding:6px 10px;")
            ctrl.addWidget(b)
        lay.addLayout(ctrl)

        plot = pg.PlotWidget()
        plot.setBackground("#19232D")
        lay.addWidget(plot, 1)
        pitem = plot.getPlotItem()
        pitem.showGrid(x=True, y=True, alpha=0.25)
        pitem.setLogMode(x=True, y=False)
        pitem.setLabel("bottom", "Frequency (Hz)", color="white")
        pitem.setLabel("left", "Noise density", color="white")
        pitem.setTitle("Electrical Noise Density", color="white")
        pitem.getAxis("left").setPen(pg.mkPen("white"))
        pitem.getAxis("left").setTextPen(pg.mkPen("white"))
        pitem.getAxis("bottom").setPen(pg.mkPen("white"))
        pitem.getAxis("bottom").setTextPen(pg.mkPen("white"))

        legend = pitem.addLegend()
        try:
            legend.setBrush(pg.mkBrush(25, 35, 45, 220))
            legend.setPen(pg.mkPen("#555"))
        except Exception:
            pass

        info = QtWidgets.QLabel(); info.setStyleSheet("color:white;padding:6px;")
        lay.addWidget(info)

        row = QtWidgets.QHBoxLayout(); btn_close = QtWidgets.QPushButton("Close")
        btn_close.setStyleSheet("background:#2b2b2b;color:white;border:1px solid #555;padding:6px 10px;")
        row.addStretch(1); row.addWidget(btn_close); lay.addLayout(row)
        btn_close.clicked.connect(dlg.accept)

        plot_state = {"f": None, "labels": [], "curves": [], "vrms": []}

        def _compute():
            pitem.clear()
            nonlocal legend
            legend = pitem.addLegend()
            try:
                legend.setBrush(pg.mkBrush(25, 35, 45, 220))
                legend.setPen(pg.mkPen("#555"))
            except Exception:
                pass

            try:
                fmin = float(fmin_edit.text()); fmax = float(fmax_edit.text())
            except Exception:
                fmin, fmax = 10.0, fs / 2.0
            fmin = max(0.0, fmin)
            fmax = min(fs / 2.0, max(fmin + 1e-6, fmax))
            units = units_cb.currentText().strip()

            lines_txt, curves = [], []
            freq_axis = None
            ylab = "Noise density"

            for i, (sig, lab) in enumerate(zip(channels, labels)):
                v = _to_volts(sig)
                nper = max(256, min(8192, len(v)))
                if nper > len(v):
                    nper = int(len(v))
                nover = max(0, min(nper // 2, nper - 1))
                f, P = welch(v, fs, nperseg=nper, noverlap=nover, scaling="density")
                P = np.maximum(P, 1e-30)
                m = (f >= fmin) & (f <= fmax) & np.isfinite(P)
                if np.count_nonzero(m) >= 2:
                    vrms = float(np.sqrt(np.trapz(P[m], f[m])))
                    lines_txt.append(f"{lab}: {vrms:.6g} Vrms")
                else:
                    vrms = None

                if units == "V²/Hz":
                    y = P; ylab = "PSD (V²/Hz)"
                elif units == "V/√Hz":
                    y = np.sqrt(P); ylab = "Noise density (V/√Hz)"
                elif units == "nV/√Hz":
                    y = np.sqrt(P) * 1e9; ylab = "Noise density (nV/√Hz)"
                else:
                    y = 20.0 * np.log10(np.sqrt(P))
                    ylab = "Noise density (dB/√Hz)" if units == "dB" else "Noise density (dBV/√Hz)"

                mask_plot = (f >= max(1e-6, fmin)) & (f <= fmax) & np.isfinite(f) & np.isfinite(y)
                fp = f[mask_plot]
                yp = y[mask_plot]
                if fp.size:
                    pitem.plot(fp, yp, pen=pg.mkPen(pal[i], width=2), name=lab)

                freq_axis = f
                curves.append((lab, y, vrms))

            if not curves:
                QtWidgets.QMessageBox.information(dlg, "Electrical Noise", "No valid data to plot.")
                return

            pitem.setLabel("left", ylab, color="white")
            pitem.setLabel("bottom", "Frequency (Hz)", color="white")
            pitem.setTitle("Electrical Noise Density", color="white")
            xmin = max(1e-6, fmin)
            xmax = min(fs / 2.0, max(xmin * 1.001, fmax))
            # In pyqtgraph log-x mode, visible range is specified in log10 space.
            pitem.setXRange(np.log10(xmin), np.log10(xmax), padding=0)
            info.setText("\n".join(lines_txt) if lines_txt else "No valid band points")

            plot_state["f"] = freq_axis
            plot_state["labels"] = [c[0] for c in curves]
            plot_state["curves"] = [c[1] for c in curves]
            plot_state["vrms"] = [c[2] for c in curves]

        def _export_jpg():
            out_dir = _export_dir()
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            units_tag = _safe(units_cb.currentText().replace("/", "_per_"))
            path = os.path.join(out_dir, f"electrical_noise_{units_tag}_{stamp}.jpg")
            try:
                exporter = ImageExporter(pitem)
                exporter.parameters()["width"] = 1800
                exporter.export(path)
            except Exception as e:
                QtWidgets.QMessageBox.critical(dlg, "Export", f"Failed to export JPG:\n{e}")
                return
            QtWidgets.QMessageBox.information(dlg, "Exported", f"Saved JPG to:\n{path}")

        def _export_csv():
            if plot_state["f"] is None or not plot_state["curves"]:
                QtWidgets.QMessageBox.information(dlg, "Export", "No computed data to export.")
                return
            out_dir = _export_dir()
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            units = units_cb.currentText().strip()
            units_tag = _safe(units.replace("/", "_per_"))
            path = os.path.join(out_dir, f"electrical_noise_{units_tag}_{stamp}.csv")
            try:
                with open(path, "w", newline="", encoding="utf-8") as fh:
                    w = csv.writer(fh)
                    header = ["frequency_hz"] + [f"{lab}_{units}" for lab in plot_state["labels"]]
                    w.writerow(header)
                    F = plot_state["f"]
                    Ys = plot_state["curves"]
                    for i in range(len(F)):
                        w.writerow([float(F[i])] + [float(y[i]) for y in Ys])
                    w.writerow([])
                    w.writerow(["channel", "vrms"])
                    for lab, v in zip(plot_state["labels"], plot_state["vrms"]):
                        w.writerow([lab, "" if v is None else float(v)])
            except Exception as e:
                QtWidgets.QMessageBox.critical(dlg, "Export", f"Failed to export CSV:\n{e}")
                return
            QtWidgets.QMessageBox.information(dlg, "Exported", f"Saved CSV to:\n{path}")

        btn_recalc.clicked.connect(_compute)
        btn_export_img.clicked.connect(_export_jpg)
        btn_export_csv.clicked.connect(_export_csv)
        btn_save_db.clicked.connect(_save_run_to_db)
        btn_history.clicked.connect(_history_compare_dialog)
        units_cb.currentIndexChanged.connect(_compute)

        _compute()
        dlg.exec_()



    def on_spec_select(self, xmin, xmax):
        """
        Called when the user drags out a region on the spectrogram.
        xmin/xmax are in seconds relative to the start of that clip.
        """
        # store for later (e.g. exporting or listening)
        self.last_spec_region = (xmin, xmax)

        # clear any previous highlight
        self.spec_canvas.ax.patches.clear()

        # compute a lighter version of the graph color for contrast
        try:
            highlight = lighten_color(self.graph_color, amount=0.4)
        except Exception:
            # fallback to using the raw graph color
            highlight = self.graph_color

        # draw the new highlighted span
        self.spec_canvas.ax.axvspan(xmin, xmax, color=highlight, alpha=0.3)
        self.spec_canvas.draw()

        # enable your Export/Listen buttons
        self.export_clip_btn.setEnabled(True)
        self.listen_btn.setEnabled(True)




    def _store_lfm_results(self, results, window_length):
        """
        Persist LFM pulse results. Supports both legacy single-channel lists and
        the newer [(ch, results)] structure for multi-channel logging.
        """

        if results and isinstance(results[0], tuple) and len(results[0]) == 3:
            # Legacy shape: [(t0, rms, freq), ...]
            results = [(None, results)]

        total = 0
        for ch, channel_results in results:
            fname = self.channel_file_label(ch) if ch is not None else self.file_name
            for t0, r, f in channel_results:
                self.log_measurement_with_project(
                    fname,
                    "LFM Pulse",
                    f,
                    t0, t0 + window_length,
                    window_length,
                    float(self.max_voltage_entry.text()),
                    float(self.bw_entry.text()),
                    r,
                    False,
                    "",
                )
                total += 1

        QtWidgets.QMessageBox.information(
            self, "Stored", f"{total} LFM pulse measurements saved."
        )

    def multi_freq_analysis(self):
        if not self.fft_mode:
            QtWidgets.QMessageBox.information(self, "Multi-Freq Analysis",
                "Please switch to FFT mode and set your start time/window first.")
            return

        # Pull window settings
        try:
            window_length = float(self.fft_length_entry.text())
            start_time    = self.fft_time_slider.value() / self.TIME_MULTIPLIER
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Error", "Invalid FFT window settings.")
            return

        # Build dialog
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Multi-Freq Analysis")
        layout = QtWidgets.QFormLayout(dlg)

        # Mode selection
        mode_group = QtWidgets.QGroupBox("Mode")
        rb_single  = QtWidgets.QRadioButton("Single Time")
        rb_scan    = QtWidgets.QRadioButton("Scan Over Time")
        rb_single.setChecked(True)
        hl = QtWidgets.QHBoxLayout(mode_group)
        hl.addWidget(rb_single)
        hl.addWidget(rb_scan)
        layout.addRow(mode_group)

        # Frequencies input
        freq_edit = QtWidgets.QLineEdit("1000,2000,3000")
        layout.addRow("Center freqs (Hz, comma-sep):", freq_edit)

        # Bandwidth half-width
        bw_edit = QtWidgets.QLineEdit("100")
        layout.addRow("Band half-width (Hz):", bw_edit)

        # Interval (only for scan mode)
        interval_edit = QtWidgets.QLineEdit("0.5")
        layout.addRow("Scan interval (s):", interval_edit)

        # Buttons
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok|QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addRow(buttons)

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        # Parse inputs
        try:
            freqs = [float(f) for f in freq_edit.text().split(",")][:10]
            half_bw   = float(bw_edit.text())
            interval  = float(interval_edit.text()) if rb_scan.isChecked() else None
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Error", "Invalid numeric entry.")
            return

        data = self.full_data
        sr   = self.sample_rate
        results = []

        def compute_vrms(seg, center):
            low, high = center - half_bw, center + half_bw
            filt = bandpass_filter(seg, low, high, sr, order=4)
            # convert to volts
            if np.issubdtype(self.original_dtype, np.integer):
                conv = float(self.max_voltage_entry.text()) / np.iinfo(self.original_dtype).max
            else:
                conv = 1.0
            volt = filt * conv
            return float(np.sqrt(np.mean(volt**2)))

        if rb_single.isChecked():
            # single snapshot
            idx0 = int(start_time * sr)
            idx1 = idx0 + int(window_length * sr)
            seg = data[idx0:idx1]
            for cf in freqs:
                vr = compute_vrms(seg, cf)
                results.append((f"{cf:.1f}", f"{start_time:.3f}", f"{vr:.6f}"))
        else:
            # full scan
            total_dur = data.shape[0] / sr
            t = 0.0
            while t + window_length <= total_dur:
                idx0 = int(t * sr)
                idx1 = idx0 + int(window_length * sr)
                seg = data[idx0:idx1]
                for cf in freqs:
                    vr = compute_vrms(seg, cf)
                    results.append((f"{cf:.1f}", f"{t:.3f}", f"{vr:.6f}"))
                t += interval

        # Show results in a table for review
        review = QtWidgets.QDialog(self)
        review.setWindowTitle("Multi-Freq Results")
        vlay = QtWidgets.QVBoxLayout(review)
        table = QtWidgets.QTableWidget(len(results), 3)
        table.setHorizontalHeaderLabels(["Freq (Hz)","Time (s)","VRMS (V)"])
        for i, (f, t, v) in enumerate(results):
            table.setItem(i, 0, QtWidgets.QTableWidgetItem(f))
            table.setItem(i, 1, QtWidgets.QTableWidgetItem(t))
            table.setItem(i, 2, QtWidgets.QTableWidgetItem(v))
        table.resizeColumnsToContents()
        vlay.addWidget(table)

        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok|QtWidgets.QDialogButtonBox.Cancel)
        vlay.addWidget(btn_box)
        btn_box.accepted.connect(review.accept)
        btn_box.rejected.connect(review.reject)

        if review.exec_() == QtWidgets.QDialog.Accepted:
            # log into DB under method "MultiFreq"
            for f, t, v in results:
                self.log_measurement_with_project(
                    self.file_name,
                    "MultiFreq",
                    float(f),
                    float(t),
                    float(t) + window_length,
                    window_length,
                    float(self.max_voltage_entry.text()),
                    half_bw*2,
                    float(v),
                    False,
                    ""
                )
            QtWidgets.QMessageBox.information(self, "Logged", f"{len(results)} measurements saved.")


    def slope_declipper_popup(self):
        """
        De-clipper with three detection modes:
        • Amplitude    : |sample| ≈ threshold (amplitude units)
        • Vrms         : sliding-window RMS ≈ threshold (V)
        • Flat Plateau : consecutive samples nearly equal (Δamplitude)
        Can operate on selected region or entire file.
        """
        # 1) Ensure data is loaded
        if self.full_data is None:
            QtWidgets.QMessageBox.critical(self, "Error", "Load a WAV file first.")
            return

        # 2) Build parameter dialog
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Slope De-clipper")
        vbox = QtWidgets.QVBoxLayout(dlg)

        # Scope: Selected region vs entire file
        scope_group = QtWidgets.QGroupBox("Scope")
        hl0 = QtWidgets.QHBoxLayout(scope_group)
        single_rb = QtWidgets.QRadioButton("Selected Region")
        batch_rb  = QtWidgets.QRadioButton("Entire File")
        single_rb.setChecked(True)
        hl0.addWidget(single_rb)
        hl0.addWidget(batch_rb)
        vbox.addWidget(scope_group)

        # Detection mode: Amplitude, Vrms, Flat Plateau
        type_group = QtWidgets.QGroupBox("Detection Mode")
        hl1 = QtWidgets.QHBoxLayout(type_group)
        amp_rb     = QtWidgets.QRadioButton("Amplitude")
        rms_rb     = QtWidgets.QRadioButton("Vrms")
        plateau_rb = QtWidgets.QRadioButton("Flat Plateau")
        amp_rb.setChecked(True)
        hl1.addWidget(amp_rb)
        hl1.addWidget(rms_rb)
        hl1.addWidget(plateau_rb)
        vbox.addWidget(type_group)

        # Parameters form
        form = QtWidgets.QFormLayout()
        thresh_edit = QtWidgets.QLineEdit("1.0")
        wig_edit    = QtWidgets.QLineEdit("0.02")
        window_edit = QtWidgets.QLineEdit("0.01")
        window_edit.setEnabled(False)

        # Labels that will update based on mode
        thresh_label = QtWidgets.QLabel("Threshold (Amplitude):")
        wig_label    = QtWidgets.QLabel("Tolerance (±Amplitude):")

        form.addRow(thresh_label, thresh_edit)
        form.addRow(wig_label,    wig_edit)
        form.addRow("Window Length (s):", window_edit)
        vbox.addLayout(form)

        # Update labels & window_field enable state on mode change
        def on_type_change():
            is_vrms_or_plateau = rms_rb.isChecked() or plateau_rb.isChecked()
            window_edit.setEnabled(is_vrms_or_plateau)

            if amp_rb.isChecked():
                thresh_label.setText("Threshold (Amplitude):")
                wig_label.   setText("Tolerance (±Amplitude):")
            elif rms_rb.isChecked():
                thresh_label.setText("Threshold (Vrms [V]):")
                wig_label.   setText("Tolerance (±V):")
            else:
                thresh_label.setText("Threshold (ΔAmplitude):")
                wig_label.   setText("Tolerance (ΔAmplitude):")

        amp_rb.toggled.    connect(on_type_change)
        rms_rb.toggled.    connect(on_type_change)
        plateau_rb.toggled.connect(on_type_change)

        # OK / Cancel buttons
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        vbox.addWidget(btns)

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        # 3) Parse inputs
        try:
            thresh   = float(thresh_edit.text())
            wiggle   = float(wig_edit.text())
            window_s = float(window_edit.text()) if (rms_rb.isChecked() or plateau_rb.isChecked()) else None
        except ValueError:
            QtWidgets.QMessageBox.critical(self, "Error", "Invalid numeric values.")
            return

        # 4) Extract data segment
        data = self.full_data
        sr   = self.sample_rate
        base_offset = 0
        if single_rb.isChecked():
            if not self.fft_mode or self.last_region is None:
                QtWidgets.QMessageBox.information(self, "Error",
                    "Switch to FFT mode and select a region first.")
                return
            start_t, end_t = self.last_region
            s0 = int(start_t * sr)
            s1 = int(end_t   * sr)
            data = data[s0:s1]
            base_offset = s0

        results = []

        # 5) Detection logic
        if amp_rb.isChecked():
            # Amplitude-based plateau detection
            mask   = np.abs(data) >= (thresh - wiggle)
            change = np.diff(mask.astype(int))
            starts = np.where(change == 1)[0] + 1
            ends   = np.where(change == -1)[0] + 1
            if mask[0]:  starts = np.insert(starts, 0, 0)
            if mask[-1]: ends   = np.append(ends, len(mask))
            for s, e in zip(starts, ends):
                if e - s < 3: continue
                seg = data[s:e]
                x   = np.arange(s, e)
                A   = np.vstack([x, np.ones_like(x)]).T
                m, b = np.linalg.lstsq(A, seg, rcond=None)[0]
                mid  = (s + e) // 2
                win  = int(0.001 * sr)
                w0, w1 = max(0, mid-win), min(len(data), mid+win)
                xw = np.arange(w0, w1)
                pred = m*xw + b
                vrms_inst = np.sqrt(np.mean(pred**2))
                results.append(((base_offset+s)/sr, (base_offset+e)/sr, vrms_inst))

        elif rms_rb.isChecked():
            # Vrms-based plateau detection
            N = max(1, int(window_s * sr))
            sq  = data**2
            cum = np.cumsum(sq)
            rms_vals = np.sqrt((cum[N:] - cum[:-N]) / N)
            mask = (rms_vals >= (thresh - wiggle)) & (rms_vals <= (thresh + wiggle))
            change = np.diff(mask.astype(int))
            starts = np.where(change == 1)[0] + 1
            ends   = np.where(change == -1)[0] + 1
            if mask[0]:  starts = np.insert(starts, 0, 0)
            if mask[-1]: ends   = np.append(ends, len(mask))
            for sw, ew in zip(starts, ends):
                s = sw
                e = ew + N
                vrms_inst = float(np.mean(rms_vals[sw:ew]))
                results.append(((base_offset+s)/sr, (base_offset+e)/sr, vrms_inst))

        else:
            # Flat Plateau detection: runs of nearly-constant amplitude
            N = max(1, int(window_s * sr))
            diff = np.abs(np.diff(data))
            mask = diff <= wiggle
            change = np.diff(mask.astype(int))
            starts = np.where(change == 1)[0] + 1
            ends   = np.where(change == -1)[0] + 1
            if mask[0]:  starts = np.insert(starts, 0, 0)
            if mask[-1]: ends   = np.append(ends, len(mask))
            for sw, ew in zip(starts, ends):
                if (ew - sw) < N:
                    continue
                s = sw
                e = ew + 1
                mean_amp = float(np.mean(data[s:e]))
                results.append(((base_offset+s)/sr, (base_offset+e)/sr, mean_amp))

        # 6) Handle no results
        if not results:
            QtWidgets.QMessageBox.information(self, "No Results",
                                            "No segments found.")
            return

        # 7) Display & log
        dlg2 = QtWidgets.QDialog(self)
        dlg2.setWindowTitle("De-clipper Results")
        layout2 = QtWidgets.QVBoxLayout(dlg2)
        txt = QtWidgets.QLabel()
        txt.setFont(QtGui.QFont("Courier", 10))
        layout2.addWidget(txt)

        pages = [results[i:i+10] for i in range(0, len(results), 10)]
        page = 0

        def show_page(i):
            nonlocal page
            page = i
            chunk = pages[page]
            s = "\n".join(f"{t0:7.3f}-{t1:7.3f}s : Value={v:6.4f}"
                        for t0, t1, v in chunk)
            txt.setText(f"<pre>{s}</pre>")
            prev_btn.setEnabled(page > 0)
            next_btn.setEnabled(page < len(pages)-1)

        nav = QtWidgets.QHBoxLayout()
        prev_btn = QtWidgets.QPushButton("← Prev")
        next_btn = QtWidgets.QPushButton("Next →")
        accept_btn = QtWidgets.QPushButton("Accept")
        cancel_btn = QtWidgets.QPushButton("Cancel")
        nav.addWidget(prev_btn)
        nav.addWidget(next_btn)
        nav.addStretch()
        nav.addWidget(accept_btn)
        nav.addWidget(cancel_btn)
        layout2.addLayout(nav)

        prev_btn.clicked.connect(lambda: show_page(page-1))
        next_btn.clicked.connect(lambda: show_page(page+1))
        accept_btn.clicked.connect(dlg2.accept)
        cancel_btn.clicked.connect(dlg2.reject)

        show_page(0)

        if dlg2.exec_() == QtWidgets.QDialog.Accepted:
            for t0, t1, v in results:
                self.log_measurement_with_project(
                    self.file_name,
                    "Slope Declipper",
                    0.0,
                    t0,
                    t1,
                    t1 - t0,
                    0.0,
                    0.0,
                    v,
                    False,
                    "",
                    misc=v
                )
            QtWidgets.QMessageBox.information(
                self,
                "Logged",
                f"{len(results)} segments logged."
            )

    ##Spl transmit analysis tool

    def spl_from_voltage_popup(self):
        """
        SPL Transmit (from DB Measurements)
        - Select measurement rows by file_name and analysis method (from tables: measurements + archive)
        - TVR source: DB curve or CSV
        - Compute SPL@1m = TVR(f) + 20*log10(Vrms) for each row
        - Plot SPL vs Frequency; support overlays; export CSV/Excel
        """
        import os, json, sqlite3
        import numpy as np
        from PyQt5 import QtWidgets, QtCore
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("SPL Transmit (DB Measurements)")
        dlg.setStyleSheet("background:#19232D; color:white;")
        dlg.setMinimumSize(1000, 700)
        layout = QtWidgets.QVBoxLayout(dlg)

        # --------- DB helpers ----------
        def _conn():
            return sqlite3.connect(DB_FILENAME)

        def list_distinct_files():
            with _conn() as c:
                cur = c.cursor()
                cur.execute("""
                    SELECT DISTINCT file_name FROM measurements
                    UNION
                    SELECT DISTINCT file_name FROM archive
                    ORDER BY file_name
                """)
                return [r[0] for r in cur.fetchall()]

        def list_methods_for_file(fname):
            with _conn() as c:
                cur = c.cursor()
                cur.execute("""
                    SELECT DISTINCT method FROM measurements WHERE file_name=?
                    UNION
                    SELECT DISTINCT method FROM archive WHERE file_name=?
                    ORDER BY method
                """, (fname, fname))
                return [r[0] for r in cur.fetchall()]

        def fetch_measurement_rows(fname, method):
            """
            Returns arrays (freqs, vrms) from both measurements + archive.
            Uses COALESCE(measured_voltage, max_voltage) as Vrms.
            Skips rows with nonpositive values.
            """
            with _conn() as c:
                cur = c.cursor()
                cur.execute("""
                    SELECT target_frequency, COALESCE(measured_voltage, max_voltage)
                    FROM measurements WHERE file_name=? AND method=?
                    UNION ALL
                    SELECT target_frequency, COALESCE(measured_voltage, max_voltage)
                    FROM archive WHERE file_name=? AND method=?
                """, (fname, method, fname, method))
                rows = cur.fetchall()
            f, v = [], []
            for (freq, vrms) in rows:
                try:
                    if freq is None or vrms is None:
                        continue
                    freq = float(freq); vrms = float(vrms)
                    if freq > 0 and vrms > 0:
                        f.append(freq); v.append(vrms)
                except Exception:
                    continue
            if not f:
                return np.array([], dtype=float), np.array([], dtype=float)
            f = np.asarray(f, dtype=float)
            v = np.asarray(v, dtype=float)
            order = np.argsort(f)
            return f[order], v[order]

        # --------- TVR builders ----------
        def build_tvr_interp_from_db(curve_name):
            with _conn() as c:
                cur = c.cursor()
                cur.execute("SELECT min_frequency, max_frequency, tvr_json FROM tvr_curves WHERE curve_name=?", (curve_name,))
                row = cur.fetchone()
            if not row:
                raise ValueError(f"TVR curve '{curve_name}' not found in database.")
            fmin, fmax, js = row
            y = np.array(json.loads(js), dtype=float)
            x = np.arange(int(fmin), int(fmax) + 1, dtype=float)
            def tvrdB(f_hz):
                f = np.asarray(f_hz, dtype=float)
                f = np.clip(f, x[0], x[-1])
                try:
                    return np.interp(np.log10(f), np.log10(x), y)
                except Exception:
                    return np.interp(f, x, y)
            return tvrdB, float(fmin), float(fmax)

        def build_tvr_interp_from_csv(csv_path):
            import csv
            rows = []
            with open(csv_path, "r", newline="", encoding="utf-8") as fh:
                rdr = csv.reader(fh)
                rows = [r for r in rdr]
            if not rows:
                raise ValueError("TVR CSV is empty.")
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
                # fallback: pick two most numeric cols
                max_cols = max(len(r) for r in data) if data else 0
                counts = []
                for ci in range(max_cols):
                    cnt = 0
                    for r in data:
                        if ci < len(r):
                            try: float(r[ci]); cnt += 1
                            except Exception: pass
                    counts.append((cnt, ci))
                counts.sort(reverse=True)
                if len(counts) < 2 or counts[0][0]==0 or counts[1][0]==0:
                    raise ValueError("Could not identify Frequency/TVR columns in CSV.")
                if freq_idx is None: freq_idx = counts[0][1]
                if tvr_idx  is None: tvr_idx  = counts[1][1]
            freqs, tvals = [], []
            for r in data:
                if max(freq_idx, tvr_idx) >= len(r): continue
                try:
                    f = float(r[freq_idx]); v = float(r[tvr_idx])
                    if f>0: freqs.append(f); tvals.append(v)
                except Exception:
                    continue
            freqs = np.asarray(freqs, dtype=float)
            tvals = np.asarray(tvals, dtype=float)
            if freqs.size < 2:
                raise ValueError("Need at least two TVR points in CSV.")
            order = np.argsort(freqs); freqs = freqs[order]; tvals = tvals[order]
            fmin = float(np.min(freqs)); fmax = float(np.max(freqs))
            def tvrdB(f_hz):
                f = np.asarray(f_hz, dtype=float)
                f = np.clip(f, fmin, fmax)
                try:
                    return np.interp(np.log10(f), np.log10(freqs), tvals)
                except Exception:
                    return np.interp(f, freqs, tvals)
            return tvrdB, fmin, fmax

        # --------- Top selectors (file + method + TVR source) ----------
        top = QtWidgets.QGridLayout()
        r = 0

        top.addWidget(QtWidgets.QLabel("File:"), r, 0)
        file_cb = QtWidgets.QComboBox(); file_cb.setMinimumWidth(260)
        file_cb.addItems(list_distinct_files())
        top.addWidget(file_cb, r, 1, 1, 3)

        r += 1
        top.addWidget(QtWidgets.QLabel("Method:"), r, 0)
        method_cb = QtWidgets.QComboBox(); method_cb.setMinimumWidth(220)
        # populate initial methods
        method_cb.addItems(list_methods_for_file(file_cb.currentText()) if file_cb.count() else [])
        top.addWidget(method_cb, r, 1, 1, 3)

        r += 1
        top.addWidget(QtWidgets.QLabel("TVR Source:"), r, 0)
        tvr_src_cb = QtWidgets.QComboBox(); tvr_src_cb.addItems(["Database curve", "CSV file"])
        top.addWidget(tvr_src_cb, r, 1)

        db_label = QtWidgets.QLabel("DB Curve:")
        top.addWidget(db_label, r, 2)
        tvr_db_cb = QtWidgets.QComboBox(); tvr_db_cb.setMinimumWidth(220)
        # load DB curve names
        with _conn() as _c:
            _cur = _c.cursor()
            _cur.execute("SELECT curve_name FROM tvr_curves ORDER BY curve_name")
            tvr_db_cb.addItems([r[0] for r in _cur.fetchall()])
        top.addWidget(tvr_db_cb, r, 3)

        r += 1
        csv_label = QtWidgets.QLabel("CSV:")
        top.addWidget(csv_label, r, 0)
        tvr_csv_edit = QtWidgets.QLineEdit(""); tvr_csv_edit.setPlaceholderText(r"C:\path\to\tvr.csv")
        tv_browse = QtWidgets.QPushButton("Browse…")
        top.addWidget(tvr_csv_edit, r, 1, 1, 2)
        top.addWidget(tv_browse, r, 3)

        # distance option (optional display)
        r += 1
        top.addWidget(QtWidgets.QLabel("Distance (m) (optional):"), r, 0)
        dist_spin = QtWidgets.QDoubleSpinBox(); dist_spin.setDecimals(3); dist_spin.setRange(0.001, 1e6); dist_spin.setValue(1.0)
        top.addWidget(dist_spin, r, 1)
        show_dist_chk = QtWidgets.QCheckBox("Show SPL@distance along with SPL@1m")
        show_dist_chk.setChecked(False)
        top.addWidget(show_dist_chk, r, 2, 1, 2)

        layout.addLayout(top)

        def _on_file_changed(_):
            method_cb.blockSignals(True)
            method_cb.clear()
            method_cb.addItems(list_methods_for_file(file_cb.currentText()))
            method_cb.blockSignals(False)
        file_cb.currentTextChanged.connect(_on_file_changed)

        def _on_src_changed(ix):
            use_db = (ix == 0)
            db_label.setVisible(use_db); tvr_db_cb.setVisible(use_db)
            csv_label.setVisible(not use_db); tvr_csv_edit.setVisible(not use_db); tv_browse.setVisible(not use_db)
        tvr_src_cb.currentIndexChanged.connect(_on_src_changed)
        _on_src_changed(tvr_src_cb.currentIndex())

        def _browse():
            path, _ = QtWidgets.QFileDialog.getOpenFileName(dlg, "Select TVR CSV", "", "CSV Files (*.csv)")
            if path: tvr_csv_edit.setText(path)
        tv_browse.clicked.connect(_browse)

        # --------- Center: plot + controls ----------
        fig = Figure(facecolor="#19232D")
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.set_facecolor("#19232D")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_color("white")
        ax.set_xlabel("Frequency (Hz)", color="white")
        ax.set_ylabel("SPL @ 1 m (dB re 1 μPa)", color="white")
        ax.grid(True, ls="--", alpha=0.35, color="gray")
        layout.addWidget(canvas, 1)

        # Overlay color handling
        color_options = getattr(self, "color_options", {
            "Aqua": "#03DFE2", "Purple": "#A78BFA", "Green": "#10B981", "Orange": "#F59E0B",
            "Blue": "#60A5FA", "Pink": "#F472B6", "Lime": "#84CC16", "Amber": "#F59E0B"
        })
        overlay_palette = list(color_options.values())
        _overlay_idx = 0
        def _next_col():
            nonlocal _overlay_idx
            if not overlay_palette: return "#AAAAAA"
            c = overlay_palette[_overlay_idx % len(overlay_palette)]
            _overlay_idx += 1
            return c

        # State for overlays: label -> (freqs, spl_1m, maybe spl_d, color)
        overlays = {}

        # --------- Bottom: actions ----------
        btn_row = QtWidgets.QHBoxLayout()
        btn_plot = QtWidgets.QPushButton("Plot")
        btn_add  = QtWidgets.QPushButton("Add Overlay")
        btn_clear= QtWidgets.QPushButton("Clear Overlays")
        btn_export = QtWidgets.QPushButton("Export…")
        btn_close = QtWidgets.QPushButton("Close")
        btn_row.addStretch(); [btn_row.addWidget(b) for b in (btn_plot, btn_add, btn_clear, btn_export, btn_close)]
        layout.addLayout(btn_row)

        # Blue buttons
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
        for b in (tv_browse, btn_plot, btn_add, btn_clear, btn_export, btn_close):
            b.setStyleSheet(BLUE_BTN)

        # --------- Compute helper ----------
        def compute_current_dataset():
            fname  = file_cb.currentText().strip()
            method = method_cb.currentText().strip()
            if not fname or not method:
                QtWidgets.QMessageBox.warning(dlg, "Select", "Choose a file and method.")
                return None
            freqs, vrms = fetch_measurement_rows(fname, method)
            if freqs.size == 0:
                QtWidgets.QMessageBox.information(dlg, "No Rows", "No measurement rows matched that file/method.")
                return None

            # TVR
            use_db = (tvr_src_cb.currentIndex() == 0)
            try:
                if use_db:
                    cname = tvr_db_cb.currentText().strip()
                    if not cname:
                        QtWidgets.QMessageBox.warning(dlg, "TVR", "Choose a TVR curve from the database.")
                        return None
                    tvrdB, fmin, fmax = build_tvr_interp_from_db(cname)
                    src_label = f"DB:{cname}"
                else:
                    path = tvr_csv_edit.text().strip()
                    if not path:
                        QtWidgets.QMessageBox.warning(dlg, "TVR", "Choose a TVR CSV file.")
                        return None
                    tvrdB, fmin, fmax = build_tvr_interp_from_csv(path)
                    src_label = f"CSV:{os.path.basename(path)}"
            except Exception as e:
                QtWidgets.QMessageBox.critical(dlg, "TVR Error", str(e))
                return None

            # Clamp to TVR range (warn if clamped)
            f_clamped = np.clip(freqs, fmin, fmax)
            clamped_n = int(np.sum((freqs < fmin) | (freqs > fmax)))
            if clamped_n > 0:
                QtWidgets.QMessageBox.information(
                    dlg, "Frequency Clamp",
                    f"{clamped_n} of {freqs.size} frequency points were outside TVR range "
                    f"({int(fmin)}–{int(fmax)} Hz) and were clamped to the edge."
                )

            tvr_db_vals = tvrdB(f_clamped)
            spl_1m = tvr_db_vals + 20.0 * np.log10(vrms)

            data = {
                "label": f"{os.path.basename(fname)} | {method} | {src_label}",
                "freqs": freqs,
                "vrms":  vrms,
                "tvr_db": tvr_db_vals,
                "spl_1m": spl_1m,
                "fmin": fmin, "fmax": fmax,
            }
            if show_dist_chk.isChecked() and dist_spin.value() > 0:
                d = float(dist_spin.value())
                # Level at distance d: L(d) = L(1m) - 20*log10(d/1)
                data["spl_d"] = spl_1m - 20.0 * np.log10(max(d, 1e-12))
                data["distance_m"] = d
            else:
                data["spl_d"] = None
                data["distance_m"] = None
            return data

        def draw_current_only(data):
            ax.clear()
            ax.set_facecolor("#19232D")
            ax.tick_params(colors="white")
            for sp in ax.spines.values():
                sp.set_color("white")
            ax.grid(True, ls="--", alpha=0.35, color="gray")
            ax.set_xlabel("Frequency (Hz)", color="white")
            ax.set_ylabel("SPL @ 1 m (dB re 1 μPa)", color="white")

            # Main in selected color
            main_col = color_options.get(next(iter(color_options.keys())), "#03DFE2")
            ax.plot(data["freqs"], data["spl_1m"], "-", lw=2, color=main_col, label=f"{data['label']} (1 m)")
            if data["spl_d"] is not None:
                ax.plot(data["freqs"], data["spl_d"], "--", lw=1.6, color=main_col, alpha=0.9,
                        label=f"{data['label']} (@ {data['distance_m']:.3f} m)")

            # Overlays
            for name, od in overlays.items():
                col = od.get("color", "#888888")
                ax.plot(od["freqs"], od["spl_1m"], "-", lw=1.8, color=col, alpha=0.95, label=f"{name} (1 m)")
                if od.get("spl_d") is not None:
                    ax.plot(od["freqs"], od["spl_d"], "--", lw=1.4, color=col, alpha=0.9,
                            label=f"{name} (@ {od['distance_m']:.3f} m)")

            ax.legend(facecolor="#19232D", edgecolor="white", labelcolor="white")
            canvas.draw_idle()

        def add_overlay_from_data(data):
            name = data["label"]
            if name in overlays:
                QtWidgets.QMessageBox.information(dlg, "Overlay", "That dataset is already in overlays.")
                return
            data = dict(data)  # copy
            data["color"] = _next_col()
            overlays[name] = data

        # --------- Export ----------
        def do_export():
            # Build datasets dictionary: name -> dataframe-ish dict
            datasets = {}
            # Current
            cur = compute_current_dataset()
            if cur is not None:
                datasets[cur["label"]] = cur
            # Overlays
            for k, v in overlays.items():
                datasets[k] = v
            if not datasets:
                QtWidgets.QMessageBox.information(dlg, "No Data", "Nothing to export.")
                return

            choices = [
                "Current dataset -> CSV",
                "All datasets -> multiple CSVs",
                "All datasets -> Excel workbook (.xlsx)"
            ]
            selection_text, ok = QtWidgets.QInputDialog.getItem(
                dlg, "Export", "Choose export format:", choices, 0, False
            )
            if not ok: return
            sel = choices.index(selection_text)

            import pandas as pd, re
            def to_df(d):
                cols = {
                    "Frequency_Hz": np.asarray(d["freqs"], dtype=float),
                    "Vrms_V": np.asarray(d["vrms"], dtype=float),
                    "TVR_dB": np.asarray(d["tvr_db"], dtype=float),
                    "SPL_1m_dB": np.asarray(d["spl_1m"], dtype=float),
                }
                if d.get("spl_d") is not None:
                    cols[f"SPL_{d['distance_m']:.3f}m_dB"] = np.asarray(d["spl_d"], dtype=float)
                return pd.DataFrame(cols)

            def sanitize_filename(s):
                s = re.sub(r"[^\w\-. ]+", "_", s.strip())
                return s or "dataset"

            if sel == 0:
                if cur is None:
                    QtWidgets.QMessageBox.warning(dlg, "No Current", "No current dataset to export.")
                    return
                path, _ = QtWidgets.QFileDialog.getSaveFileName(dlg, "Export CSV", "", "CSV Files (*.csv)")
                if not path: return
                try:
                    to_df(cur).to_csv(path, index=False)
                    QtWidgets.QMessageBox.information(dlg, "Exported", f"CSV saved to:\n{path}")
                except Exception as e:
                    QtWidgets.QMessageBox.critical(dlg, "Export Failed", str(e))
                return

            if sel == 1:
                out_dir = QtWidgets.QFileDialog.getExistingDirectory(dlg, "Choose output folder")
                if not out_dir: return
                failed = []
                for name, d in datasets.items():
                    fpath = os.path.join(out_dir, sanitize_filename(name) + ".csv")
                    try:
                        to_df(d).to_csv(fpath, index=False)
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
                        for name, d in datasets.items():
                            sheet = (name or "Sheet").strip()[:31] or "Sheet"
                            base = sheet; i = 2
                            while sheet in used:
                                sheet = (base[:28] + f"_{i}")[:31]; i += 1
                            used.add(sheet)
                            to_df(d).to_excel(xw, index=False, sheet_name=sheet)
                    QtWidgets.QMessageBox.information(dlg, "Exported", f"Workbook saved to:\n{path}")
                except Exception as e:
                    QtWidgets.QMessageBox.critical(
                        dlg, "Export Failed",
                        "Could not write Excel workbook.\n"
                        "Install an Excel engine (e.g. `pip install openpyxl` or `pip install xlsxwriter`).\n\n"
                        f"Details:\n{e}"
                    )
                return

        # --------- Wire up ----------
        def do_plot():
            data = compute_current_dataset()
            if data is None:
                return
            draw_current_only(data)

        btn_plot.clicked.connect(do_plot)

        def do_add_overlay():
            data = compute_current_dataset()
            if data is None: return
            add_overlay_from_data(data)
            draw_current_only(data)
        btn_add.clicked.connect(do_add_overlay)

        btn_clear.clicked.connect(lambda: (overlays.clear(), do_plot()))
        btn_export.clicked.connect(do_export)
        btn_close.clicked.connect(dlg.reject)

        # initial plot if possible
        if file_cb.count() and method_cb.count():
            do_plot()

        dlg.exec_()











    def peak_prominences_popup(self):
        """
        Finds peaks via scipy.signal.find_peaks, then computes prominences,
        RMS around each peak, and dominant frequency in a small window around each peak.
        Pops up results sorted by frequency, and logs each entry with the correct start/end times.
        """
        if self.full_data is None:
            QtWidgets.QMessageBox.critical(self, "Error", "No file loaded.")
            return

        # Build dialog for user parameters
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Peak Prominences Analysis")
        dlg.setModal(True)

        threshold_edit = QtWidgets.QLineEdit("0.5")
        window_edit    = QtWidgets.QLineEdit("0.1")
        threshold_edit.setFixedWidth(80)
        window_edit.setFixedWidth(80)

        form = QtWidgets.QFormLayout()
        form.addRow("Minimum Peak Height (normalized 0–1):", threshold_edit)
        form.addRow("Window around peak (s):", window_edit)

        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(dlg.accept)
        btn_box.rejected.connect(dlg.reject)

        vbox = QtWidgets.QVBoxLayout(dlg)
        vbox.addLayout(form)
        vbox.addWidget(btn_box)

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        try:
            min_height = float(threshold_edit.text())
            win_s      = float(window_edit.text())
        except ValueError:
            QtWidgets.QMessageBox.critical(self, "Error", "Invalid parameters.")
            return

        data = self.full_data.astype(np.float64)
        if np.issubdtype(self.original_dtype, np.integer):
            max_possible = float(np.iinfo(self.original_dtype).max)
            data = data / max_possible

        from scipy.signal import find_peaks, peak_prominences

        peaks, _ = find_peaks(data, height=min_height)
        if peaks.size == 0:
            QtWidgets.QMessageBox.information(self, "No Peaks", "No peaks found above threshold.")
            return

        prominences = peak_prominences(data, peaks)[0]
        results = []
        half_window = int(win_s * self.sample_rate / 2)

        for idx, pr in zip(peaks, prominences):
            peak_time = self.full_time[idx]
            start_idx = max(0, idx - half_window)
            end_idx   = min(len(data), idx + half_window)

            segment = self.full_data[start_idx:end_idx]
            if segment.size == 0:
                continue

            if np.issubdtype(self.original_dtype, np.integer):
                try:
                    conv = float(self.max_voltage_entry.text()) / np.iinfo(self.original_dtype).max
                except:
                    conv = 1.0
            else:
                conv = 1.0

            rms_voltage = np.sqrt(np.mean((segment * conv) ** 2))

            n = len(segment)
            if n < 2:
                continue
            nfft = 2 ** int(np.ceil(np.log2(n)))
            fft_result = np.fft.rfft(segment * np.hanning(n), n=nfft)
            freqs = np.fft.rfftfreq(nfft, d=1/self.sample_rate)
            dom_freq = self.refine_frequency(fft_result, freqs)

            results.append((dom_freq, rms_voltage, peak_time, pr))

        if not results:
            QtWidgets.QMessageBox.information(self, "No Results", "No valid peaks/windows found.")
            return

        # Sort by dominant frequency
        results.sort(key=lambda x: x[0])

        # Show results in a read-only text dialog
        dlg2 = QtWidgets.QDialog(self)
        dlg2.setWindowTitle("Peak Prominences Results")
        dlg2.resize(500, 400)

        txt = QtWidgets.QPlainTextEdit()
        txt.setReadOnly(True)
        out = "DomFreq (Hz)   RMSVoltage (V)   PeakTime (s)   Prominence\n"
        for df, rv, pt, pr in results:
            out += f"{df:10.2f}     {rv:10.4f}     {pt:10.3f}     {pr:10.4f}\n"
        txt.setPlainText(out)

        btn_keep = QtWidgets.QPushButton("Accept Results")
        btn_discard = QtWidgets.QPushButton("Discard")

        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addWidget(btn_keep)
        btn_layout.addWidget(btn_discard)

        v2 = QtWidgets.QVBoxLayout(dlg2)
        v2.addWidget(txt)
        v2.addLayout(btn_layout)

        def keep_action():
            """
            For each result, compute:
              start_time = peak_time - (win_s/2)
              end_time   = peak_time + (win_s/2)
            and then log it.
            """
            for df, rv, pt, pr in results:
                start_time = pt - (win_s / 2.0)
                end_time   = pt + (win_s / 2.0)

                self.log_measurement_with_project(
                    self.file_name,            # file_name
                    "PeakProminence",          # method
                    df,                        # target_frequency
                    start_time,                # start_time within file
                    end_time,                  # end_time within file
                    win_s,                     # window_length
                    float(self.max_voltage_entry.text()) if hasattr(self, 'max_voltage_entry') else 1.0,
                    0.0,                       # bandwidth (unused here)
                    rv,                        # measured_voltage (the RMS we computed)
                    False,                     # filter_applied
                    ""                         # screenshot
                )
            dlg2.accept()

        btn_keep.clicked.connect(keep_action)
        btn_discard.clicked.connect(dlg2.reject)
        dlg2.exec_()
