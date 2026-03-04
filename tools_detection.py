#!/usr/bin/env python3
"""
Detection & Classification Tools — methods for MainWindow mixin
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

class DetectionToolsMixin:
    """Mixin class providing all Detection & Classification Tools for self."""

    def cepstrum_analysis(self):
        """
        Perform real cepstrum analysis on the selected FFT window to detect echoes
        and multipath reflections. Pops up controls for quefrency range, threshold, and sound speed.
        """
        if not self.fft_mode:
            QtWidgets.QMessageBox.information(
                self, "Info", "Switch to FFT mode and select a region first.")
            return

        # Popup for user parameters
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Cepstrum Analysis")
        form = QtWidgets.QFormLayout(dlg)
        
        min_q_entry = QtWidgets.QLineEdit("0.001")  # seconds
        max_q_entry = QtWidgets.QLineEdit("0.1")    # seconds
        thresh_entry = QtWidgets.QLineEdit("0.01")  # cepstral peak threshold
        speed_entry = QtWidgets.QLineEdit("1500")   # m/s
        for w in (min_q_entry, max_q_entry, thresh_entry, speed_entry):
            w.setFixedWidth(80)

        form.addRow("Min Quefrency (s):", min_q_entry)
        form.addRow("Max Quefrency (s):", max_q_entry)
        form.addRow("Peak Threshold:", thresh_entry)
        form.addRow("Speed of Sound (m/s):", speed_entry)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        form.addWidget(btns)

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        # Read parameters
        try:
            min_q = float(min_q_entry.text())
            max_q = float(max_q_entry.text())
            thresh = float(thresh_entry.text())
            c = float(speed_entry.text())
        except ValueError:
            QtWidgets.QMessageBox.critical(self, "Error", "Invalid numeric parameters.")
            return

        # Extract data segment
        sr = self.sample_rate
        start_t = self.fft_start_time
        window_length = float(self.fft_length_entry.text())
        idx0 = int(start_t * sr)
        idx1 = idx0 + int(window_length * sr)
        segment = self.full_data[idx0:idx1]

        # Apply Hanning window and FFT
        nfft = 1 << int(np.ceil(np.log2(len(segment))))
        windowed = segment * np.hanning(len(segment))
        spectrum = np.fft.rfft(windowed, n=nfft)
        mag = np.abs(spectrum)

        # Compute log-magnitude and real cepstrum
        log_mag = np.log(mag + 1e-12)
        cep = np.fft.irfft(log_mag, n=nfft)
        q = np.arange(len(cep)) / sr

        # Find peaks within quefrency window
        mask = (q >= min_q) & (q <= max_q)
        idxs, props = sp_find_peaks(cep[mask], height=thresh)
        real_idxs = np.where(mask)[0][idxs]
        delays = q[real_idxs]
        heights = cep[real_idxs]
        distances = delays * c / 2.0

        # Plot cepstrum
        fig, ax = plt.subplots(facecolor="#19232D")
        ax.set_facecolor("#000000")
        ax.plot(q, cep, color=self.graph_color)
        ax.plot(delays, heights, 'o', color='yellow')
        ax.set_title("Real Cepstrum", color="white")
        ax.set_xlabel("Quefrency (s)", color="white")
        ax.set_ylabel("Amplitude", color="white")
        for spine in ax.spines.values(): spine.set_edgecolor("white")
        canvas = FigureCanvas(fig)

        # Results dialog
        dlg2 = QtWidgets.QDialog(self)
        dlg2.setWindowTitle("Cepstrum Results")
        vbox = QtWidgets.QVBoxLayout(dlg2)
        vbox.addWidget(canvas)

        txt = QtWidgets.QPlainTextEdit()
        txt.setReadOnly(True)
        out = "Delay (s)   Amplitude   Distance (m)\n"
        for d, h, dist in zip(delays, heights, distances):
            out += f"{d:.6f}     {h:.4f}      {dist:.2f}\n"
        txt.setPlainText(out)
        vbox.addWidget(txt)

        btns2 = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        vbox.addWidget(btns2)
        def on_accept():
            # Log each detected echo
            for d, h, dist in zip(delays, heights, distances):
                self.log_measurement_with_project(
                    self.file_name,
                    "Cepstrum",
                    0.0,
                    d, d,
                    0.0,
                    0.0, 0.0,
                    h,
                    False,
                    "",
                    misc=dist
                )
            dlg2.accept()
            plt.close(fig)
        btns2.accepted.connect(on_accept)
        btns2.rejected.connect(lambda: (dlg2.reject(), plt.close(fig)))

        dlg2.exec_()


    def automated_event_clustering_popup(self):
        if self.full_data is None:
            QtWidgets.QMessageBox.critical(self, "Error", "Load a WAV file first.")
            return

        # ─── 1) Parameter dialog with clustering mode ─────────────────────────
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Event Detection & Clustering")
        form = QtWidgets.QFormLayout(dlg)

        # mode radio buttons
        rb_rms = QtWidgets.QRadioButton("Cluster by RMS Voltage"); rb_rms.setChecked(True)
        rb_spl = QtWidgets.QRadioButton("Cluster by SPL (dB)")
        mode_layout = QtWidgets.QHBoxLayout()
        mode_layout.addWidget(rb_rms); mode_layout.addWidget(rb_spl)
        form.addRow("Clustering Mode:", mode_layout)

        thresh_edit = QtWidgets.QLineEdit("0.1")
        form.addRow("Amplitude Threshold:", thresh_edit)
        win_s_edit  = QtWidgets.QLineEdit("0.05")
        form.addRow("Min Event Length (s):", win_s_edit)
        k_edit      = QtWidgets.QLineEdit("3")
        form.addRow("Number of Clusters (k):", k_edit)

        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok |
                                            QtWidgets.QDialogButtonBox.Cancel)
        btn_box.accepted.connect(dlg.accept)
        btn_box.rejected.connect(dlg.reject)
        form.addRow(btn_box)

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        # ─── 2) Parse inputs ───────────────────────────────────────────────────
        try:
            thresh  = float(thresh_edit.text())
            min_len = float(win_s_edit.text())
            k       = int(k_edit.text())
        except ValueError:
            QtWidgets.QMessageBox.critical(self, "Error", "Invalid numeric parameters.")
            return

        data, sr = self.full_data, self.sample_rate
        min_samples = int(min_len * sr)
        use_spl = rb_spl.isChecked()

        # ─── 3) If SPL mode, prompt for hydrophone curve ─────────────────────
        if use_spl:
            curves = list(self.hydrophone_curves.values())
            names  = [c["curve_name"] for c in curves]
            choice, ok = QtWidgets.QInputDialog.getItem(
                self, "Hydrophone Curve", "Select curve for SPL calculation:", names, 0, False
            )
            if not ok:
                return
            curve = next(c for c in curves if c["curve_name"] == choice)
            minf, sens = curve["min_freq"], curve["sensitivity"]

        # ─── 4) Find all segments above threshold ─────────────────────────────
        mask  = np.abs(data) >= thresh
        edges = np.diff(mask.astype(int))
        starts = np.where(edges==1)[0]
        ends   = np.where(edges==-1)[0]
        if mask[0]:   starts = np.insert(starts, 0, 0)
        if mask[-1]:  ends   = np.append(ends, len(mask)-1)
        segs = [(s,e) for s,e in zip(starts, ends) if (e-s)>=min_samples]
        if not segs:
            QtWidgets.QMessageBox.information(self, "No Events", "No events found above threshold.")
            return

        # ─── 5) Extract features ───────────────────────────────────────────────
        feats = []; kept = []
        for s,e in segs:
            seg = data[s:e]
            # convert sample → volts
            if np.issubdtype(self.original_dtype, np.integer):
                vmax = float(self.max_voltage_entry.text())
                conv = vmax / np.iinfo(self.original_dtype).max
            else:
                conv = 1.0
            volt_seg = seg * conv

            vrms = np.sqrt(np.mean(volt_seg**2))
            if use_spl:
                # compute dominant freq
                nfft  = 1<<int(np.ceil(np.log2(len(seg))))
                spec  = np.abs(rfft(seg * np.hanning(len(seg)), n=nfft))
                freqs = rfftfreq(nfft, 1/sr)
                idx   = np.argmax(spec)
                f0    = freqs[idx]
                # lookup sensitivity
                sens_db = sens[int(round(f0)) - minf] if minf <= f0 < minf+len(sens) else sens[0]
                feature = 20*np.log10(vrms) - sens_db
            else:
                feature = vrms

            # also get dom_freq for plotting
            nfft  = 1<<int(np.ceil(np.log2(len(seg))))
            spec  = np.abs(rfft(seg * np.hanning(len(seg)), n=nfft))
            freqs = rfftfreq(nfft, 1/sr)
            dom_f = freqs[np.argmax(spec)]

            feats.append((vrms, feature, dom_f))
            kept.append((s,e))

        if not feats:
            QtWidgets.QMessageBox.information(self, "Filtered Out",
                "No valid events after applying SPL filter." if use_spl else "No events remain.")
            return

        # ─── 6) Cluster & plot ────────────────────────────────────────────────
        X = np.array([[f[1], f[2]] for f in feats])  # feature, dom_f
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=min(k, len(X)), random_state=0).fit_predict(X)

        fig, ax = plt.subplots(facecolor="#000000")
        ax.set_facecolor("#000000")
        ylabel = "SPL (dB)" if use_spl else "RMS Voltage (V)"
        for cl in np.unique(labels):
            mask = labels==cl
            ax.scatter(
                X[mask,1],    # dom_freq
                X[mask,0],    # feature
                label=f"Cluster {cl}", s=20
            )
        ax.set_title(f"Event Clusters (DomFreq vs {ylabel})", color="white")
        ax.set_xlabel("Dominant Frequency (Hz)", color="white")
        ax.set_ylabel(ylabel, color="white")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("white")
        ax.legend(facecolor="#222", labelcolor="white")
        canvas = FigureCanvas(fig)

        # ─── 7) Results dialog w/ table, Save JPG + Accept ──────────────────
        res_dlg = QtWidgets.QDialog(self)
        res_dlg.setWindowTitle("Clustered Events")
        vlay = QtWidgets.QVBoxLayout(res_dlg)
        vlay.addWidget(canvas)

        # build table
        table = QtWidgets.QTableWidget(len(kept), 5)
        table.setHorizontalHeaderLabels(["Start(s)","End(s)", ylabel, "DomFreq(Hz)", "Cluster"])
        for i, ((s,e), (vrms, feat, dom_f), lbl) in enumerate(zip(kept, feats, labels)):
            table.setItem(i, 0, QtWidgets.QTableWidgetItem(f"{s/sr:.3f}"))
            table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{e/sr:.3f}"))
            table.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{feat:.4f}"))
            table.setItem(i, 3, QtWidgets.QTableWidgetItem(f"{dom_f:.2f}"))
            table.setItem(i, 4, QtWidgets.QTableWidgetItem(str(lbl)))
        table.resizeColumnsToContents()
        vlay.addWidget(table)

        # buttons
        h = QtWidgets.QHBoxLayout()
        save_btn   = QtWidgets.QPushButton("Save as JPG")
        accept_btn = QtWidgets.QPushButton("Accept & Log")
        cancel_btn = QtWidgets.QPushButton("Close")
        h.addStretch(); h.addWidget(save_btn); h.addWidget(accept_btn); h.addWidget(cancel_btn)
        vlay.addLayout(h)

        screenshot_path = None
        def on_save():
            nonlocal screenshot_path
            path = self._save_figure_jpg(fig, parent=res_dlg)
            if path:
                screenshot_path = path

        def on_accept():
            if not screenshot_path:
                QtWidgets.QMessageBox.warning(res_dlg, "No Screenshot", "Please Save as JPG first.")
                return

            # loop and log
            if use_spl:
                # insert into spl_calculations
                conn = sqlite3.connect(DB_FILENAME); cur = conn.cursor()
                for (s,e), (vrms, feat, dom_f), lbl in zip(kept, feats, labels):
                    start, end = s/sr, e/sr
                    spl_val = float(feat)
                    cur.execute("""
                        INSERT INTO spl_calculations
                        (file_name, voltage_log_id, hydrophone_curve, target_frequency,
                        rms_voltage, spl, start_time, end_time, window_length,
                        max_voltage, bandwidth, screenshot)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                    """, (
                        self.file_name, None, curve["curve_name"], dom_f,
                        vrms, spl_val, start, end, end-start,
                        0.0, 0.0, screenshot_path
                    ))
                conn.commit(); conn.close()

            else:
                # insert into measurements via log_measurement
                for (s,e), (vrms, feat, dom_f), lbl in zip(kept, feats, labels):
                    start, end = s/sr, e/sr
                    self.log_measurement_with_project(
                        self.file_name,
                        "Event Cluster RMS",
                        dom_f,
                        start, end, end-start,
                        0.0, 0.0,
                        vrms,
                        False,
                        screenshot_path
                    )

            QtWidgets.QMessageBox.information(res_dlg, "Logged", "Cluster results logged successfully.")
            res_dlg.accept()

        save_btn.clicked.connect(on_save)
        accept_btn.clicked.connect(on_accept)
        cancel_btn.clicked.connect(res_dlg.reject)

        res_dlg.resize(800,600)
        res_dlg.exec_()
        plt.close(fig)





    def active_sonar_popup(self):
        
        def compute_vrms(signal):
            return np.sqrt(np.mean(np.square(signal)))

        def dominant_freq(signal):
            freqs = fft.rfftfreq(len(signal), d=1/self.sample_rate)
            spectrum = np.abs(fft.rfft(signal))
            return freqs[np.argmax(spectrum)]

        # Validate preconditions
        if not self.fft_mode or self.fft_start_time is None:
            QtWidgets.QMessageBox.information(self, "Active Sonar",
                "Please switch to FFT mode and select your ping region first.")
            return

        sr = self.sample_rate
        win_len = float(self.fft_length_entry.text())
        t0 = self.fft_start_time
        idx0 = int(t0 * sr)
        idx1 = idx0 + int(win_len * sr)
        ping_template = self.full_data[idx0:idx1]

        # Detect all pulses in the full recording
        corr_full = np.correlate(self.full_data, ping_template, mode='valid')
        times_full = np.arange(len(corr_full)) / sr

        pulse_thresh, ok = QtWidgets.QInputDialog.getDouble(
            self, "Pulse Detection Threshold", "Correlation threshold for pulses:",
            value=float(np.max(corr_full)) * 0.5
        )
        if not ok: return

        min_spacing = win_len  # seconds between pulses
        pulse_peaks, _ = sp_find_peaks(corr_full, height=pulse_thresh, distance=int(min_spacing * sr))
        pulse_times = times_full[pulse_peaks]

        # Create screenshot directory
        screenshots_dir = os.path.join(os.getcwd(), "screenshots")
        os.makedirs(screenshots_dir, exist_ok=True)

        # Open DB connection
        conn = sqlite3.connect("your_measurement_log.db")
        cursor = conn.cursor()
        log_entry_id = self.current_log_id

        results = []

        # Analyze each pulse and its echoes
        for i, pt in enumerate(pulse_times):
            idx_start = int(pt * sr + len(ping_template))
            idx_end = int(pulse_times[i+1] * sr) if i+1 < len(pulse_times) else len(self.full_data)
            if idx_end <= idx_start:
                continue

            segment = self.full_data[idx_start:idx_end]
            seg_corr = np.correlate(segment, ping_template, mode='valid')
            seg_times = np.arange(len(seg_corr)) / sr + pt + win_len

            echo_thresh = np.max(seg_corr) * 0.5
            echo_peaks, _ = sp_find_peaks(seg_corr, height=echo_thresh)

            echoes = []
            for ep in echo_peaks:
                etime = seg_times[ep]
                idx_echo = int(etime * sr)
                echo_win = self.full_data[idx_echo:idx_echo + len(ping_template)]
                if len(echo_win) < len(ping_template):
                    continue
                ef = dominant_freq(echo_win)
                dist = (etime - pt) * self.sound_speed / 2
                vrms_echo = compute_vrms(echo_win)
                echoes.append((etime, ef, dist, vrms_echo))

            # Pulse analysis
            pulse_idx = int(pt * sr)
            pulse_signal = self.full_data[pulse_idx:pulse_idx + len(ping_template)]
            f0 = dominant_freq(pulse_signal)
            vrms_pulse = compute_vrms(pulse_signal)

            # Generate screenshot
            img_name = f"pulse_{i+1:03d}.png"
            img_path = os.path.join(screenshots_dir, img_name)

            fig, ax = plt.subplots(figsize=(10, 4), facecolor="#19232D")
            ax.set_facecolor("#000")
            ax.plot(self.time_vector, self.full_data, color="#555555", label="Raw Signal")
            ax.axvline(pt, color="cyan", linestyle="--", label="Pulse")

            for j, (et, _, _, _) in enumerate(echoes):
                ax.axvline(et, color="yellow", linestyle="--", label="Echo" if j == 0 else "")

            ax.set_xlim(pt - 0.05, echoes[-1][0] + 0.05 if echoes else pt + 0.2)
            ax.set_title(f"Pulse {i+1} with Echoes", color="white")
            ax.set_xlabel("Time (s)", color="white")
            ax.set_ylabel("Amplitude", color="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("white")
            ax.legend()
            fig.tight_layout()

            canvas = AggCanvas(fig)
            canvas.print_png(img_path)
            plt.close(fig)

            # Log pulse
            cursor.execute("""
                INSERT INTO measurements (
                    log_entry_id, measurement_label, target_frequency,
                    measured_voltage, window_size, screenshot, misc
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                log_entry_id, f"Pulse {i+1}", f0,
                vrms_pulse, win_len, img_path, None
            ))

            cursor.execute("""
                INSERT INTO measurements (
                    log_entry_id, measurement_label, target_frequency,
                    measured_voltage, window_size, screenshot, misc
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                log_entry_id, f"Echo {j+1} Pulse {i+1}", ef,
                vrms_echo, win_len, img_path, f"{dist:.2f} m"
            ))

            results.append((pt, f0, vrms_pulse, echoes))

        conn.commit()
        conn.close()

        # Display summary in dialog
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Active Sonar Pulse Grouped Echoes")
        layout = QtWidgets.QVBoxLayout(dlg)

        fig, ax = plt.subplots(facecolor="#19232D")
        ax.set_facecolor("#000")
        ax.plot(times_full, corr_full, color=self.graph_color)
        ax.plot(pulse_times, corr_full[pulse_peaks], 'o', color='cyan')
        ax.set_title("Pulse Cross-Correlation", color="white")
        ax.set_xlabel("Time (s)", color="white")
        ax.set_ylabel("Correlation", color="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

        txt = QtWidgets.QPlainTextEdit()
        txt.setReadOnly(True)
        for i, (pt, f0, vrms_pulse, echoes) in enumerate(results):
            txt.appendPlainText(f"Pulse {i+1} @ {pt:.3f} s | {f0:.1f} Hz | Vrms: {vrms_pulse:.4f} V")
            for j, (et, ef, dist, vrms_echo) in enumerate(echoes):
                txt.appendPlainText(f"  Echo {j+1}: {et:.3f} s | {ef:.1f} Hz | {dist:.2f} m | Vrms: {vrms_echo:.4f} V")
            txt.appendPlainText("")
        layout.addWidget(txt)

        btn = QtWidgets.QPushButton("Close")
        btn.clicked.connect(dlg.accept)
        layout.addWidget(btn, alignment=QtCore.Qt.AlignRight)

        dlg.resize(900, 800)
        dlg.exec_()
        plt.close(fig)

    ## Database Tools


