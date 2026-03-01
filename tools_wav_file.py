#!/usr/bin/env python3
"""
WAV File Tools — methods for MainWindow mixin
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

class WavFileToolsMixin:
    """Mixin class providing all WAV File Tools for self."""

    def normalize_file(self):
        """
        Scale the current WAV so that its maximum absolute sample
        hits full‐scale, then save to analysis/…_normalized.wav and reload.
        """
        if self.full_data is None:
            QtWidgets.QMessageBox.critical(self, "Error", "No file loaded.")
            return

        data = self.full_data
        # Determine full‐scale for this dtype
        if np.issubdtype(self.original_dtype, np.integer):
            max_val = np.iinfo(self.original_dtype).max
        else:
            # for floats assume normalized to ±1.0
            max_val = 1.0

        peak = np.max(np.abs(data))
        if peak == 0:
            QtWidgets.QMessageBox.information(self, "Normalize", "Audio is silent, nothing to do.")
            return

        factor = max_val / peak
        normed = data * factor

        # Clip & cast back to original dtype
        if np.issubdtype(self.original_dtype, np.integer):
            normed = np.clip(normed, -max_val, max_val).astype(self.original_dtype)
        else:
            normed = normed.astype(np.float64)

        # Save into analysis subfolder
        out_dir = self._project_subdir("modified") or os.path.join(os.path.dirname(self.current_file_path), "analysis")
        os.makedirs(out_dir, exist_ok=True)
        base, ext = os.path.splitext(self.file_name)
        new_name = f"{base}_normalized{ext}"
        new_path = os.path.join(out_dir, new_name)

        try:
            wavfile.write(new_path, self.sample_rate, normed)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to write normalized WAV:\n{e}")
            return

        QtWidgets.QMessageBox.information(
            self, "Normalize Complete",
            f"Normalized file saved as:\n{new_name}\nReloading..."
        )
        self.load_wav_file(new_path)


    def remove_dc_offset_popup(self):
        if self.full_data is None:
            QtWidgets.QMessageBox.critical(self, "Error", "Load a WAV file first.")
            return

        data = self.full_data
        sr = self.sample_rate

        # Remove DC offset
        dc_offset = np.mean(data, axis=0)
        data_dc = data - dc_offset

        # Ensure correct dtype and clipping for integer formats
        if hasattr(self, 'wav_dtype') and np.issubdtype(self.wav_dtype, np.integer):
            info = np.iinfo(self.wav_dtype)
            data_dc = np.clip(data_dc, info.min, info.max)
            data_dc = data_dc.astype(self.wav_dtype)

        # Create output path
        base, ext = os.path.splitext(self.file_name)
        outpath = base + "_dc.wav"

        try:
            wavfile.write(outpath, sr, data_dc)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Write Error", f"Failed to save DC-removed WAV: {e}")
            return

        # Open the new file
        self.load_wav_file(outpath)
        QtWidgets.QMessageBox.information(self, "DC Offset Removed", f"Saved DC-removed WAV as:\n{outpath}")


    def denoise_file(self):
        if self.full_data is None:
            QtWidgets.QMessageBox.critical(self, "Error", "No file loaded.")
            return
        try:
            denoised_data = wiener(self.full_data)
            if np.issubdtype(self.original_dtype, np.integer):
                max_val = np.iinfo(self.original_dtype).max
                min_val = np.iinfo(self.original_dtype).min
                denoised_data = np.clip(denoised_data, min_val, max_val)
                denoised_data = denoised_data.astype(self.original_dtype)
            else:
                denoised_data = denoised_data.astype(np.float64)

            directory = os.path.dirname(self.current_file_path)
            new_folder = os.path.join(directory, "analysis")
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)
            base, _ = os.path.splitext(self.file_name)
            new_filename = base + "_denoised.wav"
            new_filepath = os.path.join(new_folder, new_filename)
            wavfile.write(new_filepath, self.sample_rate, denoised_data)
            QtWidgets.QMessageBox.information(
                self, "Denoising Complete",
                f"Denoised file saved as {new_filename}. It will now be loaded."
            )
            self.load_wav_file(new_filepath)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error during denoising: {e}")

    # -------------------------------------------------
    # 3) UPDATED trim_file IMPLEMENTATION
    # -------------------------------------------------

    def trim_file(self):
        """
        Opens TrimDialog so user can specify:
          1) seconds to remove from the start
          2) seconds to remove from the end
        Then writes a new WAV with both ends trimmed.
        """
        if self.full_data is None:
            QtWidgets.QMessageBox.critical(self, "Error", "No file loaded.")
            return

        max_duration = self.full_time[-1] if self.full_time is not None else 0.0
        dlg = TrimDialog(self, max_duration=max_duration)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        try:
            start_trim_s, end_trim_s = dlg.values()
        except ValueError:
            QtWidgets.QMessageBox.critical(self, "Error", "Please enter valid numbers.")
            return

        # Validate the requested trim amounts
        if start_trim_s < 0 or end_trim_s < 0:
            QtWidgets.QMessageBox.critical(self, "Error", "Trim values must be non‐negative.")
            return
        if start_trim_s + end_trim_s >= max_duration:
            QtWidgets.QMessageBox.critical(
                self, "Error",
                "Trim amounts are too large (sum exceeds or equals file duration)."
            )
            return

        # Compute sample indices to keep
        start_idx = int(start_trim_s * self.sample_rate)
        end_idx   = len(self.full_data) - int(end_trim_s * self.sample_rate)

        if start_idx >= end_idx:
            QtWidgets.QMessageBox.critical(
                self, "Error",
                "After trimming, no data would remain."
            )
            return

        # Slice out the trimmed portion
        trimmed = self.full_data[start_idx:end_idx]

        # Convert back to original dtype if needed
        if np.issubdtype(self.original_dtype, np.integer):
            max_val = np.iinfo(self.original_dtype).max
            min_val = np.iinfo(self.original_dtype).min
            trimmed = np.clip(trimmed, min_val, max_val).astype(self.original_dtype)
        else:
            trimmed = trimmed.astype(np.float64)

        # Save out a new file into a project-managed subfolder
        out_dir = self._project_subdir("modified") or os.path.join(os.path.dirname(self.current_file_path), "analysis")
        os.makedirs(out_dir, exist_ok=True)

        base, ext = os.path.splitext(self.file_name)
        new_filename = f"{base}_trimmed_{start_trim_s:.2f}–{end_trim_s:.2f}{ext}"
        new_filepath = os.path.join(out_dir, new_filename)

        try:
            wavfile.write(new_filepath, self.sample_rate, trimmed)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to write trimmed WAV:\n{e}")
            return

        QtWidgets.QMessageBox.information(
            self, "Trim Complete",
            f"Trimmed file saved as:\n{new_filename}\n\n"
            "It will now be loaded."
        )
        # Re‐load the newly trimmed file into the GUI
        self.load_wav_file(new_filepath)




    def downsample_bit_depth_popup(self):
        """
        Bulk‐convert one or more WAV files to a lower PCM bit depth.
        If only one file is selected, it will automatically load the new file.
        Requires the 'soundfile' library.
        """
        try:
            import soundfile as sf
        except ImportError:
            QtWidgets.QMessageBox.critical(
                self, "Missing Dependency",
                "The 'soundfile' library is required.\n"
                "Install it with: pip install soundfile"
            )
            return

        # Bit‐depth mapping for display
        bits_map = {
            'PCM_U8': 8, 'PCM_16': 16, 'PCM_24': 24,
            'PCM_32': 32, 'FLOAT': 32, 'DOUBLE': 64
        }
        # Allowed downsample targets
        candidates = [('PCM_U8', 8), ('PCM_16', 16), ('PCM_24', 24)]

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Downsample Bit Depth (Batch)")
        dlg.resize(450, 350)
        layout = QtWidgets.QVBoxLayout(dlg)

        # Settings form
        form = QtWidgets.QFormLayout()
        target_cb = QtWidgets.QComboBox()
        for subtype, bits in candidates:
            target_cb.addItem(f"{bits}-bit ({subtype})", userData=subtype)
        form.addRow("Target Depth:", target_cb)

        dir_le = QtWidgets.QLineEdit(os.getcwd())
        form.addRow("Output Directory:", dir_le)
        layout.addLayout(form)

        # File selection
        file_h = QtWidgets.QHBoxLayout()
        select_btn = QtWidgets.QPushButton("Select WAV Files…")
        file_h.addWidget(select_btn)
        file_h.addStretch()
        layout.addLayout(file_h)

        file_list = QtWidgets.QListWidget()
        layout.addWidget(file_list)

        # Convert action
        convert_btn = QtWidgets.QPushButton("Convert Selected Files")
        layout.addWidget(convert_btn, alignment=QtCore.Qt.AlignRight)

        batch_files = []

        def select_files():
            nonlocal batch_files
            files, _ = QtWidgets.QFileDialog.getOpenFileNames(
                dlg, "Select WAV Files", "", "WAV Files (*.wav)"
            )
            if files:
                batch_files = files
                file_list.clear()
                file_list.addItems([os.path.basename(f) for f in files])
        select_btn.clicked.connect(select_files)

        def do_convert():
            subtype = target_cb.currentData()
            bit = bits_map.get(subtype, "")
            out_dir = dir_le.text().strip()
            if not os.path.isdir(out_dir):
                QtWidgets.QMessageBox.warning(dlg, "Invalid Directory",
                                            "Please choose a valid output directory.")
                return
            if not batch_files:
                QtWidgets.QMessageBox.warning(dlg, "No Files Selected",
                                            "Please select one or more WAV files.")
                return

            errors = []
            written = []
            for path in batch_files:
                try:
                    data, fs = sf.read(path, always_2d=False)
                    base = os.path.splitext(os.path.basename(path))[0]
                    outp = os.path.join(out_dir, f"{base}_{bit}bit.wav")
                    sf.write(outp, data, fs, subtype=subtype)
                    written.append(outp)
                except Exception as e:
                    errors.append(f"{os.path.basename(path)}: {e}")

            if errors:
                QtWidgets.QMessageBox.warning(dlg, "Conversion Errors",
                                            "\n".join(errors))
            else:
                QtWidgets.QMessageBox.information(
                    dlg, "Done", f"Converted {len(written)} file(s) to {bit}-bit."
                )
                # If exactly one file was converted, load it automatically
                if len(written) == 1:
                    new_file = written[0]
                    # Assume your app has a method `load_wav_file(path)`:
                    try:
                        self.load_wav_file(new_file)
                        QtWidgets.QMessageBox.information(
                            self, "Loaded", f"Loaded {os.path.basename(new_file)}"
                        )
                    except AttributeError:
                        # Fallback: set file_name and re-run your open logic
                        self.file_name = new_file
                        QtWidgets.QMessageBox.information(
                            self, "Ready", f"Ready to analyze {os.path.basename(new_file)}"
                        )
            dlg.accept()

        convert_btn.clicked.connect(do_convert)
        dlg.exec_()



    def wav_merge_popup(self, start_tab: str = "merge"):
        """
        WAV Combine Tool with tabs for concatenation and channel stacking.

        - Merge tab: add/remove/reorder WAVs, optional resample/mix, optional gaps
        - Stack tab: stack files as separate channels in one multichannel WAV
        - Single dialog so related tools live together
        """
        from PyQt5 import QtWidgets, QtCore
        import os, re, json, math
        import numpy as np
        import soundfile as sf

        try:
            from scipy.signal import resample_poly
            _SCIPY_OK = True
        except Exception:
            _SCIPY_OK = False

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Combine WAV Files")
        dlg.setStyleSheet("background:#19232D; color:white;")
        dlg.setMinimumSize(950, 640)

        tabs = QtWidgets.QTabWidget()

        # ---------- Merge Tab ----------
        merge_tab = QtWidgets.QWidget()
        merge_layout = QtWidgets.QVBoxLayout(merge_tab)
        merge_layout.setContentsMargins(10, 10, 10, 10)
        merge_layout.setSpacing(10)

        top = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("Add WAVs"); add_btn.setStyleSheet("background:#3E6C8A;color:white;padding:6px 10px;border-radius:4px;")
        rm_btn  = QtWidgets.QPushButton("Remove Selected"); rm_btn.setStyleSheet("background:#3E6C8A;color:white;padding:6px 10px;border-radius:4px;")
        clear_btn = QtWidgets.QPushButton("Clear"); clear_btn.setStyleSheet("background:#3E6C8A;color:white;padding:6px 10px;border-radius:4px;")
        up_btn = QtWidgets.QPushButton("↑"); up_btn.setStyleSheet("background:#315870;color:white;padding:6px 10px;border-radius:4px;width:28px;")
        dn_btn = QtWidgets.QPushButton("↓"); dn_btn.setStyleSheet("background:#315870;color:white;padding:6px 10px;border-radius:4px;width:28px;")
        sort_name_btn = QtWidgets.QPushButton("Sort by Name")
        sort_time_btn = QtWidgets.QPushButton("Sort by Modified Time")
        top.addWidget(add_btn); top.addWidget(rm_btn); top.addWidget(clear_btn)
        top.addSpacing(15); top.addWidget(up_btn); top.addWidget(dn_btn)
        top.addSpacing(8); top.addWidget(sort_name_btn); top.addWidget(sort_time_btn); top.addStretch()
        merge_layout.addLayout(top)

        file_list = QtWidgets.QListWidget(); file_list.setStyleSheet("QListWidget{background:#0F1A22; color:white; border:1px solid #2D3E4F;} QListWidget::item:selected{background:#2F4A60;}"); file_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        merge_layout.addWidget(file_list, stretch=1)

        opts = QtWidgets.QGridLayout(); opts.setVerticalSpacing(8); opts.setHorizontalSpacing(10)

        enforce_chk = QtWidgets.QCheckBox("Enforce identical SR/channels (faster streaming)"); enforce_chk.setChecked(False)
        opts.addWidget(enforce_chk, 0, 0, 1, 2)

        sr_label = QtWidgets.QLabel("Target SR (Hz) – blank = match first:"); sr_label.setStyleSheet("color:#B0BEC5;")
        sr_edit  = QtWidgets.QLineEdit(""); sr_edit.setPlaceholderText("e.g., 48000"); sr_edit.setFixedWidth(120)
        opts.addWidget(sr_label, 1, 0); opts.addWidget(sr_edit, 1, 1)

        ch_label = QtWidgets.QLabel("Channel handling:"); ch_label.setStyleSheet("color:#B0BEC5;")
        ch_mode  = QtWidgets.QComboBox(); ch_mode.addItems(["Keep (no mix)", "Mono (mix)", "Stereo (mix)" ]); ch_mode.setFixedWidth(150)
        opts.addWidget(ch_label, 2, 0); opts.addWidget(ch_mode, 2, 1)

        subtype_label = QtWidgets.QLabel("Output subtype:"); subtype_label.setStyleSheet("color:#B0BEC5;")
        subtype_cb = QtWidgets.QComboBox(); subtype_cb.addItems(["Match first file", "PCM_16", "FLOAT"]); subtype_cb.setFixedWidth(150)
        opts.addWidget(subtype_label, 3, 0); opts.addWidget(subtype_cb, 3, 1)

        write_map_chk = QtWidgets.QCheckBox("Write sections map JSON"); write_map_chk.setChecked(True)
        load_after_chk = QtWidgets.QCheckBox("Load merged WAV after saving"); load_after_chk.setChecked(False)
        opts.addWidget(write_map_chk, 4, 0, 1, 2); opts.addWidget(load_after_chk, 5, 0, 1, 2)

        add_gap_chk = QtWidgets.QCheckBox("Add silence gap between files")
        add_gap_chk.setChecked(False)
        gap_spin = QtWidgets.QDoubleSpinBox()
        gap_spin.setDecimals(3)
        gap_spin.setRange(0.0, 3600.0)
        gap_spin.setSingleStep(0.1)
        gap_spin.setValue(0.5)
        gap_spin.setEnabled(False)
        gap_spin.setSuffix(" s")
        gap_layout = QtWidgets.QHBoxLayout()
        gap_layout.addWidget(add_gap_chk)
        gap_layout.addWidget(gap_spin)
        gap_layout.addStretch()
        opts.addLayout(gap_layout, 6, 0, 1, 2)

        merge_layout.addLayout(opts)

        prog = QtWidgets.QProgressBar(); prog.setRange(0, 100); prog.setValue(0); prog.setTextVisible(True); prog.setStyleSheet("QProgressBar{border:1px solid #2D3E4F; border-radius:4px; text-align:center;} QProgressBar::chunk{background-color:#3E6C8A;}")
        merge_layout.addWidget(prog)

        btns = QtWidgets.QHBoxLayout(); btns.setSpacing(10)
        merge_btn   = QtWidgets.QPushButton("Merge Files"); merge_btn.setStyleSheet("background:#2E8BC0;color:white;padding:8px 16px;border-radius:4px;font-weight:bold;")
        cancel_btn  = QtWidgets.QPushButton("Close"); cancel_btn.setStyleSheet("background:#455A64;color:white;padding:8px 16px;border-radius:4px;")
        btns.addStretch(); btns.addWidget(merge_btn); btns.addWidget(cancel_btn)
        merge_layout.addLayout(btns)

        def _add_files(paths):
            for p in paths:
                if not p:
                    continue
                it = QtWidgets.QListWidgetItem(p)
                file_list.addItem(it)

        def on_add():
            files, _ = QtWidgets.QFileDialog.getOpenFileNames(dlg, "Select WAV files", "", "WAV Files (*.wav *.wave)")
            _add_files(files)

        def on_remove():
            for it in file_list.selectedItems():
                row = file_list.row(it)
                file_list.takeItem(row)

        def on_clear():
            file_list.clear()

        def on_up():
            rows = [file_list.row(it) for it in file_list.selectedItems()]
            rows.sort()
            for r in rows:
                if r <= 0:
                    continue
                it = file_list.takeItem(r)
                file_list.insertItem(r-1, it)
                it.setSelected(True)

        def on_down():
            rows = [file_list.row(it) for it in file_list.selectedItems()]
            rows.sort(reverse=True)
            for r in rows:
                if r >= file_list.count()-1:
                    continue
                it = file_list.takeItem(r)
                file_list.insertItem(r+1, it)
                it.setSelected(True)

        def on_sort_name():
            paths = [file_list.item(i).text() for i in range(file_list.count())]
            paths.sort(key=lambda p: os.path.basename(p).lower())
            file_list.clear(); _add_files(paths)

        def on_sort_time():
            paths = [file_list.item(i).text() for i in range(file_list.count())]
            paths.sort(key=lambda p: os.path.getmtime(p))
            file_list.clear(); _add_files(paths)

        def on_toggle_enforce(state):
            enforce = (state == QtCore.Qt.Checked)
            sr_label.setEnabled(not enforce); sr_edit.setEnabled(not enforce)
            ch_label.setEnabled(not enforce); ch_mode.setEnabled(not enforce)

        def _inspect(path):
            info = sf.info(path)
            return {
                "samplerate": info.samplerate,
                "channels":   info.channels,
                "subtype":    info.subtype,
                "frames":     info.frames
            }

        def _decide_output(files, enforce, target_sr_text, ch_mode_idx, subtype_idx):
            infos = []
            for p in files:
                infos.append(_inspect(p))

            first = infos[0]
            if enforce:
                for meta in infos[1:]:
                    if meta["samplerate"] != first["samplerate"] or meta["channels"] != first["channels"]:
                        raise ValueError("When enforcing, all files must share the first file's sample rate and channel count.")

            target_sr = int(target_sr_text) if target_sr_text.strip() else first["samplerate"]

            if ch_mode_idx == 0:
                target_ch = first["channels"]
                ch_policy = "keep"
            elif ch_mode_idx == 1:
                target_ch = 1
                ch_policy = "mono"
            else:
                target_ch = 2
                ch_policy = "stereo"

            subtype_lookup = {0: "match", 1: "PCM_16", 2: "FLOAT"}
            choice = subtype_lookup.get(subtype_idx, "match")
            out_subtype = first["subtype"] if choice == "match" else choice

            return infos, first, target_sr, target_ch, ch_policy, out_subtype

        def _mix_channels(arr, policy, target_ch):
            if policy == "keep":
                return arr
            if policy == "mono":
                if arr.ndim == 1:
                    return arr[:, None]
                return np.mean(arr, axis=1, keepdims=True)
            if policy == "stereo":
                if arr.ndim == 1:
                    return np.repeat(arr[:, None], 2, axis=1)
                ch = arr.shape[1]
                if ch == 1:
                    return np.repeat(arr, 2, axis=1)
                elif ch >= 2:
                    return arr[:, :2]
            return arr

        def _resample_if_needed(arr, sr_in, sr_out):
            if sr_in == sr_out:
                return arr
            if not _SCIPY_OK:
                raise RuntimeError("Resampling required but SciPy is unavailable.")
            from fractions import Fraction
            frac = Fraction(sr_out, sr_in).limit_denominator(1000)
            up, down = frac.numerator, frac.denominator
            arrf = arr.astype(np.float32, copy=False)
            return resample_poly(arrf, up, down, axis=0)

        def _write_gap(w, target_sr, target_ch, seconds, total_frames_out):
            if seconds <= 0:
                return 0, total_frames_out
            gap_frames = int(round(seconds * target_sr))
            if gap_frames <= 0:
                return 0, total_frames_out
            block = 262144
            wrote = 0
            zero_block = np.zeros((min(block, gap_frames), target_ch), dtype='float32')
            while wrote < gap_frames:
                n = min(block, gap_frames - wrote)
                if zero_block.shape[0] != n:
                    zero_block = np.zeros((n, target_ch), dtype='float32')
                w.write(zero_block)
                wrote += n
                total_frames_out += n
            return gap_frames, total_frames_out

        def _write_stream(files, infos, out_path, target_sr, target_ch, ch_policy, out_subtype, gap_seconds):
            section_map = []
            total_frames_out = 0

            with sf.SoundFile(out_path, mode='w', samplerate=target_sr, channels=target_ch, subtype=out_subtype, format='WAV') as w:
                for idx, (path, meta) in enumerate(zip(files, infos), start=1):
                    prog.setValue(int(100 * (idx-1) / max(1, len(files))))
                    QtWidgets.QApplication.processEvents()

                    need_resample = (meta["samplerate"] != target_sr)
                    need_mix = (meta["channels"] != target_ch) if ch_policy == "keep" else (ch_policy in ("mono", "stereo"))

                    if not need_resample:
                        with sf.SoundFile(path) as r:
                            section_map.append({
                                "file": path,
                                "start_frame": int(total_frames_out),
                                "start_time_s": total_frames_out / float(target_sr),
                                "frames_in": int(len(r)),
                                "sr_in": int(r.samplerate),
                                "ch_in": int(r.channels)
                            })
                            block = 262144
                            while True:
                                data = r.read(block, dtype='float32', always_2d=True)
                                if data.size == 0:
                                    break
                                if need_mix:
                                    data = _mix_channels(data, ch_policy, target_ch)
                                w.write(data)
                                total_frames_out += data.shape[0]
                    else:
                        with sf.SoundFile(path) as r:
                            data = r.read(dtype='float32', always_2d=True)
                        if data.size == 0:
                            continue
                        if ch_policy in ("mono", "stereo"):
                            data = _mix_channels(data, ch_policy, target_ch)
                        data = _resample_if_needed(data, meta["samplerate"], target_sr)

                        section_map.append({
                            "file": path,
                            "start_frame": int(total_frames_out),
                            "start_time_s": total_frames_out / float(target_sr),
                            "frames_in": int(meta["frames"]),
                            "sr_in": int(meta["samplerate"]),
                            "ch_in": int(meta["channels"])
                        })
                        w.write(data)
                        total_frames_out += data.shape[0]

                    if gap_seconds > 0 and idx < len(files):
                        gframes, total_frames_out = _write_gap(w, target_sr, target_ch, gap_seconds, total_frames_out)
                        section_map[-1]["gap_after_frames"] = int(gframes)
                        section_map[-1]["gap_after_seconds"] = gframes / float(target_sr)

            prog.setValue(100)
            return section_map, total_frames_out

        def on_toggle_gap(state):
            gap_spin.setEnabled(state == QtCore.Qt.Checked)

        def on_merge():
            files = [file_list.item(i).text() for i in range(file_list.count())]
            if not files:
                QtWidgets.QMessageBox.information(dlg, "WAV Merge", "Add some WAV files first.")
                return

            try:
                infos, first, target_sr, target_ch, ch_policy, out_subtype = _decide_output(
                    files,
                    enforce_chk.isChecked(),
                    sr_edit.text(),
                    ch_mode.currentIndex(),
                    subtype_cb.currentIndex()
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(dlg, "Parameter Error", str(e))
                return

            out_path, _ = QtWidgets.QFileDialog.getSaveFileName(dlg, "Save merged WAV", "", "WAV (*.wav)")
            if not out_path:
                return
            if not out_path.lower().endswith(".wav"):
                out_path += ".wav"

            gap_seconds = float(gap_spin.value()) if add_gap_chk.isChecked() else 0.0

            try:
                section_map, total_frames = _write_stream(
                    files, infos, out_path, target_sr, target_ch, ch_policy, out_subtype, gap_seconds
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(dlg, "Merge failed", str(e))
                return

            if write_map_chk.isChecked():
                try:
                    with open(out_path + ".sections.json", "w") as fh:
                        json.dump({
                            "samplerate": target_sr,
                            "channels": target_ch,
                            "subtype": out_subtype,
                            "total_frames": int(total_frames),
                            "sections": section_map
                        }, fh, indent=2)
                except Exception as e:
                    QtWidgets.QMessageBox.warning(
                        dlg, "Mapping not saved", f"Merge ok, but failed to write mapping JSON:\n{e}"
                    )

            QtWidgets.QMessageBox.information(
                dlg,
                "Merge complete",
                f"Merged {len(files)} files →\n{out_path}\n\n"
                f"Duration: {total_frames/float(target_sr):.2f} s\n"
                f"SR: {target_sr} Hz, Channels: {target_ch}, Subtype: {out_subtype}"
            )

            if load_after_chk.isChecked():
                try:
                    if hasattr(self, "load_wav_file"):
                        self.load_wav_file(out_path)
                    elif hasattr(self, "open_wav") and callable(self.open_wav):
                        self.open_wav(out_path)
                except Exception as e:
                    QtWidgets.QMessageBox.warning(
                        dlg,
                        "Loaded with warning",
                        f"Merged file saved, but auto-load hit an issue:\n{e}"
                    )

        add_btn.clicked.connect(on_add)
        rm_btn.clicked.connect(on_remove)
        clear_btn.clicked.connect(on_clear)
        up_btn.clicked.connect(on_up)
        dn_btn.clicked.connect(on_down)
        sort_name_btn.clicked.connect(on_sort_name)
        sort_time_btn.clicked.connect(on_sort_time)
        enforce_chk.stateChanged.connect(on_toggle_enforce)
        add_gap_chk.stateChanged.connect(on_toggle_gap)
        merge_btn.clicked.connect(on_merge)
        cancel_btn.clicked.connect(dlg.reject)

        tabs.addTab(merge_tab, "Concatenate")

        # ---------- Stack Tab ----------
        stack_tab = QtWidgets.QWidget()
        stack_layout = QtWidgets.QVBoxLayout(stack_tab)
        stack_layout.setContentsMargins(10, 10, 10, 10)
        stack_layout.setSpacing(10)

        s_top = QtWidgets.QHBoxLayout()
        s_add_btn = QtWidgets.QPushButton("Add WAVs")
        s_remove_btn = QtWidgets.QPushButton("Remove Selected")
        s_up_btn = QtWidgets.QPushButton("↑")
        s_down_btn = QtWidgets.QPushButton("↓")
        for btn in (s_add_btn, s_remove_btn, s_up_btn, s_down_btn):
            btn.setFixedHeight(28)
        s_top.addWidget(s_add_btn)
        s_top.addWidget(s_remove_btn)
        s_top.addSpacing(8)
        s_top.addWidget(s_up_btn)
        s_top.addWidget(s_down_btn)
        s_top.addStretch()
        stack_layout.addLayout(s_top)

        s_file_list = QtWidgets.QListWidget()
        s_file_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        s_file_list.setStyleSheet("QListWidget{background:#111;color:white;border:1px solid #333;}")
        stack_layout.addWidget(s_file_list, 1)

        s_opts = QtWidgets.QFormLayout()
        pad_chk = QtWidgets.QCheckBox("Pad shorter files with silence (otherwise truncate)")
        pad_chk.setChecked(True)
        s_opts.addRow(pad_chk)

        subtype_cb = QtWidgets.QComboBox()
        subtype_cb.addItems(["Match first file", "PCM_16", "FLOAT"])
        s_opts.addRow("Output subtype", subtype_cb)

        out_path_edit = QtWidgets.QLineEdit()
        browse_btn = QtWidgets.QPushButton("Save as…")
        browse_btn.setFixedWidth(80)
        path_row = QtWidgets.QHBoxLayout()
        path_row.addWidget(out_path_edit)
        path_row.addWidget(browse_btn)
        s_opts.addRow("Output file", path_row)

        load_after_chk = QtWidgets.QCheckBox("Load combined WAV after saving")
        load_after_chk.setChecked(False)
        s_opts.addRow(load_after_chk)

        stack_layout.addLayout(s_opts)

        s_prog = QtWidgets.QProgressBar()
        s_prog.setRange(0, 100)
        s_prog.setValue(0)
        stack_layout.addWidget(s_prog)

        build_btn = QtWidgets.QPushButton("Combine")
        build_btn.setFixedHeight(32)
        build_btn.setStyleSheet("background:#2E8BC0;color:white;")
        stack_layout.addWidget(build_btn, alignment=QtCore.Qt.AlignRight)

        def _s_add_files(paths):
            for p in paths:
                if not p:
                    continue
                s_file_list.addItem(p)

        def _add_files_dialog():
            files, _ = QtWidgets.QFileDialog.getOpenFileNames(dlg, "Select WAV files", "", "WAV Files (*.wav *.wave)")
            _s_add_files(files)

        def _remove():
            for it in s_file_list.selectedItems():
                s_file_list.takeItem(s_file_list.row(it))

        def _move(delta):
            rows = [s_file_list.row(it) for it in s_file_list.selectedItems()]
            rows.sort(reverse=(delta > 0))
            new_sel = []
            for r0 in rows:
                r = max(0, min(s_file_list.count() - 1, r0 + delta))
                if r == r0:
                    new_sel.append(r)
                    continue
                it = s_file_list.takeItem(r0)
                s_file_list.insertItem(r, it)
                new_sel.append(r)
            s_file_list.clearSelection()
            for r in new_sel:
                s_file_list.item(r).setSelected(True)

        def _browse_out():
            path, _ = QtWidgets.QFileDialog.getSaveFileName(dlg, "Save combined WAV", "combined.wav", "WAV Files (*.wav)")
            if path:
                out_path_edit.setText(path)

        s_add_btn.clicked.connect(_add_files_dialog)
        s_remove_btn.clicked.connect(_remove)
        s_up_btn.clicked.connect(lambda: _move(-1))
        s_down_btn.clicked.connect(lambda: _move(1))
        browse_btn.clicked.connect(_browse_out)

        def _combine():
            files = [s_file_list.item(i).text() for i in range(s_file_list.count())]
            if not files:
                QtWidgets.QMessageBox.warning(dlg, "No files", "Add at least one WAV file.")
                return

            info_first = sf.info(files[0])
            sr = info_first.samplerate
            subtype_first = info_first.subtype
            datas = []
            lengths = []
            total_ch = 0

            for idx, path in enumerate(files):
                try:
                    data, fs = sf.read(path, always_2d=True, dtype='float64')
                except Exception as e:
                    QtWidgets.QMessageBox.critical(dlg, "Read error", f"{os.path.basename(path)}: {e}")
                    return
                if fs != sr:
                    QtWidgets.QMessageBox.critical(dlg, "Sample rate mismatch", "All files must share the same sample rate.")
                    return
                datas.append(data)
                lengths.append(data.shape[0])
                total_ch += data.shape[1]
                s_prog.setValue(int((idx + 1) / max(1, len(files)) * 40))
                QtWidgets.QApplication.processEvents()

            if not datas:
                return

            target_len = max(lengths) if pad_chk.isChecked() else min(lengths)
            if target_len <= 0:
                QtWidgets.QMessageBox.critical(dlg, "Invalid length", "Unable to determine output length.")
                return

            out = np.zeros((target_len, total_ch), dtype=np.float64)
            ch_ofs = 0
            for idx, data in enumerate(datas):
                frames = data.shape[0]
                take = min(frames, target_len)
                out[:take, ch_ofs:ch_ofs + data.shape[1]] = data[:take]
                if pad_chk.isChecked() and frames < target_len:
                    out[frames:target_len, ch_ofs:ch_ofs + data.shape[1]] = 0.0
                ch_ofs += data.shape[1]
                s_prog.setValue(40 + int((idx + 1) / max(1, len(datas)) * 40))
                QtWidgets.QApplication.processEvents()

            subtype_choice = subtype_cb.currentIndex()
            if subtype_choice == 0:
                subtype = subtype_first
            elif subtype_choice == 1:
                subtype = "PCM_16"
            else:
                subtype = "FLOAT"

            out_path = out_path_edit.text().strip()
            if not out_path:
                base = os.path.splitext(os.path.basename(files[0]))[0]
                out_path = os.path.join(os.path.dirname(files[0]), f"{base}_stacked.wav")
                out_path_edit.setText(out_path)

            try:
                sf.write(out_path, out, sr, subtype=subtype)
            except Exception as e:
                QtWidgets.QMessageBox.critical(dlg, "Write error", str(e))
                return

            s_prog.setValue(100)
            QtWidgets.QMessageBox.information(
                dlg, "Saved", f"Combined file saved to:\n{out_path}"
            )

            if load_after_chk.isChecked():
                try:
                    self.load_wav_file(out_path)
                except Exception:
                    pass

        build_btn.clicked.connect(_combine)

        tabs.addTab(stack_tab, "Stack Channels")

        layout = QtWidgets.QVBoxLayout(dlg)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(tabs)

        close_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        close_box.rejected.connect(dlg.reject)
        close_box.accepted.connect(dlg.accept)
        layout.addWidget(close_box)

        if start_tab == "stack":
            tabs.setCurrentWidget(stack_tab)
        else:
            tabs.setCurrentWidget(merge_tab)

        dlg.exec_()


    def wav_stack_channels_popup(self):
        # Backwards-compatible entry point to open the stack tab directly.
        self.wav_merge_popup(start_tab="stack")
    # ---------------------
    # SPL Tab Methods
    # ---------------------


    def scale_wav_popup(self):
        """
        Scale WAV amplitude (single file or batch) by Percent or dB.
        - Single tab:
            * Uses currently loaded file by default (or let user pick one)
            * Live peak/clip preview
            * Saves a new WAV (preserving dtype; clips ints)
            * Optional: load the scaled file after saving
        - Batch tab:
            * Add multiple WAVs (or a whole folder)
            * Global scale factor (percent or dB)
            * Preview predicted peaks + clip risk per file
            * Choose output folder (or use input folders)
            * Scales all, shows per-file status, progress bar
        """
        from PyQt5 import QtWidgets, QtCore
        import numpy as np, os
        from scipy.io import wavfile

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Scale WAV Amplitude")
        dlg.setStyleSheet("background-color:#19232D; color:white;")
        dlg.setWindowState(dlg.windowState() | QtCore.Qt.WindowMaximized)
        outer = QtWidgets.QVBoxLayout(dlg)

        tabs = QtWidgets.QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #333; }
            QTabBar::tab { background:#2A3A4A; color:white; padding:6px 12px; }
            QTabBar::tab:selected { background:#3E6C8A; }
        """)
        outer.addWidget(tabs)

        # =========================
        # Shared helpers
        # =========================
        def read_info(path):
            """Return (sr, dtype, peak_abs) without changing the data."""
            sr, data = wavfile.read(path, mmap=True)
            dt = data.dtype
            if data.ndim > 1:
                pk = float(np.max(np.abs(data), axis=0).max())
            else:
                pk = float(np.max(np.abs(data)))
            return sr, dt, pk

        def load_full(path):
            sr, data = wavfile.read(path, mmap=True)
            return sr, data, data.dtype

        def apply_scale(data, factor, dt):
            if np.issubdtype(dt, np.integer):
                info = np.iinfo(dt)
                return np.clip(data.astype(np.float64) * factor, info.min, info.max).round().astype(dt)
            else:
                return (data.astype(np.float64) * factor).astype(dt)

        def factor_from_controls(is_db_mode, pct_val, db_val):
            return (10.0 ** (db_val / 20.0)) if is_db_mode else (pct_val / 100.0)

        # =====================================================================================
        # Tab 1: Single
        # =====================================================================================
        single = QtWidgets.QWidget(); tabs.addTab(single, "Single")
        sv = QtWidgets.QVBoxLayout(single)

        # Source row
        src_row = QtWidgets.QHBoxLayout()
        src_row.addWidget(QtWidgets.QLabel("Source:"))
        loaded = getattr(self, "full_data", None) is not None
        current_path = getattr(self, "current_file_path", "")
        src_label = QtWidgets.QLabel(os.path.basename(current_path) if loaded and current_path else "<no file>")
        src_row.addWidget(src_label)
        pick_btn = QtWidgets.QPushButton("Choose File…")
        pick_btn.setStyleSheet("background-color:#3E6C8A; color:white;")
        src_row.addWidget(pick_btn); src_row.addStretch()
        sv.addLayout(src_row)

        # Mode row
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("Scale mode:"))
        rb_pct = QtWidgets.QRadioButton("Percent"); rb_db = QtWidgets.QRadioButton("dB")
        rb_pct.setChecked(True)
        mode_row.addWidget(rb_pct); mode_row.addWidget(rb_db)

        pct_spin = QtWidgets.QDoubleSpinBox(); pct_spin.setRange(0.01, 10000.0); pct_spin.setDecimals(2); pct_spin.setSingleStep(5.0); pct_spin.setValue(100.0)
        db_spin  = QtWidgets.QDoubleSpinBox(); db_spin.setRange(-120.0, 60.0);   db_spin.setDecimals(2);  db_spin.setSingleStep(0.5); db_spin.setValue(0.0)
        db_spin.setEnabled(False)
        mode_row.addWidget(QtWidgets.QLabel("Percent:")); mode_row.addWidget(pct_spin)
        mode_row.addWidget(QtWidgets.QLabel("dB:"));      mode_row.addWidget(db_spin)
        mode_row.addStretch()
        sv.addLayout(mode_row)

        # Options
        opt_row = QtWidgets.QHBoxLayout()
        load_after_cb = QtWidgets.QCheckBox("Load scaled file after saving"); load_after_cb.setChecked(True)
        opt_row.addWidget(load_after_cb); opt_row.addStretch()
        sv.addLayout(opt_row)

        # Preview group
        preview = QtWidgets.QGroupBox("Preview (peak and clip risk)")
        pv = QtWidgets.QFormLayout(preview)
        s_dtype = QtWidgets.QLabel("—"); s_pk_in = QtWidgets.QLabel("—"); s_pk_out = QtWidgets.QLabel("—"); s_clip = QtWidgets.QLabel("—")
        pv.addRow("Input dtype:", s_dtype); pv.addRow("Input peak:", s_pk_in); pv.addRow("Predicted peak:", s_pk_out); pv.addRow("Clip risk:", s_clip)
        sv.addWidget(preview)

        # Buttons
        s_btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        s_btns.button(QtWidgets.QDialogButtonBox.Ok).setText("Scale && Save…")
        sv.addWidget(s_btns)

        # Single-working state
        s_state = {
            "path": current_path if (loaded and current_path) else None,
            "sr":   self.sample_rate if loaded else None,
            "data": self.full_data if loaded else None,
            "dtype": getattr(self, "original_dtype", None) if loaded else None
        }

        def s_update_preview():
            data = s_state["data"]; dt = s_state["dtype"]
            if data is None or dt is None:
                s_dtype.setText("—"); s_pk_in.setText("—"); s_pk_out.setText("—"); s_clip.setText("—"); return
            s_dtype.setText(str(dt))
            if data.ndim > 1:
                pk = float(np.max(np.abs(data), axis=0).max())
            else:
                pk = float(np.max(np.abs(data)))
            s_pk_in.setText(f"{pk:.6g}")
            fac = factor_from_controls(rb_db.isChecked(), pct_spin.value(), db_spin.value())
            pred = pk * fac; s_pk_out.setText(f"{pred:.6g}")
            if np.issubdtype(dt, np.integer):
                info = np.iinfo(dt); s_clip.setText("YES" if (pred > info.max or pred < info.min) else "No")
            else:
                s_clip.setText("Possible (>1.0)" if pred > 1.0 else "No")

        def s_pick():
            p, _ = QtWidgets.QFileDialog.getOpenFileName(dlg, "Select WAV", "", "WAV Files (*.wav)")
            if not p: return
            try:
                sr, data, dt = load_full(p)
            except Exception as e:
                QtWidgets.QMessageBox.critical(dlg, "Read Error", str(e)); return
            s_state.update(path=p, sr=sr, data=data, dtype=dt)
            src_label.setText(os.path.basename(p))
            s_update_preview()

        pick_btn.clicked.connect(s_pick)

        def s_mode_toggle():
            if rb_pct.isChecked():
                pct_spin.setEnabled(True); db_spin.setEnabled(False)
            else:
                pct_spin.setEnabled(False); db_spin.setEnabled(True)
            s_update_preview()

        rb_pct.toggled.connect(s_mode_toggle); rb_db.toggled.connect(s_mode_toggle)
        pct_spin.valueChanged.connect(lambda _: s_update_preview())
        db_spin.valueChanged.connect(lambda _: s_update_preview())

        if s_state["data"] is None:
            QtWidgets.QMessageBox.information(dlg, "Select Source", "No WAV loaded. Click 'Choose File…' to pick a source.")
        s_update_preview()

        def s_do_scale():
            if s_state["data"] is None:
                QtWidgets.QMessageBox.warning(dlg, "No Source", "Pick or load a WAV."); return
            fac = factor_from_controls(rb_db.isChecked(), pct_spin.value(), db_spin.value())
            in_path = s_state["path"]; sr = s_state["sr"]; data = s_state["data"]; dt = s_state["dtype"]

            base_dir = os.path.dirname(in_path)
            base, ext = os.path.splitext(os.path.basename(in_path))
            suffix = f"_scaled_{db_spin.value():.2f}dB" if rb_db.isChecked() else f"_scaled_{pct_spin.value():.2f}pct"
            out_suggest = os.path.join(base_dir, base + suffix + ext)
            out_path, _ = QtWidgets.QFileDialog.getSaveFileName(dlg, "Save Scaled WAV", out_suggest, "WAV Files (*.wav)")
            if not out_path: return

            try:
                scaled = apply_scale(data, fac, dt)
                wavfile.write(out_path, sr, scaled)
            except Exception as e:
                QtWidgets.QMessageBox.critical(dlg, "Write Error", str(e)); return

            QtWidgets.QMessageBox.information(dlg, "Done", f"Saved:\n{out_path}")
            if load_after_cb.isChecked():
                try: self.load_wav_file(out_path)
                except Exception as e: QtWidgets.QMessageBox.warning(dlg, "Load Warning", f"Saved, but failed to load:\n{e}")
            dlg.accept()

        s_btns.accepted.connect(s_do_scale)
        s_btns.rejected.connect(dlg.reject)

        # =====================================================================================
        # Tab 2: Batch
        # =====================================================================================
        batch = QtWidgets.QWidget(); tabs.addTab(batch, "Batch")
        bv = QtWidgets.QVBoxLayout(batch)

        # Controls row
        ctrl = QtWidgets.QHBoxLayout()
        add_files = QtWidgets.QPushButton("Add Files…"); add_folder = QtWidgets.QPushButton("Add Folder…")
        rm_sel = QtWidgets.QPushButton("Remove Selected"); clear_btn = QtWidgets.QPushButton("Clear List")
        for b in (add_files, add_folder, rm_sel, clear_btn):
            b.setStyleSheet("background-color:#3E6C8A; color:white;")
            ctrl.addWidget(b)
        ctrl.addStretch()
        bv.addLayout(ctrl)

        # Mode row (global for batch)
        bmode = QtWidgets.QHBoxLayout()
        bmode.addWidget(QtWidgets.QLabel("Scale mode:"))
        b_rb_pct = QtWidgets.QRadioButton("Percent"); b_rb_db = QtWidgets.QRadioButton("dB")
        b_rb_pct.setChecked(True)
        bmode.addWidget(b_rb_pct); bmode.addWidget(b_rb_db)

        b_pct = QtWidgets.QDoubleSpinBox(); b_pct.setRange(0.01, 10000.0); b_pct.setDecimals(2); b_pct.setSingleStep(5.0); b_pct.setValue(100.0)
        b_db  = QtWidgets.QDoubleSpinBox(); b_db.setRange(-120.0, 60.0); b_db.setDecimals(2); b_db.setSingleStep(0.5); b_db.setValue(0.0); b_db.setEnabled(False)
        bmode.addWidget(QtWidgets.QLabel("Percent:")); bmode.addWidget(b_pct)
        bmode.addWidget(QtWidgets.QLabel("dB:"));      bmode.addWidget(b_db)
        bmode.addStretch()
        bv.addLayout(bmode)

        # Output options
        outrow = QtWidgets.QHBoxLayout()
        use_input_cb = QtWidgets.QCheckBox("Save next to input files"); use_input_cb.setChecked(True)
        outrow.addWidget(use_input_cb)
        outrow.addSpacing(20)
        outrow.addWidget(QtWidgets.QLabel("Output folder:"))
        out_edit = QtWidgets.QLineEdit(); out_edit.setPlaceholderText("Choose folder…"); out_edit.setEnabled(False); out_edit.setMinimumWidth(320)
        out_btn  = QtWidgets.QPushButton("Browse…"); out_btn.setEnabled(False); out_btn.setStyleSheet("background-color:#3E6C8A; color:white;")
        outrow.addWidget(out_edit); outrow.addWidget(out_btn); outrow.addStretch()
        bv.addLayout(outrow)

        def on_use_input_changed(chk):
            en = not chk
            out_edit.setEnabled(en); out_btn.setEnabled(en)
        use_input_cb.toggled.connect(on_use_input_changed)
        def pick_out_dir():
            d = QtWidgets.QFileDialog.getExistingDirectory(dlg, "Select Output Folder", "")
            if d: out_edit.setText(d)
        out_btn.clicked.connect(pick_out_dir)

        # File table
        table = QtWidgets.QTableWidget(0, 6)
        table.setHorizontalHeaderLabels(["File", "dtype", "Input peak", "Pred peak", "Clip risk", "Status"])
        table.setStyleSheet("QTableWidget{background:#19232D;color:white;gridline-color:#444;}")
        table.horizontalHeader().setStretchLastSection(True)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        bv.addWidget(table)

        # Progress + Run
        run_row = QtWidgets.QHBoxLayout()
        prog = QtWidgets.QProgressBar(); prog.setRange(0, 100); prog.setValue(0)
        run_btn = QtWidgets.QPushButton("Scale Files"); run_btn.setStyleSheet("background-color:#3E6C8A; color:white; padding:6px 12px;")
        run_row.addWidget(prog); run_row.addStretch(); run_row.addWidget(run_btn)
        bv.addLayout(run_row)

        # Batch state: list of dicts {path, sr, dtype, in_peak}
        items = []

        def add_paths(paths):
            nonlocal items
            new = 0
            for p in paths:
                if not p.lower().endswith(".wav"): continue
                if any(it["path"] == p for it in items): continue
                try:
                    sr, dt, pk = read_info(p)
                except Exception:
                    continue
                items.append({"path": p, "sr": sr, "dtype": dt, "in_peak": pk})
                new += 1
            if new:
                refresh_table()

        def refresh_table():
            table.setRowCount(len(items))
            fac = factor_from_controls(b_rb_db.isChecked(), b_pct.value(), b_db.value())
            for r, it in enumerate(items):
                def _it(txt): 
                    q = QtWidgets.QTableWidgetItem(txt)
                    q.setForeground(QtCore.Qt.white); 
                    return q
                pred = it["in_peak"] * fac
                if np.issubdtype(it["dtype"], np.integer):
                    info = np.iinfo(it["dtype"])
                    clip = "YES" if (pred > info.max or pred < info.min) else "No"
                else:
                    clip = "Possible (>1.0)" if pred > 1.0 else "No"
                table.setItem(r, 0, _it(os.path.basename(it["path"])))
                table.setItem(r, 1, _it(str(it["dtype"])))
                table.setItem(r, 2, _it(f"{it['in_peak']:.6g}"))
                table.setItem(r, 3, _it(f"{pred:.6g}"))
                table.setItem(r, 4, _it(clip))
                table.setItem(r, 5, _it("Queued"))
            table.resizeColumnsToContents()

        def on_add_files():
            paths, _ = QtWidgets.QFileDialog.getOpenFileNames(dlg, "Add WAV Files", "", "WAV Files (*.wav)")
            if paths: add_paths(paths)

        def on_add_folder():
            d = QtWidgets.QFileDialog.getExistingDirectory(dlg, "Select Folder", "")
            if not d: return
            found = []
            for root, _, files in os.walk(d):
                for f in files:
                    if f.lower().endswith(".wav"):
                        found.append(os.path.join(root, f))
            add_paths(found)

        def on_rm_sel():
            sel = table.selectionModel().selectedRows()
            if not sel: return
            rows = sorted([ix.row() for ix in sel], reverse=True)
            for r in rows: items.pop(r)
            refresh_table()

        def on_clear():
            items.clear(); refresh_table()

        add_files.clicked.connect(on_add_files)
        add_folder.clicked.connect(on_add_folder)
        rm_sel.clicked.connect(on_rm_sel)
        clear_btn.clicked.connect(on_clear)

        def b_mode_toggle():
            if b_rb_pct.isChecked():
                b_pct.setEnabled(True); b_db.setEnabled(False)
            else:
                b_pct.setEnabled(False); b_db.setEnabled(True)
            refresh_table()

        b_rb_pct.toggled.connect(b_mode_toggle); b_rb_db.toggled.connect(b_mode_toggle)
        b_pct.valueChanged.connect(lambda _: refresh_table())
        b_db.valueChanged.connect(lambda _: refresh_table())

        def do_batch_scale():
            if not items:
                QtWidgets.QMessageBox.information(dlg, "Nothing to do", "Add WAV files first."); return
            if not use_input_cb.isChecked():
                out_dir = out_edit.text().strip()
                if not out_dir:
                    QtWidgets.QMessageBox.warning(dlg, "Output folder", "Pick an output folder or use input folders."); return
                if not os.path.isdir(out_dir):
                    QtWidgets.QMessageBox.warning(dlg, "Output folder", "Output folder does not exist."); return

            fac = factor_from_controls(b_rb_db.isChecked(), b_pct.value(), b_db.value())
            prog.setValue(0)
            n = len(items)
            # scale each file
            for i, it in enumerate(items):
                # compute destination path
                base, ext = os.path.splitext(os.path.basename(it["path"]))
                suffix = f"_scaled_{b_db.value():.2f}dB" if b_rb_db.isChecked() else f"_scaled_{b_pct.value():.2f}pct"
                dest_dir = os.path.dirname(it["path"]) if use_input_cb.isChecked() else out_edit.text().strip()
                os.makedirs(dest_dir, exist_ok=True)
                out_path = os.path.join(dest_dir, base + suffix + ext)

                status = "OK"
                try:
                    sr, data, dt = load_full(it["path"])
                    scaled = apply_scale(data, fac, dt)
                    wavfile.write(out_path, sr, scaled)
                except Exception as e:
                    status = f"ERR: {e}"

                # update table status row
                table.item(i, 5).setText(status)
                prog.setValue(int((i+1) / n * 100))
                QtWidgets.QApplication.processEvents()

            QtWidgets.QMessageBox.information(dlg, "Done", f"Processed {n} file(s).")
            # Don't auto-load anything in batch mode.

        run_btn.clicked.connect(do_batch_scale)

        dlg.exec_()


    def wav_playlist_builder_tool(self):
        """
        One-function WAV playlist builder (self-contained):
        - GUI to add/order files with per-item Repeat and Gap(s)
        - Forces user to choose an output .wav path (can't OK without it)
        - Optional target sample rate, float32 output, peak normalization
        - Writes a single WAV assembled from the configured playlist
        """
        from PyQt5 import QtWidgets, QtCore
        import numpy as np
        from scipy.io import wavfile
        from scipy.signal import resample_poly
        import os

        # --------- inner helpers (audio) ----------
        def _to_float32(x: np.ndarray) -> np.ndarray:
            if np.issubdtype(x.dtype, np.floating):
                return x.astype(np.float32, copy=False)
            info = np.iinfo(x.dtype) if np.issubdtype(x.dtype, np.integer) else None
            scale = max(abs(info.min), info.max) if info is not None else 1.0
            return x.astype(np.float32) / scale

        def _from_float32(x: np.ndarray, want_float32: bool) -> np.ndarray:
            if want_float32:
                return x.astype(np.float32, copy=False)
            y = np.clip(x, -1.0, 1.0) * 32767.0
            return y.astype(np.int16)

        def _match_channels(x: np.ndarray, target: int) -> np.ndarray:
            if x.ndim == 1:
                x = x[:, None]
            ch = x.shape[1]
            if ch == target:
                return x
            if ch < target:
                reps = int(target // ch) + (1 if target % ch else 0)
                return np.tile(x, (1, reps))[:, :target]
            return x[:, :target]

        def _resample_audio(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
            if sr_in == sr_out:
                return x
            from math import gcd
            g = gcd(sr_in, sr_out)
            up = sr_out // g
            down = sr_in // g
            if x.ndim == 1:
                return resample_poly(x, up, down).astype(np.float32)
            chans = [resample_poly(x[:, c], up, down).astype(np.float32) for c in range(x.shape[1])]
            L = min(len(ch) for ch in chans)
            chans = [ch[:L] for ch in chans]
            return np.stack(chans, axis=1)

        def _normalize_dbfs(x: np.ndarray, target_db: float) -> np.ndarray:
            if x.size == 0:
                return x
            peak = float(np.max(np.abs(x)))
            if peak <= 0:
                return x
            target_lin = 10.0 ** (target_db / 20.0)
            return (x * (target_lin / peak)).astype(np.float32)

        # --------- inner dialog (scoped to this function) ----------
        class _Dlg(QtWidgets.QDialog):
            def __init__(self, parent=None):
                super().__init__(parent)
                self.setWindowTitle("Build WAV Playlist")
                self.resize(900, 520)
                v = QtWidgets.QVBoxLayout(self)

                # table
                self.table = QtWidgets.QTableWidget(0, 3, self)
                self.table.setHorizontalHeaderLabels(["File", "Repeat", "Gap (s)"])
                self.table.horizontalHeader().setStretchLastSection(True)
                v.addWidget(self.table)

                # row controls
                rowbar = QtWidgets.QHBoxLayout()
                self.add_btn = QtWidgets.QPushButton("Add File(s)…")
                self.rem_btn = QtWidgets.QPushButton("Remove")
                self.up_btn  = QtWidgets.QPushButton("Move Up")
                self.dn_btn  = QtWidgets.QPushButton("Move Down")
                for w in (self.add_btn, self.rem_btn, self.up_btn, self.dn_btn):
                    rowbar.addWidget(w)
                rowbar.addStretch()
                v.addLayout(rowbar)

                # options
                form = QtWidgets.QFormLayout()

                # output path
                self.out_edit = QtWidgets.QLineEdit()
                self.browse_out = QtWidgets.QPushButton("Browse…")
                out_row = QtWidgets.QHBoxLayout()
                out_row.addWidget(self.out_edit)
                out_row.addWidget(self.browse_out)
                out_w = QtWidgets.QWidget()
                out_w.setLayout(out_row)
                form.addRow("Output WAV:", out_w)

                # sample rate
                self.sr_spin = QtWidgets.QSpinBox()
                self.sr_spin.setRange(8000, 384000)
                self.sr_spin.setValue(0)  # 0 = use first file SR
                form.addRow("Sample Rate (0 = first file):", self.sr_spin)

                # float32
                self.float32_cb = QtWidgets.QCheckBox("Write 32-bit float WAV (default: 16-bit PCM)")
                form.addRow(self.float32_cb)

                # normalize
                norm_row = QtWidgets.QHBoxLayout()
                self.norm_enable = QtWidgets.QCheckBox("Normalize to (dBFS):")
                self.norm_spin = QtWidgets.QDoubleSpinBox()
                self.norm_spin.setRange(-36.0, 0.0)
                self.norm_spin.setSingleStep(0.5)
                self.norm_spin.setValue(0)
                norm_row.addWidget(self.norm_enable)
                norm_row.addWidget(self.norm_spin)
                norm_w = QtWidgets.QWidget()
                norm_w.setLayout(norm_row)
                form.addRow(norm_w)

                v.addLayout(form)

                # ok/cancel
                self.btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
                v.addWidget(self.btns)

                # wire
                self.add_btn.clicked.connect(self._add_files)
                self.rem_btn.clicked.connect(self._remove_selected)
                self.up_btn.clicked.connect(self._move_up)
                self.dn_btn.clicked.connect(self._move_down)
                self.browse_out.clicked.connect(self._browse_out)

                # accept handling: enforce output path (no close if empty)
                self.btns.accepted.connect(self._try_accept)
                self.btns.rejected.connect(self.reject)

                # disable OK until path exists
                self._ok = self.btns.button(QtWidgets.QDialogButtonBox.Ok)
                if self._ok:
                    self._ok.setEnabled(False)
                self.out_edit.textChanged.connect(lambda _: self._ok.setEnabled(bool(self.out_edit.text().strip())) if self._ok else None)

            def _add_files(self):
                paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Add WAV Files", "", "WAV Files (*.wav);;All Files (*)")
                for p in paths:
                    r = self.table.rowCount()
                    self.table.insertRow(r)
                    self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(p))
                    it_rep = QtWidgets.QTableWidgetItem("1")
                    it_gap = QtWidgets.QTableWidgetItem("0.0")
                    it_rep.setTextAlignment(QtCore.Qt.AlignCenter)
                    it_gap.setTextAlignment(QtCore.Qt.AlignCenter)
                    self.table.setItem(r, 1, it_rep)
                    self.table.setItem(r, 2, it_gap)

            def _remove_selected(self):
                rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
                for r in rows:
                    self.table.removeRow(r)

            def _move_up(self):
                r = self.table.currentRow()
                if r <= 0: return
                self._swap_rows(r, r-1)
                self.table.setCurrentCell(r-1, self.table.currentColumn())

            def _move_down(self):
                r = self.table.currentRow()
                if r < 0 or r >= self.table.rowCount()-1: return
                self._swap_rows(r, r+1)
                self.table.setCurrentCell(r+1, self.table.currentColumn())

            def _swap_rows(self, a, b):
                for c in range(self.table.columnCount()):
                    ia = self.table.takeItem(a, c)
                    ib = self.table.takeItem(b, c)
                    self.table.setItem(a, c, ib)
                    self.table.setItem(b, c, ia)

            def _browse_out(self):
                path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Output WAV", "", "WAV Files (*.wav)")
                if path:
                    if not path.lower().endswith(".wav"):
                        path += ".wav"
                    self.out_edit.setText(path)

            def playlist(self):
                items = []
                for r in range(self.table.rowCount()):
                    itf = self.table.item(r, 0)
                    itr = self.table.item(r, 1)
                    itg = self.table.item(r, 2)
                    f   = (itf.text().strip() if itf else "")
                    rep = int(itr.text()) if (itr and itr.text().strip()) else 1
                    gap = float(itg.text()) if (itg and itg.text().strip()) else 0.0
                    if f:
                        items.append({"file": f, "repeat": rep, "gap": gap})
                return items

            def _try_accept(self):
                path = self.out_edit.text().strip()
                if not path:
                    QtWidgets.QMessageBox.warning(self, "Missing Output", "Please choose an output WAV path.")
                    self._browse_out()
                    path = self.out_edit.text().strip()
                    if not path:
                        return  # keep dialog open
                if not path.lower().endswith(".wav"):
                    path += ".wav"
                    self.out_edit.setText(path)
                self.accept()

        # --------- show dialog; render if OK ----------
        dlg = _Dlg(self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        items = dlg.playlist()
        if not items:
            QtWidgets.QMessageBox.warning(self, "Empty", "No files in playlist.")
            return

        out_path = dlg.out_edit.text().strip()
        sr = int(dlg.sr_spin.value()) if hasattr(dlg, "sr_spin") else 0
        if sr <= 0:
            sr = None
        want_float32 = dlg.float32_cb.isChecked() if hasattr(dlg, "float32_cb") else False
        norm_db = dlg.norm_spin.value() if (hasattr(dlg, "norm_enable") and dlg.norm_enable.isChecked()) else None

        try:
            # Load first to set SR/CH
            sr0, d0 = wavfile.read(items[0]["file"], mmap=True)
            sr_use = sr or int(sr0)
            x0 = _to_float32(d0)
            ch_use = 1 if x0.ndim == 1 else x0.shape[1]

            segs = []
            for it in items:
                sr_i, data_i = wavfile.read(it["file"], mmap=True)
                x = _to_float32(data_i)
                x = _match_channels(x, ch_use)
                x = _resample_audio(x, int(sr_i), int(sr_use))

                reps = int(it.get("repeat", 1))
                gap_s = float(it.get("gap", 0.0))
                gap = None
                if gap_s > 0.0:
                    gap_len = int(round(sr_use * gap_s))
                    gap = np.zeros((gap_len, ch_use), dtype=np.float32)

                for r in range(max(0, reps)):
                    segs.append(x)
                    if r != reps - 1 and gap is not None and gap.size > 0:
                        segs.append(gap)

            if not segs:
                QtWidgets.QMessageBox.warning(self, "Empty", "Nothing to write.")
                return

            out = np.concatenate(segs, axis=0)
            if norm_db is not None:
                out = _normalize_dbfs(out, float(norm_db))

            audio = _from_float32(out, want_float32)
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            wavfile.write(out_path, int(sr_use), audio)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to build playlist:\n{e}")
            return

        QtWidgets.QMessageBox.information(self, "Done", f"Wrote playlist:\n{out_path}")


    def wav_secure_pack_unpack_tool_async(self):
        """
        PACK + UNPACK with background thread + progress + cancel.
        Streams TAR/LZMA/AES-GCM in chunks to keep UI responsive.
        Suffix (default '.wpk') is auto-appended silently; user never has to type it.
        Shows a clear error when password is wrong (InvalidTag -> friendly message).
        Requires: pip install cryptography
        """
        from PyQt5 import QtWidgets, QtCore
        import os, io, struct, lzma, tarfile, tempfile

        # ---------- Small 2-tab dialog (Pack/Unpack) ----------
        class _Dlg(QtWidgets.QDialog):
            def __init__(self, parent=None):
                super().__init__(parent)
                self.setWindowTitle("Secure WAV Bundle (Pack / Unpack)")
                self.resize(760, 560)
                v = QtWidgets.QVBoxLayout(self)
                self.tabs = QtWidgets.QTabWidget(); v.addWidget(self.tabs)

                # ---- PACK TAB ----
                pack = QtWidgets.QWidget(); self.tabs.addTab(pack, "Pack")
                pv = QtWidgets.QVBoxLayout(pack)
                self.pack_list = QtWidgets.QListWidget(); pv.addWidget(self.pack_list)

                ph = QtWidgets.QHBoxLayout()
                self.pack_add = QtWidgets.QPushButton("Add WAVs…")
                self.pack_rm  = QtWidgets.QPushButton("Remove")
                self.pack_up  = QtWidgets.QPushButton("Up")
                self.pack_dn  = QtWidgets.QPushButton("Down")
                for b in (self.pack_add, self.pack_rm, self.pack_up, self.pack_dn): ph.addWidget(b)
                ph.addStretch(); pv.addLayout(ph)

                formp = QtWidgets.QFormLayout()
                # Output file
                out_row = QtWidgets.QHBoxLayout()
                self.pack_out = QtWidgets.QLineEdit()
                self.pack_browse_out = QtWidgets.QPushButton("Browse…")
                out_row.addWidget(self.pack_out); out_row.addWidget(self.pack_browse_out)
                out_w = QtWidgets.QWidget(); out_w.setLayout(out_row)
                formp.addRow("Output file:", out_w)

                # Proprietary suffix
                self.pack_suffix = QtWidgets.QLineEdit(".wpk")
                formp.addRow("Proprietary suffix:", self.pack_suffix)

                # Compression level
                self.pack_preset = QtWidgets.QSpinBox()
                self.pack_preset.setRange(0,9); self.pack_preset.setValue(6)
                formp.addRow("Compression (LZMA 0–9):", self.pack_preset)

                # Password + confirm
                self.pack_pw1 = QtWidgets.QLineEdit(); self.pack_pw1.setEchoMode(QtWidgets.QLineEdit.Password)
                self.pack_pw2 = QtWidgets.QLineEdit(); self.pack_pw2.setEchoMode(QtWidgets.QLineEdit.Password)
                formp.addRow("Password:", self.pack_pw1)
                formp.addRow("Confirm:",  self.pack_pw2)

                # Iterations
                self.pack_iters = QtWidgets.QSpinBox()
                self.pack_iters.setRange(50_000, 2_000_000); self.pack_iters.setSingleStep(50_000); self.pack_iters.setValue(200_000)
                formp.addRow("PBKDF2 iterations:", self.pack_iters)

                pv.addLayout(formp)

                # ---- UNPACK TAB ----
                unpack = QtWidgets.QWidget(); self.tabs.addTab(unpack, "Unpack")
                uv = QtWidgets.QVBoxLayout(unpack)
                formu = QtWidgets.QFormLayout()

                in_row = QtWidgets.QHBoxLayout()
                self.unpack_in = QtWidgets.QLineEdit()
                self.unpack_browse_in = QtWidgets.QPushButton("Browse…")
                in_row.addWidget(self.unpack_in); in_row.addWidget(self.unpack_browse_in)
                in_w = QtWidgets.QWidget(); in_w.setLayout(in_row)
                formu.addRow("Bundle file:", in_w)

                outd_row = QtWidgets.QHBoxLayout()
                self.unpack_outdir = QtWidgets.QLineEdit()
                self.unpack_browse_outdir = QtWidgets.QPushButton("Browse…")
                outd_row.addWidget(self.unpack_outdir); outd_row.addWidget(self.unpack_browse_outdir)
                outd_w = QtWidgets.QWidget(); outd_w.setLayout(outd_row)
                formu.addRow("Extract to folder:", outd_w)

                self.unpack_pw = QtWidgets.QLineEdit(); self.unpack_pw.setEchoMode(QtWidgets.QLineEdit.Password)
                formu.addRow("Password:", self.unpack_pw)

                self.unpack_overwrite = QtWidgets.QCheckBox("Overwrite existing files")
                formu.addRow(self.unpack_overwrite)

                uv.addLayout(formu)

                # Buttons
                btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
                v.addWidget(btns)

                # wiring pack
                self.pack_add.clicked.connect(self._pack_add)
                self.pack_rm.clicked.connect(self._pack_rm)
                self.pack_up.clicked.connect(self._pack_up)
                self.pack_dn.clicked.connect(self._pack_dn)
                self.pack_browse_out.clicked.connect(self._pack_browse_out)

                # wiring unpack
                self.unpack_browse_in.clicked.connect(self._unpack_browse_in)
                self.unpack_browse_outdir.clicked.connect(self._unpack_browse_outdir)

                # ok/cancel
                btns.accepted.connect(self._try_accept)
                btns.rejected.connect(self.reject)

            # --- PACK handlers ---
            def _pack_add(self):
                paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Add WAV Files", "", "WAV Files (*.wav);;All Files (*)")
                for p in paths: self.pack_list.addItem(p)

            def _pack_rm(self):
                for it in self.pack_list.selectedItems():
                    self.pack_list.takeItem(self.pack_list.row(it))

            def _pack_up(self):
                r = self.pack_list.currentRow()
                if r <= 0: return
                it = self.pack_list.takeItem(r)
                self.pack_list.insertItem(r-1, it)
                self.pack_list.setCurrentRow(r-1)

            def _pack_dn(self):
                r = self.pack_list.currentRow()
                if r < 0 or r >= self.pack_list.count()-1: return
                it = self.pack_list.takeItem(r)
                self.pack_list.insertItem(r+1, it)
                self.pack_list.setCurrentRow(r+1)

            def _pack_browse_out(self):
                # Auto-append suffix right in the file dialog handler (quietly)
                path, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self, "Save Secure Bundle", "", "Secure Bundle (*.wpk);;All Files (*)"
                )
                if path:
                    suf = self.pack_suffix.text().strip() or ".wpk"
                    if not suf.startswith("."): suf = "." + suf
                    if not path.lower().endswith(suf.lower()):
                        path += suf
                    self.pack_out.setText(path)

            # --- UNPACK handlers ---
            def _unpack_browse_in(self):
                path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Secure Bundle", "", "Secure Bundle (*.wpk);;All Files (*)")
                if path: self.unpack_in.setText(path)

            def _unpack_browse_outdir(self):
                path = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose Output Folder", "")
                if path: self.unpack_outdir.setText(path)

            # --- Validate and close ---
            def _try_accept(self):
                if self.tabs.currentIndex() == 0:  # PACK
                    if self.pack_list.count() == 0:
                        QtWidgets.QMessageBox.warning(self, "No files", "Please add at least one WAV file."); return
                    if not self.pack_pw1.text():
                        QtWidgets.QMessageBox.warning(self, "Password", "Please enter a password."); return
                    if self.pack_pw1.text() != self.pack_pw2.text():
                        QtWidgets.QMessageBox.warning(self, "Password", "Passwords do not match."); return
                    op = self.pack_out.text().strip()
                    if not op:
                        QtWidgets.QMessageBox.warning(self, "Missing output", "Please choose an output file path."); return
                    # Silently normalize suffix on OK
                    suf = self.pack_suffix.text().strip() or ".wpk"
                    if not suf.startswith("."): suf = "." + suf
                    if not op.lower().endswith(suf.lower()):
                        op += suf
                        self.pack_out.setText(op)
                    self.accept()
                else:  # UNPACK
                    import os
                    if not os.path.isfile(self.unpack_in.text().strip()):
                        QtWidgets.QMessageBox.warning(self, "Missing input", "Please choose an existing bundle file."); return
                    if not self.unpack_outdir.text().strip():
                        QtWidgets.QMessageBox.warning(self, "Missing output folder", "Please choose an output folder."); return
                    if not self.unpack_pw.text():
                        QtWidgets.QMessageBox.warning(self, "Password", "Please enter a password."); return
                    self.accept()

        # ---------- Worker thread (streams + progress) ----------
        class _Worker(QtCore.QThread):
            progress = QtCore.pyqtSignal(int, str)   # percent, message
            done = QtCore.pyqtSignal(bool, str)      # ok, message

            def __init__(self, mode, args, parent=None):
                super().__init__(parent)
                self.mode = mode  # 'pack' or 'unpack'
                self.args = args
                self._cancel = False

            def cancel(self): self._cancel = True

            def _percent(self, i, n):
                return 0 if n <= 0 else int(min(100, max(0, round(100.0 * i / n))))

            def run(self):
                try:
                    if self.mode == 'pack':
                        self._run_pack()
                    else:
                        self._run_unpack()
                except Exception as e:
                    self.done.emit(False, f"{type(e).__name__}: {e}")

            # ----- streaming PACK -----
            def _run_pack(self):
                import os, struct, lzma, tarfile, tempfile
                from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
                from cryptography.hazmat.primitives import hashes
                from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

                files      = self.args["files"]
                out_path   = self.args["out_path"]
                preset     = self.args["preset"]
                iterations = int(self.args["iterations"])
                password   = self.args["password"].encode("utf-8")

                # total bytes for progress (sum of file sizes)
                total = 0
                sizes = []
                for p in files:
                    try: s = os.path.getsize(p)
                    except Exception: s = 0
                    sizes.append(s); total += s
                done_bytes = 0
                self.progress.emit(0, "Building TAR…")

                # 1) TAR to temp (stream)
                with tempfile.NamedTemporaryFile(delete=False) as tar_tmp:
                    tar_tmp_path = tar_tmp.name
                try:
                    with tarfile.open(tar_tmp_path, mode="w") as tf:
                        for idx, p in enumerate(files):
                            if self._cancel: raise RuntimeError("Cancelled")
                            base = os.path.basename(p)
                            ti = tarfile.TarInfo(name=base)
                            sz = sizes[idx]
                            ti.size = sz
                            ti.mtime = int(os.path.getmtime(p)) if os.path.exists(p) else 0
                            tf.addfile(ti, fileobj=open(p, "rb"))
                            done_bytes += sz
                            self.progress.emit(self._percent(done_bytes, total), f"TAR {idx+1}/{len(files)}")
                    # 2) LZMA compress tar -> temp compressed file
                    self.progress.emit(self._percent(0,1), "Compressing…")
                    with tempfile.NamedTemporaryFile(delete=False) as cmp_tmp:
                        cmp_tmp_path = cmp_tmp.name
                    CHUNK = 1<<20
                    comp = lzma.LZMACompressor(preset=preset)
                    with open(tar_tmp_path, "rb") as fin, open(cmp_tmp_path, "wb") as fout:
                        sz = os.path.getsize(tar_tmp_path)
                        pos = 0
                        while True:
                            if self._cancel: raise RuntimeError("Cancelled")
                            b = fin.read(CHUNK)
                            if not b: break
                            fout.write(comp.compress(b))
                            pos += len(b)
                            self.progress.emit(self._percent(pos, sz), "Compressing…")
                        fout.write(comp.flush())
                    # 3) AES-GCM encrypt compressed temp -> final file
                    self.progress.emit(99, "Encrypting…")
                    from os import urandom
                    salt  = urandom(16)
                    kdf   = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=iterations)
                    key   = kdf.derive(password)
                    nonce = urandom(12)
                    cipher = Cipher(algorithms.AES(key), modes.GCM(nonce))
                    enc = cipher.encryptor()
                    with open(out_path, "wb") as f_out:
                        f_out.write(b"WPK1"); f_out.write(salt); f_out.write(struct.pack(">I", iterations)); f_out.write(nonce)
                        with open(cmp_tmp_path, "rb") as f_in:
                            sz = os.path.getsize(cmp_tmp_path); pos = 0
                            while True:
                                if self._cancel: raise RuntimeError("Cancelled")
                                b = f_in.read(CHUNK)
                                if not b: break
                                ct = enc.update(b)
                                if ct: f_out.write(ct)
                                pos += len(b)
                                self.progress.emit(self._percent(pos, sz), "Encrypting…")
                            f_out.write(enc.finalize())
                            f_out.write(enc.tag)  # tag at end
                    self.progress.emit(100, "Done")
                finally:
                    try: os.remove(tar_tmp_path)
                    except Exception: pass
                    try: os.remove(cmp_tmp_path)
                    except Exception: pass
                self.done.emit(True, f"Created bundle:\n{out_path}")

            # ----- streaming UNPACK -----
            def _run_unpack(self):
                import os, struct, lzma, tarfile, tempfile
                from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
                from cryptography.hazmat.primitives import hashes
                from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
                from cryptography.exceptions import InvalidTag  # << explicit import

                in_path  = self.args["in_path"]
                out_dir  = self.args["out_dir"]
                password = self.args["password"].encode("utf-8")
                overwrite= bool(self.args["overwrite"])

                # read fixed header
                with open(in_path, "rb") as f:
                    hdr = f.read(4+16+4+12)
                    if len(hdr) < 36: raise ValueError("File too short")
                    magic, rest = hdr[:4], hdr[4:]
                    if magic != b"WPK1": raise ValueError("Invalid magic (not WPK1)")
                    salt = rest[:16]; iters = struct.unpack(">I", rest[16:20])[0]; nonce = rest[20:32]
                    ct_and_tag = f.read()

                if len(ct_and_tag) < 16:
                    raise ValueError("Ciphertext/tag missing")

                ct, tag = ct_and_tag[:-16], ct_and_tag[-16:]

                # derive key
                kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=int(iters))
                key = kdf.derive(password)

                # decrypt to temp compressed file
                self.progress.emit(0, "Decrypting…")
                with tempfile.NamedTemporaryFile(delete=False) as cmp_tmp:
                    cmp_tmp_path = cmp_tmp.name
                CHUNK = 1<<20
                cipher = Cipher(algorithms.AES(key), modes.GCM(nonce, tag))
                dec = cipher.decryptor()
                sz = len(ct); pos = 0
                with open(cmp_tmp_path, "wb") as fout:
                    mv = memoryview(ct); off = 0
                    while off < sz:
                        if self._cancel: raise RuntimeError("Cancelled")
                        chunk = mv[off:off+CHUNK]; off += len(chunk)
                        pt = dec.update(chunk)
                        if pt: fout.write(pt)
                        pos += len(chunk)
                        self.progress.emit(self._percent(pos, sz), "Decrypting…")
                    try:
                        fout.write(dec.finalize())
                    except InvalidTag:
                        # clear, user-friendly error
                        raise RuntimeError("Wrong password or file is corrupted (authentication tag mismatch).")

                # decompress LZMA to temp TAR
                self.progress.emit(0, "Decompressing…")
                with tempfile.NamedTemporaryFile(delete=False) as tar_tmp:
                    tar_tmp_path = tar_tmp.name
                with open(cmp_tmp_path, "rb") as fin, open(tar_tmp_path, "wb") as fout:
                    decomp = lzma.LZMADecompressor()
                    while True:
                        if self._cancel: raise RuntimeError("Cancelled")
                        b = fin.read(1<<20)
                        if not b: break
                        fout.write(decomp.decompress(b))
                        self.progress.emit(0, "Decompressing…")

                # extract safely
                self.progress.emit(0, "Extracting…")
                os.makedirs(out_dir, exist_ok=True)
                extracted, skipped = 0, 0
                with tarfile.open(tar_tmp_path, mode="r:*") as tf:
                    members = [m for m in tf.getmembers() if m.isfile()]
                    n = len(members)
                    for i, m in enumerate(members, 1):
                        if self._cancel: raise RuntimeError("Cancelled")
                        base = os.path.basename(m.name)
                        if not base: continue
                        dest = os.path.join(out_dir, base)
                        if (not overwrite) and os.path.exists(dest):
                            skipped += 1
                        else:
                            fobj = tf.extractfile(m)
                            if fobj is None: continue
                            with open(dest, "wb") as out:
                                while True:
                                    chunk = fobj.read(1<<20)
                                    if not chunk: break
                                    out.write(chunk)
                            extracted += 1
                        self.progress.emit(self._percent(i, n), f"Extracting {i}/{n}")

                try: os.remove(cmp_tmp_path)
                except Exception: pass
                try: os.remove(tar_tmp_path)
                except Exception: pass
                self.done.emit(True, f"Unpacked to:\n{out_dir}\n\nExtracted: {extracted}\nSkipped (exists): {skipped}")

        # ----- Show dialog, then run worker with progress -----
        dlg = _Dlg(self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        is_pack = (dlg.tabs.currentIndex() == 0)

        # deps check
        try:
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC  # noqa: F401
            from cryptography.hazmat.primitives import hashes  # noqa: F401
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes  # noqa: F401
        except Exception:
            QtWidgets.QMessageBox.critical(self, "Missing dependency",
                "This feature requires the 'cryptography' package.\n\nTry:\n   pip install cryptography")
            return

        if is_pack:
            files = [dlg.pack_list.item(i).text() for i in range(dlg.pack_list.count())]
            out_path   = dlg.pack_out.text().strip()
            suf        = dlg.pack_suffix.text().strip() or ".wpk"
            if not suf.startswith("."): suf = "." + suf
            # Final safety: silently enforce suffix
            if not out_path.lower().endswith(suf.lower()):
                out_path += suf
            args = dict(files=files,
                        out_path=out_path,
                        preset=int(dlg.pack_preset.value()),
                        iterations=int(dlg.pack_iters.value()),
                        password=dlg.pack_pw1.text())
            worker = _Worker('pack', args, self)
            label = "Packing…"
        else:
            args = dict(in_path=dlg.unpack_in.text().strip(),
                        out_dir=dlg.unpack_outdir.text().strip(),
                        password=dlg.unpack_pw.text(),
                        overwrite=dlg.unpack_overwrite.isChecked())
            worker = _Worker('unpack', args, self)
            label = "Unpacking…"

        # progress dialog
        prog = QtWidgets.QProgressDialog(label, "Cancel", 0, 100, self)
        prog.setWindowTitle(label)
        prog.setWindowModality(QtCore.Qt.ApplicationModal)
        prog.setAutoClose(False); prog.setAutoReset(False)
        prog.show()

        def on_progress(pct, msg):
            prog.setLabelText(f"{label}  {msg}")
            prog.setValue(pct)
            QtWidgets.QApplication.processEvents()

        def on_done(ok, msg):
            prog.reset()
            if ok:
                QtWidgets.QMessageBox.information(self, "Done", msg)
            else:
                QtWidgets.QMessageBox.critical(self, "Error", msg)

        prog.canceled.connect(worker.cancel)
        worker.progress.connect(on_progress)
        worker.done.connect(on_done)
        worker.start()



    def wav_channel_sync_popup(self):
        """
        WAV Channel Sync Tool

        - Full-screen style dialog.
        - Left side: large plot area (stacked waveforms + toolbar).
        - Right side: compact controls (X-axis, reference, per-channel offsets).
        - X-axis is controlled by two spinboxes: X min / X max (seconds).
        - Each channel has:
            • Editable 'Click (s)' field
            • Editable 'Offset vs Ref (s)' field
        - Left-click on each channel to set its click time (updates the field).
        - Offsets (seconds vs reference channel) can be auto-filled for that
          one channel or manually typed into spinboxes.
        - 'Save Graph…' saves the current multi-channel view to JPG/PNG.
        - 'Build Synced WAV' outputs a new multichannel WAV with each selected
          channel time-shifted according to its offset.
        """
        from PyQt5 import QtWidgets, QtCore
        import numpy as np
        import os
        from scipy.io import wavfile
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qt5agg import (
            FigureCanvasQTAgg as FigureCanvas,
            NavigationToolbar2QT as NavToolbar,
        )

        # ---- basic checks ----
        if getattr(self, "full_data", None) is None:
            QtWidgets.QMessageBox.critical(self, "No Data", "Load a WAV file first.")
            return

        sr = int(getattr(self, "sample_rate", 0)) or 0
        if sr <= 0:
            QtWidgets.QMessageBox.critical(self, "No Sample Rate", "Sample rate is not set.")
            return

        data = np.asarray(self.full_data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        n_samples, n_ch_total = data.shape

        # which channels are in play
        try:
            sel_ch = list(self.selected_channel_indices())
        except Exception:
            sel_ch = list(range(n_ch_total))
        sel_ch = [ch for ch in sel_ch if 0 <= ch < n_ch_total]
        if not sel_ch:
            sel_ch = [0]

        num_sel = len(sel_ch)
        total_time = n_samples / float(sr)

        ch_names = getattr(self, "channel_names", [])

        # ---- dialog shell ----
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Channel Sync Tool")
        dlg.setModal(True)
        dlg.setStyleSheet("background:#19232D; color:white;")

        main_layout = QtWidgets.QHBoxLayout(dlg)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, dlg)
        main_layout.addWidget(splitter)

        # =========================
        # LEFT SIDE: GRAPHS
        # =========================
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        # Matplotlib figure (one axis per channel)
        t = np.arange(n_samples, dtype=float) / float(sr)

        plt.close("all")
        fig, axes = plt.subplots(
            num_sel,
            1,
            sharex=True,
            figsize=(10, 6),
            squeeze=False,
        )
        axes = [ax for ax in axes[:, 0]]

        fig.patch.set_facecolor("#19232D")
        for ax in axes:
            ax.set_facecolor("#000000")

        # Colour palette: ch1 = current graph_color, then rotate through color_options
        if hasattr(self, "color_options") and isinstance(self.color_options, dict):
            palette = list(self.color_options.values())
        else:
            palette = [getattr(self, "graph_color", "#03DFE2")]

        base_color = getattr(self, "graph_color", palette[0])
        try:
            base_idx = palette.index(base_color)
        except ValueError:
            base_idx = 0

        max_pts = 250_000
        for idx, ch in enumerate(sel_ch):
            ax = axes[idx]
            y = data[:, ch]
            step = int(max(1, np.ceil(len(y) / max_pts)))
            if step > 1:
                y_plot = y[::step]
                t_plot = t[::step]
            else:
                y_plot = y
                t_plot = t

            col = palette[(base_idx + idx) % len(palette)]

            ax.plot(t_plot, y_plot, lw=0.8, color=col)

            if ch_names and ch < len(ch_names):
                title = ch_names[ch]
            else:
                title = f"Ch {ch+1}"

            ax.set_ylabel(title, color="white")
            ax.tick_params(colors="white")
            for sp in ax.spines.values():
                sp.set_edgecolor("white")

        axes[-1].set_xlabel("Time (s)", color="white")

        canvas = FigureCanvas(fig)
        toolbar = NavToolbar(canvas, left_widget)
        left_layout.addWidget(toolbar)
        left_layout.addWidget(canvas, 1)

        splitter.addWidget(left_widget)

        # =========================
        # RIGHT SIDE: CONTROLS
        # =========================
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(8)

        # Info
        info_lbl = QtWidgets.QLabel(
            "Workflow:\n"
            "  1) Set X min / X max to zoom time.\n"
            "  2) Use 'Click (s)' or left-click on waveforms to mark feature times.\n"
            "  3) Adjust 'Offset vs Ref (s)' per channel.\n"
            "  4) Save graph or build the synced WAV."
        )
        info_lbl.setWordWrap(True)
        info_lbl.setStyleSheet("color:white;")
        right_layout.addWidget(info_lbl)

        # X-axis controls
        x_group = QtWidgets.QGroupBox("Time Axis")
        x_group.setStyleSheet(
            "QGroupBox { border: 1px solid #444; margin-top: 6px; } "
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding:0 3px; }"
        )
        x_layout = QtWidgets.QHBoxLayout(x_group)

        x_layout.addWidget(QtWidgets.QLabel("X min (s):"))
        x_min_spin = QtWidgets.QDoubleSpinBox()
        x_min_spin.setDecimals(6)
        x_min_spin.setRange(0.0, total_time)
        x_min_spin.setSingleStep(max(1.0 / sr, 0.001))
        x_min_spin.setValue(0.0)
        x_layout.addWidget(x_min_spin)

        x_layout.addSpacing(8)

        x_layout.addWidget(QtWidgets.QLabel("X max (s):"))
        x_max_spin = QtWidgets.QDoubleSpinBox()
        x_max_spin.setDecimals(6)
        x_max_spin.setRange(0.0, total_time)
        x_max_spin.setSingleStep(max(1.0 / sr, 0.001))
        x_max_spin.setValue(total_time)
        x_layout.addWidget(x_max_spin)

        right_layout.addWidget(x_group)

        def update_xlim_from_spins():
            tmin = x_min_spin.value()
            tmax = x_max_spin.value()
            if tmax <= tmin:
                return
            for ax in axes:
                ax.set_xlim(tmin, tmax)
            canvas.draw_idle()

        x_min_spin.valueChanged.connect(lambda _: update_xlim_from_spins())
        x_max_spin.valueChanged.connect(lambda _: update_xlim_from_spins())
        update_xlim_from_spins()

        # Reference + buttons
        ref_group = QtWidgets.QGroupBox("Reference & Actions")
        ref_group.setStyleSheet(
            "QGroupBox { border: 1px solid #444; margin-top: 6px; } "
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding:0 3px; }"
        )
        ref_layout = QtWidgets.QVBoxLayout(ref_group)

        ref_row = QtWidgets.QHBoxLayout()
        ref_row.addWidget(QtWidgets.QLabel("Reference channel:"))
        ref_combo = QtWidgets.QComboBox()
        for ch in sel_ch:
            if ch_names and ch < len(ch_names):
                txt = ch_names[ch]
            else:
                txt = f"Ch {ch+1}"
            ref_combo.addItem(txt, ch)
        ref_row.addWidget(ref_combo)
        ref_layout.addLayout(ref_row)

        btn_row = QtWidgets.QHBoxLayout()

        save_graph_btn = QtWidgets.QPushButton("Save Graph…")
        save_graph_btn.setStyleSheet("background:#777;color:white;")
        btn_row.addWidget(save_graph_btn)

        clear_btn = QtWidgets.QPushButton("Clear Marks")
        clear_btn.setStyleSheet("background:#555;color:white;")
        btn_row.addWidget(clear_btn)

        build_btn = QtWidgets.QPushButton("Build Synced WAV")
        build_btn.setStyleSheet("background:#3E6C8A;color:white;font-weight:bold;")
        btn_row.addWidget(build_btn)

        ref_layout.addLayout(btn_row)
        right_layout.addWidget(ref_group)

        # Per-channel offsets in a scroll area
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll_widget = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(4)

        ch_group = QtWidgets.QGroupBox("Channel Clicks & Offsets (vs reference)")
        ch_group.setStyleSheet(
            "QGroupBox { border: 1px solid #444; margin-top: 6px; } "
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding:0 3px; }"
        )
        ch_layout = QtWidgets.QVBoxLayout(ch_group)

        scroll_layout.addWidget(ch_group)
        scroll_layout.addStretch(1)
        scroll.setWidget(scroll_widget)
        right_layout.addWidget(scroll, 1)

        right_layout.addStretch(0)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 3)  # graphs
        splitter.setStretchFactor(1, 1)  # controls

        # ---- Per-channel click times & offsets ----
        click_samples = {ch: None for ch in sel_ch}
        click_times = {ch: None for ch in sel_ch}   # seconds
        click_time_spins = {}
        offset_spins = {}
        offsets_sec = {ch: 0.0 for ch in sel_ch}
        vlines = {ch: None for ch in sel_ch}

        for ch in sel_ch:
            row = QtWidgets.QHBoxLayout()

            # Channel name
            if ch_names and ch < len(ch_names):
                ch_txt = ch_names[ch]
            else:
                ch_txt = f"Ch {ch+1}"
            ch_lbl = QtWidgets.QLabel(ch_txt)
            ch_lbl.setStyleSheet("color:white; font-weight:bold;")
            row.addWidget(ch_lbl)

            # Click spin
            row.addWidget(QtWidgets.QLabel("Click (s):"))
            click_spin = QtWidgets.QDoubleSpinBox()
            click_spin.setDecimals(6)
            click_spin.setRange(0.0, total_time)
            click_spin.setSingleStep(max(1.0 / sr, 0.0001))
            click_spin.setValue(0.0)
            row.addWidget(click_spin)
            click_time_spins[ch] = click_spin

            row.addSpacing(8)

            # Offset spinbox
            row.addWidget(QtWidgets.QLabel("Offset vs Ref (s):"))
            off_spin = QtWidgets.QDoubleSpinBox()
            off_spin.setDecimals(6)
            off_spin.setRange(-total_time, total_time)
            off_spin.setSingleStep(max(1.0 / sr, 0.0001))
            off_spin.setValue(0.0)
            row.addWidget(off_spin)
            offset_spins[ch] = off_spin

            row.addStretch(1)
            ch_layout.addLayout(row)

        # helper: apply offsets_sec -> spin widgets
        def _refresh_offset_spins(block=True):
            for ch in sel_ch:
                spin = offset_spins[ch]
                if block:
                    spin.blockSignals(True)
                spin.setValue(offsets_sec.get(ch, 0.0))
                if block:
                    spin.blockSignals(False)

            # lock reference channel offset to 0 and disable editing
            ref_ch = ref_combo.currentData()
            for ch in sel_ch:
                spin = offset_spins[ch]
                if ch == ref_ch:
                    spin.blockSignals(True)
                    spin.setValue(0.0)
                    spin.blockSignals(False)
                    spin.setEnabled(False)
                    offsets_sec[ch] = 0.0
                else:
                    spin.setEnabled(True)

        def _on_ref_changed(_=None):
            # On ref change, just enforce its offset = 0 (don't touch others).
            ref_ch = ref_combo.currentData()
            if ref_ch in sel_ch:
                offsets_sec[ref_ch] = 0.0
            _refresh_offset_spins(block=True)

        ref_combo.currentIndexChanged.connect(_on_ref_changed)

        # spinbox -> offsets_sec
        def _make_off_spin_handler(ch):
            def _handler(value):
                offsets_sec[ch] = float(value)
            return _handler

        for ch in sel_ch:
            offset_spins[ch].valueChanged.connect(_make_off_spin_handler(ch))

        _on_ref_changed()  # initial

        # ---- Canvas click handling ----
        def _update_click_spin(ch):
            """Update the click spin box for channel ch from click_times[ch]."""
            spin = click_time_spins.get(ch)
            t_ch = click_times.get(ch)
            if spin is None or t_ch is None:
                return
            spin.blockSignals(True)
            spin.setValue(t_ch)
            spin.blockSignals(False)

        def _update_vline(ch):
            """Ensure the yellow line for channel ch matches click_times[ch]."""
            t_ch = click_times.get(ch)
            if t_ch is None:
                return
            ax = axes[sel_ch.index(ch)]
            line = vlines.get(ch)
            if line is None:
                line = ax.axvline(t_ch, color="yellow", lw=1.2)
                vlines[ch] = line
            else:
                line.set_xdata([t_ch, t_ch])

        def _apply_auto_offset_for_channel(ch):
            """If ref has a click time, auto-set this channel's offset from clicks."""
            ref_ch = ref_combo.currentData()
            ref_t = click_times.get(ref_ch)
            t_ch = click_times.get(ch)
            if ref_t is None or t_ch is None or ch == ref_ch:
                return
            offsets_sec[ch] = t_ch - ref_t
            spin = offset_spins.get(ch)
            if spin is not None:
                spin.blockSignals(True)
                spin.setValue(offsets_sec[ch])
                spin.blockSignals(False)

        def _on_click(event):
            if event.button != 1:
                return
            if event.inaxes not in axes:
                return
            if event.xdata is None:
                return

            ax = event.inaxes
            ch_idx = axes.index(ax)
            ch = sel_ch[ch_idx]

            t_click = float(event.xdata)
            s = int(round(t_click * float(sr)))
            s = max(0, min(n_samples - 1, s))
            click_samples[ch] = s
            click_times[ch] = s / float(sr)

            _update_click_spin(ch)
            _update_vline(ch)
            _apply_auto_offset_for_channel(ch)

            canvas.draw_idle()

        canvas.mpl_connect("button_press_event", _on_click)

        # ---- Click spin handlers (manual edit) ----
        def _make_click_spin_handler(ch):
            def _handler(value):
                # treat this as the click time in seconds
                t_ch = float(value)
                t_ch = max(0.0, min(total_time, t_ch))
                click_times[ch] = t_ch
                click_samples[ch] = int(round(t_ch * float(sr)))
                _update_vline(ch)
                _apply_auto_offset_for_channel(ch)
                canvas.draw_idle()
            return _handler

        for ch in sel_ch:
            click_time_spins[ch].valueChanged.connect(_make_click_spin_handler(ch))

        # ---- Clear marks ----
        def _clear_marks():
            for ch in sel_ch:
                click_samples[ch] = None
                click_times[ch] = None
                spin_c = click_time_spins.get(ch)
                if spin_c is not None:
                    spin_c.blockSignals(True)
                    spin_c.setValue(0.0)
                    spin_c.blockSignals(False)
                line = vlines.get(ch)
                if line is not None:
                    try:
                        line.remove()
                    except Exception:
                        pass
                    vlines[ch] = None
                offsets_sec[ch] = 0.0
            _refresh_offset_spins(block=True)
            canvas.draw_idle()

        clear_btn.clicked.connect(_clear_marks)

        # ---- Save Graph ----
        def _save_graph():
            base_path = getattr(self, "current_file_path", "") or getattr(
                self, "wav_file_path", ""
            ) or ""
            if not base_path:
                base_path = getattr(self, "file_name", "waveform.wav")

            base_dir = os.path.dirname(base_path)
            if not base_dir:
                base_dir = os.getcwd()

            root, _ext = os.path.splitext(os.path.basename(base_path))
            default_path = os.path.join(base_dir, f"{root}_channel_sync.png")

            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                dlg,
                "Save Graph",
                default_path,
                "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)",
            )
            if not path:
                return

            try:
                fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
                QtWidgets.QMessageBox.information(
                    dlg, "Saved", f"Graph saved to:\n{path}"
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    dlg, "Save Graph", f"Could not save graph:\n{e}"
                )

        save_graph_btn.clicked.connect(_save_graph)

        # ---- Build Synced WAV ----
        def _build_synced():
            offsets_samples = {
                ch: int(round(float(offsets_sec.get(ch, 0.0)) * float(sr)))
                for ch in sel_ch
            }

            # overlapping index range
            min_n = max(-off for off in offsets_samples.values())
            max_n = min(n_samples - 1 - off for off in offsets_samples.values())
            if max_n <= min_n:
                QtWidgets.QMessageBox.critical(
                    dlg,
                    "Sync Error",
                    "Offsets leave no overlapping region between channels.\n"
                    "Try reducing absolute offsets or checking click positions.",
                )
                return

            out_len = max_n - min_n + 1
            out_ch = len(sel_ch)

            out = np.zeros((out_len, out_ch), dtype=np.float64)
            for j, ch in enumerate(sel_ch):
                off = offsets_samples[ch]
                start = min_n + off
                stop = start + out_len
                out[:, j] = data[start:stop, ch]

            dtype = getattr(self, "wav_dtype", None)
            if dtype is None:
                dtype = getattr(self, "original_dtype", None)
            if dtype is None:
                dtype = np.float64

            if np.issubdtype(dtype, np.integer):
                info = np.iinfo(dtype)
                out = np.clip(out, info.min, info.max).astype(dtype)
            else:
                out = out.astype(np.float64)

            base_dir = os.path.dirname(getattr(self, "current_file_path", "") or "")
            if not base_dir:
                base_dir = os.getcwd()
            out_dir = os.path.join(base_dir, "analysis")
            os.makedirs(out_dir, exist_ok=True)

            base_name = getattr(self, "file_name", "waveform.wav")
            root, ext = os.path.splitext(base_name)
            if not ext:
                ext = ".wav"
            out_name = f"{root}_synced.wav"
            out_path = os.path.join(out_dir, out_name)

            try:
                wavfile.write(out_path, sr, out)
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    dlg, "Write Error", f"Failed to write synced WAV:\n{e}"
                )
                return

            QtWidgets.QMessageBox.information(
                dlg,
                "Channel Sync Complete",
                f"Synced WAV saved as:\n{out_path}\n\n"
                "You can now load it into the main window if desired.",
            )

        build_btn.clicked.connect(_build_synced)

        # Show full-screen-ish and exec
        dlg.showMaximized()
        dlg.exec_()






    def pass_filter_popup(self, low_default="1000", high_default="", order_default="4"):
        """Apply optional high-pass, low-pass, or both in one step."""
        if self.full_data is None:
            QtWidgets.QMessageBox.critical(self, "Error", "No file loaded.")
            return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("High/Low Pass Filter")
        dlg.setModal(True)

        info = QtWidgets.QLabel("Leave a cutoff blank to skip that filter.")
        lowcut_edit = QtWidgets.QLineEdit(str(low_default))
        highcut_edit = QtWidgets.QLineEdit(str(high_default))
        order_edit = QtWidgets.QLineEdit(str(order_default))
        for edit in (lowcut_edit, highcut_edit, order_edit):
            edit.setFixedWidth(100)

        form = QtWidgets.QFormLayout()
        form.addRow(info)
        form.addRow("High-pass cutoff (Hz):", lowcut_edit)
        form.addRow("Low-pass cutoff (Hz):", highcut_edit)
        form.addRow("Filter order:", order_edit)

        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(dlg.accept)
        btn_box.rejected.connect(dlg.reject)

        vbox = QtWidgets.QVBoxLayout(dlg)
        vbox.addLayout(form)
        vbox.addWidget(btn_box)

        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            try:
                order = int(order_edit.text())
            except ValueError:
                QtWidgets.QMessageBox.critical(self, "Error", "Invalid filter order.")
                return

            lowcut = lowcut_edit.text().strip()
            highcut = highcut_edit.text().strip()
            low_val = float(lowcut) if lowcut else None
            high_val = float(highcut) if highcut else None

            if low_val is None and high_val is None:
                QtWidgets.QMessageBox.information(dlg, "No cutoff", "Enter at least one cutoff to apply a filter.")
                return

            nyq = 0.5 * self.sample_rate
            if low_val is not None:
                if low_val <= 0 or low_val >= nyq:
                    QtWidgets.QMessageBox.critical(dlg, "Cutoff Error", f"High-pass must be between 0 and Nyquist ({nyq:.1f} Hz).")
                    return
            if high_val is not None:
                if high_val <= 0 or high_val >= nyq:
                    QtWidgets.QMessageBox.critical(dlg, "Cutoff Error", f"Low-pass must be between 0 and Nyquist ({nyq:.1f} Hz).")
                    return
            if low_val is not None and high_val is not None and low_val >= high_val:
                QtWidgets.QMessageBox.critical(dlg, "Cutoff Error", "High-pass cutoff must be below the low-pass cutoff.")
                return

            data = self.full_data
            try:
                if low_val is not None:
                    normal_low = low_val / nyq
                    sos = butter(order, normal_low, btype="high", output="sos")
                    data = safe_sosfiltfilt(sos, data)
                if high_val is not None:
                    normal_high = high_val / nyq
                    sos = butter(order, normal_high, btype="low", output="sos")
                    data = safe_sosfiltfilt(sos, data)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Filter Error", str(e))
                return

            self.apply_filtered_data(data)


    def highpass_popup(self):
        # Backwards-compatible entry point; defaults to only a high-pass cutoff.
        self.pass_filter_popup(low_default="1000", high_default="")


    def lowpass_popup(self):
        # Backwards-compatible entry point; defaults to only a low-pass cutoff.
        self.pass_filter_popup(low_default="", high_default="1000")
    

    def anti_aliasing_popup(self):
        if self.full_data is None:
            QtWidgets.QMessageBox.critical(self, "Error", "No file loaded.")
            return

        # Ask for cutoff and decimation factor
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Anti-Aliasing Filter")
        form = QtWidgets.QFormLayout(dlg)
        cutoff_edit = QtWidgets.QLineEdit(str((self.sample_rate/2)-1))
        factor_edit = QtWidgets.QLineEdit("1")  # no decimation by default
        cutoff_edit.setFixedWidth(80)
        factor_edit.setFixedWidth(80)
        form.addRow("Cutoff Frequency (Hz):", cutoff_edit)
        form.addRow("Decimation Factor:",      factor_edit)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok|QtWidgets.QDialogButtonBox.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        form.addWidget(btns)

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        try:
            cutoff = float(cutoff_edit.text())
            decim  = max(1, int(factor_edit.text()))
        except ValueError:
            QtWidgets.QMessageBox.critical(self, "Error", "Invalid parameters.")
            return

        # Design low-pass filter at cutoff
        nyq = 0.5 * self.sample_rate
        wn = cutoff / nyq
        wn = cutoff / nyq
        if wn <= 0 or wn >= 1:
            suggested = nyq - 1.0
            QtWidgets.QMessageBox.critical(
                self, "Cutoff Error",
                f"Cutoff must be > 0 and < Nyquist ({nyq:.1f} Hz).\n"
                f"For example, try {suggested:.1f} Hz."
            )
            return
        sos = butter(8, wn, btype="low", output="sos")

        # Apply filter
        try:
            filtered = safe_sosfiltfilt(sos, self.full_data)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Filter Error", str(e))
            return

        # Decimate if requested
        if decim > 1:
            filtered = filtered[::decim]
            new_sr = self.sample_rate // decim
        else:
            new_sr = self.sample_rate

        # Cast back to original dtype
        if np.issubdtype(self.original_dtype, np.integer):
            iinfo = np.iinfo(self.original_dtype)
            filtered = np.clip(filtered, iinfo.min, iinfo.max).astype(self.original_dtype)
        else:
            filtered = filtered.astype(np.float64)

        # Save out
        directory = os.path.dirname(self.current_file_path)
        out_dir   = os.path.join(directory, "analysis")
        os.makedirs(out_dir, exist_ok=True)
        base, ext = os.path.splitext(self.file_name)
        fname     = f"{base}_aa_{int(cutoff)}Hz_x{decim}{ext}"
        path      = os.path.join(out_dir, fname)
        wavfile.write(path, new_sr, filtered)

        QtWidgets.QMessageBox.information(
            self, "Done",
            f"Anti-aliased file saved as:\n{fname}\nReloading…"
        )
        self.load_wav_file(path)


    def recommend_sample_rate_popup(self):
        if self.full_data is None:
            QtWidgets.QMessageBox.warning(self, "Error", "No file loaded.")
            return

        STANDARD_RATES = [
            2000, 4000, 8000, 11025, 16000, 22050, 32000,
            44100, 48000, 88200, 96000, 176400, 192000
        ]

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Recommend Sample Rate")
        dlg.setModal(True)
        layout = QtWidgets.QVBoxLayout(dlg)

        info_label = QtWidgets.QLabel("Computing highest significant frequency…")
        layout.addWidget(info_label)

        combo = QtWidgets.QComboBox()
        for sr in STANDARD_RATES:
            combo.addItem(f"{sr:,} Hz", sr)
        combo.setEnabled(False)
        layout.addWidget(QtWidgets.QLabel("Select Sample Rate:"))
        layout.addWidget(combo)

        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btn_box.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(False)
        layout.addWidget(btn_box)

        def compute_and_populate():
            data = self.full_data
            fs = self.sample_rate
            N = len(data)

            max_display = 200_000
            decim = max(1, N // max_display)
            if decim > 1:
                try:
                    from scipy.signal import decimate
                    coarse = decimate(data, decim, ftype='fir', zero_phase=True)
                    fs_eff = fs / decim
                except Exception:
                    coarse = data[::decim]
                    fs_eff = fs / decim
            else:
                coarse = data
                fs_eff = fs

            M = len(coarse)
            nfft = 1 << (int(np.ceil(np.log2(M))))
            windowed = coarse * np.hanning(M)
            fft_res = np.fft.rfft(windowed, n=nfft)
            freqs = np.fft.rfftfreq(nfft, d=1.0 / fs_eff)
            mag = np.abs(fft_res)

            peak_val = mag.max()
            if peak_val <= 0:
                QtWidgets.QMessageBox.information(self, "No Signal", "Waveform appears silent.")
                dlg.reject()
                return

            thresh = peak_val * 0.01
            idxs = np.where(mag >= thresh)[0]
            highest_freq = freqs[idxs[-1]] if len(idxs) else 0.0
            desired_nyq = 2.0 * highest_freq * 1.1

            recommended = next((sr for sr in STANDARD_RATES if sr >= desired_nyq), STANDARD_RATES[-1])

            info_label.setText(
                f"Highest significant frequency: {highest_freq:.2f} Hz\n"
                f"Desired Nyquist margin: {desired_nyq:.0f} Hz\n"
                f"Recommended: {recommended:,} Hz"
            )

            combo.setEnabled(True)
            idx = STANDARD_RATES.index(recommended)
            combo.setCurrentIndex(idx)
            btn_box.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(True)

        def on_accept():
            new_fs = combo.currentData()
            ratio = new_fs / self.sample_rate
            new_len = int(len(self.full_data) * ratio)

            try:
                down = np.interp(
                    np.linspace(0, len(self.full_data) - 1, new_len),
                    np.arange(len(self.full_data)),
                    self.full_data
                )
                if np.issubdtype(self.original_dtype, np.integer):
                    mx = np.iinfo(self.original_dtype).max
                    mn = np.iinfo(self.original_dtype).min
                    down = np.clip(down, mn, mx).astype(self.original_dtype)
                else:
                    down = down.astype(np.float64)

                directory = os.path.dirname(self.current_file_path)
                base, ext = os.path.splitext(self.file_name)
                new_filename = f"{base}_down_{new_fs}Hz{ext}"
                new_filepath = os.path.join(directory, new_filename)
                wavfile.write(new_filepath, new_fs, down)
                QtWidgets.QMessageBox.information(
                    self,
                    "Downsample Complete",
                    f"Saved as {new_filename}\nLoading now…"
                )
                self.load_wav_file(new_filepath)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Downsampling failed:\n{e}")

            dlg.accept()

        btn_box.accepted.connect(on_accept)
        btn_box.rejected.connect(dlg.reject)

        QtCore.QTimer.singleShot(50, compute_and_populate)
        dlg.exec_()


    def apply_filtered_data(self, filtered):
        """
        Replace full_data with filtered version, then re‐plot.
        """
        if np.issubdtype(self.original_dtype, np.integer):
            max_val = np.iinfo(self.original_dtype).max
            min_val = np.iinfo(self.original_dtype).min
            filtered = np.clip(filtered, min_val, max_val).astype(self.original_dtype)
        else:
            filtered = filtered.astype(np.float64)

        directory = os.path.dirname(self.current_file_path)
        out_dir = os.path.join(directory, "analysis")
        os.makedirs(out_dir, exist_ok=True)
        base, ext = os.path.splitext(self.file_name)
        new_filename = f"{base}_filtered{ext}"
        new_filepath = os.path.join(out_dir, new_filename)
        try:
            wavfile.write(new_filepath, self.sample_rate, filtered)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to write filtered WAV:\n{e}")
            return
        QtWidgets.QMessageBox.information(self, "Filter Complete", f"Filtered file saved as {new_filename}. It will now be loaded.")
        self.load_wav_file(new_filepath)

    # ---------------------
    # Measurement Tools Popups (continued)
    # ---------------------

