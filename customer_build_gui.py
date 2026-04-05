#!/usr/bin/env python3
"""PyQt5 configurator for customer-specific Analyze builds."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import qdarkstyle
from PyQt5 import QtCore, QtWidgets

from customer_profile import (
    PROFILE_PATH,
    TAB_ORDER,
    TOOL_CATALOG,
    default_profile,
    load_profile,
    save_profile,
)


class CustomerBuildWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analyze Customer Build Configurator")
        self.resize(1100, 760)
        self.profile = load_profile()
        self._build_ui()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        title = QtWidgets.QLabel("Customer Build Profile")
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        root.addWidget(title)

        subtitle = QtWidgets.QLabel(
            "Select tabs and tools that should be included in this customer build."
        )
        subtitle.setStyleSheet("color: #9AA3AF;")
        root.addWidget(subtitle)

        output_row = QtWidgets.QHBoxLayout()
        output_row.addWidget(QtWidgets.QLabel("Build output directory:"))
        self.output_dir_edit = QtWidgets.QLineEdit(self.profile.get("build_output_dir", "dist"))
        output_row.addWidget(self.output_dir_edit, 1)
        browse_btn = QtWidgets.QPushButton("Browse…")
        browse_btn.clicked.connect(self.choose_output_dir)
        output_row.addWidget(browse_btn)
        root.addLayout(output_row)

        content = QtWidgets.QHBoxLayout()
        root.addLayout(content, 1)

        # Tabs panel
        tabs_group = QtWidgets.QGroupBox("Enabled Main Tabs")
        tabs_layout = QtWidgets.QVBoxLayout(tabs_group)
        self.tab_checks = {}
        for tab in TAB_ORDER:
            cb = QtWidgets.QCheckBox(tab)
            cb.setChecked(tab in self.profile.get("enabled_tabs", []))
            self.tab_checks[tab] = cb
            tabs_layout.addWidget(cb)
        tabs_layout.addStretch()
        content.addWidget(tabs_group, 1)

        # Tools panel
        tools_group = QtWidgets.QGroupBox("Enabled Tools")
        tools_layout = QtWidgets.QVBoxLayout(tools_group)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Tool", "Enabled"])
        self.tree.setColumnCount(2)
        self.tree.setAlternatingRowColors(True)
        tools_layout.addWidget(self.tree)

        category_actions = QtWidgets.QHBoxLayout()
        category_actions.addWidget(QtWidgets.QLabel("Category controls:"))
        self.category_select_combo = QtWidgets.QComboBox()
        self.category_select_combo.addItems(list(TOOL_CATALOG.keys()))
        category_actions.addWidget(self.category_select_combo)
        select_all_btn = QtWidgets.QPushButton("Select All")
        select_all_btn.clicked.connect(lambda: self.set_category_checks(QtCore.Qt.Checked))
        category_actions.addWidget(select_all_btn)
        deselect_all_btn = QtWidgets.QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(lambda: self.set_category_checks(QtCore.Qt.Unchecked))
        category_actions.addWidget(deselect_all_btn)
        category_actions.addStretch()
        tools_layout.addLayout(category_actions)

        content.addWidget(tools_group, 2)

        self._populate_tool_tree()

        # Footer controls
        controls = QtWidgets.QHBoxLayout()
        root.addLayout(controls)
        reset_btn = QtWidgets.QPushButton("Reset Defaults")
        reset_btn.clicked.connect(self.reset_defaults)
        controls.addWidget(reset_btn)

        save_btn = QtWidgets.QPushButton("Save Profile")
        save_btn.clicked.connect(self.save_profile)
        controls.addWidget(save_btn)

        script_btn = QtWidgets.QPushButton("Write Build Script")
        script_btn.clicked.connect(self.write_build_script)
        controls.addWidget(script_btn)

        build_btn = QtWidgets.QPushButton("Build Executable")
        build_btn.clicked.connect(self.build_executable)
        controls.addWidget(build_btn)
        controls.addStretch()

        self.status = QtWidgets.QLabel(f"Profile file: {PROFILE_PATH.resolve()}")
        self.status.setStyleSheet("color: #6EE7B7;")
        root.addWidget(self.status)

    def _populate_tool_tree(self):
        self.tree.clear()
        enabled_tools = self.profile.get("enabled_tools", {})
        for category, tools in TOOL_CATALOG.items():
            category_item = QtWidgets.QTreeWidgetItem([category, ""])
            category_item.setFirstColumnSpanned(True)
            category_item.setExpanded(True)
            self.tree.addTopLevelItem(category_item)
            enabled_in_category = set(enabled_tools.get(category, tools))
            for tool in tools:
                item = QtWidgets.QTreeWidgetItem([tool, ""])
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                item.setCheckState(0, QtCore.Qt.Checked if tool in enabled_in_category else QtCore.Qt.Unchecked)
                category_item.addChild(item)
        self.tree.expandAll()

    def _read_profile_from_ui(self):
        enabled_tabs = [name for name, cb in self.tab_checks.items() if cb.isChecked()]
        enabled_tools = {}
        for i in range(self.tree.topLevelItemCount()):
            category_item = self.tree.topLevelItem(i)
            category = category_item.text(0)
            selected = []
            for j in range(category_item.childCount()):
                child = category_item.child(j)
                if child.checkState(0) == QtCore.Qt.Checked:
                    selected.append(child.text(0))
            enabled_tools[category] = selected
        output_dir = self.output_dir_edit.text().strip() or "dist"
        return {
            "enabled_tabs": enabled_tabs,
            "enabled_tools": enabled_tools,
            "build_output_dir": output_dir,
        }

    def choose_output_dir(self):
        chosen = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Choose Build Output Directory",
            self.output_dir_edit.text().strip() or str(Path.cwd()),
        )
        if chosen:
            self.output_dir_edit.setText(chosen)

    def set_category_checks(self, state):
        category_name = self.category_select_combo.currentText()
        for i in range(self.tree.topLevelItemCount()):
            category_item = self.tree.topLevelItem(i)
            if category_item.text(0) != category_name:
                continue
            for j in range(category_item.childCount()):
                category_item.child(j).setCheckState(0, state)
            break

    def reset_defaults(self):
        self.profile = default_profile()
        for tab, cb in self.tab_checks.items():
            cb.setChecked(tab in self.profile["enabled_tabs"])
        self.output_dir_edit.setText(self.profile.get("build_output_dir", "dist"))
        self._populate_tool_tree()
        self.status.setText("Reset to defaults (not saved yet).")

    def save_profile(self):
        self.profile = self._read_profile_from_ui()
        save_profile(self.profile)
        self.status.setText(f"Saved profile to {PROFILE_PATH.resolve()}")

    def write_build_script(self):
        self.save_profile()
        script_path = Path("build_customer_release.sh")
        output_dir = self.output_dir_edit.text().strip() or "dist"
        script_text = """#!/usr/bin/env bash
set -euo pipefail
echo "Building Analyze customer release using customer_build_config.json"
python3 -m PyInstaller --noconfirm --windowed --name AnalyzeCustomer --distpath "{output_dir}" main_app_refactored.py
echo "Build complete: {output_dir}/AnalyzeCustomer"
"""
        script_text = script_text.format(output_dir=output_dir)
        script_path.write_text(script_text, encoding="utf-8")
        script_path.chmod(0o755)
        self.status.setText(f"Wrote {script_path.resolve()}")

    def _resolve_pyinstaller_command(self):
        module_cmds = [
            ["python3", "-m", "PyInstaller"],
            [sys.executable, "-m", "PyInstaller"],
        ]
        for module_cmd in module_cmds:
            try:
                probe = subprocess.run(
                    module_cmd + ["--version"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if probe.returncode == 0:
                    return module_cmd
            except Exception:
                pass

        binary = shutil.which("pyinstaller") or shutil.which("pyinstaller.exe")
        if binary:
            return [binary]
        return None

    def build_executable(self):
        self.save_profile()
        pyinstaller_cmd = self._resolve_pyinstaller_command()
        if not pyinstaller_cmd:
            QtWidgets.QMessageBox.warning(
                self,
                "PyInstaller Not Found",
                "Couldn't find PyInstaller in this Python environment.\n"
                "Try one of these:\n"
                "1) install into this interpreter: pip install pyinstaller\n"
                "2) run with your working interpreter: python3 -m PyInstaller ...",
            )
            return
        output_dir = self.output_dir_edit.text().strip() or "dist"
        try:
            cmd = pyinstaller_cmd + [
                "--noconfirm",
                "--windowed",
                "--name",
                "AnalyzeCustomer",
                "--distpath",
                output_dir,
                "main_app_refactored.py",
            ]
            subprocess.run(cmd, check=True)
            self.status.setText(f"Build complete: {output_dir}/AnalyzeCustomer")
        except subprocess.CalledProcessError as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Build Failed",
                f"PyInstaller returned code {exc.returncode}",
            )


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    win = CustomerBuildWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
