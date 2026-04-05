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
        return {"enabled_tabs": enabled_tabs, "enabled_tools": enabled_tools}

    def reset_defaults(self):
        self.profile = default_profile()
        for tab, cb in self.tab_checks.items():
            cb.setChecked(tab in self.profile["enabled_tabs"])
        self._populate_tool_tree()
        self.status.setText("Reset to defaults (not saved yet).")

    def save_profile(self):
        self.profile = self._read_profile_from_ui()
        save_profile(self.profile)
        self.status.setText(f"Saved profile to {PROFILE_PATH.resolve()}")

    def write_build_script(self):
        self.save_profile()
        script_path = Path("build_customer_release.sh")
        script_text = """#!/usr/bin/env bash
set -euo pipefail
echo "Building Analyze customer release using customer_build_config.json"
pyinstaller --noconfirm --windowed --name AnalyzeCustomer main_app_refactored.py
echo "Build complete: dist/AnalyzeCustomer"
"""
        script_path.write_text(script_text, encoding="utf-8")
        script_path.chmod(0o755)
        self.status.setText(f"Wrote {script_path.resolve()}")

    def build_executable(self):
        self.save_profile()
        if not shutil.which("pyinstaller"):
            QtWidgets.QMessageBox.warning(
                self,
                "PyInstaller Not Found",
                "PyInstaller is not installed in this environment.\n"
                "Install it with: pip install pyinstaller",
            )
            return
        try:
            subprocess.run(
                [
                    "pyinstaller",
                    "--noconfirm",
                    "--windowed",
                    "--name",
                    "AnalyzeCustomer",
                    "main_app_refactored.py",
                ],
                check=True,
            )
            self.status.setText("Build complete: dist/AnalyzeCustomer")
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
