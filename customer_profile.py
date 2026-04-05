"""Shared tool catalog and customer profile helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

def get_profile_path() -> Path:
    """Primary profile path."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "customer_build_config.json"
    return Path(__file__).resolve().parent / "customer_build_config.json"


def get_profile_candidates() -> list[Path]:
    """All candidate locations to search for an existing profile."""
    candidates = [get_profile_path()]
    if getattr(sys, "frozen", False):
        candidates.append(Path.cwd() / "customer_build_config.json")
        candidates.append(Path(sys.executable).resolve().parent.parent / "customer_build_config.json")
    return candidates


PROFILE_PATH = get_profile_path()

TAB_ORDER = ["Analysis", "SPL", "Spectrogram", "Chart", "Logs", "Projects"]

TOOL_CATALOG: Dict[str, List[str]] = {
    "WAV File Tools": [
        "Trim File",
        "Denoise File",
        "High/Low Pass Filter",
        "Anti-Aliasing Filter",
        "Recommend Sample Rate",
        "Remove DC Offset",
        "Normalize File",
        "Downsample Bit Depth",
        "Wav File Combine",
        "Stack WAV Channels",
        "Scale WAV Amplitude",
        "Create WAV Playlist",
        "Secure WAV Bundle (Pack/Unpack)",
        "Channel Sync Tool",
    ],
    "Measurement Tools": [
        "Ambient Noise",
        "Electrical Noise",
        "Peak Prominences Analysis",
        "Interval Analysis",
        "Depth Sounder Analysis",
        "Slope De-Clipper",
        "Find Peaks",
        "Short-Time RMS",
        "Crest Factor",
        "Octave-Band Analysis",
        "SNR Estimator",
        "LFM Analysis",
        "LFM Batch Analysis",
        "HFM Analysis",
        "Multi Frequency Analysis",
        "SPL Transmit Analysis",
        "Hydrophone Calibration",
        "Duty Cycle Analysis",
        "Exceedance Curves Lx",
        "LTSA + PSD Percentiles",
        "SEL",
        "Recurrence Periodicity",
        "Dominant Frequencies Over Time",
    ],
    "Modelling & Plotting Tools": [
        "Wenz Curves",
        "Propagation Modelling",
        "Cable Loss & Hydro Sensitivity",
        "Simulated GPS Track Generator",
    ],
    "Detection & Classification Tools": [
        "Active Sonar",
        "Cepstrum Analysis",
        "Event Clustering",
        "DIFAR Processing",
    ],
    "Database Tools": [
        "Export Logs to Excel",
        "Filter Measurements",
        "Filter Logs by Date/Time",
        "Database Maintenance",
        "Clean Measurement Data",
        "Annotate Measurements",
        "Annotate SPL",
        "Calibration Curve Manager",
        "TVR Curve Manager",
        "Measurement Voltage Correction",
        "CTD Import",
    ],
}


def default_profile() -> dict:
    return {
        "enabled_tabs": list(TAB_ORDER),
        "enabled_tools": {category: list(tools) for category, tools in TOOL_CATALOG.items()},
        "build_output_dir": "dist",
    }


def load_profile(path: Path | None = None) -> dict:
    profile = default_profile()
    if path is None:
        existing = next((p for p in get_profile_candidates() if p.exists()), None)
        path = existing or get_profile_path()
    if not path.exists():
        return profile
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return profile

    enabled_tabs = [t for t in raw.get("enabled_tabs", []) if t in TAB_ORDER]
    if enabled_tabs:
        profile["enabled_tabs"] = enabled_tabs

    enabled_tools = raw.get("enabled_tools", {})
    merged_tools = {}
    for category, tools in TOOL_CATALOG.items():
        requested = enabled_tools.get(category, tools)
        merged_tools[category] = [tool for tool in requested if tool in tools]
    profile["enabled_tools"] = merged_tools
    build_output_dir = raw.get("build_output_dir", "dist")
    if isinstance(build_output_dir, str) and build_output_dir.strip():
        profile["build_output_dir"] = build_output_dir.strip()
    return profile


def save_profile(profile: dict, path: Path | None = None) -> None:
    if path is None:
        path = get_profile_path()
    normalized = default_profile()
    normalized["enabled_tabs"] = [
        t for t in profile.get("enabled_tabs", []) if t in TAB_ORDER
    ] or list(TAB_ORDER)

    normalized["enabled_tools"] = {}
    for category, tools in TOOL_CATALOG.items():
        requested = profile.get("enabled_tools", {}).get(category, tools)
        normalized["enabled_tools"][category] = [tool for tool in requested if tool in tools]
    output_dir = profile.get("build_output_dir", "dist")
    normalized["build_output_dir"] = output_dir.strip() if isinstance(output_dir, str) and output_dir.strip() else "dist"

    path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
