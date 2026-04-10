from __future__ import annotations

import os
from pathlib import Path


def get_package_dir(script_path: str) -> Path:
    override = os.environ.get("FORECAST_PACKAGE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path(script_path).resolve().parent
