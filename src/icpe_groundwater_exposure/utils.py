from pathlib import Path
import locale

import matplotlib as mpl
from matplotlib import rcParams, font_manager
import pandas as pd

from icpe_groundwater_exposure.config import ROBOTO_CONDENSED_FONT


# ============================================================
# ENV / DISPLAY
# ============================================================

def set_french_locale() -> None:
    try:
        locale.setlocale(locale.LC_TIME, "fr_FR.UTF-8")
    except Exception:
        pass


def setup_matplotlib_fonts() -> None:
    if ROBOTO_CONDENSED_FONT.exists():
        font_manager.fontManager.addfont(str(ROBOTO_CONDENSED_FONT))
        rcParams["font.family"] = "Roboto Condensed"

    mpl.rcParams["hatch.linewidth"] = 0.35


# ============================================================
# FILESYSTEM
# ============================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_dirs(paths) -> None:
    for path in paths:
        ensure_dir(path)


# ============================================================
# DATA HELPERS
# ============================================================

def format_station_count(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def find_coord_columns(df: pd.DataFrame):
    lon_candidates = ["lon", "longitude", "x"]
    lat_candidates = ["lat", "latitude", "y"]

    lon_col = next((c for c in lon_candidates if c in df.columns), None)
    lat_col = next((c for c in lat_candidates if c in df.columns), None)

    if lon_col is None or lat_col is None:
        raise ValueError(
            f"Impossible de trouver les colonnes de coordonnées. Colonnes: {df.columns.tolist()}"
        )

    return lon_col, lat_col


# ============================================================
# GEO HELPERS
# ============================================================

def normalize_departement(code):
    if code is None or pd.isna(code):
        return None

    code = str(code).strip().upper()

    if code in ["2A", "2B"]:
        return code

    return code.zfill(2)


def get_departement_name(code, mapping):
    code = normalize_departement(code)
    if code is None:
        return None
    return mapping.get(code)


def get_region_from_departement(code, mapping):
    code = normalize_departement(code)
    if code is None:
        return None
    return mapping.get(code)