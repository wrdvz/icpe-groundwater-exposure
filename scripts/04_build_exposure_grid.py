"""Build a 20 km exposure grid combining groundwater trends and IND/IRR withdrawals."""

import pandas as pd

from icpe_groundwater_exposure.config import (
    EXPOSURE_GRID_FILE,
    EXPOSURE_GRID_GEOJSON_FILE,
    NORMALIZED_BNPE_WITHDRAWALS_FILE,
    TRENDS_FILE,
)
from icpe_groundwater_exposure.grid import build_exposure_grid
from icpe_groundwater_exposure.utils import ensure_dirs


def main() -> None:
    print("\n========== BUILD EXPOSURE GRID ==========\n")
    print("Station trends input:", TRENDS_FILE)
    print("BNPE input:", NORMALIZED_BNPE_WITHDRAWALS_FILE)
    print("Grid CSV output:", EXPOSURE_GRID_FILE)
    print("Grid GeoJSON output:", EXPOSURE_GRID_GEOJSON_FILE)

    if not TRENDS_FILE.exists():
        raise FileNotFoundError(f"Station trends file not found: {TRENDS_FILE}")
    if not NORMALIZED_BNPE_WITHDRAWALS_FILE.exists():
        raise FileNotFoundError(f"BNPE normalized file not found: {NORMALIZED_BNPE_WITHDRAWALS_FILE}")

    ensure_dirs([EXPOSURE_GRID_FILE.parent, EXPOSURE_GRID_GEOJSON_FILE.parent])

    stations = pd.read_csv(TRENDS_FILE, dtype={"code_bss": "string"}, low_memory=False)
    bnpe = pd.read_csv(NORMALIZED_BNPE_WITHDRAWALS_FILE, low_memory=False)
    grid_gdf, pressure_threshold_m3 = build_exposure_grid(stations, bnpe)

    grid_gdf.drop(columns=["geometry"]).to_csv(EXPOSURE_GRID_FILE, index=False)
    grid_gdf.to_file(EXPOSURE_GRID_GEOJSON_FILE, driver="GeoJSON")

    print("Grid cells:", len(grid_gdf))
    print(f"Pressure threshold P75 (m3): {pressure_threshold_m3:,.0f}")
    print("\nExposure classes:")
    print(grid_gdf["exposure_class_2x2"].value_counts(dropna=False).to_string())
    print("\nSaved:", EXPOSURE_GRID_FILE)
    print("Saved:", EXPOSURE_GRID_GEOJSON_FILE)


if __name__ == "__main__":
    main()
