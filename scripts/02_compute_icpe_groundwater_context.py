"""Compute local groundwater context around ICPE sites.

Planned method:
- project ICPE sites and groundwater stations to Lambert-93
- find groundwater stations within 20 km of each ICPE site
- compute median and mean 20-year groundwater variation
- flag the signal as solid when at least 5 stations are available
"""

import pandas as pd

from icpe_groundwater_exposure.config import (
    ICPE_GROUNDWATER_CONTEXT_FILE,
    ICPE_RADIUS_M,
    NORMALIZED_ICPE_FILE,
    ROBUST_THRESHOLD,
    TRENDS_FILE,
)
from icpe_groundwater_exposure.exposure import compute_icpe_groundwater_context
from icpe_groundwater_exposure.utils import ensure_dirs


def main() -> None:
    print("\n========== COMPUTE ICPE GROUNDWATER CONTEXT ==========\n")
    print("ICPE input:", NORMALIZED_ICPE_FILE)
    print("Station trends input:", TRENDS_FILE)
    print("Output:", ICPE_GROUNDWATER_CONTEXT_FILE)
    print(f"Radius: {ICPE_RADIUS_M / 1000:.0f} km")
    print(f"Solid signal threshold: {ROBUST_THRESHOLD} stations")

    if not NORMALIZED_ICPE_FILE.exists():
        raise FileNotFoundError(f"Normalized ICPE file not found: {NORMALIZED_ICPE_FILE}")
    if not TRENDS_FILE.exists():
        raise FileNotFoundError(f"Station trends file not found: {TRENDS_FILE}")

    ensure_dirs([ICPE_GROUNDWATER_CONTEXT_FILE.parent])

    icpe = pd.read_csv(NORMALIZED_ICPE_FILE, dtype={"code_aiot": "string", "num_siret": "string"})
    stations = pd.read_csv(TRENDS_FILE, dtype={"code_bss": "string"})
    context = compute_icpe_groundwater_context(
        icpe=icpe,
        stations=stations,
        radius_m=ICPE_RADIUS_M,
        robust_threshold=ROBUST_THRESHOLD,
    )
    context.to_csv(ICPE_GROUNDWATER_CONTEXT_FILE, index=False)

    print("ICPE sites:", len(context))
    print("Sites with >=1 station:", int((context["n_stations_20km"] > 0).sum()))
    print("Sites with solid signal:", int(context["is_signal_solid"].sum()))
    print("\nSignal classes:")
    print(context["local_signal_class"].value_counts(dropna=False).to_string())
    print("\nSaved:", ICPE_GROUNDWATER_CONTEXT_FILE)


if __name__ == "__main__":
    main()
