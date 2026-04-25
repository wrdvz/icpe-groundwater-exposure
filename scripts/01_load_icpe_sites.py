"""Load and normalize ICPE sites for the exposure analysis."""

from icpe_groundwater_exposure.config import NORMALIZED_ICPE_FILE, RAW_ICPE_FILE
from icpe_groundwater_exposure.icpe import (
    load_raw_icpe_sites,
    normalize_icpe_sites,
    summarize_icpe_sites,
)
from icpe_groundwater_exposure.utils import ensure_dirs


def main() -> None:
    print("\n========== LOAD ICPE SITES ==========\n")
    print("Input:", RAW_ICPE_FILE)
    print("Output:", NORMALIZED_ICPE_FILE)

    if not RAW_ICPE_FILE.exists():
        raise FileNotFoundError(f"Raw ICPE file not found: {RAW_ICPE_FILE}")

    ensure_dirs([NORMALIZED_ICPE_FILE.parent])

    raw = load_raw_icpe_sites(RAW_ICPE_FILE)
    sites = normalize_icpe_sites(raw)
    sites.to_csv(NORMALIZED_ICPE_FILE, index=False)

    print("Raw rows:", len(raw))
    print("Normalized rows:", len(sites))
    print("\nSector summary:")
    print(summarize_icpe_sites(sites).to_string(index=False))
    print("\nSaved:", NORMALIZED_ICPE_FILE)


if __name__ == "__main__":
    main()
