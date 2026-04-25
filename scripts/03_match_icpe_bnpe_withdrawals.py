"""Build probabilistic ICPE to BNPE groundwater withdrawal matches."""

import pandas as pd

from icpe_groundwater_exposure.bnpe import (
    bnpe_to_lambert93,
    load_raw_bnpe_withdrawals,
    normalize_bnpe_withdrawals,
)
from icpe_groundwater_exposure.config import (
    ICPE_BNPE_BEST_MATCH_FILE,
    ICPE_BNPE_ECONOMIC_BEST_MATCH_FILE,
    ICPE_BNPE_MATCHES_FILE,
    NORMALIZED_BNPE_WITHDRAWALS_FILE,
    NORMALIZED_ICPE_FILE,
    RAW_BNPE_WITHDRAWALS_FILE,
)
from icpe_groundwater_exposure.matching import best_match_per_icpe, build_icpe_bnpe_matches
from icpe_groundwater_exposure.utils import ensure_dirs


def main() -> None:
    print("\n========== MATCH ICPE TO BNPE WITHDRAWALS ==========\n")
    print("ICPE input:", NORMALIZED_ICPE_FILE)
    print("BNPE input:", RAW_BNPE_WITHDRAWALS_FILE)
    print("BNPE normalized output:", NORMALIZED_BNPE_WITHDRAWALS_FILE)
    print("Match output:", ICPE_BNPE_MATCHES_FILE)

    if not NORMALIZED_ICPE_FILE.exists():
        raise FileNotFoundError(f"Normalized ICPE file not found: {NORMALIZED_ICPE_FILE}")
    if not RAW_BNPE_WITHDRAWALS_FILE.exists():
        raise FileNotFoundError(f"BNPE withdrawals file not found: {RAW_BNPE_WITHDRAWALS_FILE}")

    ensure_dirs([NORMALIZED_BNPE_WITHDRAWALS_FILE.parent, ICPE_BNPE_MATCHES_FILE.parent])

    icpe = pd.read_csv(NORMALIZED_ICPE_FILE, dtype={"code_aiot": "string"}, low_memory=False)
    bnpe_raw = load_raw_bnpe_withdrawals(RAW_BNPE_WITHDRAWALS_FILE)
    bnpe = normalize_bnpe_withdrawals(bnpe_raw)
    bnpe.to_csv(NORMALIZED_BNPE_WITHDRAWALS_FILE, index=False)

    matches = build_icpe_bnpe_matches(
        icpe=icpe,
        withdrawals=bnpe_to_lambert93(bnpe),
        max_distance_m=5000,
    )
    matches.to_csv(ICPE_BNPE_MATCHES_FILE, index=False)
    best_any = best_match_per_icpe(matches)
    best_any.to_csv(ICPE_BNPE_BEST_MATCH_FILE, index=False)
    best_economic = best_match_per_icpe(matches, scopes={"nearest_economic_groundwater"})
    best_economic.to_csv(ICPE_BNPE_ECONOMIC_BEST_MATCH_FILE, index=False)

    print("ICPE sites:", len(icpe))
    print("BNPE withdrawals:", len(bnpe))
    print("BNPE IND withdrawals:", int((bnpe["usage_code"] == "IND").sum()))
    print("BNPE IND+IRR withdrawals:", int(bnpe["usage_code"].isin(["IND", "IRR"]).sum()))
    print("Candidate rows:", len(matches))
    print("\nConfidence summary:")
    print(matches["match_confidence"].value_counts(dropna=False).to_string())
    print("\nBest economic match summary:")
    print(best_economic["match_confidence"].value_counts(dropna=False).to_string())
    print("\nTop probable matches:")
    top = matches[matches["match_confidence"].isin(["probable_high", "probable"])].head(12)
    cols = [
        "match_scope",
        "match_score",
        "match_distance_m",
        "nom_ets",
        "ouvrage_name",
        "usage_code",
        "volume_m3",
        "same_commune",
        "name_similarity",
    ]
    print(top[cols].to_string(index=False))
    print("\nSaved:", ICPE_BNPE_MATCHES_FILE)
    print("Saved:", ICPE_BNPE_BEST_MATCH_FILE)
    print("Saved:", ICPE_BNPE_ECONOMIC_BEST_MATCH_FILE)


if __name__ == "__main__":
    main()
