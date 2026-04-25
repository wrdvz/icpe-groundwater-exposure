from difflib import SequenceMatcher
import re
from typing import Optional
import unicodedata

import geopandas as gpd
import numpy as np
import pandas as pd

from icpe_groundwater_exposure.config import WORK_CRS


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^A-Za-z0-9]+", " ", text).upper()
    stopwords = {
        "SA",
        "SAS",
        "SARL",
        "SCI",
        "ETS",
        "ETABLISSEMENTS",
        "STE",
        "SOCIETE",
        "FRANCE",
        "USINE",
        "SITE",
    }
    tokens = [token for token in text.split() if token not in stopwords and len(token) > 1]
    return " ".join(tokens)


def text_similarity(left, right) -> float:
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    if not left_norm or not right_norm:
        return 0.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def icpe_to_lambert93(df: pd.DataFrame) -> gpd.GeoDataFrame:
    out = df.copy()
    out["x"] = pd.to_numeric(out["x"], errors="coerce")
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    out = out.dropna(subset=["x", "y"]).copy()
    return gpd.GeoDataFrame(
        out,
        geometry=gpd.points_from_xy(out["x"], out["y"]),
        crs=WORK_CRS,
    )


def nearest_withdrawal_candidates(
    icpe: pd.DataFrame,
    withdrawals: gpd.GeoDataFrame,
    match_scope: str,
    max_distance_m: int = 5000,
) -> pd.DataFrame:
    gdf_icpe = icpe_to_lambert93(icpe)
    gdf_withdrawals = withdrawals.copy()

    joined = gpd.sjoin_nearest(
        gdf_icpe,
        gdf_withdrawals,
        how="left",
        max_distance=max_distance_m,
        distance_col="match_distance_m",
        lsuffix="icpe",
        rsuffix="bnpe",
    )
    joined = pd.DataFrame(joined.drop(columns=["geometry"], errors="ignore"))
    joined["match_scope"] = match_scope
    return joined


def score_candidate(row: pd.Series) -> dict:
    if pd.isna(row.get("ouvrage_sandre")):
        return {
            "match_score": 0,
            "match_confidence": "no_candidate",
            "name_similarity": 0.0,
            "same_commune": False,
        }

    distance_m = row.get("match_distance_m")
    usage_raw = row.get("usage_code")
    usage_code = "" if pd.isna(usage_raw) else str(usage_raw)
    site_raw = row.get("site_sector")
    site_sector = "" if pd.isna(site_raw) else str(site_raw)
    name_sim = text_similarity(row.get("nom_ets"), row.get("ouvrage_name"))
    same_commune = str(row.get("cd_insee")).zfill(5) == str(row.get("code_insee")).zfill(5)

    score = 0
    if pd.notna(distance_m):
        if distance_m <= 100:
            score += 35
        elif distance_m <= 500:
            score += 28
        elif distance_m <= 1000:
            score += 22
        elif distance_m <= 2500:
            score += 12
        elif distance_m <= 5000:
            score += 6

    if same_commune:
        score += 20

    if usage_code == "IND":
        score += 15
    elif usage_code == "IRR":
        score += 10

    if site_sector in {"Industrie", "Carriere"} and usage_code == "IND":
        score += 8
    elif row.get("industrie") == 1 and usage_code == "IND":
        score += 8
    elif site_sector == "Elevage" and usage_code == "IRR":
        score += 8

    score += int(round(name_sim * 22))

    water_relevant = row.get("is_water_relevant")
    if water_relevant is True or water_relevant == 1:
        score += 5

    score = min(score, 100)
    if score >= 75:
        confidence = "probable_high"
    elif score >= 55:
        confidence = "probable"
    elif score >= 35:
        confidence = "possible"
    elif score > 0:
        confidence = "context_only"
    else:
        confidence = "no_candidate"

    return {
        "match_score": score,
        "match_confidence": confidence,
        "name_similarity": round(name_sim, 3),
        "same_commune": same_commune,
    }


def build_icpe_bnpe_matches(
    icpe: pd.DataFrame,
    withdrawals: gpd.GeoDataFrame,
    max_distance_m: int = 5000,
) -> pd.DataFrame:
    any_candidates = nearest_withdrawal_candidates(
        icpe=icpe,
        withdrawals=withdrawals,
        match_scope="nearest_any_groundwater",
        max_distance_m=max_distance_m,
    )
    ind_candidates = nearest_withdrawal_candidates(
        icpe=icpe,
        withdrawals=withdrawals[withdrawals["usage_code"] == "IND"].copy(),
        match_scope="nearest_ind_groundwater",
        max_distance_m=max_distance_m,
    )
    economic_candidates = nearest_withdrawal_candidates(
        icpe=icpe,
        withdrawals=withdrawals[withdrawals["usage_code"].isin(["IND", "IRR"])].copy(),
        match_scope="nearest_economic_groundwater",
        max_distance_m=max_distance_m,
    )

    candidates = pd.concat([any_candidates, ind_candidates, economic_candidates], ignore_index=True)
    score_df = pd.DataFrame([score_candidate(row) for _, row in candidates.iterrows()])
    candidates = pd.concat([candidates.reset_index(drop=True), score_df], axis=1)

    keep_columns = [
        "match_scope",
        "match_score",
        "match_confidence",
        "match_distance_m",
        "same_commune",
        "name_similarity",
        "code_aiot",
        "nom_ets",
        "num_siret",
        "site_sector",
        "is_water_relevant",
        "industrie",
        "ied",
        "priorite_nationale",
        "cd_regime",
        "lib_regime",
        "lib_seveso",
        "adresse",
        "cd_insee",
        "commune_icpe",
        "ouvrage_sandre",
        "ouvrage_name",
        "usage_code",
        "usage_label",
        "volume_m3",
        "longitude",
        "latitude",
        "code_insee",
        "commune_bnpe",
        "departement",
        "code_bss",
        "code_bdlisa",
        "bdlisa_label",
    ]

    candidates = candidates.rename(
        columns={
            "commune_left": "commune_icpe",
            "commune_right": "commune_bnpe",
            "commune_icpe": "commune_icpe",
            "commune_bnpe": "commune_bnpe",
        }
    )
    if "commune" in candidates.columns and "commune_icpe" not in candidates.columns:
        candidates = candidates.rename(columns={"commune": "commune_icpe"})
    if "commune_bnpe" not in candidates.columns and "commune_right" in candidates.columns:
        candidates = candidates.rename(columns={"commune_right": "commune_bnpe"})

    for column in keep_columns:
        if column not in candidates.columns:
            candidates[column] = np.nan

    candidates = candidates[keep_columns].copy()
    candidates["match_distance_m"] = candidates["match_distance_m"].round(1)
    candidates["volume_m3"] = pd.to_numeric(candidates["volume_m3"], errors="coerce")
    confidence_rank = {
        "probable_high": 0,
        "probable": 1,
        "possible": 2,
        "context_only": 3,
        "no_candidate": 4,
    }
    candidates["_confidence_rank"] = candidates["match_confidence"].map(confidence_rank).fillna(9)
    return candidates.sort_values(
        ["_confidence_rank", "match_score", "match_distance_m"],
        ascending=[True, False, True],
    ).drop(columns=["_confidence_rank"])


def best_match_per_icpe(candidates: pd.DataFrame, scopes: Optional[set[str]] = None) -> pd.DataFrame:
    out = candidates.copy()
    if scopes is not None:
        out = out[out["match_scope"].isin(scopes)].copy()

    confidence_rank = {
        "probable_high": 0,
        "probable": 1,
        "possible": 2,
        "context_only": 3,
        "no_candidate": 4,
    }
    out["_confidence_rank"] = out["match_confidence"].map(confidence_rank).fillna(9)
    out = (
        out.sort_values(
            ["code_aiot", "_confidence_rank", "match_score", "match_distance_m", "volume_m3"],
            ascending=[True, True, False, True, False],
        )
        .drop_duplicates("code_aiot")
        .drop(columns=["_confidence_rank"])
    )
    return out
