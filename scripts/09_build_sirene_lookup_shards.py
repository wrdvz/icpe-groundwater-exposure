"""Construit des shards de lookup SIRENE/ICPE pour Cloudflare R2.

Entrée attendue:
- un export SIRENE géolocalisé (CSV ou Parquet) contenant au minimum SIRET + latitude + longitude

Sortie:
- des fichiers JSON shardés par préfixe de SIRET, prêts à être uploadés dans R2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

from icpe_groundwater_exposure.config import EXPOSURE_GRID_GEOJSON_FILE

ICPE_ENRICHED_FILE = (
    Path(EXPOSURE_GRID_GEOJSON_FILE).parents[1] / "icpe" / "icpe_sites_normalized_with_water_taxonomy_7.csv"
)


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def _pick_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    existing = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in existing:
            return existing[candidate.lower()]
    return None


def _require_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    col = _pick_existing(df, candidates)
    if not col:
        raise KeyError(f"Impossible de trouver la colonne {label}. Candidats testés: {candidates}")
    return col


def _normalize_sirene(df: pd.DataFrame) -> pd.DataFrame:
    siret_col = _require_column(df, ["siret"], "siret")
    siren_col = _require_column(df, ["siren"], "siren")
    lon_col = _require_column(
        df,
        ["longitude", "longitude_ban", "x", "long"],
        "longitude",
    )
    lat_col = _require_column(
        df,
        ["latitude", "latitude_ban", "y", "lat"],
        "latitude",
    )
    naf_col = _pick_existing(df, ["code_naf", "activitePrincipaleEtablissement"])
    lib_naf_col = _pick_existing(df, ["lib_naf", "libelle_activite_principale", "libelleActivitePrincipaleEtablissement"])
    denom_col = _pick_existing(
        df,
        [
            "denomination",
            "nom_raison_sociale",
            "denominationUniteLegale",
            "nomUsageUniteLegale",
            "enseigne1Etablissement",
            "denominationUsuelleEtablissement",
        ],
    )
    geo_score_col = _pick_existing(df, ["geo_score", "score", "score_geocodage"])
    geo_type_col = _pick_existing(df, ["geo_type", "type_geocodage"])

    out = pd.DataFrame(
        {
            "siret": df[siret_col].astype("string").str.replace(r"\D+", "", regex=True),
            "siren": df[siren_col].astype("string").str.replace(r"\D+", "", regex=True),
            "denomination": df[denom_col].astype("string") if denom_col else pd.Series(pd.NA, index=df.index, dtype="string"),
            "code_naf": df[naf_col].astype("string") if naf_col else pd.Series(pd.NA, index=df.index, dtype="string"),
            "lib_naf": df[lib_naf_col].astype("string") if lib_naf_col else pd.Series(pd.NA, index=df.index, dtype="string"),
            "longitude": pd.to_numeric(df[lon_col], errors="coerce"),
            "latitude": pd.to_numeric(df[lat_col], errors="coerce"),
            "geo_score": pd.to_numeric(df[geo_score_col], errors="coerce") if geo_score_col else pd.Series(pd.NA, index=df.index),
            "geo_type": df[geo_type_col].astype("string") if geo_type_col else pd.Series(pd.NA, index=df.index, dtype="string"),
        }
    )

    out = out[out["siret"].str.len() == 14].copy()
    out = out.dropna(subset=["longitude", "latitude"]).copy()
    out = out.drop_duplicates(subset=["siret"], keep="first").copy()
    return out


def _load_icpe_lookup() -> pd.DataFrame:
    icpe = pd.read_csv(ICPE_ENRICHED_FILE, dtype={"num_siret": "string"}, low_memory=False)
    icpe["num_siret"] = icpe["num_siret"].astype("string").str.replace(r"\D+", "", regex=True)
    icpe = icpe[icpe["num_siret"].str.len() == 14].copy()
    icpe = icpe.sort_values(["num_siret", "categorie_eau_7"], na_position="last")
    icpe = icpe.drop_duplicates(subset=["num_siret"], keep="first").copy()
    return icpe.rename(
        columns={
            "num_siret": "siret",
            "categorie_eau_7": "icpe_category",
            "site_sector": "icpe_site_sector",
            "nom_ets": "icpe_nom_ets",
            "median_variation_20y_cm_20km": "groundwater_trend_cm_20y",
            "n_stations_20km": "groundwater_station_count_20km",
        }
    )[
        [
            "siret",
            "icpe_category",
            "icpe_site_sector",
            "icpe_nom_ets",
            "groundwater_trend_cm_20y",
            "groundwater_station_count_20km",
        ]
    ]


def _attach_grid_class(df: pd.DataFrame) -> pd.DataFrame:
    grid = gpd.read_file(EXPOSURE_GRID_GEOJSON_FILE)[["exposure_class_2x2", "geometry"]].to_crs("EPSG:4326")
    points = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(points, grid, how="left", predicate="within").drop(columns=["index_right"], errors="ignore")
    joined["grid_class"] = joined["exposure_class_2x2"]
    return pd.DataFrame(joined.drop(columns=["geometry", "exposure_class_2x2"]))


def _build_shards(df: pd.DataFrame, out_dir: Path, prefix_len: int, namespace: str) -> None:
    base_dir = out_dir / namespace
    base_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["shard_prefix"] = df["siret"].str.slice(0, prefix_len).str.pad(width=prefix_len, side="right", fillchar="0")

    for prefix, chunk in df.groupby("shard_prefix", dropna=True):
        shard = {}
        for _, row in chunk.iterrows():
            shard[row["siret"]] = {
                "siret": row["siret"],
                "siren": None if pd.isna(row["siren"]) else str(row["siren"]),
                "denomination": None if pd.isna(row["denomination"]) else str(row["denomination"]),
                "code_naf": None if pd.isna(row["code_naf"]) else str(row["code_naf"]),
                "lib_naf": None if pd.isna(row["lib_naf"]) else str(row["lib_naf"]),
                "longitude": None if pd.isna(row["longitude"]) else float(row["longitude"]),
                "latitude": None if pd.isna(row["latitude"]) else float(row["latitude"]),
                "geo_score": None if pd.isna(row["geo_score"]) else float(row["geo_score"]),
                "geo_type": None if pd.isna(row["geo_type"]) else str(row["geo_type"]),
                "site_icpe": bool(pd.notna(row["icpe_category"])),
                "icpe_category": None if pd.isna(row["icpe_category"]) else str(row["icpe_category"]),
                "icpe_site_sector": None if pd.isna(row["icpe_site_sector"]) else str(row["icpe_site_sector"]),
                "icpe_nom_ets": None if pd.isna(row["icpe_nom_ets"]) else str(row["icpe_nom_ets"]),
                "grid_class": None if pd.isna(row["grid_class"]) else str(row["grid_class"]),
                "groundwater_trend_cm_20y": None
                if pd.isna(row["groundwater_trend_cm_20y"])
                else float(row["groundwater_trend_cm_20y"]),
                "groundwater_station_count_20km": None
                if pd.isna(row["groundwater_station_count_20km"])
                else int(row["groundwater_station_count_20km"]),
            }

        target = base_dir / f"{prefix}.json"
        target.write_text(json.dumps(shard, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sirene", required=True, help="Chemin vers le fichier SIRENE géolocalisé (CSV ou Parquet)")
    parser.add_argument("--out-dir", required=True, help="Dossier de sortie des shards")
    parser.add_argument("--prefix-len", type=int, default=3, help="Longueur du préfixe SIRET pour le sharding")
    parser.add_argument("--namespace", default="sirene/v1", help="Sous-dossier de versionnement des shards")
    args = parser.parse_args()

    sirene = _read_table(Path(args.sirene))
    sirene_norm = _normalize_sirene(sirene)
    icpe = _load_icpe_lookup()

    merged = sirene_norm.merge(icpe, on="siret", how="left")
    merged = _attach_grid_class(merged)

    out_dir = Path(args.out_dir)
    _build_shards(merged, out_dir, args.prefix_len, args.namespace)

    print("SIRENE normalisé:", len(sirene_norm))
    print("ICPE enrichi:", len(icpe))
    print("Rows after merge:", len(merged))
    print("Shards output:", out_dir / args.namespace)


if __name__ == "__main__":
    main()
