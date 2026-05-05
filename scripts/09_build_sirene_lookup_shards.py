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
    geometry_col = _pick_existing(df, ["geometry"])
    lon_col = _pick_existing(df, ["longitude", "longitude_ban", "x", "long"])
    lat_col = _pick_existing(df, ["latitude", "latitude_ban", "y", "lat"])
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
    etat_col = _pick_existing(df, ["etatAdministratifEtablissement", "etat_administratif"])

    if geometry_col and (lon_col is None or lat_col is None):
        geom = gpd.GeoSeries.from_wkb(df[geometry_col], crs="EPSG:4326")
        longitudes = geom.x
        latitudes = geom.y
    else:
        if lon_col is None:
            raise KeyError("Impossible de trouver la colonne longitude ni une géométrie exploitable.")
        if lat_col is None:
            raise KeyError("Impossible de trouver la colonne latitude ni une géométrie exploitable.")
        longitudes = pd.to_numeric(df[lon_col], errors="coerce")
        latitudes = pd.to_numeric(df[lat_col], errors="coerce")

    out = pd.DataFrame(
        {
            "siret": df[siret_col].astype("string").str.replace(r"\D+", "", regex=True),
            "siren": df[siren_col].astype("string").str.replace(r"\D+", "", regex=True),
            "denomination": df[denom_col].astype("string") if denom_col else pd.Series(pd.NA, index=df.index, dtype="string"),
            "code_naf": df[naf_col].astype("string") if naf_col else pd.Series(pd.NA, index=df.index, dtype="string"),
            "lib_naf": df[lib_naf_col].astype("string") if lib_naf_col else pd.Series(pd.NA, index=df.index, dtype="string"),
            "longitude": longitudes,
            "latitude": latitudes,
            "geo_score": pd.to_numeric(df[geo_score_col], errors="coerce") if geo_score_col else pd.Series(pd.NA, index=df.index),
            "geo_type": df[geo_type_col].astype("string") if geo_type_col else pd.Series(pd.NA, index=df.index, dtype="string"),
            "etat_admin": df[etat_col].astype("string") if etat_col else pd.Series(pd.NA, index=df.index, dtype="string"),
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
    icpe = icpe.rename(
        columns={
            "num_siret": "siret",
            "categorie_eau_7": "icpe_category",
            "site_sector": "icpe_site_sector",
            "nom_ets": "icpe_nom_ets",
            "median_variation_20y_cm_20km": "groundwater_trend_cm_20y",
            "n_stations_20km": "groundwater_station_count_20km",
        }
    )
    for optional_col in ["groundwater_trend_cm_20y", "groundwater_station_count_20km"]:
        if optional_col not in icpe.columns:
            icpe[optional_col] = pd.NA
    return icpe[
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
            grid_code = None
            if not pd.isna(row["grid_class"]):
                grid_code = {
                    "high_pressure_declining_groundwater": "HPD",
                    "low_pressure_declining_groundwater": "LPD",
                    "high_pressure_non_declining_groundwater": "HPN",
                    "low_pressure_non_declining_groundwater": "LPN",
                    "unclassified_no_groundwater_data": "UNC",
                }.get(str(row["grid_class"]), "UNC")

            icpe_code = None
            if not pd.isna(row["icpe_category"]):
                icpe_code = {
                    "Agriculture et élevage": "AGE",
                    "Agro-industrie": "AGI",
                    "Santé, chimie, produits de synthèse": "SCP",
                    "Métallurgie, mécanique, automobile": "MMA",
                    "Papier, bois, textile, cuir": "PBTC",
                    "Extraction, carrières, eau, déchets, énergie": "ECDE",
                    "Construction et génie civil": "CGC",
                }.get(str(row["icpe_category"]), None)

            shard[row["siret"]] = {
                "n": None if pd.isna(row["denomination"]) else str(row["denomination"]),
                "a": None if pd.isna(row["code_naf"]) else str(row["code_naf"]),
                "x": None if pd.isna(row["longitude"]) else round(float(row["longitude"]), 6),
                "y": None if pd.isna(row["latitude"]) else round(float(row["latitude"]), 6),
                "gs": None if pd.isna(row["geo_score"]) else round(float(row["geo_score"]), 2),
                "gt": None if pd.isna(row["geo_type"]) else str(row["geo_type"]),
                "i": 1 if pd.notna(row["icpe_category"]) else 0,
                "ic": icpe_code,
                "g": grid_code,
            }

        target = base_dir / f"{prefix}.json"
        target.write_text(json.dumps(shard, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sirene", required=True, help="Chemin vers le fichier SIRENE géolocalisé (CSV ou Parquet)")
    parser.add_argument("--out-dir", required=True, help="Dossier de sortie des shards")
    parser.add_argument("--prefix-len", type=int, default=3, help="Longueur du préfixe SIRET pour le sharding")
    parser.add_argument("--namespace", default="sirene/v1", help="Sous-dossier de versionnement des shards")
    parser.add_argument(
        "--active-only",
        action="store_true",
        help="Ne garder que les établissements administrativement actifs",
    )
    parser.add_argument(
        "--min-geo-score",
        type=float,
        default=None,
        help="Seuil minimal de geo_score à conserver",
    )
    args = parser.parse_args()

    sirene = _read_table(Path(args.sirene))
    sirene_norm = _normalize_sirene(sirene)

    if args.active_only and "etat_admin" in sirene_norm.columns:
        sirene_norm = sirene_norm[sirene_norm["etat_admin"].fillna("").eq("A")].copy()

    if args.min_geo_score is not None:
        sirene_norm = sirene_norm[
            sirene_norm["geo_score"].notna() & (sirene_norm["geo_score"] >= args.min_geo_score)
        ].copy()

    sirene_norm = sirene_norm.drop(columns=["etat_admin"], errors="ignore")
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
