"""
Fetch Hubeau groundwater data into a local DuckDB database.

This script builds two tables in a single DuckDB file:

1. groundwater
   Raw groundwater level time series from:
   /api/v1/niveaux_nappes/chroniques

2. stations
   Station metadata from:
   /api/v1/niveaux_nappes/stations

The stations table aims to provide:
- code_bss
- x
- y
- departement
- region

Notes
-----
- x and y are stored in WGS84 longitude / latitude coordinates.
- region is derived by spatial join if a local regions GeoJSON is available.
- department is taken from Hubeau station metadata when possible, with a fallback
  from code_commune when available.
- The script is resumable:
  already loaded stations are skipped in both tables.
"""

from pathlib import Path
import time

import duckdb
import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point

from icpe_groundwater_exposure.config import (
    START_YEAR,
    END_YEAR,
    RAW_DUCKDB_FILE,
    TRENDS_FILE,
    RAW_DIR,
)
from icpe_groundwater_exposure.utils import ensure_dirs

from typing import Optional



# ============================================================
# CONFIG
# ============================================================

START_DATE = f"{START_YEAR}-01-01"
END_DATE = f"{END_YEAR}-12-31"

API_CHRONIQUES_URL = "https://hubeau.eaufrance.fr/api/v1/niveaux_nappes/chroniques"
API_STATIONS_URL = "https://hubeau.eaufrance.fr/api/v1/niveaux_nappes/stations"

RAW_TABLE = "groundwater"
STATIONS_TABLE = "stations"

OUT_SAMPLE_CSV = RAW_DIR / "sample_groundwater_timeseries.csv"

REQUEST_SLEEP_SECONDS = 0.1
SAMPLE_N_STATIONS = 5
SAMPLE_MAX_ROWS = 1000

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Adapt this path if your regions file is elsewhere
REGIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "geo"
    / "regions_france.geojson"
)

# ============================================================
# HELPERS
# ============================================================

def load_station_codes(trends_file: Path) -> list[str]:
    df = pd.read_csv(trends_file, dtype={"code_bss": "string"})
    if "code_bss" not in df.columns:
        raise ValueError(f"Missing 'code_bss' column in {trends_file}")
    return df["code_bss"].dropna().astype("string").unique().tolist()


def infer_departement_from_code_commune(code_commune) -> Optional[str]:
    if pd.isna(code_commune):
        return None

    code = str(code_commune).strip()
    if not code:
        return None

    # Very simple metropolitan logic.
    # Corsica special case kept.
    if code.startswith("2A") or code.startswith("2B"):
        return code[:2]

    return code[:2]


def first_existing_value(row: pd.Series, candidates: list[str]):
    for col in candidates:
        if col in row.index and pd.notna(row[col]):
            return row[col]
    return None


def load_regions_gdf() -> Optional[gpd.GeoDataFrame]:
    if not REGIONS_FILE.exists():
        print(f"⚠ Regions file not found: {REGIONS_FILE}")
        print("⚠ Region will remain null in stations table.")
        return None

    gdf_regions = gpd.read_file(REGIONS_FILE).to_crs("EPSG:4326")
    return gdf_regions


def extract_region_name_columns(gdf_regions: gpd.GeoDataFrame) -> Optional[str]:
    candidates = [
        "region",
        "nom_region",
        "libelle",
        "nom",
        "name",
        "NAME_1",
    ]
    for col in candidates:
        if col in gdf_regions.columns:
            return col
    return None


def fetch_station_timeseries(code_bss: str, max_retries: int = 3) -> pd.DataFrame:
    params = {
        "code_bss": code_bss,
        "date_debut_mesure": START_DATE,
        "date_fin_mesure": END_DATE,
        "size": 20000,
        "sort": "asc",
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(API_CHRONIQUES_URL, params=params, timeout=120)

            if response.status_code != 200:
                print(f"❌ Chroniques error {response.status_code} for station {code_bss}")
                return pd.DataFrame()

            payload = response.json()
            data = payload.get("data", [])

            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data)

            required_cols = ["code_bss", "date_mesure", "niveau_nappe_eau"]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                print(f"⚠ Missing chroniques columns for station {code_bss}: {missing}")
                return pd.DataFrame()

            df = df[required_cols].copy()
            df["code_bss"] = df["code_bss"].astype("string")
            df["date_mesure"] = pd.to_datetime(df["date_mesure"], errors="coerce")
            df["niveau_nappe_eau"] = pd.to_numeric(df["niveau_nappe_eau"], errors="coerce")
            df = df.dropna(subset=["code_bss", "date_mesure", "niveau_nappe_eau"]).copy()

            return df

        except requests.exceptions.ReadTimeout:
            print(f"⏳ Chroniques timeout for {code_bss} (attempt {attempt}/{max_retries})")
            time.sleep(2 * attempt)

        except requests.exceptions.RequestException as e:
            print(f"❌ Chroniques request error for {code_bss}: {e}")
            time.sleep(2 * attempt)

    print(f"❌ Chroniques failed after retries for station {code_bss}")
    return pd.DataFrame()


def fetch_station_metadata(code_bss: str, max_retries: int = 3) -> pd.DataFrame:
    params = {
        "code_bss": code_bss,
        "size": 200,
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(API_STATIONS_URL, params=params, timeout=120)

            if response.status_code != 200:
                print(f"❌ Stations error {response.status_code} for station {code_bss}")
                return pd.DataFrame()

            payload = response.json()
            data = payload.get("data", [])

            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data)

            # Keep first row only if endpoint returns duplicates
            df = df.head(1).copy()

            row = df.iloc[0]

            longitude = first_existing_value(
                row,
                ["longitude", "x", "coord_x"]
            )
            latitude = first_existing_value(
                row,
                ["latitude", "y", "coord_y"]
            )

            code_commune = first_existing_value(
                row,
                ["code_commune_insee", "code_commune", "insee_commune"]
            )

            departement = first_existing_value(
                row,
                ["code_departement", "departement", "libelle_departement", "nom_departement"]
            )

            if pd.isna(departement) or departement is None:
                departement = infer_departement_from_code_commune(code_commune)

            out = pd.DataFrame([{
                "code_bss": code_bss,
                "x": pd.to_numeric(longitude, errors="coerce"),
                "y": pd.to_numeric(latitude, errors="coerce"),
                "departement": None if pd.isna(departement) else str(departement),
                "region": None,
            }])

            out["code_bss"] = out["code_bss"].astype("string")
            out["x"] = pd.to_numeric(out["x"], errors="coerce")
            out["y"] = pd.to_numeric(out["y"], errors="coerce")

            return out

        except requests.exceptions.ReadTimeout:
            print(f"⏳ Stations timeout for {code_bss} (attempt {attempt}/{max_retries})")
            time.sleep(2 * attempt)

        except requests.exceptions.RequestException as e:
            print(f"❌ Stations request error for {code_bss}: {e}")
            time.sleep(2 * attempt)

    print(f"❌ Stations metadata failed after retries for station {code_bss}")
    return pd.DataFrame()


def enrich_region_from_geometry(df_stations: pd.DataFrame, gdf_regions: Optional[gpd.GeoDataFrame]) -> pd.DataFrame:
    if gdf_regions is None or df_stations.empty:
        return df_stations

    region_col = extract_region_name_columns(gdf_regions)
    if region_col is None:
        print("⚠ Could not detect region name column in regions file.")
        return df_stations

    gdf_pts = gpd.GeoDataFrame(
        df_stations.copy(),
        geometry=gpd.points_from_xy(df_stations["x"], df_stations["y"]),
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(
        gdf_pts,
        gdf_regions[[region_col, "geometry"]],
        how="left",
        predicate="within",
    )

    joined["region"] = joined[region_col]
    joined = joined.drop(columns=["geometry", "index_right", region_col], errors="ignore")

    return pd.DataFrame(joined)


def init_duckdb(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {RAW_TABLE} (
            code_bss VARCHAR,
            date_mesure TIMESTAMP,
            niveau_nappe_eau DOUBLE
        )
    """)

    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {STATIONS_TABLE} (
            code_bss VARCHAR,
            x DOUBLE,
            y DOUBLE,
            departement VARCHAR,
            region VARCHAR
        )
    """)


def station_timeseries_already_loaded(con: duckdb.DuckDBPyConnection, code_bss: str) -> bool:
    result = con.execute(
        f"""
        SELECT COUNT(*) AS n
        FROM {RAW_TABLE}
        WHERE code_bss = ?
        """,
        [code_bss]
    ).fetchone()

    return result[0] > 0


def station_metadata_already_loaded(con: duckdb.DuckDBPyConnection, code_bss: str) -> bool:
    result = con.execute(
        f"""
        SELECT COUNT(*) AS n
        FROM {STATIONS_TABLE}
        WHERE code_bss = ?
        """,
        [code_bss]
    ).fetchone()

    return result[0] > 0


def insert_timeseries(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> None:
    con.register("tmp_groundwater", df)
    con.execute(f"""
        INSERT INTO {RAW_TABLE}
        SELECT code_bss, date_mesure, niveau_nappe_eau
        FROM tmp_groundwater
    """)
    con.unregister("tmp_groundwater")


def insert_station_metadata(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> None:
    con.register("tmp_stations", df)
    con.execute(f"""
        INSERT INTO {STATIONS_TABLE}
        SELECT code_bss, x, y, departement, region
        FROM tmp_stations
    """)
    con.unregister("tmp_stations")


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n========== FETCH HUBEAU TO DUCKDB ==========\n")
    print("Stations source:", TRENDS_FILE)
    print("Output DuckDB:", RAW_DUCKDB_FILE)
    print("Date range:", START_DATE, "→", END_DATE)

    ensure_dirs([RAW_DIR, RAW_DUCKDB_FILE.parent])

    if not TRENDS_FILE.exists():
        raise FileNotFoundError(f"Stations source file not found: {TRENDS_FILE}")

    stations = load_station_codes(TRENDS_FILE)
    print("Nb stations to fetch:", len(stations))

    gdf_regions = load_regions_gdf()

    con = duckdb.connect(str(RAW_DUCKDB_FILE))
    init_duckdb(con)

    total_rows = 0
    total_station_metadata = 0
    sample_frames = []

    for i, code_bss in enumerate(stations, start=1):
        # ----------------------------------------------------
        # 1. Time series
        # ----------------------------------------------------
        if not station_timeseries_already_loaded(con, code_bss):
            df_station = fetch_station_timeseries(code_bss)

            if not df_station.empty:
                insert_timeseries(con, df_station)
                total_rows += len(df_station)

                if len(sample_frames) < SAMPLE_N_STATIONS:
                    sample_frames.append(df_station)

        # ----------------------------------------------------
        # 2. Station metadata
        # ----------------------------------------------------
        if not station_metadata_already_loaded(con, code_bss):
            df_meta = fetch_station_metadata(code_bss)

            if not df_meta.empty:
                df_meta = enrich_region_from_geometry(df_meta, gdf_regions)
                insert_station_metadata(con, df_meta)
                total_station_metadata += len(df_meta)

        if i % 25 == 0 or i == len(stations):
            print(
                f"{i}/{len(stations)} stations processed | "
                f"chroniques rows inserted this run: {total_rows} | "
                f"station metadata inserted this run: {total_station_metadata}"
            )

        time.sleep(REQUEST_SLEEP_SECONDS)

    print("\n=== DUCKDB CHECK ===")
    print(con.execute("SHOW TABLES").df())
    print(con.execute(f"SELECT COUNT(*) AS n_rows FROM {RAW_TABLE}").df())
    print(con.execute(f"SELECT COUNT(*) AS n_rows FROM {STATIONS_TABLE}").df())

    # Sample CSV for repo
    if sample_frames:
        df_sample = pd.concat(sample_frames, ignore_index=True).head(SAMPLE_MAX_ROWS)
        df_sample.to_csv(OUT_SAMPLE_CSV, index=False)
        print("Saved sample CSV:", OUT_SAMPLE_CSV)

    print("\nGroundwater preview:")
    print(con.execute(f"SELECT * FROM {RAW_TABLE} LIMIT 5").df())

    print("\nStations preview:")
    print(con.execute(f"SELECT * FROM {STATIONS_TABLE} LIMIT 5").df())

    con.close()

    print("\n✅ Done\n")


if __name__ == "__main__":
    main()