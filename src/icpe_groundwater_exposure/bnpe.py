import geopandas as gpd
import pandas as pd

from icpe_groundwater_exposure.config import HTML_CRS, WORK_CRS


BNPE_COLUMNS = {
    "Année": "year",
    "Code Sandre de l'ouvrage ": "ouvrage_sandre",
    "Nom de l'ouvrage": "ouvrage_name",
    "Code alternatif de l'ouvrage": "ouvrage_alt_code",
    "Origine du code alternatif": "ouvrage_alt_origin",
    "Département": "departement",
    "Code INSEE": "code_insee",
    "Commune": "commune",
    "Volume (m3)": "volume_m3",
    "Code usage BNPE": "usage_code",
    "Libellé usage BNPE": "usage_label",
    "Type d'eau": "water_type",
    "Longitude": "longitude",
    "Latitude": "latitude",
    "Précision de la localisation": "location_precision",
    "Nature du point de prélèvement": "withdrawal_point_type",
    "Code BSS": "code_bss",
    "Code BDLISA": "code_bdlisa",
    "Libellé entité hydrologique BDLISA": "bdlisa_label",
}


def load_raw_bnpe_withdrawals(path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    missing = [col for col in BNPE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing BNPE columns: {missing}")
    return df[list(BNPE_COLUMNS)].rename(columns=BNPE_COLUMNS).copy()


def normalize_bnpe_withdrawals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["volume_m3"] = pd.to_numeric(out["volume_m3"], errors="coerce").fillna(0)
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["departement"] = out["departement"].astype("string").str.zfill(2)
    out["code_insee"] = out["code_insee"].astype("string").str.zfill(5)
    out["usage_code"] = out["usage_code"].astype("string")
    out["water_type"] = out["water_type"].astype("string")
    out = out.dropna(subset=["ouvrage_sandre", "longitude", "latitude"]).copy()
    out = out[out["water_type"] == "SOUT"].copy()
    out = out.reset_index(drop=True)
    return out


def bnpe_to_lambert93(df: pd.DataFrame) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs=HTML_CRS,
    )
    return gdf.to_crs(WORK_CRS)
