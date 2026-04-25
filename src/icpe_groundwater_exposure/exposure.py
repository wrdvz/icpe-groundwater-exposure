import geopandas as gpd
import numpy as np
import pandas as pd

from icpe_groundwater_exposure.config import HTML_CRS, N_YEARS, WORK_CRS


def classify_local_signal(variation_cm) -> str:
    if pd.isna(variation_cm):
        return "No nearby station"
    if variation_cm <= -50:
        return "Strong decline"
    if variation_cm <= -20:
        return "Moderate decline"
    if variation_cm < 20:
        return "Near stable"
    if variation_cm < 50:
        return "Moderate rise"
    return "Strong rise"


def local_signal_marker(variation_cm, is_solid: bool) -> str:
    if pd.isna(variation_cm):
        return "no_signal"
    if variation_cm <= -20:
        return "decline_solid" if is_solid else "decline_limited"
    if variation_cm >= 20:
        return "rise_solid" if is_solid else "rise_limited"
    return "stable_solid" if is_solid else "stable_limited"


def prepare_station_trends(stations: pd.DataFrame) -> gpd.GeoDataFrame:
    required = ["code_bss", "x", "y", "slope_m_per_year"]
    missing = [col for col in required if col not in stations.columns]
    if missing:
        raise ValueError(f"Missing station trend columns: {missing}")

    out = stations[required].copy()
    out["x"] = pd.to_numeric(out["x"], errors="coerce")
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    out["slope_m_per_year"] = pd.to_numeric(out["slope_m_per_year"], errors="coerce")
    out = out.dropna(subset=["x", "y", "slope_m_per_year"]).copy()
    out["variation_20y_cm"] = out["slope_m_per_year"] * N_YEARS * 100

    return gpd.GeoDataFrame(
        out,
        geometry=gpd.points_from_xy(out["x"], out["y"]),
        crs=HTML_CRS,
    ).to_crs(WORK_CRS)


def prepare_icpe_sites(icpe: pd.DataFrame) -> gpd.GeoDataFrame:
    required = ["code_aiot", "x", "y"]
    missing = [col for col in required if col not in icpe.columns]
    if missing:
        raise ValueError(f"Missing ICPE columns: {missing}")

    out = icpe.copy()
    out["x"] = pd.to_numeric(out["x"], errors="coerce")
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    out = out.dropna(subset=["code_aiot", "x", "y"]).copy()

    return gpd.GeoDataFrame(
        out,
        geometry=gpd.points_from_xy(out["x"], out["y"]),
        crs=WORK_CRS,
    )


def compute_icpe_groundwater_context(
    icpe: pd.DataFrame,
    stations: pd.DataFrame,
    radius_m: int,
    robust_threshold: int,
) -> pd.DataFrame:
    gdf_icpe = prepare_icpe_sites(icpe)
    gdf_stations = prepare_station_trends(stations)

    station_sindex = gdf_stations.sindex
    records = []

    for site in gdf_icpe.itertuples():
        candidates_idx = list(station_sindex.query(site.geometry.buffer(radius_m), predicate="intersects"))
        candidates = gdf_stations.iloc[candidates_idx].copy()

        if not candidates.empty:
            distances = candidates.geometry.distance(site.geometry)
            nearby = candidates.loc[distances <= radius_m].copy()
            nearby_distances = distances.loc[nearby.index]
        else:
            nearby = candidates
            nearby_distances = pd.Series(dtype=float)

        variations = nearby["variation_20y_cm"] if not nearby.empty else pd.Series(dtype=float)
        n_stations = int(nearby["code_bss"].nunique()) if not nearby.empty else 0
        median_variation = float(np.nanmedian(variations)) if n_stations else np.nan
        mean_variation = float(np.nanmean(variations)) if n_stations else np.nan
        min_distance_km = float(nearby_distances.min() / 1000) if n_stations else np.nan
        is_solid = n_stations >= robust_threshold

        records.append(
            {
                "code_aiot": site.code_aiot,
                "n_stations_20km": n_stations,
                "median_variation_20y_cm_20km": median_variation,
                "mean_variation_20y_cm_20km": mean_variation,
                "min_station_distance_km": min_distance_km,
                "is_signal_solid": is_solid,
                "local_signal_class": classify_local_signal(median_variation),
                "local_signal_marker": local_signal_marker(median_variation, is_solid),
            }
        )

    context = pd.DataFrame(records)
    return icpe.merge(context, on="code_aiot", how="left")
