import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from icpe_groundwater_exposure.config import (
    GRID_SIZE_M,
    HTML_CRS,
    LON_MAX,
    LON_MIN,
    N_YEARS,
    ROBUST_THRESHOLD,
    WORK_CRS,
)


LAT_MAX_MAINLAND = 52
LAT_MIN_MAINLAND = 41


def classify_groundwater_signal(variation_cm) -> str:
    if pd.isna(variation_cm):
        return "no_data"
    if variation_cm <= -20:
        return "declining"
    return "not_declining"


def classify_pressure_signal(volume_m3, threshold_m3) -> str:
    if pd.isna(volume_m3) or volume_m3 <= 0:
        return "low"
    if volume_m3 >= threshold_m3:
        return "high"
    return "low"


def classify_exposure_2x2(groundwater_signal: str, pressure_signal: str) -> str:
    if groundwater_signal == "no_data":
        return "unclassified_no_groundwater_data"
    if groundwater_signal == "declining" and pressure_signal == "high":
        return "high_pressure_declining_groundwater"
    if groundwater_signal == "declining" and pressure_signal == "low":
        return "low_pressure_declining_groundwater"
    if groundwater_signal == "not_declining" and pressure_signal == "high":
        return "high_pressure_non_declining_groundwater"
    return "low_pressure_non_declining_groundwater"


def filter_mainland_wgs84(df: pd.DataFrame, lon_col: str, lat_col: str) -> pd.DataFrame:
    out = df.copy()
    out[lon_col] = pd.to_numeric(out[lon_col], errors="coerce")
    out[lat_col] = pd.to_numeric(out[lat_col], errors="coerce")
    out = out.dropna(subset=[lon_col, lat_col]).copy()
    out = out[
        (out[lon_col] >= LON_MIN)
        & (out[lon_col] <= LON_MAX)
        & (out[lat_col] >= LAT_MIN_MAINLAND)
        & (out[lat_col] <= LAT_MAX_MAINLAND)
    ].copy()
    return out


def stations_to_work_gdf(stations: pd.DataFrame) -> gpd.GeoDataFrame:
    required = ["code_bss", "x", "y", "slope_m_per_year"]
    missing = [col for col in required if col not in stations.columns]
    if missing:
        raise ValueError(f"Missing station columns: {missing}")

    out = stations[required].copy()
    out = filter_mainland_wgs84(out, "x", "y")
    out["slope_m_per_year"] = pd.to_numeric(out["slope_m_per_year"], errors="coerce")
    out = out.dropna(subset=["slope_m_per_year"]).copy()
    out["variation_20y_cm"] = out["slope_m_per_year"] * N_YEARS * 100
    return gpd.GeoDataFrame(
        out,
        geometry=gpd.points_from_xy(out["x"], out["y"]),
        crs=HTML_CRS,
    ).to_crs(WORK_CRS)


def bnpe_to_work_gdf(withdrawals: pd.DataFrame) -> gpd.GeoDataFrame:
    required = ["ouvrage_sandre", "longitude", "latitude", "volume_m3", "usage_code"]
    missing = [col for col in required if col not in withdrawals.columns]
    if missing:
        raise ValueError(f"Missing BNPE columns: {missing}")

    out = withdrawals.copy()
    out = out[out["usage_code"].isin(["IND", "IRR"])].copy()
    out = filter_mainland_wgs84(out, "longitude", "latitude")
    out["volume_m3"] = pd.to_numeric(out["volume_m3"], errors="coerce").fillna(0)
    return gpd.GeoDataFrame(
        out,
        geometry=gpd.points_from_xy(out["longitude"], out["latitude"]),
        crs=HTML_CRS,
    ).to_crs(WORK_CRS)


def create_grid(bounds, cell_size_m: int = GRID_SIZE_M) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = bounds
    minx = np.floor(minx / cell_size_m) * cell_size_m
    miny = np.floor(miny / cell_size_m) * cell_size_m
    maxx = np.ceil(maxx / cell_size_m) * cell_size_m
    maxy = np.ceil(maxy / cell_size_m) * cell_size_m

    cells = []
    cell_id = 0
    for x0 in np.arange(minx, maxx, cell_size_m):
        for y0 in np.arange(miny, maxy, cell_size_m):
            x1 = x0 + cell_size_m
            y1 = y0 + cell_size_m
            cells.append(
                {
                    "grid_id": f"grid_{int(x0)}_{int(y0)}",
                    "grid_xmin": x0,
                    "grid_ymin": y0,
                    "grid_xmax": x1,
                    "grid_ymax": y1,
                    "cell_seq": cell_id,
                    "geometry": box(x0, y0, x1, y1),
                }
            )
            cell_id += 1
    return gpd.GeoDataFrame(cells, crs=WORK_CRS)


def aggregate_stations_to_grid(grid: gpd.GeoDataFrame, stations_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    joined = gpd.sjoin(stations_gdf, grid[["grid_id", "geometry"]], how="inner", predicate="within")
    agg = (
        joined.groupby("grid_id", as_index=False)
        .agg(
            station_count=("code_bss", "nunique"),
            groundwater_median_variation_20y_cm=("variation_20y_cm", "median"),
            groundwater_mean_variation_20y_cm=("variation_20y_cm", "mean"),
        )
    )
    agg["groundwater_signal_robust"] = agg["station_count"] >= ROBUST_THRESHOLD
    agg["groundwater_signal_class"] = agg["groundwater_median_variation_20y_cm"].apply(classify_groundwater_signal)
    return agg


def aggregate_withdrawals_to_grid(grid: gpd.GeoDataFrame, bnpe_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    joined = gpd.sjoin(bnpe_gdf, grid[["grid_id", "geometry"]], how="inner", predicate="within")
    agg = (
        joined.groupby("grid_id", as_index=False)
        .agg(
            withdrawal_count=("ouvrage_sandre", "nunique"),
            withdrawal_volume_m3=("volume_m3", "sum"),
            ind_withdrawal_count=("usage_code", lambda s: int((s == "IND").sum())),
            irr_withdrawal_count=("usage_code", lambda s: int((s == "IRR").sum())),
            ind_withdrawal_volume_m3=("volume_m3", lambda s: float(s[joined.loc[s.index, "usage_code"] == "IND"].sum())),
            irr_withdrawal_volume_m3=("volume_m3", lambda s: float(s[joined.loc[s.index, "usage_code"] == "IRR"].sum())),
        )
    )
    return agg


def build_exposure_grid(stations: pd.DataFrame, withdrawals: pd.DataFrame) -> tuple[gpd.GeoDataFrame, float]:
    stations_gdf = stations_to_work_gdf(stations)
    bnpe_gdf = bnpe_to_work_gdf(withdrawals)

    bounds = (
        min(stations_gdf.total_bounds[0], bnpe_gdf.total_bounds[0]),
        min(stations_gdf.total_bounds[1], bnpe_gdf.total_bounds[1]),
        max(stations_gdf.total_bounds[2], bnpe_gdf.total_bounds[2]),
        max(stations_gdf.total_bounds[3], bnpe_gdf.total_bounds[3]),
    )
    grid = create_grid(bounds)

    station_agg = aggregate_stations_to_grid(grid, stations_gdf)
    withdrawal_agg = aggregate_withdrawals_to_grid(grid, bnpe_gdf)

    out = grid.merge(station_agg, on="grid_id", how="left").merge(withdrawal_agg, on="grid_id", how="left")

    fill_zero_cols = [
        "withdrawal_count",
        "withdrawal_volume_m3",
        "ind_withdrawal_count",
        "irr_withdrawal_count",
        "ind_withdrawal_volume_m3",
        "irr_withdrawal_volume_m3",
    ]
    for col in fill_zero_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    out["station_count"] = pd.to_numeric(out["station_count"], errors="coerce").fillna(0).astype(int)
    out["groundwater_signal_robust"] = out["groundwater_signal_robust"].fillna(False)
    out["groundwater_signal_class"] = out["groundwater_median_variation_20y_cm"].apply(classify_groundwater_signal)

    positive_volumes = out.loc[out["withdrawal_volume_m3"] > 0, "withdrawal_volume_m3"]
    pressure_threshold_m3 = float(positive_volumes.quantile(0.75)) if len(positive_volumes) else 0.0
    out["pressure_signal_class"] = out["withdrawal_volume_m3"].apply(
        lambda v: classify_pressure_signal(v, pressure_threshold_m3)
    )
    out["exposure_class_2x2"] = out.apply(
        lambda row: classify_exposure_2x2(row["groundwater_signal_class"], row["pressure_signal_class"]),
        axis=1,
    )

    out["grid_size_km"] = GRID_SIZE_M / 1000
    out["pressure_threshold_m3"] = pressure_threshold_m3
    out["has_withdrawal_data"] = out["withdrawal_count"] > 0
    out["has_groundwater_data"] = out["station_count"] > 0
    out["is_classified"] = out["has_groundwater_data"]
    out["is_robustly_classified"] = out["station_count"] >= ROBUST_THRESHOLD
    out["context_only_withdrawal"] = out["has_withdrawal_data"] & ~out["has_groundwater_data"]
    return out, pressure_threshold_m3
