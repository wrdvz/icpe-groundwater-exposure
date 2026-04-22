"""
Compute station-level groundwater trends for the France groundwater project.

What this script does
---------------------
1. Connects to the local DuckDB database containing:
   - raw Hubeau groundwater time series in the `groundwater` table
   - station metadata in the `stations` table

2. Builds annual mean groundwater levels per station using SQL.
   This reduces short-term noise and irregular measurement frequency while
   keeping a transparent and reproducible intermediate table.

3. Applies a minimal quality filter at station level:
   - at least MIN_POINTS raw observations
   - at least MIN_YEARS distinct years available

4. Runs an OLS linear regression in Python on annual means for each station.
   The regression returns:
   - slope in meters per year
   - 95% confidence interval for slope
   - intercept and intercept CI
   - R²
   - p-value
   - standardized slope on z-scored annual means

5. Merges station metadata from DuckDB (`stations` table):
   - code_bss
   - x
   - y
   - departement
   - region

6. Enriches department and region labels:
   - departement_name from DEPARTEMENT_MAPPING
   - region from DEPARTEMENT_TO_REGION

7. Filters stations to mainland France using longitude/latitude bounds.

8. Exports:
   - annual means CSV
   - station trends CSV

Why this design
---------------
- DuckDB is used for raw time series storage, station metadata storage,
  and annual aggregation.
- Python is used for the final regression because it is easier to compute
  full diagnostics there than in plain SQL.
- The output station_trends CSV is the main reusable input for downstream
  mapping and spatial aggregation scripts.
"""

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from icpe_groundwater_exposure.config import (
    START_YEAR,
    END_YEAR,
    RAW_DUCKDB_FILE,
    TRENDS_FILE,
    ANNUAL_MEANS_FILE,
    LON_MIN,
    LON_MAX,
    LAT_MIN,
    LAT_MAX,
    DEPARTEMENT_MAPPING,
    DEPARTEMENT_TO_REGION,
)
from icpe_groundwater_exposure.utils import (
    ensure_dirs,
    get_departement_name,
    get_region_from_departement,
)

# ============================================================
# CONFIG
# ============================================================

RAW_TABLE = "groundwater"
STATIONS_TABLE = "stations"
ANNUAL_TABLE = "annual_means"

MIN_POINTS = 800
MIN_YEARS = 19

ensure_dirs([ANNUAL_MEANS_FILE.parent, TRENDS_FILE.parent])

# ============================================================
# HELPERS
# ============================================================

def load_station_metadata_from_duckdb(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Load station metadata directly from the DuckDB `stations` table.

    Expected columns:
    - code_bss
    - x
    - y
    - departement
    - region
    """
    tables = con.execute("SHOW TABLES").fetchdf()

    if STATIONS_TABLE not in tables["name"].tolist():
        print("⚠ No `stations` table found in DuckDB. x/y/departement/region will be null.")
        return pd.DataFrame(columns=["code_bss", "x", "y", "departement", "region"])

    df_meta = con.execute(f"""
        SELECT
            code_bss,
            x,
            y,
            departement,
            region
        FROM {STATIONS_TABLE}
    """).df()

    if df_meta.empty:
        print("⚠ `stations` table is empty. x/y/departement/region will be null.")
        return pd.DataFrame(columns=["code_bss", "x", "y", "departement", "region"])

    df_meta["code_bss"] = df_meta["code_bss"].astype("string")
    df_meta["x"] = pd.to_numeric(df_meta["x"], errors="coerce")
    df_meta["y"] = pd.to_numeric(df_meta["y"], errors="coerce")

    if "departement" in df_meta.columns:
        df_meta["departement"] = df_meta["departement"].astype("string")

    if "region" in df_meta.columns:
        df_meta["region"] = df_meta["region"].astype("string")

    df_meta = df_meta.drop_duplicates(subset=["code_bss"]).copy()

    return df_meta


def ols_with_ci(x_years: np.ndarray, y_vals: np.ndarray, alpha: float = 0.05):
    """
    Run OLS regression and return slope/intercept diagnostics with 95% CI.
    """
    n = len(x_years)
    if n < 3:
        return None

    slope, intercept, r_value, p_value, std_err = stats.linregress(x_years, y_vals)
    r2 = r_value ** 2

    dfree = n - 2
    if dfree <= 0 or np.isnan(std_err):
        return None

    tcrit = stats.t.ppf(1 - alpha / 2, dfree)

    slope_ci_low = slope - tcrit * std_err
    slope_ci_high = slope + tcrit * std_err

    x_mean = x_years.mean()
    sxx = np.sum((x_years - x_mean) ** 2)
    if sxx == 0:
        return None

    y_hat = slope * x_years + intercept
    resid = y_vals - y_hat
    s_err = np.sqrt(np.sum(resid ** 2) / dfree)

    intercept_se = s_err * np.sqrt((1 / n) + (x_mean ** 2 / sxx))
    intercept_ci_low = intercept - tcrit * intercept_se
    intercept_ci_high = intercept + tcrit * intercept_se

    return {
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "p_value": p_value,
        "n": n,
        "slope_ci95_low": slope_ci_low,
        "slope_ci95_high": slope_ci_high,
        "intercept_ci95_low": intercept_ci_low,
        "intercept_ci95_high": intercept_ci_high,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n========== COMPUTE STATION TRENDS ==========\n")
    print("DuckDB input:", RAW_DUCKDB_FILE)
    print("Annual means output:", ANNUAL_MEANS_FILE)
    print("Station trends output:", TRENDS_FILE)
    print(f"Time window: {START_YEAR} -> {END_YEAR}")
    print(f"Quality filter: MIN_POINTS={MIN_POINTS}, MIN_YEARS={MIN_YEARS}")

    if not RAW_DUCKDB_FILE.exists():
        raise FileNotFoundError(f"DuckDB file not found: {RAW_DUCKDB_FILE}")

    con = duckdb.connect(str(RAW_DUCKDB_FILE))

    tables = con.execute("SHOW TABLES").fetchdf()
    if RAW_TABLE not in tables["name"].tolist():
        raise ValueError(f"Table '{RAW_TABLE}' not found in {RAW_DUCKDB_FILE}")

    print("\n=== RAW CHECK ===")
    print(con.execute(f"SELECT COUNT(*) AS n_rows FROM {RAW_TABLE}").df())

    # --------------------------------------------------------
    # Build annual means in DuckDB
    # --------------------------------------------------------
    print("\n=== BUILD ANNUAL MEANS ===")

    con.execute(f"DROP TABLE IF EXISTS {ANNUAL_TABLE}")

    annual_sql = f"""
    CREATE TABLE {ANNUAL_TABLE} AS
    SELECT
        code_bss,
        EXTRACT(YEAR FROM date_mesure) AS year,
        AVG(niveau_nappe_eau) AS annual_mean,
        COUNT(*) AS n_points_year
    FROM {RAW_TABLE}
    GROUP BY code_bss, year
    """

    con.execute(annual_sql)

    annual = con.execute(f"""
        SELECT *
        FROM {ANNUAL_TABLE}
        ORDER BY code_bss, year
    """).df()

    annual["code_bss"] = annual["code_bss"].astype("string")
    annual["year"] = pd.to_numeric(annual["year"], errors="coerce").astype("Int64")
    annual["annual_mean"] = pd.to_numeric(annual["annual_mean"], errors="coerce")
    annual["n_points_year"] = pd.to_numeric(annual["n_points_year"], errors="coerce").astype("Int64")

    print("Annual rows:", len(annual))
    print("Stations in annual means:", annual["code_bss"].nunique())

    annual.to_csv(ANNUAL_MEANS_FILE, index=False)
    print("Saved annual means:", ANNUAL_MEANS_FILE)

    # --------------------------------------------------------
    # Station quality diagnostics from raw table
    # --------------------------------------------------------
    print("\n=== STATION QUALITY DIAGNOSTICS ===")

    quality = con.execute(f"""
        SELECT
            code_bss,
            COUNT(*) AS n_points,
            COUNT(DISTINCT EXTRACT(YEAR FROM date_mesure)) AS n_years_raw
        FROM {RAW_TABLE}
        GROUP BY code_bss
    """).df()

    quality["code_bss"] = quality["code_bss"].astype("string")
    quality["n_points"] = pd.to_numeric(quality["n_points"], errors="coerce").astype("Int64")
    quality["n_years_raw"] = pd.to_numeric(quality["n_years_raw"], errors="coerce").astype("Int64")

    valid_codes = quality[
        (quality["n_points"] >= MIN_POINTS) &
        (quality["n_years_raw"] >= MIN_YEARS)
    ]["code_bss"].tolist()

    print("Stations passing quality filter:", len(valid_codes))
    print("Stations rejected:", quality["code_bss"].nunique() - len(valid_codes))

    annual_filtered = annual[annual["code_bss"].isin(valid_codes)].copy()

    # --------------------------------------------------------
    # OLS per station on annual means
    # --------------------------------------------------------
    print("\n=== RUN OLS PER STATION ===")

    results = []

    for code_bss, g in annual_filtered.groupby("code_bss"):
        g = g.sort_values("year").dropna(subset=["year", "annual_mean"]).copy()

        if len(g) < MIN_YEARS:
            continue

        x = g["year"].to_numpy(dtype=float)
        y = g["annual_mean"].to_numpy(dtype=float)

        fit = ols_with_ci(x, y)
        if fit is None:
            continue

        y_mean = y.mean()
        y_std = y.std(ddof=1)

        if np.isfinite(y_std) and y_std > 0:
            y_z = (y - y_mean) / y_std
            fit_z = ols_with_ci(x, y_z)
            slope_z = fit_z["slope"] if fit_z else np.nan
        else:
            slope_z = np.nan

        results.append({
            "code_bss": code_bss,
            "n_years": int(fit["n"]),
            "slope_m_per_year": fit["slope"],
            "slope_ci95_low": fit["slope_ci95_low"],
            "slope_ci95_high": fit["slope_ci95_high"],
            "intercept": fit["intercept"],
            "intercept_ci95_low": fit["intercept_ci95_low"],
            "intercept_ci95_high": fit["intercept_ci95_high"],
            "r2": fit["r2"],
            "p_value": fit["p_value"],
            "slope_z_per_year": slope_z,
            "mean_level_m": y_mean,
            "std_level_m": y_std,
            "start_year": int(g["year"].min()),
            "end_year": int(g["year"].max()),
        })

    df_trends = pd.DataFrame(results)

    print("Stations with trend computed:", len(df_trends))

    # --------------------------------------------------------
    # Merge metadata from DuckDB
    # --------------------------------------------------------
    print("\n=== MERGE METADATA ===")

    df_meta = load_station_metadata_from_duckdb(con)
    if not df_meta.empty:
        df_trends["code_bss"] = df_trends["code_bss"].astype("string")
        df_trends = df_trends.merge(df_meta, on="code_bss", how="left")

        if "departement" in df_trends.columns:
            df_trends["departement"] = df_trends["departement"].astype("string")

            df_trends["departement_name"] = df_trends["departement"].apply(
                lambda x: get_departement_name(x, DEPARTEMENT_MAPPING)
            )

            df_trends["region"] = df_trends["departement"].apply(
                lambda x: get_region_from_departement(x, DEPARTEMENT_TO_REGION)
            )

        if "x" in df_trends.columns and "y" in df_trends.columns:
            before = len(df_trends)

            df_trends = df_trends[
                df_trends["x"].notna() & df_trends["y"].notna()
            ].copy()

            df_trends = df_trends[
                (df_trends["x"] > LON_MIN) &
                (df_trends["x"] < LON_MAX) &
                (df_trends["y"] > LAT_MIN) &
                (df_trends["y"] < LAT_MAX)
            ].copy()

            print("Stations after mainland filter:", len(df_trends), f"(removed {before - len(df_trends)})")

    # --------------------------------------------------------
    # Export final trends
    # --------------------------------------------------------
    df_trends.to_csv(TRENDS_FILE, index=False)
    print("\nSaved station trends:", TRENDS_FILE)

    # --------------------------------------------------------
    # Quick stats
    # --------------------------------------------------------
    if len(df_trends) > 0:
        avg_slope = df_trends["slope_m_per_year"].mean()
        print("\n=== QUICK STATS ===")
        print(f"Mean slope: {avg_slope:.4f} m/year")
        print(f"Approx. 20-year change: {avg_slope * 20:.2f} m")
        print(df_trends["slope_m_per_year"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))

    con.close()
    print("\n✅ Done\n")


if __name__ == "__main__":
    main()