"""
Parallaxe — 610k
Carte France des variations des nappes souterraines sur 20 ans (2005-2025)
à partir de POLYG_ELEMENTAIRES + TABLE_PILE_ENTITES_NIV1

Méthode
-------
- Base géométrique : POLYG_ELEMENTAIRES, non chevauchants
- Logique verticale : TABLE_PILE_ENTITES_NIV1
- Règle de sélection : OrdRelatif = 1 (entité NV1 affleurante)
- Jointure des stations sur les polygones élémentaires
- Agrégation des tendances stationnelles par CodeEH affleurant
- Reprojection de cette agrégation sur la mosaïque exclusive

Sorties
-------
- PNG statique
- HTML interactif
- CSV station -> polygone affleurant
- CSV agrégation par CodeEH affleurant
- CSV polygone -> CodeEH affleurant
"""

from pathlib import Path
from datetime import datetime
import json
import locale
import sqlite3

import fiona
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib import rcParams, font_manager
import matplotlib as mpl
import plotly.graph_objects as go
from shapely.geometry import LineString, MultiLineString, GeometryCollection
from shapely import affinity


try:
    locale.setlocale(locale.LC_TIME, "fr_FR.UTF-8")
except Exception:
    pass


# ============================================================
# CONFIG
# ============================================================

START_YEAR = 2005
END_YEAR = 2025
N_YEARS = END_YEAR - START_YEAR

ROBUST_THRESHOLD = 5

WORK_CRS = "EPSG:2154"
HTML_CRS = "EPSG:4326"

LON_MIN, LON_MAX = -6, 10
LAT_MIN, LAT_MAX = 41, 52

NO_DATA_COLOR = "#e6e6e6"
HATCH_COLOR = "#7a7a7a"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

AQUIFER_FILE = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "geo"
    / "france_aquiferes"
    / "BDLISA_V3_METRO-gpkg"
    / "BDLISA_V3_METRO.gpkg"
)

TRENDS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "station_trends"
    / f"network70_station_trends_{START_YEAR}_{END_YEAR}.csv"
)

OUT_MAP_DIR = PROJECT_ROOT / "docs"
OUT_MAP_DIR.mkdir(parents=True, exist_ok=True)

OUT_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mapping"
)

OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_MAP_DIR / f"parallaxe_icpe_groundwater_exposure_{START_YEAR}_{END_YEAR}.png"
OUT_HTML_LIGHT = OUT_MAP_DIR / "index.html"
OUT_AGG = OUT_MAP_DIR / f"network70_affleurant_nv1_agg_{START_YEAR}_{END_YEAR}.csv"

OUT_STATION_JOIN = OUT_DATA_DIR / f"network70_station_to_polyg_affleurant_nv1_{START_YEAR}_{END_YEAR}.csv"
OUT_POLYG_AFF = OUT_DATA_DIR / f"bdlisa_polyg_affleurant_nv1_{START_YEAR}_{END_YEAR}.csv"

OUT_GEOJSON_FULL = OUT_DATA_DIR / f"bdlisa_eh_nv1_dissolved_{START_YEAR}_{END_YEAR}.geojson"
OUT_GEOJSON_LIGHT = OUT_DATA_DIR / f"bdlisa_eh_nv1_dissolved_light_{START_YEAR}_{END_YEAR}.geojson"

# Typography
robotocon_path = (
    PROJECT_ROOT
    / "assets"
    / "fonts"
    / "Roboto_Condensed"
    / "RobotoCondensed-VariableFont_wght.ttf"
)
font_manager.fontManager.addfont(str(robotocon_path))
rcParams["font.family"] = "Roboto Condensed"
mpl.rcParams["hatch.linewidth"] = 0.35


# ============================================================
# HELPERS
# ============================================================

def classify_variation_cm(v):
    if pd.isna(v):
        return "No data"
    if v <= -50:
        return "Forte baisse (≤ -50 cm)"
    if v <= -20:
        return "Baisse modérée (-50 à -20 cm)"
    if v < 20:
        return "Quasi stable (-20 à +20 cm)"
    if v < 50:
        return "Hausse modérée (+20 à +50 cm)"
    return "Forte hausse (≥ +50 cm)"


def find_coord_columns(df):
    lon_candidates = ["lon", "longitude", "x"]
    lat_candidates = ["lat", "latitude", "y"]

    lon_col = next((c for c in lon_candidates if c in df.columns), None)
    lat_col = next((c for c in lat_candidates if c in df.columns), None)

    if lon_col is None or lat_col is None:
        raise ValueError(f"Impossible de trouver les colonnes de coordonnées. Colonnes: {df.columns.tolist()}")

    return lon_col, lat_col


def format_station_count(n: int) -> str:
    return f"{n:,}".replace(",", " ")


COLOR_MAP = {
    "Forte baisse (≤ -50 cm)": "#c0392b",
    "Baisse modérée (-50 à -20 cm)": "#e26f5d",
    "Quasi stable (-20 à +20 cm)": "#f4b5b2",
    "Hausse modérée (+20 à +50 cm)": "#4e64a8",
    "Forte hausse (≥ +50 cm)": "#1c2864",
    "No data": NO_DATA_COLOR,
}

LABELS = [
    "Forte baisse (≤ -50 cm)",
    "Baisse modérée (-50 à -20 cm)",
    "Quasi stable (-20 à +20 cm)",
    "Hausse modérée (+20 à +50 cm)",
    "Forte hausse (≥ +50 cm)",
]

def make_hatch_lines_for_polygon(geom, spacing=5000, angle_deg=-45):
    if geom is None or geom.is_empty:
        return []

    minx, miny, maxx, maxy = geom.bounds
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2

    diag = ((maxx - minx)**2 + (maxy - miny)**2)**0.5
    length = diag * 2

    lines = []
    y_vals = np.arange(miny - diag, maxy + diag, spacing)

    for y in y_vals:
        base = LineString([
            (minx - diag, y),
            (maxx + diag, y)
        ])
        rotated = affinity.rotate(base, angle_deg, origin=(cx, cy))
        clipped = rotated.intersection(geom)

        if not clipped.is_empty:
            lines.append(clipped)

    return lines

# ============================================================
# DEBUG LAYERS
# ============================================================

print(fiona.listlayers(str(AQUIFER_FILE)))


# ============================================================
# LOAD STATIONS
# ============================================================

print("\n=== LOAD STATIONS ===")
df_st = pd.read_csv(TRENDS_FILE)

lon_col, lat_col = find_coord_columns(df_st)

df_st[lon_col] = pd.to_numeric(df_st[lon_col], errors="coerce")
df_st[lat_col] = pd.to_numeric(df_st[lat_col], errors="coerce")
df_st["slope_m_per_year"] = pd.to_numeric(df_st["slope_m_per_year"], errors="coerce")

df_st = df_st.dropna(subset=[lon_col, lat_col, "slope_m_per_year"]).copy()

df_st = df_st[
    (df_st[lon_col] > LON_MIN) &
    (df_st[lon_col] < LON_MAX) &
    (df_st[lat_col] > LAT_MIN) &
    (df_st[lat_col] < LAT_MAX)
].copy()

df_st["variation_20y_cm"] = df_st["slope_m_per_year"] * N_YEARS * 100
df_st["station_class"] = df_st["variation_20y_cm"].apply(classify_variation_cm)

gdf_st = gpd.GeoDataFrame(
    df_st,
    geometry=gpd.points_from_xy(df_st[lon_col], df_st[lat_col]),
    crs=HTML_CRS
).to_crs(WORK_CRS)

STATION_COUNT = len(gdf_st)

print("Stations loaded:", len(gdf_st))
print("Stations uniques:", gdf_st["code_bss"].nunique())


# ============================================================
# LOAD POLYG_ELEMENTAIRES
# ============================================================

print("\n=== LOAD POLYG_ELEMENTAIRES ===")
gdf_poly = gpd.read_file(AQUIFER_FILE, layer="polyg_elementaires")
gdf_poly.columns = [c.strip() for c in gdf_poly.columns]

print(gdf_poly.columns.tolist())

gdf_poly = gdf_poly.to_crs(WORK_CRS)
gdf_poly["codepoly"] = gdf_poly["codepoly"].astype(str)

print("Elementary polygons:", len(gdf_poly))


# ============================================================
# LOAD TABLE_PILE_ENTITES_NIV1
# ============================================================

print("\n=== LOAD TABLE_PILE_ENTITES_NIV1 ===")
with sqlite3.connect(AQUIFER_FILE) as conn:
    df_pile = pd.read_sql_query("SELECT * FROM table_pile_entites_niv1", conn)

df_pile.columns = [c.strip() for c in df_pile.columns]

print(df_pile.columns.tolist())
print(df_pile.head())

df_pile["OrdRelatif"] = pd.to_numeric(df_pile["OrdRelatif"], errors="coerce")
df_pile["CodeEH"] = pd.to_numeric(df_pile["CodeEH"], errors="coerce")
df_pile["CodePoly"] = df_pile["CodePoly"].astype(str)

df_surface = df_pile[df_pile["OrdRelatif"] == 1].copy()

print("Rows in pile:", len(df_pile))
print("Rows with OrdRelatif = 1:", len(df_surface))
print(df_surface.head())

df_surface = df_surface[["CodePoly", "CodeEH"]].drop_duplicates()

print("Unique CodePoly affleurants:", df_surface["CodePoly"].nunique())
print("Unique CodeEH affleurants:", df_surface["CodeEH"].nunique())


# ============================================================
# JOIN POLYGONES -> AQUIFERE AFFLEURANT
# ============================================================

print("\n=== JOIN POLYGONES -> AFFLEURANT NV1 ===")
gdf_poly_aff = gdf_poly.merge(
    df_surface,
    left_on="codepoly",
    right_on="CodePoly",
    how="left"
)

print("Polygones avec CodeEH affleurant:", gdf_poly_aff["CodeEH"].notna().sum(), "/", len(gdf_poly_aff))

gdf_poly_aff["CodeEH"] = pd.to_numeric(gdf_poly_aff["CodeEH"], errors="coerce")

pd.DataFrame(
    gdf_poly_aff[["codepoly", "CodeEH"]].copy()
).to_csv(OUT_POLYG_AFF, index=False)

print("Saved:", OUT_POLYG_AFF)


# ============================================================
# SPATIAL JOIN STATIONS -> POLYGONES ELEMENTAIRES
# ============================================================

print("\n=== SPATIAL JOIN STATIONS -> POLYGONES ===")
gdf_join = gpd.sjoin(
    gdf_st,
    gdf_poly_aff[["codepoly", "CodeEH", "geometry"]],
    how="left",
    predicate="within"
)

print("Stations matchées:", gdf_join["codepoly"].notna().sum())
print("Stations non matchées:", gdf_join["codepoly"].isna().sum())

gdf_join = gdf_join.dropna(subset=["codepoly", "CodeEH"]).copy()

print("Stations retenues après filtre affleurant:", len(gdf_join))
print("Stations uniques retenues:", gdf_join["code_bss"].nunique())

gdf_join[["code_bss", "variation_20y_cm", "station_class", "codepoly", "CodeEH"]].to_csv(
    OUT_STATION_JOIN,
    index=False
)

print("Saved:", OUT_STATION_JOIN)


# ============================================================
# AGGREGATION PAR CodeEH
# ============================================================

print("\n=== AGGREGATION PAR CodeEH ===")
df_agg = (
    gdf_join.groupby("CodeEH", as_index=False)
    .agg(
        n_stations=("code_bss", "nunique"),
        median_variation_20y_cm=("variation_20y_cm", "median"),
        mean_variation_20y_cm=("variation_20y_cm", "mean"),
        min_variation_20y_cm=("variation_20y_cm", "min"),
        max_variation_20y_cm=("variation_20y_cm", "max"),
        p25_variation_20y_cm=("variation_20y_cm", lambda x: np.nanpercentile(x, 25)),
        p75_variation_20y_cm=("variation_20y_cm", lambda x: np.nanpercentile(x, 75)),
    )
)

df_agg["iqr_variation_20y_cm"] = df_agg["p75_variation_20y_cm"] - df_agg["p25_variation_20y_cm"]
df_agg["median_class"] = df_agg["median_variation_20y_cm"].apply(classify_variation_cm)
df_agg["is_fragile"] = df_agg["n_stations"] < ROBUST_THRESHOLD

print("Nb lignes agg:", len(df_agg))
print("Nb CodeEH uniques agg:", df_agg["CodeEH"].nunique())

df_agg.to_csv(OUT_AGG, index=False)
print("Saved:", OUT_AGG)


# ============================================================
# BUILD FINAL MAP
# ============================================================

print("\n=== BUILD FINAL MAP ===")
gdf_map = gdf_poly_aff.merge(
    df_agg,
    on="CodeEH",
    how="left"
)

gdf_map["median_class"] = gdf_map["median_class"].fillna("No data")
gdf_map["color"] = gdf_map["median_class"].map(COLOR_MAP)
gdf_map["is_fragile"] = gdf_map["is_fragile"].fillna(False)

print(gdf_map.head())
print(gdf_map.columns.tolist())
print("Part des polygones avec donnée:", gdf_map["median_variation_20y_cm"].notna().mean())




# ============================================================
# PNG PRESENTATION CLASSIQUE
# ============================================================

print("\n=== EXPORT PNG ===")

FIG_W = 4000 / 300
FIG_H = 5000 / 300

fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=300)

map_width = 0.85
map_height = map_width * (4 / 5)
map_left = (1 - map_width) / 2
map_bottom = 0.15

ax = fig.add_axes([map_left, map_bottom, map_width, map_height])

TITLE_X = 0.07
TITLE_Y = 0.95
TITLE_SPACING = 0.032
SUB_SPACING = 0.036
BAR_SPACING = 0.018

station_count_fmt = format_station_count(STATION_COUNT)

fig.text(
    TITLE_X,
    TITLE_Y,
    "Variation des nappes souterraines sur 20 ans (2005-2025)",
    ha="left",
    va="top",
    fontsize=28,
    fontweight="medium"
)

fig.text(
    TITLE_X,
    TITLE_Y - TITLE_SPACING,
    f"Tendance sur 20 ans estimée sur {station_count_fmt} stations, mosaïque exclusive affleurante NV1",
    ha="left",
    va="top",
    fontsize=22
)

fig.add_artist(
    Line2D(
        [TITLE_X, TITLE_X + 0.18],
        [TITLE_Y - TITLE_SPACING - SUB_SPACING,
         TITLE_Y - TITLE_SPACING - SUB_SPACING],
        transform=fig.transFigure,
        linewidth=1.2,
        color="#071633"
    )
)

fig.text(
    TITLE_X,
    TITLE_Y - TITLE_SPACING - SUB_SPACING - BAR_SPACING,
    "Réseau 070 - Surveillance de l'état quantitatif des eaux souterraines de la France",
    ha="left",
    va="top",
    fontsize=13
)

fig.text(
    TITLE_X,
    TITLE_Y - TITLE_SPACING - SUB_SPACING - BAR_SPACING - 0.014,
    f"Support exclusif = POLYG_ELEMENTAIRES + TABLE_PILE_ENTITES_NIV1 ordre 1, hachures si < {ROBUST_THRESHOLD} stations",
    ha="left",
    va="top",
    fontsize=13
)

fig.text(
    TITLE_X,
    TITLE_Y - TITLE_SPACING - SUB_SPACING - BAR_SPACING - 0.028,
    "Source: ADES / Hubeau API - BDLISA / Parallaxe processing",
    ha="left",
    va="top",
    fontsize=13
)

gdf_map.plot(
    ax=ax,
    color=gdf_map["color"].fillna(NO_DATA_COLOR),
    edgecolor="none",
    linewidth=0,
    zorder=1
)

fragile = gdf_map[gdf_map["is_fragile"] == True].copy()
if len(fragile) > 0:
    fragile.plot(
        ax=ax,
        facecolor="none",
        edgecolor=HATCH_COLOR,
        linewidth=0,
        hatch="/////",
        zorder=2
    )

ax.axis("off")

legend_patches = [
    mpatches.Patch(color=COLOR_MAP[label], label=label)
    for label in LABELS
]
legend_patches.append(
    mpatches.Patch(
        facecolor="white",
        edgecolor=HATCH_COLOR,
        hatch="/////",
        label=f"Faible support (< {ROBUST_THRESHOLD} stations)"
    )
)
legend_patches.append(
    mpatches.Patch(
        color=NO_DATA_COLOR,
        label="Aucune donnée"
    )
)

fig.legend(
    handles=legend_patches,
    loc="lower right",
    bbox_to_anchor=(0.95, 0.03),
    frameon=False,
    fontsize=12
)

fig.text(
    0.07,
    0.06,
    f"Carte établie le {datetime.now().strftime('%d %B %Y')}",
    ha="left",
    va="bottom",
    fontsize=13
)

fig.text(
    0.07,
    0.045,
    "Parallaxe processing",
    ha="left",
    va="bottom",
    fontsize=13,
    color="#071633"
)

plt.savefig(OUT_PNG, dpi=300, facecolor="white")
plt.close(fig)

print("Saved PNG:", OUT_PNG)




# ============================================================
# HTML LIGHT (DISSOLVED BY CodeEH)
# ============================================================

print("\n=== LOAD LIBELLES NV1 ===")
gdf_nv1 = gpd.read_file(AQUIFER_FILE, layer="entites_niveau1_extension")
gdf_nv1.columns = [c.strip() for c in gdf_nv1.columns]

print(gdf_nv1.columns.tolist())

# on garde seulement la table attributaire utile
df_nv1_labels = (
    gdf_nv1[["codeeh", "libelleeh"]]
    .drop_duplicates()
    .copy()
)

df_nv1_labels["codeeh"] = pd.to_numeric(df_nv1_labels["codeeh"], errors="coerce")

gdf_map = gdf_map.merge(
    df_nv1_labels,
    left_on="CodeEH",
    right_on="codeeh",
    how="left"
).drop(columns=["codeeh"])







print("\n=== EXPORT HTML LIGHT ===")

gdf_html = gdf_map.copy()
gdf_html = gdf_html[gdf_html["CodeEH"].notna()].copy()

gdf_html["geometry"] = gdf_html["geometry"].make_valid()
gdf_html["geometry"] = gdf_html["geometry"].buffer(0)

gdf_html = gdf_html[~gdf_html.geometry.is_empty].copy()
gdf_html = gdf_html[gdf_html.geometry.notna()].copy()

gdf_html = gdf_html.dissolve(
    by="CodeEH",
    aggfunc={
        "libelleeh": "first",
        "median_variation_20y_cm": "first",
        "mean_variation_20y_cm": "first",
        "min_variation_20y_cm": "first",
        "max_variation_20y_cm": "first",
        "p25_variation_20y_cm": "first",
        "p75_variation_20y_cm": "first",
        "iqr_variation_20y_cm": "first",
        "n_stations": "first",
        "median_class": "first",
        "is_fragile": "first",
        "color": "first",
    }
).reset_index()

print("HTML entities:", len(gdf_html))

# ------------------------------------------------------------
# PREPARE FRAGILE GEOMETRIES FOR HATCHING (in WORK_CRS)
# ------------------------------------------------------------

gdf_fragile_work = (
    gdf_html[gdf_html["is_fragile"] == True]
    .copy()
    .to_crs(WORK_CRS)
)

gdf_html_full = gdf_html.copy()
gdf_html_full.to_file(OUT_GEOJSON_FULL, driver="GeoJSON")
print("Saved GeoJSON full:", OUT_GEOJSON_FULL)

gdf_html = gdf_html.to_crs(HTML_CRS).copy()
gdf_html = gdf_html.sort_values("CodeEH").reset_index(drop=True)
gdf_html["obj_id"] = np.arange(1, len(gdf_html) + 1)
gdf_html["obj_label"] = gdf_html["obj_id"].astype(str) + "/" + str(len(gdf_html))
gdf_html["feature_id"] = gdf_html["CodeEH"].astype(str)


class_order = [
    "Forte baisse (≤ -50 cm)",
    "Baisse modérée (-50 à -20 cm)",
    "Quasi stable (-20 à +20 cm)",
    "Hausse modérée (+20 à +50 cm)",
    "Forte hausse (≥ +50 cm)",
    "No data",
]

class_to_num = {cls: i for i, cls in enumerate(class_order)}
gdf_html["class_num"] = gdf_html["median_class"].map(class_to_num)

gdf_html_light = gdf_html.copy()
gdf_html_light["geometry"] = gdf_html_light["geometry"].simplify(
    0.005,
    preserve_topology=True
)

gdf_html_light.to_file(OUT_GEOJSON_LIGHT, driver="GeoJSON")
print("Saved GeoJSON light:", OUT_GEOJSON_LIGHT)

geojson = json.loads(gdf_html_light.to_json())





colorscale = [
    [0/5, COLOR_MAP["Forte baisse (≤ -50 cm)"]],
    [1/5, COLOR_MAP["Baisse modérée (-50 à -20 cm)"]],
    [2/5, COLOR_MAP["Quasi stable (-20 à +20 cm)"]],
    [3/5, COLOR_MAP["Hausse modérée (+20 à +50 cm)"]],
    [4/5, COLOR_MAP["Forte hausse (≥ +50 cm)"]],
    [5/5, COLOR_MAP["No data"]],
]


all_hatch_geoms = []

for geom in gdf_fragile_work.geometry:
    lines = make_hatch_lines_for_polygon(geom, spacing=7000, angle_deg=-45)
    all_hatch_geoms.extend(lines)

gdf_hatch = gpd.GeoDataFrame(
    geometry=all_hatch_geoms,
    crs=WORK_CRS
).to_crs(HTML_CRS)

hatch_lons = []
hatch_lats = []

for geom in gdf_hatch.geometry:

    if geom.geom_type == "LineString":
        xs, ys = geom.xy
        hatch_lons.extend(xs)
        hatch_lats.extend(ys)
        hatch_lons.append(None)
        hatch_lats.append(None)

    elif geom.geom_type == "MultiLineString":
        for part in geom.geoms:
            xs, ys = part.xy
            hatch_lons.extend(xs)
            hatch_lats.extend(ys)
            hatch_lons.append(None)
            hatch_lats.append(None)


fig_html = go.Figure()



fig_html.add_trace(
    go.Choroplethmapbox(
        geojson=geojson,
        locations=gdf_html_light["feature_id"],
        z=gdf_html_light["class_num"],
        featureidkey="properties.feature_id",
        colorscale=colorscale,
        zmin=0,
        zmax=5,
        showscale=False,
        marker_opacity=0.85,
        marker_line_width=0.4,
        marker_line_color="rgba(255,255,255,0.25)",
        customdata=gdf_html_light[
            ["obj_label", "CodeEH", "libelleeh",
             "median_variation_20y_cm", "n_stations",
             "median_class", "is_fragile"]
        ].to_numpy(),
        hoverinfo="none",
        hovertemplate=None,
    )
)

highlight_trace_index = len(fig_html.data)



fig_html.add_trace(
    go.Choroplethmapbox(
        geojson=geojson,
        locations=[],
        z=[],
        featureidkey="properties.feature_id",
        colorscale=[
            [0, "rgba(135,206,250,0.8)"],
            [1, "rgba(135,206,250,0.8)"],
        ],
        showscale=False,
        marker_opacity=1,
        marker_line_width=0,
        marker_line_color="rgba(0,0,0,0)",
        hoverinfo="skip",
        hovertemplate=None,
        showlegend=False,
        name="hover-highlight",
    )
)

fig_html.add_trace(
    go.Scattermapbox(
        lon=hatch_lons,
        lat=hatch_lats,
        mode="lines",
        line=dict(
            width=0.5,
            color="rgba(170,170,170,0.7)"
        ),
        hoverinfo="skip",
        showlegend=False,
        name="fragile-hatch"
    )
)




fig_html.update_layout(
    mapbox_style="carto-positron",
    mapbox_zoom=4.7,
    mapbox_center={"lat": 46.5, "lon": 2.2},
    margin=dict(l=0, r=0, t=0, b=0),
    title=None,
    font=dict(
        family="Roboto Condensed, Arial, sans-serif",
        size=14,
        color="#222"
    )
)


fig_html.write_html(
    OUT_HTML_LIGHT,
    include_plotlyjs=True,
    full_html=False
)

date_str = datetime.now().strftime("%d %B %Y")

panel_html = """



<div id="side-panel">

    <!-- LEGEND -->
    <div>
        <div style="font-weight:600;font-size:18px;margin-bottom:16px;">Légende</div>

        <div><span style="display:inline-block;width:14px;height:14px;background:#c0392b;margin-right:8px;vertical-align:middle;"></span>Forte baisse (≤ -50 cm)</div>
        <div><span style="display:inline-block;width:14px;height:14px;background:#e26f5d;margin-right:8px;vertical-align:middle;"></span>Baisse modérée (-50 à -20 cm)</div>
        <div><span style="display:inline-block;width:14px;height:14px;background:#f4b5b2;margin-right:8px;vertical-align:middle;"></span>Quasi stable (-20 à +20 cm)</div>
        <div><span style="display:inline-block;width:14px;height:14px;background:#4e64a8;margin-right:8px;vertical-align:middle;"></span>Hausse modérée (+20 à +50 cm)</div>
        <div><span style="display:inline-block;width:14px;height:14px;background:#1c2864;margin-right:8px;vertical-align:middle;"></span>Forte hausse (≥ +50 cm)</div>

        <div style="margin-top:8px;">
            <span style="display:inline-block;width:14px;height:14px;background:white;border:1px solid #bbb;background-image:repeating-linear-gradient(-45deg,#bbb,#bbb 1px,transparent 1px,transparent 4px);margin-right:8px;vertical-align:middle;"></span>
            Faible support (&lt; 5 stations)
        </div>

        <div>
            <span style="display:inline-block;width:14px;height:14px;background:#e6e6e6;margin-right:8px;vertical-align:middle;"></span>
            Aucune donnée
        </div>
    </div>

    <!-- SEPARATOR -->
    <div style="border-top:1px solid #111;margin:28px 0;"></div>

    <!-- HOVER DYNAMIC -->
    <div id="info-panel" style="font-size:15px;line-height:1.4;">
        <b>Survoler un aquifère</b><br>
        Les informations détaillées apparaîtront ici.
    </div>

    <!-- FOOTER MENTIONS -->
    <div style="margin-top:auto;font-size:13px;line-height:1.45;color:#333;">
        <div>Réseau 070 - Surveillance de l'état quantitatif des eaux souterraines de la France</div>
        <div style="margin-top:8px;">Support exclusif = POLYG_ELEMENTAIRES + TABLE_PILE_ENTITES_NIV1 ordre 1, hachures si &lt; 5 stations</div>
        <div style="margin-top:8px;">Source: ADES / Hubeau API - BDLISA / Parallaxe processing</div>
        <div style="margin-top:8px;">Carte établie le """ + date_str + """</div>
    </div>

</div>
"""



plot_div = OUT_HTML_LIGHT.read_text(encoding="utf-8")



final_html = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Roboto+Condensed:wght@400;500;600&display=swap" rel="stylesheet">

<style>
html, body {
    margin: 0;
    padding: 0;
    height: 100%;
    font-family: 'Roboto Condensed', Arial, sans-serif;
    background: #f5f5f5;
}

.app {
    height: 100vh;
    display: flex;
    flex-direction: column;
    padding: 10px;
    box-sizing: border-box;
}

.title {
    font-size: 28px;
    margin-bottom: 10px;
}

.content {
    flex: 1;
    display: flex;
    gap: 10px;
    min-height: 0;
}

#side-panel {
    width: 320px;
    background: white;
    padding: 22px 18px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.12);
    border-radius: 4px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
}

.map-container {
    flex: 1;
    background: white;
    padding: 10px;
    border-radius: 4px;
    display: flex;
    min-width: 0;
    min-height: 0;
    overflow: hidden;   /* 👈 IMPORTANT */
}

.map-container > div {
    width: 100% !important;
    height: 100% !important;
}

</style>
</head>

<body>

<div class="app">

    <div class="title">
        Variation des nappes souterraines sur 20 ans (2005-2025)
    </div>

    <div class="content">

        __PANEL_HTML__

        <div class="map-container">
            __PLOT_DIV__
        </div>

    </div>

</div>

<script>
document.addEventListener("DOMContentLoaded", function() {

    const gd = document.querySelector(".js-plotly-plot");
    if (!gd) return;

    const highlightTrace = __HIGHLIGHT__;
    const infoPanel = document.getElementById("info-panel");

    Plotly.Plots.resize(gd);

    window.addEventListener("resize", function() {
        Plotly.Plots.resize(gd);
    });

    gd.on("plotly_hover", function(evt) {
        if (!evt.points || !evt.points.length) return;

        const p = evt.points[0];
        const code = p.location;
        const cd = p.customdata || [];

        Plotly.restyle(
            gd,
            { locations: [[code]], z: [[1]] },
            [highlightTrace]
        );

        if (infoPanel) {
            infoPanel.innerHTML = `
                <b>Aquifère affleurant NV1</b><br>
                Objet: ${cd[0] ?? ""}<br>
                CodeEH: ${cd[1] ?? ""}<br>
                Libellé: ${cd[2] ?? ""}<br>
                Médiane 20 ans: ${cd[3] !== null && cd[3] !== undefined ? cd[3].toFixed(0) : ""} cm<br>
                Nb stations: ${cd[4] ?? ""}<br>
                Classe: ${cd[5] ?? ""}<br>
                Fragile: ${cd[6] ? "oui" : "non"}
            `;
        }
    });

    gd.on("plotly_unhover", function() {

        Plotly.restyle(
            gd,
            { locations: [[]], z: [[]] },
            [highlightTrace]
        );

        if (infoPanel) {
            infoPanel.innerHTML =
                "<b>Survoler un aquifère</b><br>Les informations détaillées apparaîtront ici.";
        }
    });

});
</script>

</body>
</html>
"""


final_html = final_html.replace("__HIGHLIGHT__", str(highlight_trace_index))
final_html = final_html.replace("__PANEL_HTML__", panel_html)
final_html = final_html.replace("__PLOT_DIV__", plot_div)

OUT_HTML_LIGHT.write_text(final_html, encoding="utf-8")

print("Saved HTML light:", OUT_HTML_LIGHT)
print("\n=== DONE ===")

