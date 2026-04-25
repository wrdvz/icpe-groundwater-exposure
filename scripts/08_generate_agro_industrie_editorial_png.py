from __future__ import annotations

from datetime import date

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd

from icpe_groundwater_exposure.config import (
    EXPOSURE_GRID_GEOJSON_FILE,
    PROJECT_ROOT,
    ROBOTO_CONDENSED_FONT,
    WORK_CRS,
)


REGIONS_FILE = PROJECT_ROOT / "data" / "external" / "geo" / "regions_france.geojson"
ICPE_ENRICHED_FILE = PROJECT_ROOT / "data" / "processed" / "icpe" / "icpe_sites_normalized_with_water_taxonomy_7.csv"
OUT_PNG = PROJECT_ROOT / "docs" / "parallaxe_icpe_agro_industrie_groundwater_exposure_2023.png"

METRO_REGION_CODES = {"11", "24", "27", "28", "32", "44", "52", "53", "75", "76", "84", "93", "94"}

COULEURS = {
    "forte_pression_nappe_baisse": "#9b2c2c",
    "faible_pression_nappe_baisse": "#e49a9a",
    "forte_pression_nappe_non_baissiere": "#355f7c",
    "faible_pression_nappe_non_baissiere": "#b9d2df",
    "non_classe": "#e6e6e6",
    "points_icpe": "#1c2864",
    "fond": "#ffffff",
    "texte": "#111111",
    "ligne": "#4f648c",
}

LIBELLES = {
    "forte_pression_nappe_baisse": "Forte pression IND+IRR + nappe en baisse",
    "faible_pression_nappe_baisse": "Faible pression IND+IRR + nappe en baisse",
    "forte_pression_nappe_non_baissiere": "Forte pression IND+IRR + nappe non baissière",
    "faible_pression_nappe_non_baissiere": "Faible pression IND+IRR + nappe non baissière",
    "non_classe": "Aucune donnée nappe",
}


def charger_police() -> FontProperties | None:
    if ROBOTO_CONDENSED_FONT.exists():
        fontManager.addfont(str(ROBOTO_CONDENSED_FONT))
        return FontProperties(fname=str(ROBOTO_CONDENSED_FONT))
    return None


def silhouette_france_metropolitaine() -> gpd.GeoDataFrame:
    regions = gpd.read_file(REGIONS_FILE)
    regions["code"] = regions["code"].astype(str).str.zfill(2)
    metro = regions[regions["code"].isin(METRO_REGION_CODES)].copy().to_crs(WORK_CRS)
    return gpd.GeoDataFrame({"nom": ["France métropolitaine"], "geometry": [metro.union_all()]}, crs=WORK_CRS)


def reclasser_grille_mixte(france: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, float]:
    grid = gpd.read_file(EXPOSURE_GRID_GEOJSON_FILE).to_crs(WORK_CRS)
    grid = gpd.clip(grid, france)

    for col in ["station_count", "groundwater_median_variation_20y_cm", "withdrawal_volume_m3"]:
        grid[col] = pd.to_numeric(grid[col], errors="coerce")

    volumes_positifs = grid.loc[grid["withdrawal_volume_m3"] > 0, "withdrawal_volume_m3"]
    seuil_p75 = float(volumes_positifs.quantile(0.75)) if len(volumes_positifs) else 0.0

    def classer_ligne(row: pd.Series) -> str:
        if pd.isna(row["station_count"]) or row["station_count"] <= 0:
            return "non_classe"
        nappe_en_baisse = pd.notna(row["groundwater_median_variation_20y_cm"]) and row["groundwater_median_variation_20y_cm"] <= -10
        forte_pression = pd.notna(row["withdrawal_volume_m3"]) and row["withdrawal_volume_m3"] > seuil_p75
        if forte_pression and nappe_en_baisse:
            return "forte_pression_nappe_baisse"
        if nappe_en_baisse:
            return "faible_pression_nappe_baisse"
        if forte_pression:
            return "forte_pression_nappe_non_baissiere"
        return "faible_pression_nappe_non_baissiere"

    grid["classe_mixte_png"] = grid.apply(classer_ligne, axis=1)
    return grid, seuil_p75


def charger_points_agro_industrie(france: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, int]:
    points = pd.read_csv(ICPE_ENRICHED_FILE, low_memory=False)
    points = points[points["categorie_eau_7"] == "Agro-industrie"].copy()
    points["x"] = pd.to_numeric(points["x"], errors="coerce")
    points["y"] = pd.to_numeric(points["y"], errors="coerce")
    points = points.dropna(subset=["x", "y"]).copy()
    n_sites = len(points)

    points_agg = (
        points.groupby(["x", "y"], as_index=False)
        .agg(
            n_sites=("code_aiot", "nunique"),
        )
    )
    points_agg["taille_point"] = points_agg["n_sites"].clip(upper=40).pow(0.7) * 4.2

    gdf = gpd.GeoDataFrame(points_agg, geometry=gpd.points_from_xy(points_agg["x"], points_agg["y"]), crs=WORK_CRS)
    return gpd.clip(gdf, france), n_sites


def formatter_nombre(n: float | int) -> str:
    return f"{int(round(n)):,}".replace(",", " ")


def ajouter_bloc_texte(fig: plt.Figure, police: FontProperties | None, n_sites: int, n_points_agreges: int, seuil_p75: float) -> None:
    prop = {"fontproperties": police} if police else {}
    fig.text(
        0.07,
        0.93,
        "Exposition des sites agro-industriels aux tensions sur les eaux souterraines",
        fontsize=28,
        ha="left",
        va="top",
        color=COULEURS["texte"],
        **prop,
    )
    fig.text(
        0.07,
        0.895,
        "Grille 20 km croisant les volumes de prélèvements IND+IRR 2023 et la tendance des nappes sur 20 ans (2005-2025)",
        fontsize=18,
        ha="left",
        va="top",
        color=COULEURS["texte"],
        **prop,
    )
    fig.add_artist(
        Line2D([0.07, 0.25], [0.84, 0.84], transform=fig.transFigure, color=COULEURS["ligne"], linewidth=1.6)
    )
    notes = (
        "Points = sites ICPE classés en Agro-industrie via la NAF\n"
        "Forte pression = volume IND+IRR par maille > P75 des mailles non nulles ; baisse = variation médiane des nappes ≤ -10 cm\n"
        f"France métropolitaine ; {formatter_nombre(n_sites)} sites, ramenés à {formatter_nombre(n_points_agreges)} positions ; seuil P75 IND+IRR = {formatter_nombre(seuil_p75)} m3"
    )
    fig.text(
        0.07, 0.82, notes, fontsize=12.5, ha="left", va="top", linespacing=1.35, color=COULEURS["texte"], **prop
    )


def ajouter_signature(fig: plt.Figure, police: FontProperties | None) -> None:
    prop = {"fontproperties": police} if police else {}
    aujourd_hui = date.today().strftime("%d/%m/%Y")
    fig.text(0.07, 0.07, f"Carte établie le {aujourd_hui}\nParallaxe processing", fontsize=12.5, ha="left", va="bottom", color=COULEURS["texte"], **prop)


def ajouter_legende(fig: plt.Figure, police: FontProperties | None) -> None:
    handles = [
        Patch(facecolor=COULEURS["forte_pression_nappe_baisse"], edgecolor="none", label=LIBELLES["forte_pression_nappe_baisse"]),
        Patch(facecolor=COULEURS["faible_pression_nappe_baisse"], edgecolor="none", label=LIBELLES["faible_pression_nappe_baisse"]),
        Patch(facecolor=COULEURS["faible_pression_nappe_non_baissiere"], edgecolor="none", label=LIBELLES["faible_pression_nappe_non_baissiere"]),
        Patch(facecolor=COULEURS["forte_pression_nappe_non_baissiere"], edgecolor="none", label=LIBELLES["forte_pression_nappe_non_baissiere"]),
        Patch(facecolor=COULEURS["non_classe"], edgecolor="none", label=LIBELLES["non_classe"]),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COULEURS["points_icpe"], markeredgecolor=COULEURS["points_icpe"], markersize=7, label="Sites Agro-industrie"),
    ]
    leg = fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.93, 0.08), frameon=False, fontsize=12, handlelength=2.0, labelspacing=0.65)
    if police:
        for text in leg.get_texts():
            text.set_fontproperties(police)


def dessiner_carte(ax: plt.Axes, france: gpd.GeoDataFrame, grille: gpd.GeoDataFrame, points: gpd.GeoDataFrame) -> None:
    ax.set_facecolor(COULEURS["fond"])
    france.plot(ax=ax, color=COULEURS["non_classe"], edgecolor="none", zorder=0)
    ordre = [
        "non_classe",
        "faible_pression_nappe_non_baissiere",
        "forte_pression_nappe_non_baissiere",
        "faible_pression_nappe_baisse",
        "forte_pression_nappe_baisse",
    ]
    for classe in ordre:
        sous = grille[grille["classe_mixte_png"] == classe]
        if len(sous):
            sous.plot(ax=ax, color=COULEURS[classe], edgecolor="white", linewidth=0.15, zorder=1)
    france.boundary.plot(ax=ax, color="#ffffff", linewidth=0.8, zorder=2)
    if len(points):
        ax.scatter(points.geometry.x, points.geometry.y, s=points["taille_point"], c=COULEURS["points_icpe"], alpha=0.6, linewidths=0, zorder=3)
    minx, miny, maxx, maxy = france.total_bounds
    pad_x = (maxx - minx) * 0.06
    pad_y = (maxy - miny) * 0.04
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    police = charger_police()
    france = silhouette_france_metropolitaine()
    grille, seuil_p75 = reclasser_grille_mixte(france)
    points, n_sites = charger_points_agro_industrie(france)

    fig = plt.figure(figsize=(14, 18), facecolor=COULEURS["fond"])
    ax = fig.add_axes([0.07, 0.14, 0.86, 0.63])
    dessiner_carte(ax, france, grille, points)
    ajouter_bloc_texte(fig, police, n_sites, len(points), seuil_p75)
    ajouter_signature(fig, police)
    ajouter_legende(fig, police)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=220, facecolor=COULEURS["fond"], bbox_inches="tight")
    print(f"PNG écrit: {OUT_PNG}")
    print(f"Seuil P75 IND+IRR (m3): {seuil_p75:,.0f}")
    print(f"Sites Agro-industrie affichés: {n_sites:,}")
    print(f"Positions agrégées affichées: {len(points):,}")


if __name__ == "__main__":
    main()
