"""Génère la carte HTML d'exposition ICPE / nappes / prélèvements en Lambert-93."""

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

from icpe_groundwater_exposure.config import (
    EXPOSURE_GRID_GEOJSON_FILE,
    EXPOSURE_MAP_HTML_FILE,
    ICPE_BNPE_ECONOMIC_BEST_MATCH_FILE,
    ICPE_GROUNDWATER_CONTEXT_FILE,
    NORMALIZED_BNPE_WITHDRAWALS_FILE,
    TRENDS_FILE,
)
from icpe_groundwater_exposure.utils import ensure_dirs

USAGES_PRESSION = ["AEP", "IND", "IRR"]
ICPE_ENRICHED_FILE = (
    Path(EXPOSURE_GRID_GEOJSON_FILE).parents[1] / "icpe" / "icpe_sites_normalized_with_water_taxonomy_7.csv"
)

COULEURS_FOND = {
    "high_pressure_declining_groundwater": "#9b2c2c",
    "low_pressure_declining_groundwater": "#e49a9a",
    "high_pressure_non_declining_groundwater": "#355f7c",
    "low_pressure_non_declining_groundwater": "#b9d2df",
    "unclassified_no_groundwater_data": "#e5e7eb",
}

LIBELLES_FOND = {
    "high_pressure_declining_groundwater": "Forte pression + nappe en baisse",
    "low_pressure_declining_groundwater": "Faible pression + nappe en baisse",
    "high_pressure_non_declining_groundwater": "Forte pression + nappe non baissière",
    "low_pressure_non_declining_groundwater": "Faible pression + nappe non baissière",
    "unclassified_no_groundwater_data": "Non classé (pas de donnée nappe)",
}

COULEURS_POINTS = {
    "PRELEVEMENT": "#2563eb",
    "BSS": "#7c3aed",
    "GRIS": "#9ca3af",
}

COULEURS_SECTEURS = {
    "Agriculture et élevage": "#1d4ed8",
    "Agro-industrie": "#2563eb",
    "Santé, chimie, produits de synthèse": "#7c3aed",
    "Métallurgie, mécanique, automobile": "#b45309",
    "Papier, bois, textile, cuir": "#059669",
    "Extraction, carrières, eau, déchets, énergie": "#dc2626",
    "Construction et génie civil": "#475569",
}


def _fmt_int(value) -> str:
    if pd.isna(value):
        return "n.d."
    return f"{int(value):,}".replace(",", " ")


def _fmt_float(value, digits=1, suffix="") -> str:
    if pd.isna(value):
        return "n.d."
    return f"{value:.{digits}f}{suffix}"


def popup_grille(row) -> str:
    robuste = "oui" if bool(row.get("groundwater_signal_robust")) else "non"
    return (
        f"<strong>{LIBELLES_FOND.get(row.get('exposure_class_2x2'), row.get('exposure_class_2x2'))}</strong><br>"
        f"Volume AEP+IND+IRR : {_fmt_int(row.get('withdrawal_pressure_volume_m3'))} m3<br>"
        f"Nombre de prélèvements AEP+IND+IRR : {_fmt_int(row.get('withdrawal_pressure_count'))}<br>"
        f"Volume AEP : {_fmt_int(row.get('aep_withdrawal_volume_m3'))} m3<br>"
        f"Volume IND : {_fmt_int(row.get('ind_withdrawal_volume_m3'))} m3<br>"
        f"Volume IRR : {_fmt_int(row.get('irr_withdrawal_volume_m3'))} m3<br>"
        f"Tendance médiane des nappes : {_fmt_float(row.get('groundwater_median_variation_20y_cm'), 1, ' cm')}<br>"
        f"Nombre de stations : {_fmt_int(row.get('station_count'))}<br>"
        f"Signal robuste (>= 5 stations) : {robuste}"
    )


def resume_grille(row) -> str:
    robuste = "oui" if bool(row.get("groundwater_signal_robust")) else "non"
    return (
        f"<strong>{LIBELLES_FOND.get(row.get('exposure_class_2x2'), row.get('exposure_class_2x2'))}</strong><br>"
        f"Volume AEP+IND+IRR : {_fmt_int(row.get('withdrawal_pressure_volume_m3'))} m3<br>"
        f"Prélèvements AEP+IND+IRR : {_fmt_int(row.get('withdrawal_pressure_count'))}<br>"
        f"Tendance médiane des nappes : {_fmt_float(row.get('groundwater_median_variation_20y_cm'), 1, ' cm')}<br>"
        f"Stations : {_fmt_int(row.get('station_count'))}<br>"
        f"Signal robuste : {robuste}"
    )


def popup_icpe(row) -> str:
    robuste = "oui" if bool(row.get("is_signal_solid")) else "non"
    return (
        f"<strong>{row.get('nom_ets', 'Site sans nom')}</strong><br>"
        f"Catégorie eau : {row.get('categorie_eau_7', 'n.d.')}<br>"
        f"Secteur ICPE d'origine : {row.get('site_sector', 'n.d.')}<br>"
        f"NAF : {row.get('code_naf', 'n.d.')} - {row.get('lib_naf', 'n.d.')}<br>"
        f"Commune : {row.get('commune', 'n.d.')}<br>"
        f"Tendance médiane des nappes (20 km) : {_fmt_float(row.get('median_variation_20y_cm_20km'), 1, ' cm')}<br>"
        f"Stations nappes (20 km) : {_fmt_int(row.get('n_stations_20km'))}<br>"
        f"Signal nappe robuste : {robuste}"
    )


def resume_icpe(row) -> str:
    robuste = "oui" if bool(row.get("is_signal_solid")) else "non"
    classe_grille = LIBELLES_FOND.get(row.get("exposure_class_2x2"), "n.d.")
    return (
        f"<strong>{row.get('nom_ets', 'Site sans nom')}</strong><br>"
        f"{row.get('categorie_eau_7', 'n.d.')}<br>"
        f"Commune : {row.get('commune', 'n.d.')}<br>"
        f"Classe de grille : {classe_grille}<br>"
        f"NAF : {row.get('code_naf', 'n.d.')} - {row.get('lib_naf', 'n.d.')}<br>"
        f"Tendance nappe (20 km) : {_fmt_float(row.get('median_variation_20y_cm_20km'), 1, ' cm')}<br>"
        f"Stations (20 km) : {_fmt_int(row.get('n_stations_20km'))}<br>"
        f"Signal robuste : {robuste}"
    )


def popup_prelevement(row) -> str:
    return (
        f"<strong>{row.get('ouvrage_name', 'Ouvrage BNPE')}</strong><br>"
        f"Usage : {row.get('usage_code', 'n.d.')} - {row.get('usage_label', 'n.d.')}<br>"
        f"Volume : {_fmt_int(row.get('volume_m3'))} m3<br>"
        f"Commune : {row.get('commune', 'n.d.')}<br>"
        f"Code Sandre ouvrage : {row.get('ouvrage_sandre', 'n.d.')}<br>"
        f"Code BSS : {row.get('code_bss', 'n.d.')}"
    )


def popup_bss(row) -> str:
    return (
        f"<strong>Point BSS {row.get('code_bss', 'n.d.')}</strong><br>"
        f"Tendance 2005-2025 : {_fmt_float(row.get('variation_20y_cm'), 1, ' cm')}<br>"
        f"Nombre d'années : {_fmt_int(row.get('n_years'))}<br>"
        f"Département : {row.get('departement_name', 'n.d.')}<br>"
        f"Région : {row.get('region', 'n.d.')}"
    )


def _classify_groundwater_signal(variation_cm) -> str:
    if pd.isna(variation_cm):
        return "no_data"
    if variation_cm <= -10:
        return "declining"
    return "not_declining"


def _classify_pressure_signal(volume_m3, threshold_m3) -> str:
    if pd.isna(volume_m3) or volume_m3 <= 0:
        return "low"
    if volume_m3 > threshold_m3:
        return "high"
    return "low"


def _classify_exposure_2x2(groundwater_signal: str, pressure_signal: str) -> str:
    if groundwater_signal == "no_data":
        return "unclassified_no_groundwater_data"
    if groundwater_signal == "declining" and pressure_signal == "high":
        return "high_pressure_declining_groundwater"
    if groundwater_signal == "declining" and pressure_signal == "low":
        return "low_pressure_declining_groundwater"
    if groundwater_signal == "not_declining" and pressure_signal == "high":
        return "high_pressure_non_declining_groundwater"
    return "low_pressure_non_declining_groundwater"


def charger_grille():
    grille = gpd.read_file(EXPOSURE_GRID_GEOJSON_FILE).to_crs("EPSG:2154")
    bnpe = pd.read_csv(NORMALIZED_BNPE_WITHDRAWALS_FILE, low_memory=False)
    bnpe = bnpe[bnpe["usage_code"].isin(USAGES_PRESSION)].copy()
    bnpe["longitude"] = pd.to_numeric(bnpe["longitude"], errors="coerce")
    bnpe["latitude"] = pd.to_numeric(bnpe["latitude"], errors="coerce")
    bnpe["volume_m3"] = pd.to_numeric(bnpe["volume_m3"], errors="coerce").fillna(0)
    bnpe = bnpe.dropna(subset=["longitude", "latitude"]).copy()

    bnpe_gdf = gpd.GeoDataFrame(
        bnpe,
        geometry=gpd.points_from_xy(bnpe["longitude"], bnpe["latitude"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:2154")

    joined = gpd.sjoin(bnpe_gdf, grille[["grid_id", "geometry"]], how="inner", predicate="within")
    agg = (
        joined.groupby("grid_id", as_index=False)
        .agg(
            withdrawal_pressure_count=("ouvrage_sandre", "nunique"),
            withdrawal_pressure_volume_m3=("volume_m3", "sum"),
            aep_withdrawal_count=("usage_code", lambda s: int((s == "AEP").sum())),
            ind_withdrawal_count=("usage_code", lambda s: int((s == "IND").sum())),
            irr_withdrawal_count=("usage_code", lambda s: int((s == "IRR").sum())),
            aep_withdrawal_volume_m3=("volume_m3", lambda s: float(s[joined.loc[s.index, "usage_code"] == "AEP"].sum())),
            ind_withdrawal_volume_m3=("volume_m3", lambda s: float(s[joined.loc[s.index, "usage_code"] == "IND"].sum())),
            irr_withdrawal_volume_m3=("volume_m3", lambda s: float(s[joined.loc[s.index, "usage_code"] == "IRR"].sum())),
        )
    )

    grille = grille.drop(
        columns=[
            "withdrawal_count",
            "withdrawal_volume_m3",
            "ind_withdrawal_count",
            "irr_withdrawal_count",
            "ind_withdrawal_volume_m3",
            "irr_withdrawal_volume_m3",
            "pressure_signal_class",
            "exposure_class_2x2",
            "pressure_threshold_m3",
            "has_withdrawal_data",
            "context_only_withdrawal",
        ],
        errors="ignore",
    )
    grille = grille.merge(agg, on="grid_id", how="left")

    fill_zero_cols = [
        "withdrawal_pressure_count",
        "withdrawal_pressure_volume_m3",
        "aep_withdrawal_count",
        "ind_withdrawal_count",
        "irr_withdrawal_count",
        "aep_withdrawal_volume_m3",
        "ind_withdrawal_volume_m3",
        "irr_withdrawal_volume_m3",
    ]
    for col in fill_zero_cols:
        grille[col] = pd.to_numeric(grille[col], errors="coerce").fillna(0)

    positive_volumes = grille.loc[grille["withdrawal_pressure_volume_m3"] > 0, "withdrawal_pressure_volume_m3"]
    pressure_threshold_m3 = float(positive_volumes.quantile(0.75)) if len(positive_volumes) else 0.0
    grille["groundwater_signal_class"] = grille["groundwater_median_variation_20y_cm"].apply(_classify_groundwater_signal)
    grille["pressure_signal_class"] = grille["withdrawal_pressure_volume_m3"].apply(
        lambda v: _classify_pressure_signal(v, pressure_threshold_m3)
    )
    grille["exposure_class_2x2"] = grille.apply(
        lambda row: _classify_exposure_2x2(row["groundwater_signal_class"], row["pressure_signal_class"]),
        axis=1,
    )
    grille["pressure_threshold_m3"] = pressure_threshold_m3
    grille["has_withdrawal_data"] = grille["withdrawal_pressure_count"] > 0
    grille["context_only_withdrawal"] = grille["has_withdrawal_data"] & ~grille["has_groundwater_data"]
    grille["popup_html"] = grille.apply(popup_grille, axis=1)
    grille["hover_html"] = grille.apply(resume_grille, axis=1)
    grille = grille.to_crs("EPSG:4326")
    return json.loads(grille.to_json())


def charger_icpe():
    points = pd.read_csv(ICPE_ENRICHED_FILE, dtype={"code_aiot": "string"}, low_memory=False)
    points = points[points["categorie_eau_7"].notna()].copy()
    points["num_siret"] = points["num_siret"].astype("string").str.replace(r"\D+", "", regex=True)
    points["x"] = pd.to_numeric(points["x"], errors="coerce")
    points["y"] = pd.to_numeric(points["y"], errors="coerce")
    points = points.dropna(subset=["x", "y"]).copy()
    gdf = gpd.GeoDataFrame(points, geometry=gpd.points_from_xy(points["x"], points["y"]), crs="EPSG:2154")
    grille = gpd.read_file(EXPOSURE_GRID_GEOJSON_FILE).to_crs("EPSG:2154")[["grid_id", "exposure_class_2x2", "geometry"]]
    gdf = gpd.sjoin(gdf, grille, how="left", predicate="within").drop(columns=["index_right"], errors="ignore")
    gdf = gdf.to_crs("EPSG:4326")
    gdf["lon"] = gdf.geometry.x
    gdf["lat"] = gdf.geometry.y
    gdf["popup_html"] = gdf.apply(popup_icpe, axis=1)
    gdf["hover_html"] = gdf.apply(resume_icpe, axis=1)
    return [
        {
            "lat": row["lat"],
            "lon": row["lon"],
            "categorie": row["categorie_eau_7"],
            "couleur": COULEURS_SECTEURS.get(row["categorie_eau_7"], COULEURS_POINTS["GRIS"]),
            "siret": None if pd.isna(row.get("num_siret")) else str(row.get("num_siret")),
            "nom_ets": row.get("nom_ets"),
            "commune": row.get("commune"),
            "site_sector": row.get("site_sector"),
            "code_naf": row.get("code_naf"),
            "lib_naf": row.get("lib_naf"),
            "grid_class": LIBELLES_FOND.get(row.get("exposure_class_2x2"), "n.d."),
            "popup_html": row["popup_html"],
            "hover_html": row["hover_html"],
        }
        for _, row in gdf.iterrows()
    ]


def charger_prelevements():
    df = pd.read_csv(NORMALIZED_BNPE_WITHDRAWALS_FILE, low_memory=False)
    df = df[df["usage_code"].isin(USAGES_PRESSION)].copy()
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df = df.dropna(subset=["longitude", "latitude"]).copy()
    df["popup_html"] = df.apply(popup_prelevement, axis=1)
    return [
        {
            "lat": row["latitude"],
            "lon": row["longitude"],
            "forme": "cercle",
            "couleur": COULEURS_POINTS["PRELEVEMENT"],
            "popup_html": row["popup_html"],
        }
        for _, row in df.iterrows()
    ]


def charger_bss():
    df = pd.read_csv(TRENDS_FILE, dtype={"code_bss": "string"}, low_memory=False)
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["variation_20y_cm"] = pd.to_numeric(df["slope_m_per_year"], errors="coerce") * 20 * 100
    df = df.dropna(subset=["x", "y", "variation_20y_cm"]).copy()
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["x"], df["y"]), crs="EPSG:4326")
    gdf["lon"] = gdf.geometry.x
    gdf["lat"] = gdf.geometry.y
    gdf["popup_html"] = gdf.apply(popup_bss, axis=1)
    return [
        {
            "lat": row["lat"],
            "lon": row["lon"],
            "forme": "losange",
            "couleur": COULEURS_POINTS["BSS"],
            "popup_html": row["popup_html"],
        }
        for _, row in gdf.iterrows()
    ]


def construire_html(grille_geojson, points_icpe) -> str:
    points_icpe_par_categorie = {}
    for point in points_icpe:
        points_icpe_par_categorie.setdefault(point["categorie"], []).append(point)

    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Exposition des sites ICPE aux nappes souterraines</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    html, body {{ height: 100%; margin: 0; font-family: Inter, system-ui, sans-serif; color: #16202a; }}
    #map {{ position: absolute; inset: 0; background: #f5f5f4; }}
    .panel {{
      position: absolute; top: 16px; left: 16px; bottom: 16px; z-index: 1000; width: 390px;
      background: rgba(255,255,255,0.95); border: 1px solid #d6dde5; border-radius: 8px;
      padding: 14px 16px; box-shadow: 0 8px 24px rgba(0,0,0,0.08); overflow: auto;
    }}
    .panel h1 {{ margin: 0 0 10px; font-size: 20px; line-height: 1.1; }}
    .panel p {{ margin: 0 0 10px; font-size: 13px; line-height: 1.45; color: #455468; }}
    .legend {{ margin-top: 12px; display: grid; gap: 6px; font-size: 12px; }}
    .legend-row {{ display: flex; align-items: center; gap: 8px; }}
    .swatch {{ width: 14px; height: 14px; border-radius: 2px; border: 1px solid rgba(0,0,0,0.12); flex: 0 0 auto; }}
    .section-title {{ font-size: 14px; font-weight: 700; margin-top: 8px; margin-bottom: 10px; }}
    .divider {{ height: 1px; background: #0f172a; opacity: 0.18; margin: 18px 0; }}
    .hover-box {{ font-size: 13px; line-height: 1.45; color: #334155; min-height: 120px; }}
    .portfolio-box {{ display: grid; gap: 8px; }}
    .portfolio-actions {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    .button-link, .file-label {{
      display: inline-flex; align-items: center; justify-content: center;
      min-height: 34px; padding: 0 12px; border-radius: 6px; text-decoration: none;
      border: 1px solid #cbd5e1; background: #f8fafc; color: #0f172a; font-size: 12px; font-weight: 600;
      cursor: pointer;
    }}
    .button-link:hover, .file-label:hover {{ background: #eef2f7; }}
    .file-input {{ display: none; }}
    .portfolio-status {{ font-size: 12px; line-height: 1.45; color: #475569; }}
    .sources {{ margin-top: 24px; font-size: 12px; line-height: 1.55; color: #334155; }}
    .sources strong {{ display: block; margin-top: 8px; }}
    .leaflet-popup-content {{ margin: 10px 12px; line-height: 1.35; }}
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="panel">
    <h1>Exposition des sites ICPE aux nappes souterraines</h1>
    <p>Grille de 20 km combinant la tendance des nappes et les volumes de prélèvements BNPE pour les usages eau potable, industriels et d'irrigation. Seules les cellules avec observations piézométriques sont classées.</p>
    <p>Le fond représente une pression territoriale plus systémique sur la ressource. La carte charge un seul secteur ICPE au démarrage ; les autres secteurs se chargent à la demande via le sélecteur de couches.</p>
    <div class="section-title">Légende</div>
    <div class="legend">
      <div class="legend-row"><span class="swatch" style="background:{COULEURS_FOND['high_pressure_declining_groundwater']}"></span>Forte pression + nappe en baisse</div>
      <div class="legend-row"><span class="swatch" style="background:{COULEURS_FOND['low_pressure_declining_groundwater']}"></span>Faible pression + nappe en baisse</div>
      <div class="legend-row"><span class="swatch" style="background:{COULEURS_FOND['high_pressure_non_declining_groundwater']}"></span>Forte pression + nappe non baissière</div>
      <div class="legend-row"><span class="swatch" style="background:{COULEURS_FOND['low_pressure_non_declining_groundwater']}"></span>Faible pression + nappe non baissière</div>
      <div class="legend-row"><span class="swatch" style="background:{COULEURS_FOND['unclassified_no_groundwater_data']}"></span>Non classé (pas de donnée nappe)</div>
      <div class="legend-row"><span class="swatch" style="background:{COULEURS_SECTEURS['Agriculture et élevage']}"></span>Agriculture et élevage</div>
      <div class="legend-row"><span class="swatch" style="background:{COULEURS_SECTEURS['Agro-industrie']}"></span>Agro-industrie</div>
      <div class="legend-row"><span class="swatch" style="background:{COULEURS_SECTEURS['Santé, chimie, produits de synthèse']}"></span>Santé, chimie, produits de synthèse</div>
      <div class="legend-row"><span class="swatch" style="background:{COULEURS_SECTEURS['Métallurgie, mécanique, automobile']}"></span>Métallurgie, mécanique, automobile</div>
      <div class="legend-row"><span class="swatch" style="background:{COULEURS_SECTEURS['Papier, bois, textile, cuir']}"></span>Papier, bois, textile, cuir</div>
      <div class="legend-row"><span class="swatch" style="background:{COULEURS_SECTEURS['Extraction, carrières, eau, déchets, énergie']}"></span>Extraction, carrières, eau, déchets, énergie</div>
      <div class="legend-row"><span class="swatch" style="background:{COULEURS_SECTEURS['Construction et génie civil']}"></span>Construction et génie civil</div>
    </div>
    <div class="divider"></div>
    <div class="section-title">Portefeuille</div>
    <div class="portfolio-box">
      <p>Charge un fichier portefeuille simple avec une seule colonne obligatoire : <strong>siret</strong>. La carte fait un rapprochement exact avec les sites ICPE géolocalisés et déduit le reste quand il y a match.</p>
      <div class="portfolio-actions">
        <a class="button-link" href="./portfolio_template.xlsx" download>Télécharger le modèle XLSX</a>
        <label class="file-label" for="portfolio-upload">Charger un portefeuille</label>
        <input id="portfolio-upload" class="file-input" type="file" accept=".xlsx,.xls,.csv">
      </div>
      <div id="portfolio-status" class="portfolio-status">Aucun portefeuille chargé.</div>
    </div>
    <div class="divider"></div>
    <div class="section-title">Survoler une maille ou un site</div>
    <div id="hover-box" class="hover-box">Les informations détaillées apparaîtront ici.</div>
    <div class="sources">
      <strong>Sources</strong>
      BNPE 2023 - eaux souterraines (AEP, IND, IRR)<br>
      Réseau 070 - surveillance de l'état quantitatif des eaux souterraines<br>
      ICPE + NAF retraitées pour l'analyse sectorielle<br>
      Traitements : Parallaxe processing
    </div>
  </div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/xlsx@0.18.5/dist/xlsx.full.min.js"></script>
  <script>
    const grilleGeojson = {json.dumps(grille_geojson)};
    const pointsICPEParCategorie = {json.dumps(points_icpe_par_categorie)};
    const couleursFond = {json.dumps(COULEURS_FOND)};
    const hoverDefaultHtml = 'Les informations détaillées apparaîtront ici.';

    const map = L.map('map', {{
      center: [46.6, 2.2],
      zoom: 5.7,
      minZoom: 4,
      zoomSnap: 0.5
    }});

    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
      subdomains: 'abcd',
      tileSize: 256,
      attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
      maxZoom: 20
    }}).addTo(map);

    const paneGrille = map.createPane('pane-grille');
    paneGrille.style.zIndex = 350;

    const panePoints = map.createPane('pane-points');
    panePoints.style.zIndex = 520;

    const panePortfolio = map.createPane('pane-portfolio');
    panePortfolio.style.zIndex = 620;
    const lookupApiUrl = 'https://icpe-groundwater-lookup.edward-vizard.workers.dev/lookup-siret';

    function setHoverBox(html) {{
      document.getElementById('hover-box').innerHTML = html;
    }}

    function resetHoverBox() {{
      setHoverBox(hoverDefaultHtml);
    }}

    function styleGrille(feature) {{
      const cls = feature.properties.exposure_class_2x2;
      return {{
        color: 'transparent',
        weight: 0,
        fillColor: couleursFond[cls] || '#cccccc',
        fillOpacity: cls === 'unclassified_no_groundwater_data' ? 0.22 : 0.52
      }};
    }}

    const coucheGrille = L.geoJSON(grilleGeojson, {{
      pane: 'pane-grille',
      style: styleGrille,
      onEachFeature: function(feature, layer) {{
        layer.on('mouseover', function() {{
          setHoverBox(feature.properties.hover_html || feature.properties.popup_html || hoverDefaultHtml);
        }});
        layer.on('mouseout', function() {{
          resetHoverBox();
        }});
      }}
    }}).addTo(map);

    const renderer = L.canvas({{ padding: 0.5 }});
    const portfolioRenderer = L.canvas({{ padding: 0.5 }});

    function pointRadiusForZoom(zoom) {{
      if (zoom >= 11) return 6.2;
      if (zoom >= 10) return 5.0;
      if (zoom >= 9) return 4.0;
      if (zoom >= 8) return 3.2;
      if (zoom >= 7) return 2.4;
      if (zoom >= 6) return 1.7;
      return 1.1;
    }}

    function currentPointRadius() {{
      return pointRadiusForZoom(map.getZoom());
    }}

    function ajoutePointsICPE(records, couleur) {{
      const layer = L.layerGroup();
      records.forEach((row) => {{
        const marker = L.circleMarker([row.lat, row.lon], {{
          pane: 'pane-points',
          renderer: renderer,
          radius: currentPointRadius(),
          stroke: false,
          fillColor: couleur,
          fillOpacity: 0.7
        }});
        marker.on('mouseover', function() {{
          setHoverBox(row.hover_html || row.popup_html || hoverDefaultHtml);
        }});
        marker.on('mouseout', function() {{
          resetHoverBox();
        }});
        layer.addLayer(marker);
      }});
      layer._loaded = true;
      return layer;
    }}

    const couchesICPE = {{}};
    Object.entries(pointsICPEParCategorie).forEach(([categorie, records]) => {{
      couchesICPE[categorie] = L.layerGroup();
    }});
    const categorieParDefaut = 'Agriculture et élevage';

    const overlays = {{
      'Grille d\\'exposition': coucheGrille,
      ...couchesICPE
    }};

    L.control.layers(null, overlays, {{ collapsed: false }}).addTo(map);

    function chargeCategorieSiBesoin(categorie) {{
      const couche = couchesICPE[categorie];
      if (!couche || couche._loaded) return;
      const records = pointsICPEParCategorie[categorie] || [];
      if (!records.length) {{
        couche._loaded = true;
        return;
      }}
      const built = ajoutePointsICPE(records, records[0].couleur);
      built.eachLayer((layer) => couche.addLayer(layer));
    }}

    chargeCategorieSiBesoin(categorieParDefaut);
    couchesICPE[categorieParDefaut].addTo(map);

    map.on('overlayadd', function(e) {{
      Object.entries(couchesICPE).forEach(([categorie, couche]) => {{
        if (e.layer === couche) {{
          chargeCategorieSiBesoin(categorie);
        }}
      }});
    }});

    function updatePointRadii() {{
      const radius = currentPointRadius();
      Object.values(couchesICPE).forEach((couche) => {{
        couche.eachLayer((layer) => {{
          if (layer.setRadius) layer.setRadius(radius);
        }});
      }});
    }}

    map.on('zoomend', updatePointRadii);

    const portfolioLayer = L.layerGroup().addTo(map);
    const allICPEPoints = Object.values(pointsICPEParCategorie).flat();
    const icpeBySiret = new Map();
    allICPEPoints.forEach((row) => {{
      const siret = (row.siret || '').replace(/\\D+/g, '');
      if (!siret) return;
      if (!icpeBySiret.has(siret)) icpeBySiret.set(siret, []);
      icpeBySiret.get(siret).push(row);
    }});

    function portfolioRadiusForZoom(zoom) {{
      if (zoom >= 11) return 8.2;
      if (zoom >= 10) return 7.0;
      if (zoom >= 9) return 6.0;
      if (zoom >= 8) return 5.0;
      if (zoom >= 7) return 4.0;
      if (zoom >= 6) return 3.0;
      return 2.2;
    }}

    function updatePortfolioRadii() {{
      const radius = portfolioRadiusForZoom(map.getZoom());
      portfolioLayer.eachLayer((layer) => {{
        if (layer.setRadius) layer.setRadius(radius);
      }});
    }}

    map.on('zoomend', updatePortfolioRadii);

    function escapeHtml(value) {{
      return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }}

    function formatGridClass(value) {{
      const labels = {{
        high_pressure_declining_groundwater: 'Forte pression + nappe en baisse',
        low_pressure_declining_groundwater: 'Faible pression + nappe en baisse',
        high_pressure_non_declining_groundwater: 'Forte pression + nappe non baissière',
        low_pressure_non_declining_groundwater: 'Faible pression + nappe non baissière',
        unclassified_no_groundwater_data: 'Non classé (pas de donnée nappe)'
      }};
      return labels[value] || value || 'n.d.';
    }}

    function formatGeoScore(value) {{
      if (value === null || value === undefined || value === '') return 'n.d.';
      const num = Number(value);
      if (!Number.isFinite(num)) return 'n.d.';
      return num.toFixed(2);
    }}

    function buildPortfolioHover(row, portfolioRow, localIcpeMatch) {{
      const portfolioName = portfolioRow.company_name || portfolioRow.nom_societe || portfolioRow.portfolio_name || '';
      const siteName = localIcpeMatch?.nom_ets || row.denomination || 'Site sans nom';
      const waterCategory = localIcpeMatch?.categorie || row.icpe_category || 'n.d.';
      const commune = localIcpeMatch?.commune || 'n.d.';
      const gridClass = localIcpeMatch?.grid_class || formatGridClass(row.grid_class);
      return `
        <strong>${{escapeHtml(siteName)}}</strong><br>
        ${{portfolioName ? `Portefeuille : ${{escapeHtml(portfolioName)}}<br>` : ''}}
        SIRET : ${{escapeHtml(portfolioRow.siret)}}<br>
        Site ICPE : ${{row.site_icpe ? 'oui' : 'non'}}<br>
        Catégorie eau : ${{escapeHtml(waterCategory)}}<br>
        Commune : ${{escapeHtml(commune)}}<br>
        Qualité géoloc : ${{escapeHtml(row.geo_type || 'n.d.')}} (score ${{escapeHtml(formatGeoScore(row.geo_score))}})<br>
        Classe de grille : ${{escapeHtml(gridClass)}}
      `;
    }}

    function clearPortfolioLayer() {{
      portfolioLayer.clearLayers();
      updatePortfolioRadii();
    }}

    async function lookupPortfolioRows(rows) {{
      const response = await fetch(lookupApiUrl, {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ sirets: rows.map((row) => row.siret) }})
      }});
      if (!response.ok) {{
        throw new Error(`Lookup API error: ${{response.status}}`);
      }}
      return await response.json();
    }}

    function renderPortfolioRows(rows, lookupResults) {{
      clearPortfolioLayer();
      const status = document.getElementById('portfolio-status');
      if (!rows.length) {{
        status.innerHTML = 'Aucune ligne exploitable trouvée dans le fichier.';
        return;
      }}

      const lookupBySiret = new Map((lookupResults?.results || []).map((result) => [String(result.siret || ''), result]));
      let displayed = 0;
      let icpeCount = 0;
      let unmatchedCount = 0;

      rows.forEach((row) => {{
        const siret = String(row.siret || '').replace(/\\D+/g, '');
        if (!siret) return;
        const result = lookupBySiret.get(siret);
        if (!result || !result.found || !Number.isFinite(result.latitude) || !Number.isFinite(result.longitude)) {{
          unmatchedCount += 1;
          return;
        }}

        const localIcpeMatch = (icpeBySiret.get(siret) || [])[0] || null;
        if (result.site_icpe) icpeCount += 1;
        displayed += 1;

        const marker = L.circleMarker([result.latitude, result.longitude], {{
          pane: 'pane-portfolio',
          renderer: portfolioRenderer,
          radius: portfolioRadiusForZoom(map.getZoom()),
          stroke: true,
          weight: 1.6,
          color: '#0f172a',
          fillColor: result.site_icpe ? '#facc15' : '#fde68a',
          fillOpacity: 0.92
        }});
        const hoverHtml = buildPortfolioHover(result, row, localIcpeMatch);
        marker.on('mouseover', function() {{
          setHoverBox(hoverHtml);
        }});
        marker.on('mouseout', function() {{
          resetHoverBox();
        }});
        portfolioLayer.addLayer(marker);
      }});

      status.innerHTML = `
        <strong>${{rows.length}}</strong> lignes lues<br>
        <strong>${{displayed}}</strong> points affichés<br>
        <strong>${{icpeCount}}</strong> sites ICPE reconnus<br>
        <strong>${{unmatchedCount}}</strong> SIRET sans correspondance
      `;

      if (portfolioLayer.getLayers().length) {{
        map.fitBounds(portfolioLayer.getBounds(), {{ padding: [32, 32], maxZoom: 9 }});
      }}
    }}

    function normalizePortfolioRow(raw) {{
      const normalized = {{}};
      Object.entries(raw || {{}}).forEach(([key, value]) => {{
        normalized[String(key).trim().toLowerCase()] = value;
      }});
      const siret = String(
        normalized.siret ?? normalized.num_siret ?? normalized.identifiant ?? normalized.id_siret ?? ''
      ).replace(/\\D+/g, '');
      if (!siret) return null;
      return {{
        siret,
        company_name: normalized.company_name ?? normalized.nom_societe ?? normalized.nom_entreprise ?? '',
        nom_societe: normalized.nom_societe ?? normalized.company_name ?? '',
        portfolio_name: normalized.portfolio_name ?? normalized.portefeuille ?? normalized.site_label ?? '',
      }};
    }}

    function parsePortfolioWorkbook(file) {{
      return new Promise((resolve, reject) => {{
        const reader = new FileReader();
        reader.onload = function(event) {{
          try {{
            const data = new Uint8Array(event.target.result);
            const workbook = XLSX.read(data, {{ type: 'array' }});
            const firstSheet = workbook.SheetNames[0];
            const sheet = workbook.Sheets[firstSheet];
            const jsonRows = XLSX.utils.sheet_to_json(sheet, {{ defval: '' }});
            resolve(jsonRows.map(normalizePortfolioRow).filter(Boolean));
          }} catch (error) {{
            reject(error);
          }}
        }};
        reader.onerror = reject;
        reader.readAsArrayBuffer(file);
      }});
    }}

    document.getElementById('portfolio-upload').addEventListener('change', async function(event) {{
      const file = event.target.files && event.target.files[0];
      if (!file) return;
      const status = document.getElementById('portfolio-status');
      status.textContent = 'Chargement du portefeuille...';
      try {{
        let rows = [];
        if (file.name.toLowerCase().endsWith('.csv')) {{
          const text = await file.text();
          const workbook = XLSX.read(text, {{ type: 'string' }});
          const firstSheet = workbook.SheetNames[0];
          rows = XLSX.utils.sheet_to_json(workbook.Sheets[firstSheet], {{ defval: '' }})
            .map(normalizePortfolioRow)
            .filter(Boolean);
        }} else {{
          rows = await parsePortfolioWorkbook(file);
        }}
        const lookupResults = await lookupPortfolioRows(rows);
        renderPortfolioRows(rows, lookupResults);
      }} catch (error) {{
        status.textContent = 'Le portefeuille n’a pas pu être chargé. Vérifie le format, la colonne SIRET et la disponibilité du service de lookup.';
        console.error(error);
      }}
    }});

    map.fitBounds(coucheGrille.getBounds(), {{ padding: [24, 24] }});
  </script>
</body>
</html>"""


def main() -> None:
    print("\\n========== GENERATE ICPE EXPOSURE MAP ==========\\n")
    print("Grid input:", EXPOSURE_GRID_GEOJSON_FILE)
    print("ICPE matches input:", ICPE_BNPE_ECONOMIC_BEST_MATCH_FILE)
    print("ICPE groundwater context input:", ICPE_GROUNDWATER_CONTEXT_FILE)
    print("BNPE input:", NORMALIZED_BNPE_WITHDRAWALS_FILE)
    print("BSS input:", TRENDS_FILE)
    print("HTML output:", EXPOSURE_MAP_HTML_FILE)

    for path in [
        EXPOSURE_GRID_GEOJSON_FILE,
        ICPE_BNPE_ECONOMIC_BEST_MATCH_FILE,
        ICPE_GROUNDWATER_CONTEXT_FILE,
        NORMALIZED_BNPE_WITHDRAWALS_FILE,
        TRENDS_FILE,
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing input: {path}")

    ensure_dirs([Path(EXPOSURE_MAP_HTML_FILE).parent])

    grille_geojson = charger_grille()
    points_icpe = charger_icpe()
    html = construire_html(grille_geojson, points_icpe)
    Path(EXPOSURE_MAP_HTML_FILE).write_text(html, encoding="utf-8")

    print("Cellules de grille:", len(grille_geojson["features"]))
    print("Points ICPE:", len(points_icpe))
    print("Saved:", EXPOSURE_MAP_HTML_FILE)


if __name__ == "__main__":
    main()
