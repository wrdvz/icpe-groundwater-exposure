from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ============================================================
# TIME WINDOW
# ============================================================

START_YEAR = 2005
END_YEAR = 2025
N_YEARS = END_YEAR - START_YEAR

# ============================================================
# ROBUSTNESS
# ============================================================

ROBUST_THRESHOLD = 5
ICPE_RADIUS_KM = 20
ICPE_RADIUS_M = ICPE_RADIUS_KM * 1000
GRID_SIZE_KM = 20
GRID_SIZE_M = GRID_SIZE_KM * 1000

# ============================================================
# CRS
# ============================================================

WORK_CRS = "EPSG:2154"
HTML_CRS = "EPSG:4326"

# ============================================================
# MAINLAND FRANCE BBOX
# ============================================================

LON_MIN, LON_MAX = -6, 10
LAT_MIN, LAT_MAX = 41, 52

# ============================================================
# STYLES
# ============================================================

NO_DATA_COLOR = "#e6e6e6"
HATCH_COLOR = "#7a7a7a"

CLASS_ORDER = [
    "Forte baisse (≤ -50 cm)",
    "Baisse modérée (-50 à -20 cm)",
    "Quasi stable (-20 à +20 cm)",
    "Hausse modérée (+20 à +50 cm)",
    "Forte hausse (≥ +50 cm)",
]

COLOR_MAP = {
    "Forte baisse (≤ -50 cm)": "#c0392b",
    "Baisse modérée (-50 à -20 cm)": "#e26f5d",
    "Quasi stable (-20 à +20 cm)": "#f4b5b2",
    "Hausse modérée (+20 à +50 cm)": "#4e64a8",
    "Forte hausse (≥ +50 cm)": "#1c2864",
    "No data": NO_DATA_COLOR,
}

# ============================================================
# ASSETS
# ============================================================

ROBOTO_CONDENSED_FONT = (
    PROJECT_ROOT
    / "assets"
    / "fonts"
    / "Roboto_Condensed"
    / "RobotoCondensed-VariableFont_wght.ttf"
)

# ============================================================
# INPUT DATA
# ============================================================

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

ANNUAL_MEANS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "station_trends"
    / f"network70_station_annual_means_{START_YEAR}_{END_YEAR}.csv"
)

RAW_DUCKDB_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "water"
    / f"groundwater_network70_{START_YEAR}_{END_YEAR}.duckdb"
)

RAW_ICPE_FILE = PROJECT_ROOT / "data" / "raw" / "icpe" / "icpe_installation_classee.csv"
RAW_BNPE_WITHDRAWALS_FILE = PROJECT_ROOT / "data" / "raw" / "prelevements" / "prelevements.csv"

# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

STATION_TRENDS_DIR = PROCESSED_DIR / "station_trends"
MAPPING_DIR = PROCESSED_DIR / "mapping"
AGGREGATION_DIR = PROCESSED_DIR / "aggregation"
GEOMETRY_DIR = PROCESSED_DIR / "geometry"
MAPS_DIR = OUTPUTS_DIR / "maps"
ICPE_PROCESSED_DIR = PROCESSED_DIR / "icpe"

# ============================================================
# OUTPUT FILES
# ============================================================

OUT_STATION_JOIN = (
    MAPPING_DIR
    / f"network70_station_to_polyg_affleurant_nv1_{START_YEAR}_{END_YEAR}.csv"
)

OUT_AGG = (
    AGGREGATION_DIR
    / f"network70_affleurant_nv1_agg_{START_YEAR}_{END_YEAR}.csv"
)

OUT_POLYG_AFF = (
    MAPPING_DIR
    / f"bdlisa_polyg_affleurant_nv1_{START_YEAR}_{END_YEAR}.csv"
)

OUT_GEOJSON_FULL = (
    GEOMETRY_DIR
    / f"bdlisa_eh_nv1_dissolved_{START_YEAR}_{END_YEAR}.geojson"
)

OUT_GEOJSON_LIGHT = (
    GEOMETRY_DIR
    / f"bdlisa_eh_nv1_dissolved_light_{START_YEAR}_{END_YEAR}.geojson"
)

OUT_PNG = (
    MAPS_DIR
    / f"groundwater_france_exclusive_polygons_{START_YEAR}_{END_YEAR}.png"
)

OUT_HTML = (
    MAPS_DIR
    / f"groundwater_france_exclusive_polygons_{START_YEAR}_{END_YEAR}.html"
)

NORMALIZED_ICPE_FILE = ICPE_PROCESSED_DIR / "icpe_sites_normalized.csv"

ICPE_GROUNDWATER_CONTEXT_FILE = (
    ICPE_PROCESSED_DIR
    / f"icpe_groundwater_context_{START_YEAR}_{END_YEAR}_{ICPE_RADIUS_KM}km.csv"
)

NORMALIZED_BNPE_WITHDRAWALS_FILE = (
    PROCESSED_DIR / "prelevements" / "bnpe_groundwater_withdrawals_normalized.csv"
)

ICPE_BNPE_MATCHES_FILE = (
    ICPE_PROCESSED_DIR / "icpe_bnpe_withdrawal_matches_2023.csv"
)

ICPE_BNPE_BEST_MATCH_FILE = (
    ICPE_PROCESSED_DIR / "icpe_bnpe_best_match_2023.csv"
)

ICPE_BNPE_ECONOMIC_BEST_MATCH_FILE = (
    ICPE_PROCESSED_DIR / "icpe_bnpe_economic_best_match_2023.csv"
)

EXPOSURE_GRID_FILE = (
    PROCESSED_DIR / "grid" / f"icpe_exposure_grid_{GRID_SIZE_KM}km.csv"
)

EXPOSURE_GRID_GEOJSON_FILE = (
    PROCESSED_DIR / "grid" / f"icpe_exposure_grid_{GRID_SIZE_KM}km.geojson"
)

EXPOSURE_MAP_HTML_FILE = DOCS_HTML_FILE = PROJECT_ROOT / "docs" / "index.html"

# ============================================================
# GEO DATA
# ============================================================

DEPARTEMENT_MAPPING = {
    "01": "Ain",
    "02": "Aisne",
    "03": "Allier",
    "04": "Alpes-de-Haute-Provence",
    "05": "Hautes-Alpes",
    "06": "Alpes-Maritimes",
    "07": "Ardèche",
    "08": "Ardennes",
    "09": "Ariège",
    "10": "Aube",
    "11": "Aude",
    "12": "Aveyron",
    "13": "Bouches-du-Rhône",
    "14": "Calvados",
    "15": "Cantal",
    "16": "Charente",
    "17": "Charente-Maritime",
    "18": "Cher",
    "19": "Corrèze",
    "2A": "Corse-du-Sud",
    "2B": "Haute-Corse",
    "21": "Côte-d'Or",
    "22": "Côtes-d'Armor",
    "23": "Creuse",
    "24": "Dordogne",
    "25": "Doubs",
    "26": "Drôme",
    "27": "Eure",
    "28": "Eure-et-Loir",
    "29": "Finistère",
    "30": "Gard",
    "31": "Haute-Garonne",
    "32": "Gers",
    "33": "Gironde",
    "34": "Hérault",
    "35": "Ille-et-Vilaine",
    "36": "Indre",
    "37": "Indre-et-Loire",
    "38": "Isère",
    "39": "Jura",
    "40": "Landes",
    "41": "Loir-et-Cher",
    "42": "Loire",
    "43": "Haute-Loire",
    "44": "Loire-Atlantique",
    "45": "Loiret",
    "46": "Lot",
    "47": "Lot-et-Garonne",
    "48": "Lozère",
    "49": "Maine-et-Loire",
    "50": "Manche",
    "51": "Marne",
    "52": "Haute-Marne",
    "53": "Mayenne",
    "54": "Meurthe-et-Moselle",
    "55": "Meuse",
    "56": "Morbihan",
    "57": "Moselle",
    "58": "Nièvre",
    "59": "Nord",
    "60": "Oise",
    "61": "Orne",
    "62": "Pas-de-Calais",
    "63": "Puy-de-Dôme",
    "64": "Pyrénées-Atlantiques",
    "65": "Hautes-Pyrénées",
    "66": "Pyrénées-Orientales",
    "67": "Bas-Rhin",
    "68": "Haut-Rhin",
    "69": "Rhône",
    "70": "Haute-Saône",
    "71": "Saône-et-Loire",
    "72": "Sarthe",
    "73": "Savoie",
    "74": "Haute-Savoie",
    "75": "Paris",
    "76": "Seine-Maritime",
    "77": "Seine-et-Marne",
    "78": "Yvelines",
    "79": "Deux-Sèvres",
    "80": "Somme",
    "81": "Tarn",
    "82": "Tarn-et-Garonne",
    "83": "Var",
    "84": "Vaucluse",
    "85": "Vendée",
    "86": "Vienne",
    "87": "Haute-Vienne",
    "88": "Vosges",
    "89": "Yonne",
    "90": "Territoire de Belfort",
    "91": "Essonne",
    "92": "Hauts-de-Seine",
    "93": "Seine-Saint-Denis",
    "94": "Val-de-Marne",
    "95": "Val-d'Oise",
}

DEPARTEMENT_TO_REGION = {
    # Île-de-France
    "75": "Île-de-France",
    "77": "Île-de-France",
    "78": "Île-de-France",
    "91": "Île-de-France",
    "92": "Île-de-France",
    "93": "Île-de-France",
    "94": "Île-de-France",
    "95": "Île-de-France",

    # Hauts-de-France
    "02": "Hauts-de-France",
    "59": "Hauts-de-France",
    "60": "Hauts-de-France",
    "62": "Hauts-de-France",
    "80": "Hauts-de-France",

    # Grand Est
    "08": "Grand Est",
    "10": "Grand Est",
    "51": "Grand Est",
    "52": "Grand Est",
    "54": "Grand Est",
    "55": "Grand Est",
    "57": "Grand Est",
    "67": "Grand Est",
    "68": "Grand Est",
    "88": "Grand Est",

    # Normandie
    "14": "Normandie",
    "27": "Normandie",
    "50": "Normandie",
    "61": "Normandie",
    "76": "Normandie",

    # Bretagne
    "22": "Bretagne",
    "29": "Bretagne",
    "35": "Bretagne",
    "56": "Bretagne",

    # Pays de la Loire
    "44": "Pays de la Loire",
    "49": "Pays de la Loire",
    "53": "Pays de la Loire",
    "72": "Pays de la Loire",
    "85": "Pays de la Loire",

    # Centre-Val de Loire
    "18": "Centre-Val de Loire",
    "28": "Centre-Val de Loire",
    "36": "Centre-Val de Loire",
    "37": "Centre-Val de Loire",
    "41": "Centre-Val de Loire",
    "45": "Centre-Val de Loire",

    # Bourgogne-Franche-Comté
    "21": "Bourgogne-Franche-Comté",
    "25": "Bourgogne-Franche-Comté",
    "39": "Bourgogne-Franche-Comté",
    "58": "Bourgogne-Franche-Comté",
    "70": "Bourgogne-Franche-Comté",
    "71": "Bourgogne-Franche-Comté",
    "89": "Bourgogne-Franche-Comté",
    "90": "Bourgogne-Franche-Comté",

    # Nouvelle-Aquitaine
    "16": "Nouvelle-Aquitaine",
    "17": "Nouvelle-Aquitaine",
    "19": "Nouvelle-Aquitaine",
    "23": "Nouvelle-Aquitaine",
    "24": "Nouvelle-Aquitaine",
    "33": "Nouvelle-Aquitaine",
    "40": "Nouvelle-Aquitaine",
    "47": "Nouvelle-Aquitaine",
    "64": "Nouvelle-Aquitaine",
    "79": "Nouvelle-Aquitaine",
    "86": "Nouvelle-Aquitaine",
    "87": "Nouvelle-Aquitaine",

    # Occitanie
    "09": "Occitanie",
    "11": "Occitanie",
    "12": "Occitanie",
    "30": "Occitanie",
    "31": "Occitanie",
    "32": "Occitanie",
    "34": "Occitanie",
    "46": "Occitanie",
    "48": "Occitanie",
    "65": "Occitanie",
    "66": "Occitanie",
    "81": "Occitanie",
    "82": "Occitanie",

    # Auvergne-Rhône-Alpes
    "01": "Auvergne-Rhône-Alpes",
    "03": "Auvergne-Rhône-Alpes",
    "07": "Auvergne-Rhône-Alpes",
    "15": "Auvergne-Rhône-Alpes",
    "26": "Auvergne-Rhône-Alpes",
    "38": "Auvergne-Rhône-Alpes",
    "42": "Auvergne-Rhône-Alpes",
    "43": "Auvergne-Rhône-Alpes",
    "63": "Auvergne-Rhône-Alpes",
    "69": "Auvergne-Rhône-Alpes",
    "73": "Auvergne-Rhône-Alpes",
    "74": "Auvergne-Rhône-Alpes",

    # Provence-Alpes-Côte d'Azur
    "04": "Provence-Alpes-Côte d'Azur",
    "05": "Provence-Alpes-Côte d'Azur",
    "06": "Provence-Alpes-Côte d'Azur",
    "13": "Provence-Alpes-Côte d'Azur",
    "83": "Provence-Alpes-Côte d'Azur",
    "84": "Provence-Alpes-Côte d'Azur",

    # Corse
    "2A": "Corse",
    "2B": "Corse",
}
