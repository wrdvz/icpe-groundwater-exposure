import pandas as pd


CORE_COLUMNS = [
    "code_aiot",
    "x",
    "y",
    "code_epsg",
    "nom_ets",
    "num_dep",
    "adresse",
    "cd_insee",
    "cd_postal",
    "commune",
    "code_naf",
    "lib_naf",
    "num_siret",
    "cd_regime",
    "lib_regime",
    "seveso",
    "lib_seveso",
    "bovins",
    "porcs",
    "volailles",
    "carriere",
    "eolienne",
    "industrie",
    "ied",
    "priorite_nationale",
    "rubriques_autorisation",
    "rubriques_enregistrement",
    "rubriques_declaration",
    "date_modification",
    "derniere_inspection",
    "url_fiche",
]


def classify_site_sector(row: pd.Series) -> str:
    if row.get("industrie") == 1:
        return "Industrie"
    if row.get("carriere") == 1:
        return "Carriere"
    if row.get("eolienne") == 1:
        return "Eolien"
    if row.get("bovins") == 1 or row.get("porcs") == 1 or row.get("volailles") == 1:
        return "Elevage"
    return "Autre"


def load_raw_icpe_sites(path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=";",
        dtype={
            "code_aiot": "string",
            "num_dep": "string",
            "cd_insee": "string",
            "cd_postal": "string",
            "num_siret": "string",
        },
    )
    missing = [col for col in CORE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing ICPE columns: {missing}")
    return df[CORE_COLUMNS].copy()


def normalize_icpe_sites(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in ["x", "y", "code_epsg", "num_dep", "code_naf", "seveso"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in [
        "bovins",
        "porcs",
        "volailles",
        "carriere",
        "eolienne",
        "industrie",
        "ied",
        "priorite_nationale",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)

    out = out.dropna(subset=["code_aiot", "x", "y"]).copy()
    out = out[out["code_epsg"] == 2154].copy()

    out["site_sector"] = out.apply(classify_site_sector, axis=1)
    out["is_seveso"] = out["seveso"].fillna(3).astype(int) != 3
    out["is_water_relevant"] = (
        (out["industrie"] == 1)
        | (out["ied"] == 1)
        | (out["priorite_nationale"] == 1)
        | out["rubriques_autorisation"].notna()
        | out["rubriques_enregistrement"].notna()
    )

    out = out.sort_values(["site_sector", "code_aiot"]).reset_index(drop=True)
    return out


def summarize_icpe_sites(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("site_sector", as_index=False)
        .agg(
            n_sites=("code_aiot", "nunique"),
            n_industrie=("industrie", "sum"),
            n_ied=("ied", "sum"),
            n_priorite_nationale=("priorite_nationale", "sum"),
            n_seveso=("is_seveso", "sum"),
        )
        .sort_values("n_sites", ascending=False)
    )
