# MVP Cloudflare - lookup SIRET geolocalise

Ce dossier prepare le prochain palier du projet :

```text
front statique -> Worker Cloudflare -> shards SIRENE/ICPE dans R2
```

L'objectif est de reconnaitre :
- les `SIRET` ICPE deja dans la base
- mais aussi les `SIRET` non-ICPE, via SIRENE geolocalisee

## Architecture retenue

- **Cloudflare Pages** pour le front
- **Cloudflare Worker** pour l'API `POST /lookup-siret`
- **Cloudflare R2** pour stocker les shards de lookup
- **DuckDB / pandas / geopandas** en build-time seulement, pas en runtime

On evite volontairement de mettre toute la base SIRENE :
- dans `index.html`
- ou dans D1

## Arborescence

```text
cloudflare/
  lookup-worker/
    package.json
    wrangler.jsonc
    src/index.js
```

## 1. Creer le bucket R2

Nom conseille :

```text
icpe-groundwater-sirene-shards
```

## 2. Construire les shards localement

Le script prend un export SIRENE geolocalise (CSV ou Parquet), le reduit, le joint a :
- la base ICPE enrichie
- la grille d'exposition

puis il ecrit des shards JSON par prefixe de SIRET.

Exemple :

```bash
PYTHONPATH=src python3 scripts/09_build_sirene_lookup_shards.py \
  --sirene /chemin/vers/sirene_geolocalisee.parquet \
  --out-dir outputs/sirene_shards \
  --prefix-len 3 \
  --namespace sirene/v1
```

Sortie typique :

```text
outputs/sirene_shards/sirene/v1/000.json
outputs/sirene_shards/sirene/v1/001.json
...
outputs/sirene_shards/sirene/v1/999.json
```

## 3. Uploader les shards dans R2

Test sur quelques fichiers :

```bash
cd cloudflare/lookup-worker
npm install
npx wrangler r2 object put icpe-groundwater-sirene-shards/sirene/v1/000.json --file ../../outputs/sirene_shards/sirene/v1/000.json
```

Upload batch avec le helper du repo :

```bash
cd /Users/wrdvz/dev/parallaxe/_portfolio/icpe-groundwater-exposure
/usr/bin/python3 cloudflare/scripts/upload_r2_shards.py \
  --bucket icpe-groundwater-sirene-shards \
  --source-dir outputs/sirene_shards_active_g08/sirene/v1 \
  --prefix sirene/v1
```

Test limité sur 20 fichiers :

```bash
cd /Users/wrdvz/dev/parallaxe/_portfolio/icpe-groundwater-exposure
/usr/bin/python3 cloudflare/scripts/upload_r2_shards.py \
  --bucket icpe-groundwater-sirene-shards \
  --source-dir outputs/sirene_shards_active_g08/sirene/v1 \
  --prefix sirene/v1 \
  --limit 20
```

## 4. Deployer le Worker

```bash
cd cloudflare/lookup-worker
npm install
npx wrangler deploy
```

Le Worker expose :

### `GET /health`

Test simple

### `POST /lookup-siret`

Body :

```json
{
  "sirets": ["12345678901234", "98765432109876"]
}
```

Reponse :

```json
{
  "query_count": 2,
  "matched_count": 1,
  "unmatched_count": 1,
  "results": [
    {
      "found": true,
      "siret": "12345678901234",
      "siren": "123456789",
      "denomination": "Societe Exemple",
      "latitude": 48.85,
      "longitude": 2.35,
      "geo_score": 0.93,
      "site_icpe": true,
      "icpe_category": "Agro-industrie",
      "grid_class": "high_pressure_declining_groundwater"
    },
    {
      "found": false,
      "siret": "98765432109876"
    }
  ]
}
```

## 5. Branchement au front

La carte HTML actuelle peut evoluer ainsi :

1. on garde l'import `.xlsx`
2. au lieu de matcher seulement les SIRET ICPE embarques
3. on appelle le Worker avec la liste des `SIRET`
4. on affiche :
   - les matchs ICPE
   - les matchs SIRENE non-ICPE
   - le statut `site ICPE : oui/non`

## Remarques

- le sharding par prefixe de `SIRET` permet d'eviter de charger une grosse base complete en runtime
- `R2` sert de stockage d'objets
- le Worker reste ultra leger
- si plus tard on veut consolider au niveau `SIREN` ou groupe, on ajoutera une seconde couche d'enrichissement
