#!/usr/bin/env bash
set -euo pipefail

BUCKET="${1:-icpe-groundwater-sirene-shards}"
SOURCE_DIR="${2:-/Users/wrdvz/dev/parallaxe/_portfolio/icpe-groundwater-exposure/outputs/sirene_shards_active_g08/sirene/v1}"
PREFIX="${3:-sirene/v1}"

cd /Users/wrdvz/dev/parallaxe/_portfolio/icpe-groundwater-exposure/cloudflare/lookup-worker

/usr/bin/python3 ../scripts/upload_r2_shards.py \
  --bucket "$BUCKET" \
  --source-dir "$SOURCE_DIR" \
  --prefix "$PREFIX"
