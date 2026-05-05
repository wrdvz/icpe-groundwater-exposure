const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      ...corsHeaders,
    },
  });
}

function normalizeSiret(value) {
  return String(value ?? "").replace(/\D+/g, "");
}

function shardKeyForSiret(siret, prefixLength, namespace) {
  const prefix = siret.slice(0, prefixLength).padEnd(prefixLength, "0");
  return `${namespace}/${prefix}.json`;
}

async function loadShard(env, objectKey) {
  const object = await env.SIRENE_SHARDS.get(objectKey);
  if (!object) return {};
  return await object.json();
}

function pickResult(record) {
  return {
    siret: record.siret,
    siren: record.siren ?? null,
    denomination: record.denomination ?? null,
    code_naf: record.code_naf ?? null,
    lib_naf: record.lib_naf ?? null,
    latitude: record.latitude ?? null,
    longitude: record.longitude ?? null,
    geo_score: record.geo_score ?? null,
    geo_type: record.geo_type ?? null,
    site_icpe: Boolean(record.site_icpe),
    icpe_category: record.icpe_category ?? null,
    icpe_site_sector: record.icpe_site_sector ?? null,
    icpe_nom_ets: record.icpe_nom_ets ?? null,
    grid_class: record.grid_class ?? null,
    groundwater_trend_cm_20y: record.groundwater_trend_cm_20y ?? null,
    groundwater_station_count_20km: record.groundwater_station_count_20km ?? null,
  };
}

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders });
    }

    const url = new URL(request.url);
    if (request.method === "GET" && url.pathname === "/health") {
      return json({ ok: true, service: "icpe-groundwater-lookup" });
    }

    if (request.method !== "POST" || url.pathname !== "/lookup-siret") {
      return json({ error: "Not found" }, 404);
    }

    let body;
    try {
      body = await request.json();
    } catch {
      return json({ error: "Body JSON invalide" }, 400);
    }

    const sirets = Array.isArray(body?.sirets)
      ? body.sirets.map(normalizeSiret).filter((s) => s.length === 14)
      : [];

    if (!sirets.length) {
      return json({ error: "Aucun SIRET exploitable fourni" }, 400);
    }

    const prefixLength = Number.parseInt(env.SHARD_PREFIX_LENGTH || "3", 10);
    const namespace = env.SHARD_NAMESPACE || "sirene/v1";
    const shardKeys = [...new Set(sirets.map((siret) => shardKeyForSiret(siret, prefixLength, namespace)))];

    const shardEntries = await Promise.all(
      shardKeys.map(async (key) => [key, await loadShard(env, key)])
    );
    const shardMap = new Map(shardEntries);

    const results = sirets.map((siret) => {
      const shardKey = shardKeyForSiret(siret, prefixLength, namespace);
      const shard = shardMap.get(shardKey) || {};
      const record = shard[siret];
      if (!record) {
        return { siret, found: false };
      }
      return { found: true, ...pickResult(record) };
    });

    const matched = results.filter((row) => row.found).length;
    return json({
      query_count: sirets.length,
      matched_count: matched,
      unmatched_count: sirets.length - matched,
      results,
    });
  },
};
