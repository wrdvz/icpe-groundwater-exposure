"""Upload en batch des shards SIRENE vers un bucket R2 via wrangler."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True, help="Nom du bucket R2")
    parser.add_argument("--source-dir", required=True, help="Dossier local contenant les shards")
    parser.add_argument("--prefix", default="", help="Préfixe optionnel dans le bucket")
    parser.add_argument("--wrangler-bin", default="npx wrangler", help="Commande wrangler à utiliser")
    parser.add_argument("--limit", type=int, default=None, help="Limiter le nombre de fichiers pour un test")
    parser.add_argument(
        "--remote",
        action="store_true",
        default=True,
        help="Uploader vers le vrai bucket R2 distant (recommandé).",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    files = sorted(source_dir.rglob("*.json"))
    if args.limit:
        files = files[: args.limit]

    if not files:
        raise SystemExit(f"Aucun shard JSON trouvé dans {source_dir}")

    uploaded = 0
    for file_path in files:
        rel = file_path.relative_to(source_dir).as_posix()
        object_key = f"{args.prefix.rstrip('/')}/{rel}" if args.prefix else rel
        remote_flag = " --remote" if args.remote else ""
        command = (
            f'{args.wrangler_bin} r2 object put {args.bucket}/{object_key}'
            f' --file "{file_path}"{remote_flag}'
        )
        print(f"[{uploaded + 1}/{len(files)}] {object_key}")
        subprocess.run(command, shell=True, check=True)
        uploaded += 1

    print(f"Upload terminé: {uploaded} fichiers envoyés vers {args.bucket}")


if __name__ == "__main__":
    main()
