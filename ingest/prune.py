"""Remove index chunks and manifest rows for files that are gone or excluded.

A file is pruned when it is in the manifest but (a) no longer exists in the
source folder, or (b) its catalog row now says index != yes.

Run:  uv run python -m ingest.prune            # dry run - shows what would go
      uv run python -m ingest.prune --apply    # actually delete
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from common import Manifest, chunks_table, load_catalog, load_config, sql_str

console = Console()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="delete; without this flag it is a dry run")
    args = ap.parse_args()

    cfg = load_config()
    man = Manifest(cfg["data_dir"])
    catalog = load_catalog(cfg)          # None = no catalog: prune only deleted files
    src_root = Path(cfg["source_dir"])

    victims = []
    for path, h, chunks, status in man.all():
        missing = not (src_root / path).exists()
        excluded = catalog is not None and path not in catalog
        if missing or excluded:
            victims.append((path, chunks, "deleted from disk" if missing else "catalog index=no"))

    if not victims:
        console.print("Nothing to prune - index matches the source folder and catalog.")
        return 0

    for path, chunks, why in victims:
        console.print(f"{'PRUNE' if args.apply else 'would prune'}: {path}  ({chunks} chunks, {why})")

    if not args.apply:
        console.print(f"\nDry run. {len(victims)} file(s) would be pruned. Re-run with --apply to delete.")
        return 0

    tbl = chunks_table(cfg)
    for path, _, _ in victims:
        tbl.delete(f"file_path = {sql_str(path)}")
        man.con.execute("DELETE FROM files WHERE path=?", (path,))
    man.con.commit()
    tbl.create_fts_index("text", replace=True)
    console.print(f"Pruned {len(victims)} file(s). Chunks now in index: {tbl.count_rows()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
