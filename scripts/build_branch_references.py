"""Generate or check the classified branch-reference companion without modelling."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from force_consistency.data.endpoint_references import (
    build_reference_manifest, load_reference_manifest,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=ROOT / "manifests/landscape17-aspirin-split.json")
    parser.add_argument("--classification", type=Path, default=ROOT / "reports/landscape17-endpoints.json")
    parser.add_argument("--output", type=Path, default=ROOT / "manifests/landscape17-aspirin-branch-references.json")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check:
        manifest = build_reference_manifest(args.split, args.classification)
        # Existing evaluation companions are protected, including stale ones.
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    verified = load_reference_manifest(args.output, args.split, args.classification)
    print(json.dumps({"status": "verified", "branches": len(verified.references), "identity": verified.identity}))


if __name__ == "__main__":
    main()
