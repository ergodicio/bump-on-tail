"""Recompute numerical experiments in a new output directory."""

import argparse
import json
from pathlib import Path

from bot.paths import DATA, MODELS
from bot.provenance import record

from . import comparisons, validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment", choices=["all", "validation", "coverage", "scan", "twomode", "standing"]
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = (
        ["validation", "coverage", "scan", "twomode", "standing"]
        if args.experiment == "all"
        else [args.experiment]
    )
    expected = {
        "validation": ["validation_kappa", "validation_itg_trace", "validation_itg_scan"],
        "coverage": ["coverage"],
        "scan": ["growth_scan"],
        "twomode": ["twomode"],
        "standing": ["standing"],
    }
    for name in names:
        for stem in expected[name]:
            if (args.output / f"{stem}.npz").exists():
                parser.error(f"Refusing to overwrite {stem}.npz; select a new output directory")
    for name in names:
        if name == "validation":
            validation.main(args.output)
        else:
            getattr(comparisons, name)(args.output)
        print(f"Completed {name}", flush=True)
    inputs = list(MODELS.glob("*.eqx")) + [DATA / "scan_parameters.json"]
    provenance = record(inputs)
    provenance["experiments"] = names
    (args.output / f"provenance_{args.experiment}.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
