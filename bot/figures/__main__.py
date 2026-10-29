"""Redraw the paper figures using bundled or explicitly selected data."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from bot.paths import ASSETS, DATA

from . import comparisons, validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure",
        nargs="+",
        choices=["all", "validation", "equivariance", "coverage", "scan", "twomode", "standing"],
        default=["all"],
    )
    parser.add_argument(
        "--data", type=Path, default=DATA, help="Directory containing numerical NPZ inputs"
    )
    parser.add_argument("--output", type=Path, default=Path("build/figures"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = (
        ["validation", "equivariance", "coverage", "scan", "twomode", "standing"]
        if "all" in args.figure
        else args.figure
    )
    for name in names:
        if name == "validation":
            validation.main(args.output, args.data)
        elif name == "equivariance":
            shutil.copy2(
                ASSETS / "fig_equivariance_serif.png", args.output / "fig_equivariance_serif.png"
            )
        else:
            getattr(comparisons, name)(args.data, args.output)
        print(f"Rendered {name}")
    inputs = list(args.data.glob("*.npz")) + [ASSETS / "fig_equivariance_serif.png"]
    record = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    (args.output / "render_inputs.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
