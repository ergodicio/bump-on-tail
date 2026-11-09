"""Repeat the paper's parity-constrained fit at the neural training frequencies."""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    from bot.closures.homog_nn import make_dataset
    from bot.closures.opt_hunana import fit_coefficients, opt_coefficients

    result = {}
    for n in (3, 4):
        frequencies = make_dataset(n, 400_000, seed=0, return_zetas=True)[-1]
        fitted = fit_coefficients(n, frequencies)
        print(
            f"N={n}: {len(frequencies)} frequencies; max coefficient difference {max(abs(fitted - opt_coefficients(n))):.3e}"
        )
        result[f"a{n}"] = fitted
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)


if __name__ == "__main__":
    main()
