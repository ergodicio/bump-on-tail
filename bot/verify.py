"""Compare recomputed arrays with the fixed numerical inputs for the paper."""

import argparse
from pathlib import Path

import numpy as np

from .paths import DATA


def compare(data):
    """Validate all fields, shapes, finite-value masks and numerical agreement.

    Tolerances allow platform-level roundoff in nonlinear time integration;
    integer arrays and initial-condition parameter arrays must agree exactly.
    """
    results = {}
    for reference_path in sorted(DATA.glob("*.npz")):
        candidate_path = Path(data) / reference_path.name
        with (
            np.load(reference_path, allow_pickle=False) as reference,
            np.load(candidate_path, allow_pickle=False) as candidate,
        ):
            if set(reference.files) != set(candidate.files):
                raise AssertionError(f"{reference_path.name}: array keys differ")
            largest = 0.0
            for key in reference.files:
                expected = reference[key]
                actual = candidate[key]
                if expected.shape != actual.shape:
                    raise AssertionError(f"{reference_path.name}:{key}: shape changed")
                np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
                if (
                    expected.dtype.kind in "iub"
                    or key.endswith("params")
                    or key in ("u_b_vals", "eps_vals", "k_star", "t")
                ):
                    np.testing.assert_array_equal(actual, expected)
                else:
                    mask = np.isfinite(expected)
                    scale = max(float(np.max(abs(expected[mask]))), 1e-30)
                    # Preserve near-zero crossings while allowing JAX/BLAS roundoff.
                    atol = 5e-8 if reference_path.name == "growth_scan.npz" else 1e-7 * scale
                    np.testing.assert_allclose(
                        actual,
                        expected,
                        rtol=1e-5,
                        atol=atol,
                        equal_nan=True,
                        err_msg=f"{reference_path.name}:{key}",
                    )
                    largest = max(
                        largest, float(np.max(abs(actual[mask] - expected[mask]))) / scale
                    )
            results[reference_path.name] = largest
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    for name, error in compare(args.data).items():
        print(f"{name}: passed (maximum absolute difference / array peak {error:.3e})")


if __name__ == "__main__":
    main()
