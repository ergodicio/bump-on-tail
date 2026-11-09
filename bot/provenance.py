"""Compact source, environment, and input records for numerical runs."""

import hashlib
import importlib.metadata
import platform
import subprocess

from .paths import PACKAGE, ROOT


def record(inputs):
    def version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit = None
    source = hashlib.sha256()
    for path in sorted(PACKAGE.rglob("*.py")):
        source.update(str(path.relative_to(PACKAGE)).encode())
        source.update(path.read_bytes())
    return dict(
        source_commit=commit,
        python=platform.python_version(),
        python_source_sha256=source.hexdigest(),
        versions={
            name: version(name)
            for name in ["numpy", "scipy", "matplotlib", "jax", "equinox", "optax"]
        },
        inputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
    )
