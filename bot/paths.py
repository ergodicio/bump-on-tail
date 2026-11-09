"""Bundled inputs, located independently of the current working directory."""

from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent
DATA = PACKAGE / "data"
MODELS = PACKAGE / "models"
ASSETS = PACKAGE / "assets"
