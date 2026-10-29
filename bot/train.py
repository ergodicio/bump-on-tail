"""Retrain a paper neural closure without replacing the published checkpoint."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=int, choices=[3, 4], required=True)
    parser.add_argument("--samples", type=int, default=400_000)
    parser.add_argument("--steps", type=int, default=150_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    from bot.closures.homog_nn import save, train
    from bot.provenance import record

    model = train(args.order, n_samples=args.samples, n_steps=args.steps, seed=args.seed)
    save(model, args.order, args.output)
    config = record([])
    config.update(vars(args))
    config.update(
        hidden=[192, 192, 192],
        batch=1024,
        learning_rate=2e-3,
        single_loss_weight=3.0,
        single_fraction=0.5,
        mirror_fraction=0.0,
        impulse_fraction=0.0,
        potential_weight=1.0,
    )
    config["output"] = str(args.output)
    args.output.with_suffix(".json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
