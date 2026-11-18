"""Command line: python -m fml {download,build,baseline,demo}."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fml.config import DEFAULT_CONFIG, load_config


def main() -> None:
    parser = argparse.ArgumentParser(prog="fml", description=__doc__)
    parser.add_argument("command", choices=["download", "build", "baseline", "demo"])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--overwrite", action="store_true", help="re-download existing files")
    args = parser.parse_args()
    cfg = load_config(args.config)
    paths = cfg["paths"]

    if args.command == "download":
        from fml.data.download import download

        download(cfg["leagues"], cfg["seasons"], paths["raw"], overwrite=args.overwrite)

    elif args.command == "build":
        from fml.data.clean import build_matches

        matches = build_matches(cfg["leagues"], cfg["seasons"], paths["raw"])
        Path(paths["processed"]).parent.mkdir(parents=True, exist_ok=True)
        matches.to_csv(paths["processed"], index=False)
        print(f"{len(matches)} matches -> {paths['processed']}")
        print(matches.groupby(["league", "season"]).size().unstack(0))

    elif args.command in ("baseline", "demo"):
        from fml.pipeline import run_baseline

        if args.command == "demo":
            from fml.data.synthetic import generate

            matches = generate()
            reports = Path(paths["reports"]) / "demo"
            print("Running on SYNTHETIC data (no internet needed).")
        else:
            matches = pd.read_csv(paths["processed"], parse_dates=["date"])
            reports = Path(paths["reports"])
        results = run_baseline(matches, cfg, reports)
        with pd.option_context("display.width", 120, "display.precision", 4):
            print(results.to_string(index=False))
        print(f"\nSaved metrics, predictions and calibration plot to {reports}/")


if __name__ == "__main__":
    main()
