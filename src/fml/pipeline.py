"""Phase 1 pipeline: market baseline vs Elo vs base rates, scored walk-forward."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from fml.evaluation.metrics import calibration_table, score_all  # noqa: E402
from fml.evaluation.walkforward import walk_forward  # noqa: E402
from fml.market.margin import market_probabilities  # noqa: E402
from fml.models.elo import EloParams, compute_elo  # noqa: E402
from fml.models.outcome import BaseRateModel, EloOutcomeModel  # noqa: E402

PROB_COLS = ["p_h", "p_d", "p_a"]


def run_baseline(
    matches: pd.DataFrame, cfg: dict[str, Any], reports_dir: str | Path
) -> pd.DataFrame:
    """Score the market, Elo and base-rate forecasts on the same out-of-sample matches."""
    reports_dir = Path(reports_dir)
    (reports_dir / "figures").mkdir(parents=True, exist_ok=True)
    m_cfg, wf_cfg = cfg["market"], cfg["walk_forward"]

    market = market_probabilities(
        matches, m_cfg["closing_cols"], m_cfg.get("closing_fallback"), m_cfg["margin_method"]
    )
    market["match_id"] = matches["match_id"]

    data = matches.join(compute_elo(matches, EloParams(**cfg["elo"])))
    wf_kwargs = dict(
        test_start=wf_cfg["test_start"],
        window_years=wf_cfg["window_years"],
        step=wf_cfg["step"],
        half_life_days=wf_cfg.get("half_life_days"),
    )
    forecasts = {
        "Market (closing odds, margin removed)": market[["match_id", *PROB_COLS]],
        "Elo + logistic (walk-forward)": walk_forward(
            data, EloOutcomeModel, ["elo_diff"], **wf_kwargs
        )[["match_id", *PROB_COLS]],
        "Base rates (walk-forward)": walk_forward(data, BaseRateModel, ["elo_diff"], **wf_kwargs)[
            ["match_id", *PROB_COLS]
        ],
    }

    # Score every forecast on exactly the same matches: those every forecast covers.
    common = set(matches["match_id"])
    for f in forecasts.values():
        common &= set(f.dropna(subset=PROB_COLS)["match_id"])
    base = matches[matches["match_id"].isin(common)][["match_id", "league", "season", "result"]]

    rows, all_preds = [], []
    for name, f in forecasts.items():
        merged = base.merge(f, on="match_id")
        all_preds.append(merged.assign(model=name))
        groups = [("All leagues", merged)] + list(merged.groupby("league"))
        for league, g in groups:
            rows.append(
                {
                    "model": name,
                    "league": league,
                    **score_all(g[PROB_COLS].to_numpy(), g["result"].to_numpy()),
                }
            )

    results = pd.DataFrame(rows)
    results.to_csv(reports_dir / "baseline_metrics.csv", index=False)
    pd.concat(all_preds).to_csv(reports_dir / "predictions.csv", index=False)

    # Margins and odds sources per league: the baseline may mix Pinnacle and market-average
    # closing odds, so report how often each was used rather than hiding it.
    sources = matches.assign(margin=market["margin"], odds_source=market["odds_source"])
    summary = sources.groupby(["league", "odds_source"]).agg(
        n=("margin", "size"), mean_margin=("margin", "mean")
    )
    summary.to_csv(reports_dir / "market_margins.csv")

    _write_run_info(matches, cfg, reports_dir)

    _plot_calibration(all_preds, reports_dir / "figures" / "calibration.png")
    return results


def _write_run_info(matches: pd.DataFrame, cfg: dict[str, Any], reports_dir: Path) -> None:
    """Register the run: when, which code, which config, which data."""
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    data_bytes = pd.util.hash_pandas_object(matches, index=False).to_numpy().tobytes()
    info = {
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_revision": revision,
        "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16],
        "data_sha256": hashlib.sha256(data_bytes).hexdigest()[:16],
        "n_matches": int(len(matches)),
        "date_range": [str(matches["date"].min()), str(matches["date"].max())],
        "config": cfg,
    }
    (reports_dir / "run_info.json").write_text(json.dumps(info, indent=2, default=str))


def _plot_calibration(all_preds: list[pd.DataFrame], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="Perfect calibration")
    for preds in all_preds:
        name = preds["model"].iloc[0]
        if name.startswith("Base rates"):
            continue
        table = calibration_table(preds[PROB_COLS].to_numpy(), preds["result"].to_numpy())
        ax.plot(table["predicted"], table["observed"], marker="o", label=name)
    ax.set_xlabel("Forecast probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration, out-of-sample matches")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
