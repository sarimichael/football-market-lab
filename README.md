# Football Market Lab

**Can a rolling-window model beat the football betting market?** This project prices
matches in four European leagues (Premier League, Bundesliga, Ligue 1, Eredivisie;
2020/21–2025/26), scores those prices against the market's own probabilities, and then
tests how bet-sizing rules (flat stakes and fractional Kelly) behave when the edge is uncertain.

The football is the lab. The real subject is the one trading firms care about: pricing
against a market maker, honest out-of-sample evaluation, calibration, and risk.

> **Status: Phase 1 of 4 complete** — data pipeline, market baseline, Elo model,
> rolling walk-forward evaluation, leakage tests and CI. Models, staking and the
> write-up follow (see [Roadmap](#roadmap)).

## Quickstart

```bash
git clone https://github.com/sarimichael/football-market-lab.git
cd football-market-lab
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

make demo        # whole pipeline on synthetic data, no internet needed
make data        # download + clean real data from football-data.co.uk
make baseline    # score market vs Elo vs base rates on real data
make test        # unit and leakage tests
```

No `make`? Use `python -m fml demo`, `python -m fml download`, `python -m fml build`,
`python -m fml baseline` and `pytest`.

## Method

| Step | What happens | Where |
| --- | --- | --- |
| Data | Results, match stats and opening/closing odds for 4 leagues × 6 seasons from [football-data.co.uk](https://www.football-data.co.uk/) | `src/fml/data/` |
| Market baseline | Bookmaker margin removed from Pinnacle closing odds (multiplicative, power or Shin method) | `src/fml/market/margin.py` |
| Elo | Pre-match ratings per league, home advantage, goal-difference multiplier, regression to the mean each summer, promoted teams start low | `src/fml/models/elo.py` |
| Outcome model | Multinomial logistic regression from Elo difference to home/draw/away | `src/fml/models/outcome.py` |
| Walk-forward | Refit every month on the previous **3 years only**; predict the next month; the window rolls forward with time | `src/fml/evaluation/walkforward.py` |
| Scoring | Ranked probability score, log loss, Brier score, calibration curve | `src/fml/evaluation/metrics.py` |

Every setting (leagues, seasons, window length, Elo parameters, margin method, recency
half-life) lives in [`configs/base.yaml`](configs/base.yaml).

### Guarding against look-ahead bias

- Elo ratings attached to a match are computed **before** kickoff. A test changes a
  match's result and checks that no rating up to that date moves.
- The walk-forward loop never trains on a match on or after the refit date, and the
  window is exactly 3 years. Both are tested.
- Promoted teams are rated from a snapshot taken at the start of the season, never from
  ratings already updated by other same-day results (this bug was caught by the tests).

## Results

Run `make baseline` to produce, in `reports/`:

- `baseline_metrics.csv` — RPS, log loss and Brier score per model and league
- `market_margins.csv` — average bookmaker margin per league
- `figures/calibration.png` — reliability curve for the market and the model
- `predictions.csv` — every out-of-sample forecast

Results on real data will be added here with confidence intervals once Phase 2 is done.

## Roadmap

1. **Foundations** ✅ data pipeline, market baseline, Elo, walk-forward, tests, CI
2. **Models** — Dixon–Coles with time decay, gradient boosting on lagged features, calibration
3. **Betting and risk** — edge filter, closing line value, flat vs full/half/quarter Kelly,
   Baker–McHale shrinkage, bootstrap and Monte Carlo bankroll paths
4. **Release** — results with confidence intervals, limitations, write-up

## Limitations

- The market baseline uses **closing** odds, which are known only at kickoff. Bets placed
  earlier must be tested at earlier prices.
- Synthetic data (`make demo`) only checks that the code runs; its numbers mean nothing.
- Historical backtests ignore bookmaker limits on winning accounts.

## Data and licence

Match data: [football-data.co.uk](https://www.football-data.co.uk/) (Joseph Buchdahl).
The raw files are downloaded by the scripts and are not redistributed in this repository.
Code: MIT licence.

This is a research project on forecasting and market efficiency, not betting advice.
