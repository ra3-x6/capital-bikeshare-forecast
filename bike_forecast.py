"""Backtest a one-day-ahead forecast of Capital Bikeshare rentals."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_DATA = Path(__file__).resolve().parent / "data" / "day.csv"
TEST_START = pd.Timestamp("2012-09-01")
FEATURES = [
    "lag_1", "lag_7", "mean_previous_7", "weekend", "annual_sin", "annual_cos",
]
METHODS = ["yesterday", "last_week", "ridge"]


def prepare_daily(source: pd.DataFrame) -> pd.DataFrame:
    """Validate daily granularity before shifting, with one record per calendar day."""
    dates = pd.to_datetime(source["dteday"], format="%Y-%m-%d", errors="raise")
    counts = pd.to_numeric(source["cnt"], errors="raise")
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError("Rental dates must be present and unique.")
    if not np.isfinite(counts).all() or (counts < 0).any() or (counts % 1 != 0).any():
        raise ValueError("Rental counts must be finite, nonnegative integers.")

    data = pd.DataFrame({"date": dates, "actual": counts}).set_index("date").sort_index()
    if len(data) < 9 or not data.index.to_series().diff().iloc[1:].eq(pd.Timedelta(days=1)).all():
        raise ValueError("Provide at least nine consecutive daily observations.")
    return data


def build_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Build features for date t using known calendar values and counts through t-1."""
    actual = daily["actual"]
    phase = 2 * np.pi * (daily.index.dayofyear - 1) / 365.25
    features = daily.assign(
        lag_1=actual.shift(1),
        lag_7=actual.shift(7),
        mean_previous_7=actual.shift(1).rolling(7).mean(),
        weekend=(daily.index.dayofweek >= 5).astype(int),
        annual_sin=np.sin(phase),
        annual_cos=np.cos(phase),
    )
    return features.dropna(subset=FEATURES)


def backtest(features: pd.DataFrame, test_start: pd.Timestamp = TEST_START) -> pd.DataFrame:
    """Fit once on earlier rows and predict each later day using its available history."""
    train = features.loc[features.index < test_start]
    test = features.loc[features.index >= test_start]
    if len(train) < 7 or test.empty:
        raise ValueError("The split needs at least seven training rows and one test row.")
    # A fixed, modest penalty keeps correlated lag features from dominating the fit.
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(train[FEATURES], train["actual"])
    return pd.DataFrame({
        "actual": test["actual"],
        "yesterday": test["lag_1"],
        "last_week": test["lag_7"],
        "ridge": model.predict(test[FEATURES]).clip(min=0),
    })


def evaluate(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return MAE in rentals per day and forecast bias (prediction minus actual)."""
    errors = predictions[METHODS].subtract(predictions["actual"], axis="index")
    return pd.DataFrame({
        "mae": errors.abs().mean(),
        "bias": errors.mean(),
    }).rename_axis("method")


def evaluate_by_day_type(predictions: pd.DataFrame) -> pd.DataFrame:
    """Show where each method's errors are concentrated."""
    groups = np.where(predictions.index.dayofweek >= 5, "Weekend", "Weekday")
    segments = {
        day_type: evaluate(group).assign(days=len(group))
        for day_type, group in predictions.groupby(groups)
    }
    return pd.concat(segments, names=["day_type", "method"])


def plot_backtest(predictions: pd.DataFrame, path: Path) -> None:
    """Plot the held-out predictions and their MAE comparison."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig, (series, comparison) = plt.subplots(
        2, 1, figsize=(9.4, 6.2), height_ratios=(2, 1), layout="constrained",
    )
    fig.set_facecolor("#f8fafc")
    for ax in (series, comparison):
        ax.set_facecolor("#f8fafc")
        ax.spines[["top", "right"]].set_visible(False)
    series.plot(predictions.index, predictions["actual"], color="#172d47", label="Observed")
    series.plot(
        predictions.index, predictions["ridge"],
        color="#b34f15", label="Ridge forecast", linewidth=1.35,
    )
    series.set_title("Capital Bikeshare next-day forecasts", loc="left", pad=14)
    series.set_ylabel("Daily rentals")
    series.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    series.legend(frameon=False, ncol=2)
    series.yaxis.grid(alpha=0.25)

    scores = evaluate(predictions)["mae"]
    bars = comparison.barh(
        ["Yesterday", "Same day last week", "Ridge"],
        scores,
        color=["#71859b", "#71859b", "#b34f15"],
    )
    comparison.invert_yaxis()
    comparison.bar_label(bars, fmt="%.0f", padding=7)
    comparison.set_xlim(0, max(scores.max() * 1.16, 1))
    comparison.set_xlabel("Mean absolute error (rentals per day)")
    comparison.xaxis.grid(alpha=0.25)
    comparison.set_axisbelow(True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--test-start", type=pd.Timestamp, default=TEST_START)
    parser.add_argument("--output", type=Path, default=Path("reports"))
    args = parser.parse_args()

    features = build_features(prepare_daily(pd.read_csv(args.input)))
    predictions = backtest(features, args.test_start)
    metrics = evaluate(predictions)
    args.output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output / "holdout-predictions.csv", float_format="%.3f")
    metrics.to_csv(args.output / "metrics.csv", float_format="%.3f")
    evaluate_by_day_type(predictions).to_csv(args.output / "day-type-metrics.csv", float_format="%.3f")
    plot_backtest(predictions, args.output / "forecast-backtest.png")
    print(f"Evaluated {len(predictions)} days.\n{metrics.round(2)}")


if __name__ == "__main__":
    main()
