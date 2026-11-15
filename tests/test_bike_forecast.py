import pandas as pd
import pytest

from bike_forecast import FEATURES, backtest, build_features, evaluate, prepare_daily


def sample():
    return pd.DataFrame({
        "dteday": pd.date_range("2012-01-01", periods=25).strftime("%Y-%m-%d"),
        "cnt": range(25),
    })


def test_features_do_not_use_target_or_future_counts():
    source = sample()
    date = pd.Timestamp("2012-01-10")
    features = build_features(prepare_daily(source))
    source.loc[9:, "cnt"] = 1000
    changed = build_features(prepare_daily(source))
    pd.testing.assert_series_equal(features.loc[date, FEATURES], changed.loc[date, FEATURES])
    assert features.loc[date, ["lag_1", "lag_7", "mean_previous_7"]].tolist() == [8, 2, 5]


def test_backtest_excludes_holdout_targets_from_fit():
    source = sample()
    date = pd.Timestamp("2012-01-20")
    original = backtest(build_features(prepare_daily(source)), date)
    source.loc[19:, "cnt"] += 1000
    changed = backtest(build_features(prepare_daily(source)), date)
    assert original.index.min() == date
    assert original.loc[date, "ridge"] == changed.loc[date, "ridge"]
    assert original.loc[date, "yesterday"] == 18
    assert original.loc[date, "last_week"] == 12
    assert changed.loc[date + pd.Timedelta(days=1), "yesterday"] == 1019


def test_gaps_and_duplicate_dates_are_invalid():
    source = sample()
    with pytest.raises(ValueError, match="consecutive"):
        prepare_daily(source.drop(index=4))
    with pytest.raises(ValueError, match="unique"):
        prepare_daily(pd.concat([source, source.iloc[:1]]))


def test_metric_units_and_bias_direction():
    predictions = pd.DataFrame({
        "actual": [10, 20], "yesterday": [12, 18], "last_week": [15, 25], "ridge": [9, 17],
    })
    results = evaluate(predictions)
    assert results.loc["yesterday"].to_dict() == {"mae": 2, "bias": 0}
    assert results.loc["last_week"].to_dict() == {"mae": 5, "bias": 5}
    assert results.loc["ridge"].to_dict() == {"mae": 2, "bias": -2}
