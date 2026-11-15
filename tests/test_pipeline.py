import numpy as np
from sklearn.dummy import DummyClassifier

from mathgrades.dataset import to_band
from mathgrades.preprocessing import (ENGINEERED_FROM, GradeFeatures, build_pipeline,
                                      raw_inputs_needed)
from mathgrades.selection import run_feature_tests
from mathgrades.training import ordinal_probabilities


def test_filter_and_scaler_learn_from_training_rows_only(split):
    X_train, _, y_train, _ = split
    first_rows, first_bands = X_train.iloc[:150], y_train.iloc[:150]
    pipeline = build_pipeline(DummyClassifier()).fit(first_rows, first_bands)

    expected = run_feature_tests(first_rows, first_bands, pipeline.named_steps["select"].level)
    assert pipeline.named_steps["select"].kept_ == expected.loc[expected["kept"], "column"].tolist()

    scaler = pipeline.named_steps["encode"].named_transformers_["scale"]
    numeric = [c for c in pipeline.named_steps["select"].kept_ if first_rows[c].dtype.kind in "iuf"]
    assert np.array_equal(scaler.data_min_, first_rows[numeric].min().to_numpy())
    assert np.array_equal(scaler.data_max_, first_rows[numeric].max().to_numpy())


def test_grade_features_and_raw_inputs(split):
    X_train, _, y_train, _ = split
    pipeline = build_pipeline(DummyClassifier(), trend=True, bands=True, level=0.001)
    pipeline.fit(X_train, y_train)
    rows = X_train.head(5)
    added = pipeline.named_steps["grades"].transform(rows)
    assert (added["G2_minus_G1"] == rows["G2"] - rows["G1"]).all()
    assert (added["G2_band"] == to_band(rows["G2"])).all()

    needed = raw_inputs_needed(pipeline)
    assert not set(needed) & set(ENGINEERED_FROM)  # the app never asks for an engineered column
    assert set(needed) <= set(X_train.columns)


def test_grade_features_skip_missing_grades(split):
    X_train = split[0]
    without_grades = X_train.drop(columns=["G1", "G2"]).head(3)
    unchanged = GradeFeatures(trend=True, bands=True).transform(without_grades)
    assert list(unchanged.columns) == list(without_grades.columns)


def test_ordinal_probabilities_are_valid():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(50, 1)) * 3
    logits = scores + np.array([1.5, 0.0, -2.0])  # decreasing thresholds, as the network builds them
    probabilities = ordinal_probabilities(logits)
    assert probabilities.shape == (50, 4)
    assert np.allclose(probabilities.sum(axis=1), 1)
    assert (probabilities >= 0).all()
