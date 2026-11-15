import numpy as np
import pandas as pd
import pytest

from mathgrades.inference import GradePredictor
from mathgrades.preprocessing import build_pipeline, raw_inputs_needed
from mathgrades.settings import SETTINGS
from mathgrades.training import GradeNetClassifier


def tiny_network(seed: int = SETTINGS.seed) -> GradeNetClassifier:
    return GradeNetClassifier(hidden_sizes=(8,), max_epochs=5, seed=seed)


def test_same_seed_gives_identical_results(split):
    X_train, X_test, y_train, _ = split
    first = build_pipeline(tiny_network()).fit(X_train, y_train).predict_proba(X_test)
    second = build_pipeline(tiny_network()).fit(X_train, y_train).predict_proba(X_test)
    other_seed = build_pipeline(tiny_network(seed=99)).fit(X_train, y_train).predict_proba(X_test)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other_seed)


def test_softmax_and_ordinal_heads_give_probabilities(split):
    X_train, X_test, y_train, _ = split
    for head in ("ordinal", "softmax"):
        model = GradeNetClassifier(hidden_sizes=(8,), max_epochs=5, head=head)
        probabilities = build_pipeline(model).fit(X_train, y_train).predict_proba(X_test)
        assert probabilities.shape == (len(X_test), SETTINGS.n_classes)
        assert np.allclose(probabilities.sum(axis=1), 1)


saved = pytest.mark.skipif(not SETTINGS.pipeline_path.exists(),
                           reason="run python -m mathgrades.train first")


@saved
def test_saved_predictor_returns_four_band_probabilities():
    predictor = GradePredictor()
    probabilities = predictor.probabilities(predictor.typical)
    assert len(probabilities) == len(SETTINGS.band_names) == 4
    assert np.isclose(probabilities.sum(), 1)
    assert 0 <= predictor.band(predictor.typical) < 4


@saved
def test_app_form_matches_what_the_model_reads():
    predictor = GradePredictor()
    assert [field["name"] for field in predictor.form] == raw_inputs_needed(predictor.pipeline)
    for field in predictor.form:
        if field["kind"] == "number":
            assert field["low"] <= field["start"] <= field["high"]
        else:
            assert field["start"] in [value for value, _ in field["options"]]
    predictor.pipeline.predict(pd.DataFrame([predictor.typical]))  # runs without missing columns
