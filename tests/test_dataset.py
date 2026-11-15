import numpy as np

from mathgrades.attributes import ATTRIBUTES
from mathgrades.dataset import to_band
from mathgrades.settings import SETTINGS


def test_dataset_matches_the_brief(students):
    assert students.shape == (395, 33)
    assert students.isna().sum().sum() == 0
    assert set(students.columns) == {a.name for a in ATTRIBUTES} | {"G3"}


def test_band_edges():
    grades = [0, 9, 10, 12, 13, 15, 16, 20]
    assert to_band(grades).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert [SETTINGS.grade_range(band) for band in range(4)] == ["0-9", "10-12", "13-15", "16-20"]


def test_band_counts(students):
    assert np.bincount(to_band(students["G3"])).tolist() == [130, 134, 91, 40]


def test_split_is_stratified(split):
    X_train, X_test, y_train, y_test = split
    assert (len(X_train), len(X_test)) == (296, 99)
    train_share = np.bincount(y_train) / len(y_train)
    test_share = np.bincount(y_test) / len(y_test)
    assert np.abs(train_share - test_share).max() < 0.02
