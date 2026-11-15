"""Feature selection with two classic statistical tests.

For every column we ask: "is this column related to the grade band, or could
the differences we see be chance?" The p-value is the probability of seeing a
relationship at least this strong if there were none. A small p-value
(below 0.05) means the relationship is unlikely to be chance, so we keep it.

- Numeric columns (age, absences, G1, coded scales such as studytime):
  one-way ANOVA F-test. It compares the column's mean across the four bands.
- Text columns (school, Mjob, yes/no answers): chi-squared test of
  independence. It compares the observed counts in a column x band table
  with the counts we would expect if the two were unrelated.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway
from sklearn.base import BaseEstimator, TransformerMixin

from mathgrades.settings import SETTINGS


def is_numeric(column: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(column)


def test_relationship(column: pd.Series, bands: pd.Series) -> tuple[str, float, float]:
    """Return (test name, test statistic, p-value) for one column."""
    if is_numeric(column):
        values_per_band = [column[bands == band] for band in sorted(bands.unique())]
        statistic, p_value = f_oneway(*values_per_band)
        return "ANOVA F", float(statistic), float(p_value)

    counts = pd.crosstab(column, bands)
    statistic, p_value, _, _ = chi2_contingency(counts)
    return "chi-squared", float(statistic), float(p_value)


def run_feature_tests(X: pd.DataFrame, bands: pd.Series, level: float) -> pd.DataFrame:
    """One row per column, most significant first, with a 'kept' flag (p < level)."""
    rows = []
    for name in X.columns:
        test, statistic, p_value = test_relationship(X[name], bands)
        rows.append({"column": name, "test": test, "statistic": statistic, "p_value": p_value})

    table = pd.DataFrame(rows).sort_values("p_value", ignore_index=True)
    table["kept"] = table["p_value"] < level
    return table


class SignificanceFilter(BaseEstimator, TransformerMixin):
    """Pipeline step that keeps the columns whose test p-value is below `level`.

    Because it is a pipeline step, cross-validation refits it on each fold's
    training part: the students being scored never influence which columns
    are kept. That is what stops feature selection from leaking.
    """

    def __init__(self, level: float = SETTINGS.significance_level):
        self.level = level

    def fit(self, X: pd.DataFrame, y):
        self.tests_ = run_feature_tests(X, pd.Series(np.asarray(y), index=X.index), self.level)
        self.kept_ = self.tests_.loc[self.tests_["kept"], "column"].tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.kept_]
