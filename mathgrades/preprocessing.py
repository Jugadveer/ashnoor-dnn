"""The model pipeline: grade features -> select columns -> encode them -> model.

Every model in the project (network, SVM, gradient boosting, dummies) is
wrapped in the same pipeline. scikit-learn clones and refits the whole
pipeline on each cross-validation fold, so the selected columns, the
scaling and the one-hot categories are always learnt from training rows only.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from mathgrades.dataset import to_band
from mathgrades.selection import SignificanceFilter

# engineered column -> the raw columns it is computed from
ENGINEERED_FROM = {"G2_minus_G1": ["G1", "G2"], "G1_band": ["G1"], "G2_band": ["G2"]}


class GradeFeatures(BaseEstimator, TransformerMixin):
    """Optional features built from the term grades (Optuna decides which help):

    trend=True: G2_minus_G1, is the student improving or slipping between terms?
    bands=True: G1_band and G2_band, each term grade put through the same band
                edges as the final grade (0 = At risk ... 3 = Excellent).

    The bands give the network the band edges directly. Without them it has to
    learn four sharp steps from a smooth 0-1 input, which 236 students do not
    teach well. A feature is skipped when its source grade is not in the data
    (the "no grades" experiments). Learns nothing, so fit() does nothing.
    """

    def __init__(self, trend: bool = False, bands: bool = False):
        self.trend = trend
        self.bands = bands

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        added = {}
        if self.trend and {"G1", "G2"} <= set(X.columns):
            added["G2_minus_G1"] = X["G2"] - X["G1"]
        if self.bands:
            for grade in ("G1", "G2"):
                if grade in X.columns:
                    added[f"{grade}_band"] = to_band(X[grade])
        return X.assign(**added)


def build_encoder() -> ColumnTransformer:
    """Numeric columns -> MinMaxScaler, (x - min) / (max - min), so training
    values land in 0-1; text columns -> one 0/1 column per category.

    The column selectors decide numeric vs text when the encoder is fitted,
    so it works on whichever columns the filter kept in that fold.
    handle_unknown="ignore" turns a category never seen in training into all zeros.
    """
    return ColumnTransformer([
        ("scale", MinMaxScaler(), make_column_selector(dtype_include="number")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         make_column_selector(dtype_exclude="number")),
    ])


def build_pipeline(model, trend: bool = False, bands: bool = False,
                   level: float | None = None) -> Pipeline:
    """level=None keeps the filter's default significance level."""
    select = SignificanceFilter() if level is None else SignificanceFilter(level)
    return Pipeline([
        ("grades", GradeFeatures(trend, bands)),
        ("select", select),
        ("encode", build_encoder()),
        ("model", model),
    ])


def raw_inputs_needed(pipeline: Pipeline) -> list[str]:
    """The raw columns a fitted pipeline reads: the kept columns, with each
    engineered column replaced by the grades it is computed from."""
    needed = []
    for column in pipeline.named_steps["select"].kept_:
        for raw in ENGINEERED_FROM.get(column, [column]):
            if raw not in needed:
                needed.append(raw)
    return needed
