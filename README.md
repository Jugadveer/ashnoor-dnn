# Maths Grade Bands with an Ordinal PyTorch Network

GradeNet predicts which of four ordered bands a student's final maths grade
(G3, 0-20) will land in: At risk 0-9, Average 10-12, Good 13-15, Excellent
16-20. It is a three-hidden-layer PyTorch network with an ordinal output: it
answers "is the band above At risk / above Average / above Good?" with one
shared score and ordered thresholds, so the answers never contradict each other.
A Streamlit app shows the band, the chance of passing and what drove the
prediction for one student.

`REPORT.md` is the full project report; `VIVA_PREP.md` is the study guide.

## Results first

Repeated stratified 5-fold cross-validation, 3 repeats, on the 296 training
students (15 scores per model, the same splits for every model):

| | GradeNet | Copying the G2 band | Tuned SVM | Tuned gradient boosting |
|---|---|---|---|---|
| Pass / fail correct | **91.4 %** | 92.2 % | 92.1 % | 92.0 % |
| Within one band | **100 %** | 100 % | 99.9 % | 100 % |
| Exact band | **83.6 %** | 84.1 % | 83.1 % | 81.8 % |
| Weighted kappa (main score) | **0.915** ± 0.016 | 0.918 | 0.913 | 0.910 |

On the 99 held-out test students GradeNet scores kappa 0.884 (95 % CI
0.825-0.930), exact band 78.8 % (70.7-86.9 %), pass/fail 87.9 % (81.8-93.9 %),
within one band 100 %, and ranks failing students above passing ones with
ROC-AUC 0.967.

**What this means, plainly:**
- GradeNet is **statistically tied** with the tuned SVM, tuned gradient
  boosting and the naive rule "final band = band of the second-term grade G2"
  (McNemar p = 1.00 and 0.18; 5x2cv t-test p = 0.84 and 0.28).
- It is **significantly better** than copying the band of G1, the first-term
  grade (5x2cv p = 0.005; on the test set it was right for 18 students G1
  got wrong, against 8 the other way, McNemar p = 0.076).
- It is **not** better than copying G2. In Maths the final grade follows G2
  very closely (Spearman 0.96), so copying G2 is close to the best any model
  can do from these columns. What the network adds is a calibrated
  probability for every band and a chance of passing, which the copying rule
  cannot give.

## Brief checklist

| The brief asks for | Where |
|---|---|
| Data preprocessing | `mathgrades/dataset.py` (bands), `mathgrades/preprocessing.py` (grade features, MinMax scaling, one-hot) |
| Exploratory data analysis | `mathgrades/explore.py`, `reports/figures/eda_*.png` |
| Feature selection | `mathgrades/selection.py`: ANOVA F and chi-squared tests, inside every CV fold |
| DNN architecture design | `mathgrades/network.py`: `GradeNet`, `OrderedThresholds` |
| Model training | `mathgrades/training.py`: hand-written loop, AdamW, scheduler, early stopping, temperature scaling |
| Hyperparameter tuning | `mathgrades/search.py`: Optuna, 40 trials per model family |
| Performance evaluation | `mathgrades/metrics.py`, `mathgrades/significance.py`, `mathgrades/experiments.py` |
| Comparison with traditional ML | `mathgrades/classical.py`: tuned SVM and gradient boosting, dummy and naive baselines |
| Streamlit app | `streamlit_app.py` |

## Dataset

[UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance),
Mathematics file (the brief allows Maths or Portuguese).

| | Brief | This file |
|---|---|---|
| Records | 395 | 395 |
| Input attributes | about 30 | 30 + period grades G1, G2 |
| Target | G3 | G3, 0-20 |
| Missing / duplicate rows | | 0 / 0 |

| Band | At risk | Average | Good | Excellent |
|---|---|---|---|---|
| G3 | 0-9 | 10-12 | 13-15 | 16-20 |
| Students | 130 (33 %) | 134 (34 %) | 91 (23 %) | 40 (10 %) |

38 students have G3 = 0, and every one of them has 0 recorded absences:
probably students who left the course. They stay in as At risk, but they hide
the absences effect (Spearman 0.02 overall, -0.24 without them).

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

python -m mathgrades.explore       # dataset checks and EDA charts (seconds)
python -m mathgrades.train         # tuning, comparison, test, saved model (about 10-14 minutes)
python -m mathgrades.experiments   # ablation, learning curve, calibration, fairness (about 8 minutes)
python -m pytest                   # 12 tests, about 15 seconds
streamlit run streamlit_app.py     # the app
```

Training is deterministic: two full runs wrote identical `results.json`
(apart from the run time).

## Layout

```
streamlit_app.py        Report card (band, pass chance, drivers, what-if) and Model results pages
.streamlit/             config.toml (colours, STIX Two font), exercise_book.css (squared paper, red-pen answer)
mathgrades/
  settings.py           every setting, one frozen dataclass
  attributes.py         column meanings, the brief's four groups, readable answers
  dataset.py            download, load, G3 -> band, 75/25 stratified split
  explore.py            EDA (run as a module)
  selection.py          ANOVA / chi-squared tests; SignificanceFilter pipeline step
  preprocessing.py      GradeFeatures, encoder, the shared pipeline
  network.py            GradeNet with the ordinal head and skip path
  training.py           GradeNetClassifier (scikit-learn wrapper), loss, temperature scaling
  search.py             Optuna search spaces and studies
  classical.py          SVM, gradient boosting, dummy and naive baselines
  metrics.py            kappa, pass/fail, within-one, calibration scores
  significance.py       bootstrap intervals, McNemar, 5x2cv t-test
  explain.py            permutation importance, per-student drivers, worst mistakes
  charts.py             every saved chart, in the app's colours
  inference.py          input form and GradePredictor for the app
  train.py              main run (python -m mathgrades.train)
  experiments.py        long analyses (python -m mathgrades.experiments)
tests/                  pytest suite
models/                 grade_pipeline.joblib, input_form.json
reports/                results.json, experiments.json, trials/, feature_tests.csv,
                        worst_mistakes.csv, figures/
```

## Method

1. **Split once:** 296 training and 99 test students, stratified by band.
2. **One pipeline for every model**, refitted inside every CV fold:
   grade features -> significance filter -> MinMax / one-hot -> model.
   Nothing is learnt from the students being scored.
3. **Grade features** (Optuna decides): `G2_minus_G1` and the band of G1 and
   G2. The bands hand the network the band edges, which a smooth 0-1 input
   made hard to learn from 236 students.
4. **Feature tests:** ANOVA F (numeric) and chi-squared (text), keep p < level,
   level tuned. The final GradeNet keeps G2_band, G2, G1, G1_band, failures,
   G2_minus_G1, Medu and age (level 0.001).
5. **GradeNet:** 8 inputs -> 32 -> 16 -> 8 (each Linear, BatchNorm, ReLU,
   dropout 0.16) -> shared score (+ linear skip from the inputs) -> 3
   threshold logits with decreasing biases. 1,083 trainable parameters.
   Loss: binary cross-entropy over the thresholds, label smoothing 0.1.
   AdamW, learning rate 0.0129, weight decay 2.9e-6, batch 32;
   ReduceLROnPlateau; early stopping (76 epochs). Temperature scaling on the
   stop set (T = 0.39).
6. **Tuning:** Optuna TPE, 40 trials each for GradeNet, SVM (kernel, C, gamma,
   degree) and gradient boosting (learning rate, depth, leaves, iterations, L2),
   all scored by mean quadratic weighted kappa on the same 3 folds.
   Chosen: SVM linear kernel, C = 0.81; boosting 412 trees, depth 8, 13
   leaves, learning rate 0.013.
7. **Comparison:** repeated 5x3 CV, then one test, with bootstrap intervals
   (2,000 resamples), McNemar's test and Dietterich's 5x2cv t-test.

## Full test-set table

| Model | CV kappa | Test kappa (95 % CI) | Exact (95 % CI) | Within one | Pass/fail | At-risk recall | At-risk AUC |
|---|---|---|---|---|---|---|---|
| GradeNet (ordinal) | 0.915 ± 0.016 | 0.884 (0.825-0.930) | 0.788 (0.707-0.869) | 1.000 | 0.879 | 0.750 | 0.967 |
| SVM (tuned) | 0.913 ± 0.021 | 0.894 (0.843-0.936) | 0.798 (0.717-0.879) | 1.000 | 0.909 | 0.875 | - |
| Gradient boosting (tuned) | 0.910 ± 0.013 | 0.894 (0.841-0.936) | 0.798 (0.717-0.869) | 1.000 | 0.919 | 0.844 | 0.972 |
| Naive: band of G2 | 0.918 ± 0.015 | 0.910 (0.859-0.952) | 0.838 (0.768-0.909) | 1.000 | 0.909 | 0.875 | - |
| Naive: band of G1 | 0.825 ± 0.029 | 0.837 (0.768-0.891) | 0.687 (0.596-0.778) | 1.000 | 0.869 | 0.781 | - |
| Dummy: most frequent | 0.000 | 0.000 | 0.343 (0.253-0.444) | 0.899 | 0.677 | 0.000 | 0.500 |
| Dummy: stratified | 0.055 ± 0.092 | -0.061 (-0.241-0.123) | 0.313 (0.222-0.404) | 0.677 | 0.586 | 0.375 | 0.531 |

The SVM has no probabilities (no AUC); the naive rules and SVM output only a band.

## Experiments (`python -m mathgrades.experiments`, 472 s)

**Leakage check.** Same model and folds: feature selection inside the folds
0.921, fitted once outside 0.914. On this data the leak made no measurable
difference; inside-the-fold selection is still the only correct procedure.
The first version of this project reported CV kappa 0.874 (softmax network,
selection outside the folds, 3-fold CV); the gain comes from the ordinal head,
the grade-band features and the stricter filter, not from leakage.

**Ablation** (3 folds x 3 repeats, change in CV kappa vs the full model, 0.919):

| Change | CV kappa | Change | Takeaway |
|---|---|---|---|
| No feature selection | 0.848 | -0.071 | Noise columns hurt a small-data network most |
| Band features removed | 0.896 | -0.023 | Band edges are hard to learn from 236 students |
| Softmax head instead of ordinal | 0.907 | -0.012 | Using the band order helps |
| No dropout | 0.908 | -0.011 | Regularisation still matters at 1,083 parameters |
| Seed ensemble of 5 | 0.914 | -0.005 | No gain, so not used |
| One hidden layer instead of 3 | 0.915 | -0.004 | Depth adds little here |
| No linear skip path | 0.915 | -0.004 | Small help |
| No BatchNorm | 0.917 | -0.003 | Within noise |
| Label smoothing off | 0.917 | -0.002 | Within noise |
| Class weights on | 0.918 | -0.002 | Within noise; tuning switched them off |
| Trend removed | 0.918 | -0.001 | Within noise |

**Grade setups** (as in Cortez and Silva, 2008; CV kappa):

| Model | G1 and G2 | G1 only | No grades |
|---|---|---|---|
| GradeNet | 0.919 | 0.810 | 0.209 |
| Best baseline (SVM) | 0.911 | 0.787 | 0.220 |
| Naive band of latest grade | 0.918 | 0.825 | - |

Without any grades, no model gets far (kappa about 0.2): background alone
says little about the maths grade.

**Learning curve** (`learning_curve.png`): GradeNet is flat at 0.91-0.92 from
93 training students on; the SVM keeps rising, to 0.921 at 197. More data
would help the SVM more than this network.

**Calibration** (test set, `reliability.png`): temperature scaling (T = 0.39,
fitted on the stop set) sharpened under-confident probabilities. Expected
calibration error 0.217 -> 0.090, Brier score 0.381 -> 0.309, log loss
0.671 -> 0.501.

**Fairness** (test set): kappa is the same for girls and boys (0.882), but
the model under-predicts At risk for girls (32 % predicted vs 40 % actual;
boys 24 % vs 24 %). Urban students: 29 % predicted vs 34 % actual; rural
25 % vs 25 %. MS has only 7 test students, too few to judge.

**Error analysis** (`reports/worst_mistakes.csv`): all 21 test mistakes are
one band off. The most confident ones are students whose final grade crossed a
band edge that their G2 did not: G2 of 8 or 9 then a final 10 (predicted At
risk, actually passed), G2 of 10 then a final 9 (predicted Average, actually
failed), or G2 of 12 then a final 13. The three most confident wrong "At risk"
calls were students with 1-3 past failures, which pushes the model towards At risk.

## Design choices and limitations

- Ordinal head because the bands are ordered; kappa because it charges far misses more.
- Tuning-fold optimism: the best tuning score (0.921) is higher than the
  repeated-CV score (0.915), because picking the best of 40 trials on one
  split also picks some luck. The repeated CV with different splits is the
  number to trust. Rerunning the same network with a different row order
  moves its score by about 0.01.
- 99 test students: intervals are about ±0.05 kappa wide.
- Chi-squared assumes at least 5 expected students per cell; that fails for
  some rare categories, so those p-values are rough.
- Probabilities are calibrated on 60 stop-set students only.
- The app's inputs only offer values seen in training.
