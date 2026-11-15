# Project Report: Predicting Maths Grade Bands with an Ordinal Neural Network

## Abstract

This project predicts which of four ordered bands a secondary-school student's
final mathematics grade will fall in, using the UCI Student Performance
dataset (395 students). The model, GradeNet, is a three-hidden-layer PyTorch
network with an ordinal output head. It was tuned with Optuna and compared,
on the same folds and with the same pipeline, against a tuned support vector
machine, tuned gradient boosting, two dummy classifiers and two naive rules
that copy the band of an earlier term grade. Over repeated 5-fold
cross-validation GradeNet reaches a quadratic weighted kappa of 0.915, 83.6 %
exact-band accuracy, 91.4 % pass/fail accuracy and 100 % within one band. On
99 held-out students it scores kappa 0.884 (95 % CI 0.825-0.930) and ranks
failing students above passing ones with ROC-AUC 0.967. It is statistically
tied with the SVM and with copying the band of the second-term grade G2, and
significantly better than copying the first-term grade G1 (5x2cv t-test
p = 0.005). The project's conclusion is that on this dataset G2 sets a
ceiling that no model clearly beats. What the network adds is a calibrated
probability for every band.

## 1. Problem

The brief asks for a deep neural network that predicts a student's academic
performance category from demographic, academic, social and behavioural
attributes. The network must be compared with at least one traditional
machine-learning model and served through a Streamlit app.

The final grade G3 (0-20) is grouped into four bands:

| Band | At risk | Average | Good | Excellent |
|---|---|---|---|---|
| G3 | 0-9 | 10-12 | 13-15 | 16-20 |
| Students | 130 (33 %) | 134 (34 %) | 91 (23 %) | 40 (10 %) |

The bands are ordered, so predicting Excellent for an At-risk student is a
worse error than predicting Average. For that reason the main score is
quadratic weighted kappa, which charges an error by the square of its
distance. Exact accuracy, pass/fail accuracy (G3 >= 10) and within-one-band
accuracy are reported alongside it.

## 2. Data

[UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance),
Mathematics course file (`student-mat.csv`): 395 students from two Portuguese
schools, 30 attributes, the period grades G1 and G2, and the target G3. There
are no missing values and no duplicate rows.

Findings from the exploratory analysis (`python -m mathgrades.explore`) that
shaped the design:

- **G3 follows G2 very closely** (Spearman 0.96). Any model that sees G2 starts
  near the top, and the naive rule "final band = band of G2" is the bar to
  beat.
- **38 students have G3 = 0, and all of them have 0 recorded absences.** They
  most likely left the course. They stay in the data as At risk. They also hide the
  effect of absences: the correlation is 0.02 with them and -0.24 without.
- **Background attributes alone carry little signal.** Past failures, wanting
  higher education and mother's education are the strongest of them, and
  their effect is small next to G1 and G2.

## 3. Method

### 3.1 Evaluation protocol

1. **One split:** 296 training and 99 test students (75/25, stratified by
   band). The test set is scored once, at the end.
2. **One pipeline for every model,** refitted inside every cross-validation
   fold: grade features -> significance filter -> MinMax scaling and one-hot
   encoding -> model. No step learns from the students it is scored on. An
   experiment (section 5.1) measures what fitting the filter outside the folds
   would have changed.
3. **Tuning** on 3 folds; **comparison** on repeated stratified 5-fold CV with
   3 repeats (15 scores per model, the same splits for every model).

### 3.2 Features

- **Grade features**, each switched on or off by the tuner: the trend
  `G2_minus_G1`, and the band of G1 and of G2. The band features hand the
  network the band edges, which are hard to learn from 236 students.
- **Feature selection:** an ANOVA F-test for numeric columns and a chi-squared
  test for categorical ones, keeping every column with p below a tuned level.
  At the chosen level (0.001) the final model keeps G2_band, G2, G1, G1_band,
  failures, G2_minus_G1, Medu and age.

### 3.3 GradeNet

| Part | Choice | Reason |
|---|---|---|
| Inputs | 8 encoded columns | Output of the filter above |
| Hidden layers | 32 -> 16 -> 8, each Linear, BatchNorm, ReLU, dropout 0.16 | Tuned; small because there are only 236 training rows per fold |
| Skip path | A linear layer from the inputs straight to the score | Lets the network pass the strong G2 signal through unchanged |
| Output | One shared score and 3 thresholds with ordered biases ("is the band above At risk / above Average / above Good?") | The ordinal approach of Cao, Mirjalili and Raschka (2020): the three answers can never contradict each other |
| Loss | Binary cross-entropy over the three thresholds, label smoothing 0.1 | Matches the ordinal output |
| Optimiser | AdamW, learning rate 0.0129, weight decay 2.9e-6, batch 32, ReduceLROnPlateau | Tuned |
| Stopping | Early stopping on a held-back stop set (stopped after 76 epochs) | Prevents overfitting |
| Calibration | Temperature scaling on the stop set (T = 0.39) | Makes the band probabilities trustworthy |
| Size | 1,083 trainable parameters | |

### 3.4 Comparison models

All are tuned with Optuna (TPE sampler, 40 trials each) on the same folds and
with the same objective as GradeNet:

- **SVM:** kernel, C, gamma, degree. Chosen: linear kernel, C = 0.81.
- **Histogram gradient boosting:** learning rate, depth, leaves, iterations,
  L2. Chosen: 412 trees, depth 8, 13 leaves, learning rate 0.013.
- **Dummy classifiers:** most frequent band, and random by class share.
- **Naive rules:** copy the band of G2; copy the band of G1.

### 3.5 Statistics

- 95 % bootstrap confidence intervals on the test set (2,000 resamples).
- McNemar's test on the test set (which students one model gets right and the
  other wrong).
- Dietterich's 5x2cv paired t-test on the training data.

## 4. Results

### 4.1 Cross-validation (5 folds x 3 repeats, 296 students)

| Model | Weighted kappa | Exact band | Pass/fail | Within one |
|---|---|---|---|---|
| **GradeNet** | **0.915 ± 0.016** | **83.6 %** | **91.4 %** | **100 %** |
| SVM (tuned) | 0.913 ± 0.021 | 83.1 % | 92.1 % | 99.9 % |
| Gradient boosting (tuned) | 0.910 ± 0.013 | 81.8 % | 92.0 % | 100 % |
| Naive: band of G2 | 0.918 ± 0.015 | 84.1 % | 92.2 % | 100 % |
| Naive: band of G1 | 0.825 ± 0.029 | 65.9 % | 82.8 % | 100 % |
| Dummy: most frequent | 0.000 | 33.8 % | | 89.9 % |
| Dummy: stratified | 0.055 ± 0.092 | 28.6 % | | 74.0 % |

### 4.2 Test set (99 students, used once)

| Model | Kappa (95 % CI) | Exact (95 % CI) | Pass/fail | At-risk recall | At-risk AUC |
|---|---|---|---|---|---|
| **GradeNet** | **0.884 (0.825-0.930)** | **0.788 (0.707-0.869)** | **0.879** | **0.750** | **0.967** |
| SVM | 0.894 (0.843-0.936) | 0.798 (0.717-0.879) | 0.909 | 0.875 | - |
| Gradient boosting | 0.894 (0.841-0.936) | 0.798 (0.717-0.869) | 0.919 | 0.844 | 0.972 |
| Naive: band of G2 | 0.910 (0.859-0.952) | 0.838 (0.768-0.909) | 0.909 | 0.875 | - |
| Naive: band of G1 | 0.837 (0.768-0.891) | 0.687 (0.596-0.778) | 0.869 | 0.781 | - |
| Dummy: most frequent | 0.000 | 0.343 (0.253-0.444) | 0.677 | 0.000 | 0.500 |
| Dummy: stratified | -0.061 (-0.241-0.123) | 0.313 (0.222-0.404) | 0.586 | 0.375 | 0.531 |

Every learned model and both naive rules place every test student within one
band of the truth. A dash means the model gives a band but no probability, so
it has no AUC.

### 4.3 Is GradeNet better? Paired tests

| GradeNet vs | Test: right only for GradeNet / only for the other | McNemar p | 5x2cv t-test p | Verdict |
|---|---|---|---|---|
| SVM | 4 / 5 | 1.00 | 0.84 | Tied |
| Naive: band of G2 | 2 / 7 | 0.18 | 0.28 | Tied |
| Naive: band of G1 | 18 / 8 | 0.076 | **0.005** | GradeNet better (significant in CV) |

### 4.4 What the model relies on

Permutation importance on the test set (drop in kappa when a column is
shuffled): G2 0.69, G1 0.02, every other column about 0. Once G2 is known,
the background columns add almost nothing to the prediction.

## 5. Experiments (`python -m mathgrades.experiments`)

### 5.1 Leakage check

Same model, same folds. Feature selection fitted inside each fold gives kappa
0.921; fitted once on all training students before the split gives 0.914. On
this data the leak made no measurable difference, but only the inside-the-fold
version is a correct estimate. An earlier softmax version of this project
scored 0.874 (selection outside the folds, 3-fold CV). The improvement comes
from the ordinal head, the band features and the stricter filter, not from
leakage.

### 5.2 Ablation (change in CV kappa from the full model, 0.919)

| Change | CV kappa | Change |
|---|---|---|
| No feature selection | 0.848 | -0.071 |
| Band-of-term-grade features removed | 0.896 | -0.023 |
| Softmax head instead of ordinal | 0.907 | -0.012 |
| No dropout | 0.908 | -0.011 |
| Seed ensemble of 5 (not used) | 0.914 | -0.005 |
| One hidden layer instead of 3 | 0.915 | -0.004 |
| No linear skip path | 0.915 | -0.004 |
| No BatchNorm | 0.917 | -0.003 |
| Label smoothing off | 0.917 | -0.002 |
| Class weights on | 0.918 | -0.002 |
| G2 - G1 trend removed | 0.918 | -0.001 |

Feature selection matters most, because irrelevant columns let a small-data
network fit noise. The band features and the ordinal head come next. Changes
under about 0.005 are within the fold-to-fold noise.

### 5.3 Grade setups (after Cortez and Silva, 2008; CV kappa)

| Model | G1 and G2 | G1 only | No grades |
|---|---|---|---|
| GradeNet | 0.919 | 0.810 | 0.209 |
| Best baseline (SVM) | 0.911 | 0.787 | 0.220 |
| Naive: band of latest grade | 0.918 | 0.825 | - |

The earlier the prediction, the harder it is. Without any period grade, no
model gets far.

### 5.4 Learning curve

GradeNet is flat at kappa 0.91-0.92 from 93 training students upward. The SVM
keeps rising, to 0.921 at 197 students. More data would help the SVM more
than this network (`reports/figures/learning_curve.png`).

### 5.5 Calibration (test set)

| | Brier score | Expected calibration error | Log loss |
|---|---|---|---|
| Before (T = 1) | 0.381 | 0.217 | 0.671 |
| After temperature scaling (T = 0.39) | 0.309 | 0.090 | 0.501 |

The raw network was under-confident. Label smoothing pulls probabilities
towards the middle, and T < 1 sharpens them back.

### 5.6 Fairness (test set)

Kappa is the same for girls and boys (0.882), but the model under-predicts
At risk for girls: it predicts 32 % At risk against 40 % actual (boys: 24 %
against 24 %). Urban students: 29 % predicted against 34 % actual; rural:
25 % against 25 %. The second school (MS) has only 7 test students, too few to
judge. A school using this model should check the At-risk list for girls by
hand rather than trust it alone.

### 5.7 Error analysis

All 21 test mistakes are exactly one band off. The most confident mistakes are
students whose final grade crossed a band edge that G2 did not: a G2 of 8 or 9
followed by a final 10, a G2 of 10 followed by a final 9, a G2 of 12 followed
by a final 13. These students are close to an edge by construction, and
nothing in the other columns announces the crossing
(`reports/worst_mistakes.csv`).

## 6. Discussion

**Why the network does not beat copying G2.** In Maths the final grade is
almost a copy of the second-term grade. About 84 % of students end in the
same band as their G2 (the naive rule's CV accuracy). The other 16 % cross a band edge for reasons the
dataset does not record (exam-day performance, effort in the final term). No
model in this project, neural or classical, separates those students
reliably. The same ceiling appears in the dataset's original paper.

**What the network adds anyway:**
- a probability for every band and a calibrated chance of passing, which the
  copying rule cannot give;
- a ranking of students by risk (AUC 0.967), so a teacher can work down a list;
- a significant gain over copying the first-term grade (p = 0.005), which is
  what a school would have at an earlier point in the year.

**Limitations:**
- 99 test students give confidence intervals about ±0.05 kappa wide.
- The best tuning score (0.921) is above the repeated-CV score (0.915):
  choosing the best of 40 trials on one split also picks some luck. The
  repeated-CV figure is the one to quote.
- Temperature scaling is fitted on only 60 stop-set students.
- The chi-squared test assumes at least 5 expected students per cell, which
  fails for some rare categories. Those p-values are approximate.
- The app's inputs only offer values seen in training.
- One quirk: in the app, setting past failures to 3 slightly raises the pass
  chance. Once G2 is known, past failures carry almost no information, and the
  small effect the network learned is noise.

## 7. Conclusion

GradeNet meets every part of the brief: preprocessing, EDA, feature
selection, an ordinal deep network, training with early stopping, Optuna
tuning, evaluation with confidence intervals and significance tests, and a
comparison with tuned traditional models and naive rules. It predicts the
exact band for 83.6 % of students in cross-validation, gets pass/fail right
for 91.4 %, and is never more than one band off. The honest finding is that
the period grade G2 already holds almost all of the predictable signal, so
every competent model ties with simply copying its band. The network's value
lies in calibrated probabilities and risk ranking, not in a higher accuracy.

## References

- Cortez, P. and Silva, A. (2008). Using Data Mining to Predict Secondary
  School Student Performance. *Proceedings of 5th FUture BUsiness TEChnology
  Conference*, 5-12.
- Cao, W., Mirjalili, V. and Raschka, S. (2020). Rank consistent ordinal
  regression for neural networks with application to age estimation.
  *Pattern Recognition Letters*, 140, 325-331.
- Dietterich, T. G. (1998). Approximate statistical tests for comparing
  supervised classification learning algorithms. *Neural Computation*, 10(7),
  1895-1923.
- Akiba, T. et al. (2019). Optuna: A Next-generation Hyperparameter
  Optimization Framework. *KDD 2019*.
- Guo, C. et al. (2017). On Calibration of Modern Neural Networks. *ICML 2017*.
