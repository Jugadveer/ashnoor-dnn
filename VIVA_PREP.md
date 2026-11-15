# Viva guide: Maths grade bands with an ordinal PyTorch network

Everything an examiner can ask about this project, in four parts: facts to
know cold, the project stage by stage, practice questions, and the day itself.
Every number comes from `reports/results.json` and `reports/experiments.json`.

| Mark | Where to look |
|---|---|
| P1 Core Architecture | B4 network, B5 training, B6 tuning |
| P2 Problem Solving | B1 framing, B2 data, B3 leakage and features, B7-B9 results and experiments |
| P3 Tech Stack | B10 tools |
| P4 System Design | B10 pipeline, files, app, tests |
| P5 Tech Communication | A1 pitch, part C |

---

# Part A: know these cold

## A1. Pitch

**Short.** "I predict which of four ordered bands a student's final maths
grade falls in: At risk, Average, Good or Excellent. The data is the UCI Maths
file, 395 students. My model is an ordinal neural network in PyTorch: three
hidden layers and an output that asks 'is the band above At risk, above
Average, above Good?' with ordered thresholds. Feature selection, scaling and
encoding run inside every cross-validation fold, so nothing leaks. I tuned the
network, an SVM and gradient boosting with Optuna on the same folds, and
compared them with dummy and naive baselines using repeated cross-validation,
bootstrap intervals, McNemar's test and the 5x2cv t-test. The network gets
91 % pass/fail accuracy, 84 % exact band and 100 % within one band in
cross-validation, with kappa 0.915. It ties the tuned models and the rule
'copy the band of the second-term grade', and beats copying the first-term
grade. What it adds over copying is a calibrated chance of passing."

**Long** adds: the zero-grade dropouts, the band features, the ablation
(feature selection matters most), calibration (ECE 0.22 -> 0.09) and the
fairness gap for girls.

## A2. Numbers

| Item | Value |
|---|---|
| Students / columns | 395 / 33 (30 attributes + G1 + G2 + G3) |
| Bands (students) | At risk 0-9 (130), Average 10-12 (134), Good 13-15 (91), Excellent 16-20 (40) |
| Train / test | 296 / 99, stratified, seed 7 |
| Zero final grades | 38, all with 0 absences (probable dropouts) |
| Kept inputs | G2_band, G2, G1, G1_band, failures, G2_minus_G1, Medu, age (8; from 5 raw columns) |
| Network | 8 -> 32 -> 16 -> 8 -> score -> 3 thresholds; 1,083 parameters |
| Dropout / LR / weight decay / batch | 0.16 / 0.0129 / 2.9e-6 / 32 |
| Label smoothing / class weights | 0.1 / off |
| Epochs (final) / temperature | 76 / 0.39 |
| Optuna | 40 trials per family, 3-fold CV, TPE seed 7 |
| CV (5x3) kappa | GradeNet 0.915 ± 0.016, G2 rule 0.918, SVM 0.913, boosting 0.910, G1 rule 0.825 |
| CV pass/fail / exact / within one | 0.914 / 0.836 / 1.000 |
| Test kappa (95 % CI) | 0.884 (0.825-0.930) |
| Test exact / pass-fail / within one | 0.788 / 0.879 / 1.000 |
| At-risk ROC-AUC (test) | 0.967 |
| vs G2 rule | McNemar p = 0.18, 5x2cv p = 0.28 (tie) |
| vs G1 rule | McNemar p = 0.076, 5x2cv p = 0.005 (better in CV) |
| vs SVM | McNemar p = 1.00, 5x2cv p = 0.84 (tie) |
| ECE before / after temperature | 0.217 / 0.090 |
| Tests | 12 pytest tests, about 15 s |
| Run times | train 674 s and 829 s on two identical runs (CPU shared), experiments 472 s |

---

# Part B: the project, stage by stage

## B1. Framing

- 4 ordered bands from G3. 10 is the pass mark on Portugal's 0-20 scale, so
  "At risk" is exactly the failing range; three passing levels follow.
  `band_starts = (10, 13, 16)`; `np.digitize` counts the band starts a grade
  has reached (8 -> 0, 11 -> 1, 14 -> 2, 17 -> 3).
- **Ordinal**, not just multi-class: the bands have an order, so the model
  (B4) and the main metric (B7) use it.
- Three levels of detail are reported: 4 bands (kappa, exact accuracy),
  within one band, and pass/fail. They answer different teacher questions.

## B2. Data

- UCI Student Performance (Cortez and Silva, 2008), Maths file, 395 students
  from two Portuguese schools (GP 349, MS 46), 208 female and 187 male, age 15-22.
- Groups from the brief (see `attributes.py`): Demographic 13, Academic 10,
  Social 5, Behavioural 4.
- **Zero grades:** 38 students have G3 = 0 and 0 absences. Probably they left.
  They make absences look unrelated (Spearman 0.02; -0.24 without them).
- EDA: Spearman with G3: G2 0.96, G1 0.88, failures -0.36, Medu 0.23.
  Mean G3 by failures 11.25 / 8.12 / 6.24 / 5.69. School support goes with
  lower grades (9.43 vs 10.56) because struggling students get it.

## B3. Leakage, features, preprocessing

**The pipeline** (`preprocessing.py`), the same for every model:
`GradeFeatures -> SignificanceFilter -> ColumnTransformer -> model`.
scikit-learn clones and refits it on every fold. So in cross-validation the
columns kept, the MinMax ranges and the one-hot categories are learnt from the
fold's training rows only. That is what "no leakage" means here.

**Leakage check** (experiments): same model, same folds. Selection inside the
folds 0.921, fitted once on all training rows 0.914. No measurable inflation
on this data, but inside is the only correct way. The first version's CV of
0.874 came from a different model (softmax, no band features), not from leakage.

**GradeFeatures** (Optuna switches them on or off):
- `G2_minus_G1`: improving or slipping.
- `G1_band`, `G2_band`: the term grade put through the band edges. Why: the
  band edges are sharp steps; learning four steps from a smooth 0-1 input with
  236 students is hard. Removing them costs 0.023 kappa (ablation).

**SignificanceFilter** (`selection.py`): ANOVA F-test for numeric columns
(between-band variance / within-band variance), chi-squared test of
independence for text columns (sum of (observed - expected)^2 / expected).
Keep p < level; level tuned in {0.001, 0.01, 0.05}; the network uses 0.001.
Without the filter kappa drops by 0.071: the largest effect in the ablation.

**Encoding:** MinMaxScaler `(x - min) / (max - min)` for numbers, one-hot for
text, unseen categories -> all zeros.

## B4. The network

```
8 inputs
Linear 8->32   288   BatchNorm 64   ReLU   Dropout 0.16
Linear 32->16  528   BatchNorm 32   ReLU   Dropout 0.16
Linear 16->8   136   BatchNorm 16   ReLU   Dropout 0.16
score = w . hidden (8) + v . inputs (8, skip path)
threshold logits = score + b_k, k = 1..3  (3 biases)
total 1,083
```

- **Ordinal head (CORAL-style; Cao, Mirjalili and Raschka, 2020).** One
  shared score per student, three biases. P(band > k) = sigmoid(score + b_k).
  Biases built as b1, b1 - softplus(d1), b1 - softplus(d1) - softplus(d2):
  softplus is always positive, so b1 > b2 > b3 always, so
  P(> At risk) >= P(> Average) >= P(> Good): no contradictions (rank consistency).
- **Band probabilities:** P(At risk) = 1 - P(>0); P(Average) = P(>0) - P(>1);
  P(Good) = P(>1) - P(>2); P(Excellent) = P(>2). Never negative.
- **Prediction:** the number of thresholds with P > 0.5 = the median band of the
  predicted distribution. It can differ from the single most likely band;
  the median is the best guess when distance matters.
- **Linear skip path:** lets the network copy a straight-line rule (final
  grade follows G2) directly; hidden layers learn corrections.
- **BatchNorm:** normalises each neuron over the batch, then learnt scale and
  shift (2 parameters per neuron); running averages used at prediction time.
- **ReLU** max(0, x); **Dropout** 0.16, off at prediction time.
- **Why 3 hidden layers?** The brief asks for a deep network; Optuna searched
  2-3 layers. The ablation shows one layer costs only 0.004: depth adds little
  on 296 students, and I say so.
- **Softmax version** (old): 4 outputs, cross-entropy. Ablation: 0.907 vs 0.919.

## B5. Training (`training.py`)

- `GradeNetClassifier` wraps the network as a scikit-learn estimator (fit,
  predict, predict_proba), so it fits in the same pipeline, CV, Optuna and
  permutation importance as every other model.
- **Loss:** binary cross-entropy over the three threshold questions. A Good
  student has targets [1, 1, 0]. Label smoothing 0.1 moves targets to 0.95 / 0.05.
- **AdamW** (Loshchilov and Hutter, 2019): Adam (Kingma and Ba, 2015) with weight
  decay applied directly to the weights rather than through the gradient.
- **Stop set:** 20 % of training rows (60 of 296). ReduceLROnPlateau halves the
  learning rate after 8 epochs without improvement; early stopping after 20;
  best weights restored. Final run: 76 epochs.
- `drop_last=True`: BatchNorm cannot normalise a batch of one student.
- **Temperature scaling** (Guo et al., 2017): divide logits by T, choose T to
  minimise log loss on the stop set. T = 0.39 < 1 means the raw model was
  under-confident, probably because label smoothing trains it never to say
  0 or 1. T never changes the predicted band.
- **Determinism:** seeds for weights and batch order, deterministic algorithms;
  two full runs gave identical results.

## B6. Tuning (`search.py`, `classical.py`)

- Optuna (Akiba et al., 2019), TPE: 10 random trials, then proposals that
  resemble the best trials. 40 trials per family; objective = mean quadratic
  weighted kappa on 3 stratified folds; the whole pipeline is refitted per fold.
- Every family also tunes the filter level and the two grade features: fair.
- Spaces: network (2-3 layers, width 16/32/64 halving, dropout 0-0.4, LR
  1e-3 to 5e-2 log, weight decay 1e-6 to 1e-2 log, batch 16/32/64, class
  weights, smoothing 0/0.05/0.1); SVM (linear/rbf/poly, C 0.01-1000 log,
  gamma, degree); boosting (learning rate, depth 2-8, leaves 4-63,
  iterations 50-500, L2).
- Chosen: network above; SVM linear kernel C = 0.81; boosting 412 trees,
  depth 8, 13 leaves, learning rate 0.013, L2 6.2.
- **Optimism of tuning:** best tuning score 0.921, repeated-CV 0.915. Picking
  the best of 40 on one split also picks luck. Trust the repeated CV.

## B7. Metrics and statistics

- **Quadratic weighted kappa** (Cohen, 1968): cost of predicting band j for
  true band i is (i - j)^2 / 9. kappa = 1 - observed cost / chance cost.
  1 perfect, 0 chance. Two bands off costs 4 times one band off.
- **Within one band**, **pass/fail accuracy**, **at-risk recall**, **at-risk
  ROC-AUC** (chance a random failing student gets a higher P(At risk) than a
  random passing one).
- **Repeated stratified 5-fold CV x 3:** 15 scores per model on the same
  splits; more stable than one 99-student test set.
- **Bootstrap 95 % intervals** (Efron and Tibshirani, 1993): 2,000 test sets
  drawn with replacement; 2.5th and 97.5th percentiles.
- **McNemar (1947):** only students where exactly one model is right count;
  under "equally good" each is a coin flip; exact binomial test.
- **5x2cv t-test (Dietterich, 1998):** 5 repetitions of 2-fold CV; t uses the
  first difference over the average per-repetition variance; 5 degrees of
  freedom; designed to keep false alarms near 5 %.

## B8. Results and how to defend them

| Model | CV kappa | Test kappa (CI) | Test exact | Pass/fail | At-risk recall |
|---|---|---|---|---|---|
| GradeNet | 0.915 | 0.884 (0.825-0.930) | 0.788 | 0.879 | 0.750 |
| SVM | 0.913 | 0.894 (0.843-0.936) | 0.798 | 0.909 | 0.875 |
| Gradient boosting | 0.910 | 0.894 (0.841-0.936) | 0.798 | 0.919 | 0.844 |
| Naive band of G2 | 0.918 | 0.910 (0.859-0.952) | 0.838 | 0.909 | 0.875 |
| Naive band of G1 | 0.825 | 0.837 (0.768-0.891) | 0.687 | 0.869 | 0.781 |
| Dummy most frequent | 0.000 | 0.000 | 0.343 | 0.677 | 0.000 |

- **Tie with copying G2**, said plainly: in Maths G3 follows G2 almost exactly,
  so copying G2 is close to the ceiling for exact bands. Permutation importance
  agrees: shuffling G2 drops test kappa by 0.69; G1 by 0.02; nothing else matters.
- **Better than copying G1** in CV (5x2cv p = 0.005).
- **Added value:** a calibrated probability per band and a pass chance (AUC 0.967).
- **Lower at-risk recall on test (0.75 vs 0.875)**: the network calls some
  borderline students Average; the confident errors in B9 are exactly these.
- Every model is within one band for 100 % of test students.

## B9. Experiments

- **Ablation** (3x3 CV; full model 0.919): no feature selection -0.071; band
  features removed -0.023; softmax head -0.012; no dropout -0.011; seed
  ensemble of 5 -0.005 (not used); one hidden layer -0.004; no skip path
  -0.004; no BatchNorm -0.003; label smoothing off and class weights on -0.002;
  trend removed -0.001: all within noise.
- **Grade setups** (CV kappa): with G1 and G2: GradeNet 0.919, SVM 0.911,
  copy G2 0.918. G1 only: 0.810, 0.787, copy G1 0.825. No grades: 0.209, 0.220.
- **Learning curve:** GradeNet flat at 0.91-0.92 from 93 students; the SVM
  still rising (0.921 at 197).
- **Calibration:** ECE 0.217 -> 0.090, Brier 0.381 -> 0.309, log loss
  0.671 -> 0.501 after temperature scaling.
- **Fairness:** kappa equal for girls and boys (0.882), but predicted At-risk
  share for girls 32 % vs actual 40 % (boys 24 % vs 24 %): the model misses
  more failing girls. Urban 29 % vs 34 %. MS has 7 test students: too few.
- **Error analysis:** all 21 errors one band off; students crossing a band
  edge between G2 and G3 (G2 8-9 -> G3 10; G2 10 -> 9; G2 12 -> 13); past
  failures push towards At risk.

## B10. System, tools, tests

```
python -m mathgrades.explore      checks + EDA
python -m mathgrades.train        tuning, CV, test, saves models/grade_pipeline.joblib + input_form.json
python -m mathgrades.experiments  ablation, setups, learning curve, calibration, fairness, errors
streamlit run streamlit_app.py    loads the pipeline once (st.cache_resource)
```

- **App:** styled as a page of a squared maths exercise book (theme in
  `.streamlit/config.toml`, squared paper, margin line and red-pen answer in
  `.streamlit/exercise_book.css`, one typeface: STIX Two). Laid out like a
  worked exercise: Given (the inputs, only the 5 raw columns the model reads,
  ranges limited to training values), Answer, Working, What if. Report card: band,
  likely grade range, pass chance, probability chart, "what pushed this
  prediction" (reset each input to the typical value; change in the band's
  probability), what-if table. Results page: headlines with CIs, full table,
  significance in words, permutation importance, ablation, calibration,
  fairness, confusion matrices. Friendly message if the model is missing.
- **Tests** (`pytest`, 12 tests, ~15 s): dataset counts and columns, band
  edges, stratified split, filter and scaler learn from training rows only,
  grade features, ordinal probabilities valid, same seed same result, both
  heads, saved predictor gives 4 probabilities summing to 1, app form matches
  the model's inputs.
- **Tools:** PyTorch 2.15 (Paszke et al., 2019), scikit-learn 1.9 (Pedregosa
  et al., 2011), Optuna 5.0, SciPy 1.17, pandas 3.1, Streamlit 1.67, Altair,
  matplotlib, joblib, pytest.

## B11. Limitations and ethics

- Small data; tuning-fold optimism; a single run varies by about 0.01.
- Ties the copying rule on exact bands; the gain is probabilities.
- Chi-squared expected-count assumption fails for rare categories.
- Under-prediction of At risk for girls: needs a check on more data before any use.
- Calibration fitted on 60 students.
- **Ethics:** inputs include family background (Medu) and age. Use it to
  offer help early, never to judge a student's "employability".

---

# Part C: practice questions

1. Why an ordinal model? The bands are ordered; it uses that (ablation +0.012 over softmax).
2. How does CORAL avoid contradictory answers? One score, ordered biases.
3. How do you get probabilities from three sigmoids? Differences of cumulative probabilities.
4. Why is your prediction the median band? It is where the cumulative probability crosses 0.5.
5. Count the first layer's parameters. 8 x 32 + 32 = 288; BatchNorm 64.
6. What is the skip path for? Copying a linear rule directly.
7. What is leakage and how did you prevent it? Pipeline refitted per fold; test used once.
8. Did fixing leakage change the score? Inside 0.921 vs outside 0.914: no measurable inflation here.
9. Why band features? Sharp edges are hard to learn from 236 students; -0.023 without.
10. Why does feature selection help so much? Noise columns hurt a small network (-0.071).
11. Why is the network not better than copying G2? G3 follows G2 almost exactly; the ceiling is close.
12. Then why use a network? Calibrated pass probability, AUC 0.967, drivers per student.
13. What do McNemar and 5x2cv test? Paired differences on the same students / same splits.
14. What does the bootstrap interval mean? The range the test score could take with other test students.
15. What is temperature scaling? One number that rescales logits; fitted on the stop set; T = 0.39.
16. Is the model fair? Same kappa by sex, but it misses more failing girls.
17. What do the worst errors have in common? Grades crossing a band edge between G2 and G3.
18. What do your tests check? See B10.
19. Why AdamW? Decoupled weight decay.
20. Why was the seed ensemble dropped? CV showed no gain (-0.005).
21. Why repeated CV and not just the test set? 99 students is noisy; 15 CV scores are steadier.
22. What would you do next? More data; ordinal-aware baselines; check the girls gap; Fisher's exact test.
23. In the app, 3 past failures *raise* the pass chance slightly (95 % -> 97 % for the
    typical student). Why? Once G2 is known, past failures carry almost no extra
    information: permutation importance for failures is -0.012 on the test set
    (shuffling it does not hurt). The small effect the network learnt is noise.
    Say it plainly; it shows the drivers panel exposes what the model really uses.

# Part D: on the day

- [ ] Run `python -m mathgrades.train` and `python -m mathgrades.experiments` at home (about 20 minutes).
- [ ] `python -m pytest` (12 passed).
- [ ] Demo: typical student -> Average, 95 % chance of passing; open Academic and
      raise the second-term grade to show Good, then lower it to 8 to show At risk;
      read the "What pushed this prediction" lines; open Model results and read
      the headline table and the significance sentences.
- [ ] Be ready for question 23 if the examiner plays with past failures.
- [ ] Draw the network and count its 1,083 parameters once on paper.
