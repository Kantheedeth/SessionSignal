# SessionSignal: online purchase intent proof of concept

DES423 group assignment — predictive modeling pitch.

| Student ID | Name | Main contribution |
|---|---|---|
| 6622782298 | Sirawich Narakulsookpipat | Problem research and framing |
| 6622780342 | Chaitawat Boonkitticharoen | Dataset audit and target definition |
| 6622782215 | Bhuchi Kumprakob | Exploratory data analysis |
| 6622780276 | Nonthaphat Chuathong | Evaluation design and model selection |
| 6622770657 | Kamin Sirisuwapong | PyCaret comparison, results and interpretation |
| 6622772455 | Poonnawit Wongprateepwilai | Improvement and business solution |
| 6622790267 | Choltawat Jullakasevi | Streamlit app, demonstration and limitations |

This project uses the supplied `online_shoppers_intention.csv` from the [UCI Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset). It predicts whether a completed browsing session ended in a purchase (`Revenue`).

The proposed user is an ecommerce conversion analyst. The business question is which completed sessions deserve investigation when analyst time is limited. A proposed workflow ranks sessions by model score, examines high-score sessions with no recorded purchase, and reviews user paths or support logs where those are available. The analyst can then form a checkout or support hypothesis and test a change in an A/B experiment. The app implements scoring and batch ranking today: the **Find lost sales** tab ranks all 2,441 held-out sessions and returns the high-scoring ones that did not buy. Investigation tooling and the A/B experiment remain next steps, and would need journey or support logs the dataset does not include. The model does not establish causes of abandonment or a revenue increase.

## Run the demonstration

Use Python 3.10 or 3.11. From this folder:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python train.py
.venv/bin/streamlit run app.py
```

Open the local URL printed by Streamlit. The app has five tabs:

- **Score a session** — pick a held-out session, see what that visitor did, and get an estimated chance of buying. Every control is live. The app labels the outcome automatically: *caught a buyer*, *correctly set aside*, or **worth investigating** when a high-scoring session did not convert.
- **Find lost sales** — ranks all 2,441 held-out sessions. Set how many your team can review and it returns the worklist: the high-intent sessions that bought nothing. Reviewing the top 50 surfaces 25 real buyers against 8 expected at random.
- **How it works** — the modelling steps, what the model looks at, and why accuracy is the wrong measure.
- **Results** — model comparison with bootstrap intervals, feature importance, the threshold study, the PageValues ablation, PyCaret's 14-model sweep, and the EDA charts.
- **About** — the problem with sources, the dataset, the team, and the honest limitations.

The theme is pinned in `.streamlit/config.toml` so the app renders identically on any machine, which matters when recording.

For the demonstration use examples **27** (score 68%, bought), **9** (1%, did not buy) and **8** (64%, did **not** buy — the session an analyst should investigate).

## Presentation files

| File | What it is |
|---|---|
| [presentation.html](presentation.html) | The 17-slide deck. Open in a browser, arrow keys to advance. |
| [SessionSignal.pptx](SessionSignal.pptx) | The same deck as editable PowerPoint, for import into Canva. |
| [SessionSignal_slides.pdf](SessionSignal_slides.pdf) | Flattened slides, as a presenting backup. |
| [SessionSignal_Presentation_Script.pdf](SessionSignal_Presentation_Script.pdf) | Word-for-word speaking script: every line to say, every stage direction, timed per speaker. |
| [SessionSignal_Project_Summary.pdf](SessionSignal_Project_Summary.pdf) | Plain-language summary of the whole project. |
| [PRESENTATION_GUIDE.md](PRESENTATION_GUIDE.md) | Run of show, speaker allocation, Thai notes, demo script, recording steps. |

The talk is **10–15 minutes at 1× speed** with the demonstration included; the run of show is timed to **14:00**. The notebook asks for a YouTube video and the class message says present live in the last session — the deck works for either, and [PRESENTATION_GUIDE.md](PRESENTATION_GUIDE.md) has a checklist for both.

The already trained model and results are in `artifacts/`. You can run the app without retraining after installing dependencies. Results are deterministic with seed 42, subject to library and platform versions.

## Method and results

- 12,330 source sessions, 18 columns including target; 125 exact duplicate rows removed.
- `PageValues` is excluded because it is a revenue-derived analytics measure. This leaves 16 input features.
- Stratified 80/20 train/test split: 9,764 training rows and 2,441 held-out test rows.
- Four-fold training cross-validation compares a no-purchase baseline, weighted logistic regression, random forest and histogram gradient boosting. The strongest family by training-fold average precision is tuned with a small grid search. The test set is evaluated once per candidate.
- The tuned random forest is selected by training-fold average precision (0.391). On the test set it scores **0.374 average precision**, **0.787 ROC AUC**, **37.1% precision** and **46.3% recall** at a 0.5 threshold. Its top-scored 10% of sessions has a **43.7% purchase rate**, compared with **15.6% overall**.
- The forest tuning grid is `max_depth=[8, 16, None]` and `min_samples_leaf=[2, 5]`. The selected values are `None` and `5`. Training-fold AP improves from 0.386 to 0.391 and test AP from 0.372 to 0.374: a modest improvement.
- **Model differences are not statistically significant.** A 2,000-sample paired bootstrap puts the tuned forest just **+0.0028** ahead of the runner-up, 95% CI **[−0.008, +0.014]**, winning 68% of resamples. Every real model beats the baseline decisively; between themselves they are indistinguishable, and we say so rather than claiming a winner.
- **Permutation importance** names `Month` as the strongest feature (−0.060 AP when shuffled), ahead of `ProductRelated_Duration` and `ExitRates`. The model has partly learned one store's seasonality, which is why later-period validation is the first next step.
- **The threshold mattered more than the hyperparameters.** Grid search moved AP by +0.002 (noise). Moving the cutoff from 0.5 to **0.446** lifted recall from **46.3% to 62.3%** and F1 from **0.412 to 0.444**, at no training cost. A capacity-based cutoff of 0.607 flags the top 10%.
- **Excluding `PageValues` costs roughly half the measured performance** (AP 0.723 → 0.374). It is computed from revenue, so we treat the higher score as leakage rather than an achievement. Both numbers are reproduced by `train.py`.
- The full comparison and confusion matrices are in [artifacts/results.json](artifacts/results.json). Re-run `train.py` to reproduce the saved model, both EDA figures, the importance chart and the report.

Accuracy is misleading here: predicting no purchase for everyone gives 84.4% test accuracy and zero purchase recall. Average precision measures how well a model ranks rare purchases. A higher score does not imply an intervention will cause sales.

## PyCaret tutorial connection

The original linked workshop uses the removed `pycaret.nlp` module. The current [PyCaret classification tutorial](https://pycaret.gitbook.io/docs/get-started/tutorials) covers this tabular problem. [pycaret_comparison.py](pycaret_comparison.py) runs `setup` → `compare_models` → `tune_model` → `predict_model` on the same cleaned data with the same `PageValues` exclusion, with `fix_imbalance=True` because purchases are only 15.6% of rows.

It compares **all 14 classifiers PyCaret ships with**, not a shortlist. Best CV AUC is **0.7732** (LightGBM), then gradient boosting **0.7641** and random forest **0.7495**, with the dummy baseline at **0.5000**. Against our scikit-learn holdout ROC AUC of **0.787**, the two toolchains reach the same conclusion: no model pulls clearly ahead. The saved sweep is [artifacts/pycaret_leaderboard.csv](artifacts/pycaret_leaderboard.csv) and the tuned holdout result is [artifacts/pycaret_tuned_holdout.csv](artifacts/pycaret_tuned_holdout.csv).

**PyCaret lands on exactly the same 2,441 holdout rows as `train.py`** — both pass seed 42 to the same underlying splitter, and we verified the two row sets are identical. What differs is everything after the split: PyCaret keeps the seven categorical columns as native pandas categories (16 columns) where we one-hot them (73), leaves the numeric columns unscaled, balances classes with SMOTE rather than class weights, ranks by AUC rather than average precision, and reports cross-validation rather than a held-out score. Read the two tables alongside each other, not merged. To show them during rehearsal no installation is needed:

```bash
cat artifacts/pycaret_leaderboard.csv
```

To reproduce the sweep, use a separate Python 3.10 or 3.11 environment so its dependencies do not disturb the demo environment:

```bash
python -m pip install 'pycaret==3.3.2' six
python pycaret_comparison.py
```

`six` is required explicitly: PyCaret 3.3.2 imports `scikitplot`, which needs it but does not declare it. On macOS the optional LightGBM import may also need `libomp`.

## Was it unfair to tune only the forest?

`train.py` tunes the strongest family and leaves the rest at their initial settings, which is a reasonable
thing to question. [fair_tuning_check.py](fair_tuning_check.py) removes the asymmetry: every family gets a
grid search under the same protocol, with gradient boosting given the largest grid on purpose.

| Family | Untuned CV AP | Tuned CV AP | Gain | Grid size |
|---|---:|---:|---:|---:|
| Logistic regression | 0.3258 | 0.3281 | +0.0023 | 5 |
| Gradient boosting | 0.3725 | 0.3829 | **+0.0105** | 18 |
| **Random forest** | 0.3860 | **0.3907** | +0.0047 | 6 |

Gradient boosting gained the most from tuning and still came second. The forest also led *before* anyone was
tuned (0.386 vs 0.372), so the selection is not an artefact of the asymmetry. With both tuned, the holdout
gap is +0.0069 with a 95% bootstrap interval of −0.014 to +0.029: still inside the tie.

## Is LightGBM better?

PyCaret's sweep puts LightGBM first by AUC, and people reproducing this project have landed on it, so [lightgbm_check.py](lightgbm_check.py) runs it through the same protocol as `train.py`: same split, same folds, same average-precision metric, same untouched holdout.

| Model | CV AP | Test AP | Test AUC |
|---|---:|---:|---:|
| LightGBM, defaults | 0.3549 | 0.3599 | 0.7804 |
| LightGBM, class-balanced | 0.3641 | 0.3620 | 0.7760 |
| LightGBM, tuned | 0.3738 | 0.3768 | 0.7925 |
| **Ours, tuned forest** | **0.3907** | 0.3739 | 0.7875 |

Selection uses cross-validated AP on training data only, where ours leads 0.391 to 0.374. On the holdout LightGBM is +0.0034 ahead, with a 95% bootstrap interval of −0.020 to +0.026 and a 62% win rate — inside the same tie as every other model here. LightGBM is deliberately not in `requirements.txt`; it needs an OpenMP runtime that is awkward on macOS and the demo must not depend on it.

```bash
.venv/bin/python -m pip install lightgbm
.venv/bin/python lightgbm_check.py
```

## Practical limit

The UCI rows describe completed sessions. The demonstration is a retrospective classifier, not a validated real-time recommender. To intervene while a visitor is still active, collect timestamped partial-session features, retrain and test on a later time period, then run an A/B experiment for any proposed action. `PageValues` was excluded to limit a potential source of target leakage, but this alone does not validate real-time availability of every remaining feature.

## Sources

- **Dataset attribution:** Sakar, C. & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset*. UCI Machine Learning Repository. [DOI: 10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q). The included `online_shoppers_intention.csv` is the original source file; cleaning is performed in code. UCI publishes it under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- [UCI dataset description and citation](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
- [Baymard Institute cart abandonment study list](https://baymard.com/lists/cart-abandonment-rate) — business context only; cart abandonment differs from non-purchase sessions
- [Current PyCaret tutorials](https://pycaret.gitbook.io/docs/get-started/tutorials)
- [PyCaret classification quickstart](https://pycaret.gitbook.io/docs/get-started/quickstart)
- [Original assignment notebook and rubric](https://colab.research.google.com/drive/1pys5EpelJBWDcHZNxpeUcDDybJnLbYTV) — presentation logistics follow the latest 7–8 member, in-session assignment.
