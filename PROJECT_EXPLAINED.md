# SessionSignal: the project explained from start to finish

This is a plain-language guide for our DES423 group project. It follows the assignment questions, explains what we actually built, and gives the numbers behind our claims. Use it to prepare a report or answer questions during the presentation. The main experiment is in [`train.py`](train.py); its saved numbers are in [`artifacts/results.json`](artifacts/results.json).

## The project in one minute

An online store receives many visits, but most do not end in a purchase. Our project, **SessionSignal**, learns from past browsing sessions and gives each completed session a score for how likely it was to end in a purchase. An ecommerce analyst could use the ranking to decide which unsuccessful sessions to investigate first. The present app demonstrates this on historical data; it does **not** yet predict during a live visit or prove that an intervention increases sales.

This is **supervised binary classification**. “Supervised” means the training examples already have the correct answer. “Binary” means there are two possible outcomes: purchase (`Revenue = TRUE`) or no purchase (`Revenue = FALSE`). It is not regression (predicting a money amount), time-series forecasting, or deep learning. Our main models use scikit-learn; a separate PyCaret experiment connects the project to the course tutorial.

## 1. What is the dataset, and what do we predict?

We use the [UCI Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset), originally donated in 2018. The file [`online_shoppers_intention.csv`](online_shoppers_intention.csv) has **12,330 rows**, one for each recorded shopping session, and **18 columns**: 17 candidate inputs and the `Revenue` outcome. UCI reports **10,422 no-purchase sessions** and **1,908 purchase sessions** in the original data. It reports no missing values. The records are from a one-year period, with each session assigned to a different user in the dataset.

Our question is: **Given a session's browsing information, can we rank sessions by their likelihood of ending in a purchase?** The outcome is whether a transaction happened, not how much the person spent. A prediction score such as `0.64` means the model assigns a 64% purchase score to that session; it is an estimate from historical patterns, not a guaranteed or independently calibrated probability.

### What each column means

The duration fields are shown in seconds in our app. For the rate fields, a value such as `0.05` corresponds to 5%. The numeric codes for operating system, browser, region, and traffic source are **category IDs**, not amounts or rankings.

| Field | Simple meaning |
|---|---|
| `Administrative` | Number of administrative pages visited. |
| `Administrative_Duration` | Total time spent on administrative pages. |
| `Informational` | Number of information pages visited. |
| `Informational_Duration` | Total time spent on information pages. |
| `ProductRelated` | Number of product pages visited. |
| `ProductRelated_Duration` | Total time spent on product pages. |
| `BounceRates` | Summary of the visited pages' bounce rates. A page “bounces” when a visit starts there and ends without another interaction. |
| `ExitRates` | Summary of the visited pages' exit rates. A page has an “exit” when it is the last page of a visit. |
| `PageValues` | Analytics estimate of the average value attributed to pages visited before a transaction. **It is not this visitor's spending. We exclude it from model inputs.** |
| `SpecialDay` | Closeness to a special shopping day, from 0 to 1. |
| `Month` | Month of the visit. This CSV has ten months; January and April are absent. |
| `OperatingSystems` | Category code for the visitor's operating system. |
| `Browser` | Category code for the visitor's browser. |
| `Region` | Category code for the visitor's geographic region. |
| `TrafficType` | Category code for how the visitor reached the site. |
| `VisitorType` | `New_Visitor`, `Returning_Visitor`, or `Other`. Returning means a previous visit, not necessarily a previous purchase. |
| `Weekend` | `TRUE` if the visit happened on a weekend. |
| `Revenue` | **The answer we predict:** `TRUE` if the session ended in a purchase. |

These definitions follow the [UCI variable notes](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset). The publisher does not provide names for the numbered operating system, browser, region, or traffic categories, so we do not invent them.

## 2. What patterns did we find? Exploratory data analysis (EDA)

**EDA means looking at the data before training a model.** We check its size and quality, count the outcomes, and compare browsing patterns between purchasers and nonpurchasers. Our EDA figures are [`artifacts/eda.png`](artifacts/eda.png) and [`artifacts/eda_detail.png`](artifacts/eda_detail.png). The figures are produced by `train.py`.

| Finding | What the data shows | Why it matters |
|---|---|---|
| Purchases are uncommon | In the original data, 1,908 of 12,330 sessions purchased: **15.5%**. The remaining **84.5%** did not. | A model that always says “no purchase” looks accurate but finds no buyers. |
| Data quality | **No missing values** and **125 exact duplicate rows**. After removing duplicates, we use **12,205 rows**: 1,908 purchases and 10,297 nonpurchases. | Remove repeated identical records before splitting, so an identical row cannot appear in both train and test. |
| Product browsing | Purchasers visited a median of **29 product pages**, versus **16** for nonpurchasers. Their median time on product pages was about **1,110 seconds**, versus **526 seconds**. | Browsing depth and time may help rank purchase intent. These are associations, not causes. |
| Exit behaviour | Average `ExitRates` was about **1.96%** for purchasers and **4.55%** for nonpurchasers. | Sessions associated with frequent exits were less likely to end in purchase in this dataset. |
| Seasonality | Purchase rate was about **25.5% in November** but **1.7% in February** after duplicate removal. | Month may be useful, but a model can learn one store's past seasonality and fail in a later year. February has only 181 sessions. |
| Visitor type | New visitors purchased at about **24.9%**, returning visitors at **14.1%**. | Visitor type contains information here; “returning” does not mean an existing customer. |
| Traffic source | For categories with at least 100 sessions, purchase rates vary; for example, `TrafficType` 8 was about **27.7%** and type 13 about **5.9%**. | The source of visits may matter, but the category codes cannot be translated into channel names from this file. |

The strongest simple numeric correlation with `Revenue` among the plotted browsing fields is `ExitRates`, about **−0.20**. This is a weak-to-moderate association, not proof that changing exit rate will cause purchases. The EDA compares groups and helps us choose questions; the held-out test set is what checks prediction on unseen rows.

## 3. What real-world problem are we trying to solve?

An ecommerce team cannot manually inspect every visit that ends without a purchase. In this source dataset, **10,422 of 12,330 sessions** ended without buying. If an analyst has time to review only a small number of sessions, a ranking could help them focus on visits that *looked* promising but did not convert.

There is also broader evidence that lost purchases are common: the [Baymard Institute's cart abandonment review](https://baymard.com/blog/cart-abandonment-rate) reports a **70.22% average** across 50 published cart-abandonment studies. **Cart abandonment is different from our dataset's no-purchase sessions**: we do not know which UCI sessions had a cart. We use Baymard only to show the general business context, not as a measurement of our model's impact.

Our proposed user is a **conversion analyst**. The practical question is: *Which completed no-purchase sessions should the analyst examine first for possible checkout, navigation, pricing, or support problems?* The current CSV does not record why visitors left. A high score is a lead for investigation, not an explanation or proof of recoverable revenue.

## 4. How did we prepare the data and measure the models?

1. Read the CSV and remove **125 exact duplicate rows**. This leaves **12,205** sessions.
2. Set `Revenue` as the answer (`TRUE = 1`, `FALSE = 0`). Exclude `PageValues` from the inputs because it is an analytics measure derived from transaction value and may give the model information too close to the outcome. We therefore train with **16 input fields**.
3. Make a **stratified 80/20 split**, preserving roughly the same purchase fraction in each part: **9,764 training rows** and **2,441 test rows**. `random_state = 42` makes this split reproducible.
4. Within the training part, use **four-fold stratified cross-validation**. In each round, fit on three folds and score on the fourth; average the four results. This guides model-family choice without using the held-out test rows for training.
5. Preprocess inside a scikit-learn pipeline. Numeric fields get median imputation and standard scaling. Categorical fields get most-frequent imputation and one-hot encoding, so a code like browser `3` is treated as a category rather than “three times” browser `1`. The dataset has no missing values, but the pipeline can handle them at prediction time.
6. Tune the best model family by training-fold **average precision**. Refit the candidates on the training rows and score them on the **same 2,441 held-out test sessions**. This makes the main comparison fair. The final model is selected by training cross-validation, not by looking for the highest test score.

**What “held out” means:** the model never sees the test outcomes while fitting. The test set acts like a final exam. If we repeatedly make choices after looking at its scores, it stops being a clean final exam. Some extra analyses below (threshold choice, importance, and ablation) use the test set and should be treated as exploratory.

### Our evaluation measures, in simple terms

| Measure | Simple explanation | Why we use it |
|---|---|---|
| **Average precision (AP), primary** | Summarizes the precision–recall curve across score cutoffs. It rewards ranking actual purchasers near the top. Higher is better. | Purchasers are only about 15.6% after cleaning. A random or constant ranking gets AP near that prevalence. |
| **ROC AUC** | Chance that a randomly selected purchaser receives a higher score than a randomly selected nonpurchaser. `0.5` is chance ranking; `1.0` is perfect. | Gives a second view of ranking across all cutoffs. |
| **Precision at a cutoff** | Of sessions predicted “purchase,” what fraction actually purchased? | Useful when false alarms consume analyst time. |
| **Recall at a cutoff** | Of all actual purchasers, what fraction did the model identify? | Useful when missing purchasers matters. |
| **F1 at a cutoff** | A combined score for precision and recall. | Helps discuss the trade-off when one cutoff is required. |
| **Accuracy** | Fraction of all yes/no labels that are correct. | We show it for context, but it is misleading by itself with many nonpurchasers. |
| **Lift in the top 10%** | Purchase rate among the highest-scored 10% divided by the overall purchase rate. | Matches the idea of limited review capacity. |

For precision, recall, F1, and accuracy in the comparison table, we label a session “purchase” when its model score is at least **0.5**. AP and ROC AUC use the full ranking, so they do not depend on that cutoff. The no-purchase baseline gets **84.4% test accuracy** simply because most sessions do not buy, but its **recall is 0%**: it finds no purchasers.

## 5. Which models did we compare, and how do they work?

All five rows below use the **same cleaned data, features, train/test split, and test sessions**. The fifth row is the tuned version of one of the three real model families.

| Model | How it works, simply | Test AP ↑ | Test ROC AUC ↑ | Precision / Recall / F1 at 0.5 |
|---|---|---:|---:|---|
| Always no purchase | A dummy reference that uses the training purchase rate as its score and predicts “no purchase” at 0.5. | 0.156 | 0.500 | 0% / 0% / 0% |
| Logistic regression | Learns a weight for each input, adds the weighted values, and converts the result into a purchase score. `class_weight="balanced"` gives more weight to the rarer purchase class during fitting. | 0.331 | 0.756 | 27.9% / 78.3% / 0.411 |
| Random forest | Trains **200 decision trees**. Each tree follows simple rules and sees a random selection of inputs at its splits; the forest averages their purchase votes. | 0.372 | 0.786 | 39.9% / 34.6% / 0.370 |
| Histogram gradient boosting | Builds small trees **one after another**. Each new tree aims to reduce mistakes left by the current combined model. | 0.370 | 0.792 | 45.8% / 7.1% / 0.122 |
| **Tuned random forest** | The same forest idea, after trying different tree depths and minimum leaf sizes on the training folds. The selected settings are unlimited depth and at least five training samples per leaf. | **0.374** | 0.787 | 37.1% / 46.3% / 0.412 |

The tuned forest had the highest **training cross-validation AP** (0.391), so it is the model saved for the app. Its held-out AP is **0.374**. Gradient boosting has a slightly higher **test ROC AUC** (0.792 versus 0.787), which is possible because AUC and AP measure ranking differently. At a 0.5 cutoff, boosting predicts very few purchases, so its recall is low even though its ranking AUC is good.

The tuned forest's 0.5-cutoff confusion matrix on 2,441 test sessions is:

| Actual outcome | Predicted no purchase | Predicted purchase |
|---|---:|---:|
| No purchase | 1,759 | 300 |
| Purchase | 205 | 177 |

In words, it caught **177 of 382** actual purchasers and missed **205**. Of the **477** sessions it labeled as purchases, **177** really bought. These counts explain its 37.1% precision and 46.3% recall.

### How confident are we that the tuned forest is best?

The tuned forest's AP is only about **0.002** above the untuned forest on this test set. We repeatedly resampled the **same paired test sessions** 2,000 times and recalculated the AP difference. The mean difference was about **+0.0028**, with a 95% bootstrap interval of **−0.008 to +0.014**. Because that interval includes zero, the data do **not** give convincing evidence that tuning made the forest better than the untuned version. The practical conclusion is that the real models improve substantially over the dummy baseline, while their relative differences are small and uncertain. This bootstrap check is exploratory because the comparison to the runner-up was identified after inspecting test results.

### How does PyCaret fit in?

[`pycaret_comparison.py`](pycaret_comparison.py) runs PyCaret's classification workflow: `setup` → `compare_models` → `tune_model` → `predict_model`. It removes duplicates and `PageValues` as the main experiment does, enables imbalance handling, and compares **14 classifiers**. Its cross-validation AUC leaderboard starts with LightGBM **0.7732**, gradient boosting **0.7641**, and random forest **0.7495**. This is a useful wider search. **PyCaret makes its own split and preprocessing choices**, and sorts this sweep by AUC, so do not put these numbers in the same head-to-head table as the scikit-learn test AP values. See [`artifacts/pycaret_leaderboard.csv`](artifacts/pycaret_leaderboard.csv).

## 6. How does the model help solve the problem?

The saved forest scores each completed session from low to high purchase intent. The [Streamlit app](app.py) has a **Score a session** view for an individual held-out example and a **Find lost sales** view that ranks all **2,441 held-out sessions**. The latter can show high-scoring sessions whose recorded outcome was no purchase. An analyst could open those sessions first, then check fuller journey or support records to form a hypothesis about friction.

For example, among the **top 245 scored test sessions** (about 10% of the test set), **107 purchased**: a **43.7% purchase rate**, compared with **15.6%** in the full test set, or about **2.79× lift**. Among the **top 50**, **25 purchased**, compared with about **8 expected** from a random set of 50. The other 25 top-50 sessions did **not** purchase and are candidates for investigation. These are retrospective ranking results, not evidence that we recovered 25 sales.

An eventual business workflow would be: **score completed sessions → prioritize a manageable worklist → investigate customer paths and support evidence → design a fix → run an A/B test → measure whether conversion actually changes**. The current dataset provides the first step and a demo of prioritization. It does not contain cart contents, session replay, reasons for leaving, intervention costs, or revenue amounts.

## 7. How could we improve the project?

| Next step | Why it helps | How to check it fairly |
|---|---|---|
| **Choose a cutoff for the real review capacity** | A 0.5 cutoff is arbitrary. On this test set, an exploratory cutoff of **0.446** raises recall from **46.3% to 62.3%** and F1 from **0.412 to 0.444**, while reducing precision from 37.1% to 34.5%. A cutoff near **0.607** flags the top 10% (245 sessions). | Choose the cutoff on a **separate validation set**, then report its performance once on a new untouched test set. The current cutoff was chosen on the test set, so its gain is optimistic. |
| **Test on later dates and another store** | `Month` is the strongest feature in our permutation check: shuffling it lowered test AP by about **0.060**. The model may depend on seasonality specific to this one dataset. | Train on earlier periods; test on later, unseen periods and other stores. Compare AP, lift, and calibration. |
| **Collect live, time-stamped partial sessions** | Our rows describe completed sessions. A live app must know what a shopper has done **so far**, without using information recorded after the prediction moment. | Define a prediction time, rebuild features using only earlier events, and test on future sessions. |
| **Improve data and feature quality** | The CSV lacks reasons for leaving, cart status, item prices, and checkout events. These may be more actionable than anonymous category IDs. | Add only privacy-appropriate fields available at prediction time; compare with the same validation design and check for leakage. |
| **Check calibration and costs** | A score used as a probability should match real purchase frequencies. Review effort and possible interventions have different costs. | Make a calibration plot and choose a worklist size using measured analyst capacity or expected value, not AP alone. |
| **Test an intervention** | Good ranking does not show that contacting or helping shoppers changes behaviour. | Randomly assign a proposed intervention in an A/B experiment and measure actual incremental conversions and costs. |

We deliberately exclude `PageValues`. A separate experiment keeping it raises test AP from **0.374 to 0.723**, but this feature is derived from transaction value. That higher score may reflect information too close to the answer, so we do **not** treat it as a deployable improvement. Removing it does not by itself prove that every other feature is available early enough for live prediction.

## 8. Main conclusion and important limits

**Conclusion:** On this dataset, browsing behaviour can rank purchasing sessions better than a no-purchase baseline. Our selected tuned random forest achieved **0.374 AP** and **0.787 ROC AUC** on a held-out set without `PageValues`. Its top 10% of sessions had about **2.79×** the overall purchase rate. The margin over the other serious models is small, so we should present it as the **selected demonstration model**, not a proven universal winner.

**Limits:** This is one historical dataset of **completed** sessions. Its associations do not explain why someone left. The app's “lost sales” worklist uses the known outcomes in the test data to show what an analyst might inspect; it does not recover sales automatically. A real-time or causal claim would need new partial-session data, later-period testing, and an experiment on the proposed action.

## Reproducing the evidence

From the project folder, using Python 3.10 or 3.11:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python train.py
.venv/bin/streamlit run app.py
```

`train.py` creates the model, EDA figures, importance chart, scored test sessions, and [`artifacts/results.json`](artifacts/results.json). Running it will overwrite those generated files. The [README](README.md) explains the demo and separate PyCaret run; the [presentation guide](PRESENTATION_GUIDE.md) has the group speaking plan.

### Sources

- [UCI dataset description and citation](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset): Sakar, C. & Kastro, Y. (2018), *Online Shoppers Purchasing Intention Dataset*, DOI [10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q), CC BY 4.0.
- [Baymard Institute, cart abandonment review](https://baymard.com/blog/cart-abandonment-rate): business context, not the source of our session-level outcomes.
- [PyCaret tutorials](https://pycaret.gitbook.io/docs/get-started/tutorials): current classification workflow used as the course-related comparison.
- Project results: [`artifacts/results.json`](artifacts/results.json), produced by [`train.py`](train.py). The EDA values above are calculated from the supplied CSV after the same duplicate removal.
