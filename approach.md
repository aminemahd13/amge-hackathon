# How We Predict Delivery Time — Explained From Scratch

*Written for someone brand new to machine learning. No prior knowledge assumed.
Every technical word is explained the first time it appears.*

---

## 1. What are we actually trying to do?

The competition gives us information about online shopping orders (who bought what,
from where, to where, how much shipping cost, when it was ordered…) and asks one question:

> **How many days will it take for this package to arrive?**

The catch: we have to guess this **at the moment the order is placed** — before the package
has even been picked up. So we can't use anything that happens after purchase. We only get to
use facts known at checkout time.

This is a **prediction** problem (also called a "regression" problem, because the answer is a
number — days — not a category like "yes/no").

---

## 2. How is our guess scored?

The competition uses **MAE — Mean Absolute Error**. In plain English:

> For each order, take how many days you were *off* by (ignoring whether you guessed too high
> or too low). Average that across all orders.

- If the real delivery was 10 days and you guessed 12, your error is **2 days**.
- If you guessed 7, your error is **3 days**.
- MAE is just the average of all those errors.

**Lower is better. 0 would be a perfect psychic. Our best (locked-in) score is 2.741**, meaning our
guesses are on average within **2.74 days** of reality.

For comparison: if we lazily guessed "11 days" (the typical delivery time) for *every* order,
we'd be off by **6.4 days** on average. So our model is **more than twice as accurate** as the
lazy guess.

---

## 3. What data do we have?

The data is real orders from **Olist**, a big Brazilian online marketplace.

- **Training data**: ~85,000 orders where we **already know** the answer (how many days they took).
  This is what the model learns from. Think of it as 85,000 solved practice questions with the
  answers on the back.
- **Test data**: ~15,000 orders where we **don't** know the answer. We have to predict these.
  This is the real exam. We submit our 15,000 guesses and the competition grades them.

Each order comes with columns like:

| What we know | Example |
|---|---|
| Where the customer is | city + GPS coordinates (latitude/longitude) |
| Where the seller is | city + GPS coordinates |
| The product | category, weight, size, number of photos |
| The money | price, shipping cost ("freight") |
| The timing | exact date & time the order was placed |

---

## 4. The core idea behind any prediction model

A model is basically a **machine that finds patterns**. We show it thousands of examples like:

> "Order from a far-away seller, heavy product, placed in December → took 25 days"
> "Order from a nearby seller, light product, placed in May → took 5 days"

After seeing enough examples, it learns rules of thumb like *"farther away = slower"* and
*"December orders are slower"* — but far more subtle and numerous than a human could write down.

Then for a brand-new order, it applies everything it learned to produce a number.

The two ingredients that make this work well are:
1. **Good features** — giving the model the *right clues* to look at (Section 5).
2. **A good learning algorithm** — the actual pattern-finding machine (Section 6).

---

## 5. Feature Engineering — giving the model good clues

"Features" are the columns of information we feed the model. **Feature engineering** means
*creating new, smarter columns* from the raw data. This is where most of the winning happens —
a great clue beats a fancy algorithm. Here's what we built and *why*:

### 5a. Distance (the single most powerful clue)
The raw data gives GPS coordinates of the seller and the customer, but not the distance between
them. So we **calculate the straight-line distance** (the "haversine" distance — the proper way
to measure distance on a globe). Intuitively: the farther the package travels, the longer it takes.
This turned out to be our strongest single clue.

### 5b. Region & route — the journey, not just the distance
Two orders can both travel 500 km, but one runs along a busy, well-served corridor between major
cities and the other ends in a remote town. So besides raw distance we group every location into a
**region** (using a clustering algorithm that finds ~30 natural geographic zones), and for each
**origin-region → destination-region route** we compute the *historical average delivery time*.
This captures the idea that real logistics is about routes, not just raw kilometers.

> *(We also tested a much finer version — splitting Brazil into small 1°×1° grid squares for a more
> precise "corridor" clue. It looked better in local testing but made the real leaderboard score
> slightly **worse**, so we kept the coarser, more robust region version. See the "Lesson learned"
> box in Section 10.)*

### 5c. Timing & seasonality
Delivery speed changes over time:
- It got **faster** from 2016 → 2018 as Olist matured.
- It **slows down** around **Black Friday and Christmas** (everyone's ordering, warehouses clog up).

So we extract: month, day of week, hour, "is it a weekend", "is it the holiday season", "how many
days until Christmas", "how close to a Brazilian public holiday", and a steady "days since the
start" counter to capture the long-term speed-up trend.

### 5d. Money as a hidden clue
**Shipping cost ("freight")** is secretly very informative — Olist sets it based on distance and
weight, so it encodes route difficulty. We use it directly, plus ratios like *shipping-cost-per-kg*
and *shipping-cost-per-km*.

### 5e. Seller behaviour
Some sellers ship fast, some are slow. Since many test sellers also appear in the training data,
we compute each **seller's historical average delivery time** as a clue. Same idea for product
category and city.

### 5f. Filling in missing information
A few hundred orders are missing GPS coordinates. Instead of throwing them away, we fill the gap
with the **average coordinates of that city** (a reasonable stand-in).

> **The important safety rule:** any clue based on "historical average" (like seller average, or
> corridor average) is computed **without ever letting an order peek at its own answer**. We use a
> technique called *out-of-fold encoding* — explained in Section 8 — so the model can't cheat.

In total the winning model uses **~67 engineered clues** built from the original ~25 raw columns.

---

## 6. The Model — "Gradient Boosting" explained simply

The actual pattern-finding machine we use is called **Gradient Boosting**. Our final model uses two
"brands" of it — **LightGBM** and **XGBoost** (a third, **CatBoost**, exists too; we tried it but it
didn't make our final blend). They're different implementations of the same core idea.

Here's the intuition, no math:

1. Start with a dumb guess for every order (e.g. "11 days for everyone").
2. Look at where that guess was **wrong**, and build a simple **decision tree** that corrects the
   biggest mistakes a little.
   - A *decision tree* is just a flowchart of yes/no questions: *"Is distance > 800 km? → yes →
     Is it December? → yes → add 4 days."*
3. Now we have a slightly better guess. Look at the *remaining* mistakes, build another small tree
   to fix those.
4. Repeat **hundreds of times**. Each tree is weak on its own, but stacked together — each one
   cleaning up the leftover errors of the previous — they become extremely accurate.

That's "boosting": a team of hundreds of tiny flowcharts, each focused on fixing what the others
missed. It's the go-to method for this kind of spreadsheet-style data because it's accurate, fast,
and handles missing values and weird distributions gracefully.

### One important tuning choice: optimizing for *our* score
Because we're judged on MAE (average error), we tell the model to specifically minimize *absolute
error*. We also found that **transforming the target with a logarithm** helps — delivery times are
"lopsided" (most are short, a few are huge 100+ day outliers), and the log transform tames those
outliers so they don't distort the model. (Don't worry about the math — it just measurably lowered
our error.)

---

## 7. Why we use *several* models, not one (the "ensemble")

We don't train just one model — we train a handful of slightly different ones:
- different "brands" (LightGBM, XGBoost),
- different error-handling settings,
- with and without the log transform.

Each makes slightly different mistakes. When we **average their predictions together** (an
"ensemble" or "blend"), the random mistakes partly cancel out, and the result is more accurate and
more stable than any single model. It's the "wisdom of the crowd" — ask several decent experts and
average their answers.

We don't average them equally — we **find the best weighting** (maybe model A gets 47%, model B
34%, etc.) by checking which combination scored best on our validation (Section 8).

---

## 8. Validation — how we know it works *without* cheating (the most important part)

The biggest danger in machine learning is **fooling yourself**. A model can *memorize* the training
answers and look perfect on training data, yet fail miserably on new orders. We need an honest
estimate of how it'll do on data it has never seen.

### 8a. Hold-out testing
We hide part of the training data from the model during learning, then test on that hidden part —
because we know the true answers there, we can measure the real error. It's like keeping some
practice questions aside to use as a mock exam.

### 8b. We do it *by time* (because the real test is in the future)
Here's a subtlety specific to this problem: the real test orders happen **later in time** than most
training orders (the test even extends ~2 months past the training data). Predicting the *future*
is harder than filling in the *past*.

So our main mock exam is a **chronological hold-out**: we train on the *earliest* 85% of orders and
test on the *most recent* 15%. This mimics the real situation — "learn from the past, predict the
future" — and gives an honest, slightly pessimistic estimate. (Our chronological score is ~3.87,
which is why we were pleasantly surprised the real public score was a better 2.74 — the actual test
turned out a bit easier than our deliberately-tough mock exam.)

### 8c. Cross-validation (using all the data fairly)
We also use **5-fold cross-validation**: split the data into 5 equal chunks, then train 5 times —
each time holding out a different chunk as the mini-test. This way every order gets a prediction
from a model that *didn't* see it, and we get predictions for the whole dataset without cheating.
We use these "honest" predictions to choose the ensemble weights.

### 8d. The no-cheating rule for our "historical average" clues
Remember the seller-average and corridor-average clues from Section 5? If we naively computed
"this seller's average delivery time" using *all* data, each order would partly be looking at its
own answer — that's cheating, and it makes the model look better than it really is. To prevent this,
we use **out-of-fold encoding**: when computing the average clue for an order, we only use *other*
orders, never itself. This keeps the validation honest.

---

## 9. The actual pipeline — what each script does

If you open the project folder, here's the assembly line (run in this order to reproduce the
winning **2.741** submission):

1. **`build_features.py`** → reads `data/train.csv` & `data/test.csv` and produces the ~67
   engineered clues. Saves them as fast-loading `.parquet` files.
2. **`model_lib.py`** → the shared engine: defines the models and the validation harness
   (the chronological hold-out + 5-fold cross-validation logic).
3. **`train_zoo.py`** → trains the family of models (LightGBM variants + XGBoost) and saves each
   one's predictions.
4. **`blend.py`** → finds the best weighting to combine the models, and writes the final
   **`submission.csv`** (the file you upload).

The output, `submission.csv`, is simply: `id, delivery_time_days` — our predicted days for each of
the 15,000 test orders.

> *Note:* the "corridor" (v2) feature experiment lived in extra scripts that we removed after it
> failed to improve the real leaderboard (see the Lesson-learned box below). The four scripts above
> are the clean, winning pipeline.

### 9b. The SAVED, reusable model (predict new orders without retraining)

The trained model is saved on disk in the **`model/`** folder, so you can score brand-new orders
instantly — no 15-minute retrain needed.

- **`model/`** → the actual trained model: 4 booster files (`lgb_l1_log.txt`, `lgb_l1_deep.txt`,
  `lgb_l2_log.txt`, `xgb_mae.json`) plus `meta.json` (the blend weights + settings).
- **`pipeline.py`** → rebuilds the exact same features for any new orders (uses `data/train.csv` to
  recompute the historical-average clues, so training and prediction stay identical).
- **`train_and_save.py`** → run once to (re)create the `model/` folder from scratch.
- **`predict.py`** → loads `model/` and scores a CSV of new orders.

**To predict new orders:**
```
python predict.py  my_new_orders.csv  my_predictions.csv
```
(The input CSV must have the same columns as `data/test.csv` — everything known at order time.)
Run with no arguments, it re-scores `data/test.csv`.

The saved model matches the locked submission almost exactly (prediction correlation **0.996**,
average difference **0.25 days**), so it's a faithful, reusable copy of our winning model — handy
for the business stage (e.g. "if this seller were closer, how much faster would it arrive?").

---

## 10. Where we stand

| Version | What's new | Validation error | Public score |
|---|---|---|---|
| Baseline (1 model) | distance + time + basic clues | 3.90 | — |
| **v1 blend (BEST — keep this)** | blend of 4 LightGBM/XGBoost models (picked from 8) | 3.87 | **2.741** ✅ |
| v2 blend (reverted) | + fine 1° "corridor" clues | 3.86 *(looked better)* | 2.763 ❌ worse |

> **Lesson learned (important):** the corridor features *lowered* our local validation error but
> *raised* the real public score. This means our local mock-exam is slightly **optimistic** and
> doesn't perfectly mirror the real test set. **The leaderboard is the ground truth**, so we kept
> the simpler v1 blend that actually scored best (2.741). Takeaway: trust the real score over local
> validation, and don't add complexity that only helps the mock exam.

**Lower = better in every column.** We're improving the number step by step, and each real
improvement is confirmed by *submitting and watching the public score drop* — because our local
mock exam is a bit harsher than the real test, the leaderboard is our ground truth.

---

## 11. Mini-glossary

- **Feature / clue**: a column of input information the model looks at.
- **Feature engineering**: creating smarter clues from the raw data (the highest-value work).
- **Target**: the thing we're predicting — here, delivery time in days.
- **MAE (Mean Absolute Error)**: average size of our mistakes, in days. Lower is better.
- **Model**: the pattern-finding machine.
- **Gradient Boosting (LightGBM/XGBoost/CatBoost)**: our model type — a team of hundreds of tiny
  decision-tree flowcharts that each fix the previous ones' mistakes.
- **Decision tree**: a flowchart of yes/no questions ending in a number.
- **Ensemble / blend**: averaging several models for a better, steadier answer.
- **Validation / hold-out**: testing on hidden data to honestly estimate real-world accuracy.
- **Cross-validation**: rotating the hidden part so every order gets an honest prediction.
- **Chronological split**: validating "train on past, predict future" to match the real test.
- **Out-of-fold encoding**: computing "historical average" clues without an order peeking at its
  own answer (anti-cheating).
- **Overfitting**: when a model memorizes the training data and fails on new data — the thing all
  our validation is designed to catch.
- **Submission**: the `submission.csv` file of predictions we upload to be scored.

---

## 12. Project files — what's in the folder

| File / folder | What it is |
|---|---|
| **`submission.csv`** | ⭐ The final answer we submit — the locked **2.741** predictions |
| **`approach.md`** | This document (plain-English) |
| **`technical.md`** | The engineering-detail version (features, model configs, validation, math) |
| **`delivery_model.ipynb`** | 📓 Local notebook: loads the saved model, predicts, what-if analysis, feature importance (imports the project's `.py` modules) |
| **`delivery_model_kaggle.ipynb`** | ☁️ **Self-contained Kaggle notebook** (no imports; auto-detects inputs). Upload this one to Kaggle. |
| **`data/`** | The raw competition data (`train.csv`, `test.csv`, `sample_submission.csv`) |
| **`model/`** | 💾 The **saved trained model** — 4 booster files + `meta.json` (blend weights). Reusable. |
| | |
| *Code — reproduce the exact submission:* | *(run in this order)* |
| `build_features.py` | Turns raw data into the ~67 engineered clues |
| `model_lib.py` | Defines the models + the validation harness |
| `train_zoo.py` | Trains the family of models |
| `blend.py` | Combines them and writes `submission.csv` |
| | |
| *Code — use / rebuild the saved model:* | |
| `pipeline.py` | Rebuilds features identically for training **and** prediction |
| `train_and_save.py` | Run once to (re)create the `model/` folder |
| `predict.py` | Loads `model/` and scores new orders → predictions CSV |
| | |
| *(Your own business-analysis work — left untouched)* | `make_plots.py`, `seller_plots.py`, `_seller_analysis.py`, `plots/`, `plots_sellers/` |
| `hackathon-amge-caravane.zip` | The original download (redundant with `data/` — safe to delete) |

**Two ways to use the project:**
1. **Just submit** → upload `submission.csv` (already done, scored 2.741).
2. **Predict new orders** → `python predict.py my_orders.csv my_predictions.csv` (uses the saved `model/`, no retraining).
