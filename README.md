# ✈️ Airline Complaint Classifier

Automatically classifies a customer complaint (e.g. a tweet directed at an
airline) into a specific category — **Late Flight**, **Lost Luggage**,
**Customer Service Issue**, **Cancelled Flight**, and more — so it could be
routed to the right team instead of a human manually reading and sorting
every complaint.

**[🔗 Live demo](#)** — *(add your deployed Streamlit link here once deployed)*

![Python](https://img.shields.io/badge/python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![Streamlit](https://img.shields.io/badge/streamlit-app-red)

---

## Problem Statement

Support and social-media teams receive a high volume of customer complaints
that need to be triaged before reaching the right team. This project builds
a text classifier that predicts a complaint's category directly from its
text, and wraps it in a simple app that mimics how it could be used in a
real routing pipeline.

## Dataset

[Twitter US Airline Sentiment (Kaggle, crowdflower)](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
— ~14,600 real tweets directed at 6 US airlines (Feb 2015), human-labeled
with sentiment and, for negative tweets, a specific complaint reason.

> **Before running the notebook:** download the CSV from the link above and
> save it as `data/airline_tweets.csv` (see `data/README.md`). This file
> isn't included in the repo — download it yourself.

**A note on dataset choice:** this project originally used a different
Kaggle "customer support ticket" dataset. Investigation during EDA revealed
that dataset's ticket text didn't actually correlate with its category
labels — identical ticket descriptions appeared under different categories,
and every model (regardless of complexity) scored at random-chance level.
Rather than ship a misleading result, the project was switched to this
dataset, where labels come from real human annotators reading real tweets,
so the text-label relationship is genuine. That diagnostic process is
documented in the first notebook cell — worth reading, since "the model
looks broken, is it the model or the data?" is a real skill, not a footnote.

## Approach

1. **Filtering** — kept only negative tweets with a specific complaint
   reason; dropped `"Can't Tell"` (an annotator-uncertainty flag, not a real
   category)
2. **EDA** — class balance (highly imbalanced, ~39x between largest and
   smallest class — a realistic scenario), tweet length distribution, most
   common words per category
3. **Cleaning** — removed @mentions (so the model can't "cheat" by learning
   airline identity), URLs, and punctuation noise
4. **Baseline model** — TF-IDF + Logistic Regression
5. **Class imbalance handling** — compared unweighted vs.
   `class_weight="balanced"`
6. **Optional stronger model** — sentence embeddings
   (`sentence-transformers`, `all-MiniLM-L6-v2`) + Logistic Regression,
   included in the notebook but requires internet to download the pretrained
   model on first run
7. **Evaluation** — F1-macro / F1-weighted (not just accuracy, since classes
   are heavily imbalanced) + confusion matrix
8. **Error analysis** — inspected misclassified tweets to understand *why*
   the model confuses certain categories
9. **Deployment** — best model saved and served through a Streamlit app

## Results

*(From an actual run on the full dataset — see `notebooks/ticket_classification.ipynb` for full output.)*

| Model                                      | F1-macro | F1-weighted |
|---------------------------------------------|----------|-------------|
| TF-IDF + Logistic Regression (baseline)      | 0.444    | 0.627       |
| TF-IDF + Logistic Regression (balanced)      | **0.564**| 0.651       |

**Best model: TF-IDF + Logistic Regression, class-balanced** — 63.8% overall
accuracy, 0.564 F1-macro across 9 categories (random chance ≈ 0.11).

**Class balancing had a large effect on rare categories:**
- *Damaged Luggage* (15 test examples): F1 went from **0.00 → 0.46** — the
  unweighted model never predicted this class at all
- *Flight Booking Problems*: F1 went from 0.32 → 0.47
- *Bad Flight*: F1 went from 0.43 → 0.48

This is the clearest evidence in the whole project that class weighting
matters: on a class this rare, an unweighted model just gives up and never
predicts it, no matter how relevant the text is.

**Key error pattern:** the single most common confusion is *Customer Service
Issue* being predicted for tweets actually labeled *Flight Booking Problems*
(84 cases) — both categories often involve complaints about being unable to
get help or resolve something through the airline's support channels, so the
vocabulary genuinely overlaps. Other confusions cluster similarly around
semantically related categories (e.g. *Late Flight* vs. *Cancelled Flight*).

**Note on the embeddings model:** the notebook includes code for a sentence-
embeddings model, but it requires internet access to download the pretrained
model on first run — if you run the notebook with internet access, re-run
Section 9 to see whether it beats the TF-IDF model, and update this table.

## Project Structure

```
support-ticket-classifier/
├── data/
│   ├── README.md                              # where to download the dataset
│   └── airline_tweets.csv                      # you add this (downloaded from Kaggle, gitignored)
├── notebooks/
│   └── ticket_classification.ipynb             # full EDA + modeling pipeline
├── models/                                     # trained model artifacts (created by the notebook)
│   ├── ticket_classifier.joblib
│   ├── tfidf_vectorizer.joblib
│   ├── model_type.joblib
│   └── categories.joblib
├── app/
│   └── app.py                                  # Streamlit demo app
├── requirements.txt
└── README.md
```

## How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/support-ticket-classifier.git
cd support-ticket-classifier

# 2. Create a virtual environment and install dependencies
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Download the dataset (see data/README.md) and place it at:
#    data/airline_tweets.csv

# 4. Run the notebook to train and save the model
jupyter notebook notebooks/ticket_classification.ipynb
#    -> Run All cells. This creates the files in models/

# 5. Launch the app
streamlit run app/app.py
```

## Deploying (Streamlit Community Cloud — free)

1. Push this repo to GitHub (see below if you haven't already).
2. **Run the notebook locally first** and commit the resulting `models/`
   folder — Streamlit Cloud only serves the app, it doesn't run your
   notebook for you.
3. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, click **"New app"**.
4. Select this repository, branch `main`, and set the main file path to
   `app/app.py`.
5. Click **Deploy**. Streamlit Cloud installs everything from
   `requirements.txt` automatically.
6. Once live, copy the app URL back into the "Live demo" link at the top of
   this README.

*(Alternative: Hugging Face Spaces also supports Streamlit apps for free.)*

## Pushing This Repo to GitHub

```bash
cd support-ticket-classifier
git init
git add .
git commit -m "Initial commit: airline complaint classification project"
git branch -M main
git remote add origin https://github.com/<your-username>/support-ticket-classifier.git
git push -u origin main
```

> **Note on the model files:** `ticket_classifier.joblib` and
> `tfidf_vectorizer.joblib` are small (a few hundred KB) and safe to commit
> directly — no Git LFS needed for the TF-IDF model. If you switch to the
> embeddings model, the `sentence_embedder` folder can be ~80MB; use
> [Git LFS](https://git-lfs.com/) or exclude it via `.gitignore` in that case.

## Future Improvements

- Get the sentence-embeddings model actually running (needs internet on
  first run) and compare it fairly against the TF-IDF baseline.
- Try a fine-tuned transformer (e.g. `distilbert-base-uncased`) directly on
  the full dataset.
- Investigate the *Customer Service Issue* ↔ *Flight Booking Problems*
  confusion further — possibly merge them, or add features that distinguish
  "can't get help" from "can't complete a booking."
- Cost-sensitive evaluation: weight misclassifications by how expensive a
  misroute actually is for the business.
- Try oversampling (SMOTE) as an alternative to class weighting for the
  rarest categories (Damaged Luggage, longlines).

## Author

*(Your name, LinkedIn, portfolio link)*
