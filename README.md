# ✈️ Airline Complaint Classifier

An NLP-based Machine Learning project that automatically classifies
airline-related customer complaints into categories such as **Late Flight,
Lost Luggage, Cancelled Flight, Customer Service Issue**, and more.

The system helps airlines automatically identify the type of complaint
and route it to the appropriate support team.

---

### Objective

The objective is to build an automated complaint classification system
that can:

- Process raw customer complaint text
- Clean and preprocess the text
- Extract meaningful NLP features
- Predict the complaint category
- Provide fast and consistent classification

## 🗂️ Dataset

**[Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)** — ~14,600 real tweets about 6 US airlines, human-labeled for sentiment and complaint reason (`negativereason`). The complaint reason is the target predicted by this project.

## 🔍 Approach

1. **Data Filtering** — Kept negative tweets with a specific complaint reason and removed `"Can't Tell"`.
2. **EDA** — Analyzed class distribution, tweet length, and common words by category.
3. **Text Cleaning** — Removed mentions, URLs, and punctuation noise.
4. **Baseline Model** — TF-IDF with Logistic Regression.
5. **Imbalance Handling** — Compared standard and `class_weight="balanced"` models.
6. **Semantic Model** — Evaluated `all-MiniLM-L6-v2` sentence embeddings with Logistic Regression.
7. **Evaluation** — Used Macro F1, Weighted F1, classification report, and confusion matrix.
8. **Error Analysis** — Examined misclassified tweets to identify common sources of confusion.
9. **Deployment** — Saved the best model and deployed it using Streamlit.

## Results

| Model                                      | F1-macro | F1-weighted |
|---------------------------------------------|----------|-------------|
| TF-IDF + Logistic Regression (baseline)      | 0.444    | 0.627       |
| TF-IDF + Logistic Regression (balanced)      | **0.564**| 0.651       |

**Best model: TF-IDF + Logistic Regression, class-balanced** — 63.8% overall
accuracy, 0.564 F1-macro across 9 categories (random chance ≈ 0.11).

```

```

## 🚀 Deployed App

Try it live: **https://support-ticket-classifier-nxfkihjkjg2dudusjdtlxk.streamlit.app/**

## Review
[](images/image1.png)
[](images/image2.png)

```
```

## Author
**Indradev Kumar**
