"""
Streamlit app: Airline Complaint Classifier

Loads the model trained in notebooks/ticket_classification.ipynb and lets a
user paste a customer complaint (like a tweet) to get a predicted category +
confidence scores.

Run locally:
    streamlit run app/app.py
"""

import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

st.set_page_config(page_title="Airline Complaint Classifier", page_icon="✈️", layout="centered")


@st.cache_resource
def load_model():
    model_type_path = MODEL_DIR / "model_type.joblib"
    if not model_type_path.exists():
        return None

    model_type = joblib.load(model_type_path)
    model = joblib.load(MODEL_DIR / "ticket_classifier.joblib")

    if model_type == "tfidf":
        vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
        return {"type": "tfidf", "model": model, "vectorizer": vectorizer}
    else:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(str(MODEL_DIR / "sentence_embedder"))
        return {"type": "embeddings", "model": model, "embedder": embedder}


bundle = load_model()

st.title("✈️ Airline Complaint Classifier")
st.write(
    "Paste a customer complaint below and the model will predict which "
    "category it falls under — e.g. **Late Flight**, **Lost Luggage**, "
    "**Customer Service Issue**, **Cancelled Flight**, and more."
)

if bundle is None:
    st.error(
        "No trained model found in `models/`. Run all cells in "
        "`notebooks/ticket_classification.ipynb` first — the last cell "
        "saves the model files this app needs."
    )
    st.stop()

example_tickets = {
    "-- pick an example --": "",
    "Lost luggage example": "My bag never showed up at baggage claim and nobody can tell me where it is.",
    "Late flight example": "This is the third time my flight has been delayed for over 3 hours with no updates.",
    "Customer service example": "I've been on hold for over an hour trying to reach someone about my booking.",
    "Cancelled flight example": "My flight got cancelled with no notice and now I'm stuck at the airport.",
    "Flight booking example": "I tried to book a flight online three times and the payment kept failing.",
}

choice = st.selectbox("Try an example, or write your own below:", list(example_tickets.keys()))
default_text = example_tickets[choice]

ticket_text = st.text_area(
    "Complaint text",
    value=default_text,
    height=140,
    placeholder="e.g. My connecting flight was cancelled and no one at the gate could help...",
)


def predict(text: str):
    if bundle["type"] == "tfidf":
        vec = bundle["vectorizer"].transform([text])
        proba = bundle["model"].predict_proba(vec)[0]
    else:
        emb = bundle["embedder"].encode([text])
        proba = bundle["model"].predict_proba(emb)[0]
    return pd.DataFrame(
        {"Category": bundle["model"].classes_, "Probability": proba}
    ).sort_values("Probability", ascending=False).reset_index(drop=True)


if st.button("Classify complaint", type="primary"):
    if not ticket_text.strip():
        st.warning("Please enter some complaint text first.")
    else:
        with st.spinner("Classifying..."):
            prob_df = predict(ticket_text)

        st.success(f"**Predicted category:** {prob_df.iloc[0]['Category']}")
        st.caption(f"Confidence: {prob_df.iloc[0]['Probability']:.1%}")

        st.bar_chart(prob_df.set_index("Category"))
        st.dataframe(prob_df.style.format({"Probability": "{:.1%}"}), use_container_width=True)

st.divider()
st.caption(
    "Portfolio project · Model: TF-IDF + Logistic Regression (class-balanced) · "
    "Trained on the [Twitter US Airline Sentiment dataset](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) · "
    "[View the training notebook & code on GitHub](#)"
)
