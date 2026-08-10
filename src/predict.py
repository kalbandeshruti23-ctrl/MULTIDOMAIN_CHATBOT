import os
import re
import joblib
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

VECTORIZER_PATH = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer.pkl"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "baseline_intent_model.pkl"
)

INTENT_MAPPING_PATH = os.path.join(
    PROCESSED_DIR,
    "intent_mapping.csv"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            "TF-IDF vectorizer not found. Run train_baseline.py first."
        )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Intent model not found. Run train_baseline.py first."
        )

    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)

    return vectorizer, model


# ============================================================
# LOAD INTENT MAPPING
# ============================================================

def load_intent_mapping():

    if not os.path.exists(INTENT_MAPPING_PATH):
        return {}

    mapping = pd.read_csv(INTENT_MAPPING_PATH)

    return dict(
        zip(
            mapping["intent_id"],
            mapping["intent"]
        )
    )


# ============================================================
# PREPROCESS QUERY
# ============================================================

def preprocess_query(query):

    query = query.lower().strip()

    replacements = {

        "expire medicine": "expired medicine",
        "expire tablet": "expired tablet",
        "expire": "expired",

        "medicne": "medicine",
        "medicin": "medicine",

        "appt": "appointment",
        "appointmnt": "appointment",

        "delievery": "delivery",
        "refunding": "refund",

        "cant": "can't",
        "wont": "won't"

    }

    for wrong, correct in replacements.items():
        query = query.replace(wrong, correct)

    query = re.sub(r"[^a-z0-9\s]", " ", query)

    query = re.sub(r"\s+", " ", query)

    return query.strip()


# ============================================================
# PREDICT INTENT
# ============================================================

def predict_intent(
    query,
    vectorizer,
    model
):

    query = preprocess_query(query)

    query_vector = vectorizer.transform(
        [query]
    )

    predicted_intent = model.predict(
        query_vector
    )[0]

    probabilities = model.predict_proba(
        query_vector
    )[0]

    confidence = float(probabilities.max())

    return predicted_intent, confidence


# ============================================================
# OPTIONAL KEYWORD FALLBACK
# ============================================================

def keyword_fallback(query):

    text = query.lower()

    if any(word in text for word in [
        "expired",
        "expiry",
        "expire"
    ]):
        return (
            "expired_medicine",
            0.99
        )

    return None


# ============================================================
# SMART PREDICTION
# ============================================================

def smart_predict(
    query,
    vectorizer,
    model
):

    fallback = keyword_fallback(query)

    if fallback is not None:
        return fallback

    return predict_intent(
        query,
        vectorizer,
        model
    )


# ============================================================
# INTERACTIVE MODE
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("MULTI-DOMAIN CHATBOT")
    print("INTENT PREDICTION")
    print("=" * 60)

    vectorizer, model = load_model()

    print("\nModel loaded successfully.")

    print("\nType your query.")

    print("Type 'exit' to stop.")

    while True:

        query = input("\nYou : ").strip()

        if query.lower() == "exit":
            print("\nPrediction stopped.")
            break

        if not query:
            print("Please enter a query.")
            continue

        intent, confidence = smart_predict(
            query,
            vectorizer,
            model
        )

        print("\nDetected Intent :", intent)

        print(
            "Confidence : {:.2f}%".format(
                confidence * 100
            )
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()