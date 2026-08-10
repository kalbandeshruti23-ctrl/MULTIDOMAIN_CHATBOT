import os
import json
import torch
import pandas as pd

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "distilbert_intent_model"
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "multidomain_chatbot_dataset_FINAL_v2.csv"
)


# ============================================================
# GLOBAL MODEL VARIABLES
# ============================================================

_model = None
_tokenizer = None
_intent_mapping = None


# ============================================================
# LOAD TRANSFORMER MODEL
# ============================================================

def load_model():

    global _model
    global _tokenizer

    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(
            "\nDistilBERT model not found.\n\n"
            f"Expected location:\n{MODEL_DIR}\n\n"
            "Please run train_transformer.py first."
        )

    print("Loading DistilBERT transformer model...")

    try:

        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR,
            local_files_only=True
        )

        _model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR,
            local_files_only=True
        )

        _model.eval()

        print("DistilBERT model loaded successfully.")

        return _model, _tokenizer

    except Exception as e:

        raise RuntimeError(
            "\nUnable to load DistilBERT model.\n\n"
            f"Model path: {MODEL_DIR}\n\n"
            f"Error: {str(e)}"
        )


# ============================================================
# LOAD INTENT MAPPING
# ============================================================

def load_intent_mapping():

    global _intent_mapping

    if _intent_mapping is not None:
        return _intent_mapping

    # --------------------------------------------------------
    # Method 1: Read id2label from model config
    # --------------------------------------------------------

    config_file = os.path.join(
        MODEL_DIR,
        "config.json"
    )

    if os.path.exists(config_file):

        try:

            with open(
                config_file,
                "r",
                encoding="utf-8"
            ) as file:

                config = json.load(file)

            id2label = config.get("id2label")

            if id2label:

                mapping = {}

                for key, value in id2label.items():

                    mapping[int(key)] = value

                _intent_mapping = mapping

                return _intent_mapping

        except Exception as e:

            print(
                "Warning: Could not read intent mapping "
                f"from config.json: {e}"
            )

    # --------------------------------------------------------
    # Method 2: Read intent classes from dataset
    # --------------------------------------------------------

    if not os.path.exists(DATA_FILE):

        raise FileNotFoundError(
            "\nDataset not found.\n\n"
            f"Expected location:\n{DATA_FILE}"
        )

    try:

        df = pd.read_csv(DATA_FILE)

    except Exception as e:

        raise RuntimeError(
            f"Unable to read dataset:\n{e}"
        )

    if "intent" not in df.columns:

        raise ValueError(
            "The dataset does not contain an 'intent' column."
        )

    intents = sorted(
        df["intent"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    _intent_mapping = {
        index: intent
        for index, intent in enumerate(intents)
    }

    return _intent_mapping


# ============================================================
# GET DOMAIN FROM INTENT
# ============================================================

def get_domain(intent):

    intent = str(intent).strip().lower()

    # --------------------------------------------------------
    # Finance
    # --------------------------------------------------------

    finance_intents = {
        "cash_withdrawal",
        "cash_withdrawal_charge",
        "card_arrival",
        "card_payment_not_recognised",
        "payment_issue",
        "transfer_pending"
    }

    # --------------------------------------------------------
    # E-Commerce
    # --------------------------------------------------------

    ecommerce_intents = {
        "track_order",
        "return_product",
        "refund",
        "cancel_order"
    }

    # --------------------------------------------------------
    # Healthcare
    # --------------------------------------------------------

    healthcare_intents = {
        "appointment",
        "symptom_query",
        "emergency",
        "medical_report",
        "medicine_information"
    }

    if intent in finance_intents:
        return "Finance"

    if intent in ecommerce_intents:
        return "E-commerce"

    if intent in healthcare_intents:
        return "Healthcare"

    return "General"


# ============================================================
# PREDICT INTENT
# ============================================================

def predict_intent(text):

    if text is None:

        return {
            "intent": "unknown",
            "confidence": 0.0,
            "domain": "General"
        }

    text = str(text).strip()

    if not text:

        return {
            "intent": "unknown",
            "confidence": 0.0,
            "domain": "General"
        }

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model, tokenizer = load_model()

    # --------------------------------------------------------
    # Load intent mapping
    # --------------------------------------------------------

    intent_mapping = load_intent_mapping()

    # --------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )

        confidence, predicted_id = torch.max(
            probabilities,
            dim=-1
        )

    predicted_id = predicted_id.item()

    confidence = confidence.item()

    # --------------------------------------------------------
    # Intent
    # --------------------------------------------------------

    intent = intent_mapping.get(
        predicted_id,
        f"class_{predicted_id}"
    )

    # --------------------------------------------------------
    # Domain
    # --------------------------------------------------------

    domain = get_domain(intent)

    return {
        "intent": intent,
        "confidence": confidence,
        "domain": domain
    }


# ============================================================
# PREDICT - SIMPLE FUNCTION FOR APP
# ============================================================

def predict(text):

    return predict_intent(text)


# ============================================================
# GET CONFIDENCE AS PERCENTAGE
# ============================================================

def get_confidence_percentage(text):

    result = predict_intent(text)

    return result["confidence"] * 100


# ============================================================
# TEST THE MODEL
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("       MULTI-DOMAIN TRANSFORMER CHATBOT")
    print("       DistilBERT Intent Prediction")
    print("=" * 60)

    test_messages = [

        "Where is my order?",

        "My money transfer is pending",

        "How can I schedule an appointment?",

        "I want to return my product",

        "My card payment was not recognised",

        "I need information about my medicine"

    ]

    print()

    for message in test_messages:

        try:

            result = predict_intent(message)

            print(f"User       : {message}")
            print(f"Domain     : {result['domain']}")
            print(f"Intent     : {result['intent']}")
            print(
                f"Confidence : "
                f"{result['confidence'] * 100:.2f}%"
            )

            print("-" * 60)

        except Exception as e:

            print(f"ERROR: {e}")
            print("-" * 60)