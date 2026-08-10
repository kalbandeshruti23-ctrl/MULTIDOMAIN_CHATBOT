import os
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\nLoading processed datasets...")

    train_path = os.path.join(
        PROCESSED_DIR,
        "train.csv"
    )

    validation_path = os.path.join(
        PROCESSED_DIR,
        "validation.csv"
    )

    test_path = os.path.join(
        PROCESSED_DIR,
        "test.csv"
    )

    train_df = pd.read_csv(train_path)
    validation_df = pd.read_csv(validation_path)
    test_df = pd.read_csv(test_path)

    print(
        f"Training records   : {len(train_df)}"
    )

    print(
        f"Validation records : {len(validation_df)}"
    )

    print(
        f"Test records       : {len(test_df)}"
    )

    return (
        train_df,
        validation_df,
        test_df
    )


# ============================================================
# TF-IDF
# ============================================================

def create_vectorizer():

    print("\nCreating TF-IDF vectorizer...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True
    )

    return vectorizer


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(X_train, y_train):

    print("\nTraining Logistic Regression model...")

    model = LogisticRegression(
        max_iter=2000,
        C=5.0,
        class_weight="balanced",
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "Model training completed."
    )

    return model


# ============================================================
# EVALUATE
# ============================================================

def evaluate_model(
    model,
    X,
    y,
    dataset_name
):

    print(
        f"\nEvaluating on {dataset_name}..."
    )

    predictions = model.predict(X)

    accuracy = accuracy_score(
        y,
        predictions
    )

    print(
        f"{dataset_name} Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print("\nClassification Report:")

    report = classification_report(
        y,
        predictions,
        zero_division=0
    )

    print(report)

    return (
        predictions,
        accuracy,
        report
    )


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    y_true,
    y_pred,
    model,
    filename
):

    print(
        "\nCreating confusion matrix..."
    )

    labels = model.classes_

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    plt.figure(
        figsize=(18, 15)
    )

    plt.imshow(
        matrix,
        interpolation="nearest"
    )

    plt.title(
        "Baseline Intent Classification - Confusion Matrix"
    )

    plt.colorbar()

    plt.xticks(
        range(len(labels)),
        labels,
        rotation=90,
        fontsize=7
    )

    plt.yticks(
        range(len(labels)),
        labels,
        fontsize=7
    )

    plt.xlabel(
        "Predicted Intent"
    )

    plt.ylabel(
        "Actual Intent"
    )

    plt.tight_layout()

    path = os.path.join(
        RESULTS_DIR,
        filename
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Confusion matrix saved:\n{path}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    vectorizer,
    model
):

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    vectorizer_path = os.path.join(
        MODEL_DIR,
        "tfidf_vectorizer.pkl"
    )

    model_path = os.path.join(
        MODEL_DIR,
        "baseline_intent_model.pkl"
    )

    joblib.dump(
        vectorizer,
        vectorizer_path
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        "\nModel files saved:"
    )

    print(
        vectorizer_path
    )

    print(
        model_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("MULTI-DOMAIN CHATBOT")
    print("BASELINE INTENT CLASSIFICATION")
    print("=" * 60)

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    (
        train_df,
        validation_df,
        test_df
    ) = load_data()

    # --------------------------------------------------------
    # Text and labels
    # --------------------------------------------------------

    X_train_text = train_df[
        "user_query"
    ]

    y_train = train_df[
        "intent"
    ]

    X_validation_text = validation_df[
        "user_query"
    ]

    y_validation = validation_df[
        "intent"
    ]

    X_test_text = test_df[
        "user_query"
    ]

    y_test = test_df[
        "intent"
    ]

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    vectorizer = create_vectorizer()

    X_train = vectorizer.fit_transform(
        X_train_text
    )

    X_validation = vectorizer.transform(
        X_validation_text
    )

    X_test = vectorizer.transform(
        X_test_text
    )

    print(
        f"\nTF-IDF training matrix shape: "
        f"{X_train.shape}"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model = train_model(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    (
        validation_predictions,
        validation_accuracy,
        validation_report
    ) = evaluate_model(
        model,
        X_validation,
        y_validation,
        "Validation"
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    (
        test_predictions,
        test_accuracy,
        test_report
    ) = evaluate_model(
        model,
        X_test,
        y_test,
        "Test"
    )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    save_confusion_matrix(
        y_test,
        test_predictions,
        model,
        "baseline_confusion_matrix.png"
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    save_model(
        vectorizer,
        model
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics = pd.DataFrame({
        "Dataset": [
            "Validation",
            "Test"
        ],
        "Accuracy": [
            validation_accuracy,
            test_accuracy
        ]
    })

    metrics_path = os.path.join(
        RESULTS_DIR,
        "baseline_metrics.csv"
    )

    metrics.to_csv(
        metrics_path,
        index=False
    )

    print(
        f"\nMetrics saved:\n{metrics_path}"
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("BASELINE TRAINING COMPLETED")
    print("=" * 60)

    print(
        f"\nValidation Accuracy : "
        f"{validation_accuracy * 100:.2f}%"
    )

    print(
        f"Test Accuracy       : "
        f"{test_accuracy * 100:.2f}%"
    )

    print(
        "\nNext step:"
    )

    print(
        "Run predict.py to test individual user queries."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()