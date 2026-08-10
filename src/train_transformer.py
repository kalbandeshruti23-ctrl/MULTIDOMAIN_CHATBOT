import os
import json
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "distilbert-base-uncased"

RANDOM_STATE = 42

MAX_LENGTH = 128


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
    "models",
    "distilbert_intent_model"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
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

    train_df = pd.read_csv(
        train_path
    )

    validation_df = pd.read_csv(
        validation_path
    )

    test_df = pd.read_csv(
        test_path
    )

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
# CREATE LABEL MAPPING
# ============================================================

def create_label_mapping(
    train_df,
    validation_df,
    test_df
):

    all_intents = sorted(
        set(train_df["intent"])
        | set(validation_df["intent"])
        | set(test_df["intent"])
    )

    label2id = {
        intent: index
        for index, intent
        in enumerate(all_intents)
    }

    id2label = {
        index: intent
        for intent, index
        in label2id.items()
    }

    print(
        f"\nNumber of intents: "
        f"{len(all_intents)}"
    )

    return (
        label2id,
        id2label
    )


# ============================================================
# CONVERT DATAFRAME TO HUGGING FACE DATASET
# ============================================================

def prepare_dataset(
    dataframe,
    label2id
):

    dataframe = dataframe.copy()

    dataframe["label"] = dataframe[
        "intent"
    ].map(label2id)

    dataframe = dataframe[
        [
            "user_query",
            "label"
        ]
    ]

    return Dataset.from_pandas(
        dataframe,
        preserve_index=False
    )


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_datasets(
    train_dataset,
    validation_dataset,
    test_dataset,
    tokenizer
):

    def tokenize_function(batch):

        return tokenizer(
            batch["user_query"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )

    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True
    )

    validation_dataset = validation_dataset.map(
        tokenize_function,
        batched=True
    )

    test_dataset = test_dataset.map(
        tokenize_function,
        batched=True
    )

    columns = [
        "input_ids",
        "attention_mask",
        "label"
    ]

    train_dataset.set_format(
        type="torch",
        columns=columns
    )

    validation_dataset.set_format(
        type="torch",
        columns=columns
    )

    test_dataset.set_format(
        type="torch",
        columns=columns
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset
    )


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    evaluation_result
):

    logits, labels = evaluation_result

    predictions = np.argmax(
        logits,
        axis=-1
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="weighted",
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 65)
    print("MULTI-DOMAIN CHATBOT")
    print("DISTILBERT INTENT CLASSIFICATION")
    print("=" * 65)

    print(
        f"\nDevice: {DEVICE}"
    )

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    (
        train_df,
        validation_df,
        test_df
    ) = load_data()

    # --------------------------------------------------------
    # Label mapping
    # --------------------------------------------------------

    (
        label2id,
        id2label
    ) = create_label_mapping(
        train_df,
        validation_df,
        test_df
    )

    # --------------------------------------------------------
    # Save mappings
    # --------------------------------------------------------

    with open(
        os.path.join(
            MODEL_DIR,
            "label2id.json"
        ),
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            label2id,
            file,
            indent=4
        )

    with open(
        os.path.join(
            MODEL_DIR,
            "id2label.json"
        ),
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                str(key): value
                for key, value
                in id2label.items()
            },
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Convert datasets
    # --------------------------------------------------------

    train_dataset = prepare_dataset(
        train_df,
        label2id
    )

    validation_dataset = prepare_dataset(
        validation_df,
        label2id
    )

    test_dataset = prepare_dataset(
        test_df,
        label2id
    )

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    print(
        "\nLoading tokenizer..."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # Tokenize
    # --------------------------------------------------------

    print(
        "Tokenizing datasets..."
    )

    (
        train_dataset,
        validation_dataset,
        test_dataset
    ) = tokenize_datasets(
        train_dataset,
        validation_dataset,
        test_dataset,
        tokenizer
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        "\nLoading pretrained DistilBERT..."
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    # --------------------------------------------------------
    # Training arguments
    # --------------------------------------------------------

    training_args = TrainingArguments(
        output_dir=MODEL_DIR,

        eval_strategy="epoch",

        save_strategy="epoch",

        logging_strategy="steps",

        logging_steps=25,

        learning_rate=2e-5,

        per_device_train_batch_size=8,

        per_device_eval_batch_size=8,

        num_train_epochs=3,

        weight_decay=0.01,

        load_best_model_at_end=True,

        metric_for_best_model="f1",

        greater_is_better=True,

        save_total_limit=2,

        report_to="none",

        fp16=False,

        seed=RANDOM_STATE
    )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

    trainer = Trainer(
        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=validation_dataset,

        processing_class=tokenizer,

        compute_metrics=compute_metrics
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("STARTING TRANSFORMER TRAINING")
    print("=" * 65)

    trainer.train()

    # --------------------------------------------------------
    # Validation evaluation
    # --------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("VALIDATION RESULTS")
    print("=" * 65)

    validation_results = trainer.evaluate(
        eval_dataset=validation_dataset
    )

    for key, value in validation_results.items():

        if isinstance(value, float):

            print(
                f"{key}: {value:.4f}"
            )

    # --------------------------------------------------------
    # Test evaluation
    # --------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("TEST RESULTS")
    print("=" * 65)

    test_results = trainer.evaluate(
        eval_dataset=test_dataset
    )

    for key, value in test_results.items():

        if isinstance(value, float):

            print(
                f"{key}: {value:.4f}"
            )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    print(
        "\nSaving Transformer model..."
    )

    trainer.save_model(
        MODEL_DIR
    )

    tokenizer.save_pretrained(
        MODEL_DIR
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics = {

        "model":
            MODEL_NAME,

        "validation_accuracy":
            validation_results.get(
                "eval_accuracy",
                None
            ),

        "validation_precision":
            validation_results.get(
                "eval_precision",
                None
            ),

        "validation_recall":
            validation_results.get(
                "eval_recall",
                None
            ),

        "validation_f1":
            validation_results.get(
                "eval_f1",
                None
            ),

        "test_accuracy":
            test_results.get(
                "eval_accuracy",
                None
            ),

        "test_precision":
            test_results.get(
                "eval_precision",
                None
            ),

        "test_recall":
            test_results.get(
                "eval_recall",
                None
            ),

        "test_f1":
            test_results.get(
                "eval_f1",
                None
            )
    }

    metrics_path = os.path.join(
        RESULTS_DIR,
        "transformer_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("TRANSFORMER TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 65)

    print(
        f"\nModel saved at:\n"
        f"{MODEL_DIR}"
    )

    print(
        f"\nMetrics saved at:\n"
        f"{metrics_path}"
    )

    print("\n")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()