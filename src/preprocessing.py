import os
import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "multidomain_chatbot_dataset_FINAL_v2.csv"
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Clean chatbot text while preserving useful information.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Remove leading/trailing spaces
    text = text.strip()

    # Convert multiple spaces into one
    text = re.sub(r"\s+", " ", text)

    return text


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():
    """
    Load the final chatbot dataset.
    """

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            "\nDataset not found!\n"
            f"Expected location:\n{DATA_PATH}\n\n"
            "Please make sure the CSV file is inside the data folder."
        )

    print("\nLoading dataset...")
    df = pd.read_csv(DATA_PATH)

    print(f"Dataset loaded successfully: {len(df)} rows")

    return df


# ============================================================
# VALIDATE DATASET
# ============================================================

def validate_dataset(df):
    """
    Check whether all required columns exist.
    """

    required_columns = [
        "id",
        "domain",
        "intent",
        "user_query",
        "response",
        "source",
        "split"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"\nMissing required columns: {missing_columns}"
        )

    print("\nDataset validation successful.")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")


# ============================================================
# PREPROCESS DATA
# ============================================================

def preprocess_data(df):

    print("\nCleaning text data...")

    # Clean text columns
    df["user_query"] = df["user_query"].apply(clean_text)
    df["response"] = df["response"].apply(clean_text)
    df["domain"] = df["domain"].apply(clean_text)
    df["intent"] = df["intent"].apply(clean_text)

    # Remove rows with empty user queries
    before = len(df)

    df = df[df["user_query"] != ""]

    removed_empty = before - len(df)

    print(
        f"Empty queries removed: {removed_empty}"
    )

    # Remove duplicate queries
    before = len(df)

    df = df.drop_duplicates(
        subset=["user_query"],
        keep="first"
    )

    removed_duplicates = before - len(df)

    print(
        f"Duplicate queries removed: {removed_duplicates}"
    )

    # Reset index
    df = df.reset_index(drop=True)

    return df


# ============================================================
# ENCODE INTENTS
# ============================================================

def encode_labels(df):

    print("\nEncoding intent labels...")

    intent_encoder = LabelEncoder()

    df["intent_id"] = intent_encoder.fit_transform(
        df["intent"]
    )

    print(
        f"Number of intents: "
        f"{len(intent_encoder.classes_)}"
    )

    # Create intent mapping
    intent_mapping = pd.DataFrame({
        "intent_id": range(
            len(intent_encoder.classes_)
        ),
        "intent": intent_encoder.classes_
    })

    return df, intent_mapping


# ============================================================
# ENCODE DOMAINS
# ============================================================

def encode_domains(df):

    print("\nEncoding domain labels...")

    domain_encoder = LabelEncoder()

    df["domain_id"] = domain_encoder.fit_transform(
        df["domain"]
    )

    print(
        f"Number of domains: "
        f"{len(domain_encoder.classes_)}"
    )

    # Create domain mapping
    domain_mapping = pd.DataFrame({
        "domain_id": range(
            len(domain_encoder.classes_)
        ),
        "domain": domain_encoder.classes_
    })

    return df, domain_mapping


# ============================================================
# CREATE TRAIN / VALIDATION / TEST DATA
# ============================================================

def create_splits(df):

    print("\nCreating dataset splits...")

    train_df = df[
        df["split"].str.lower() == "train"
    ].copy()

    validation_df = df[
        df["split"].str.lower() == "validation"
    ].copy()

    test_df = df[
        df["split"].str.lower() == "test"
    ].copy()

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
# SAVE PROCESSED DATA
# ============================================================

def save_processed_data(
    df,
    train_df,
    validation_df,
    test_df,
    intent_mapping,
    domain_mapping
):

    print("\nSaving processed files...")

    # Create directory
    os.makedirs(
        PROCESSED_DIR,
        exist_ok=True
    )

    # Complete processed dataset
    df.to_csv(
        os.path.join(
            PROCESSED_DIR,
            "processed_dataset.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # Training data
    train_df.to_csv(
        os.path.join(
            PROCESSED_DIR,
            "train.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # Validation data
    validation_df.to_csv(
        os.path.join(
            PROCESSED_DIR,
            "validation.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # Test data
    test_df.to_csv(
        os.path.join(
            PROCESSED_DIR,
            "test.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # Intent mapping
    intent_mapping.to_csv(
        os.path.join(
            PROCESSED_DIR,
            "intent_mapping.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # Domain mapping
    domain_mapping.to_csv(
        os.path.join(
            PROCESSED_DIR,
            "domain_mapping.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"\nFiles saved successfully in:\n"
        f"{PROCESSED_DIR}"
    )


# ============================================================
# DATASET SUMMARY
# ============================================================

def display_summary(df):

    print("\n")
    print("=" * 60)
    print("MULTI-DOMAIN CHATBOT DATASET SUMMARY")
    print("=" * 60)

    print(
        f"\nTotal records      : {len(df)}"
    )

    print(
        f"Total domains      : "
        f"{df['domain'].nunique()}"
    )

    print(
        f"Total intents      : "
        f"{df['intent'].nunique()}"
    )

    print("\nDomain distribution:")

    domain_counts = (
        df["domain"]
        .value_counts()
    )

    for domain, count in domain_counts.items():

        print(
            f"  {domain:<15} : {count}"
        )

    print("\nIntent distribution:")

    intent_counts = (
        df["intent"]
        .value_counts()
    )

    for intent, count in intent_counts.items():

        print(
            f"  {intent:<30} : {count}"
        )

    print("\nDataset split:")

    split_counts = (
        df["split"]
        .value_counts()
    )

    for split, count in split_counts.items():

        print(
            f"  {split:<15} : {count}"
        )


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("MULTI-DOMAIN CHATBOT")
    print("DATA PREPROCESSING")
    print("=" * 60)

    try:

        # 1. Load dataset
        df = load_dataset()

        # 2. Validate dataset
        validate_dataset(df)

        # 3. Preprocess text
        df = preprocess_data(df)

        # 4. Encode intents
        df, intent_mapping = encode_labels(df)

        # 5. Encode domains
        df, domain_mapping = encode_domains(df)

        # 6. Create train/validation/test split
        (
            train_df,
            validation_df,
            test_df
        ) = create_splits(df)

        # 7. Save processed files
        save_processed_data(
            df,
            train_df,
            validation_df,
            test_df,
            intent_mapping,
            domain_mapping
        )

        # 8. Display summary
        display_summary(df)

        print("\n")
        print("=" * 60)
        print("PREPROCESSING COMPLETED SUCCESSFULLY")
        print("=" * 60)

    except Exception as e:

        print("\nERROR:")
        print(e)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()