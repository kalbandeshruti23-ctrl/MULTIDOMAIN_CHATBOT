import os
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_PER_INTENT = 100


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

INPUT_FILE = os.path.join(
    DATA_DIR,
    "multidomain_chatbot_dataset_FINAL.csv"
)

OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "multidomain_chatbot_dataset_FINAL.csv"
)


# ============================================================
# ADDITIONAL EXAMPLES
# ============================================================

ADDITIONAL_EXAMPLES = {

    "lost_or_stolen_card": [
        {
            "domain": "Finance",
            "intent": "lost_or_stolen_card",
            "user_query": "My debit card has been stolen",
            "response": (
                "If your card is lost or stolen, block it immediately "
                "through your bank's official app or contact the bank's "
                "customer support."
            )
        },
        {
            "domain": "Finance",
            "intent": "lost_or_stolen_card",
            "user_query": "I cannot find my bank card",
            "response": (
                "If your card is missing, block it immediately through "
                "your bank's official app or contact your bank."
            )
        },
        {
            "domain": "Finance",
            "intent": "lost_or_stolen_card",
            "user_query": "Someone took my ATM card",
            "response": (
                "Please block your ATM card immediately and report the "
                "incident to your bank through an official channel."
            )
        }
    ]
}


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 65)
    print("FINAL DATASET BALANCING")
    print("=" * 65)

    # --------------------------------------------------------
    # Check input file
    # --------------------------------------------------------

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"\nDataset not found:\n{INPUT_FILE}"
        )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"\nOriginal records: {len(df)}"
    )

    # --------------------------------------------------------
    # Normalize columns
    # --------------------------------------------------------

    required_columns = [
        "domain",
        "intent",
        "user_query",
        "response"
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Missing required column: {column}"
            )

    # --------------------------------------------------------
    # Clean queries
    # --------------------------------------------------------

    df["user_query"] = (
        df["user_query"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Remove duplicate queries
    # --------------------------------------------------------

    df = df.drop_duplicates(
        subset=["user_query"],
        keep="first"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Add source if missing
    # --------------------------------------------------------

    if "source" not in df.columns:

        df["source"] = (
            "existing_project_dataset"
        )

    # --------------------------------------------------------
    # Add split if missing
    # --------------------------------------------------------

    if "split" not in df.columns:

        df["split"] = "train"

    # --------------------------------------------------------
    # Add missing examples
    # --------------------------------------------------------

    for intent, examples in ADDITIONAL_EXAMPLES.items():

        for example in examples:

            query_exists = (
                df["user_query"]
                .str.lower()
                .eq(
                    example["user_query"].lower()
                )
                .any()
            )

            if not query_exists:

                new_row = {
                    "domain":
                        example["domain"],

                    "intent":
                        example["intent"],

                    "user_query":
                        example["user_query"],

                    "response":
                        example["response"],

                    "source":
                        "curated_project_extension",

                    "split":
                        "train"
                }

                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            [new_row]
                        )
                    ],
                    ignore_index=True
                )

    # --------------------------------------------------------
    # Recreate IDs
    # --------------------------------------------------------

    df["id"] = range(
        1,
        len(df) + 1
    )

    # --------------------------------------------------------
    # Reorder columns
    # --------------------------------------------------------

    df = df[
        [
            "id",
            "domain",
            "intent",
            "user_query",
            "response",
            "source",
            "split"
        ]
    ]

    # --------------------------------------------------------
    # Display intent distribution
    # --------------------------------------------------------

    print("\nIntent distribution BEFORE balancing:")

    counts = (
        df["intent"]
        .value_counts()
        .sort_index()
    )

    print(
        counts.to_string()
    )

    # --------------------------------------------------------
    # Check all intents
    # --------------------------------------------------------

    insufficient = (
        counts[
            counts < TARGET_PER_INTENT
        ]
    )

    if len(insufficient) > 0:

        print("\nWARNING:")
        print(
            "The following intents have fewer "
            f"than {TARGET_PER_INTENT} examples:"
        )

        print(
            insufficient.to_string()
        )

    else:

        print(
            f"\nAll intents have at least "
            f"{TARGET_PER_INTENT} examples."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    final_counts = (
        df["intent"]
        .value_counts()
        .sort_index()
    )

    print()
    print("=" * 65)
    print("FINAL DATASET")
    print("=" * 65)

    print(
        f"\nTotal records: {len(df)}"
    )

    print(
        f"Total domains: "
        f"{df['domain'].nunique()}"
    )

    print(
        f"Total intents: "
        f"{df['intent'].nunique()}"
    )

    print(
        "\nIntent distribution:"
    )

    print(
        final_counts.to_string()
    )

    print(
        "\nDataset saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print("=" * 65)
    print("DATASET FIX COMPLETED")
    print("=" * 65)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()