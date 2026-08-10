import os
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

INPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "multidomain_chatbot_dataset_FINAL_v2.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "multidomain_chatbot_dataset_FINAL.csv"
)


# ============================================================
# TARGET SIZE
# ============================================================

SAMPLES_PER_INTENT = 100

RANDOM_STATE = 42


# ============================================================
# ADDITIONAL EXAMPLES
# ============================================================

extra_examples = {

    # --------------------------------------------------------
    # FINANCE
    # --------------------------------------------------------

    "lost_or_stolen_card": [
        "I lost my bank card",
        "My debit card is missing",
        "Someone stole my card",
        "I cannot find my credit card",
        "My ATM card was stolen",
        "I misplaced my bank card",
        "My credit card is missing",
        "What should I do if I lose my card?",
        "My debit card disappeared",
        "I lost my ATM card",
        "My banking card is gone",
        "I think my card was stolen",
        "I cannot find my debit card",
        "My card has been lost",
        "Someone took my bank card",
        "How do I report a lost card?",
        "What should I do about a stolen card?",
        "My credit card was stolen",
        "My card is missing",
        "I need to block my lost card",
        "I lost my payment card",
        "My card was taken",
        "My bank card disappeared",
        "I need help with a stolen card",
        "I have misplaced my credit card",
        "My debit card cannot be found",
        "I want to report my card missing",
        "My ATM card disappeared",
        "I lost my banking card yesterday",
        "Someone stole my debit card",
        "My card is nowhere to be found",
        "I need to block my stolen card",
        "My card went missing",
        "How can I block a lost card?",
        "My credit card is nowhere",
        "I accidentally lost my card",
        "My debit card was stolen",
        "I need help because my card is missing",
        "What happens if my card gets stolen?",
        "I cannot locate my ATM card"
    ],

    # --------------------------------------------------------
    # E-COMMERCE
    # --------------------------------------------------------

    "damaged_product": [
        "The product arrived damaged",
        "My item is broken",
        "I received a defective product",
        "The product I received is faulty",
        "My delivery was damaged",
        "The item does not work",
        "The product is damaged",
        "My package contains a broken item",
        "The item arrived broken",
        "I received a damaged item",
        "The product is defective",
        "My order arrived damaged",
        "The item is not working",
        "I got a faulty product",
        "The package was damaged",
        "The product was broken during delivery",
        "I received a damaged package",
        "My new product is defective",
        "The item I ordered is broken",
        "There is damage to my product",
        "The product has been damaged",
        "My item arrived with damage",
        "I received a faulty item",
        "The package contains a damaged product",
        "The product does not work properly",
        "My order contains a broken product",
        "The item is defective",
        "The delivered item is damaged",
        "My product arrived in bad condition",
        "The product was damaged in transit",
        "The item I received is faulty",
        "I want to report a damaged product",
        "My purchase arrived broken",
        "The item is damaged and unusable",
        "The product has a defect",
        "My delivery contains a broken item",
        "The product arrived with damage",
        "The item I purchased is faulty",
        "I got a damaged item",
        "My order was delivered damaged"
    ]
}


# ============================================================
# RESPONSE TEMPLATES
# ============================================================

responses = {

    "lost_or_stolen_card":
        "If your card is lost or stolen, block it immediately through your bank's official app or contact the bank's customer support.",

    "damaged_product":
        "I'm sorry the product arrived damaged. Please use the order's return or replacement option and provide the required order details."
}


# ============================================================
# CREATE VARIATIONS
# ============================================================

def create_variations(base_examples, target_count):

    examples = list(dict.fromkeys(base_examples))

    if len(examples) >= target_count:
        return examples[:target_count]

    prefixes = [
        "Please help, ",
        "I need help: ",
        "Can you help? ",
        "I have a problem: ",
        "I need assistance. ",
        "Could you help me? ",
        "Please assist me. "
    ]

    suffixes = [
        "",
        " Please help.",
        " What should I do?",
        " Can you assist?",
        " I need help with this.",
        " Please tell me what to do."
    ]

    generated = []

    for example in examples:

        generated.append(example)

        for prefix in prefixes:

            candidate = prefix + example

            if candidate not in generated:
                generated.append(candidate)

            if len(generated) >= target_count:
                return generated[:target_count]

        for suffix in suffixes:

            candidate = example + suffix

            if candidate not in generated:
                generated.append(candidate)

            if len(generated) >= target_count:
                return generated[:target_count]

    return generated[:target_count]


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 65)
    print("FINAL MULTI-DOMAIN CHATBOT DATASET CREATION")
    print("=" * 65)

    # --------------------------------------------------------
    # Load existing dataset
    # --------------------------------------------------------

    if not os.path.exists(INPUT_PATH):

        raise FileNotFoundError(
            f"\nInput dataset not found:\n{INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print(
        f"\nInput records: {len(df)}"
    )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    df["user_query"] = (
        df["user_query"]
        .astype(str)
        .str.strip()
    )

    df = df.drop_duplicates(
        subset=["user_query"],
        keep="first"
    )

    # --------------------------------------------------------
    # Build final dataset intent by intent
    # --------------------------------------------------------

    final_parts = []

    all_intents = sorted(
        df["intent"].unique()
    )

    print(
        f"Existing intents: {len(all_intents)}"
    )

    for intent in all_intents:

        intent_df = df[
            df["intent"] == intent
        ].copy()

        current_count = len(intent_df)

        # ----------------------------------------------------
        # Already enough examples
        # ----------------------------------------------------

        if current_count >= SAMPLES_PER_INTENT:

            selected = intent_df.sample(
                n=SAMPLES_PER_INTENT,
                random_state=RANDOM_STATE
            )

            final_parts.append(
                selected
            )

            continue

        # ----------------------------------------------------
        # Need additional examples
        # ----------------------------------------------------

        additional_needed = (
            SAMPLES_PER_INTENT -
            current_count
        )

        print(
            f"{intent}: "
            f"{current_count} → "
            f"{SAMPLES_PER_INTENT}"
        )

        if intent in extra_examples:

            examples = create_variations(
                extra_examples[intent],
                additional_needed
            )

        else:

            # Use existing examples to create
            # controlled text variations.
            base_queries = (
                intent_df["user_query"]
                .tolist()
            )

            examples = create_variations(
                base_queries,
                additional_needed
            )

        # ----------------------------------------------------
        # Create additional rows
        # ----------------------------------------------------

        new_rows = []

        for query in examples:

            if query in set(
                intent_df["user_query"]
            ):
                continue

            domain = intent_df[
                "domain"
            ].mode()[0]

            if intent in responses:

                response = responses[intent]

            else:

                response = intent_df[
                    "response"
                ].iloc[0]

            new_rows.append({
                "domain": domain,
                "intent": intent,
                "user_query": query,
                "response": response,
                "source": "curated_project_extension"
            })

        if new_rows:

            additional_df = pd.DataFrame(
                new_rows
            )

            intent_df = pd.concat(
                [
                    intent_df,
                    additional_df
                ],
                ignore_index=True
            )

        # ----------------------------------------------------
        # Final sampling
        # ----------------------------------------------------

        if len(intent_df) >= SAMPLES_PER_INTENT:

            intent_df = intent_df.sample(
                n=SAMPLES_PER_INTENT,
                random_state=RANDOM_STATE
            )

        else:

            print(
                f"WARNING: {intent} has only "
                f"{len(intent_df)} examples."
            )

        final_parts.append(
            intent_df
        )

    # --------------------------------------------------------
    # Combine all intents
    # --------------------------------------------------------

    final_df = pd.concat(
        final_parts,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Shuffle
    # --------------------------------------------------------

    final_df = final_df.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Create IDs
    # --------------------------------------------------------

    final_df["id"] = range(
        1,
        len(final_df) + 1
    )

    # --------------------------------------------------------
    # Create train / validation / test
    # --------------------------------------------------------

    final_df["split"] = ""

    for intent in final_df["intent"].unique():

        indices = final_df[
            final_df["intent"] == intent
        ].index

        train_idx, temp_idx = train_test_split(
            indices,
            test_size=0.20,
            random_state=RANDOM_STATE
        )

        validation_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.50,
            random_state=RANDOM_STATE
        )

        final_df.loc[
            train_idx,
            "split"
        ] = "train"

        final_df.loc[
            validation_idx,
            "split"
        ] = "validation"

        final_df.loc[
            test_idx,
            "split"
        ] = "test"

    # --------------------------------------------------------
    # Column order
    # --------------------------------------------------------

    final_df = final_df[
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
    # Save
    # --------------------------------------------------------

    final_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("FINAL DATASET STATISTICS")
    print("=" * 65)

    print(
        f"\nTotal records: "
        f"{len(final_df)}"
    )

    print(
        f"Domains: "
        f"{final_df['domain'].nunique()}"
    )

    print(
        f"Intents: "
        f"{final_df['intent'].nunique()}"
    )

    print("\nDomain distribution:")

    print(
        final_df[
            "domain"
        ].value_counts()
    )

    print("\nIntent distribution:")

    print(
        final_df[
            "intent"
        ].value_counts()
        .sort_index()
        .to_string()
    )

    print("\nDataset split:")

    print(
        final_df[
            "split"
        ].value_counts()
    )

    print(
        f"\nFinal dataset saved to:\n"
        f"{OUTPUT_PATH}"
    )

    print("\n")
    print("=" * 65)
    print("FINAL DATASET CREATED SUCCESSFULLY")
    print("=" * 65)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()