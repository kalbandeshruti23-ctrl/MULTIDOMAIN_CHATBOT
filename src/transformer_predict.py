import os
import json
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ============================================================
# CONFIGURATION
# ============================================================

CONFIDENCE_THRESHOLD = 0.60
MAX_LENGTH = 128


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "distilbert_intent_model"
)


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading Transformer model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR
)


# ============================================================
# LOAD MODEL
# ============================================================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)

model.eval()


# ============================================================
# LOAD LABEL MAPPING
# ============================================================

LABEL_PATH = os.path.join(
    MODEL_DIR,
    "id2label.json"
)

if not os.path.exists(LABEL_PATH):

    raise FileNotFoundError(
        f"id2label.json not found:\n{LABEL_PATH}"
    )


with open(
    LABEL_PATH,
    "r",
    encoding="utf-8"
) as file:

    id2label = json.load(file)


# ============================================================
# DOMAIN MAPPING
# ============================================================

FINANCE_INTENTS = {

    "balance_inquiry",
    "bank_account",
    "card_payment",
    "cash_withdrawal",
    "credit_card",
    "fraud_transaction",
    "loan_information",
    "lost_or_stolen_card",
    "transaction_issue",
    "transfer_failed",
    "transfer_pending"
}


ECOMMERCE_INTENTS = {

    "account_issue",
    "cancel_order",
    "damaged_product",
    "delivery_time",
    "payment_failed",
    "product_information",
    "refund",
    "return_order",
    "track_order",
    "wrong_product"
}


HEALTHCARE_INTENTS = {

    "appointment_booking",
    "appointment_cancel",
    "appointment_reschedule",
    "emergency_guidance",
    "health_insurance",
    "lab_test",
    "medical_report",
    "medication_information",
    "prescription",
    "symptoms_information"
}


# ============================================================
# RESPONSE DATABASE
# ============================================================

RESPONSES = {

    # --------------------------------------------------------
    # FINANCE
    # --------------------------------------------------------

    "balance_inquiry":
        "You can check your account balance through your bank's official mobile app, website, ATM, or branch.",

    "bank_account":
        "I can help with general bank account questions. Please describe the account issue you are facing.",

    "card_payment":
        "I can help with card payment issues. Please provide details about the payment or transaction.",

    "cash_withdrawal":
        "I can help with cash withdrawal issues. Please describe what happened during the withdrawal.",

    "credit_card":
        "I can help with general credit card questions. Please tell me what you need assistance with.",

    "fraud_transaction":
        "If you notice an unauthorized transaction, contact your bank immediately through an official channel and report the transaction.",

    "loan_information":
        "I can provide general information about loans, including eligibility, repayment, and interest-related questions.",

    "lost_or_stolen_card":
        "If your card is lost or stolen, block it immediately through your bank's official app or contact your bank's customer support.",

    "transaction_issue":
        "I can help with transaction issues. Please provide more details about the transaction and what went wrong.",

    "transfer_failed":
        "If your transfer failed, verify the recipient and account details and check the transaction status with your bank.",

    "transfer_pending":
        "A pending transfer may take some time depending on the bank and transfer type. Please check the transaction status with your bank.",


    # --------------------------------------------------------
    # E-COMMERCE
    # --------------------------------------------------------

    "account_issue":
        "I can help with your account-related issue. Please describe the problem you are experiencing.",

    "cancel_order":
        "I can help with order cancellation. Please provide your order details and check whether cancellation is still available.",

    "damaged_product":
        "I'm sorry the product arrived damaged or defective. Please use the order's return or replacement option and provide your order details.",

    "delivery_time":
        "Delivery time depends on the product, seller, and delivery location. Please check the estimated delivery date in your order details.",

    "payment_failed":
        "If your payment failed, verify your payment details and try again or use another supported payment method.",

    "product_information":
        "I can provide information about a product. Please tell me the product name or what information you need.",

    "refund":
        "Refund processing time depends on the payment method and seller. Please check your order's refund status.",

    "return_order":
        "You can request a return through the order's return option, subject to the applicable return policy.",

    "track_order":
        "I can help with order tracking. Please provide your order number or tracking details.",

    "wrong_product":
        "I'm sorry you received the wrong product. Please check your order details and request a replacement or return.",


    # --------------------------------------------------------
    # HEALTHCARE
    # --------------------------------------------------------

    "appointment_booking":
        "I can help you with appointment booking. Please provide your preferred date, time, and type of appointment.",

    "appointment_cancel":
        "I can help with cancelling an appointment. Please provide your appointment details.",

    "appointment_reschedule":
        "I can help with rescheduling an appointment. Please provide your current appointment details and preferred new time.",

    "emergency_guidance":
        "If you are experiencing a medical emergency, seek immediate help from local emergency medical services or go to the nearest emergency department.",

    "health_insurance":
        "I can help with general health-insurance information. Please describe what you want to know.",

    "lab_test":
        "I can provide general information about laboratory tests. Please tell me which test you are asking about.",

    "medical_report":
        "I can help explain general medical-report terminology, but a qualified healthcare professional should interpret your results in clinical context.",

    "medication_information":
        "I can provide general information about medications. For personalized medication advice, please consult a qualified healthcare professional.",

    "prescription":
        "Prescription decisions should be made by a qualified healthcare professional. Please consult your doctor or pharmacist.",

    "symptoms_information":
        "I can provide general information about symptoms, but I cannot provide a medical diagnosis. For concerning symptoms, consult a qualified healthcare professional."
}


# ============================================================
# CLARIFICATION QUESTIONS
# ============================================================

CLARIFICATION_RESPONSES = {

    "Healthcare": (
        "I can help with your healthcare question. "
        "Could you tell me whether you need help with "
        "symptoms, medication, an appointment, a lab test, "
        "or health insurance?"
    ),

    "E-commerce": (
        "I'd be happy to help with your product or order issue. "
        "Could you tell me whether the problem is about a "
        "damaged product, wrong product, return/refund, "
        "delivery, payment, or product information?"
    ),

    "Finance": (
        "I can help with your banking question. "
        "Could you tell me whether the issue is about a "
        "card, transaction, transfer, account, cash withdrawal, "
        "loan, or fraud?"
    ),

    "Unknown": (
        "I'd be happy to help. Could you provide a little "
        "more detail about your question?"
    )
}


# ============================================================
# DOMAIN DETECTION
# ============================================================

def get_domain(intent):

    if intent in FINANCE_INTENTS:
        return "Finance"

    if intent in ECOMMERCE_INTENTS:
        return "E-commerce"

    if intent in HEALTHCARE_INTENTS:
        return "Healthcare"

    return "Unknown"


# ============================================================
# KEYWORD-BASED DOMAIN HINT
#
# This is NOT replacing the Transformer.
# It is used only to improve clarification for
# very vague queries.
# ============================================================

def detect_domain_hint(text):

    text = text.lower()

    finance_words = [
        "bank",
        "banking",
        "account",
        "card",
        "debit",
        "credit",
        "atm",
        "cash",
        "transfer",
        "transaction",
        "loan",
        "fraud",
        "money"
    ]

    ecommerce_words = [
        "product",
        "order",
        "delivery",
        "package",
        "item",
        "purchase",
        "return",
        "refund",
        "seller",
        "shopping",
        "payment"
    ]

    healthcare_words = [
        "health",
        "doctor",
        "hospital",
        "medicine",
        "medication",
        "symptom",
        "fever",
        "pain",
        "appointment",
        "medical",
        "prescription",
        "lab",
        "test",
        "insurance"
    ]

    finance_score = sum(
        1
        for word in finance_words
        if word in text
    )

    ecommerce_score = sum(
        1
        for word in ecommerce_words
        if word in text
    )

    healthcare_score = sum(
        1
        for word in healthcare_words
        if word in text
    )

    scores = {
        "Finance": finance_score,
        "E-commerce": ecommerce_score,
        "Healthcare": healthcare_score
    }

    best_domain = max(
        scores,
        key=scores.get
    )

    if scores[best_domain] == 0:
        return "Unknown"

    return best_domain


# ============================================================
# TRANSFORMER PREDICTION
# ============================================================

def predict_intent(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    )

    with torch.no_grad():

        outputs = model(
            **inputs
        )

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

    intent = id2label[
        str(predicted_id)
    ]

    domain = get_domain(
        intent
    )

    return (
        domain,
        intent,
        confidence
    )


# ============================================================
# RESPONSE GENERATION
# ============================================================

def get_response(intent):

    return RESPONSES.get(
        intent,
        "I'm sorry, I don't have a suitable response for that query yet."
    )


# ============================================================
# FOLLOW-UP CONTEXT
# ============================================================

def resolve_follow_up(
    user_input,
    previous_domain
):

    text = user_input.lower().strip()

    follow_up_words = [
        "it",
        "this",
        "that",
        "same",
        "again",
        "there",
        "please do it",
        "what about it",
        "tell me more"
    ]

    is_follow_up = any(
        phrase in text
        for phrase in follow_up_words
    )

    if is_follow_up and previous_domain:

        return (
            True,
            previous_domain
        )

    return (
        False,
        None
    )


# ============================================================
# CHATBOT
# ============================================================

def chatbot():

    print("\n")
    print("=" * 65)
    print("MULTI-DOMAIN TRANSFORMER CHATBOT")
    print("=" * 65)

    print("\nSupported domains:")
    print("1. Finance")
    print("2. E-commerce")
    print("3. Healthcare")

    print(
        "\nConfidence threshold:",
        f"{CONFIDENCE_THRESHOLD * 100:.0f}%"
    )

    print(
        "\nType 'exit' to stop the chatbot."
    )

    print(
        "Type 'reset' to clear conversation context."
    )

    # --------------------------------------------------------
    # Context variables
    # --------------------------------------------------------

    previous_domain = None
    previous_intent = None

    conversation_history = []

    # --------------------------------------------------------
    # Conversation loop
    # --------------------------------------------------------

    while True:

        try:

            user_input = input(
                "\nYou: "
            ).strip()

        except KeyboardInterrupt:

            print(
                "\n\nBot: Goodbye!"
            )

            break

        # ----------------------------------------------------
        # Empty input
        # ----------------------------------------------------

        if not user_input:

            print(
                "Bot: Please enter a query."
            )

            continue

        # ----------------------------------------------------
        # Exit
        # ----------------------------------------------------

        if user_input.lower() in {
            "exit",
            "quit",
            "bye"
        }:

            print(
                "Bot: Goodbye! Have a nice day."
            )

            break

        # ----------------------------------------------------
        # Reset context
        # ----------------------------------------------------

        if user_input.lower() == "reset":

            previous_domain = None
            previous_intent = None
            conversation_history = []

            print(
                "Bot: Conversation context has been reset."
            )

            continue

        # ----------------------------------------------------
        # Save user message
        # ----------------------------------------------------

        conversation_history.append(
            {
                "role": "user",
                "message": user_input
            }
        )

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        (
            domain,
            intent,
            confidence
        ) = predict_intent(
            user_input
        )

        # ----------------------------------------------------
        # Domain hint
        # ----------------------------------------------------

        domain_hint = detect_domain_hint(
            user_input
        )

        # ----------------------------------------------------
        # Follow-up detection
        # ----------------------------------------------------

        (
            is_follow_up,
            follow_up_domain
        ) = resolve_follow_up(
            user_input,
            previous_domain
        )

        if is_follow_up:

            domain = follow_up_domain

        # ----------------------------------------------------
        # Display prediction
        # ----------------------------------------------------

        print(
            f"\nDetected Domain : {domain}"
        )

        print(
            f"Detected Intent : {intent}"
        )

        print(
            f"Confidence      : "
            f"{confidence * 100:.2f}%"
        )

        # ----------------------------------------------------
        # LOW CONFIDENCE
        # ----------------------------------------------------

        if confidence < CONFIDENCE_THRESHOLD:

            if domain_hint != "Unknown":

                clarification = (
                    CLARIFICATION_RESPONSES[
                        domain_hint
                    ]
                )

                print(
                    f"\nBot: {clarification}"
                )

            else:

                print(
                    f"\nBot: "
                    f"{CLARIFICATION_RESPONSES['Unknown']}"
                )

            conversation_history.append(
                {
                    "role": "bot",
                    "message": "Clarification requested"
                }
            )

            continue

        # ----------------------------------------------------
        # HIGH CONFIDENCE
        # ----------------------------------------------------

        response = get_response(
            intent
        )

        print(
            f"\nBot: {response}"
        )

        # ----------------------------------------------------
        # Update context
        # ----------------------------------------------------

        previous_domain = domain
        previous_intent = intent

        conversation_history.append(
            {
                "role": "bot",
                "message": response,
                "domain": domain,
                "intent": intent,
                "confidence": confidence
            }
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    chatbot()