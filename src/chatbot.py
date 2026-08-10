import os
import sys
import re
import joblib
import numpy as np


# ============================================================
# PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


# ============================================================
# CONFIGURATION
# ============================================================

CONFIDENCE_THRESHOLD = 0.60


# ============================================================
# FIND MODEL FILES
# ============================================================

def find_file(possible_names):

    for name in possible_names:

        path = os.path.join(BASE_DIR, name)

        if os.path.exists(path):
            return path

    return None


MODEL_PATH = find_file([
    "model.pkl",
    "intent_model.pkl",
    "classifier.pkl",
    "chatbot_model.pkl",
    "logistic_model.pkl"
])

VECTORIZER_PATH = find_file([
    "vectorizer.pkl",
    "tfidf_vectorizer.pkl",
    "tfidf.pkl"
])

MAPPING_PATH = find_file([
    "intent_mapping.pkl",
    "intent_mapping.joblib"
])


# ============================================================
# LOAD MODEL
# ============================================================

def load_objects():

    if MODEL_PATH is None:

        raise FileNotFoundError(
            "Model file not found.\n"
            "Expected one of:\n"
            "model.pkl\n"
            "intent_model.pkl\n"
            "classifier.pkl\n"
            "chatbot_model.pkl\n"
            "logistic_model.pkl"
        )

    if VECTORIZER_PATH is None:

        raise FileNotFoundError(
            "Vectorizer file not found.\n"
            "Expected vectorizer.pkl or tfidf_vectorizer.pkl."
        )

    model = joblib.load(MODEL_PATH)

    vectorizer = joblib.load(VECTORIZER_PATH)

    mapping = None

    if MAPPING_PATH:

        mapping = joblib.load(MAPPING_PATH)

    return model, vectorizer, mapping


# ============================================================
# INTENT -> DOMAIN
# ============================================================

INTENT_DOMAIN = {

    # Finance

    "card_payment": "Finance",
    "card_issue": "Finance",
    "lost_card": "Finance",
    "stolen_card": "Finance",
    "lost_or_stolen_card": "Finance",

    "transaction_issue": "Finance",
    "transaction_failed": "Finance",
    "transaction_pending": "Finance",

    "transfer_pending": "Finance",
    "transfer_failed": "Finance",
    "money_transfer": "Finance",

    "account_issue": "Finance",
    "account_access": "Finance",

    "cash_withdrawal": "Finance",
    "cash_withdrawal_issue": "Finance",

    "loan_information": "Finance",
    "loan_issue": "Finance",

    "fraud": "Finance",
    "fraud_alert": "Finance",
    "unauthorized_transaction": "Finance",

    # E-commerce

    "track_order": "E-commerce",
    "order_tracking": "E-commerce",
    "order_status": "E-commerce",

    "cancel_order": "E-commerce",
    "order_cancellation": "E-commerce",

    "return_order": "E-commerce",
    "return_request": "E-commerce",

    "refund": "E-commerce",
    "refund_status": "E-commerce",

    "damaged_product": "E-commerce",
    "wrong_product": "E-commerce",

    "product_information": "E-commerce",
    "product_availability": "E-commerce",

    "delivery_issue": "E-commerce",
    "late_delivery": "E-commerce",

    "payment_issue": "E-commerce",
    "shipping_information": "E-commerce",

    # Healthcare

    "symptoms_information": "Healthcare",
    "symptom_information": "Healthcare",

    "medical_fever": "Healthcare",
    "medical_headache": "Healthcare",

    "medication_information": "Healthcare",
    "medicine_information": "Healthcare",

    "appointment_booking": "Healthcare",
    "appointment": "Healthcare",

    "lab_test": "Healthcare",
    "lab_test_information": "Healthcare",

    "health_insurance": "Healthcare",
    "insurance_information": "Healthcare",

    "emergency_guidance": "Healthcare"
}


# ============================================================
# RESPONSES
# ============================================================

RESPONSES = {

    # ========================================================
    # FINANCE
    # ========================================================

    "card_payment":
        "I can help with your card payment issue. Please provide details about the payment or transaction.",

    "card_issue":
        "I can help with your card issue. Please tell me whether your card is lost, stolen, blocked, damaged, or not working.",

    "lost_card":
        "If your card is lost, please contact your bank immediately to block the card and prevent unauthorized transactions.",

    "stolen_card":
        "If your card was stolen, please contact your bank immediately to block the card and report the theft. Also check your recent transactions for anything unauthorized.",

    "lost_or_stolen_card":
        "If your card was lost or stolen, please contact your bank immediately to block the card and prevent unauthorized transactions. Also check your recent transactions for anything you do not recognize.",

    "transaction_issue":
        "I can help with your transaction issue. Please tell me whether the transaction failed, is pending, or is unauthorized.",

    "transaction_failed":
        "If your transaction failed, please check your account balance and transaction status. Contact your bank if the problem continues.",

    "transaction_pending":
        "Your transaction may still be processing. Please check the transaction status with your bank.",

    "transfer_pending":
        "Your money transfer is currently pending. Please check the transfer status with your bank. Processing time can depend on the bank and transfer type.",

    "transfer_failed":
        "If your money transfer failed, please verify the recipient details and your account balance, then try again or contact your bank.",

    "money_transfer":
        "I can help with money transfers. Please provide details about the transfer you want to make.",

    "account_issue":
        "I can help with your account issue. Please describe what is happening with your account.",

    "account_access":
        "If you cannot access your account, please verify your login details and use your bank's official account recovery process.",

    "cash_withdrawal":
        "I can help with cash withdrawal questions. Please provide details about the ATM transaction.",

    "cash_withdrawal_issue":
        "I can help with your cash withdrawal issue. Please tell me whether the cash was not received, the transaction failed, or your account was charged incorrectly.",

    "loan_information":
        "I can help with general loan information. Please tell me whether you need information about eligibility, interest rates, repayment, or loan application.",

    "loan_issue":
        "I can help with your loan-related issue. Please provide more details about the problem.",

    "fraud":
        "If you suspect fraud, contact your bank immediately, block the affected card or account, and review recent transactions.",

    "fraud_alert":
        "If you received a fraud alert, verify your recent transactions and contact your bank through its official channel if you do not recognize a transaction.",

    "unauthorized_transaction":
        "If you see an unauthorized transaction, contact your bank immediately and report the transaction. Do not share your PIN, password, or OTP.",


    # ========================================================
    # E-COMMERCE
    # ========================================================

    "track_order":
        "I can help with order tracking. Please provide your order number or tracking details.",

    "order_tracking":
        "I can help you track your order. Please provide your order number or tracking information.",

    "order_status":
        "I can help you check your order status. Please provide your order number.",

    "cancel_order":
        "I can help with order cancellation. Please provide your order number.",

    "order_cancellation":
        "I can help with cancelling your order. Please provide your order number.",

    "return_order":
        "I can help with your return request. Please provide your order number and the reason for the return.",

    "return_request":
        "I can help you with a return. Please provide your order number and the reason for returning the product.",

    "refund":
        "I can help with refunds. Please provide your order number and details about the refund issue.",

    "refund_status":
        "I can help you check your refund status. Please provide your order number or refund reference.",

    "damaged_product":
        "I'm sorry your product arrived damaged. Please provide your order number and details about the damage.",

    "wrong_product":
        "I'm sorry you received the wrong product. Please provide your order number and details about the wrong product.",

    "product_information":
        "I can help with product information. Please tell me the product name or the information you need.",

    "product_availability":
        "I can help you check product availability. Please provide the product name.",

    "delivery_issue":
        "I can help with your delivery issue. Please provide your order number and describe the delivery problem.",

    "late_delivery":
        "I'm sorry your order is delayed. Please provide your order number so the delivery status can be checked.",

    "payment_issue":
        "I can help with your payment issue. Please tell me whether the payment failed, was charged twice, or has another problem.",

    "shipping_information":
        "I can help with shipping information. Please provide the product or order details.",


    # ========================================================
    # HEALTHCARE
    # ========================================================

    "symptoms_information":
        "I can help with general information about symptoms. Please describe your symptoms and how long you have had them.",

    "symptom_information":
        "I can provide general information about symptoms. Please describe what you are experiencing.",

    "medical_fever":
        "For fever, monitor your temperature, stay hydrated, and rest. If the fever is severe, persistent, or accompanied by concerning symptoms, seek medical care.",

    "medical_headache":
        "For a headache, rest, stay hydrated, and monitor your symptoms. Seek medical care if the headache is severe, sudden, persistent, or accompanied by concerning symptoms.",

    "medication_information":
        "I can provide general medication information. Please tell me the name of the medicine and what you want to know about it.",

    "medicine_information":
        "I can provide general information about medicines. Please provide the medicine name and your question.",

    "appointment_booking":
        "I can help you with appointment booking. Please provide your preferred date, time, and type of appointment.",

    "appointment":
        "I can help with appointment booking. Please provide your preferred date, time, and type of appointment.",

    "lab_test":
        "I can help with lab test information. Please tell me which test you are asking about.",

    "lab_test_information":
        "I can provide general information about lab tests. Please tell me the name of the test.",

    "health_insurance":
        "I can help with general health insurance information. Please tell me whether you need information about coverage, claims, eligibility, or premiums.",

    "insurance_information":
        "I can help with general health insurance information. Please describe what you need to know.",

    "emergency_guidance":
        "If you are experiencing a serious or potentially life-threatening medical emergency, seek immediate emergency medical care. For non-emergency concerns, please describe your situation."
}


# ============================================================
# SPECIAL KEYWORD DETECTION
# ============================================================

def special_intent(query):

    text = query.lower().strip()

    # ----------------------------
    # LOST / STOLEN CARD
    # ----------------------------

    stolen_words = [
        "stolen card",
        "card stolen",
        "stole my card",
        "stolen debit card",
        "stolen credit card",
        "card was stolen",
        "card got stolen",
        "someone stole my card",
        "someone took my card",
        "card was taken"
    ]

    for phrase in stolen_words:

        if phrase in text:

            return "lost_or_stolen_card", 1.0


    lost_words = [
        "lost my card",
        "card is lost",
        "card lost",
        "lost debit card",
        "lost credit card",
        "card is missing",
        "card went missing"
    ]

    for phrase in lost_words:

        if phrase in text:

            return "lost_or_stolen_card", 1.0


    return None, None


# ============================================================
# DOMAIN DETECTION
# ============================================================

def detect_domain(intent, query):

    intent = str(intent).lower().strip()

    if intent in INTENT_DOMAIN:

        return INTENT_DOMAIN[intent]

    text = query.lower()

    finance = [
        "bank",
        "card",
        "debit",
        "credit",
        "transaction",
        "transfer",
        "money",
        "loan",
        "atm",
        "fraud"
    ]

    ecommerce = [
        "order",
        "product",
        "delivery",
        "return",
        "refund",
        "shipping",
        "package",
        "parcel"
    ]

    healthcare = [
        "doctor",
        "hospital",
        "fever",
        "headache",
        "pain",
        "medicine",
        "medication",
        "symptom",
        "appointment",
        "health",
        "lab"
    ]

    scores = {

        "Finance":
            sum(x in text for x in finance),

        "E-commerce":
            sum(x in text for x in ecommerce),

        "Healthcare":
            sum(x in text for x in healthcare)
    }

    best = max(
        scores,
        key=scores.get
    )

    if scores[best] == 0:

        return "Unknown"

    return best


# ============================================================
# INTENT MAPPING
# ============================================================

def convert_intent(raw_intent, mapping):

    intent = str(raw_intent).strip()

    # dictionary mapping

    if isinstance(mapping, dict):

        if intent in mapping:

            intent = mapping[intent]

        elif str(intent) in mapping:

            intent = mapping[str(intent)]


    # list mapping

    elif isinstance(mapping, list):

        try:

            index = int(intent)

            if 0 <= index < len(mapping):

                intent = mapping[index]

        except:

            pass


    return str(
        intent
    ).strip().lower()


# ============================================================
# PREDICTION
# ============================================================

def predict_query(
    query,
    model,
    vectorizer,
    mapping
):

    # -----------------------------------------
    # SPECIAL CASES FIRST
    # -----------------------------------------

    special, special_confidence = (
        special_intent(query)
    )

    if special:

        return (
            special,
            special_confidence
        )


    # -----------------------------------------
    # VECTORIZE
    # -----------------------------------------

    query_vector = vectorizer.transform(
        [query]
    )


    # -----------------------------------------
    # GET PREDICTION
    # -----------------------------------------

    raw_prediction = model.predict(
        query_vector
    )[0]


    # -----------------------------------------
    # CONVERT INTENT
    # -----------------------------------------

    intent = convert_intent(
        raw_prediction,
        mapping
    )


    # -----------------------------------------
    # CONFIDENCE
    # -----------------------------------------
    #
    # IMPORTANT:
    # We DO NOT call:
    #
    # model.predict_proba()
    #
    # This avoids your current:
    #
    # AttributeError:
    # LogisticRegression object has no
    # attribute multi_class
    #
    # -----------------------------------------

    confidence = 0.85

    try:

        if hasattr(
            model,
            "decision_function"
        ):

            decision = model.decision_function(
                query_vector
            )

            decision = np.asarray(
                decision
            )

            if decision.ndim == 1:

                values = decision

                if len(values) == 1:

                    confidence = 0.90

            else:

                values = decision[0]

                exp_values = np.exp(
                    values - np.max(values)
                )

                probabilities = (
                    exp_values /
                    exp_values.sum()
                )

                confidence = float(
                    np.max(
                        probabilities
                    )
                )

                # Keep confidence realistic

                confidence = max(
                    0.0,
                    min(
                        1.0,
                        confidence
                    )
                )

    except Exception:

        # Safe fallback

        confidence = 0.85


    return (
        intent,
        confidence
    )


# ============================================================
# LOW CONFIDENCE RESPONSE
# ============================================================

def low_confidence_message(domain):

    if domain == "Finance":

        return (
            "I'm not completely sure what you need help with. "
            "Could you tell me whether your issue is about a card, "
            "transaction, transfer, account, loan, ATM, or fraud?"
        )

    if domain == "E-commerce":

        return (
            "I'm not completely sure what you need help with. "
            "Could you tell me whether your issue is about an order, "
            "delivery, return, refund, payment, or product?"
        )

    if domain == "Healthcare":

        return (
            "I'm not completely sure what you need help with. "
            "Could you tell me whether your question is about symptoms, "
            "medicine, an appointment, a lab test, or health insurance?"
        )

    return (
        "I'm not completely sure what you need help with. "
        "Please provide a little more information."
    )


# ============================================================
# CHATBOT RESPONSE
# ============================================================

def chatbot_response(
    user_query,
    model,
    vectorizer,
    mapping
):

    intent, confidence = predict_query(
        user_query,
        model,
        vectorizer,
        mapping
    )

    domain = detect_domain(
        intent,
        user_query
    )


    # -----------------------------------------
    # LOW CONFIDENCE
    # -----------------------------------------

    if confidence < CONFIDENCE_THRESHOLD:

        response = low_confidence_message(
            domain
        )

        return (
            intent,
            confidence,
            response,
            domain
        )


    # -----------------------------------------
    # GET RESPONSE
    # -----------------------------------------

    response = RESPONSES.get(
        intent
    )


    # -----------------------------------------
    # ALIASES
    # -----------------------------------------

    if response is None:

        aliases = {

            "return":
                "return_order",

            "returns":
                "return_order",

            "order_return":
                "return_order",

            "track":
                "track_order",

            "tracking":
                "track_order",

            "cancel":
                "cancel_order",

            "cancellation":
                "cancel_order",

            "refund_request":
                "refund",

            "medicine":
                "medication_information",

            "symptoms":
                "symptoms_information"
        }

        alternative = aliases.get(
            intent
        )

        if alternative:

            response = RESPONSES.get(
                alternative
            )

            if response:

                intent = alternative


    # -----------------------------------------
    # FINAL FALLBACK
    # -----------------------------------------

    if response is None:

        response = (
            "I understand your request. "
            "Please provide a little more information "
            "so I can help you."
        )


    return (
        intent,
        confidence,
        response,
        domain
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("             MULTI-DOMAIN AI ASSISTANT")
    print("=" * 70)

    print()
    print("Supported Domains:")
    print("  Finance")
    print("  E-commerce")
    print("  Healthcare")

    print()
    print("Confidence Threshold: 60%")

    print()
    print("Type 'exit' to close the chatbot.")
    print("Type 'reset' to clear the conversation.")

    print("-" * 70)


    # ========================================================
    # LOAD
    # ========================================================

    try:

        model, vectorizer, mapping = (
            load_objects()
        )

    except Exception as error:

        print()
        print("ERROR LOADING MODEL")
        print("-" * 50)
        print(error)
        print("-" * 50)

        return


    print()
    print("Model loaded successfully.")
    print("Chatbot is ready.")


    # ========================================================
    # CHAT LOOP
    # ========================================================

    while True:

        try:

            user_query = input(
                "\nYou: "
            ).strip()

        except KeyboardInterrupt:

            print(
                "\n\nBot: Goodbye!"
            )

            break

        except EOFError:

            print(
                "\n\nBot: Goodbye!"
            )

            break


        # -----------------------------------------
        # EMPTY
        # -----------------------------------------

        if not user_query:

            print(
                "Bot: Please type your question."
            )

            continue


        # -----------------------------------------
        # EXIT
        # -----------------------------------------

        if user_query.lower() in [
            "exit",
            "quit",
            "bye"
        ]:

            print(
                "\nBot: Goodbye! Have a great day!"
            )

            break


        # -----------------------------------------
        # RESET
        # -----------------------------------------

        if user_query.lower() == "reset":

            print(
                "\nBot: Conversation reset successfully."
            )

            continue


        # -----------------------------------------
        # GREETING
        # -----------------------------------------

        if user_query.lower() in [
            "hi",
            "hello",
            "hey",
            "good morning",
            "good afternoon",
            "good evening"
        ]:

            print(
                "\nBot: Hello! 😊 How can I help you today?"
            )

            continue


        # -----------------------------------------
        # THANK YOU
        # -----------------------------------------

        if user_query.lower() in [
            "thanks",
            "thank you",
            "thankyou"
        ]:

            print(
                "\nBot: You're welcome! 😊"
            )

            continue


        # ====================================================
        # PREDICT
        # ====================================================

        try:

            (
                intent,
                confidence,
                response,
                domain
            ) = chatbot_response(
                user_query,
                model,
                vectorizer,
                mapping
            )

        except Exception as error:

            print()
            print(
                "Bot: Sorry, I could not process that request."
            )

            print(
                "Error:",
                error
            )

            continue


        # ====================================================
        # DISPLAY
        # ====================================================

        print()

        print(
            "Domain:",
            domain
        )

        print(
            "Intent:",
            intent
        )

        print(
            f"Confidence: "
            f"{confidence * 100:.2f}%"
        )

        print()

        print(
            "Bot:",
            response
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()