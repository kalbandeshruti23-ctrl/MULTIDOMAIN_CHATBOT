
import os
import sys
import streamlit as st
import base64

# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ============================================================
# IMPORT MODEL
# ============================================================

try:
    from predict import (
        load_model,
        load_intent_mapping,
        predict_intent
    )

except ImportError as e:
    st.error("Unable to import predict.py")
    st.code(str(e))
    st.stop()


# ============================================================
# CONFIGURATION
# ============================================================

CONFIDENCE_THRESHOLD = 0.60

# ============================================================
# BACKGROUND IMAGE
# ============================================================

def add_bg_from_local(image_file):

    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()

    st.markdown(
        f"""
        <style>

        .stApp {{
            background: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* Make containers slightly transparent */

        .main .block-container {{
            background: rgba(255,255,255,0.82);
            padding: 2rem;
            border-radius: 20px;
        }}

        section[data-testid="stSidebar"] {{
            background: rgba(20,20,20,0.88);
        }}

        section[data-testid="stSidebar"] * {{
            color: white !important;
        }}

        </style>
        """,
        unsafe_allow_html=True,
    )
# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Multi-Domain AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.set_page_config(
    page_title="Multi-Domain AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

add_bg_from_local("assets/ai_chat_background.png.png")
# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #f7f9fc;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
    }

    /* Main header */
    .main-header {
        background: linear-gradient(
            135deg,
            #4f46e5,
            #7c3aed
        );

        padding: 30px;
        border-radius: 20px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 8px 25px rgba(79, 70, 229, 0.20);
    }

    .main-header h1 {
        margin: 0;
        font-size: 34px;
        font-weight: 700;
    }

    .main-header p {
        margin-top: 8px;
        font-size: 16px;
        opacity: 0.9;
    }

    /* Domain cards */
    .card {
        background-color: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        min-height: 150px;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.04);
    }

    .card h3 {
        margin-bottom: 8px;
    }

    .card p {
        color: #6b7280;
        font-size: 14px;
    }

    /* Status */
    .status-high {
        color: #15803d;
        font-weight: 700;
    }
    .main-header{
    background: rgba(79,70,229,0.85);
    backdrop-filter: blur(10px);
}

.card{
    background: rgba(255,255,255,0.92);
    backdrop-filter: blur(8px);
}
/* ==========================
   CHAT TEXT
========================== */

.stChatMessage {
    background: rgba(255,255,255,0.95);
    border-radius: 15px;
    padding: 12px;
    margin-bottom: 12px;
}

/* User and bot message text */
.stChatMessage p,
.stChatMessage div,
.stMarkdown p {
    color: #000000 !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    line-height: 1.7;
}

/* Chat input */
.stChatInput input {
    font-size: 18px !important;
    font-weight: 600 !important;
    color: black !important;
}

/* Placeholder */
.stChatInput input::placeholder {
    color: #555 !important;
    font-weight: 600;
}

/* Assistant message */
[data-testid="stChatMessageContent"] {
    color: black !important;
    font-size: 18px !important;
    font-weight: 700 !important;
}

/* User message */
[data-testid="stChatMessageContent"] p {
    color: black !important;
    font-weight: 700 !important;
}

/* Markdown text */
.stMarkdown {
    color: black !important;
    font-size: 18px !important;
    font-weight: 700 !important;
}
    .status-low {
        color: #b45309;
        font-weight: 700;
    }

    /* Details */
    .details {
        font-size: 13px;
        color: #64748b;
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 8px 12px;
        margin-top: 8px;
        border: 1px solid #e5e7eb;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_chatbot():

    vectorizer, model = load_model()

    intent_mapping = load_intent_mapping()

    return vectorizer, model, intent_mapping


try:

    vectorizer, model, intent_mapping = load_chatbot()

except Exception as e:

    st.error("Model could not be loaded.")

    st.code(str(e))

    st.info(
        "Check that your model files are inside the models folder."
    )

    st.stop()


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
# DOMAIN DETECTION
# ============================================================

def get_domain(intent):

    if intent is None:
        return None

    intent = str(intent).strip().lower()

    return INTENT_DOMAIN.get(intent)


def detect_domain_from_query(query):

    query = query.lower()

    finance_words = [
        "bank", "card", "debit", "credit",
        "transaction", "transfer", "money",
        "loan", "cash", "atm", "account",
        "fraud", "stolen", "lost"
    ]

    ecommerce_words = [
        "product", "order", "delivery",
        "return", "refund", "shipping",
        "package", "parcel", "shopping"
    ]

    healthcare_words = [
        "health", "doctor", "hospital",
        "fever", "headache", "pain",
        "medicine", "medication",
        "symptom", "appointment",
        "lab", "disease", "sick"
    ]

    scores = {
        "Finance": sum(
            word in query
            for word in finance_words
        ),

        "E-commerce": sum(
            word in query
            for word in ecommerce_words
        ),

        "Healthcare": sum(
            word in query
            for word in healthcare_words
        )
    }

    best_domain = max(
        scores,
        key=scores.get
    )

    if scores[best_domain] == 0:
        return None

    return best_domain


# ============================================================
# STOLEN / LOST CARD DETECTION
# ============================================================

def detect_critical_finance_intent(query):

    query = query.lower().strip()

    phrases = [

        "stole my card",
        "card was stolen",
        "card got stolen",
        "card is stolen",
        "card was taken",
        "card got taken",
        "card has been stolen",
        "card has been taken",

        "someone took my card",
        "someone stole my card",
        "someone has taken my card",
        "someone has stolen my card",

        "my card is missing",
        "my card went missing",

        "debit card stolen",
        "credit card stolen",

        "debit card was stolen",
        "credit card was stolen",

        "debit card was taken",
        "credit card was taken",

        "lost my card",
        "i lost my card",
        "my card is lost",

        "lost debit card",
        "lost credit card",

        "debit card is missing",
        "credit card is missing"
    ]

    for phrase in phrases:

        if phrase in query:
            return "lost_or_stolen_card"

    return None


# ============================================================
# BOT RESPONSES
# ============================================================

RESPONSES = {

    "card_payment":
        "I can help with card payment issues. Please provide details about the payment or transaction.",

    "card_issue":
        "I can help with your card issue. Please tell me whether your card is lost, stolen, blocked, damaged, or not working.",

    "lost_card":
        "If your card is lost, please contact your bank immediately to block the card and prevent unauthorized transactions.",

    "stolen_card":
        "If your card was stolen, please contact your bank immediately to block the card and report the theft.",

    "lost_or_stolen_card":
        "If your card was lost or stolen, please contact your bank immediately to block the card and prevent unauthorized transactions. Also check your recent transactions for anything you do not recognize.",

    "transaction_issue":
        "I can help with your banking question. Could you tell me whether the issue is about a card, transaction, transfer, account, cash withdrawal, loan, or fraud?",

    "transaction_failed":
        "If your transaction failed, please check your account balance and transaction status. Contact your bank if the problem continues.",

    "transaction_pending":
        "Your transaction may still be processing. Please check the transaction status with your bank.",

    "transfer_pending":
        "Pending transfers can take time depending on the bank and transfer type. Please check the transaction status with your bank.",

    "transfer_failed":
        "If your money transfer failed, please verify the recipient details and account balance, then try again or contact your bank.",

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
        "I'm sorry your product arrived damaged. Please provide your order number and details about the damage so the return or replacement process can be started.",

    "wrong_product":
        "I'm sorry you received the wrong product. Please provide your order number and the product you received.",

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

    "symptoms_information":
        "I can help with general information about symptoms. Please describe your symptoms, how long you have had them, and their severity.",

    "symptom_information":
        "I can provide general information about symptoms. Please describe what you are experiencing.",

    "medical_fever":
        "For fever, monitor your temperature, stay hydrated, and rest. If the fever is severe, persistent, or accompanied by concerning symptoms, seek medical care.",

    "medical_headache":
        "For a headache, rest, stay hydrated, and monitor your symptoms. Seek medical care if the headache is severe, sudden, persistent, or accompanied by other concerning symptoms.",

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
        "If you are experiencing a serious or potentially life-threatening medical emergency, seek immediate emergency medical care. For non-emergency concerns, please describe your symptoms or situation."
}


# ============================================================
# LOW CONFIDENCE RESPONSE
# ============================================================

def low_confidence_response(domain):

    if domain == "Finance":

        return (
            "I'm not completely sure what you need help with. "
            "Could you tell me whether your issue is about a card, "
            "transaction, transfer, account, cash withdrawal, loan, or fraud?"
        )

    if domain == "E-commerce":

        return (
            "I'm not completely sure what you need help with. "
            "Could you tell me whether the problem is about a damaged product, "
            "wrong product, return/refund, delivery, payment, or product information?"
        )

    if domain == "Healthcare":

        return (
            "I'm not completely sure what you need help with. "
            "Could you tell me whether you need help with symptoms, "
            "medication, an appointment, a lab test, or health insurance?"
        )

    return (
        "I'm not completely sure what you need help with. "
        "Could you please provide a little more information?"
    )


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🤖 AI Assistant")

    st.caption(
        "Multi-Domain Customer Service"
    )

    st.divider()

    st.subheader("Supported Domains")

    st.write("🏦 **Finance**")
    st.caption(
        "Cards • Transactions • Transfers • Fraud"
    )

    st.write("🛒 **E-commerce**")
    st.caption(
        "Orders • Delivery • Returns • Refunds"
    )

    st.write("🏥 **Healthcare**")
    st.caption(
        "Symptoms • Appointments • Medication"
    )

    st.divider()

    st.subheader("Model")

    st.metric(
        "Confidence Threshold",
        "60%"
    )

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-header">
        <h1>🤖 Multi-Domain AI Customer Assistant</h1>
        <p>
            Intelligent intent recognition across
            Finance, E-commerce and Healthcare
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DOMAIN CARDS
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    st.markdown(
        """
        <div class="card">
            <h3>🏦 Finance</h3>
            <p>
                Cards • Transactions • Transfers • Fraud
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        """
        <div class="card">
            <h3>🛒 E-commerce</h3>
            <p>
                Orders • Delivery • Returns • Refunds
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        """
        <div class="card">
            <h3>🏥 Healthcare</h3>
            <p>
                Symptoms • Appointments • Medication
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SECTION
# ============================================================

st.subheader("💬 Chat with the Assistant")


# ============================================================
# WELCOME
# ============================================================

if not st.session_state.messages:

    st.info(
        "👋 Welcome! Ask me a question about Finance, "
        "E-commerce or Healthcare."
    )


# ============================================================
# DISPLAY CHAT
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and "details" in message
        ):

            details = message["details"]

            confidence = details[
                "confidence"
            ]

            if confidence >= 60:

                status = "🟢 High confidence"

            else:

                status = "🟠 Low confidence"

            st.markdown(
                f"""
                <div class="details">

                <b>Domain:</b>
                {details['domain']}

                &nbsp;&nbsp;|&nbsp;&nbsp;

                <b>Intent:</b>
                {details['intent']}

                &nbsp;&nbsp;|&nbsp;&nbsp;

                {status}

                &nbsp;&nbsp;|&nbsp;&nbsp;

                <b>{confidence:.2f}%</b>

                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# CHAT INPUT
# ============================================================

user_query = st.chat_input(
    "Type your question..."
)


# ============================================================
# PROCESS QUERY
# ============================================================

if user_query:

    user_query = user_query.strip()

    if not user_query:
        st.stop()


    # ========================================================
    # USER MESSAGE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query
        }
    )


    command = user_query.lower()


    # ========================================================
    # RESET
    # ========================================================

    if command == "reset":

        st.session_state.messages = []

        st.rerun()


    # ========================================================
    # EXIT
    # ========================================================

    if command == "exit":

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content":
                    "Goodbye! Have a great day."
            }
        )

        st.rerun()


    # ========================================================
    # GREETING
    # ========================================================

    if command in [
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening"
    ]:

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content":
                    "Hello! 👋 How can I help you today?"
            }
        )

        st.rerun()


    # ========================================================
    # THANK YOU
    # ========================================================

    if command in [
        "thanks",
        "thank you",
        "thankyou"
    ]:

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content":
                    "You're welcome! 😊 "
                    "Is there anything else I can help you with?"
            }
        )

        st.rerun()


    # ========================================================
    # SPECIAL STOLEN CARD DETECTION
    # ========================================================

    critical_intent = (
        detect_critical_finance_intent(
            user_query
        )
    )


    if critical_intent:

        predicted_intent = critical_intent

        confidence = 1.0

        domain = "Finance"


    else:

        # ====================================================
        # MODEL
        # ====================================================

        try:

            predicted_intent, confidence = (
                predict_intent(
                    user_query,
                    vectorizer,
                    model
                )
            )

        except Exception as e:

            st.error(
                "Unable to process the query."
            )

            st.code(str(e))

            st.stop()


        # ====================================================
        # INTENT MAPPING
        # ====================================================

        if predicted_intent in intent_mapping:

            predicted_intent = (
                intent_mapping[
                    predicted_intent
                ]
            )

        elif str(
            predicted_intent
        ) in intent_mapping:

            predicted_intent = (
                intent_mapping[
                    str(predicted_intent)
                ]
            )


        # ====================================================
        # CONFIDENCE
        # ====================================================

        confidence = float(
            confidence
        )

        if confidence > 1:

            confidence = (
                confidence / 100
            )


        # ====================================================
        # DOMAIN
        # ====================================================

        domain = get_domain(
            predicted_intent
        )


        if domain is None:

            domain = detect_domain_from_query(
                user_query
            )


    # ========================================================
    # RESPONSE
    # ========================================================

    if confidence < CONFIDENCE_THRESHOLD:

        displayed_intent = "Uncertain"

        response = (
            low_confidence_response(
                domain
            )
        )

    else:

        displayed_intent = predicted_intent

        response = RESPONSES.get(
            str(
                predicted_intent
            ).lower(),

            "I can help with that. "
            "Please provide more information."
        )


    # ========================================================
    # SAVE RESPONSE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "assistant",

            "content": response,

            "details": {

                "domain":
                    domain
                    if domain
                    else "Uncertain",

                "intent":
                    displayed_intent,

                "confidence":
                    confidence * 100
            }
        }
    )


    # ========================================================
    # REFRESH
    # ========================================================

    st.rerun()