# Multi-Domain Transformer Chatbot

## 📌 Project Overview

The **Multi-Domain Transformer Chatbot** is an AI-based customer service chatbot designed to understand and respond to user queries across multiple domains.

The system uses a **pretrained DistilBERT Transformer model** for intent classification and **Streamlit** to provide an interactive web-based chatbot interface.

The chatbot currently supports three major domains:

- 💰 Finance
- 🛒 E-commerce
- 🏥 Healthcare

The system identifies the user's intent from the input query and generates an appropriate response based on the predicted intent.

---

## 🎯 Objectives

The main objectives of this project are:

- To develop an intelligent multi-domain customer service chatbot.
- To classify user queries using a pretrained Transformer model.
- To use DistilBERT for intent recognition.
- To provide confidence scores for predictions.
- To provide an interactive and user-friendly Streamlit interface.
- To handle customer queries from Finance, E-commerce, and Healthcare domains.
- To reduce the need for manual customer support for common queries.

---

## 🧠 System Architecture

```text
                 User
                  │
                  ▼
          Streamlit Chat UI
                  │
                  ▼
            User Query
                  │
                  ▼
          Text Preprocessing
                  │
                  ▼
        DistilBERT Transformer
                  │
                  ▼
          Intent Classification
                  │
          ┌───────┴────────┐
          ▼                ▼
       Intent          Confidence
          │
          ▼
     Response Mapping
          │
          ▼
     Chatbot Response
          │
          ▼
          User