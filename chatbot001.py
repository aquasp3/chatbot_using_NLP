import csv
import datetime
import json
import os
import random
from typing import Any

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


DATA_DIR = os.path.dirname(os.path.abspath(__file__))
INTENTS_PATH = os.path.join(DATA_DIR, "intents001.json")
CHAT_LOG_PATH = os.path.join(DATA_DIR, "chat_log.csv")


def load_intents(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file:
        loaded = json.load(file)

    if not isinstance(loaded, list):
        raise ValueError("The intents file should contain a list of intent objects.")

    for intent in loaded:
        if "tag" not in intent or "patterns" not in intent or "responses" not in intent:
            raise ValueError("Each intent must include 'tag', 'patterns', and 'responses' keys.")

    return loaded


intents = load_intents(INTENTS_PATH)
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)


def _build_training_data() -> tuple[list[str], list[str]]:
    tags: list[str] = []
    patterns: list[str] = []
    for intent in intents:
        for pattern in intent["patterns"]:
            tags.append(intent["tag"])
            patterns.append(pattern)
    return patterns, tags


patterns, tags = _build_training_data()
x = vectorizer.fit_transform(patterns)
clf.fit(x, tags)


def ensure_chat_log_exists() -> None:
    if not os.path.exists(CHAT_LOG_PATH):
        with open(CHAT_LOG_PATH, "w", newline="", encoding="utf-8") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["User Input", "Chatbot Response", "Timestamp"])


def chatbot(input_text: str, threshold: float = 0.30) -> str:
    input_features = vectorizer.transform([input_text])
    probabilities = clf.predict_proba(input_features)[0]
    best_index = probabilities.argmax()
    confidence = probabilities[best_index]
    tag = clf.classes_[best_index]

    if confidence < threshold:
        return "I’m not fully sure I understood that. Could you rephrase your question?"

    for intent in intents:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "I couldn’t find a matching response right now."


def append_history(user_input: str, response: str) -> None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(CHAT_LOG_PATH, "a", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([user_input, response, timestamp])


def render_conversation_history() -> None:
    ensure_chat_log_exists()

    with open(CHAT_LOG_PATH, "r", encoding="utf-8") as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader, None)
        has_rows = False
        for row in csv_reader:
            has_rows = True
            st.text(f"User: {row[0]}")
            st.text(f"Chatbot: {row[1]}")
            st.text(f"Timestamp: {row[2]}")
            st.markdown("---")

        if not has_rows:
            st.info("No conversation history yet. Start a chat from the Home page.")


def main() -> None:
    st.title("Intents of Chatbot using NLP")

    if "chat_counter" not in st.session_state:
        st.session_state.chat_counter = 0

    st.sidebar.subheader("Model Settings")
    threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.30, 0.05)

    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")

        ensure_chat_log_exists()
        st.session_state.chat_counter += 1
        user_input = st.text_input("You:", key=f"user_input_{st.session_state.chat_counter}")

        if user_input:
            response = chatbot(user_input, threshold=threshold)
            st.text_area(
                "Chatbot:",
                value=response,
                height=120,
                max_chars=None,
                key=f"chatbot_response_{st.session_state.chat_counter}",
            )
            append_history(str(user_input), response)

            if response.lower() in ["goodbye", "bye"]:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    elif choice == "Conversation History":
        st.header("Conversation History")
        render_conversation_history()

    elif choice == "About":
        st.write(
            "The goal of this project is to create a chatbot that can understand and respond to user input based on intents. "
            "The chatbot is built using NLP and Logistic Regression for intent classification, and Streamlit for the UI."
        )

        st.subheader("Project Overview")
        st.write(
            """
            1. NLP techniques and Logistic Regression are used to train the chatbot on labeled intents.
            2. Streamlit is used to build a web-based chatbot interface for interactive conversations.
            """
        )


if __name__ == "__main__":
    main()
