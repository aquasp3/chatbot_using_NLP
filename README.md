# chatbot_using_NLP

A simple intent-based chatbot built with **NLP + Logistic Regression** and served using **Streamlit**.

## Features
- Intent classification with `TfidfVectorizer` + `LogisticRegression`
- Adjustable confidence threshold from the sidebar
- Chat UI built with Streamlit
- Conversation history stored in `chat_log.csv`
- Fallback response for low-confidence predictions

## Project files
- `chatbot001.py` – app and model training code
- `intents001.json` – intents dataset
- `chat_log.csv` – generated chat history file

## Setup
1. Create and activate a virtual environment (recommended)
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run
```bash
streamlit run chatbot001.py
```

Then open the local URL shown by Streamlit in your browser.
