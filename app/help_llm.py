import streamlit as st
from groq import Groq

# ============================================================
#  CLIENT GROQ (CACHE)
# ============================================================
@st.cache_resource(show_spinner=False)
def get_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])


# ============================================================
#  EXPLICAȚII TEHNICE FIXE (fără Groq)
# ============================================================
TECHNICAL_TOPICS = {
    "VADER": {
        "RO": (
            "VADER (Valence Aware Dictionary and sEntiment Reasoner) este un algoritm "
            "lexicon‑based pentru analiză de sentiment. Folosește un lexicon pre‑annotat "
            "și reguli lingvistice (negări, intensificatori, majuscule, punctuație) pentru "
            "a ajusta polaritatea. Produce un scor compound între -1 și 1, optimizat pentru "
            "texte scurte și limbaj informal."
        ),
        "EN": (
            "VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon‑based "
            "sentiment analysis algorithm. It uses a pre‑annotated lexicon and linguistic "
            "heuristics (negations, intensifiers, capitalization, punctuation) to adjust "
            "polarity. It outputs a compound score between -1 and 1, optimized for short, "
            "informal text."
        ),
    },

    "Logistic Regression": {
        "RO": (
            "Logistic Regression este un model liniar de clasificare care folosește "
            "vectorizarea TF‑IDF pentru a transforma textul în caracteristici numerice. "
            "Învață limite de decizie între clasele Pozitiv, Neutru și Negativ. Este rapid, "
            "stabil și interpretabil, potrivit pentru seturi de date moderate."
        ),
        "EN": (
            "Logistic Regression is a linear classification model that uses TF‑IDF "
            "vectorization to convert text into numerical features. It learns decision "
            "boundaries between Positive, Neutral, and Negative classes. It is fast, "
            "stable, and interpretable, suitable for medium‑sized datasets."
        ),
    },

    "Transformer": {
        "RO": (
            "Modelul Transformer folosește mecanisme de self‑attention pentru a analiza "
            "contextul global al textului. Poate surprinde relații semantice complexe și "
            "oferă cea mai mare acuratețe în clasificarea sentimentului."
        ),
        "EN": (
            "The Transformer model uses self‑attention mechanisms to analyze the global "
            "context of text. It captures complex semantic relationships and provides the "
            "highest accuracy in sentiment classification."
        ),
    },
}


# ============================================================
#  FUNCTIA PRINCIPALĂ
# ============================================================
def ask_groq(topic: str, lang: str) -> str:
    lang = lang.upper()

    # 1. Dacă topicul are explicație tehnică → returnăm direct
    if topic in TECHNICAL_TOPICS:
        return TECHNICAL_TOPICS[topic][lang]

    # 2. Pentru restul topicurilor → folosim Groq
    client = get_groq_client()

    # Prompt adaptat limbii
    if lang == "RO":
        user_prompt = f"Explică foarte pe scurt secțiunea '{topic}' în 2–3 fraze."
    else:
        user_prompt = f"Briefly explain the '{topic}' section in 2–3 sentences."

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a micro‑assistant for a sentiment analysis dashboard. "
                        "Your job is ONLY to explain the selected section in 2–3 sentences. "
                        "No greetings, no questions, no chit‑chat, no marketing tone. "
                        "Be factual, concise, and neutral. "
                        "If the user writes in Romanian, answer in Romanian. "
                        "If the user writes in English, answer in English."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=150,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"⚠️ Eroare Groq: {e}"
