import streamlit as st
from groq import Groq

@st.cache_resource(show_spinner=False)
def get_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

def ask_groq(topic: str, lang: str) -> str:
    client = get_groq_client()

    # Construim prompt-ul în funcție de limba selectată
    if lang.upper() == "RO":
        user_prompt = f"Explică foarte pe scurt secțiunea '{topic}'."
    else:
        user_prompt = f"Briefly explain the '{topic}' section."

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a micro‑assistant for a sentiment analysis dashboard. "
                        "Your job is ONLY to explain the selected section in 2–3 sentences maximum. "
                        "No greetings, no questions, no chit‑chat, no marketing tone. "
                        "Be factual, concise, neutral. "
                        "If the user writes in Romanian, answer in Romanian. "
                        "If the user writes in English, answer in English."
                    )
                },
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=120,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"⚠️ Eroare Groq: {e}"
