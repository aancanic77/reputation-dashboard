import streamlit as st
from groq import Groq

@st.cache_resource(show_spinner=False)
def get_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

def ask_groq(question: str) -> str:
    client = get_groq_client()

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant for a sentiment dashboard. "
                        "Keep answers short, clear, and friendly. "
                        "If the user writes in Romanian, answer in Romanian. "
                        "If the user writes in English, answer in English."
                    )
                },
                {"role": "user", "content": question},
            ],
            temperature=0.3,
            max_tokens=200,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"EROARE GROQ: {e}"
