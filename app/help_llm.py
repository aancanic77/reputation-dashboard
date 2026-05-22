import streamlit as st
from groq import Groq

# ============================================================
#  CLIENT GROQ (cache permanent)
# ============================================================
@st.cache_resource(show_spinner=False)
def get_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])


# ============================================================
#  LLM HELP (Groq direct, rapid, stabil)
# ============================================================
@st.cache_data(show_spinner=False)
def ask_groq(question: str) -> str:
    """
    Returnează un răspuns scurt de la Groq.
    Cache-ul evită apelurile repetate pentru aceeași întrebare.
    """
    client = get_groq_client()

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",   # MODEL CORECT
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

    except Exception:
        return "⚠️ Groq este indisponibil momentan — încearcă din nou în câteva secunde."
