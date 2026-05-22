import streamlit as st
from groq import Groq

# Cache pentru client
@st.cache_resource(show_spinner=False)
def get_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

# Cache pentru răspunsuri
@st.cache_data(show_spinner=False)
def ask_groq(question: str) -> str:
    client = get_groq_client()

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Keep answers short, clear, and friendly."},
                {"role": "user", "content": question},
            ],
            temperature=0.4,
            max_tokens=300,
        )
        return response.choices[0].message.content

    except Exception:
        return "Groq is temporarily unavailable. Please try again in a moment."
