import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd

# ============================================================
#  STATIC FALLBACK HELP
# ============================================================
def app_guide_answer(topic: str) -> str:
    guides = {
        "Dashboard": "Dashboard-ul arată analiza sentimentului pe modele ML.",
        "Logistic Regression": "Model clasic ML pe TF-IDF, 3 clase echilibrate.",
        "VADER": "Analizor rule-based optimizat pentru social media.",
        "Transformer": "Model contextual (DistilBERT) pentru sentiment.",
        "Live Pipeline": "ETL complet: colectare → extragere → mapare → analiză.",
        "Proof of Source": "Afișează comentariile reale din Reddit folosite în metrici.",
        "AI Insights": "Generează insight-uri marketing cu LLM.",
    }
    return guides.get(topic, "❓ Subiect necunoscut.")


# ============================================================
#  ULTRA-RAPID GROQ HELP (cache permanent + pre-prompt + bilingv)
# ============================================================
from groq import Groq

HELP_SYSTEM_PROMPT = """
You are a concise assistant for a sentiment dashboard.
Rules:
- Answer in 2–3 sentences maximum.
- No marketing language.
- No invented features.
- Stay factual and minimal.
- If user language is Romanian, answer in Romanian.
- If user language is English, answer in English.
"""

@st.cache_resource
def cached_help(topic: str, lang: str) -> str:
    """
    Cache permanent: dacă Groq răspunde o dată, nu mai este apelat niciodată
    pentru același topic + limbă.
    """
    if lang == "RO":
        lang_prompt = f"Explică foarte pe scurt secțiunea '{topic}'."
    else:
        lang_prompt = f"Briefly explain the '{topic}' section."

    try:
        client = Groq(api_key=st.secrets.get("GROQ_API_KEY"))
        if client is None:
            raise ValueError("Missing Groq key")

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": HELP_SYSTEM_PROMPT},
                {"role": "user", "content": lang_prompt}
            ],
            temperature=0.1,
            max_tokens=60
        )

        text = response.choices[0].message["content"].strip()
        if not text:
            raise ValueError("Empty response")

        return text

    except Exception:
        static_text = app_guide_answer(topic)
        if lang == "RO":
            return f"⚠️ Groq indisponibil — folosesc explicația standard.\n\n{static_text}"
        else:
            return f"⚠️ Groq unavailable — using standard explanation.\n\n{static_text}"


def help_llm(topic: str, lang: str) -> str:
    return cached_help(topic, lang)


# ============================================================
#  PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Reputation & Sentiment Dashboard",
    layout="wide",
    page_icon="🧠",
)

# ============================================================
#  IMPORTS
# ============================================================
from utils import load_reddit_dataframe
from tab0_home import render_home
from tabs.tab_dashboard import render_dashboard
from tabs.tab2_ai_insights import render_tab2
from tabs.tab3_interactive_demo import render_tab3
from tabs.tab4_live_pipeline import render_tab4
from tabs.tab5_proof_of_source import render_tab5

from components.layout import render_base_styles
from components.header import render_header
from components.footer import render_footer
from components.sidebar import render_sidebar
from translations import t


# ============================================================
#  SESSION STATE
# ============================================================
if "entered" not in st.session_state:
    st.session_state.entered = False

if "show_controls" not in st.session_state:
    st.session_state.show_controls = True

if "lang" not in st.session_state:
    st.session_state.lang = "RO"

if "help_open" not in st.session_state:
    st.session_state.help_open = False


# ============================================================
#  LANDING PAGE
# ============================================================
if not st.session_state.entered:
    render_home()
    st.stop()


# ============================================================
#  GLOBAL STYLES
# ============================================================
# ============================================================
#  GLOBAL STYLES
# ============================================================
if "_components_styles_loaded" not in st.session_state:
    render_base_styles()
    st.session_state["_components_styles_loaded"] = True




st.markdown(
    "<script>document.body.classList.remove('landing-page');</script>",
    unsafe_allow_html=True,
)


# ============================================================
#  GLOBAL HEADER
# ============================================================
render_header()


# ============================================================
#  HELP DIALOG (cu Groq + fallback + cache permanent)
# ============================================================
@st.dialog("Asistentul tău")
def help_dialog():
    st.markdown("### Alege un subiect pentru explicații")

    topic = st.selectbox(
        "Subiect",
        [
            "Dashboard",
            "Logistic Regression",
            "VADER",
            "Transformer",
            "Live Pipeline",
            "Proof of Source",
            "AI Insights",
        ],
        key="help_topic_select"
    )

    if st.button("Trimite", type="primary"):
        answer = help_llm(topic, st.session_state.lang)
        st.markdown(f"### Explicație\n{answer}")


# ============================================================
#  SIDEBAR CONTROLS
# ============================================================

if st.session_state.show_controls:
    sidebar_values = render_sidebar()
else:
    sidebar_values = {
        "dashboard_method": "Logistic Regression 3-class balanced",
        "company_filter": "All",
        "rows_slider": 50,
        "dashboard_limit_rows": 50,
        "dashboard_refresh": False,
        "lang": st.session_state.lang,
    }

# Trigger dialog
if st.session_state.help_open:
    help_dialog()
    st.session_state.help_open = False


dashboard_method = sidebar_values["dashboard_method"]
company_filter = sidebar_values["company_filter"]
rows_slider = sidebar_values["rows_slider"]
lang = sidebar_values["lang"]
dashboard_limit_rows = sidebar_values["dashboard_limit_rows"]
dashboard_refresh = sidebar_values["dashboard_refresh"]


# ============================================================
#  LOAD BASE DATA
# ============================================================
@st.cache_data(show_spinner=False)
def get_base_df_full():
    df = load_reddit_dataframe()

    if df is None or df.empty:
        st.warning(
            "⚠️ Unable to load data from database. Using mock data for demonstration."
        )
        return pd.DataFrame()

    return df


base_df_full = get_base_df_full()
base_df = base_df_full.copy() if base_df_full is not None else None

if base_df is not None and company_filter != "All":
    base_df = base_df[base_df["company"] == company_filter]


# ============================================================
#  TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        t("tab_dashboard", lang),
        t("tab_ai_insights", lang),
        t("tab_interactive_demo", lang),
        t("tab_live_pipeline", lang),
        t("tab_proof_source", lang),
    ]
)

with tab1:
    render_dashboard(
        base_df_full,
        method_name=dashboard_method,
        company=company_filter,
        limit_rows=dashboard_limit_rows,
        refresh=dashboard_refresh,
    )

with tab2:
    render_tab2()

with tab3:
    render_tab3()

with tab4:
    render_tab4(rows_slider, lang)

with tab5:
    render_tab5(base_df, rows_slider)


# ============================================================
#  GLOBAL FOOTER
# ============================================================
render_footer()
