import os
from dotenv import load_dotenv

load_dotenv()

import streamlit as st
import pandas as pd

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Reputation & Sentiment Dashboard",
    layout="wide",
    page_icon="🧠",
)

# ============================================================
# SESSION STATE
# ============================================================
state = st.session_state

state.setdefault("entered", False)
state.setdefault("show_controls", True)
state.setdefault("lang", "RO")
state.setdefault("help_open", False)

# ============================================================
# IMPORTS LIGHT
# ============================================================
from translations import t
from components.layout import render_base_styles
from components.header import render_header
from components.footer import render_footer
from components.sidebar import render_sidebar
from utils import load_reddit_dataframe

# ============================================================
# STATIC FALLBACK HELP
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
# GROQ CLIENT (CACHED)
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
def get_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])


# ============================================================
# CACHED HELP RESPONSES
# ============================================================
@st.cache_data(ttl=86400)
def cached_help(topic: str, lang: str) -> str:

    if lang == "RO":
        user_prompt = f"Explică foarte pe scurt secțiunea '{topic}'."
    else:
        user_prompt = f"Briefly explain the '{topic}' section."

    try:
        client = get_groq_client()

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {
                    "role": "system",
                    "content": HELP_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0.1,
            max_tokens=60,
        )

        text = response.choices[0].message.content.strip()

        if not text:
            raise ValueError("Empty response")

        return text

    except Exception:

        static_text = app_guide_answer(topic)

        if lang == "RO":
            return (
                "⚠️ Groq indisponibil — folosesc explicația standard.\n\n"
                f"{static_text}"
            )

        return (
            "⚠️ Groq unavailable — using standard explanation.\n\n"
            f"{static_text}"
        )


def help_llm(topic: str, lang: str) -> str:
    return cached_help(topic, lang)


# ============================================================
# LANDING PAGE
# ============================================================
if not state.entered:

    from tab0_home import render_home

    render_home()
    st.stop()


# ============================================================
# GLOBAL STYLES
# ============================================================
render_base_styles()


# ============================================================
# HEADER
# ============================================================
render_header()


# ============================================================
# HELP DIALOG
# ============================================================
@st.dialog("Asistentul tău")
def help_dialog():

    st.markdown("### Alege un subiect pentru explicații")

    with st.form("help_form"):

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
        )

        submitted = st.form_submit_button(
            "Trimite",
            type="primary"
        )

    if submitted:

        with st.spinner("Generare răspuns..."):

            answer = help_llm(
                topic,
                state.lang
            )

        st.markdown(f"### Explicație\n{answer}")


# ============================================================
# SIDEBAR
# ============================================================
if state.show_controls:
    sidebar_values = render_sidebar()
else:
    sidebar_values = {
        "dashboard_method": "Logistic Regression 3-class balanced",
        "company_filter": "All",
        "rows_slider": 50,
        "dashboard_limit_rows": 50,
        "dashboard_refresh": False,
        "lang": state.lang,
    }

# ============================================================
# OPEN HELP DIALOG
# ============================================================
if state.help_open:
    help_dialog()
    state.help_open = False


# ============================================================
# SIDEBAR VALUES
# ============================================================
dashboard_method = sidebar_values["dashboard_method"]
company_filter = sidebar_values["company_filter"]
rows_slider = sidebar_values["rows_slider"]
dashboard_limit_rows = sidebar_values["dashboard_limit_rows"]
dashboard_refresh = sidebar_values["dashboard_refresh"]

lang = state.lang




# ============================================================
# OPTIONAL CACHE CLEAR
# ============================================================
if dashboard_refresh:
    st.cache_data.clear()


# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data(ttl=600, show_spinner=False)
def get_base_df_full():

    df = load_reddit_dataframe()

    if df is None or df.empty:

        st.warning(
            "⚠️ Unable to load data from database. "
            "Using mock data for demonstration."
        )

        return pd.DataFrame()

    # MEMORY OPTIMIZATION
    if "company" in df.columns:
        df["company"] = df["company"].astype("category")

    if "sentiment" in df.columns:
        df["sentiment"] = df["sentiment"].astype("category")

    return df


base_df_full = get_base_df_full()

# ============================================================
# FILTERED DATA
# ============================================================
if (
    base_df_full is not None
    and not base_df_full.empty
    and company_filter != "All"
):
    filtered_df = base_df_full[
        base_df_full["company"] == company_filter
    ]
else:
    filtered_df = base_df_full


# ============================================================
# NAVIGATION (FASTER THAN TABS)
# ============================================================
active_tab = st.radio(
    "",
    [
        t("tab_dashboard", lang),
        t("tab_ai_insights", lang),
        t("tab_interactive_demo", lang),
        t("tab_live_pipeline", lang),
        t("tab_proof_source", lang),
    ],
    horizontal=True,
)

# ============================================================
# DASHBOARD
# ============================================================
if active_tab == t("tab_dashboard", lang):

    from tabs.tab_dashboard import render_dashboard

    render_dashboard(
        base_df_full,
        method_name=dashboard_method,
        company=company_filter,
        limit_rows=dashboard_limit_rows,
        refresh=dashboard_refresh,
    )

# ============================================================
# AI INSIGHTS
# ============================================================
elif active_tab == t("tab_ai_insights", lang):

    st.markdown("### AI Insights")

    generate_ai = st.button(
        "Generate AI Insights",
        type="primary"
    )

    if generate_ai:

        with st.spinner("Generating AI insights..."):

            from tabs.tab2_ai_insights import render_tab2

            render_tab2()

# ============================================================
# INTERACTIVE DEMO
# ============================================================
elif active_tab == t("tab_interactive_demo", lang):

    from tabs.tab3_interactive_demo import render_tab3

    render_tab3()

# ============================================================
# LIVE PIPELINE
# ============================================================
elif active_tab == t("tab_live_pipeline", lang):

    from tabs.tab4_live_pipeline import render_tab4

    render_tab4(
        rows_slider,
        lang
    )

# ============================================================
# PROOF OF SOURCE
# ============================================================
elif active_tab == t("tab_proof_source", lang):

    from tabs.tab5_proof_of_source import render_tab5

    render_tab5(
        filtered_df,
        rows_slider
    )

# ============================================================
# CLEANUP
# ============================================================
del filtered_df

# ============================================================
# FOOTER
# ============================================================
render_footer()
