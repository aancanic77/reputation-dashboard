import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd

# ============================================================
#  STATIC FALLBACK HELP (folosit dacă Groq pică)
# ============================================================
def app_guide_answer(topic: str) -> str:
    guides = {
        "Dashboard": "📊 Dashboard shows sentiment analysis across 3 ML models.",
        "Logistic Regression": "🤖 Logistic Regression (3-class) uses TF-IDF vectors.",
        "VADER": "⚖️ VADER is a rule-based sentiment analyzer optimized for social media.",
        "Transformer": "🧠 Transformer (DistilBERT) provides contextual sentiment understanding.",
        "Live Pipeline": "🔄 Live Pipeline shows the full ETL: Collect → Extract → Map → Analyze → Result.",
        "Proof of Source": "📁 Proof of Source displays real Reddit comments behind each metric.",
        "AI Insights": "💡 AI Insights uses LLMs to generate marketing intelligence.",
    }

    return guides.get(topic, "❓ Subiect necunoscut.")


# ============================================================
#  GROQ HELP (LLM) + FALLBACK
# ============================================================
from groq import Groq

def help_llm(topic: str) -> str:
    """
    Încearcă să explice topicul folosind Groq.
    Dacă Groq pică → fallback la app_guide_answer().
    """

    prompt = f"""
You are a technical assistant for a sentiment analysis dashboard.
Explain the section called "{topic}" clearly and concisely.
Do NOT invent features that do not exist in the app.
Do NOT add marketing fluff.
Focus only on what the section actually does in the dashboard.
"""

    try:
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=250
        )

        return response.choices[0].message["content"]

    except Exception as e:
        # fallback automat
        return app_guide_answer(topic)


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
st.session_state["_components_styles_loaded"] = False
render_base_styles()

st.markdown(
    "<script>document.body.classList.remove('landing-page');</script>",
    unsafe_allow_html=True,
)


# ============================================================
#  GLOBAL HEADER
# ============================================================
render_header()


# ============================================================
#  HELP DIALOG (cu Groq + fallback)
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
        answer = help_llm(topic)
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
