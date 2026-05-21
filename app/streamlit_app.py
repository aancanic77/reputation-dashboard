import os
from dotenv import load_dotenv

load_dotenv()  # ← ÎN PRIMUL RÂND

import streamlit as st


# ============================================================
#  GUIDE ASSISTANT HELPER
# ============================================================
def app_guide_answer(question: str) -> str:
    q = question.lower()

    guides = {
        "dashboard": "📊 **Dashboard** shows sentiment analysis across 3 ML models (Logistic Regression, VADER, Transformer). Use filters to analyze Apple, Google, or Samsung reputation.",
        "logistic regression": "🤖 **Logistic Regression (3-class)** is a balanced machine learning model trained on TF-IDF vectors. It classifies comments as Positive, Negative, or Neutral.",
        "vader": "⚖️ **VADER** (Valence Aware Dictionary and sEntiment Reasoner) is a rule-based sentiment analyzer optimized for social media text. Fast and interpretable!",
        "transformer": "🧠 **Transformer model** uses deep learning (DistilBERT) for nuanced sentiment understanding. More accurate but slower than rule-based approaches.",
        "live pipeline": "🔄 **Live Pipeline** demos the full 5-step ETL: Collect → Extract → Map → Analyze (3 models) → Result. Shows real-time processing.",
        "proof of source": "📁 **Proof of Source** validates findings by showing actual Reddit comments behind each dashboard metric. Ensures transparency and academic rigor.",
        "ai insights": "💡 **AI Insights** uses LLM (Groq/LangChain) to generate marketing intelligence and contextual interpretations of sentiment trends.",
        "help": "👋 Hi! Ask me about Dashboard, Logistic Regression, VADER, Transformer, Live Pipeline, Proof of Source, or AI Insights. I'm here to help!",
    }

    for key, answer in guides.items():
        if key in q:
            return answer

    return "❓ I didn't find that topic. Try asking about: Dashboard, Logistic Regression, VADER, Transformer, Live Pipeline, Proof of Source, or AI Insights."


# ============================================================
#  PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Reputation & Sentiment Dashboard",
    layout="wide",
    page_icon="🧠",
)


from utils import load_reddit_dataframe



# Tabs
from tab0_home import render_home
from tabs.tab_dashboard import render_dashboard
from tabs.tab2_ai_insights import render_tab2
from tabs.tab3_interactive_demo import render_tab3
from tabs.tab4_live_pipeline import render_tab4
from tabs.tab5_proof_of_source import render_tab5
#st.write("Secrets loaded:", list(st.secrets.keys()))
# UI components
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
#  LANDING PAGE (NO SIDEBAR)
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
    """
    <script>
    document.body.classList.remove('landing-page');
    </script>
    """,
    unsafe_allow_html=True,
)


# ============================================================
#  GLOBAL HEADER
# ============================================================
render_header()

# ============================================================
#  FLOATING HELP BUTTON (HTML ONLY — CSS ESTE ÎN style.css)
# ============================================================
st.markdown(
    '<div id="floating-help-btn">💬 Help</div>',
    unsafe_allow_html=True
)

if "help_open" not in st.session_state:
    st.session_state.help_open = False


# ============================================================
#  JS: CLICK → SET STREAMLIT SESSION STATE DIRECT
# ============================================================
help_js = """
<script>
document.addEventListener("DOMContentLoaded", function() {
    const interval = setInterval(() => {
        const helpBtn = document.getElementById("floating-help-btn");
        if (helpBtn) {
            helpBtn.onclick = () => {
                const streamlitEvent = new CustomEvent("streamlit:message", {
                    detail: {type: "help_open", value: true}
                });
                window.dispatchEvent(streamlitEvent);
            };
            clearInterval(interval);
        }
    }, 200);
});
</script>
"""
st.markdown(help_js, unsafe_allow_html=True)


# ============================================================
#  PYTHON LISTENER (SAFE)
# ============================================================
listener_js = """
<script>
window.addEventListener("streamlit:message", (event) => {
    if (event.detail.type === "help_open") {
        window.streamlitSendMessage({
            type: "streamlit:setComponentValue",
            value: true
        });
    }
});
</script>
"""
st.markdown(listener_js, unsafe_allow_html=True)


# ============================================================
#  TRIGGER PYTHON STATE
# ============================================================
if st.session_state.get("_component_value"):
    st.session_state.help_open = True
    st.session_state["_component_value"] = False


# ============================================================
#  TRANSLATION FUNCTION (EN → RO)
# ============================================================
def translate_to_ro(text: str) -> str:
    import requests
    headers = {"Authorization": f"Bearer " + st.secrets["HF_API_KEY"]}
    payload = {"inputs": text}
    r = requests.post(
        "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-ro",
        headers=headers,
        json=payload
    )
    try:
        return r.json()[0]["translation_text"]
    except:
        return text


# ============================================================
#  POPUP DIALOG
# ============================================================
@st.dialog("Asistentul tău")
def help_dialog():
    lang = st.session_state.get("lang", "RO")

    st.write("Întreabă-mă orice despre aplicație.")

    q = st.text_input("Întrebare")
    if st.button("Trimite"):
        answer = app_guide_answer(q)

        if lang == "RO":
            answer = translate_to_ro(answer)

        st.write(answer)


if st.session_state.help_open:
    help_dialog()
    st.session_state.help_open = False


# ============================================================
#  HIDDEN STREAMLIT BUTTON (REAL TOGGLE)
# ============================================================
if st.button(" ", key="togglebtn", help=""):
    st.session_state.show_controls = not st.session_state.show_controls
    st.rerun()




# ============================================================
#  SIDEBAR CONTROLS (conditionally visible)
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

dashboard_method = sidebar_values["dashboard_method"]
company_filter = sidebar_values["company_filter"]
rows_slider = sidebar_values["rows_slider"]
lang = sidebar_values["lang"]
dashboard_limit_rows = sidebar_values["dashboard_limit_rows"]
dashboard_refresh = sidebar_values["dashboard_refresh"]


# ============================================================
#  LOAD BASE DATA (CACHED)
# ============================================================
@st.cache_data(show_spinner=False)
def get_base_df_full():
    df = load_reddit_dataframe()
    
    if df is None or df.empty:
        st.warning(
            "⚠️ Unable to load data from database. Using mock data for demonstration.\n\n"
            "**Possible causes:**\n"
            "- Database is not running or unreachable\n"
            "- Connection credentials are incorrect\n"
            "- For Neon connections: ensure you're using the unpooled endpoint (not the pooler endpoint)\n"
            "- Check that `.env` contains the correct `PG_DSN`\n\n"
            "**To debug:** Run `python test_db_fix.py` from the project root"
        )
        # Return empty dataframe - dashboard will handle gracefully
        return pd.DataFrame()
    
    return df


base_df_full = get_base_df_full()
base_df = base_df_full.copy() if base_df_full is not None else None

if base_df is not None and company_filter != "All":
    base_df = base_df[base_df["company"] == company_filter]


# ============================================================
#  PAGE CONTENT (TABS)
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
