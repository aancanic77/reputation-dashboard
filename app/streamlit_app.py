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
#  FLOATING GUIDE CHAT DIALOG
# ============================================================

def render_guide_dialog():
    lang = st.session_state["lang"]

    with st.expander(t("help_dialog_title", lang), expanded=True):
        st.info(t("help_intro_1", lang))
        st.info(t("help_intro_2", lang))

        with st.form("guide_chat_form", clear_on_submit=True):
            guide_question = st.text_input(
                t("help_input_label", lang),
                placeholder=t("help_input_placeholder", lang),
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button(t("help_send_button", lang), use_container_width=True)

        if submitted and guide_question.strip():
            st.info(app_guide_answer(guide_question))


# ============================================================
#  FLOATING HELP BUTTON STYLING & RENDERING
# ============================================================
if st.button(
    "💬 Help",
    key="floating_help_button_main_unique",
    type="primary",
    use_container_width=False,
):
    st.session_state.help_open = not st.session_state.help_open
    st.experimental_rerun()

if st.session_state.help_open:
    render_guide_dialog()

st.markdown(
    """
    <style>
    .help-fix-button {
        font-size: 18px !important;
    }
    .help-dialog-card {
        position: fixed !important;
        right: 22px !important;
        bottom: 92px !important;
        z-index: 999998 !important;
        max-width: 380px !important;
        width: calc(100vw - 44px) !important;
        border-radius: 24px !important;
        background: rgba(255, 255, 255, 0.98) !important;
        box-shadow: 0 24px 60px rgba(0, 0, 0, 0.18) !important;
        padding: 20px 24px !important;
    }
    </style>
    <script>
    const fixHelpButton = () => {
        const buttons = [...document.querySelectorAll('button')];
        const helpButton = buttons.find(b => b.innerText.trim() === '💬 Help');
        if (helpButton) {
            helpButton.style.position = 'fixed';
            helpButton.style.right = '22px';
            helpButton.style.bottom = '22px';
            helpButton.style.zIndex = '999999';
            helpButton.style.borderRadius = '999px';
            helpButton.style.padding = '16px 24px';
            helpButton.style.background = 'linear-gradient(135deg, #FF8A00 0%, #FFC300 100%)';
            helpButton.style.color = 'white';
            helpButton.style.boxShadow = '0 14px 35px rgba(255, 138, 0, 0.35)';
            helpButton.style.fontWeight = '700';
            helpButton.style.cursor = 'pointer';
            helpButton.style.transition = 'transform 0.2s ease';
        }
    };
    setTimeout(fixHelpButton, 100);
    setTimeout(fixHelpButton, 500);
    setInterval(fixHelpButton, 1000);
    </script>
    """,
    unsafe_allow_html=True,
)


# ============================================================
#  FLOATING ROUND BUTTON (HTML)
# ============================================================
st.markdown(
    """
    <div class="toggle-controls-btn" onclick="document.querySelector('button[data-testid=togglebtn]').click()">
        ⚙️
    </div>
    """,
    unsafe_allow_html=True,
)


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
