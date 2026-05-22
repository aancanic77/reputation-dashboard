import streamlit as st
from translations import t

def render_sidebar() -> dict:
    state = st.session_state

    # ============================
    # LIMBA CURENTĂ
    # ============================
    lang = state.get("lang", "RO")

    st.sidebar.title(t("sidebar_title", lang))

    # ============================
    # SELECTOR LIMBĂ
    # ============================
    new_lang = st.sidebar.radio(
        t("sidebar_language_selector", lang),
        ["RO", "EN"],
        index=0 if lang == "RO" else 1,
        key="lang"
    )

    # ============================
    # METODĂ SENTIMENT
    # ============================
    method = st.sidebar.radio(
        t("sidebar_method", new_lang),
        [
            "Logistic Regression 3-class balanced",
            "VADER",
            "Deep Learning Transformer",
        ],
        key="dashboard_method"
    )

    # ============================
    # SELECTOR COMPANIE
    # ============================
    company = st.sidebar.selectbox(
        t("sidebar_company", new_lang),
        ["All", "Apple", "Google", "Samsung"],
        key="company_filter"
    )

    # ============================
    # SLIDER NUMĂR RÂNDURI
    # ============================
    rows_slider = st.sidebar.slider(
        t("sidebar_rows_slider", new_lang),
        min_value=10,
        max_value=200,
        value=state.get("rows_slider", 50),
        key="rows_slider"
    )

    # ============================
    # SLIDER LIMITĂ RÂNDURI
    # ============================
    dashboard_limit_rows = st.sidebar.slider(
        t("sidebar_limit_rows", new_lang),
        min_value=20,
        max_value=500,
        value=state.get("dashboard_limit_rows", 50),
        key="dashboard_limit_rows"
    )

    st.sidebar.markdown("---")

    # ============================
    # REFRESH BUTTON
    # ============================
    refresh = st.sidebar.button(
        t("sidebar_refresh", new_lang),
        type="primary",
        key="dashboard_refresh"
    )

    # ============================
    # HELP BUTTON
    # ============================
    if st.sidebar.button("💬 Help", key="help_btn_sidebar"):
        state.help_open = True

    st.sidebar.caption(t("sidebar_caption", new_lang))

    return {
        "dashboard_method": method,
        "company_filter": company,
        "rows_slider": rows_slider,
        "dashboard_limit_rows": dashboard_limit_rows,
        "dashboard_refresh": refresh,
        "lang": new_lang,
    }
