import streamlit as st
from components.layout import render_base_styles
from translations import t


def render_sidebar() -> dict:
    render_base_styles()

    # ============================
    # LIMBA CURENTĂ
    # ============================
    lang = st.session_state.get("lang", "RO")

    # ============================
    # TITLU SIDEBAR
    # ============================
    st.sidebar.title(t("sidebar_title", lang))

    # ============================
    # SELECTOR LIMBĂ
    # ============================
    new_lang = st.sidebar.radio(
        t("sidebar_language_selector", lang),
        ["RO", "EN"],
        key="lang",
    )
    lang = new_lang

    # ============================
    # METODĂ SENTIMENT
    # ============================
    method = st.sidebar.radio(
        t("sidebar_method", lang),
        [
            "Logistic Regression 3-class balanced",
            "VADER",
            "Deep Learning Transformer",
        ],
        index=0,
        key="dashboard_method",
    )

    # ============================
    # SELECTOR COMPANIE
    # ============================
    company = st.sidebar.selectbox(
        t("sidebar_company", lang),
        ["All", "Apple", "Google", "Samsung"],
        index=0,
        key="company_filter",
    )

    # ============================
    # SLIDER NUMĂR RÂNDURI
    # ============================
    rows_slider = st.sidebar.slider(
        t("sidebar_rows_slider", lang),
        min_value=10,
        max_value=200,
        value=50,
        key="rows_slider",
    )

    # ============================
    # SLIDER LIMITĂ RÂNDURI DISPUTE
    # ============================
    dashboard_limit_rows = st.sidebar.slider(
        t("sidebar_limit_rows", lang),
        min_value=20,
        max_value=500,
        value=50,
        key="dashboard_limit_rows",
    )

    # ============================
    # BUTON REFRESH
    # ============================
    refresh = st.sidebar.button(
        t("sidebar_refresh", lang),
        type="primary"
    )

    # ============================
    # HELP BUTTON — AICI ESTE LOCUL CORECT
    # ============================
    st.sidebar.markdown("---")
    if st.sidebar.button("💬 Help", key="help_btn_sidebar", type="primary"):

        st.session_state.help_open = True

    # ============================
    # CAPTION
    # ============================
    st.sidebar.caption(t("sidebar_caption", lang))

    # ============================
    # RETURN VALUES
    # ============================
    return {
        "dashboard_method": method,
        "company_filter": company,
        "rows_slider": rows_slider,
        "dashboard_limit_rows": dashboard_limit_rows,
        "dashboard_refresh": refresh,
        "lang": lang,
    }
