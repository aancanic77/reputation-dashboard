import time
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from translations import t
from reddit_pipeline import run_reddit_pipeline_live, DEMO_MODE
from sentiment_models import (
    predict_logreg,
    predict_vader,
    predict_transformer,
)

STATUS_COLORS = {
    "Pending": "#9CA3AF",
    "Running": "#0EA5E9",
    "Completed": "#16A34A",
    "Error": "#DC2626",
}


def time_ago(ts):
    try:
        ts = float(ts)
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    except Exception:
        return "unknown"

    now = datetime.now(timezone.utc)
    diff = now - dt
    sec = diff.total_seconds()

    if sec < 60:
        return "acum câteva secunde"
    if sec < 3600:
        return f"acum {int(sec // 60)} minute"
    if sec < 86400:
        return f"acum {int(sec // 3600)} ore"
    return f"acum {int(sec // 86400)} zile"


def render_step_card(title: str, description: str, status: str) -> str:
    color = STATUS_COLORS.get(status, "#9CA3AF")
    return f"""
    <div style="background:#FFFFFF; border-radius:14px; padding:18px; margin-bottom:14px;
                border-left:6px solid {color}; box-shadow:0 2px 8px rgba(0,0,0,0.06);">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
            <div>
                <div style="font-size:16px; font-weight:700; color:#111827; margin-bottom:4px;">{title}</div>
                <div style="font-size:14px; color:#4B5563; line-height:1.5;">{description}</div>
            </div>
            <div style="font-size:13px; font-weight:700; color:{color}; padding:6px 10px; border-radius:999px; background:rgba(0,0,0,0.03);">
                {status}
            </div>
        </div>
    </div>
    """


def update_step(placeholder, title: str, description: str, status: str):
    placeholder.markdown(
        render_step_card(title, description, status),
        unsafe_allow_html=True,
    )


def render_tab4(rows_slider: int, lang: str):

    # ============================
    # CSS
    # ============================
    st.markdown(
        """
        <style>
            :root { --primary: #FF8A00; --accent: #007BFF; }
            .card { background:#fff; border-radius:12px; padding:24px; margin-bottom:24px;
                    box-shadow:0 2px 6px rgba(0,0,0,0.06); }
            h2.section-title { color:var(--accent); margin-top:0; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ============================
    # INTRO
    # ============================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<h2 class="section-title">{t("tab4_title", lang)}</h2>', unsafe_allow_html=True)
    st.write(t("tab4_description", lang))
    if DEMO_MODE:
        st.caption(t("tab4_demo_mode_caption", lang))
    st.markdown("</div>", unsafe_allow_html=True)

    # ============================
    # SLIDER
    # ============================
    comments_per_company = st.slider(
        t("tab4_comments_per_company", lang),
        min_value=10, max_value=40, value=20, step=5,
    )
    st.caption(t("tab4_expected_sample_caption", lang).format(count=comments_per_company * 3))

    # ============================
    # RUN BUTTON
    # ============================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<h2 class="section-title">{t("tab4_run_pipeline_title", lang)}</h2>', unsafe_allow_html=True)
    run_clicked = st.button(t("tab4_run_button", lang), type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if not run_clicked:
        st.info(t("tab4_press_button_info", lang))
        return

    # ============================
    # STEP CARDS
    # ============================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<h2 class="section-title">{t("tab4_pipeline_steps_title", lang)}</h2>', unsafe_allow_html=True)

    steps = [
        ("collect", "desc_collect"),
        ("extract", "desc_extract"),
        ("map", "desc_map"),
        ("analyze", "desc_analyze"),
        ("result", "desc_result"),
    ]

    placeholders = []
    for key, desc in steps:
        ph = st.empty()
        placeholders.append(ph)
        update_step(ph, t(f"tab4_step_{key}", lang), t(f"tab4_step_{desc}", lang), "Pending")

    progress = st.progress(0)

    # ============================
    # STEP 1 — COLLECT
    # ============================
    update_step(placeholders[0], t("tab4_step_collect", lang), t("tab4_step_desc_collect", lang), "Running")
    progress.progress(10)

    live_df = run_reddit_pipeline_live(limit_per_sub=comments_per_company)

    update_step(placeholders[0], t("tab4_step_collect", lang), t("tab4_step_desc_collect", lang), "Completed")
    progress.progress(20)

    # ============================
    # STEP 2 — EXTRACT
    # ============================
    update_step(placeholders[1], t("tab4_step_extract", lang), t("tab4_step_desc_extract", lang), "Running")
    progress.progress(30)

    extracted_df = live_df.copy()

    update_step(placeholders[1], t("tab4_step_extract", lang), t("tab4_step_desc_extract", lang), "Completed")
    progress.progress(40)

    # ============================
    # STEP 3 — MAP
    # ============================
    update_step(placeholders[2], t("tab4_step_map", lang), t("tab4_step_desc_map", lang), "Running")
    progress.progress(50)

    mapped_df = extracted_df.copy()

    # LIMITARE PENTRU CLOUD
    MAX_ROWS = 25
    if len(mapped_df) > MAX_ROWS:
        mapped_df = mapped_df.head(MAX_ROWS)
        st.warning(f"Au fost procesate doar primele {MAX_ROWS} comentarii pentru performanță.")

    update_step(placeholders[2], t("tab4_step_map", lang), t("tab4_step_desc_map", lang), "Completed")
    progress.progress(60)

    # ============================
    # STEP 4 — ANALYZE
    # ============================
    update_step(placeholders[3], t("tab4_step_analyze", lang), t("tab4_step_desc_analyze", lang), "Running")
    progress.progress(70)

    mapped_df["lr_label"] = mapped_df["text"].apply(lambda t: predict_logreg(t)["label"])
    mapped_df["vader_label"] = mapped_df["text"].apply(lambda t: predict_vader(t)["label"])

    if DEMO_MODE:
        mapped_df["dl_label"] = "disabled"
    else:
        mapped_df["dl_label"] = mapped_df["text"].apply(lambda t: predict_transformer(t)["label"])

    update_step(placeholders[3], t("tab4_step_analyze", lang), t("tab4_step_desc_analyze", lang), "Completed")
    progress.progress(85)

    # -------------------------------
    #  STEP 5 — RESULT (cu protecție de erori)
    # -------------------------------
    steps[4]["status"] = "Running"
    update_step(placeholders[4], **steps[4])
    progress.progress(95)
    
    try:
        final_df = mapped_df.copy()
    
        # conversie timp
        final_df["created_at"] = pd.to_datetime(
            final_df["created_utc"], unit="s", errors="coerce"
        ).dt.strftime("%Y-%m-%d %H:%M:%S")
    
        final_df["time_ago"] = final_df["created_utc"].apply(time_ago)
    
        # succes
        steps[4]["status"] = "Completed"
        update_step(placeholders[4], **steps[4])
        progress.progress(100)
    
    except Exception as e:
        steps[4]["status"] = "Error"
        update_step(placeholders[4], **steps[4])
        st.error("A apărut o eroare în etapa finală a pipeline-ului.")
        st.exception(e)
        return


    # ============================
    # RESULTS
    # ============================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<h2 class="section-title">{t("tab4_results_title", lang)}</h2>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric(t("tab4_comments_metric", lang), len(final_df))
    col2.metric(t("tab4_companies_metric", lang), final_df["company"].nunique())
    col3.metric(t("tab4_models_metric", lang),
                t("tab4_models_value_default", lang) if not DEMO_MODE else t("tab4_models_value_demo", lang))

    # ============================
    # CHARTS (LIMITATE)
    # ============================
    st.markdown(f"### {t('tab4_comments_over_time', lang)}")

    time_df = final_df.copy()
    time_df["created_dt"] = pd.to_datetime(time_df["created_utc"], unit="s", errors="coerce")
    time_df = time_df.dropna(subset=["created_dt"])
    time_df = time_df.set_index("created_dt").resample("5min").size().rename("comments").to_frame()

    if not time_df.empty:
        st.line_chart(time_df.tail(50))
    else:
        st.caption(t("tab4_no_valid_timestamps_caption", lang))

    st.markdown(f"### {t('tab4_comments_per_company_chart', lang)}")
    st.bar_chart(final_df["company"].value_counts())

    # ============================
    # SAMPLE COMMENTS (LIMITATE)
    # ============================
    st.markdown(f"### {t('tab4_sample_comments', lang)}")

    preview_cols = [
        "company", "subreddit", "author", "created_at",
        "time_ago", "text", "lr_label", "vader_label", "dl_label",
    ]

    st.caption("Se afișează doar primele 10 comentarii pentru performanță.")
    st.dataframe(final_df[preview_cols].head(10), use_container_width=True, hide_index=True)

    # ============================
    # DISTRIBUȚII (LIMITATE)
    # ============================
    st.markdown(f"### {t('tab4_sentiment_distribution', lang)}")

    st.markdown(f"**{t('tab4_logistic_regression', lang)}**")
    st.dataframe(final_df.groupby(["company", "lr_label"]).size().reset_index(name="count").head(10),
                 use_container_width=True, hide_index=True)

    st.markdown(f"**{t('tab4_vader', lang)}**")
    st.dataframe(final_df.groupby(["company", "vader_label"]).size().reset_index(name="count").head(10),
                 use_container_width=True, hide_index=True)

    st.markdown(f"**{t('tab4_transformer', lang)}**")
    st.dataframe(final_df.groupby(["company", "dl_label"]).size().reset_index(name="count").head(10),
                 use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)
