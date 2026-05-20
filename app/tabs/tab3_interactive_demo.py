import streamlit as st
import pandas as pd

from translations import t
from sentiment_models import (
    predict_logreg,
    predict_vader,
    predict_transformer,
    predict_logreg_word_contrib,
)

from visualization import (
    plot_word_contributions,
    plot_transformer_attention_heatmap,
)


def render_tab3():
    lang = st.session_state["lang"]

    # ============================
    #  INTRO CARD
    # ============================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        f'<h2 class="section-title">{t("tab3_title", lang)}</h2>',
        unsafe_allow_html=True,
    )
    st.write(t("tab3_intro_description", lang))
    st.markdown('</div>', unsafe_allow_html=True)

    # ============================
    #  INPUT CARD
    # ============================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        f'<h2 class="section-title">{t("tab3_input_section", lang)}</h2>',
        unsafe_allow_html=True,
    )

    # Exemplele rămân în ENGLEZĂ — NU se traduc
    example_options = [
        "I love Google products, they are amazing and reliable",
        "I hate slow updates and bugs in software",
        "Samsung phones are good but overpriced",
        "The new iPhone is excellent, fast, and absolutely perfect",
        "The Pixel phone is awful, slow, and disappointing",
    ]

    example = st.selectbox(
        t("tab3_choose_example", lang),
        example_options,
    )

    user_text = st.text_area(
        t("tab3_write_custom", lang),
        value=example,
        height=100
    )

    st.markdown('</div>', unsafe_allow_html=True)

    # ============================
    #  ANALYSIS CARD
    # ============================
    if st.button(t("tab3_analyze_button", lang), type="primary", use_container_width=True, key="analyze_btn"):

        if not user_text.strip():
            st.warning(t("tab3_empty_text_warning", lang))
            return

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            f'<h2 class="section-title">{t("tab3_results_section", lang)}</h2>',
            unsafe_allow_html=True,
        )

        # ============================================================
        # LOGISTIC REGRESSION
        # ============================================================
        st.markdown(f"### {t('tab3_logreg_title', lang)}")

        logreg_res = predict_logreg(user_text)
        lr_label = logreg_res["label"]

        color_class = (
            "label-positive" if lr_label == "positive"
            else "label-negative" if lr_label == "negative"
            else "label-neutral"
        )

        st.markdown(f"""
        <div class="result-box">
            <b>{t('tab3_prediction_label', lang)}</b> <span class="{color_class}">{lr_label.upper()}</span>
        </div>
        """, unsafe_allow_html=True)

        contrib_df = predict_logreg_word_contrib(user_text)
        fig_lr = plot_word_contributions(
            contrib_df["token"].tolist(),
            contrib_df["contribution"].tolist(),
            t("tab3_logreg_word_contrib_title", lang),
            t("tab3_logreg_word_contrib_subtitle", lang),
        )
        st.pyplot(fig_lr, use_container_width=True)

        st.divider()

        # ============================================================
        # VADER
        # ============================================================
        st.markdown(f"### {t('tab3_vader_title', lang)}")

        vader_res = predict_vader(user_text)
        vader_label = vader_res["label"]

        color_class = (
            "label-positive" if vader_label == "positive"
            else "label-negative" if vader_label == "negative"
            else "label-neutral"
        )

        st.markdown(f"""
        <div class="result-box">
            <b>{t('tab3_prediction_label', lang)}</b> <span class="{color_class}">{vader_label.upper()}</span>
        </div>
        """, unsafe_allow_html=True)

        vader_words = vader_res["tokens"]
        vader_scores = vader_res["scores"]

        fig_vader = plot_word_contributions(
            vader_words,
            vader_scores,
            t("tab3_vader_word_contrib_title", lang),
            t("tab3_vader_word_contrib_subtitle", lang),
        )
        st.pyplot(fig_vader, use_container_width=True)

        st.divider()

        # ============================================================
        # TRANSFORMER
        # ============================================================
        st.markdown(f"### {t('tab3_transformer_title', lang)}")

        transf_res = predict_transformer(user_text)
        t_label = transf_res["label"]

        color_class = (
            "label-positive" if t_label == "positive"
            else "label-negative" if t_label == "negative"
            else "label-neutral"
        )

        st.markdown(f"""
        <div class="result-box">
            <b>{t('tab3_prediction_label', lang)}</b> <span class="{color_class}">{t_label.upper()}</span>
        </div>
        """, unsafe_allow_html=True)

        tokens = transf_res["tokens"]
        if len(tokens) > 1:
            fig_heatmap = plot_transformer_attention_heatmap(tokens, transf_res["attention"])
            st.pyplot(fig_heatmap, use_container_width=True)
        else:
            st.info(t("tab3_longer_sentence_info", lang))

        st.markdown('</div>', unsafe_allow_html=True)

        # ============================================================
        # FINAL COMPARISON CARD
        # ============================================================
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            f'<h2 class="section-title">{t("tab3_comparison_section", lang)}</h2>',
            unsafe_allow_html=True,
        )

        comparison_df = pd.DataFrame([
            {"Model": "Logistic Regression", "Prediction": lr_label},
            {"Model": "VADER", "Prediction": vader_label},
            {"Model": "Transformer", "Prediction": t_label},
        ])

        st.dataframe(comparison_df, hide_index=True, use_container_width=True)

        st.success(t("tab3_comparison_success", lang))

        st.markdown('</div>', unsafe_allow_html=True)
