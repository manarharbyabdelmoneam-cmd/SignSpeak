"""
UI Page: Dashboard
====================
Two things:
    1. Model performance summary — read from the training notebook's
       exported deployment_config.json / final_test_metrics.json (if
       present in models/), so real evaluation numbers are shown instead
       of made-up ones.
    2. Session history — every translation done during the current
       browser session (kept in st.session_state, not persisted to disk).

If the metrics files aren't shipped with the deployed model, that section
is skipped gracefully instead of crashing the page.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from config.settings import MODELS_DIR

DEPLOYMENT_CONFIG_FILENAME = "deployment_config.json"
FINAL_TEST_METRICS_FILENAME = "final_test_metrics.json"

SESSION_HISTORY_KEY = "signbridge_prediction_history"


def _load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _render_model_info_section() -> None:
    st.markdown("### معلومات الموديل")

    deployment_config = _load_json_if_exists(MODELS_DIR / DEPLOYMENT_CONFIG_FILENAME)
    test_metrics = _load_json_if_exists(MODELS_DIR / FINAL_TEST_METRICS_FILENAME)

    if deployment_config is None and test_metrics is None:
        st.caption("معلومات تقييم الموديل مش متاحة في نسخة الـ deployment دي.")
        return

    col1, col2, col3, col4 = st.columns(4)

    if deployment_config:
        with col1:
            st.metric("المعمارية", deployment_config.get("architecture", "—"))
        with col2:
            st.metric("عدد الإشارات", deployment_config.get("number_of_classes", "—"))

    if test_metrics:
        with col3:
            accuracy = test_metrics.get("test_accuracy")
            st.metric("Accuracy (Internal Test)", f"{accuracy:.1%}" if accuracy else "—")
        with col4:
            f1 = test_metrics.get("test_macro_f1")
            st.metric("Macro F1 (Internal Test)", f"{f1:.2f}" if f1 else "—")

    st.caption(
        "⚠️ النتائج دي من الـ internal holdout test set (17 عينة بس) - "
        "مؤشر أولي، مش تقييم مستقل على بيانات جديدة تماما."
    )


def _render_session_history_section() -> None:
    st.markdown("### سجل الترجمات في الجلسة دي")

    history = st.session_state.get(SESSION_HISTORY_KEY, [])

    if not history:
        st.caption("لسه معملتيش أي ترجمة في الجلسة دي. جربي ارفعي فيديو من تبويب رفع فيديو.")
        return

    df = pd.DataFrame(history)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("عدد الترجمات", len(df))
    with col2:
        st.metric("متوسط نسبة الثقة", f"{df['confidence'].mean() * 100:.1f}%")
    with col3:
        confident_ratio = df["is_confident"].mean() if "is_confident" in df else 0
        st.metric("نسبة النتائج الواثقة", f"{confident_ratio * 100:.1f}%")

    st.dataframe(
        df[["predicted_label_ar", "predicted_label_en", "confidence", "is_confident"]]
        .rename(columns={
            "predicted_label_ar": "الجملة (عربي)",
            "predicted_label_en": "Phrase (EN)",
            "confidence": "نسبة الثقة",
            "is_confident": "واثق؟",
        }),
        use_container_width=True,
        hide_index=True,
    )

    if st.button("امسحي السجل"):
        st.session_state[SESSION_HISTORY_KEY] = []
        st.rerun()


def render_dashboard_page() -> None:
    """Renders the statistics/dashboard page."""
    st.markdown("## 📊 الإحصائيات")

    _render_model_info_section()
    st.divider()
    _render_session_history_section()


def log_prediction_to_session(result: dict) -> None:
    """
    Call this from upload_page.py after a successful prediction, so the
    dashboard's session history stays in sync. Not called automatically
    from here to keep this module free of side effects on import.
    """
    history = st.session_state.get(SESSION_HISTORY_KEY, [])
    history.append({
        "predicted_label_ar": result["predicted_label_ar"],
        "predicted_label_en": result["predicted_label_en"],
        "confidence": result["confidence"],
        "is_confident": result["is_confident"],
    })
    st.session_state[SESSION_HISTORY_KEY] = history
