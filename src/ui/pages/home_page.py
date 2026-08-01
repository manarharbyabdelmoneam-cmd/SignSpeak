"""
UI Page: Home
===============
Landing page: introduces SignBridge AI, explains how it works, and lists
the sign phrases the model currently supports (so users know what to
expect before they try the upload page).
"""

import json

import streamlit as st

from config.settings import LABEL_MAPPING_PATH, APP_TITLE, CONFIDENCE_THRESHOLD


def _load_supported_phrases() -> list[dict]:
    """Read config/label_mapping.json and return a list of {ar, en} dicts."""
    with open(LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels = data["labels"]
    # Sort by numeric class index so the list has a stable, predictable order
    ordered_indices = sorted(labels.keys(), key=int)
    return [labels[idx] for idx in ordered_indices]


def render_home_page() -> None:
    """Renders the landing/home page."""
    st.markdown(f"# 🤟 {APP_TITLE}")
    st.markdown(
        "### مساعد ذكي لترجمة **لغة الإشارة العربية (ArSL)** لجمل مكتوبة، "
        "باستخدام تحليل حركة اليدين والجسم عن طريق الذكاء الاصطناعي."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### إزاي يشتغل؟")
        st.markdown(
            "1. ارفعي فيديو قصير (لحد 15 ثانية) بتأدي فيه الإشارة\n"
            "2. النظام بيستخرج حركة اليدين والجسم باستخدام MediaPipe\n"
            "3. موديل الذكاء الاصطناعي (BiLSTM) بيتنبأ بالجملة المقصودة\n"
            "4. بتشوفي النتيجة بالعربي والإنجليزي مع نسبة الثقة"
        )

    with col2:
        st.markdown("#### نصايح لأفضل نتيجة")
        st.markdown(
            "- إضاءة كويسة وواضحة\n"
            "- إيدك وجسمك الفوقاني ظاهرين قدام الكاميرا\n"
            "- أدي الإشارة بوضوح ومن غير حركة زيادة حواليها\n"
            "- فيديو قصير ومركز على الإشارة نفسها"
        )

    st.divider()

    st.markdown("#### الإشارات المتاحة حاليًا")
    st.caption(
        f"الموديل مدرّب حاليًا على {len(_load_supported_phrases())} جملة/عبارة. "
        f"أي تنبؤ بنسبة ثقة أقل من {int(CONFIDENCE_THRESHOLD * 100)}% هيتطلب منك تعيدي المحاولة."
    )

    phrases = _load_supported_phrases()
    phrase_cols = st.columns(3)
    for index, phrase in enumerate(phrases):
        with phrase_cols[index % 3]:
            st.markdown(f"- **{phrase['ar']}** *({phrase['en']})*")

    st.divider()
    st.info("ابدئي من تبويب **رفع فيديو** في القائمة الجانبية 👈")
