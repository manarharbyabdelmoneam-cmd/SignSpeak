"""
UI Component: Result
======================
Renders a prediction result dict (from VideoService/ModelService) as a
clean result card: the predicted phrase (Arabic + English), a confidence
indicator, and — when confidence is low — a friendly prompt asking the
user to repeat the sign instead of trusting the prediction.
"""

import streamlit as st


def render_result_component(result: dict) -> None:
    """
    Renders the prediction result.

    Expects the dict shape returned by Predictor.predict() /
    ModelService.process_video() / VideoService.process_uploaded_video():
        {
            "predicted_index": int,
            "predicted_label_en": str,
            "predicted_label_ar": str,
            "confidence": float,        # 0.0 - 1.0
            "is_confident": bool,
            "top_3": [ {index, en, ar, confidence}, ... ],
            "metadata": {...},          # optional, not displayed here
        }
    """
    if result is None:
        return

    st.markdown("### النتيجة")

    confidence_percent = result["confidence"] * 100

    if result["is_confident"]:
        st.success(
            f"## {result['predicted_label_ar']}\n"
            f"**{result['predicted_label_en']}**"
        )
    else:
        st.warning(
            "مش متأكدين قوي من الإشارة دي 🤔\n\n"
            f"أقرب تخمين: **{result['predicted_label_ar']}** "
            f"({result['predicted_label_en']})\n\n"
            "جربي تعيدي الإشارة تاني بإضاءة أوضح، وخلي إيدك في نص الكاميرا."
        )

    # --- Confidence bar ---
    st.progress(min(result["confidence"], 1.0))
    st.caption(f"نسبة الثقة: {confidence_percent:.1f}%")

    # --- Top 3 alternatives ---
    top_3 = result.get("top_3", [])
    if len(top_3) > 1:
        with st.expander("احتمالات تانية"):
            for candidate in top_3:
                st.markdown(
                    f"- **{candidate['ar']}** ({candidate['en']}) "
                    f"— {candidate['confidence'] * 100:.1f}%"
                )
