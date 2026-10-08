import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

from captionmodel import analyzeimage
from utils.translator import translatecaption, SUPPORTEDLANGUAGES
from utils.texttospeech import generateaudio
from utils.sentiment import get_sentiment
from utils.evaluation import calculate_scores

st.set_page_config(page_title="Advanced AI Captioning", layout="wide")

st.title("🧠 Next-Gen AI Image Captioning System")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(image, use_container_width=True)

    # analyzeimage now returns xai_explanations as well
    caption, annotatedimg, df, heatmap, xai_explanations = analyzeimage(image)

    st.subheader("📝 Caption")
    st.success(caption)

    # Language
    language = st.selectbox("🌍 Select Language", sorted(SUPPORTEDLANGUAGES.keys()))
    translated = translatecaption(caption, language)

    st.subheader("🌐 Translation")
    st.info(translated)

    # Audio
    lang_code = SUPPORTEDLANGUAGES[language]
    audio = generateaudio(translated, lang_code)
    st.audio(audio)

    # Sentiment
    st.subheader("😊 Sentiment")
    st.write(get_sentiment(caption))

    # Detection
    st.subheader("📦 Detection")
    st.image(annotatedimg)

    # XAI Heatmap
    st.subheader("🔥 Explainable AI — Attention Heatmap")
    st.caption("Warmer regions (red/yellow) indicate areas where the model focused most attention during detection.")
    st.image(heatmap)

    # ── NEW: Per-Object XAI Explanations ──────────────────────────────────────
    st.subheader("🔍 Why Did the Model Detect These Objects?")
    st.caption(
        "Each card below explains **why** YOLOv8 predicted that particular object — "
        "including confidence level, size, position, and the visual reasoning behind the decision."
    )

    if xai_explanations:
        for exp in xai_explanations:
            with st.expander(
                f"{exp['conf_icon']} **{exp['object'].upper()}** — "
                f"{exp['confidence']:.1%} confidence ({exp['conf_label']})",
                expanded=True
            ):
                col1, col2 = st.columns([1, 3])

                with col1:
                    if exp["crop"] is not None and exp["crop"].size > 0:
                        st.image(
                            exp["crop"],
                            caption=f"Detected: {exp['object']}",
                            use_container_width=True
                        )
                    else:
                        st.write("_(crop unavailable)_")

                with col2:
                    st.markdown(exp["explanation"])

                    # Confidence bar
                    st.progress(
                        float(exp["confidence"]),
                        text=f"Confidence: {exp['confidence']:.1%}"
                    )

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("📐 Size", exp["size"].split("(")[0].strip())
                    with col_b:
                        st.metric("📍 Position", exp["position"].replace("in the ", "").replace("at the ", ""))
    else:
        st.info("No objects were detected in this image.")
    # ── END XAI ───────────────────────────────────────────────────────────────

    # Data
    st.dataframe(df)

    if not df.empty:
        counts = df["Object"].value_counts()

        fig, ax = plt.subplots()
        counts.plot(kind="bar", ax=ax)
        st.pyplot(fig)

        st.success(f"Top Object: {counts.idxmax()}")

    # Evaluation
    st.subheader("📏 Evaluation")
    ref = st.text_input("Enter Reference Caption")

    if ref:
        scores = calculate_scores(ref, caption)
        st.write(scores)