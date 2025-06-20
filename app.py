import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import os

# --- Load Trained Model ---
@st.cache_resource  # Efficient caching in Streamlit
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "text_emotion.pkl")
    if os.path.exists(model_path):
        return joblib.load(open(model_path, "rb"))
    else:
        st.error(f"Model file not found at: {model_path}")
        st.stop()

pipe_lr = load_model()

# --- Emoji Dictionary ---
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂",
    "neutral": "😐", "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"
}

# --- Prediction Functions ---
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

# --- Main App ---
def main():
    st.title("🧠 Text Emotion Detection")
    st.subheader("Detect the emotion behind any message!")

    with st.form(key='my_form'):
        raw_text = st.text_area("Enter your text below 👇", height=150)
        submit_text = st.form_submit_button(label='Analyze Emotion')

    if submit_text and raw_text.strip():
        col1, col2 = st.columns(2)

        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        with col1:
            st.success("🔤 Original Text")
            st.write(raw_text)

            st.success("🔮 Prediction")
            emoji_icon = emotions_emoji_dict.get(prediction, "")
            st.write(f"**{prediction.capitalize()}** {emoji_icon}")
            st.write(f"**Confidence:** {np.max(probability):.2f}")

        with col2:
            st.success("📊 Prediction Probability")
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["Emotion", "Probability"]

            chart = alt.Chart(proba_df_clean).mark_bar().encode(
                x=alt.X('Emotion', sort='-y'),
                y='Probability',
                color='Emotion'
            )
            st.altair_chart(chart, use_container_width=True)

    elif submit_text:
        st.warning("⚠️ Please enter some text before submitting.")

if __name__ == '__main__':
    main()