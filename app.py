import streamlit as st
import joblib
import os

# --- Load models ---
svm_model = joblib.load("svm_model.pkl")
mnb_model = joblib.load("mnb_model.pkl")
log_reg_model = joblib.load("log_reg_model.pkl")

# --- Try loading vectorizer ---
vectorizer = None
for vec_name in ["vectorizer.pkl", "cv.pkl", "tfidf.pkl"]:
    if os.path.exists(vec_name):
        vectorizer = joblib.load(vec_name)
        break

if vectorizer is None:
    st.error("❌ Could not find vectorizer.pkl (or cv.pkl / tfidf.pkl). Please save your vectorizer from the notebook.")
    st.stop()

# --- Dictionary for model selection ---
models = {
    "Support Vector Machine": svm_model,
    "Naive Bayes": mnb_model,
    "Logistic Regression": log_reg_model,
}

# --- Streamlit UI ---
st.set_page_config(page_title="Amazon Food Review Sentiment", page_icon="🍴", layout="wide")
st.title("🍴 Amazon Food Review Sentiment Analysis")

# Dropdown for model selection
selected_model_name = st.selectbox("🔽 Choose a Model:", list(models.keys()))
selected_model = models[selected_model_name]

st.write("Enter a food review below and analyze sentiment using your chosen model.")

# User input text
review_text = st.text_area("✍️ Write your review here:", height=150)

if st.button("🔍 Analyze Sentiment"):
    if review_text.strip():
        # Vectorize text
        X = vectorizer.transform([review_text])

        # Prediction
        prediction = selected_model.predict(X)[0]

        # Confidence score (if available)
        try:
            prob = selected_model.predict_proba(X)[0]
            confidence = max(prob) * 100
        except:
            confidence = None

        # Display result
        if prediction == 1:
            st.success(f"✅ Positive Review" + (f" (Confidence: {confidence:.2f}%)" if confidence else ""))
        else:
            st.error(f"❌ Negative Review" + (f" (Confidence: {confidence:.2f}%)" if confidence else ""))
    else:
        st.warning("⚠️ Please enter a review to analyze.")
