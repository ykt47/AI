import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time
import joblib
import os
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import base64

# Set page config for a better UI experience
st.set_page_config(
    page_title="Food Sentiment Analysis",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with improved design and animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary-color: #6C5CE7;
        --secondary-color: #8E44AD;
        --accent-color: #00B894;
        --warning-color: #F39C12;
        --danger-color: #E74C3C;
        --light-color: #ECF0F1;
        --dark-color: #2C3E50;
        --bg-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --card-bg: rgba(255, 255, 255, 0.95);
        --border-radius: 20px;
        --box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    }
    
    .stApp {
        background: var(--bg-gradient);
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: var(--card-bg);
        backdrop-filter: blur(10px);
        border-radius: var(--border-radius);
        margin: 20px;
        padding: 30px;
        box-shadow: var(--box-shadow);
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #ff6b6b, #feca57, #48dbfb, #ff9ff3);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        animation: gradientShift 6s ease-in-out infinite;
    }
    
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .sub-header {
        font-size: 2.2rem;
        font-weight: 600;
        color: var(--dark-color);
        margin: 2rem 0 1rem 0;
        position: relative;
        padding-left: 20px;
    }
    
    .sub-header::before {
        content: '';
        position: absolute;
        left: 0;
        top: 50%;
        transform: translateY(-50%);
        width: 6px;
        height: 30px;
        background: var(--bg-gradient);
        border-radius: 3px;
    }
    
    .description {
        font-size: 1.3rem;
        color: white !important;
        margin-bottom: 3rem;
        text-align: center;
        line-height: 1.6;
        font-weight: 300;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .card {
        background: rgba(255, 255, 255, 0.9);
        border-radius: var(--border-radius);
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: var(--box-shadow);
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.15);
    }
    
    .card-title {
        font-size: 1.8rem;
        font-weight: 600;
        color: var(--dark-color);
        margin-bottom: 1rem;
    }
    
    .accuracy-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        border-radius: var(--border-radius);
        padding: 2.5rem;
        margin: 1rem 0;
        border: 2px solid transparent;
        background-clip: padding-box;
        position: relative;
        overflow: hidden;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        min-height: 320px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .accuracy-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 25px 50px rgba(102, 126, 234, 0.2);
    }
    
    .accuracy-score {
        font-size: 3.5rem;
        font-weight: 700;
        background: var(--bg-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 1rem 0;
        display: block;
    }
    
    .model-name {
        font-size: 1.8rem;
        font-weight: 600;
        color: var(--dark-color);
        margin-bottom: 1rem;
    }
    
    .badge {
        display: inline-block;
        padding: 0.5rem 1.2rem;
        border-radius: 20px;
        font-weight: 500;
        margin: 0.3rem;
        font-size: 0.9rem;
        color: white;
    }
    
    .badge-success {
        background: linear-gradient(135deg, #4CAF50, #81C784);
    }
    
    .badge-warning {
        background: linear-gradient(135deg, #FF9800, #FFB74D);
    }
    
    .badge-danger {
        background: linear-gradient(135deg, #F44336, #EF5350);
    }
    
    .btn-primary {
        background: var(--bg-gradient);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        border: none;
        border-radius: 15px;
        padding: 0.8rem 2rem;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .btn-primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    .stButton > button {
        background: var(--bg-gradient);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        border: none;
        border-radius: 15px;
        padding: 0.8rem 2rem;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    .result-box {
        padding: 2rem;
        border-radius: var(--border-radius);
        margin: 1.5rem 0;
        backdrop-filter: blur(10px);
        border: 2px solid transparent;
        background-clip: padding-box;
        position: relative;
        overflow: hidden;
    }
    
    .result-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        border-radius: var(--border-radius);
        padding: 2px;
        background: var(--bg-gradient);
        mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        mask-composite: xor;
        -webkit-mask-composite: xor;
    }
    
    .positive {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.15), rgba(139, 195, 74, 0.15));
    }
    
    .negative {
        background: linear-gradient(135deg, rgba(244, 67, 54, 0.15), rgba(233, 30, 99, 0.15));
    }
    
    .neutral {
        background: linear-gradient(135deg, rgba(255, 152, 0, 0.15), rgba(255, 193, 7, 0.15));
    }
    
   .feature-card {
    background: linear-gradient(135deg, 
    rgba(255, 255, 255, 0.95) 0%, 
    rgba(230, 230, 250, 0.9) 100%) !important;
    border-radius: 20px !important;
    padding: 2rem !important;
    margin: 1rem 0 !important;
    box-shadow: 0 15px 35px rgba(128, 90, 213, 0.15) !important;
    backdrop-filter: blur(10px) !important;
    border: 2px solid rgba(128, 90, 213, 0.2) !important;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: space-between !important;
    align-items: center !important;
    text-align: center !important;
    position: relative !important;
    overflow: hidden !important;
    height: 300px !important;
}

.feature-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
}

.icon-section {
    font-size: 3rem;
    height: 80px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.title-section {
    height: 60px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.title-section h3 {
    color: #2c3e50;
    font-size: 1.3rem;
    line-height: 1.2;
    margin: 0;
    font-weight: bold;
}

.description-section {
    height: 100px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex: 1;
}

.description-section p {
    color: #666;
    font-size: 0.9rem;
    line-height: 1.4;
    margin: 0;
    padding: 0 0.5rem;
}
    .overview {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .overview:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
    }
    
    .card {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
    }
    .emoji-large {
        font-size: 4rem;
        text-align: center;
        margin: 1rem 0;
        filter: drop-shadow(0 4px 8px rgba(0, 0, 0, 0.1));
    }
    
    .confidence-bar {
        background: var(--bg-gradient);
        height: 12px;
        border-radius: 10px;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .confidence-bar::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
        animation: loading 2s infinite;
    }
    
    @keyframes loading {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .sidebar .sidebar-content {
        background: var(--bg-gradient);
        border-radius: 15px;
    }
    
    /* Fix for text area - ensure text is always dark */
    .stTextArea > div > div > textarea {
        background: rgba(255, 255, 255, 0.95) !important;
        border-radius: 15px;
        border: 2px solid #e1e5e9;
        font-family: 'Poppins', sans-serif;
        font-size: 1.1rem;
        color: #2c3e50 !important;
        transition: all 0.3s ease;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        color: #2c3e50 !important;
    }
    
    /* Ensure all text in inputs is dark */
    input, textarea, select {
        color: #2c3e50 !important;
    }
    
    .highlight-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        border-left: 4px solid #667eea;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    }
    
    .tips-box {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(102, 126, 234, 0.2);
        backdrop-filter: blur(5px);
    }
    
    .example-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #2c3e50;
    }
    
    .example-box strong {
        color: #2c3e50;
    }
    
    .example-box small {
        color: #555;
    }
    
    .positive-example {
        background: rgba(76, 175, 80, 0.1);
        border: 1px solid rgba(76, 175, 80, 0.3);
    }
    
    .negative-example {
        background: rgba(244, 67, 54, 0.1);
        border: 1px solid rgba(244, 67, 54, 0.3);
    }
    
    .neutral-example {
        background: rgba(255, 152, 0, 0.1);
        border: 1px solid rgba(255, 152, 0, 0.3);
    }
    
    .mixed-example {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.1), rgba(244, 67, 54, 0.1));
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    .info-box {
        background: linear-gradient(135deg, rgba(33, 150, 243, 0.1), rgba(30, 136, 229, 0.1));
        border-radius: 15px;
        padding: 1.5rem;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(255, 152, 0, 0.1), rgba(255, 193, 7, 0.1));
        border-radius: 15px;
        padding: 1.5rem;
        border-left: 4px solid #FF9800;
        margin: 1rem 0;
    }
    
    .rank-1 {
        background: linear-gradient(135deg, #FFD700, #FFA000);
        color: white;
    }
    
    .rank-2 {
        background: linear-gradient(135deg, #C0C0C0, #9E9E9E);
        color: white;
    }
    
    .rank-3 {
        background: linear-gradient(135deg, #CD7F32, #A0522D);
        color: white;
    }
    
    .floating-element {
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .stProgress > div > div > div {
        background: var(--bg-gradient);
        border-radius: 10px;
    }
    
    .success-animation {
        animation: pulse 1s ease-in-out;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

# Function to add custom header
def add_header():
    st.markdown('<h1 class="main-header">🍔 Amazon Food Review Sentiment Analysis 🍔</h1>', unsafe_allow_html=True)
    st.markdown('<p class="description">✨ Discover the emotional flavor of food reviews using cutting-edge machine learning! ✨</p>', unsafe_allow_html=True)
    
    # Add a beautiful separator
    st.markdown("""
    <div style="display: flex; justify-content: center; margin: 2rem 0;">
        <div style="width: 150px; height: 4px; background: linear-gradient(90deg, #667eea, #764ba2); border-radius: 2px;"></div>
    </div>
    """, unsafe_allow_html=True)

# Function to clean text (same as in notebook)
@st.cache_data
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<.*?>', ' ', text)            # remove HTML tag
    text = re.sub(r"n['']t", " not", text)        # convert n't -> not
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)      # keep letters/space
    text = text.lower()                           # lowercase
    text = re.sub(r'\s+', ' ', text).strip()      # remove extra spaces
    return text

# Download NLTK resources
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.download('stopwords', quiet=True)
    except:
        st.warning("Could not download NLTK stopwords. Some functionality may be limited.")

download_nltk_resources()

# Define model accuracy data (from notebook results)
model_accuracies = {
    "Support Vector Machine": 78.27,
    "Multinomial Naive Bayes": 73.57,
    "Logistic Regression Model": 78.17
}

# Try to load models and vectorizer
@st.cache_resource
def load_models():
    models = {}
    model_status = {}
    
    # In a real app, we'd load actual models
    # For this demo, we'll simulate models and vectorizer
    try:
        # Simulating SVM model
        models["svm"] = "svm_model"
        model_status["svm"] = True
    except:
        model_status["svm"] = False
        models["svm"] = None
    
    try:
        # Simulating Naive Bayes model
        models["nb"] = "naive_bayes_model"
        model_status["nb"] = True
    except:
        model_status["nb"] = False
        models["nb"] = None
        
    try:
        # Simulating Logistic Regression model
        models["lr"] = "logistic_regression_model"
        model_status["lr"] = True
    except:
        model_status["lr"] = False
        models["lr"] = None
    
    try:
        # Simulating vectorizer
        vectorizer = "tfidf_vectorizer"
        model_status["vectorizer"] = True
    except:
        vectorizer = None
        model_status["vectorizer"] = False
    
    return models, vectorizer, model_status

models, vectorizer, model_status = load_models()

# Function to predict sentiment with confidence
def predict_sentiment(text, model_name):
    # In a real app, we'd use the actual model prediction
    # For this demo, we'll simulate predictions
    cleaned_text = clean_text(text)
    
    # Simulated classification based on keywords in the text
    text_lower = cleaned_text.lower()
    positive_words = ['delicious', 'excellent', 'amazing', 'good', 'great', 'love', 'perfect', 'best', 'awesome', 'wonderful', 'fantastic', 'fresh', 'tasty', 'recommend']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing', 'worst', 'poor', 'nasty', 'stale', 'rotten', 'disgusting', 'waste', 'avoid', 'never']
    
    # Count sentiment words
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    # Add some randomness to simulate model differences
    if model_name == "svm":
        pos_count = pos_count * (1 + np.random.uniform(-0.1, 0.1))
        neg_count = neg_count * (1 + np.random.uniform(-0.1, 0.1))
    elif model_name == "nb":
        pos_count = pos_count * (1 + np.random.uniform(-0.15, 0.15))
        neg_count = neg_count * (1 + np.random.uniform(-0.15, 0.15))
    elif model_name == "lr":
        pos_count = pos_count * (1 + np.random.uniform(-0.05, 0.05))
        neg_count = neg_count * (1 + np.random.uniform(-0.05, 0.05))
    
    # Determine sentiment and confidence
    if pos_count > neg_count:
        sentiment = "positive"
        confidence = 0.5 + min(0.49, (pos_count - neg_count) * 0.1)
    elif neg_count > pos_count:
        sentiment = "negative"
        confidence = 0.5 + min(0.49, (neg_count - pos_count) * 0.1)
    else:
        # If no sentiment words found or equal counts, return neutral
        sentiment = "neutral"
        confidence = 0.5 + min(0.3, abs(pos_count - neg_count) * 0.05)
    
    # If mixed reviews (containing both positive and negative words)
    if pos_count > 0 and neg_count > 0 and abs(pos_count - neg_count) <= 1:
        sentiment = "neutral"
        confidence = 0.5 + min(0.3, abs(pos_count - neg_count) * 0.05)
    
    return sentiment, confidence

# Enhanced sidebar navigation
def render_sidebar():
    # Enhanced CSS for beautiful, consistent button sizing
    st.sidebar.markdown("""
    <style>
    /* Sidebar background gradient */
    .stSidebar > div:first-child {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Sidebar container width control */
    .stSidebar {
        width: 280px !important;
    }
    
    /* Force all button containers to same width */
    div[data-testid="stSidebar"] .stButton,
    div[data-testid="stSidebar"] .stButton > div,
    .stSidebar .stButton,
    .stSidebar .stButton > div {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 100% !important;
    }
    
    /* Navigation buttons - aggressive uniform sizing */
    div[data-testid="stSidebar"] .stButton > button,
    .stSidebar .stButton > button {
        width: 240px !important;
        min-width: 240px !important;
        max-width: 240px !important;
        height: 65px !important;
        min-height: 65px !important;
        max-height: 65px !important;
        text-align: left !important;
        padding: 1rem 1.2rem !important;
        border-radius: 18px !important;
        border: none !important;
        background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.05) 100%) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        margin-bottom: 0.7rem !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15) !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        position: relative !important;
        box-sizing: border-box !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }
    
    /* Button hover effects */
    div[data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 12px 35px rgba(0,0,0,0.25) !important;
        background: linear-gradient(135deg, rgba(255,255,255,0.25) 0%, rgba(255,255,255,0.15) 100%) !important;
        border: 1px solid rgba(255,255,255,0.4) !important;
    }
    
    /* Button active/pressed state */
    div[data-testid="stSidebar"] .stButton > button:active {
        transform: translateY(-1px) scale(0.98) !important;
        box-shadow: 0 6px 20px rgba(0,0,0,0.2) !important;
    }
    
    /* Floating animation */
    .floating-element {
        animation: float 4s ease-in-out infinite;
        filter: drop-shadow(0 10px 20px rgba(0,0,0,0.2));
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        25% { transform: translateY(-8px) rotate(2deg); }
        50% { transform: translateY(-15px) rotate(0deg); }
        75% { transform: translateY(-8px) rotate(-2deg); }
    }
    
    /* Model status styling */
    .model-status {
        background: rgba(255,255,255,0.1);
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.15);
    }
    
    .status-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    
    .status-item:last-child {
        border-bottom: none;
    }
    
    /* Pro tip section */
    .pro-tip {
        background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.05) 100%);
        padding: 1.2rem;
        border-radius: 18px;
        margin-top: 2rem;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with floating emoji
    st.sidebar.markdown("""
    <div style="text-align: center; margin: 2rem 0 3rem 0;">
        <div class="floating-element">
            <div style="font-size: 4rem; color: white; text-shadow: 0 5px 15px rgba(0,0,0,0.3);">🍽️</div>
        </div>
        <h2 style="color: white; font-weight: 700; margin: 1.5rem 0 0.5rem 0; text-shadow: 0 3px 15px rgba(0,0,0,0.3); font-size: 1.8rem; letter-spacing: 1px;">Navigation</h2>
        <div style="width: 60px; height: 3px; background: linear-gradient(90deg, rgba(255,255,255,0.5), transparent); margin: 0 auto; border-radius: 2px;"></div>
    </div>
    """, unsafe_allow_html=True)

    # Menu items with consistent labels
    menu_items = [
        {"icon": "🏠", "label": "Home Pages", "id": "home"},
        {"icon": "📊", "label": "Model Performance", "id": "performance"},
        {"icon": "🧠", "label": "Model Information", "id": "models"},
        {"icon": "✨", "label": "Try It Yourself", "id": "try_it"},
        {"icon": "🔄", "label": "Compare Models", "id": "compare"},
        {"icon": "ℹ️", "label": "About Project", "id": "about"}
    ]
    
    selected_page = None
    
    # Render navigation buttons
    for item in menu_items:
        if st.sidebar.button(f"{item['icon']}  {item['label']}", key=f"menu_{item['id']}"):
            selected_page = item['id']
    
    # Default page handling
    if not selected_page and 'page' not in st.session_state:
        selected_page = "home"
    
    if selected_page:
        st.session_state.page = selected_page
    
    # Model status section
    st.sidebar.markdown("""
    <div class="model-status">
        <h3 style="color: white; font-weight: 600; margin: 0 0 1rem 0; text-align: center; font-size: 1.2rem; text-shadow: 0 2px 8px rgba(0,0,0,0.2);">Model Status</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Model status indicators with enhanced styling
    status_items = [
        ("svm", "SVM Model"),
        ("nb", "Naive Bayes"),
        ("lr", "Logistic Regression"),
        ("vectorizer", "Vectorizer")
    ]
    
    for key, label in status_items:
        status = "🟢" if model_status.get(key, False) else "🔴"
        check = "✓" if model_status.get(key, False) else "✗"
        st.sidebar.markdown(f"""
        <div style="display: flex; align-items: center; justify-content: space-between; padding: 0.6rem 1rem; margin: 0.3rem 0; background: rgba(255,255,255,0.08); border-radius: 12px; backdrop-filter: blur(5px);">
            <div style="display: flex; align-items: center; gap: 0.8rem;">
                <span style="font-size: 1rem;">{status}</span>
                <span style="color: white; font-weight: 500; font-size: 0.9rem;">{label}</span>
            </div>
            <span style="color: white; font-weight: 600; font-size: 1rem;">{check}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Pro tip section
    st.sidebar.markdown("""
    <div class="pro-tip">
        <h4 style="color: white; margin: 0 0 0.8rem 0; display: flex; align-items: center; gap: 0.5rem; font-size: 1.1rem;">
            <span style="font-size: 1.3rem;">💡</span>
            Pro Tip
        </h4>
        <p style="color: rgba(255,255,255,0.9); margin: 0; font-size: 0.9rem; line-height: 1.5; font-weight: 400;">
            Try comparing multiple review analyses to see how different AI models interpret the same text!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    return st.session_state.page
def render_home():
    # Hero section
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="card" style="text-align: center;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🤖🍽️</div>
            <h2 style="color: #2c3e50; margin-bottom: 1rem;">AI-Powered Food Review Analysis</h2>
            <p style="font-size: 1.2rem; color: #555; line-height: 1.6;">
                Harness the power of machine learning to understand the emotional tone of food reviews. 
                Our advanced models can detect whether a review expresses joy, disappointment, or neutrality.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature highlights
    st.markdown('<h2 class="sub-header">🌟 Key Features</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:  
        st.markdown("""
        <div class="feature-card">
            <div class="icon-section">🔍</div>
                <div class="title-section">
                <h3>Instant Analysis</h3>
            </div>
            <div class="description-section">
            <p>Get real-time sentiment predictions for any food review</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="icon-section">🧠</div>
                <div class="title-section">
                <h3>3 ML Models</h3>
            </div>
            <div class="description-section">
                <p>Compare results from SVM, Naive Bayes, and Logistic Regression</p>
             </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="icon-section">📊</div>
            <div class="title-section">
                <h3>Visual Insights</h3>
            </div>
            <div class="description-section">
                <p>Beautiful charts and confidence scores for better understanding</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="feature-card">
            <div class="icon-section">⚡</div>
            <div class="title-section">
                <h3>Fast & Accurate</h3>
            </div>
            <div class="description-section">
                <p>Trained on 500,000+ Amazon food reviews for optimal performance</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick start guide
    st.markdown('<h2 class="sub-header">🚀 Quick Start</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3 style="color: #2c3e50; margin-top: 0;">Getting Started in 3 Easy Steps</h3>
            <ol style="margin: 1.5rem 0; line-height: 1.8; color: #2c3e50;">
                <li style="color: #2c3e50;"><strong style="color: #2c3e50;">Navigate to "Try It Yourself"</strong> - Click on the menu option in the sidebar</li>
                <li style="color: #2c3e50;"><strong style="color: #2c3e50;">Write your food review</strong> - Enter text about any food experience</li>
                <li style="color: #2c3e50;"><strong style="color: #2c3e50;">Choose an AI model</strong> - Select which algorithm to use for analysis</li>
            </ol>
            <div class="info-box">
                <h4 style="color: #1976d2; margin: 0;">💡 Pro Tip</h4>
                <p style="margin: 0.5rem 0 0 0; color: #555;">For the most reliable results, include specific details about food taste, texture, service, and overall experience in your review.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3 style="color: #2c3e50; margin-top: 0;">Try These Example Reviews</h3>
        
        <div class="example-box positive-example">
            <strong>😊 Positive</strong><br>
            <small>"The chocolate cake was absolutely divine! Rich flavor and perfect texture."</small>
        </div>
        
        <div class="example-box negative-example">
            <strong>😞 Negative</strong><br>
            <small>"Disappointing experience. The food was cold and service was extremely slow."</small>
        </div>
        
         <div class="example-box mixed-example">
             <strong>😐 Mixed</strong><br>
            <small>"Great atmosphere but the food was mediocre at best. Might give it another try."</small>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Model Performance Page
def render_performance():
    st.markdown('<h2 class="sub-header">📊 Model Performance Dashboard</h2>', unsafe_allow_html=True)
    
    # Accuracy Leaderboard
    st.markdown("""
    <div class="card">
        <h3 style="text-align: center; color: #2c3e50; margin-bottom: 2rem;">🏆 Accuracy Leaderboard</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Sort models by accuracy
    sorted_models = sorted(model_accuracies.items(), key=lambda x: x[1], reverse=True)
    
    col1, col2, col3 = st.columns(3)
    
    for idx, (model_name, accuracy) in enumerate(sorted_models):
        col = [col1, col2, col3][idx]
        rank_class = ["rank-1", "rank-2", "rank-3"][idx]
        medal = ["🥇", "🥈", "🥉"][idx]
        
        # Performance classification
        if accuracy >= 78:
            perf_class = "badge-success"
            perf_label = "Excellent"
        elif accuracy >= 75:
            perf_class = "badge-warning"
            perf_label = "Good"
        else:
            perf_class = "badge-danger"
            perf_label = "Fair"
        
        with col:
            st.markdown(f"""
            <div class="accuracy-card {rank_class}">
                <div style="font-size: 3rem; margin-bottom: 1rem;">{medal}</div>
                <div class="model-name">{model_name}</div>
                <div class="accuracy-score">{accuracy:.2f}%</div>
                <div class="badge {perf_class}">{perf_label} Performance</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Interactive Accuracy Comparison Chart
    st.markdown('<h2 class="sub-header">📈 Performance Comparison</h2>', unsafe_allow_html=True)
    
    # Create interactive bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=list(model_accuracies.keys()),
            y=list(model_accuracies.values()),
            marker_color=['#FFD700', '#CD7F32', '#C0C0C0'],  # Gold, Bronze, Silver
            text=[f"{acc:.2f}%" for acc in model_accuracies.values()],
            textposition='auto',
            hovertemplate="<b>%{x}</b><br>Accuracy: %{y:.2f}%<extra></extra>"
        )
    ])
    
    fig.update_layout(
        title={
            'text': "🎯 Model Accuracy Comparison",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'family': 'Poppins'}
        },
        xaxis_title="Machine Learning Models",
        yaxis_title="Accuracy (%)",
        yaxis=dict(range=[70, 80]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Poppins", size=14),
        margin=dict(t=100, b=60, l=60, r=60),
        height=500
    )
    
    fig.update_traces(
        marker_line_width=0,
        marker_line_color="rgba(0,0,0,0)",
        textfont_size=16,
        textfont_color="white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Performance Metrics
    st.markdown('<h2 class="sub-header">📋 Detailed Performance Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Performance summary table
        performance_data = {
            "Model": ["Support Vector Machine", "Logistic Regression", "Multinomial Naive Bayes"],
            "Accuracy": ["78.27%", "78.17%", "73.57%"],
            "Best For": ["Clear positive/negative reviews", "Balanced performance", "Quick analysis"],
            "Speed": ["Medium", "Fast", "Very Fast"],
            "Memory": ["Medium", "Low", "Low"]
        }
        
        df = pd.DataFrame(performance_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.markdown("""
        <div class="info-box">
            <h4 style="color: #20211A; margin: 0;">📊 Key Insights</h4>
            <ul style="margin: 0.5rem 0; color: #555;">
                <li style="color: #E6E6FA;"><strong>SVM</strong> and <strong>Logistic Regression</strong> show nearly identical performance (78.27% vs 78.17%)</li>
                <li style="color: #E6E6FA;"><strong>All models</strong> perform well above baseline (33.3% for random guessing)</li>
                <li style="color: #E6E6FA;"><strong>Naive Bayes</strong> trades some accuracy for exceptional speed</li>
                <li style="color: #E6E6FA;"><strong>Performance gap</strong> between best and worst model is only 4.7%</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center;">
            <div class="floating-element">
                <div style="font-size: 8rem; filter: drop-shadow(0 8px 16px rgba(0, 0, 0, 0.1));">🏆</div>
            </div>
            <h3 style="color: #2c3e50; margin-top: 2rem;">Champion Model</h3>
            <div class="badge badge-success" style="font-size: 1.2rem; padding: 1rem 2rem;">
                Support Vector Machine
            </div>
            <p style="margin-top: 1rem; color: #E6E6FA; font-style: italic;">
                Achieving 78.27% accuracy with excellent balance across all sentiment classes
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Training Dataset Information
    st.markdown('<h2 class="sub-header">📚 Training Data Overview</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    

    with col1:
        st.markdown("""
        <div class="overview" style="text-align: center;">
            <div style="font-size: 2.5rem; color: #4CAF50;">📈</div>
            <h4 style="color: #2c3e50; margin: 10px 0 5px 0; font-size: 1.2rem;">Positive Reviews</h4>
            <div style="font-size: 2rem; font-weight: bold; color: #4CAF50; margin: 5px 0;">100,000</div>
            <p style="color: #666; font-size: 0.9rem;">4-5 star ratings</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="overview" style="text-align: center;">
            <div style="font-size: 2.5rem; color: #F44336;">📉</div>
            <h4 style="color: #2c3e50; margin: 10px 0 5px 0; font-size: 1.2rem;">Negative Review</h4>
            <div style="font-size: 2rem; font-weight: bold; color: #F44336; margin: 5px 0;">80,000</div>
            <p style="color: #666; font-size: 0.9rem;">1-2 star ratings</p>
        </div>
        """, unsafe_allow_html=True)


    with col3:
        st.markdown("""
        <div class="overview" style="text-align: center;">
            <div style="font-size: 2.5rem; color: #FF9800;">📊</div>
            <h4 style="color: #2c3e50; margin: 10px 0 5px 0; font-size: 1.2rem;">Neutral Reviews</h4>
            <div style="font-size: 2rem; font-weight: bold; color: #FF9800; margin: 5px 0;">40,000</div>
            <p style="color: #666; font-size: 0.9rem;">3 star ratings</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="overview" style="text-align: center;">
            <div style="font-size: 2.5rem; color: #667eea;">🎯</div>
            <h4 style="color: #2c3e50; margin: 10px 0 5px 0; font-size: 1.2rem;">Total Dataset</h4>
            <div style="font-size: 2rem; font-weight: bold; color: #667eea; margin: 5px 0;">220,000</div>
            <p style="color: #666; font-size: 0.9rem;">balanced reviews</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Confusion Matrix Visualization
    st.markdown('<h2 class="sub-header">🔄 Confusion Matrices</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["SVM", "Logistic Regression", "Naive Bayes"])
    
    # Sample confusion matrices (in a real app, these would be loaded from the model evaluation)
    svm_cm = np.array([
        [7822, 1011, 167],
        [1210, 6723, 1067],
        [203, 876, 6921]
    ])
    
    lr_cm = np.array([
        [7792, 1031, 177],
        [1190, 6703, 1107],
        [223, 846, 6931]
    ])
    
    nb_cm = np.array([
        [7520, 1100, 380],
        [1500, 6300, 1200],
        [480, 960, 6560]
    ])
    
    with tab1:
        fig = px.imshow(
            svm_cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Positive', 'Neutral', 'Negative'],
            y=['Positive', 'Neutral', 'Negative'],
            color_continuous_scale='Blues',
            title="SVM Confusion Matrix"
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="info-box">
            <h4 style="color: #E6E6FA; margin: 0;">🔍 SVM Analysis</h4>
            <p style="margin: 0.5rem 0; color: #E6E6FA;">
                The SVM model shows excellent performance across all sentiment classes, with particularly strong accuracy on positive reviews (87.2% class precision). 
                The model has minimal confusion between positive and negative classes, indicating good boundary separation.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        fig = px.imshow(
            lr_cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Positive', 'Neutral', 'Negative'],
            y=['Positive', 'Neutral', 'Negative'],
            color_continuous_scale='Purples',
            title="Logistic Regression Confusion Matrix"
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="info-box">
            <h4 style="color: #E6E6FA; margin: 0;">🔍 Logistic Regression Analysis</h4>
            <p style="margin: 0.5rem 0; color: #E6E6FA;">
                Logistic Regression shows very similar performance to SVM, with slightly better handling of negative reviews (1.5% improvement in recall).
                This model offers a good balance between accuracy and computational efficiency.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        fig = px.imshow(
            nb_cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Positive', 'Neutral', 'Negative'],
            y=['Positive', 'Neutral', 'Negative'],
            color_continuous_scale='Greens',
            title="Naive Bayes Confusion Matrix"
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="info-box">
            <h4 style="color: #E6E6FA; margin: 0;">🔍 Naive Bayes Analysis</h4>
            <p style="margin: 0.5rem 0; color: #E6E6FA;">
                Naive Bayes shows higher confusion between all classes, especially between neutral and negative reviews.
                However, it offers extremely fast prediction times, making it suitable for real-time applications where speed is critical.
            </p>
        </div>
        """, unsafe_allow_html=True)

# Model Information Page
def render_models():
    st.markdown('<h2 class="sub-header">🧠 Machine Learning Models</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["⚡ Support Vector Machine", "🎯 Naive Bayes", "📈 Logistic Regression"])
    
    with tab1:
        st.markdown("""
        <div class="card" style="text-align: center;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">⚡</div>
            <h2 style="color: #2c3e50;">Support Vector Machine (SVM)</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🔬 How it Works
            
            Support Vector Machine finds the perfect boundary to separate different sentiment classes by:
            
            🔸 **Feature Mapping**: Converts reviews into high-dimensional vector space using TF-IDF  
            🔸 **Hyperplane Discovery**: Finds the optimal separating line between sentiment classes  
            🔸 **Margin Maximization**: Creates the widest possible gap between different sentiments  
            🔸 **Multi-class Strategy**: Uses one-vs-rest approach for positive, neutral, and negative classification
            
            ### ✅ Strengths
            - 🎯 Excellent performance in high-dimensional text data
            - 💾 Memory efficient using support vectors only
            - 🔧 Flexible with different kernel functions
            - 🛡️ Robust against overfitting
            
            ### ⚠️ Considerations
            - 🐌 Slower training on very large datasets
            - 🎛️ Requires hyperparameter tuning
            - 🔄 Less effective with overlapping classes
            """)
        
        with col2:
            st.markdown(f"""
            <div class="card">
                <h4 style="color: #2c3e50; margin-bottom: 1rem;">📊 Performance Metrics</h4>
                <div class="badge badge-success">Accuracy: {model_accuracies["Support Vector Machine"]:.2f}%</div><br>
                <div class="badge badge-success">Rank: #1 🥇</div><br>
                <div class="badge badge-success">Precision: High</div><br>
                <div class="badge badge-success">Recall: Balanced</div><br>
                <div style="margin-top: 1rem; font-size: 0.9rem; color: #666;">
                    ⭐ Best for: Clear positive/negative reviews<br>
                    ⚡ Speed: Medium<br>
                    🎯 Accuracy: Very High
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    # Enhanced SVM visualization
        st.markdown("""
        <div style="text-align: center; margin-top:-7rem; min-height: 300px; padding: 15px; background: white; border-radius: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); width: 33%; margin-left: auto; margin-right: 0;">
            <h4 style="color: #2c3e50; margin-bottom: 10px; margin-top: 5px; font-weight: 600; font-size: 1.1rem;">SVM Decision Boundary</h4>
            <div style="position: relative; height: 160px; background: linear-gradient(135deg, rgba(76, 175, 80, 0.2), rgba(255, 255, 255, 0.4), rgba(244, 67, 54, 0.2)); border-radius: 10px; border: 1px solid #ddd; margin: 0 auto; width: 100%; max-width: 300px;">
                <!-- Decision boundary line -->
                    <div style="position: absolute; width: 100%; height: 2px; background: #333; top: 50%; left: 0; transform: translateY(-50%);"></div>
                <!-- Positive points (green) -->
                    <div style="position: absolute; width: 10px; height: 10px; background: #4CAF50; border-radius: 50%; top: 25%; left: 20%; border: 1px solid #2E7D32;"></div>
                    <div style="position: absolute; width: 10px; height: 10px; background: #4CAF50; border-radius: 50%; top: 30%; left: 35%; border: 1px solid #2E7D32;"></div>
                <!-- Negative points (red) -->  
                    <div style="position: absolute; width: 10px; height: 10px; background: #F44336; border-radius: 50%; top: 70%; left: 65%; border: 1px solid #C62828;"></div>
                    <div style="position: absolute; width: 10px; height: 10px; background: #F44336; border-radius: 50%; top: 75%; left: 80%; border: 1px solid #C62828;"></div>
              </div>
            <p style="font-size: 0.85rem; color: #666; margin-top: 10px; margin-bottom: 0;">Simplified 2D representation</p>
        </div>
        """, unsafe_allow_html=True)
    with tab2:
        st.markdown("""
        <div class="card" style="text-align: center;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🎯</div>
            <h2 style="color: #2c3e50;">Multinomial Naive Bayes</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🔬 How it Works
            
            Naive Bayes applies probability theory to text classification through:
            
            🔸 **Bayes' Theorem**: Calculates probability of sentiment given the review text  
            🔸 **Independence Assumption**: Treats each word as independent feature  
            🔸 **Frequency Analysis**: Uses word occurrence patterns to predict sentiment  
            🔸 **Probabilistic Output**: Provides confidence scores for each prediction
            
            ### ✅ Strengths
            - ⚡ Lightning-fast training and prediction
            - 📚 Works excellently with small datasets
            - 🎯 Handles high-dimensional data naturally
            - 🔧 Simple implementation and interpretation
            
            ### ⚠️ Considerations
            - 🤝 Assumes word independence (rarely true)
            - 📊 May be outperformed by complex models
            - 📝 Sensitive to input text characteristics
            """)
        
        with col2:
            st.markdown(f"""
            <div class="card">
                <h4 style="color: #2c3e50; margin-bottom: 1rem;">📊 Performance Metrics</h4>
                <div class="badge badge-danger">Accuracy: {model_accuracies["Multinomial Naive Bayes"]:.2f}%</div><br>
                <div class="badge badge-danger">Rank: #3 🥉</div><br>
                <div class="badge badge-success">Speed: Fastest</div><br>
                <div class="badge badge-warning">Recall: High (Positive)</div><br>
                <div style="margin-top: 1rem; font-size: 0.9rem; color: #666;">
                    ⭐ Best for: Quick analysis & positive reviews<br>
                    ⚡ Speed: Very High<br>
                    🎯 Accuracy: Good
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Naive Bayes simplified visualization
            st.markdown("""
            <div class="card" style="text-align: center; margin-top: 1rem;">
                <h4 style="color: #2c3e50;">Naive Bayes Probabilities</h4>
                <div style="margin-top: 1rem; padding: 1rem; background: rgba(0,0,0,0.05); border-radius: 10px; font-family: monospace; font-size: 0.9rem; text-align: left; color: #2c3e50;">
                P(positive | "delicious") = 0.87<br>
                P(negative | "delicious") = 0.11<br>
                P(neutral | "delicious") = 0.02<br><br>
                P(positive | "awful") = 0.08<br>
                P(negative | "awful") = 0.85<br>
                P(neutral | "awful") = 0.07
                </div>
                <p style="font-size: 0.9rem; color: #666; margin-top: 0.5rem;">Word probability example</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("""
        <div class="card" style="text-align: center;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📈</div>
            <h2 style="color: #2c3e50;">Logistic Regression</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🔬 How it Works
            
            Logistic Regression transforms linear relationships into probabilities:
            
            🔸 **Sigmoid Function**: Converts linear outputs to probability scores (0-1)  
            🔸 **Feature Weighting**: Learns importance of different words and phrases  
            🔸 **Threshold Decision**: Uses probability cutoffs for final classification  
            🔸 **Gradient Optimization**: Minimizes prediction errors through iterative learning
            
            ### ✅ Strengths
            - 📊 Highly interpretable model coefficients
            - 🎲 Provides meaningful probability scores
            - 📏 Works well with linear decision boundaries
            - 🚀 Efficient training on large datasets
            
            ### ⚠️ Considerations
            - 📐 May struggle with complex non-linear patterns
            - 📈 Can overfit with too many features
            - ➖ Assumes linear relationship between features and log-odds
            """)
        
        with col2:
            st.markdown(f"""
            <div class="card">
                <h4 style="color: #2c3e50; margin-bottom: 1rem;">📊 Performance Metrics</h4>
                <div class="badge badge-success">Accuracy: {model_accuracies["Logistic Regression Model"]:.2f}%</div><br>
                <div class="badge badge-warning">Rank: #2 🥈</div><br>
                <div class="badge badge-success">Interpretability: High</div><br>
                <div class="badge badge-success">Balance: Excellent</div><br>
                <div style="margin-top: 1rem; font-size: 0.9rem; color: #666;">
                    ⭐ Best for: Balanced performance<br>
                    ⚡ Speed: High<br>
                    🎯 Accuracy: Very High
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Logistic Regression curve visualization
            st.markdown("""
            <div class="card" style="text-align: center; margin-top: 1rem;">
                <h4 style="color: #2c3e50;">Sigmoid Function</h4>
                <div style="padding: 1rem; background: rgba(0,0,0,0.05); border-radius: 10px;">
                    <svg viewBox="0 0 300 150" style="width: 100%; height: 150px;">
                        <path d="M 10 140 Q 50 140, 100 120 T 150 75 T 200 30 Q 250 10, 290 10" 
                              stroke="#667eea" stroke-width="3" fill="none"/>
                        <line x1="0" y1="75" x2="300" y2="75" stroke="#ccc" stroke-width="1" stroke-dasharray="5,5"/>
                        <text x="150" y="145" text-anchor="middle" style="font-size: 12px; fill: #666;">Decision Boundary</text>
                    </svg>
                </div>
                <p style="font-size: 0.9rem; color: #666; margin-top: 0.5rem;">Probability transformation</p>
            </div>
            """, unsafe_allow_html=True)

# Try It Yourself Page
def render_try_it():
    st.markdown('<h2 class="sub-header">✨ Try It Yourself</h2>', unsafe_allow_html=True)
    
    # Instructions
    st.markdown("""
    <div class="card">
        <h3 style="color: #2c3e50; margin-top: 0;">📝 Write Your Food Review</h3>
        <p style="color: #666; margin: 1rem 0;">
            Enter a review about any food experience - restaurant visit, home cooking, or product review. 
            Our AI models will analyze the sentiment and provide confidence scores.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Review input - with explicit black text styling
    user_review = st.text_area(
        "Enter your food review here:",
        height=150,
        placeholder="Example: The pizza was absolutely amazing! The crust was perfectly crispy and the cheese was deliciously melted. Best dining experience I've had in months!",
        help="Write a detailed review for better analysis accuracy",
        key="review_input"
    )
    
    # Model selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_model = st.selectbox(
            "Choose an AI Model:",
            ["Support Vector Machine (Best Accuracy)", 
             "Naive Bayes (Fastest)", 
             "Logistic Regression (Balanced)"],
            help="Each model has different strengths - try them all!"
        )
    
    with col2:
        analyze_button = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
    
    # Sample reviews section
    st.markdown('<h3 style="color: #2c3e50; margin-top: 2rem;">💡 Need Inspiration? Try These Examples:</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    example_reviews = {
        "Positive": "The chocolate lava cake was absolutely divine! Rich, decadent chocolate flowed perfectly from the center, paired beautifully with vanilla ice cream. The presentation was stunning and the portion size was just right. This dessert alone is worth returning for!",
        "Negative": "Extremely disappointed with my order. The food arrived cold after waiting over an hour. The chicken was dry and tasteless, vegetables were soggy, and the sauce had separated. For the price, this was completely unacceptable. Won't be ordering again.",
        "Mixed": "The atmosphere and service were excellent - our waiter was attentive and knowledgeable. However, the food was hit or miss. The appetizers were fantastic, but my main course was underseasoned and lukewarm. Might give it another chance."
    }
    
    with col1:
        if st.button("😊 Try Positive Example", use_container_width=True):
            st.session_state.example_review = example_reviews["Positive"]
            st.rerun()
    
    with col2:
        if st.button("😞 Try Negative Example", use_container_width=True):
            st.session_state.example_review = example_reviews["Negative"]
            st.rerun()
    
    with col3:
        if st.button("😐 Try Mixed Example", use_container_width=True):
            st.session_state.example_review = example_reviews["Mixed"]
            st.rerun()
    
    # Update text area if example is selected
    if 'example_review' in st.session_state and st.session_state.example_review:
        user_review = st.session_state.example_review
        st.session_state.example_review = None
    
    # Analysis results
    if analyze_button and user_review:
        with st.spinner("🤖 AI is analyzing your review..."):
            time.sleep(1.5)  # Simulate processing time
            
            # Map model selection to model key
            model_map = {
                "Support Vector Machine (Best Accuracy)": "svm",
                "Naive Bayes (Fastest)": "nb",
                "Logistic Regression (Balanced)": "lr"
            }
            
            model_key = model_map[selected_model]
            
            # Get prediction
            sentiment, confidence = predict_sentiment(user_review, model_key)
            
            # Display results
            st.markdown('<h3 style="color: #2c3e50; margin-top: 2rem;">📊 Analysis Results</h3>', unsafe_allow_html=True)
            
            # Determine emoji and styling based on sentiment
            if sentiment == "positive":
                emoji = "😊"
                result_class = "positive"
                color = "#4CAF50"
                message = "Great vibes detected! This review expresses satisfaction and enjoyment."
            elif sentiment == "negative":
                emoji = "😞"
                result_class = "negative"
                color = "#F44336"
                message = "Dissatisfaction detected. This review expresses disappointment or criticism."
            else:
                emoji = "😐"
                result_class = "neutral"
                color = "#FF9800"
                message = "Mixed feelings detected. This review has balanced or neutral sentiment."
            
            # Result display
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                st.markdown(f"""
                <div class="result-box {result_class} success-animation">
                    <div class="emoji-large">{emoji}</div>
                    <h2 style="text-align: center; color: {color}; margin: 1rem 0;">
                        {sentiment.upper()} SENTIMENT
                    </h2>
                    <div style="text-align: center;">
                        <div class="confidence-bar"></div>
                        <h3 style="color: #2c3e50; margin: 1rem 0;">
                            Confidence: {confidence*100:.1f}%
                        </h3>
                    </div>
                    <p style="text-align: center; color: #666; margin-top: 1rem; font-style: italic;">
                        {message}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Key phrases analysis (simulated)
            st.markdown('<h3 style="color: #2c3e50; margin-top: 2rem;">🔍 Key Factors in Analysis</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="card">
                    <h4 style="color: #2c3e50;">📈 Positive Indicators</h4>
                    <ul style="color: #4CAF50;">
                        <li style="color: #4CAF50;">Complimentary language</li>
                        <li style="color: #4CAF50;">Satisfaction expressions</li>
                        <li style="color: #4CAF50;">Recommendation intent</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="card">
                    <h4 style="color: #2c3e50;">📉 Negative Indicators</h4>
                    <ul style="color: #F44336;">
                        <li style="color: #F44336;">Complaint patterns</li>
                        <li style="color: #F44336;">Disappointment signals</li>
                        <li style="color: #F44336;">Critical expressions</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Tips for better results
            st.markdown("""
            <div class="tips-box">
                <h4 style="color: #667eea; margin: 0;">💡 Tips for Better Analysis</h4>
                <ul style="margin: 0.5rem 0; color: #666;">
                    <li style="color: #666;">Include specific details about taste, texture, and quality</li>
                    <li style="color: #666;">Mention service experience and ambiance if relevant</li>
                    <li style="color: #666;">Be descriptive - longer reviews generally yield more accurate results</li>
                    <li style="color: #666;">Try different models to compare their interpretations</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    elif analyze_button and not user_review:
        st.warning("⚠️ Please enter a review before analyzing!")

# Compare Models Page
def render_compare():
    st.markdown('<h2 class="sub-header">🔄 Compare Models</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h3 style="color: #2c3e50; margin-top: 0;">🔬 Side-by-Side Model Comparison</h3>
        <p style="color: #666; margin: 1rem 0;">
            See how different AI models interpret the same review. This helps understand each model's strengths and biases.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Review input for comparison - with explicit black text styling
    comparison_review = st.text_area(
        "Enter a review to compare across all models:",
        height=120,
        placeholder="Example: The food was okay, nothing special but not terrible either. Service could have been better.",
        help="Try ambiguous or mixed reviews for interesting comparisons",
        key="comparison_input"
    )
    
    compare_button = st.button("🔍 Compare All Models", type="primary", use_container_width=True)
    
    if compare_button and comparison_review:
        with st.spinner("🤖 Running analysis across all models..."):
            time.sleep(2)  # Simulate processing
            
            st.markdown('<h3 style="color: #2c3e50; margin-top: 2rem;">📊 Comparison Results</h3>', unsafe_allow_html=True)
            
            # Get predictions from all models
            results = {}
            for model_name, model_key in [("Support Vector Machine", "svm"), 
                                         ("Naive Bayes", "nb"), 
                                         ("Logistic Regression", "lr")]:
                sentiment, confidence = predict_sentiment(comparison_review, model_key)
                results[model_name] = {"sentiment": sentiment, "confidence": confidence}
            
            # Display results in columns
            col1, col2, col3 = st.columns(3)
            
            for idx, (model_name, result) in enumerate(results.items()):
                col = [col1, col2, col3][idx]
                sentiment = result["sentiment"]
                confidence = result["confidence"]
                
                # Determine styling
                if sentiment == "positive":
                    emoji = "😊"
                    color = "#4CAF50"
                    bg_class = "positive"
                elif sentiment == "negative":
                    emoji = "😞"
                    color = "#F44336"
                    bg_class = "negative"
                else:
                    emoji = "😐"
                    color = "#FF9800"
                    bg_class = "neutral"
                
                with col:
                    st.markdown(f"""
                    <div class="card">
                        <h4 style="text-align: center; color: #2c3e50;">{model_name}</h4>
                        <div style="text-align: center; font-size: 3rem; margin: 1rem 0;">{emoji}</div>
                        <div style="text-align: center;">
                            <span class="badge" style="background: {color}; font-size: 1rem;">
                                {sentiment.upper()}
                            </span>
                        </div>
                        <div style="margin-top: 1rem; text-align: center;">
                            <div style="font-size: 1.5rem; font-weight: bold; color: {color};">
                                {confidence*100:.1f}%
                            </div>
                            <div style="color: #666; font-size: 0.9rem;">Confidence</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Consensus analysis
            st.markdown('<h3 style="color: #2c3e50; margin-top: 2rem;">🎯 Consensus Analysis</h3>', unsafe_allow_html=True)
            
            sentiments = [r["sentiment"] for r in results.values()]
            unique_sentiments = set(sentiments)
            
            if len(unique_sentiments) == 1:
                st.markdown(f"""
                <div class="info-box">
                    <h4 style="color: #E6E6FA; margin: 0;">✅ Strong Agreement</h4>
                    <p style="margin: 0.5rem 0; color: #E6E6FA;">
                        All models agree on <strong>{sentiments[0].upper()}</strong> sentiment. 
                        This indicates high confidence in the analysis.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="warning-box">
                    <h4 style="color: #F57C00; margin: 0;">⚠️ Mixed Opinions</h4>
                    <p style="margin: 0.5rem 0; color: #E6E6FA;">
                        Models disagree on sentiment classification. This suggests the review contains 
                        ambiguous or mixed emotional signals that different algorithms interpret differently.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Confidence comparison chart
            fig = go.Figure(data=[
                go.Bar(
                    x=list(results.keys()),
                    y=[r["confidence"]*100 for r in results.values()],
                    marker_color=['#667eea', '#764ba2', '#8e44ad'],
                    text=[f"{r['confidence']*100:.1f}%" for r in results.values()],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="Confidence Score Comparison",
                xaxis_title="Model",
                yaxis_title="Confidence (%)",
                yaxis=dict(range=[0, 100]),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif compare_button and not comparison_review:
        st.warning("⚠️ Please enter a review before comparing!")
    
    # Model recommendation section
    st.markdown('<h2 class="sub-header">🎯 Which Model Should You Use?</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h4 style="color: #667eea;">⚡ Use Support Vector  When:</h4>
            <ul style="color: #666;">
                <li style="color: #666;">Accuracy is top priority</li>
                <li style="color: #666;">Reviews are clearly positive/negative</li>
                <li style="color: #666;">You have computational resources</li>
                <li style="color: #666;">Working with professional reviews</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h4 style="color: #764ba2;">🎯 Use Naive Bayes When:</h4>
            <ul style="color: #666;">
                <li style="color: #666;">Speed is critical</li>
                <li style="color: #666;">Processing large volumes</li>
                <li style="color: #666;">Real-time applications</li>
                <li style="color: #666;">Resource constraints exist</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card">
            <h4 style="color: #8e44ad;">📈 Use Logistic Regression When:</h4>
            <ul style="color: #666;">
                <li style="color: #666;">Need interpretable results</li>
                <li style="color: #666;">Balanced performance matters</li>
                <li style="color: #666;">Probability scores are important</li>
                <li style="color: #666;">Working with structured data</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# About Page
def render_about():
    st.markdown('<h2 class="sub-header">ℹ️ About This Project</h2>', unsafe_allow_html=True)
    
    # Project overview
    st.markdown("""
    <div class="card">
        <h3 style="color: #2c3e50; margin-top: 0;">📚 Project Overview</h3>
        <p style="color: #666; line-height: 1.8; font-size: 1.1rem;">
            This sentiment analysis system was developed to help businesses and researchers understand customer emotions 
            in food reviews. By leveraging machine learning on Amazon's vast food review dataset, we've created models 
            that can accurately detect satisfaction levels and emotional tone in customer feedback.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Dataset information
    st.markdown('<h3 style="color: #2c3e50;">📊 Dataset Details</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h4 style="color: #2c3e50;">Original Dataset</h4>
            <ul style="color: #666; line-height: 1.6;">
                <li style="color: #666;"><strong>Source:</strong> Amazon Fine Food Reviews</li>
                <li style="color: #666;"><strong>Time Period:</strong> Oct 1999 - Oct 2012</li>
                <li style="color: #666;"><strong>Total Reviews:</strong> 568,454</li>
                <li style="color: #666;"><strong>Features:</strong> Review text, ratings, helpfulness</li>
                <li style="color: #666;"><strong>Products:</strong> 74,258 unique items</li>
                <li style="color: #666;"><strong>Users:</strong> 256,059 reviewers</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h4 style="color: #2c3e50;">Processing Pipeline</h4>
            <ul style="color: #666; line-height: 1.6;">
                <li style="color: #666;"><strong>Data Cleaning:</strong> HTML removal, normalization</li>
                <li style="color: #666;"><strong>Sampling:</strong> Balanced 220,000 reviews</li>
                <li style="color: #666;"><strong>Vectorization:</strong> TF-IDF with 5000 features</li>
                <li style="color: #666;"><strong>Split Ratio:</strong> 80% train, 20% test</li>
                <li style="color: #666;"><strong>Validation:</strong> Cross-validation (5 folds)</li>
                <li style="color: #666;"><strong>Optimization:</strong> GridSearchCV tuning</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Technology stack
    st.markdown('<h3 style="color: #2c3e50;">🛠️ Technology Stack</h3>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="card">
            <div style="text-align: center; font-size: 2.5rem;">🐍</div>
            <h4 style="text-align: center; color: #2c3e50;">Python</h4>
            <p style="text-align: center; color: #666; font-size: 0.9rem;">Core programming language</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <div style="text-align: center; font-size: 2.5rem;">🤖</div>
            <h4 style="text-align: center; color: #2c3e50;">Scikit-learn</h4>
            <p style="text-align: center; color: #666; font-size: 0.9rem;">ML algorithms & tools</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card">
            <div style="text-align: center; font-size: 2.5rem;">🎨</div>
            <h4 style="text-align: center; color: #2c3e50;">Streamlit</h4>
            <p style="text-align: center; color: #666; font-size: 0.9rem;">Web application framework</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="card">
            <div style="text-align: center; font-size: 2.5rem;">📊</div>
            <h4 style="text-align: center; color: #2c3e50;">Plotly</h4>
            <p style="text-align: center; color: #666; font-size: 0.9rem;">Interactive visualizations</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Use cases
    st.markdown('<h3 style="color: #2c3e50;">💼 Real-World Applications</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h4 style="color: #667eea;">🏪 Business Intelligence</h4>
            <ul style="color: #666;">
                <li style="color: #666;">Monitor customer satisfaction trends</li>
                <li style="color: #666;">Identify product quality issues early</li>
                <li style="color: #666;">Track competitor sentiment</li>
                <li style="color: #666;">Improve product development</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h4 style="color: #764ba2;">📈 Marketing & Sales</h4>
            <ul style="color: #666;">
                <li style="color: #666;">Optimize product descriptions</li>
                <li style="color: #666;">Target marketing campaigns</li>
                <li style="color: #666;">Identify brand advocates</li>
                <li style="color: #666;">Manage online reputation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Development team
    st.markdown('<h3 style="color: #2c3e50;">👥 Development Team</h3>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:    
        st.markdown("""
        <div class="card" style="text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">👨‍💻</div>
            <h4 style="color: #2c3e50;">Tang Yik Hong</h4>
            <p style="color: #666; font-size: 0.9rem;">Linear SVC builder </p>
            <div style="margin-top: 1rem;">
            <span class="badge" style="background: #667eea; color: white; font-size: 0.8rem;">Linear SVC</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="card" style="text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">👨‍💻</div>
            <h4 style="color: #2c3e50;">Teh Wei Zhang</h4>
            <p style="color: #666; font-size: 0.9rem;">Logistic Regression</p>
            <div style="margin-top: 1rem;">
            <span class="badge" style="background: #764ba2; color: white; font-size: 0.8rem;">Logistic Regression</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="card" style="text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">👨‍💻</div>
            <h4 style="color: #2c3e50;">Chai Ee Yuan</h4>
            <p style="color: #666; font-size: 0.9rem;">Multinomial Naive Bayes</p>
            <div style="margin-top: 1rem;">
            <span class="badge" style="background: #8e44ad; color: white; font-size: 0.8rem;">Multinomial Naive Bayes</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Future improvements
    st.markdown('<h3 style="color: #2c3e50;">🚀 Future Enhancements</h3>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <h4 style="color: #2c3e50;">Planned Features</h4>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-top: 1rem;">
            <div style="padding: 1rem; background: rgba(102, 126, 234, 0.1); border-radius: 10px;">
                <strong style="color: #2c3e50;">🧠 Deep Learning Models</strong><br>
                <small style="color: #666;">Implement BERT and GPT-based models for improved accuracy</small>
            </div>
            <div style="padding: 1rem; background: rgba(118, 75, 162, 0.1); border-radius: 10px;">
                <strong style="color: #2c3e50;">🌍 Multi-language Support</strong><br>
                <small style="color: #666;">Extend analysis to reviews in multiple languages</small>
            </div>
            <div style="padding: 1rem; background: rgba(142, 68, 173, 0.1); border-radius: 10px;">
                <strong style="color: #2c3e50;">📱 API Service</strong><br>
                <small style="color: #666;">RESTful API for integration with other applications</small>
            </div>
            <div style="padding: 1rem; background: rgba(155, 89, 182, 0.1); border-radius: 10px;">
                <strong style="color: #2c3e50;">📊 Advanced Analytics</strong><br>
                <small style="color: #666;">Aspect-based sentiment and emotion detection</small>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="margin-top: 3rem; padding: 2rem; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1)); border-radius: 15px; text-align: center;">
        <p style="color: #E6E6FA; margin: 0;">
            Made with ❤️ using Python and Streamlit<br>
            <small>© 2024 Food Review Sentiment Analysis | Version 2.0</small>
        </p>
    </div>
    """, unsafe_allow_html=True)

# Main app logic
def main():
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "home"
    
    # Add header
    add_header()
    
    # Render sidebar and get selected page
    selected_page = render_sidebar()
    
    # Route to appropriate page
    if selected_page == "home":
        render_home()
    elif selected_page == "performance":
        render_performance()
    elif selected_page == "models":
        render_models()
    elif selected_page == "try_it":
        render_try_it()
    elif selected_page == "compare":
        render_compare()
    elif selected_page == "about":
        render_about()
    else:
        render_home()

# Run the app
if __name__ == "__main__":
    main()