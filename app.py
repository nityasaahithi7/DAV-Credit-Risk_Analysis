import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import load_data, preprocess_data
from models import train_logistic_regression, train_random_forest

# --------------------------------------------------
# COLOR PALETTE DEFINITION (CHIC SAGE & PINK BEIGE)
# --------------------------------------------------
COLOR_SAGE_DARK  = "#586F57"
COLOR_SAGE_MAIN  = "#849782"
COLOR_SAGE_LIGHT = "#A3B18A"
COLOR_BEIGE_DARK = "#E8D8D0"
COLOR_BEIGE_MAIN = "#F7EBE8"
COLOR_TEXT_MAIN  = "#2B2D42"
COLOR_TEXT_MUTED = "#6C757D"

palette_sage_beige = sns.light_palette(COLOR_SAGE_MAIN, as_cmap=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.facecolor"] = "#FFFFFF"
plt.rcParams["axes.facecolor"] = "#FFFFFF"
plt.rcParams["text.color"] = COLOR_TEXT_MAIN
plt.rcParams["axes.labelcolor"] = COLOR_TEXT_MAIN
plt.rcParams["xtick.color"] = COLOR_TEXT_MAIN
plt.rcParams["ytick.color"] = COLOR_TEXT_MAIN

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Analysis System",
    page_icon="🌿",
    layout="wide"
)

# Custom CSS for Minimal Chic Cards
st.markdown(f"""
    <style>
    /* Background & Main Layout */
    .stApp {{
        background-color: #F9F6F0;
    }}
    .main-title {{
        text-align: center;
        font-size: 2.8rem;
        font-weight: 700;
        color: {COLOR_TEXT_MAIN};
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }}
    .sub-title {{
        text-align: center;
        font-size: 1.05rem;
        color: {COLOR_TEXT_MUTED};
        margin-bottom: 2rem;
    }}
    .section-label {{
        text-align: center;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        color: {COLOR_SAGE_DARK};
        text-transform: uppercase;
        margin-bottom: 2rem;
    }}
    
    /* Streamlit Buttons Styling Override */
    div.stButton > button {{
        background-color: {COLOR_SAGE_MAIN} !important;
        color: #FFFFFF !important;
        border-radius: 10px !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 0.7rem 1rem !important;
        transition: all 0.3s ease !important;
    }}
    div.stButton > button:hover {{
        background-color: {COLOR_SAGE_DARK} !important;
        box-shadow: 0 4px 12px rgba(88, 111, 87, 0.25) !important;
    }}
    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER SECTION
# --------------------------------------------------
st.markdown('<div class="main-title">Credit Risk Analysis System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Data-Driven Insights into Loan Defaults & Applicant Risk Modeling</div>', unsafe_allow_html=True)
st.markdown('<div class="section-label">SELECT ALGORITHM TO EXPLORE THE ANALYSIS</div>', unsafe_allow_html=True)

# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------
DATA_PATH = "data/credit_risk_dataset.csv"

@st.cache_data
def get_data():
    df_orig = load_data(DATA_PATH)
    return df_orig, preprocess_data(df_orig)

# --------------------------------------------------
# CLEAN ALGORITHM CARDS
# --------------------------------------------------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("<h2 style='text-align: center; margin-bottom: 0.5rem;'>📈</h2>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center; color: {COLOR_TEXT_MAIN}; font-weight: 600; margin-bottom: 0.5rem;'>Logistic Regression</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: {COLOR_TEXT_MUTED}; height: 45px; margin-bottom: 1.5rem;'>Classifies default probability using scaled linear boundary optimization.</p>", unsafe_allow_html=True)
    btn_lr = st.button("View Logistic Regression Results →", key="btn_lr", width="stretch")

with col2:
    st.markdown("<h2 style='text-align: center; margin-bottom: 0.5rem;'>🌿</h2>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center; color: {COLOR_TEXT_MAIN}; font-weight: 600; margin-bottom: 0.5rem;'>Random Forest</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: {COLOR_TEXT_MUTED}; height: 45px; margin-bottom: 1.5rem;'>Evaluates risk through an ensemble of decision trees for non-linear patterns.</p>", unsafe_allow_html=True)
    btn_rf = st.button("View Random Forest Results →", key="btn_rf", width="stretch")

st.divider()

# --------------------------------------------------
# RESULTS DISPLAY AREA
# --------------------------------------------------
if btn_lr or btn_rf:
    df_original, (
        X_train, X_test, X_train_scaled, X_test_scaled,
        y_train, y_test, scaler, df_processed
    ) = get_data()

    if btn_lr:
        st.header("📈 Logistic Regression Analysis Results")
        with st.spinner("Training Logistic Regression Model..."):
            model, predictions, metrics, cm = train_logistic_regression(
                X_train_scaled, X_test_scaled, y_train, y_test
            )

        # Metrics Overview
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics['Accuracy']:.2%}")
        m2.metric("Precision", f"{metrics['Precision']:.2%}")
        m3.metric("Recall", f"{metrics['Recall']:.2%}")
        m4.metric("F1 Score", f"{metrics['F1 Score']:.2%}")

        st.write("")
        
        # Visualizations
        c_chart, c_preview = st.columns([1, 1], gap="medium")
        
        with c_chart:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap=palette_sage_beige, ax=ax, cbar=False)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            
        with c_preview:
            st.subheader("Preprocessed Sample Preview")
            st.dataframe(df_processed.head(8), width="stretch")

    elif btn_rf:
        st.header("🌿 Random Forest Analysis Results")
        with st.spinner("Training Random Forest Classifier..."):
            model, predictions, metrics, cm = train_random_forest(
                X_train, X_test, y_train, y_test
            )

        # Metrics Overview
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics['Accuracy']:.2%}")
        m2.metric("Precision", f"{metrics['Precision']:.2%}")
        m3.metric("Recall", f"{metrics['Recall']:.2%}")
        m4.metric("F1 Score", f"{metrics['F1 Score']:.2%}")

        st.write("")
        
        # Visualizations
        c_chart, c_preview = st.columns([1, 1], gap="medium")
        
        with c_chart:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap=palette_sage_beige, ax=ax, cbar=False)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            
        with c_preview:
            st.subheader("Preprocessed Sample Preview")
            st.dataframe(df_processed.head(8), width="stretch")

else:
    st.info("💡 Click one of the cards above to run the algorithm and view model performance.")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.caption("Credit Risk Analysis | DAV Mini Project")