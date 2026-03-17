import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.express as px
import pandas as pd

MODEL_NAME = "unitary/toxic-bert"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

labels = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]


def predict(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
    results = dict(zip(labels, probs))

    severity = max(probs)

    if results["threat"] > 0.5:
        violation = "Threat"
    elif results["identity_hate"] > 0.5:
        violation = "Identity Hate"
    elif results["insult"] > 0.4:
        violation = "Insult"
    elif results["obscene"] > 0.4:
        violation = "Obscene"
    elif results["toxic"] > 0.3:
        violation = "Toxic Language"
    else:
        violation = "None"

# ---- Status Logic ---- #

    if severity > 0.8:
        status = "Severe Policy Violation"

    elif severity > 0.5:
        status = "Policy Violation"

    # detect indirect threats
    elif results["threat"] > 0.05:
        status = "Potential Threat"

    elif results["toxic"] > 0.2:
        status = "Toxic Content"

    else:
        status = "Safe Content"

    return results, severity, violation, status


# ---------------- UI ---------------- #

st.set_page_config(page_title="PolicyGuard NLP", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better styling
st.markdown("""
<style>
    .big-font {
        font-size:35px !important;
        font-weight: bold;
        color: #1E3A8A;
    }
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: None;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #2563eb;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Reduce metric value text size */
    [data-testid="stMetricValue"] {
        font-size: 26px !important;
    }

    /* Reduce metric label size */
    [data-testid="stMetricLabel"] {
        font-size: 14px !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for history and current analysis
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None

# Sidebar
with st.sidebar:

    st.markdown("## 🛡️ PolicyGuard NLP")

    st.markdown("""
    **Automatic Detection of Policy Violations in Social Media Content**
    """)

    st.markdown("""
    <div style="
        background:#1f2937;
        padding:15px;
        border-radius:10px;
        border:1px solid #374151;
        line-height:1.6;
    ">
    This system uses a <b>BERT-based NLP model</b> to detect harmful content such as toxic language,
    threats, insults, obscene text, and identity-based hate in social media posts.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 📊 Content Policy Categories")

    def category_card(title, description):
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(description)

    c1, c2 = st.columns(2)

    with c1:
        category_card("Toxic", "General rude or offensive language.")
        category_card("Obscene", "Vulgar or sexually explicit language.")
        category_card("Insult", "Language meant to humiliate or demean someone.")

    with c2:
        category_card("Severe Toxic", "Extremely abusive or aggressive language.")
        category_card("Threat", "Statements expressing intent to harm someone.")
        category_card("Identity Hate", "Hate speech targeting groups based on identity.")
    st.markdown("---") 

# Main Header
st.markdown('<p class="big-font">🛡️ PolicyGuard Content Moderation</p>', unsafe_allow_html=True)
st.markdown("Analyze social media content for toxicity, threats, and policy violations using advanced NLP.")
st.divider()

# Create Tabs for different sections
tab1, tab2 = st.tabs(["🔍 Analyze Content", "📚 History"])

with tab1:
    col1, col2 = st.columns([1, 1.2], gap="large")

# -------- INPUT -------- #

with col1:
    st.subheader("✏️ Input Content")

    text = st.text_area(
        "Enter social media content to analyze",
        height=220,
        placeholder="Type a comment, tweet, or message..."
    )

    analyze = st.button("Analyze Content", use_container_width=True)

    if analyze and text.strip():
        with st.spinner("Analyzing content using AI..."):
            results, severity, violation, status = predict(text)

            st.session_state.current_analysis = {
                "text": text,
                "results": results,
                "severity": severity,
                "violation": violation,
                "status": status
            }

            st.session_state.history.insert(0, st.session_state.current_analysis)
            st.session_state.history = st.session_state.history[:10]

    # Show currently analyzed text
    if st.session_state.current_analysis:
        with st.container(border=True):
            st.markdown("### Currently Analyzing:")
            st.info(st.session_state.current_analysis["text"])
    else:
        with st.container(border=True):
            st.info("👈 Type a message above and click **Analyze Content** to analyze.")
    

    
    # -------- RESULTS -------- #
    with col2:
        if st.session_state.current_analysis:
            data = st.session_state.current_analysis
            text = data["text"]
            results = data["results"]
            severity = data["severity"]
            violation = data["violation"]
            status = data["status"]

            st.subheader("📋 Analysis Results")
            
            # Status Banner
            if status == "Severe Policy Violation":
                st.error("**Severe Policy Violation** — Immediate moderation required!", icon="🚨")

            elif status == "Policy Violation":
                st.warning("**Policy Violation Detected** — Please review this content.", icon="⚠️")

            elif status == "Potential Threat":
                st.warning("**Potential Threat Detected** — Monitor this content carefully.", icon="⚠️")
            elif status == "Toxic Content":
                st.warning("⚠️ **Toxic Language Detected** — Content may violate community guidelines.")
            else:
                st.success("**Safe Content** — No significant policy violations detected.", icon="✅")

            # Metrics
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Severity Score", f"{severity:.2f}", 
                          delta="High Risk" if severity > 0.5 else "Low Risk", 
                          delta_color="inverse")
            with c2:
                st.metric("Primary Violation", violation)
            with c3:
                st.metric("Toxic Language", "Detected" if results["toxic"] > 0.3 else "Not Detected")

            st.markdown("---")

            # Category Breakdown Chart
            st.subheader("📊 Detailed Category Breakdown")

            df = pd.DataFrame({
                "Category": ["Severe Toxic", "Obscene", "Threat", "Insult", "Identity Hate"],
                "Probability": [
                    results["severe_toxic"],
                    results["obscene"],
                    results["threat"],
                    results["insult"],
                    results["identity_hate"],
                ]
            }).sort_values(by="Probability", ascending=True)

            fig = px.bar(
                df,
                x="Probability",
                y="Category",
                orientation="h",
                color="Probability",
                color_continuous_scale="Reds",
                text="Probability"
            )

            fig.update_layout(
                xaxis_range=[0,1],
                height=400,
                margin=dict(l=20,r=20,t=20,b=20),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            
            st.plotly_chart(fig, use_container_width=True)
            
        elif not st.session_state.current_analysis:
            st.info("👈 Enter some text on the left and press **Enter** to see results here.")

with tab2:
    st.subheader("📚 Analysis History")
    if not st.session_state.history:
        st.info("No history yet. Analyze some content to see it here.")
    else:
        for i, item in enumerate(st.session_state.history):
            with st.container(border=True):
                st.markdown(f"**Text:** `{item['text']}`")
                st.markdown(f"**Status:** {item['status']} | **Severity:** {item['severity']:.2f} | **Primary Violation:** {item['violation']}")

