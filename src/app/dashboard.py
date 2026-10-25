"""
Interactive VC Decision-Support System
======================================
Streamlit Dashboard for Early-Stage (Seed) Startup Investment Screening,
TreeSHAP Explainability, and Corea 21-Factor Venture Checklist Scorecard.
"""

import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.explainability.shap_explainer import VentureExplainer

st.set_page_config(
    page_title="VenturePulse - Seed Decision Support",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1E222D;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #2962FF;
    }
    .stProgress > div > div > div > div {
        background-color: #00E676;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_explainer():
    return VentureExplainer()

try:
    explainer = load_explainer()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# Header
st.title("🚀 VenturePulse: Seed-Stage Investment Decision-Support System")
st.markdown("""
*An explainable, time-aware machine learning framework for evaluating early-stage venture capital deals,
addressing class imbalance and look-ahead bias as proposed in our research survey.*
""")

st.divider()

# Sidebar: Deal Input Parameters
st.sidebar.header("📋 Candidate Startup Profile")
startup_name = st.sidebar.text_input("Startup Name", "OmniAI Technologies")

seed_amount = st.sidebar.number_input("Seed Round Capital (USD $)", min_value=10_000, max_value=20_000_000, value=2_000_000, step=100_000)
angel_amount = st.sidebar.number_input("Pre-Seed / Angel Capital (USD $)", min_value=0, max_value=5_000_000, value=250_000, step=50_000)
months_to_seed = st.sidebar.slider("Months from Founding to Seed Close", min_value=1, max_value=60, value=9)

market = st.sidebar.selectbox(
    "Market Vertical",
    ['Enterprise Software', 'Software', 'Biotechnology', 'Mobile', 'Analytics', 'Finance', 'Health Care', 'Security', 'E-Commerce', 'Other']
)

is_hub = st.sidebar.radio("Headquarters Location", ["Top Tier Venture Hub (SF, NY, Boston, London)", "Secondary / Emerging Market"], index=0)
is_top_hub = 1 if "Top Tier" in is_hub else 0

country = st.sidebar.selectbox("Country Tier", ['USA', 'GBR', 'CAN', 'IND', 'DEU', 'FRA', 'ISR', 'OTHER'])

deal_input = {
    'seed_usd': float(seed_amount),
    'angel_usd': float(angel_amount),
    'months_to_first_funding': float(months_to_seed),
    'is_top_hub': is_top_hub,
    'country_tier': country,
    'market_group': market
}

# Run Evaluation
res = explainer.evaluate_deal(deal_input)
prob = res['probability_series_a']
tier = res['conviction_tier']
scorecard = res['corea_scorecard']

# Main Display
col1, col2, col3 = st.columns([1.2, 1, 1.2])

with col1:
    st.subheader("🎯 Investment Recommendation")
    if "Tier 1" in tier:
        st.success(f"### {tier}")
    elif "Tier 2" in tier:
        st.warning(f"### {tier}")
    else:
        st.error(f"### {tier}")

    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Series A Graduation Probability (%)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#2962FF"},
            'steps': [
                {'range': [0, 20], 'color': "#37474F"},
                {'range': [20, 40], 'color': "#455A64"},
                {'range': [40, 100], 'color': "#1B5E20"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 11.29  # Historical test baseline
            }
        }
    ))
    fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)
    st.caption("🔴 Red line indicates historical market average exit/progression rate (11.3%).")

with col2:
    st.subheader("📊 21-Factor Venture Score")
    v_score = scorecard['venture_score_100']
    st.metric(label="Corea Venture Heuristic Score", value=f"{v_score} / 100", delta=f"{v_score - 50:+d} vs Benchmark")
    
    st.markdown("**Checklist Pillar Results:**")
    for item in scorecard['checklist_items']:
        icon = "✅" if item['Pass'] else "⚠️"
        st.markdown(f"{icon} **{item['Pillar']}**: {item['Check']}")

with col3:
    st.subheader("⚡ Signal Overview")
    st.markdown(f"- **Startup**: `{startup_name}`")
    st.markdown(f"- **Total Early Capital**: `${(seed_amount + angel_amount):,}`")
    st.markdown(f"- **Traction Velocity**: `{months_to_seed} months` to seed check")
    st.markdown(f"- **Geography**: `{country}` ({'Hub' if is_top_hub else 'Non-Hub'})")
    st.markdown(f"- **Sector**: `{market}`")

st.divider()

# Explainability Section
st.subheader("🔍 TreeSHAP Feature Attributions (Why did the model assign this score?)")
st.markdown("Detailed breakdown showing which features increased confidence (+) or introduced risk penalties (-).")

shap_df = res['shap_dataframe'].head(10).copy()
shap_df['Color'] = shap_df['Attribution'].apply(lambda x: '#00E676' if x >= 0 else '#FF5252')

fig_shap = go.Figure(go.Bar(
    x=shap_df['Attribution'],
    y=shap_df['Feature'],
    orientation='h',
    marker_color=shap_df['Color']
))
fig_shap.update_layout(
    title="Top 10 Feature Attribution Contributions to Predicted Probability",
    xaxis_title="SHAP Value (Contribution to Log-Odds)",
    yaxis_title="Feature",
    yaxis=dict(autorange="reversed"),
    height=400,
    margin=dict(l=20, r=20, t=40, b=20)
)
st.plotly_chart(fig_shap, use_container_width=True)

# Footnote
st.caption("Model: Random Forest & LightGBM trained with scale_pos_weight under strict 2012+ chronological out-of-time holdout.")
