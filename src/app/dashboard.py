"""
VenturePulse: Seed-Stage Investment Decision-Support Platform
============================================================
A production-grade, multi-tab decision-support interface for evaluating
early-stage venture capital deals, inspecting TreeSHAP explainability,
exploring historical Crunchbase cohorts, and reviewing empirical benchmarks.

Run:
    .venv/bin/streamlit run src/app/dashboard.py
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.explainability.shap_explainer import VentureExplainer

# Page configuration
st.set_page_config(
    page_title="VenturePulse | Seed-Stage Decision Support",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for institutional typography & metric cards
st.markdown("""
<style>
    /* Metric Cards */
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 20px;
        color: #F8FAFC;
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.82rem;
        color: #94A3B8;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #FFFFFF;
    }
    .metric-sub {
        font-size: 0.8rem;
        margin-top: 4px;
    }
    /* Status Badges */
    .badge-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-blue { background-color: #1E3A8A; color: #93C5FD; border: 1px solid #2563EB; }
    .badge-green { background-color: #064E3B; color: #6EE7B7; border: 1px solid #059669; }
    .badge-purple { background-color: #4C1D95; color: #C4B5FD; border: 1px solid #7C3AED; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_explainer():
    return VentureExplainer()


@st.cache_data
def load_historical_data():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(root, "data", "processed", "seed_features.parquet")
    if os.path.exists(data_path):
        return pd.read_parquet(data_path)
    return None


@st.cache_data
def load_benchmark_tables():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    art_dir = os.path.join(root, "artifacts")
    tables = {}
    for name in ["test_results.csv", "ablation_results.csv", "leakage_check.csv"]:
        p = os.path.join(art_dir, name)
        if os.path.exists(p):
            tables[name] = pd.read_csv(p)
    return tables


try:
    explainer = load_explainer()
    bundle = explainer.bundle
    metrics = bundle["test_metrics"]
    ci = bundle["test_ci"]
except Exception as e:
    st.error(f"Error loading model artifacts: {e}\n\nPlease run: `.venv/bin/python -m src.models.train_models`")
    st.stop()

hist_df = load_historical_data()
bench_tables = load_benchmark_tables()

# ==============================================================================
# SIDEBAR: Deal Entry & Quick Presets
# ==============================================================================
st.sidebar.title("🚀 VenturePulse")
st.sidebar.markdown(
    '<span class="badge-pill badge-blue">Audit Verified: No Leakage</span> '
    '<span class="badge-pill badge-green">15.6k Startups</span>',
    unsafe_allow_html=True
)
st.sidebar.markdown("---")

st.sidebar.subheader("⚡ Quick Deal Presets")
preset_options = [
    "Custom Evaluation",
    "🚀 AI Infrastructure ($2.5M Seed, 7 mos, SF Bay Area)",
    "🧬 BioTech Therapeutics ($3.5M Seed, 14 mos, Boston)",
    "🛒 Emerging E-Commerce ($600k Seed, 22 mos, Non-Hub)",
    "🛠️ Developer Tools ($1.2M Seed, 5 mos, NYC)",
]
selected_preset = st.sidebar.selectbox("Load Preconfigured Startup", preset_options)

# Presets mapping
if selected_preset == "🚀 AI Infrastructure ($2.5M Seed, 7 mos, SF Bay Area)":
    init_name = "Synthetix AI"
    init_seed = 2_500_000
    init_angel = 250_000
    init_months = 7
    init_sector = "Enterprise & B2B SaaS"
    init_ai = True
    init_b2b = True
    init_bio = False
    init_fin = False
    init_mob = False
    init_ecom = False
    init_hub = True
    init_country = "USA"
    init_web = True
elif selected_preset == "🧬 BioTech Therapeutics ($3.5M Seed, 14 mos, Boston)":
    init_name = "OncoGenix Labs"
    init_seed = 3_500_000
    init_angel = 0
    init_months = 14
    init_sector = "Healthcare & BioTech"
    init_ai = True
    init_b2b = False
    init_bio = True
    init_fin = False
    init_mob = False
    init_ecom = False
    init_hub = True
    init_country = "USA"
    init_web = True
elif selected_preset == "🛒 Emerging E-Commerce ($600k Seed, 22 mos, Non-Hub)":
    init_name = "DirectCart Logistics"
    init_seed = 600_000
    init_angel = 100_000
    init_months = 22
    init_sector = "E-Commerce & Retail"
    init_ai = False
    init_b2b = False
    init_bio = False
    init_fin = False
    init_mob = True
    init_ecom = True
    init_hub = False
    init_country = "CAN"
    init_web = True
elif selected_preset == "🛠️ Developer Tools ($1.2M Seed, 5 mos, NYC)":
    init_name = "DevPulse Cloud"
    init_seed = 1_200_000
    init_angel = 150_000
    init_months = 5
    init_sector = "Enterprise & B2B SaaS"
    init_ai = True
    init_b2b = True
    init_bio = False
    init_fin = False
    init_mob = False
    init_ecom = False
    init_hub = True
    init_country = "USA"
    init_web = True
else:
    init_name = "OmniAI Technologies"
    init_seed = 2_000_000
    init_angel = 250_000
    init_months = 9
    init_sector = "Enterprise & B2B SaaS"
    init_ai = True
    init_b2b = True
    init_bio = False
    init_fin = False
    init_mob = False
    init_ecom = False
    init_hub = True
    init_country = "USA"
    init_web = True

st.sidebar.markdown("### 📋 Candidate Profile")
name = st.sidebar.text_input("Startup Name", init_name)
seed_amount = st.sidebar.number_input("Seed Round Capital ($ USD)", 0, 25_000_000, init_seed, 100_000)
angel_amount = st.sidebar.number_input("Pre-Seed / Angel Capital ($ USD)", 0, 5_000_000, init_angel, 50_000)

timing_known = st.sidebar.checkbox("Founding Date Documented", True)
months = st.sidebar.slider("Months from Founding to Seed Check", 0, 120, init_months, disabled=not timing_known)

has_homepage = st.sidebar.checkbox("Verified Domain / Active Website", init_web)

macro_sectors = [
    "Enterprise & B2B SaaS", "Consumer Web & Mobile", "Healthcare & BioTech",
    "Fintech & Financial", "E-Commerce & Retail", "General Software & Tech",
    "Hardware & CleanTech", "Media & Advertising", "EdTech", "Other / Emerging"
]
sector_index = macro_sectors.index(init_sector) if init_sector in macro_sectors else 0
sector = st.sidebar.selectbox("Macro Industry Sector", macro_sectors, index=sector_index)

st.sidebar.markdown("**Cross-Cutting Domain Signals:**")
c_t1, c_t2 = st.sidebar.columns(2)
with c_t1:
    tag_ai = st.checkbox("AI / Analytics", value=init_ai)
    tag_b2b = st.checkbox("B2B Enterprise", value=init_b2b)
    tag_bio = st.checkbox("Biotech/Health", value=init_bio)
with c_t2:
    tag_fin = st.checkbox("Fintech", value=init_fin)
    tag_mob = st.checkbox("Mobile / Apps", value=init_mob)
    tag_ecom = st.checkbox("E-Commerce", value=init_ecom)

countries = ["USA", "GBR", "CAN", "IND", "DEU", "FRA", "ISR", "ESP", "OTHER"]
country_index = countries.index(init_country) if init_country in countries else 0
country = st.sidebar.selectbox("Country Tier", countries, index=country_index)
hub = st.sidebar.radio(
    "Ecosystem Location",
    ["Top Venture Hub (SF, NY, Boston, London, BLR)", "Other / Emerging"],
    index=0 if init_hub else 1
).startswith("Top")
n_cat = st.sidebar.slider("Number of Category Taxonomy Tags", 1, 10, 3)

# Evaluate Deal
deal = {
    "seed_usd": float(seed_amount),
    "angel_usd": float(angel_amount),
    "months_to_first_funding": float(months) if timing_known else None,
    "is_top_hub": int(hub),
    "country_tier": country,
    "macro_sector": sector,
    "is_b2b_saas": int(tag_b2b),
    "is_ai_analytics": int(tag_ai),
    "is_biotech_health": int(tag_bio),
    "is_fintech": int(tag_fin),
    "is_consumer_mobile": int(tag_mob),
    "is_ecommerce": int(tag_ecom),
    "num_categories": n_cat,
    "has_homepage": int(has_homepage),
}

if seed_amount + angel_amount <= 0:
    st.warning("⚠️ Enter a seed or angel amount greater than zero to screen this deal.")
    st.stop()

res = explainer.evaluate_deal(deal)
prob = res["probability_series_a"]
base = res["base_rate"]
lift = res["lift_vs_base"]
tier = res["conviction_tier"]
cl = res["checklist"]

# ==============================================================================
# MAIN NAVIGATION TABS
# ==============================================================================
tabs = st.tabs([
    "🎯 Deal Evaluator & XAI",
    "📊 Historical Cohort Explorer",
    "⚖️ Scientific Rigor & Benchmarks",
    "📚 Research Survey Alignment"
])

# ------------------------------------------------------------------------------
# TAB 1: DEAL EVALUATOR & XAI
# ------------------------------------------------------------------------------
with tabs[0]:
    st.subheader(f"Investment Evaluation: `{name}`")
    st.caption("Point-in-time machine learning prediction combined with TreeSHAP feature attributions and investment heuristics.")

    # High-level Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Follow-on Probability</div>
            <div class="metric-val">{prob:.1%}</div>
            <div class="metric-sub" style="color: {'#4ADE80' if lift >= 1.0 else '#F87171'};">
                {lift:+.2f}x vs Historical Base ({base:.1%})
            </div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        tier_color = "#34D399" if "Tier 1" in tier else ("#FBBF24" if "Tier 2" in tier else "#F87171")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Conviction Tier</div>
            <div class="metric-val" style="color: {tier_color}; font-size: 1.25rem;">{tier}</div>
            <div class="metric-sub" style="color: #94A3B8;">Calibrated Decision Support</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Heuristic Screen</div>
            <div class="metric-val">{cl['score']} / {cl['max_score']}</div>
            <div class="metric-sub" style="color: #94A3B8;">Disciplined VC Rules Passed</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Early Capital</div>
            <div class="metric-val">${(seed_amount + angel_amount):,}</div>
            <div class="metric-sub" style="color: #94A3B8;">Seed: ${(seed_amount):,} | Angel: ${(angel_amount):,}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Middle Row: Gauge & Heuristic Screen
    g_col, c_col = st.columns([1.1, 1.2])

    with g_col:
        st.markdown("#### 🎯 Calibrated Success Likelihood")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 36, "color": "#F8FAFC"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94A3B8"},
                "bar": {"color": "#3B82F6", "thickness": 0.28},
                "bgcolor": "#1E293B",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 25], "color": "rgba(239, 68, 68, 0.25)"},
                    {"range": [25, 45], "color": "rgba(245, 158, 11, 0.25)"},
                    {"range": [45, 100], "color": "rgba(16, 185, 129, 0.25)"}
                ],
                "threshold": {
                    "line": {"color": "#EF4444", "width": 3},
                    "thickness": 0.8,
                    "value": base * 100
                }
            }
        ))
        fig_gauge.update_layout(
            height=270,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#F8FAFC"}
        )
        st.plotly_chart(fig_gauge, width='stretch')
        st.caption(f"🔴 Red line indicates the training cohort baseline graduation rate ({base:.1%}).")

    with c_col:
        st.markdown("#### ✅ Heuristic Investment Checklist")
        st.caption("Transparent, rule-based screening pillars maintained independently from the machine learning model.")
        for item in cl["items"]:
            icon = "✅" if item["Pass"] else "⚠️"
            bg = "rgba(16, 185, 129, 0.08)" if item["Pass"] else "rgba(245, 158, 11, 0.08)"
            border = "#059669" if item["Pass"] else "#D97706"
            st.markdown(
                f'<div style="background:{bg}; border-left: 4px solid {border}; padding: 8px 12px; border-radius: 4px; margin-bottom: 6px; font-size: 0.88rem;">'
                f'{icon} <strong>{item["Pillar"]}</strong>: {item["Check"]}'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown("---")

    # Bottom Row: TreeSHAP Attributions
    st.markdown("#### 🔍 TreeSHAP Feature Attributions (Why did the model assign this score?)")
    st.caption("Local feature-level explanations computed via Shapley values, showing exact positive drivers vs. risk penalties.")

    shap_df = res["shap_dataframe"].head(10).copy()
    shap_df["Color"] = shap_df["Attribution"].apply(lambda v: "#10B981" if v >= 0 else "#EF4444")
    shap_df["Direction"] = shap_df["Attribution"].apply(lambda v: "Positive Tailwind (+)" if v >= 0 else "Risk Penalty (-)")

    fig_shap = go.Figure(go.Bar(
        x=shap_df["Attribution"],
        y=shap_df["Feature"],
        orientation="h",
        marker_color=shap_df["Color"],
        text=[f"{v:+.3f}" for v in shap_df["Attribution"]],
        textposition="outside"
    ))
    fig_shap.update_layout(
        height=380,
        yaxis=dict(autorange="reversed", title="Feature"),
        xaxis=dict(title="Contribution to Model Log-Odds (SHAP Value)"),
        margin=dict(l=20, r=40, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#F8FAFC"}
    )
    st.plotly_chart(fig_shap, width='stretch')


# ------------------------------------------------------------------------------
# TAB 2: HISTORICAL COHORT EXPLORER
# ------------------------------------------------------------------------------
with tabs[1]:
    st.subheader("📊 Crunchbase Historical Cohort Explorer (N = 15,633)")
    st.caption("Explore verified empirical patterns across early-stage investments in the public Crunchbase snapshot.")

    if hist_df is not None:
        c1, c2 = st.columns([1, 1])

        with c1:
            st.markdown("##### Follow-on Rate by Cleaned Macro Sector")
            sector_stats = (
                hist_df[hist_df["is_mature"] == 1]
                .groupby("macro_sector")["target_followon"]
                .agg(["mean", "count"])
                .reset_index()
                .sort_values(by="mean", ascending=True)
            )
            sector_stats["mean_pct"] = (sector_stats["mean"] * 100).round(1)

            fig_sec = px.bar(
                sector_stats,
                x="mean_pct",
                y="macro_sector",
                orientation="h",
                text="mean_pct",
                labels={"mean_pct": "Follow-on Rate (%)", "macro_sector": "Sector"},
                color="mean_pct",
                color_continuous_scale="Viridis"
            )
            fig_sec.update_layout(
                height=380,
                margin=dict(l=10, r=20, t=10, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#F8FAFC"},
                showlegend=False
            )
            st.plotly_chart(fig_sec, width='stretch')

        with c2:
            st.markdown("##### Seed Capital Distribution by Outcome")
            hist_df["Outcome"] = hist_df["target_followon"].apply(lambda v: "Raised Follow-on" if v == 1 else "No Follow-on")
            fig_box = px.box(
                hist_df[hist_df["is_mature"] == 1],
                x="Outcome",
                y="early_funding_usd",
                color="Outcome",
                log_y=True,
                labels={"early_funding_usd": "Early Funding ($ USD, Log Scale)"},
                color_discrete_map={"Raised Follow-on": "#10B981", "No Follow-on": "#64748B"}
            )
            fig_box.update_layout(
                height=380,
                margin=dict(l=10, r=20, t=10, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#F8FAFC"},
                showlegend=False
            )
            st.plotly_chart(fig_box, width='stretch')

        st.markdown("##### Ecosystem Hub Advantage: Hub vs. Non-Hub")
        hub_summary = (
            hist_df[hist_df["is_mature"] == 1]
            .groupby(["macro_sector", "is_top_hub"])["target_followon"]
            .mean()
            .unstack()
            .rename(columns={0: "Non-Hub", 1: "Top Tier Hub"})
            .apply(lambda col: (col * 100).round(1))
            .reset_index()
        )
        st.dataframe(hub_summary, width='stretch')
    else:
        st.info("Processed dataset file not found at `data/processed/seed_features.parquet`.")


# ------------------------------------------------------------------------------
# TAB 3: SCIENTIFIC RIGOR & BENCHMARKS
# ------------------------------------------------------------------------------
with tabs[2]:
    st.subheader("⚖️ Empirical Benchmark Results & Verification")
    st.caption("Rigorous evaluation on the 2011 out-of-time holdout test cohort (N = 2,160, Test Base Rate = 24.07%).")

    if "test_results.csv" in bench_tables:
        st.markdown("##### 1. Model Family Comparison Table")
        t_df = bench_tables["test_results.csv"].copy()
        if "Unnamed: 0" in t_df.columns:
            t_df = t_df.rename(columns={"Unnamed: 0": "Model"})
        st.dataframe(t_df.style.format({
            "PR-AUC": "{:.4f}",
            "ROC-AUC": "{:.4f}",
            "P@10%": "{:.2%}",
            "Lift@10%": "{:.2f}x",
            "Brier": "{:.4f}"
        }), width='stretch')

    b_c1, b_c2 = st.columns(2)

    with b_c1:
        st.markdown("##### 2. Feature Group Ablation Study")
        if "ablation_results.csv" in bench_tables:
            a_df = bench_tables["ablation_results.csv"].copy()
            if "Unnamed: 0" in a_df.columns:
                a_df = a_df.rename(columns={"Unnamed: 0": "Feature Set"})
            st.dataframe(a_df.style.format({
                "PR-AUC": "{:.4f}",
                "ROC-AUC": "{:.4f}",
                "P@10%": "{:.2%}",
                "Lift@10%": "{:.2f}x"
            }), width='stretch')
            st.caption("Shows monotonic PR-AUC improvements as timing, missingness, and multi-label taxonomy are added.")

    with b_c2:
        st.markdown("##### 3. Empirical Proof of Look-Ahead Bias")
        if "leakage_check.csv" in bench_tables:
            l_df = bench_tables["leakage_check.csv"].copy()
            if "Unnamed: 0" in l_df.columns:
                l_df = l_df.rename(columns={"Unnamed: 0": "Validation Strategy"})
            st.dataframe(l_df.style.format({
                "PR-AUC": "{:.4f}",
                "ROC-AUC": "{:.4f}",
                "P@10%": "{:.2%}",
                "Lift@10%": "{:.2f}x"
            }), width='stretch')
            st.caption("Standard random 5-fold CV inflates PR-AUC to 0.5405 (+26%), proving why time-split validation is critical.")


# ------------------------------------------------------------------------------
# TAB 4: RESEARCH SURVEY ALIGNMENT
# ------------------------------------------------------------------------------
with tabs[3]:
    st.subheader("📚 Alignment with Research Survey Literature")
    st.markdown("""
    This platform directly operationalizes the research recommendations from our survey paper
    (*"Machine Learning-Based Startup Success and Investment Prediction: A Survey"*).
    """)

    paper_alignment = pd.DataFrame([
        {
            "Surveyed Paper": "Ulrich Kaiser & Johan Kuhn (2020) [1]",
            "Problem / Focus": "Early textual & non-textual indicators for Danish ventures",
            "How Implemented in VenturePulse": "Multi-label category tag extraction and web-presence indicators (`has_homepage`)."
        },
        {
            "Surveyed Paper": "Francesco Corea et al. (2021) [2]",
            "Problem / Focus": "Heuristic checklist & data-driven VC framework (600k firms)",
            "How Implemented in VenturePulse": "Automated transparent investment checklist scorecard covering capital, velocity, and ecosystem."
        },
        {
            "Surveyed Paper": "J. Park, S. Choi & Y. Feng (2024) [3]",
            "Problem / Focus": "Resolving class imbalance (9:1 ratio) & early-stage signals",
            "How Implemented in VenturePulse": "Cost-sensitive tree ensembles evaluated strictly on PR-AUC & Precision@Top-10%."
        },
        {
            "Surveyed Paper": "L. Gautam & N. Wattanapongsakorn (2024) [4]",
            "Problem / Focus": "Gradient boosting & Crunchbase venture benchmarks",
            "How Implemented in VenturePulse": "Comparison of LightGBM, XGBoost, and Random Forest on mature Crunchbase cohorts."
        },
        {
            "Surveyed Paper": "A. Thirupathi, T. Alhanai & M. Ghassemi (2022) [10]",
            "Problem / Focus": "Early signs of venture success & time-varying factor decay",
            "How Implemented in VenturePulse": "Strict point-in-time features; complete masking of all post-seed rounds."
        }
    ])

    st.dataframe(paper_alignment, width='stretch')
    st.markdown("""
    > **Key Conclusion from Survey**: Machine learning algorithms excel at early deal screening (providing a ~2.0x lift in hit-rate),
    > but because of high real-world venture uncertainty, the system functions as **decision support for human VCs**, not a black-box replacement.
    """)
