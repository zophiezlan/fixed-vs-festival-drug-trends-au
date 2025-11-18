"""
Interactive Streamlit Dashboard for Drug Checking Services Analysis.

This dashboard provides real-time visualization and analysis of drug checking data
with AI-powered insights.

Run with: streamlit run streamlit_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    st.warning("Plotly not installed. Some visualizations will be limited.")

from analysis import DrugCheckingAnalyzer
from qualitative_analysis import QualitativeAnalyzer

try:
    from ai_nlp_analysis import AIQualitativeAnalyzer
    from ml_predictive_models import NPSTrendPredictor, AnomalyDetector
    from ai_research_assistant import ResearchAssistant
    HAS_AI_MODULES = True
except ImportError:
    HAS_AI_MODULES = False


# Page configuration
st.set_page_config(
    page_title="Australian Drug Checking Services Dashboard",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and cache data."""
    try:
        quant_df = pd.read_csv('data/combined_data.csv')
        quant_df['date'] = pd.to_datetime(quant_df['date'])

        qual_df = pd.read_csv('data/all_interviews.csv')
        qual_df['interview_date'] = pd.to_datetime(qual_df['interview_date'])

        return quant_df, qual_df, True
    except FileNotFoundError:
        return None, None, False


def main_dashboard():
    """Main dashboard view."""
    # Header
    st.markdown('<div class="main-header">🔬 Australian Drug Checking Services Dashboard</div>',
                unsafe_allow_html=True)
    st.markdown("### AI-Powered Analysis of Fixed-Site vs Festival Services")
    st.markdown("---")

    # Load data
    quant_df, qual_df, data_loaded = load_data()

    if not data_loaded:
        st.error("⚠️ Data not found. Please run `python main.py` first to generate data.")
        st.info("Run: `python main.py` in the terminal to generate analysis data.")
        return

    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150x150.png?text=Drug+Checking", use_container_width=True)
        st.title("Navigation")

        page = st.radio(
            "Select View:",
            ["📊 Overview", "📈 Quantitative Analysis", "💬 Qualitative Insights",
             "🤖 AI Analysis", "🔮 Predictions", "📋 Policy Recommendations"]
        )

        st.markdown("---")
        st.markdown("### Filters")

        # Service type filter
        service_filter = st.multiselect(
            "Service Type",
            options=quant_df['service_type'].unique().tolist(),
            default=quant_df['service_type'].unique().tolist()
        )

        # Date range filter
        date_range = st.date_input(
            "Date Range",
            value=(quant_df['date'].min(), quant_df['date'].max()),
            min_value=quant_df['date'].min(),
            max_value=quant_df['date'].max()
        )

        # Apply filters
        filtered_df = quant_df[
            (quant_df['service_type'].isin(service_filter)) &
            (quant_df['date'] >= pd.Timestamp(date_range[0])) &
            (quant_df['date'] <= pd.Timestamp(date_range[1]))
        ]

        st.markdown("---")
        st.markdown(f"**Samples:** {len(filtered_df)}")
        st.markdown(f"**Date Range:** {len(pd.date_range(date_range[0], date_range[1], freq='D'))} days")

    # Render selected page
    if page == "📊 Overview":
        show_overview(filtered_df, qual_df)
    elif page == "📈 Quantitative Analysis":
        show_quantitative_analysis(filtered_df)
    elif page == "💬 Qualitative Insights":
        show_qualitative_insights(qual_df)
    elif page == "🤖 AI Analysis":
        show_ai_analysis(filtered_df, qual_df)
    elif page == "🔮 Predictions":
        show_predictions(filtered_df)
    elif page == "📋 Policy Recommendations":
        show_policy_recommendations(filtered_df, qual_df)


def show_overview(df, qual_df):
    """Overview dashboard."""
    st.header("📊 Overview Dashboard")

    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Samples", len(df))

    with col2:
        st.metric("Unique Substances", df['substance_detected'].nunique())

    with col3:
        nps_rate = df['is_nps'].mean() * 100
        st.metric("NPS Detection Rate", f"{nps_rate:.1f}%")

    with col4:
        st.metric("Service Types", df['service_type'].nunique())

    st.markdown("---")

    # Comparison by service type
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Samples by Service Type")
        service_counts = df['service_type'].value_counts()

        if HAS_PLOTLY:
            fig = px.pie(values=service_counts.values, names=service_counts.index,
                        title="Sample Distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(service_counts)

    with col2:
        st.subheader("NPS Detection by Service")
        nps_by_service = df.groupby('service_type')['is_nps'].agg(['sum', 'count'])
        nps_by_service['rate'] = nps_by_service['sum'] / nps_by_service['count'] * 100

        if HAS_PLOTLY:
            fig = px.bar(nps_by_service.reset_index(), x='service_type', y='rate',
                        labels={'rate': 'NPS Rate (%)', 'service_type': 'Service Type'},
                        title="NPS Detection Rates")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(nps_by_service['rate'])

    # Timeline
    st.subheader("Detection Timeline")
    df['month'] = df['date'].dt.to_period('M').astype(str)
    timeline = df.groupby(['month', 'service_type']).size().reset_index(name='count')

    if HAS_PLOTLY:
        fig = px.line(timeline, x='month', y='count', color='service_type',
                     title="Samples Over Time", labels={'count': 'Number of Samples'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        pivot = timeline.pivot(index='month', columns='service_type', values='count')
        st.line_chart(pivot)

    # Key Findings
    st.markdown("---")
    st.subheader("🔍 Key Findings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("**Substance Diversity**")

        for service_type in df['service_type'].unique():
            unique_subs = df[df['service_type'] == service_type]['substance_detected'].nunique()
            st.markdown(f"- {service_type}: **{unique_subs}** unique substances")

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("**Adulterant Detection**")

        for service_type in df['service_type'].unique():
            service_data = df[df['service_type'] == service_type]
            adulterant_pct = (service_data['num_adulterants'] > 0).mean() * 100
            st.markdown(f"- {service_type}: **{adulterant_pct:.1f}%** with adulterants")

        st.markdown('</div>', unsafe_allow_html=True)


def show_quantitative_analysis(df):
    """Detailed quantitative analysis."""
    st.header("📈 Quantitative Analysis")

    # Diversity Analysis
    st.subheader("Substance Diversity Analysis")

    analyzer = DrugCheckingAnalyzer(dataframe=df)
    diversity_results = {}

    for service_type in df['service_type'].unique():
        diversity = analyzer.calculate_diversity_index(service_type)
        diversity_results[service_type] = diversity

    # Display diversity metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Shannon Diversity Index**")
        for service_type, metrics in diversity_results.items():
            st.metric(service_type, f"{metrics['shannon_diversity']:.3f}")

    with col2:
        st.markdown("**Simpson Diversity Index**")
        for service_type, metrics in diversity_results.items():
            st.metric(service_type, f"{metrics['simpson_diversity']:.3f}")

    with col3:
        st.markdown("**Species Richness**")
        for service_type, metrics in diversity_results.items():
            st.metric(service_type, metrics['species_richness'])

    # Top substances
    st.markdown("---")
    st.subheader("Most Detected Substances")

    tab1, tab2 = st.tabs(["Fixed-site", "Festival"])

    with tab1:
        if 'Fixed-site' in df['service_type'].unique():
            top_subs = analyzer.get_top_substances('Fixed-site', top_n=10)
            if HAS_PLOTLY:
                fig = px.bar(x=top_subs.values, y=top_subs.index, orientation='h',
                           labels={'x': 'Detections', 'y': 'Substance'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(top_subs)

    with tab2:
        if 'Festival' in df['service_type'].unique():
            top_subs = analyzer.get_top_substances('Festival', top_n=10)
            if HAS_PLOTLY:
                fig = px.bar(x=top_subs.values, y=top_subs.index, orientation='h',
                           labels={'x': 'Detections', 'y': 'Substance'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(top_subs)

    # Early warning analysis
    st.markdown("---")
    st.subheader("Early Warning Capability")

    detection_advantage = analyzer.calculate_detection_time_advantage()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Fixed-site Detected First", detection_advantage.get('Fixed-site', 0))
    with col2:
        st.metric("Festival Detected First", detection_advantage.get('Festival', 0))
    with col3:
        st.metric("Same Time", detection_advantage.get('Same', 0))


def show_qualitative_insights(qual_df):
    """Qualitative analysis insights."""
    st.header("💬 Qualitative Insights")

    analyzer = QualitativeAnalyzer(dataframe=qual_df)

    # Participant summary
    st.subheader("Participant Overview")

    summary = analyzer.get_participant_summary()

    for participant_type, services in summary.items():
        st.markdown(f"**{participant_type}:**")
        cols = st.columns(len(services))
        for i, (service_type, data) in enumerate(services.items()):
            with cols[i]:
                st.metric(f"{service_type}", f"{data['count']} interviews")
                st.caption(f"Avg duration: {data['avg_duration']:.0f} min")

    # Themes
    st.markdown("---")
    st.subheader("Thematic Analysis")

    themes = analyzer.extract_themes()

    for theme_name, theme_data in themes.items():
        with st.expander(f"📌 {theme_name}"):
            for service_type, data in theme_data.items():
                st.markdown(f"**{service_type}**: {data['response_count']} responses")
                if data['sample_quotes']:
                    st.info(f"💭 \"{data['sample_quotes'][0][:150]}...\"")


def show_ai_analysis(df, qual_df):
    """AI-powered analysis."""
    st.header("🤖 AI-Powered Analysis")

    if not HAS_AI_MODULES:
        st.warning("AI modules not available. Please ensure all dependencies are installed.")
        return

    tab1, tab2, tab3 = st.tabs(["Sentiment Analysis", "Topic Modeling", "Named Entities"])

    ai_analyzer = AIQualitativeAnalyzer(dataframe=qual_df)

    with tab1:
        st.subheader("Sentiment Analysis")
        sentiment = ai_analyzer.perform_sentiment_analysis()

        if 'by_service_type' in sentiment:
            for service_type, data in sentiment['by_service_type'].items():
                st.markdown(f"**{service_type}**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg Sentiment", f"{data.get('avg_sentiment', 0):.3f}")
                with col2:
                    st.metric("Positive", data.get('positive_responses', 0))
                with col3:
                    st.metric("Negative", data.get('negative_responses', 0))

    with tab2:
        st.subheader("Topic Modeling")
        topics = ai_analyzer.perform_topic_modeling(n_topics=5)

        if 'topics' in topics and 'error' not in topics:
            for topic_name, topic_data in topics['topics'].items():
                st.markdown(f"**{topic_name}:**")
                st.markdown(f"Keywords: {', '.join(topic_data['keywords'][:8])}")

    with tab3:
        st.subheader("Named Entity Recognition")
        entities = ai_analyzer.extract_named_entities()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Substances Mentioned:**")
            for substance, count in list(entities['substances'].items())[:10]:
                st.markdown(f"- {substance}: {count}")

        with col2:
            st.markdown("**Service Type Mentions:**")
            for service, count in entities['service_types'].items():
                st.markdown(f"- {service}: {count}")


def show_predictions(df):
    """ML predictions and forecasting."""
    st.header("🔮 Predictive Analytics")

    if not HAS_AI_MODULES:
        st.warning("ML modules not available.")
        return

    predictor = NPSTrendPredictor(dataframe=df)

    st.subheader("NPS Trend Forecasting")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Fixed-site Services**")
        if 'Fixed-site' in df['service_type'].unique():
            forecast = predictor.forecast_nps_trend('Fixed-site', periods_ahead=6)
            if 'error' not in forecast:
                st.metric("Current NPS Rate", f"{forecast['current_rate']:.1%}")
                st.metric("Predicted Trend", forecast['trend'].upper())
                st.line_chart(forecast['predictions'])

    with col2:
        st.markdown("**Festival Services**")
        if 'Festival' in df['service_type'].unique():
            forecast = predictor.forecast_nps_trend('Festival', periods_ahead=6)
            if 'error' not in forecast:
                st.metric("Current NPS Rate", f"{forecast['current_rate']:.1%}")
                st.metric("Predicted Trend", forecast['trend'].upper())
                st.line_chart(forecast['predictions'])

    # Anomaly detection
    st.markdown("---")
    st.subheader("Anomaly Detection")

    anomaly_detector = AnomalyDetector(dataframe=df)
    emerging = anomaly_detector.detect_emerging_substances(threshold_days=30)

    for service_type, data in emerging.items():
        with st.expander(f"🚨 {service_type} - Emerging Substances"):
            st.markdown(f"**{data['emerging_count']} new substances detected in last 30 days**")
            for sub in data['substances']:
                st.markdown(f"- {sub['substance']} (NPS: {sub['is_nps']}) - First detected: {sub['first_detected']}")


def show_policy_recommendations(df, qual_df):
    """Policy recommendations."""
    st.header("📋 Policy Recommendations")

    if not HAS_AI_MODULES:
        st.warning("Research assistant module not available.")
        return

    assistant = ResearchAssistant(quantitative_data=df, qualitative_data=qual_df)
    recommendations = assistant.generate_policy_recommendations()

    for rec in recommendations:
        with st.expander(f"{rec['id']}: {rec['recommendation']} [{rec['priority'].upper()}]"):
            st.markdown("**Rationale:**")
            for rationale in rec['rationale']:
                st.markdown(f"- {rationale}")

            st.markdown("**Implementation:**")
            for impl in rec['implementation']:
                st.markdown(f"→ {impl}")

            st.markdown("**Expected Outcomes:**")
            for outcome in rec['expected_outcomes']:
                st.markdown(f"✓ {outcome}")


def show_about():
    """About page."""
    st.title("About This Dashboard")

    st.markdown("""
    This interactive dashboard provides comprehensive analysis of drug checking services
    in Australia, comparing fixed-site and festival-based models.

    ### Features:
    - **Real-time Data Visualization**: Interactive charts and metrics
    - **AI-Powered Analysis**: NLP, sentiment analysis, topic modeling
    - **Predictive Analytics**: ML-based trend forecasting
    - **Policy Insights**: Evidence-based recommendations

    ### Data Sources:
    - Quantitative drug checking data
    - Qualitative stakeholder interviews
    - Mixed-methods integration

    ### Technologies:
    - Streamlit for interactive dashboards
    - Plotly for advanced visualizations
    - Scikit-learn for machine learning
    - Custom NLP and research assistant modules
    """)


if __name__ == "__main__":
    main_dashboard()
