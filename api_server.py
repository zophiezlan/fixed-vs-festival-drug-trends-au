"""
REST API for Drug Checking Services Data.

Provides programmatic access to quantitative data, qualitative insights,
AI analysis, and predictions.

Run with: python api_server.py
Access at: http://localhost:5000
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from analysis import DrugCheckingAnalyzer
from qualitative_analysis import QualitativeAnalyzer

try:
    from ai_nlp_analysis import AIQualitativeAnalyzer
    from ml_predictive_models import NPSTrendPredictor, AnomalyDetector
    from ai_research_assistant import ResearchAssistant
    from network_analysis import SubstanceNetworkAnalyzer
    HAS_AI_MODULES = True
except ImportError:
    HAS_AI_MODULES = False

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
app.config['JSON_SORT_KEYS'] = False


# Helper functions
def load_data():
    """Load quantitative and qualitative data."""
    try:
        quant_df = pd.read_csv('data/combined_data.csv')
        quant_df['date'] = pd.to_datetime(quant_df['date'])

        qual_df = pd.read_csv('data/all_interviews.csv')
        qual_df['interview_date'] = pd.to_datetime(qual_df['interview_date'])

        return quant_df, qual_df, True
    except FileNotFoundError:
        return None, None, False


# Root endpoint
@app.route('/')
def index():
    """API documentation."""
    return jsonify({
        'name': 'Australian Drug Checking Services API',
        'version': '1.0.0',
        'description': 'REST API for accessing drug checking data and AI-powered insights',
        'endpoints': {
            'health': {
                'url': '/api/health',
                'method': 'GET',
                'description': 'Check API health status'
            },
            'quantitative': {
                'url': '/api/quantitative/summary',
                'method': 'GET',
                'description': 'Get quantitative analysis summary',
                'params': ['service_type (optional)']
            },
            'qualitative': {
                'url': '/api/qualitative/summary',
                'method': 'GET',
                'description': 'Get qualitative analysis summary'
            },
            'ai_analysis': {
                'url': '/api/ai/sentiment',
                'method': 'GET',
                'description': 'Get AI sentiment analysis'
            },
            'predictions': {
                'url': '/api/predictions/forecast',
                'method': 'GET',
                'description': 'Get NPS trend forecasts',
                'params': ['service_type', 'periods (optional)']
            },
            'network': {
                'url': '/api/network/cooccurrence',
                'method': 'GET',
                'description': 'Get substance co-occurrence network'
            },
            'data': {
                'url': '/api/data/samples',
                'method': 'GET',
                'description': 'Get raw sample data',
                'params': ['service_type (optional)', 'limit (optional)', 'offset (optional)']
            }
        },
        'documentation': 'Visit /docs for full API documentation'
    })


@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    quant_df, qual_df, data_loaded = load_data()

    return jsonify({
        'status': 'healthy' if data_loaded else 'degraded',
        'timestamp': datetime.now().isoformat(),
        'data_loaded': data_loaded,
        'ai_modules_available': HAS_AI_MODULES,
        'quantitative_samples': len(quant_df) if data_loaded else 0,
        'qualitative_interviews': len(qual_df) if data_loaded else 0
    })


@app.route('/api/quantitative/summary')
def quantitative_summary():
    """Get quantitative analysis summary."""
    quant_df, _, data_loaded = load_data()

    if not data_loaded:
        return jsonify({'error': 'Data not available'}), 503

    service_type = request.args.get('service_type')

    if service_type:
        quant_df = quant_df[quant_df['service_type'] == service_type]

    analyzer = DrugCheckingAnalyzer(dataframe=quant_df)
    comparison = analyzer.get_service_comparison()

    # Diversity analysis
    diversity = {}
    for stype in quant_df['service_type'].unique():
        diversity[stype] = analyzer.calculate_diversity_index(stype)

    # Early warning
    detection_advantage = analyzer.calculate_detection_time_advantage()

    return jsonify({
        'service_comparison': comparison,
        'diversity_analysis': diversity,
        'early_warning': detection_advantage,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/quantitative/substances')
def top_substances():
    """Get top detected substances."""
    quant_df, _, data_loaded = load_data()

    if not data_loaded:
        return jsonify({'error': 'Data not available'}), 503

    service_type = request.args.get('service_type')
    top_n = int(request.args.get('top_n', 10))

    analyzer = DrugCheckingAnalyzer(dataframe=quant_df)
    top_subs = analyzer.get_top_substances(service_type, top_n)

    return jsonify({
        'service_type': service_type or 'all',
        'top_substances': top_subs.to_dict(),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/qualitative/summary')
def qualitative_summary():
    """Get qualitative analysis summary."""
    _, qual_df, data_loaded = load_data()

    if not data_loaded:
        return jsonify({'error': 'Data not available'}), 503

    analyzer = QualitativeAnalyzer(dataframe=qual_df)

    summary = analyzer.get_participant_summary()
    themes = analyzer.extract_themes()
    differences = analyzer.identify_key_differences()

    return jsonify({
        'participant_summary': summary,
        'themes': themes,
        'key_differences': differences,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/ai/sentiment')
def ai_sentiment():
    """Get AI sentiment analysis."""
    if not HAS_AI_MODULES:
        return jsonify({'error': 'AI modules not available'}), 503

    _, qual_df, data_loaded = load_data()

    if not data_loaded:
        return jsonify({'error': 'Data not available'}), 503

    analyzer = AIQualitativeAnalyzer(dataframe=qual_df)
    sentiment = analyzer.perform_sentiment_analysis()

    return jsonify({
        'sentiment_analysis': sentiment,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/ai/topics')
def ai_topics():
    """Get AI topic modeling results."""
    if not HAS_AI_MODULES:
        return jsonify({'error': 'AI modules not available'}), 503

    _, qual_df, data_loaded = load_data()

    if not data_loaded:
        return jsonify({'error': 'Data not available'}), 503

    n_topics = int(request.args.get('n_topics', 5))
    method = request.args.get('method', 'lda')

    analyzer = AIQualitativeAnalyzer(dataframe=qual_df)
    topics = analyzer.perform_topic_modeling(n_topics=n_topics, method=method)

    return jsonify({
        'topic_modeling': topics,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/ai/entities')
def ai_entities():
    """Get named entity recognition results."""
    if not HAS_AI_MODULES:
        return jsonify({'error': 'AI modules not available'}), 503

    _, qual_df, data_loaded = load_data()

    if not data_loaded:
        return jsonify({'error': 'Data not available'}), 503

    analyzer = AIQualitativeAnalyzer(dataframe=qual_df)
    entities = analyzer.extract_named_entities()

    return jsonify({
        'named_entities': entities,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/predictions/forecast')
def nps_forecast():
    """Get NPS trend forecast."""
    if not HAS_AI_MODULES:
        return jsonify({'error': 'AI modules not available'}), 503

    quant_df, _, data_loaded = load_data()

    if not data_loaded:
        return jsonify({'error': 'Data not available'}), 503

    service_type = request.args.get('service_type', 'Fixed-site')
    periods_ahead = int(request.args.get('periods', 6))

    predictor = NPSTrendPredictor(dataframe=quant_df)
    forecast = predictor.forecast_nps_trend(service_type, periods_ahead)

    return jsonify({
        'forecast': forecast,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/predictions/anomalies')
def detect_anomalies():
    """Detect anomalies and emerging substances."""
    if not HAS_AI_MODULES:
        return jsonify({'error': 'AI modules not available'}), 503

    quant_df, _, data_loaded = load_data()

    if not data_loaded:
        return jsonify({'error': 'Data not available'}), 503

    threshold_days = int(request.args.get('threshold_days', 30))

    detector = AnomalyDetector(dataframe=quant_df)
    emerging = detector.detect_emerging_substances(threshold_days)
    anomalies = detector.detect_statistical_anomalies()

    return jsonify({
        'emerging_substances': emerging,
        'statistical_anomalies': anomalies,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/network/cooccurrence')
def network_cooccurrence():
    """Get substance co-occurrence network."""
    if not HAS_AI_MODULES:
        return jsonify({'error': 'AI modules not available'}), 503

    quant_df, _, data_loaded = load_data()

    if not data_loaded:
        return jsonify({'error': 'Data not available'}), 503

    time_window = request.args.get('time_window', 'W')
    min_cooccurrence = int(request.args.get('min_cooccurrence', 2))

    analyzer = SubstanceNetworkAnalyzer(dataframe=quant_df)
    networks = analyzer.build_temporal_cooccurrence_network(time_window, min_cooccurrence)

    # Remove network objects (not JSON serializable)
    for service_type in networks:
        if 'network_object' in networks[service_type]:
            del networks[service_type]['network_object']

    return jsonify({
        'networks': networks,
        'parameters': {
            'time_window': time_window,
            'min_cooccurrence': min_cooccurrence
        },
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/network/diffusion')
def nps_diffusion():
    """Get NPS diffusion analysis."""
    if not HAS_AI_MODULES:
        return jsonify({'error': 'AI modules not available'}), 503

    quant_df, _, data_loaded = load_data()

    if not data_loaded:
        return jsonify({'error': 'Data not available'}), 503

    analyzer = SubstanceNetworkAnalyzer(dataframe=quant_df)
    diffusion = analyzer.analyze_nps_diffusion()

    return jsonify({
        'nps_diffusion': diffusion,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/research/recommendations')
def policy_recommendations():
    """Get policy recommendations."""
    if not HAS_AI_MODULES:
        return jsonify({'error': 'AI modules not available'}), 503

    quant_df, qual_df, data_loaded = load_data()

    if not data_loaded:
        return jsonify({'error': 'Data not available'}), 503

    assistant = ResearchAssistant(quantitative_data=quant_df, qualitative_data=qual_df)
    recommendations = assistant.generate_policy_recommendations()

    return jsonify({
        'recommendations': recommendations,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/research/hypotheses')
def research_hypotheses():
    """Get generated research hypotheses."""
    if not HAS_AI_MODULES:
        return jsonify({'error': 'AI modules not available'}), 503

    quant_df, qual_df, data_loaded = load_data()

    if not data_loaded:
        return jsonify({'error': 'Data not available'}), 503

    assistant = ResearchAssistant(quantitative_data=quant_df, qualitative_data=qual_df)
    hypotheses = assistant.generate_hypotheses()

    return jsonify({
        'hypotheses': hypotheses,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/data/samples')
def get_samples():
    """Get raw sample data."""
    quant_df, _, data_loaded = load_data()

    if not data_loaded:
        return jsonify({'error': 'Data not available'}), 503

    service_type = request.args.get('service_type')
    limit = int(request.args.get('limit', 100))
    offset = int(request.args.get('offset', 0))

    if service_type:
        quant_df = quant_df[quant_df['service_type'] == service_type]

    # Pagination
    total = len(quant_df)
    samples = quant_df.iloc[offset:offset + limit]

    return jsonify({
        'total': total,
        'limit': limit,
        'offset': offset,
        'count': len(samples),
        'data': samples.to_dict(orient='records'),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/data/statistics')
def get_statistics():
    """Get dataset statistics."""
    quant_df, qual_df, data_loaded = load_data()

    if not data_loaded:
        return jsonify({'error': 'Data not available'}), 503

    stats = {
        'quantitative': {
            'total_samples': len(quant_df),
            'date_range': {
                'start': quant_df['date'].min().isoformat(),
                'end': quant_df['date'].max().isoformat()
            },
            'service_types': quant_df['service_type'].value_counts().to_dict(),
            'unique_substances': quant_df['substance_detected'].nunique(),
            'nps_samples': int(quant_df['is_nps'].sum()),
            'nps_rate': float(quant_df['is_nps'].mean())
        },
        'qualitative': {
            'total_interviews': len(qual_df),
            'participant_types': qual_df['participant_type'].value_counts().to_dict(),
            'service_types': qual_df['service_type'].value_counts().to_dict(),
            'avg_duration_minutes': float(qual_df['interview_duration_minutes'].mean())
        },
        'timestamp': datetime.now().isoformat()
    }

    return jsonify(stats)


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("=" * 80)
    print("Australian Drug Checking Services API")
    print("=" * 80)
    print("\nStarting API server...")
    print("Access API at: http://localhost:5000")
    print("Documentation: http://localhost:5000")
    print("\nPress Ctrl+C to stop")
    print("=" * 80)

    app.run(debug=True, host='0.0.0.0', port=5000)
