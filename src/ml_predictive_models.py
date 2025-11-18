"""
Machine Learning Predictive Models for Drug Checking Data Analysis.

This module provides ML-based predictive capabilities including:
- Time series forecasting for NPS trends
- Classification models for substance detection
- Anomaly detection for emerging threats
- Clustering for pattern identification
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML libraries
try:
    from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import classification_report, accuracy_score, mean_squared_error
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not installed. ML features will be limited.")

try:
    from scipy import stats
    from scipy.signal import find_peaks
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class NPSTrendPredictor:
    """
    Predict Novel Psychoactive Substance trends using time series analysis and ML.
    """

    def __init__(self, data_path=None, dataframe=None):
        """Initialize with drug checking data."""
        if dataframe is not None:
            self.df = dataframe.copy()
        elif data_path:
            self.df = pd.read_csv(data_path)
        else:
            raise ValueError("Must provide either data_path or dataframe")

        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date')
        self.models = {}

    def prepare_time_series_features(self, freq='M'):
        """
        Prepare time series features for ML models.

        Args:
            freq: Frequency for aggregation ('D', 'W', 'M')

        Returns:
            DataFrame with engineered features
        """
        # Aggregate by time period
        self.df['period'] = self.df['date'].dt.to_period(freq)

        ts_data = self.df.groupby(['period', 'service_type']).agg({
            'is_nps': ['sum', 'count'],
            'substance_detected': 'nunique',
            'num_adulterants': 'mean'
        }).reset_index()

        ts_data.columns = ['period', 'service_type', 'nps_count', 'total_samples',
                           'unique_substances', 'avg_adulterants']

        # Calculate NPS rate
        ts_data['nps_rate'] = ts_data['nps_count'] / ts_data['total_samples']

        # Add time-based features
        ts_data['period_str'] = ts_data['period'].astype(str)
        ts_data['time_index'] = range(len(ts_data))

        # Lag features
        for col in ['nps_rate', 'nps_count', 'unique_substances']:
            ts_data[f'{col}_lag1'] = ts_data.groupby('service_type')[col].shift(1)
            ts_data[f'{col}_lag2'] = ts_data.groupby('service_type')[col].shift(2)

        # Rolling statistics
        ts_data['nps_rate_rolling_mean'] = ts_data.groupby('service_type')['nps_rate'].rolling(window=3, min_periods=1).mean().reset_index(drop=True)
        ts_data['nps_rate_rolling_std'] = ts_data.groupby('service_type')['nps_rate'].rolling(window=3, min_periods=1).std().reset_index(drop=True)

        return ts_data.fillna(0)

    def forecast_nps_trend(self, service_type, periods_ahead=6):
        """
        Forecast NPS detection rates for future periods.

        Args:
            service_type: 'Fixed-site' or 'Festival'
            periods_ahead: Number of periods to forecast

        Returns:
            Dictionary with predictions and confidence intervals
        """
        if not HAS_SKLEARN:
            return self._simple_trend_forecast(service_type, periods_ahead)

        # Prepare data
        ts_data = self.prepare_time_series_features()
        service_data = ts_data[ts_data['service_type'] == service_type].copy()

        if len(service_data) < 5:
            return {'error': 'Insufficient data for forecasting'}

        # Features for model
        feature_cols = ['time_index', 'total_samples', 'unique_substances',
                       'nps_rate_lag1', 'nps_rate_lag2', 'nps_rate_rolling_mean']
        feature_cols = [col for col in feature_cols if col in service_data.columns]

        X = service_data[feature_cols].fillna(0)
        y = service_data['nps_rate']

        # Train model
        model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=3)
        model.fit(X, y)
        self.models[f'nps_forecast_{service_type}'] = model

        # Generate future predictions
        last_row = service_data.iloc[-1]
        predictions = []
        lower_bounds = []
        upper_bounds = []

        current_features = last_row[feature_cols].values.reshape(1, -1)

        for i in range(periods_ahead):
            pred = model.predict(current_features)[0]
            predictions.append(pred)

            # Simple confidence intervals (can be improved with quantile regression)
            std_error = 0.05 * (i + 1)  # Increasing uncertainty
            lower_bounds.append(max(0, pred - 1.96 * std_error))
            upper_bounds.append(min(1, pred + 1.96 * std_error))

            # Update features for next prediction
            if len(current_features[0]) > 2:
                current_features[0, 0] += 1  # time_index
                if 'nps_rate_lag1' in feature_cols:
                    lag1_idx = feature_cols.index('nps_rate_lag1')
                    current_features[0, lag1_idx] = pred

        return {
            'service_type': service_type,
            'periods_ahead': periods_ahead,
            'predictions': predictions,
            'lower_bound': lower_bounds,
            'upper_bound': upper_bounds,
            'current_rate': float(service_data['nps_rate'].iloc[-1]),
            'trend': 'increasing' if predictions[-1] > predictions[0] else 'decreasing'
        }

    def _simple_trend_forecast(self, service_type, periods_ahead=6):
        """Simple linear trend when sklearn unavailable."""
        service_data = self.df[self.df['service_type'] == service_type]

        # Calculate monthly NPS rate
        service_data['month'] = service_data['date'].dt.to_period('M')
        monthly = service_data.groupby('month').agg({
            'is_nps': 'sum',
            'substance_detected': 'count'
        })
        monthly['nps_rate'] = monthly['is_nps'] / monthly['substance_detected']

        if len(monthly) < 3:
            return {'error': 'Insufficient data'}

        # Simple linear extrapolation
        rates = monthly['nps_rate'].values
        x = np.arange(len(rates))
        slope = (rates[-1] - rates[0]) / len(rates)

        predictions = []
        for i in range(periods_ahead):
            pred = rates[-1] + slope * (i + 1)
            predictions.append(max(0, min(1, pred)))

        return {
            'service_type': service_type,
            'predictions': predictions,
            'trend': 'increasing' if slope > 0 else 'decreasing'
        }

    def detect_trend_changes(self):
        """
        Detect significant changes in NPS detection trends.

        Returns:
            Dictionary with change points and trend analysis
        """
        results = {}

        for service_type in self.df['service_type'].unique():
            service_data = self.df[self.df['service_type'] == service_type].copy()
            service_data['month'] = service_data['date'].dt.to_period('M')

            monthly = service_data.groupby('month').agg({
                'is_nps': 'sum',
                'substance_detected': 'count'
            })
            monthly['nps_rate'] = monthly['is_nps'] / monthly['substance_detected']

            rates = monthly['nps_rate'].values

            if len(rates) < 5:
                continue

            # Simple change point detection
            changes = []
            for i in range(2, len(rates) - 2):
                before_mean = np.mean(rates[max(0, i-3):i])
                after_mean = np.mean(rates[i:min(len(rates), i+3)])

                if abs(after_mean - before_mean) > 0.1:  # 10% threshold
                    changes.append({
                        'period': i,
                        'before_rate': float(before_mean),
                        'after_rate': float(after_mean),
                        'change': float(after_mean - before_mean)
                    })

            results[service_type] = {
                'change_points': changes,
                'overall_trend': 'increasing' if rates[-1] > rates[0] else 'decreasing',
                'volatility': float(np.std(rates)),
                'current_rate': float(rates[-1])
            }

        return results


class AnomalyDetector:
    """
    Detect anomalous patterns in drug checking data using ML techniques.
    """

    def __init__(self, data_path=None, dataframe=None):
        """Initialize with drug checking data."""
        if dataframe is not None:
            self.df = dataframe.copy()
        elif data_path:
            self.df = pd.read_csv(data_path)
        else:
            raise ValueError("Must provide either data_path or dataframe")

        self.df['date'] = pd.to_datetime(self.df['date'])

    def detect_emerging_substances(self, threshold_days=30):
        """
        Detect newly appearing substances that may indicate emerging threats.

        Args:
            threshold_days: Number of days to look back for "new" substances

        Returns:
            Dictionary with emerging substances by service type
        """
        recent_date = self.df['date'].max() - pd.Timedelta(days=threshold_days)
        recent_data = self.df[self.df['date'] >= recent_date]
        historical_data = self.df[self.df['date'] < recent_date]

        results = {}

        for service_type in self.df['service_type'].unique():
            recent_substances = set(
                recent_data[recent_data['service_type'] == service_type]['substance_detected'].unique()
            )
            historical_substances = set(
                historical_data[historical_data['service_type'] == service_type]['substance_detected'].unique()
            )

            emerging = recent_substances - historical_substances

            # Get details for emerging substances
            emerging_details = []
            for substance in emerging:
                substance_data = recent_data[
                    (recent_data['service_type'] == service_type) &
                    (recent_data['substance_detected'] == substance)
                ]

                emerging_details.append({
                    'substance': substance,
                    'first_detected': substance_data['date'].min().strftime('%Y-%m-%d'),
                    'detection_count': len(substance_data),
                    'is_nps': bool(substance_data['is_nps'].iloc[0])
                })

            results[service_type] = {
                'emerging_count': len(emerging),
                'substances': emerging_details
            }

        return results

    def detect_statistical_anomalies(self):
        """
        Detect statistical anomalies in substance detection patterns.

        Returns:
            Dictionary with anomalous patterns
        """
        if not HAS_SKLEARN or not HAS_SCIPY:
            return self._simple_anomaly_detection()

        anomalies = {}

        for service_type in self.df['service_type'].unique():
            service_data = self.df[self.df['service_type'] == service_type].copy()

            # Aggregate by week
            service_data['week'] = service_data['date'].dt.to_period('W')
            weekly = service_data.groupby('week').agg({
                'is_nps': 'sum',
                'substance_detected': 'count',
                'num_adulterants': 'mean'
            }).reset_index()

            weekly['nps_rate'] = weekly['is_nps'] / weekly['substance_detected']

            # Use Isolation Forest for anomaly detection
            if len(weekly) < 5:
                continue

            features = weekly[['nps_rate', 'substance_detected', 'num_adulterants']].fillna(0)

            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = iso_forest.fit_predict(features)

            # Identify anomalous weeks
            anomalous_weeks = weekly[anomaly_labels == -1]

            anomalies[service_type] = {
                'anomalous_periods': len(anomalous_weeks),
                'periods': [
                    {
                        'week': str(week),
                        'nps_rate': float(rate),
                        'samples': int(samples)
                    }
                    for week, rate, samples in zip(
                        anomalous_weeks['week'],
                        anomalous_weeks['nps_rate'],
                        anomalous_weeks['substance_detected']
                    )
                ]
            }

        return anomalies

    def _simple_anomaly_detection(self):
        """Simple z-score based anomaly detection."""
        anomalies = {}

        for service_type in self.df['service_type'].unique():
            service_data = self.df[self.df['service_type'] == service_type].copy()
            service_data['week'] = service_data['date'].dt.to_period('W')

            weekly = service_data.groupby('week').agg({
                'is_nps': 'sum',
                'substance_detected': 'count'
            })
            weekly['nps_rate'] = weekly['is_nps'] / weekly['substance_detected']

            # Simple z-score approach
            mean_rate = weekly['nps_rate'].mean()
            std_rate = weekly['nps_rate'].std()

            anomalous = weekly[abs(weekly['nps_rate'] - mean_rate) > 2 * std_rate]

            anomalies[service_type] = {
                'anomalous_periods': len(anomalous),
                'threshold': float(mean_rate + 2 * std_rate)
            }

        return anomalies


class SubstanceClusterAnalyzer:
    """
    Perform clustering analysis to identify patterns in substance detection.
    """

    def __init__(self, data_path=None, dataframe=None):
        """Initialize with drug checking data."""
        if dataframe is not None:
            self.df = dataframe.copy()
        elif data_path:
            self.df = pd.read_csv(data_path)
        else:
            raise ValueError("Must provide either data_path or dataframe")

    def cluster_by_detection_patterns(self, n_clusters=4):
        """
        Cluster substances by their detection patterns.

        Args:
            n_clusters: Number of clusters to identify

        Returns:
            Dictionary with cluster assignments and characteristics
        """
        if not HAS_SKLEARN:
            return {'error': 'sklearn required for clustering'}

        # Create substance feature matrix
        substance_features = []
        substances = []

        for substance in self.df['substance_detected'].unique():
            sub_data = self.df[self.df['substance_detected'] == substance]

            features = {
                'total_detections': len(sub_data),
                'fixed_site_pct': len(sub_data[sub_data['service_type'] == 'Fixed-site']) / len(sub_data),
                'is_nps': sub_data['is_nps'].iloc[0],
                'avg_adulterants': sub_data['num_adulterants'].mean(),
                'detection_span_days': (sub_data['date'].max() - sub_data['date'].min()).days
            }

            substance_features.append(list(features.values()))
            substances.append(substance)

        # Standardize and cluster
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(substance_features)

        kmeans = KMeans(n_clusters=min(n_clusters, len(substances)), random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)

        # Analyze clusters
        clusters = {}
        for i in range(n_clusters):
            cluster_substances = [substances[j] for j in range(len(substances)) if cluster_labels[j] == i]

            if not cluster_substances:
                continue

            cluster_data = self.df[self.df['substance_detected'].isin(cluster_substances)]

            clusters[f'Cluster {i+1}'] = {
                'substances': cluster_substances,
                'count': len(cluster_substances),
                'characteristics': {
                    'avg_detections': float(cluster_data.groupby('substance_detected').size().mean()),
                    'nps_percentage': float(cluster_data['is_nps'].mean() * 100),
                    'fixed_site_percentage': float(
                        len(cluster_data[cluster_data['service_type'] == 'Fixed-site']) / len(cluster_data) * 100
                    )
                }
            }

        return clusters

    def find_co_occurring_substances(self, min_support=5):
        """
        Identify substances that frequently co-occur as adulterants.

        Args:
            min_support: Minimum number of co-occurrences

        Returns:
            Dictionary with co-occurrence patterns
        """
        # This would require more detailed adulterant data
        # For now, analyze based on temporal co-occurrence

        co_occurrence = {}

        # Group by week to find substances detected together
        self.df['week'] = pd.to_datetime(self.df['date']).dt.to_period('W')

        for service_type in self.df['service_type'].unique():
            service_data = self.df[self.df['service_type'] == service_type]

            weekly_substances = service_data.groupby('week')['substance_detected'].apply(list)

            pairs = {}
            for substances in weekly_substances:
                unique_subs = list(set(substances))
                for i in range(len(unique_subs)):
                    for j in range(i + 1, len(unique_subs)):
                        pair = tuple(sorted([unique_subs[i], unique_subs[j]]))
                        pairs[pair] = pairs.get(pair, 0) + 1

            # Filter by min support
            significant_pairs = {k: v for k, v in pairs.items() if v >= min_support}

            co_occurrence[service_type] = {
                'pair_count': len(significant_pairs),
                'top_pairs': sorted(significant_pairs.items(), key=lambda x: x[1], reverse=True)[:10]
            }

        return co_occurrence


def generate_ml_predictions_report(data_path):
    """Generate comprehensive ML predictions report."""
    report = []
    report.append("=" * 80)
    report.append("MACHINE LEARNING PREDICTIVE ANALYTICS")
    report.append("NPS Trend Forecasting & Anomaly Detection")
    report.append("=" * 80)
    report.append("")

    # Initialize predictors
    predictor = NPSTrendPredictor(data_path=data_path)
    anomaly_detector = AnomalyDetector(data_path=data_path)
    cluster_analyzer = SubstanceClusterAnalyzer(data_path=data_path)

    # Forecast NPS trends
    report.append("1. NPS TREND FORECASTING (6 Periods Ahead)")
    report.append("-" * 80)

    for service_type in ['Fixed-site', 'Festival']:
        forecast = predictor.forecast_nps_trend(service_type, periods_ahead=6)

        if 'error' not in forecast:
            report.append(f"\n{service_type}:")
            report.append(f"  Current NPS Rate: {forecast['current_rate']:.1%}")
            report.append(f"  Predicted Trend: {forecast['trend'].upper()}")
            report.append(f"  6-Period Forecast: {forecast['predictions'][-1]:.1%}")
            if 'lower_bound' in forecast:
                report.append(f"  Confidence Interval: [{forecast['lower_bound'][-1]:.1%}, {forecast['upper_bound'][-1]:.1%}]")

    # Trend changes
    report.append("\n")
    report.append("2. TREND CHANGE DETECTION")
    report.append("-" * 80)

    trend_changes = predictor.detect_trend_changes()
    for service_type, data in trend_changes.items():
        report.append(f"\n{service_type}:")
        report.append(f"  Overall Trend: {data['overall_trend'].upper()}")
        report.append(f"  Volatility: {data['volatility']:.3f}")
        report.append(f"  Significant Changes Detected: {len(data['change_points'])}")

    # Anomaly detection
    report.append("\n")
    report.append("3. ANOMALY DETECTION")
    report.append("-" * 80)

    emerging = anomaly_detector.detect_emerging_substances(threshold_days=30)
    for service_type, data in emerging.items():
        report.append(f"\n{service_type}:")
        report.append(f"  Emerging Substances (last 30 days): {data['emerging_count']}")
        if data['substances']:
            for sub in data['substances'][:3]:
                report.append(f"    - {sub['substance']} (NPS: {sub['is_nps']})")

    # Clustering
    report.append("\n")
    report.append("4. SUBSTANCE CLUSTERING ANALYSIS")
    report.append("-" * 80)

    clusters = cluster_analyzer.cluster_by_detection_patterns(n_clusters=4)
    if 'error' not in clusters:
        for cluster_name, data in clusters.items():
            report.append(f"\n{cluster_name}:")
            report.append(f"  Substances: {data['count']}")
            report.append(f"  NPS %: {data['characteristics']['nps_percentage']:.1f}%")
            report.append(f"  Fixed-site %: {data['characteristics']['fixed_site_percentage']:.1f}%")

    report.append("\n")
    report.append("=" * 80)
    report.append("KEY ML INSIGHTS:")
    report.append("✓ Predictive models forecast future NPS detection trends")
    report.append("✓ Anomaly detection identifies emerging substance threats")
    report.append("✓ Clustering reveals distinct substance detection patterns")
    report.append("✓ Trend analysis highlights changes in drug market dynamics")
    report.append("=" * 80)

    return "\n".join(report)


if __name__ == "__main__":
    print("Machine Learning Predictive Models Module")
    print("Advanced analytics for drug checking data")
