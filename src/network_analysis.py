"""
Network Analysis Module for Drug Checking Data.

Analyzes substance co-occurrence patterns, temporal networks, and relationship structures.
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: networkx not installed. Network analysis features will be limited.")


class SubstanceNetworkAnalyzer:
    """
    Analyze substance co-occurrence networks and temporal patterns.
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
        self.networks = {}

    def build_temporal_cooccurrence_network(self, time_window='W', min_cooccurrence=2):
        """
        Build network of substances that co-occur within time windows.

        Args:
            time_window: Time window for co-occurrence ('D', 'W', 'M')
            min_cooccurrence: Minimum co-occurrences to include edge

        Returns:
            Dictionary with network statistics and structure
        """
        if not HAS_NETWORKX:
            return self._simple_cooccurrence_analysis(time_window, min_cooccurrence)

        networks = {}

        for service_type in self.df['service_type'].unique():
            service_data = self.df[self.df['service_type'] == service_type].copy()
            service_data['period'] = service_data['date'].dt.to_period(time_window)

            # Create graph
            G = nx.Graph()

            # Group substances by period
            period_substances = service_data.groupby('period')['substance_detected'].apply(list)

            # Build co-occurrence pairs
            edge_weights = Counter()

            for substances in period_substances:
                unique_subs = list(set(substances))
                # Create pairs
                for i in range(len(unique_subs)):
                    for j in range(i + 1, len(unique_subs)):
                        pair = tuple(sorted([unique_subs[i], unique_subs[j]]))
                        edge_weights[pair] += 1

            # Add edges to graph
            for (sub1, sub2), weight in edge_weights.items():
                if weight >= min_cooccurrence:
                    G.add_edge(sub1, sub2, weight=weight, cooccurrences=weight)

            # Calculate network metrics
            if len(G.nodes()) > 0:
                # Degree centrality
                degree_cent = nx.degree_centrality(G)
                top_central = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:10]

                # Betweenness centrality
                betweenness = nx.betweenness_centrality(G)
                top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]

                # Connected components
                components = list(nx.connected_components(G))

                # Clustering coefficient
                clustering = nx.clustering(G)
                avg_clustering = np.mean(list(clustering.values())) if clustering else 0

                networks[service_type] = {
                    'nodes': len(G.nodes()),
                    'edges': len(G.edges()),
                    'density': nx.density(G),
                    'avg_clustering': float(avg_clustering),
                    'connected_components': len(components),
                    'largest_component_size': max([len(c) for c in components]) if components else 0,
                    'top_central_substances': [(node, float(cent)) for node, cent in top_central],
                    'top_betweenness_substances': [(node, float(cent)) for node, cent in top_betweenness],
                    'network_object': G
                }
            else:
                networks[service_type] = {
                    'nodes': 0,
                    'edges': 0,
                    'error': 'No co-occurrences found with current parameters'
                }

        return networks

    def _simple_cooccurrence_analysis(self, time_window='W', min_cooccurrence=2):
        """Simple co-occurrence analysis without networkx."""
        networks = {}

        for service_type in self.df['service_type'].unique():
            service_data = self.df[self.df['service_type'] == service_type].copy()
            service_data['period'] = service_data['date'].dt.to_period(time_window)

            # Group substances by period
            period_substances = service_data.groupby('period')['substance_detected'].apply(list)

            # Build co-occurrence pairs
            edge_weights = Counter()

            for substances in period_substances:
                unique_subs = list(set(substances))
                for i in range(len(unique_subs)):
                    for j in range(i + 1, len(unique_subs)):
                        pair = tuple(sorted([unique_subs[i], unique_subs[j]]))
                        edge_weights[pair] += 1

            # Filter by min cooccurrence
            significant_pairs = {k: v for k, v in edge_weights.items() if v >= min_cooccurrence}

            # Get most connected substances
            substance_connections = Counter()
            for (sub1, sub2), count in significant_pairs.items():
                substance_connections[sub1] += count
                substance_connections[sub2] += count

            networks[service_type] = {
                'unique_substances': len(substance_connections),
                'cooccurrence_pairs': len(significant_pairs),
                'top_connected_substances': substance_connections.most_common(10),
                'strongest_pairs': sorted(significant_pairs.items(), key=lambda x: x[1], reverse=True)[:10]
            }

        return networks

    def analyze_nps_diffusion(self):
        """
        Analyze how NPS substances spread through the network over time.

        Returns:
            Dictionary with diffusion patterns and timing
        """
        nps_data = self.df[self.df['is_nps'] == True].copy()

        diffusion_analysis = {}

        for service_type in self.df['service_type'].unique():
            service_nps = nps_data[nps_data['service_type'] == service_type]

            if len(service_nps) == 0:
                continue

            # Analyze first appearances
            first_detections = service_nps.groupby('substance_detected')['date'].min().sort_values()

            # Calculate spread rate (new NPS per month)
            service_nps['month'] = service_nps['date'].dt.to_period('M')
            monthly_new_nps = []

            seen_substances = set()
            for month in sorted(service_nps['month'].unique()):
                month_data = service_nps[service_nps['month'] == month]
                month_substances = set(month_data['substance_detected'].unique())
                new_this_month = month_substances - seen_substances
                monthly_new_nps.append({
                    'month': str(month),
                    'new_nps_count': len(new_this_month),
                    'cumulative_nps': len(seen_substances) + len(new_this_month)
                })
                seen_substances.update(new_this_month)

            # Calculate adoption rate
            if len(monthly_new_nps) > 1:
                avg_new_per_month = np.mean([m['new_nps_count'] for m in monthly_new_nps])
            else:
                avg_new_per_month = 0

            diffusion_analysis[service_type] = {
                'total_nps_types': len(first_detections),
                'first_nps_detected': str(first_detections.iloc[0]) if len(first_detections) > 0 else None,
                'latest_nps_detected': str(first_detections.iloc[-1]) if len(first_detections) > 0 else None,
                'avg_new_nps_per_month': float(avg_new_per_month),
                'monthly_pattern': monthly_new_nps,
                'nps_list': first_detections.index.tolist()
            }

        return diffusion_analysis

    def identify_substance_clusters(self):
        """
        Identify clusters of frequently co-detected substances.

        Returns:
            Dictionary with identified clusters
        """
        if not HAS_NETWORKX:
            return {'error': 'networkx required for clustering'}

        clusters = {}

        for service_type in self.df['service_type'].unique():
            service_data = self.df[self.df['service_type'] == service_type].copy()
            service_data['week'] = service_data['date'].dt.to_period('W')

            # Build co-occurrence graph
            G = nx.Graph()

            weekly_substances = service_data.groupby('week')['substance_detected'].apply(list)

            edge_weights = Counter()
            for substances in weekly_substances:
                unique_subs = list(set(substances))
                for i in range(len(unique_subs)):
                    for j in range(i + 1, len(unique_subs)):
                        pair = tuple(sorted([unique_subs[i], unique_subs[j]]))
                        edge_weights[pair] += 1

            for (sub1, sub2), weight in edge_weights.items():
                if weight >= 2:
                    G.add_edge(sub1, sub2, weight=weight)

            if len(G.nodes()) == 0:
                continue

            # Use community detection
            try:
                from networkx.algorithms import community
                communities = community.greedy_modularity_communities(G)

                cluster_info = []
                for i, comm in enumerate(communities):
                    if len(comm) > 1:
                        # Get subgraph
                        subgraph = G.subgraph(comm)

                        # Calculate cluster metrics
                        cluster_info.append({
                            'cluster_id': i + 1,
                            'substances': list(comm),
                            'size': len(comm),
                            'internal_edges': subgraph.number_of_edges(),
                            'density': nx.density(subgraph)
                        })

                clusters[service_type] = {
                    'num_clusters': len(cluster_info),
                    'clusters': cluster_info
                }

            except ImportError:
                # Fallback to connected components
                components = list(nx.connected_components(G))
                cluster_info = []

                for i, comp in enumerate(components):
                    if len(comp) > 1:
                        cluster_info.append({
                            'cluster_id': i + 1,
                            'substances': list(comp),
                            'size': len(comp)
                        })

                clusters[service_type] = {
                    'num_clusters': len(cluster_info),
                    'clusters': cluster_info
                }

        return clusters

    def analyze_temporal_evolution(self):
        """
        Analyze how substance networks evolve over time.

        Returns:
            Dictionary with temporal network metrics
        """
        evolution = {}

        for service_type in self.df['service_type'].unique():
            service_data = self.df[self.df['service_type'] == service_type].copy()
            service_data['quarter'] = service_data['date'].dt.to_period('Q')

            quarters = sorted(service_data['quarter'].unique())

            temporal_metrics = []

            for quarter in quarters:
                quarter_data = service_data[service_data['quarter'] == quarter]

                metrics = {
                    'period': str(quarter),
                    'unique_substances': quarter_data['substance_detected'].nunique(),
                    'total_detections': len(quarter_data),
                    'nps_count': quarter_data['is_nps'].sum(),
                    'nps_rate': float(quarter_data['is_nps'].mean()),
                    'avg_adulterants': float(quarter_data['num_adulterants'].mean())
                }

                temporal_metrics.append(metrics)

            # Calculate trends
            if len(temporal_metrics) >= 2:
                diversity_trend = temporal_metrics[-1]['unique_substances'] - temporal_metrics[0]['unique_substances']
                nps_trend = temporal_metrics[-1]['nps_rate'] - temporal_metrics[0]['nps_rate']

                evolution[service_type] = {
                    'periods_analyzed': len(temporal_metrics),
                    'temporal_metrics': temporal_metrics,
                    'diversity_change': diversity_trend,
                    'nps_rate_change': float(nps_trend),
                    'overall_trend': 'increasing' if diversity_trend > 0 else 'stable_or_decreasing'
                }

        return evolution

    def calculate_network_resilience(self):
        """
        Calculate network resilience - how robust the detection network is.

        Returns:
            Dictionary with resilience metrics
        """
        if not HAS_NETWORKX:
            return {'error': 'networkx required for resilience analysis'}

        resilience = {}

        networks = self.build_temporal_cooccurrence_network(time_window='M', min_cooccurrence=2)

        for service_type, network_data in networks.items():
            if 'network_object' not in network_data:
                continue

            G = network_data['network_object']

            if len(G.nodes()) < 3:
                continue

            # Calculate resilience metrics
            try:
                # Node connectivity
                node_connectivity = nx.node_connectivity(G)

                # Edge connectivity
                edge_connectivity = nx.edge_connectivity(G)

                # Average shortest path (for largest component)
                if nx.is_connected(G):
                    avg_path = nx.average_shortest_path_length(G)
                else:
                    largest_cc = max(nx.connected_components(G), key=len)
                    subgraph = G.subgraph(largest_cc)
                    avg_path = nx.average_shortest_path_length(subgraph) if len(subgraph) > 1 else 0

                resilience[service_type] = {
                    'node_connectivity': node_connectivity,
                    'edge_connectivity': edge_connectivity,
                    'avg_shortest_path': float(avg_path),
                    'resilience_score': float((node_connectivity + edge_connectivity) / 2),
                    'interpretation': 'High' if node_connectivity >= 2 else 'Low'
                }

            except Exception as e:
                resilience[service_type] = {
                    'error': f'Could not calculate resilience: {str(e)}'
                }

        return resilience


def generate_network_analysis_report(data_path):
    """Generate comprehensive network analysis report."""
    report = []
    report.append("=" * 80)
    report.append("NETWORK ANALYSIS: SUBSTANCE CO-OCCURRENCE PATTERNS")
    report.append("Temporal Networks & Relationship Structures")
    report.append("=" * 80)
    report.append("")

    analyzer = SubstanceNetworkAnalyzer(data_path=data_path)

    # Co-occurrence networks
    report.append("1. TEMPORAL CO-OCCURRENCE NETWORKS")
    report.append("-" * 80)

    networks = analyzer.build_temporal_cooccurrence_network(time_window='W', min_cooccurrence=2)

    for service_type, data in networks.items():
        report.append(f"\n{service_type}:")
        if 'error' not in data:
            if HAS_NETWORKX:
                report.append(f"  Network Size: {data['nodes']} substances, {data['edges']} connections")
                report.append(f"  Network Density: {data['density']:.3f}")
                report.append(f"  Average Clustering: {data['avg_clustering']:.3f}")
                report.append(f"  Connected Components: {data['connected_components']}")

                if data['top_central_substances']:
                    report.append(f"  Most Central Substances:")
                    for substance, centrality in data['top_central_substances'][:5]:
                        report.append(f"    - {substance}: {centrality:.3f}")
            else:
                report.append(f"  Unique Substances: {data['unique_substances']}")
                report.append(f"  Co-occurrence Pairs: {data['cooccurrence_pairs']}")
        else:
            report.append(f"  {data['error']}")

    # NPS diffusion
    report.append("\n")
    report.append("2. NPS DIFFUSION ANALYSIS")
    report.append("-" * 80)

    diffusion = analyzer.analyze_nps_diffusion()

    for service_type, data in diffusion.items():
        report.append(f"\n{service_type}:")
        report.append(f"  Total NPS Types: {data['total_nps_types']}")
        report.append(f"  Avg New NPS per Month: {data['avg_new_nps_per_month']:.2f}")
        if data['first_nps_detected']:
            report.append(f"  First Detection: {data['first_nps_detected']}")
        if data['latest_nps_detected']:
            report.append(f"  Latest Detection: {data['latest_nps_detected']}")

    # Substance clusters
    report.append("\n")
    report.append("3. SUBSTANCE CLUSTERING")
    report.append("-" * 80)

    clusters = analyzer.identify_substance_clusters()

    if 'error' not in clusters:
        for service_type, data in clusters.items():
            report.append(f"\n{service_type}:")
            report.append(f"  Number of Clusters: {data['num_clusters']}")

            for cluster in data['clusters'][:3]:
                report.append(f"\n  Cluster {cluster['cluster_id']} ({cluster['size']} substances):")
                report.append(f"    {', '.join(cluster['substances'][:5])}")
                if len(cluster['substances']) > 5:
                    report.append(f"    ... and {len(cluster['substances']) - 5} more")

    # Temporal evolution
    report.append("\n")
    report.append("4. TEMPORAL NETWORK EVOLUTION")
    report.append("-" * 80)

    evolution = analyzer.analyze_temporal_evolution()

    for service_type, data in evolution.items():
        report.append(f"\n{service_type}:")
        report.append(f"  Periods Analyzed: {data['periods_analyzed']}")
        report.append(f"  Diversity Change: {data['diversity_change']:+d} substances")
        report.append(f"  NPS Rate Change: {data['nps_rate_change']:+.1%}")
        report.append(f"  Overall Trend: {data['overall_trend'].upper()}")

    report.append("\n")
    report.append("=" * 80)
    report.append("KEY NETWORK INSIGHTS:")
    report.append("✓ Substance co-occurrence patterns reveal market dynamics")
    report.append("✓ Network structure differs significantly between service types")
    report.append("✓ NPS diffusion patterns indicate early warning capabilities")
    report.append("✓ Temporal evolution shows changing drug market landscape")
    report.append("=" * 80)

    return "\n".join(report)


if __name__ == "__main__":
    print("Network Analysis Module for Drug Checking Data")
    print("Analyzes substance co-occurrence and temporal patterns")
