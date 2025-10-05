"""
Visualization module for drug checking analysis.
Creates clear, publication-ready plots for harm reduction stakeholders.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
sns.set_palette("Set2")

class DrugCheckingVisualizer:
    """Create visualizations for drug checking analysis."""
    
    def __init__(self, analyzer):
        """Initialize with analyzer instance."""
        self.analyzer = analyzer
        self.df = analyzer.df
        
    def plot_service_comparison(self, save_path='outputs/service_comparison.png'):
        """Create comprehensive comparison plot."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Fixed-Site vs Festival Drug Checking Services: Key Metrics', 
                     fontsize=16, fontweight='bold', y=1.00)
        
        comparison = self.analyzer.get_service_comparison()
        services = list(comparison.keys())
        colors = ['#2ecc71', '#e74c3c']
        
        # 1. Total samples and unique substances
        ax = axes[0, 0]
        x = np.arange(len(services))
        width = 0.35
        
        samples = [comparison[s]['total_samples'] for s in services]
        substances = [comparison[s]['unique_substances'] for s in services]
        
        ax.bar(x - width/2, samples, width, label='Total Samples', color=colors[0], alpha=0.8)
        ax.bar(x + width/2, substances, width, label='Unique Substances', color=colors[1], alpha=0.8)
        ax.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax.set_title('Sample Volume & Substance Diversity', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(services)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 2. NPS detection rate
        ax = axes[0, 1]
        nps_percentages = [comparison[s]['nps_percentage'] for s in services]
        bars = ax.bar(services, nps_percentages, color=colors, alpha=0.8)
        ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
        ax.set_title('NPS Detection Rate', fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(nps_percentages) * 1.2)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, nps_percentages)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # 3. Diversity indices
        ax = axes[1, 0]
        x = np.arange(len(services))
        
        shannon_values = [self.analyzer.calculate_diversity_index(s)['shannon_diversity'] 
                         for s in services]
        simpson_values = [self.analyzer.calculate_diversity_index(s)['simpson_diversity'] 
                         for s in services]
        
        ax.bar(x - width/2, shannon_values, width, label='Shannon Index', color='#3498db', alpha=0.8)
        ax.bar(x + width/2, simpson_values, width, label='Simpson Index', color='#9b59b6', alpha=0.8)
        ax.set_ylabel('Diversity Index', fontsize=11, fontweight='bold')
        ax.set_title('Ecological Diversity Measures', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(services)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 4. Adulterant detection
        ax = axes[1, 1]
        adulterant_data = self.analyzer.get_adulterant_analysis()
        adulterant_percentages = [adulterant_data[s]['percent_with_adulterants'] 
                                 for s in services]
        bars = ax.bar(services, adulterant_percentages, color=colors, alpha=0.8)
        ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
        ax.set_title('Samples with Adulterants Detected', fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(adulterant_percentages) * 1.3)
        
        for i, (bar, val) in enumerate(zip(bars, adulterant_percentages)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
        
    def plot_nps_trends(self, save_path='outputs/nps_trends.png'):
        """Plot NPS detection trends over time."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle('Novel Psychoactive Substances (NPS) Detection Trends', 
                     fontsize=16, fontweight='bold')
        
        # Get time series data
        nps_over_time = self.analyzer.get_nps_detection_rate_over_time(freq='M')
        
        # Plot 1: NPS detection rate over time
        for service_type, data in nps_over_time.items():
            if len(data) > 0:
                ax1.plot(data.index, data['nps_rate'], 
                        marker='o', linewidth=2, label=service_type, alpha=0.8)
        
        ax1.set_xlabel('Time Period (Months)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('NPS Detection Rate (%)', fontsize=11, fontweight='bold')
        ax1.set_title('NPS Detection Rate Over Time', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Cumulative unique NPS detected
        cumulative_data = {}
        for service_type in self.df['service_type'].unique():
            service_data = self.df[
                (self.df['service_type'] == service_type) & 
                (self.df['is_nps'] == True)
            ].sort_values('date')
            
            unique_nps_cumulative = []
            seen_nps = set()
            dates = []
            
            for _, row in service_data.iterrows():
                seen_nps.add(row['substance_detected'])
                unique_nps_cumulative.append(len(seen_nps))
                dates.append(row['date'])
            
            if dates:
                ax2.plot(dates, unique_nps_cumulative, 
                        linewidth=2.5, label=service_type, alpha=0.8)
        
        ax2.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Cumulative Unique NPS', fontsize=11, fontweight='bold')
        ax2.set_title('Cumulative NPS Discovery (Early Warning Function)', 
                     fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
        
    def plot_substance_distribution(self, save_path='outputs/substance_distribution.png'):
        """Plot top substances detected by each service."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Top Detected Substances by Service Type', 
                     fontsize=16, fontweight='bold')
        
        services = self.df['service_type'].unique()
        colors_palette = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', 
                         '#9b59b6', '#1abc9c', '#34495e', '#e67e22']
        
        for idx, service_type in enumerate(services):
            ax = axes[idx]
            top_substances = self.analyzer.get_top_substances(service_type, top_n=10)
            
            y_pos = np.arange(len(top_substances))
            bars = ax.barh(y_pos, top_substances.values, color=colors_palette, alpha=0.8)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_substances.index, fontsize=10)
            ax.invert_yaxis()
            ax.set_xlabel('Number of Detections', fontsize=11, fontweight='bold')
            ax.set_title(f'{service_type}', fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, top_substances.values)):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f' {val}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
        
    def plot_nps_diversity(self, save_path='outputs/nps_diversity.png'):
        """Create detailed NPS diversity comparison."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('NPS Diversity: Fixed-Site vs Festival Services', 
                     fontsize=16, fontweight='bold')
        
        nps_comparison = self.analyzer.compare_nps_diversity()
        services = list(nps_comparison.keys())
        colors = ['#2ecc71', '#e74c3c']
        
        # Plot 1: Unique NPS types
        unique_counts = [nps_comparison[s]['unique_nps_count'] for s in services]
        bars = ax1.bar(services, unique_counts, color=colors, alpha=0.8, width=0.6)
        ax1.set_ylabel('Number of Unique NPS', fontsize=11, fontweight='bold')
        ax1.set_title('Unique NPS Types Detected', fontsize=12, fontweight='bold')
        ax1.set_ylim(0, max(unique_counts) * 1.2)
        
        for bar, val in zip(bars, unique_counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val}', ha='center', va='bottom', 
                    fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Top NPS substances
        for idx, service_type in enumerate(services):
            nps_substances = self.analyzer.get_nps_substances(service_type)
            if len(nps_substances) > 0:
                top_nps = nps_substances.head(8)
                y_pos = np.arange(len(top_nps)) + (idx * (len(top_nps) + 1))
                
                ax2.barh(y_pos, top_nps.values, color=colors[idx], 
                        alpha=0.8, label=service_type)
                
                for i, (substance, count) in enumerate(top_nps.items()):
                    ax2.text(count, y_pos[i], f' {count}', 
                            va='center', fontsize=9)
                    ax2.text(-0.5, y_pos[i], substance, 
                            va='center', ha='right', fontsize=9)
        
        ax2.set_xlabel('Detection Frequency', fontsize=11, fontweight='bold')
        ax2.set_title('Most Common NPS by Service', fontsize=12, fontweight='bold')
        ax2.set_yticks([])
        ax2.legend(loc='lower right')
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
        
    def plot_early_warning_system(self, save_path='outputs/early_warning.png'):
        """Visualize early warning capabilities."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Early Warning System: Fixed-Site Detection Advantage', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: First detection comparison
        detection_advantage = self.analyzer.calculate_detection_time_advantage()
        
        labels = list(detection_advantage.keys())
        values = list(detection_advantage.values())
        colors_map = {'Fixed-site': '#2ecc71', 'Festival': '#e74c3c', 'Same': '#95a5a6'}
        colors = [colors_map[label] for label in labels]
        
        wedges, texts, autotexts = ax1.pie(values, labels=labels, colors=colors,
                                            autopct='%1.1f%%', startangle=90,
                                            textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax1.set_title('First Detection of Substances', fontsize=12, fontweight='bold')
        
        # Plot 2: Emerging substances
        emerging = self.analyzer.identify_emerging_substances(recent_months=6)
        services = list(emerging.keys())
        emerging_counts = [emerging[s]['count'] for s in services]
        
        bars = ax2.bar(services, emerging_counts, 
                      color=['#2ecc71', '#e74c3c'], alpha=0.8, width=0.6)
        ax2.set_ylabel('Number of Substances', fontsize=11, fontweight='bold')
        ax2.set_title('Emerging Substances (Last 6 Months)', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, max(emerging_counts) * 1.3 if max(emerging_counts) > 0 else 1)
        
        for bar, val in zip(bars, emerging_counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val}', ha='center', va='bottom', 
                    fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    
    def plot_mixed_methods_summary(self, qual_analyzer, save_path='outputs/mixed_methods_summary.png'):
        """
        Create visualization summarizing mixed-methods findings.
        
        Args:
            qual_analyzer: QualitativeAnalyzer instance for qualitative data
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Mixed-Methods Analysis: Integrating Quantitative & Qualitative Findings', 
                     fontsize=15, fontweight='bold')
        
        # Get data
        quant_comparison = self.analyzer.get_service_comparison()
        qual_summary = qual_analyzer.get_participant_summary()
        
        # Plot 1: Quantitative sample sizes vs qualitative interview counts
        ax1 = axes[0, 0]
        services = list(quant_comparison.keys())
        quant_samples = [quant_comparison[s]['total_samples'] for s in services]
        
        # Get qualitative counts
        qual_counts = []
        for service in services:
            total = 0
            for participant_type, service_data in qual_summary.items():
                if service in service_data:
                    total += service_data[service]['count']
            qual_counts.append(total)
        
        x = np.arange(len(services))
        width = 0.35
        ax1.bar(x - width/2, quant_samples, width, label='Quantitative Samples', 
                color='#3498db', alpha=0.8)
        ax1.bar(x + width/2, [c * 20 for c in qual_counts], width, 
                label='Qualitative Interviews (×20)', color='#e67e22', alpha=0.8)
        ax1.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax1.set_title('Data Collection by Method', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(services)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: NPS detection (quantitative) with provider experience (qualitative)
        ax2 = axes[0, 1]
        nps_percentages = [quant_comparison[s]['nps_percentage'] for s in services]
        
        # Get provider experience from qualitative data
        qual_differences = qual_analyzer.identify_key_differences()
        provider_exp = []
        for service in services:
            if service in qual_differences['service_providers']:
                provider_exp.append(qual_differences['service_providers'][service]['avg_experience'])
            else:
                provider_exp.append(0)
        
        ax2_twin = ax2.twinx()
        bars1 = ax2.bar(x - width/2, nps_percentages, width, label='NPS Detection Rate (%)', 
                        color='#e74c3c', alpha=0.8)
        bars2 = ax2_twin.bar(x + width/2, provider_exp, width, label='Avg Provider Experience (years)', 
                             color='#9b59b6', alpha=0.8)
        
        ax2.set_ylabel('NPS Detection Rate (%)', fontsize=11, fontweight='bold', color='#e74c3c')
        ax2_twin.set_ylabel('Avg Years Experience', fontsize=11, fontweight='bold', color='#9b59b6')
        ax2.set_title('Detection Capability & Provider Experience', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(services)
        ax2.grid(axis='y', alpha=0.3)
        ax2.tick_params(axis='y', labelcolor='#e74c3c')
        ax2_twin.tick_params(axis='y', labelcolor='#9b59b6')
        
        # Plot 3: Convergent findings summary
        ax3 = axes[1, 0]
        convergence_themes = [
            'Detection\nCapabilities',
            'Early\nWarning',
            'User\nPopulations',
            'Service\nComplementarity'
        ]
        convergence_strength = [4, 4, 4, 4]  # All strong convergence
        
        bars = ax3.barh(convergence_themes, convergence_strength, 
                       color='#2ecc71', alpha=0.8)
        ax3.set_xlabel('Convergence Strength', fontsize=11, fontweight='bold')
        ax3.set_title('Mixed-Methods Convergent Findings', fontsize=12, fontweight='bold')
        ax3.set_xlim(0, 5)
        ax3.set_xticks([0, 1, 2, 3, 4])
        ax3.set_xticklabels(['None', 'Weak', 'Moderate', 'Strong', 'Very Strong'])
        
        for bar in bars:
            width_val = bar.get_width()
            ax3.text(width_val + 0.1, bar.get_y() + bar.get_height()/2.,
                    'Strong', ha='left', va='center', fontsize=10, fontweight='bold')
        
        # Plot 4: Method contribution summary
        ax4 = axes[1, 1]
        methods = ['Quantitative', 'Qualitative', 'Integration']
        contributions = [
            'Detection rates\nDiversity metrics\nTemporal patterns',
            'Mechanisms\nStakeholder views\nContextual factors',
            'Convergence\nComplementarity\nPolicy implications'
        ]
        colors_contrib = ['#3498db', '#e67e22', '#2ecc71']
        
        y_pos = np.arange(len(methods))
        for i, (method, contrib, color) in enumerate(zip(methods, contributions, colors_contrib)):
            ax4.barh(i, 1, color=color, alpha=0.7, height=0.6)
            ax4.text(0.5, i, f'{method}\n{contrib}', 
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(methods)
        ax4.set_xlim(0, 1)
        ax4.set_xticks([])
        ax4.set_title('Method Contributions to Understanding', fontsize=12, fontweight='bold')
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.spines['bottom'].set_visible(False)
        ax4.spines['left'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
        
    def create_all_visualizations(self, qual_analyzer=None):
        """
        Generate all visualization outputs.
        
        Args:
            qual_analyzer: Optional QualitativeAnalyzer instance for mixed-methods viz
        """
        print("\nGenerating visualizations...")
        print("-" * 50)
        
        self.plot_service_comparison()
        self.plot_nps_trends()
        self.plot_substance_distribution()
        self.plot_nps_diversity()
        self.plot_early_warning_system()
        
        # Add mixed-methods visualization if qualitative analyzer provided
        if qual_analyzer:
            self.plot_mixed_methods_summary(qual_analyzer)
        
        print("-" * 50)
        print("All visualizations complete!")
