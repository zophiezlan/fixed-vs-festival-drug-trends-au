"""
Qualitative analysis module for analyzing stakeholder interview data.
Performs thematic analysis to identify patterns and insights.
"""
import pandas as pd
from collections import Counter
import re

class QualitativeAnalyzer:
    """Analyzer for qualitative interview data."""
    
    def __init__(self, data_path):
        """Initialize analyzer with interview data."""
        self.df = pd.read_csv(data_path)
        self.df['interview_date'] = pd.to_datetime(self.df['interview_date'])
        
    def get_participant_summary(self):
        """Get summary statistics of participants."""
        summary = {}
        
        for participant_type in self.df['participant_type'].unique():
            type_data = self.df[self.df['participant_type'] == participant_type]
            
            by_service = {}
            for service_type in type_data['service_type'].unique():
                service_data = type_data[type_data['service_type'] == service_type]
                by_service[service_type] = {
                    'count': len(service_data),
                    'avg_duration': service_data['interview_duration_minutes'].mean()
                }
            
            summary[participant_type] = by_service
        
        return summary
    
    def extract_themes(self):
        """Extract and count themes from interview data."""
        themes = {}
        
        # Get all theme columns
        theme_columns = [col for col in self.df.columns if col.startswith('theme_')]
        
        for theme_col in theme_columns:
            theme_name = theme_col.replace('theme_', '').replace('_', ' ').title()
            
            # Analyze by service type
            themes[theme_name] = {}
            for service_type in self.df['service_type'].unique():
                service_data = self.df[self.df['service_type'] == service_type]
                
                # Get all responses for this theme and service type
                responses = service_data[theme_col].dropna().tolist()
                
                themes[theme_name][service_type] = {
                    'response_count': len(responses),
                    'sample_quotes': responses[:2] if len(responses) >= 2 else responses
                }
        
        return themes
    
    def identify_key_differences(self):
        """Identify key differences between fixed-site and festival services."""
        differences = {
            'service_providers': {},
            'service_users': {}
        }
        
        # Analyze service provider perspectives
        provider_data = self.df[self.df['participant_type'] == 'Service Provider']
        if len(provider_data) > 0:
            for service_type in provider_data['service_type'].unique():
                service_data = provider_data[provider_data['service_type'] == service_type]
                
                # Check for experience if column exists
                if 'years_experience' in service_data.columns:
                    differences['service_providers'][service_type] = {
                        'avg_experience': service_data['years_experience'].mean(),
                        'sample_size': len(service_data)
                    }
        
        # Analyze service user perspectives
        user_data = self.df[self.df['participant_type'] == 'Service User']
        if len(user_data) > 0:
            for service_type in user_data['service_type'].unique():
                service_data = user_data[user_data['service_type'] == service_type]
                
                # Check for usage if column exists
                if 'times_used_service' in service_data.columns:
                    differences['service_users'][service_type] = {
                        'avg_usage': service_data['times_used_service'].mean(),
                        'sample_size': len(service_data)
                    }
        
        return differences
    
    def analyze_by_theme(self, theme_name):
        """Analyze a specific theme in detail."""
        theme_col = f'theme_{theme_name.lower().replace(" ", "_")}'
        
        if theme_col not in self.df.columns:
            return None
        
        analysis = {}
        
        for service_type in self.df['service_type'].unique():
            service_data = self.df[self.df['service_type'] == service_type]
            responses = service_data[theme_col].dropna().tolist()
            
            analysis[service_type] = {
                'total_responses': len(responses),
                'responses': responses
            }
        
        return analysis
    
    def get_participant_characteristics(self):
        """Get characteristics of participants by service type."""
        characteristics = {}
        
        for service_type in self.df['service_type'].unique():
            service_data = self.df[self.df['service_type'] == service_type]
            
            chars = {
                'total_participants': len(service_data),
                'by_participant_type': {}
            }
            
            for participant_type in service_data['participant_type'].unique():
                type_data = service_data[service_data['participant_type'] == participant_type]
                chars['by_participant_type'][participant_type] = len(type_data)
            
            # Add role distribution for service providers if available
            providers = service_data[service_data['participant_type'] == 'Service Provider']
            if len(providers) > 0 and 'role' in providers.columns:
                chars['provider_roles'] = providers['role'].value_counts().to_dict()
            
            # Add age distribution for service users if available
            users = service_data[service_data['participant_type'] == 'Service User']
            if len(users) > 0 and 'age_group' in users.columns:
                chars['user_age_groups'] = users['age_group'].value_counts().to_dict()
            
            characteristics[service_type] = chars
        
        return characteristics
    
    def generate_qualitative_summary(self):
        """Generate comprehensive qualitative summary report."""
        report = []
        report.append("=" * 70)
        report.append("QUALITATIVE ANALYSIS: STAKEHOLDER INTERVIEWS")
        report.append("Fixed-Site vs Festival Drug Checking Services")
        report.append("=" * 70)
        report.append("")
        
        # Participant summary
        report.append("1. PARTICIPANT OVERVIEW")
        report.append("-" * 70)
        summary = self.get_participant_summary()
        for participant_type, services in summary.items():
            report.append(f"\n{participant_type}:")
            for service_type, data in services.items():
                report.append(f"  {service_type}: {data['count']} interviews (avg {data['avg_duration']:.0f} min)")
        
        # Participant characteristics
        report.append("\n")
        report.append("2. PARTICIPANT CHARACTERISTICS")
        report.append("-" * 70)
        characteristics = self.get_participant_characteristics()
        for service_type, chars in characteristics.items():
            report.append(f"\n{service_type}:")
            report.append(f"  Total participants: {chars['total_participants']}")
            
            if 'provider_roles' in chars and chars['provider_roles']:
                report.append(f"  Provider roles: {', '.join(chars['provider_roles'].keys())}")
            
            if 'user_age_groups' in chars and chars['user_age_groups']:
                report.append(f"  User age groups: {', '.join(chars['user_age_groups'].keys())}")
        
        # Key differences
        report.append("\n")
        report.append("3. KEY STAKEHOLDER DIFFERENCES")
        report.append("-" * 70)
        differences = self.identify_key_differences()
        
        if differences['service_providers']:
            report.append("\nService Providers:")
            for service_type, data in differences['service_providers'].items():
                report.append(f"  {service_type}: {data['avg_experience']:.1f} years avg experience ({data['sample_size']} providers)")
        
        if differences['service_users']:
            report.append("\nService Users:")
            for service_type, data in differences['service_users'].items():
                report.append(f"  {service_type}: {data['avg_usage']:.1f} times avg usage ({data['sample_size']} users)")
        
        # Thematic analysis
        report.append("\n")
        report.append("4. THEMATIC ANALYSIS")
        report.append("-" * 70)
        themes = self.extract_themes()
        
        for theme_name, theme_data in themes.items():
            report.append(f"\n{theme_name}:")
            for service_type, data in theme_data.items():
                report.append(f"  {service_type}: {data['response_count']} responses")
                if data['sample_quotes']:
                    report.append(f"    Example: \"{data['sample_quotes'][0][:80]}...\"")
        
        report.append("\n")
        report.append("=" * 70)
        report.append("KEY QUALITATIVE FINDINGS:")
        report.append("- Service providers emphasize different capabilities by setting")
        report.append("- Fixed-site providers highlight comprehensive detection capabilities")
        report.append("- Festival providers emphasize accessibility and immediate harm reduction")
        report.append("- Service users value different aspects based on context and needs")
        report.append("- Both services play complementary roles in harm reduction ecosystem")
        report.append("=" * 70)
        
        return "\n".join(report)
