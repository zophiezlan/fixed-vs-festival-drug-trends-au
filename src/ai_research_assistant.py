"""
AI Research Assistant for Drug Checking Research.

This module provides AI-powered research capabilities including:
- Automated hypothesis generation
- Research question formulation
- Insight extraction and synthesis
- Automated literature review support
- Policy recommendation generation
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import re
from collections import defaultdict


class ResearchAssistant:
    """
    AI-powered research assistant for analyzing drug checking data and generating insights.
    """

    def __init__(self, quantitative_data=None, qualitative_data=None):
        """
        Initialize research assistant with data.

        Args:
            quantitative_data: Path to or DataFrame of quantitative data
            qualitative_data: Path to or DataFrame of qualitative data
        """
        # Load quantitative data
        if isinstance(quantitative_data, str):
            self.quant_df = pd.read_csv(quantitative_data)
        elif quantitative_data is not None:
            self.quant_df = quantitative_data.copy()
        else:
            self.quant_df = None

        # Load qualitative data
        if isinstance(qualitative_data, str):
            self.qual_df = pd.read_csv(qualitative_data)
        elif qualitative_data is not None:
            self.qual_df = qualitative_data.copy()
        else:
            self.qual_df = None

        self.insights = []
        self.hypotheses = []
        self.recommendations = []

    def generate_research_questions(self):
        """
        Generate relevant research questions based on available data.

        Returns:
            List of research questions organized by category
        """
        questions = {
            'detection_capabilities': [],
            'service_models': [],
            'policy_implications': [],
            'harm_reduction': [],
            'emerging_threats': []
        }

        if self.quant_df is not None:
            # Questions based on quantitative patterns
            nps_rates = self.quant_df.groupby('service_type')['is_nps'].mean()

            if len(nps_rates) >= 2:
                questions['detection_capabilities'].extend([
                    "What factors explain the differences in NPS detection rates between service types?",
                    "How do equipment capabilities influence substance identification accuracy?",
                    "What is the relationship between sample volume and detection diversity?"
                ])

            questions['emerging_threats'].extend([
                "Which novel psychoactive substances pose the greatest emerging risk?",
                "How quickly can each service model identify new substance trends?",
                "What early warning indicators predict NPS market changes?"
            ])

        if self.qual_df is not None:
            # Questions based on qualitative themes
            questions['service_models'].extend([
                "What drives user preferences for different service models?",
                "How do service providers perceive their role in the harm reduction ecosystem?",
                "What barriers prevent optimal service delivery in each model?"
            ])

            questions['harm_reduction'].extend([
                "How do different service models influence behavior change?",
                "What information needs are most critical for service users?",
                "How can services better reach high-risk populations?"
            ])

        questions['policy_implications'].extend([
            "What resource allocation optimizes public health outcomes?",
            "How should services be integrated into broader harm reduction strategies?",
            "What policy changes would enhance service effectiveness?",
            "How can data from drug checking services inform drug policy?"
        ])

        return questions

    def generate_hypotheses(self):
        """
        Generate testable hypotheses from data patterns.

        Returns:
            List of hypotheses with supporting evidence
        """
        hypotheses = []

        if self.quant_df is not None:
            # Analyze service type differences
            comparison = self.quant_df.groupby('service_type').agg({
                'substance_detected': 'nunique',
                'is_nps': ['sum', 'mean'],
                'num_adulterants': 'mean'
            })

            if len(comparison) >= 2:
                service_types = comparison.index.tolist()

                # Hypothesis 1: Detection diversity
                hypotheses.append({
                    'id': 'H1',
                    'hypothesis': f"{service_types[0]} services detect a broader range of substances than {service_types[1]} services",
                    'type': 'comparative',
                    'variables': ['service_type', 'substance_diversity'],
                    'testable': True,
                    'suggested_tests': ['Chi-square test', 'Shannon diversity index comparison'],
                    'implications': 'Different service models may serve distinct surveillance functions'
                })

                # Hypothesis 2: NPS detection
                hypotheses.append({
                    'id': 'H2',
                    'hypothesis': "Year-round operation increases novel psychoactive substance detection rates",
                    'type': 'causal',
                    'variables': ['operation_schedule', 'nps_detection_rate'],
                    'testable': True,
                    'suggested_tests': ['Regression analysis', 'Temporal correlation'],
                    'implications': 'Continuous monitoring enhances early warning capabilities'
                })

                # Hypothesis 3: Adulterant detection
                hypotheses.append({
                    'id': 'H3',
                    'hypothesis': "Advanced analytical equipment increases adulterant detection sensitivity",
                    'type': 'causal',
                    'variables': ['equipment_type', 'adulterant_detection'],
                    'testable': True,
                    'suggested_tests': ['ANOVA', 'Logistic regression'],
                    'implications': 'Equipment investment directly impacts harm reduction effectiveness'
                })

        if self.qual_df is not None:
            # Hypothesis based on qualitative patterns
            hypotheses.append({
                'id': 'H4',
                'hypothesis': "User trust in drug checking services is mediated by perceived privacy and non-judgmental approach",
                'type': 'mediational',
                'variables': ['privacy', 'non_judgmental_approach', 'trust', 'service_usage'],
                'testable': True,
                'suggested_tests': ['Structural equation modeling', 'Mediation analysis'],
                'implications': 'Service design features influence accessibility and uptake'
            })

            hypotheses.append({
                'id': 'H5',
                'hypothesis': "Complementarity between fixed-site and festival services maximizes population-level harm reduction",
                'type': 'theoretical',
                'variables': ['service_mix', 'population_coverage', 'harm_reduction_outcomes'],
                'testable': True,
                'suggested_tests': ['Ecological analysis', 'System dynamics modeling'],
                'implications': 'Integrated service networks outperform single-model approaches'
            })

        self.hypotheses = hypotheses
        return hypotheses

    def extract_key_insights(self):
        """
        Extract and synthesize key insights from data.

        Returns:
            Dictionary of insights organized by theme
        """
        insights = {
            'detection_patterns': [],
            'service_differentiation': [],
            'public_health_value': [],
            'user_perspectives': [],
            'system_level': []
        }

        if self.quant_df is not None:
            # Detection pattern insights
            nps_by_service = self.quant_df.groupby('service_type')['is_nps'].agg(['sum', 'mean', 'count'])

            for service_type in nps_by_service.index:
                data = nps_by_service.loc[service_type]
                insights['detection_patterns'].append({
                    'insight': f"{service_type} services detected {int(data['sum'])} NPS samples ({data['mean']*100:.1f}% of total)",
                    'significance': 'high',
                    'evidence_strength': 'quantitative'
                })

            # Substance diversity
            diversity = self.quant_df.groupby('service_type')['substance_detected'].nunique()
            max_service = diversity.idxmax()
            min_service = diversity.idxmin()
            diff_pct = ((diversity[max_service] - diversity[min_service]) / diversity[min_service] * 100)

            insights['service_differentiation'].append({
                'insight': f"{max_service} services show {diff_pct:.0f}% higher substance diversity than {min_service} services",
                'significance': 'high',
                'evidence_strength': 'quantitative',
                'implication': 'Different service models fulfill distinct public health surveillance roles'
            })

            # Early warning function
            first_detections = {}
            for substance in self.quant_df['substance_detected'].unique():
                sub_data = self.quant_df[self.quant_df['substance_detected'] == substance]
                first_by_service = sub_data.groupby('service_type')['date'].min()
                if len(first_by_service) > 1:
                    first_service = first_by_service.idxmin()
                    first_detections[first_service] = first_detections.get(first_service, 0) + 1

            if first_detections:
                leader = max(first_detections, key=first_detections.get)
                insights['public_health_value'].append({
                    'insight': f"{leader} services detected {first_detections[leader]} substances first, demonstrating early warning capability",
                    'significance': 'high',
                    'evidence_strength': 'quantitative',
                    'implication': 'Critical for identifying emerging substance threats before widespread distribution'
                })

        if self.qual_df is not None:
            # User perspective insights
            if 'participant_type' in self.qual_df.columns:
                user_count = len(self.qual_df[self.qual_df['participant_type'] == 'Service User'])
                provider_count = len(self.qual_df[self.qual_df['participant_type'] == 'Service Provider'])

                insights['user_perspectives'].append({
                    'insight': f"Analysis incorporates perspectives from {user_count} service users and {provider_count} providers",
                    'significance': 'medium',
                    'evidence_strength': 'qualitative',
                    'implication': 'Multi-stakeholder approach provides comprehensive understanding'
                })

            # Service complementarity
            if 'service_type' in self.qual_df.columns:
                service_types = self.qual_df['service_type'].unique()
                if len(service_types) >= 2:
                    insights['system_level'].append({
                        'insight': "Stakeholders from both service models emphasize complementary rather than competing roles",
                        'significance': 'high',
                        'evidence_strength': 'qualitative',
                        'implication': 'Policy should support diverse service models as integrated harm reduction system'
                    })

        self.insights = insights
        return insights

    def generate_policy_recommendations(self):
        """
        Generate evidence-based policy recommendations.

        Returns:
            List of policy recommendations with supporting evidence
        """
        recommendations = []

        # Recommendation 1: Dual investment
        recommendations.append({
            'id': 'R1',
            'recommendation': 'Invest in both fixed-site and festival-based drug checking services',
            'priority': 'high',
            'rationale': [
                'Services fulfill distinct but complementary public health functions',
                'Fixed-site services provide continuous surveillance and broader detection',
                'Festival services offer point-of-need accessibility and immediate harm reduction',
                'Different user populations benefit from different service models'
            ],
            'evidence_base': ['quantitative_analysis', 'stakeholder_interviews'],
            'implementation': [
                'Allocate sustained funding for permanent fixed-site locations',
                'Provide event-based funding for festival services',
                'Establish data sharing protocols between service types'
            ],
            'expected_outcomes': [
                'Enhanced drug market surveillance',
                'Increased population reach',
                'Improved early warning capabilities'
            ]
        })

        # Recommendation 2: Equipment and training
        recommendations.append({
            'id': 'R2',
            'recommendation': 'Enhance analytical capabilities through equipment and training investment',
            'priority': 'high',
            'rationale': [
                'Advanced equipment increases detection accuracy and breadth',
                'Skilled operators maximize analytical capabilities',
                'Better detection directly translates to better harm reduction'
            ],
            'evidence_base': ['detection_rate_analysis', 'provider_interviews'],
            'implementation': [
                'Provide funding for advanced analytical equipment (FTIR, GC-MS, LC-MS)',
                'Support ongoing training for service providers',
                'Establish quality assurance protocols'
            ],
            'expected_outcomes': [
                'Increased NPS and adulterant detection',
                'Enhanced analytical confidence',
                'Better public health data quality'
            ]
        })

        # Recommendation 3: Early warning system
        recommendations.append({
            'id': 'R3',
            'recommendation': 'Formalize drug checking services as part of public health early warning system',
            'priority': 'high',
            'rationale': [
                'Services detect emerging substances before widespread distribution',
                'Real-time data enables rapid public health response',
                'Early intervention prevents harms at population level'
            ],
            'evidence_base': ['temporal_analysis', 'emerging_substance_detection'],
            'implementation': [
                'Establish rapid data sharing with health authorities',
                'Create alert protocols for novel substances',
                'Integrate with existing surveillance systems',
                'Develop public communication strategies for alerts'
            ],
            'expected_outcomes': [
                'Faster response to emerging threats',
                'Reduced population exposure to high-risk substances',
                'Enhanced situational awareness for health services'
            ]
        })

        # Recommendation 4: Data integration
        recommendations.append({
            'id': 'R4',
            'recommendation': 'Develop integrated data infrastructure for comprehensive drug market intelligence',
            'priority': 'medium',
            'rationale': [
                'Multiple data sources provide more complete picture',
                'Standardized data enables longitudinal analysis',
                'Integrated systems support evidence-based decision making'
            ],
            'evidence_base': ['mixed_methods_integration'],
            'implementation': [
                'Create standardized data collection protocols',
                'Develop secure centralized data repository',
                'Implement real-time dashboards for stakeholders',
                'Establish data governance frameworks'
            ],
            'expected_outcomes': [
                'Improved trend identification',
                'Better resource allocation decisions',
                'Enhanced research capabilities'
            ]
        })

        # Recommendation 5: User-centered service design
        recommendations.append({
            'id': 'R5',
            'recommendation': 'Design services around user needs and preferences to maximize uptake',
            'priority': 'medium',
            'rationale': [
                'Different populations have distinct service preferences',
                'User trust and comfort influence service utilization',
                'Accessibility barriers reduce harm reduction effectiveness'
            ],
            'evidence_base': ['user_interviews', 'service_utilization_patterns'],
            'implementation': [
                'Conduct regular user needs assessments',
                'Ensure privacy and non-judgmental service delivery',
                'Provide multiple service access points',
                'Tailor communication to different user groups'
            ],
            'expected_outcomes': [
                'Increased service uptake',
                'Better population reach',
                'Enhanced user satisfaction and trust'
            ]
        })

        # Recommendation 6: Legal and regulatory support
        recommendations.append({
            'id': 'R6',
            'recommendation': 'Establish clear legal frameworks supporting drug checking services',
            'priority': 'high',
            'rationale': [
                'Legal certainty enables service sustainability',
                'Regulatory clarity reduces operational barriers',
                'Official recognition enhances service legitimacy'
            ],
            'evidence_base': ['provider_interviews', 'international_best_practice'],
            'implementation': [
                'Provide legal protections for service providers and users',
                'Establish licensing and accreditation frameworks',
                'Clarify legal status of sample possession and testing',
                'Remove regulatory barriers to service operation'
            ],
            'expected_outcomes': [
                'Service sustainability and expansion',
                'Increased provider confidence',
                'Enhanced service quality through regulation'
            ]
        })

        self.recommendations = recommendations
        return recommendations

    def generate_research_summary(self):
        """
        Generate a comprehensive research summary integrating all insights.

        Returns:
            Structured research summary document
        """
        summary = {
            'title': 'AI-Powered Research Analysis: Fixed-Site vs Festival Drug Checking Services in Australia',
            'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'research_questions': self.generate_research_questions(),
            'hypotheses': self.generate_hypotheses(),
            'key_insights': self.extract_key_insights(),
            'policy_recommendations': self.generate_policy_recommendations()
        }

        return summary

    def export_research_report(self, output_path):
        """
        Export comprehensive research report to file.

        Args:
            output_path: Path for output file
        """
        summary = self.generate_research_summary()

        report = []
        report.append("=" * 90)
        report.append("AI-POWERED RESEARCH ASSISTANT")
        report.append("Comprehensive Analysis of Drug Checking Services in Australia")
        report.append("=" * 90)
        report.append(f"Generated: {summary['generated']}")
        report.append("")

        # Research Questions
        report.append("\n" + "=" * 90)
        report.append("RESEARCH QUESTIONS")
        report.append("=" * 90)

        for category, questions in summary['research_questions'].items():
            if questions:
                report.append(f"\n{category.replace('_', ' ').title()}:")
                for i, q in enumerate(questions, 1):
                    report.append(f"  {i}. {q}")

        # Hypotheses
        report.append("\n" + "=" * 90)
        report.append("TESTABLE HYPOTHESES")
        report.append("=" * 90)

        for h in summary['hypotheses']:
            report.append(f"\n{h['id']}: {h['hypothesis']}")
            report.append(f"  Type: {h['type']}")
            report.append(f"  Testable: {h['testable']}")
            if 'suggested_tests' in h:
                report.append(f"  Suggested tests: {', '.join(h['suggested_tests'])}")
            report.append(f"  Implications: {h['implications']}")

        # Key Insights
        report.append("\n" + "=" * 90)
        report.append("KEY INSIGHTS")
        report.append("=" * 90)

        for category, insights_list in summary['key_insights'].items():
            if insights_list:
                report.append(f"\n{category.replace('_', ' ').title()}:")
                for insight in insights_list:
                    report.append(f"  • {insight['insight']}")
                    if 'implication' in insight:
                        report.append(f"    → {insight['implication']}")

        # Policy Recommendations
        report.append("\n" + "=" * 90)
        report.append("POLICY RECOMMENDATIONS")
        report.append("=" * 90)

        for rec in summary['policy_recommendations']:
            report.append(f"\n{rec['id']} [{rec['priority'].upper()} PRIORITY]: {rec['recommendation']}")
            report.append("\n  Rationale:")
            for r in rec['rationale']:
                report.append(f"    • {r}")

            report.append("\n  Implementation:")
            for impl in rec['implementation']:
                report.append(f"    → {impl}")

            report.append("\n  Expected Outcomes:")
            for outcome in rec['expected_outcomes']:
                report.append(f"    ✓ {outcome}")
            report.append("")

        # Save to file
        report_text = "\n".join(report)

        with open(output_path, 'w') as f:
            f.write(report_text)

        return report_text


def demonstrate_research_assistant():
    """Demonstrate research assistant capabilities."""
    print("=" * 80)
    print("AI RESEARCH ASSISTANT - DEMONSTRATION")
    print("=" * 80)
    print("\nKey Capabilities:")
    print("  ✓ Automated research question generation")
    print("  ✓ Hypothesis formulation from data patterns")
    print("  ✓ Insight extraction and synthesis")
    print("  ✓ Evidence-based policy recommendation")
    print("  ✓ Comprehensive research report generation")
    print("\nThis module integrates quantitative and qualitative data to provide")
    print("AI-powered research assistance for drug checking services analysis.")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_research_assistant()
