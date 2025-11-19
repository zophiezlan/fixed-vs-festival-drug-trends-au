"""
AI-Powered Natural Language Processing Module for Qualitative Data Analysis.

This module provides advanced NLP capabilities for automated analysis of interview data,
including sentiment analysis, topic modeling, named entity recognition, and semantic analysis.
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import re
import warnings
warnings.filterwarnings('ignore')

# NLP libraries
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation, NMF
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not installed. Some NLP features will be limited.")

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False
    print("Warning: textblob not installed. Sentiment analysis will be limited.")


class AIQualitativeAnalyzer:
    """
    Advanced AI-powered qualitative data analyzer using NLP techniques.
    """

    def __init__(self, data_path=None, dataframe=None):
        """
        Initialize the AI analyzer with interview data.

        Args:
            data_path: Path to CSV file with interview data
            dataframe: Or provide DataFrame directly
        """
        if dataframe is not None:
            self.df = dataframe
        elif data_path:
            self.df = pd.read_csv(data_path)
            if 'interview_date' in self.df.columns:
                self.df['interview_date'] = pd.to_datetime(self.df['interview_date'])
        else:
            raise ValueError("Must provide either data_path or dataframe")

        self.theme_columns = [col for col in self.df.columns if col.startswith('theme_')]
        self.all_text = self._extract_all_text()

    def _extract_all_text(self):
        """Extract all text responses from theme columns."""
        all_text = []
        for col in self.theme_columns:
            all_text.extend(self.df[col].dropna().tolist())
        return all_text

    def perform_sentiment_analysis(self):
        """
        Perform sentiment analysis on interview responses.

        Returns:
            Dictionary with sentiment scores by service type and theme
        """
        if not HAS_TEXTBLOB:
            return self._simple_sentiment_analysis()

        results = {
            'by_service_type': {},
            'by_theme': {},
            'overall': {}
        }

        # Analyze by service type
        for service_type in self.df['service_type'].unique():
            service_data = self.df[self.df['service_type'] == service_type]
            sentiments = []
            subjectivity = []

            for col in self.theme_columns:
                responses = service_data[col].dropna().tolist()
                for response in responses:
                    blob = TextBlob(str(response))
                    sentiments.append(blob.sentiment.polarity)
                    subjectivity.append(blob.sentiment.subjectivity)

            if sentiments:
                results['by_service_type'][service_type] = {
                    'avg_sentiment': np.mean(sentiments),
                    'sentiment_std': np.std(sentiments),
                    'avg_subjectivity': np.mean(subjectivity),
                    'positive_responses': sum(1 for s in sentiments if s > 0.1),
                    'negative_responses': sum(1 for s in sentiments if s < -0.1),
                    'neutral_responses': sum(1 for s in sentiments if -0.1 <= s <= 0.1)
                }

        # Analyze by theme
        for theme_col in self.theme_columns:
            theme_name = theme_col.replace('theme_', '').replace('_', ' ').title()
            sentiments = []

            responses = self.df[theme_col].dropna().tolist()
            for response in responses:
                blob = TextBlob(str(response))
                sentiments.append(blob.sentiment.polarity)

            if sentiments:
                results['by_theme'][theme_name] = {
                    'avg_sentiment': np.mean(sentiments),
                    'sentiment_std': np.std(sentiments)
                }

        # Overall sentiment
        all_sentiments = []
        for text in self.all_text:
            blob = TextBlob(str(text))
            all_sentiments.append(blob.sentiment.polarity)

        results['overall'] = {
            'avg_sentiment': np.mean(all_sentiments),
            'sentiment_distribution': {
                'positive': sum(1 for s in all_sentiments if s > 0.1),
                'negative': sum(1 for s in all_sentiments if s < -0.1),
                'neutral': sum(1 for s in all_sentiments if -0.1 <= s <= 0.1)
            }
        }

        return results

    def _simple_sentiment_analysis(self):
        """Simple rule-based sentiment when TextBlob unavailable."""
        positive_words = ['good', 'great', 'excellent', 'better', 'effective', 'helpful',
                         'positive', 'important', 'valuable', 'comprehensive', 'thorough']
        negative_words = ['bad', 'poor', 'limited', 'difficult', 'challenge', 'problem',
                         'negative', 'insufficient', 'inadequate']

        results = {'by_service_type': {}, 'overall': {}}

        for service_type in self.df['service_type'].unique():
            service_data = self.df[self.df['service_type'] == service_type]
            scores = []

            for col in self.theme_columns:
                responses = service_data[col].dropna().tolist()
                for response in responses:
                    text_lower = str(response).lower()
                    pos_count = sum(text_lower.count(word) for word in positive_words)
                    neg_count = sum(text_lower.count(word) for word in negative_words)
                    scores.append(pos_count - neg_count)

            if scores:
                results['by_service_type'][service_type] = {
                    'avg_sentiment': np.mean(scores),
                    'positive_responses': sum(1 for s in scores if s > 0),
                    'negative_responses': sum(1 for s in scores if s < 0)
                }

        return results

    def perform_topic_modeling(self, n_topics=5, method='lda'):
        """
        Perform topic modeling on interview responses using LDA or NMF.

        Args:
            n_topics: Number of topics to extract
            method: 'lda' (Latent Dirichlet Allocation) or 'nmf' (Non-negative Matrix Factorization)

        Returns:
            Dictionary with topics, keywords, and document-topic distributions
        """
        if not HAS_SKLEARN:
            return self._simple_topic_extraction(n_topics)

        # Prepare text data
        texts = [str(text) for text in self.all_text if len(str(text)) > 20]

        if len(texts) < n_topics:
            return {'error': 'Not enough text data for topic modeling'}

        # Vectorize
        if method == 'lda':
            vectorizer = CountVectorizer(max_features=1000, stop_words='english',
                                        min_df=2, max_df=0.8)
            doc_term_matrix = vectorizer.fit_transform(texts)

            # LDA model
            model = LatentDirichletAllocation(n_components=n_topics, random_state=42,
                                             max_iter=20, learning_method='online')
            doc_topics = model.fit_transform(doc_term_matrix)

        else:  # NMF
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english',
                                        min_df=2, max_df=0.8)
            doc_term_matrix = vectorizer.fit_transform(texts)

            model = NMF(n_components=n_topics, random_state=42, max_iter=200)
            doc_topics = model.fit_transform(doc_term_matrix)

        # Extract topics and keywords
        feature_names = vectorizer.get_feature_names_out()
        topics = {}

        for topic_idx, topic in enumerate(model.components_):
            top_indices = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            topics[f'Topic {topic_idx + 1}'] = {
                'keywords': top_words,
                'weight': float(topic[top_indices[0]])
            }

        # Analyze topic distribution by service type
        topic_by_service = {}
        for i, service_type in enumerate(self.df['service_type'].unique()):
            service_data = self.df[self.df['service_type'] == service_type]
            service_texts = []

            for col in self.theme_columns:
                service_texts.extend(service_data[col].dropna().tolist())

            service_texts = [str(t) for t in service_texts if len(str(t)) > 20]

            if service_texts:
                service_dtm = vectorizer.transform(service_texts)
                service_topics = model.transform(service_dtm)

                topic_by_service[service_type] = {
                    'dominant_topics': [f'Topic {i+1}' for i in np.argsort(service_topics.mean(axis=0))[-3:][::-1]],
                    'topic_distribution': service_topics.mean(axis=0).tolist()
                }

        return {
            'topics': topics,
            'method': method,
            'n_topics': n_topics,
            'topic_by_service': topic_by_service
        }

    def _simple_topic_extraction(self, n_topics=5):
        """Simple keyword-based topic extraction when sklearn unavailable."""
        from collections import Counter

        # Simple word frequency analysis
        all_words = []
        for text in self.all_text:
            words = re.findall(r'\b[a-z]{4,}\b', str(text).lower())
            all_words.extend(words)

        # Remove common words
        stop_words = {'that', 'this', 'with', 'have', 'from', 'they', 'been',
                     'were', 'their', 'there', 'would', 'about', 'which'}
        all_words = [w for w in all_words if w not in stop_words]

        # Get most common
        word_freq = Counter(all_words)
        top_words = word_freq.most_common(n_topics * 10)

        topics = {}
        for i in range(min(n_topics, len(top_words))):
            start = i * 10
            end = start + 10
            topics[f'Topic {i+1}'] = {
                'keywords': [word for word, _ in top_words[start:end]]
            }

        return {'topics': topics, 'method': 'simple_frequency'}

    def extract_named_entities(self):
        """
        Extract named entities (organizations, substances, locations) from text.

        Returns:
            Dictionary with entity counts and examples
        """
        # Simple rule-based NER for drug checking context
        entities = {
            'substances': Counter(),
            'organizations': Counter(),
            'locations': Counter(),
            'service_types': Counter()
        }

        # Known patterns
        substance_patterns = [
            r'\b(?:mdma|cocaine|heroin|methamphetamine|ketamine|lsd|dmt|ghb)\b',
            r'\b(?:ecstasy|molly|speed|ice|crystal|acid|mushrooms)\b',
            r'\b(?:fentanyl|carfentanil|synthetic|cathinone|cannabinoid)\b'
        ]

        org_patterns = [
            r'\b(?:pill testing australia|cantest|dancewize|harm reduction|festival)\b',
        ]

        location_patterns = [
            r'\b(?:canberra|sydney|melbourne|act|nsw|victoria|australia)\b'
        ]

        for text in self.all_text:
            text_lower = str(text).lower()

            # Extract substances
            for pattern in substance_patterns:
                matches = re.findall(pattern, text_lower)
                entities['substances'].update(matches)

            # Extract organizations
            for pattern in org_patterns:
                matches = re.findall(pattern, text_lower)
                entities['organizations'].update(matches)

            # Extract locations
            for pattern in location_patterns:
                matches = re.findall(pattern, text_lower)
                entities['locations'].update(matches)

            # Extract service mentions
            if 'fixed' in text_lower or 'fixed-site' in text_lower:
                entities['service_types']['fixed-site'] += 1
            if 'festival' in text_lower:
                entities['service_types']['festival'] += 1

        return {
            'substances': dict(entities['substances'].most_common(20)),
            'organizations': dict(entities['organizations'].most_common(10)),
            'locations': dict(entities['locations'].most_common(10)),
            'service_types': dict(entities['service_types'])
        }

    def calculate_text_similarity(self):
        """
        Calculate semantic similarity between responses from different service types.

        Returns:
            Similarity scores and analysis
        """
        if not HAS_SKLEARN:
            return {'error': 'sklearn required for similarity analysis'}

        # Get responses by service type
        service_texts = {}
        for service_type in self.df['service_type'].unique():
            service_data = self.df[self.df['service_type'] == service_type]
            texts = []
            for col in self.theme_columns:
                texts.extend(service_data[col].dropna().tolist())
            service_texts[service_type] = ' '.join(str(t) for t in texts)

        if len(service_texts) < 2:
            return {'error': 'Need at least 2 service types for comparison'}

        # Vectorize
        vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
        texts_list = list(service_texts.values())
        tfidf_matrix = vectorizer.fit_transform(texts_list)

        # Calculate similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)

        service_types = list(service_texts.keys())
        results = {
            'similarity_scores': {},
            'distinctive_terms': {}
        }

        for i, service1 in enumerate(service_types):
            for j, service2 in enumerate(service_types):
                if i < j:
                    results['similarity_scores'][f'{service1} vs {service2}'] = float(similarity_matrix[i, j])

        # Find distinctive terms for each service
        feature_names = vectorizer.get_feature_names_out()
        for i, service_type in enumerate(service_types):
            scores = tfidf_matrix[i].toarray()[0]
            top_indices = scores.argsort()[-10:][::-1]
            results['distinctive_terms'][service_type] = [feature_names[idx] for idx in top_indices]

        return results

    def generate_ai_insights(self):
        """
        Generate comprehensive AI-powered insights from qualitative data.

        Returns:
            Dictionary with multiple AI analysis results
        """
        insights = {
            'sentiment_analysis': self.perform_sentiment_analysis(),
            'topic_modeling': self.perform_topic_modeling(n_topics=5, method='lda'),
            'named_entities': self.extract_named_entities(),
            'text_similarity': self.calculate_text_similarity()
        }

        return insights

    def generate_ai_summary_report(self):
        """Generate comprehensive AI analysis report."""
        report = []
        report.append("=" * 80)
        report.append("AI-POWERED QUALITATIVE ANALYSIS")
        report.append("Natural Language Processing & Machine Learning Insights")
        report.append("=" * 80)
        report.append("")

        # Sentiment Analysis
        report.append("1. SENTIMENT ANALYSIS")
        report.append("-" * 80)
        sentiment_results = self.perform_sentiment_analysis()

        if 'by_service_type' in sentiment_results:
            for service_type, data in sentiment_results['by_service_type'].items():
                report.append(f"\n{service_type}:")
                report.append(f"  Average Sentiment: {data.get('avg_sentiment', 0):.3f}")
                if 'positive_responses' in data:
                    report.append(f"  Positive Responses: {data['positive_responses']}")
                    report.append(f"  Negative Responses: {data['negative_responses']}")
                    if 'neutral_responses' in data:
                        report.append(f"  Neutral Responses: {data['neutral_responses']}")

        # Topic Modeling
        report.append("\n")
        report.append("2. TOPIC MODELING (Latent Dirichlet Allocation)")
        report.append("-" * 80)
        topic_results = self.perform_topic_modeling()

        if 'topics' in topic_results and 'error' not in topic_results:
            for topic_name, topic_data in topic_results['topics'].items():
                keywords = topic_data['keywords'][:5]
                report.append(f"\n{topic_name}: {', '.join(keywords)}")

            if 'topic_by_service' in topic_results:
                report.append("\nDominant Topics by Service Type:")
                for service_type, data in topic_results['topic_by_service'].items():
                    topics = ', '.join(data['dominant_topics'][:3])
                    report.append(f"  {service_type}: {topics}")

        # Named Entity Recognition
        report.append("\n")
        report.append("3. NAMED ENTITY RECOGNITION")
        report.append("-" * 80)
        entities = self.extract_named_entities()

        if entities['substances']:
            top_substances = list(entities['substances'].items())[:5]
            report.append(f"\nMost Mentioned Substances:")
            for substance, count in top_substances:
                report.append(f"  {substance}: {count} mentions")

        if entities['service_types']:
            report.append(f"\nService Type Mentions:")
            for service, count in entities['service_types'].items():
                report.append(f"  {service}: {count} mentions")

        # Text Similarity
        report.append("\n")
        report.append("4. SEMANTIC SIMILARITY ANALYSIS")
        report.append("-" * 80)
        similarity = self.calculate_text_similarity()

        if 'similarity_scores' in similarity:
            for comparison, score in similarity['similarity_scores'].items():
                report.append(f"\n{comparison}: {score:.3f} similarity")

        if 'distinctive_terms' in similarity:
            report.append("\nDistinctive Terms by Service Type:")
            for service_type, terms in similarity['distinctive_terms'].items():
                report.append(f"  {service_type}: {', '.join(terms[:5])}")

        report.append("\n")
        report.append("=" * 80)
        report.append("KEY AI INSIGHTS:")
        report.append("✓ Sentiment patterns reveal stakeholder attitudes toward each service model")
        report.append("✓ Topic modeling uncovers hidden themes in interview responses")
        report.append("✓ Named entity recognition identifies key substances and organizations")
        report.append("✓ Semantic analysis highlights distinctive language patterns by service type")
        report.append("=" * 80)

        return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    print("AI-Powered NLP Analysis Module")
    print("This module provides advanced qualitative data analysis capabilities")
    print("\nKey Features:")
    print("  - Sentiment Analysis (TextBlob)")
    print("  - Topic Modeling (LDA/NMF)")
    print("  - Named Entity Recognition")
    print("  - Semantic Similarity Analysis")
    print("  - Automated insight generation")
