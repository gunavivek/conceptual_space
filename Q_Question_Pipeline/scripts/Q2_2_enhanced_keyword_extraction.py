"""
Q2.2: Enhanced Keyword Extraction
Identifies and weights critical terms from questions for geometric positioning
"""

import json
import os
import re
import numpy as np
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional


class DomainTermExtractor:
    """
    Extracts domain-specific terminology from questions
    """

    def __init__(self):
        # Financial domain terms with importance weights
        self.financial_terms = {
            'revenue': 1.0, 'income': 0.9, 'expense': 0.9, 'profit': 0.9,
            'loss': 0.9, 'cost': 0.8, 'asset': 0.8, 'liability': 0.8,
            'equity': 0.8, 'cash': 0.7, 'investment': 0.7, 'dividend': 0.7,
            'earnings': 0.8, 'margin': 0.7, 'capital': 0.7, 'debt': 0.7
        }

        # Temporal domain terms
        self.temporal_terms = {
            'year': 0.9, 'quarter': 0.8, 'month': 0.7, 'annual': 0.8,
            'fiscal': 0.8, 'period': 0.7, 'date': 0.6, 'time': 0.6,
            'yearly': 0.8, 'quarterly': 0.8, 'monthly': 0.7
        }

        # Quantitative domain terms
        self.quantitative_terms = {
            'percentage': 1.0, 'percent': 1.0, 'ratio': 0.9, 'rate': 0.8,
            'change': 0.8, 'growth': 0.8, 'increase': 0.7, 'decrease': 0.7,
            'total': 0.7, 'sum': 0.6, 'average': 0.6, 'difference': 0.7,
            'variance': 0.7, 'deviation': 0.7, 'mean': 0.6, 'median': 0.6
        }

    def extract_domain_terms(self, question_text: str) -> List[Dict]:
        """
        Extract and weight domain-specific terms
        """
        text_lower = question_text.lower()
        domain_terms = []

        # Extract financial terms
        for term, weight in self.financial_terms.items():
            if term in text_lower:
                domain_terms.append({
                    'term': term,
                    'domain': 'financial',
                    'importance': weight,
                    'normalized_form': self._normalize_term(term)
                })

        # Extract temporal terms
        for term, weight in self.temporal_terms.items():
            if term in text_lower:
                domain_terms.append({
                    'term': term,
                    'domain': 'temporal',
                    'importance': weight,
                    'normalized_form': self._normalize_term(term)
                })

        # Extract quantitative terms
        for term, weight in self.quantitative_terms.items():
            if term in text_lower:
                domain_terms.append({
                    'term': term,
                    'domain': 'quantitative',
                    'importance': weight,
                    'normalized_form': self._normalize_term(term)
                })

        return domain_terms

    def _normalize_term(self, term: str) -> str:
        """
        Normalize term to canonical form
        """
        # Simple normalization mapping
        normalization_map = {
            'revenues': 'revenue',
            'incomes': 'income',
            'expenses': 'expense',
            'profits': 'profit',
            'losses': 'loss',
            'costs': 'cost',
            'percentage': 'percent',
            'yearly': 'annual',
            'quarterly': 'quarter',
            'monthly': 'month'
        }
        return normalization_map.get(term, term)


class SemanticKeywordAnalyzer:
    """
    Analyzes semantic importance of keywords
    """

    def __init__(self):
        # Stopwords to exclude
        self.stopwords = {
            'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'and', 'or', 'as', 'is', 'was', 'are', 'were'
        }

        # Interrogative words (lower weight as they're structural)
        self.interrogatives = {
            'what', 'when', 'where', 'who', 'why', 'how', 'which'
        }

    def extract_semantic_keywords(self, question_text: str) -> List[Dict]:
        """
        Extract keywords with semantic importance scores
        """
        # Tokenize (simple split - in production would use NLTK/spaCy)
        tokens = question_text.lower().split()
        keywords = []

        for token in tokens:
            # Remove punctuation
            clean_token = re.sub(r'[^\w\s%]', '', token)

            # Skip empty tokens and stopwords
            if not clean_token or clean_token in self.stopwords:
                continue

            # Calculate semantic weight
            weight = self._calculate_semantic_weight(clean_token)

            # Only include keywords above threshold
            if weight > 0.3:
                keyword_info = {
                    'keyword': clean_token,
                    'weight': weight,
                    'category': self._categorize_keyword(clean_token),
                    'pos_tag': self._get_pos_tag(clean_token),
                    'semantic_role': self._determine_semantic_role(clean_token, question_text)
                }
                keywords.append(keyword_info)

        return keywords

    def _calculate_semantic_weight(self, token: str) -> float:
        """
        Calculate semantic importance weight
        """
        # Numbers get high weight
        if any(char.isdigit() for char in token):
            return 0.9

        # Interrogatives get low weight (they're structural, not semantic)
        if token in self.interrogatives:
            return 0.3

        # Percentage signs indicate importance
        if '%' in token:
            return 0.95

        # Length-based weight (longer words often more specific)
        length_weight = min(1.0, len(token) / 8.0)

        # Base weight
        return max(0.4, length_weight)

    def _categorize_keyword(self, token: str) -> str:
        """
        Categorize keyword type
        """
        if any(char.isdigit() for char in token):
            return 'entity'
        elif token in self.interrogatives:
            return 'interrogative'
        elif token in ['change', 'increase', 'decrease', 'growth']:
            return 'relational'
        else:
            return 'concept'

    def _get_pos_tag(self, token: str) -> str:
        """
        Get simplified POS tag (in production would use NLTK)
        """
        if any(char.isdigit() for char in token):
            return 'NUM'
        elif token in self.interrogatives:
            return 'WH'
        elif token.endswith('ing'):
            return 'VBG'
        elif token.endswith('ed'):
            return 'VBD'
        elif token.endswith('s') and len(token) > 2:
            return 'NNS'
        else:
            return 'NN'

    def _determine_semantic_role(self, token: str, question_text: str) -> str:
        """
        Determine semantic role in question
        """
        if token in self.interrogatives:
            return 'question_word'
        elif any(char.isdigit() for char in token):
            return 'value'
        elif token in ['change', 'increase', 'decrease']:
            return 'operation'
        else:
            return 'entity'


class EntityKeywordRecognizer:
    """
    Recognizes named entities and special keywords
    """

    def extract_entity_keywords(self, question_text: str) -> List[Dict]:
        """
        Extract entity keywords from question
        """
        entities = []

        # Extract year entities (1900-2099)
        year_pattern = r'\b(19|20)\d{2}\b'
        for match in re.finditer(year_pattern, question_text):
            entities.append({
                'entity': match.group(),
                'entity_type': 'temporal',
                'confidence': 0.95,
                'span': [match.start(), match.end()]
            })

        # Extract percentage entities
        percentage_pattern = r'\b\d+\.?\d*\s*%'
        for match in re.finditer(percentage_pattern, question_text):
            entities.append({
                'entity': match.group().strip(),
                'entity_type': 'percentage',
                'confidence': 0.95,
                'span': [match.start(), match.end()]
            })

        # Extract general number entities
        number_pattern = r'\b\d+\.?\d*\b'
        for match in re.finditer(number_pattern, question_text):
            # Skip if already captured as year or percentage
            if not any(match.start() >= e['span'][0] and match.end() <= e['span'][1] for e in entities):
                entities.append({
                    'entity': match.group(),
                    'entity_type': 'numeric',
                    'confidence': 0.9,
                    'span': [match.start(), match.end()]
                })

        # Extract financial concept entities
        financial_concepts = ['revenue', 'income', 'profit', 'expense', 'cost', 'asset', 'liability']
        text_lower = question_text.lower()
        for concept in financial_concepts:
            if concept in text_lower:
                # Find all occurrences
                start = 0
                while True:
                    pos = text_lower.find(concept, start)
                    if pos == -1:
                        break
                    entities.append({
                        'entity': concept,
                        'entity_type': 'financial_concept',
                        'confidence': 0.85,
                        'span': [pos, pos + len(concept)]
                    })
                    start = pos + 1

        return entities


class KeywordEmbeddingGenerator:
    """
    Generates embedding vectors for keywords
    """

    def __init__(self):
        self.embedding_dim = 50

    def generate_embeddings(self, keywords: List[Dict], domain_terms: List[Dict]) -> Dict:
        """
        Generate embedding vectors for keywords
        """
        keyword_vectors = {}

        # Generate embeddings for semantic keywords
        for kw_info in keywords:
            keyword = kw_info['keyword']
            embedding = self._create_keyword_embedding(keyword, kw_info)
            keyword_vectors[keyword] = embedding

        # Add embeddings for domain terms
        for term_info in domain_terms:
            term = term_info['term']
            if term not in keyword_vectors:
                embedding = self._create_domain_embedding(term, term_info)
                keyword_vectors[term] = embedding

        # Create aggregate embedding
        if keyword_vectors:
            aggregate_vector = self._aggregate_embeddings(list(keyword_vectors.values()))
        else:
            aggregate_vector = [0.0] * self.embedding_dim

        return {
            'aggregate_vector': aggregate_vector,
            'keyword_vectors': keyword_vectors,
            'vector_dimension': self.embedding_dim
        }

    def _create_keyword_embedding(self, keyword: str, kw_info: Dict) -> List[float]:
        """
        Create embedding vector for keyword
        """
        # Use hash for reproducible pseudo-random embedding
        hash_val = int(hashlib.md5(keyword.encode()).hexdigest()[:8], 16)
        np.random.seed(hash_val)

        # Generate base embedding
        embedding = np.random.randn(self.embedding_dim) * 0.1

        # Adjust based on keyword properties
        if kw_info.get('category') == 'entity':
            embedding[0:10] += 0.3  # Boost entity dimensions
        if kw_info.get('weight', 0) > 0.7:
            embedding = embedding * (1 + kw_info['weight'])  # Scale by importance

        return embedding.tolist()

    def _create_domain_embedding(self, term: str, term_info: Dict) -> List[float]:
        """
        Create embedding for domain-specific term
        """
        hash_val = int(hashlib.md5(term.encode()).hexdigest()[:8], 16)
        np.random.seed(hash_val)

        embedding = np.random.randn(self.embedding_dim) * 0.1

        # Boost based on domain
        if term_info['domain'] == 'financial':
            embedding[10:20] += 0.5
        elif term_info['domain'] == 'temporal':
            embedding[20:30] += 0.5
        elif term_info['domain'] == 'quantitative':
            embedding[30:40] += 0.5

        # Scale by importance
        embedding = embedding * term_info['importance']

        return embedding.tolist()

    def _aggregate_embeddings(self, embeddings: List[List[float]]) -> List[float]:
        """
        Aggregate multiple embeddings into single vector
        """
        if not embeddings:
            return [0.0] * self.embedding_dim

        # Weighted mean aggregation
        aggregate = np.mean(embeddings, axis=0)

        # Normalize
        norm = np.linalg.norm(aggregate)
        if norm > 0:
            aggregate = aggregate / norm

        return aggregate.tolist()


class SemanticClusteringEngine:
    """
    Clusters keywords by semantic similarity
    """

    def cluster_keywords(self, keywords: List[Dict], domain_terms: List[Dict]) -> List[Dict]:
        """
        Group keywords into semantic clusters
        """
        # Predefined semantic clusters for financial QA
        cluster_definitions = {
            'temporal_cluster': {
                'keywords': ['year', 'period', 'quarter', 'annual', 'month', 'date', 'fiscal'],
                'theme': 'temporal'
            },
            'financial_cluster': {
                'keywords': ['revenue', 'income', 'expense', 'profit', 'cost', 'asset', 'liability'],
                'theme': 'financial'
            },
            'quantitative_cluster': {
                'keywords': ['percentage', 'percent', 'change', 'growth', 'increase', 'decrease', 'total'],
                'theme': 'quantitative'
            },
            'query_cluster': {
                'keywords': ['what', 'how', 'when', 'where', 'why', 'which'],
                'theme': 'interrogative'
            }
        }

        # Collect all keyword strings
        all_keyword_strings = [kw['keyword'] for kw in keywords]
        all_keyword_strings.extend([dt['term'] for dt in domain_terms])

        clusters = []
        for cluster_id, cluster_def in cluster_definitions.items():
            matched_keywords = []

            for kw_str in all_keyword_strings:
                # Check if keyword matches cluster pattern
                for cluster_kw in cluster_def['keywords']:
                    if cluster_kw in kw_str.lower() or kw_str.lower() in cluster_kw:
                        matched_keywords.append(kw_str)
                        break

            # Also check for years (special case for temporal)
            if cluster_id == 'temporal_cluster':
                year_pattern = r'\b(19|20)\d{2}\b'
                for kw_str in all_keyword_strings:
                    if re.match(year_pattern, kw_str):
                        matched_keywords.append(kw_str)

            if matched_keywords:
                # Remove duplicates
                matched_keywords = list(set(matched_keywords))
                clusters.append({
                    'cluster_id': cluster_id,
                    'cluster_theme': cluster_def['theme'],
                    'keywords': matched_keywords,
                    'coherence_score': len(matched_keywords) / max(1, len(cluster_def['keywords']))
                })

        return clusters


class Q2_2_EnhancedKeywordExtraction:
    """
    Main Q2.2 Enhanced Keyword Extraction processor
    """

    def __init__(self):
        self.domain_extractor = DomainTermExtractor()
        self.semantic_analyzer = SemanticKeywordAnalyzer()
        self.entity_recognizer = EntityKeywordRecognizer()
        self.embedding_generator = KeywordEmbeddingGenerator()
        self.clustering_engine = SemanticClusteringEngine()

    def extract_keywords(self, question_id: str) -> Dict:
        """
        Main processing function for enhanced keyword extraction
        """
        start_time = datetime.now()

        try:
            # Load question data from Q1
            question_data = self._load_question_from_q1(question_id)
            question_text = question_data['question_text']
            doc_id = question_data['doc_id']

            # Extract domain-specific terms
            domain_terms = self.domain_extractor.extract_domain_terms(question_text)

            # Extract semantic keywords
            semantic_keywords = self.semantic_analyzer.extract_semantic_keywords(question_text)

            # Extract entity keywords
            entity_keywords = self.entity_recognizer.extract_entity_keywords(question_text)

            # Cluster keywords semantically
            semantic_clusters = self.clustering_engine.cluster_keywords(
                semantic_keywords, domain_terms
            )

            # Generate keyword embeddings
            keyword_embeddings = self.embedding_generator.generate_embeddings(
                semantic_keywords, domain_terms
            )

            # Calculate keyword features
            keyword_features = self._calculate_keyword_features(
                semantic_keywords, domain_terms, entity_keywords
            )

            # Calculate processing metadata
            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            result = {
                'question_id': question_id,
                'doc_id': doc_id,
                'question_text': question_text,
                'keyword_extraction': {
                    'primary_keywords': semantic_keywords,
                    'domain_specific_terms': domain_terms,
                    'entity_keywords': entity_keywords,
                    'semantic_clusters': semantic_clusters
                },
                'keyword_features': keyword_features,
                'keyword_embeddings': keyword_embeddings,
                'processing_metadata': {
                    'extraction_timestamp': datetime.now().isoformat(),
                    'processing_time_ms': processing_time,
                    'extraction_method': 'enhanced_semantic',
                    'confidence_score': self._calculate_confidence(keyword_features)
                }
            }

            return result

        except Exception as e:
            print(f"Error in Q2.2 processing: {e}")
            return self._get_default_output(question_id)

    def _load_question_from_q1(self, question_id: str) -> Dict:
        """Load question data from Q1 output - NO ANSWER DATA"""
        try:
            q1_path = "../outputs/Q1_Question_ingestion.json"
            with open(q1_path, 'r') as f:
                q1_data = json.load(f)

            # Handle both single and multi-question formats
            if isinstance(q1_data, dict):
                if 'question_id' in q1_data and q1_data['question_id'] == question_id:
                    # Single question format
                    raw_data = q1_data
                elif question_id in q1_data:
                    # Multi-question format
                    raw_data = q1_data[question_id]
                else:
                    raw_data = q1_data  # Assume it's the right question

            # Extract only safe fields (no answer data)
            return {
                'question_id': raw_data.get('question_id', question_id),
                'doc_id': raw_data.get('doc_id', question_id),
                'question_text': raw_data.get('question_text', '')
            }

        except Exception as e:
            print(f"Error loading Q1 data: {e}")
            # Return default question
            return {
                'question_id': question_id,
                'doc_id': question_id,
                'question_text': 'What is the percentage change in the revenue from 2018 to 2019?'
            }

    def _calculate_keyword_features(self, semantic_keywords: List[Dict],
                                   domain_terms: List[Dict],
                                   entity_keywords: List[Dict]) -> Dict:
        """
        Calculate keyword feature statistics
        """
        total_keywords = len(semantic_keywords)
        unique_keywords = len(set(k['keyword'] for k in semantic_keywords))
        domain_count = len(domain_terms)
        entity_count = len(entity_keywords)

        return {
            'total_keywords': total_keywords,
            'unique_keywords': unique_keywords,
            'domain_term_ratio': domain_count / max(1, total_keywords),
            'semantic_density': unique_keywords / max(1, total_keywords),
            'keyword_diversity': 1.0 - (1.0 / max(1, unique_keywords)),
            'entity_ratio': entity_count / max(1, total_keywords)
        }

    def _calculate_confidence(self, keyword_features: Dict) -> float:
        """
        Calculate confidence in keyword extraction
        """
        # Base confidence on keyword richness
        if keyword_features['total_keywords'] == 0:
            return 0.0

        confidence = 0.5  # Base confidence

        # Boost for domain terms
        if keyword_features['domain_term_ratio'] > 0.3:
            confidence += 0.2

        # Boost for good keyword diversity
        if keyword_features['keyword_diversity'] > 0.5:
            confidence += 0.2

        # Boost for entity presence
        if keyword_features.get('entity_ratio', 0) > 0.2:
            confidence += 0.1

        return min(1.0, confidence)

    def _get_default_output(self, question_id: str) -> Dict:
        """Return default output on error"""
        return {
            'question_id': question_id,
            'doc_id': question_id,
            'question_text': 'Error in processing',
            'keyword_extraction': {
                'primary_keywords': [],
                'domain_specific_terms': [],
                'entity_keywords': [],
                'semantic_clusters': []
            },
            'keyword_features': {
                'total_keywords': 0,
                'unique_keywords': 0,
                'domain_term_ratio': 0.0,
                'semantic_density': 0.0,
                'keyword_diversity': 0.0
            },
            'keyword_embeddings': {
                'aggregate_vector': [0.0] * 50,
                'keyword_vectors': {},
                'vector_dimension': 50
            },
            'processing_metadata': {
                'extraction_timestamp': datetime.now().isoformat(),
                'processing_time_ms': 0.0,
                'extraction_method': 'error_fallback',
                'confidence_score': 0.0
            }
        }

    def save_output(self, result: Dict, output_path: str = "../outputs/Q2.2_enhanced_keyword_extraction.json"):
        """Save Q2.2 output to file"""
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Wrap in question_id structure for consistency
            output_data = {result['question_id']: result}

            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)

            print(f"Q2.2 output saved to {output_path}")

        except Exception as e:
            print(f"Error saving Q2.2 output: {e}")


def main():
    """Process all questions from Q1 output"""
    print("=" * 70)
    print("Q2.2: Enhanced Keyword Extraction - Processing All Questions from Q1")
    print("=" * 70)

    # Initialize Q2.2
    q22 = Q2_2_EnhancedKeywordExtraction()

    try:
        # Load all questions from Q1 output
        q1_path = "../outputs/Q1_Question_ingestion.json"
        with open(q1_path, 'r') as f:
            q1_data = json.load(f)

        questions = q1_data.get('questions', [])
        print(f"Found {len(questions)} questions from Q1")

        all_results = {}
        successful = 0
        failed = 0

        for i, question_data in enumerate(questions, 1):
            question_id = question_data.get('question_id', f'q_{i}')
            question_text = question_data.get('question_text', '')

            print(f"\n[{i}/{len(questions)}] Processing: {question_id}")
            print(f"Question: {question_text[:80]}...")

            try:
                # Modify the load function temporarily to use the question data directly
                original_load = q22._load_question_from_q1
                q22._load_question_from_q1 = lambda qid: {
                    'question_id': question_data.get('question_id', qid),
                    'doc_id': question_data.get('doc_id', qid),
                    'question_text': question_data.get('question_text', '')
                }

                # Run keyword extraction
                result = q22.extract_keywords(question_id)
                all_results[question_id] = result

                # Show brief summary
                kw_count = len(result['keyword_extraction']['primary_keywords'])
                domain_count = len(result['keyword_extraction']['domain_specific_terms'])
                entity_count = len(result['keyword_extraction']['entity_keywords'])

                print(f"  -> Keywords: {kw_count}, Domain terms: {domain_count}, Entities: {entity_count}")
                successful += 1

                # Restore original function
                q22._load_question_from_q1 = original_load

            except Exception as e:
                print(f"  -> ERROR: {e}")
                failed += 1
                # Restore original function
                q22._load_question_from_q1 = original_load

        # Save all results
        output_path = "../outputs/Q2.2_enhanced_keyword_extraction.json"
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\n" + "=" * 70)
        print("Q2.2 BATCH PROCESSING COMPLETE")
        print("=" * 70)
        print(f"Total questions: {len(questions)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Success rate: {successful/len(questions)*100:.1f}%")
        print(f"Results saved to: {output_path}")

        # Show summary statistics
        if all_results:
            total_keywords = sum(len(r['keyword_extraction']['primary_keywords']) for r in all_results.values())
            total_domain_terms = sum(len(r['keyword_extraction']['domain_specific_terms']) for r in all_results.values())
            total_entities = sum(len(r['keyword_extraction']['entity_keywords']) for r in all_results.values())

            print(f"\nExtraction Summary:")
            print(f"  Total keywords extracted: {total_keywords}")
            print(f"  Total domain terms: {total_domain_terms}")
            print(f"  Total entities: {total_entities}")
            print(f"  Avg keywords per question: {total_keywords/len(all_results):.1f}")

        return all_results

    except Exception as e:
        print(f"Error in Q2.2 batch processing: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("Q2.2 ENHANCED KEYWORD EXTRACTION")
    print("=" * 50)

    result = main()

    if result:
        print("Q2.2_enhanced_keyword_extraction.json created successfully")
        print("Keyword extraction complete - ready for Q2.5 coordinate calculation")
    else:
        print("Failed to create Q2.2 output")