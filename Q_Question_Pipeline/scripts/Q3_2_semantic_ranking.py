"""
Q3.2: Semantic Ranking Module
Second stage of hybrid chunk retrieval - ranks filtered chunks using semantic similarity
Migrated from B-Pipeline logic but integrated into Q-Pipeline
"""

import json
import os
import re
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class Q32_SemanticRanking:
    """
    Semantic ranking using embedding similarity and answer capability assessment.
    Migrated from B-Pipeline's proven matching techniques.
    """

    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',
                 q_pipeline_path: str = "Q_Question_Pipeline/outputs"):
        """
        Initialize semantic ranking module.

        Args:
            model_name: Sentence transformer model for embeddings
            q_pipeline_path: Path to Q-Pipeline outputs
        """
        self.q_pipeline_path = q_pipeline_path
        self.model = SentenceTransformer(model_name)
        self.embedding_cache = {}

    def load_filtered_chunks(self, question_id: str) -> Tuple[List[Dict], Dict]:
        """
        Load Q3.1 geometric filtering results.

        Args:
            question_id: Question identifier

        Returns:
            Tuple of (filtered_chunks, filter_metrics)
        """
        q31_path = os.path.join(self.q_pipeline_path, "Q3.1_geometric_filtering.json")

        if not os.path.exists(q31_path):
            raise FileNotFoundError(f"Q3.1 output not found: {q31_path}")

        with open(q31_path, 'r') as f:
            q31_data = json.load(f)

        return q31_data['filtered_chunks'], q31_data['filter_metrics']

    def calculate_embedding_similarity(self,
                                      question_text: str,
                                      chunks: List[Dict]) -> Dict[str, float]:
        """
        Calculate cosine similarity between question and chunk embeddings.

        Args:
            question_text: Question text
            chunks: List of chunks to rank

        Returns:
            Dictionary of chunk_id -> similarity_score
        """
        # Get question embedding
        question_embedding = self.model.encode([question_text])[0]

        # Get chunk embeddings
        chunk_texts = [chunk.get('content', '') for chunk in chunks]
        chunk_ids = [chunk.get('chunk_id', f'chunk_{i}') for i, chunk in enumerate(chunks)]

        if not chunk_texts:
            return {}

        # Calculate embeddings (with caching)
        chunk_embeddings = []
        for chunk_id, text in zip(chunk_ids, chunk_texts):
            if chunk_id in self.embedding_cache:
                chunk_embeddings.append(self.embedding_cache[chunk_id])
            else:
                embedding = self.model.encode([text])[0]
                self.embedding_cache[chunk_id] = embedding
                chunk_embeddings.append(embedding)

        # Calculate cosine similarities
        similarities = cosine_similarity([question_embedding], chunk_embeddings)[0]

        return dict(zip(chunk_ids, similarities))

    def check_temporal_alignment(self,
                                 question_text: str,
                                 chunk_content: str) -> float:
        """
        Check if chunk contains temporal information aligned with question.

        Args:
            question_text: Question text
            chunk_content: Chunk content

        Returns:
            Temporal alignment score (0-1)
        """
        # Extract years from question
        question_years = set(re.findall(r'\b(19|20)\d{2}\b', question_text))

        if not question_years:
            # No temporal requirement in question
            return 0.5  # Neutral score

        # Extract years from chunk
        chunk_years = set(re.findall(r'\b(19|20)\d{2}\b', chunk_content))

        if not chunk_years:
            # No temporal info in chunk
            return 0.0

        # Calculate overlap
        overlap = len(question_years.intersection(chunk_years))
        total = len(question_years)

        return overlap / total if total > 0 else 0.0

    def assess_answer_capability(self,
                                 question_text: str,
                                 chunk_content: str) -> Dict[str, any]:
        """
        Assess whether chunk can potentially answer the question.
        Migrated from B3.3 answer capability assessment.

        Args:
            question_text: Question text
            chunk_content: Chunk content

        Returns:
            Dictionary with capability assessment details
        """
        assessment = {
            'can_provide_answer': False,
            'confidence': 0.0,
            'reasons': []
        }

        question_lower = question_text.lower()
        chunk_lower = chunk_content.lower()

        # Check for question type indicators
        if 'percentage change' in question_lower:
            # Need numbers and years for percentage calculation
            has_numbers = bool(re.findall(r'\d+[\d,\.]*', chunk_content))
            has_years = bool(re.findall(r'\b(19|20)\d{2}\b', chunk_content))
            has_revenue_data = 'revenue' in chunk_lower

            if has_numbers and has_years and has_revenue_data:
                assessment['can_provide_answer'] = True
                assessment['confidence'] = 0.9
                assessment['reasons'].append('Contains numerical data with years')

        elif 'what is' in question_lower or 'what was' in question_lower:
            # Lookup question - check for relevant terms
            key_terms = ['revenue', 'income', 'expense', 'profit', 'loss', 'total']
            matching_terms = sum(1 for term in key_terms if term in chunk_lower)

            if matching_terms > 0:
                assessment['can_provide_answer'] = True
                assessment['confidence'] = min(0.3 * matching_terms, 1.0)
                assessment['reasons'].append(f'Contains {matching_terms} relevant terms')

        # Check for data tables (common in financial documents)
        if '[[' in chunk_content or '","' in chunk_content:
            assessment['confidence'] += 0.2
            assessment['reasons'].append('Contains structured data/table')
            assessment['can_provide_answer'] = True

        # Check for specific numerical patterns
        if re.findall(r'\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?', chunk_content):
            assessment['confidence'] = min(assessment['confidence'] + 0.1, 1.0)
            assessment['reasons'].append('Contains formatted numbers')

        return assessment

    def calculate_semantic_scores(self,
                                  question_text: str,
                                  chunks: List[Dict]) -> List[Dict]:
        """
        Calculate comprehensive semantic scores for chunks.

        Args:
            question_text: Question text
            chunks: Filtered chunks from Q3.1

        Returns:
            Chunks with semantic scores added
        """
        print(f"\n[Q3.2] Calculating semantic scores for {len(chunks)} chunks")

        # Calculate embedding similarities
        similarities = self.calculate_embedding_similarity(question_text, chunks)

        # Process each chunk
        for chunk in chunks:
            chunk_id = chunk.get('chunk_id', '')
            content = chunk.get('content', '')

            # Embedding similarity
            chunk['embedding_similarity'] = similarities.get(chunk_id, 0.0)

            # Temporal alignment
            chunk['temporal_alignment'] = self.check_temporal_alignment(question_text, content)

            # Answer capability
            capability = self.assess_answer_capability(question_text, content)
            chunk['answer_capability'] = capability

            # Combined semantic score
            chunk['semantic_score'] = (
                0.4 * chunk['embedding_similarity'] +
                0.2 * chunk['temporal_alignment'] +
                0.4 * capability['confidence']
            )

        return chunks

    def rank_chunks_semantically(self,
                                question_id: str,
                                question_text: str,
                                top_k: int = 10) -> Tuple[List[Dict], Dict]:
        """
        Apply semantic ranking to geometrically filtered chunks.

        Args:
            question_id: Question identifier
            question_text: Full question text
            top_k: Number of top chunks to return

        Returns:
            Tuple of (ranked_chunks, ranking_metrics)
        """
        print(f"\n[Q3.2] Starting semantic ranking for {question_id}")

        # Load Q3.1 filtered chunks
        filtered_chunks, filter_metrics = self.load_filtered_chunks(question_id)
        print(f"[Q3.2] Loaded {len(filtered_chunks)} geometrically filtered chunks")

        if not filtered_chunks:
            print("[Q3.2] WARNING: No chunks to rank")
            empty_metrics = {
                'total_chunks_ranked': 0,
                'top_k_selected': 0,
                'avg_semantic_score': 0.0,
                'max_semantic_score': 0.0,
                'min_semantic_score': 0.0,
                'chunks_with_answer_capability': 0,
                'timestamp': datetime.now().isoformat(),
                'error': 'No chunks from geometric filtering'
            }
            return [], empty_metrics

        # Calculate semantic scores
        chunks_with_scores = self.calculate_semantic_scores(question_text, filtered_chunks)

        # Sort by semantic score
        chunks_with_scores.sort(key=lambda x: x.get('semantic_score', 0), reverse=True)

        # Select top K
        top_chunks = chunks_with_scores[:top_k]

        # Calculate ranking metrics
        ranking_metrics = {
            'total_chunks_ranked': len(chunks_with_scores),
            'top_k_selected': len(top_chunks),
            'avg_semantic_score': np.mean([c.get('semantic_score', 0) for c in top_chunks]) if top_chunks else 0,
            'max_semantic_score': max([c.get('semantic_score', 0) for c in chunks_with_scores]) if chunks_with_scores else 0,
            'min_semantic_score': min([c.get('semantic_score', 0) for c in chunks_with_scores]) if chunks_with_scores else 0,
            'chunks_with_answer_capability': sum(1 for c in chunks_with_scores if c.get('answer_capability', {}).get('can_provide_answer', False)),
            'timestamp': datetime.now().isoformat()
        }

        print(f"[Q3.2] Semantic ranking complete:")
        print(f"       - Top {top_k} chunks selected")
        print(f"       - Avg semantic score: {ranking_metrics['avg_semantic_score']:.3f}")
        print(f"       - Chunks with answer capability: {ranking_metrics['chunks_with_answer_capability']}")

        return top_chunks, ranking_metrics

    def save_results(self,
                    question_id: str,
                    ranked_chunks: List[Dict],
                    ranking_metrics: Dict,
                    output_dir: str = "Q_Question_Pipeline/outputs"):
        """
        Save semantic ranking results.

        Args:
            question_id: Question identifier
            ranked_chunks: Semantically ranked chunks
            ranking_metrics: Ranking statistics
            output_dir: Output directory
        """
        output_data = {
            'question_id': question_id,
            'stage': 'Q3.2_semantic_ranking',
            'ranked_chunks': ranked_chunks,
            'ranking_metrics': ranking_metrics
        }

        output_path = os.path.join(output_dir, "Q3.2_semantic_ranking.json")

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)

        print(f"[Q3.2] Results saved to: {output_path}")


def main():
    """Test Q3.2 semantic ranking on sample question."""

    # Initialize module with correct paths
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    q_pipeline_path = os.path.join(base_path, "outputs")

    q32 = Q32_SemanticRanking(q_pipeline_path=q_pipeline_path)

    # Test on sample question
    question_id = "finqa_test_1630"
    question_text = "What is the percentage change in the revenue from 2018 to 2019?"

    try:
        # Apply semantic ranking
        ranked_chunks, metrics = q32.rank_chunks_semantically(question_id, question_text)

        # Save results
        q32.save_results(question_id, ranked_chunks, metrics, q_pipeline_path)

        # Display summary
        print("\n" + "="*60)
        print(f"Q3.2 SEMANTIC RANKING SUMMARY")
        print("="*60)
        print(f"Question: {question_text}")
        print(f"Total chunks ranked: {metrics['total_chunks_ranked']}")
        print(f"Top K selected: {metrics['top_k_selected']}")
        print(f"Avg semantic score: {metrics['avg_semantic_score']:.3f}")

        if ranked_chunks:
            print(f"\nTop 3 chunks by semantic score:")
            for i, chunk in enumerate(ranked_chunks[:3], 1):
                print(f"  {i}. {chunk['chunk_id']}")
                print(f"     Semantic score: {chunk.get('semantic_score', 0):.3f}")
                print(f"     Embedding similarity: {chunk.get('embedding_similarity', 0):.3f}")
                print(f"     Can answer: {chunk.get('answer_capability', {}).get('can_provide_answer', False)}")
                print(f"     Content preview: {chunk.get('content', '')[:100]}...")

    except Exception as e:
        print(f"Error in Q3.2 semantic ranking: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()