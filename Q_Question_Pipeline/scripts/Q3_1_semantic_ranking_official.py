"""
Q3.1: Simplified Semantic Ranking Module
Uses enhanced Q2.5 output eliminating A3 dependency

REVOLUTIONARY IMPROVEMENT:
- Single input source: Enhanced Q2.5 output only
- No cross-pipeline dependencies (A3, A4)
- Geometric filtering already completed by Q2.5
- Focus purely on semantic ranking of pre-filtered chunks
- Self-contained and simplified architecture

This simplified Q3.1 takes geometrically filtered chunks from enhanced Q2.5
and applies semantic ranking for final chunk selection.
"""

import json
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class Q31_SimplifiedSemanticRanking:
    """
    Simplified Q3.1 - Semantic ranking of pre-filtered chunks from enhanced Q2.5.

    This eliminates the A3 dependency by using chunks already geometrically
    filtered and included in Q2.5 output.
    """

    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',
                 q_pipeline_path: str = "Q_Question_Pipeline/outputs"):
        """
        Initialize simplified semantic ranking module.

        Args:
            model_name: Sentence transformer model for embeddings
            q_pipeline_path: Path to Q-Pipeline outputs
        """
        self.q_pipeline_path = q_pipeline_path
        self.model = SentenceTransformer(model_name)

    def load_enhanced_q25_output(self, question_id: str) -> Dict:
        """
        Load enhanced Q2.5 output with geometric filtering results.

        Args:
            question_id: Question identifier

        Returns:
            Enhanced Q2.5 data with filtered chunks
        """
        q25_path = os.path.join(self.q_pipeline_path, f"Q2.5_document_aware_assignment_{question_id}.json")

        if not os.path.exists(q25_path):
            raise FileNotFoundError(f"Enhanced Q2.5 output not found: {q25_path}")

        with open(q25_path, 'r') as f:
            q25_data = json.load(f)

        # Verify it's the enhanced version with geometric filtering
        if 'geometric_filtering' not in q25_data:
            raise ValueError(f"Q2.5 output is not enhanced version. Missing geometric_filtering data.")

        return q25_data

    def calculate_semantic_similarity(self, question_text: str, chunk_content: str) -> float:
        """
        Calculate semantic similarity between question and chunk content.

        Args:
            question_text: Question text
            chunk_content: Chunk content

        Returns:
            Similarity score (0-1)
        """
        # Create embeddings
        question_embedding = self.model.encode([question_text])[0]
        chunk_embedding = self.model.encode([chunk_content])[0]

        # Calculate cosine similarity
        similarity = cosine_similarity([question_embedding], [chunk_embedding])[0][0]
        return float(similarity)

    def assess_answer_capability(self, chunk: Dict, question_text: str) -> Tuple[bool, float, List[str]]:
        """
        Assess if chunk can provide answer to the question.

        Args:
            chunk: Chunk data
            question_text: Question text

        Returns:
            Tuple of (can_answer, confidence, reasons)
        """
        content = chunk.get('content', '').lower()
        question_lower = question_text.lower()
        reasons = []
        confidence = 0.0

        # Check for numerical data (important for financial questions)
        if any(char.isdigit() for char in content):
            reasons.append("Contains numerical data")
            confidence += 0.3

        # Check for year mentions (temporal questions)
        if any(year in content for year in ['2018', '2019', '2020', '2021']):
            reasons.append("Contains temporal data with years")
            confidence += 0.2

        # Check for structured data patterns
        if any(pattern in content for pattern in ['[[', ']]', 'revenue', 'income']):
            reasons.append("Contains structured data/table")
            confidence += 0.2

        # Check for formatted numbers
        if any(pattern in content for pattern in [',000', 'us$', '$']):
            reasons.append("Contains formatted numbers")
            confidence += 0.2

        # Check for question keywords
        question_words = set(question_lower.split())
        content_words = set(content.split())
        overlap = question_words.intersection(content_words)
        if len(overlap) > 2:
            reasons.append(f"High keyword overlap ({len(overlap)} words)")
            confidence += 0.1

        can_answer = confidence >= 0.4
        return can_answer, min(confidence, 1.0), reasons

    def apply_semantic_ranking(self,
                             question_id: str,
                             top_k: int = 10) -> Tuple[List[Dict], Dict]:
        """
        Apply semantic ranking to geometrically filtered chunks from Q2.5.

        Args:
            question_id: Question identifier
            top_k: Number of top chunks to select

        Returns:
            Tuple of (ranked_chunks, ranking_metrics)
        """
        print(f"\n[Q3.1-Simplified] Starting semantic ranking for {question_id}")

        # Load enhanced Q2.5 output
        q25_data = self.load_enhanced_q25_output(question_id)

        question_text = q25_data.get('question_text', '')
        filtered_chunks = q25_data['geometric_filtering']['filtered_chunks']
        filter_metrics = q25_data['geometric_filtering']['filter_metrics']

        print(f"[Q3.1-Simplified] Loaded {len(filtered_chunks)} pre-filtered chunks from enhanced Q2.5")
        print(f"[Q3.1-Simplified] Original reduction: {filter_metrics['reduction_percentage']:.1f}%")

        # Calculate semantic scores for each chunk
        for chunk in filtered_chunks:
            # Semantic similarity
            semantic_score = self.calculate_semantic_similarity(question_text, chunk['content'])
            chunk['embedding_similarity'] = semantic_score

            # Answer capability assessment
            can_answer, answer_confidence, reasons = self.assess_answer_capability(chunk, question_text)
            chunk['answer_capability'] = {
                'can_provide_answer': can_answer,
                'confidence': answer_confidence,
                'reasons': reasons
            }

            # Combined semantic score (embedding + answer capability)
            chunk['semantic_score'] = (
                0.6 * semantic_score +           # Embedding similarity
                0.4 * answer_confidence          # Answer capability
            )

        # Sort by semantic score
        filtered_chunks.sort(key=lambda x: x.get('semantic_score', 0), reverse=True)

        # Select top K chunks
        top_chunks = filtered_chunks[:top_k]

        # Calculate ranking metrics
        ranking_metrics = {
            'input_chunks': len(filtered_chunks),
            'output_chunks': len(top_chunks),
            'further_reduction_percentage': (1 - len(top_chunks) / len(filtered_chunks)) * 100 if filtered_chunks else 0,
            'avg_semantic_score': np.mean([c.get('semantic_score', 0) for c in top_chunks]) if top_chunks else 0,
            'avg_embedding_similarity': np.mean([c.get('embedding_similarity', 0) for c in top_chunks]) if top_chunks else 0,
            'chunks_with_answer_capability': sum(1 for c in top_chunks if c['answer_capability']['can_provide_answer']),
            'timestamp': datetime.now().isoformat()
        }

        print(f"[Q3.1-Simplified] Semantic ranking complete:")
        print(f"       - Further reduced from {len(filtered_chunks)} to {len(top_chunks)} chunks")
        print(f"       - Additional reduction: {ranking_metrics['further_reduction_percentage']:.1f}%")
        print(f"       - Avg semantic score: {ranking_metrics['avg_semantic_score']:.3f}")
        print(f"       - Chunks with answer capability: {ranking_metrics['chunks_with_answer_capability']}")

        return top_chunks, ranking_metrics

    def save_results(self,
                    question_id: str,
                    ranked_chunks: List[Dict],
                    ranking_metrics: Dict,
                    output_dir: str = "Q_Question_Pipeline/outputs"):
        """
        Save simplified semantic ranking results.

        Args:
            question_id: Question identifier
            ranked_chunks: Semantically ranked chunks
            ranking_metrics: Ranking statistics
            output_dir: Output directory
        """
        output_data = {
            'question_id': question_id,
            'stage': 'Q3.1_simplified_semantic_ranking',
            'methodology': 'enhanced_q25_input_only',
            'ranked_chunks': ranked_chunks,
            'ranking_metrics': ranking_metrics,
            'pipeline_improvement': {
                'eliminated_dependencies': ['A3_raw_chunks_no_dedup.json', 'A4_geometric_concept_space.json'],
                'single_input_source': 'Q2.5_enhanced_document_aware_assignment',
                'architectural_benefit': 'self_sufficient_pipeline_stage'
            }
        }

        output_path = os.path.join(output_dir, f"Q3.1_simplified_semantic_ranking_{question_id}.json")

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"[Q3.1-Simplified] Results saved to: {output_path}")


def main():
    """Test simplified Q3.1 semantic ranking on sample question."""

    # Initialize module
    q31_simplified = Q31_SimplifiedSemanticRanking()

    # Test on sample question
    question_id = "finqa_test_1630"

    try:
        # Apply semantic ranking using enhanced Q2.5 output
        ranked_chunks, metrics = q31_simplified.apply_semantic_ranking(question_id)

        # Save results
        q31_simplified.save_results(question_id, ranked_chunks, metrics)

        # Display summary
        print("\n" + "="*70)
        print(f"Q3.1 SIMPLIFIED SEMANTIC RANKING SUMMARY")
        print("="*70)
        print(f"Question ID: {question_id}")
        print(f"Input chunks (from Q2.5): {metrics['input_chunks']}")
        print(f"Output chunks: {metrics['output_chunks']}")
        print(f"Further reduction: {metrics['further_reduction_percentage']:.1f}%")
        print(f"Avg semantic score: {metrics['avg_semantic_score']:.3f}")

        if ranked_chunks:
            print(f"\nTop 3 semantically ranked chunks:")
            for i, chunk in enumerate(ranked_chunks[:3], 1):
                print(f"  {i}. {chunk['chunk_id']}")
                print(f"     Semantic score: {chunk['semantic_score']:.3f}")
                print(f"     Can answer: {chunk['answer_capability']['can_provide_answer']}")
                print(f"     Geometric score: {chunk.get('geometric_score', 0):.3f}")

        print(f"\n" + "="*70)
        print(f"ARCHITECTURAL IMPROVEMENT ACHIEVED")
        print("="*70)
        print(f"[SUCCESS] Eliminated A3 dependency")
        print(f"[SUCCESS] Eliminated A4 dependency")
        print(f"[SUCCESS] Single input: Enhanced Q2.5 output")
        print(f"[SUCCESS] Self-sufficient pipeline stage")

    except Exception as e:
        print(f"Error in Q3.1 simplified semantic ranking: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()