"""
Q3.3: Concept-Based Boosting Module
Third stage of hybrid chunk retrieval - boosts chunks based on concept membership strength
Combines geometric and semantic scores with concept importance
"""

import json
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict


class Q33_ConceptBoosting:
    """
    Concept-based boosting using membership scores and concept importance.
    Final ranking stage combining all signals.
    """

    def __init__(self,
                 a_pipeline_path: str = "A_Concept_pipeline/outputs",
                 q_pipeline_path: str = "Q_Question_Pipeline/outputs"):
        """
        Initialize concept boosting module.

        Args:
            a_pipeline_path: Path to A-Pipeline outputs
            q_pipeline_path: Path to Q-Pipeline outputs
        """
        self.a_pipeline_path = a_pipeline_path
        self.q_pipeline_path = q_pipeline_path
        self.concept_importance = {}

    def load_ranked_chunks(self, question_id: str) -> Tuple[List[Dict], Dict]:
        """
        Load Q3.2 semantic ranking results.

        Args:
            question_id: Question identifier

        Returns:
            Tuple of (ranked_chunks, ranking_metrics)
        """
        q32_path = os.path.join(self.q_pipeline_path, "Q3.2_semantic_ranking.json")

        if not os.path.exists(q32_path):
            raise FileNotFoundError(f"Q3.2 output not found: {q32_path}")

        with open(q32_path, 'r') as f:
            q32_data = json.load(f)

        return q32_data['ranked_chunks'], q32_data['ranking_metrics']

    def calculate_concept_importance(self, doc_id: str) -> Dict[str, float]:
        """
        Calculate importance scores for each concept based on document analysis.

        Args:
            doc_id: Document identifier

        Returns:
            Dictionary of concept_id -> importance_score
        """
        # Load A3 chunks to analyze concept distribution
        chunks_path = os.path.join(self.a_pipeline_path, "A3_raw_chunks_no_dedup.json")

        if not os.path.exists(chunks_path):
            print(f"[Q3.3] WARNING: Could not load chunks for concept importance")
            return {}

        with open(chunks_path, 'r') as f:
            chunks_data = json.load(f)

        # Count concept occurrences across chunks
        concept_counts = defaultdict(int)
        concept_avg_scores = defaultdict(list)

        for chunk in chunks_data.get('chunks', []):
            if chunk.get('doc_id') != doc_id:
                continue

            # Track concept memberships
            for concept in chunk.get('concept_memberships', []):
                concept_counts[concept] += 1

            # Track membership scores
            membership_scores = chunk.get('membership_scores', {})
            for concept, score in membership_scores.items():
                concept_avg_scores[concept].append(score)

        # Calculate importance scores
        importance_scores = {}
        total_chunks = sum(1 for c in chunks_data.get('chunks', []) if c.get('doc_id') == doc_id)

        for concept in concept_counts:
            # Frequency component (how often concept appears)
            frequency = concept_counts[concept] / total_chunks if total_chunks > 0 else 0

            # Strength component (average membership score)
            avg_score = np.mean(concept_avg_scores[concept]) if concept_avg_scores[concept] else 0

            # Combined importance
            importance_scores[concept] = 0.5 * frequency + 0.5 * avg_score

        self.concept_importance = importance_scores
        return importance_scores

    def boost_chunk_scores(self, chunks: List[Dict], doc_id: str) -> List[Dict]:
        """
        Apply concept-based boosting to chunk scores.

        Args:
            chunks: Semantically ranked chunks
            doc_id: Document identifier

        Returns:
            Chunks with boosted scores
        """
        # Calculate concept importance if not already done
        if not self.concept_importance:
            self.concept_importance = self.calculate_concept_importance(doc_id)

        print(f"[Q3.3] Applying concept boosting to {len(chunks)} chunks")

        for chunk in chunks:
            # Get chunk's concept memberships and scores
            concept_memberships = chunk.get('concept_memberships', [])
            membership_scores = chunk.get('membership_scores', {})

            # Calculate concept boost
            concept_boost = 0.0
            boost_components = []

            for concept in concept_memberships:
                # Get concept importance
                importance = self.concept_importance.get(concept, 0.5)

                # Get membership strength
                membership_strength = membership_scores.get(concept, 0.5)

                # Calculate boost contribution
                contribution = importance * membership_strength
                concept_boost += contribution

                boost_components.append({
                    'concept': concept,
                    'importance': importance,
                    'membership': membership_strength,
                    'contribution': contribution
                })

            # Normalize boost (average across concepts)
            if concept_memberships:
                concept_boost /= len(concept_memberships)

            # Store boost information
            chunk['concept_boost'] = concept_boost
            chunk['boost_components'] = boost_components

            # Calculate final hybrid score
            # Ensure scores are floats (handle potential JSON serialization issues)
            geometric_score = float(chunk.get('geometric_score', 0.0))
            semantic_score = float(chunk.get('semantic_score', 0.0))

            # Weighted combination
            chunk['hybrid_score'] = (
                0.3 * geometric_score +      # Spatial proximity
                0.4 * semantic_score +        # Semantic similarity
                0.3 * concept_boost          # Concept importance
            )

        return chunks

    def apply_final_ranking(self,
                           question_id: str,
                           doc_id: str,
                           top_k: int = 5) -> Tuple[List[Dict], Dict]:
        """
        Apply concept-based boosting and produce final ranking.

        Args:
            question_id: Question identifier
            doc_id: Document identifier
            top_k: Number of top chunks for final selection

        Returns:
            Tuple of (final_ranked_chunks, boosting_metrics)
        """
        print(f"\n[Q3.3] Starting concept-based boosting for {question_id}")

        # Load Q3.2 ranked chunks
        ranked_chunks, ranking_metrics = self.load_ranked_chunks(question_id)
        print(f"[Q3.3] Loaded {len(ranked_chunks)} semantically ranked chunks")

        if not ranked_chunks:
            print("[Q3.3] WARNING: No chunks to boost")
            empty_metrics = {
                'total_chunks_boosted': 0,
                'final_chunks_selected': 0,
                'avg_hybrid_score': 0.0,
                'max_hybrid_score': 0.0,
                'avg_concept_boost': 0.0,
                'unique_concepts': 0,
                'concept_boost_applied': False,
                'boosting_method': 'none',
                'score_components': {
                    'geometric_weight': 0.3,
                    'semantic_weight': 0.4,
                    'concept_weight': 0.3
                },
                'timestamp': datetime.now().isoformat(),
                'error': 'No chunks from semantic ranking'
            }
            return [], empty_metrics

        # Apply concept boosting
        boosted_chunks = self.boost_chunk_scores(ranked_chunks, doc_id)

        # Final ranking by hybrid score
        boosted_chunks.sort(key=lambda x: x.get('hybrid_score', 0), reverse=True)

        # Select top K for final answer generation
        final_chunks = boosted_chunks[:top_k]

        # Calculate boosting metrics
        boosting_metrics = {
            'total_chunks_processed': len(boosted_chunks),
            'final_chunks_selected': len(final_chunks),
            'avg_hybrid_score': np.mean([c.get('hybrid_score', 0) for c in final_chunks]) if final_chunks else 0,
            'max_hybrid_score': max([c.get('hybrid_score', 0) for c in boosted_chunks]) if boosted_chunks else 0,
            'avg_concept_boost': np.mean([c.get('concept_boost', 0) for c in final_chunks]) if final_chunks else 0,
            'unique_concepts': len(set(c for chunk in final_chunks for c in chunk.get('concept_memberships', []))),
            'score_components': {
                'geometric_weight': 0.3,
                'semantic_weight': 0.4,
                'concept_weight': 0.3
            },
            'timestamp': datetime.now().isoformat()
        }

        print(f"[Q3.3] Concept boosting complete:")
        print(f"       - Final {top_k} chunks selected")
        print(f"       - Avg hybrid score: {boosting_metrics['avg_hybrid_score']:.3f}")
        print(f"       - Avg concept boost: {boosting_metrics['avg_concept_boost']:.3f}")
        print(f"       - Unique concepts covered: {boosting_metrics['unique_concepts']}")

        return final_chunks, boosting_metrics

    def generate_retrieval_summary(self,
                                   final_chunks: List[Dict],
                                   boosting_metrics: Dict) -> Dict:
        """
        Generate comprehensive summary of the retrieval process.

        Args:
            final_chunks: Final selected chunks
            boosting_metrics: Boosting statistics

        Returns:
            Summary dictionary
        """
        summary = {
            'retrieval_pipeline': 'Q3_Hybrid_Retrieval',
            'stages_completed': ['Q3.1_geometric_filtering', 'Q3.2_semantic_ranking', 'Q3.3_concept_boosting'],
            'final_selection': {
                'num_chunks': len(final_chunks),
                'chunk_ids': [c.get('chunk_id', '') for c in final_chunks],
                'avg_scores': {
                    'geometric': np.mean([float(c.get('geometric_score', 0)) for c in final_chunks]) if final_chunks else 0,
                    'semantic': np.mean([float(c.get('semantic_score', 0)) for c in final_chunks]) if final_chunks else 0,
                    'concept': np.mean([float(c.get('concept_boost', 0)) for c in final_chunks]) if final_chunks else 0,
                    'hybrid': boosting_metrics['avg_hybrid_score']
                }
            },
            'top_chunk_analysis': []
        }

        # Analyze top chunks
        for i, chunk in enumerate(final_chunks[:3], 1):
            chunk_analysis = {
                'rank': i,
                'chunk_id': chunk.get('chunk_id', ''),
                'hybrid_score': chunk.get('hybrid_score', 0),
                'shared_convex_balls': chunk.get('shared_convex_balls', []),
                'can_answer': chunk.get('answer_capability', {}).get('can_provide_answer', False),
                'content_preview': chunk.get('content', '')[:150] + '...'
            }
            summary['top_chunk_analysis'].append(chunk_analysis)

        return summary

    def save_results(self,
                    question_id: str,
                    final_chunks: List[Dict],
                    boosting_metrics: Dict,
                    retrieval_summary: Dict,
                    output_dir: str = "Q_Question_Pipeline/outputs"):
        """
        Save concept boosting and final retrieval results.

        Args:
            question_id: Question identifier
            final_chunks: Final ranked chunks
            boosting_metrics: Boosting statistics
            retrieval_summary: Comprehensive retrieval summary
            output_dir: Output directory
        """
        output_data = {
            'question_id': question_id,
            'stage': 'Q3.3_concept_boosting',
            'final_ranked_chunks': final_chunks,
            'boosting_metrics': boosting_metrics,
            'retrieval_summary': retrieval_summary
        }

        # Save detailed results
        output_path = os.path.join(output_dir, "Q3.3_concept_boosting.json")
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)

        # Save final Q3 results (combined)
        q3_final_path = os.path.join(output_dir, f"Q3_final_retrieval_{question_id}.json")
        q3_final_data = {
            'question_id': question_id,
            'pipeline': 'Q3_Hybrid_Retrieval',
            'final_chunks': final_chunks,
            'summary': retrieval_summary,
            'timestamp': datetime.now().isoformat()
        }
        with open(q3_final_path, 'w') as f:
            json.dump(q3_final_data, f, indent=2, default=str)

        print(f"[Q3.3] Results saved to: {output_path}")
        print(f"[Q3.3] Final Q3 results saved to: {q3_final_path}")


def main():
    """Test Q3.3 concept boosting on sample question."""

    # Initialize module with correct paths
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    a_pipeline_path = os.path.join(base_path, "A_Concept_pipeline", "outputs")
    q_pipeline_path = os.path.join(base_path, "outputs")

    q33 = Q33_ConceptBoosting(
        a_pipeline_path=a_pipeline_path,
        q_pipeline_path=q_pipeline_path
    )

    # Test on sample question
    question_id = "finqa_test_1630"
    doc_id = "finqa_test_1630"

    try:
        # Apply concept boosting
        final_chunks, metrics = q33.apply_final_ranking(question_id, doc_id)

        # Generate summary
        summary = q33.generate_retrieval_summary(final_chunks, metrics)

        # Save results
        q33.save_results(question_id, final_chunks, metrics, summary, q_pipeline_path)

        # Display summary
        print("\n" + "="*60)
        print(f"Q3.3 CONCEPT BOOSTING SUMMARY")
        print("="*60)
        print(f"Question: {question_id}")
        print(f"Final chunks selected: {metrics['final_chunks_selected']}")
        print(f"Avg hybrid score: {metrics['avg_hybrid_score']:.3f}")
        print(f"Unique concepts: {metrics['unique_concepts']}")

        if final_chunks:
            print(f"\nTop 3 final chunks:")
            for i, chunk in enumerate(final_chunks[:3], 1):
                print(f"  {i}. {chunk['chunk_id']}")
                print(f"     Hybrid score: {float(chunk.get('hybrid_score', 0)):.3f}")
                print(f"     - Geometric: {float(chunk.get('geometric_score', 0)):.3f}")
                print(f"     - Semantic: {float(chunk.get('semantic_score', 0)):.3f}")
                print(f"     - Concept boost: {float(chunk.get('concept_boost', 0)):.3f}")

    except Exception as e:
        print(f"Error in Q3.3 concept boosting: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()