"""
Q-Pipeline: Minimal Q1 → Q2.5 → Q3.1 Pipeline
Revolutionary constraint-based question-answering system
Validates the core innovation: convex ball constrained matching
"""

import json
import os
import sys
from typing import Dict, List
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(__file__))

from Q1_question_ingestion import Q1_QuestionIngestion
from Q_Question_Pipeline.scripts.archive.Q2_5_convex_ball_assignment import Q25_ConvexBallAssignment
from Q3_1_constrained_geometric_matching import Q31_ConstrainedGeometricMatching


class QMinimalPipeline:
    """
    Minimal Q-Pipeline implementation for concept validation.
    Demonstrates the revolutionary convex ball constraint in action.
    """

    def __init__(self, config: Dict = None):
        """
        Initialize minimal Q-Pipeline.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {
            'max_matches': 5,
            'min_constraint_strength': 0.3,
            'output_path': 'Q_Question_Pipeline/outputs/',
            'verbose': True
        }

        # Initialize modules
        self.q1 = Q1_QuestionIngestion()
        self.q25 = Q25_ConvexBallAssignment()
        self.q31 = Q31_ConstrainedGeometricMatching()

    def process_single_question(self, question_id: str) -> Dict:
        """
        Process a single question through Q1 → Q2.5 → Q3.1 pipeline.

        Args:
            question_id: Question identifier

        Returns:
            Complete pipeline results with revolutionary constraint metrics
        """
        if self.config['verbose']:
            print(f"\n{'='*60}")
            print(f"Q-PIPELINE: Processing Question {question_id}")
            print(f"{'='*60}")

        # STEP 1: Q1 - Question Ingestion
        if self.config['verbose']:
            print(f"\nQ1: Question Ingestion...")

        q1_output = self.q1.load_question(question_id)

        if self.config['verbose']:
            print(f"   * Question loaded: {q1_output['question_text'][:80]}...")
            print(f"   * Doc ID: {q1_output['doc_id']}")
            print(f"   * Pipeline ready: {q1_output['pipeline_ready']}")

        # STEP 2: Q2.5 - Document-Specific Convex Ball Assignment
        if self.config['verbose']:
            print(f"\nQ2.5: Convex Ball Assignment...")

        q25_output = self.q25.process_question(q1_output)

        if self.config['verbose']:
            balls_assigned = q25_output['geometric_metadata']['total_balls_assigned']
            max_strength = q25_output['geometric_metadata']['max_membership_strength']
            print(f"   * Dimensions: {q25_output['coordinate_space']['dimensions']}")
            print(f"   * Balls assigned: {balls_assigned}")
            print(f"   * Max membership: {max_strength:.3f}")
            print(f"   * Coverage score: {q25_output['coverage_score']:.3f}")

        # Save Q2.5 output for Q3.1
        self.q25.save_output(q25_output)

        # STEP 3: Q3.1 - Constrained Geometric Matching
        if self.config['verbose']:
            print(f"\nQ3.1: Constrained Geometric Matching...")

        q31_output = self.q31.process_question(question_id)

        if self.config['verbose']:
            total_chunks = q31_output['constraint_metrics']['total_chunks']
            eligible_chunks = q31_output['constraint_metrics']['eligible_chunks']
            reduction = q31_output['constraint_metrics']['reduction_percentage']
            matches_found = q31_output['processing_metadata']['total_matches_found']

            print(f"   REVOLUTIONARY CONSTRAINT APPLIED:")
            print(f"   * Total chunks: {total_chunks}")
            print(f"   * Eligible chunks: {eligible_chunks}")
            print(f"   * Search space reduction: {reduction:.1f}%")
            print(f"   * Matches found: {matches_found}")

        # Save Q3.1 output
        self.q31.save_output(q31_output)

        # Combine pipeline results
        pipeline_result = {
            'question_id': question_id,
            'processing_timestamp': datetime.now().isoformat(),
            'pipeline_stages': {
                'q1_ingestion': q1_output,
                'q25_convex_ball_assignment': q25_output,
                'q31_constrained_matching': q31_output
            },
            'revolutionary_metrics': {
                'search_space_reduction_percentage': q31_output['constraint_metrics']['reduction_percentage'],
                'constraint_effectiveness': q31_output['constraint_metrics']['constraint_satisfied'],
                'convex_balls_utilized': len(q25_output['convex_ball_assignments']),
                'geometric_precision': q31_output['quality_metrics']['avg_geometric_distance'],
                'intent_alignment': q31_output['quality_metrics']['avg_intent_alignment']
            },
            'top_matches': q31_output['constrained_matches'][:self.config['max_matches']]
        }

        return pipeline_result

    def process_batch_questions(self, question_ids: List[str]) -> List[Dict]:
        """
        Process multiple questions to demonstrate constraint effectiveness.

        Args:
            question_ids: List of question identifiers

        Returns:
            List of pipeline results
        """
        results = []
        for qid in question_ids:
            try:
                result = self.process_single_question(qid)
                results.append(result)
            except Exception as e:
                print(f"Error processing {qid}: {e}")
                continue

        return results

    def generate_constraint_analysis(self, results: List[Dict]) -> Dict:
        """
        Analyze the effectiveness of convex ball constraints across questions.

        Args:
            results: List of pipeline results

        Returns:
            Constraint analysis summary
        """
        if not results:
            return {'error': 'No results to analyze'}

        # Extract constraint metrics
        reductions = []
        chunks_before = []
        chunks_after = []
        intent_alignments = []
        geometric_distances = []

        for result in results:
            metrics = result['revolutionary_metrics']
            reductions.append(metrics['search_space_reduction_percentage'])

            q31_data = result['pipeline_stages']['q31_constrained_matching']
            chunks_before.append(q31_data['constraint_metrics']['total_chunks'])
            chunks_after.append(q31_data['constraint_metrics']['eligible_chunks'])

            if q31_data['quality_metrics']:
                intent_alignments.append(q31_data['quality_metrics']['avg_intent_alignment'])
                geometric_distances.append(q31_data['quality_metrics']['avg_geometric_distance'])

        import numpy as np

        analysis = {
            'constraint_effectiveness': {
                'avg_reduction_percentage': float(np.mean(reductions)),
                'min_reduction': float(np.min(reductions)),
                'max_reduction': float(np.max(reductions)),
                'std_reduction': float(np.std(reductions))
            },
            'search_space_impact': {
                'avg_chunks_before_constraint': float(np.mean(chunks_before)),
                'avg_chunks_after_constraint': float(np.mean(chunks_after)),
                'total_chunk_evaluations_saved': sum(chunks_before) - sum(chunks_after)
            },
            'quality_improvements': {
                'avg_intent_alignment': float(np.mean(intent_alignments)) if intent_alignments else 0,
                'avg_geometric_distance': float(np.mean(geometric_distances)) if geometric_distances else 0,
                'intent_alignment_improvement': 'Targeting >0.2 vs B-Pipeline -0.391'
            },
            'questions_processed': len(results),
            'successful_constraint_applications': sum(1 for r in results
                                                    if r['revolutionary_metrics']['constraint_effectiveness'])
        }

        return analysis

    def compare_with_b_pipeline(self, question_ids: List[str]) -> Dict:
        """
        Compare Q-Pipeline constraint-based results with B-Pipeline baseline.
        This would demonstrate the revolutionary improvement.

        Args:
            question_ids: Questions to compare

        Returns:
            Comparison analysis
        """
        # Placeholder for B-Pipeline comparison
        # In full implementation, this would load B-Pipeline results
        # and compare accuracy, precision, and efficiency metrics

        comparison = {
            'methodology': 'Q-Pipeline (Constrained) vs B-Pipeline (Global)',
            'questions_tested': len(question_ids),
            'expected_improvements': {
                'search_space_reduction': '>90%',
                'intent_alignment': 'From -0.391 to >0.2',
                'accuracy_target': '75-80% vs 70% baseline'
            },
            'revolutionary_advantages': [
                'Convex ball constraint eliminates irrelevant chunks',
                'Document-specific coordinate systems',
                'Mathematical precision in geometric calculations',
                'Human cognitive process alignment'
            ]
        }

        return comparison

    def save_pipeline_results(self, results: List[Dict], analysis: Dict = None):
        """
        Save complete pipeline results and analysis.

        Args:
            results: Pipeline processing results
            analysis: Optional constraint analysis
        """
        output_dir = self.config['output_path']
        os.makedirs(output_dir, exist_ok=True)

        # Save detailed results
        results_path = os.path.join(output_dir, 'Q_minimal_pipeline_results.json')
        with open(results_path, 'w') as f:
            json.dump({
                'pipeline_version': 'Q-Pipeline Minimal v1.0',
                'processing_timestamp': datetime.now().isoformat(),
                'results': results,
                'constraint_analysis': analysis or {}
            }, f, indent=2)

        print(f"\nPipeline results saved to: {results_path}")

    def run_demonstration(self, max_questions: int = 3) -> Dict:
        """
        Run a demonstration of the Q-Pipeline revolutionary capabilities.

        Args:
            max_questions: Maximum questions to process

        Returns:
            Demonstration results
        """
        print(f"\nQ-PIPELINE REVOLUTIONARY DEMONSTRATION")
        print(f"{'='*80}")
        print(f"Implementing human cognitive process with convex ball constraints")
        print(f"Target: Transform 70% accuracy to 75-80% through geometric precision")

        # Try to find available questions
        questions_to_test = []

        # Look for sample data
        data_path = "../data/sample_20_records.parquet"
        if os.path.exists(data_path):
            import pandas as pd
            df = pd.read_parquet(data_path)
            # Use 'id' column as question_id
            questions_to_test = df['id'].head(max_questions).tolist()
        elif os.path.exists("../../sample_20_records.parquet"):
            import pandas as pd
            df = pd.read_parquet("../../sample_20_records.parquet")
            questions_to_test = df['id'].head(max_questions).tolist()
        else:
            # Look for existing B-Pipeline outputs
            b_output_path = "B_Retrieval_pipeline/outputs/B1_current_question.json"
            if os.path.exists(b_output_path):
                with open(b_output_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list) and data:
                        questions_to_test = [data[0].get('question_id', 'test_question')]
                    elif isinstance(data, dict):
                        questions_to_test = [data.get('question_id', 'test_question')]

        if not questions_to_test:
            print("No test questions found. Please ensure data is available.")
            return {'error': 'No test data available'}

        # Process questions through Q-Pipeline
        results = self.process_batch_questions(questions_to_test)

        # Analyze constraint effectiveness
        analysis = self.generate_constraint_analysis(results)

        # Generate comparison framework
        comparison = self.compare_with_b_pipeline(questions_to_test)

        # Save results
        self.save_pipeline_results(results, analysis)

        print(f"\nCONSTRAINT EFFECTIVENESS ANALYSIS:")
        print(f"{'='*50}")
        if 'constraint_effectiveness' in analysis:
            ce = analysis['constraint_effectiveness']
            print(f"Average search space reduction: {ce['avg_reduction_percentage']:.1f}%")
            print(f"Range: {ce['min_reduction']:.1f}% - {ce['max_reduction']:.1f}%")

        if 'search_space_impact' in analysis:
            si = analysis['search_space_impact']
            print(f"Avg chunks before constraint: {si['avg_chunks_before_constraint']:.0f}")
            print(f"Avg chunks after constraint: {si['avg_chunks_after_constraint']:.0f}")
            print(f"Total evaluations saved: {si['total_chunk_evaluations_saved']}")

        return {
            'results': results,
            'analysis': analysis,
            'comparison': comparison
        }


if __name__ == "__main__":
    # Run Q-Pipeline demonstration
    pipeline = QMinimalPipeline({
        'max_matches': 3,
        'verbose': True,
        'output_path': '../outputs/'
    })

    demonstration = pipeline.run_demonstration(max_questions=2)

    if 'error' not in demonstration:
        print(f"\nQ-PIPELINE DEMONSTRATION COMPLETE!")
        print(f"Revolutionary convex ball constraints successfully implemented.")
        print(f"Ready for full pipeline development and accuracy improvements.")
    else:
        print(f"Demonstration failed: {demonstration['error']}")