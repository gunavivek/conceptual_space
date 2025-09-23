"""
Q3.1: Q2.5-Only Constrained Geometric Matching Module
SIMPLIFIED: Only processes Q2.5 output without accessing other pipeline files
"""

import json
import numpy as np
import os
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict


class Q31_ConstrainedGeometricMatching:
    """
    Simplified constraint-based matching module.
    Only processes Q2.5 output data without external dependencies.
    """

    def __init__(self, q_pipeline_path: str = "../outputs"):
        """
        Initialize constrained geometric matching.

        Args:
            q_pipeline_path: Path to Q-Pipeline outputs (Q2.5 results)
        """
        self.q_pipeline_path = q_pipeline_path

    def load_q25_data(self) -> Dict:
        """
        Load Q2.5 document aware assignment data.

        Returns:
            Complete Q2.5 data structure
        """
        q25_path = os.path.join(self.q_pipeline_path, "Q2.5_document_aware_assignment.json")

        if not os.path.exists(q25_path):
            raise FileNotFoundError(f"Q2.5 output not found: {q25_path}")

        with open(q25_path, 'r') as f:
            q25_data = json.load(f)

        # Handle nested structure: data is in question_results
        if 'question_results' in q25_data:
            return q25_data['question_results']
        else:
            return q25_data

    def extract_convex_ball_constraints(self, question_data: Dict) -> Dict:
        """
        Extract convex ball constraints from Q2.5 question data.

        Args:
            question_data: Single question data from Q2.5

        Returns:
            Constraint analysis for the question
        """
        # Extract convex ball assignments from nested structure
        assignments = question_data.get('multi_dimensional_analysis', {}).get(
            'document_aware_assignment', {}).get('convex_ball_assignments', [])

        if not assignments:
            return {
                'has_constraints': False,
                'constraint_balls': [],
                'constraint_strength': 0.0,
                'message': 'No convex ball assignments found'
            }

        # Analyze constraint strength
        strong_assignments = [a for a in assignments if a.get('membership_strength', 0) >= 0.3]
        constraint_balls = [a['ball_id'] for a in strong_assignments]

        avg_strength = np.mean([a.get('membership_strength', 0) for a in strong_assignments]) if strong_assignments else 0

        return {
            'has_constraints': len(strong_assignments) > 0,
            'constraint_balls': constraint_balls,
            'total_assignments': len(assignments),
            'strong_assignments': len(strong_assignments),
            'constraint_strength': float(avg_strength),
            'assignments': strong_assignments,
            'all_assignments': assignments
        }

    def analyze_geometric_filtering_potential(self, question_data: Dict) -> Dict:
        """
        Analyze the geometric filtering potential from Q2.5 data.

        Args:
            question_data: Single question data from Q2.5

        Returns:
            Analysis of geometric filtering potential
        """
        # Extract geometric filtering data from Q2.5
        geometric_data = question_data.get('multi_dimensional_analysis', {}).get(
            'geometric_filtering', {})

        if not geometric_data:
            return {
                'has_geometric_filtering': False,
                'message': 'No geometric filtering data found in Q2.5'
            }

        return {
            'has_geometric_filtering': True,
            'original_chunks': geometric_data.get('original_chunks', 0),
            'filtered_chunks': geometric_data.get('filtered_chunks', 0),
            'reduction_percentage': geometric_data.get('reduction_percentage', 0),
            'chunks_per_ball': geometric_data.get('chunks_per_ball', {}),
            'filtering_effectiveness': geometric_data.get('reduction_percentage', 0) > 50
        }

    def simulate_constrained_matching(self, constraint_data: Dict, geometric_data: Dict) -> Dict:
        """
        Simulate constrained geometric matching based on Q2.5 data.

        Args:
            constraint_data: Convex ball constraint analysis
            geometric_data: Geometric filtering analysis

        Returns:
            Simulated matching results
        """
        if not constraint_data['has_constraints']:
            return {
                'matching_feasible': False,
                'reason': 'No strong convex ball constraints available',
                'simulation_quality': 'poor'
            }

        if not geometric_data['has_geometric_filtering']:
            return {
                'matching_feasible': False,
                'reason': 'No geometric filtering data available from Q2.5',
                'simulation_quality': 'poor'
            }

        # Simulate matching effectiveness
        constraint_strength = constraint_data['constraint_strength']
        reduction_effectiveness = geometric_data['reduction_percentage'] / 100.0

        # Calculate simulated matching metrics
        simulated_precision = min(0.95, constraint_strength + reduction_effectiveness * 0.3)
        simulated_efficiency = reduction_effectiveness
        simulated_quality = (simulated_precision + simulated_efficiency) / 2

        return {
            'matching_feasible': True,
            'constraint_strength': constraint_strength,
            'geometric_reduction': reduction_effectiveness,
            'simulated_precision': float(simulated_precision),
            'simulated_efficiency': float(simulated_efficiency),
            'simulated_quality': float(simulated_quality),
            'constraint_balls': constraint_data['constraint_balls'],
            'chunks_per_ball': geometric_data.get('chunks_per_ball', {}),
            'simulation_quality': 'excellent' if simulated_quality > 0.8 else 'good' if simulated_quality > 0.6 else 'fair'
        }

    def process_question(self, question_id: str, question_data: Dict) -> Dict:
        """
        Complete Q3.1 processing for a question using only Q2.5 data.

        Args:
            question_id: Question identifier
            question_data: Question data from Q2.5

        Returns:
            Complete Q3.1 analysis based on Q2.5 constraints
        """
        # Extract convex ball constraints
        constraint_analysis = self.extract_convex_ball_constraints(question_data)

        # Analyze geometric filtering potential
        geometric_analysis = self.analyze_geometric_filtering_potential(question_data)

        # Simulate constrained matching
        matching_simulation = self.simulate_constrained_matching(constraint_analysis, geometric_analysis)

        return {
            'question_id': question_id,
            'doc_id': question_data.get('doc_id', 'unknown'),
            'question_text': question_data.get('question_text', ''),
            'constraint_analysis': constraint_analysis,
            'geometric_analysis': geometric_analysis,
            'matching_simulation': matching_simulation,
            'q31_recommendation': self._generate_recommendation(constraint_analysis, geometric_analysis, matching_simulation),
            'processing_metadata': {
                'input_source': 'Q2.5_document_aware_assignment',
                'processing_mode': 'Q2.5_only_simulation',
                'dependencies': ['Q2.5'],
                'constraint_driven': constraint_analysis['has_constraints'],
                'geometric_enabled': geometric_analysis['has_geometric_filtering']
            }
        }

    def _generate_recommendation(self, constraint_analysis: Dict, geometric_analysis: Dict,
                               matching_simulation: Dict) -> Dict:
        """
        Generate Q3.1 processing recommendation based on analysis.

        Args:
            constraint_analysis: Convex ball constraint analysis
            geometric_analysis: Geometric filtering analysis
            matching_simulation: Matching simulation results

        Returns:
            Processing recommendation
        """
        if not constraint_analysis['has_constraints']:
            return {
                'recommendation': 'fallback_to_traditional_matching',
                'reason': 'Insufficient convex ball constraints',
                'confidence': 'low',
                'next_steps': ['Enhance Q2.5 ball assignment', 'Consider broader constraint thresholds']
            }

        if not geometric_analysis['has_geometric_filtering']:
            return {
                'recommendation': 'develop_geometric_filtering',
                'reason': 'Missing geometric filtering from Q2.5',
                'confidence': 'medium',
                'next_steps': ['Implement geometric filtering in Q2.5', 'Add spatial indexing']
            }

        quality = matching_simulation['simulation_quality']

        if quality == 'excellent':
            return {
                'recommendation': 'proceed_with_constrained_matching',
                'reason': f'High constraint quality ({matching_simulation["simulated_quality"]:.3f})',
                'confidence': 'high',
                'expected_performance': f'{matching_simulation["geometric_reduction"]*100:.1f}% reduction, {matching_simulation["simulated_precision"]*100:.1f}% precision'
            }
        elif quality == 'good':
            return {
                'recommendation': 'proceed_with_monitoring',
                'reason': f'Adequate constraint quality ({matching_simulation["simulated_quality"]:.3f})',
                'confidence': 'medium',
                'monitoring_needed': ['Precision tracking', 'Constraint effectiveness']
            }
        else:
            return {
                'recommendation': 'optimize_constraints_first',
                'reason': f'Suboptimal constraint quality ({matching_simulation["simulated_quality"]:.3f})',
                'confidence': 'low',
                'optimization_needed': ['Strengthen ball assignments', 'Improve geometric filtering']
            }

    def save_output(self, output_data: Dict, output_path: str = None):
        """
        Save Q3.1 output for downstream modules.

        Args:
            output_data: Q3.1 processing results
            output_path: Output file path
        """
        if output_path is None:
            output_path = os.path.join(self.q_pipeline_path, "Q3.1_constrained_geometric_matches.json")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Load existing data
        existing_data = {}
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                existing_data = json.load(f)

        # Add/update this question's results
        existing_data[output_data['question_id']] = output_data

        with open(output_path, 'w') as f:
            json.dump(existing_data, f, indent=2)

        print(f"Q3.1 output saved to {output_path}")


if __name__ == "__main__":
    # Test Q3.1 module with Q2.5-only processing
    print("="*60)
    print("Q3.1: Q2.5-Only Constrained Geometric Matching Test")
    print("="*60)

    q31 = Q31_ConstrainedGeometricMatching()

    try:
        # Load Q2.5 data
        q25_data = q31.load_q25_data()
        print(f"Loaded Q2.5 data with {len(q25_data)} questions")

        # Process first question
        first_question_id = list(q25_data.keys())[0]
        question_data = q25_data[first_question_id]

        print(f"\nProcessing question: {first_question_id}")
        print(f"Question: {question_data.get('question_text', 'N/A')}")

        # Process through Q3.1
        q31_output = q31.process_question(first_question_id, question_data)

        print(f"\n{'='*40}")
        print("Q3.1 OUTPUT - Constraint Analysis:")
        print(f"{'='*40}")

        constraint_analysis = q31_output['constraint_analysis']
        print(f"Has Constraints: {constraint_analysis['has_constraints']}")
        print(f"Strong Assignments: {constraint_analysis['strong_assignments']}")
        print(f"Constraint Strength: {constraint_analysis['constraint_strength']:.3f}")
        print(f"Constraint Balls: {constraint_analysis['constraint_balls']}")

        geometric_analysis = q31_output['geometric_analysis']
        print(f"\nGeometric Filtering: {geometric_analysis['has_geometric_filtering']}")
        if geometric_analysis['has_geometric_filtering']:
            print(f"Reduction: {geometric_analysis['reduction_percentage']:.1f}%")
            print(f"Chunks per ball: {geometric_analysis['chunks_per_ball']}")

        matching_simulation = q31_output['matching_simulation']
        print(f"\nMatching Feasible: {matching_simulation['matching_feasible']}")
        if matching_simulation['matching_feasible']:
            print(f"Simulated Quality: {matching_simulation['simulation_quality']}")
            print(f"Simulated Precision: {matching_simulation['simulated_precision']:.3f}")
            print(f"Simulated Efficiency: {matching_simulation['simulated_efficiency']:.3f}")

        recommendation = q31_output['q31_recommendation']
        print(f"\nRecommendation: {recommendation['recommendation']}")
        print(f"Reason: {recommendation['reason']}")
        print(f"Confidence: {recommendation['confidence']}")

        # Save output
        q31.save_output(q31_output)

        print(f"\n{'='*40}")
        print("Q3.1 Processing Complete")
        print(f"{'='*40}")

    except Exception as e:
        print(f"Error in Q3.1 testing: {e}")
        import traceback
        traceback.print_exc()