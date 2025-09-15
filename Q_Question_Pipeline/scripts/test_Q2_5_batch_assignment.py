#!/usr/bin/env python3
"""
Comprehensive Q2.5 Batch Testing Script
Tests question-to-convex-ball assignment for all 20 test questions

This script:
1. Loads all 20 test questions from Q1 output
2. Processes each through the complete Q2.x pipeline
3. Performs convex ball assignment using Q2.5 enhanced
4. Generates comprehensive analysis report
5. Validates assignment quality metrics

Author: Claude (Anthropic)
Date: 2025-09-14
"""

import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Q-Pipeline modules
from scripts.Q2_5_enhanced_convex_ball_assignment import EnhancedQ25ConvexBallAssignment

class Q25BatchTester:
    """Batch testing for Q2.5 convex ball assignment"""

    def __init__(self):
        self.q25_processor = EnhancedQ25ConvexBallAssignment()
        self.results = []
        self.statistics = {
            'total_processed': 0,
            'successful_assignments': 0,
            'failed_assignments': 0,
            'avg_confidence': 0.0,
            'containment_types': {},
            'fallback_usage': {},
            'dimension_performance': {}
        }

    def load_test_questions(self) -> List[Dict]:
        """Load 20 test questions from Q1 output"""
        q1_test_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'outputs', 'Q1_20_records_test_results.json'
        )

        try:
            with open(q1_test_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('processed_questions', [])
        except Exception as e:
            print(f"Error loading test questions: {e}")
            return []

    def process_question(self, question_data: Dict) -> Dict:
        """Process single question through Q2.5"""
        question_id = question_data['question_id']

        print(f"\n  Processing: {question_id}")
        print(f"    Question: {question_data['question_text'][:80]}...")

        start_time = time.time()

        try:
            # Process through Q2.5
            result = self.q25_processor.process_question(question_id)

            if 'error' not in result:
                processing_time = (time.time() - start_time) * 1000
                result['batch_test_metadata'] = {
                    'processing_time_ms': processing_time,
                    'question_text': question_data['question_text'],
                    'doc_id': question_data['doc_id']
                }

                # Extract key metrics
                self.update_statistics(result)

                print(f"    [SUCCESS] Assignment successful (confidence: {result.get('assignment_confidence', 0):.3f})")
                return result
            else:
                print(f"    [FAILED] Assignment failed: {result.get('error', 'Unknown error')}")
                return result

        except Exception as e:
            print(f"    [ERROR] Processing error: {e}")
            return {
                'question_id': question_id,
                'error': str(e),
                'batch_test_metadata': {
                    'question_text': question_data['question_text']
                }
            }

    def update_statistics(self, result: Dict):
        """Update batch statistics from result"""
        if 'error' not in result:
            self.statistics['successful_assignments'] += 1

            # Update confidence
            confidence = result.get('assignment_confidence', 0)
            current_avg = self.statistics['avg_confidence']
            n = self.statistics['successful_assignments']
            self.statistics['avg_confidence'] = ((n-1) * current_avg + confidence) / n

            # Track multi-dimensional analysis
            multi_dim = result.get('multi_dimensional_analysis', {})
            for dim_type, dim_analysis in multi_dim.items():
                if dim_type not in self.statistics['dimension_performance']:
                    self.statistics['dimension_performance'][dim_type] = {
                        'total_balls_assigned': 0,
                        'avg_membership': 0,
                        'containment_count': 0,
                        'fallback_count': 0
                    }

                stats = dim_analysis.get('membership_statistics', {})
                dim_perf = self.statistics['dimension_performance'][dim_type]

                dim_perf['total_balls_assigned'] += stats.get('total_balls_assigned', 0)
                if stats.get('avg_membership_strength', 0) > 0:
                    dim_perf['avg_membership'] = (
                        (dim_perf['avg_membership'] * (n-1) + stats.get('avg_membership_strength', 0)) / n
                    )

                if dim_analysis.get('containment_status') == 'contained':
                    dim_perf['containment_count'] += 1

                if dim_analysis.get('fallback_applied'):
                    dim_perf['fallback_count'] += 1

            # Track fusion strategy
            fusion = result.get('fusion_analysis', {})
            strategy = fusion.get('fusion_strategy', 'unknown')
            self.statistics['containment_types'][strategy] = \
                self.statistics['containment_types'].get(strategy, 0) + 1

        else:
            self.statistics['failed_assignments'] += 1

    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 80)
        print("Q2.5 BATCH TESTING REPORT - QUESTION TO CONVEX BALL ASSIGNMENT")
        print("=" * 80)

        print(f"\nTest Summary:")
        print(f"  Total Questions Processed: {self.statistics['total_processed']}")
        print(f"  Successful Assignments: {self.statistics['successful_assignments']}")
        print(f"  Failed Assignments: {self.statistics['failed_assignments']}")
        print(f"  Success Rate: {(self.statistics['successful_assignments'] / max(1, self.statistics['total_processed'])) * 100:.1f}%")
        print(f"  Average Confidence: {self.statistics['avg_confidence']:.3f}")

        print(f"\nDimensional Performance:")
        for dim_type, perf in self.statistics['dimension_performance'].items():
            print(f"  {dim_type}:")
            print(f"    - Total balls assigned: {perf['total_balls_assigned']}")
            print(f"    - Avg membership strength: {perf['avg_membership']:.3f}")
            print(f"    - Direct containment rate: {(perf['containment_count'] / max(1, self.statistics['successful_assignments'])) * 100:.1f}%")
            print(f"    - Fallback usage rate: {(perf['fallback_count'] / max(1, self.statistics['successful_assignments'])) * 100:.1f}%")

        print(f"\nFusion Strategies Used:")
        for strategy, count in self.statistics['containment_types'].items():
            percentage = (count / max(1, self.statistics['successful_assignments'])) * 100
            print(f"  {strategy}: {count} ({percentage:.1f}%)")

        # Identify best and worst performing questions
        if self.results:
            sorted_results = sorted(
                [r for r in self.results if 'error' not in r],
                key=lambda x: x.get('assignment_confidence', 0),
                reverse=True
            )

            if sorted_results:
                print(f"\nTop 3 High-Confidence Assignments:")
                for i, result in enumerate(sorted_results[:3], 1):
                    meta = result.get('batch_test_metadata', {})
                    print(f"  {i}. {result['question_id']} (confidence: {result.get('assignment_confidence', 0):.3f})")
                    print(f"     Question: {meta.get('question_text', '')[:60]}...")

                print(f"\nBottom 3 Low-Confidence Assignments:")
                for i, result in enumerate(sorted_results[-3:], 1):
                    meta = result.get('batch_test_metadata', {})
                    print(f"  {i}. {result['question_id']} (confidence: {result.get('assignment_confidence', 0):.3f})")
                    print(f"     Question: {meta.get('question_text', '')[:60]}...")

        print("\n" + "=" * 80)
        print("ASSESSMENT: Q2.5 READINESS STATUS")
        print("=" * 80)

        # Readiness assessment
        readiness_score = 0
        readiness_criteria = []

        # Check success rate
        success_rate = (self.statistics['successful_assignments'] / max(1, self.statistics['total_processed'])) * 100
        if success_rate >= 90:
            readiness_score += 25
            readiness_criteria.append("[PASS] High success rate (>90%)")
        elif success_rate >= 70:
            readiness_score += 15
            readiness_criteria.append("[MODERATE] Moderate success rate (70-90%)")
        else:
            readiness_criteria.append("[FAIL] Low success rate (<70%)")

        # Check confidence levels
        if self.statistics['avg_confidence'] >= 0.7:
            readiness_score += 25
            readiness_criteria.append("[PASS] High average confidence (>0.7)")
        elif self.statistics['avg_confidence'] >= 0.5:
            readiness_score += 15
            readiness_criteria.append("[MODERATE] Moderate average confidence (0.5-0.7)")
        else:
            readiness_criteria.append("[FAIL] Low average confidence (<0.5)")

        # Check dimensional coverage
        active_dimensions = sum(
            1 for perf in self.statistics['dimension_performance'].values()
            if perf['total_balls_assigned'] > 0
        )
        if active_dimensions >= 3:
            readiness_score += 25
            readiness_criteria.append("[PASS] Good dimensional coverage (3+ active)")
        elif active_dimensions >= 2:
            readiness_score += 15
            readiness_criteria.append("[MODERATE] Fair dimensional coverage (2 active)")
        else:
            readiness_criteria.append("[FAIL] Poor dimensional coverage (<2 active)")

        # Check A-Pipeline integration
        if self.statistics['successful_assignments'] > 0:
            readiness_score += 25
            readiness_criteria.append("[PASS] A-Pipeline integration working")
        else:
            readiness_criteria.append("[FAIL] A-Pipeline integration failed")

        print("\nReadiness Criteria:")
        for criterion in readiness_criteria:
            print(f"  {criterion}")

        print(f"\nOverall Readiness Score: {readiness_score}/100")

        if readiness_score >= 80:
            print("\n[READY] RESULT: Q2.5 is READY for production use!")
            print("The system can successfully assign questions to convex balls.")
        elif readiness_score >= 60:
            print("\n[WARNING] RESULT: Q2.5 is PARTIALLY READY")
            print("The system works but may need optimization for better performance.")
        else:
            print("\n[NOT READY] RESULT: Q2.5 NEEDS IMPROVEMENT")
            print("The system requires further development before production use.")

        return readiness_score

    def save_results(self):
        """Save batch test results to file"""
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'outputs'
        )
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, 'Q2.5_batch_test_results.json')

        output_data = {
            'test_metadata': {
                'test_date': datetime.now().isoformat(),
                'test_type': 'Q2.5_comprehensive_batch_test',
                'total_questions': self.statistics['total_processed']
            },
            'statistics': self.statistics,
            'results': self.results
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\nDetailed results saved to: {output_file}")

    def run_batch_test(self):
        """Run complete batch test"""
        print("\n" + "=" * 80)
        print("STARTING Q2.5 BATCH TEST - CONVEX BALL ASSIGNMENT VALIDATION")
        print("=" * 80)

        # Load test questions
        test_questions = self.load_test_questions()

        if not test_questions:
            print("ERROR: No test questions found!")
            return 0

        print(f"\nLoaded {len(test_questions)} test questions")
        print("Processing questions through Q2.5 enhanced convex ball assignment...")

        # Process each question
        for question_data in test_questions:
            self.statistics['total_processed'] += 1
            result = self.process_question(question_data)
            self.results.append(result)

        # Generate report
        readiness_score = self.generate_report()

        # Save results
        self.save_results()

        return readiness_score

def main():
    """Main execution"""
    tester = Q25BatchTester()
    readiness_score = tester.run_batch_test()

    return 0 if readiness_score >= 60 else 1

if __name__ == "__main__":
    sys.exit(main())