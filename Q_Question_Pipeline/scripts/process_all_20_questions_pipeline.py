#!/usr/bin/env python3
"""
Complete Q-Pipeline Processing for 20 Test Questions
Processes all questions through Q1 → Q2.1 → Q2.2 → Q2.3 → Q2.4 → Q2.5

This script ensures all necessary preprocessing is done before Q2.5 convex ball assignment.

Author: Claude (Anthropic)
Date: 2025-09-14
"""

import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all Q-Pipeline modules
from scripts.Q1_question_ingestion import Q1_QuestionIngestion
from scripts.Q2_1_enhanced_intent_layer import Q2_1_EnhancedIntentLayer
from scripts.Q2_2_enhanced_keyword_extraction import Q2_2_EnhancedKeywordExtraction
from scripts.Q2_3_question_structure_analysis import Q2_3_QuestionStructureAnalysis
from scripts.Q2_4_temporal_coordinate_mapping import Q24TemporalCoordinateMapping
from scripts.Q2_5_enhanced_convex_ball_assignment import EnhancedQ25ConvexBallAssignment

class CompletePipelineProcessor:
    """Processes all 20 questions through complete Q-Pipeline"""

    def __init__(self):
        print("Initializing Q-Pipeline processors...")
        self.q1_processor = Q1_QuestionIngestion()
        self.q21_processor = Q2_1_EnhancedIntentLayer()
        self.q22_processor = Q2_2_EnhancedKeywordExtraction()
        self.q23_processor = Q2_3_QuestionStructureAnalysis()
        self.q24_processor = Q24TemporalCoordinateMapping()
        self.q25_processor = EnhancedQ25ConvexBallAssignment()

        self.pipeline_results = {}
        self.statistics = {
            'q1_processed': 0,
            'q21_processed': 0,
            'q22_processed': 0,
            'q23_processed': 0,
            'q24_processed': 0,
            'q25_processed': 0,
            'q25_successful': 0,
            'total_processing_time': 0
        }

    def load_test_questions(self) -> List[Dict]:
        """Load 20 test questions"""
        test_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'outputs', 'Q1_20_records_test_results.json'
        )

        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('processed_questions', [])
        except Exception as e:
            print(f"Error loading test questions: {e}")
            return []

    def process_q1(self, question_data: Dict) -> Dict:
        """Process through Q1 ingestion"""
        try:
            # Save Q1 data for this question
            q1_output = {
                'question_id': question_data['question_id'],
                'question_text': question_data['question_text'],
                'doc_id': question_data['doc_id'],
                'pipeline_ready': True,
                'metadata': question_data.get('metadata', {})
            }

            # Save to Q1 output file (required by Q2.x modules)
            output_file = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'outputs', 'Q1_Question_ingestion.json'
            )

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(q1_output, f, indent=2, ensure_ascii=False)

            self.statistics['q1_processed'] += 1
            return q1_output

        except Exception as e:
            print(f"    Error in Q1: {e}")
            return None

    def process_q21(self, question_id: str) -> Dict:
        """Process through Q2.1 Intent Layer"""
        try:
            result = self.q21_processor.process_question(question_id)
            if 'error' not in result:
                self.statistics['q21_processed'] += 1
            return result
        except Exception as e:
            print(f"    Error in Q2.1: {e}")
            return {'error': str(e)}

    def process_q22(self, question_id: str) -> Dict:
        """Process through Q2.2 Keyword Extraction"""
        try:
            result = self.q22_processor.process_question(question_id)
            if 'error' not in result:
                self.statistics['q22_processed'] += 1
            return result
        except Exception as e:
            print(f"    Error in Q2.2: {e}")
            return {'error': str(e)}

    def process_q23(self, question_id: str) -> Dict:
        """Process through Q2.3 Structure Analysis"""
        try:
            result = self.q23_processor.process_question(question_id)
            if 'error' not in result:
                self.statistics['q23_processed'] += 1
            return result
        except Exception as e:
            print(f"    Error in Q2.3: {e}")
            return {'error': str(e)}

    def process_q24(self, question_id: str) -> Dict:
        """Process through Q2.4 Temporal Mapping"""
        try:
            result = self.q24_processor.process_question(question_id)
            if 'error' not in result:
                self.statistics['q24_processed'] += 1
            return result
        except Exception as e:
            print(f"    Error in Q2.4: {e}")
            return {'error': str(e)}

    def process_q25(self, question_id: str) -> Dict:
        """Process through Q2.5 Convex Ball Assignment"""
        try:
            result = self.q25_processor.process_question(question_id)
            if 'error' not in result:
                self.statistics['q25_processed'] += 1
                if result.get('assignment_confidence', 0) > 0:
                    self.statistics['q25_successful'] += 1
            return result
        except Exception as e:
            print(f"    Error in Q2.5: {e}")
            return {'error': str(e)}

    def process_single_question(self, question_data: Dict) -> Dict:
        """Process single question through complete pipeline"""
        question_id = question_data['question_id']
        print(f"\nProcessing {question_id}:")
        print(f"  Question: {question_data['question_text'][:60]}...")

        results = {'question_id': question_id}

        # Q1 Ingestion
        print("  [Q1] Ingesting question...")
        q1_result = self.process_q1(question_data)
        if not q1_result:
            results['error'] = "Q1 ingestion failed"
            return results
        results['q1'] = q1_result

        # Q2.1 Intent
        print("  [Q2.1] Analyzing intent...")
        results['q21'] = self.process_q21(question_id)

        # Q2.2 Keywords
        print("  [Q2.2] Extracting keywords...")
        results['q22'] = self.process_q22(question_id)

        # Q2.3 Structure
        print("  [Q2.3] Analyzing structure...")
        results['q23'] = self.process_q23(question_id)

        # Q2.4 Temporal
        print("  [Q2.4] Mapping temporal coordinates...")
        results['q24'] = self.process_q24(question_id)

        # Q2.5 Convex Ball Assignment
        print("  [Q2.5] Assigning to convex balls...")
        results['q25'] = self.process_q25(question_id)

        # Check if Q2.5 was successful
        if 'error' not in results['q25']:
            confidence = results['q25'].get('assignment_confidence', 0)
            print(f"  [SUCCESS] Assignment confidence: {confidence:.3f}")
        else:
            print(f"  [FAILED] Q2.5 assignment failed")

        return results

    def generate_report(self):
        """Generate comprehensive pipeline report"""
        print("\n" + "=" * 80)
        print("Q-PIPELINE COMPLETE PROCESSING REPORT")
        print("=" * 80)

        print("\nPipeline Statistics:")
        print(f"  Q1 Processed:  {self.statistics['q1_processed']}/20")
        print(f"  Q2.1 Processed: {self.statistics['q21_processed']}/20")
        print(f"  Q2.2 Processed: {self.statistics['q22_processed']}/20")
        print(f"  Q2.3 Processed: {self.statistics['q23_processed']}/20")
        print(f"  Q2.4 Processed: {self.statistics['q24_processed']}/20")
        print(f"  Q2.5 Processed: {self.statistics['q25_processed']}/20")
        print(f"  Q2.5 Successful Assignments: {self.statistics['q25_successful']}/20")

        success_rate = (self.statistics['q25_successful'] / 20) * 100
        print(f"\nOverall Success Rate: {success_rate:.1f}%")

        # Analyze Q2.5 results
        q25_confidences = []
        for result in self.pipeline_results.values():
            if 'q25' in result and 'error' not in result['q25']:
                confidence = result['q25'].get('assignment_confidence', 0)
                if confidence > 0:
                    q25_confidences.append(confidence)

        if q25_confidences:
            avg_confidence = sum(q25_confidences) / len(q25_confidences)
            print(f"Average Q2.5 Confidence: {avg_confidence:.3f}")
            print(f"Max Confidence: {max(q25_confidences):.3f}")
            print(f"Min Confidence: {min(q25_confidences):.3f}")

        print("\n" + "=" * 80)
        print("Q2.5 READINESS ASSESSMENT")
        print("=" * 80)

        if success_rate >= 80:
            print("[READY] Q2.5 is ready to assign questions into convex balls!")
            print("The pipeline successfully processes questions through all stages.")
        elif success_rate >= 50:
            print("[PARTIALLY READY] Q2.5 shows promise but needs optimization.")
            print("Some questions are successfully assigned, but coverage needs improvement.")
        else:
            print("[NOT READY] Q2.5 requires further development.")
            print("The pipeline needs debugging to handle more questions successfully.")

        return success_rate

    def save_results(self):
        """Save complete pipeline results"""
        output_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'outputs', 'complete_pipeline_results.json'
        )

        output_data = {
            'test_metadata': {
                'test_date': datetime.now().isoformat(),
                'test_type': 'complete_q_pipeline_20_questions',
                'total_questions': 20
            },
            'statistics': self.statistics,
            'results': self.pipeline_results
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)

        print(f"\nComplete results saved to: {output_file}")

    def run_pipeline(self):
        """Run complete pipeline on all 20 questions"""
        print("\n" + "=" * 80)
        print("STARTING COMPLETE Q-PIPELINE PROCESSING")
        print("Processing 20 questions through Q1 -> Q2.1 -> Q2.2 -> Q2.3 -> Q2.4 -> Q2.5")
        print("=" * 80)

        start_time = time.time()

        # Load test questions
        test_questions = self.load_test_questions()
        if not test_questions:
            print("ERROR: No test questions found!")
            return 0

        print(f"\nLoaded {len(test_questions)} test questions")

        # Process each question
        for i, question_data in enumerate(test_questions, 1):
            print(f"\n[{i}/20] " + "-" * 70)
            result = self.process_single_question(question_data)
            self.pipeline_results[question_data['question_id']] = result

        self.statistics['total_processing_time'] = time.time() - start_time

        # Generate report
        success_rate = self.generate_report()

        # Save results
        self.save_results()

        print(f"\nTotal processing time: {self.statistics['total_processing_time']:.1f} seconds")

        return success_rate

def main():
    """Main execution"""
    processor = CompletePipelineProcessor()
    success_rate = processor.run_pipeline()

    return 0 if success_rate >= 50 else 1

if __name__ == "__main__":
    sys.exit(main())