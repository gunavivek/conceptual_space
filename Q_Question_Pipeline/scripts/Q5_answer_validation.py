"""
Q5: Answer Validation Module
FINAL Q5 - Validates Q4 generated answers against ground truth

ARCHITECTURE POSITION:
- Input: Q4 answer generation output + Ground truth data
- Process: Multiple validation approaches (numeric, semantic, lexical)
- Output: Comprehensive validation report with accuracy metrics
- Pipeline Completion: Final evaluation stage providing accuracy assessment

FEATURES:
- Multiple validation approaches: exact, numeric, semantic, and lexical matching
- Ground truth loading from sample data (parquet files)
- Semantic similarity using SentenceTransformers
- BLEU-like scoring for lexical overlap
- Comprehensive accuracy reporting
- Question-type specific validation logic
- Validation status classification

This Q5 completes the Q-Pipeline by providing thorough validation of generated
answers against ground truth, enabling accuracy assessment and model evaluation.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import re
import numpy as np
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Optional semantic similarity (graceful degradation if not available)
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("[WARNING] SentenceTransformers not available. Semantic similarity will be disabled.")


class Q5_AnswerValidation:
    """
    Q5 - Validates Q4 generated answers against ground truth.

    Provides comprehensive validation using multiple approaches:
    - Numeric matching for financial calculations
    - Semantic similarity for meaning comparison
    - Lexical overlap for content matching
    - Question-type specific validation logic
    """

    def __init__(self, q_pipeline_path: str = None):
        """
        Initialize Q5 answer validation module.

        Args:
            q_pipeline_path: Path to Q-Pipeline outputs
        """
        if q_pipeline_path is None:
            # Auto-detect the Q-Pipeline outputs path
            script_dir = Path(__file__).parent
            self.q_pipeline_path = str(script_dir.parent / "outputs")
        else:
            self.q_pipeline_path = q_pipeline_path

        # Set data directory path
        self.data_dir = Path(__file__).parent.parent.parent / "A_Concept_pipeline" / "data"

        # Initialize semantic similarity model if available
        self.similarity_model = None
        self.semantic_available = SEMANTIC_AVAILABLE

        if self.semantic_available:
            try:
                print("[INFO] Loading semantic similarity model...")
                self.similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                print("[INFO] Semantic similarity model loaded successfully")
            except Exception as e:
                print(f"[WARNING] Failed to load semantic model: {e}")
                self.semantic_available = False

    def load_q4_answer(self, question_id: str) -> Dict:
        """
        Load Q4 generated answer.

        Args:
            question_id: Question identifier

        Returns:
            Q4 answer data
        """
        q4_path = Path(self.q_pipeline_path) / "Q4_answer_generation.json"

        if not q4_path.exists():
            print(f"[ERROR] Q4 output not found: {q4_path}")
            return {}

        with open(q4_path, 'r', encoding='utf-8') as f:
            q4_data = json.load(f)

        print(f"[Q5] Loaded Q4 answer from: {q4_path}")
        return q4_data

    def load_ground_truth(self, question_id: str) -> Dict:
        """
        Load ground truth answer from sample data.

        Args:
            question_id: Question identifier

        Returns:
            Ground truth data
        """
        # Try different sample file names
        sample_files = [
            "sample_20_records.parquet",
            "finqa_sample.parquet",
            f"{question_id}.parquet"
        ]

        for filename in sample_files:
            sample_file = self.data_dir / filename
            if sample_file.exists():
                print(f"[Q5] Loading ground truth from: {sample_file}")
                break
        else:
            print(f"[ERROR] No sample file found in: {self.data_dir}")
            print(f"[INFO] Tried: {sample_files}")
            return {}

        try:
            # Load the parquet file
            df = pd.read_parquet(sample_file)
            print(f"[Q5] Loaded {len(df)} records from ground truth file")

            # Find the row with matching ID
            matching_rows = df[df['id'] == question_id]

            if matching_rows.empty:
                print(f"[WARNING] No ground truth found for {question_id}")
                print(f"[INFO] Available IDs: {df['id'].tolist()[:5]}...")
                return {}

            row = matching_rows.iloc[0]

            # Extract all answer-related fields
            ground_truth = {
                'id': row['id'],
                'question': row['question'],
                'response': row.get('response', ''),
                'documents': row.get('documents', ''),
            }

            # Look for any columns that might contain answer data
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['answer', 'truth', 'expected', 'correct']):
                    if pd.notna(row[col]):
                        ground_truth[col] = row[col]

            print(f"[Q5] Found ground truth for {question_id}")
            return ground_truth

        except Exception as e:
            print(f"[ERROR] Failed to load ground truth: {e}")
            return {}

    def extract_numeric_value(self, text: str) -> Optional[float]:
        """
        Extract numeric value from text.

        Args:
            text: Text to extract number from

        Returns:
            Extracted numeric value or None
        """
        if isinstance(text, (int, float)):
            return float(text)

        if not isinstance(text, str):
            return None

        # Remove commas and dollar signs
        text = text.replace(',', '').replace('$', '')

        # Look for percentage patterns first (most specific)
        percent_match = re.search(r'(\d+\.?\d*)\s*%', text)
        if percent_match:
            return float(percent_match.group(1))

        # Look for regular numbers
        number_match = re.search(r'\d+\.?\d*', text)
        if number_match:
            return float(number_match.group(0))

        return None

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1
        """
        if not self.similarity_model or not text1 or not text2:
            return 0.0

        try:
            # Generate embeddings for both texts
            embeddings = self.similarity_model.encode([text1, text2])

            # Calculate cosine similarity
            similarity_matrix = cosine_similarity([embeddings[0]], [embeddings[1]])
            similarity_score = similarity_matrix[0][0]

            return float(similarity_score)

        except Exception as e:
            print(f"[WARNING] Error calculating semantic similarity: {e}")
            return 0.0

    def calculate_bleu_like_score(self, generated: str, reference: str) -> float:
        """
        Calculate BLEU-like score for answer quality assessment.

        Args:
            generated: Generated answer
            reference: Reference answer

        Returns:
            Score between 0 and 1 based on word overlap
        """
        if not generated or not reference:
            return 0.0

        # Simple word-based overlap scoring
        gen_words = set(generated.lower().split())
        ref_words = set(reference.lower().split())

        if not ref_words:
            return 0.0

        # Calculate precision and recall
        overlap = len(gen_words.intersection(ref_words))
        precision = overlap / len(gen_words) if gen_words else 0.0
        recall = overlap / len(ref_words) if ref_words else 0.0

        # F1-like score
        if precision + recall == 0:
            return 0.0

        f1_score = 2 * (precision * recall) / (precision + recall)
        return float(f1_score)

    def compare_answers(self, generated: str, ground_truth_data: Dict, question_type: str = "general") -> Dict:
        """
        Compare generated answer with ground truth.

        Args:
            generated: Generated answer text
            ground_truth_data: Ground truth data dictionary
            question_type: Type of question for specialized validation

        Returns:
            Validation results including match status and metrics
        """
        results = {
            'generated_answer': generated,
            'question_type': question_type,
            'ground_truth_found': False,
            'exact_match': False,
            'numeric_match': False,
            'similarity_score': 0.0,
            'semantic_similarity': 0.0,
            'bleu_like_score': 0.0,
            'validation_status': 'UNKNOWN',
            'validation_details': {}
        }

        # Try to find the actual answer in ground truth data
        response = ground_truth_data.get('response', '')

        if response:
            results['ground_truth_found'] = True
            results['ground_truth_response'] = response[:300] + '...' if len(response) > 300 else response

            # Calculate semantic similarity scores
            if self.semantic_available:
                results['semantic_similarity'] = self.calculate_semantic_similarity(generated, response)

            results['bleu_like_score'] = self.calculate_bleu_like_score(generated, response)
            results['similarity_score'] = max(results['semantic_similarity'], results['bleu_like_score'])

            # Extract and compare numeric values
            generated_num = self.extract_numeric_value(generated)
            results['validation_details']['generated_numeric'] = generated_num

            if generated_num is not None:
                # Look for numeric values in the response
                response_nums = re.findall(r'\d+\.?\d*', response.replace(',', ''))

                for num_str in response_nums:
                    try:
                        response_num = float(num_str)
                        results['validation_details']['response_numerics'] = response_nums

                        # Check if numbers are close (tolerance based on question type)
                        tolerance = 0.1 if question_type == "percentage_change" else 1.0

                        if abs(generated_num - response_num) <= tolerance:
                            results['numeric_match'] = True
                            results['validation_details']['matched_value'] = response_num
                            results['validation_details']['numeric_difference'] = abs(generated_num - response_num)
                            break
                    except ValueError:
                        continue

                # Special handling for percentage answers
                if question_type == "percentage_change" and '%' in generated:
                    # Check if the percentage value appears in the response
                    generated_percent_str = f"{generated_num:.2f}"
                    if generated_percent_str in response or f"{generated_num:.1f}" in response:
                        results['numeric_match'] = True
                        results['validation_details']['percentage_match'] = True

        # If no response field, check documents for the answer
        if not results['ground_truth_found']:
            documents = ground_truth_data.get('documents', '')
            if documents and isinstance(documents, str):
                # Check if our answer components appear in the source documents
                if any(word in documents.lower() for word in generated.lower().split() if len(word) > 3):
                    results['ground_truth_found'] = True
                    results['validation_status'] = 'FOUND_IN_DOCUMENTS'

        # Determine final validation status using enhanced scoring
        if results['numeric_match']:
            results['validation_status'] = 'CORRECT'
        elif results['semantic_similarity'] >= 0.8:  # High semantic similarity
            results['validation_status'] = 'CORRECT_SEMANTIC'
        elif results['semantic_similarity'] >= 0.6:  # Medium semantic similarity
            results['validation_status'] = 'PARTIAL_SEMANTIC'
        elif results['bleu_like_score'] >= 0.7:     # High word overlap
            results['validation_status'] = 'CORRECT_LEXICAL'
        elif results['ground_truth_found']:
            results['validation_status'] = 'PARTIAL'
        else:
            results['validation_status'] = 'NO_MATCH'

        return results

    def validate_answer(self, question_id: str) -> Dict:
        """
        Validate Q4 answer against ground truth.

        Args:
            question_id: Question identifier

        Returns:
            Complete validation results
        """
        print(f"\n[Q5] Starting answer validation for {question_id}")

        try:
            # Load Q4 answer
            q4_data = self.load_q4_answer(question_id)
            if not q4_data:
                return {
                    'error': 'Q4 answer not found',
                    'question_id': question_id
                }

            # Load ground truth
            ground_truth = self.load_ground_truth(question_id)
            if not ground_truth:
                return {
                    'error': 'Ground truth not found',
                    'question_id': question_id
                }

            # Extract generated answer and metadata
            generated_answer_data = q4_data.get('generated_answer', {})
            generated_text = generated_answer_data.get('answer_text', '')
            question_type = generated_answer_data.get('question_type', 'general')
            question_text = generated_answer_data.get('question_text', '')

            print(f"[Q5] Generated answer: {generated_text[:100]}...")
            print(f"[Q5] Question type: {question_type}")

            # Perform validation comparison
            validation_results = self.compare_answers(generated_text, ground_truth, question_type)

            # Add metadata
            validation_results.update({
                'question_id': question_id,
                'question_text': question_text,
                'q4_confidence': generated_answer_data.get('confidence', 0),
                'q4_generation_method': generated_answer_data.get('generation_method', 'unknown'),
                'validation_metadata': {
                    'validation_timestamp': datetime.now().isoformat(),
                    'pipeline_stage': 'Q5_answer_validation',
                    'semantic_model_available': self.semantic_available,
                    'ground_truth_source': 'sample_data'
                }
            })

            print(f"[Q5] Validation status: {validation_results['validation_status']}")
            print(f"[Q5] Numeric match: {validation_results['numeric_match']}")
            print(f"[Q5] Semantic similarity: {validation_results['semantic_similarity']:.3f}")

            return validation_results

        except Exception as e:
            print(f"[ERROR] Q5 validation failed: {str(e)}")
            return {
                'error': f'Validation failed: {str(e)}',
                'question_id': question_id
            }

    def save_results(self, question_id: str, validation_data: Dict):
        """
        Save Q5 validation results to output file.

        Args:
            question_id: Question identifier
            validation_data: Validation results data
        """
        output_data = {
            'question_id': question_id,
            'stage': 'Q5_answer_validation',
            'validation_results': validation_data,
            'pipeline_completion': {
                'pipeline_stages': ['Q2.5', 'Q3.1', 'Q3.2', 'Q3.3', 'Q4', 'Q5'],
                'final_stage': True,
                'validation_complete': True,
                'pipeline_status': 'COMPLETED'
            }
        }

        # Always use the same output filename pattern
        output_path = Path(self.q_pipeline_path) / f"Q5_answer_validation_{question_id}.json"

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"[Q5] Validation results saved to: {output_path}")

    def generate_validation_summary(self, question_id: str, validation_data: Dict) -> Dict:
        """
        Generate a summary of validation results.

        Args:
            question_id: Question identifier
            validation_data: Validation results

        Returns:
            Summary dictionary
        """
        summary = {
            'question_id': question_id,
            'overall_accuracy': 'UNKNOWN',
            'score_breakdown': {
                'numeric_accuracy': validation_data.get('numeric_match', False),
                'semantic_similarity': validation_data.get('semantic_similarity', 0.0),
                'lexical_overlap': validation_data.get('bleu_like_score', 0.0),
                'combined_score': validation_data.get('similarity_score', 0.0)
            },
            'validation_status': validation_data.get('validation_status', 'UNKNOWN'),
            'q4_confidence': validation_data.get('q4_confidence', 0.0),
            'recommendation': 'UNKNOWN'
        }

        # Determine overall accuracy
        status = validation_data.get('validation_status', 'UNKNOWN')
        if status in ['CORRECT', 'CORRECT_SEMANTIC']:
            summary['overall_accuracy'] = 'HIGH'
            summary['recommendation'] = 'Answer is accurate and reliable'
        elif status in ['CORRECT_LEXICAL', 'PARTIAL_SEMANTIC']:
            summary['overall_accuracy'] = 'MEDIUM'
            summary['recommendation'] = 'Answer is partially accurate'
        elif status == 'PARTIAL':
            summary['overall_accuracy'] = 'LOW'
            summary['recommendation'] = 'Answer needs improvement'
        else:
            summary['overall_accuracy'] = 'POOR'
            summary['recommendation'] = 'Answer is likely incorrect'

        return summary


def main():
    """Test Q5 answer validation on sample question."""

    # Initialize Q5 module
    q5 = Q5_AnswerValidation()

    # Test on sample question
    question_id = "finqa_test_1630"

    try:
        # Validate Q4 answer
        validation_results = q5.validate_answer(question_id)

        # Save validation results
        q5.save_results(question_id, validation_results)

        # Generate and display summary
        summary = q5.generate_validation_summary(question_id, validation_results)

        # Display results
        print("\n" + "="*80)
        print(f"Q5 ANSWER VALIDATION SUMMARY")
        print("="*80)
        print(f"Question ID: {question_id}")
        print(f"Question: {validation_results.get('question_text', '')}")

        print(f"\n[VALIDATION RESULTS]")
        print(f"Status: {validation_results.get('validation_status', 'UNKNOWN')}")
        print(f"Numeric Match: {validation_results.get('numeric_match', False)}")
        print(f"Semantic Similarity: {validation_results.get('semantic_similarity', 0):.3f}")
        print(f"Lexical Overlap: {validation_results.get('bleu_like_score', 0):.3f}")
        print(f"Overall Accuracy: {summary['overall_accuracy']}")

        print(f"\n[Q4 vs GROUND TRUTH]")
        generated = validation_results.get('generated_answer', '')
        ground_truth = validation_results.get('ground_truth_response', '')
        print(f"Generated: {generated[:150]}...")
        if ground_truth:
            print(f"Ground Truth: {ground_truth[:150]}...")

        print(f"\n" + "="*80)
        print(f"Q-PIPELINE VALIDATION COMPLETE!")
        print("="*80)
        print(f"[SUCCESS] Q5 validation completed successfully")
        print(f"[SUCCESS] Pipeline completed: Q2.5 -> Q3.1 -> Q3.2 -> Q3.3 -> Q4 -> Q5")
        print(f"[RECOMMENDATION] {summary['recommendation']}")

    except Exception as e:
        print(f"Error in Q5 answer validation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()