"""
Q3.2: Advanced Chunk Validation & Data Extraction Preparation
OFFICIAL Q3.2 - Advanced analysis and validation stage in Q-Pipeline

ARCHITECTURE POSITION:
- Input: Q3.1 semantically ranked chunks
- Process: Data validation + extraction preparation + chunk quality assessment
- Output: Validated chunks with extraction metadata for Q3.3
- Next Stage: Q3.3 (Final Selection & Boosting)

FEATURES:
- Data completeness validation for financial calculations
- Extraction metadata preparation (identifies tables, numbers, dates)
- Chunk quality scoring based on answer requirements
- Question-specific validation (percentage change, lookup, etc.)
- Preparation for accurate data extraction in downstream stages

This Q3.2 ensures chunks not only match semantically but contain the exact
data needed to answer the question, with metadata for extraction.
"""

import json
import os
import re
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class Q32_ChunkValidation:
    """
    Q3.2 - Advanced chunk validation and data extraction preparation.

    Validates that semantically ranked chunks contain the specific data
    needed to answer the question and prepares extraction metadata.
    """

    def __init__(self, q_pipeline_path: str = "Q_Question_Pipeline/outputs"):
        """
        Initialize Q3.2 chunk validation module.

        Args:
            q_pipeline_path: Path to Q-Pipeline outputs
        """
        self.q_pipeline_path = q_pipeline_path

    def load_q31_output(self, question_id: str) -> Dict:
        """
        Load Q3.1 semantic ranking results.

        Args:
            question_id: Question identifier

        Returns:
            Q3.1 data with semantically ranked chunks
        """
        q31_path = os.path.join(self.q_pipeline_path, f"Q3.1_semantic_ranking_{question_id}.json")

        if not os.path.exists(q31_path):
            raise FileNotFoundError(f"Q3.1 output not found: {q31_path}")

        with open(q31_path, 'r') as f:
            q31_data = json.load(f)

        return q31_data

    def extract_financial_data(self, content: str) -> Dict[str, List]:
        """
        Extract structured financial data from chunk content.

        Args:
            content: Chunk content

        Returns:
            Dictionary with extracted financial data
        """
        extraction_data = {
            'revenue_figures': [],
            'years': [],
            'currency_amounts': [],
            'tables': [],
            'percentages': []
        }

        # Extract years
        years = re.findall(r'\b(20\d{2}|19\d{2})\b', content)
        extraction_data['years'] = list(set(years))

        # Extract currency amounts
        currency_patterns = [
            r'\$[\d,]+(?:\.\d+)?',  # $123,456.78
            r'US\$[\d,]+',          # US$123,456
            r'[\d,]+\s*(?:million|thousand|billion)',  # 123 million
        ]

        for pattern in currency_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            extraction_data['currency_amounts'].extend(matches)

        # Extract percentages
        percentages = re.findall(r'\d+(?:\.\d+)?%', content)
        extraction_data['percentages'] = percentages

        # Check for table structures
        if '[[' in content or '","' in content:
            # Structured table detected
            table_patterns = re.findall(r'\[\[.*?\]\]', content, re.DOTALL)
            extraction_data['tables'] = table_patterns

        # Extract revenue-specific figures
        revenue_patterns = [
            r'[Rr]evenue["\s]*[,"]?\s*["\[]?[\d,]+',
            r'[\d,]+["\s]*[,"]?\s*[\d,]+["\s]*(?=.*revenue)',
        ]

        for pattern in revenue_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            extraction_data['revenue_figures'].extend(matches)

        return extraction_data

    def validate_percentage_change_capability(self, chunk: Dict, question_text: str) -> Dict[str, any]:
        """
        Validate if chunk can answer percentage change questions.

        Args:
            chunk: Chunk data
            question_text: Question text

        Returns:
            Validation results
        """
        content = chunk.get('content', '')
        extraction_data = self.extract_financial_data(content)

        validation = {
            'can_calculate_percentage_change': False,
            'confidence': 0.0,
            'missing_elements': [],
            'data_quality': 'incomplete',
            'extraction_ready': False
        }

        # For percentage change, need: base year value, target year value
        question_lower = question_text.lower()

        if 'percentage change' in question_lower or '% change' in question_lower:
            # Extract required years from question
            question_years = set(re.findall(r'\b(20\d{2}|19\d{2})\b', question_text))
            chunk_years = set(extraction_data['years'])

            # Check year coverage
            year_coverage = len(question_years.intersection(chunk_years)) / len(question_years) if question_years else 0

            # Check for numerical data
            has_currency_data = len(extraction_data['currency_amounts']) >= 2
            has_table_structure = len(extraction_data['tables']) > 0

            # Calculate confidence
            confidence = 0.0
            if year_coverage >= 1.0:  # All required years present
                confidence += 0.4
            elif year_coverage >= 0.5:  # Some years present
                confidence += 0.2
            else:
                validation['missing_elements'].append('required_years')

            if has_currency_data:
                confidence += 0.3
            else:
                validation['missing_elements'].append('currency_amounts')

            if has_table_structure:
                confidence += 0.2
            else:
                validation['missing_elements'].append('structured_table')

            if 'revenue' in content.lower():
                confidence += 0.1
            else:
                validation['missing_elements'].append('revenue_context')

            validation['confidence'] = confidence
            validation['can_calculate_percentage_change'] = confidence >= 0.6
            validation['data_quality'] = 'complete' if confidence >= 0.8 else 'partial' if confidence >= 0.4 else 'incomplete'
            validation['extraction_ready'] = confidence >= 0.6

        return validation

    def validate_chunk_data_completeness(self, chunk: Dict, question_text: str) -> Dict[str, any]:
        """
        Validate chunk data completeness for the specific question type.

        Args:
            chunk: Chunk data
            question_text: Question text

        Returns:
            Completeness validation results
        """
        question_lower = question_text.lower()
        content = chunk.get('content', '')

        # Determine question type and validate accordingly
        if any(phrase in question_lower for phrase in ['percentage change', '% change', 'percent change']):
            return self.validate_percentage_change_capability(chunk, question_text)

        elif any(phrase in question_lower for phrase in ['what is', 'what was', 'how much']):
            # Lookup question validation
            extraction_data = self.extract_financial_data(content)
            return {
                'can_provide_lookup': len(extraction_data['currency_amounts']) > 0,
                'confidence': min(0.3 * len(extraction_data['currency_amounts']), 1.0),
                'extraction_ready': True,
                'data_quality': 'complete' if extraction_data['currency_amounts'] else 'incomplete'
            }

        else:
            # Generic validation
            return {
                'can_answer_generic': True,
                'confidence': 0.5,
                'extraction_ready': True,
                'data_quality': 'partial'
            }

    def apply_chunk_validation(self, question_id: str) -> Tuple[List[Dict], Dict]:
        """
        Apply advanced validation to Q3.1 semantically ranked chunks.

        Args:
            question_id: Question identifier

        Returns:
            Tuple of (validated_chunks, validation_metrics)
        """
        print(f"\n[Q3.2] Starting chunk validation for {question_id}")

        # Load Q3.1 output
        q31_data = self.load_q31_output(question_id)

        ranked_chunks = q31_data['ranked_chunks']
        ranking_metrics = q31_data['ranking_metrics']

        # Get question text from Q2.5 (should be accessible via Q3.1 metadata)
        question_text = ""
        # For now, we'll load from Q2.5 directly
        q25_path = os.path.join(self.q_pipeline_path, f"Q2.5_document_aware_assignment_{question_id}.json")
        if os.path.exists(q25_path):
            with open(q25_path, 'r') as f:
                q25_data = json.load(f)
                question_text = q25_data.get('question_text', '')

        print(f"[Q3.2] Loaded {len(ranked_chunks)} semantically ranked chunks")
        print(f"[Q3.2] Question: {question_text}")

        # Validate each chunk
        validated_chunks = []

        for chunk in ranked_chunks:
            # Extract financial data
            extraction_data = self.extract_financial_data(chunk['content'])
            chunk['extraction_metadata'] = extraction_data

            # Validate data completeness
            validation_result = self.validate_chunk_data_completeness(chunk, question_text)
            chunk['validation_result'] = validation_result

            # Calculate validation score
            chunk['validation_score'] = validation_result.get('confidence', 0.0)

            # Mark as extraction ready
            chunk['extraction_ready'] = validation_result.get('extraction_ready', False)

            validated_chunks.append(chunk)

        # Sort by combined score (semantic + validation)
        for chunk in validated_chunks:
            semantic_score = chunk.get('semantic_score', 0.0)
            validation_score = chunk.get('validation_score', 0.0)
            chunk['combined_score'] = 0.6 * semantic_score + 0.4 * validation_score

        validated_chunks.sort(key=lambda x: x.get('combined_score', 0), reverse=True)

        # Calculate validation metrics
        validation_metrics = {
            'input_chunks': len(ranked_chunks),
            'validated_chunks': len(validated_chunks),
            'extraction_ready_chunks': sum(1 for c in validated_chunks if c.get('extraction_ready', False)),
            'avg_validation_score': np.mean([c.get('validation_score', 0) for c in validated_chunks]) if validated_chunks else 0,
            'avg_combined_score': np.mean([c.get('combined_score', 0) for c in validated_chunks]) if validated_chunks else 0,
            'complete_data_chunks': sum(1 for c in validated_chunks if c.get('validation_result', {}).get('data_quality') == 'complete'),
            'timestamp': datetime.now().isoformat()
        }

        print(f"[Q3.2] Chunk validation complete:")
        print(f"       - Validated chunks: {validation_metrics['validated_chunks']}")
        print(f"       - Extraction ready: {validation_metrics['extraction_ready_chunks']}")
        print(f"       - Complete data chunks: {validation_metrics['complete_data_chunks']}")
        print(f"       - Avg validation score: {validation_metrics['avg_validation_score']:.3f}")

        return validated_chunks, validation_metrics

    def save_results(self,
                    question_id: str,
                    validated_chunks: List[Dict],
                    validation_metrics: Dict,
                    output_dir: str = "Q_Question_Pipeline/outputs"):
        """
        Save Q3.2 chunk validation results.

        Args:
            question_id: Question identifier
            validated_chunks: Validated chunks with extraction metadata
            validation_metrics: Validation statistics
            output_dir: Output directory
        """
        output_data = {
            'question_id': question_id,
            'stage': 'Q3.2_chunk_validation',
            'methodology': 'data_extraction_preparation_and_validation',
            'validated_chunks': validated_chunks,
            'validation_metrics': validation_metrics,
            'pipeline_position': {
                'previous_stage': 'Q3.1_semantic_ranking',
                'next_stage': 'Q3.3_concept_boosting_final_selection',
                'architecture_role': 'advanced_validation_and_extraction_prep'
            }
        }

        output_path = os.path.join(output_dir, f"Q3.2_chunk_validation_{question_id}.json")

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"[Q3.2] Results saved to: {output_path}")


def main():
    """Test Q3.2 chunk validation on sample question."""

    # Initialize module
    q32 = Q32_ChunkValidation()

    # Test on sample question
    question_id = "finqa_test_1630"

    try:
        # Apply chunk validation
        validated_chunks, metrics = q32.apply_chunk_validation(question_id)

        # Save results
        q32.save_results(question_id, validated_chunks, metrics)

        # Display summary
        print("\n" + "="*70)
        print(f"Q3.2 CHUNK VALIDATION SUMMARY")
        print("="*70)
        print(f"Question ID: {question_id}")
        print(f"Validated chunks: {metrics['validated_chunks']}")
        print(f"Extraction ready: {metrics['extraction_ready_chunks']}")
        print(f"Complete data chunks: {metrics['complete_data_chunks']}")
        print(f"Avg validation score: {metrics['avg_validation_score']:.3f}")

        if validated_chunks:
            print(f"\nTop 3 validated chunks:")
            for i, chunk in enumerate(validated_chunks[:3], 1):
                print(f"  {i}. {chunk['chunk_id']}")
                print(f"     Combined score: {chunk['combined_score']:.3f}")
                print(f"     Validation score: {chunk['validation_score']:.3f}")
                print(f"     Extraction ready: {chunk['extraction_ready']}")
                print(f"     Data quality: {chunk['validation_result'].get('data_quality', 'unknown')}")

        print(f"\n" + "="*70)
        print(f"Q3.2 READY FOR Q3.3 FINAL SELECTION")
        print("="*70)

    except Exception as e:
        print(f"Error in Q3.2 chunk validation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()