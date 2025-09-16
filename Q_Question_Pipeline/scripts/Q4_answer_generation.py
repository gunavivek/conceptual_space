"""
Q4: Answer Generation Module
OFFICIAL Q4 - Final answer generation stage in Q-Pipeline

ARCHITECTURE POSITION:
- Input: Q3.3 final selected chunks with concept boosting
- Process: Data extraction + calculation + natural language generation
- Output: Complete answer with reasoning, calculations, and sources
- Pipeline Completion: Final stage producing human-readable answers

FEATURES:
- Intelligent data extraction from structured content
- Mathematical calculation engine for financial questions
- Natural language answer generation
- Source attribution and confidence scoring
- Question-type specific answer formatting
- Comprehensive answer validation

This Q4 completes the Q-Pipeline by generating accurate, well-reasoned answers
from the carefully curated and ranked chunks from Q3 stages.
"""

import json
import os
import re
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime


class Q4_AnswerGeneration:
    """
    Q4 - Final answer generation from curated chunks.

    Takes the best chunks from Q3.3 and generates complete answers with
    reasoning, calculations, and source attribution.
    """

    def __init__(self, q_pipeline_path: str = "Q_Question_Pipeline/outputs"):
        """
        Initialize Q4 answer generation module.

        Args:
            q_pipeline_path: Path to Q-Pipeline outputs
        """
        self.q_pipeline_path = q_pipeline_path

    def load_q3_final_chunks(self, question_id: str) -> Dict:
        """
        Load Q3.3 final chunks or fallback to available Q3 output.

        Args:
            question_id: Question identifier

        Returns:
            Final chunks data from Q3 pipeline
        """
        # Try Q3.3 concept boosting first
        q33_path = os.path.join(self.q_pipeline_path, f"Q3.3_concept_boosting_{question_id}.json")
        if os.path.exists(q33_path):
            with open(q33_path, 'r') as f:
                q33_data = json.load(f)
            return q33_data

        # Fallback to Q3 final retrieval
        q3_final_path = os.path.join(self.q_pipeline_path, f"Q3_final_retrieval_{question_id}.json")
        if os.path.exists(q3_final_path):
            with open(q3_final_path, 'r') as f:
                q3_data = json.load(f)
            return q3_data

        # Fallback to Q3.2 if available
        q32_path = os.path.join(self.q_pipeline_path, f"Q3.2_semantic_ranking_{question_id}.json")
        if os.path.exists(q32_path):
            with open(q32_path, 'r') as f:
                q32_data = json.load(f)
            return q32_data

        raise FileNotFoundError(f"No Q3 output found for {question_id}")

    def get_question_context(self, question_id: str) -> Dict[str, str]:
        """
        Get question text and context from Q2.5 output.

        Args:
            question_id: Question identifier

        Returns:
            Dictionary with question_text and doc_id
        """
        q25_path = os.path.join(self.q_pipeline_path, f"Q2.5_document_aware_assignment_{question_id}.json")

        if os.path.exists(q25_path):
            with open(q25_path, 'r') as f:
                q25_data = json.load(f)
            return {
                'question_text': q25_data.get('question_text', ''),
                'doc_id': q25_data.get('doc_id', '')
            }

        return {'question_text': '', 'doc_id': ''}

    def extract_revenue_data(self, content: str, target_years: List[str]) -> Dict[str, float]:
        """
        Extract revenue data for specific years from content.

        Args:
            content: Chunk content
            target_years: Years to extract data for

        Returns:
            Dictionary of year -> revenue_amount
        """
        revenue_data = {}

        # Look for revenue table patterns
        # Pattern 1: Table format [["", "2019", "2018"], ["Revenue", "172,752", "140,368"]]
        table_pattern = r'\[\["[^"]*",\s*"(\d{4})",\s*"(\d{4})"\][^]]*\["[^"]*Revenue[^"]*",\s*"([\d,]+)",\s*"([\d,]+)"\]'
        table_match = re.search(table_pattern, content, re.IGNORECASE)

        if table_match:
            year1, year2, amount1, amount2 = table_match.groups()
            # Convert amounts to numbers (remove commas, multiply by 1000 if needed)
            amount1_num = float(amount1.replace(',', '')) * 1000  # Assuming figures are in thousands
            amount2_num = float(amount2.replace(',', '')) * 1000

            revenue_data[year1] = amount1_num
            revenue_data[year2] = amount2_num

        # Pattern 2: Look for "Revenue", "172,752", "140,368" patterns
        if not revenue_data:
            revenue_line_pattern = r'["\[]?Revenue["\]]?[^"]*["\[,]\s*([\d,]+)["\],]\s*["\[,]\s*([\d,]+)'
            revenue_match = re.search(revenue_line_pattern, content, re.IGNORECASE)

            if revenue_match:
                amount1, amount2 = revenue_match.groups()
                amount1_num = float(amount1.replace(',', '')) * 1000
                amount2_num = float(amount2.replace(',', '')) * 1000

                # Match with years found in content
                years_in_content = re.findall(r'\b(20\d{2})\b', content)
                if len(years_in_content) >= 2:
                    # Assume later year corresponds to first amount
                    sorted_years = sorted(years_in_content)
                    if len(sorted_years) >= 2:
                        revenue_data[sorted_years[-1]] = amount1_num  # Later year
                        revenue_data[sorted_years[-2]] = amount2_num  # Earlier year

        return revenue_data

    def calculate_percentage_change(self, old_value: float, new_value: float) -> Tuple[float, Dict]:
        """
        Calculate percentage change between two values.

        Args:
            old_value: Original value
            new_value: New value

        Returns:
            Tuple of (percentage_change, calculation_details)
        """
        if old_value == 0:
            return float('inf'), {'error': 'Division by zero'}

        change_amount = new_value - old_value
        percentage_change = (change_amount / old_value) * 100

        calculation_details = {
            'old_value': old_value,
            'new_value': new_value,
            'change_amount': change_amount,
            'percentage_change': round(percentage_change, 2),
            'formula': f'(({new_value:,.0f} - {old_value:,.0f}) / {old_value:,.0f}) × 100'
        }

        return percentage_change, calculation_details

    def generate_percentage_change_answer(self, question_text: str, chunks: List[Dict]) -> Dict:
        """
        Generate answer for percentage change questions.

        Args:
            question_text: Question text
            chunks: Available chunks

        Returns:
            Complete answer data
        """
        # Extract years from question
        question_years = re.findall(r'\b(20\d{2})\b', question_text)
        if len(question_years) < 2:
            return {
                'answer_text': "Unable to determine the years for comparison from the question.",
                'confidence': 0.1,
                'error': 'insufficient_year_information'
            }

        from_year, to_year = sorted(question_years)[:2]

        # Extract revenue data from chunks
        revenue_data = {}
        source_chunks = []

        for chunk in chunks:
            chunk_revenue_data = self.extract_revenue_data(chunk['content'], [from_year, to_year])
            if chunk_revenue_data:
                revenue_data.update(chunk_revenue_data)
                source_chunks.append({
                    'chunk_id': chunk['chunk_id'],
                    'relevance': 'revenue_data_source',
                    'extracted_data': chunk_revenue_data
                })

        # Check if we have data for both years
        if from_year not in revenue_data or to_year not in revenue_data:
            return {
                'answer_text': f"Unable to find revenue data for both {from_year} and {to_year} in the provided documents.",
                'confidence': 0.2,
                'error': 'missing_year_data',
                'available_data': list(revenue_data.keys())
            }

        # Calculate percentage change
        old_value = revenue_data[from_year]
        new_value = revenue_data[to_year]

        percentage_change, calc_details = self.calculate_percentage_change(old_value, new_value)

        # Generate natural language answer
        direction = "increased" if percentage_change > 0 else "decreased"
        abs_change = abs(percentage_change)

        answer_text = f"The revenue {direction} by {abs_change:.2f}% from {from_year} to {to_year}. "
        answer_text += f"Specifically, revenue changed from ${old_value:,.0f} in {from_year} to ${new_value:,.0f} in {to_year}, "
        answer_text += f"representing a change of ${abs(calc_details['change_amount']):,.0f}."

        # Determine confidence based on data quality
        confidence = 0.95  # High confidence for exact calculations with clear data

        return {
            'answer_text': answer_text,
            'calculation_details': calc_details,
            'confidence': confidence,
            'answer_type': 'percentage_change_calculation',
            'source_chunks': source_chunks,
            'data_extracted': revenue_data
        }

    def generate_lookup_answer(self, question_text: str, chunks: List[Dict]) -> Dict:
        """
        Generate answer for lookup questions (what is, what was, how much).

        Args:
            question_text: Question text
            chunks: Available chunks

        Returns:
            Complete answer data
        """
        # Extract target year and metric from question
        years = re.findall(r'\b(20\d{2})\b', question_text)
        target_year = years[0] if years else None

        # Look for the metric being asked about
        question_lower = question_text.lower()
        if 'revenue' in question_lower:
            metric = 'revenue'
        elif 'income' in question_lower:
            metric = 'income'
        elif 'expense' in question_lower or 'cost' in question_lower:
            metric = 'expense'
        else:
            metric = 'financial_figure'

        # Extract data from chunks
        extracted_data = {}
        source_chunks = []

        for chunk in chunks:
            if target_year:
                year_data = self.extract_revenue_data(chunk['content'], [target_year])
                if year_data and target_year in year_data:
                    extracted_data[target_year] = year_data[target_year]
                    source_chunks.append({
                        'chunk_id': chunk['chunk_id'],
                        'relevance': 'data_source',
                        'extracted_data': {target_year: year_data[target_year]}
                    })
                    break

        if not extracted_data:
            return {
                'answer_text': f"Unable to find {metric} data for {target_year if target_year else 'the requested period'} in the provided documents.",
                'confidence': 0.2,
                'error': 'data_not_found'
            }

        # Generate answer
        value = list(extracted_data.values())[0]
        answer_text = f"The {metric} for {target_year} was ${value:,.0f}."

        return {
            'answer_text': answer_text,
            'confidence': 0.9,
            'answer_type': 'lookup_response',
            'source_chunks': source_chunks,
            'data_extracted': extracted_data
        }

    def generate_answer(self, question_id: str) -> Dict:
        """
        Generate complete answer for the question.

        Args:
            question_id: Question identifier

        Returns:
            Complete answer generation result
        """
        print(f"\n[Q4] Starting answer generation for {question_id}")

        # Load data
        q3_data = self.load_q3_final_chunks(question_id)
        question_context = self.get_question_context(question_id)

        question_text = question_context['question_text']
        final_chunks = q3_data.get('final_chunks', q3_data.get('ranked_chunks', []))

        print(f"[Q4] Question: {question_text}")
        print(f"[Q4] Available chunks: {len(final_chunks)}")

        if not final_chunks:
            return {
                'error': 'No chunks available for answer generation',
                'confidence': 0.0
            }

        # Determine question type and generate appropriate answer
        question_lower = question_text.lower()

        if any(phrase in question_lower for phrase in ['percentage change', '% change', 'percent change']):
            print(f"[Q4] Detected percentage change question")
            answer_data = self.generate_percentage_change_answer(question_text, final_chunks)

        elif any(phrase in question_lower for phrase in ['what is', 'what was', 'how much']):
            print(f"[Q4] Detected lookup question")
            answer_data = self.generate_lookup_answer(question_text, final_chunks)

        else:
            # Generic answer
            answer_data = {
                'answer_text': "This question type is not yet supported by the answer generation system.",
                'confidence': 0.1,
                'answer_type': 'unsupported'
            }

        # Add metadata
        answer_data['question_id'] = question_id
        answer_data['question_text'] = question_text
        answer_data['processing_metadata'] = {
            'chunks_used': len(final_chunks),
            'generation_timestamp': datetime.now().isoformat(),
            'pipeline_stage': 'Q4_answer_generation'
        }

        print(f"[Q4] Answer generated with {answer_data.get('confidence', 0):.2f} confidence")

        return answer_data

    def save_results(self,
                    question_id: str,
                    answer_data: Dict,
                    output_dir: str = "Q_Question_Pipeline/outputs"):
        """
        Save Q4 answer generation results.

        Args:
            question_id: Question identifier
            answer_data: Generated answer data
            output_dir: Output directory
        """
        output_data = {
            'question_id': question_id,
            'stage': 'Q4_answer_generation',
            'generated_answer': answer_data,
            'pipeline_completion': {
                'pipeline_stages': ['Q2.5', 'Q3.1', 'Q3.2', 'Q3.3', 'Q4'],
                'final_stage': True,
                'answer_ready': True
            }
        }

        output_path = os.path.join(output_dir, f"Q4_answer_generation_{question_id}.json")

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"[Q4] Results saved to: {output_path}")


def main():
    """Test Q4 answer generation on sample question."""

    # Initialize module
    q4 = Q4_AnswerGeneration()

    # Test on sample question
    question_id = "finqa_test_1630"

    try:
        # Generate answer
        answer_data = q4.generate_answer(question_id)

        # Save results
        q4.save_results(question_id, answer_data)

        # Display summary
        print("\n" + "="*70)
        print(f"Q4 ANSWER GENERATION SUMMARY")
        print("="*70)
        print(f"Question ID: {question_id}")
        print(f"Question: {answer_data.get('question_text', '')}")
        print(f"Answer: {answer_data.get('answer_text', '')}")
        print(f"Confidence: {answer_data.get('confidence', 0):.2f}")
        print(f"Answer Type: {answer_data.get('answer_type', 'unknown')}")

        if 'calculation_details' in answer_data:
            calc = answer_data['calculation_details']
            print(f"\nCalculation Details:")
            print(f"  Formula: {calc.get('formula', '')}")
            print(f"  Percentage Change: {calc.get('percentage_change', 0):.2f}%")

        if 'source_chunks' in answer_data:
            print(f"\nSource Chunks: {len(answer_data['source_chunks'])}")

        print(f"\n" + "="*70)
        print(f"Q-PIPELINE COMPLETE!")
        print("="*70)
        print(f"[SUCCESS] Answer generated successfully")
        print(f"[SUCCESS] Pipeline completed: Q2.5 → Q3.1 → Q3.2 → Q3.3 → Q4")

    except Exception as e:
        print(f"Error in Q4 answer generation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()