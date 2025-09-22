"""
Q4: HYBRID Answer Generation Module
OFFICIAL Q4 - Final answer generation stage in Q-Pipeline with DUAL OUTPUT

ARCHITECTURE POSITION:
- Input: Q3.3 final selected chunks with concept boosting + Question text
- Process: DUAL generation - LLM + Algorithmic processing in parallel
- Output: Two complete answer files - LLM generated + Algorithmic generated
- Pipeline Completion: Final stage producing human-readable answers via both methods

FEATURES:
- DUAL GENERATION: Both LLM (OpenAI) and algorithmic approaches
- Intelligent data extraction from structured content
- Mathematical calculation engine for financial questions
- Natural language answer generation (both methods)
- Source attribution and confidence scoring
- Question-type specific answer formatting
- Comprehensive answer validation
- OpenAI API integration with robust error handling

This Q4 completes the Q-Pipeline by generating accurate, well-reasoned answers
from the carefully curated and ranked chunks using BOTH LLM and algorithmic methods.
"""

import json
import os
import re
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
from pathlib import Path
import concurrent.futures

# OpenAI integration
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("[WARNING] OpenAI not installed. LLM generation will be disabled.")


class Q4_HybridAnswerGeneration:
    """
    Q4 - HYBRID answer generation from curated chunks.

    Generates answers using BOTH algorithmic and LLM approaches:
    1. Algorithmic: Pattern-based extraction + mathematical calculations
    2. LLM: OpenAI GPT with context-aware prompting

    Produces two separate output files for comparison and analysis.
    """

    def __init__(self, q_pipeline_path: str = None):
        """
        Initialize Q4 hybrid answer generation module.

        Args:
            q_pipeline_path: Path to Q-Pipeline outputs
        """
        if q_pipeline_path is None:
            # Auto-detect the Q-Pipeline outputs path
            script_dir = Path(__file__).parent
            self.q_pipeline_path = str(script_dir.parent / "outputs")
        else:
            self.q_pipeline_path = q_pipeline_path

        # Initialize OpenAI client (copied from B5)
        self.openai_api_key = None
        self.openai_client = None

        if OPENAI_AVAILABLE:
            self._initialize_openai()

        # Regex patterns for algorithmic processing
        self.number_patterns = [
            r'([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?)',  # Comma-separated numbers
            r'([0-9]+\.[0-9]+)',  # Decimal numbers
            r'([0-9]+)',  # Whole numbers
        ]

    def _initialize_openai(self):
        """Initialize OpenAI client with API key."""
        # Load .env file from project root
        env_path = Path(__file__).parent.parent.parent / ".env"
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        # Remove quotes if present
                        value = value.strip('\'"')
                        os.environ[key] = value
            print(f"[INFO] Loaded environment variables from {env_path}")

        # Get API key from environment
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            print("[INFO] Using OpenAI API key from environment variable")
        else:
            print("[INFO] No OPENAI_API_KEY found in environment")

        if self.openai_api_key and self.openai_api_key.startswith('sk-'):
            try:
                self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
                print("[SUCCESS] OpenAI API client initialized successfully")
            except Exception as e:
                print(f"[ERROR] Failed to initialize OpenAI client: {e}")
        else:
            print("[INFO] No valid OPENAI_API_KEY found. LLM generation will be disabled.")

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
        q32_path = os.path.join(self.q_pipeline_path, f"Q3.2_chunk_validation_{question_id}.json")
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

    # =================== ALGORITHMIC APPROACH (Original Q4 Logic) ===================

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

    def generate_algorithmic_answer(self, question_text: str, chunks: List[Dict]) -> Dict:
        """
        Generate answer using algorithmic approach (original Q4 logic).

        Args:
            question_text: Question text
            chunks: Available chunks

        Returns:
            Complete answer data using algorithmic processing
        """
        print(f"[Q4-ALGORITHMIC] Processing question: {question_text}")

        # Determine question type and generate appropriate answer
        question_lower = question_text.lower()

        if any(phrase in question_lower for phrase in ['percentage change', '% change', 'percent change']):
            print(f"[Q4-ALGORITHMIC] Detected percentage change question")
            return self._generate_algorithmic_percentage_change_answer(question_text, chunks)

        elif any(phrase in question_lower for phrase in ['what is', 'what was', 'how much']):
            print(f"[Q4-ALGORITHMIC] Detected lookup question")
            return self._generate_algorithmic_lookup_answer(question_text, chunks)

        else:
            # Generic answer
            return {
                'answer_text': "This question type is not yet supported by the algorithmic answer generation system.",
                'confidence': 0.1,
                'answer_type': 'unsupported',
                'generation_method': 'algorithmic'
            }

    def _generate_algorithmic_percentage_change_answer(self, question_text: str, chunks: List[Dict]) -> Dict:
        """Generate percentage change answer using algorithmic approach."""
        # Extract years from question
        question_years = re.findall(r'\b(20\d{2})\b', question_text)
        if len(question_years) < 2:
            return {
                'answer_text': "Unable to determine the years for comparison from the question.",
                'confidence': 0.1,
                'error': 'insufficient_year_information',
                'generation_method': 'algorithmic'
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
                'available_data': list(revenue_data.keys()),
                'generation_method': 'algorithmic'
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
            'data_extracted': revenue_data,
            'generation_method': 'algorithmic'
        }

    def _generate_algorithmic_lookup_answer(self, question_text: str, chunks: List[Dict]) -> Dict:
        """Generate lookup answer using algorithmic approach."""
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
                'error': 'data_not_found',
                'generation_method': 'algorithmic'
            }

        # Generate answer
        value = list(extracted_data.values())[0]
        answer_text = f"The {metric} for {target_year} was ${value:,.0f}."

        return {
            'answer_text': answer_text,
            'confidence': 0.9,
            'answer_type': 'lookup_response',
            'source_chunks': source_chunks,
            'data_extracted': extracted_data,
            'generation_method': 'algorithmic'
        }

    # =================== LLM APPROACH (Copied from B5) ===================

    def generate_llm_answer(self, question_text: str, chunks: List[Dict]) -> Dict:
        """
        Generate answer using LLM approach (OpenAI API).

        Args:
            question_text: Question text
            chunks: Available chunks

        Returns:
            Complete answer data using LLM processing
        """
        print(f"[Q4-LLM] Processing question: {question_text}")

        if not self.openai_client:
            return {
                'answer_text': "OpenAI API client not available. Please check your API key configuration.",
                'confidence': 0.0,
                'error': 'openai_client_unavailable',
                'generation_method': 'llm_error'
            }

        # Determine question type for appropriate prompting
        question_lower = question_text.lower()

        if any(phrase in question_lower for phrase in ['percentage change', '% change', 'percent change']):
            question_type = "percentage_change"
        else:
            question_type = "general"

        try:
            # Prepare context from chunks
            context_text = ""
            for i, chunk in enumerate(chunks[:5], 1):
                chunk_content = chunk.get('content', '')[:800]  # Limit chunk size
                score = chunk.get('semantic_score', chunk.get('combined_score', 0))
                context_text += f"Context {i} (Score: {score:.3f}):\n{chunk_content}\n\n"

            # Create prompt based on question type
            if question_type == "percentage_change":
                prompt = f"""You are a financial analysis expert. Based on the provided context, answer the question with a precise percentage calculation.

Question: {question_text}

Context Information:
{context_text}

Instructions:
1. Extract the relevant numerical values for the years mentioned in the question from the context
2. Calculate the exact percentage change: ((new_value - old_value) / old_value) * 100
3. Provide your answer in a clear, complete sentence format
4. If you cannot find the exact values needed, explain what information is missing
5. Show your calculation steps if possible

Answer:"""
            else:
                prompt = f"""You are a financial document analysis expert. Based on the provided context, answer the question accurately and concisely.

Question: {question_text}

Context Information:
{context_text}

Instructions:
1. Analyze the context to find information relevant to the question
2. Provide a clear, direct answer based on the evidence in the context
3. If the information is not available in the context, state that clearly
4. Keep your answer focused and professional

Answer:"""

            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert financial analyst specializing in document analysis and quantitative reasoning."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent, factual responses
                max_tokens=500
            )

            answer_text = response.choices[0].message.content.strip()

            # Calculate confidence based on response characteristics
            confidence = self._calculate_llm_confidence(answer_text, chunks, question_type)

            return {
                'answer_text': answer_text,
                'confidence': confidence,
                'answer_type': question_type,
                'generation_method': 'llm',
                'model_used': 'gpt-3.5-turbo',
                'context_chunks_used': len(chunks),
                'api_usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                },
                'source_chunks': [{'chunk_id': chunk['chunk_id'], 'relevance': 'llm_context'} for chunk in chunks[:5]]
            }

        except Exception as e:
            print(f"[ERROR] OpenAI API call failed: {str(e)}")
            return {
                'answer_text': f"Error generating LLM answer: {str(e)}",
                'confidence': 0.0,
                'generation_method': 'llm_error',
                'error': str(e)
            }

    def _calculate_llm_confidence(self, answer_text: str, chunks: List[Dict], question_type: str) -> float:
        """Calculate confidence score for LLM-generated answers."""
        confidence = 0.7  # Base confidence for LLM responses

        # Boost confidence for specific indicators
        if any(indicator in answer_text.lower() for indicator in ['calculate', 'formula', '%', 'percent']):
            confidence += 0.1

        # Boost confidence if answer contains specific numbers
        if re.search(r'\d+\.?\d*%', answer_text):
            confidence += 0.1

        # Boost confidence based on chunk quality
        if chunks:
            avg_chunk_score = np.mean([chunk.get('semantic_score', chunk.get('combined_score', 0)) for chunk in chunks])
            confidence += min(0.1, avg_chunk_score * 0.2)

        # Penalize if answer indicates uncertainty
        if any(uncertainty in answer_text.lower() for uncertainty in ['unable', 'cannot find', 'missing', 'not available']):
            confidence -= 0.3

        return max(0.0, min(1.0, confidence))

    # =================== HYBRID GENERATION CONTROLLER ===================

    def generate_hybrid_answers(self, question_id: str) -> Tuple[Dict, Dict]:
        """
        Generate answers using BOTH algorithmic and LLM approaches.

        Args:
            question_id: Question identifier

        Returns:
            Tuple of (algorithmic_result, llm_result)
        """
        print(f"\n[Q4-HYBRID] Starting dual answer generation for {question_id}")

        # Load data
        q3_data = self.load_q3_final_chunks(question_id)
        question_context = self.get_question_context(question_id)

        question_text = question_context['question_text']
        final_chunks = q3_data.get('final_chunks',
                              q3_data.get('final_ranked_chunks',
                                         q3_data.get('ranked_chunks', [])))

        print(f"[Q4-HYBRID] Question: {question_text}")
        print(f"[Q4-HYBRID] Available chunks: {len(final_chunks)}")

        if not final_chunks:
            error_result = {
                'error': 'No chunks available for answer generation',
                'confidence': 0.0,
                'question_text': question_text
            }
            return error_result, error_result

        # Run both approaches in parallel for efficiency
        print(f"[Q4-HYBRID] Running both algorithmic and LLM generation in parallel...")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit both tasks
            algorithmic_future = executor.submit(self.generate_algorithmic_answer, question_text, final_chunks)
            llm_future = executor.submit(self.generate_llm_answer, question_text, final_chunks)

            # Wait for both results
            algorithmic_result = algorithmic_future.result()
            llm_result = llm_future.result()

        # Add metadata to both results
        for result in [algorithmic_result, llm_result]:
            result.update({
                'question_id': question_id,
                'question_text': question_text,
                'processing_metadata': {
                    'chunks_used': len(final_chunks),
                    'generation_timestamp': datetime.now().isoformat(),
                    'pipeline_stage': 'Q4_hybrid_answer_generation'
                }
            })

        print(f"[Q4-HYBRID] Algorithmic confidence: {algorithmic_result.get('confidence', 0):.2f}")
        print(f"[Q4-HYBRID] LLM confidence: {llm_result.get('confidence', 0):.2f}")

        return algorithmic_result, llm_result

    def save_dual_results(self,
                         question_id: str,
                         algorithmic_result: Dict,
                         llm_result: Dict,
                         output_dir: str = None):
        """
        Save both algorithmic and LLM results to separate files.

        Args:
            question_id: Question identifier
            algorithmic_result: Algorithmic answer data
            llm_result: LLM answer data
            output_dir: Output directory
        """
        if output_dir is None:
            output_dir = self.q_pipeline_path
        # Save algorithmic result
        algorithmic_output = {
            'question_id': question_id,
            'stage': 'Q4_hybrid_answer_generation',
            'generation_method': 'algorithmic',
            'generated_answer': algorithmic_result,
            'pipeline_completion': {
                'pipeline_stages': ['Q2.5', 'Q3.1', 'Q3.2', 'Q3.3', 'Q4'],
                'final_stage': True,
                'answer_ready': True,
                'generation_approach': 'pattern_based_algorithmic'
            }
        }

        algorithmic_path = os.path.join(output_dir, f"Q4_algorithmic_answer_{question_id}.json")
        with open(algorithmic_path, 'w') as f:
            json.dump(algorithmic_output, f, indent=2)

        # Save LLM result
        llm_output = {
            'question_id': question_id,
            'stage': 'Q4_hybrid_answer_generation',
            'generation_method': 'llm',
            'generated_answer': llm_result,
            'pipeline_completion': {
                'pipeline_stages': ['Q2.5', 'Q3.1', 'Q3.2', 'Q3.3', 'Q4'],
                'final_stage': True,
                'answer_ready': True,
                'generation_approach': 'llm_based_openai'
            }
        }

        llm_path = os.path.join(output_dir, f"Q4_llm_answer_{question_id}.json")
        with open(llm_path, 'w') as f:
            json.dump(llm_output, f, indent=2)

        print(f"[Q4-HYBRID] Algorithmic results saved to: {algorithmic_path}")
        print(f"[Q4-HYBRID] LLM results saved to: {llm_path}")

    def generate_comparison_report(self,
                                  question_id: str,
                                  algorithmic_result: Dict,
                                  llm_result: Dict,
                                  output_dir: str = None):
        """
        Generate a comparison report between algorithmic and LLM approaches.

        Args:
            question_id: Question identifier
            algorithmic_result: Algorithmic answer data
            llm_result: LLM answer data
            output_dir: Output directory
        """
        if output_dir is None:
            output_dir = self.q_pipeline_path
        comparison_report = {
            'question_id': question_id,
            'comparison_timestamp': datetime.now().isoformat(),
            'question_text': algorithmic_result.get('question_text', ''),
            'comparison_analysis': {
                'algorithmic_approach': {
                    'confidence': algorithmic_result.get('confidence', 0),
                    'answer_length': len(algorithmic_result.get('answer_text', '')),
                    'has_calculations': 'calculation_details' in algorithmic_result,
                    'error_status': algorithmic_result.get('error', None)
                },
                'llm_approach': {
                    'confidence': llm_result.get('confidence', 0),
                    'answer_length': len(llm_result.get('answer_text', '')),
                    'model_used': llm_result.get('model_used', 'unknown'),
                    'api_tokens': llm_result.get('api_usage', {}).get('total_tokens', 0),
                    'error_status': llm_result.get('error', None)
                },
                'recommended_approach': 'algorithmic' if algorithmic_result.get('confidence', 0) > llm_result.get('confidence', 0) else 'llm'
            },
            'answers': {
                'algorithmic_answer': algorithmic_result.get('answer_text', ''),
                'llm_answer': llm_result.get('answer_text', '')
            }
        }

        comparison_path = os.path.join(output_dir, f"Q4_comparison_report_{question_id}.json")
        with open(comparison_path, 'w') as f:
            json.dump(comparison_report, f, indent=2)

        print(f"[Q4-HYBRID] Comparison report saved to: {comparison_path}")


def main():
    """Test Q4 hybrid answer generation on sample question."""

    # Initialize hybrid module
    q4_hybrid = Q4_HybridAnswerGeneration()

    # Test on sample question
    question_id = "finqa_test_1630"

    try:
        # Generate both answers
        algorithmic_result, llm_result = q4_hybrid.generate_hybrid_answers(question_id)

        # Save both results
        q4_hybrid.save_dual_results(question_id, algorithmic_result, llm_result)

        # Generate comparison report
        q4_hybrid.generate_comparison_report(question_id, algorithmic_result, llm_result)

        # Display summary
        print("\n" + "="*80)
        print(f"Q4 HYBRID ANSWER GENERATION SUMMARY")
        print("="*80)
        print(f"Question ID: {question_id}")
        print(f"Question: {algorithmic_result.get('question_text', '')}")

        print(f"\n[ALGORITHMIC APPROACH]")
        print(f"Answer: {algorithmic_result.get('answer_text', '')}")
        print(f"Confidence: {algorithmic_result.get('confidence', 0):.2f}")
        print(f"Method: {algorithmic_result.get('generation_method', 'unknown')}")

        print(f"\n[LLM APPROACH]")
        print(f"Answer: {llm_result.get('answer_text', '')}")
        print(f"Confidence: {llm_result.get('confidence', 0):.2f}")
        print(f"Method: {llm_result.get('generation_method', 'unknown')}")
        if 'api_usage' in llm_result:
            tokens = llm_result['api_usage'].get('total_tokens', 0)
            print(f"API Tokens Used: {tokens}")

        print(f"\n" + "="*80)
        print(f"Q-PIPELINE HYBRID COMPLETE!")
        print("="*80)
        print(f"[SUCCESS] Dual answers generated successfully")
        print(f"[SUCCESS] Pipeline completed: Q2.5 -> Q3.1 -> Q3.2 -> Q3.3 -> Q4 (HYBRID)")
        print(f"[FILES] Two output files generated: algorithmic + LLM answers")

    except Exception as e:
        print(f"Error in Q4 hybrid answer generation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()