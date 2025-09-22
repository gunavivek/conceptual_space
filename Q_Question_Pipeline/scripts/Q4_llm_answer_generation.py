"""
Q4: LLM-Only Answer Generation Module
SIMPLIFIED Q4 - Final answer generation using OpenAI LLM only

ARCHITECTURE POSITION:
- Input: Q3.3 final selected chunks with concept boosting + Question text
- Process: OpenAI LLM generation with context-aware prompting
- Output: Single complete answer file (overwrites existing)
- Pipeline Completion: Final stage producing human-readable answers

FEATURES:
- LLM-ONLY GENERATION: OpenAI GPT with robust prompting
- Context preparation from curated chunks
- Question-type specific prompt engineering
- Natural language answer generation with reasoning
- Source attribution and confidence scoring
- Error handling and graceful degradation
- Single output file approach (no suffixes)

This simplified Q4 completes the Q-Pipeline by generating accurate, natural language
answers using proven OpenAI LLM approach, focusing on simplicity and reliability.
"""

import json
import os
import re
# import numpy as np  # Removed to avoid dtype issues
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

# OpenAI integration
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("[WARNING] OpenAI not installed. Q4 will be disabled.")


class Q4_LLM_AnswerGeneration:
    """
    Q4 - Simplified LLM-only answer generation from curated chunks.

    Uses OpenAI GPT with context-aware prompting to generate natural language
    answers from the carefully curated and ranked chunks from Q3 stages.
    """

    def __init__(self, q_pipeline_path: str = None):
        """
        Initialize Q4 LLM answer generation module.

        Args:
            q_pipeline_path: Path to Q-Pipeline outputs
        """
        if q_pipeline_path is None:
            # Auto-detect the Q-Pipeline outputs path
            script_dir = Path(__file__).parent
            self.q_pipeline_path = str(script_dir.parent / "outputs")
        else:
            self.q_pipeline_path = q_pipeline_path

        # Initialize OpenAI client
        self.openai_api_key = None
        self.openai_client = None

        if OPENAI_AVAILABLE:
            self._initialize_openai()

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
            print("[INFO] No valid OPENAI_API_KEY found. Q4 will be disabled.")

    def load_q3_final_chunks(self, question_id: str) -> Dict:
        """
        Load Q3.3 final chunks or fallback to available Q3 output.

        Args:
            question_id: Question identifier

        Returns:
            Final chunks data from Q3 pipeline
        """
        # Try Q3.3 concept boosting first
        q33_path = os.path.join(self.q_pipeline_path, "Q3.3_concept_boosting.json")
        if os.path.exists(q33_path):
            with open(q33_path, 'r') as f:
                q33_data = json.load(f)
            print(f"[Q4-LLM] Loaded chunks from Q3.3 concept boosting")
            return q33_data

        # Fallback to Q3 final retrieval
        q3_final_path = os.path.join(self.q_pipeline_path, "Q3_final_retrieval.json")
        if os.path.exists(q3_final_path):
            with open(q3_final_path, 'r') as f:
                q3_data = json.load(f)
            print(f"[Q4-LLM] Loaded chunks from Q3 final retrieval")
            return q3_data

        # Fallback to Q3.2 if available
        q32_path = os.path.join(self.q_pipeline_path, "Q3.2_chunk_validation.json")
        if os.path.exists(q32_path):
            with open(q32_path, 'r') as f:
                q32_data = json.load(f)
            print(f"[Q4-LLM] Loaded chunks from Q3.2 chunk validation")
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

    def determine_question_type(self, question_text: str) -> str:
        """
        Determine the type of question for appropriate prompting.

        Args:
            question_text: Question text

        Returns:
            Question type identifier
        """
        question_lower = question_text.lower()

        if any(phrase in question_lower for phrase in ['percentage change', '% change', 'percent change']):
            return "percentage_change"
        elif any(phrase in question_lower for phrase in ['what is', 'what was']):
            return "lookup"
        elif any(phrase in question_lower for phrase in ['how much', 'amount']):
            return "amount_lookup"
        elif any(phrase in question_lower for phrase in ['compare', 'comparison', 'difference']):
            return "comparison"
        elif any(phrase in question_lower for phrase in ['calculate', 'computation']):
            return "calculation"
        else:
            return "general"

    def create_context_from_chunks(self, chunks: List[Dict], max_chunks: int = 5) -> str:
        """
        Create formatted context text from chunks.

        Args:
            chunks: Available chunks
            max_chunks: Maximum number of chunks to include

        Returns:
            Formatted context string
        """
        context_text = ""
        for i, chunk in enumerate(chunks[:max_chunks], 1):
            chunk_content = chunk.get('content', '')[:800]  # Limit chunk size

            # Get relevance score from various possible fields
            score = float(chunk.get('semantic_score', 0) or
                         chunk.get('combined_score', 0) or
                         chunk.get('concept_boost_score', 0) or
                         chunk.get('relevance_score', 0) or 0)

            context_text += f"Context {i} (Relevance Score: {float(score):.3f}):\n{chunk_content}\n\n"

        return context_text

    def create_prompt(self, question_text: str, context_text: str, question_type: str) -> str:
        """
        Create appropriate prompt based on question type.

        Args:
            question_text: Question text
            context_text: Formatted context
            question_type: Type of question

        Returns:
            Formatted prompt for OpenAI
        """
        if question_type == "percentage_change":
            return f"""You are a financial analysis expert. Based on the provided context, answer the question with precise calculations and clear explanations.

Question: {question_text}

Context Information:
{context_text}

Instructions:
1. Extract the relevant numerical values for the years mentioned in the question from the context
2. Calculate the exact percentage change using the formula: ((new_value - old_value) / old_value) * 100
3. Provide your answer in a complete sentence format that includes:
   - The calculated percentage change with 2 decimal places
   - The direction (increased/decreased)
   - The specific values for both years
   - The absolute change amount
4. Show your reasoning and calculation steps
5. If you cannot find the exact values needed, explain what information is missing

Answer:"""

        elif question_type in ["lookup", "amount_lookup"]:
            return f"""You are a financial document analysis expert. Based on the provided context, answer the question accurately with specific numerical information.

Question: {question_text}

Context Information:
{context_text}

Instructions:
1. Analyze the context to find the specific information requested in the question
2. Provide the exact numerical value with appropriate units (dollars, percentages, etc.)
3. Include the relevant time period or year if applicable
4. If multiple values are relevant, explain the relationship between them
5. If the information is not available in the context, state that clearly

Answer:"""

        elif question_type == "comparison":
            return f"""You are a financial analysis expert specializing in comparative analysis. Based on the provided context, answer the question with detailed comparisons.

Question: {question_text}

Context Information:
{context_text}

Instructions:
1. Identify the items, periods, or metrics being compared
2. Extract the relevant numerical values from the context
3. Provide a clear comparison with specific numbers
4. Calculate any differences or ratios if relevant
5. Explain the significance of the comparison

Answer:"""

        else:  # general
            return f"""You are a financial document analysis expert. Based on the provided context, answer the question accurately and comprehensively.

Question: {question_text}

Context Information:
{context_text}

Instructions:
1. Analyze the context to find information relevant to the question
2. Provide a clear, direct answer based on the evidence in the context
3. Include specific numerical values, dates, or other concrete details when available
4. If calculations are needed, show your work
5. If the information is not available in the context, state that clearly
6. Keep your answer focused and professional

Answer:"""

    def generate_llm_answer(self, question_text: str, chunks: List[Dict]) -> Dict:
        """
        Generate answer using OpenAI LLM.

        Args:
            question_text: Question text
            chunks: Available chunks

        Returns:
            Complete answer data
        """
        print(f"[Q4-LLM] Processing question: {question_text}")

        if not self.openai_client:
            return {
                'answer_text': "OpenAI API client not available. Please check your API key configuration.",
                'confidence': 0.0,
                'error': 'openai_client_unavailable',
                'generation_method': 'llm_error'
            }

        try:
            # Determine question type and create context
            question_type = self.determine_question_type(question_text)
            context_text = self.create_context_from_chunks(chunks)

            print(f"[Q4-LLM] Question type: {question_type}")
            print(f"[Q4-LLM] Using {len(chunks)} chunks for context")

            # Create appropriate prompt
            prompt = self.create_prompt(question_text, context_text, question_type)

            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert financial analyst specializing in document analysis and quantitative reasoning. Always provide accurate, well-reasoned answers based on the provided context."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistent, factual responses
                max_tokens=600
            )

            answer_text = response.choices[0].message.content.strip()

            # Calculate confidence based on response characteristics
            confidence = self._calculate_confidence(answer_text, chunks, question_type)

            print(f"[Q4-LLM] Generated answer with confidence: {confidence:.2f}")

            return {
                'answer_text': answer_text,
                'confidence': confidence,
                'question_type': question_type,
                'generation_method': 'llm',
                'model_used': 'gpt-3.5-turbo',
                'context_chunks_used': len(chunks),
                'api_usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                },
                'llm_inputs': {
                    'question': question_text,
                    'context_text': context_text,
                    'full_prompt': prompt,
                    'question_type_detected': question_type
                },
                'source_chunks': [
                    {
                        'chunk_id': chunk.get('chunk_id', f'chunk_{i}'),
                        'relevance_score': (chunk.get('semantic_score', 0) or
                                           chunk.get('combined_score', 0) or
                                           chunk.get('concept_boost_score', 0)),
                        'content_preview': chunk.get('content', '')[:200] + '...' if len(chunk.get('content', '')) > 200 else chunk.get('content', '')
                    }
                    for i, chunk in enumerate(chunks[:5])
                ]
            }

        except Exception as e:
            print(f"[ERROR] OpenAI API call failed: {str(e)}")
            return {
                'answer_text': f"Error generating answer: {str(e)}",
                'confidence': 0.0,
                'generation_method': 'llm_error',
                'error': str(e)
            }

    def _calculate_confidence(self, answer_text: str, chunks: List[Dict], question_type: str) -> float:
        """Calculate confidence score for LLM-generated answers."""
        confidence = 0.7  # Base confidence for LLM responses

        # Boost confidence for specific indicators
        if any(indicator in answer_text.lower() for indicator in ['calculate', 'formula', '%', 'percent']):
            confidence += 0.1

        # Boost confidence if answer contains specific numbers
        if re.search(r'\d+\.?\d*%', answer_text) or re.search(r'\$[\d,]+', answer_text):
            confidence += 0.1

        # Boost confidence based on chunk quality
        if chunks:
            chunk_scores = []
            for chunk in chunks:
                score = float(chunk.get('semantic_score', 0) or
                             chunk.get('combined_score', 0) or
                             chunk.get('concept_boost_score', 0) or 0)
                chunk_scores.append(score)

            if chunk_scores:
                avg_chunk_score = sum(chunk_scores) / len(chunk_scores)
                confidence += min(0.15, avg_chunk_score * 0.3)

        # Boost confidence for percentage change questions with calculations
        if question_type == "percentage_change" and "%" in answer_text:
            confidence += 0.1

        # Penalize if answer indicates uncertainty
        if any(uncertainty in answer_text.lower() for uncertainty in
               ['unable', 'cannot find', 'missing', 'not available', 'unclear']):
            confidence -= 0.3

        # Penalize very short answers
        if len(answer_text) < 50:
            confidence -= 0.1

        return max(0.0, min(1.0, confidence))

    def generate_answer(self, question_id: str) -> Dict:
        """
        Generate complete answer for the question.

        Args:
            question_id: Question identifier

        Returns:
            Complete answer generation result
        """
        print(f"\n[Q4-LLM] Starting answer generation for {question_id}")

        try:
            # Load data
            q3_data = self.load_q3_final_chunks(question_id)
            question_context = self.get_question_context(question_id)

            question_text = question_context['question_text']

            # Extract chunks from Q3 data structure
            final_chunks = (q3_data.get('final_chunks') or
                           q3_data.get('final_ranked_chunks') or
                           q3_data.get('ranked_chunks') or [])

            print(f"[Q4-LLM] Question: {question_text}")
            print(f"[Q4-LLM] Available chunks: {len(final_chunks)}")

            if not final_chunks:
                return {
                    'error': 'No chunks available for answer generation',
                    'confidence': 0.0,
                    'question_text': question_text,
                    'generation_method': 'error'
                }

            # Generate LLM answer
            answer_data = self.generate_llm_answer(question_text, final_chunks)

            # Add metadata
            answer_data.update({
                'question_id': question_id,
                'question_text': question_text,
                'processing_metadata': {
                    'chunks_used': len(final_chunks),
                    'generation_timestamp': datetime.now().isoformat(),
                    'pipeline_stage': 'Q4_llm_answer_generation'
                }
            })

            return answer_data

        except Exception as e:
            print(f"[ERROR] Q4 answer generation failed: {str(e)}")
            return {
                'error': f'Answer generation failed: {str(e)}',
                'confidence': 0.0,
                'question_text': question_context.get('question_text', ''),
                'generation_method': 'error'
            }

    def save_results(self, question_id: str, answer_data: Dict):
        """
        Save Q4 answer generation results to output file.

        Args:
            question_id: Question identifier
            answer_data: Generated answer data
        """
        output_data = {
            'question_id': question_id,
            'stage': 'Q4_answer_generation',
            'generated_answer': answer_data,
            'pipeline_completion': {
                'pipeline_stages': ['Q2.5', 'Q3.1', 'Q3.2', 'Q3.3', 'Q4'],
                'final_stage': True,
                'answer_ready': True,
                'generation_approach': 'llm_based_openai'
            }
        }

        # Always use the same output filename (no suffixes)
        output_path = os.path.join(self.q_pipeline_path, f"Q4_answer_generation_{question_id}.json")

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"[Q4-LLM] Results saved to: {output_path}")


def main():
    """Test Q4 LLM answer generation on sample question."""

    # Initialize LLM module
    q4_llm = Q4_LLM_AnswerGeneration()

    # Test on sample question
    question_id = "finqa_test_1630"

    try:
        # Generate answer
        answer_data = q4_llm.generate_answer(question_id)

        # Save results
        q4_llm.save_results(question_id, answer_data)

        # Display summary
        print("\n" + "="*70)
        print(f"Q4 LLM ANSWER GENERATION SUMMARY")
        print("="*70)
        print(f"Question ID: {question_id}")
        print(f"Question: {answer_data.get('question_text', '')}")
        print(f"Answer: {answer_data.get('answer_text', '')}")
        print(f"Confidence: {answer_data.get('confidence', 0):.2f}")
        print(f"Question Type: {answer_data.get('question_type', 'unknown')}")

        if 'api_usage' in answer_data:
            tokens = answer_data['api_usage'].get('total_tokens', 0)
            print(f"API Tokens Used: {tokens}")

        if 'source_chunks' in answer_data:
            print(f"Source Chunks: {len(answer_data['source_chunks'])}")

        print(f"\n" + "="*70)
        print(f"Q-PIPELINE COMPLETE!")
        print("="*70)
        print(f"[SUCCESS] LLM answer generated successfully")
        print(f"[SUCCESS] Pipeline completed: Q2.5 -> Q3.1 -> Q3.2 -> Q3.3 -> Q4 (LLM)")

    except Exception as e:
        print(f"Error in Q4 LLM answer generation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()