#!/usr/bin/env python3
"""
B5.2: Direct Answer Generation from B2 Outputs
Bypasses B3/B4 and uses B2 intent analysis to directly rank chunks and generate answers
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

class B52AnswerGenerator:
    """
    Generates answers directly from B2 outputs without B3/B4 intermediaries
    """
    
    def __init__(self):
        self.output_dir = Path(__file__).parent.parent / "outputs"
        
    def load_b2_outputs(self) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """Load B1 and all B2 sub-module outputs, including B2.4 if available"""
        # Load B1 question - try B1_current_question.json first
        b1_file = self.output_dir / "B1_current_question.json"
        if not b1_file.exists():
            b1_file = self.output_dir / "B1_question_analysis.json"
        
        with open(b1_file, 'r', encoding='utf-8') as f:
            b1_data = json.load(f)
            
        # Load B2.1 intent analysis
        with open(self.output_dir / "B2.1_intent_layer_output.json", 'r', encoding='utf-8') as f:
            b2_1_data = json.load(f)
            
        # Load B2.2 declarative transformation
        with open(self.output_dir / "B2.2_declarative_output.json", 'r', encoding='utf-8') as f:
            b2_2_data = json.load(f)
            
        # Load B2.3 answer expectation
        with open(self.output_dir / "B2.3_answer_expectation_output.json", 'r', encoding='utf-8') as f:
            b2_3_data = json.load(f)
            
        # Load B2.4 temporal analysis (optional)
        b2_4_data = None
        b2_4_path = self.output_dir / "B2.4_temporal_analysis_output.json"
        if b2_4_path.exists():
            with open(b2_4_path, 'r', encoding='utf-8') as f:
                b2_4_data = json.load(f)
            
        return b1_data, b2_1_data, b2_2_data, b2_3_data, b2_4_data
    
    def load_a3_chunks(self, target_record_id: str = None) -> List[Dict]:
        """Load chunks from A3 output with optional record ID filtering"""
        # Use absolute path relative to script location
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent  # Go up to conceptual_space/
        
        chunk_path = project_root / "A_Concept_pipeline" / "outputs" / "A3_multi_strategy_chunks.json"
        if not chunk_path.exists():
            chunk_path = project_root / "A_Concept_pipeline" / "outputs" / "A3_raw_chunks_no_dedup.json"
        
        if chunk_path.exists():
            with open(chunk_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                chunks = data.get('chunks', [])
                
                # Apply record ID filtering to prevent data leakage
                if target_record_id:
                    filtered_chunks = []
                    for chunk in chunks:
                        chunk_id = chunk.get('chunk_id', '')
                        # Check if chunk belongs to target record
                        if chunk_id.startswith(target_record_id):
                            filtered_chunks.append(chunk)
                    
                    return filtered_chunks
                
                return chunks
        return []
    
    def score_chunk_with_b2_criteria(self, chunk: Dict, b2_1: Dict, b2_2: Dict, b2_3: Dict, b2_4: Dict = None) -> float:
        """
        Score a chunk based on B2 intent analysis criteria
        """
        score = 0.0
        content = chunk.get('content', '')
        content_lower = content.lower()
        
        # B2.1 Intent-based scoring
        intent_analysis = b2_1.get('intent_analysis', {})
        key_entities = b2_1.get('key_entities', [])
        
        # Check for key entities from B2.1
        for entity in key_entities:
            entity_value = entity.get('value', '').lower()
            if entity_value and entity_value in content_lower:
                score += 0.25
                
        # B2.3 Answer expectation-based scoring
        answer_pred = b2_3.get('answer_prediction', {})
        format_spec = b2_3.get('format_specification', {})
        
        # If expecting numeric answer, check for numbers
        if answer_pred.get('primary_type') == 'numeric':
            # Check for dollar amounts
            if '$' in content and re.search(r'\d+(?:\.\d+)?', content):
                score += 0.25
                
        # If comparative structure, check for multiple time periods
        if format_spec.get('structure') == 'comparative':
            years_found = re.findall(r'\b(20\d{2})\b', content)
            if len(set(years_found)) >= 2:  # At least 2 different years
                score += 0.35  # High score for comparison data
                
        # If requires calculation, prioritize chunks with complete data
        complexity = b2_3.get('complexity_analysis', {})
        if complexity.get('requires_calculation'):
            # Check if chunk has both entity and values
            has_entity = any(e.get('value', '').lower() in content_lower for e in key_entities)
            has_values = bool(re.findall(r'\$\d+(?:\.\d+)?', content))
            if has_entity and has_values:
                score += 0.15
                
        # Bonus for chunks that match declarative patterns from B2.2
        declarative_forms = b2_2.get('declarative_forms', [])
        for dec_form in declarative_forms[:2]:  # Check top 2 declarative forms
            # Simple keyword matching for declarative alignment
            dec_keywords = dec_form.lower().split()
            matches = sum(1 for kw in dec_keywords if kw in content_lower)
            if matches >= 2:
                score += 0.1
                break
        
        # Enhanced temporal scoring using B2.4
        if b2_4 and b2_4.get('is_temporal_question'):
            temporal_score = self.calculate_temporal_score(chunk, b2_4)
            score += temporal_score
                
        return min(score, 1.0)  # Cap at 1.0
    
    def calculate_temporal_score(self, chunk: Dict, b2_4: Dict) -> float:
        """Calculate additional score for temporal questions using B2.4 analysis"""
        content = chunk.get('content', '').lower()
        score = 0.0
        
        # Check for compound temporal terms from B2.4
        compound_terms = b2_4.get('temporal_entities', {}).get('compound_temporal_terms', [])
        for term_info in compound_terms:
            term = term_info.get('term', '').lower()
            if term and term in content:
                score += 0.3  # High boost for compound temporal terms
                
        # Check for enhanced search terms from B2.4
        enhanced_terms = b2_4.get('enhanced_search_terms', [])
        for term in enhanced_terms:
            if term.lower() in content:
                score += 0.1
                
        # Check for business periods
        business_periods = b2_4.get('temporal_entities', {}).get('business_periods', [])
        for period_info in business_periods:
            term = period_info.get('term', '').lower()
            if term and term in content:
                score += 0.2
                
        # Check for timing relationships
        timing_relations = b2_4.get('temporal_entities', {}).get('timing_relationships', [])
        for relation_info in timing_relations:
            term = relation_info.get('term', '').lower()
            if term and term in content:
                score += 0.15
                
        return min(score, 0.4)  # Cap temporal bonus at 0.4
    
    def rank_chunks_directly(self, chunks: List[Dict], b2_1: Dict, b2_2: Dict, b2_3: Dict, b2_4: Dict = None) -> List[Dict]:
        """
        Rank chunks directly using B2 criteria without B3/B4
        """
        scored_chunks = []
        
        for chunk in chunks:
            score = self.score_chunk_with_b2_criteria(chunk, b2_1, b2_2, b2_3, b2_4)
            scored_chunks.append({
                'chunk_id': chunk.get('chunk_id', 'unknown'),
                'content': chunk.get('content', ''),
                'score': score,
                'strategy': chunk.get('strategy', 'unknown')
            })
        
        # Sort by score descending
        scored_chunks.sort(key=lambda x: x['score'], reverse=True)
        
        return scored_chunks
    
    def extract_values_for_comparison(self, chunks: List[Dict], entity: str) -> Dict[str, float]:
        """
        Extract values for comparison questions
        """
        values = {}
        entity_lower = entity.lower()
        
        for chunk in chunks[:5]:  # Check top 5 chunks
            content = chunk['content']
            content_lower = content.lower()
            
            # Only process if chunk contains the entity
            if entity_lower not in content_lower:
                continue
                
            # Look for year-value patterns
            # Pattern 1: "for YEAR is $VALUE"
            pattern1 = r'for (\d{4}) is \$(\d+(?:\.\d+)?)'
            matches = re.findall(pattern1, content_lower)
            for year, value in matches:
                if year not in values:
                    values[year] = float(value)
                    
            # Pattern 2: "YEAR: $VALUE"
            pattern2 = r'(\d{4}):\s*\$(\d+(?:\.\d+)?)'
            matches = re.findall(pattern2, content)
            for year, value in matches:
                if year not in values:
                    values[year] = float(value)
                    
        return values
    
    def generate_temporal_answer(self, question: str, ranked_chunks: List[Dict], 
                                b2_1: Dict, b2_4: Dict) -> str:
        """Generate answer specifically for temporal questions using B2.4 analysis"""
        # Get temporal context from B2.4
        temporal_context = b2_4.get('temporal_context', {})
        compound_terms = b2_4.get('temporal_entities', {}).get('compound_temporal_terms', [])
        
        # Look for direct temporal answers in top chunks
        for chunk in ranked_chunks[:3]:
            content = chunk['content']
            content_lower = content.lower()
            
            # Check if chunk contains the compound temporal term
            for term_info in compound_terms:
                term = term_info.get('term', '')
                if term.lower() in content_lower:
                    # Look for timing patterns in the content
                    timing_patterns = [
                        r'(annually|yearly|every year)\s+at\s+the\s+(beginning|start)',
                        r'generally\s+invoice\s+customers\s+(annually|yearly|monthly|quarterly)',
                        r'at\s+the\s+(beginning|start|end)\s+of\s+(each|every)\s+(annual|year|quarter|month)',
                        r'(annually|yearly|monthly|quarterly)\s+at\s+the\s+(beginning|start)'
                    ]
                    
                    for pattern in timing_patterns:
                        match = re.search(pattern, content_lower)
                        if match:
                            answer = f"Based on the provided context regarding {term}:\n\n"
                            answer += f"{content}\n\n"
                            
                            # Extract the specific timing from the match
                            if 'annually' in match.group() or 'yearly' in match.group():
                                if 'beginning' in match.group() or 'start' in match.group():
                                    timing_detail = "annually at the beginning of each coverage period"
                                else:
                                    timing_detail = "annually"
                            else:
                                timing_detail = match.group()
                            
                            answer += f"For {term}, customers are invoiced {timing_detail}."
                            return answer
            
        # Fallback: General temporal answer
        if ranked_chunks and ranked_chunks[0]['score'] > 0.2:
            answer = f"Based on the temporal analysis of the question:\n\n"
            answer += f"{ranked_chunks[0]['content']}\n\n"
            answer += f"The question involves temporal aspects related to {', '.join(term['term'] for term in compound_terms)}."
            return answer
            
        return "Unable to find specific temporal information to answer this question."
    
    def generate_answer(self, question: str, ranked_chunks: List[Dict], 
                       b2_1: Dict, b2_2: Dict, b2_3: Dict, b2_4: Dict = None) -> str:
        """
        Generate answer based on B2 analysis and ranked chunks
        """
        intent = b2_1.get('intent_analysis', {}).get('primary_intent', 'factual')
        answer_type = b2_3.get('answer_prediction', {}).get('primary_type', 'text')
        format_spec = b2_3.get('format_specification', {})
        
        # Handle temporal questions with B2.4 if available
        if (intent == 'temporal' or 
            (b2_4 and b2_4.get('is_temporal_question') and b2_4.get('temporal_confidence', 0) > 0.5)):
            return self.generate_temporal_answer(question, ranked_chunks, b2_1, b2_4)
        
        # Handle different intent types
        if intent == 'comparison' or format_spec.get('structure') == 'comparative':
            # Extract entity for comparison
            entities = b2_1.get('key_entities', [])
            if entities:
                entity = entities[0].get('value', '')
                values = self.extract_values_for_comparison(ranked_chunks, entity)
                
                if len(values) >= 2:
                    years = sorted(values.keys())
                    old_val = values[years[0]]
                    new_val = values[years[-1]]
                    change = new_val - old_val
                    
                    answer = f"Based on the financial data from the provided context:\n\n"
                    for year in years:
                        answer += f"• {entity} deferred income for {year}: ${values[year]} million\n"
                    
                    direction = "increase" if change > 0 else "decrease"
                    answer += f"\nThe change in {entity} deferred income was a {direction} of "
                    answer += f"${abs(round(change, 1))} million "
                    answer += f"(${old_val} million in {years[0]} to ${new_val} million in {years[-1]}).\n\n"
                    answer += f"Data source: {ranked_chunks[0]['chunk_id']}"
                    
                    return answer
                    
        elif answer_type == 'numeric':
            # Look for numeric values in top chunks
            for chunk in ranked_chunks[:3]:
                numbers = re.findall(r'\$(\d+(?:\.\d+)?)', chunk['content'])
                if numbers:
                    answer = f"Based on the context:\n\n"
                    answer += f"{chunk['content']}\n\n"
                    answer += f"The relevant value is ${numbers[0]} million."
                    return answer
                    
        # Default: Use top chunks as context
        if ranked_chunks:
            answer = "Based on the provided context:\n\n"
            for i, chunk in enumerate(ranked_chunks[:3], 1):
                if chunk['score'] > 0.3:  # Only include reasonably scored chunks
                    answer += f"{i}. {chunk['content']}\n\n"
            
            if not any(chunk['score'] > 0.3 for chunk in ranked_chunks[:3]):
                answer = "Unable to find sufficiently relevant information in the context to answer the question."
        else:
            answer = "No relevant chunks found to answer the question."
            
        return answer
    
    def process(self) -> Dict[str, Any]:
        """
        Main processing method
        """
        # Load B2 outputs (including B2.4 if available)
        b1_data, b2_1_data, b2_2_data, b2_3_data, b2_4_data = self.load_b2_outputs()
        
        # Extract question record ID for chunk filtering
        question_id = b1_data.get('question_id', '')
        
        # Load A3 chunks with record ID filtering to prevent data leakage
        chunks = self.load_a3_chunks(target_record_id=question_id)
        
        print(f"Loaded {len(chunks)} chunks from A3 (filtered by record ID: {question_id})")
        print(f"Intent: {b2_1_data.get('intent_analysis', {}).get('primary_intent', 'unknown')}")
        print(f"Expected answer: {b2_3_data.get('answer_prediction', {}).get('primary_type', 'unknown')}")
        if b2_4_data:
            print(f"Temporal analysis: {b2_4_data.get('is_temporal_question', False)} (confidence: {b2_4_data.get('temporal_confidence', 0):.2f})")
        
        # Rank chunks directly using B2 criteria (including B2.4 temporal analysis)
        ranked_chunks = self.rank_chunks_directly(chunks, b2_1_data, b2_2_data, b2_3_data, b2_4_data)
        
        print(f"\nTop 5 chunks after B2-based ranking:")
        for i, chunk in enumerate(ranked_chunks[:5], 1):
            print(f"  {i}. {chunk['chunk_id']} (score: {chunk['score']:.3f})")
        
        # Generate answer (including temporal processing)
        question = b1_data['question']
        answer = self.generate_answer(question, ranked_chunks, b2_1_data, b2_2_data, b2_3_data, b2_4_data)
        
        # Prepare output
        result = {
            "question": question,
            "answer": answer,
            "intent_type": b2_1_data.get('intent_analysis', {}).get('primary_intent', 'unknown'),
            "answer_type": b2_3_data.get('answer_prediction', {}).get('primary_type', 'unknown'),
            "pipeline_path": "B1 → B2 → Direct Ranking → B5.2",
            "chunks_evaluated": len(chunks),
            "top_chunks": [
                {
                    "rank": i+1,
                    "chunk_id": chunk['chunk_id'],
                    "content": chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content'],
                    "score": chunk['score']
                }
                for i, chunk in enumerate(ranked_chunks[:5])
            ],
            "confidence": ranked_chunks[0]['score'] if ranked_chunks else 0.0,
            "b2_criteria_used": {
                "intent": b2_1_data.get('intent_analysis', {}).get('primary_intent'),
                "key_entities": [e.get('value') for e in b2_1_data.get('key_entities', [])],
                "expected_format": b2_3_data.get('format_specification', {}).get('structure'),
                "answer_type": b2_3_data.get('answer_prediction', {}).get('primary_type'),
                "temporal_analysis": b2_4_data.get('is_temporal_question') if b2_4_data else False
            },
            "generation_timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def save_output(self, result: Dict):
        """Save the output to file"""
        output_path = self.output_dir / "B5.2_direct_answer_output.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            
        print(f"\n[OK] Saved output to {output_path}")

def main():
    """Main execution"""
    print("="*60)
    print("B5.2: DIRECT ANSWER GENERATION")
    print("Using B2 outputs directly (bypassing B3/B4)")
    print("="*60)
    
    generator = B52AnswerGenerator()
    
    try:
        print("\nProcessing with simplified pipeline...")
        result = generator.process()
        
        # Display answer
        print("\n" + "="*60)
        print("FINAL ANSWER:")
        print("="*60)
        print(result['answer'])
        print("="*60)
        
        # Save output
        generator.save_output(result)
        
        print(f"\nB5.2 Answer Generation Complete!")
        print(f"Pipeline: {result['pipeline_path']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Top chunk: {result['top_chunks'][0]['chunk_id'] if result['top_chunks'] else 'None'}")
        
    except Exception as e:
        print(f"Error in B5.2: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()