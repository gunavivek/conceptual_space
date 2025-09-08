#!/usr/bin/env python3
"""
B-Pipeline Orchestrator
Coordinates B1-B4 scripts to process questions through intent analysis
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import importlib.util

class BPipelineOrchestrator:
    """
    Orchestrates the B-Pipeline (Intent Space) processing
    B1 → B2 (parallel) → B3 (parallel) → B4
    """
    
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.output_dir = self.script_dir.parent / "outputs"
        self.output_dir.mkdir(exist_ok=True)
        
        # Pipeline timing
        self.timing = {}
        
    def has_temporal_indicators(self, question):
        """Check if question has temporal indicators that warrant B2.4 analysis"""
        question_lower = question.lower()
        
        # Basic temporal indicators
        temporal_words = [
            "when", "time", "timing", "date", "period", "annual", "annually", 
            "yearly", "monthly", "quarterly", "frequency", "how often"
        ]
        
        # Compound temporal indicators
        compound_indicators = [
            "multi-year", "long-term", "coverage period", "billing period",
            "service period", "agreement period", "contract period"
        ]
        
        # Check for basic temporal words
        has_basic_temporal = any(word in question_lower for word in temporal_words)
        
        # Check for compound temporal terms
        has_compound_temporal = any(term in question_lower for term in compound_indicators)
        
        return has_basic_temporal or has_compound_temporal
        
    def run_b1_question_input(self, question_index=0):
        """Run B1: Question Input"""
        print("\n" + "="*60)
        print("B1: QUESTION INPUT LAYER")
        print("="*60)
        
        start_time = datetime.now()
        
        try:
            # Load question from B1 output (already processed by B1 script)
            b1_output_path = self.output_dir / "B1_current_question.json"
            
            if b1_output_path.exists():
                # Load from B1 output file
                with open(b1_output_path, 'r') as f:
                    question_data = json.load(f)
                print(f"[OK] Loaded question from B1 output: {b1_output_path}")
            else:
                # Fallback: Import and run B1 if output doesn't exist
                spec = importlib.util.spec_from_file_location("B1", self.script_dir / "B1_read_question.py")
                B1 = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(B1)
                
                # Load question
                question_data = B1.load_question_from_parquet(
                    data_path="../A_Concept_pipeline/data/single_record_finqa_test_617.parquet",
                    question_index=question_index
                )
                
                # Analyze question
                analysis = B1.analyze_question(question_data["question"])
                question_data["analysis"] = analysis
                
                # Save B1 output
                with open(b1_output_path, 'w') as f:
                    json.dump(question_data, f, indent=2)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            self.timing["B1"] = elapsed
            
            print(f"[OK] B1 Complete: Question loaded in {elapsed:.3f}s")
            print(f"   Question: {question_data['question']}")
            
            # Get analysis from question_data
            analysis = question_data.get('analysis', {})
            print(f"   Type: {analysis.get('question_type', 'unknown')}")
            print(f"   Expected Answer: {analysis.get('expected_answer_type', 'unknown')}")
            print(f"   Output: {b1_output_path}")
            
            return question_data
            
        except Exception as e:
            print(f"[X] B1 Failed: {e}")
            return None
    
    def run_b2_intent_processing(self, question_data):
        """Run B2: Parallel Intent Processing (B2.1, B2.2, B2.3)"""
        print("\n" + "="*60)
        print("B2: PARALLEL INTENT PROCESSING")
        print("="*60)
        
        start_time = datetime.now()
        b2_results = {}
        
        question_text = question_data["question"]
        
        # B2.1: Intent Layer Modeling
        try:
            spec = importlib.util.spec_from_file_location("B2_1", self.script_dir / "B2_1_intent_layer_modeling.py")
            B2_1 = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(B2_1)
            
            intent_analysis = B2_1.analyze_intent(question_text)
            entities = B2_1.extract_key_entities(question_text)
            
            b2_1_output = {
                "intent_analysis": intent_analysis,
                "key_entities": entities,
                "question": question_text,
                "processing_timestamp": datetime.now().isoformat()
            }
            
            # Save B2.1 individual output
            b2_1_path = self.output_dir / "B2.1_intent_layer_output.json"
            with open(b2_1_path, 'w') as f:
                json.dump(b2_1_output, f, indent=2)
            
            b2_results["intent_modeling"] = b2_1_output
            print(f"[OK] B2.1: Intent = {intent_analysis.get('primary_intent', 'unknown')}")
            print(f"   Output: {b2_1_path}")
            
        except Exception as e:
            print(f"[X] B2.1 Failed: {e}")
            b2_results["intent_modeling"] = {"error": str(e)}
        
        # B2.2: Declarative Transformation
        try:
            spec = importlib.util.spec_from_file_location("B2_2", self.script_dir / "B2_2_declarative_transformation.py")
            B2_2 = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(B2_2)
            
            declarative = B2_2.transform_to_declarative(question_text)
            
            b2_2_output = {
                "question": question_text,
                "declarative_forms": declarative,
                "processing_timestamp": datetime.now().isoformat()
            }
            
            # Save B2.2 individual output
            b2_2_path = self.output_dir / "B2.2_declarative_output.json"
            with open(b2_2_path, 'w') as f:
                json.dump(b2_2_output, f, indent=2)
            
            b2_results["declarative_transformation"] = declarative
            print(f"[OK] B2.2: Declarative transformation completed")
            print(f"   Output: {b2_2_path}")
            
        except Exception as e:
            print(f"[X] B2.2 Failed: {e}")
            b2_results["declarative_transformation"] = {"error": str(e)}
        
        # B2.3: Answer Expectation Prediction
        try:
            spec = importlib.util.spec_from_file_location("B2_3", self.script_dir / "B2_3_answer_expectation_prediction.py")
            B2_3 = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(B2_3)
            
            # Prepare input data for B2.3
            b2_3_input = {
                "question": question_text,
                "intent_analysis": b2_results.get("intent_modeling", {}).get("intent_analysis", {}),
                "declarative_forms": [{"declarative": d} for d in b2_results.get("declarative_transformation", [])]
            }
            answer_prediction = B2_3.process_answer_expectation(b2_3_input)
            
            # Save B2.3 individual output
            b2_3_path = self.output_dir / "B2.3_answer_expectation_output.json"
            with open(b2_3_path, 'w') as f:
                json.dump(answer_prediction, f, indent=2)
            
            b2_results["answer_expectation"] = answer_prediction
            print(f"[OK] B2.3: Expected answer format = {answer_prediction.get('answer_prediction', {}).get('primary_type', 'unknown')}")
            print(f"   Output: {b2_3_path}")
            
        except Exception as e:
            print(f"[X] B2.3 Failed: {e}")
            b2_results["answer_expectation"] = {"error": str(e)}
        
        # B2.4: Temporal Analysis (conditional - only for temporal questions)
        temporal_analysis = None
        intent_primary = b2_results.get("intent_modeling", {}).get("intent_analysis", {}).get("primary_intent")
        
        # Check if question needs temporal analysis
        if intent_primary == "temporal" or self.has_temporal_indicators(question_text):
            try:
                spec = importlib.util.spec_from_file_location("B2_4", self.script_dir / "B2_4_temporal_analysis.py")
                B2_4 = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(B2_4)
                
                temporal_analysis = B2_4.process_temporal_question(question_text)
                
                # Save B2.4 individual output
                b2_4_path = self.output_dir / "B2.4_temporal_analysis_output.json"
                with open(b2_4_path, 'w') as f:
                    json.dump(temporal_analysis, f, indent=2)
                
                b2_results["temporal_analysis"] = temporal_analysis
                print(f"[OK] B2.4: Temporal analysis completed (confidence: {temporal_analysis.get('temporal_confidence', 0):.2f})")
                print(f"   Compound terms: {len(temporal_analysis.get('temporal_entities', {}).get('compound_temporal_terms', []))}")
                print(f"   Output: {b2_4_path}")
                
            except Exception as e:
                print(f"[X] B2.4 Failed: {e}")
                b2_results["temporal_analysis"] = {"error": str(e)}
        else:
            print(f"[--] B2.4: Skipped (not a temporal question)")
        
        # Save B2 output
        b2_output_path = self.output_dir / "B2_intent_processing.json"
        with open(b2_output_path, 'w') as f:
            json.dump(b2_results, f, indent=2)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        self.timing["B2"] = elapsed
        
        print(f"[OK] B2 Complete: Intent processing in {elapsed:.3f}s")
        print(f"   Output: {b2_output_path}")
        
        return b2_results
    
    def run_b3_concept_matching(self, question_data, b2_results):
        """Run B3: Multi-Strategy Concept Matching (B3.1, B3.2, B3.3)"""
        print("\n" + "="*60)
        print("B3: MULTI-STRATEGY CONCEPT MATCHING")
        print("="*60)
        
        start_time = datetime.now()
        b3_results = {}
        
        # Load A3 chunks for matching
        chunk_path = Path("../../A_Concept_pipeline/outputs/A3_raw_chunks_no_dedup.json")
        if not chunk_path.exists():
            chunk_path = Path("../../A_Concept_pipeline/outputs/A3_multi_strategy_chunks.json")
        
        if chunk_path.exists():
            with open(chunk_path, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
                chunks = chunk_data.get('chunks', [])
            print(f"   Loaded {len(chunks)} chunks from A-Pipeline")
        else:
            print("❌ No A3 chunks found - cannot perform matching")
            chunks = []
        
        question_text = question_data["question"]
        
        # B3.1: Intent-based Matching
        try:
            spec = importlib.util.spec_from_file_location("B3_1", self.script_dir / "B3.1_intent_matching.py")
            B3_1 = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(B3_1)
            
            intent_matches = B3_1.match_chunks_by_intent(
                question_text, 
                chunks, 
                b2_results.get("intent_modeling", {})
            )
            
            # Save B3.1 individual output
            b3_1_path = self.output_dir / "B3.1_intent_matching_output.json"
            with open(b3_1_path, 'w') as f:
                json.dump(intent_matches, f, indent=2)
            
            b3_results["intent_matching"] = intent_matches
            print(f"[OK] B3.1: Found {len(intent_matches.get('ranked_chunks', []))} intent matches")
            print(f"   Output: {b3_1_path}")
            
        except Exception as e:
            print(f"[X] B3.1 Failed: {e}")
            b3_results["intent_matching"] = {"error": str(e)}
        
        # B3.2: Declarative Matching
        try:
            spec = importlib.util.spec_from_file_location("B3_2", self.script_dir / "B3.2_declarative_matching.py")
            B3_2 = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(B3_2)
            
            declarative_matches = B3_2.match_declarative_patterns(
                chunks,
                b2_results.get("declarative_transformation", {})
            )
            
            # Save B3.2 individual output
            b3_2_path = self.output_dir / "B3.2_declarative_matching_output.json"
            with open(b3_2_path, 'w') as f:
                json.dump(declarative_matches, f, indent=2)
            
            b3_results["declarative_matching"] = declarative_matches
            print(f"[OK] B3.2: Found {len(declarative_matches.get('ranked_chunks', []))} declarative matches")
            print(f"   Output: {b3_2_path}")
            
        except Exception as e:
            print(f"[X] B3.2 Failed: {e}")
            b3_results["declarative_matching"] = {"error": str(e)}
        
        # B3.3: Answer Backward Matching
        try:
            spec = importlib.util.spec_from_file_location("B3_3", self.script_dir / "B3.3_answer_backward_matching.py")
            B3_3 = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(B3_3)
            
            backward_matches = B3_3.match_by_answer_expectations(
                chunks,
                b2_results.get("answer_expectation", {})
            )
            
            # Save B3.3 individual output
            b3_3_path = self.output_dir / "B3.3_answer_backward_output.json"
            with open(b3_3_path, 'w') as f:
                json.dump(backward_matches, f, indent=2)
            
            b3_results["answer_backward"] = backward_matches
            print(f"[OK] B3.3: Found {len(backward_matches.get('ranked_chunks', []))} backward matches")
            print(f"   Output: {b3_3_path}")
            
        except Exception as e:
            print(f"[X] B3.3 Failed: {e}")
            b3_results["answer_backward"] = {"error": str(e)}
        
        # Save B3 output
        b3_output_path = self.output_dir / "B3_concept_matching.json"
        with open(b3_output_path, 'w') as f:
            json.dump(b3_results, f, indent=2)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        self.timing["B3"] = elapsed
        
        print(f"[OK] B3 Complete: Concept matching in {elapsed:.3f}s")
        print(f"   Output: {b3_output_path}")
        
        return b3_results
    
    def run_b4_weighted_combination(self, b3_results):
        """Run B4: Weighted Strategy Combination"""
        print("\n" + "="*60)
        print("B4: WEIGHTED STRATEGY COMBINATION")
        print("="*60)
        
        start_time = datetime.now()
        
        try:
            spec = importlib.util.spec_from_file_location("B4", self.script_dir / "B4_weighted_strategy_combination.py")
            B4 = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(B4)
            
            # Combine strategy results
            final_ranking = B4.combine_strategy_results(b3_results)
            
            # Save B4 output
            b4_output_path = self.output_dir / "B4_final_ranking.json"
            with open(b4_output_path, 'w') as f:
                json.dump(final_ranking, f, indent=2)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            self.timing["B4"] = elapsed
            
            ranked_chunks = final_ranking.get('ranked_chunks', [])
            print(f"[OK] B4 Complete: Final ranking in {elapsed:.3f}s")
            print(f"   Top-ranked chunks: {len(ranked_chunks)}")
            if ranked_chunks:
                top_chunk = ranked_chunks[0]
                print(f"   Best match: {top_chunk.get('chunk_id', 'unknown')} (score: {top_chunk.get('combined_score', 0):.3f})")
            print(f"   Output: {b4_output_path}")
            
            return final_ranking
            
        except Exception as e:
            print(f"[X] B4 Failed: {e}")
            return {"error": str(e)}
    
    def run_b52_direct_answer_generation(self):
        """Run B5.2: Direct Answer Generation using B2 outputs"""
        print("\n" + "="*60)
        print("B5.2: DIRECT ANSWER GENERATION")
        print("="*60)
        
        start_time = datetime.now()
        
        try:
            # Import and run B5.2
            spec = importlib.util.spec_from_file_location("B5_2", self.script_dir / "B5.2_generate_answer.py")
            B5_2 = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(B5_2)
            
            # Create generator and process
            generator = B5_2.B52AnswerGenerator()
            result = generator.process()
            
            # Save output
            generator.save_output(result)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            self.timing["B5.2"] = elapsed
            
            print(f"[OK] B5.2 Complete: Answer generated in {elapsed:.3f}s")
            print(f"   Confidence: {result.get('confidence', 0):.3f}")
            print(f"   Top chunk: {result['top_chunks'][0]['chunk_id'] if result.get('top_chunks') else 'None'}")
            
            return result
            
        except Exception as e:
            print(f"[X] B5.2 Failed: {e}")
            return {"error": str(e)}

    def orchestrate(self, question_index=0):
        """Main orchestration method - Simplified Pipeline"""
        print("="*80)
        print("B-PIPELINE ORCHESTRATOR: SIMPLIFIED INTENT SPACE PROCESSING")
        print("B1 -> B2 (parallel) -> B5.2 (Direct Answer Generation)")
        print("="*80)
        
        start_time = datetime.now()
        
        # B1: Question Input
        question_data = self.run_b1_question_input(question_index)
        if not question_data:
            print("[X] Pipeline failed at B1")
            return
        
        # B2: Parallel Intent Processing
        b2_results = self.run_b2_intent_processing(question_data)
        
        # B5.2: Direct Answer Generation (bypassing B3/B4)
        final_results = self.run_b52_direct_answer_generation()
        
        # Summary
        total_elapsed = (datetime.now() - start_time).total_seconds()
        
        print("\n" + "="*80)
        print("B-PIPELINE SUMMARY (SIMPLIFIED)")
        print("="*80)
        print(f"Question: {question_data.get('question', 'Unknown')}")
        print(f"Total processing time: {total_elapsed:.3f}s")
        print(f"Pipeline: B1 -> B2 -> B5.2 (Direct)")
        print("\nStage timing:")
        for stage, elapsed in self.timing.items():
            print(f"  {stage}: {elapsed:.3f}s")
        
        if final_results and 'top_chunks' in final_results:
            top_chunks = final_results['top_chunks'][:3]
            print(f"\nTop 3 chunks with B2-based ranking:")
            for chunk in top_chunks:
                print(f"  {chunk['rank']}. {chunk['chunk_id']} (score: {chunk['score']:.3f})")
        
        print(f"\nFinal Answer Generated:")
        if final_results and 'answer' in final_results:
            # Show first 100 chars of answer
            answer_preview = final_results['answer'][:100] + "..." if len(final_results['answer']) > 100 else final_results['answer']
            print(f"  {answer_preview}")
        
        print("\n[OK] B-Pipeline complete - Answer ready!")

def main():
    """Main execution"""
    orchestrator = BPipelineOrchestrator()
    orchestrator.orchestrate(question_index=0)

if __name__ == "__main__":
    main()