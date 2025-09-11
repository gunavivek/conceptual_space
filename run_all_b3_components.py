#!/usr/bin/env python3
"""
Run all B3 components (B3.1, B3.2, B3.3) using B2 outputs and A3 chunks
Updates existing output files with overwriting behavior
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add scripts directory to path
sys.path.append("B_Retrieval_pipeline/scripts")

# Import all B3 components using importlib (files have dots in names)
import importlib.util

def load_b3_module(script_name, alias):
    """Load B3 module with dots in filename"""
    script_path = Path("B_Retrieval_pipeline/scripts") / script_name
    spec = importlib.util.spec_from_file_location(alias, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

B3_1 = load_b3_module("B3.1_intent_matching.py", "B3_1")
B3_2 = load_b3_module("B3.2_declarative_matching.py", "B3_2") 
B3_3 = load_b3_module("B3.3_answer_backward_matching.py", "B3_3")

def load_a3_chunks():
    """Load A3 chunks with proper path resolution"""
    # Use absolute paths like the orchestrator does
    script_dir = Path(__file__).parent
    project_root = script_dir
    
    chunk_path = project_root / "A_Concept_pipeline" / "outputs" / "A3_multi_strategy_chunks.json"
    if not chunk_path.exists():
        chunk_path = project_root / "A_Concept_pipeline" / "outputs" / "A3_raw_chunks_no_dedup.json"
    
    if chunk_path.exists():
        with open(chunk_path, 'r', encoding='utf-8') as f:
            chunk_data = json.load(f)
            return chunk_data.get('chunks', [])
    else:
        print(f"[WARNING] No A3 chunks found at {chunk_path}")
        return []

def filter_chunks_by_question_id(chunks, question_id):
    """Filter chunks to prevent cross-document contamination"""
    filtered_chunks = []
    for chunk in chunks:
        chunk_id = chunk.get('chunk_id', '')
        if chunk_id.startswith(question_id):
            filtered_chunks.append(chunk)
    return filtered_chunks

def run_all_b3_components():
    """Run all B3 components on B2 outputs and A3 chunks"""
    
    # Load B2 consolidated output
    b2_file = Path("B_Retrieval_pipeline/outputs/B2_current_output.json")
    with open(b2_file, 'r', encoding='utf-8') as f:
        b2_data = json.load(f)
    
    # Load A3 chunks
    all_chunks = load_a3_chunks()
    
    print("=" * 80)
    print("RUNNING ALL B3 COMPONENTS - CONCEPT MATCHING")
    print("=" * 80)
    print(f"B2 Input: {b2_file}")
    print(f"Total questions: {b2_data['total_questions']}")
    print(f"A3 chunks loaded: {len(all_chunks)}")
    print()
    
    # Process each question through all B3 components
    all_b3_results = []
    
    for idx, b2_result in enumerate(b2_data['all_b2_results'], 1):
        question_id = b2_result['question_id']
        question = b2_result['question']
        
        print(f"\\n[{idx}/20] Processing: {question_id}")
        print(f"Question: {question[:60]}...")
        
        # Filter chunks by question ID to prevent data leakage
        filtered_chunks = filter_chunks_by_question_id(all_chunks, question_id)
        print(f"Filtered chunks: {len(all_chunks)} -> {len(filtered_chunks)} (target: {question_id})")
        
        if len(filtered_chunks) == 0:
            print(f"[WARNING] No chunks found for {question_id} - skipping B3 processing")
            b3_combined = {
                "question_id": question_id,
                "question": question,
                "b3_1_intent_matching": {"error": "No chunks found for this question_id"},
                "b3_2_declarative_matching": {"error": "No chunks found for this question_id"},
                "b3_3_answer_backward": {"error": "No chunks found for this question_id"},
                "processing_timestamp": datetime.now().isoformat()
            }
            all_b3_results.append(b3_combined)
            continue
        
        print("-" * 60)
        
        # B3.1: Intent-Based Matching
        try:
            intent_matches = B3_1.match_chunks_by_intent(
                question,
                filtered_chunks,
                b2_result.get("b2_1_intent", {}),
                b2_result.get("b2_4_temporal", {})
            )
            b3_1_result = intent_matches
            ranked_count = len(intent_matches.get('ranked_chunks', []))
            print(f"B3.1 Intent Matching: {ranked_count} chunks matched")
        except Exception as e:
            print(f"B3.1 Error: {e}")
            b3_1_result = {"error": str(e)}
        
        # B3.2: Declarative Pattern Matching
        try:
            declarative_matches = B3_2.match_declarative_patterns(
                filtered_chunks,
                b2_result.get("b2_2_declarative", {})
            )
            b3_2_result = declarative_matches
            ranked_count = len(declarative_matches.get('ranked_chunks', []))
            print(f"B3.2 Declarative Matching: {ranked_count} chunks matched")
        except Exception as e:
            print(f"B3.2 Error: {e}")
            b3_2_result = {"error": str(e)}
        
        # B3.3: Answer Backward Matching
        try:
            backward_matches = B3_3.match_by_answer_expectations(
                filtered_chunks,
                b2_result.get("b2_3_answer_expectation", {})
            )
            b3_3_result = backward_matches
            ranked_count = len(backward_matches.get('ranked_chunks', []))
            print(f"B3.3 Answer Backward Matching: {ranked_count} chunks matched")
        except Exception as e:
            print(f"B3.3 Error: {e}")
            b3_3_result = {"error": str(e)}
        
        # Combine all B3 results for this question
        b3_combined = {
            "question_id": question_id,
            "question": question,
            "chunks_available": len(filtered_chunks),
            "b3_1_intent_matching": b3_1_result,
            "b3_2_declarative_matching": b3_2_result,
            "b3_3_answer_backward": b3_3_result,
            "processing_timestamp": datetime.now().isoformat()
        }
        
        all_b3_results.append(b3_combined)
    
    # Save individual B3 component outputs (overwriting)
    output_dir = Path("B_Retrieval_pipeline/outputs")
    
    # Save B3.1 outputs
    b3_1_output = {
        "component": "B3.1_intent_matching",
        "timestamp": datetime.now().isoformat(),
        "total_processed": len(all_b3_results),
        "results": [{"question_id": r["question_id"], 
                    "question": r["question"],
                    "chunks_available": r.get("chunks_available", 0),
                    **r["b3_1_intent_matching"]} for r in all_b3_results]
    }
    with open(output_dir / "B3.1_intent_matching_output.json", 'w', encoding='utf-8') as f:
        json.dump(b3_1_output, f, indent=2)
    
    # Save B3.2 outputs
    b3_2_output = {
        "component": "B3.2_declarative_matching",
        "timestamp": datetime.now().isoformat(),
        "total_processed": len(all_b3_results),
        "results": [{"question_id": r["question_id"],
                    "question": r["question"],
                    "chunks_available": r.get("chunks_available", 0),
                    **r["b3_2_declarative_matching"]} for r in all_b3_results]
    }
    with open(output_dir / "B3.2_declarative_matching_output.json", 'w', encoding='utf-8') as f:
        json.dump(b3_2_output, f, indent=2)
    
    # Save B3.3 outputs
    b3_3_output = {
        "component": "B3.3_answer_backward_matching",
        "timestamp": datetime.now().isoformat(),
        "total_processed": len(all_b3_results),
        "results": [{"question_id": r["question_id"],
                    "question": r["question"],
                    "chunks_available": r.get("chunks_available", 0),
                    **r["b3_3_answer_backward"]} for r in all_b3_results]
    }
    with open(output_dir / "B3.3_answer_backward_matching_output.json", 'w', encoding='utf-8') as f:
        json.dump(b3_3_output, f, indent=2)
    
    # Save consolidated B3 output (overwriting)
    consolidated_output = {
        "timestamp": datetime.now().isoformat(),
        "input_b2_file": "B2_current_output.json",
        "a3_chunks_loaded": len(all_chunks),
        "total_questions": len(all_b3_results),
        "all_b3_results": all_b3_results
    }
    with open(output_dir / "B3_current_output.json", 'w', encoding='utf-8') as f:
        json.dump(consolidated_output, f, indent=2)
    
    print("\\n" + "=" * 80)
    print("B3 PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total questions processed: {len(all_b3_results)}")
    print(f"A3 chunks available: {len(all_chunks)}")
    print("\\nOutput files created/updated:")
    print("  - B3.1_intent_matching_output.json")
    print("  - B3.2_declarative_matching_output.json") 
    print("  - B3.3_answer_backward_matching_output.json")
    print("  - B3_current_output.json (consolidated)")
    
    # Display summary statistics
    print("\\n" + "=" * 80)
    print("CHUNK MATCHING SUMMARY")
    print("=" * 80)
    
    questions_with_chunks = sum(1 for r in all_b3_results if r.get("chunks_available", 0) > 0)
    questions_without_chunks = len(all_b3_results) - questions_with_chunks
    
    print(f"Questions with chunks available: {questions_with_chunks}")
    print(f"Questions without chunks: {questions_without_chunks}")
    
    if questions_with_chunks > 0:
        total_chunks_used = sum(r.get("chunks_available", 0) for r in all_b3_results)
        avg_chunks_per_question = total_chunks_used / questions_with_chunks
        print(f"Average chunks per question: {avg_chunks_per_question:.1f}")

if __name__ == "__main__":
    run_all_b3_components()