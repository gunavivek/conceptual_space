#!/usr/bin/env python3
"""
Run all B2 components (B2.1, B2.2, B2.3, B2.4) using B1_all_20_records.json as input
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add scripts directory to path
sys.path.append("B_Retrieval_pipeline/scripts")

# Import all B2 components
import B2_1_intent_layer_modeling as B2_1
import B2_2_declarative_transformation as B2_2
import B2_3_answer_expectation_prediction as B2_3
import B2_4_temporal_analysis as B2_4

def run_all_b2_components():
    """Run all B2 components on B1 output"""
    
    # Load B1 outputs
    b1_file = Path("B_Retrieval_pipeline/outputs/B1_all_20_records.json")
    with open(b1_file, 'r', encoding='utf-8') as f:
        b1_data = json.load(f)
    
    print("=" * 80)
    print("RUNNING ALL B2 COMPONENTS ON 20 QUESTIONS")
    print("=" * 80)
    print(f"Input: {b1_file}")
    print(f"Total questions: {b1_data['total_records']}")
    print()
    
    # Process each question through all B2 components
    all_b2_results = []
    
    for idx, b1_output in enumerate(b1_data['b1_outputs'], 1):
        question_id = b1_output['question_id']
        question = b1_output['question']
        
        print(f"\n[{idx}/20] Processing: {question_id}")
        print(f"Question: {question[:60]}...")
        print("-" * 60)
        
        # B2.1: Intent Layer Modeling
        try:
            intent_analysis = B2_1.analyze_intent(question)
            entities = B2_1.extract_key_entities(question)
            b2_1_result = {
                "intent_analysis": intent_analysis,
                "key_entities": entities
            }
            print(f"B2.1 Intent: {intent_analysis.get('primary_intent', 'unknown')}")
        except Exception as e:
            print(f"B2.1 Error: {e}")
            b2_1_result = {"error": str(e)}
        
        # B2.2: Declarative Transformation
        try:
            declarative_forms = B2_2.transform_to_declarative(question)
            b2_2_result = {
                "declarative_forms": declarative_forms
            }
            print(f"B2.2 Declarative: Generated {len(declarative_forms)} forms")
        except Exception as e:
            print(f"B2.2 Error: {e}")
            b2_2_result = {"error": str(e)}
        
        # B2.3: Answer Expectation Prediction
        try:
            # Prepare input for B2.3
            b2_3_input = {
                "question": question,
                "intent_analysis": b2_1_result.get("intent_analysis", {}),
                "declarative_forms": [{"declarative": d} for d in b2_2_result.get("declarative_forms", [])]
            }
            answer_expectation = B2_3.process_answer_expectation(b2_3_input)
            b2_3_result = answer_expectation
            primary_type = answer_expectation.get('answer_prediction', {}).get('primary_type', 'unknown')
            print(f"B2.3 Expected Answer: {primary_type}")
        except Exception as e:
            print(f"B2.3 Error: {e}")
            b2_3_result = {"error": str(e)}
        
        # B2.4: Temporal Analysis (conditional)
        b2_4_result = None
        # Check if temporal analysis is needed
        question_lower = question.lower()
        temporal_indicators = ["when", "year", "2018", "2019", "2020", "change", "between", "from", "to"]
        has_temporal = any(indicator in question_lower for indicator in temporal_indicators)
        
        if has_temporal or intent_analysis.get('primary_intent') == 'temporal':
            try:
                temporal_result = B2_4.process_temporal_question(question)
                b2_4_result = temporal_result
                print(f"B2.4 Temporal: Confidence {temporal_result.get('temporal_confidence', 0):.2f}")
            except Exception as e:
                print(f"B2.4 Error: {e}")
                b2_4_result = {"error": str(e)}
        else:
            print("B2.4 Temporal: Skipped (no temporal indicators)")
        
        # Combine all B2 results for this question
        b2_combined = {
            "question_id": question_id,
            "question": question,
            "b2_1_intent": b2_1_result,
            "b2_2_declarative": b2_2_result,
            "b2_3_answer_expectation": b2_3_result,
            "b2_4_temporal": b2_4_result,
            "processing_timestamp": datetime.now().isoformat()
        }
        
        all_b2_results.append(b2_combined)
    
    # Save individual B2 component outputs
    output_dir = Path("B_Retrieval_pipeline/outputs")
    
    # Save B2.1 outputs
    b2_1_output = {
        "component": "B2.1_intent_layer_modeling",
        "total_processed": len(all_b2_results),
        "results": [{"question_id": r["question_id"], 
                    "question": r["question"],
                    **r["b2_1_intent"]} for r in all_b2_results]
    }
    with open(output_dir / "B2.1_all_20_records.json", 'w', encoding='utf-8') as f:
        json.dump(b2_1_output, f, indent=2)
    
    # Save B2.2 outputs
    b2_2_output = {
        "component": "B2.2_declarative_transformation",
        "total_processed": len(all_b2_results),
        "results": [{"question_id": r["question_id"],
                    "question": r["question"],
                    **r["b2_2_declarative"]} for r in all_b2_results]
    }
    with open(output_dir / "B2.2_all_20_records.json", 'w', encoding='utf-8') as f:
        json.dump(b2_2_output, f, indent=2)
    
    # Save B2.3 outputs
    b2_3_output = {
        "component": "B2.3_answer_expectation_prediction",
        "total_processed": len(all_b2_results),
        "results": [{"question_id": r["question_id"],
                    "question": r["question"],
                    **r["b2_3_answer_expectation"]} for r in all_b2_results]
    }
    with open(output_dir / "B2.3_all_20_records.json", 'w', encoding='utf-8') as f:
        json.dump(b2_3_output, f, indent=2)
    
    # Save B2.4 outputs (only for questions that had temporal analysis)
    temporal_results = [r for r in all_b2_results if r["b2_4_temporal"] is not None]
    b2_4_output = {
        "component": "B2.4_temporal_analysis",
        "total_processed": len(temporal_results),
        "results": [{"question_id": r["question_id"],
                    "question": r["question"],
                    **r["b2_4_temporal"]} for r in temporal_results]
    }
    with open(output_dir / "B2.4_all_temporal_records.json", 'w', encoding='utf-8') as f:
        json.dump(b2_4_output, f, indent=2)
    
    # Save consolidated B2 output
    consolidated_output = {
        "timestamp": datetime.now().isoformat(),
        "input_file": "B1_all_20_records.json",
        "total_questions": len(all_b2_results),
        "temporal_questions": len(temporal_results),
        "all_b2_results": all_b2_results
    }
    with open(output_dir / "B2_all_20_records_complete.json", 'w', encoding='utf-8') as f:
        json.dump(consolidated_output, f, indent=2)
    
    print("\n" + "=" * 80)
    print("B2 PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total questions processed: {len(all_b2_results)}")
    print(f"Temporal questions identified: {len(temporal_results)}")
    print("\nOutput files created:")
    print("  - B2.1_all_20_records.json")
    print("  - B2.2_all_20_records.json") 
    print("  - B2.3_all_20_records.json")
    print(f"  - B2.4_all_temporal_records.json ({len(temporal_results)} questions)")
    print("  - B2_all_20_records_complete.json (consolidated)")
    
    # Display summary statistics
    print("\n" + "=" * 80)
    print("INTENT DISTRIBUTION (B2.1)")
    print("=" * 80)
    intent_counts = {}
    for result in all_b2_results:
        if "intent_analysis" in result["b2_1_intent"]:
            intent = result["b2_1_intent"]["intent_analysis"].get("primary_intent", "unknown")
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    for intent, count in sorted(intent_counts.items()):
        print(f"  {intent}: {count}")
    
    print("\n" + "=" * 80)
    print("EXPECTED ANSWER TYPES (B2.3)")
    print("=" * 80)
    answer_type_counts = {}
    for result in all_b2_results:
        if "answer_prediction" in result["b2_3_answer_expectation"]:
            answer_type = result["b2_3_answer_expectation"]["answer_prediction"].get("primary_type", "unknown")
            answer_type_counts[answer_type] = answer_type_counts.get(answer_type, 0) + 1
    
    for answer_type, count in sorted(answer_type_counts.items()):
        print(f"  {answer_type}: {count}")

if __name__ == "__main__":
    run_all_b2_components()