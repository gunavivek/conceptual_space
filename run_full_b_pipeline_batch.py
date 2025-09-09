#!/usr/bin/env python3
"""
Run FULL B-Pipeline (B1->B2->B3->B4->B5) for all 13 questions
Using the fixed orchestrator with proper doc_id filtering and B2.4 temporal integration
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

def run_full_b_pipeline_batch():
    """Run the complete B-pipeline B1->B2->B3->B4->B5 for all 13 questions"""
    print("="*80)
    print("FULL B-PIPELINE BATCH EXECUTION: B1 -> B2 -> B3 -> B4 -> B5")
    print("Processing 13 questions with fixed doc_id filtering")
    print("="*80)
    
    # The 13 records that have chunks in A3
    target_records = [
        'finqa_test_1212', 'finqa_test_1395', 'finqa_test_1431',
        'finqa_test_1485', 'finqa_test_1552', 'finqa_test_1630',
        'finqa_test_462', 'finqa_test_487', 'finqa_test_515',
        'finqa_test_607', 'finqa_test_723', 'finqa_test_734',
        'finqa_test_869'
    ]
    
    print(f"\\nProcessing {len(target_records)} records through Full B-Pipeline")
    
    # Load the sample data
    sample_file = Path("sample_20_records.parquet")
    df = pd.read_parquet(sample_file)
    
    # Filter to our target records
    filtered_df = df[df['id'].isin(target_records)]
    print(f"Loaded {len(filtered_df)} records with questions\\n")
    
    batch_start = datetime.now()
    
    # Process each question through full B-pipeline
    results = []
    successful = 0
    
    for idx, (_, row) in enumerate(filtered_df.iterrows(), 1):
        record_id = row['id']
        question = row['question']
        
        print(f"\\n[{idx}/{len(filtered_df)}] Processing: {record_id}")
        print(f"  Question: {question[:80]}...")
        
        # Create B1 input for this question
        b1_data = {
            "question_id": record_id,
            "question": question,
            "metadata": {
                "source": "sample_20_records.parquet",
                "record_id": record_id,
                "loaded_at": datetime.now().isoformat()
            }
        }
        
        # Save as current question for B1
        b1_path = Path("B_Retrieval_pipeline/outputs/B1_current_question.json")
        b1_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(b1_path, 'w', encoding='utf-8') as f:
            json.dump(b1_data, f, indent=2)
        
        # Create a custom orchestrator script for full pipeline
        orchestrator_script = '''
import sys
sys.path.append("scripts")
from B_pipeline_orchestrator import BPipelineOrchestrator

class FullBPipelineOrchestrator(BPipelineOrchestrator):
    """Extended orchestrator to run full B1->B2->B3->B4->B5 pipeline"""
    
    def orchestrate_full_pipeline(self, question_index=0):
        """Run complete pipeline with B3 and B4"""
        from datetime import datetime
        
        print("="*80)
        print("FULL B-PIPELINE ORCHESTRATOR: B1 -> B2 -> B3 -> B4 -> B5")
        print("="*80)
        
        start_time = datetime.now()
        
        # B1: Question Input
        question_data = self.run_b1_question_input(question_index)
        if not question_data:
            return None
        
        # B2: Parallel Intent Processing
        b2_results = self.run_b2_intent_processing(question_data)
        
        # B3: Multi-Strategy Concept Matching (NEW!)
        b3_results = self.run_b3_concept_matching(question_data, b2_results)
        
        # B4: Weighted Strategy Combination (NEW!)
        b4_results = self.run_b4_weighted_combination(b3_results)
        
        # B5: Full Answer Generation (not B5.2)
        b5_results = self.run_b5_full_answer_generation(b4_results)
        
        # Summary
        total_elapsed = (datetime.now() - start_time).total_seconds()
        
        print("\\n" + "="*80)
        print("FULL B-PIPELINE SUMMARY")
        print("="*80)
        print(f"Question: {question_data.get('question', 'Unknown')}")
        print(f"Total processing time: {total_elapsed:.3f}s")
        print(f"Pipeline: B1 -> B2 -> B3 -> B4 -> B5 (COMPLETE)")
        
        return {
            'b1': question_data,
            'b2': b2_results,
            'b3': b3_results,
            'b4': b4_results,
            'b5': b5_results,
            'total_time': total_elapsed
        }
    
    def run_b5_full_answer_generation(self, b4_results):
        """Run B5 full answer generation"""
        print("\\n" + "="*60)
        print("B5: FULL ANSWER GENERATION")
        print("="*60)
        
        start_time = datetime.now()
        
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("B5", self.script_dir / "B5_answer_generation.py")
            B5 = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(B5)
            
            # Generate answer using B4 results
            result = B5.generate_answer_from_ranking(b4_results)
            
            # Save B5 output
            b5_output_path = self.output_dir / "B5_full_answer_output.json"
            with open(b5_output_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            self.timing["B5"] = elapsed
            
            print(f"[OK] B5 Complete: Full answer generated in {elapsed:.3f}s")
            print(f"   Output: {b5_output_path}")
            
            return result
            
        except Exception as e:
            print(f"[X] B5 Failed: {e}")
            return {"error": str(e)}

# Run the full pipeline
orchestrator = FullBPipelineOrchestrator()
result = orchestrator.orchestrate_full_pipeline()
'''
        
        # Save the orchestrator script
        temp_script = Path("temp_full_orchestrator.py")
        with open(temp_script, 'w') as f:
            f.write(orchestrator_script)
        
        try:
            # Run the full B-pipeline orchestrator
            result = subprocess.run([sys.executable, str(temp_script)], 
                                   capture_output=True, text=True, cwd="B_Retrieval_pipeline",
                                   timeout=120)
            
            if result.returncode == 0:
                print("  [OK] Full B-pipeline completed")
                
                # Check outputs and collect results
                outputs_to_check = [
                    "B5_full_answer_output.json",
                    "B4_final_ranking.json", 
                    "B3.1_intent_matching_output.json",
                    "B2_intent_processing.json"
                ]
                
                output_status = {}
                for output_file in outputs_to_check:
                    output_path = Path(f"B_Retrieval_pipeline/outputs/{output_file}")
                    if output_path.exists():
                        with open(output_path, 'r') as f:
                            output_status[output_file] = json.load(f)
                    else:
                        output_status[output_file] = None
                
                # Extract key metrics
                confidence = 0.0
                chunks_found = 0
                answer = "No answer generated"
                
                if output_status.get("B5_full_answer_output.json"):
                    b5_data = output_status["B5_full_answer_output.json"]
                    confidence = b5_data.get('confidence', 0.0)
                    chunks_found = len(b5_data.get('top_chunks', []))
                    answer = b5_data.get('answer', 'No answer generated')
                    
                    successful += 1
                    status = "success"
                else:
                    status = "no_output"
                
                print(f"  [OK] Chunks found: {chunks_found}, Confidence: {confidence:.2f}")
                print(f"  Answer: {answer[:100]}...")
                
                results.append({
                    'record_id': record_id,
                    'question': question,
                    'chunks_found': chunks_found,
                    'confidence': confidence,
                    'answer': answer,
                    'status': status,
                    'pipeline': 'B1->B2->B3->B4->B5'
                })
                
            else:
                print(f"  [ERROR] Full B-pipeline failed: {result.stderr[:200]}")
                results.append({
                    'record_id': record_id,
                    'question': question,
                    'status': 'error',
                    'error': result.stderr[:200],
                    'pipeline': 'B1->B2->B3->B4->B5'
                })
                
        except subprocess.TimeoutExpired:
            print(f"  [TIMEOUT] B-pipeline timed out after 120s")
            results.append({
                'record_id': record_id,
                'question': question,
                'status': 'timeout',
                'pipeline': 'B1->B2->B3->B4->B5'
            })
        
        finally:
            # Clean up temp script
            if temp_script.exists():
                temp_script.unlink()
    
    # Save batch results
    total_elapsed = (datetime.now() - batch_start).total_seconds()
    batch_results = {
        'timestamp': datetime.now().isoformat(),
        'total_processed': len(results),
        'successful': successful,
        'total_time': total_elapsed,
        'pipeline_type': 'FULL_B_PIPELINE_B1_B2_B3_B4_B5',
        'results': results
    }
    
    batch_output = Path("B_Retrieval_pipeline/outputs/B_full_pipeline_batch_results.json")
    with open(batch_output, 'w', encoding='utf-8') as f:
        json.dump(batch_results, f, indent=2, ensure_ascii=False)
    
    # Summary
    print("\\n" + "="*80)
    print("FULL B-PIPELINE BATCH SUMMARY")
    print("="*80)
    print(f"Total processed: {len(results)}")
    print(f"Successful (with answers): {successful}/{len(results)}")
    print(f"Total batch time: {total_elapsed:.1f}s")
    print(f"Average time per question: {total_elapsed/len(results):.1f}s")
    
    print("\\nPer-record results:")
    for r in results:
        if r['status'] == 'success':
            print(f"  [OK] {r['record_id']}: {r['chunks_found']} chunks, confidence: {r['confidence']:.2f}")
        elif r['status'] == 'timeout':
            print(f"  [TIMEOUT] {r['record_id']}: Processing timed out")
        else:
            print(f"  [ERROR] {r['record_id']}: {r.get('error', 'Unknown error')[:100]}")
    
    print(f"\\nResults saved to: {batch_output}")
    print("\\n" + "="*80)
    print("FULL B-PIPELINE BATCH PROCESSING COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    run_full_b_pipeline_batch()