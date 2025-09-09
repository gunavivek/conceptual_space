#!/usr/bin/env python3
"""
Run FULL B-Pipeline (B1->B2->B3->B4->B5) for all 13 questions
Direct approach using existing orchestrator
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime
import pandas as pd

# Set UTF-8 encoding for Windows compatibility
import locale
if os.name == 'nt':  # Windows
    try:
        locale.setlocale(locale.LC_ALL, 'C.UTF-8')
    except:
        try:
            locale.setlocale(locale.LC_ALL, 'en_US.UTF-8') 
        except:
            pass  # Use system default
    # Set environment variables for UTF-8
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add scripts directory to path
sys.path.append("B_Retrieval_pipeline/scripts")
from B_pipeline_orchestrator import BPipelineOrchestrator

class FullBatchOrchestrator(BPipelineOrchestrator):
    """Extended orchestrator for full B-pipeline batch processing"""
    
    def orchestrate_full_pipeline_single(self, question_data):
        """Run complete pipeline B1->B2->B3->B4->B5 for single question"""
        print(f"\n{'='*80}")
        print(f"FULL B-PIPELINE: {question_data.get('question_id', 'Unknown')}")
        print(f"Question: {question_data.get('question', 'Unknown')[:60]}...")
        print(f"{'='*80}")
        
        start_time = datetime.now()
        
        try:
            # B2: Parallel Intent Processing (uses current question from B1 output)
            b2_results = self.run_b2_intent_processing(question_data)
            if not b2_results:
                return {"error": "B2 failed", "stage": "B2"}
            
            # B3: Multi-Strategy Concept Matching
            b3_results = self.run_b3_concept_matching(question_data, b2_results)
            if not b3_results:
                return {"error": "B3 failed", "stage": "B3"}
            
            # B4: Weighted Strategy Combination
            b4_results = self.run_b4_weighted_combination(b3_results)
            if not b4_results:
                return {"error": "B4 failed", "stage": "B4"}
            
            # B5: Full Answer Generation
            b5_results = self.run_b5_full_answer_generation(question_data, b4_results)
            
            # Calculate metrics
            total_elapsed = (datetime.now() - start_time).total_seconds()
            
            result = {
                'question_id': question_data.get('question_id'),
                'question': question_data.get('question'),
                'status': 'success',
                'total_time': total_elapsed,
                'pipeline': 'B1->B2->B3->B4->B5',
                'b2_results': b2_results,
                'b3_results': b3_results, 
                'b4_results': b4_results,
                'b5_results': b5_results
            }
            
            # Extract key metrics
            if b5_results and not b5_results.get('error'):
                result['chunks_found'] = len(b5_results.get('top_chunks', []))
                result['confidence'] = b5_results.get('confidence', 0.0)
                result['answer'] = b5_results.get('answer', '')
            else:
                result['chunks_found'] = 0
                result['confidence'] = 0.0
                result['answer'] = 'No answer generated'
                if b5_results and b5_results.get('error'):
                    result['b5_error'] = b5_results['error']
            
            print(f"[OK] Full pipeline completed in {total_elapsed:.3f}s")
            print(f"   Chunks found: {result['chunks_found']}")
            print(f"   Confidence: {result['confidence']:.2f}")
            
            return result
            
        except Exception as e:
            total_elapsed = (datetime.now() - start_time).total_seconds()
            print(f"[ERROR] Pipeline failed: {str(e)}")
            return {
                'question_id': question_data.get('question_id'),
                'question': question_data.get('question'),
                'status': 'error',
                'error': str(e),
                'total_time': total_elapsed,
                'pipeline': 'B1->B2->B3->B4->B5'
            }
    
    def run_b5_full_answer_generation(self, question_data, b4_results):
        """Run B5 full answer generation"""
        print(f"\n{'='*60}")
        print("B5: FULL ANSWER GENERATION")
        print(f"{'='*60}")
        
        start_time = datetime.now()
        
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("B5", self.script_dir / "B5_answer_generation.py")
            B5 = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(B5)
            
            # Generate answer using B4 results
            result = B5.generate_answer(question_data, b4_results)
            
            # Save B5 output
            b5_output_path = self.output_dir / "B5_full_answer_output.json"
            with open(b5_output_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            self.timing["B5"] = elapsed
            
            print(f"[OK] B5 Complete: Full answer generated in {elapsed:.3f}s")
            
            return result
            
        except Exception as e:
            print(f"[X] B5 Failed: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

def run_batch_full_b_pipeline():
    """Run full B-pipeline for all 13 questions"""
    print(f"{'='*80}")
    print("FULL B-PIPELINE BATCH EXECUTION: B1 -> B2 -> B3 -> B4 -> B5")
    print("Processing 13 questions with doc_id filtering and B2.4 integration")
    print(f"{'='*80}")
    
    # Target records that have chunks in A3
    target_records = [
        'finqa_test_1212', 'finqa_test_1395', 'finqa_test_1431',
        'finqa_test_1485', 'finqa_test_1552', 'finqa_test_1630',
        'finqa_test_462', 'finqa_test_487', 'finqa_test_515',
        'finqa_test_607', 'finqa_test_723', 'finqa_test_734',
        'finqa_test_869'
    ]
    
    # Change to correct working directory
    os.chdir("B_Retrieval_pipeline")
    
    # Initialize orchestrator
    orchestrator = FullBatchOrchestrator()
    
    # Load sample data
    sample_file = Path("../sample_20_records.parquet")
    df = pd.read_parquet(sample_file)
    
    # Filter to target records
    filtered_df = df[df['id'].isin(target_records)]
    print(f"\nLoaded {len(filtered_df)} records for processing\n")
    
    batch_start = datetime.now()
    results = []
    successful = 0
    
    for idx, (_, row) in enumerate(filtered_df.iterrows(), 1):
        record_id = row['id']
        question = row['question']
        
        print(f"\n[{idx}/{len(filtered_df)}] Processing: {record_id}")
        
        # Create question data for B1
        question_data = {
            "question_id": record_id,
            "question": question,
            "metadata": {
                "source": "sample_20_records.parquet",
                "record_id": record_id,
                "loaded_at": datetime.now().isoformat()
            }
        }
        
        # Save as current question for B1
        b1_path = Path("outputs/B1_current_question.json")
        b1_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(b1_path, 'w', encoding='utf-8') as f:
            json.dump(question_data, f, indent=2)
        
        # Run full pipeline
        result = orchestrator.orchestrate_full_pipeline_single(question_data)
        results.append(result)
        
        if result.get('status') == 'success':
            successful += 1
    
    # Calculate batch metrics
    total_elapsed = (datetime.now() - batch_start).total_seconds()
    
    batch_results = {
        'timestamp': datetime.now().isoformat(),
        'total_processed': len(results),
        'successful': successful,
        'failed': len(results) - successful,
        'success_rate': successful / len(results) * 100 if results else 0,
        'total_time': total_elapsed,
        'avg_time_per_question': total_elapsed / len(results) if results else 0,
        'pipeline_type': 'FULL_B_PIPELINE_B1_B2_B3_B4_B5',
        'features': [
            'doc_id_filtering_enabled',
            'B2.4_temporal_integration',
            'multi_strategy_concept_matching',
            'weighted_combination_ranking',
            'full_answer_generation'
        ],
        'results': results
    }
    
    # Save batch results
    batch_output = Path("outputs/B_full_pipeline_batch_results.json")
    with open(batch_output, 'w', encoding='utf-8') as f:
        json.dump(batch_results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n{'='*80}")
    print("FULL B-PIPELINE BATCH SUMMARY")
    print(f"{'='*80}")
    print(f"Total processed: {len(results)}")
    print(f"Successful: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
    print(f"Failed: {len(results) - successful}")
    print(f"Total batch time: {total_elapsed:.1f}s")
    print(f"Average time per question: {total_elapsed/len(results):.1f}s")
    
    print(f"\nDetailed results:")
    for r in results:
        if r.get('status') == 'success':
            chunks = r.get('chunks_found', 0)
            conf = r.get('confidence', 0.0)
            time_taken = r.get('total_time', 0.0)
            print(f"  [OK] {r['question_id']}: {chunks} chunks, conf: {conf:.2f}, time: {time_taken:.1f}s")
        else:
            error = r.get('error', 'Unknown error')
            time_taken = r.get('total_time', 0.0)
            print(f"  [ERROR] {r['question_id']}: {error[:60]}..., time: {time_taken:.1f}s")
    
    print(f"\nResults saved to: {batch_output}")
    print(f"{'='*80}")
    print("FULL B-PIPELINE BATCH PROCESSING COMPLETE!")
    print(f"{'='*80}")

if __name__ == "__main__":
    run_batch_full_b_pipeline()