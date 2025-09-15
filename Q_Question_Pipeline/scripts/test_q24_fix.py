#!/usr/bin/env python3
"""
Quick Q2.4 Fix Verification Test
Tests Q2.4 with multiple questions to verify the fix works
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.Q2_4_temporal_coordinate_mapping import Q24TemporalCoordinateMapping

def test_q24_fix():
    """Test Q2.4 with multiple questions"""
    processor = Q24TemporalCoordinateMapping()

    test_questions = [
        "finqa_test_1630",
        "finqa_test_1431",
        "finqa_test_1212",
        "finqa_test_462",
        "finqa_test_1552"
    ]

    print("Q2.4 FIX VERIFICATION TEST")
    print("=" * 40)

    success_count = 0

    for i, question_id in enumerate(test_questions, 1):
        print(f"\n[{i}/5] Testing {question_id}")

        # Load question data (test the fix)
        question_data = processor.load_question_from_q1(question_id)
        if question_data:
            print(f"  [SUCCESS] Question loaded: {question_data['question_text'][:50]}...")

            # Test temporal analysis
            try:
                result = processor.analyze_temporal_coordinates(question_id)
                if question_id in result and 'error' not in result[question_id]:
                    confidence = result[question_id]['processing_metadata']['temporal_extraction_confidence']
                    print(f"  [SUCCESS] Analysis successful (confidence: {confidence:.3f})")
                    success_count += 1
                else:
                    print(f"  [FAILED] Analysis failed: {result.get(question_id, {}).get('error', 'Unknown')}")
            except Exception as e:
                print(f"  [ERROR] Analysis exception: {e}")
        else:
            print(f"  [FAILED] Failed to load question data")

    print(f"\n" + "=" * 40)
    print(f"Q2.4 FIX RESULTS: {success_count}/5 questions successful")

    if success_count >= 4:
        print("[SUCCESS] Q2.4 fix is working! Ready for full pipeline test.")
        return True
    else:
        print("[FAILURE] Q2.4 still has issues.")
        return False

if __name__ == "__main__":
    success = test_q24_fix()
    sys.exit(0 if success else 1)