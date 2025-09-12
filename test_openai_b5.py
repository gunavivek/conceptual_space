#!/usr/bin/env python3
"""
Direct test of B5 with OpenAI API using Config.py key
"""

import sys
import os
from pathlib import Path

# Add the main directory to path
sys.path.append(str(Path(__file__).parent))

# Import the API key directly from Config.py
from Config import OPENAI_API_KEY

print(f"API Key found in Config.py: {OPENAI_API_KEY[:20]}..." if OPENAI_API_KEY.startswith('sk-') else "Invalid API key format")

# Set environment variable for this session
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Now import and run B5
sys.path.append(str(Path(__file__).parent / "B_Retrieval_pipeline" / "scripts"))

try:
    from B5_enhanced_answer_generation import EnhancedB5AnswerGenerator
    
    print("\n" + "="*60)
    print("TESTING B5 WITH OPENAI API FROM CONFIG.PY")
    print("="*60)
    
    # Create generator instance
    generator = EnhancedB5AnswerGenerator()
    
    # Test a single question
    print("\nTesting with a sample question...")
    
    # Create mock data for testing
    question_data = {
        "question_id": "test_1",
        "question": "What is the percentage change in the revenue from 2018 to 2019?"
    }
    
    b4_ranking = {
        "ranked_chunks": [
            {
                "chunk_id": "finqa_test_1630_paragraph_aware_0",
                "content": 'Note 3. Revenue [["", "Consolidated", ""], ["", "2019", "2018"], ["", "US$000", "US$000"], ["Software license revenue", "82,575", "64,420"], ["Subscription and maintenance revenue", "64,955", "56,996"], ["Search advertising revenue", "17,940", "11,968"], ["Service revenue", "3,655", "5,532"], ["Other revenue", "2,694", "1,260"], ["", "171,819", "140,176"], ["Interest income", "933", "192"], ["Revenue", "172,752", "140,368"]]',
                "combined_score": 0.89
            },
            {
                "chunk_id": "finqa_test_1630_semantic_sentence_0",
                "content": "Revenue [table with 2018=$140,368 and 2019=$172,752] for financial analysis",
                "combined_score": 0.75
            }
        ]
    }
    
    # Generate answer
    result = generator.generate_answer(question_data, b4_ranking)
    
    print("\nResult:")
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Generation Method: {result.get('generation_method', 'unknown')}")
    
    if 'api_usage' in result:
        print(f"API Usage: {result['api_usage']}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()