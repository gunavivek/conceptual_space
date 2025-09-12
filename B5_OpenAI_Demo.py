#!/usr/bin/env python3
"""
Demo script showing how to use B5 with OpenAI API
Set your OPENAI_API_KEY environment variable and run this script
"""

import os
import sys
from pathlib import Path

# Add the scripts directory to Python path
sys.path.append(str(Path(__file__).parent / "B_Retrieval_pipeline" / "scripts"))

# Example of setting API key programmatically (replace with your actual key)
# os.environ["OPENAI_API_KEY"] = "sk-your-openai-api-key-here"

def demo_openai_integration():
    """Demonstrate OpenAI API integration with B5"""
    
    print("="*70)
    print("B5 ENHANCED ANSWER GENERATION WITH OPENAI API DEMO")
    print("="*70)
    
    # Check if API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n[INFO] OPENAI_API_KEY not found!")
        print("\nTo test OpenAI integration:")
        print("1. Get an OpenAI API key from https://platform.openai.com/api-keys")
        print("2. Set the environment variable:")
        print("   Windows: set OPENAI_API_KEY=sk-your-key-here")
        print("   Linux/Mac: export OPENAI_API_KEY=sk-your-key-here")
        print("3. Run this demo again")
        print("\n[SUCCESS] For now, B5 will use rule-based fallback (which is working perfectly!)")
        return
    
    print(f"[SUCCESS] OpenAI API Key found: {api_key[:20]}...")
    print("\n[INFO] B5 will use OpenAI API to generate enhanced answers!")
    
    # Import and run B5
    try:
        from B5_enhanced_answer_generation import main
        main()
    except ImportError as e:
        print(f"[ERROR] Error importing B5: {e}")
    except Exception as e:
        print(f"[ERROR] Error running B5: {e}")

if __name__ == "__main__":
    demo_openai_integration()