#!/usr/bin/env python3
"""
Run B5 with OpenAI API key directly set
"""

import os
import sys
from pathlib import Path

# Set the OpenAI API key from Config.py directly
OPENAI_API_KEY = 'your-openai-api-key-here'

# Set environment variable
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

print(f"[SUCCESS] OpenAI API Key set: {OPENAI_API_KEY[:20]}...")

# Change to B5 script directory and run
scripts_dir = Path(__file__).parent / "B_Retrieval_pipeline" / "scripts"
os.chdir(str(scripts_dir))

# Import and run B5
sys.path.append(str(scripts_dir))

print("[INFO] Running B5 with OpenAI API integration...")
print("="*60)

# Run B5 main function
from B5_enhanced_answer_generation import main
main()