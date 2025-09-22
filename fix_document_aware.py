#!/usr/bin/env python3
"""
Fix document-aware concept filtering in all chunking strategies
"""

import os
import re

def fix_strategy_file(filepath, strategy_name):
    """Fix a single strategy file to use document-aware filtering"""

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern to find extract_concept_memberships calls without doc_id
    # Skip the ones in base_strategy.py and ones that already have doc_id

    if strategy_name == "paragraph_aware.py":
        # Special case for paragraph_aware which has comparison calls
        content = re.sub(
            r'memberships1, scores1 = self\.extract_concept_memberships\(para1, concepts\)',
            r'# Note: For similarity calculation, we use generic matching\n        memberships1, scores1 = self.extract_concept_memberships(para1, concepts)',
            content
        )
        content = re.sub(
            r'memberships2, scores2 = self\.extract_concept_memberships\(para2, concepts\)',
            r'memberships2, scores2 = self.extract_concept_memberships(para2, concepts)',
            content
        )
        # Fix the main one in merge
        content = re.sub(
            r'memberships, scores = self\.extract_concept_memberships\(merged_text, concepts\)',
            r'# DOCUMENT-AWARE: Only match concepts from same document\n            memberships, scores = self.extract_concept_memberships(merged_text, concepts, doc_id=doc_id)',
            content
        )

    elif strategy_name == "contextual_overlap.py":
        # Fix overlap calculations
        content = re.sub(
            r'prev_concepts, prev_scores = self\.extract_concept_memberships\(overlap_text_prev, concepts\)',
            r'# Note: For overlap, we keep generic matching\n        prev_concepts, prev_scores = self.extract_concept_memberships(overlap_text_prev, concepts)',
            content
        )
        content = re.sub(
            r'curr_concepts, curr_scores = self\.extract_concept_memberships\(overlap_text_curr, concepts\)',
            r'curr_concepts, curr_scores = self.extract_concept_memberships(overlap_text_curr, concepts)',
            content
        )
        # Fix main assignment
        content = re.sub(
            r'memberships, scores = self\.extract_concept_memberships\(segment_text, concepts\)(?!\s*,\s*doc_id)',
            r'# DOCUMENT-AWARE: Only match concepts from same document\n            memberships, scores = self.extract_concept_memberships(segment_text, concepts, doc_id=doc_id)',
            content
        )

    elif strategy_name == "document_structure.py":
        content = re.sub(
            r'memberships, scores = self\.extract_concept_memberships\(section_text, concepts\)',
            r'# DOCUMENT-AWARE: Only match concepts from same document\n            memberships, scores = self.extract_concept_memberships(section_text, concepts, doc_id=doc_id)',
            content
        )

    elif strategy_name == "concept_aware.py":
        content = re.sub(
            r'memberships, scores = self\.extract_concept_memberships\(region_text, concepts\)',
            r'# DOCUMENT-AWARE: Only match concepts from same document\n            memberships, scores = self.extract_concept_memberships(region_text, concepts, doc_id=doc_id)',
            content
        )

    elif strategy_name == "quality_based.py":
        # First occurrence for quality calculation
        content = re.sub(
            r'memberships, scores = self\.extract_concept_memberships\(text, concepts, threshold=0\.2\)',
            r'# Note: For quality calculation, we use generic matching\n        memberships, scores = self.extract_concept_memberships(text, concepts, threshold=0.2)',
            content
        )
        # Main assignment
        content = re.sub(
            r'(\s+)memberships, scores = self\.extract_concept_memberships\(chunk_text, concepts\)(?!\s*,\s*doc_id)',
            r'\1# DOCUMENT-AWARE: Only match concepts from same document\n\1memberships, scores = self.extract_concept_memberships(chunk_text, concepts, doc_id=doc_id)',
            content
        )

    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Fixed {strategy_name}")

# Fix all strategy files
strategies_dir = r"C:\AiSearch\conceptual_space\A_Concept_pipeline\scripts\chunking_strategies"

files_to_fix = [
    "paragraph_aware.py",
    "contextual_overlap.py",
    "document_structure.py",
    "concept_aware.py",
    "quality_based.py"
]

for filename in files_to_fix:
    filepath = os.path.join(strategies_dir, filename)
    if os.path.exists(filepath):
        fix_strategy_file(filepath, filename)

print("\nAll strategy files updated with document-aware filtering!")