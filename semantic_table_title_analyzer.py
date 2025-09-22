"""
Semantic Table Title Detection - Human-like Intelligence

This module provides semantic analysis to differentiate between:
1. Section headers (document structure): "Note 8", "16. INCOME TAXES"
2. Table titles (content description): "Net Income per Share", "Revenue by geographic area"
3. Context descriptions: "The following table presents..."

Key Principle: Table titles describe WHAT the data represents, not WHERE it appears in the document.
"""

import re
from typing import Dict, List, Tuple, Optional

def analyze_table_context_semantically(text: str, table_start: int) -> Dict[str, str]:
    """
    Semantic analysis to extract meaningful table title without section noise

    Args:
        text: Full document text
        table_start: Starting position of table

    Returns:
        Dict with semantic components:
        - 'pure_table_title': Actual table content description
        - 'section_reference': Document structure reference
        - 'semantic_context': Human-readable table description
    """

    # Extract wider context for semantic analysis
    context_window = 300
    pre_text = text[max(0, table_start - context_window):table_start].strip()

    # Semantic patterns for different content types
    patterns = {
        'section_headers': [
            r'Note\s+\d+[:\.]?',           # "Note 8:", "Note 16."
            r'\d+\.\s+[A-Z][A-Z\s]+',      # "16. INCOME TAXES"
            r'[A-Z]{2,}\s+[A-Z]{2,}',      # "CONSOLIDATED STATEMENTS"
            r'Table\s+\d+[:\.]?',          # "Table 1:"
        ],
        'table_descriptors': [
            r'following\s+table[s]?\s+(?:presents?|shows?|summarizes?)\s+([^.]+)',
            r'table[s]?\s+(?:below|above)\s+(?:presents?|shows?|summarizes?)\s+([^.]+)',
            r'(?:presents?|shows?|summarizes?)\s+([^.]+)\s+(?:in the )?(?:following )?table',
        ],
        'content_titles': [
            # Things that describe actual data content
            r'(?:by\s+geographic\s+area|per\s+share|and\s+subsidiaries)',
            r'(?:revenue|income|expense|cost|balance|cash|assets|liabilities)',
            r'(?:years?\s+ended|fiscal\s+year|period)',
        ]
    }

    # Step 1: Identify and extract section headers
    section_refs = []
    cleaned_text = pre_text

    for pattern in patterns['section_headers']:
        matches = list(re.finditer(pattern, cleaned_text, re.IGNORECASE))
        for match in matches:
            section_refs.append(match.group().strip())
            # Remove section headers from text for cleaner title extraction
            cleaned_text = cleaned_text.replace(match.group(), ' ', 1)

    # Step 2: Look for explicit table descriptors
    table_descriptor = None
    for pattern in patterns['table_descriptors']:
        match = re.search(pattern, cleaned_text, re.IGNORECASE)
        if match:
            table_descriptor = match.group(1).strip()
            break

    # Step 3: Extract pure content title (closest to table, not section header)
    lines = cleaned_text.split('\n')
    content_title = None

    # Search backwards for meaningful content title
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue

        # Skip if it looks like a section header we missed
        if re.match(r'^(Note|Table|Section|\d+\.)', line, re.IGNORECASE):
            continue

        # Check if it contains content-related terms
        has_content_terms = any(
            re.search(pattern, line, re.IGNORECASE)
            for pattern in patterns['content_titles']
        )

        # If it has content terms OR is the immediate line before table
        if has_content_terms or line == lines[-1].strip():
            content_title = line
            break

    # Step 4: Semantic title extraction from compound phrases
    pure_title = None
    if content_title:
        # Handle patterns like "Note 8: Net Income per Share"
        colon_split = content_title.split(':')
        if len(colon_split) == 2:
            section_part = colon_split[0].strip()
            title_part = colon_split[1].strip()

            # If first part looks like section ref, use second part
            if re.match(r'^(Note|Table|Section|\d+)', section_part, re.IGNORECASE):
                pure_title = title_part
                if section_part not in section_refs:
                    section_refs.append(section_part)
            else:
                pure_title = content_title
        else:
            pure_title = content_title

    # Use table descriptor if no pure title found
    if not pure_title and table_descriptor:
        pure_title = table_descriptor

    # Final cleanup
    if pure_title:
        pure_title = re.sub(r'^\W+|\W+$', '', pure_title)  # Remove leading/trailing punctuation
        pure_title = re.sub(r'\s+', ' ', pure_title)       # Normalize whitespace

    return {
        'pure_table_title': pure_title or '',
        'section_reference': ', '.join(section_refs) if section_refs else '',
        'semantic_context': pure_title or '',
        'table_descriptor': table_descriptor or '',
        'confidence': calculate_title_confidence(pure_title, section_refs, table_descriptor)
    }

def calculate_title_confidence(title: Optional[str], sections: List[str], descriptor: Optional[str]) -> float:
    """Calculate confidence score for extracted title"""
    confidence = 0.0

    if title:
        confidence += 0.4

        # Higher confidence for content-rich titles
        content_indicators = ['income', 'revenue', 'balance', 'cash', 'share', 'expense', 'cost']
        if any(indicator in title.lower() for indicator in content_indicators):
            confidence += 0.3

        # Higher confidence if we successfully separated from section
        if sections:
            confidence += 0.2

    if descriptor:
        confidence += 0.1

    return min(confidence, 1.0)

def demonstrate_semantic_analysis():
    """Demonstrate the semantic analysis on sample texts"""

    test_cases = [
        'Note 8: Net Income per Share [["", "", "YearEnded", ""]',
        '16. INCOME TAXES (Continued) The following table presents federal tax rates [["Federal statutory',
        'Revenue by geographic area are as follows (in thousands): [["", "", "Year Ended',
        'The following table summarizes information regarding shares of common stock granted and vested [["", ""',
        'CONSOLIDATED STATEMENTS OF OPERATIONS Revenue by geographic area [["United States"'
    ]

    print("Semantic Table Title Analysis Demonstration:")
    print("=" * 60)

    for i, text in enumerate(test_cases, 1):
        result = analyze_table_context_semantically(text, len(text) - 10)
        print(f"\nTest Case {i}:")
        print(f"Input: {text[:60]}...")
        print(f"Pure Table Title: '{result['pure_table_title']}'")
        print(f"Section Reference: '{result['section_reference']}'")
        print(f"Confidence: {result['confidence']:.2f}")

if __name__ == "__main__":
    demonstrate_semantic_analysis()