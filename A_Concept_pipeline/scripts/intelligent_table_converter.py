#!/usr/bin/env python3
"""
Intelligent Table Converter Module
A drop-in replacement for table-to-text conversion with semantic intelligence

This module can be imported by A2.1 to enhance table conversion without
modifying the core preprocessing logic.
"""

import re
import ast
from typing import List, Dict, Tuple, Optional, Any

def intelligent_convert_table_to_text(
    table_str: str,
    context: str = "",
    domain: str = "finance"
) -> str:
    """
    Drop-in replacement for convert_table_to_text with intelligent parsing

    This function intelligently handles:
    1. Section headers (Note 8) vs table titles (Net Income per Share)
    2. Denomination scaling (in thousands, millions, etc.)
    3. Currency detection and formatting
    4. Mixed units (per share exceptions)

    Args:
        table_str: String representation of nested list table
        context: Pre-table context for semantic analysis
        domain: Business domain for specialized handling

    Returns:
        str: Natural language representation of table
    """

    try:
        # Parse the table structure
        table_data = ast.literal_eval(table_str)

        # Extract metadata using intelligent parsing
        metadata = extract_table_metadata(table_data, context)

        # Convert to natural language with proper formatting
        return build_natural_language(table_data, metadata, domain)

    except Exception as e:
        print(f"[WARNING] Intelligent conversion failed, using basic conversion: {e}")
        # Fallback to basic conversion
        return basic_table_to_text(table_str)

def extract_table_metadata(
    table_data: List[List[str]],
    context: str
) -> Dict[str, Any]:
    """
    Extract semantic metadata from table and context

    Returns metadata dictionary with:
    - table_title: Actual content description
    - section_ref: Document section reference
    - currency: Currency symbol
    - scale_factor: Numeric multiplier
    - scale_description: Human readable scale
    - exceptions: List of exception patterns
    """

    metadata = {
        'table_title': '',
        'section_ref': '',
        'currency': '$',
        'scale_factor': 1,
        'scale_description': '',
        'exceptions': [],
        'confidence': 0.0
    }

    # Step 1: Separate section reference from table title
    section_pattern = r'(Note\s+\d+|Table\s+\d+|Section\s+[\dA-Z]+)'
    section_match = re.search(section_pattern, context, re.IGNORECASE)

    if section_match:
        metadata['section_ref'] = section_match.group(1)
        metadata['confidence'] += 0.2

    # Clean context by removing section reference
    clean_context = context
    if metadata['section_ref']:
        clean_context = re.sub(
            re.escape(metadata['section_ref']) + r'[:\.]?\s*',
            '',
            context,
            flags=re.IGNORECASE
        )

    # Step 2: Extract actual table title
    title_found = False

    # Method 1: Look for title after colon
    colon_match = re.search(r':\s*([A-Z][^.\n]{5,50})', clean_context)
    if colon_match and not title_found:
        potential_title = colon_match.group(1).strip()
        # Verify it contains semantic content
        if any(word in potential_title.lower() for word in
               ['revenue', 'income', 'balance', 'cash', 'asset', 'liability',
                'expense', 'cost', 'share', 'operation', 'statement']):
            metadata['table_title'] = potential_title
            title_found = True
            metadata['confidence'] += 0.3

    # Method 2: Look for descriptive patterns
    if not title_found:
        desc_patterns = [
            r'following table (?:presents?|shows?) ([^.]+)',
            r'table below (?:presents?|shows?) ([^.]+)',
            r'(?:presents?|shows?) ([^.]+) (?:in the )?(?:following )?table'
        ]

        for pattern in desc_patterns:
            match = re.search(pattern, clean_context, re.IGNORECASE)
            if match:
                metadata['table_title'] = match.group(1).strip()
                title_found = True
                metadata['confidence'] += 0.25
                break

    # Method 3: Extract from first semantic row
    if not title_found:
        for row in table_data[:4]:
            row_text = ' '.join(str(cell) for cell in row if cell)
            # Skip if it's a date or year row
            if re.match(r'^[\d\s,/\-]+$', row_text):
                continue
            # Check for semantic content
            if any(word in row_text.lower() for word in
                   ['revenue', 'income', 'balance', 'expense', 'asset']):
                # Clean section references
                cleaned_title = re.sub(section_pattern, '', row_text, flags=re.IGNORECASE)
                if cleaned_title.strip():
                    metadata['table_title'] = cleaned_title.strip()
                    title_found = True
                    metadata['confidence'] += 0.2
                    break

    # Step 3: Extract denomination and currency
    full_text = context + ' '.join(str(cell) for row in table_data[:4] for cell in row)

    # Currency detection
    currency_patterns = {
        r'\$|USD?': '$',
        r'£|GBP': '£',
        r'€|EUR': '€',
        r'¥|JPY': '¥'
    }

    for pattern, symbol in currency_patterns.items():
        if re.search(pattern, full_text):
            metadata['currency'] = symbol
            metadata['confidence'] += 0.1
            break

    # Scale detection with exception handling
    scale_patterns = [
        (r"in thousands(?:,?\s*except\s+([^).\n]+))?", 1000, "in thousands"),
        (r"in millions(?:,?\s*except\s+([^).\n]+))?", 1000000, "in millions"),
        (r"in billions(?:,?\s*except\s+([^).\n]+))?", 1000000000, "in billions"),
        (r"\(000'?s?\)(?:,?\s*except\s+([^).\n]+))?", 1000, "in thousands"),
        (r"000's(?:,?\s*except\s+([^).\n]+))?", 1000, "in thousands"),
        (r"\('?000'?s?\)(?:,?\s*except\s+([^).\n]+))?", 1000, "in thousands")
    ]

    for pattern, scale, description in scale_patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            metadata['scale_factor'] = scale
            metadata['scale_description'] = description
            metadata['confidence'] += 0.2

            # Extract exceptions if present
            if match.group(1):
                exception = match.group(1).strip()
                metadata['exceptions'].append(exception)
                metadata['confidence'] += 0.1
            break

    # Special case: Check for 'US$000' or similar patterns
    if re.search(r'US\$000|USD\s*000|\$000', full_text):
        metadata['scale_factor'] = 1000
        metadata['scale_description'] = "in thousands"
        metadata['confidence'] += 0.15

    return metadata

def build_natural_language(
    table_data: List[List[str]],
    metadata: Dict[str, Any],
    domain: str
) -> str:
    """
    Build natural language representation using metadata

    Creates clean, readable text without section header pollution
    and with proper value scaling
    """

    sentences = []

    # Add table title if found (without section reference)
    if metadata['table_title']:
        # Clean any remaining section references
        clean_title = re.sub(r'Note\s+\d+[:\.]?\s*', '', metadata['table_title'], flags=re.IGNORECASE)
        sentences.append(f"This table shows {clean_title}.")

    # Add denomination context
    if metadata['scale_factor'] > 1:
        scale_text = metadata['scale_description'] or f"in {metadata['scale_factor']:,}s"
        currency_name = get_currency_name(metadata['currency'])

        if metadata['exceptions']:
            exceptions_text = ', '.join(metadata['exceptions'])
            sentences.append(f"Values are {scale_text} of {currency_name}, except {exceptions_text}.")
        else:
            sentences.append(f"Values are {scale_text} of {currency_name}.")

    # Process data rows
    for i, row in enumerate(table_data):
        # Skip empty rows
        if not any(cell for cell in row):
            continue

        # Skip header rows (usually first 1-2 rows)
        if i < 2 and any(word in ' '.join(str(c) for c in row).lower()
                         for word in ['year', 'ended', 'june', 'december', 'march']):
            continue

        # Skip denomination rows
        row_text = ' '.join(str(cell) for cell in row if cell)
        if re.search(r'in thousands|in millions|\(000\)', row_text, re.IGNORECASE):
            continue

        # Process data row
        processed_row = process_data_row(row, metadata)
        if processed_row:
            sentences.append(processed_row)

    # Join sentences with proper spacing
    result = ' '.join(sentences)

    # Final cleanup - remove any repeated "Note X" patterns
    result = re.sub(r'(Note\s+\d+[:\.]?\s*)+', '', result)

    return result.strip()

def process_data_row(
    row: List[str],
    metadata: Dict[str, Any]
) -> Optional[str]:
    """
    Process a single data row with intelligent value formatting
    """

    if not row:
        return None

    processed_items = []
    row_label = None

    for j, cell in enumerate(row):
        if not cell:
            continue

        cell_str = str(cell).strip()

        # Skip section references
        if re.match(r'^Note\s+\d+', cell_str, re.IGNORECASE):
            continue

        # First non-empty cell is usually the row label
        if row_label is None:
            row_label = cell_str
            # Clean section references from label
            row_label = re.sub(r'Note\s+\d+[:\.]?\s*', '', row_label, flags=re.IGNORECASE)
            processed_items.append(row_label)
        else:
            # Process value cells
            formatted_value = format_value(
                cell_str,
                metadata,
                is_exception=check_if_exception(row_label, metadata)
            )
            if formatted_value:
                processed_items.append(formatted_value)

    # Only return if we have meaningful data
    if len(processed_items) > 1:
        return ' '.join(processed_items) + '.'

    return None

def format_value(
    value_str: str,
    metadata: Dict[str, Any],
    is_exception: bool = False
) -> Optional[str]:
    """
    Format a value with proper scaling and currency
    """

    # Remove existing currency symbols
    clean_value = re.sub(r'[\$£€¥,]', '', value_str)

    try:
        value = float(clean_value)

        # Apply scaling if not an exception
        if not is_exception and metadata['scale_factor'] > 1:
            actual_value = value * metadata['scale_factor']

            # Format based on magnitude
            if actual_value >= 1_000_000_000:
                return f"{metadata['currency']}{actual_value/1_000_000_000:.1f} billion"
            elif actual_value >= 1_000_000:
                return f"{metadata['currency']}{actual_value/1_000_000:.1f} million"
            elif actual_value >= 1_000:
                return f"{metadata['currency']}{actual_value/1_000:.0f}k"
            else:
                return f"{metadata['currency']}{actual_value:.0f}"
        else:
            # No scaling or is exception (like per-share data)
            if value < 100 and 'per' in value_str.lower():
                return f"{metadata['currency']}{value:.2f}"
            else:
                return f"{metadata['currency']}{value:,.0f}"

    except ValueError:
        # Non-numeric value - return as is if meaningful
        if len(value_str) > 1 and not value_str.isspace():
            return value_str

    return None

def check_if_exception(
    row_label: str,
    metadata: Dict[str, Any]
) -> bool:
    """
    Check if a row label indicates it should be treated as an exception
    to the normal scaling rules
    """

    if not row_label:
        return False

    row_lower = row_label.lower()

    # Check explicit exceptions
    for exception in metadata.get('exceptions', []):
        if exception.lower() in row_lower:
            return True

    # Check common exception patterns
    exception_patterns = ['per share', 'per unit', 'percentage', 'ratio', 'rate']
    return any(pattern in row_lower for pattern in exception_patterns)

def get_currency_name(symbol: str) -> str:
    """
    Get the full currency name from symbol
    """

    currency_names = {
        '$': 'US dollars',
        '£': 'British pounds',
        '€': 'euros',
        '¥': 'Japanese yen'
    }

    return currency_names.get(symbol, 'dollars')

def basic_table_to_text(table_str: str) -> str:
    """
    Basic fallback conversion without intelligent parsing
    """

    try:
        table_data = ast.literal_eval(table_str)
        sentences = []

        for row in table_data:
            if any(cell for cell in row):
                row_text = ' '.join(str(cell) for cell in row if cell)
                sentences.append(row_text + '.')

        return ' '.join(sentences)

    except:
        return table_str

# Test function for standalone testing
def test_intelligent_converter():
    """
    Test the intelligent converter with sample data
    """

    # Sample table from user's example
    sample_table = """[
        ['', '', 'YearEnded', ''],
        ['', 'June 30, 2019', 'June 24, 2018', 'June 25, 2017'],
        ['', '', '(in thousands, except per share data)', ''],
        ['Net income', '$2,191,430', '$2,380,681', '$1,697,763'],
        ['Basic average shares outstanding', '152,478', '161,643', '162,222'],
        ['Net income per share-basic', '$14.37', '$14.73', '$10.47']
    ]"""

    context = "Note 8: Net Income per Share. The following table presents the calculation of basic and diluted net income per share:"

    result = intelligent_convert_table_to_text(sample_table, context, "finance")

    print("Input Context:", context)
    print("\nIntelligent Conversion Result:")
    print(result)

    return result

if __name__ == "__main__":
    test_intelligent_converter()