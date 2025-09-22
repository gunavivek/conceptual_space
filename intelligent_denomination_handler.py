"""
Intelligent Denomination Handler for Financial Tables

Handles complex denomination scenarios:
1. Embedded currency symbols in data ($2,191,430)
2. Scale factors in separate rows (in thousands)
3. Mixed units (thousands vs per share)
4. Smart value conversion and formatting
"""

import re
from typing import Dict, List, Tuple, Optional

def parse_denomination_context(table_data: List[List[str]]) -> Dict[str, any]:
    """
    Intelligently parse denomination information from table structure

    Args:
        table_data: Nested list table structure

    Returns:
        Dict with denomination intelligence:
        - 'base_currency': Primary currency symbol
        - 'scale_factor': Multiplier (thousands, millions, etc.)
        - 'mixed_units': Different units for different rows
        - 'unit_exceptions': Special cases (per share, etc.)
    """

    denom_info = {
        'base_currency': '',
        'scale_factor': 1,
        'scale_description': '',
        'mixed_units': False,
        'unit_exceptions': [],
        'smart_formatting': True
    }

    # Scan all rows for denomination patterns
    for row in table_data:
        for cell in row:
            if not cell:
                continue

            cell_str = str(cell).strip()

            # Look for scale factor descriptions
            scale_patterns = {
                r'in thousands?(?:,?\s*except\s+([^)]+))?': {'scale': 1000, 'exceptions': []},
                r'in millions?(?:,?\s*except\s+([^)]+))?': {'scale': 1000000, 'exceptions': []},
                r'in billions?(?:,?\s*except\s+([^)]+))?': {'scale': 1000000000, 'exceptions': []},
                r'\(thousands?\)': {'scale': 1000, 'exceptions': []},
                r'\(millions?\)': {'scale': 1000000, 'exceptions': []},
            }

            for pattern, info in scale_patterns.items():
                match = re.search(pattern, cell_str, re.IGNORECASE)
                if match:
                    denom_info['scale_factor'] = info['scale']
                    denom_info['scale_description'] = cell_str

                    # Check for exceptions like "except per share data"
                    if match.group(1):
                        exception_text = match.group(1).strip()
                        denom_info['unit_exceptions'].append(exception_text)
                        denom_info['mixed_units'] = True

                    break

    # Detect base currency from data values
    currency_found = set()
    for row in table_data:
        for cell in row:
            if cell and re.search(r'[\$£€¥]', str(cell)):
                currency_symbols = re.findall(r'[\$£€¥]', str(cell))
                currency_found.update(currency_symbols)

    if currency_found:
        denom_info['base_currency'] = list(currency_found)[0]  # Use most common

    return denom_info

def format_financial_value(value: str, denom_info: Dict, row_label: str = '') -> str:
    """
    Intelligently format financial values based on denomination context

    Args:
        value: Raw value string (e.g., "$2,191,430")
        denom_info: Denomination context from parse_denomination_context
        row_label: Row context for unit exception handling

    Returns:
        str: Intelligently formatted value
    """

    if not value or not str(value).strip():
        return value

    # Clean the value
    clean_value = str(value).strip()

    # Extract numeric part
    numeric_match = re.search(r'[\d,.-]+', clean_value)
    if not numeric_match:
        return value

    numeric_str = numeric_match.group().replace(',', '')

    try:
        numeric_val = float(numeric_str)
    except ValueError:
        return value

    # Check if this row should be treated differently (e.g., per share data)
    is_exception = False
    if denom_info.get('unit_exceptions'):
        for exception in denom_info['unit_exceptions']:
            if 'per share' in exception.lower() and 'per share' in row_label.lower():
                is_exception = True
                break

    # Format based on context
    currency = denom_info.get('base_currency', '$')

    if is_exception:
        # Don't scale per-share values
        return f"{currency}{numeric_val:,.2f}"

    # Apply scale factor for monetary values
    scale = denom_info.get('scale_factor', 1)

    if scale > 1:
        # Convert to more readable format
        actual_value = numeric_val * scale

        if actual_value >= 1_000_000_000:
            return f"{currency}{actual_value/1_000_000_000:.2f} billion"
        elif actual_value >= 1_000_000:
            return f"{currency}{actual_value/1_000_000:.1f} million"
        elif actual_value >= 1_000:
            return f"{currency}{actual_value/1_000:.0f} thousand"
        else:
            return f"{currency}{actual_value:,.0f}"
    else:
        return f"{currency}{numeric_val:,.2f}"

def demonstrate_intelligent_denomination():
    """Demonstrate intelligent denomination handling"""

    # Your example table
    sample_table = [
        ['', '', 'YearEnded', ''],
        ['', 'June 30, 2019', 'June 24, 2018', 'June 25, 2017'],
        ['', '', '(in thousands, except per share data)', ''],
        ['Numerator:', '', '', ''],
        ['Net income', '$2,191,430', '$2,380,681', '$1,697,763'],
        ['Denominator:', '', '', ''],
        ['Basic average shares outstanding', '152,478', '161,643', '162,222'],
        ['Net income per share-basic', '$14.37', '$14.73', '$10.47'],
        ['Net income per share-diluted', '$13.70', '$13.17', '$9.24']
    ]

    print("Intelligent Denomination Analysis:")
    print("=" * 50)

    # Parse denomination context
    denom_info = parse_denomination_context(sample_table)
    print(f"Detected currency: {denom_info['base_currency']}")
    print(f"Scale factor: {denom_info['scale_factor']:,}")
    print(f"Scale description: {denom_info['scale_description']}")
    print(f"Mixed units: {denom_info['mixed_units']}")
    print(f"Unit exceptions: {denom_info['unit_exceptions']}")

    print("\nSmart Value Formatting:")
    print("-" * 30)

    # Test different row types
    test_cases = [
        ('Net income', '$2,191,430'),
        ('Basic average shares outstanding', '152,478'),
        ('Net income per share-basic', '$14.37'),
        ('Net income per share-diluted', '$13.70')
    ]

    for row_label, value in test_cases:
        formatted = format_financial_value(value, denom_info, row_label)
        print(f"{row_label}: {value} -> {formatted}")

if __name__ == "__main__":
    demonstrate_intelligent_denomination()