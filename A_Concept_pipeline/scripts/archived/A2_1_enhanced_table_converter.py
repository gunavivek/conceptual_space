"""
Enhanced Table-to-Text Converter for A2.1
This module can be integrated into A2.1 WITHOUT disturbing existing logic
It provides enhanced context extraction and more meaningful natural language generation
"""

import re
from typing import Dict, List, Any, Tuple, Optional

def extract_enhanced_table_context(text: str, table_start: int, table_end: int) -> Dict[str, str]:
    """
    Extract comprehensive table context including title, note references, and post-table info

    This function looks for:
    1. Table title/header (up to 500 chars before table)
    2. Note references (e.g., "Note 3. Revenue")
    3. Section headers (e.g., "Consolidated Financial Statements")
    4. Post-table explanatory text (up to 200 chars after)

    Args:
        text: Full document text
        table_start: Starting position of table
        table_end: Ending position of table

    Returns:
        Dict containing:
        - 'title': Main table title
        - 'note_reference': Note number if applicable
        - 'section': Section header if found
        - 'post_context': Explanatory text after table
        - 'full_context': Combined meaningful context
    """
    context = {}

    # Extract pre-table text (expand search window)
    pre_context_window = 500
    pre_text = text[max(0, table_start - pre_context_window):table_start].strip()

    # Look for Note references (e.g., "Note 3. Revenue")
    note_pattern = r'Note\s+(\d+)[.\s]+([^\n]+?)(?:\n|$)'
    note_match = re.search(note_pattern, pre_text, re.IGNORECASE)
    if note_match:
        context['note_reference'] = f"Note {note_match.group(1)}"
        context['title'] = note_match.group(2).strip()

    # Look for section headers (often in all caps or with specific formatting)
    section_pattern = r'([A-Z][A-Z\s]{10,})|(\d+\.\s+[A-Z][^.\n]+)'
    section_matches = list(re.finditer(section_pattern, pre_text))
    if section_matches:
        last_section = section_matches[-1].group().strip()
        context['section'] = last_section

    # If no Note found, look for immediate title (last line before table)
    if 'title' not in context:
        lines = pre_text.split('\n')
        # Search backwards for non-empty line
        for line in reversed(lines):
            line = line.strip()
            if line and not line.isspace():
                # Remove common prefixes
                line = re.sub(r'^\d+[.)]\s*', '', line)  # Remove "1." or "1)"
                line = re.sub(r'^Table\s+\d+[.:]\s*', '', line, re.IGNORECASE)  # Remove "Table 1:"
                if line:
                    context['title'] = line
                    break

    # Extract post-table context (often contains explanatory notes)
    post_context_window = 200
    post_text = text[table_end:min(len(text), table_end + post_context_window)].strip()

    # Look for explanatory text after table
    if post_text:
        # Take first sentence or paragraph
        first_sentence = post_text.split('.')[0].strip()
        if first_sentence and len(first_sentence) > 10:
            context['post_context'] = first_sentence

    # Build full context string
    full_context_parts = []
    if 'note_reference' in context:
        full_context_parts.append(context['note_reference'])
    if 'title' in context:
        full_context_parts.append(context['title'])
    elif 'section' in context:
        full_context_parts.append(context['section'])

    context['full_context'] = ' - '.join(full_context_parts) if full_context_parts else ""

    return context

def generate_enhanced_table_text(table_info: Dict[str, Any], context: Dict[str, str]) -> str:
    """
    Generate meaningful natural language from table with proper financial intelligence

    This function creates rich descriptions that include:
    1. Proper header detection (skip units rows)
    2. Financial formatting intelligence (thousands, millions)
    3. Trend analysis with natural language flow
    4. Subtotal and total recognition

    Args:
        table_info: Parsed table structure with headers and data
        context: Enhanced context dictionary from extract_enhanced_table_context

    Returns:
        str: Rich natural language description of the table
    """
    headers = table_info.get('headers', [])
    data_rows = table_info.get('data_rows', [])

    if not data_rows:
        return ""

    # Enhanced header processing to properly identify units and years
    years = []
    units = ""
    scale_factor = ""

    # Look through ALL rows (including data_rows) for headers that might have been misclassified
    all_rows = headers + data_rows
    actual_data_start = 0

    for i, row in enumerate(all_rows):
        row_str = ' '.join(str(cell) for cell in row if cell)

        # Check if this is a units row (should be skipped as data)
        if re.search(r'US\$\d+|USD\s*\d+|thousands?|millions?', row_str, re.IGNORECASE):
            units = row_str
            if '000' in row_str:
                scale_factor = "thousands"
            elif 'million' in row_str.lower():
                scale_factor = "millions"
            actual_data_start = max(actual_data_start, i + 1)
            continue

        # Check for years
        years_in_row = [cell for cell in row if cell and re.match(r'^\d{4}$', str(cell))]
        if years_in_row:
            years.extend(years_in_row)
            actual_data_start = max(actual_data_start, i + 1)
            continue

        # Check for mostly empty header rows
        if sum(1 for cell in row if not cell or str(cell).strip() == "") > len(row) * 0.6:
            actual_data_start = max(actual_data_start, i + 1)
            continue

        # This is actual data
        break

    # Get actual data rows (skip misclassified headers)
    if actual_data_start < len(all_rows):
        actual_data_rows = all_rows[actual_data_start:]
    else:
        actual_data_rows = data_rows

    text_parts = []

    # Add introduction with context (WITHOUT repeating it for every row)
    intro_text = ""
    if context.get('full_context'):
        intro_text = f"The following presents {context['full_context']} data"
    elif context.get('title'):
        intro_text = f"The following presents {context['title']} data"

    if intro_text:
        text_parts.append(intro_text)

    # Process actual data rows with financial intelligence
    revenue_items = []
    totals = []

    for row_idx, row in enumerate(actual_data_rows):
        if not row or len(row) < 2:
            continue

        row_label = str(row[0]).strip() if row[0] else ""

        # Extract values
        values = []
        for i in range(1, len(row)):
            cell = str(row[i]).strip() if row[i] else ""
            if cell and cell != "":
                values.append(cell)

        if not values or len(values) < 2:
            continue

        # Classify row type
        if not row_label or row_label.strip() == "":
            # Empty label = subtotal
            totals.append((f"subtotal", values))
        elif row_label.lower() in ['total', 'revenue', 'income']:
            # Final totals
            totals.append((row_label, values))
        else:
            # Regular revenue item
            revenue_items.append((row_label, values))

    # Generate natural language for revenue items
    if revenue_items and len(years) >= 2:
        year1, year2 = years[0], years[1]

        for item_label, values in revenue_items:
            if len(values) >= 2:
                try:
                    val1 = float(values[0].replace(',', ''))  # First value corresponds to year1
                    val2 = float(values[1].replace(',', ''))  # Second value corresponds to year2

                    # Format values with scale
                    if scale_factor == "thousands":
                        val1_millions = val1 / 1000
                        val2_millions = val2 / 1000
                        formatted_val1 = f"${val1_millions:.1f} million" if val1_millions < 1000 else f"${val1_millions/1000:.1f} billion"
                        formatted_val2 = f"${val2_millions:.1f} million" if val2_millions < 1000 else f"${val2_millions/1000:.1f} billion"
                    else:
                        formatted_val1 = f"${val1:,.0f}"
                        formatted_val2 = f"${val2:,.0f}"

                    # Calculate trend (val2 - val1 because chronologically year2 > year1)
                    change = val2 - val1
                    pct_change = (change / val1 * 100) if val1 != 0 else 0

                    if abs(pct_change) > 5:  # Significant change
                        if change > 0:
                            trend_text = f"increased to {formatted_val2} in {year2} from {formatted_val1} in {year1}"
                        else:
                            trend_text = f"declined to {formatted_val2} in {year2} from {formatted_val1} in {year1}"
                    else:
                        trend_text = f"was {formatted_val2} in {year2} compared to {formatted_val1} in {year1}"

                    text_parts.append(f"{item_label} {trend_text}")

                except (ValueError, IndexError):
                    # Fallback for non-numeric values
                    text_parts.append(f"{item_label}: {values[0]} in {year1}, {values[1]} in {year2}")

    # Add totals summary
    if totals and len(years) >= 2:
        for total_label, values in totals:
            if len(values) >= 2:
                try:
                    val1 = float(values[0].replace(',', ''))
                    val2 = float(values[1].replace(',', ''))

                    if scale_factor == "thousands":
                        val1_millions = val1 / 1000
                        val2_millions = val2 / 1000
                        formatted_val1 = f"${val1_millions:.1f} million"
                        formatted_val2 = f"${val2_millions:.1f} million"
                    else:
                        formatted_val1 = f"${val1:,.0f}"
                        formatted_val2 = f"${val2:,.0f}"

                    change = val2 - val1
                    pct_change = (change / val1 * 100) if val1 != 0 else 0

                    if total_label == "subtotal":
                        text_parts.append(f"These revenue streams totaled {formatted_val2} in {years[1]}, compared to {formatted_val1} in {years[0]}")
                    else:
                        text_parts.append(f"Overall {total_label.lower()} was {formatted_val2} in {years[1]}, an increase from {formatted_val1} in {years[0]}" if change > 0 else f"Overall {total_label.lower()} was {formatted_val2} in {years[1]}, compared to {formatted_val1} in {years[0]}")

                except (ValueError, IndexError):
                    text_parts.append(f"{total_label}: {values[0]} in {years[0]}, {values[1]} in {years[1]}")

    # Combine with proper flow
    if text_parts:
        if len(text_parts) > 1:
            # Combine intro with narrative flow
            result = text_parts[0] + ": " + ". ".join(text_parts[1:]) + "."
        else:
            result = text_parts[0] + "."

        # Clean up formatting
        result = re.sub(r':\s*:', ':', result)  # Remove double colons
        result = re.sub(r'\.\s*\.', '.', result)  # Remove double periods
        result = re.sub(r'\s+', ' ', result)  # Normalize whitespace

        return result

    return ""

def enhanced_convert_table_to_text(table_info: Dict[str, Any], text: str,
                                  table_start: int, table_end: int,
                                  existing_context: str = "") -> str:
    """
    Enhanced wrapper function that can replace the existing convert_table_to_text
    while maintaining backward compatibility

    Args:
        table_info: Parsed table structure
        text: Full document text
        table_start: Table start position
        table_end: Table end position
        existing_context: Context from existing extraction (for compatibility)

    Returns:
        str: Enhanced natural language description
    """
    # Extract enhanced context
    enhanced_context = extract_enhanced_table_context(text, table_start, table_end)

    # If no enhanced context found, use existing context
    if not enhanced_context.get('full_context') and existing_context:
        enhanced_context['full_context'] = existing_context

    # Generate enhanced text
    return generate_enhanced_table_text(table_info, enhanced_context)

# Integration function that can be called from A2.1
def integrate_enhanced_converter(original_function):
    """
    Decorator to enhance existing convert_table_to_text function
    Can be applied without modifying core logic
    """
    def wrapper(table_info: Dict[str, Any], context: str = "", **kwargs):
        # Check if we have additional context available
        if 'full_text' in kwargs and 'table_position' in kwargs:
            # Use enhanced converter
            full_text = kwargs['full_text']
            start, end = kwargs['table_position']
            return enhanced_convert_table_to_text(table_info, full_text, start, end, context)
        else:
            # Fall back to original function
            return original_function(table_info, context)
    return wrapper

# Example usage notes for integration:
"""
To integrate into A2.1 WITHOUT modifying existing code:

1. Import this module at the top of A2.1:
   from A2_1_enhanced_table_converter import enhanced_convert_table_to_text

2. In the convert_tables_to_text function, modify the call (around line 505):

   CURRENT:
   table_text = convert_table_to_text(parsed_table, pre_table_text)

   ENHANCED:
   table_text = enhanced_convert_table_to_text(
       parsed_table,
       text,
       table_pattern['start'],
       table_pattern['end'],
       pre_table_text
   )

This provides richer context extraction and more meaningful natural language
while maintaining full backward compatibility.
"""