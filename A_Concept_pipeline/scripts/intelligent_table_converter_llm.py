#!/usr/bin/env python3
"""
Intelligent Table Converter with LLM Support
Combines rule-based heuristics with optional LLM analysis for complex cases

This production-ready solution uses:
1. Fast rule-based extraction for simple cases
2. LLM fallback for complex semantic analysis
3. Caching to minimize API calls
4. Graceful degradation when LLM unavailable
"""

import re
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import hashlib

# Optional LLM imports
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

class ComplexityLevel(Enum):
    """Determine when to use LLM vs rules"""
    SIMPLE = 1  # Clear patterns, use rules
    MODERATE = 2  # Mixed patterns, try rules first
    COMPLEX = 3  # Ambiguous, use LLM

@dataclass
class TableMetadata:
    """Structured table metadata"""
    table_title: str
    section_reference: str
    currency: str
    scale_factor: int
    scale_description: str
    unit_exceptions: List[str]
    confidence: float
    extraction_method: str

class IntelligentTableConverter:
    """
    Production-ready table converter with LLM support
    """

    def __init__(self, llm_provider: str = "openai", api_key: Optional[str] = None):
        self.llm_provider = llm_provider
        self.api_key = api_key
        self.cache = {}  # Cache LLM results

        # Initialize LLM client if available
        if llm_provider == "openai" and OPENAI_AVAILABLE and api_key:
            openai.api_key = api_key
            self.llm_available = True
        elif llm_provider == "anthropic" and ANTHROPIC_AVAILABLE and api_key:
            self.claude = anthropic.Anthropic(api_key=api_key)
            self.llm_available = True
        else:
            self.llm_available = False
            print(f"Warning: LLM provider {llm_provider} not available. Using rule-based extraction only.")

    def analyze_table(self,
                     table_data: List[List[str]],
                     context: str = "",
                     force_llm: bool = False) -> TableMetadata:
        """
        Intelligently analyze table with hybrid approach

        Args:
            table_data: Nested list table structure
            context: Surrounding text context
            force_llm: Force LLM analysis even for simple cases

        Returns:
            TableMetadata with extracted information
        """

        # Check cache first
        cache_key = self._get_cache_key(table_data, context)
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Determine complexity
        complexity = self._assess_complexity(table_data, context)

        # Try rule-based extraction first (fast)
        if not force_llm and complexity != ComplexityLevel.COMPLEX:
            metadata = self._rule_based_extraction(table_data, context)
            if metadata.confidence > 0.7:  # High confidence threshold
                self.cache[cache_key] = metadata
                return metadata

        # Use LLM for complex cases or low confidence
        if self.llm_available and (force_llm or complexity == ComplexityLevel.COMPLEX):
            metadata = self._llm_extraction(table_data, context)
            self.cache[cache_key] = metadata
            return metadata

        # Fallback to enhanced rules
        metadata = self._enhanced_rule_extraction(table_data, context)
        self.cache[cache_key] = metadata
        return metadata

    def _assess_complexity(self, table_data: List[List[str]], context: str) -> ComplexityLevel:
        """Assess table complexity to determine extraction method"""

        indicators = {
            'has_nested_headers': False,
            'mixed_denominations': False,
            'multiple_sections': False,
            'ambiguous_title': False
        }

        # Check for nested headers
        if len(table_data) > 2:
            first_rows = str(table_data[:3])
            if first_rows.count('Note') > 2 or first_rows.count('Table') > 2:
                indicators['has_nested_headers'] = True

        # Check for mixed denominations
        text_sample = str(table_data)
        denom_patterns = ['thousands', 'millions', 'per share', 'except']
        if sum(1 for p in denom_patterns if p in text_sample.lower()) >= 2:
            indicators['mixed_denominations'] = True

        # Check for multiple section references
        if context.count('Note') > 1 or context.count('Table') > 1:
            indicators['multiple_sections'] = True

        # Check for ambiguous title
        if ':' in context and '.' in context and len(context.split('\n')) > 2:
            indicators['ambiguous_title'] = True

        # Determine complexity level
        true_count = sum(indicators.values())
        if true_count >= 3:
            return ComplexityLevel.COMPLEX
        elif true_count >= 1:
            return ComplexityLevel.MODERATE
        else:
            return ComplexityLevel.SIMPLE

    def _rule_based_extraction(self, table_data: List[List[str]], context: str) -> TableMetadata:
        """Fast rule-based extraction for simple cases"""

        # Extract section reference
        section_ref = ""
        section_pattern = r'(Note\s+\d+|Table\s+\d+|Section\s+[\dA-Z]+)'
        section_match = re.search(section_pattern, context, re.IGNORECASE)
        if section_match:
            section_ref = section_match.group(1)

        # Extract table title (after removing section)
        title = ""
        clean_context = re.sub(section_pattern, '', context, flags=re.IGNORECASE)

        # Look for title patterns
        title_patterns = [
            r'([A-Z][A-Za-z\s]+(?:Revenue|Income|Balance|Cash|Assets|Liabilities)[\w\s]*)',
            r':[\s]*([^.\n]+)',  # After colon
            r'presents[\s]+([^.\n]+)',  # After "presents"
        ]

        for pattern in title_patterns:
            match = re.search(pattern, clean_context)
            if match:
                title = match.group(1).strip()
                break

        # Extract currency
        currency = "$"  # Default
        currency_symbols = re.findall(r'[\$£€¥]', str(table_data))
        if currency_symbols:
            currency = currency_symbols[0]

        # Extract scale factor
        scale_factor = 1
        scale_desc = ""
        scale_patterns = {
            r'in thousands': (1000, 'in thousands'),
            r'in millions': (1000000, 'in millions'),
            r'\(000\)': (1000, 'in thousands'),
            r"000's": (1000, "in thousands"),
        }

        full_text = context + ' ' + str(table_data[:3])
        for pattern, (scale, desc) in scale_patterns.items():
            if re.search(pattern, full_text, re.IGNORECASE):
                scale_factor = scale
                scale_desc = desc
                break

        # Check for exceptions
        exceptions = []
        if 'except per share' in full_text.lower():
            exceptions.append('per share data')

        # Calculate confidence
        confidence = 0.0
        if title: confidence += 0.4
        if section_ref: confidence += 0.2
        if scale_factor > 1: confidence += 0.2
        if currency != "$": confidence += 0.1
        if exceptions: confidence += 0.1

        return TableMetadata(
            table_title=title,
            section_reference=section_ref,
            currency=currency,
            scale_factor=scale_factor,
            scale_description=scale_desc,
            unit_exceptions=exceptions,
            confidence=min(confidence, 1.0),
            extraction_method="rule_based"
        )

    def _enhanced_rule_extraction(self, table_data: List[List[str]], context: str) -> TableMetadata:
        """Enhanced rule extraction with semantic patterns"""

        # Start with basic rule extraction
        base_metadata = self._rule_based_extraction(table_data, context)

        # Enhance with semantic analysis
        # Remove section headers from potential titles
        if base_metadata.table_title and base_metadata.section_reference:
            if base_metadata.section_reference in base_metadata.table_title:
                base_metadata.table_title = base_metadata.table_title.replace(
                    base_metadata.section_reference, ''
                ).strip(': ')

        # Look for semantic title indicators
        semantic_indicators = [
            'revenue', 'income', 'expense', 'balance', 'cash',
            'assets', 'liabilities', 'equity', 'operations', 'comprehensive'
        ]

        if not base_metadata.table_title:
            # Try to find title from first non-empty, non-numeric row
            for row in table_data[:5]:
                row_text = ' '.join(str(cell) for cell in row if cell).lower()
                if any(indicator in row_text for indicator in semantic_indicators):
                    # Clean and extract
                    potential_title = row_text
                    for pattern in [r'note \d+', r'table \d+', r'^\d+\.']:
                        potential_title = re.sub(pattern, '', potential_title, flags=re.IGNORECASE)
                    base_metadata.table_title = potential_title.strip().title()
                    base_metadata.confidence += 0.2
                    break

        # Validate denomination context
        if base_metadata.scale_factor > 1:
            # Check if denomination applies to all values
            value_pattern = r'\d+[,\d]*\.?\d*'
            values = re.findall(value_pattern, str(table_data))

            # If we have per-share values (typically < 100), note exception
            small_values = [float(v.replace(',', '')) for v in values[:10] if float(v.replace(',', '')) < 100]
            if len(small_values) > len(values[:10]) * 0.3:  # 30% are small values
                base_metadata.unit_exceptions.append('per share values detected')

        base_metadata.extraction_method = "enhanced_rules"
        return base_metadata

    def _llm_extraction(self, table_data: List[List[str]], context: str) -> TableMetadata:
        """Use LLM for complex semantic extraction"""

        if self.llm_provider == "openai":
            return self._openai_extraction(table_data, context)
        elif self.llm_provider == "anthropic":
            return self._claude_extraction(table_data, context)
        else:
            # Fallback to enhanced rules if LLM unavailable
            return self._enhanced_rule_extraction(table_data, context)

    def _openai_extraction(self, table_data: List[List[str]], context: str) -> TableMetadata:
        """Use OpenAI GPT for extraction"""

        if not OPENAI_AVAILABLE:
            return self._enhanced_rule_extraction(table_data, context)

        try:
            # Prepare concise prompt
            table_preview = json.dumps(table_data[:5])

            prompt = f"""Analyze this financial table and extract metadata.

Context before table: {context[:500]}
Table preview: {table_preview}

Extract and return as JSON:
1. table_title: The actual content description (not section header like "Note 8")
2. section_reference: Document section (e.g., "Note 8", "Table 2")
3. currency: Currency symbol
4. scale_factor: Number (1000 for thousands, 1000000 for millions, 1 for no scaling)
5. scale_description: Human readable (e.g., "in thousands")
6. unit_exceptions: List of exceptions (e.g., ["per share data"])

Be precise and extract only what's clearly indicated."""

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Use 3.5 for cost efficiency
                messages=[
                    {"role": "system", "content": "You are a financial document analyst. Extract table metadata precisely."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )

            # Parse response
            result_text = response.choices[0].message.content

            # Try to parse as JSON
            import json
            result = json.loads(result_text)

            return TableMetadata(
                table_title=result.get('table_title', ''),
                section_reference=result.get('section_reference', ''),
                currency=result.get('currency', '$'),
                scale_factor=result.get('scale_factor', 1),
                scale_description=result.get('scale_description', ''),
                unit_exceptions=result.get('unit_exceptions', []),
                confidence=0.9,  # High confidence for LLM
                extraction_method="openai_gpt"
            )

        except Exception as e:
            print(f"OpenAI extraction failed: {e}")
            return self._enhanced_rule_extraction(table_data, context)

    def _claude_extraction(self, table_data: List[List[str]], context: str) -> TableMetadata:
        """Use Anthropic Claude for extraction"""

        if not ANTHROPIC_AVAILABLE:
            return self._enhanced_rule_extraction(table_data, context)

        try:
            table_preview = json.dumps(table_data[:5])

            prompt = f"""Analyze this financial table to extract metadata.

Context: {context[:500]}
Table: {table_preview}

Return a JSON object with these exact keys:
- table_title: actual table content description (NOT section headers like "Note 8")
- section_reference: document section reference (e.g., "Note 8")
- currency: currency symbol
- scale_factor: 1000 for thousands, 1000000 for millions, or 1
- scale_description: human readable like "in thousands"
- unit_exceptions: array of exceptions like ["per share data"]

Focus on semantic understanding to differentiate section structure from content meaning."""

            response = self.claude.messages.create(
                model="claude-3-haiku-20240307",  # Use Haiku for cost efficiency
                max_tokens=200,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )

            result = json.loads(response.content[0].text)

            return TableMetadata(
                table_title=result.get('table_title', ''),
                section_reference=result.get('section_reference', ''),
                currency=result.get('currency', '$'),
                scale_factor=result.get('scale_factor', 1),
                scale_description=result.get('scale_description', ''),
                unit_exceptions=result.get('unit_exceptions', []),
                confidence=0.95,  # Highest confidence for Claude
                extraction_method="anthropic_claude"
            )

        except Exception as e:
            print(f"Claude extraction failed: {e}")
            return self._enhanced_rule_extraction(table_data, context)

    def _get_cache_key(self, table_data: List[List[str]], context: str) -> str:
        """Generate cache key for table + context"""
        content = json.dumps(table_data[:5]) + context[:200]
        return hashlib.md5(content.encode()).hexdigest()

    def convert_table_to_natural_language(self,
                                         table_data: List[List[str]],
                                         context: str = "") -> str:
        """
        Convert table to natural language using intelligent metadata extraction

        Returns:
            Natural language representation of the table
        """

        # Get metadata
        metadata = self.analyze_table(table_data, context)

        # Build natural language representation
        sentences = []

        # Add title if found
        if metadata.table_title:
            sentences.append(f"This table shows {metadata.table_title}.")

        # Add denomination context
        if metadata.scale_factor > 1:
            scale_text = metadata.scale_description or f"in {metadata.scale_factor:,}s"
            sentences.append(f"All values are {scale_text} of {metadata.currency} unless otherwise noted.")

        # Add exceptions
        if metadata.unit_exceptions:
            exceptions = ', '.join(metadata.unit_exceptions)
            sentences.append(f"Exceptions: {exceptions}.")

        # Convert data rows
        for i, row in enumerate(table_data):
            if i == 0:  # Headers
                continue

            # Build row description
            row_text = []
            for j, cell in enumerate(row):
                if not cell:
                    continue

                # Check if numeric
                try:
                    value = float(str(cell).replace(',', ''))

                    # Apply scaling if not an exception
                    is_exception = any(
                        exc in str(row[0]).lower()
                        for exc in ['per share', 'percentage', 'ratio']
                        if metadata.unit_exceptions
                    )

                    if not is_exception and metadata.scale_factor > 1:
                        actual_value = value * metadata.scale_factor
                        if actual_value >= 1_000_000_000:
                            formatted = f"{metadata.currency}{actual_value/1_000_000_000:.1f} billion"
                        elif actual_value >= 1_000_000:
                            formatted = f"{metadata.currency}{actual_value/1_000_000:.1f} million"
                        else:
                            formatted = f"{metadata.currency}{actual_value:,.0f}"
                    else:
                        formatted = f"{metadata.currency}{value:,.2f}"

                    row_text.append(formatted)

                except ValueError:
                    # Non-numeric cell
                    row_text.append(str(cell))

            if row_text:
                sentences.append(' '.join(row_text))

        return ' '.join(sentences)

def demonstrate_intelligent_conversion():
    """Demonstrate the intelligent table converter"""

    # Sample table from the user's example
    sample_table = [
        ['', '', 'YearEnded', ''],
        ['', 'June 30, 2019', 'June 24, 2018', 'June 25, 2017'],
        ['', '', '(in thousands, except per share data)', ''],
        ['Software license revenue', '82,575', '64,420', '51,234'],
        ['Subscription revenue', '64,955', '56,996', '48,123'],
        ['Net income per share-basic', '$14.37', '$14.73', '$10.47']
    ]

    context = "Note 3: Revenue. The following table presents revenue by category:"

    # Initialize converter (would use API key in production)
    converter = IntelligentTableConverter(llm_provider="openai")

    # Analyze table
    metadata = converter.analyze_table(sample_table, context)

    print("Extracted Metadata:")
    print(f"  Table Title: {metadata.table_title}")
    print(f"  Section Ref: {metadata.section_reference}")
    print(f"  Currency: {metadata.currency}")
    print(f"  Scale: {metadata.scale_factor:,} ({metadata.scale_description})")
    print(f"  Exceptions: {metadata.unit_exceptions}")
    print(f"  Confidence: {metadata.confidence:.2f}")
    print(f"  Method: {metadata.extraction_method}")

    # Convert to natural language
    print("\nNatural Language Conversion:")
    nl_text = converter.convert_table_to_natural_language(sample_table, context)
    print(nl_text)

if __name__ == "__main__":
    demonstrate_intelligent_conversion()