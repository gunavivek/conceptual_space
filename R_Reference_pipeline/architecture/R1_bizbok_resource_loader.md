# R1: BIZBOK Resource Loader

## Purpose
Load and process BIZBOK concepts from Excel file into structured JSON format for semantic ontology building.

## Input
- **Source:** `data/Information Map for all industry and common.xlsx`
- **Required Columns:**
  - `Industry Domain` - Business domain classification
  - `Information Concept` - Concept name
  - `Information Concept Definition` - Concept definition  
  - `Related Information Concepts` - Related concepts (comma/semicolon separated)
- **Volume:** 516 rows with 97 unique concepts

## Output Files
1. **R1_CONCEPTS.json** - Complete concept definitions with keywords and relationships
2. **R1_DOMAINS.json** - Domain structure and concept mappings
3. **R1_KEYWORDS.json** - Keyword index for concept discovery
4. **R1_processing_report.json** - Processing statistics and quality metrics

## Processing Logic

### 1. Data Loading
- Load Excel with flexible column mapping
- Handle duplicates by preserving first occurrence
- Track skipped rows with detailed logging

### 2. Keyword Extraction
- Extract from concept name (split, clean, lowercase)
- Extract from definition (top 20 meaningful words)
- Filter stop words and short terms (< 3 chars)
- Generate compound terms (2-grams) from definition

### 3. Relationship Processing
- Parse related concepts from delimited strings
- Convert to standardized concept IDs (lowercase, underscores)
- Validate relationships against existing concepts
- Optional: Add bidirectional relationships (toggle: `add_bidirectional`)

### 4. Domain Organization
- Group concepts by domain
- Count concepts per domain
- Generate domain descriptions

## Configuration Options
- **add_bidirectional** (bool): Enable/disable automatic bidirectional relationships
  - `True`: Creates bidirectional links (ontology mode)
  - `False`: Preserves exact Excel relationships (data fidelity mode)

## Performance
- **Processing Time:** < 2 seconds for 516 rows
- **Memory Usage:** < 50MB
- **Scalability:** Linear O(n) complexity

## Quality Metrics
- Tracks valid vs skipped concepts
- Reports duplicate handling
- Validates relationship integrity
- Provides keyword extraction statistics

## Script Location
`Scripts/R1_bizbok_resource_loader.py`