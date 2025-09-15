# Q1: Question Ingestion - Architecture Specification

## Core Purpose

Q1 is the **foundation module** of the Q-Pipeline, responsible for loading questions with **critical doc_id linkage** that enables document-specific concept space alignment. This module transforms raw question data into Q-Pipeline format with essential metadata for downstream geometric processing.

## Key Innovation: Document Association

Unlike traditional QA systems that treat questions independently, Q1 ensures every question is **linked to its source document**, enabling:
- Document-specific coordinate system alignment in Q2.5
- Constraint-based matching within document boundaries in Q3.1
- Mathematical precision through shared concept spaces

## Input/Output Specification

### Output File
- **Primary**: `Q_Question_Pipeline/outputs/Q1_Question_ingestion.json`
- **Contains**: Structured question data with document associations and pipeline readiness validation

### Input Sources
```python
supported_formats = {
    "parquet": "Pandas DataFrame format (.parquet)",
    "json": "JSON format with question arrays",
    "b_pipeline_fallback": "B-Pipeline outputs for compatibility"
}

expected_columns = {
    "question_id": ["id", "question_id"],  # Question identifier
    "question_text": ["question", "question_text"],  # Raw question
    "doc_id": ["doc_id", "document_id"],  # Document reference
    "answer": ["answer", "response"],  # Ground truth (optional)
    "metadata": ["source", "dataset_name"]  # Additional info
}
```

### Output Structure
```python
{
    "question_id": str,          # Unique question identifier
    "question_text": str,        # Raw question text
    "doc_id": str,               # CRITICAL: Document reference for concept space
    "answer": str,               # Ground truth answer (for validation)
    "metadata": {
        "source": str,           # Data source identifier
        "ingestion_timestamp": str,  # ISO timestamp
        "raw_data_keys": List[str]   # Original data fields
    },
    "pipeline_ready": bool       # Validation flag for downstream processing
}
```

## Core Architecture Components

### 1. Data Source Abstraction Layer

```python
class DataSourceHandler:
    """
    Abstraction layer for multiple data formats
    """

    def load_parquet_data(self, path: str) -> pd.DataFrame:
        """Load parquet format with column mapping"""

    def load_json_data(self, path: str) -> List[Dict]:
        """Load JSON format with question arrays"""

    def load_b_pipeline_fallback(self, path: str) -> Dict:
        """Fallback to existing B-Pipeline outputs"""
```

### 2. Question Processing Engine

```python
class QuestionProcessor:
    """
    Core processing engine for question transformation
    """

    def process_question(self, raw_data: Dict) -> Dict:
        """
        Transform raw question data into Q-Pipeline format

        CRITICAL STEPS:
        1. Extract question_id (handle multiple column names)
        2. Extract question_text
        3. Determine doc_id (essential for concept space alignment)
        4. Extract ground truth answer (for validation)
        5. Preserve metadata for traceability
        """
```

### 3. Document ID Resolution System

```python
class DocIdResolver:
    """
    CRITICAL: Ensures every question has proper document association
    """

    def resolve_doc_id(self, raw_data: Dict) -> str:
        """
        Resolve document ID using multiple strategies:

        Priority Order:
        1. Explicit doc_id field
        2. document_id field
        3. Extract from question_id pattern
        4. Use question_id as doc reference (fallback)
        """

    def validate_doc_id(self, doc_id: str) -> bool:
        """Validate doc_id format and existence"""
```

## Implementation Algorithm

### Step 1: Data Source Detection
```python
def detect_data_source(self, path: str) -> str:
    """
    Detect data format and select appropriate loader
    """
    if path.endswith('.parquet'):
        return 'parquet'
    elif path.endswith('.json'):
        return 'json'
    elif 'B_Retrieval_pipeline' in path:
        return 'b_pipeline_fallback'
    else:
        raise ValueError(f"Unsupported data format: {path}")
```

### Step 2: Column Mapping and Extraction
```python
def extract_question_fields(self, raw_data: Dict) -> Dict:
    """
    Extract question fields with flexible column mapping
    """
    # Question ID extraction with fallbacks
    question_id = (
        raw_data.get('question_id') or
        raw_data.get('id') or
        'unknown'
    )

    # Question text extraction
    question_text = (
        raw_data.get('question') or
        raw_data.get('question_text') or
        ''
    )

    # Answer extraction (for validation)
    answer = (
        raw_data.get('answer') or
        raw_data.get('response') or
        ''
    )

    return {
        'question_id': question_id,
        'question_text': question_text,
        'answer': answer
    }
```

### Step 3: Document ID Resolution (CRITICAL)
```python
def resolve_document_association(self, raw_data: Dict) -> str:
    """
    CRITICAL: Resolve document association for concept space alignment
    """
    # Strategy 1: Explicit doc_id
    doc_id = raw_data.get('doc_id')
    if doc_id:
        return str(doc_id)

    # Strategy 2: Alternative column names
    doc_id = raw_data.get('document_id')
    if doc_id:
        return str(doc_id)

    # Strategy 3: Extract from question_id pattern
    question_id = raw_data.get('question_id') or raw_data.get('id')
    if question_id:
        # For patterns like 'finqa_test_1630'
        if '_' in str(question_id):
            parts = str(question_id).rsplit('_', 1)
            if len(parts) > 1:
                return str(question_id)  # Use full ID as doc reference

    # Strategy 4: Fallback - use question_id as doc reference
    if question_id:
        return str(question_id)

    raise ValueError("Cannot determine doc_id for concept space alignment")
```

### Step 4: Validation and Quality Control
```python
def validate_processed_question(self, processed: Dict) -> bool:
    """
    Validate processed question meets Q-Pipeline requirements
    """
    required_fields = ['question_id', 'question_text', 'doc_id']

    for field in required_fields:
        if field not in processed or not processed[field]:
            return False

    # Additional validations
    if len(processed['question_text']) < 5:
        return False  # Question too short

    if not isinstance(processed['doc_id'], str):
        return False  # Doc ID must be string

    return True
```

## Caching and Performance Optimization

### Question Cache System
```python
class QuestionCache:
    """
    LRU cache for frequently accessed questions
    """

    def __init__(self, cache_size: int = 1000):
        self.cache = {}
        self.cache_size = cache_size
        self.access_order = []

    def get_question(self, question_id: str) -> Optional[Dict]:
        """Get cached question with LRU update"""

    def cache_question(self, question_id: str, question_data: Dict):
        """Cache processed question with LRU eviction"""
```

### Batch Processing Optimization
```python
def load_batch_questions(self, question_ids: List[str]) -> List[Dict]:
    """
    Optimized batch loading for multiple questions

    Optimizations:
    1. Single file read for parquet sources
    2. Vectorized pandas operations
    3. Parallel processing for large batches
    4. Cache utilization for repeated requests
    """
```

## Integration Points

## Output File Format

The Q1 module produces a structured JSON output file at:
`Q_Question_Pipeline/outputs/Q1_Question_ingestion.json`

### File Structure
```json
{
  "question_id": {
    "question_id": "string",
    "question_text": "string",
    "doc_id": "string",
    "answer": "string",
    "metadata": {
      "source": "string",
      "ingestion_timestamp": "ISO_timestamp",
      "raw_data_keys": ["array_of_original_fields"]
    },
    "pipeline_ready": boolean
  }
}
```

### Integration Points

### Downstream Module Requirements

#### Q2.5 Dependencies
```python
q25_requirements = {
    "question_id": "Unique identifier for processing chain",
    "question_text": "Raw text for semantic analysis",
    "doc_id": "CRITICAL: Document reference for concept space loading"
}
```

#### Q3.1 Dependencies
```python
q31_requirements = {
    "question_id": "Question identifier for result tracking",
    "doc_id": "Document scope for chunk filtering"
}
```

### A-Pipeline Coordination
```python
def validate_doc_id_alignment(self, doc_id: str) -> bool:
    """
    Validate that doc_id exists in A-Pipeline outputs
    """
    a_pipeline_docs = self.load_a_pipeline_document_list()
    return doc_id in a_pipeline_docs
```

## Error Handling and Fallback Strategies

### Data Loading Failures
```python
def handle_data_loading_error(self, error: Exception) -> Dict:
    """
    Graceful degradation for data loading failures
    """
    return {
        "error_type": "data_loading_failure",
        "fallback_strategy": "attempt_alternative_sources",
        "retry_options": ["b_pipeline_fallback", "manual_input"]
    }
```

### Missing Doc ID Resolution
```python
def handle_missing_doc_id(self, raw_data: Dict) -> str:
    """
    Fallback strategy for missing document association
    """
    # Create synthetic doc_id based on question characteristics
    question_hash = hashlib.md5(
        raw_data.get('question', '').encode()
    ).hexdigest()[:8]

    synthetic_doc_id = f"synthetic_{question_hash}"

    # Log warning for analysis
    self.log_warning(f"Created synthetic doc_id: {synthetic_doc_id}")

    return synthetic_doc_id
```

## Quality Metrics and Monitoring

### Processing Metrics
```python
processing_metrics = {
    "ingestion_success_rate": float,    # % successfully processed
    "doc_id_resolution_rate": float,    # % with valid doc_id
    "cache_hit_rate": float,            # Cache efficiency
    "avg_processing_time": float,       # Performance metric
    "data_quality_score": float         # Overall quality assessment
}
```

### Data Quality Indicators
```python
quality_indicators = {
    "missing_doc_id_count": int,        # Questions without doc association
    "empty_question_count": int,        # Questions with no text
    "duplicate_question_count": int,    # Duplicate question_ids
    "invalid_format_count": int         # Malformed data entries
}
```

## Testing Requirements

### Unit Tests
1. **Column Mapping Tests**: Verify flexible column name handling
2. **Doc ID Resolution Tests**: Test all resolution strategies
3. **Data Format Tests**: Validate parquet, JSON, fallback loading
4. **Cache Tests**: Verify LRU cache behavior
5. **Validation Tests**: Ensure processed questions meet requirements

### Integration Tests
1. **A-Pipeline Alignment**: Verify doc_ids exist in A-Pipeline outputs
2. **Q2.5 Compatibility**: Ensure output format matches Q2.5 expectations
3. **Batch Processing**: Test large-scale question loading
4. **Error Scenarios**: Validate graceful failure handling

### Performance Tests
1. **Load Testing**: Process 1000+ questions within time limits
2. **Memory Testing**: Monitor cache memory usage
3. **Concurrent Access**: Verify thread safety for parallel processing

## Success Criteria

### Functional Requirements
1. **100% Doc ID Resolution**: Every question must have valid document association
2. **Format Flexibility**: Support multiple input data formats seamlessly
3. **Cache Efficiency**: >70% cache hit rate for repeated access
4. **Processing Speed**: <10ms per question for cached data
5. **Data Integrity**: Zero data corruption during transformation

### Quality Requirements
1. **Validation Rate**: >95% of processed questions pass validation
2. **Error Recovery**: Graceful handling of all expected error scenarios
3. **Compatibility**: Seamless integration with Q2.5 and downstream modules
4. **Traceability**: Complete audit trail for data transformations

## Configuration Parameters

```python
q1_config = {
    "cache_size": 1000,                 # Question cache size
    "batch_size": 100,                  # Optimal batch processing size
    "validation_strict": True,          # Strict validation mode
    "fallback_enabled": True,           # Enable fallback strategies
    "logging_level": "INFO",            # Logging verbosity
    "data_quality_threshold": 0.95      # Minimum quality score
}
```

## Summary

Q1 Question Ingestion is the **critical foundation** of the Q-Pipeline, ensuring every question has proper document association for downstream geometric processing. The module's flexible architecture handles diverse data formats while maintaining the strict requirement for doc_id linkage that enables the revolutionary convex ball constraint system.

**Key Success Factor**: The quality of Q1's doc_id resolution directly impacts the effectiveness of Q2.5's concept space alignment and Q3.1's constraint-based matching.

---
*Q1 Architecture Specification v1.0*
*Foundation Module for Revolutionary Geometric Question Processing*