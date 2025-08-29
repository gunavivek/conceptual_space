# Conceptual Space Pipeline System

A dual-pipeline system for concept extraction and retrieval-augmented generation (RAG).

## System Overview

### Pipeline A: Concept Building Pipeline
Processes documents to extract, group, and embed concepts for building a conceptual space.

### Pipeline B: Retrieval & QA Pipeline  
Handles question processing, concept matching, and answer generation using the conceptual space.

## Directory Structure

```
conceptual_space/
├── A_concept_pipeline/       # Pipeline A: Concept extraction and processing
│   ├── scripts/             # A1.x, A2.x, A3.x processing scripts
│   ├── outputs/             # Generated concept files and embeddings
│   ├── data/                # Input documents (parquet files)
│   ├── config/              # Pipeline configuration
│   ├── architecture/        # Design documents
│   └── tests/               # Unit tests
│
├── B_retrieval_pipeline/    # Pipeline B: Question answering
│   ├── scripts/             # B1-B6 processing scripts
│   ├── outputs/             # Question processing results
│   ├── data/                # Question datasets
│   ├── config/              # Pipeline configuration
│   ├── architecture/        # Design documents
│   └── tests/               # Unit tests
│
├── shared/                  # Shared utilities
│   ├── utils/              # Common utility functions
│   ├── embeddings/         # Embedding models and utilities
│   └── config/             # Shared configuration
│
├── docs/                    # Documentation
└── logs/                    # Execution logs
```

## Pipeline A: Concept Building (A-Series)

### Stage 1: Document Processing
- **A1.1**: Document Reader - Load documents from parquet files
- **A1.2**: Domain Detection - Identify document domains

### Stage 2: Concept Extraction
- **A2.1**: Document Preprocessing - Clean and normalize text
- **A2.2**: Keyword Extraction - Extract key terms and phrases
- **A2.3**: Concept Grouping - Group related concepts thematically
- **A2.4**: Core Concept Synthesis - Identify main themes
- **A2.5**: Concept Expansion - Expand concepts using multiple strategies
- **A2.6**: Business Concept Centroids - Generate concept embeddings
- **A2.7**: Semantic Chunking - Create semantic chunks
- **A2.8**: Quality Validation - Validate extraction quality

### Stage 3: Optimization
- **A3**: Centroid Validation & Optimization

## Pipeline B: Retrieval & QA (B-Series)

### Question Processing
- **B1**: Read Question - Load and parse user question
- **B2.1**: Intent Layer Modeling - Analyze question intent
- **B2.2**: Declarative Transformation - Convert to declarative form
- **B2.3**: Answer Expectation - Predict expected answer type

### Concept Matching
- **B3.1**: Intent-Based Matching - Match based on intent
- **B3.2**: Declarative Form Matching - Match declarative patterns
- **B3.3**: Answer-Backwards Matching - Reverse-engineer from answer

### Answer Generation
- **B4**: Weighted Strategy Combination - Combine matching strategies
- **B5**: Answer Generation - Generate final answer
- **B6**: Answer Comparison - Compare with ground truth

## Quick Start

### Running Pipeline A
```bash
cd A_concept_pipeline/scripts
python A1.1_document_reader.py
python A1.2_document_domain_detector.py
# ... continue with A2.x scripts
```

### Running Pipeline B
```bash
cd B_retrieval_pipeline/scripts
python B1_read_question.py
python B2_1_intent_layer_modeling.py
# ... continue with remaining B scripts
```

## Configuration

### Environment Variables
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Data Sources
- Place input documents in `A_concept_pipeline/data/`
- Default test file: `test_5_records.parquet`

## Dependencies

```python
pandas
numpy
scikit-learn
sentence-transformers
openai
nltk
spacy
```

## Development Status

🔄 **Rebuilding from scratch** - Pipeline structure recreated after data loss

## License

MIT

## Contact

For questions or issues, please contact the repository owner.