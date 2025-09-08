# B-Pipeline (Intent Space) - Complete Walkthrough
**Last Updated**: 2025-09-07  
**Status**: ✅ **OPERATIONAL - Clean B1-B4 Architecture**

## Overview

The B-Pipeline represents the **Intent Space** of the tri-semantic architecture. Its purpose is to:
1. Understand what the user is asking (intent)
2. Transform the question into multiple representations
3. Match these representations against concept-enriched chunks from A-Pipeline
4. Produce weighted concept rankings for retrieval

## Pipeline Architecture

```
B1: Question Input
    ↓
B2: Parallel Intent Processing
    ├── B2.1: Intent Layer Modeling
    ├── B2.2: Declarative Transformation  
    └── B2.3: Answer Expectation Prediction
    ↓
B3: Multi-Strategy Concept Matching
    ├── B3.1: Intent-based Matching
    ├── B3.2: Declarative Matching
    └── B3.3: Answer Backward Matching
    ↓
B4: Weighted Strategy Combination
    ↓
Output: Ranked chunks for I-Pipeline
```

## Component Details

### B1: Question Input Layer (`B1_read_question.py`)

**Purpose**: Load and analyze incoming questions

**Key Functions**:
- `load_question_from_parquet()`: Load questions from test data
- `analyze_question()`: Basic question analysis

**Analysis Performed**:
- **Question Type Detection**: what, how, why, when, where, who, yes/no
- **Answer Type Prediction**: numeric, percentage, date, boolean, text
- **Keyword Extraction**: Important terms from the question

**Example Output**:
```json
{
    "question_id": "finqa_test_617",
    "question": "What was the total deferred income in 2019?",
    "question_type": "what",
    "answer_type": "numeric",
    "keywords": ["total", "deferred", "income", "2019"]
}
```

### B2: Parallel Intent Processing (3 Strategies Run in Parallel)

#### B2.1: Intent Layer Modeling (`B2_1_intent_layer_modeling.py`)

**Purpose**: Analyze the deep intent behind the question

**Intent Categories**:
- **Comparison**: compare, difference, versus, change, growth
- **Calculation**: calculate, total, sum, average, percentage
- **Definition**: what is, define, explain, describe
- **Identification**: which, who, what, identify, list
- **Temporal**: when, date, year, period, quarter
- **Causal**: why, reason, cause, result
- **Procedural**: how, process, method, steps
- **Factual**: is, are, was, were, does

**Output**:
```json
{
    "primary_intent": "calculation",
    "all_intents": ["calculation", "temporal"],
    "intent_scores": {"calculation": 2, "temporal": 1},
    "expects_numeric": true,
    "is_comparative": false,
    "confidence": 0.6
}
```

#### B2.2: Declarative Transformation (`B2_2_declarative_transformation.py`)

**Purpose**: Convert questions into declarative statements for better matching

**Transformation Rules**:
- "What was X?" → "X was [ANSWER]"
- "How much did Y increase?" → "Y increased by [ANSWER]"
- "What is the total of Z?" → "The total of Z is [ANSWER]"

**Key Features**:
- Pattern-based transformation
- Preserves key entities
- Creates matchable statement templates

**Example**:
```
Question: "What was the total deferred income in 2019?"
Declarative: "The total deferred income in 2019 was [ANSWER]"
```

#### B2.3: Answer Expectation Prediction (`B2_3_answer_expectation_prediction.py`)

**Purpose**: Predict what form the answer should take

**Predictions Made**:
- **Format**: number, text, date, percentage, currency
- **Magnitude**: units, thousands, millions, billions
- **Structure**: single value, list, comparison, calculation
- **Confidence**: How certain about the prediction

**Example Output**:
```json
{
    "expected_format": "currency",
    "expected_magnitude": "millions",
    "expected_structure": "single_value",
    "units": "dollars",
    "confidence": 0.85
}
```

### B3: Multi-Strategy Concept Matching (3 Strategies Run in Parallel)

#### B3.1: Intent-based Matching (`B3.1_intent_matching.py`)

**Purpose**: Match chunks based on intent alignment

**Process**:
1. Takes intent analysis from B2.1
2. Scores chunks based on intent-concept alignment
3. Prioritizes chunks matching the primary intent

**Scoring Factors**:
- Intent-concept relevance
- Intent keyword presence
- Semantic similarity to intent

#### B3.2: Declarative Matching (`B3.2_declarative_matching.py`)

**Purpose**: Match chunks using declarative patterns

**Process**:
1. Takes declarative statement from B2.2
2. Finds chunks that could complete the statement
3. Pattern matching and template filling

**Advantages**:
- Better for factual questions
- Improved precision for specific queries
- Natural language pattern matching

#### B3.3: Answer Backward Matching (`B3.3_answer_backward_matching.py`)

**Purpose**: Match chunks that could contain the expected answer type

**Process**:
1. Takes answer expectations from B2.3
2. Finds chunks with matching answer patterns
3. Backward reasoning from answer to question

**Example**:
- If expecting currency in millions → prioritize chunks with "$XX.X million"
- If expecting percentage → prioritize chunks with "XX%"
- If expecting date → prioritize chunks with temporal markers

### B4: Weighted Strategy Combination (`B4_weighted_strategy_combination.py`)

**Purpose**: Combine results from all B3 strategies

**Weighting Scheme**:
```python
default_weights = {
    'intent_matching': 0.4,      # B3.1
    'declarative_matching': 0.35, # B3.2
    'answer_backward': 0.25      # B3.3
}
```

**Combination Process**:
1. Collect scores from all B3 strategies
2. Apply strategy weights
3. Normalize scores
4. Rank chunks by combined score
5. Return top-k chunks

**Output Format**:
```json
{
    "ranked_chunks": [
        {
            "chunk_id": "finqa_test_617_semantic_sentence_0",
            "combined_score": 0.87,
            "strategy_scores": {
                "intent": 0.9,
                "declarative": 0.85,
                "backward": 0.82
            },
            "concept_memberships": ["core_1"],
            "relevance_explanation": "High alignment with calculation intent"
        }
    ],
    "processing_time": 0.6,
    "strategies_used": 3
}
```

## Data Flow Example

**Input Question**: "What was the change in deferred income between 2018 and 2019?"

### B1 Output:
```
question_type: "what"
answer_type: "numeric"
keywords: ["change", "deferred", "income", "2018", "2019"]
```

### B2 Parallel Processing:
- **B2.1**: Intent = "comparison/calculation"
- **B2.2**: Declarative = "The change in deferred income between 2018 and 2019 was [ANSWER]"
- **B2.3**: Expects = "numeric value, possibly negative, in millions"

### B3 Parallel Matching:
- **B3.1**: Finds chunks with comparison/calculation concepts
- **B3.2**: Finds chunks matching the declarative pattern
- **B3.3**: Finds chunks containing year-over-year changes

### B4 Combination:
- Aggregates all scores
- Applies weights
- Returns top 5 chunks ranked by relevance

## Performance Characteristics

- **B1**: ~0.05s (question analysis)
- **B2**: ~0.2s (parallel intent processing)
- **B3**: ~0.3s (parallel concept matching)
- **B4**: ~0.1s (weighted combination)
- **Total**: ~0.65s typical processing time

## Integration Points

### Input from A-Pipeline:
- Chunks from A3 (preferably raw chunks for diversity)
- Concepts from A2.4 and A2.5
- Document metadata

### Output to I-Pipeline:
- Ranked chunks with scores
- Intent analysis metadata
- Strategy contributions
- Ready for I5 answer generation

## Key Design Principles

1. **Parallel Processing**: B2 and B3 run strategies in parallel
2. **Multi-Strategy Approach**: Different strategies for different question types
3. **Intent-Driven**: Understanding what the user wants guides retrieval
4. **Clean Boundaries**: B1-B4 only, no answer generation (that's I5)
5. **Weighted Fusion**: Combines multiple signals for robust ranking

## Configuration Options

### Adjustable Parameters:
```python
# B2.1 Intent Modeling
intent_confidence_threshold = 0.5

# B2.2 Declarative Transformation  
use_advanced_patterns = True

# B2.3 Answer Prediction
prediction_confidence_min = 0.3

# B3 Matching Strategies
matching_threshold = 0.4

# B4 Weights (can be adjusted per domain)
strategy_weights = {
    'intent': 0.4,
    'declarative': 0.35,
    'backward': 0.25
}
```

## Summary

The B-Pipeline transforms user questions into multiple representations, analyzes intent from different angles, and matches against concept-enriched chunks using three parallel strategies. It produces weighted rankings that combine intent understanding, declarative patterns, and answer expectations - all optimized for sub-second processing to enable real-time question answering.