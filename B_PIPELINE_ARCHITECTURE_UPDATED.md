# B-Pipeline Architecture Document
**Last Updated**: September 12, 2025
**Status**: FULLY OPERATIONAL

## Executive Summary
The B-Pipeline implements a sophisticated Retrieval and Answer Generation system using a tri-semantic matching approach with weighted combination strategies. The pipeline processes financial questions through multiple semantic layers before generating answers using either OpenAI GPT-3.5 or rule-based methods.

## Pipeline Overview

```
┌──────────────────────────────────────────────────────────────┐
│                     B-PIPELINE FLOW                          │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  B1 → B2 (3 layers) → B3 (3 strategies) → B4 → B5 → B6      │
│                                                               │
│  B1: Question Loading                                        │
│  B2: Context Splitting & Transformation                      │
│  B3: Tri-Semantic Matching                                   │
│  B4: Weighted Strategy Combination                           │
│  B5: Answer Generation (OpenAI/Rule-based)                   │
│  B6: Answer Validation                                       │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

## Component Specifications

### B1: Question Loader
- **Script**: `B1_read_question.py`
- **Input**: `A_Concept_pipeline/data/sample_20_records.parquet`
- **Output**: `outputs/B1_current_question.json`
- **Function**: Loads financial questions from ground truth dataset
- **Supports**: Single question or batch processing (all 20 questions)

### B2: Context Splitting Layer
Three sub-components process questions through different semantic lenses:

#### B2.1: Intent Layer Modeling
- **Script**: `B2_1_intent_layer_modeling.py`
- **Input**: `B1_current_question.json`
- **Output**: `B2.1_intent_layer_output.json`
- **Function**: Extracts question intent and key terms

#### B2.2: Declarative Transformation
- **Script**: `B2_2_declarative_transformation.py`
- **Input**: `B2.1_intent_layer_output.json`
- **Output**: `B2.2_declarative_transformation_output.json`
- **Function**: Transforms question into declarative statements

#### B2.3: Answer Expectation Prediction
- **Script**: `B2_3_answer_expectation_prediction.py`
- **Input**: `B2.2_declarative_transformation_output.json`
- **Output**: `B2.3_answer_expectation_output.json`
- **Function**: Predicts expected answer format and type

### B3: Tri-Semantic Matching Strategies
Three parallel strategies match questions to concept chunks:

#### B3.1: Intent Matching (53.8% weight)
- **Script**: `B3.1_intent_matching.py`
- **Inputs**: 
  - `B2.1_intent_layer_output.json`
  - `A_Concept_pipeline/outputs/A3_multi_strategy_chunks.json`
- **Output**: `B3.1_intent_matching_output.json`
- **Function**: Matches based on question intent

#### B3.2: Declarative Matching (36.2% weight)
- **Script**: `B3.2_declarative_matching.py`
- **Inputs**: 
  - `B2.2_declarative_transformation_output.json`
  - A-pipeline chunks
- **Output**: `B3.2_declarative_matching_output.json`
- **Function**: Matches using declarative patterns

#### B3.3: Answer Capability Assessment (10% weight → **RECOMMENDED: 100%**)
- **Script**: `B3.3_answer_capability_assessment.py`
- **Inputs**: 
  - `B2.3_answer_expectation_output.json`
  - `A_Concept_pipeline/outputs/A3_multi_strategy_chunks.json` (with concept memberships)
- **Output**: `B3.3_answer_capability_assessment_output.json`
- **Function**: Assesses chunks' capability to provide required answer types using semantic analysis and concept integration
- **Enhancement**: Now integrates A-pipeline concept memberships and importance scores

### B4: Weighted Strategy Combination
- **Script**: `B4_weighted_strategy_combination.py`
- **Inputs**: All three B3 outputs
- **Output**: `B4_weighted_combination_output.json`
- **Function**: Combines strategies using weighted scoring
- **Weights**: B3.1 (53.8%), B3.2 (36.2%), B3.3 (10%)

### B5: Enhanced Answer Generation
- **Script**: `B5_enhanced_answer_generation.py`
- **Input**: `B4_weighted_combination_output.json`
- **Output**: `B5_enhanced_answer_output.json`
- **Function**: Generates answers using:
  - Primary: OpenAI GPT-3.5-turbo (if API key configured)
  - Fallback: Rule-based generation for specific question types
- **Configuration**: API key in `Config.py`

### B6: Answer Validation
- **Script**: `B6_answer_validation.py`
- **Input**: `B5_enhanced_answer_output.json`
- **Output**: `B6_validation_results.json`
- **Function**: Validates generated answers against ground truth
- **Metrics**: Accuracy rate, correct count, partial matches

## Data Flow Summary

```
Ground Truth Data (parquet)
         ↓
    B1: Load Questions
         ↓
    B2: Context Split
      ↙  ↓  ↘
   B2.1 B2.2 B2.3
     ↓   ↓   ↓
   B3.1 B3.2 B3.3  ← [A3 Chunks]
      ↘  ↓  ↙
    B4: Combine
         ↓
    B5: Generate ← [OpenAI API]
         ↓
    B6: Validate
```

## Key Features

1. **Tri-Semantic Processing**: Three parallel semantic strategies for comprehensive understanding
2. **Weighted Combination**: Optimized weights based on strategy performance
3. **Hybrid Generation**: OpenAI API with rule-based fallback
4. **Batch Processing**: Supports single or multiple question processing
5. **Validation System**: Automatic accuracy assessment against ground truth

## Performance Metrics
- **Current Accuracy**: 55% (11/20 correct)
- **Processing Time**: ~2-3 minutes for 20 questions
- **OpenAI Usage**: 100% when API key configured

## Configuration Requirements
- **OpenAI API Key**: Set in `Config.py` (line 41)
- **Input Data**: `sample_20_records.parquet` required
- **A-Pipeline Dependency**: `A3_multi_strategy_chunks.json` must exist

## Execution Order
```bash
python B1_read_question.py
python B2_1_intent_layer_modeling.py
python B2_2_declarative_transformation.py
python B2_3_answer_expectation_prediction.py
python B3.1_intent_matching.py
python B3.2_declarative_matching.py
python B3.3_answer_capability_assessment.py
python B4_weighted_strategy_combination.py
python B5_enhanced_answer_generation.py
python B6_answer_validation.py
```

## Output Files Location
All outputs stored in: `B_Retrieval_pipeline/outputs/`

## Notes for Developers
- All components include error handling and logging
- Each script can run independently with proper inputs
- Unicode handling implemented throughout
- Paths use relative references for portability