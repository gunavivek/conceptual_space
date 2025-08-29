# B6: Answer Comparison - Architecture Design

## Name
**B6: Answer Comparison**

## Purpose
Compares generated answers with ground truth responses to evaluate answer quality and accuracy through similarity analysis, numeric validation, and comprehensive quality assessment for system performance measurement.

## Input Files
- **Primary**: `outputs/B5_generated_answer.json`
- **Secondary**: `outputs/B1_current_question.json`
- **Contains**: Generated answers and ground truth responses for comparison analysis

## Output Files
- **Primary**: `outputs/B6_answer_comparison.json`
- **Contains**: Detailed comparison results with similarity scores, accuracy assessments, and evaluation metrics

## Processing Logic

### Answer Normalization Framework
- Implements **text standardization** including case conversion, punctuation removal, and whitespace normalization
- Applies **numerical preservation** maintaining quantitative precision while normalizing surrounding text
- Performs **semantic equivalence preparation** standardizing terminology and expression variations
- Creates **comparable text representations** optimized for similarity calculation algorithms

### Similarity Calculation System
- Uses **sequence-based similarity** (SequenceMatcher) for overall text comparison with character-level analysis
- Implements **numeric value extraction** using regex patterns for financial figures, percentages, and quantities
- Calculates **numeric similarity** through proportional difference analysis with tolerance for minor variations
- Combines **text and numeric similarity** using weighted integration (30% text, 70% numeric for numeric answers)

### Accuracy Assessment Framework
- Determines **exact match accuracy** through normalized text comparison for precise answer validation
- Evaluates **numeric accuracy** using threshold-based comparison (±0.01 tolerance) for quantitative answers
- Assesses **semantic correctness** through similarity threshold analysis (>0.8 for correct classification)
- Generates **partial correctness** identification for answers meeting moderate similarity requirements (>0.5)

### Quality Evaluation Integration
- Combines **multiple accuracy dimensions** (exact, numeric, semantic) for comprehensive correctness assessment
- Calculates **overall similarity scores** integrating text and numeric components based on answer type
- Generates **confidence-weighted evaluations** incorporating generation confidence in accuracy assessment
- Produces **comparative quality metrics** showing performance against baseline and expected accuracy levels

## Key Decisions

### Similarity Metric Selection
- **Decision**: Use SequenceMatcher for text similarity rather than semantic embedding similarity
- **Rationale**: Provides interpretable character-level comparison suitable for factual answer evaluation
- **Impact**: Enables precise textual comparison but may miss semantically equivalent but differently worded correct answers

### Numeric Tolerance Policy
- **Decision**: Apply 0.01 absolute difference tolerance for numeric accuracy rather than percentage-based tolerance
- **Rationale**: Accommodates minor rounding differences while maintaining strict accuracy standards for financial data
- **Impact**: Provides appropriate precision for financial answers but may be too strict for approximate quantities

### Weighted Similarity Integration
- **Decision**: Weight numeric similarity (70%) higher than text similarity (30%) for numeric answers
- **Rationale**: Numeric accuracy is typically more critical than textual expression for quantitative questions
- **Impact**: Optimizes evaluation for numeric answers but may undervalue important contextual information

### Correctness Threshold Selection
- **Decision**: Use 0.8 similarity threshold for correctness determination rather than higher/lower thresholds
- **Rationale**: Balances precision requirements with reasonable tolerance for expression variations
- **Impact**: Provides practical correctness assessment but may misclassify borderline cases

### Ground Truth Integration Strategy
- **Decision**: Support both explicit ground truth and fallback mock answers rather than requiring ground truth
- **Rationale**: Enables evaluation in both testing and demonstration scenarios without strict ground truth requirements
- **Impact**: Provides evaluation flexibility but may reduce evaluation quality when using mock data

### Evaluation Completeness Approach
- **Decision**: Generate comprehensive evaluation covering multiple accuracy dimensions rather than single accuracy score
- **Rationale**: Provides detailed analysis supporting both automated evaluation and human review
- **Impact**: Enables thorough answer quality assessment but increases evaluation complexity and output size