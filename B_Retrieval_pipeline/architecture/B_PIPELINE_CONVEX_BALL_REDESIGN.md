# B-Pipeline Convex Ball Architecture: Complete Redesign Specification

## Document Version
- **Version**: 3.0 - Convex Ball Architecture
- **Date**: September 13, 2025
- **Status**: ARCHITECTURAL REDESIGN SPECIFICATION
- **Purpose**: Complete B-Pipeline rebuild based on human cognitive process analysis

---

## Executive Summary

This document specifies the complete architectural redesign of the B-Pipeline to implement true **convex ball constrained matching** based on human cognitive process analysis. The redesign fundamentally shifts from global similarity matching to **intra-convex-ball geometric matching** within document-specific concept spaces.

### Key Innovation
Transform B-Pipeline from traditional information retrieval to **human-aligned cognitive processing** where questions and chunks are matched within mathematically defined convex balls centered on semantic concept centroids.

---

## Architectural Foundation

### Core Principles

1. **Document-Specific Concept Spaces**: Each document establishes its own n-dimensional concept space with centroids
2. **Convex Ball Constraints**: All matching occurs within shared convex boundaries
3. **Coordinate System Alignment**: Questions map to same coordinate system as document chunks
4. **Geometric Precision**: True Euclidean distance calculations within constrained spaces
5. **Human Process Mirroring**: Architecture mirrors step-by-step human cognitive analysis

### Human Cognitive Process Mapping
```
Human Process → B-Pipeline Implementation
────────────────────────────────────────────
1. Question Intent Analysis → B2.1 Enhanced Intent Layer
2. Question Semantic Analysis → B2.2-B2.4 Concept Feature Extraction
3. Question-Concept Mapping → B2.5 Convex Ball Coordinate Assignment
4. Intra-Ball Chunk Finding → B3.1 Constrained Geometric Matching
5. Semantic Refinement → B3.2 Intra-Ball Semantic Scoring
6. Answer Extraction → B3.3 Structured Data Capability Assessment
7. Final Ranking → B4 Convex Ball Constrained Fusion
```

---

## Module-by-Module Redesign Specifications

### B2.1 Intent Layer - MINOR ENHANCEMENTS
**Current Role**: Question intent analysis → Intent vector
**Redesign Requirements**: Enhanced intent categories for structured data

#### New Intent Categories
```python
intent_categories = {
    # Existing categories (keep)
    "comparison": float,
    "calculation": float,
    "definition": float,
    "identification": float,
    "factual": float,

    # NEW categories for structured data
    "table_intersection": float,     # NEW: Table coordinate lookup intent
    "temporal_lookup": float,        # NEW: Year/date specific queries
    "numerical_extraction": float,   # ENHANCED: Numeric value retrieval
    "contextual_integration": float  # NEW: Cross-chunk context combination
}
```

#### Implementation Changes
- Add detection logic for table-based queries
- Enhanced temporal intent scoring
- Output 8-dimensional intent vectors (vs current 5-dimensional)
- **Integration Point**: Intent vectors feed into B3.1 geometric weighting

---

### B2.2 Semantic Layer - MODERATE REDESIGN
**Current Role**: Semantic similarity analysis
**New Role**: Question semantic fingerprinting for concept space mapping

#### Major Changes
```python
# OLD: Global semantic similarity scoring
def analyze_semantic_similarity(question, chunks): pass

# NEW: Semantic concept fingerprinting
def extract_question_semantic_features(question, doc_concept_space):
    """Extract semantic features aligned with document's concept centroids"""
    return {
        "concept_alignments": calculate_semantic_alignment_per_concept(question),
        "domain_indicators": extract_domain_semantic_markers(question),
        "structural_hints": detect_semantic_structure_references(question),
        "semantic_coordinates": map_semantics_to_concept_dimensions(question)
    }
```

#### Integration Changes
- Remove duplicate concept mapping functionality (delegated to B2.5)
- Focus on semantic feature extraction for B2.5 input
- Output semantic fingerprints rather than similarity scores

---

### B2.3 Contextual Layer - MAJOR REDESIGN
**Current Role**: General context analysis
**New Role**: Question context alignment with document structure

#### Complete Architecture Change
```python
def analyze_question_context(question, doc_id):
    """Context analysis aligned with specific document structure"""
    return {
        "document_scope": {
            "target_doc_id": doc_id,  # CRITICAL: Must match A-pipeline doc_id
            "cross_doc_scope": False  # Questions constrained to single document
        },
        "temporal_context": {
            "target_years": extract_years_periods(question),
            "temporal_scope": identify_time_range_constraints(question),
            "relative_temporal": detect_comparison_periods(question)
        },
        "structural_context": {
            "table_references": detect_table_navigation_hints(question),
            "coordinate_hints": extract_row_column_references(question),
            "denomination_context": identify_unit_requirements(question)
        },
        "scope_constraints": {
            "category_filters": identify_category_constraints(question),
            "jurisdiction_scope": extract_geographic_constraints(question),
            "hierarchy_level": detect_aggregation_level(question)
        }
    }
```

#### Key Innovation
- **Document-Specific Context**: All context analysis tied to specific doc_id
- **Structural Intelligence**: Detection of table navigation requirements
- **Constraint Identification**: Scope limitations that inform convex ball selection

---

### B2.4 Temporal Layer - ENHANCE EXISTING
**Current Role**: Temporal analysis
**Enhanced Role**: Temporal coordinate mapping for convex ball positioning

#### Enhancement Requirements
```python
def enhanced_temporal_analysis(question, doc_temporal_space):
    """Temporal analysis with coordinate system integration"""
    temporal_features = existing_temporal_analysis(question)  # Keep existing

    # NEW: Temporal coordinate mapping
    temporal_coordinates = {
        "temporal_position": map_time_references_to_coordinates(question),
        "temporal_scope_vector": calculate_time_range_vector(question),
        "relative_temporal_vector": compute_comparison_temporal_vector(question)
    }

    # NEW: Integration with concept space
    concept_temporal_contribution = calculate_temporal_concept_weights(
        temporal_coordinates, doc_temporal_space
    )

    return {
        **temporal_features,  # Existing functionality preserved
        "temporal_coordinates": temporal_coordinates,
        "concept_space_contribution": concept_temporal_contribution
    }
```

---

### B2.5 Question-Concept Mapping - FUNDAMENTAL REDESIGN
**Current Role**: Global fuzzy concept membership
**New Role**: Document-specific convex ball coordinate assignment

#### Revolutionary Architecture Change
```python
class ConvexBallQuestionMapper:
    def map_question_to_convex_balls(self, question_id, doc_id):
        """CRITICAL: Map question to document's specific concept space"""

        # STEP 1: Load A-pipeline concept centroids for this document
        doc_concept_centroids = self.load_a_pipeline_centroids(doc_id)

        # STEP 2: Aggregate B2.1-B2.4 question features
        question_features = self.aggregate_b2_features(question_id)

        # STEP 3: Map to same coordinate system as A-pipeline chunks
        question_coords = self.calculate_question_coordinates(
            question_features, doc_concept_centroids
        )

        # STEP 4: Determine convex ball memberships
        ball_memberships = {}
        for concept_id, centroid in doc_concept_centroids.items():
            distance = euclidean_distance(question_coords, centroid.coordinates)
            if distance <= centroid.radius:
                membership_score = self.calculate_membership_score(distance, centroid.radius)
                ball_memberships[concept_id] = {
                    "membership_score": membership_score,
                    "within_ball": True,
                    "distance_from_centroid": distance,
                    "centroid_coordinates": centroid.coordinates,
                    "ball_radius": centroid.radius
                }

        return {
            "question_id": question_id,
            "doc_id": doc_id,
            "question_coordinates": question_coords,
            "convex_ball_memberships": ball_memberships,
            "coordinate_system_dimension": len(question_coords)
        }
```

#### Critical Integration Points
- **A-Pipeline Dependency**: MUST load concept centroids established by A-pipeline
- **Coordinate System Alignment**: Question coordinates MUST use same dimensions as chunk coordinates
- **Document Scope**: Each question maps to specific document's concept space

---

## B3.x Layer Revolutionary Redesign

### B3.1 Geometric Intent Matching - COMPLETE OVERHAUL
**Current**: Global concept space geometric matching
**New**: Intra-convex-ball constrained geometric matching

#### New Architecture
```python
class ConvexBallGeometricMatcher:
    def geometric_intent_matching_constrained(self, question_id, doc_id):
        """Geometric matching within convex ball constraints"""

        # STEP 1: Load question coordinates and ball memberships from B2.5
        question_data = self.load_b25_output(question_id, doc_id)
        question_coords = question_data["question_coordinates"]
        question_balls = question_data["convex_ball_memberships"]

        # STEP 2: Load document chunks from A3 for same doc_id
        doc_chunks = self.load_a3_chunks(doc_id)

        # STEP 3: CONSTRAINT - Only consider chunks in shared convex balls
        eligible_chunks = self.filter_chunks_by_shared_balls(doc_chunks, question_balls)

        # STEP 4: Calculate geometric distances within constrained space
        geometric_matches = []
        for chunk in eligible_chunks:
            shared_balls = self.get_shared_convex_balls(question_balls, chunk.convex_ball_memberships)
            if shared_balls:
                # Calculate distance within shared convex ball
                distance = euclidean_distance(question_coords, chunk.coordinates)

                # Intent alignment within ball
                intent_vector = self.load_intent_vector(question_id)
                direction_vector = chunk.coordinates - question_coords
                intent_alignment = self.calculate_intent_alignment(intent_vector, direction_vector)

                # Angular analysis from shared centroids only
                concept_angles = self.calculate_concept_angles_constrained(
                    question_coords, chunk.coordinates, shared_balls
                )

                geometric_matches.append({
                    "chunk_id": chunk.chunk_id,
                    "geometric_distance": distance,
                    "intent_alignment": intent_alignment,
                    "concept_angles": concept_angles,
                    "shared_convex_balls": shared_balls,
                    "constraint_satisfied": True
                })

        return self.rank_by_geometric_proximity(geometric_matches)
```

#### Key Innovation Features
- **Convex Ball Constraint**: Only chunks within shared balls are considered
- **Document-Specific Matching**: All calculations within single document's concept space
- **Geometric Precision**: True Euclidean distances within mathematically defined constraints
- **Intent Weighting**: Intent vectors applied within constrained space

---

### B3.2 Semantic Similarity - MAJOR REDESIGN
**Current**: Global semantic similarity
**New**: Intra-ball semantic refinement

#### New Role Architecture
```python
def intra_ball_semantic_refinement(geometric_results, question_id):
    """Semantic scoring within geometrically constrained matches"""

    refined_results = []
    for result in geometric_results:
        if result["constraint_satisfied"]:  # Only process constrained matches

            # Semantic refinement within geometric constraints
            semantic_score = self.calculate_constrained_semantic_similarity(
                question_id, result["chunk_id"], result["shared_convex_balls"]
            )

            # Tie-breaking for geometrically similar chunks
            if result["geometric_distance"] < similarity_threshold:
                semantic_weight = 0.7  # Higher weight for tie-breaking
            else:
                semantic_weight = 0.3  # Lower weight for distinct geometric matches

            combined_score = (
                (1 - semantic_weight) * result["geometric_score"] +
                semantic_weight * semantic_score
            )

            refined_results.append({
                **result,
                "semantic_refinement_score": semantic_score,
                "combined_geometric_semantic": combined_score
            })

    return refined_results
```

---

### B3.3 Answer Capability Assessment - STRATEGIC REDESIGN
**Current**: General chunk capability scoring
**New**: Structured data extraction capability assessment

#### New Focus Architecture
```python
def assess_structured_extraction_capability(chunk, question, question_coords):
    """Assess capability for structured data extraction (table intersection)"""

    capabilities = {
        # Table intersection capability
        "table_intersection_ready": {
            "has_table_structure": self.detect_table_structure(chunk.content),
            "coordinate_extractable": self.can_extract_table_coordinates(chunk, question),
            "intersection_feasible": self.validate_intersection_possibility(chunk, question)
        },

        # Context integration capability
        "contextual_integration": {
            "denomination_context": self.has_unit_denomination_context(chunk),
            "scope_context": self.has_appropriate_scope_context(chunk, question),
            "temporal_alignment": self.matches_temporal_scope(chunk, question)
        },

        # Structured extraction capability
        "extraction_precision": {
            "exact_match_possible": self.can_extract_exact_value(chunk, question),
            "calculation_required": self.requires_calculation_step(chunk, question),
            "multi_chunk_dependency": self.requires_cross_chunk_context(chunk, question)
        }
    }

    # Calculate structured extraction score
    extraction_score = self.calculate_structured_extraction_score(capabilities)

    return {
        "chunk_id": chunk.chunk_id,
        "structured_extraction_score": extraction_score,
        "capability_breakdown": capabilities,
        "extraction_confidence": self.calculate_extraction_confidence(capabilities)
    }
```

---

### B4 Strategy Combination - ARCHITECTURAL REDESIGN
**Current**: Global weighted combination of strategy scores
**New**: Convex ball constrained strategy fusion

#### New Fusion Architecture
```python
class ConvexBallStrategyFusion:
    def combine_strategies_within_convex_balls(self, question_id, doc_id):
        """Strategy combination within convex ball constraints"""

        # All strategies operate within same convex ball constraints
        geometric_results = self.b31_constrained_geometric_matching(question_id, doc_id)
        semantic_results = self.b32_intra_ball_semantic_refinement(geometric_results)
        capability_results = self.b33_structured_extraction_assessment(semantic_results)

        # Load shared convex ball constraints
        shared_balls = self.load_shared_convex_balls(question_id, doc_id)

        # Constrained fusion within mathematical boundaries
        final_ranking = self.weighted_fusion_constrained(
            geometric_results=geometric_results,
            semantic_results=semantic_results,
            capability_results=capability_results,
            convex_ball_constraints=shared_balls,
            fusion_weights={
                "geometric_weight": 0.5,      # Primary within constraints
                "semantic_weight": 0.3,       # Refinement within constraints
                "capability_weight": 0.2      # Extraction feasibility
            }
        )

        return final_ranking
```

---

## Implementation Phase Strategy

### Phase 1: Core Infrastructure (Immediate Priority)
1. **B2.5 Complete Redesign**: Document-specific convex ball coordinate mapping
2. **B3.1 Constraint Implementation**: Convex ball filtering before geometric calculations
3. **A-Pipeline Integration**: Establish coordinate system alignment protocols

### Phase 2: Strategy Layer Integration (Next Sprint)
4. **B3.2 Intra-Ball Operations**: Semantic refinement within constraints
5. **B3.3 Structured Assessment**: Table intersection capability scoring
6. **B4 Constrained Fusion**: Strategy combination within mathematical boundaries

### Phase 3: Intelligence Enhancement (Future Optimization)
7. **B2.1-B2.4 Enhancements**: Enhanced intent/context/temporal alignment
8. **Performance Optimization**: Vectorization and caching within constraints
9. **Accuracy Tuning**: Parameter optimization for 75-80% target

---

## Critical Integration Points

### A-Pipeline Dependencies
```python
# B2.5 MUST load A-pipeline outputs
doc_concept_centroids = load_from_a_pipeline(
    doc_id=doc_id,
    outputs_required=[
        "concept_centroids",      # Centroid coordinates and radii
        "concept_space_dimensions", # n-dimensional space definition
        "chunk_coordinates"       # For alignment validation
    ]
)
```

### B-Pipeline Data Flow
```
B2.1 → B2.2 → B2.3 → B2.4 → B2.5 (Coordinate Assignment)
                                    ↓
B4 ← B3.3 ← B3.2 ← B3.1 (Constrained Matching)
```

### Cross-Module Coordination
- **Coordinate System Consistency**: All modules use identical n-dimensional space
- **Document Scope Enforcement**: All processing constrained to single doc_id
- **Convex Ball Constraints**: Mathematical boundaries enforced at every stage

---

## Expected Outcomes

### Accuracy Improvements
- **Precision Enhancement**: Matching within concept constraints eliminates irrelevant chunks
- **Human Alignment**: System mirrors human cognitive process step-by-step
- **Mathematical Rigor**: True geometric calculations within defined spaces
- **Target Achievement**: 75-80% accuracy through constrained matching

### System Benefits
- **Computational Efficiency**: Reduced search space through convex ball constraints
- **Interpretability**: Clear geometric reasoning for every match decision
- **Scalability**: Document-specific processing allows parallel scaling
- **Maintainability**: Clear separation of concerns between constraint and scoring logic

---

## Success Criteria

### Validation Requirements
1. **Geometric Consistency**: All distances satisfy triangle inequality within balls
2. **Constraint Compliance**: No matches occur outside convex ball boundaries
3. **Coordinate Alignment**: Question and chunk coordinates in identical space
4. **Human Process Mirroring**: Each step maps to identified human cognitive process

### Performance Benchmarks
- **Processing Speed**: <2 seconds for 20 questions with constraints
- **Memory Efficiency**: <500MB with coordinate caching
- **Accuracy Target**: 75-80% exact answer matching
- **Precision Improvement**: >90% of matches within shared convex balls

---

## Document Control
- **Type**: ARCHITECTURAL REDESIGN SPECIFICATION
- **Scope**: Complete B-Pipeline Convex Ball Architecture
- **Implementation Status**: Ready for Clean Rebuild
- **Dependencies**: A-Pipeline concept centroids, coordinate system alignment
- **Author**: System Architect
- **Review Status**: Ready for Implementation Phase 1

---

**NEXT SESSION START POINT**: Begin with Phase 1 implementation - B2.5 redesign and B3.1 constraint implementation, using this document as the complete architectural blueprint.