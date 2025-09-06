# A-Pipeline Migration Guide

## Component Mapping (Old → New)

### Active Components (Current Implementation)
| Old Reference | Current Location | Status | Notes |
|--------------|------------------|--------|-------|
| A1 | `scripts/` (various loaders) | ✅ Active | Document loading |
| A2.1 | `scripts/A2.1_concept_aware_preprocessing.py` | ✅ Active | Enhanced preprocessing |
| A2.2 | `scripts/A2.2_concept_identification.py` | ✅ Active | Concept extraction |
| A2.3 | `scripts/A2.3_intra_document_clustering.py` | ✅ Active | Document clustering |
| A2.4 | `scripts/A2.4_core_concept_identification.py` | ✅ Active | 10 core concepts |
| A2.5 | `scripts/A2.5_expanded_concepts_orchestrator.py` | ✅ Redesigned | Concept entity generation |
| A2.5.1-3 | `scripts/A2.5.[1-3]_*.py` | ✅ Redesigned | Generate new concepts |
| A2.5.4-5 | `scripts/A2.5.[4-5]_*.py` | ⚠️ Legacy | Need redesign |
| A3 | `scripts/A3_concept_based_chunking.py` | ✅ Active | 26-concept chunking |

### Deprecated/Archived Components
| Component | Original Purpose | Current Solution | Location |
|-----------|-----------------|------------------|----------|
| A2.6 | Relationship Builder | Integrated in A3 | Never implemented |
| A2.7 | Cross-Validator | Embedded validation | Never implemented |
| A2.8 | Semantic Chunking | Replaced by A3 | Never implemented |
| A2.9 | R4X Enhancement | Moved to I-Pipeline | `archived_scripts/` |

## Key Changes in Current Implementation

### A2.5: From Term Expansion to Concept Generation
**Old Approach**: Added keywords to existing 10 concepts
```python
# Old: Expanding terms within same concept
concept["expanded_terms"] = original_terms + new_terms
```

**New Approach**: Generates NEW concept entities
```python
# New: Creating new concept entities
new_concept = {
    "concept_id": "a251_core_1_semantic_neighbor",
    "canonical_name": "New Concept Entity",
    "primary_keywords": [...]
}
```

### A3: Enhanced Multi-Layered Chunking
**Capabilities**:
- Processes 26 concepts (10 A2.4 + 16 A2.5)
- Convex ball boundaries for each concept
- Overlapping membership scoring
- Multi-concept chunk detection

**Replaces**:
- A2.6 functionality (relationships via overlaps)
- A2.7 functionality (validation built-in)
- A2.8 functionality (semantic chunking)

## Finding Functionality

### If you're looking for...

**Concept Relationships** → Check A3
- `A3_concept_based_chunking.py`: `concept_memberships` and `convex_ball_memberships`

**Validation Logic** → Embedded in components
- A2.4: `validate_concepts()` method
- A2.5: Strategy-specific validation
- A3: `_validate_chunk()` method

**Semantic Chunking** → Use A3
- `A3_concept_based_chunking.py`: Full implementation with multi-layered approach

**R4X/Cross-Pipeline Integration** → I-Pipeline
- `I_InterSpace_pipeline/scripts/I1_cross_pipeline_semantic_integrator.py`
- `I_InterSpace_pipeline/scripts/I2_system_validation.py`
- `I_InterSpace_pipeline/scripts/I3_tri_semantic_visualizer.py`

## Output File Locations

### Current Outputs
```
A_Concept_pipeline/outputs/
├── A2.1_preprocessed_documents.json
├── A2.2_identified_concepts.json
├── A2.3_document_clusters.json
├── A2.4_core_concepts.json           # 10 core concepts
├── A2.5_expanded_concepts.json       # Orchestrator output (legacy format)
├── A2.5.1_semantic_expansion.json    # 6 new concepts
├── A2.5.2_domain_expansion.json      # 9 new concepts
├── A2.5.3_hierarchical_expansion.json # 1 new concept
├── A3_concept_based_chunks.json      # Main chunking output
├── A3_chunking_statistics.json       # Chunking metrics
└── A3_concept_chunks_summary.csv     # Summary table
```

### Visualization Outputs
```
A_Concept_pipeline/outputs/
├── A3_convex_ball_3d_visualization.html  # 3D concept space
└── A3_chunking_analysis_dashboard.html   # Analysis dashboard
```

## Running the Current Pipeline

### Full Pipeline Execution
```bash
# Run A2.1-A2.4 (core concept extraction)
cd A_Concept_pipeline/scripts
python A2.1_concept_aware_preprocessing.py
python A2.2_concept_identification.py
python A2.3_intra_document_clustering.py
python A2.4_core_concept_identification.py

# Run A2.5 strategies (concept generation)
python A2.5.1_semantic_similarity_expansion.py
python A2.5.2_domain_knowledge_expansion.py
python A2.5.3_hierarchical_clustering_expansion.py

# Run A3 (multi-layered chunking)
python A3_concept_based_chunking.py

# Generate visualizations
python ../A32_convex_ball_visualization.py
```

### Quick Test (A3 with existing concepts)
```bash
# Assumes A2.4 and A2.5 outputs exist
python A3_concept_based_chunking.py
```

## Common Issues and Solutions

### Issue: A3 only shows 10 concepts
**Solution**: Ensure A2.5.1-3 have been run to generate new concept entities

### Issue: Empty convex balls
**Expected**: New concept entities may not match document content exactly (exploration space)

### Issue: A2.5 orchestrator shows old format
**Known Issue**: Orchestrator needs update; A3 reads directly from strategy files

## Future Development

### Priority Tasks
1. Redesign A2.5.4 and A2.5.5 for concept generation
2. Update A2.5 orchestrator for new format
3. Optimize convex ball radii in A3
4. Target: 50+ total concept entities