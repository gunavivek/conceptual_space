# A4: Geometric Concept Space Generation - Architecture Specification

## Core Purpose

A4 is the **geometric concept space generation module** that creates mathematically precise coordinate systems and convex ball definitions by combining A2.4 core concepts, A2.5 expanded concepts, and A3 chunk distributions using geometric drawing principles.

## Key Innovation: Definition-Based Geometric Embedding

A4 introduces revolutionary geometric concept space generation that creates authentic coordinate systems by embedding concept definitions (not just labels) and mapping chunks geometrically to ensure mathematical consistency with Q-Pipeline question mapping.

### Traditional vs. A4 Geometric Approach

#### Traditional Keyword-Based Approach
```python
traditional_concept_space = {
    "concept_representation": "keyword_frequency",  # Surface-level
    "coordinate_system": "arbitrary_dimensions",     # No geometric basis
    "chunk_assignment": "text_similarity",          # Semantic only
    "convex_balls": "estimated_parameters"          # No mathematical foundation
}
```

#### A4 Geometric Concept Space (Revolutionary)
```python
a4_geometric_concept_space = {
    "concept_representation": {
        "core_definition_embedding": "A2.4_canonical_definition",
        "expanded_definition_embedding": "A2.5_enriched_definition",
        "combined_semantic_vector": "definition_fusion_embedding"
    },
    "coordinate_system": {
        "embedding_model": "all-MiniLM-L6-v2",
        "dimensions": 384,
        "mathematical_basis": "semantic_vector_space",
        "consistency_guarantee": "identical_question_chunk_mapping"
    },
    "chunk_geometric_mapping": {
        "chunk_coordinates": "same_embedding_model_as_concepts",
        "distance_calculation": "euclidean_geometric_distance",
        "membership_determination": "convex_ball_containment"
    },
    "convex_ball_optimization": {
        "centroid_calculation": "concept_definition_embedding",
        "radius_optimization": "chunk_distribution_based",
        "geometric_properties": "volume_density_coverage_analysis"
    }
}
```

## Input/Output Specification

### Input Files
- **A2.4**: `A_Concept_pipeline/outputs/A2.4_core_concepts.json`
- **A2.5**: `A_Concept_pipeline/outputs/A2.5_expanded_concepts.json`
- **A3**: `A_Concept_pipeline/outputs/A3_multi_strategy_chunks.json`

### Output File
- **Target**: `A_Concept_pipeline/outputs/A4_geometric_concept_space.json`
- **Contains**: Complete geometric concept spaces with centroids, convex balls, and chunk memberships

### Output Structure
```python
{
    "document_id": {
        "geometric_concept_space": {
            "coordinate_system": {
                "dimensions": int,                    # Embedding model dimensions
                "embedding_model": str,               # Model used for consistency
                "space_type": "semantic_concept_embedding",
                "mathematical_properties": {
                    "metric": "euclidean",
                    "normalized": bool,
                    "dimension_scaling": "unit_sphere"
                }
            },
            "concept_centroids": {
                "concept_id": {
                    "concept_id": str,
                    "canonical_name": str,
                    "centroid_coordinates": List[float],    # Definition-based embedding
                    "definition_text": str,                 # Combined A2.4 + A2.5 definitions
                    "geometric_properties": {
                        "magnitude": float,
                        "unit_vector": List[float],
                        "embedding_confidence": float
                    },
                    "concept_metadata": {
                        "importance_score": float,          # From A2.4
                        "expansion_enrichment": float,      # From A2.5
                        "definition_completeness": float
                    }
                }
            },
            "convex_balls": {
                "concept_id": {
                    "centroid": List[float],               # Same as concept_centroids
                    "radius": float,                       # Optimized from chunk distribution
                    "member_chunks": [
                        {
                            "chunk_id": str,
                            "chunk_coordinates": List[float],      # Geometric position
                            "membership_strength": float,          # 1 - (distance/radius)
                            "distance_to_centroid": float,
                            "geometric_confidence": float,
                            "chunk_properties": {
                                "content_preview": str,
                                "chunk_type": str,
                                "semantic_alignment": float
                            }
                        }
                    ],
                    "geometric_properties": {
                        "volume": float,                   # n-dimensional ball volume
                        "chunk_density": float,            # chunks per unit volume
                        "coverage_completeness": float,    # % of concept space covered
                        "boundary_tightness": float,       # optimal radius efficiency
                        "dimensional_variance": List[float] # per-dimension spread
                    },
                    "optimization_metadata": {
                        "radius_calculation_method": str,
                        "outlier_chunks_excluded": int,
                        "geometric_quality_score": float
                    }
                }
            },
            "document_metadata": {
                "total_concepts": int,
                "total_chunks": int,
                "average_chunk_per_concept": float,
                "geometric_space_utilization": float,
                "coordinate_system_efficiency": float
            }
        }
    }
}
```

## Core Architecture Components

### 1. ConceptDefinitionEmbedder
```python
class ConceptDefinitionEmbedder:
    """Creates geometric embeddings from concept definitions"""

    def combine_concept_definitions(self, a24_concept: Dict, a25_expanded: Dict) -> str:
        """Combine A2.4 canonical + A2.5 expanded definitions"""

    def create_concept_embedding(self, combined_definition: str) -> np.ndarray:
        """Generate semantic embedding for concept definition"""

    def calculate_concept_centroid(self, concept_embedding: np.ndarray) -> np.ndarray:
        """Create geometric centroid from definition embedding"""
```

### 2. ChunkGeometricMapper
```python
class ChunkGeometricMapper:
    """Maps chunks to geometric concept space coordinates"""

    def map_chunk_to_coordinates(self, chunk_content: str) -> np.ndarray:
        """Map chunk using SAME embedding model as concepts"""

    def calculate_chunk_concept_distance(self, chunk_coords: np.ndarray,
                                       concept_centroid: np.ndarray) -> float:
        """Calculate geometric distance between chunk and concept"""

    def assign_chunk_to_concepts(self, chunk_coords: np.ndarray,
                               concept_centroids: Dict) -> Dict:
        """Determine concept memberships for chunk"""
```

### 3. ConvexBallOptimizer
```python
class ConvexBallOptimizer:
    """Optimizes convex ball parameters using geometric principles"""

    def calculate_optimal_radius(self, centroid: np.ndarray,
                               member_chunk_coords: List[np.ndarray]) -> float:
        """Calculate optimal radius from chunk distribution"""

    def analyze_geometric_properties(self, centroid: np.ndarray,
                                   radius: float, chunks: List) -> Dict:
        """Calculate volume, density, coverage properties"""

    def optimize_ball_parameters(self, concept_id: str, centroid: np.ndarray,
                               all_chunks: List) -> Dict:
        """Optimize convex ball for maximum geometric efficiency"""
```

### 4. A4GeometricConceptSpace
```python
class A4GeometricConceptSpace:
    """Main A4 geometric concept space generator"""

    def load_a_pipeline_inputs(self) -> Tuple[Dict, Dict, Dict]:
        """Load A2.4, A2.5, and A3 outputs"""

    def build_concept_embedding_space(self, a24_data: Dict, a25_data: Dict) -> Dict:
        """Create concept centroids from combined definitions"""

    def map_chunks_geometrically(self, a3_chunks: Dict, concept_centroids: Dict) -> Dict:
        """Map all chunks to geometric concept space"""

    def optimize_convex_balls(self, concept_centroids: Dict, chunk_mappings: Dict) -> Dict:
        """Create optimized convex balls with geometric properties"""

    def generate_complete_geometric_space(self, doc_id: str) -> Dict:
        """Generate complete A4 output for document"""
```

## Geometric Principles Implementation

### 1. Definition-Based Centroid Calculation
```python
def create_concept_centroid(self, a24_concept: Dict, a25_expanded: Dict) -> np.ndarray:
    # Combine definitions
    core_definition = a24_concept.get('canonical_name', '') + ' ' + \
                     a24_concept.get('concept_definition', {}).get('definition', '')

    expanded_terms = a25_expanded.get('all_expanded_terms', [])
    expanded_definition = ' '.join(expanded_terms)

    combined_definition = core_definition + ' ' + expanded_definition

    # Create embedding using same model as Q2.5 will use
    concept_embedding = self.semantic_model.encode(combined_definition)

    # Normalize to unit sphere for geometric consistency
    centroid = concept_embedding / np.linalg.norm(concept_embedding)

    return centroid
```

### 2. Optimal Radius Calculation
```python
def calculate_optimal_radius(self, centroid: np.ndarray, member_chunks: List) -> float:
    if not member_chunks:
        return 1.0  # Default radius

    # Calculate distances from centroid to all member chunks
    distances = [np.linalg.norm(chunk_coords - centroid)
                for chunk_coords in member_chunks]

    # Use 95th percentile to exclude outliers
    optimal_radius = np.percentile(distances, 95)

    # Ensure minimum radius for geometric stability
    return max(optimal_radius, 0.5)
```

### 3. Geometric Properties Analysis
```python
def analyze_geometric_properties(self, centroid: np.ndarray, radius: float,
                               chunks: List) -> Dict:
    # n-dimensional ball volume: V = π^(n/2) * r^n / Γ(n/2 + 1)
    n = len(centroid)
    volume = (np.pi ** (n/2)) * (radius ** n) / math.gamma(n/2 + 1)

    # Chunk density
    chunk_density = len(chunks) / volume if volume > 0 else 0

    # Coverage completeness
    distances = [np.linalg.norm(chunk['coordinates'] - centroid) for chunk in chunks]
    coverage_completeness = len([d for d in distances if d <= radius]) / len(distances)

    return {
        'volume': volume,
        'chunk_density': chunk_density,
        'coverage_completeness': coverage_completeness,
        'boundary_tightness': np.mean(distances) / radius if radius > 0 else 0
    }
```

## Integration with Q-Pipeline

### Coordinate System Consistency
```python
# A4 creates embeddings using model X
a4_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
concept_centroid = a4_embedding_model.encode(concept_definition)

# Q2.5 MUST use identical model for questions
q25_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # SAME MODEL
question_coordinates = q25_embedding_model.encode(question_text)

# Guaranteed coordinate system compatibility
distance = np.linalg.norm(question_coordinates - concept_centroid)  # Valid calculation
```

### Q2.5 Integration Points
```python
# Q2.5 loads A4 output directly
def load_document_concept_space(self, doc_id: str) -> Dict:
    a4_path = "A_Concept_pipeline/outputs/A4_geometric_concept_space.json"
    with open(a4_path, 'r') as f:
        a4_data = json.load(f)
    return a4_data[doc_id]['geometric_concept_space']

# Q2.5 uses A4 coordinate system
def calculate_question_coordinates(self, question_text: str, concept_space: Dict) -> np.ndarray:
    # Use same embedding model as A4
    model_name = concept_space['coordinate_system']['embedding_model']
    embedding_model = SentenceTransformer(model_name)
    return embedding_model.encode(question_text)
```

## Performance Targets

### Processing Metrics
- **A4 Generation Time**: <30 seconds per document with 50+ concepts
- **Memory Usage**: <1GB for geometric space generation
- **Coordinate Precision**: Float64 precision for geometric calculations
- **Embedding Consistency**: 100% identical coordinate systems A4↔Q2.5

### Quality Indicators
- **Convex Ball Coverage**: >85% chunks contained within optimized radius
- **Geometric Efficiency**: >70% space utilization without overlap
- **Definition Integration**: >90% successful A2.4+A2.5 combination
- **Q-Pipeline Compatibility**: 100% coordinate system consistency

## Success Criteria

### Functional Requirements
1. **Complete Geometric Space Generation**: All concepts from A2.4+A2.5 converted to coordinates
2. **Optimized Convex Ball Parameters**: Radius optimization based on chunk distributions
3. **Chunk Geometric Mapping**: All A3 chunks mapped to concept space coordinates
4. **Q-Pipeline Integration**: Compatible coordinate systems for Q2.5 consumption

### Quality Requirements
1. **Mathematical Precision**: Geometric properties calculated with mathematical rigor
2. **Definition-Based Embeddings**: Concept centroids based on semantic definitions, not labels
3. **Optimal Space Utilization**: Efficient convex ball parameters without excessive overlap
4. **Coordinate System Consistency**: Guaranteed compatibility between A4 and Q2.5

## Configuration Parameters

```python
a4_geometric_config = {
    "embedding_model": "all-MiniLM-L6-v2",
    "coordinate_normalization": "unit_sphere",
    "radius_calculation": {
        "method": "percentile_based",
        "percentile_threshold": 95,
        "minimum_radius": 0.5,
        "outlier_exclusion": True
    },
    "geometric_optimization": {
        "volume_calculation": "n_dimensional_ball",
        "density_analysis": True,
        "coverage_target": 0.85,
        "efficiency_threshold": 0.70
    },
    "definition_combination": {
        "a24_weight": 0.7,
        "a25_weight": 0.3,
        "max_definition_length": 1000,
        "semantic_deduplication": True
    }
}
```

## Summary

A4 Geometric Concept Space Generation is the **revolutionary geometric foundation module** that creates mathematically precise concept spaces by combining A2.4 core concepts, A2.5 expanded concepts, and A3 chunk distributions. By using definition-based semantic embeddings and geometric optimization principles, A4 ensures Q-Pipeline operates on authentic coordinate systems with optimal convex ball parameters.

**Key Innovation**: Definition-based geometric embedding that creates concept centroids from semantic meaning rather than keyword frequency, combined with chunk distribution analysis for optimal convex ball parameters, ensuring mathematical consistency between A-Pipeline chunk coordinates and Q-Pipeline question coordinates.

---
*A4 Geometric Concept Space Generation Architecture v1.0*
*Revolutionary Definition-Based Geometric Embedding*