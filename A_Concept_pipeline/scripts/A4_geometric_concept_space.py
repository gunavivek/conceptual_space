"""
A4: Geometric Concept Space Generation Module
Revolutionary definition-based geometric embedding with optimized convex balls
Creates mathematical foundation for Q-Pipeline geometric constraint satisfaction
"""

import json
import numpy as np
import os
import math
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from datetime import datetime
from collections import defaultdict


class ConceptDefinitionEmbedder:
    """Creates geometric embeddings from concept definitions"""

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with same embedding model as Q-Pipeline will use"""
        self.semantic_model = SentenceTransformer(embedding_model_name)
        self.embedding_model_name = embedding_model_name

    def create_concept_definition_text(self, concept: Dict, concept_source: str = "unknown") -> str:
        """Create definition text for independent A2.4 or A2.5 concepts using name + definition"""

        canonical_name = concept.get('canonical_name', '')
        concept_def = concept.get('concept_definition', {})

        # Extract definition text
        if isinstance(concept_def, dict):
            definition_text = concept_def.get('definition', '')
            synonyms = concept_def.get('synonyms', [])
            synonyms_text = ' '.join(synonyms[:3]) if synonyms else ""  # Top 3 synonyms
        else:
            definition_text = str(concept_def)
            synonyms_text = ""

        # Include primary keywords for additional context
        primary_keywords = concept.get('primary_keywords', [])
        keywords_text = ' '.join(primary_keywords[:5]) if primary_keywords else ""  # Top 5 keywords

        # Create comprehensive definition text: name + definition + synonyms + keywords
        concept_definition_text = f"{canonical_name} {definition_text} {synonyms_text} {keywords_text}".strip()

        # Limit length to prevent embedding issues
        if len(concept_definition_text) > 1000:
            concept_definition_text = concept_definition_text[:1000]

        print(f"A4: Created {concept_source} concept text for '{canonical_name}': {concept_definition_text[:100]}...")

        return concept_definition_text

    def create_concept_embedding(self, combined_definition: str) -> np.ndarray:
        """Generate semantic embedding for concept definition"""
        if not combined_definition.strip():
            # Fallback for empty definitions
            return np.random.randn(self.semantic_model.get_sentence_embedding_dimension()) * 0.1

        # Generate embedding
        concept_embedding = self.semantic_model.encode(combined_definition)

        return concept_embedding

    def calculate_concept_centroid(self, concept_embedding: np.ndarray) -> np.ndarray:
        """Create geometric centroid from definition embedding with normalization"""
        # Normalize to unit sphere for geometric consistency
        norm = np.linalg.norm(concept_embedding)
        if norm > 0:
            centroid = concept_embedding / norm
        else:
            # Fallback for zero vectors
            centroid = np.random.randn(len(concept_embedding))
            centroid = centroid / np.linalg.norm(centroid)

        return centroid


class ChunkGeometricMapper:
    """Maps chunks to geometric concept space coordinates"""

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with SAME embedding model as concepts for consistency"""
        self.semantic_model = SentenceTransformer(embedding_model_name)

    def map_chunk_to_coordinates(self, chunk_content: str) -> np.ndarray:
        """Map chunk using SAME embedding model as concepts"""
        if not chunk_content.strip():
            return np.zeros(self.semantic_model.get_sentence_embedding_dimension())

        # Limit chunk content length for embedding efficiency
        if len(chunk_content) > 2000:
            chunk_content = chunk_content[:2000]

        chunk_embedding = self.semantic_model.encode(chunk_content)

        # Normalize to unit sphere for consistency with concept centroids
        norm = np.linalg.norm(chunk_embedding)
        if norm > 0:
            chunk_coords = chunk_embedding / norm
        else:
            chunk_coords = np.zeros(len(chunk_embedding))

        return chunk_coords

    def calculate_chunk_concept_distance(self, chunk_coords: np.ndarray,
                                       concept_centroid: np.ndarray) -> float:
        """Calculate geometric distance between chunk and concept"""
        # Ensure same dimensionality
        min_dim = min(len(chunk_coords), len(concept_centroid))
        if min_dim == 0:
            return float('inf')

        chunk_vec = chunk_coords[:min_dim]
        concept_vec = concept_centroid[:min_dim]

        # Euclidean distance in normalized space
        distance = np.linalg.norm(chunk_vec - concept_vec)
        return float(distance)

    def assign_chunk_to_concepts(self, chunk_coords: np.ndarray,
                               concept_centroids: Dict) -> Dict:
        """Determine concept memberships for chunk based on geometric distance"""
        memberships = {}

        for concept_id, centroid in concept_centroids.items():
            distance = self.calculate_chunk_concept_distance(chunk_coords, centroid)
            memberships[concept_id] = {
                'distance': distance,
                'coordinates': chunk_coords.tolist()
            }

        return memberships


class ConvexBallOptimizer:
    """Optimizes convex ball parameters using geometric principles"""

    def calculate_optimal_radius(self, centroid: np.ndarray,
                               member_chunk_coords: List[np.ndarray],
                               percentile: float = 95.0) -> float:
        """Calculate optimal radius from chunk distribution using percentile method"""
        if not member_chunk_coords:
            return 1.0  # Default radius for concepts without chunks

        # Calculate distances from centroid to all member chunks
        distances = []
        for chunk_coords in member_chunk_coords:
            distance = np.linalg.norm(chunk_coords - centroid)
            distances.append(distance)

        if not distances:
            return 1.0

        # Use percentile to exclude outliers while covering most chunks
        optimal_radius = np.percentile(distances, percentile)

        # Ensure minimum radius for geometric stability
        return max(float(optimal_radius), 0.5)

    def analyze_geometric_properties(self, centroid: np.ndarray, radius: float,
                                   chunks: List[Dict]) -> Dict:
        """Calculate volume, density, coverage properties"""
        n = len(centroid)

        # n-dimensional ball volume: V = π^(n/2) * r^n / Γ(n/2 + 1)
        try:
            volume = (np.pi ** (n/2)) * (radius ** n) / math.gamma(n/2 + 1)
        except (OverflowError, ZeroDivisionError):
            volume = 0.0

        # Chunk density
        chunk_density = len(chunks) / volume if volume > 0 else 0

        # Coverage completeness - what percentage of chunks fit within radius
        if chunks:
            distances = []
            for chunk in chunks:
                chunk_coords = np.array(chunk.get('coordinates', []))
                if len(chunk_coords) > 0:
                    distance = np.linalg.norm(chunk_coords - centroid)
                    distances.append(distance)

            if distances:
                contained_chunks = len([d for d in distances if d <= radius])
                coverage_completeness = contained_chunks / len(distances)
                boundary_tightness = np.mean(distances) / radius if radius > 0 else 0
            else:
                coverage_completeness = 0.0
                boundary_tightness = 0.0
        else:
            coverage_completeness = 0.0
            boundary_tightness = 0.0

        return {
            'volume': float(volume),
            'chunk_density': float(chunk_density),
            'coverage_completeness': float(coverage_completeness),
            'boundary_tightness': float(boundary_tightness),
            'dimensional_variance': np.var(centroid).tolist() if len(centroid) > 0 else 0.0
        }

    def optimize_ball_parameters(self, concept_id: str, centroid: np.ndarray,
                               member_chunks: List[Dict]) -> Dict:
        """Optimize convex ball for maximum geometric efficiency"""

        # Extract chunk coordinates
        chunk_coordinates = []
        for chunk in member_chunks:
            coords = chunk.get('coordinates', [])
            if coords and len(coords) > 0:
                chunk_coordinates.append(np.array(coords))

        # Calculate optimal radius
        optimal_radius = self.calculate_optimal_radius(centroid, chunk_coordinates)

        # Create enhanced chunk memberships with geometric properties
        enhanced_chunks = []
        for chunk in member_chunks:
            chunk_coords = np.array(chunk.get('coordinates', []))
            if len(chunk_coords) > 0:
                distance = np.linalg.norm(chunk_coords - centroid)
                membership_strength = max(0.0, 1.0 - (distance / optimal_radius)) if optimal_radius > 0 else 0.0

                enhanced_chunk = {
                    'chunk_id': chunk.get('chunk_id', 'unknown'),
                    'chunk_coordinates': chunk_coords.tolist(),
                    'membership_strength': float(membership_strength),
                    'distance_to_centroid': float(distance),
                    'geometric_confidence': float(membership_strength),
                    'chunk_properties': {
                        'content_preview': chunk.get('content', '')[:100] + '...' if chunk.get('content') else '',
                        'chunk_type': chunk.get('chunk_type', 'unknown'),
                        'semantic_alignment': float(membership_strength)
                    }
                }
                enhanced_chunks.append(enhanced_chunk)

        # Analyze geometric properties
        geometric_properties = self.analyze_geometric_properties(centroid, optimal_radius, enhanced_chunks)

        # Calculate quality score
        geometric_quality_score = (
            geometric_properties['coverage_completeness'] * 0.4 +
            min(1.0, geometric_properties['chunk_density'] / 10.0) * 0.3 +
            (1.0 - geometric_properties['boundary_tightness']) * 0.3
        )

        return {
            'centroid': centroid.tolist(),
            'radius': float(optimal_radius),
            'member_chunks': enhanced_chunks,
            'geometric_properties': geometric_properties,
            'optimization_metadata': {
                'radius_calculation_method': 'percentile_95',
                'outlier_chunks_excluded': len(member_chunks) - len(enhanced_chunks),
                'geometric_quality_score': float(geometric_quality_score)
            }
        }


class A4GeometricConceptSpace:
    """Main A4 geometric concept space generator"""

    def __init__(self, a_pipeline_path: str = None):
        """Initialize A4 geometric concept space generator"""
        if a_pipeline_path is None:
            # Use absolute path to outputs directory
            script_dir = os.path.dirname(os.path.abspath(__file__))  # scripts directory
            a_pipeline_dir = os.path.dirname(script_dir)  # A_Concept_pipeline directory
            self.a_pipeline_path = os.path.join(a_pipeline_dir, "outputs")
        else:
            self.a_pipeline_path = a_pipeline_path
        self.embedding_model_name = "all-MiniLM-L6-v2"

        # Initialize components
        self.definition_embedder = ConceptDefinitionEmbedder(self.embedding_model_name)
        self.chunk_mapper = ChunkGeometricMapper(self.embedding_model_name)
        self.ball_optimizer = ConvexBallOptimizer()

    def load_a_pipeline_inputs(self) -> Tuple[Dict, Dict, Dict]:
        """Load A2.4, A2.5, and A3 outputs"""

        # Load A2.4 core concepts
        a24_path = os.path.join(self.a_pipeline_path, "A2.4_core_concepts.json")
        with open(a24_path, 'r', encoding='utf-8') as f:
            a24_data = json.load(f)

        # Load A2.5 expanded concepts
        a25_path = os.path.join(self.a_pipeline_path, "A2.5_expanded_concepts.json")
        with open(a25_path, 'r', encoding='utf-8') as f:
            a25_data = json.load(f)

        # Load A3 chunks
        a3_path = os.path.join(self.a_pipeline_path, "A3_multi_strategy_chunks.json")
        with open(a3_path, 'r', encoding='utf-8') as f:
            a3_data = json.load(f)

        return a24_data, a25_data, a3_data

    def build_concept_embedding_space(self, a24_data: Dict, a25_data: Dict) -> Tuple[Dict, Dict]:
        """Create concept centroids from A2.4 core concepts and A2.5 surrounding concepts independently
        Returns: (concept_centroids, old_to_new_id_mapping)
        """
        concept_centroids = {}
        old_to_new_id_mapping = {}  # Map old IDs to new readable IDs

        # Process A2.4 core concepts (document-central)
        if 'core_concepts' in a24_data:
            print(f"A4: Processing {len(a24_data['core_concepts'])} A2.4 core concepts...")
            for a24_concept in a24_data['core_concepts']:
                original_id = a24_concept.get('concept_id')
                if not original_id:
                    continue

                # Use readable concept name as ID (make it safe for use as ID)
                canonical_name = a24_concept.get('canonical_name', original_id)
                readable_id = canonical_name.lower().replace(' ', '_').replace('-', '_').replace('/', '_')

                # Check for duplicates and add source prefix if needed
                if readable_id in concept_centroids:
                    readable_id = f"a24_{readable_id}"

                # Store mapping from old ID to new readable ID
                old_to_new_id_mapping[original_id] = readable_id

                # Create definition text using name + definition + synonyms + keywords
                definition_text = self.definition_embedder.create_concept_definition_text(
                    a24_concept, "A2.4 core"
                )

                # Create embedding and centroid
                concept_embedding = self.definition_embedder.create_concept_embedding(definition_text)
                concept_centroid = self.definition_embedder.calculate_concept_centroid(concept_embedding)

                # Store centroid with A2.4 metadata
                concept_centroids[readable_id] = {
                    'concept_id': readable_id,
                    'original_id': original_id,
                    'canonical_name': canonical_name,
                    'centroid_coordinates': concept_centroid.tolist(),
                    'definition_text': definition_text,
                    'concept_source': 'A2.4_core',
                    'geometric_properties': {
                        'magnitude': float(np.linalg.norm(concept_embedding)),
                        'unit_vector': concept_centroid.tolist(),
                        'embedding_confidence': 1.0 if len(definition_text) > 10 else 0.5
                    },
                    'concept_metadata': {
                        'importance_score': a24_concept.get('importance_score', 0.5),
                        'document_count': a24_concept.get('document_count', 1),
                        'coverage_ratio': a24_concept.get('coverage_ratio', 0.1),
                        'definition_completeness': min(1.0, len(definition_text) / 500.0)
                    }
                }

        # Process A2.5 surrounding concepts (contextual)
        if 'expanded_concepts' in a25_data:
            print(f"A4: Processing {len(a25_data['expanded_concepts'])} A2.5 surrounding concepts...")
            for original_id, a25_concept_data in a25_data['expanded_concepts'].items():
                # Extract the original concept from A2.5 structure
                a25_concept = a25_concept_data.get('original_concept', {})
                if not a25_concept or not a25_concept.get('concept_id'):
                    continue

                # Use readable concept name as ID (make it safe for use as ID)
                canonical_name = a25_concept.get('canonical_name', original_id)
                readable_id = canonical_name.lower().replace(' ', '_').replace('-', '_').replace('/', '_')

                # Check for duplicates and add source prefix if needed
                if readable_id in concept_centroids:
                    readable_id = f"a25_{readable_id}"

                # Store mapping from old ID to new readable ID
                old_to_new_id_mapping[original_id] = readable_id

                # Create definition text using name + definition + synonyms + keywords
                definition_text = self.definition_embedder.create_concept_definition_text(
                    a25_concept, "A2.5 surrounding"
                )

                # Create embedding and centroid
                concept_embedding = self.definition_embedder.create_concept_embedding(definition_text)
                concept_centroid = self.definition_embedder.calculate_concept_centroid(concept_embedding)

                # Store centroid with A2.5 metadata
                concept_centroids[readable_id] = {
                    'concept_id': readable_id,
                    'original_id': original_id,
                    'canonical_name': canonical_name,
                    'centroid_coordinates': concept_centroid.tolist(),
                    'definition_text': definition_text,
                    'concept_source': 'A2.5_surrounding',
                    'geometric_properties': {
                        'magnitude': float(np.linalg.norm(concept_embedding)),
                        'unit_vector': concept_centroid.tolist(),
                        'embedding_confidence': 1.0 if len(definition_text) > 10 else 0.5
                    },
                    'concept_metadata': {
                        'importance_score': a25_concept.get('importance_score', 0.5),
                        'document_count': a25_concept.get('document_count', 1),
                        'coverage_ratio': a25_concept.get('coverage_ratio', 0.1),
                        'definition_completeness': min(1.0, len(definition_text) / 500.0)
                    }
                }

        print(f"A4: Created {len(concept_centroids)} total concept centroids from A2.4 + A2.5")
        return concept_centroids, old_to_new_id_mapping

    def map_chunks_geometrically(self, a3_chunks: Dict, concept_centroids: Dict, old_to_new_id_mapping: Dict = None) -> Dict:
        """Map all chunks to geometric concept space"""
        chunk_mappings = defaultdict(list)

        # Process each chunk
        for chunk in a3_chunks.get('chunks', []):
            chunk_id = chunk.get('chunk_id')
            chunk_content = chunk.get('content', '')
            doc_id = chunk.get('doc_id', '')

            if not chunk_content:
                continue

            # Map chunk to coordinates
            chunk_coords = self.chunk_mapper.map_chunk_to_coordinates(chunk_content)

            # Find concept memberships from existing A3 data
            existing_memberships = chunk.get('concept_memberships', [])

            # For each concept the chunk belongs to, add to mapping
            for old_concept_id in existing_memberships:
                # Translate old ID to new readable ID if mapping provided
                if old_to_new_id_mapping:
                    concept_id = old_to_new_id_mapping.get(old_concept_id, old_concept_id)
                else:
                    concept_id = old_concept_id

                if concept_id in concept_centroids:
                    chunk_data = {
                        'chunk_id': chunk_id,
                        'coordinates': chunk_coords.tolist(),
                        'content': chunk_content,
                        'chunk_type': chunk.get('chunk_type', 'unknown'),
                        'doc_id': doc_id,
                        'membership_score': chunk.get('membership_scores', {}).get(concept_id, 1.0)
                    }
                    chunk_mappings[concept_id].append(chunk_data)

        return dict(chunk_mappings)

    def optimize_convex_balls(self, concept_centroids: Dict, chunk_mappings: Dict) -> Dict:
        """Create optimized convex balls with geometric properties"""
        convex_balls = {}

        for concept_id, centroid_info in concept_centroids.items():
            centroid = np.array(centroid_info['centroid_coordinates'])
            member_chunks = chunk_mappings.get(concept_id, [])

            # Optimize convex ball parameters
            optimized_ball = self.ball_optimizer.optimize_ball_parameters(
                concept_id, centroid, member_chunks
            )

            convex_balls[concept_id] = optimized_ball

        return convex_balls

    def generate_complete_geometric_space(self, doc_id: str = None) -> Dict:
        """Generate complete A4 output"""

        print(f"A4: Generating geometric concept space...")
        start_time = datetime.now()

        # Load A-Pipeline inputs
        a24_data, a25_data, a3_data = self.load_a_pipeline_inputs()
        print(f"A4: Loaded A-Pipeline inputs - A2.4, A2.5, A3")

        # Build concept embedding space
        concept_centroids, old_to_new_id_mapping = self.build_concept_embedding_space(a24_data, a25_data)
        print(f"A4: Created {len(concept_centroids)} concept centroids from definitions")

        # Map chunks geometrically
        chunk_mappings = self.map_chunks_geometrically(a3_data, concept_centroids, old_to_new_id_mapping)
        total_mapped_chunks = sum(len(chunks) for chunks in chunk_mappings.values())
        print(f"A4: Mapped {total_mapped_chunks} chunks to geometric concept space")

        # Optimize convex balls
        convex_balls = self.optimize_convex_balls(concept_centroids, chunk_mappings)
        print(f"A4: Optimized {len(convex_balls)} convex balls with geometric properties")

        # Calculate document-level metrics
        total_chunks = len(a3_data.get('chunks', []))
        avg_chunks_per_concept = total_mapped_chunks / len(concept_centroids) if concept_centroids else 0

        # Create coordinate system info
        embedding_dimensions = self.definition_embedder.semantic_model.get_sentence_embedding_dimension()

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        # Build complete geometric space
        geometric_space = {
            'coordinate_system': {
                'dimensions': embedding_dimensions,
                'embedding_model': self.embedding_model_name,
                'space_type': 'semantic_concept_embedding',
                'mathematical_properties': {
                    'metric': 'euclidean',
                    'normalized': True,
                    'dimension_scaling': 'unit_sphere'
                }
            },
            'concept_centroids': concept_centroids,
            'convex_balls': convex_balls,
            'document_metadata': {
                'total_concepts': len(concept_centroids),
                'total_chunks': total_chunks,
                'total_mapped_chunks': total_mapped_chunks,
                'average_chunk_per_concept': float(avg_chunks_per_concept),
                'geometric_space_utilization': float(total_mapped_chunks / total_chunks) if total_chunks > 0 else 0.0,
                'coordinate_system_efficiency': 1.0,  # Perfect consistency by design
                'processing_time_ms': float(processing_time)
            }
        }

        return geometric_space

    def save_output(self, geometric_spaces: Dict, output_path: str = None):
        """Save A4 geometric concept spaces"""
        if output_path is None:
            output_path = os.path.join(self.a_pipeline_path, "A4_geometric_concept_space.json")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(geometric_spaces, f, indent=2, ensure_ascii=False)

        print(f"A4: Geometric concept spaces saved to {output_path}")
        return output_path


def process_all_documents():
    """Process all documents and generate geometric concept spaces"""

    print("="*70)
    print("A4: GEOMETRIC CONCEPT SPACE GENERATION")
    print("="*70)

    # Initialize A4 processor
    a4_processor = A4GeometricConceptSpace()

    try:
        # Generate geometric spaces (processes all documents from A-Pipeline data)
        geometric_spaces = {}

        # For now, create a general geometric space
        # In practice, this could be per-document if needed
        general_space = a4_processor.generate_complete_geometric_space()

        # Get document IDs from A3 data to organize by document
        _, _, a3_data = a4_processor.load_a_pipeline_inputs()
        document_ids = set()
        for chunk in a3_data.get('chunks', []):
            doc_id = chunk.get('doc_id', '')
            if doc_id:
                document_ids.add(doc_id)

        # Assign the geometric space to each document
        for doc_id in document_ids:
            geometric_spaces[doc_id] = {
                'geometric_concept_space': general_space
            }

        # Save output
        output_file = a4_processor.save_output(geometric_spaces)

        # Print summary
        print(f"\n{'='*50}")
        print("A4 GEOMETRIC CONCEPT SPACE SUMMARY:")
        print(f"{'='*50}")
        metadata = general_space['document_metadata']
        print(f"Documents processed: {len(document_ids)}")
        print(f"Total concepts: {metadata['total_concepts']}")
        print(f"Total chunks mapped: {metadata['total_mapped_chunks']}")
        print(f"Average chunks per concept: {metadata['average_chunk_per_concept']:.1f}")
        print(f"Space utilization: {metadata['geometric_space_utilization']:.1%}")
        print(f"Processing time: {metadata['processing_time_ms']:.1f}ms")
        print(f"Coordinate dimensions: {general_space['coordinate_system']['dimensions']}")
        print(f"Embedding model: {general_space['coordinate_system']['embedding_model']}")
        print(f"\nA4 geometric concept spaces ready for Q2.5 integration!")

    except Exception as e:
        print(f"Error in A4 processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    process_all_documents()