#!/usr/bin/env python3
"""
B3.1: Geometric Intent Matching
Implements true geometric distance and angular calculations in concept space
for intent-driven question-chunk matching.

Architecture: Based on B3.1_geometric_intent_matching.md
Technical Spec: Implementation follows B3.1_geometric_intent_techspec.md
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "concept_ball_threshold": 0.3,
    "max_chunks_to_process": 1000,
    "intent_weight_factor": 0.5,
    "distance_to_similarity_scale": 1.0,
    "max_output_chunks": 10,
    "enable_caching": True,
    "parallel_processing": False
}


class GeometricIntentMatcher:
    """
    Main class for geometric intent-based question-chunk matching in concept space
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the geometric matcher with configuration"""
        self.config = CONFIG.copy()
        if config:
            self.config.update(config)

        self.concept_dimension_map = {}
        self.concept_centroids = {}
        self.cache = {} if self.config["enable_caching"] else None

        logger.info("Geometric Intent Matcher initialized")

    def load_concept_space_dimensions(self, a3_chunks: List[Dict], b25_mappings: Dict) -> None:
        """
        Build concept space dimensions from all available concepts

        Args:
            a3_chunks: Chunks with concept memberships
            b25_mappings: Question concept mappings
        """
        all_concepts = set()

        # Collect concepts from chunks
        for chunk in a3_chunks:
            all_concepts.update(chunk.get("concept_memberships", []))

        # Collect concepts from question mappings
        if "fuzzy_memberships" in b25_mappings:
            all_concepts.update(b25_mappings["fuzzy_memberships"].keys())

        # Create dimension mapping
        self.concept_dimension_map = {
            concept: idx for idx, concept in enumerate(sorted(all_concepts))
        }

        # Initialize concept centroids (using importance scores when available)
        n_dims = len(self.concept_dimension_map)
        for concept, idx in self.concept_dimension_map.items():
            centroid = np.zeros(n_dims)
            centroid[idx] = 1.0  # Unit vector in concept's dimension
            self.concept_centroids[concept] = centroid

        logger.info(f"Concept space initialized with {n_dims} dimensions")

    def map_to_concept_space(self, entity_memberships: Dict[str, float]) -> np.ndarray:
        """
        Convert entity membership scores to coordinate vector in concept space

        Args:
            entity_memberships: Dict of concept_id -> membership_score

        Returns:
            np.ndarray: Coordinate vector in concept space
        """
        n_dims = len(self.concept_dimension_map)
        coords = np.zeros(n_dims)

        for concept_id, score in entity_memberships.items():
            if concept_id in self.concept_dimension_map:
                idx = self.concept_dimension_map[concept_id]
                # Clamp membership scores to [0, 1]
                coords[idx] = max(0.0, min(1.0, score))

        return coords

    def extract_intent_vector(self, b21_intent: Dict) -> np.ndarray:
        """
        Convert B2.1 intent analysis to geometric intent vector

        Args:
            b21_intent: Intent analysis from B2.1

        Returns:
            np.ndarray: Normalized intent direction vector
        """
        intent_types = ["comparison", "calculation", "definition", "identification", "factual"]
        intent_analysis = b21_intent.get("intent_analysis", {})
        intent_scores = intent_analysis.get("intent_scores", {})

        # Build intent vector from scores
        intent_vector = np.array([
            intent_scores.get(intent_type, 0.0) for intent_type in intent_types
        ])

        # Normalize to unit vector
        norm = np.linalg.norm(intent_vector)
        if norm > 0:
            intent_vector = intent_vector / norm
        else:
            # Default uniform intent if no specific intent found
            intent_vector = np.ones(len(intent_types)) / np.sqrt(len(intent_types))

        return intent_vector

    def calculate_geometric_distance(
        self,
        question_coords: np.ndarray,
        chunk_coords: np.ndarray,
        intent_vector: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calculate intent-weighted geometric distance between question and chunk

        Args:
            question_coords: Question coordinates in concept space
            chunk_coords: Chunk coordinates in concept space
            intent_vector: Normalized intent direction vector

        Returns:
            Tuple of (weighted_distance, intent_alignment)
        """
        # Calculate base Euclidean distance
        base_distance = np.linalg.norm(question_coords - chunk_coords)

        # Calculate chunk direction from question
        direction = chunk_coords - question_coords
        direction_norm = np.linalg.norm(direction)

        # Calculate intent alignment
        if direction_norm > 0:
            # Extend intent vector to match concept space dimensions if needed
            extended_intent = np.zeros(len(question_coords))
            extended_intent[:min(len(intent_vector), len(extended_intent))] = intent_vector[:min(len(intent_vector), len(extended_intent))]

            # Normalize direction
            direction_normalized = direction / direction_norm

            # Calculate alignment (cosine similarity)
            intent_alignment = np.dot(extended_intent, direction_normalized)
            intent_alignment = float(np.clip(intent_alignment, -1.0, 1.0))
        else:
            # Question and chunk at same position
            intent_alignment = 1.0

        # Apply intent weighting (closer when aligned with intent)
        weight_factor = self.config["intent_weight_factor"]
        intent_weight = weight_factor + (1 - weight_factor) * max(0, intent_alignment)

        # Weighted distance (lower when better aligned)
        weighted_distance = base_distance / intent_weight if intent_weight > 0 else base_distance

        return float(weighted_distance), float(intent_alignment)

    def calculate_concept_angles(
        self,
        question_coords: np.ndarray,
        chunk_coords: np.ndarray,
        primary_concepts: List[str]
    ) -> Dict[str, float]:
        """
        Calculate angles from multiple concept centroids

        Args:
            question_coords: Question coordinates
            chunk_coords: Chunk coordinates
            primary_concepts: List of primary concept IDs

        Returns:
            Dict of concept_id -> angle in radians
        """
        angles = {}

        for concept_id in primary_concepts:
            if concept_id not in self.concept_centroids:
                continue

            centroid = self.concept_centroids[concept_id]

            # Vectors from centroid to question and chunk
            q_vector = question_coords - centroid
            c_vector = chunk_coords - centroid

            # Calculate angle between vectors
            q_norm = np.linalg.norm(q_vector)
            c_norm = np.linalg.norm(c_vector)

            if q_norm > 0 and c_norm > 0:
                cos_angle = np.dot(q_vector, c_vector) / (q_norm * c_norm)
                # Clamp for numerical stability
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = float(np.arccos(cos_angle))
            else:
                angle = 0.0

            angles[f"{concept_id}_angle"] = angle

        return angles

    def filter_chunks_by_concept_balls(
        self,
        chunks: List[Dict],
        primary_concepts: List[str],
        threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Filter chunks to those within relevant concept balls

        Args:
            chunks: All available chunks
            primary_concepts: Primary concepts defining the balls
            threshold: Minimum membership score for inclusion

        Returns:
            List of chunks within the concept balls
        """
        if threshold is None:
            threshold = self.config["concept_ball_threshold"]

        filtered_chunks = []

        for chunk in chunks:
            chunk_memberships = chunk.get("concept_memberships", [])
            membership_scores = chunk.get("membership_scores", {})

            # Check if chunk belongs to any primary concept ball
            for concept in primary_concepts:
                if concept in chunk_memberships:
                    score = membership_scores.get(concept, 0.0)
                    if score >= threshold:
                        filtered_chunks.append(chunk)
                        break

        logger.debug(f"Filtered {len(chunks)} chunks to {len(filtered_chunks)} within concept balls")

        # If filtering removes all chunks, return original set
        if not filtered_chunks:
            logger.warning("Concept ball filtering removed all chunks, using full set")
            return chunks

        return filtered_chunks

    def distance_to_similarity(self, distance: float) -> float:
        """
        Convert geometric distance to similarity score

        Args:
            distance: Geometric distance

        Returns:
            Similarity score in [0, 1]
        """
        scale = self.config["distance_to_similarity_scale"]
        similarity = 1.0 / (1.0 + distance * scale)
        return float(similarity)

    def process_question(
        self,
        question_id: str,
        a3_chunks: List[Dict],
        b25_mappings: Dict,
        b21_intent: Dict
    ) -> Dict[str, Any]:
        """
        Main processing function for geometric intent matching

        Args:
            question_id: Question identifier
            a3_chunks: Chunks with concept memberships from A3
            b25_mappings: Question concept mappings from B2.5
            b21_intent: Intent analysis from B2.1

        Returns:
            Dict containing ranked chunks with geometric scores
        """
        start_time = time.perf_counter()

        try:
            # Initialize concept space dimensions
            self.load_concept_space_dimensions(a3_chunks, b25_mappings)

            # Extract question concept mappings
            question_fuzzy_memberships = b25_mappings.get("fuzzy_memberships", {})
            primary_concepts = b25_mappings.get("primary_concepts", [])

            # Build question membership scores dict
            question_memberships = {
                concept_id: data.get("membership_score", 0.0)
                for concept_id, data in question_fuzzy_memberships.items()
            }

            # Map question to concept space coordinates
            question_coords = self.map_to_concept_space(question_memberships)

            # Extract intent vector
            intent_vector = self.extract_intent_vector(b21_intent)

            # Filter chunks to relevant concept balls
            filtered_chunks = self.filter_chunks_by_concept_balls(
                a3_chunks, primary_concepts
            )

            # Process each chunk
            results = []
            for chunk in filtered_chunks[:self.config["max_chunks_to_process"]]:
                # Get chunk membership scores
                chunk_memberships = {}
                for concept in chunk.get("concept_memberships", []):
                    score = chunk.get("membership_scores", {}).get(concept, 1.0)
                    chunk_memberships[concept] = score

                # Map chunk to concept space
                chunk_coords = self.map_to_concept_space(chunk_memberships)

                # Calculate geometric distance with intent weighting
                distance, intent_alignment = self.calculate_geometric_distance(
                    question_coords, chunk_coords, intent_vector
                )

                # Calculate angles from concept centroids
                angles = self.calculate_concept_angles(
                    question_coords, chunk_coords, primary_concepts
                )

                # Convert distance to similarity
                similarity = self.distance_to_similarity(distance)

                # Store result
                results.append({
                    "chunk_id": chunk.get("chunk_id", ""),
                    "content": chunk.get("content", "")[:200] + "..." if len(chunk.get("content", "")) > 200 else chunk.get("content", ""),
                    "geometric_distance": round(distance, 4),
                    "intent_alignment": round(intent_alignment, 4),
                    "similarity_score": round(similarity, 4),
                    "concept_angles": {k: round(v, 4) for k, v in angles.items()},
                    "geometric_metadata": {
                        "question_coords": question_coords.tolist()[:5],  # Show first 5 dims
                        "chunk_coords": chunk_coords.tolist()[:5],  # Show first 5 dims
                        "intent_vector": intent_vector.tolist(),
                        "concept_dimensions": len(self.concept_dimension_map)
                    },
                    "doc_id": chunk.get("doc_id", ""),
                    "chunk_type": chunk.get("chunk_type", "")
                })

            # Sort by geometric distance (ascending - closer is better)
            results.sort(key=lambda x: x["geometric_distance"])

            # Limit to top N results
            results = results[:self.config["max_output_chunks"]]

            # Calculate processing time
            processing_time = (time.perf_counter() - start_time) * 1000  # Convert to ms

            return {
                "question_id": question_id,
                "strategy": "geometric_intent_matching",
                "ranked_chunks": results,
                "total_matches": len(results),
                "processing_metrics": {
                    "chunks_processed": len(filtered_chunks),
                    "chunks_in_concept_balls": len(filtered_chunks),
                    "total_available_chunks": len(a3_chunks),
                    "processing_time_ms": round(processing_time, 2),
                    "concept_space_dimensions": len(self.concept_dimension_map),
                    "primary_concepts_used": len(primary_concepts)
                },
                "quality_indicators": {
                    "avg_geometric_distance": round(np.mean([r["geometric_distance"] for r in results]), 4) if results else 0.0,
                    "avg_intent_alignment": round(np.mean([r["intent_alignment"] for r in results]), 4) if results else 0.0,
                    "best_similarity": round(results[0]["similarity_score"], 4) if results else 0.0
                },
                "processing_timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error processing question {question_id}: {e}")
            return {
                "question_id": question_id,
                "strategy": "geometric_intent_matching",
                "error": str(e),
                "ranked_chunks": [],
                "total_matches": 0,
                "processing_timestamp": datetime.now().isoformat()
            }


def load_inputs():
    """Load input files for B3.1 processing"""
    script_dir = Path(__file__).parent.parent

    # Load A3 chunks
    a3_path = script_dir.parent / "A_Concept_pipeline" / "outputs" / "A3_multi_strategy_chunks.json"
    if not a3_path.exists():
        raise FileNotFoundError(f"A3 output not found: {a3_path}")

    with open(a3_path, 'r', encoding='utf-8') as f:
        a3_data = json.load(f)

    # Extract chunks
    chunks = []
    if isinstance(a3_data, dict) and "chunks" in a3_data:
        chunks = a3_data["chunks"]
    elif isinstance(a3_data, list):
        chunks = a3_data

    # Load B2.5 concept mappings
    b25_path = script_dir / "outputs" / "B2.5_question_concept_mapping_output.json"
    if not b25_path.exists():
        raise FileNotFoundError(f"B2.5 output not found: {b25_path}")

    with open(b25_path, 'r', encoding='utf-8') as f:
        b25_data = json.load(f)

    # Load B2.1 intent analysis
    b21_path = script_dir / "outputs" / "B2.1_intent_layer_output.json"
    if not b21_path.exists():
        raise FileNotFoundError(f"B2.1 output not found: {b21_path}")

    with open(b21_path, 'r', encoding='utf-8') as f:
        b21_data = json.load(f)

    return chunks, b25_data, b21_data


def save_output(results: List[Dict], output_path: str = "outputs/B3.1_geometric_intent_output.json"):
    """Save processing results"""
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / output_path

    # Ensure output directory exists
    full_path.parent.mkdir(parents=True, exist_ok=True)

    # Save results
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {full_path}")


def main():
    """Main execution function"""
    print("=" * 80)
    print("B3.1: Geometric Intent Matching")
    print("Implementing true geometric distance calculations in concept space")
    print("=" * 80)

    try:
        # Load inputs
        print("\nLoading inputs from A3, B2.5, and B2.1...")
        chunks, b25_data, b21_data = load_inputs()

        print(f"Loaded {len(chunks)} chunks from A3")
        print(f"Loaded {len(b25_data.get('results', []))} question mappings from B2.5")
        print(f"Loaded {len(b21_data)} intent analyses from B2.1")

        # Initialize matcher
        matcher = GeometricIntentMatcher()

        # Process each question
        all_results = []
        questions_processed = 0

        for b25_result in b25_data.get("results", []):
            question_id = b25_result.get("question_id")
            if not question_id:
                continue

            # Find corresponding B2.1 intent analysis
            b21_intent = None
            for b21_item in b21_data:
                if b21_item.get("question_id") == question_id:
                    b21_intent = b21_item
                    break

            if not b21_intent:
                logger.warning(f"No B2.1 intent analysis found for {question_id}")
                continue

            # Extract B2.5 concept mappings for this question
            b25_mappings = b25_result.get("concept_mappings", {})

            print(f"\nProcessing {question_id}...")

            # Process question
            result = matcher.process_question(
                question_id=question_id,
                a3_chunks=chunks,
                b25_mappings=b25_mappings,
                b21_intent=b21_intent
            )

            all_results.append(result)
            questions_processed += 1

            # Display summary
            if result.get("ranked_chunks"):
                top_chunk = result["ranked_chunks"][0]
                print(f"  Top match: {top_chunk['chunk_id']}")
                print(f"  Geometric distance: {top_chunk['geometric_distance']:.3f}")
                print(f"  Intent alignment: {top_chunk['intent_alignment']:.3f}")
                print(f"  Similarity score: {top_chunk['similarity_score']:.3f}")

        # Save results
        save_output(all_results)

        # Print summary
        print("\n" + "=" * 80)
        print("PROCESSING SUMMARY")
        print("=" * 80)
        print(f"Questions processed: {questions_processed}")

        if all_results:
            avg_distance = np.mean([
                r["quality_indicators"]["avg_geometric_distance"]
                for r in all_results if "quality_indicators" in r
            ])
            avg_alignment = np.mean([
                r["quality_indicators"]["avg_intent_alignment"]
                for r in all_results if "quality_indicators" in r
            ])

            print(f"Average geometric distance: {avg_distance:.3f}")
            print(f"Average intent alignment: {avg_alignment:.3f}")

        print("\nB3.1 Geometric Intent Matching completed successfully!")

    except Exception as e:
        logger.error(f"Fatal error in B3.1 processing: {e}")
        raise


if __name__ == "__main__":
    main()