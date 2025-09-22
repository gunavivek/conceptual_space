"""
Inference Engine: Identifies and fills conceptual gaps and logical inference bridges
Implements gap detection, implicit concept discovery, and conceptual bridge filling
"""

import json
import re
from collections import defaultdict, Counter
from itertools import combinations
import math

class InferenceEngine:
    """Advanced inference engine for conceptual gap identification and filling"""

    def __init__(self, gap_threshold=0.3, bridge_confidence=0.6):
        """
        Initialize inference engine

        Args:
            gap_threshold: Threshold for identifying conceptual gaps
            bridge_confidence: Minimum confidence for inference bridges
        """
        self.gap_threshold = gap_threshold
        self.bridge_confidence = bridge_confidence

        # Logical relationship patterns
        self.logical_patterns = self._initialize_logical_patterns()

        # Inference rules for gap filling
        self.inference_rules = self._initialize_inference_rules()

        # Gap detection metrics
        self.gap_metrics = {}

    def _initialize_logical_patterns(self):
        """Initialize logical relationship patterns for inference"""
        return {
            "causal_patterns": [
                r"\b(\w+)\s+(causes?|leads?\s+to|results?\s+in|triggers?)\s+(\w+)\b",
                r"\b(\w+)\s+(because\s+of|due\s+to|owing\s+to)\s+(\w+)\b",
                r"\bif\s+(\w+)\s+then\s+(\w+)\b"
            ],
            "temporal_patterns": [
                r"\b(\w+)\s+(before|after|during|while)\s+(\w+)\b",
                r"\b(\w+)\s+(precedes|follows|occurs\s+with)\s+(\w+)\b"
            ],
            "conditional_patterns": [
                r"\bwhen\s+(\w+)\s+.*?\s+(\w+)\b",
                r"\bunless\s+(\w+)\s+.*?\s+(\w+)\b",
                r"\bprovided\s+(\w+)\s+.*?\s+(\w+)\b"
            ],
            "similarity_patterns": [
                r"\b(\w+)\s+(similar\s+to|like|resembles)\s+(\w+)\b",
                r"\b(\w+)\s+and\s+(\w+)\s+(both|share|have)\b"
            ]
        }

    def _initialize_inference_rules(self):
        """Initialize inference rules for gap filling"""
        return {
            "transitivity": {
                "is_a": {"rule": "if A is_a B and B is_a C, then A is_a C", "confidence": 0.9},
                "part_of": {"rule": "if A part_of B and B part_of C, then A part_of C", "confidence": 0.8},
                "causes": {"rule": "if A causes B and B causes C, then A may cause C", "confidence": 0.7}
            },
            "inheritance": {
                "properties": {"rule": "if A is_a B, then A inherits properties of B", "confidence": 0.8},
                "methods": {"rule": "if A is_a B, then A inherits methods of B", "confidence": 0.8}
            },
            "composition": {
                "aggregation": {"rule": "if A part_of B, then B contains A", "confidence": 0.9},
                "dependency": {"rule": "if A part_of B, then A depends on B", "confidence": 0.7}
            }
        }

    def detect_conceptual_gaps(self, concepts):
        """
        Detect conceptual gaps in the concept collection

        Args:
            concepts: List of concepts to analyze

        Returns:
            dict: Detected gaps with gap types and confidence scores
        """
        gaps = {
            "missing_intermediates": [],
            "missing_generalizations": [],
            "missing_specializations": [],
            "missing_relationships": [],
            "orphaned_concepts": []
        }

        # Build concept relationships
        concept_graph = self._build_concept_graph(concepts)

        # Detect missing intermediates
        gaps["missing_intermediates"] = self._detect_missing_intermediates(concept_graph)

        # Detect missing generalizations
        gaps["missing_generalizations"] = self._detect_missing_generalizations(concepts)

        # Detect missing specializations
        gaps["missing_specializations"] = self._detect_missing_specializations(concepts)

        # Detect missing relationships
        gaps["missing_relationships"] = self._detect_missing_relationships(concepts)

        # Detect orphaned concepts
        gaps["orphaned_concepts"] = self._detect_orphaned_concepts(concept_graph)

        return gaps

    def _build_concept_graph(self, concepts):
        """Build a graph representation of concept relationships"""
        graph = defaultdict(list)
        all_keywords = set()

        for concept in concepts:
            concept_id = concept.get("concept_id", "")
            keywords = concept.get("keywords", [])
            canonical_name = concept.get("canonical_name", "")

            all_keywords.update(keywords)
            all_keywords.add(canonical_name)

            # Find relationships based on keyword overlap
            for other_concept in concepts:
                if other_concept.get("concept_id") == concept_id:
                    continue

                other_keywords = other_concept.get("keywords", [])
                overlap = set(keywords) & set(other_keywords)

                if overlap:
                    overlap_ratio = len(overlap) / max(len(keywords), len(other_keywords))
                    if overlap_ratio > 0.3:  # Significant overlap
                        graph[concept_id].append({
                            "target": other_concept.get("concept_id"),
                            "overlap_ratio": overlap_ratio,
                            "shared_keywords": list(overlap)
                        })

        return graph

    def _detect_missing_intermediates(self, concept_graph):
        """Detect missing intermediate concepts between related concepts"""
        missing_intermediates = []

        for concept_id, connections in concept_graph.items():
            if len(connections) >= 2:
                # Check for potential missing intermediates
                for i, conn1 in enumerate(connections):
                    for conn2 in connections[i+1:]:
                        # If two concepts are related to this one but not to each other
                        if conn2["target"] not in [c["target"] for c in concept_graph.get(conn1["target"], [])]:
                            # Calculate potential intermediate concept
                            shared_keywords = set(conn1["shared_keywords"]) & set(conn2["shared_keywords"])
                            if shared_keywords:
                                missing_intermediates.append({
                                    "concept1": conn1["target"],
                                    "concept2": conn2["target"],
                                    "bridge_concept": concept_id,
                                    "potential_keywords": list(shared_keywords),
                                    "confidence": min(conn1["overlap_ratio"], conn2["overlap_ratio"])
                                })

        return missing_intermediates

    def _detect_missing_generalizations(self, concepts):
        """Detect missing generalization concepts"""
        missing_generalizations = []
        keyword_clusters = defaultdict(list)

        # Group concepts by shared keywords
        for concept in concepts:
            keywords = concept.get("keywords", [])
            for keyword in keywords:
                keyword_clusters[keyword].append(concept)

        # Find keywords shared by multiple concepts (potential generalizations)
        for keyword, concept_list in keyword_clusters.items():
            if len(concept_list) >= 3:  # Multiple concepts share this keyword
                # Check if there's a generalization concept for this keyword
                has_generalization = any(
                    keyword.lower() in concept.get("canonical_name", "").lower()
                    for concept in concept_list
                )

                if not has_generalization:
                    missing_generalizations.append({
                        "potential_generalization": keyword,
                        "specialized_concepts": [c.get("concept_id") for c in concept_list],
                        "concept_count": len(concept_list),
                        "confidence": min(0.9, len(concept_list) / 10)
                    })

        return missing_generalizations

    def _detect_missing_specializations(self, concepts):
        """Detect missing specialization concepts"""
        missing_specializations = []

        for concept in concepts:
            keywords = concept.get("keywords", [])
            canonical_name = concept.get("canonical_name", "")

            # Look for general terms that might need specializations
            general_indicators = ["system", "method", "process", "technique", "approach"]

            for indicator in general_indicators:
                if indicator in canonical_name.lower() or any(indicator in kw.lower() for kw in keywords):
                    # This is a general concept, check for specializations
                    specialization_count = sum(
                        1 for other_concept in concepts
                        if other_concept.get("concept_id") != concept.get("concept_id")
                        and any(kw in other_concept.get("keywords", []) for kw in keywords)
                    )

                    if specialization_count < 2:  # Few specializations found
                        missing_specializations.append({
                            "general_concept": concept.get("concept_id"),
                            "general_term": indicator,
                            "existing_specializations": specialization_count,
                            "confidence": 0.7 if specialization_count == 0 else 0.5
                        })

        return missing_specializations

    def _detect_missing_relationships(self, concepts):
        """Detect missing relationships between concepts"""
        missing_relationships = []

        # Analyze text content for implicit relationships
        for concept1, concept2 in combinations(concepts, 2):
            keywords1 = concept1.get("keywords", [])
            keywords2 = concept2.get("keywords", [])

            # Look for logical relationship patterns
            relationship_hints = self._find_relationship_patterns(keywords1, keywords2)

            if relationship_hints:
                missing_relationships.append({
                    "concept1": concept1.get("concept_id"),
                    "concept2": concept2.get("concept_id"),
                    "suggested_relationships": relationship_hints,
                    "confidence": max(hint["confidence"] for hint in relationship_hints)
                })

        return missing_relationships

    def _detect_orphaned_concepts(self, concept_graph):
        """Detect concepts with insufficient connections (orphaned concepts)"""
        orphaned_concepts = []

        for concept_id, connections in concept_graph.items():
            if len(connections) < 2:  # Poorly connected
                orphaned_concepts.append({
                    "concept_id": concept_id,
                    "connection_count": len(connections),
                    "isolation_score": 1.0 - (len(connections) / 5),  # Normalize to 0-1
                    "confidence": 0.8 if len(connections) == 0 else 0.6
                })

        return orphaned_concepts

    def _find_relationship_patterns(self, keywords1, keywords2):
        """Find potential relationship patterns between keyword sets"""
        relationship_hints = []
        all_text = " ".join(keywords1 + keywords2)

        for pattern_type, patterns in self.logical_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, all_text, re.IGNORECASE)
                for match in matches:
                    groups = match.groups()
                    if len(groups) >= 2:
                        relationship_hints.append({
                            "type": pattern_type.replace("_patterns", ""),
                            "source_term": groups[0],
                            "target_term": groups[-1],
                            "pattern": pattern,
                            "confidence": 0.7
                        })

        return relationship_hints

    def fill_conceptual_gaps(self, concepts, detected_gaps):
        """
        Fill conceptual gaps by generating bridge concepts

        Args:
            concepts: Original concept list
            detected_gaps: Detected gaps from gap detection

        Returns:
            dict: Generated bridge concepts and filled gaps
        """
        bridge_concepts = []
        filled_gaps = {
            "intermediate_bridges": [],
            "generalization_bridges": [],
            "relationship_bridges": []
        }

        # Fill missing intermediates
        for gap in detected_gaps["missing_intermediates"]:
            if gap["confidence"] >= self.bridge_confidence:
                bridge_concept = self._create_intermediate_bridge(gap, concepts)
                bridge_concepts.append(bridge_concept)
                filled_gaps["intermediate_bridges"].append(gap)

        # Fill missing generalizations
        for gap in detected_gaps["missing_generalizations"]:
            if gap["confidence"] >= self.bridge_confidence:
                bridge_concept = self._create_generalization_bridge(gap, concepts)
                bridge_concepts.append(bridge_concept)
                filled_gaps["generalization_bridges"].append(gap)

        # Fill missing relationships
        for gap in detected_gaps["missing_relationships"]:
            if gap["confidence"] >= self.bridge_confidence:
                relationship_bridge = self._create_relationship_bridge(gap, concepts)
                filled_gaps["relationship_bridges"].append(relationship_bridge)

        return {
            "bridge_concepts": bridge_concepts,
            "filled_gaps": filled_gaps,
            "gap_fill_statistics": {
                "total_gaps_detected": sum(len(gaps) for gaps in detected_gaps.values()),
                "gaps_filled": len(bridge_concepts) + len(filled_gaps["relationship_bridges"]),
                "fill_ratio": len(bridge_concepts) / max(sum(len(gaps) for gaps in detected_gaps.values()), 1)
            }
        }

    def _create_intermediate_bridge(self, gap, concepts):
        """Create an intermediate bridge concept"""
        return {
            "concept_id": f"bridge_intermediate_{len(gap['potential_keywords'])}",
            "canonical_name": f"Bridge: {' + '.join(gap['potential_keywords'])}",
            "keywords": gap["potential_keywords"],
            "concept_type": "inferred_intermediate",
            "confidence": gap["confidence"],
            "source_concepts": [gap["concept1"], gap["concept2"]],
            "bridge_type": "intermediate"
        }

    def _create_generalization_bridge(self, gap, concepts):
        """Create a generalization bridge concept"""
        return {
            "concept_id": f"bridge_general_{gap['potential_generalization']}",
            "canonical_name": f"General {gap['potential_generalization'].title()}",
            "keywords": [gap["potential_generalization"], "general", "category"],
            "concept_type": "inferred_generalization",
            "confidence": gap["confidence"],
            "specialized_concepts": gap["specialized_concepts"],
            "bridge_type": "generalization"
        }

    def _create_relationship_bridge(self, gap, concepts):
        """Create a relationship bridge between concepts"""
        return {
            "concept1": gap["concept1"],
            "concept2": gap["concept2"],
            "relationship_type": gap["suggested_relationships"][0]["type"],
            "confidence": gap["confidence"],
            "bridge_type": "relationship",
            "inference_rule": "pattern_based_inference"
        }

    def detect_concept_specific_gaps(self, concept, all_concepts):
        """
        Detect gaps specific to a single concept

        Args:
            concept: Target concept
            all_concepts: All available concepts

        Returns:
            dict: Concept-specific gaps
        """
        # Use existing gap detection but filter for this concept
        all_gaps = self.detect_conceptual_gaps(all_concepts)

        concept_id = concept.get("concept_id", "")
        concept_specific_gaps = {
            "missing_intermediates": [],
            "missing_relationships": [],
            "isolation_issues": []
        }

        # Filter gaps relevant to this concept
        for gap in all_gaps["missing_intermediates"]:
            if (gap.get("concept1") == concept_id or
                gap.get("concept2") == concept_id or
                gap.get("bridge_concept") == concept_id):
                concept_specific_gaps["missing_intermediates"].append(gap)

        for gap in all_gaps["missing_relationships"]:
            if (gap.get("concept1") == concept_id or
                gap.get("concept2") == concept_id):
                concept_specific_gaps["missing_relationships"].append(gap)

        for gap in all_gaps["orphaned_concepts"]:
            if gap.get("concept_id") == concept_id:
                concept_specific_gaps["isolation_issues"].append(gap)

        return concept_specific_gaps

    def fill_concept_gaps(self, concept, concept_gaps):
        """
        Fill gaps for a specific concept

        Args:
            concept: Target concept
            concept_gaps: Gaps specific to this concept

        Returns:
            dict: Filled gaps and bridge concepts
        """
        bridge_concepts = []
        filled_gaps = {
            "intermediate_bridges": [],
            "relationship_bridges": []
        }

        # Fill intermediate gaps
        for gap in concept_gaps["missing_intermediates"]:
            if gap.get("confidence", 0) >= self.bridge_confidence:
                bridge_concept = self._create_intermediate_bridge(gap, [concept])
                bridge_concepts.append(bridge_concept)
                filled_gaps["intermediate_bridges"].append(gap)

        # Fill relationship gaps
        for gap in concept_gaps["missing_relationships"]:
            if gap.get("confidence", 0) >= self.bridge_confidence:
                relationship_bridge = self._create_relationship_bridge(gap, [concept])
                filled_gaps["relationship_bridges"].append(relationship_bridge)

        return {
            "bridge_concepts": bridge_concepts,
            "filled_gaps": filled_gaps
        }

    def get_inference_info(self):
        """Get information about the inference engine"""
        return {
            "gap_threshold": self.gap_threshold,
            "bridge_confidence": self.bridge_confidence,
            "logical_patterns_count": sum(len(patterns) for patterns in self.logical_patterns.values()),
            "inference_rules_count": sum(len(rules) for rules in self.inference_rules.values())
        }