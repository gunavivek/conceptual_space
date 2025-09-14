#!/usr/bin/env python3
"""
B2.5: Question-Concept Mapping
Implements fuzzy membership-based question-to-concept space mapping

Architecture: Based on B2.5_QUESTION_CONCEPT_MAPPING.md design principles
Technical Spec: Implementation follows B2.5_Question_Concept_Map_TechSpec.md
"""

import json
import math
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging

class PerformanceMonitor:
    """Track and log performance metrics"""

    def __init__(self):
        self.start_time = None
        self.checkpoints = {}

    def start_timing(self):
        """Start performance measurement"""
        self.start_time = time.perf_counter()

    def checkpoint(self, name: str):
        """Record timing checkpoint"""
        if self.start_time:
            elapsed = (time.perf_counter() - self.start_time) * 1000  # ms
            self.checkpoints[name] = elapsed

    def get_metrics(self) -> Dict[str, float]:
        """Get all performance metrics"""
        total_time = max(self.checkpoints.values()) if self.checkpoints else 0
        return {
            "total_processing_time_ms": total_time,
            **{f"{name}_time_ms": time for name, time in self.checkpoints.items()}
        }

class QuestionConceptMapper:
    """
    Main class implementing fuzzy membership-based question-to-concept mapping

    Architecture: Implements design decisions from B2.5_QUESTION_CONCEPT_MAPPING.md
    - Fuzzy set membership model for concept mapping
    - Multi-source feature fusion from B2.1-B2.4
    - Structured output for downstream B3.x consumption
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize mapper with configuration

        Args:
            config_path: Optional path to configuration file
        """
        self.config = self._load_configuration(config_path)
        self.concepts_lookup = {}
        self.expanded_concepts = {}
        self.processing_stats = {}
        self.performance_monitor = PerformanceMonitor()

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _load_configuration(self, config_path: Optional[str] = None) -> Dict:
        """
        Load and validate configuration

        Args:
            config_path: Optional custom config path

        Returns:
            dict: Validated configuration
        """
        # Default configuration aligned with technical specification
        default_config = {
            "fuzzy_parameters": {
                "importance_scaling": 1.0,
                "distance_weights": {
                    "keyword_overlap": 0.6,
                    "semantic_similarity": 0.3,
                    "answer_relevance": 0.25,
                    "temporal_alignment": 0.1
                },
                "confidence_thresholds": {
                    "high_confidence": 0.8,
                    "medium_confidence": 0.6,
                    "low_confidence": 0.4
                }
            },
            "membership_thresholds": {
                "strong_membership": 0.7,
                "medium_membership": 0.3,
                "weak_membership": 0.1,
                "max_concepts_output": 5,
                "min_concepts_output": 1
            },
            "processing": {
                "enable_temporal_context": True,
                "enable_declarative_elements": True,
                "cache_concept_data": True
            },
            "file_paths": {
                "b21_output": "B2.1_intent_layer_output.json",
                "b22_output": "B2.2_declarative_transformation_output.json",
                "b23_output": "B2.3_answer_expectation_output.json",
                "b24_output": "B2.4_temporal_analysis_output.json",
                "output_path": "B2.5_question_concept_mapping_output.json"
            },
            "performance": {
                "max_processing_time_ms": 50,
                "enable_performance_logging": True
            }
        }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                # Merge with defaults (simplified merge)
                return {**default_config, **user_config}
            except Exception as e:
                self.logger.warning(f"Config loading failed, using defaults: {e}")

        return default_config

    def load_concept_data(self) -> bool:
        """
        Load A2.4 and A2.5 concept definitions
        Both A2.4 and A2.5 are mandatory inputs

        Returns:
            bool: Success status of concept loading
        """
        try:
            script_dir = Path(__file__).parent.parent

            # Load A2.4 core concepts (mandatory)
            a24_path = script_dir.parent / "A_Concept_pipeline" / "outputs" / "A2.4_core_concepts.json"
            if not a24_path.exists():
                self.logger.error(f"A2.4 core concepts not found (REQUIRED): {a24_path}")
                return False

            with open(a24_path, 'r', encoding='utf-8') as f:
                a24_data = json.load(f)

            # Convert to lookup dict
            self.concepts_lookup = {}
            for concept in a24_data.get("core_concepts", []):
                concept_id = concept.get("concept_id")
                if concept_id:
                    self.concepts_lookup[concept_id] = {
                        "canonical_name": concept.get("canonical_name", ""),
                        "importance_score": concept.get("importance_score", 0.5),
                        "primary_keywords": concept.get("primary_keywords", []),
                        "keyword_frequencies": concept.get("keyword_frequencies", {}),
                        "document_count": concept.get("document_count", 0)
                    }

            self.logger.info(f"Loaded {len(self.concepts_lookup)} core concepts from A2.4")

            # Load A2.5 expanded concepts (mandatory)
            a25_path = script_dir.parent / "A_Concept_pipeline" / "outputs" / "A2.5_expanded_concepts.json"
            if not a25_path.exists():
                self.logger.error(f"A2.5 expanded concepts not found (REQUIRED): {a25_path}")
                return False

            with open(a25_path, 'r', encoding='utf-8') as f:
                a25_data = json.load(f)
            self.expanded_concepts = a25_data.get("expanded_concepts", {})
            self.logger.info(f"Loaded expanded concepts for {len(self.expanded_concepts)} core concepts from A2.5")

            # Validate that we have both concept sources (both are mandatory)
            if len(self.concepts_lookup) == 0:
                self.logger.error("No core concepts loaded from A2.4 - REQUIRED for operation")
                return False

            if len(self.expanded_concepts) == 0:
                self.logger.error("No expanded concepts loaded from A2.5 - REQUIRED for operation")
                return False

            self.performance_monitor.checkpoint("concept_loading")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load concept data: {e}")
            return False

    def _load_json_file(self, file_path: Path) -> Optional[Dict]:
        """Load and validate JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load {file_path}: {e}")
            return None

    def _find_question_data(self, data: Any, question_id: str) -> Optional[Dict]:
        """Find question data by ID in various data structures"""
        # Handle direct list format (B2.1, B2.2, B2.3 outputs)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get("question_id") == question_id:
                    return item

        # Handle single question format
        elif isinstance(data, dict):
            if data.get("question_id") == question_id:
                return data

            # Handle list of questions/results nested in dict (B2.4 output)
            if "results" in data and isinstance(data["results"], list):
                for item in data["results"]:
                    if isinstance(item, dict) and item.get("question_id") == question_id:
                        return item

        return None

    def _validate_inputs(self, question_id: str) -> Dict[str, Any]:
        """
        Validate and load all required and optional inputs

        Returns:
            dict: Validated input data with status flags
        """
        script_dir = Path(__file__).parent.parent
        outputs_dir = script_dir / "outputs"

        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "data": {}
        }

        # Required inputs
        required_files = {
            "b21": outputs_dir / self.config["file_paths"]["b21_output"],
            "b23": outputs_dir / self.config["file_paths"]["b23_output"]
        }

        for key, file_path in required_files.items():
            data = self._load_json_file(file_path)
            if data is None:
                validation_result["errors"].append(f"Required file not found or invalid: {file_path}")
                validation_result["valid"] = False
                continue

            question_data = self._find_question_data(data, question_id)
            if question_data is None:
                validation_result["errors"].append(f"Question {question_id} not found in {key}")
                validation_result["valid"] = False
            else:
                validation_result["data"][key] = question_data

        # Optional inputs
        optional_files = {
            "b22": outputs_dir / self.config["file_paths"]["b22_output"],
            "b24": outputs_dir / self.config["file_paths"]["b24_output"]
        }

        for key, file_path in optional_files.items():
            data = self._load_json_file(file_path)
            if data is not None:
                question_data = self._find_question_data(data, question_id)
                if question_data is not None:
                    validation_result["data"][key] = question_data
                else:
                    validation_result["warnings"].append(f"Question {question_id} not found in optional {key}")
            else:
                validation_result["warnings"].append(f"Optional file not available: {file_path}")

        return validation_result

    def aggregate_question_features(self, question_data: Dict) -> Dict[str, Any]:
        """
        Combine B2.1-B2.4 outputs into unified question representation

        Architecture: Implements multi-source feature fusion strategy

        Args:
            question_data: Dict containing all B2.x outputs

        Returns:
            dict: Aggregated features with weights and priorities
        """
        # Primary signals (required)
        b21_data = question_data.get("b21", {})
        b23_data = question_data.get("b23", {})

        # Enhancement signals (optional)
        b22_data = question_data.get("b22", {})
        b24_data = question_data.get("b24", {})

        # Extract primary keywords from B2.1 (using focus_terms)
        primary_keywords = b21_data.get("focus_terms", [])
        # Extract entities as additional keywords
        entities = b21_data.get("entities", [])
        entity_values = [e.get("value", "") for e in entities if isinstance(e, dict) and e.get("value")]
        primary_keywords.extend(entity_values)

        # No keyword weights in B2.1 output, so we'll use uniform weights
        keyword_weights = {kw: 1.0 for kw in primary_keywords}

        # Add declarative entities if available
        if b22_data and self.config["processing"]["enable_declarative_elements"]:
            # B2.2 has key_components with subjects, objects, modifiers
            key_components = b22_data.get("key_components", {})
            subjects = key_components.get("subjects", [])
            objects = key_components.get("objects", [])
            modifiers = key_components.get("modifiers", [])
            primary_keywords.extend(subjects + objects + modifiers)

        # Extract answer context
        answer_prediction = b23_data.get("answer_prediction", {})
        answer_type = answer_prediction.get("primary_type", "text")

        # Extract temporal context if available
        temporal_context = {}
        if b24_data and self.config["processing"]["enable_temporal_context"]:
            # B2.4 has temporal_entities and temporal_patterns
            temporal_entities = b24_data.get("temporal_entities", {})
            temporal_patterns = b24_data.get("temporal_patterns", {})
            temporal_context = {
                "entities": temporal_entities,
                "patterns": temporal_patterns,
                "confidence": b24_data.get("temporal_confidence", 0.5)
            }

        # Calculate feature importance weights
        feature_weights = self._calculate_feature_importance(b21_data, b23_data, b24_data)

        return {
            "primary_keywords": list(set(primary_keywords)),  # Remove duplicates
            "keyword_weights": keyword_weights,
            "answer_type": answer_type,
            "answer_context": answer_prediction,
            "temporal_context": temporal_context,
            "feature_weights": feature_weights,
            "input_completeness": {
                "b21_available": bool(b21_data),
                "b22_available": bool(b22_data),
                "b23_available": bool(b23_data),
                "b24_available": bool(b24_data)
            }
        }

    def _calculate_feature_importance(self, b21_data: Dict, b23_data: Dict, b24_data: Dict) -> Dict[str, float]:
        """Calculate dynamic feature importance weights"""
        weights = {
            "keyword_importance": 0.6,
            "answer_type_importance": 0.3,
            "temporal_importance": 0.1
        }

        # Boost temporal importance if temporal data available with high confidence
        if b24_data:
            temporal_confidence = b24_data.get("temporal_confidence", 0.5)
            if temporal_confidence > 0.7:
                weights["temporal_importance"] = 0.2
                weights["keyword_importance"] = 0.5

        # Boost answer type importance if high confidence
        answer_prediction = b23_data.get("answer_prediction", {})
        answer_confidence = answer_prediction.get("confidence", 0.5)
        if answer_confidence > 0.8:
            weights["answer_type_importance"] = 0.4
            weights["keyword_importance"] = 0.5

        return weights

    def _compute_keyword_distance(self, question_keywords: List[str],
                                concept_keywords: List[str],
                                concept_weights: Dict[str, int],
                                concept_id: str = None) -> float:
        """
        Calculate keyword-based distance between question and concept
        Enhanced with A2.5 expanded concepts when available

        Returns:
            float: Distance score (0.0 = perfect match, 1.0 = no match)
        """
        if not question_keywords or not concept_keywords:
            return 1.0

        # Normalize to lowercase
        question_set = set(kw.lower().strip() for kw in question_keywords if kw.strip())
        concept_set = set(kw.lower().strip() for kw in concept_keywords if kw.strip())

        # Enhance with A2.5 expanded concepts if available
        if concept_id and concept_id in self.expanded_concepts:
            expanded_data = self.expanded_concepts[concept_id]

            # Add semantic expansions
            semantic_expansions = expanded_data.get("semantic_expansions", [])
            concept_set.update(exp.lower().strip() for exp in semantic_expansions)

            # Add domain expansions
            domain_expansions = expanded_data.get("domain_expansions", [])
            concept_set.update(exp.lower().strip() for exp in domain_expansions)

        overlap = question_set.intersection(concept_set)

        if not overlap:
            return 1.0

        # Weight overlap by concept keyword frequencies
        weighted_overlap = sum(concept_weights.get(kw, 1) for kw in overlap)
        total_concept_weight = sum(concept_weights.values()) if concept_weights else len(concept_keywords)

        if total_concept_weight == 0:
            return 1.0

        overlap_ratio = weighted_overlap / total_concept_weight
        return 1.0 - min(overlap_ratio, 1.0)

    def _compute_semantic_distance(self, answer_type: str, concept_name: str) -> float:
        """
        Calculate semantic distance between expected answer type and concept

        Returns:
            float: Distance score (0.0 = highly relevant, 1.0 = irrelevant)
        """
        # Define answer type to concept relevance mappings
        type_concept_mappings = {
            "percentage": ["change", "ratio", "metrics", "performance", "growth", "rate"],
            "numeric": ["amount", "count", "value", "total", "sum", "number"],
            "date": ["time", "period", "schedule", "timeline", "when", "timing"],
            "text": ["description", "policy", "procedure", "definition", "explanation"]
        }

        relevant_concepts = type_concept_mappings.get(answer_type.lower(), [])
        concept_lower = concept_name.lower()

        # Check for direct relevance
        for relevant in relevant_concepts:
            if relevant in concept_lower:
                return 0.2  # High relevance

        # Check for partial relevance (financial context)
        if answer_type.lower() in ["percentage", "numeric"]:
            financial_terms = ["revenue", "cost", "income", "expense", "profit", "loss", "sales", "earnings"]
            if any(term in concept_lower for term in financial_terms):
                return 0.4  # Medium relevance

        return 0.8  # Low relevance

    def _compute_temporal_distance(self, temporal_context: Dict, concept_definition: Dict) -> float:
        """
        Calculate temporal alignment distance

        Returns:
            float: Distance score (0.0 = perfect alignment, 1.0 = no alignment)
        """
        if not temporal_context:
            return 0.5  # Neutral when no temporal context

        # Check if concept relates to temporal aspects
        concept_name = concept_definition.get("canonical_name", "").lower()
        concept_keywords = [kw.lower() for kw in concept_definition.get("primary_keywords", [])]

        temporal_indicators = ["time", "period", "annual", "quarterly", "monthly", "schedule", "timing",
                              "year", "date", "fiscal", "quarter"]

        has_temporal_aspect = any(indicator in concept_name for indicator in temporal_indicators) or \
                             any(indicator in " ".join(concept_keywords) for indicator in temporal_indicators)

        # Check if temporal entities exist in context
        temporal_entities = temporal_context.get("entities", {})
        has_specific_times = bool(temporal_entities.get("specific_times", []))
        has_periods = bool(temporal_entities.get("period_mentions", []))

        # High temporal relevance if both concept and question have temporal aspects
        if (has_specific_times or has_periods) and has_temporal_aspect:
            return 0.2  # Strong alignment
        elif has_temporal_aspect:
            return 0.4  # Moderate alignment
        else:
            return 0.6  # Weak alignment

    def calculate_fuzzy_membership(self, question_features: Dict, concept_definition: Dict, concept_id: str = None) -> Tuple[float, Dict]:
        """
        Implement fuzzy membership function: μ_concept(q) = exp(-d²/2σ²)

        Architecture Reference: Core Design Decision in B2.5_QUESTION_CONCEPT_MAPPING.md

        Args:
            question_features: Aggregated question representation
            concept_definition: A2.4 concept data
            concept_id: Concept identifier for A2.5 lookup

        Returns:
            tuple: (membership_score, calculation_details)
        """
        # Extract concept parameters
        sigma = concept_definition["importance_score"] * self.config["fuzzy_parameters"]["importance_scaling"]
        concept_keywords = concept_definition["primary_keywords"]
        concept_weights = concept_definition["keyword_frequencies"]

        # Calculate distance components with A2.5 enhancement
        keyword_distance = self._compute_keyword_distance(
            question_features["primary_keywords"],
            concept_keywords,
            concept_weights,
            concept_id
        )

        semantic_distance = self._compute_semantic_distance(
            question_features["answer_type"],
            concept_definition["canonical_name"]
        )

        temporal_distance = self._compute_temporal_distance(
            question_features.get("temporal_context", {}),
            concept_definition
        )

        # Weighted distance combination
        weights = self.config["fuzzy_parameters"]["distance_weights"]
        total_distance = (
            keyword_distance * weights["keyword_overlap"] +
            semantic_distance * weights["semantic_similarity"] +
            temporal_distance * weights.get("temporal_alignment", 0.1)
        )

        # Fuzzy membership calculation
        if sigma == 0:
            sigma = 0.5  # Prevent division by zero

        membership = math.exp(-(total_distance ** 2) / (2 * sigma ** 2))
        membership = min(1.0, max(0.0, membership))  # Clamp to [0,1]

        # Calculate individual similarities for details
        calculation_details = {
            "keyword_overlap": 1.0 - keyword_distance,
            "semantic_similarity": 1.0 - semantic_distance,
            "temporal_alignment": 1.0 - temporal_distance,
            "answer_type_relevance": 1.0 - semantic_distance,
            "distance_components": {
                "keyword_distance": keyword_distance,
                "semantic_distance": semantic_distance,
                "temporal_distance": temporal_distance,
                "weighted_distance": total_distance
            }
        }

        return membership, calculation_details

    def calculate_fuzzy_memberships(self, question_features: Dict, concepts: Dict) -> Dict[str, Any]:
        """
        Core fuzzy membership calculation implementing architectural design

        Args:
            question_features: Aggregated question features
            concepts: Loaded concept definitions

        Returns:
            dict: Fuzzy membership scores for all concepts
        """
        memberships = {}

        for concept_id, concept_def in concepts.items():
            membership_score, details = self.calculate_fuzzy_membership(question_features, concept_def, concept_id)

            # Calculate confidence based on input completeness and score
            confidence = self._calculate_membership_confidence(membership_score, question_features, details)

            memberships[concept_id] = {
                "membership_score": round(membership_score, 4),
                "confidence": round(confidence, 4),
                "concept_name": concept_def["canonical_name"],
                "calculation_details": {k: round(v, 4) if isinstance(v, float) else v for k, v in details.items()}
            }

        self.performance_monitor.checkpoint("fuzzy_calculation")
        return memberships

    def _calculate_membership_confidence(self, membership_score: float, question_features: Dict, details: Dict) -> float:
        """Calculate confidence in membership score"""
        base_confidence = membership_score

        # Boost confidence if multiple input sources available
        completeness = question_features["input_completeness"]
        available_inputs = sum(completeness.values())
        confidence_boost = min(0.2, available_inputs * 0.05)

        # Boost confidence if multiple distance components agree
        distances = details["distance_components"]
        agreement = 1.0 - abs(distances["keyword_distance"] - distances["semantic_distance"])
        confidence_boost += agreement * 0.1

        return min(1.0, base_confidence + confidence_boost)

    def rank_concept_memberships(self, memberships: Dict) -> List[Dict]:
        """
        Sort and filter concepts by membership strength

        Args:
            memberships: Raw fuzzy membership scores

        Returns:
            list: Ranked concepts with scores and metadata
        """
        # Filter by minimum threshold
        min_threshold = self.config["membership_thresholds"]["weak_membership"]
        filtered_memberships = {
            k: v for k, v in memberships.items()
            if v["membership_score"] >= min_threshold
        }

        # Sort by membership score
        ranked = sorted(
            filtered_memberships.items(),
            key=lambda x: x[1]["membership_score"],
            reverse=True
        )

        # Limit to max concepts
        max_concepts = self.config["membership_thresholds"]["max_concepts_output"]
        ranked = ranked[:max_concepts]

        # Format output
        ranked_concepts = []
        for rank, (concept_id, membership_data) in enumerate(ranked, 1):
            ranked_concepts.append({
                "concept_id": concept_id,
                "membership_score": membership_data["membership_score"],
                "rank": rank,
                "concept_name": membership_data["concept_name"],
                "confidence": membership_data["confidence"]
            })

        return ranked_concepts

    def _generate_concept_space_summary(self, memberships: Dict, ranked_concepts: List) -> Dict:
        """Generate summary statistics about concept space mapping"""
        thresholds = self.config["membership_thresholds"]

        membership_distribution = {
            "strong": len([m for m in memberships.values() if m["membership_score"] >= thresholds["strong_membership"]]),
            "medium": len([m for m in memberships.values() if thresholds["medium_membership"] <= m["membership_score"] < thresholds["strong_membership"]]),
            "weak": len([m for m in memberships.values() if thresholds["weak_membership"] <= m["membership_score"] < thresholds["medium_membership"]]),
            "filtered": len([m for m in memberships.values() if m["membership_score"] < thresholds["weak_membership"]])
        }

        scores = [m["membership_score"] for m in memberships.values()]
        avg_membership = sum(scores) / len(scores) if scores else 0

        return {
            "total_concepts_evaluated": len(memberships),
            "concepts_with_membership": len([m for m in memberships.values() if m["membership_score"] > 0]),
            "strong_memberships": membership_distribution["strong"],
            "average_membership": round(avg_membership, 4),
            "membership_distribution": membership_distribution
        }

    def process_question(self, question_id: str) -> Dict[str, Any]:
        """
        Complete question-to-concept mapping pipeline

        Args:
            question_id: Identifier for question to process

        Returns:
            dict: Complete mapping results
        """
        self.performance_monitor.start_timing()

        try:
            # Validate inputs
            input_validation = self._validate_inputs(question_id)
            if not input_validation["valid"]:
                # Try to get question text even in error case
                question_text = input_validation["data"].get("b21", {}).get("question", "Question text not available")
                return {
                    "error": "Input validation failed",
                    "details": input_validation["errors"],
                    "question_id": question_id,
                    "question": question_text,
                    "processing_timestamp": datetime.now().isoformat()
                }

            # Log warnings for missing optional inputs
            for warning in input_validation["warnings"]:
                self.logger.warning(warning)

            self.performance_monitor.checkpoint("input_validation")

            # Aggregate question features
            question_features = self.aggregate_question_features(input_validation["data"])
            self.performance_monitor.checkpoint("feature_aggregation")

            # Calculate fuzzy memberships
            fuzzy_memberships = self.calculate_fuzzy_memberships(question_features, self.concepts_lookup)

            # Rank concepts
            ranked_concepts = self.rank_concept_memberships(fuzzy_memberships)
            self.performance_monitor.checkpoint("concept_ranking")

            # Generate primary concepts list
            primary_concepts = [c["concept_id"] for c in ranked_concepts[:2]]  # Top 2

            # Generate summary
            concept_space_summary = self._generate_concept_space_summary(fuzzy_memberships, ranked_concepts)

            # Calculate overall mapping confidence
            mapping_confidence = self._calculate_overall_confidence(ranked_concepts, question_features)

            # Extract question text from B2.1 data for manual review
            question_text = input_validation["data"].get("b21", {}).get("question", "Question text not available")

            # Compile results
            result = {
                "question_id": question_id,
                "question": question_text,
                "processing_timestamp": datetime.now().isoformat(),
                "question_features": {
                    "primary_keywords": question_features["primary_keywords"],
                    "answer_type": question_features["answer_type"],
                    "temporal_context": question_features.get("temporal_context", {}),
                    "feature_weights": question_features["feature_weights"]
                },
                "concept_mappings": {
                    "fuzzy_memberships": fuzzy_memberships,
                    "ranked_concepts": ranked_concepts,
                    "primary_concepts": primary_concepts,
                    "concept_space_summary": concept_space_summary
                },
                "processing_metrics": self.performance_monitor.get_metrics(),
                "quality_indicators": {
                    "mapping_confidence": round(mapping_confidence, 4),
                    "concept_coverage": round(len(ranked_concepts) / max(len(self.concepts_lookup), 1), 4),
                    "input_completeness": question_features["input_completeness"]
                },
                "configuration_info": {
                    "core_concepts_count": len(self.concepts_lookup),
                    "expanded_concepts_count": len(self.expanded_concepts),
                    "concept_sources": "A2.4 + A2.5 (both mandatory)"
                }
            }

            return result

        except Exception as e:
            self.logger.error(f"Error processing question {question_id}: {e}")
            # Try to get question text for error reporting
            try:
                question_text = self._validate_inputs(question_id)["data"].get("b21", {}).get("question", "Question text not available")
            except:
                question_text = "Question text not available"

            return {
                "error": f"Processing failed: {str(e)}",
                "question_id": question_id,
                "question": question_text,
                "processing_timestamp": datetime.now().isoformat()
            }

    def _calculate_overall_confidence(self, ranked_concepts: List, question_features: Dict) -> float:
        """Calculate overall confidence in concept mapping"""
        if not ranked_concepts:
            return 0.0

        # Base confidence from top concept
        base_confidence = ranked_concepts[0].get("confidence", 0.0)

        # Boost if multiple strong concepts
        strong_concepts = len([c for c in ranked_concepts if c["membership_score"] >= 0.7])
        diversity_boost = min(0.2, strong_concepts * 0.1)

        # Boost for input completeness
        available_inputs = sum(question_features["input_completeness"].values())
        completeness_boost = min(0.2, available_inputs * 0.05)

        return min(1.0, base_confidence + diversity_boost + completeness_boost)

    def generate_output(self, results: Dict, output_path: Optional[str] = None) -> bool:
        """
        Create standardized output format and save to file

        Args:
            results: Processing results
            output_path: Optional custom output path

        Returns:
            bool: Success status
        """
        try:
            if output_path is None:
                script_dir = Path(__file__).parent.parent
                output_path = script_dir / "outputs" / self.config["file_paths"]["output_path"]
            else:
                output_path = Path(output_path)

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save results
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Results saved to: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save output: {e}")
            return False

def load_questions_from_b1():
    """Load questions from B1 output"""
    script_dir = Path(__file__).parent.parent
    b1_path = script_dir / "outputs" / "B1_current_question.json"

    if not b1_path.exists():
        print(f"B1 output not found: {b1_path}")
        return []

    try:
        with open(b1_path, 'r', encoding='utf-8') as f:
            b1_data = json.load(f)

        questions = []
        if isinstance(b1_data, list):
            questions = [q.get("question_id") for q in b1_data if q.get("question_id")]
        elif isinstance(b1_data, dict):
            if "questions" in b1_data:
                questions = [q.get("question_id") for q in b1_data["questions"] if q.get("question_id")]
            elif "question_id" in b1_data:
                questions = [b1_data["question_id"]]

        return questions
    except Exception as e:
        print(f"Error loading B1 questions: {e}")
        return []

def main():
    """Main execution function"""
    print("=" * 80)
    print("B2.5: Question-Concept Mapping - Fuzzy Membership Implementation")
    print("=" * 80)

    # Initialize mapper
    mapper = QuestionConceptMapper()

    # Display configuration
    print("Configuration: Both A2.4 and A2.5 are MANDATORY inputs")
    print("Expanded concepts: REQUIRED for enhanced mapping accuracy")

    # Load concept data
    print("\nLoading concept data from A-pipeline...")
    if not mapper.load_concept_data():
        print("ERROR: Failed to load concept data. Ensure A-pipeline has been executed.")
        return

    print(f"Successfully loaded {len(mapper.concepts_lookup)} core concepts (A2.4)")
    if mapper.expanded_concepts:
        print(f"Successfully loaded expanded concepts for {len(mapper.expanded_concepts)} concepts (A2.5)")
    else:
        print("ERROR: A2.5 expanded concepts not loaded - REQUIRED for operation!")
        return

    # Load questions from B1
    question_ids = load_questions_from_b1()
    if not question_ids:
        # Fallback: process a sample question if available
        print("No questions found in B1 output. Using sample question ID...")
        question_ids = ["finqa_test_1630"]  # Default sample

    print(f"Processing {len(question_ids)} questions...")

    # Process all questions
    all_results = []
    successful_processing = 0

    for question_id in question_ids:
        print(f"\nProcessing question: {question_id}")

        result = mapper.process_question(question_id)

        if "error" not in result:
            successful_processing += 1
            # Display summary
            concept_mappings = result["concept_mappings"]
            ranked_concepts = concept_mappings["ranked_concepts"]

            print(f"  Mapped to {len(ranked_concepts)} concepts")
            if ranked_concepts:
                top_concept = ranked_concepts[0]
                print(f"  Top concept: {top_concept['concept_name']} (score: {top_concept['membership_score']:.3f})")

            processing_time = result["processing_metrics"]["total_processing_time_ms"]
            print(f"  Processing time: {processing_time:.1f}ms")
        else:
            print(f"  ERROR: {result['error']}")

        all_results.append(result)

    # Save batch results
    batch_output = {
        "component": "B2.5_question_concept_mapping",
        "timestamp": datetime.now().isoformat(),
        "total_questions": len(question_ids),
        "successful_processing": successful_processing,
        "results": all_results,
        "configuration": mapper.config
    }

    # Save output
    success = mapper.generate_output(batch_output)

    print("\n" + "=" * 80)
    print("B2.5 PROCESSING SUMMARY")
    print("=" * 80)
    print(f"Questions processed: {len(question_ids)}")
    print(f"Successful: {successful_processing}")
    print(f"Failed: {len(question_ids) - successful_processing}")

    # Configuration summary
    print(f"\nCONCEPT CONFIGURATION:")
    print(f"A2.4 Core concepts: {len(mapper.concepts_lookup)}")
    print(f"A2.5 Expanded concepts: {len(mapper.expanded_concepts)} loaded")
    print(f"Configuration: Both A2.4 and A2.5 MANDATORY")

    if successful_processing > 0:
        avg_concepts_per_question = sum(
            len(r.get("concept_mappings", {}).get("ranked_concepts", []))
            for r in all_results if "error" not in r
        ) / successful_processing
        print(f"\nPERFORMANCE METRICS:")
        print(f"Average concepts per question: {avg_concepts_per_question:.1f}")

        # Show average confidence
        avg_confidence = sum(
            r.get("quality_indicators", {}).get("mapping_confidence", 0)
            for r in all_results if "error" not in r
        ) / successful_processing
        print(f"Average mapping confidence: {avg_confidence:.3f}")

    if success:
        print(f"\nOutput saved successfully!")
    else:
        print("WARNING: Output saving failed")

    print(f"\nB2.5 Question-Concept Mapping completed successfully!")

if __name__ == "__main__":
    main()