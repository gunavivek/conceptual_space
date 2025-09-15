#!/usr/bin/env python3
"""
Q2.4: Temporal Coordinate Mapping - Implementation Script

This module implements temporal intelligence for the Q-Pipeline, analyzing temporal patterns
in questions and mapping them to geometric coordinates in the concept space.

Key Features:
- Advanced temporal entity extraction (dates, quarters, relative references)
- Temporal relationship analysis (comparisons, sequences, durations)
- 8-dimensional geometric temporal coordinate mapping
- Temporal constraint generation for Q3.1 geometric matching
- Chronological structure analysis and complexity scoring

Author: Claude (Anthropic)
Date: 2025-09-14
Version: 1.0
"""

import json
import re
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import math

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class TemporalEntity:
    """Represents a detected temporal entity"""
    text: str
    entity_type: str
    start_pos: int
    end_pos: int
    normalized_value: Any
    confidence: float

@dataclass
class TemporalRelationship:
    """Represents a relationship between temporal entities"""
    relationship_type: str
    source_entity: TemporalEntity
    target_entity: TemporalEntity
    relationship_strength: float

class TemporalEntityExtractor:
    """Advanced temporal entity extraction and classification"""

    def __init__(self):
        self.temporal_patterns = {
            'absolute_years': [
                (r'\b(20[0-2]\d)\b', 'YEAR'),           # 2000-2029
                (r'\b(19[8-9]\d)\b', 'YEAR'),           # 1980-1999
            ],
            'quarters': [
                (r'\b([Qq][1-4])\s*(20[0-2]\d)\b', 'QUARTER'),  # Q1 2019
                (r'\b(20[0-2]\d)\s*([Qq][1-4])\b', 'QUARTER'),  # 2019 Q1
            ],
            'relative_time': [
                (r'\b(last|previous|prior)\s+(year|quarter|month)\b', 'RELATIVE_TIME'),
                (r'\b(next|following|subsequent)\s+(year|quarter|month)\b', 'RELATIVE_TIME'),
                (r'\b(current|this)\s+(year|quarter|month)\b', 'RELATIVE_TIME')
            ],
            'temporal_modifiers': [
                (r'\b(beginning|start|end|close)\s+of\b', 'TEMPORAL_MODIFIER'),
                (r'\b(during|throughout|within)\b', 'TEMPORAL_MODIFIER'),
                (r'\b(from|to|through|until)\b', 'TEMPORAL_MODIFIER')
            ],
            'comparison_patterns': [
                (r'\bfrom\s+(20[0-2]\d)\s+to\s+(20[0-2]\d)\b', 'TEMPORAL_COMPARISON'),
                (r'\b(20[0-2]\d)\s+vs\s+(20[0-2]\d)\b', 'TEMPORAL_COMPARISON'),
                (r'\bcompared?\s+to\s+(previous|last)\b', 'TEMPORAL_COMPARISON')
            ]
        }

    def extract_temporal_entities(self, question_text: str) -> Dict[str, List[TemporalEntity]]:
        """Extract comprehensive temporal entities from question text"""
        entities = {
            'extracted_dates': [],
            'temporal_expressions': [],
            'relative_time_references': [],
            'temporal_modifiers': [],
            'chronological_indicators': []
        }

        # Extract absolute years
        for pattern, entity_type in self.temporal_patterns['absolute_years']:
            matches = re.finditer(pattern, question_text, re.IGNORECASE)
            for match in matches:
                entity = TemporalEntity(
                    text=match.group(1),
                    entity_type=entity_type,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    normalized_value=int(match.group(1)),
                    confidence=0.95
                )
                entities['extracted_dates'].append(entity)

        # Extract quarters
        for pattern, entity_type in self.temporal_patterns['quarters']:
            matches = re.finditer(pattern, question_text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 2:
                    quarter_text = match.group(1) if match.group(1).lower().startswith('q') else match.group(2)
                    year_text = match.group(2) if match.group(1).lower().startswith('q') else match.group(1)

                    entity = TemporalEntity(
                        text=match.group(0),
                        entity_type=entity_type,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        normalized_value={'quarter': quarter_text, 'year': int(year_text)},
                        confidence=0.9
                    )
                    entities['temporal_expressions'].append(entity)

        # Extract relative time references
        for pattern, entity_type in self.temporal_patterns['relative_time']:
            matches = re.finditer(pattern, question_text, re.IGNORECASE)
            for match in matches:
                entity = TemporalEntity(
                    text=match.group(0),
                    entity_type=entity_type,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    normalized_value={'modifier': match.group(1), 'unit': match.group(2)},
                    confidence=0.8
                )
                entities['relative_time_references'].append(entity)

        # Extract temporal modifiers
        for pattern, entity_type in self.temporal_patterns['temporal_modifiers']:
            matches = re.finditer(pattern, question_text, re.IGNORECASE)
            for match in matches:
                entity = TemporalEntity(
                    text=match.group(0),
                    entity_type=entity_type,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    normalized_value=match.group(1) if len(match.groups()) > 0 else match.group(0),
                    confidence=0.75
                )
                entities['temporal_modifiers'].append(entity)

        # Extract chronological indicators
        chronological_keywords = ['change', 'from', 'to', 'between', 'compare', 'versus', 'vs']
        for keyword in chronological_keywords:
            pattern = rf'\b{keyword}\b'
            matches = re.finditer(pattern, question_text, re.IGNORECASE)
            for match in matches:
                entities['chronological_indicators'].append(match.group(0))

        return entities

class TemporalRelationshipAnalyzer:
    """Analyzes temporal relationships and sequences"""

    def analyze_temporal_relationships(self, temporal_entities: Dict, question_text: str) -> Dict:
        """Analyze complex temporal relationships"""

        # Identify comparison timeframes
        comparison_timeframes = self.identify_comparison_timeframes(temporal_entities)

        # Build temporal sequences
        temporal_sequences = self.build_temporal_sequences(temporal_entities)

        # Analyze duration specifications
        duration_specs = self.analyze_duration_specifications(temporal_entities)

        # Determine temporal granularity
        temporal_granularity = self.determine_temporal_granularity(temporal_entities)

        # Establish reference timeframe
        reference_timeframe = self.establish_reference_timeframe(temporal_entities)

        return {
            'comparison_timeframes': comparison_timeframes,
            'temporal_sequences': temporal_sequences,
            'duration_specifications': duration_specs,
            'temporal_granularity': temporal_granularity,
            'reference_time_frame': reference_timeframe
        }

    def identify_comparison_timeframes(self, temporal_entities: Dict) -> List[Dict]:
        """Identify temporal comparison patterns (e.g., "2018 to 2019")"""
        comparisons = []
        extracted_dates = temporal_entities.get('extracted_dates', [])

        # Look for "from X to Y" patterns
        if len(extracted_dates) >= 2:
            # Sort dates chronologically (handle both dict and object formats)
            if extracted_dates and isinstance(extracted_dates[0], dict):
                sorted_dates = sorted(extracted_dates, key=lambda x: x['normalized_value'])
            else:
                sorted_dates = sorted(extracted_dates, key=lambda x: x.normalized_value)

            for i in range(len(sorted_dates) - 1):
                start_date = sorted_dates[i]
                end_date = sorted_dates[i + 1]

                # Handle both dict and object formats
                if isinstance(start_date, dict):
                    start_value = start_date['normalized_value']
                    start_text = start_date['text']
                    start_confidence = start_date['confidence']
                    end_value = end_date['normalized_value']
                    end_text = end_date['text']
                    end_confidence = end_date['confidence']
                else:
                    start_value = start_date.normalized_value
                    start_text = start_date.text
                    start_confidence = start_date.confidence
                    end_value = end_date.normalized_value
                    end_text = end_date.text
                    end_confidence = end_date.confidence

                comparisons.append({
                    'comparison_type': 'temporal_range',
                    'start_timeframe': {
                        'value': start_value,
                        'text': start_text,
                        'confidence': start_confidence
                    },
                    'end_timeframe': {
                        'value': end_value,
                        'text': end_text,
                        'confidence': end_confidence
                    },
                    'comparison_direction': 'chronological_progression',
                    'geometric_constraint_type': 'temporal_span',
                    'span_years': end_value - start_value
                })

        return comparisons

    def build_temporal_sequences(self, temporal_entities: Dict) -> List[Dict]:
        """Build temporal sequences from detected entities"""
        sequences = []

        # Combine all temporal entities with temporal ordering
        all_temporal = []

        # Add extracted dates
        for entity in temporal_entities.get('extracted_dates', []):
            # Handle both dict and object formats
            if isinstance(entity, dict):
                temporal_order = entity['normalized_value']
            else:
                temporal_order = entity.normalized_value

            all_temporal.append({
                'entity': entity,
                'temporal_order': temporal_order,
                'sequence_type': 'absolute_date'
            })

        # Add temporal expressions (quarters)
        for entity in temporal_entities.get('temporal_expressions', []):
            # Handle both dict and object formats
            if isinstance(entity, dict):
                normalized_value = entity['normalized_value']
            else:
                normalized_value = entity.normalized_value

            if isinstance(normalized_value, dict) and 'year' in normalized_value:
                year = normalized_value['year']
                quarter = normalized_value.get('quarter', 'Q1')
                # Convert quarter to fractional year for ordering
                quarter_offset = {'Q1': 0.0, 'Q2': 0.25, 'Q3': 0.5, 'Q4': 0.75}.get(quarter.upper(), 0.0)
                temporal_order = year + quarter_offset

                all_temporal.append({
                    'entity': entity,
                    'temporal_order': temporal_order,
                    'sequence_type': 'quarterly_date'
                })

        # Sort by temporal order
        if all_temporal:
            sorted_temporal = sorted(all_temporal, key=lambda x: x['temporal_order'])

            sequences.append({
                'sequence_id': 'primary_chronological',
                'sequence_length': len(sorted_temporal),
                'sequence_span': sorted_temporal[-1]['temporal_order'] - sorted_temporal[0]['temporal_order'],
                'sequence_entities': sorted_temporal
            })

        return sequences

    def analyze_duration_specifications(self, temporal_entities: Dict) -> List[Dict]:
        """Analyze duration specifications in temporal references"""
        durations = []

        comparison_timeframes = self.identify_comparison_timeframes(temporal_entities)
        for comparison in comparison_timeframes:
            if 'span_years' in comparison:
                durations.append({
                    'duration_type': 'year_span',
                    'duration_value': comparison['span_years'],
                    'duration_unit': 'years',
                    'source_comparison': comparison
                })

        return durations

    def determine_temporal_granularity(self, temporal_entities: Dict) -> str:
        """Determine the temporal granularity of the question"""
        # Check for quarters
        if temporal_entities.get('temporal_expressions'):
            for expr in temporal_entities['temporal_expressions']:
                # Handle both dict and object formats
                if isinstance(expr, dict):
                    normalized_value = expr['normalized_value']
                else:
                    normalized_value = expr.normalized_value

                if isinstance(normalized_value, dict) and 'quarter' in normalized_value:
                    return 'quarter'

        # Check for years
        if temporal_entities.get('extracted_dates'):
            return 'year'

        # Check for relative time units
        for ref in temporal_entities.get('relative_time_references', []):
            # Handle both dict and object formats
            if isinstance(ref, dict):
                normalized_value = ref['normalized_value']
            else:
                normalized_value = ref.normalized_value

            if isinstance(normalized_value, dict) and 'unit' in normalized_value:
                return normalized_value['unit']

        return 'year'  # Default

    def establish_reference_timeframe(self, temporal_entities: Dict) -> Dict:
        """Establish the primary reference timeframe"""
        # Find the most recent or most prominent temporal reference
        all_dates = temporal_entities.get('extracted_dates', [])
        if all_dates:
            # Use the first mentioned date as primary reference
            primary_date = all_dates[0]

            # Handle both dict and object formats
            if isinstance(primary_date, dict):
                return {
                    'reference_type': 'absolute_date',
                    'reference_value': primary_date['normalized_value'],
                    'reference_text': primary_date['text'],
                    'confidence': primary_date['confidence']
                }
            else:
                return {
                    'reference_type': 'absolute_date',
                    'reference_value': primary_date.normalized_value,
                    'reference_text': primary_date.text,
                    'confidence': primary_date.confidence
                }

        # Fall back to relative references
        relative_refs = temporal_entities.get('relative_time_references', [])
        if relative_refs:
            primary_ref = relative_refs[0]

            # Handle both dict and object formats
            if isinstance(primary_ref, dict):
                return {
                    'reference_type': 'relative_time',
                    'reference_value': primary_ref['normalized_value'],
                    'reference_text': primary_ref['text'],
                    'confidence': primary_ref['confidence']
                }
            else:
                return {
                    'reference_type': 'relative_time',
                    'reference_value': primary_ref.normalized_value,
                    'reference_text': primary_ref.text,
                    'confidence': primary_ref.confidence
                }

        return {
            'reference_type': 'none',
            'reference_value': None,
            'reference_text': '',
            'confidence': 0.0
        }

class GeometricTemporalCoordinator:
    """Maps temporal analysis to geometric coordinates"""

    def __init__(self):
        self.temporal_coordinate_dimensions = 8
        self.chronological_reference_epoch = 2000

    def map_temporal_to_geometric(self, temporal_analysis: Dict) -> Dict:
        """Map temporal analysis to geometric coordinate vectors"""

        # Generate primary temporal vector
        primary_vector = self.generate_primary_temporal_vector(temporal_analysis)

        # Generate comparative temporal vectors
        comparative_vectors = self.generate_comparative_temporal_vectors(temporal_analysis)

        # Generate constraint boundaries
        constraint_boundaries = self.generate_temporal_constraint_boundaries(temporal_analysis)

        # Generate chronological ordering vectors
        ordering_vectors = self.generate_chronological_ordering_vectors(temporal_analysis)

        # Calculate precision weights
        precision_weights = self.calculate_temporal_precision_weights(temporal_analysis)

        return {
            'primary_temporal_vector': primary_vector,
            'comparative_temporal_vectors': comparative_vectors,
            'temporal_constraint_boundaries': constraint_boundaries,
            'chronological_ordering_vectors': ordering_vectors,
            'temporal_precision_weights': precision_weights
        }

    def generate_primary_temporal_vector(self, temporal_analysis: Dict) -> List[float]:
        """Generate primary 8-dimensional temporal coordinate vector"""
        vector = [0.0] * self.temporal_coordinate_dimensions

        # Dimension 0-1: Primary timeframe coordinates
        primary_dates = temporal_analysis['temporal_entities'].get('extracted_dates', [])
        if primary_dates:
            # Handle both dict and object formats
            if isinstance(primary_dates[0], dict):
                primary_year = primary_dates[0]['normalized_value']
            else:
                primary_year = primary_dates[0].normalized_value

            # Normalize years to [0, 1] range relative to reference epoch
            normalized_year = (primary_year - self.chronological_reference_epoch) / 50.0
            vector[0] = max(0.0, min(1.0, normalized_year))

            # Secondary temporal coordinate (for multi-year contexts)
            if len(primary_dates) > 1:
                if isinstance(primary_dates[1], dict):
                    secondary_year = primary_dates[1]['normalized_value']
                else:
                    secondary_year = primary_dates[1].normalized_value

                normalized_secondary = (secondary_year - self.chronological_reference_epoch) / 50.0
                vector[1] = max(0.0, min(1.0, normalized_secondary))

        # Dimension 2-3: Temporal comparison coordinates
        comparison_timeframes = temporal_analysis['temporal_relationships'].get('comparison_timeframes', [])
        if comparison_timeframes:
            comparison = comparison_timeframes[0]
            start_year = comparison['start_timeframe']['value']
            end_year = comparison['end_timeframe']['value']

            # Encode temporal span
            span_years = end_year - start_year
            normalized_span = min(abs(span_years) / 10.0, 1.0)  # 10-year max span
            vector[2] = normalized_span

            # Encode temporal direction (forward/backward in time)
            vector[3] = 1.0 if span_years > 0 else 0.0

        # Dimension 4: Temporal granularity
        granularity = temporal_analysis['temporal_relationships'].get('temporal_granularity', 'year')
        granularity_map = {'day': 1.0, 'month': 0.75, 'quarter': 0.5, 'year': 0.25, 'multi_year': 0.0}
        vector[4] = granularity_map.get(granularity, 0.25)

        # Dimension 5: Temporal complexity
        complexity = temporal_analysis['chronological_structure'].get('temporal_complexity', 0.0)
        vector[5] = min(complexity, 1.0)

        # Dimension 6: Temporal precision
        precision = temporal_analysis['chronological_structure'].get('time_reference_precision', 0.5)
        vector[6] = precision

        # Dimension 7: Temporal disambiguation
        disambiguation = temporal_analysis['chronological_structure'].get('temporal_disambiguation', 0.5)
        vector[7] = disambiguation

        return vector

    def generate_comparative_temporal_vectors(self, temporal_analysis: Dict) -> List[List[float]]:
        """Generate vectors for temporal comparisons"""
        comparative_vectors = []

        comparison_timeframes = temporal_analysis['temporal_relationships'].get('comparison_timeframes', [])
        for comparison in comparison_timeframes:
            vector = [0.0] * 4  # 4D comparative vector

            # Start timeframe coordinates
            start_year = comparison['start_timeframe']['value']
            normalized_start = (start_year - self.chronological_reference_epoch) / 50.0
            vector[0] = max(0.0, min(1.0, normalized_start))

            # End timeframe coordinates
            end_year = comparison['end_timeframe']['value']
            normalized_end = (end_year - self.chronological_reference_epoch) / 50.0
            vector[1] = max(0.0, min(1.0, normalized_end))

            # Span magnitude
            span_years = abs(end_year - start_year)
            vector[2] = min(span_years / 10.0, 1.0)

            # Confidence score
            avg_confidence = (comparison['start_timeframe']['confidence'] +
                            comparison['end_timeframe']['confidence']) / 2.0
            vector[3] = avg_confidence

            comparative_vectors.append(vector)

        return comparative_vectors

    def generate_temporal_constraint_boundaries(self, temporal_analysis: Dict) -> List[Dict]:
        """Generate temporal constraint boundaries for Q3.1"""
        boundaries = []

        # Primary temporal boundary
        primary_dates = temporal_analysis['temporal_entities'].get('extracted_dates', [])
        if primary_dates:
            # Handle both dict and object formats
            if isinstance(primary_dates[0], dict):
                primary_year = primary_dates[0]['normalized_value']
            else:
                primary_year = primary_dates[0].normalized_value

            normalized_year = (primary_year - self.chronological_reference_epoch) / 50.0

            boundaries.append({
                'boundary_type': 'primary_temporal',
                'center_coordinates': [normalized_year],
                'boundary_radius': 0.05,  # Tight boundary
                'constraint_strength': 0.9,
                'temporal_year': primary_year
            })

        # Comparative temporal boundaries
        comparison_timeframes = temporal_analysis['temporal_relationships'].get('comparison_timeframes', [])
        for i, comparison in enumerate(comparison_timeframes):
            start_year = comparison['start_timeframe']['value']
            end_year = comparison['end_timeframe']['value']

            start_normalized = (start_year - self.chronological_reference_epoch) / 50.0
            end_normalized = (end_year - self.chronological_reference_epoch) / 50.0

            boundaries.append({
                'boundary_type': 'comparative_temporal',
                'center_coordinates': [start_normalized, end_normalized],
                'boundary_radius': 0.1,
                'constraint_strength': 0.8,
                'comparison_id': i,
                'temporal_span': [start_year, end_year]
            })

        return boundaries

    def generate_chronological_ordering_vectors(self, temporal_analysis: Dict) -> List[Dict]:
        """Generate vectors representing chronological ordering"""
        ordering_vectors = []

        temporal_sequences = temporal_analysis['temporal_relationships'].get('temporal_sequences', [])
        for sequence in temporal_sequences:
            if 'sequence_entities' in sequence and len(sequence['sequence_entities']) > 1:
                entities = sequence['sequence_entities']

                for i in range(len(entities) - 1):
                    current_entity = entities[i]
                    next_entity = entities[i + 1]

                    ordering_vectors.append({
                        'ordering_type': 'chronological_sequence',
                        'source_temporal_order': current_entity['temporal_order'],
                        'target_temporal_order': next_entity['temporal_order'],
                        'ordering_direction': 'forward',
                        'temporal_distance': next_entity['temporal_order'] - current_entity['temporal_order'],
                        'sequence_position': i
                    })

        return ordering_vectors

    def calculate_temporal_precision_weights(self, temporal_analysis: Dict) -> List[float]:
        """Calculate precision weights for temporal coordinates"""
        weights = []

        # Primary temporal entity confidence
        primary_dates = temporal_analysis['temporal_entities'].get('extracted_dates', [])
        if primary_dates:
            # Handle both dict and object formats
            if isinstance(primary_dates[0], dict):
                weights.append(primary_dates[0]['confidence'])
            else:
                weights.append(primary_dates[0].confidence)
        else:
            weights.append(0.5)  # Default weight

        # Comparative temporal confidence
        comparison_timeframes = temporal_analysis['temporal_relationships'].get('comparison_timeframes', [])
        for comparison in comparison_timeframes:
            avg_confidence = (comparison['start_timeframe']['confidence'] +
                            comparison['end_timeframe']['confidence']) / 2.0
            weights.append(avg_confidence)

        # Temporal granularity precision
        granularity = temporal_analysis['temporal_relationships'].get('temporal_granularity', 'year')
        granularity_precision = {'day': 1.0, 'month': 0.9, 'quarter': 0.8, 'year': 0.7, 'multi_year': 0.5}
        weights.append(granularity_precision.get(granularity, 0.7))

        return weights

class ChronologicalStructureAnalyzer:
    """Analyzes the chronological structure and complexity of temporal references"""

    def analyze_chronological_structure(self, temporal_entities: Dict, temporal_relationships: Dict) -> Dict:
        """Comprehensive chronological structure analysis"""

        # Calculate temporal complexity
        temporal_complexity = self.calculate_temporal_complexity(temporal_entities, temporal_relationships)

        # Determine chronological depth
        chronological_depth = self.determine_chronological_depth(temporal_entities)

        # Analyze temporal scope
        temporal_scope = self.analyze_temporal_scope(temporal_relationships)

        # Calculate time reference precision
        time_reference_precision = self.calculate_time_reference_precision(temporal_entities)

        # Assess temporal disambiguation
        temporal_disambiguation = self.assess_temporal_disambiguation(temporal_entities, temporal_relationships)

        return {
            'temporal_complexity': temporal_complexity,
            'chronological_depth': chronological_depth,
            'temporal_scope': temporal_scope,
            'time_reference_precision': time_reference_precision,
            'temporal_disambiguation': temporal_disambiguation
        }

    def calculate_temporal_complexity(self, entities: Dict, relationships: Dict) -> float:
        """Calculate complexity score based on temporal references"""
        complexity_score = 0.0

        # Base complexity from number of temporal entities
        num_dates = len(entities.get('extracted_dates', []))
        num_expressions = len(entities.get('temporal_expressions', []))
        num_relative_refs = len(entities.get('relative_time_references', []))
        num_modifiers = len(entities.get('temporal_modifiers', []))

        entity_complexity = min((num_dates + num_expressions + num_relative_refs + num_modifiers) / 6.0, 1.0)
        complexity_score += entity_complexity * 0.4

        # Relationship complexity
        num_comparisons = len(relationships.get('comparison_timeframes', []))
        num_sequences = len(relationships.get('temporal_sequences', []))

        relationship_complexity = min((num_comparisons + num_sequences) / 3.0, 1.0)
        complexity_score += relationship_complexity * 0.6

        return min(complexity_score, 1.0)

    def determine_chronological_depth(self, temporal_entities: Dict) -> int:
        """Determine the chronological depth (nesting levels) of temporal references"""
        depth = 0

        # Base depth from entity types
        if temporal_entities.get('extracted_dates'):
            depth += 1
        if temporal_entities.get('temporal_expressions'):
            depth += 1
        if temporal_entities.get('relative_time_references'):
            depth += 1
        if temporal_entities.get('temporal_modifiers'):
            depth += 1

        return min(depth, 4)  # Max depth of 4

    def analyze_temporal_scope(self, temporal_relationships: Dict) -> str:
        """Analyze the temporal scope (local vs global time context)"""

        # Check for comparison timeframes spanning multiple years
        comparison_timeframes = temporal_relationships.get('comparison_timeframes', [])
        if comparison_timeframes:
            for comparison in comparison_timeframes:
                if 'span_years' in comparison and abs(comparison['span_years']) > 5:
                    return 'global'

        # Check temporal sequences
        temporal_sequences = temporal_relationships.get('temporal_sequences', [])
        if temporal_sequences:
            for sequence in temporal_sequences:
                if sequence.get('sequence_span', 0) > 3:
                    return 'global'

        return 'local'

    def calculate_time_reference_precision(self, temporal_entities: Dict) -> float:
        """Calculate precision of time references"""
        total_precision = 0.0
        num_entities = 0

        # Precision from extracted dates (high precision)
        for entity in temporal_entities.get('extracted_dates', []):
            total_precision += 0.9
            num_entities += 1

        # Precision from temporal expressions (medium-high precision)
        for entity in temporal_entities.get('temporal_expressions', []):
            total_precision += 0.8
            num_entities += 1

        # Precision from relative time references (medium precision)
        for entity in temporal_entities.get('relative_time_references', []):
            total_precision += 0.6
            num_entities += 1

        # Precision from temporal modifiers (low precision)
        for entity in temporal_entities.get('temporal_modifiers', []):
            total_precision += 0.4
            num_entities += 1

        if num_entities == 0:
            return 0.5  # Default precision

        return total_precision / num_entities

    def assess_temporal_disambiguation(self, temporal_entities: Dict, temporal_relationships: Dict) -> float:
        """Assess temporal disambiguation (clarity of time references)"""
        disambiguation_score = 1.0  # Start with perfect clarity

        # Reduce score for ambiguous relative references
        relative_refs = temporal_entities.get('relative_time_references', [])
        if relative_refs:
            disambiguation_score -= len(relative_refs) * 0.1

        # Reduce score for temporal modifiers without clear anchors
        temporal_modifiers = temporal_entities.get('temporal_modifiers', [])
        extracted_dates = temporal_entities.get('extracted_dates', [])

        if temporal_modifiers and not extracted_dates:
            disambiguation_score -= 0.3

        # Improve score for clear temporal comparisons
        comparison_timeframes = temporal_relationships.get('comparison_timeframes', [])
        if comparison_timeframes:
            disambiguation_score += len(comparison_timeframes) * 0.1

        return max(0.0, min(disambiguation_score, 1.0))

class TemporalConstraintGenerator:
    """Generates geometric constraints based on temporal analysis"""

    def generate_temporal_constraints(self, geometric_coordinates: Dict, temporal_analysis: Dict) -> Dict:
        """Generate comprehensive temporal constraint specifications"""

        # Generate hard temporal constraints
        hard_constraints = self.generate_hard_temporal_constraints(geometric_coordinates, temporal_analysis)

        # Generate soft temporal preferences
        soft_preferences = self.generate_soft_temporal_preferences(geometric_coordinates, temporal_analysis)

        # Generate exclusion zones
        exclusion_zones = self.generate_temporal_exclusion_zones(geometric_coordinates, temporal_analysis)

        # Generate precedence rules
        precedence_rules = self.generate_chronological_precedence_rules(geometric_coordinates, temporal_analysis)

        # Generate matching priorities
        matching_priorities = self.generate_temporal_matching_priorities(geometric_coordinates, temporal_analysis)

        return {
            'hard_temporal_constraints': hard_constraints,
            'soft_temporal_preferences': soft_preferences,
            'temporal_exclusion_zones': exclusion_zones,
            'chronological_precedence_rules': precedence_rules,
            'temporal_matching_priorities': matching_priorities
        }

    def generate_hard_temporal_constraints(self, coordinates: Dict, analysis: Dict) -> List[Dict]:
        """Generate mandatory temporal constraints for Q3.1 geometric matching"""
        constraints = []

        # Primary temporal boundary constraints
        primary_vector = coordinates.get('primary_temporal_vector', [])
        if len(primary_vector) >= 2:
            constraints.append({
                'constraint_type': 'temporal_boundary',
                'constraint_dimension': [0, 1],  # First two temporal dimensions
                'constraint_center': primary_vector[:2],
                'constraint_radius': 0.08,       # Tight temporal boundary
                'constraint_strength': 'mandatory',
                'constraint_description': 'Primary temporal reference constraint',
                'temporal_importance': 0.9
            })

        # Temporal comparison constraints
        comparative_vectors = coordinates.get('comparative_temporal_vectors', [])
        for i, comp_vector in enumerate(comparative_vectors):
            if len(comp_vector) >= 2:
                constraints.append({
                    'constraint_type': 'temporal_comparison',
                    'constraint_dimension': [2, 3],
                    'constraint_center': comp_vector[:2],
                    'constraint_radius': 0.12,
                    'constraint_strength': 'mandatory',
                    'constraint_description': f'Temporal comparison constraint {i+1}',
                    'temporal_importance': 0.8
                })

        return constraints

    def generate_soft_temporal_preferences(self, coordinates: Dict, analysis: Dict) -> List[Dict]:
        """Generate preferred temporal regions for matching"""
        preferences = []

        # Temporal granularity preferences
        granularity = analysis['temporal_relationships'].get('temporal_granularity', 'year')
        if granularity in ['quarter', 'month']:
            preferences.append({
                'preference_type': 'temporal_granularity',
                'preference_description': f'Prefer {granularity}-level temporal matching',
                'granularity_boost': 0.2,
                'temporal_dimension_focus': [4]  # Granularity dimension
            })

        # Chronological ordering preferences
        ordering_vectors = coordinates.get('chronological_ordering_vectors', [])
        for i, ordering in enumerate(ordering_vectors):
            if ordering.get('temporal_distance', 0) > 0:
                preferences.append({
                    'preference_type': 'chronological_ordering',
                    'preference_description': f'Prefer chronological sequence {i+1}',
                    'ordering_boost': 0.15,
                    'sequence_position': ordering.get('sequence_position', 0)
                })

        return preferences

    def generate_temporal_exclusion_zones(self, coordinates: Dict, analysis: Dict) -> List[Dict]:
        """Generate temporal exclusion zones to avoid during matching"""
        exclusion_zones = []

        # Exclude temporal regions too far from primary timeframe
        primary_vector = coordinates.get('primary_temporal_vector', [])
        if len(primary_vector) >= 1:
            primary_temporal_coord = primary_vector[0]

            # Exclude regions more than 0.3 units away in temporal space
            exclusion_zones.append({
                'exclusion_type': 'distant_temporal',
                'exclusion_center': [primary_temporal_coord],
                'exclusion_radius': 0.3,
                'exclusion_strength': 'moderate',
                'exclusion_description': 'Exclude temporally distant regions'
            })

        return exclusion_zones

    def generate_chronological_precedence_rules(self, coordinates: Dict, analysis: Dict) -> List[Dict]:
        """Generate chronological precedence rules for ordering"""
        precedence_rules = []

        # Rules based on temporal sequences
        ordering_vectors = coordinates.get('chronological_ordering_vectors', [])
        for ordering in ordering_vectors:
            if ordering.get('ordering_direction') == 'forward':
                precedence_rules.append({
                    'rule_type': 'chronological_precedence',
                    'source_temporal_order': ordering.get('source_temporal_order'),
                    'target_temporal_order': ordering.get('target_temporal_order'),
                    'precedence_strength': 0.8,
                    'rule_description': 'Maintain chronological ordering'
                })

        return precedence_rules

    def generate_temporal_matching_priorities(self, coordinates: Dict, analysis: Dict) -> List[Dict]:
        """Generate temporal matching priorities for Q3.1"""
        priorities = []

        # High priority for exact temporal matches
        primary_dates = analysis['temporal_entities'].get('extracted_dates', [])
        for i, date_entity in enumerate(primary_dates):
            # Handle both dict and object formats
            if isinstance(date_entity, dict):
                temporal_value = date_entity['normalized_value']
                text = date_entity['text']
            else:
                temporal_value = date_entity.normalized_value
                text = date_entity.text

            priorities.append({
                'priority_type': 'exact_temporal_match',
                'priority_level': 'high',
                'temporal_value': temporal_value,
                'priority_boost': 0.3,
                'match_description': f'Exact match for {text}'
            })

        # Medium priority for temporal range matches
        comparison_timeframes = analysis['temporal_relationships'].get('comparison_timeframes', [])
        for comparison in comparison_timeframes:
            priorities.append({
                'priority_type': 'temporal_range_match',
                'priority_level': 'medium',
                'temporal_start': comparison['start_timeframe']['value'],
                'temporal_end': comparison['end_timeframe']['value'],
                'priority_boost': 0.2,
                'match_description': f"Range match for {comparison['start_timeframe']['text']} to {comparison['end_timeframe']['text']}"
            })

        return priorities

class Q24TemporalCoordinateMapping:
    """Main Q2.4 Temporal Coordinate Mapping processor"""

    def __init__(self, config: Dict = None):
        self.config = config or {
            "temporal_coordinate_dimensions": 8,
            "chronological_reference_epoch": 2000,
            "temporal_precision_threshold": 0.8,
            "max_temporal_span_years": 10,
            "constraint_boundary_radius": 0.1,
            "processing_timeout": 100
        }

        # Initialize components
        self.temporal_extractor = TemporalEntityExtractor()
        self.relationship_analyzer = TemporalRelationshipAnalyzer()
        self.geometric_coordinator = GeometricTemporalCoordinator()
        self.structure_analyzer = ChronologicalStructureAnalyzer()
        self.constraint_generator = TemporalConstraintGenerator()

    def analyze_temporal_coordinates(self, question_id: str) -> Dict:
        """Main temporal coordinate analysis method"""
        start_time = datetime.now()

        # Load question data from Q1
        question_data = self.load_question_from_q1(question_id)
        if not question_data:
            return self.create_error_response(question_id, "Failed to load question data from Q1")

        question_text = question_data['question_text']
        doc_id = question_data['doc_id']

        try:
            # Step 1: Extract temporal entities
            temporal_entities_raw = self.temporal_extractor.extract_temporal_entities(question_text)

            # Convert TemporalEntity objects to dictionaries for JSON serialization
            temporal_entities = self.convert_temporal_entities_to_dict(temporal_entities_raw)

            # Step 2: Analyze temporal relationships
            temporal_relationships = self.relationship_analyzer.analyze_temporal_relationships(
                temporal_entities, question_text
            )

            # Step 3: Analyze chronological structure
            chronological_structure = self.structure_analyzer.analyze_chronological_structure(
                temporal_entities, temporal_relationships
            )

            # Combine temporal analysis
            temporal_analysis = {
                'temporal_entities': temporal_entities,
                'temporal_relationships': temporal_relationships,
                'chronological_structure': chronological_structure
            }

            # Step 4: Map to geometric coordinates
            geometric_coordinates = self.geometric_coordinator.map_temporal_to_geometric(temporal_analysis)

            # Step 5: Generate temporal constraints
            constraint_specifications = self.constraint_generator.generate_temporal_constraints(
                geometric_coordinates, temporal_analysis
            )

            # Calculate processing time
            end_time = datetime.now()
            processing_time_ms = (end_time - start_time).total_seconds() * 1000

            # Build complete response
            return {
                question_id: {
                    'question_id': question_id,
                    'doc_id': doc_id,
                    'question_text': question_text,
                    'temporal_analysis': temporal_analysis,
                    'geometric_temporal_coordinates': geometric_coordinates,
                    'constraint_specifications': constraint_specifications,
                    'processing_metadata': {
                        'analysis_timestamp': datetime.now().isoformat(),
                        'processing_time_ms': round(processing_time_ms, 3),
                        'temporal_extraction_confidence': self.calculate_extraction_confidence(temporal_entities),
                        'geometric_mapping_status': 'complete'
                    }
                }
            }

        except Exception as e:
            return self.create_error_response(question_id, f"Temporal analysis error: {str(e)}")

    def convert_temporal_entities_to_dict(self, temporal_entities_raw: Dict) -> Dict:
        """Convert TemporalEntity objects to dictionaries for JSON serialization"""
        temporal_entities = {}

        for entity_type, entities in temporal_entities_raw.items():
            if entity_type == 'chronological_indicators':
                # Already strings, keep as is
                temporal_entities[entity_type] = entities
            else:
                # Convert TemporalEntity objects to dictionaries
                temporal_entities[entity_type] = []
                for entity in entities:
                    temporal_entities[entity_type].append({
                        'text': entity.text,
                        'entity_type': entity.entity_type,
                        'start_pos': entity.start_pos,
                        'end_pos': entity.end_pos,
                        'normalized_value': entity.normalized_value,
                        'confidence': entity.confidence
                    })

        return temporal_entities

    def load_question_from_q1(self, question_id: str) -> Optional[Dict]:
        """Load question data from Q1 output or fallback to test dataset"""
        # First try to load from Q1 output
        q1_output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'outputs', 'Q1_Question_ingestion.json'
        )

        try:
            with open(q1_output_path, 'r', encoding='utf-8') as f:
                q1_data = json.load(f)

                # Handle both nested and flat Q1 structures
                if question_id in q1_data:
                    return q1_data[question_id]
                elif 'question_id' in q1_data and q1_data['question_id'] == question_id:
                    return q1_data
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass

        # Fallback: Load from original test dataset
        test_file_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'outputs', 'Q1_20_records_test_results.json'
        )

        try:
            with open(test_file_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
                processed_questions = test_data.get('processed_questions', [])

                for question_data in processed_questions:
                    if question_data.get('question_id') == question_id:
                        # Return in Q1 format
                        return {
                            'question_id': question_data['question_id'],
                            'question_text': question_data['question_text'],
                            'doc_id': question_data['doc_id'],
                            'pipeline_ready': True,
                            'metadata': question_data.get('metadata', {})
                        }
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass

        return None

    def calculate_extraction_confidence(self, temporal_entities: Dict) -> float:
        """Calculate overall confidence in temporal extraction"""
        total_confidence = 0.0
        total_entities = 0

        for entity_type, entities in temporal_entities.items():
            if entity_type == 'chronological_indicators':
                continue  # Skip string lists

            for entity in entities:
                # Handle both dict and object formats
                if isinstance(entity, dict) and 'confidence' in entity:
                    total_confidence += entity['confidence']
                    total_entities += 1
                elif hasattr(entity, 'confidence'):
                    total_confidence += entity.confidence
                    total_entities += 1

        if total_entities == 0:
            return 0.5  # Default confidence

        return total_confidence / total_entities

    def create_error_response(self, question_id: str, error_message: str) -> Dict:
        """Create error response for failed processing"""
        return {
            question_id: {
                'question_id': question_id,
                'error': error_message,
                'processing_metadata': {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'processing_time_ms': 0,
                    'temporal_extraction_confidence': 0.0,
                    'geometric_mapping_status': 'failed'
                }
            }
        }

    def save_output(self, result: Dict) -> str:
        """Save Q2.4 output to JSON file"""
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'outputs'
        )
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, 'Q2.4_temporal_coordinate_mapping.json')

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        return output_file

def main():
    """Main execution function"""
    print("Q2.4 TEMPORAL COORDINATE MAPPING")
    print("=" * 50)

    # Initialize Q2.4 processor
    processor = Q24TemporalCoordinateMapping()

    # Sample question ID for testing
    question_id = "finqa_test_1630"

    print("=" * 60)
    print("Q2.4: Temporal Coordinate Mapping Test")
    print("=" * 60)
    print(f"Processing Q2.4 for question: {question_id}")

    # Load and display question
    question_data = processor.load_question_from_q1(question_id)
    if question_data:
        question_text = question_data['question_text']
        print(f"Question: {question_text[:60]}...")
        print()

    # Process temporal coordinate mapping
    result = processor.analyze_temporal_coordinates(question_id)

    if question_id in result and 'error' not in result[question_id]:
        analysis_result = result[question_id]

        print("=" * 40)
        print("Q2.4 OUTPUT - Temporal Coordinate Mapping:")
        print("=" * 40)
        print(f"Question ID: {analysis_result['question_id']}")
        print()

        # Display temporal entities
        temporal_entities = analysis_result['temporal_analysis']['temporal_entities']
        print("Temporal Entities:")
        for entity_type, entities in temporal_entities.items():
            if entities:
                if entity_type == 'chronological_indicators':
                    print(f"  {entity_type}: {', '.join(entities)}")
                else:
                    print(f"  {entity_type}: {len(entities)} detected")
                    for entity in entities[:2]:  # Show first 2
                        if hasattr(entity, 'text'):
                            print(f"    - {entity.text} ({entity.entity_type})")
        print()

        # Display temporal relationships
        temporal_relationships = analysis_result['temporal_analysis']['temporal_relationships']
        print("Temporal Relationships:")
        for rel_type, relationships in temporal_relationships.items():
            if isinstance(relationships, list) and relationships:
                print(f"  {rel_type}: {len(relationships)} found")
            elif relationships:
                print(f"  {rel_type}: {relationships}")
        print()

        # Display geometric coordinates
        geometric_coords = analysis_result['geometric_temporal_coordinates']
        print("Geometric Temporal Coordinates:")
        primary_vector = geometric_coords.get('primary_temporal_vector', [])
        if primary_vector:
            print(f"  Primary temporal vector: [{'  '.join([f'{x:.2f}' for x in primary_vector[:5]])}...]")

        comparative_vectors = geometric_coords.get('comparative_temporal_vectors', [])
        if comparative_vectors:
            print(f"  Comparative vectors: {len(comparative_vectors)} vectors")

        constraint_boundaries = geometric_coords.get('temporal_constraint_boundaries', [])
        if constraint_boundaries:
            print(f"  Constraint boundaries: {len(constraint_boundaries)} boundaries")
        print()

        # Display constraint specifications
        constraints = analysis_result['constraint_specifications']
        print("Temporal Constraints:")
        hard_constraints = constraints.get('hard_temporal_constraints', [])
        if hard_constraints:
            print(f"  Hard constraints: {len(hard_constraints)}")

        soft_preferences = constraints.get('soft_temporal_preferences', [])
        if soft_preferences:
            print(f"  Soft preferences: {len(soft_preferences)}")
        print()

        # Display processing metadata
        metadata = analysis_result['processing_metadata']
        print(f"Processing Time: {metadata['processing_time_ms']:.1f}ms")
        print(f"Extraction Confidence: {metadata['temporal_extraction_confidence']:.2f}")
        print(f"Mapping Status: {metadata['geometric_mapping_status']}")

    else:
        print("ERROR in Q2.4 processing:")
        print(result.get(question_id, {}).get('error', 'Unknown error'))
        return 1

    # Save output
    output_file = processor.save_output(result)
    print(f"Q2.4 output saved to {output_file}")
    print(f"Q2.4_temporal_coordinate_mapping.json created successfully")

    print("Temporal coordinate mapping complete - ready for Q2.5 integration")

    return 0

if __name__ == "__main__":
    sys.exit(main())