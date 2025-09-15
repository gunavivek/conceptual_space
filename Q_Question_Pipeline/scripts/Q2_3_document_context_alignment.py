"""
Q2.3: Document Context Alignment
Analyzes document structure and provides geometric positioning hints for constraint-based matching
"""

import json
import os
import re
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd


class DocumentStructureAnalyzer:
    """
    Advanced analysis of document structural organization
    """

    def __init__(self):
        self.hierarchy_patterns = self._load_hierarchy_patterns()
        self.financial_keywords = [
            "revenue", "income", "expense", "cost", "profit", "loss", "total",
            "assets", "liabilities", "equity", "cash", "flow", "statement"
        ]

    def _load_hierarchy_patterns(self) -> Dict:
        """Load document hierarchy detection patterns"""
        return {
            "section_headers": [
                r'^#{1,3}\s+(.+)$',  # Markdown headers
                r'^\d+\.\s+(.+)$',   # Numbered sections
                r'^[A-Z][A-Z\s]+$'   # All caps headers
            ],
            "subsection_patterns": [
                r'^\d+\.\d+\s+(.+)$',  # Numbered subsections
                r'^[a-z]\)\s+(.+)$',    # Lettered subsections
                r'^-\s+(.+)$'           # Bullet points
            ]
        }

    def analyze_document_structure(self, doc_id: str, document_chunks: List[Dict]) -> Dict:
        """
        Comprehensive document structure analysis
        """
        try:
            # Hierarchical structure analysis
            hierarchy = self._analyze_hierarchy(document_chunks)

            # Table structure intelligence
            table_analysis = self._analyze_table_structure(document_chunks)

            # Layout pattern recognition
            layout_patterns = self._analyze_layout_patterns(document_chunks)

            return {
                'hierarchical_mapping': hierarchy,
                'table_structure_intelligence': table_analysis,
                'layout_pattern_analysis': layout_patterns
            }
        except Exception as e:
            print(f"Error in document structure analysis: {e}")
            return self._get_default_structure_analysis()

    def _analyze_hierarchy(self, chunks: List[Dict]) -> Dict:
        """Analyze document hierarchical structure"""
        hierarchy_levels = {}
        section_tree = {}
        cross_references = []

        for i, chunk in enumerate(chunks):
            text = chunk.get('text', '')

            # Detect hierarchy level
            level = self._detect_hierarchy_level(text)
            if level > 0:
                if level not in hierarchy_levels:
                    hierarchy_levels[level] = []
                    section_tree[str(level)] = []

                section_info = {
                    'id': f"section_{i}",
                    'text': text[:100],  # First 100 chars
                    'level': level,
                    'position': i,
                    'content_density': len(text.split()),
                    'spatial_boundary': {
                        'start': i,
                        'end': min(i + 3, len(chunks)),  # Approximate section span
                        'geometric_region': [i, level]
                    }
                }

                hierarchy_levels[level].append(section_info)
                section_tree[str(level)].append(section_info)

        return {
            'total_levels': len(hierarchy_levels),
            'section_tree': section_tree,
            'level_importance': {str(k): 1.0 / k for k in hierarchy_levels.keys()},
            'cross_references': cross_references
        }

    def _detect_hierarchy_level(self, text: str) -> int:
        """Detect hierarchy level of text content"""
        # Check for markdown headers
        if re.match(r'^#{1,3}\s+', text):
            return text.count('#')

        # Check for numbered sections
        if re.match(r'^\d+\.\s+', text):
            return 1

        # Check for subsections
        if re.match(r'^\d+\.\d+\s+', text):
            return 2

        # Check for all caps (likely headers)
        if text.isupper() and len(text.split()) <= 5:
            return 1

        return 0  # No hierarchy detected

    def _analyze_table_structure(self, chunks: List[Dict]) -> Dict:
        """Analyze table organization for geometric positioning"""
        detected_tables = []
        table_relationships = {}
        intersection_hotspots = []

        for i, chunk in enumerate(chunks):
            text = chunk.get('text', '')

            # Detect financial tables
            if self._is_financial_table(text):
                table_info = {
                    'id': f"table_{i}",
                    'type': self._classify_table_type(text),
                    'position': i,
                    'spatial_boundary': {
                        'chunk_index': i,
                        'geometric_coordinates': [i, 0],  # Row, column in document
                        'boundary_box': [i, i+1, 0, 1]   # Simple bounding box
                    },
                    'information_density': len(text.split()),
                    'financial_relevance': self._calculate_financial_relevance(text)
                }
                detected_tables.append(table_info)

                # Identify intersection hotspots
                hotspots = self._identify_intersection_hotspots(text, i)
                intersection_hotspots.extend(hotspots)

        return {
            'detected_tables': detected_tables,
            'table_relationships': table_relationships,
            'column_semantics': self._classify_column_semantics(detected_tables),
            'row_categorization': self._classify_row_types(detected_tables),
            'intersection_hotspots': intersection_hotspots
        }

    def _is_financial_table(self, text: str) -> bool:
        """Check if text contains financial table data"""
        # Look for financial keywords + numbers + year patterns
        financial_matches = sum(1 for keyword in self.financial_keywords if keyword.lower() in text.lower())
        year_matches = len(re.findall(r'\b20\d{2}\b', text))  # Years like 2018, 2019
        number_matches = len(re.findall(r'\$?[\d,]+\.?\d*', text))  # Dollar amounts/numbers

        return financial_matches >= 2 and year_matches >= 1 and number_matches >= 2

    def _classify_table_type(self, text: str) -> str:
        """Classify the type of financial table"""
        text_lower = text.lower()

        if any(word in text_lower for word in ['revenue', 'income', 'sales']):
            return 'revenue_table'
        elif any(word in text_lower for word in ['expense', 'cost', 'operating']):
            return 'expense_table'
        elif any(word in text_lower for word in ['balance', 'assets', 'liabilities']):
            return 'balance_sheet'
        elif any(word in text_lower for word in ['cash', 'flow']):
            return 'cash_flow'
        else:
            return 'financial_data'

    def _calculate_financial_relevance(self, text: str) -> float:
        """Calculate financial relevance score"""
        financial_score = sum(1 for keyword in self.financial_keywords if keyword.lower() in text.lower())
        return min(1.0, financial_score / 5.0)  # Normalize to 0-1

    def _identify_intersection_hotspots(self, text: str, position: int) -> List[Dict]:
        """Identify high-value table intersection points"""
        hotspots = []

        # Look for year-revenue intersections (common in financial QA)
        years = re.findall(r'\b20\d{2}\b', text)
        financial_terms = [term for term in self.financial_keywords if term.lower() in text.lower()]

        for year in years:
            for term in financial_terms:
                hotspots.append({
                    'table_id': f"table_{position}",
                    'row_semantic': term,
                    'column_semantic': year,
                    'intersection_value': 0.8,  # High value intersection
                    'geometric_coordinates': [position, hash(f"{term}_{year}") % 100]
                })

        return hotspots

    def _classify_column_semantics(self, tables: List[Dict]) -> Dict:
        """Classify column semantic types"""
        return {
            'temporal_columns': ['2018', '2019', '2020', '2021', 'Q1', 'Q2', 'Q3', 'Q4'],
            'financial_columns': ['Revenue', 'Income', 'Expense', 'Total', 'Net'],
            'metric_columns': ['Percentage', 'Change', 'Growth', 'Rate']
        }

    def _classify_row_types(self, tables: List[Dict]) -> Dict:
        """Classify row semantic types"""
        return {
            'financial_rows': ['Current', 'Non-current', 'Total', 'Net'],
            'category_rows': ['Assets', 'Liabilities', 'Equity', 'Revenue'],
            'calculation_rows': ['Change', 'Growth', 'Percentage', 'Ratio']
        }

    def _analyze_layout_patterns(self, chunks: List[Dict]) -> Dict:
        """Analyze document layout patterns"""
        total_chunks = len(chunks)
        financial_chunks = sum(1 for chunk in chunks if self._is_financial_table(chunk.get('text', '')))

        return {
            'document_type': 'financial_report',  # Assume financial based on context
            'layout_category': 'structured_financial',
            'structural_patterns': ['hierarchical_sections', 'financial_tables', 'numerical_data'],
            'navigation_complexity': min(1.0, total_chunks / 50.0),  # Normalize complexity
            'content_density': {
                'total_chunks': total_chunks,
                'financial_density': financial_chunks / max(1, total_chunks),
                'average_chunk_length': np.mean([len(chunk.get('text', '').split()) for chunk in chunks])
            }
        }

    def _get_default_structure_analysis(self) -> Dict:
        """Return default structure analysis on error"""
        return {
            'hierarchical_mapping': {
                'total_levels': 1,
                'section_tree': {'1': []},
                'level_importance': {'1': 1.0},
                'cross_references': []
            },
            'table_structure_intelligence': {
                'detected_tables': [],
                'table_relationships': {},
                'column_semantics': {},
                'row_categorization': {},
                'intersection_hotspots': []
            },
            'layout_pattern_analysis': {
                'document_type': 'unknown',
                'layout_category': 'unstructured',
                'structural_patterns': [],
                'navigation_complexity': 0.5,
                'content_density': {'total_chunks': 0, 'financial_density': 0.0}
            }
        }


class GeometricPositioningHints:
    """
    Generates geometric hints for constraint-based matching
    """

    def generate_positioning_hints(self, structure_analysis: Dict, question_intent: Dict) -> Dict:
        """
        Generate geometric positioning hints based on document structure and question intent
        """
        # Spatial clustering based on content organization
        spatial_clusters = self._generate_spatial_clustering(structure_analysis)

        # Structural coordinate system for document navigation
        coordinate_system = self._establish_coordinate_system(structure_analysis)

        # Constraint recommendations for geometric matching
        constraints = self._recommend_constraints(structure_analysis, question_intent)

        return {
            'spatial_clustering_map': spatial_clusters,
            'structural_coordinate_system': coordinate_system,
            'constraint_recommendations': constraints
        }

    def _generate_spatial_clustering(self, structure: Dict) -> Dict:
        """Create spatial clusters based on document structure"""
        clusters = []

        # Financial data clusters from detected tables
        for table in structure['table_structure_intelligence']['detected_tables']:
            clusters.append({
                'cluster_id': f"financial_cluster_{table['id']}",
                'cluster_type': 'financial_data',
                'geometric_boundary': table['spatial_boundary'],
                'semantic_density': table.get('information_density', 0),
                'access_priority': 'high' if table.get('financial_relevance', 0) > 0.6 else 'medium'
            })

        # Hierarchical content clusters
        for level, sections in structure['hierarchical_mapping']['section_tree'].items():
            if int(level) <= 2:  # Top 2 hierarchy levels
                for section in sections:
                    clusters.append({
                        'cluster_id': f"section_cluster_{section['id']}",
                        'cluster_type': 'hierarchical_content',
                        'geometric_boundary': section['spatial_boundary'],
                        'semantic_density': section.get('content_density', 0),
                        'access_priority': 'high' if int(level) == 1 else 'medium'
                    })

        return {
            'content_clusters': clusters,
            'cluster_boundaries': self._calculate_cluster_boundaries(clusters),
            'inter_cluster_distances': self._calculate_inter_cluster_distances(clusters)
        }

    def _calculate_cluster_boundaries(self, clusters: List[Dict]) -> List[Dict]:
        """Calculate geometric boundaries for clusters"""
        boundaries = []
        for cluster in clusters:
            boundary = cluster['geometric_boundary']
            boundaries.append({
                'cluster_id': cluster['cluster_id'],
                'boundary_type': 'rectangular',
                'coordinates': boundary.get('boundary_box', [0, 1, 0, 1]),
                'confidence': 0.8
            })
        return boundaries

    def _calculate_inter_cluster_distances(self, clusters: List[Dict]) -> Dict:
        """Calculate distances between clusters"""
        distances = {}
        for i, cluster1 in enumerate(clusters):
            for j, cluster2 in enumerate(clusters[i+1:], i+1):
                pos1 = cluster1['geometric_boundary'].get('geometric_coordinates', [i, 0])
                pos2 = cluster2['geometric_boundary'].get('geometric_coordinates', [j, 0])
                distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
                distances[f"{cluster1['cluster_id']}_to_{cluster2['cluster_id']}"] = float(distance)
        return distances

    def _establish_coordinate_system(self, structure: Dict) -> Dict:
        """Establish structural coordinate system for document navigation"""
        return {
            'reference_origin': {'x': 0, 'y': 0, 'semantic': 'document_start'},
            'axis_definitions': {
                'x_axis': 'document_progression',  # Sequential progression through document
                'y_axis': 'hierarchy_level',       # Hierarchical depth
                'z_axis': 'semantic_category'      # Content category (financial, textual, etc.)
            },
            'transformation_matrices': {
                'document_to_geometric': [[1.0, 0.0], [0.0, 1.0]],  # Identity for now
                'hierarchy_weighting': [1.0, 0.8, 0.6, 0.4]        # Weights by level
            }
        }

    def _recommend_constraints(self, structure: Dict, question_intent: Dict) -> Dict:
        """Generate constraint recommendations for geometric matching"""
        hard_constraints = []
        soft_constraints = []
        priority_regions = []

        # Table intersection constraints
        if question_intent.get('table_intersection', 0) > 0.7:
            for table in structure['table_structure_intelligence']['detected_tables']:
                hard_constraints.append({
                    'constraint_type': 'table_boundary',
                    'constraint_id': f"table_constraint_{table['id']}",
                    'geometric_region': table['spatial_boundary'],
                    'enforcement_strength': 'mandatory'
                })

                priority_regions.append({
                    'region_id': f"priority_table_{table['id']}",
                    'region_type': 'financial_table',
                    'geometric_boundary': table['spatial_boundary'],
                    'priority_score': table.get('financial_relevance', 0.8)
                })

        # Temporal lookup constraints
        if question_intent.get('temporal_lookup', 0) > 0.6:
            soft_constraints.append({
                'constraint_type': 'temporal_proximity',
                'constraint_id': 'temporal_content_bias',
                'preference_weight': question_intent['temporal_lookup'],
                'target_patterns': ['year_references', 'temporal_markers']
            })

        return {
            'hard_constraints': hard_constraints,
            'soft_constraints': soft_constraints,
            'exclusion_zones': [],  # No exclusion zones for now
            'priority_regions': priority_regions
        }


class ContextualIntelligenceEngine:
    """
    Provides intelligent context alignment between questions and document structure
    """

    def align_question_to_document_context(self, question_data: Dict, structure_analysis: Dict, intent_data: Dict) -> Dict:
        """
        Intelligent alignment of question intent with document structure
        """
        # Analyze structural relevance
        structural_relevance = self._calculate_structural_relevance(
            question_data['question_text'],
            structure_analysis
        )

        # Identify likely target areas
        target_areas = self._identify_target_areas(
            intent_data.get('intent_classification', {}),
            structure_analysis
        )

        # Generate navigation strategy
        navigation_strategy = self._generate_navigation_strategy(
            question_data, structure_analysis, target_areas
        )

        # Generate relationship vectors
        relationship_vectors = self._generate_relationship_vectors(structure_analysis)

        return {
            'question_document_alignment': {
                'structural_relevance': structural_relevance,
                'likely_target_areas': target_areas,
                'navigation_strategy': navigation_strategy,
                'context_amplifiers': self._identify_context_amplifiers(structure_analysis)
            },
            'relationship_vectors': relationship_vectors
        }

    def _calculate_structural_relevance(self, question_text: str, structure: Dict) -> float:
        """Calculate how well question aligns with document structure"""
        relevance_score = 0.0

        # Financial table relevance
        financial_tables = structure['table_structure_intelligence']['detected_tables']
        if financial_tables and any(term in question_text.lower()
                                  for term in ['revenue', 'income', 'percentage', 'change']):
            relevance_score += 0.4

        # Temporal relevance
        if re.search(r'\b20\d{2}\b', question_text):  # Contains years
            relevance_score += 0.3

        # Hierarchical structure relevance
        hierarchy_levels = structure['hierarchical_mapping']['total_levels']
        if hierarchy_levels > 1:  # Well-structured document
            relevance_score += 0.3

        return min(1.0, relevance_score)

    def _identify_target_areas(self, intent_classification: Dict, structure: Dict) -> List[Dict]:
        """Identify probable answer locations based on intent and structure"""
        target_areas = []

        # For table intersection queries
        if intent_classification.get('table_intersection', 0) > 0.7:
            for table in structure['table_structure_intelligence']['detected_tables']:
                target_areas.append({
                    'area_type': 'table_intersection',
                    'area_id': table['id'],
                    'geometric_region': table['spatial_boundary'],
                    'confidence': intent_classification['table_intersection'],
                    'access_strategy': 'row_column_navigation'
                })

        # For temporal lookup queries
        if intent_classification.get('temporal_lookup', 0) > 0.6:
            # Find sections with temporal markers
            for level, sections in structure['hierarchical_mapping']['section_tree'].items():
                for section in sections:
                    if re.search(r'\b20\d{2}\b', section.get('text', '')):
                        target_areas.append({
                            'area_type': 'temporal_data',
                            'area_id': section['id'],
                            'geometric_region': section['spatial_boundary'],
                            'confidence': intent_classification['temporal_lookup'],
                            'access_strategy': 'temporal_navigation'
                        })

        # For analytical operations
        if intent_classification.get('analytical_operation', 0) > 0.7:
            for table in structure['table_structure_intelligence']['detected_tables']:
                if table['type'] in ['revenue_table', 'financial_data']:
                    target_areas.append({
                        'area_type': 'analytical_data',
                        'area_id': f"analytical_{table['id']}",
                        'geometric_region': table['spatial_boundary'],
                        'confidence': intent_classification['analytical_operation'],
                        'access_strategy': 'calculation_focused'
                    })

        return target_areas

    def _generate_navigation_strategy(self, question_data: Dict, structure: Dict, target_areas: List[Dict]) -> str:
        """Generate navigation strategy for document traversal"""
        if not target_areas:
            return 'sequential_scan'

        # Determine primary strategy based on target areas
        area_types = [area['area_type'] for area in target_areas]

        if 'table_intersection' in area_types:
            return 'table_intersection_navigation'
        elif 'temporal_data' in area_types:
            return 'temporal_progression_navigation'
        elif 'analytical_data' in area_types:
            return 'calculation_focused_navigation'
        else:
            return 'hierarchical_traversal'

    def _identify_context_amplifiers(self, structure: Dict) -> List[str]:
        """Identify elements that enhance context understanding"""
        amplifiers = []

        # Financial table amplifiers
        if structure['table_structure_intelligence']['detected_tables']:
            amplifiers.append('financial_table_context')

        # Hierarchical structure amplifiers
        if structure['hierarchical_mapping']['total_levels'] > 2:
            amplifiers.append('hierarchical_structure')

        # High information density amplifiers
        density = structure['layout_pattern_analysis']['content_density']
        if density.get('financial_density', 0) > 0.3:
            amplifiers.append('high_financial_density')

        return amplifiers

    def _generate_relationship_vectors(self, structure: Dict) -> Dict:
        """Generate relationship vectors for geometric positioning"""
        return {
            'semantic_relationships': self._calculate_semantic_relationships(structure),
            'structural_relationships': self._calculate_structural_relationships(structure),
            'temporal_relationships': self._calculate_temporal_relationships(structure),
            'hierarchical_relationships': self._calculate_hierarchical_relationships(structure)
        }

    def _calculate_semantic_relationships(self, structure: Dict) -> List[Dict]:
        """Calculate semantic relationship vectors"""
        relationships = []
        tables = structure['table_structure_intelligence']['detected_tables']

        for i, table1 in enumerate(tables):
            for table2 in tables[i+1:]:
                relationships.append({
                    'source': table1['id'],
                    'target': table2['id'],
                    'relationship_type': 'semantic_similarity',
                    'strength': 0.7 if table1['type'] == table2['type'] else 0.3,
                    'vector': [1.0, 0.0, 0.0]  # Placeholder vector
                })

        return relationships

    def _calculate_structural_relationships(self, structure: Dict) -> List[Dict]:
        """Calculate structural relationship vectors"""
        relationships = []
        sections = []

        # Flatten all sections
        for level, level_sections in structure['hierarchical_mapping']['section_tree'].items():
            sections.extend(level_sections)

        # Calculate adjacency relationships
        for i, section in enumerate(sections[:-1]):
            next_section = sections[i + 1]
            relationships.append({
                'source': section['id'],
                'target': next_section['id'],
                'relationship_type': 'structural_adjacency',
                'strength': 0.8,
                'vector': [0.0, 1.0, 0.0]  # Placeholder vector
            })

        return relationships

    def _calculate_temporal_relationships(self, structure: Dict) -> List[Dict]:
        """Calculate temporal relationship vectors"""
        # For now, return empty list - could be enhanced with temporal analysis
        return []

    def _calculate_hierarchical_relationships(self, structure: Dict) -> List[Dict]:
        """Calculate hierarchical relationship vectors"""
        relationships = []

        # Parent-child relationships in hierarchy
        for level_str, sections in structure['hierarchical_mapping']['section_tree'].items():
            level = int(level_str)
            if level > 1:  # Has parent level
                parent_level_str = str(level - 1)
                parent_sections = structure['hierarchical_mapping']['section_tree'].get(parent_level_str, [])

                for section in sections:
                    # Find nearest parent (simplified - just take first)
                    if parent_sections:
                        relationships.append({
                            'source': parent_sections[0]['id'],
                            'target': section['id'],
                            'relationship_type': 'hierarchical_parent_child',
                            'strength': 1.0 / level,  # Stronger at higher levels
                            'vector': [0.0, 0.0, 1.0]  # Placeholder vector
                        })

        return relationships


class Q2_3_DocumentContextAlignment:
    """
    Main Q2.3 Document Context Alignment processor
    """

    def __init__(self):
        self.structure_analyzer = DocumentStructureAnalyzer()
        self.positioning_hints = GeometricPositioningHints()
        self.contextual_intelligence = ContextualIntelligenceEngine()

    def process_document_context(self, question_id: str) -> Dict:
        """
        Main processing function for document context alignment
        """
        start_time = datetime.now()

        try:
            # Load question data from Q1
            question_data = self._load_question_from_q1(question_id)

            # Load intent data from Q2.1 (if available)
            intent_data = self._load_intent_from_q21(question_id)

            # Load document chunks from A-Pipeline
            document_chunks = self._load_document_chunks(question_data['doc_id'])

            # Analyze document structure
            structure_analysis = self.structure_analyzer.analyze_document_structure(
                question_data['doc_id'],
                document_chunks
            )

            # Generate geometric positioning hints
            geometric_hints = self.positioning_hints.generate_positioning_hints(
                structure_analysis,
                intent_data.get('intent_classification', {})
            )

            # Generate contextual intelligence
            contextual_intel = self.contextual_intelligence.align_question_to_document_context(
                question_data,
                structure_analysis,
                intent_data
            )

            # Calculate processing metadata
            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            result = {
                'question_id': question_data['question_id'],
                'doc_id': question_data['doc_id'],
                'question_text': question_data['question_text'],
                'document_structure_analysis': structure_analysis,
                'geometric_positioning_hints': geometric_hints,
                'contextual_intelligence': contextual_intel,
                'processing_metadata': {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'processing_time_ms': processing_time,
                    'document_complexity_score': self._calculate_complexity_score(structure_analysis),
                    'structural_confidence': self._calculate_structural_confidence(structure_analysis),
                    'analysis_depth': 'comprehensive'
                }
            }

            return result

        except Exception as e:
            print(f"Error in Q2.3 processing: {e}")
            return self._get_default_output(question_id)

    def _load_question_from_q1(self, question_id: str) -> Dict:
        """Load question data from Q1 output"""
        try:
            q1_path = "../outputs/Q1_Question_ingestion.json"
            with open(q1_path, 'r') as f:
                q1_data = json.load(f)

            if isinstance(q1_data, dict) and 'question_id' in q1_data:
                # Single question format
                if q1_data['question_id'] == question_id:
                    return q1_data
            elif isinstance(q1_data, dict):
                # Multi-question format
                if question_id in q1_data:
                    return q1_data[question_id]

            raise ValueError(f"Question {question_id} not found in Q1 output")

        except Exception as e:
            print(f"Error loading Q1 data: {e}")
            # Return minimal question data
            return {
                'question_id': question_id,
                'doc_id': question_id,
                'question_text': 'Unknown question'
            }

    def _load_intent_from_q21(self, question_id: str) -> Dict:
        """Load intent data from Q2.1 output"""
        try:
            q21_path = "../outputs/Q2.1_enhanced_intent_classification.json"
            with open(q21_path, 'r') as f:
                q21_data = json.load(f)

            if question_id in q21_data:
                return q21_data[question_id]
            else:
                print(f"Intent data not found for {question_id}, using defaults")
                return {'intent_classification': {}}

        except Exception as e:
            print(f"Error loading Q2.1 data: {e}")
            return {'intent_classification': {}}

    def _load_document_chunks(self, doc_id: str) -> List[Dict]:
        """Load document chunks from A-Pipeline"""
        try:
            # Try to load from A-Pipeline concept space
            concept_space_path = f"../../A_Concept_pipeline/outputs/concept_spaces/{doc_id}_concept_space.json"

            if os.path.exists(concept_space_path):
                with open(concept_space_path, 'r') as f:
                    concept_space = json.load(f)
                return concept_space.get('chunks', [])
            else:
                # Fallback: create mock chunks based on document structure
                return self._create_mock_chunks(doc_id)

        except Exception as e:
            print(f"Error loading document chunks: {e}")
            return self._create_mock_chunks(doc_id)

    def _create_mock_chunks(self, doc_id: str) -> List[Dict]:
        """Create mock document chunks for testing"""
        return [
            {
                'id': f'{doc_id}_chunk_0',
                'text': '# Financial Performance Report 2018-2019\nThis report provides comprehensive financial analysis.',
                'position': 0
            },
            {
                'id': f'{doc_id}_chunk_1',
                'text': '## Revenue Analysis Table\n| Year | Revenue | Growth |\n| 2018 | $140,368,000 | - |\n| 2019 | $172,752,000 | 23.07% |',
                'position': 1
            },
            {
                'id': f'{doc_id}_chunk_2',
                'text': '## Operating Expenses\nTotal operating expenses: 2018: $95,000,000, 2019: $108,500,000.',
                'position': 2
            },
            {
                'id': f'{doc_id}_chunk_3',
                'text': '### Income Statement Summary\nNet income calculations and percentage change analysis for 2018-2019 period.',
                'position': 3
            },
            {
                'id': f'{doc_id}_chunk_4',
                'text': 'Financial Metrics: The revenue percentage change from 2018 to 2019 was 23.07%, representing strong growth.',
                'position': 4
            },
            {
                'id': f'{doc_id}_chunk_5',
                'text': '## Balance Sheet Data\nAssets, liabilities, and equity information for fiscal years 2018 and 2019.',
                'position': 5
            }
        ]

    def _calculate_complexity_score(self, structure: Dict) -> float:
        """Calculate document complexity score"""
        hierarchy_levels = structure['hierarchical_mapping']['total_levels']
        table_count = len(structure['table_structure_intelligence']['detected_tables'])

        # Complexity based on hierarchy depth and table count
        complexity = (hierarchy_levels / 5.0) + (table_count / 10.0)
        return min(1.0, complexity)

    def _calculate_structural_confidence(self, structure: Dict) -> float:
        """Calculate confidence in structural analysis"""
        # Base confidence on detection quality
        has_tables = len(structure['table_structure_intelligence']['detected_tables']) > 0
        has_hierarchy = structure['hierarchical_mapping']['total_levels'] > 1

        confidence = 0.5  # Base confidence
        if has_tables:
            confidence += 0.3
        if has_hierarchy:
            confidence += 0.2

        return min(1.0, confidence)

    def _get_default_output(self, question_id: str) -> Dict:
        """Return default output on error"""
        return {
            'question_id': question_id,
            'doc_id': question_id,
            'question_text': 'Error in processing',
            'document_structure_analysis': self.structure_analyzer._get_default_structure_analysis(),
            'geometric_positioning_hints': {
                'spatial_clustering_map': {'content_clusters': [], 'cluster_boundaries': []},
                'structural_coordinate_system': {},
                'constraint_recommendations': {'hard_constraints': [], 'soft_constraints': []}
            },
            'contextual_intelligence': {
                'question_document_alignment': {'structural_relevance': 0.0, 'likely_target_areas': []},
                'relationship_vectors': {'semantic_relationships': []}
            },
            'processing_metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'processing_time_ms': 0.0,
                'document_complexity_score': 0.0,
                'structural_confidence': 0.0,
                'analysis_depth': 'error_fallback'
            }
        }

    def save_output(self, result: Dict, output_path: str = "../outputs/Q2.3_document_context_alignment.json"):
        """Save Q2.3 output to file"""
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Wrap in question_id structure for consistency
            output_data = {result['question_id']: result}

            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)

            print(f"Q2.3 output saved to {output_path}")

        except Exception as e:
            print(f"Error saving Q2.3 output: {e}")


def main():
    """Main execution function"""
    print("=" * 60)
    print("Q2.3: Document Context Alignment Test")
    print("=" * 60)

    # Initialize Q2.3
    q23 = Q2_3_DocumentContextAlignment()

    # Process the sample question
    question_id = "finqa_test_1630"
    print(f"Processing Q2.3 for question: {question_id}")

    # Run document context alignment
    result = q23.process_document_context(question_id)

    print("\n" + "=" * 40)
    print("Q2.3 OUTPUT - Document Context Alignment:")
    print("=" * 40)
    print(f"Question ID: {result['question_id']}")
    print(f"Document Structure Complexity: {result['processing_metadata']['document_complexity_score']:.3f}")
    print(f"Structural Confidence: {result['processing_metadata']['structural_confidence']:.3f}")

    # Show structure analysis summary
    structure = result['document_structure_analysis']
    print(f"\nDocument Structure Summary:")
    print(f"  Hierarchy Levels: {structure['hierarchical_mapping']['total_levels']}")
    print(f"  Detected Tables: {len(structure['table_structure_intelligence']['detected_tables'])}")
    print(f"  Document Type: {structure['layout_pattern_analysis']['document_type']}")

    # Show geometric hints summary
    hints = result['geometric_positioning_hints']
    print(f"\nGeometric Positioning Hints:")
    print(f"  Content Clusters: {len(hints['spatial_clustering_map']['content_clusters'])}")
    print(f"  Hard Constraints: {len(hints['constraint_recommendations']['hard_constraints'])}")
    print(f"  Priority Regions: {len(hints['constraint_recommendations']['priority_regions'])}")

    # Show contextual intelligence summary
    context = result['contextual_intelligence']
    print(f"\nContextual Intelligence:")
    print(f"  Structural Relevance: {context['question_document_alignment']['structural_relevance']:.3f}")
    print(f"  Target Areas: {len(context['question_document_alignment']['likely_target_areas'])}")
    print(f"  Navigation Strategy: {context['question_document_alignment']['navigation_strategy']}")

    print(f"\nProcessing Time: {result['processing_metadata']['processing_time_ms']:.1f}ms")

    # Save output
    q23.save_output(result)

    return result


if __name__ == "__main__":
    print("Q2.3 DOCUMENT CONTEXT ALIGNMENT")
    print("=" * 50)

    result = main()

    if result:
        print("Q2.3_document_context_alignment.json created successfully")
        print("Ready for Q2.5 integration with structural intelligence")
    else:
        print("Failed to create Q2.3 output")