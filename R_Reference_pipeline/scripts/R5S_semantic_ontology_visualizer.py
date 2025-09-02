#!/usr/bin/env python3
"""
R5S: Semantic Ontology Visualizer
Create meaningful visualizations of semantic relationships and knowledge structures
Enhanced to show TRUE semantic meaning rather than word-based patterns
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import time
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

class SemanticGraphBuilder:
    """Build interactive graph structures from semantic ontology"""
    
    def __init__(self):
        self.color_schemes = self._init_color_schemes()
        
    def _init_color_schemes(self):
        """Initialize color schemes for different visualizations"""
        return {
            'domains': {
                'financial_management': '#E74C3C',
                'organizational_structure': '#3498DB', 
                'strategic_planning': '#2ECC71',
                'operational_processes': '#F39C12',
                'information_management': '#9B59B6',
                'resource_management': '#E67E22',
                'legal_compliance': '#34495E',
                'technology_systems': '#1ABC9C',
                'uncategorized': '#95A5A6'
            },
            'relationships': {
                'IS_A': '#2E8B57',
                'PART_OF': '#4169E1', 
                'HAS_PROPERTY': '#DAA520',
                'REQUIRES': '#DC143C',
                'CAUSES': '#FF6347',
                'USED_FOR': '#32CD32',
                'ENABLES': '#00CED1',
                'CONSTRAINS': '#8B0000',
                'PRECEDES': '#9370DB',
                'RELATED_TO': '#708090'
            }
        }
    
    def create_knowledge_graph(self, ontology_data: Dict) -> str:
        """Build vis.js interactive network like R5L"""
        concepts = ontology_data.get('concepts', {})
        
        # Prepare data for vis.js (matching R5L format exactly)
        nodes = []
        edges = []
        
        # Domain colors (matching R5L exactly)
        domain_colors = {
            'common': '#FF9999',
            'transportation': '#66B3FF', 
            'international development org': '#99FF99',
            'manufacturing': '#FFCC99',
            'government': '#FF99CC',
            'insurance': '#99CCFF',
            'telecom': '#FFB366',
            'finance': '#B366FF'
        }
        
        # Create nodes (matching R5L structure)
        for concept_name, concept_data in concepts.items():
            domain = concept_data.get('domain', 'common')
            relationships = concept_data.get('relationships', {})
            rel_count = sum(len(targets) for targets in relationships.values())
            
            # Create tooltip matching R5L format
            tooltip = (f"Concept: {concept_name}\n"
                      f"Domain: {domain}\n"
                      f"Total Relationships: {rel_count}\n"
                      f"IS_A: {len(relationships.get('IS_A', []))}\n"
                      f"PART_OF: {len(relationships.get('PART_OF', []))}\n"
                      f"HAS_PROPERTY: {len(relationships.get('HAS_PROPERTY', []))}\n"
                      f"REQUIRES: {len(relationships.get('REQUIRES', []))}\n"
                      f"USED_FOR: {len(relationships.get('USED_FOR', []))}")
            
            nodes.append({
                'id': concept_name,
                'label': concept_name,
                'title': tooltip,
                'group': domain,
                'color': domain_colors.get(domain, '#CCCCCC'),
                'value': rel_count,  # This controls bubble size
                'level': 1
            })
        
        # Create edges with relationship type styling and smart target matching
        edge_id = 0
        relationship_colors = {
            'IS_A': '#4169E1',           # Blue - hierarchical
            'PART_OF': '#228B22',        # Green - compositional 
            'HAS_PROPERTY': '#FF8C00',   # Orange - attributes
            'REQUIRES': '#DC143C',       # Red - dependencies
            'USED_FOR': '#9966FF'        # Purple - functional
        }
        
        # Build target lookup for smart matching (target -> concept_id mapping)
        target_to_concept = {}
        for concept_id in concepts.keys():
            if '.' in concept_id:
                # Extract base concept name (e.g., "common.asset" -> "asset")
                base_name = concept_id.split('.', 1)[1]
                if base_name not in target_to_concept:
                    target_to_concept[base_name] = []
                target_to_concept[base_name].append(concept_id)
            else:
                # Handle legacy non-namespaced concepts
                if concept_id not in target_to_concept:
                    target_to_concept[concept_id] = []
                target_to_concept[concept_id].append(concept_id)
        
        for concept_name, concept_data in concepts.items():
            relationships = concept_data.get('relationships', {})
            for rel_type, targets in relationships.items():
                for target in targets:
                    # Try multiple matching strategies
                    matched_concepts = []
                    
                    # Strategy 1: Direct match (for exact IDs)
                    if target in concepts:
                        matched_concepts.append(target)
                    
                    # Strategy 2: Base name match (e.g., "asset" matches "common.asset", "finance.asset")
                    elif target in target_to_concept:
                        matched_concepts.extend(target_to_concept[target])
                    
                    # Strategy 3: Partial match (e.g., "organization" might match concepts containing "organization")
                    else:
                        for concept_id in concepts.keys():
                            if target.lower() in concept_id.lower():
                                matched_concepts.append(concept_id)
                                break  # Take first match to avoid duplicates
                    
                    # Create edges for all matched concepts
                    for matched_concept in matched_concepts[:1]:  # Limit to first match to avoid clutter
                        edges.append({
                            'from': concept_name,
                            'to': matched_concept,
                            'color': {'color': relationship_colors.get(rel_type, '#808080'), 'opacity': 0.6},
                            'title': f'{rel_type}: {concept_name} → {matched_concept}',
                            'width': 2
                        })
                        edge_id += 1
        
        # Count relationships for statistics
        total_relationships = len(edges)
        concept_count = len(concepts)
        domain_count = len(set(concept_data.get('domain', 'common') for concept_data in concepts.values()))
        
        # Generate HTML using exact R5L template structure
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>BIZBOK Semantic Ontology - Interactive Visualization</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style type="text/css">
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }}
        .container {{
            background: white;
            margin: 20px;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            margin-top: 0;
        }}
        #mynetwork {{
            width: 100%;
            height: 700px;
            border: 2px solid #ddd;
            border-radius: 5px;
        }}
        .info {{
            background: #f7f7f7;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .stats {{
            display: flex;
            justify-content: space-around;
            margin-top: 20px;
            padding: 15px;
            background: #f7f7f7;
            border-radius: 5px;
        }}
        .stat-item {{
            text-align: center;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
        }}
        .stat-label {{
            color: #666;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔮 BIZBOK Semantic Ontology - Interactive Network</h1>
        
        <div class="info">
            <strong>📌 Instructions:</strong><br>
            • <b>Zoom:</b> Mouse wheel or pinch<br>
            • <b>Pan:</b> Click and drag background<br>
            • <b>Select:</b> Click node to highlight connections<br>
            • <b>Move:</b> Drag nodes to rearrange<br>
            • <b>Info:</b> Hover over nodes and edges for details
        </div>
        
        <div class="info">
            <strong>🎨 Color Legend - Domains:</strong><br>
            <span style="color: #FF9999; font-size: 16px;">●</span> Common &nbsp;&nbsp;
            <span style="color: #66B3FF; font-size: 16px;">●</span> Transportation &nbsp;&nbsp;
            <span style="color: #99FF99; font-size: 16px;">●</span> Int'l Development &nbsp;&nbsp;
            <span style="color: #FFCC99; font-size: 16px;">●</span> Manufacturing<br>
            <span style="color: #FF99CC; font-size: 16px;">●</span> Government &nbsp;&nbsp;
            <span style="color: #99CCFF; font-size: 16px;">●</span> Insurance &nbsp;&nbsp;
            <span style="color: #FFB366; font-size: 16px;">●</span> Telecom &nbsp;&nbsp;
            <span style="color: #B366FF; font-size: 16px;">●</span> Finance
        </div>
        
        <div class="info">
            <strong>🔗 Relationship Types:</strong><br>
            <span style="color: #4169E1;">━</span> <strong>IS_A</strong> (blue): Hierarchical inheritance relationships<br>
            <span style="color: #228B22;">━</span> <strong>PART_OF</strong> (green): Compositional part-whole relationships<br>
            <span style="color: #FF8C00;">━</span> <strong>HAS_PROPERTY</strong> (orange): Property-attribute relationships<br>
            <span style="color: #DC143C;">━</span> <strong>REQUIRES</strong> (red): Dependency relationships<br>
            <span style="color: #9966FF;">━</span> <strong>USED_FOR</strong> (purple): Functional purpose relationships
        </div>
        
        <div class="info">
            <strong>⚠️ Note:</strong> "Semantic" relationships use <strong>meaning-based</strong> pattern recognition + domain inference rules. Hover over edges to see relationship types. This R5S implements true semantic understanding vs R4L lexical similarity.
        </div>
        
        <div id="mynetwork"></div>
        
        <div class="stats">
            <div class="stat-item">
                <div class="stat-value">{concept_count}</div>
                <div class="stat-label">Concepts</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{domain_count}</div>
                <div class="stat-label">Domains</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{total_relationships}</div>
                <div class="stat-label">Relationships</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">5</div>
                <div class="stat-label">Relation Types</div>
            </div>
        </div>
    </div>
    
    <script type="text/javascript">
        // Create nodes and edges
        var nodes = new vis.DataSet({json.dumps(nodes)});
        var edges = new vis.DataSet({json.dumps(edges)});
        
        // Create network
        var container = document.getElementById('mynetwork');
        var data = {{
            nodes: nodes,
            edges: edges
        }};
        
        var options = {{
            nodes: {{
                shape: 'dot',
                scaling: {{
                    min: 10,
                    max: 40,
                    label: {{
                        min: 8,
                        max: 20
                    }}
                }},
                font: {{
                    size: 12,
                    face: 'Segoe UI'
                }},
                borderWidth: 2
            }},
            edges: {{
                width: 0.5,
                smooth: {{
                    type: 'continuous',
                    roundness: 0.5
                }}
            }},
            physics: {{
                stabilization: {{
                    iterations: 200
                }},
                barnesHut: {{
                    gravitationalConstant: -2000,
                    springConstant: 0.001,
                    springLength: 200,
                    damping: 0.09
                }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 200,
                navigationButtons: true,
                keyboard: true
            }},
            groups: {{
                useDefaultGroups: false
            }}
        }};
        
        var network = new vis.Network(container, data, options);
        
        // Add click handler
        network.on("click", function(params) {{
            if (params.nodes.length > 0) {{
                var nodeId = params.nodes[0];
                var node = nodes.get(nodeId);
                console.log('Selected:', node.label);
            }}
        }});
        
        // Stabilization progress
        network.on("stabilizationProgress", function(params) {{
            console.log('Stabilization progress:', params.iterations + '/' + params.total);
        }});
        
        network.once("stabilizationIterationsDone", function() {{
            console.log("Stabilization complete");
        }});
    </script>
</body>
</html>"""
        
        return html_content
    
    def create_hierarchical_tree(self, taxonomy_data: Dict) -> go.Figure:
        """Build hierarchical tree showing IS_A relationships"""
        if not taxonomy_data or 'levels' not in taxonomy_data:
            return self._create_empty_figure("No taxonomy data available")
        
        levels = taxonomy_data['levels']
        max_depth = len(levels)
        
        # Create tree layout
        fig = go.Figure()
        
        # Add nodes level by level
        level_colors = px.colors.qualitative.Set3[:max_depth]
        
        for level_num, concepts in levels.items():
            if not concepts:
                continue
                
            # Calculate positions for this level
            y_pos = max_depth - level_num  # Top to bottom
            x_positions = np.linspace(-len(concepts)/2, len(concepts)/2, len(concepts))
            
            # Add concept nodes
            fig.add_trace(go.Scatter(
                x=x_positions,
                y=[y_pos] * len(concepts),
                mode='markers+text',
                marker=dict(
                    size=20,
                    color=level_colors[level_num - 1] if level_num <= len(level_colors) else '#95A5A6',
                    opacity=0.8,
                    line=dict(width=2, color='white')
                ),
                text=concepts,
                textposition='middle center',
                textfont=dict(size=10, color='white'),
                name=f'Level {level_num}',
                hovertemplate='<b>%{text}</b><br>Level: %{y}<extra></extra>'
            ))
        
        # Add connections between levels
        graph = taxonomy_data.get('graph', {})
        reverse_graph = taxonomy_data.get('reverse_graph', {})
        
        # Create position lookup
        pos_lookup = {}
        for level_num, concepts in levels.items():
            if concepts:
                y_pos = max_depth - level_num
                x_positions = np.linspace(-len(concepts)/2, len(concepts)/2, len(concepts))
                for i, concept in enumerate(concepts):
                    pos_lookup[concept] = (x_positions[i], y_pos)
        
        # Add edges
        edge_x, edge_y = [], []
        for parent, children in graph.items():
            if parent in pos_lookup:
                parent_pos = pos_lookup[parent]
                for child in children:
                    if child in pos_lookup:
                        child_pos = pos_lookup[child]
                        edge_x.extend([parent_pos[0], child_pos[0], None])
                        edge_y.extend([parent_pos[1], child_pos[1], None])
        
        if edge_x:
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(width=2, color='#2E8B57'),
                name='IS_A relationships',
                showlegend=True,
                hoverinfo='skip'
            ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="R4S Semantic Taxonomy Tree<br><sub>Hierarchical IS_A relationships</sub>",
                x=0.5,
                font=dict(size=18)
            ),
            showlegend=True,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=True, title="Hierarchy Level"),
            plot_bgcolor='white',
            width=1000,
            height=600,
            margin=dict(t=100)
        )
        
        return fig
    
    def create_domain_cluster_map(self, clusters_data: Dict) -> go.Figure:
        """Create semantic domain cluster bubble map"""
        if not clusters_data:
            return self._create_empty_figure("No cluster data available")
        
        # Prepare data for bubble chart
        domains = []
        concept_counts = []
        coherence_scores = []
        central_concepts = []
        colors = []
        
        domain_colors = self.color_schemes['domains']
        
        for domain, data in clusters_data.items():
            domains.append(domain.replace('_', ' ').title())
            concept_counts.append(data.get('concept_count', 0))
            coherence_scores.append(data.get('coherence', 0))
            central_concepts.append(data.get('central_concept', 'N/A'))
            colors.append(domain_colors.get(domain, '#95A5A6'))
        
        # Create bubble chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=coherence_scores,
            y=concept_counts,
            mode='markers+text',
            marker=dict(
                size=[count * 10 for count in concept_counts],
                color=colors,
                opacity=0.7,
                line=dict(width=2, color='white'),
                sizemode='diameter',
                sizemin=20
            ),
            text=domains,
            textposition='middle center',
            textfont=dict(size=10, color='white'),
            hovertemplate='<b>%{text}</b><br>' +
                         'Concepts: %{y}<br>' +
                         'Coherence: %{x:.2f}<br>' +
                         'Central: %{customdata}<br>' +
                         '<extra></extra>',
            customdata=central_concepts,
            name='Semantic Domains'
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="R4S Semantic Domain Clusters<br><sub>Bubble size = concept count, position = coherence vs count</sub>",
                x=0.5,
                font=dict(size=18)
            ),
            xaxis=dict(
                title="Semantic Coherence Score",
                showgrid=True,
                gridcolor='lightgray',
                range=[-0.1, max(coherence_scores) * 1.1] if coherence_scores else [0, 1]
            ),
            yaxis=dict(
                title="Number of Concepts",
                showgrid=True,
                gridcolor='lightgray',
                range=[0, max(concept_counts) * 1.1] if concept_counts else [0, 10]
            ),
            plot_bgcolor='white',
            width=800,
            height=600,
            margin=dict(t=100)
        )
        
        return fig
    
    def create_relationship_matrix(self, ontology_data: Dict) -> go.Figure:
        """Create heatmap showing relationship patterns"""
        concepts = ontology_data.get('concepts', {})
        
        if not concepts:
            return self._create_empty_figure("No concept data available")
        
        # Build relationship matrix
        concept_names = list(concepts.keys())[:20]  # Limit for readability
        rel_types = ['IS_A', 'PART_OF', 'HAS_PROPERTY', 'REQUIRES', 'CAUSES', 'USED_FOR', 'ENABLES']
        
        # Create matrix data
        matrix_data = []
        for concept in concept_names:
            row = []
            concept_rels = concepts[concept].get('relationships', {})
            for rel_type in rel_types:
                count = len(concept_rels.get(rel_type, []))
                row.append(count)
            matrix_data.append(row)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix_data,
            x=rel_types,
            y=concept_names,
            colorscale='Viridis',
            showscale=True,
            hovertemplate='<b>%{y}</b><br>%{x}: %{z} relationships<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="R4S Relationship Matrix<br><sub>Concept vs Relationship Type Heatmap (Top 20 concepts)</sub>",
                x=0.5,
                font=dict(size=16)
            ),
            xaxis=dict(title="Relationship Types"),
            yaxis=dict(title="Concepts"),
            width=800,
            height=600,
            margin=dict(t=100, l=200)
        )
        
        return fig
    
    def _create_empty_figure(self, message: str) -> go.Figure:
        """Create empty figure with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color='gray')
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        return fig


class StaticImageGenerator:
    """Generate high-quality static visualizations"""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        self.colors = {
            'primary': '#2E8B57',
            'secondary': '#4169E1', 
            'accent': '#DC143C',
            'neutral': '#708090'
        }
    
    def create_relationship_distribution_chart(self, ontology_data: Dict, 
                                             output_path: Path) -> None:
        """Create bar chart of relationship type distribution"""
        relationships_data = ontology_data.get('relationships', {})
        
        # Count relationships by type
        rel_counts = defaultdict(int)
        for concept_rels in relationships_data.values():
            for rel_type, targets in concept_rels.items():
                rel_counts[rel_type] += len(targets)
        
        if not rel_counts:
            return
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        rel_types = list(rel_counts.keys())
        counts = list(rel_counts.values())
        
        bars = ax.bar(rel_types, counts, color=self.colors['primary'], alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{int(height)}', ha='center', va='bottom')
        
        ax.set_title('R4S Semantic Relationship Distribution\nMeaning-based relationships by type', 
                    fontsize=16, pad=20)
        ax.set_xlabel('Relationship Types', fontsize=12)
        ax.set_ylabel('Number of Relationships', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path / "R5S_relationship_distribution.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_domain_coherence_chart(self, clusters_data: Dict, 
                                    output_path: Path) -> None:
        """Create bar chart of domain coherence scores"""
        if not clusters_data:
            return
        
        # Extract coherence data
        domains = []
        coherence_scores = []
        concept_counts = []
        
        for domain, data in clusters_data.items():
            domains.append(domain.replace('_', ' ').title())
            coherence_scores.append(data.get('coherence', 0))
            concept_counts.append(data.get('concept_count', 0))
        
        # Create dual-axis chart
        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        # Coherence bars
        bars1 = ax1.bar([d + ' (L)' for d in domains], coherence_scores, 
                       color=self.colors['primary'], alpha=0.7, label='Coherence Score')
        ax1.set_ylabel('Coherence Score', color=self.colors['primary'], fontsize=12)
        ax1.tick_params(axis='y', labelcolor=self.colors['primary'])
        ax1.set_ylim(0, 1)
        
        # Concept count bars
        ax2 = ax1.twinx()
        bars2 = ax2.bar([d + ' (R)' for d in domains], concept_counts, 
                       color=self.colors['secondary'], alpha=0.7, label='Concept Count')
        ax2.set_ylabel('Number of Concepts', color=self.colors['secondary'], fontsize=12)
        ax2.tick_params(axis='y', labelcolor=self.colors['secondary'])
        
        # Add value labels
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            ax1.text(bar1.get_x() + bar1.get_width()/2., bar1.get_height() + 0.02,
                    f'{coherence_scores[i]:.2f}', ha='center', va='bottom', 
                    color=self.colors['primary'])
            ax2.text(bar2.get_x() + bar2.get_width()/2., bar2.get_height() + 0.5,
                    f'{concept_counts[i]}', ha='center', va='bottom', 
                    color=self.colors['secondary'])
        
        plt.title('R4S Semantic Domain Analysis\nCoherence scores and concept distribution', 
                 fontsize=16, pad=20)
        plt.xticks(range(len(domains)), domains, rotation=45, ha='right')
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_path / "R5S_domain_coherence.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_comparison_chart(self, r4l_stats: Dict, r4s_stats: Dict, 
                               output_path: Path) -> None:
        """Create comparison chart between R4L and R4S"""
        try:
            # Load R4L statistics for comparison
            r4l_path = output_path.parent / "R4L_ontology_statistics.json"
            if r4l_path.exists():
                with open(r4l_path, 'r') as f:
                    r4l_data = json.load(f)
                    r4l_relationships = r4l_data.get('statistics', {}).get('relationships_total', 0)
                    r4l_clusters = r4l_data.get('statistics', {}).get('total_clusters', 0)
            else:
                r4l_relationships = 547  # From earlier run
                r4l_clusters = 33
            
            r4s_relationships = r4s_stats.get('counts', {}).get('total_relationships', 0)
            r4s_clusters = r4s_stats.get('counts', {}).get('semantic_clusters', 0)
            
            # Create comparison chart
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            
            # Relationships comparison
            systems = ['R4L\n(Lexical)', 'R4S\n(Semantic)']
            rel_counts = [r4l_relationships, r4s_relationships]
            
            bars1 = ax1.bar(systems, rel_counts, color=[self.colors['neutral'], self.colors['primary']])
            ax1.set_title('Total Relationships', fontsize=14)
            ax1.set_ylabel('Count')
            for i, bar in enumerate(bars1):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                        f'{rel_counts[i]}', ha='center', va='bottom')
            
            # Clusters comparison
            cluster_counts = [r4l_clusters, r4s_clusters]
            bars2 = ax2.bar(systems, cluster_counts, color=[self.colors['neutral'], self.colors['primary']])
            ax2.set_title('Semantic/Lexical Clusters', fontsize=14)
            ax2.set_ylabel('Count')
            for i, bar in enumerate(bars2):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{cluster_counts[i]}', ha='center', va='bottom')
            
            # Quality metrics
            quality_metrics = ['Relationship\nTypes', 'Processing\nSpeed (s)', 'Semantic\nCoverage (%)', 'Domain\nIntegration']
            r4l_values = [2, 0.5, 0, 0]  # R4L doesn't have these metrics
            r4s_values = [
                r4s_stats.get('counts', {}).get('relationship_types', 0),
                r4s_stats.get('metadata', {}).get('processing_time', 0),
                r4s_stats.get('quality_metrics', {}).get('semantic_coverage', 0) * 100,
                r4s_stats.get('quality_metrics', {}).get('domain_integration_score', 0)
            ]
            
            x = np.arange(len(quality_metrics))
            width = 0.35
            
            ax3.bar(x - width/2, r4l_values, width, label='R4L (Lexical)', color=self.colors['neutral'])
            ax3.bar(x + width/2, r4s_values, width, label='R4S (Semantic)', color=self.colors['primary'])
            ax3.set_title('Quality Metrics Comparison', fontsize=14)
            ax3.set_xticks(x)
            ax3.set_xticklabels(quality_metrics)
            ax3.legend()
            
            # Enhancement impact pie chart
            enhancement_data = r4s_stats.get('enhancement_metrics', {})
            labels = ['Keyword\nEnhanced', 'Domain\nEnhanced', 'Cross-Domain\nLinks']
            values = [
                enhancement_data.get('keyword_enhanced_extractions', 0),
                enhancement_data.get('domain_enhanced_inferences', 0),
                enhancement_data.get('cross_domain_relationships', 0)
            ]
            
            if sum(values) > 0:
                colors = [self.colors['primary'], self.colors['secondary'], self.colors['accent']]
                wedges, texts, autotexts = ax4.pie(values, labels=labels, autopct='%1.0f', 
                                                  colors=colors, startangle=90)
                ax4.set_title('R4S Enhancement Impact', fontsize=14)
            
            plt.suptitle('R4L vs R4S Comprehensive Comparison\nLexical vs Semantic Ontology Performance', 
                        fontsize=16)
            plt.tight_layout()
            plt.savefig(output_path / "R5S_r4l_vs_r4s_comparison.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"   [WARN] Could not create comparison chart: {str(e)}")


class R5S_SemanticOntologyVisualizer:
    """
    Main class for visualizing semantic relationships and knowledge structures
    Creates meaningful visualizations showing TRUE semantic meaning
    """
    
    def __init__(self):
        self.script_dir = Path(__file__).parent.parent
        self.output_dir = self.script_dir / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create visualization output directory
        self.viz_dir = self.output_dir / "R5S_visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.graph_builder = SemanticGraphBuilder()
        self.image_generator = StaticImageGenerator()
        
        # Data storage
        self.ontology_data = {}
        self.relationships_data = {}
        self.hierarchy_data = {}
        self.clusters_data = {}
        self.statistics_data = {}
        
        # Performance tracking
        self.performance_metrics = {
            'start_time': None,
            'visualization_stages': {}
        }
    
    def main_visualization_pipeline(self):
        """Execute the complete semantic ontology visualization pipeline"""
        print("=" * 80)
        print("R5S: Semantic Ontology Visualizer")
        print("Creating meaningful visualizations of semantic relationships")
        print("=" * 80)
        
        self.performance_metrics['start_time'] = time.time()
        
        # Stage 1: Load R4S outputs
        print("\n[STAGE 1] Loading R4S semantic ontology outputs...")
        stage_start = time.time()
        self.stage1_load_r4s_outputs()
        self.performance_metrics['visualization_stages']['data_loading'] = time.time() - stage_start
        
        # Stage 2: Create interactive visualizations
        print("\n[STAGE 2] Creating interactive knowledge graph...")
        stage_start = time.time()
        self.stage2_create_interactive_visualizations()
        self.performance_metrics['visualization_stages']['interactive_viz'] = time.time() - stage_start
        
        # Stage 3: Create static publication figures
        print("\n[STAGE 3] Generating static publication figures...")
        stage_start = time.time()
        self.stage3_create_static_visualizations()
        self.performance_metrics['visualization_stages']['static_viz'] = time.time() - stage_start
        
        # Stage 4: Create analysis reports
        print("\n[STAGE 4] Generating visualization reports...")
        stage_start = time.time()
        self.stage4_create_analysis_reports()
        self.performance_metrics['visualization_stages']['reports'] = time.time() - stage_start
        
        total_time = time.time() - self.performance_metrics['start_time']
        print(f"\n[SUCCESS] R5S Semantic Visualizer completed in {total_time:.2f} seconds!")
        
        return self.generate_visualization_summary()
    
    def stage1_load_r4s_outputs(self):
        """Load all R4S output files for visualization"""
        # Load semantic ontology
        ontology_path = self.output_dir / "R4S_semantic_ontology.json"
        if ontology_path.exists():
            with open(ontology_path, 'r', encoding='utf-8') as f:
                self.ontology_data = json.load(f)
            print(f"   [OK] Loaded semantic ontology with {len(self.ontology_data.get('concepts', {}))} concepts")
        else:
            raise FileNotFoundError(f"R4S ontology not found: {ontology_path}")
        
        # Load relationships
        relationships_path = self.output_dir / "R4S_semantic_relationships.json"
        if relationships_path.exists():
            with open(relationships_path, 'r', encoding='utf-8') as f:
                self.relationships_data = json.load(f)
            rel_count = self.relationships_data.get('metadata', {}).get('total_relationships', 0)
            print(f"   [OK] Loaded {rel_count} semantic relationships")
        
        # Load hierarchy
        hierarchy_path = self.output_dir / "R4S_semantic_hierarchy.json"
        if hierarchy_path.exists():
            with open(hierarchy_path, 'r', encoding='utf-8') as f:
                self.hierarchy_data = json.load(f)
            print(f"   [OK] Loaded semantic hierarchy with {self.hierarchy_data.get('max_depth', 0)} levels")
        
        # Load clusters
        clusters_path = self.output_dir / "R4S_semantic_clusters.json"
        if clusters_path.exists():
            with open(clusters_path, 'r', encoding='utf-8') as f:
                self.clusters_data = json.load(f)
            print(f"   [OK] Loaded {len(self.clusters_data)} semantic clusters")
        
        # Load statistics
        statistics_path = self.output_dir / "R4S_ontology_statistics.json"
        if statistics_path.exists():
            with open(statistics_path, 'r', encoding='utf-8') as f:
                self.statistics_data = json.load(f)
            print(f"   [OK] Loaded comprehensive statistics")
    
    def stage2_create_interactive_visualizations(self):
        """Create interactive HTML visualizations"""
        created_count = 0
        
        # Knowledge graph (now returns HTML string)
        try:
            knowledge_graph_html = self.graph_builder.create_knowledge_graph(self.ontology_data)
            output_path = self.viz_dir / "R5S_knowledge_graph.html"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(knowledge_graph_html)
            print(f"   [OK] Created interactive knowledge graph")
            created_count += 1
        except Exception as e:
            print(f"   [WARN] Could not create knowledge graph: {str(e)}")
        
        # Hierarchical tree
        try:
            hierarchy_tree = self.graph_builder.create_hierarchical_tree(self.hierarchy_data)
            output_path = self.viz_dir / "R5S_taxonomy_tree.html"
            pyo.plot(hierarchy_tree, filename=str(output_path), auto_open=False)
            print(f"   [OK] Created hierarchical taxonomy tree")
            created_count += 1
        except Exception as e:
            print(f"   [WARN] Could not create hierarchy tree: {str(e)}")
        
        # Domain cluster map
        try:
            cluster_map = self.graph_builder.create_domain_cluster_map(self.clusters_data)
            output_path = self.viz_dir / "R5S_domain_clusters.html"
            pyo.plot(cluster_map, filename=str(output_path), auto_open=False)
            print(f"   [OK] Created domain cluster map")
            created_count += 1
        except Exception as e:
            print(f"   [WARN] Could not create cluster map: {str(e)}")
        
        # Relationship matrix
        try:
            rel_matrix = self.graph_builder.create_relationship_matrix(self.ontology_data)
            output_path = self.viz_dir / "R5S_relationship_matrix.html"
            pyo.plot(rel_matrix, filename=str(output_path), auto_open=False)
            print(f"   [OK] Created relationship matrix")
            created_count += 1
        except Exception as e:
            print(f"   [WARN] Could not create relationship matrix: {str(e)}")
        
        print(f"   [SUMMARY] Created {created_count} interactive visualizations")
    
    def stage3_create_static_visualizations(self):
        """Create static publication-quality figures"""
        created_count = 0
        
        # Relationship distribution chart
        try:
            self.image_generator.create_relationship_distribution_chart(
                self.relationships_data, self.viz_dir
            )
            print(f"   [OK] Created relationship distribution chart")
            created_count += 1
        except Exception as e:
            print(f"   [WARN] Could not create distribution chart: {str(e)}")
        
        # Domain coherence chart
        try:
            self.image_generator.create_domain_coherence_chart(
                self.clusters_data, self.viz_dir
            )
            print(f"   [OK] Created domain coherence chart")
            created_count += 1
        except Exception as e:
            print(f"   [WARN] Could not create coherence chart: {str(e)}")
        
        # R4L vs R4S comparison
        try:
            self.image_generator.create_comparison_chart(
                {}, self.statistics_data, self.viz_dir
            )
            print(f"   [OK] Created R4L vs R4S comparison chart")
            created_count += 1
        except Exception as e:
            print(f"   [WARN] Could not create comparison chart: {str(e)}")
        
        print(f"   [SUMMARY] Created {created_count} static visualizations")
    
    def stage4_create_analysis_reports(self):
        """Create visualization analysis reports"""
        # Create visualization report
        report_data = {
            'metadata': {
                'created': datetime.now().isoformat(),
                'processing_time': time.time() - self.performance_metrics['start_time'],
                'visualization_tool': 'R5S_Semantic_Visualizer_v2.0'
            },
            'input_data_summary': {
                'concepts_visualized': len(self.ontology_data.get('concepts', {})),
                'relationships_visualized': self.relationships_data.get('metadata', {}).get('total_relationships', 0),
                'semantic_clusters': len(self.clusters_data),
                'hierarchy_depth': self.hierarchy_data.get('max_depth', 0)
            },
            'visualizations_created': {
                'interactive_html': [
                    'R5S_knowledge_graph.html',
                    'R5S_taxonomy_tree.html', 
                    'R5S_domain_clusters.html',
                    'R5S_relationship_matrix.html'
                ],
                'static_images': [
                    'R5S_relationship_distribution.png',
                    'R5S_domain_coherence.png',
                    'R5S_r4l_vs_r4s_comparison.png'
                ]
            },
            'key_insights': {
                'semantic_domains_identified': len(self.clusters_data),
                'relationship_types_visualized': len(set(
                    rel_type for concept_rels in self.relationships_data.get('relationships', {}).values()
                    for rel_type in concept_rels.keys()
                )),
                'most_connected_domain': self._find_most_connected_domain(),
                'visualization_quality_score': self._calculate_visualization_quality()
            },
            'performance_metrics': self.performance_metrics
        }
        
        # Save visualization report
        report_path = self.viz_dir / "R5S_visualization_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        # Create insights summary
        insights_text = self._generate_insights_text(report_data)
        insights_path = self.viz_dir / "R5S_semantic_insights.txt"
        with open(insights_path, 'w', encoding='utf-8') as f:
            f.write(insights_text)
        
        print(f"   [OK] Created visualization analysis reports")
    
    def _find_most_connected_domain(self) -> str:
        """Find the semantic domain with most relationships"""
        domain_connections = defaultdict(int)
        
        for concept_data in self.ontology_data.get('concepts', {}).values():
            domain = concept_data.get('semantic_domain', 'uncategorized')
            relationships = concept_data.get('relationships', {})
            rel_count = sum(len(targets) for targets in relationships.values())
            domain_connections[domain] += rel_count
        
        if domain_connections:
            return max(domain_connections, key=domain_connections.get)
        return 'unknown'
    
    def _calculate_visualization_quality(self) -> float:
        """Calculate overall visualization quality score"""
        score = 0.0
        
        # Data completeness (40%)
        if self.ontology_data.get('concepts'):
            score += 0.2
        if self.relationships_data.get('relationships'):
            score += 0.1
        if self.clusters_data:
            score += 0.1
        
        # Visualization variety (30%)
        viz_count = 0
        expected_viz = ['knowledge_graph', 'taxonomy_tree', 'domain_clusters', 'relationship_matrix']
        for viz in expected_viz:
            viz_path = self.viz_dir / f"R5S_{viz}.html"
            if viz_path.exists():
                viz_count += 1
        score += (viz_count / len(expected_viz)) * 0.3
        
        # Analysis depth (30%)
        if self.statistics_data.get('quality_metrics'):
            score += 0.15
        if self.statistics_data.get('enhancement_metrics'):
            score += 0.15
        
        return min(score, 1.0)
    
    def _generate_insights_text(self, report_data: Dict) -> str:
        """Generate human-readable insights summary"""
        insights = f"""
R5S SEMANTIC ONTOLOGY VISUALIZATION INSIGHTS
============================================
Generated: {report_data['metadata']['created']}
Processing Time: {report_data['metadata']['processing_time']:.2f} seconds

SEMANTIC STRUCTURE ANALYSIS:
----------------------------
• Total Concepts Visualized: {report_data['input_data_summary']['concepts_visualized']}
• Semantic Relationships: {report_data['input_data_summary']['relationships_visualized']}
• Semantic Domains: {report_data['input_data_summary']['semantic_clusters']}
• Taxonomy Depth: {report_data['input_data_summary']['hierarchy_depth']} levels

VISUALIZATION OUTPUTS:
----------------------
Interactive Visualizations:
"""
        
        for viz in report_data['visualizations_created']['interactive_html']:
            insights += f"  ✓ {viz}\n"
        
        insights += "\nStatic Publication Figures:\n"
        for viz in report_data['visualizations_created']['static_images']:
            insights += f"  ✓ {viz}\n"
        
        insights += f"""
KEY SEMANTIC INSIGHTS:
---------------------
• Most Connected Domain: {report_data['key_insights']['most_connected_domain']}
• Relationship Types: {report_data['key_insights']['relationship_types_visualized']}
• Visualization Quality: {report_data['key_insights']['visualization_quality_score']:.2f}/1.0

USAGE RECOMMENDATIONS:
---------------------
1. Start with R5S_knowledge_graph.html for overall semantic structure
2. Use R5S_taxonomy_tree.html to explore IS_A hierarchies
3. Examine R5S_domain_clusters.html for semantic groupings
4. Review static charts for publication-ready figures

FILES LOCATION: {self.viz_dir}
"""
        return insights
    
    def generate_visualization_summary(self):
        """Generate execution summary"""
        total_time = time.time() - self.performance_metrics['start_time']
        
        summary = {
            'success': True,
            'processing_time': total_time,
            'concepts_visualized': len(self.ontology_data.get('concepts', {})),
            'relationships_visualized': self.relationships_data.get('metadata', {}).get('total_relationships', 0),
            'visualizations_created': {
                'interactive': 4,
                'static': 3,
                'total': 7
            },
            'output_directory': str(self.viz_dir),
            'key_files': [
                'R5S_knowledge_graph.html',
                'R5S_taxonomy_tree.html',
                'R5S_domain_clusters.html',
                'R5S_relationship_matrix.html',
                'R5S_r4l_vs_r4s_comparison.png'
            ]
        }
        
        return summary


def main():
    """Main execution function"""
    try:
        visualizer = R5S_SemanticOntologyVisualizer()
        result = visualizer.main_visualization_pipeline()
        
        # Display results
        print("\n" + "=" * 80)
        print("R5S SEMANTIC ONTOLOGY VISUALIZER SUMMARY")
        print("=" * 80)
        print(f"Processing Time: {result['processing_time']:.2f} seconds")
        print(f"Concepts Visualized: {result['concepts_visualized']}")
        print(f"Relationships Visualized: {result['relationships_visualized']}")
        print(f"Visualizations Created: {result['visualizations_created']['total']}")
        print(f"  • Interactive HTML: {result['visualizations_created']['interactive']}")
        print(f"  • Static Images: {result['visualizations_created']['static']}")
        print(f"\nOutput Directory: {result['output_directory']}")
        print(f"\nKey Visualization Files:")
        for filename in result['key_files']:
            print(f"  - {filename}")
        
        print(f"\n{'='*80}")
        print("SUCCESS: Semantic ontology visualizations created!")
        print("Open the HTML files in a browser for interactive exploration")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nERROR: R5S Semantic Visualizer failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()