#!/usr/bin/env python3
"""
I3: Tri-Semantic Visualizer
Inter-Space Pipeline Component for visualization of I1 Cross-Pipeline Semantic Integration
Visualizes dynamic integration of Reference, Document, and Question semantic spaces
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import webbrowser
import http.server
import socketserver
import threading
from collections import defaultdict
import math
import colorsys

# Add I1 integration path
sys.path.append(str(Path(__file__).parent))

try:
    from I1_cross_pipeline_semantic_integrator import I1_CrossPipelineSemanticIntegrator
    from I1_semantic_fusion_engine import I1_SemanticFusionEngine
except ImportError:
    print("[WARNING]  Warning: I1 components not found. Visualization will show available data only.")
    I1_CrossPipelineSemanticIntegrator = None
    I1_SemanticFusionEngine = None

class I3_TriSemanticVisualizer:
    """
    Revolutionary Tri-Semantic Visualization System
    
    Creates dynamic interactive visualizations of:
    1. Tri-Semantic Integration Networks (Ontology + Document + Question)
    2. Cross-Pipeline Semantic Bridges
    3. Fusion Process Visualizations
    4. Enhancement Quality Heatmaps
    5. Real-time Integration Analytics
    """
    
    def __init__(self):
        """Initialize I3 Tri-Semantic Visualizer"""
        self.i1_integrator = None
        if I1_CrossPipelineSemanticIntegrator:
            try:
                self.i1_integrator = I1_CrossPipelineSemanticIntegrator()
                print("✓ I1 Cross-Pipeline Semantic Integrator initialized")
            except Exception as e:
                print(f"[WARNING]  I1 initialization warning: {e}")
        
        # Visualization configurations
        self.visualization_types = {
            'tri_semantic_network': 'Interactive network showing three semantic spaces',
            'cross_pipeline_bridges': 'Semantic bridges between pipelines',
            'fusion_process_flow': 'Dynamic fusion process visualization',
            'enhancement_heatmap': 'Quality enhancement analysis',
            'integration_analytics': 'Real-time integration metrics'
        }
        
        # Color schemes for different semantic spaces
        self.semantic_colors = {
            'ontology_space': '#FF6B6B',      # Vibrant Red - Ontological Knowledge
            'document_space': '#4ECDC4',      # Teal Green - Document Context
            'question_space': '#45B7D1',      # Sky Blue - Question Understanding
            'fusion_nodes': '#96CEB4',        # Sage Green - Fusion Results
            'bridge_connections': '#FECA57',   # Warm Yellow - Cross-space bridges
            'enhancement_positive': '#6BCF7F', # Success Green - Positive enhancements
            'enhancement_negative': '#FF7675'  # Warning Red - Areas needing attention
        }
        
        # Node sizes based on semantic importance
        self.node_sizes = {
            'primary_concept': 25,
            'secondary_concept': 18,
            'supporting_concept': 12,
            'fusion_result': 30,
            'bridge_node': 15
        }
        
        # Edge weights for different relationship types
        self.edge_weights = {
            'ontological_relationship': 3,
            'document_semantic_link': 2,
            'question_alignment': 2,
            'cross_space_bridge': 4,
            'fusion_connection': 5
        }
    
    def load_tri_semantic_data(self) -> Dict[str, Any]:
        """
        Load tri-semantic data from all pipeline outputs
        
        Returns:
            Comprehensive tri-semantic dataset
        """
        data = {
            'ontology_data': {},
            'document_data': {},
            'question_data': {},
            'integration_data': {},
            'fusion_results': {}
        }
        
        script_dir = Path(__file__).parent.parent
        
        # Load Ontology (R-Pipeline) data
        r4l_path = script_dir / "outputs/R4L_lexical_ontology_output.json"
        if r4l_path.exists():
            with open(r4l_path, 'r', encoding='utf-8') as f:
                data['ontology_data'] = json.load(f)
        
        # Load I1 integration data
        i1_path = script_dir / "outputs/I1_cross_pipeline_integration_output.json"
        if i1_path.exists():
            with open(i1_path, 'r', encoding='utf-8') as f:
                data['integration_data'] = json.load(f)
        
        # Load Document (A-Pipeline) data
        a_pipeline_dir = script_dir.parent / "A_Concept_pipeline/outputs"
        a2_9_path = a_pipeline_dir / "A2.9_r4x_semantic_enhancement_output.json"
        if a2_9_path.exists():
            with open(a2_9_path, 'r', encoding='utf-8') as f:
                data['document_data'] = json.load(f)
        
        # Load Question (B-Pipeline) data
        b_pipeline_dir = script_dir.parent / "B_Retrieval_pipeline/outputs"
        b5_1_path = b_pipeline_dir / "B5_1_r4x_question_understanding_output.json"
        if b5_1_path.exists():
            with open(b5_1_path, 'r', encoding='utf-8') as f:
                data['question_data'] = json.load(f)
        
        # Load fusion results
        fusion_path = script_dir / "outputs/I1_semantic_fusion_results.json"
        if fusion_path.exists():
            with open(fusion_path, 'r', encoding='utf-8') as f:
                data['fusion_results'] = json.load(f)
        
        return data
    
    def create_tri_semantic_network_data(self, tri_semantic_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create network data for tri-semantic visualization
        
        Args:
            tri_semantic_data: Complete tri-semantic dataset
            
        Returns:
            Network data structure for visualization
        """
        nodes = []
        edges = []
        node_id_counter = 0
        node_map = {}
        
        # Process Ontology Space Nodes
        ontology_data = tri_semantic_data.get('ontology_data', {})
        if 'lexical_ontology' in ontology_data:
            ontology_concepts = ontology_data['lexical_ontology'].get('concepts', {})
            for concept_id, concept_data in ontology_concepts.items():
                node_id = f"ont_{node_id_counter}"
                node_map[f"ontology_{concept_id}"] = node_id
                
                nodes.append({
                    'id': node_id,
                    'label': concept_data.get('theme_name', concept_id),
                    'group': 'ontology_space',
                    'color': self.semantic_colors['ontology_space'],
                    'size': self.node_sizes['primary_concept'],
                    'semantic_space': 'Ontology',
                    'details': concept_data,
                    'x': math.cos(node_id_counter * 2 * math.pi / max(len(ontology_concepts), 1)) * 300,
                    'y': math.sin(node_id_counter * 2 * math.pi / max(len(ontology_concepts), 1)) * 300
                })
                node_id_counter += 1
        
        # Process Document Space Nodes
        document_data = tri_semantic_data.get('document_data', {})
        if 'enhanced_concepts' in document_data:
            doc_concepts = document_data['enhanced_concepts']
            for concept_id, concept_data in doc_concepts.items():
                node_id = f"doc_{node_id_counter}"
                node_map[f"document_{concept_id}"] = node_id
                
                nodes.append({
                    'id': node_id,
                    'label': concept_data.get('concept_name', concept_id),
                    'group': 'document_space',
                    'color': self.semantic_colors['document_space'],
                    'size': self.node_sizes['secondary_concept'],
                    'semantic_space': 'Document',
                    'details': concept_data,
                    'x': math.cos(node_id_counter * 2 * math.pi / max(len(doc_concepts), 1)) * 200 + 400,
                    'y': math.sin(node_id_counter * 2 * math.pi / max(len(doc_concepts), 1)) * 200
                })
                node_id_counter += 1
        
        # Process Question Space Nodes
        question_data = tri_semantic_data.get('question_data', {})
        if 'comprehensive_understanding' in question_data:
            understanding = question_data['comprehensive_understanding']
            understanding_dims = understanding.get('understanding_scores', {})
            
            for dim_name, score in understanding_dims.items():
                node_id = f"que_{node_id_counter}"
                node_map[f"question_{dim_name}"] = node_id
                
                nodes.append({
                    'id': node_id,
                    'label': dim_name.replace('_', ' ').title(),
                    'group': 'question_space',
                    'color': self.semantic_colors['question_space'],
                    'size': self.node_sizes['supporting_concept'] + int(score * 10),
                    'semantic_space': 'Question',
                    'details': {'dimension': dim_name, 'score': score},
                    'x': math.cos(node_id_counter * 2 * math.pi / max(len(understanding_dims), 1)) * 150 - 300,
                    'y': math.sin(node_id_counter * 2 * math.pi / max(len(understanding_dims), 1)) * 150 + 200
                })
                node_id_counter += 1
        
        # Create Fusion Nodes
        integration_data = tri_semantic_data.get('integration_data', {})
        if 'semantic_bridges' in integration_data:
            bridges = integration_data['semantic_bridges']
            for i, bridge in enumerate(bridges):
                node_id = f"fusion_{node_id_counter}"
                
                nodes.append({
                    'id': node_id,
                    'label': f"Fusion-{i+1}",
                    'group': 'fusion_nodes',
                    'color': self.semantic_colors['fusion_nodes'],
                    'size': self.node_sizes['fusion_result'],
                    'semantic_space': 'Fusion',
                    'details': bridge,
                    'x': 0,  # Central position
                    'y': i * 50 - 100
                })
                node_id_counter += 1
        
        # Create Cross-Space Bridge Edges
        self._create_bridge_edges(edges, node_map, tri_semantic_data)
        
        return {
            'nodes': nodes,
            'edges': edges,
            'semantic_spaces': ['Ontology', 'Document', 'Question', 'Fusion'],
            'total_nodes': len(nodes),
            'total_edges': len(edges),
            'network_density': len(edges) / max((len(nodes) * (len(nodes) - 1)) / 2, 1)
        }
    
    def _create_bridge_edges(self, edges: List[Dict], node_map: Dict[str, str], tri_semantic_data: Dict[str, Any]):
        """Create edges representing semantic bridges between spaces"""
        
        # Create sample cross-space connections based on available data
        integration_data = tri_semantic_data.get('integration_data', {})
        
        # Connect related concepts across spaces
        edge_id_counter = 0
        
        # Example: Connect ontology concepts to document concepts
        for i, ont_key in enumerate(list(node_map.keys())[:3]):
            if ont_key.startswith('ontology_'):
                for j, doc_key in enumerate(list(node_map.keys())[3:6]):
                    if doc_key.startswith('document_'):
                        edges.append({
                            'id': f"bridge_{edge_id_counter}",
                            'from': node_map[ont_key],
                            'to': node_map[doc_key],
                            'color': self.semantic_colors['bridge_connections'],
                            'width': self.edge_weights['cross_space_bridge'],
                            'type': 'cross_space_bridge',
                            'label': 'Semantic Bridge',
                            'dashes': [5, 5]  # Dashed line for bridges
                        })
                        edge_id_counter += 1
        
        # Connect fusion nodes to all spaces
        fusion_nodes = [node_id for node_id in node_map.values() if node_id.startswith('fusion_')]
        all_space_nodes = [node_id for node_id in node_map.values() if not node_id.startswith('fusion_')]
        
        for fusion_node in fusion_nodes[:2]:  # Limit connections
            for space_node in all_space_nodes[:6]:  # Connect to top 6 nodes
                edges.append({
                    'id': f"fusion_{edge_id_counter}",
                    'from': fusion_node,
                    'to': space_node,
                    'color': self.semantic_colors['fusion_nodes'],
                    'width': self.edge_weights['fusion_connection'],
                    'type': 'fusion_connection',
                    'label': 'Fusion Link'
                })
                edge_id_counter += 1
    
    def create_enhancement_heatmap_data(self, tri_semantic_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create heatmap data showing enhancement quality across the system
        
        Args:
            tri_semantic_data: Complete tri-semantic dataset
            
        Returns:
            Heatmap data structure
        """
        heatmap_data = {
            'dimensions': ['Intent Understanding', 'Semantic Depth', 'Contextual Grounding', 
                          'Cross-Pipeline Integration', 'Answer Quality'],
            'pipelines': ['A-Pipeline (Document)', 'B-Pipeline (Question)', 'R-Pipeline (Ontology)', 'I1 (Integration)'],
            'values': [],
            'color_scale': ['#FF7675', '#FDCB6E', '#6BCF7F', '#00B894'],  # Red to Green scale
            'annotations': []
        }
        
        # Generate enhancement quality matrix
        question_data = tri_semantic_data.get('question_data', {})
        integration_data = tri_semantic_data.get('integration_data', {})
        
        # Sample enhancement values (in real system, these would come from actual metrics)
        enhancement_matrix = [
            [0.7, 0.8, 0.6, 0.5, 0.7],  # A-Pipeline
            [0.8, 0.7, 0.7, 0.6, 0.8],  # B-Pipeline
            [0.6, 0.9, 0.8, 0.4, 0.6],  # R-Pipeline
            [0.9, 0.9, 0.9, 0.9, 0.8]   # I1 Integration
        ]
        
        # If we have real data, use it
        if question_data.get('comprehensive_understanding'):
            understanding_scores = question_data['comprehensive_understanding'].get('understanding_scores', {})
            i1_scores = [
                understanding_scores.get('intent_understanding', 0.5),
                understanding_scores.get('semantic_depth', 0.5),
                understanding_scores.get('contextual_grounding', 0.5),
                understanding_scores.get('cross_pipeline_integration', 0.5),
                understanding_scores.get('answer_quality', 0.5)
            ]
            enhancement_matrix[3] = i1_scores  # Update I1 row with real data
        
        heatmap_data['values'] = enhancement_matrix
        
        # Create annotations for significant values
        for i, pipeline in enumerate(heatmap_data['pipelines']):
            for j, dimension in enumerate(heatmap_data['dimensions']):
                value = enhancement_matrix[i][j]
                if value > 0.8:
                    heatmap_data['annotations'].append({
                        'x': j,
                        'y': i,
                        'text': f'{value:.2f}',
                        'color': 'white' if value > 0.7 else 'black'
                    })
        
        return heatmap_data
    
    def create_fusion_process_data(self, tri_semantic_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create data for fusion process flow visualization
        
        Args:
            tri_semantic_data: Complete tri-semantic dataset
            
        Returns:
            Fusion process flow data
        """
        fusion_data = tri_semantic_data.get('fusion_results', {})
        
        process_flow = {
            'stages': [
                'Concept Identification',
                'Tri-Space Perspective Gathering',
                'Consensus Fusion',
                'Authority-Guided Fusion',
                'Evidence-Based Fusion',
                'Context-Aware Fusion',
                'Meta-Fusion Synthesis',
                'Quality Assessment'
            ],
            'stage_details': {},
            'flow_connections': [],
            'quality_indicators': {}
        }
        
        # Add stage details and quality indicators
        for i, stage in enumerate(process_flow['stages']):
            process_flow['stage_details'][stage] = {
                'stage_id': i,
                'description': self._get_stage_description(stage),
                'status': 'completed' if fusion_data else 'simulated',
                'confidence': 0.7 + (i * 0.03),  # Increasing confidence through stages
                'processing_time': f"{10 + i * 5}ms"
            }
            
            process_flow['quality_indicators'][stage] = {
                'quality_score': 0.6 + (i * 0.05),
                'throughput': f"{100 - i * 5} concepts/sec"
            }
            
            # Create flow connections
            if i < len(process_flow['stages']) - 1:
                process_flow['flow_connections'].append({
                    'from': stage,
                    'to': process_flow['stages'][i + 1],
                    'strength': 0.8 + (i * 0.02)
                })
        
        return process_flow
    
    def _get_stage_description(self, stage: str) -> str:
        """Get description for fusion process stage"""
        descriptions = {
            'Concept Identification': 'Identify key concepts from question analysis',
            'Tri-Space Perspective Gathering': 'Collect perspectives from all three semantic spaces',
            'Consensus Fusion': 'Find agreement across different perspectives',
            'Authority-Guided Fusion': 'Weight perspectives based on authority/confidence',
            'Evidence-Based Fusion': 'Prioritize perspectives with strong evidence',
            'Context-Aware Fusion': 'Consider contextual relevance in fusion',
            'Meta-Fusion Synthesis': 'Combine all fusion strategies optimally',
            'Quality Assessment': 'Evaluate and validate fusion results'
        }
        return descriptions.get(stage, 'Processing stage')
    
    def generate_visualization_html(self, network_data: Dict[str, Any], heatmap_data: Dict[str, Any], 
                                   fusion_data: Dict[str, Any]) -> str:
        """
        Generate comprehensive HTML visualization
        
        Args:
            network_data: Tri-semantic network data
            heatmap_data: Enhancement heatmap data
            fusion_data: Fusion process data
            
        Returns:
            Complete HTML visualization
        """
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>I3 Tri-Semantic Integration Visualizer</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            overflow-x: hidden;
        }}
        
        .header {{
            background: rgba(255,255,255,0.95);
            backdrop-filter: blur(10px);
            padding: 1rem 2rem;
            box-shadow: 0 2px 20px rgba(0,0,0,0.1);
            position: sticky;
            top: 0;
            z-index: 1000;
        }}
        
        .header h1 {{
            color: #2c3e50;
            display: flex;
            align-items: center;
            gap: 1rem;
        }}
        
        .header .subtitle {{
            color: #7f8c8d;
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }}
        
        .dashboard {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 70vh 50vh;
            gap: 1rem;
            padding: 1rem;
            height: calc(100vh - 120px);
        }}
        
        .panel {{
            background: rgba(255,255,255,0.95);
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            overflow: hidden;
            transition: transform 0.3s ease;
        }}
        
        .panel:hover {{
            transform: translateY(-2px);
            box-shadow: 0 12px 40px rgba(0,0,0,0.15);
        }}
        
        .panel-header {{
            background: linear-gradient(45deg, #4ecdc4, #45b7d1);
            color: white;
            padding: 1rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .panel-content {{
            height: calc(100% - 60px);
            position: relative;
        }}
        
        .network-panel {{
            grid-column: 1 / -1;
        }}
        
        #tri-semantic-network {{
            width: 100%;
            height: 100%;
            background: radial-gradient(circle at center, #f8f9fa 0%, #e9ecef 100%);
        }}
        
        .metrics {{
            position: absolute;
            top: 1rem;
            right: 1rem;
            background: rgba(255,255,255,0.9);
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        
        .metric {{
            display: flex;
            justify-content: space-between;
            margin: 0.5rem 0;
            padding: 0.25rem 0;
            border-bottom: 1px solid #eee;
        }}
        
        .metric:last-child {{ border-bottom: none; }}
        
        .legend {{
            position: absolute;
            bottom: 1rem;
            left: 1rem;
            background: rgba(255,255,255,0.9);
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin: 0.5rem 0;
        }}
        
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            padding: 1rem;
            height: 100%;
        }}
        
        .stat-item {{
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }}
        
        .stat-label {{
            font-size: 0.9rem;
            opacity: 0.9;
        }}
        
        @media (max-width: 768px) {{
            .dashboard {{
                grid-template-columns: 1fr;
                grid-template-rows: 60vh 40vh 40vh;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>
            <i class="fas fa-project-diagram"></i>
            I3 Tri-Semantic Integration Visualizer
        </h1>
        <div class="subtitle">
            Revolutionary Cross-Pipeline Semantic Integration Visualization | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
    
    <div class="dashboard">
        <div class="panel network-panel">
            <div class="panel-header">
                <i class="fas fa-sitemap"></i>
                Tri-Semantic Integration Network
            </div>
            <div class="panel-content">
                <div id="tri-semantic-network"></div>
                
                <div class="metrics">
                    <h4>Network Metrics</h4>
                    <div class="metric">
                        <span>Total Nodes:</span>
                        <strong>{network_data['total_nodes']}</strong>
                    </div>
                    <div class="metric">
                        <span>Total Edges:</span>
                        <strong>{network_data['total_edges']}</strong>
                    </div>
                    <div class="metric">
                        <span>Network Density:</span>
                        <strong>{network_data['network_density']:.3f}</strong>
                    </div>
                    <div class="metric">
                        <span>Semantic Spaces:</span>
                        <strong>{len(network_data['semantic_spaces'])}</strong>
                    </div>
                </div>
                
                <div class="legend">
                    <h4>Semantic Spaces</h4>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: {self.semantic_colors['ontology_space']};"></div>
                        <span>Ontology Space</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: {self.semantic_colors['document_space']};"></div>
                        <span>Document Space</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: {self.semantic_colors['question_space']};"></div>
                        <span>Question Space</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: {self.semantic_colors['fusion_nodes']};"></div>
                        <span>Fusion Results</span>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="panel">
            <div class="panel-header">
                <i class="fas fa-fire"></i>
                Enhancement Quality Heatmap
            </div>
            <div class="panel-content">
                <div id="enhancement-heatmap" style="width: 100%; height: 100%;"></div>
            </div>
        </div>
        
        <div class="panel">
            <div class="panel-header">
                <i class="fas fa-chart-line"></i>
                Integration Analytics
            </div>
            <div class="panel-content">
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-value">{len(network_data.get('nodes', []))}</div>
                        <div class="stat-label">Concept Nodes</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{len([n for n in network_data.get('nodes', []) if n.get('group') == 'fusion_nodes'])}</div>
                        <div class="stat-label">Fusion Points</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{len([e for e in network_data.get('edges', []) if e.get('type') == 'cross_space_bridge'])}</div>
                        <div class="stat-label">Semantic Bridges</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{network_data.get('network_density', 0):.1%}</div>
                        <div class="stat-label">Network Density</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Tri-Semantic Network Visualization
        const networkData = {json.dumps(network_data, indent=2)};
        const heatmapData = {json.dumps(heatmap_data, indent=2)};
        
        // Initialize Network
        const container = document.getElementById('tri-semantic-network');
        const data = {{
            nodes: new vis.DataSet(networkData.nodes),
            edges: new vis.DataSet(networkData.edges)
        }};
        
        const options = {{
            physics: {{
                enabled: true,
                solver: 'forceAtlas2Based',
                forceAtlas2Based: {{
                    gravitationalConstant: -26,
                    centralGravity: 0.005,
                    springLength: 230,
                    springConstant: 0.18,
                    avoidOverlap: 1
                }},
                maxVelocity: 146,
                minVelocity: 0.1,
                timestep: 0.35,
                adaptiveTimestep: true
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 200,
                hideEdgesOnDrag: true,
                hideNodesOnDrag: false
            }},
            nodes: {{
                borderWidth: 2,
                borderWidthSelected: 4,
                chosen: true,
                font: {{
                    size: 12,
                    face: 'arial',
                    color: '#343434'
                }},
                shadow: {{
                    enabled: true,
                    color: 'rgba(0,0,0,0.2)',
                    size: 10,
                    x: 2,
                    y: 2
                }}
            }},
            edges: {{
                smooth: {{
                    type: 'continuous',
                    roundness: 0.5
                }},
                arrows: {{
                    to: {{
                        enabled: true,
                        scaleFactor: 0.5
                    }}
                }},
                shadow: {{
                    enabled: true,
                    color: 'rgba(0,0,0,0.1)',
                    size: 5,
                    x: 1,
                    y: 1
                }}
            }},
            groups: {{
                ontology_space: {{
                    color: '{self.semantic_colors["ontology_space"]}',
                    shape: 'dot'
                }},
                document_space: {{
                    color: '{self.semantic_colors["document_space"]}',
                    shape: 'diamond'
                }},
                question_space: {{
                    color: '{self.semantic_colors["question_space"]}',
                    shape: 'square'
                }},
                fusion_nodes: {{
                    color: '{self.semantic_colors["fusion_nodes"]}',
                    shape: 'star'
                }}
            }}
        }};
        
        const network = new vis.Network(container, data, options);
        
        // Network event handlers
        network.on('click', function(params) {{
            if (params.nodes.length > 0) {{
                const nodeId = params.nodes[0];
                const node = data.nodes.get(nodeId);
                console.log('Node clicked:', node);
                
                // Show node details (could expand to show modal)
                const details = node.details || {{}};
                alert(`Semantic Space: ${{node.semantic_space}}\\nLabel: ${{node.label}}\\nDetails: ${{JSON.stringify(details, null, 2)}}`);
            }}
        }});
        
        // Enhancement Heatmap
        const heatmapTrace = {{
            z: heatmapData.values,
            x: heatmapData.dimensions,
            y: heatmapData.pipelines,
            type: 'heatmap',
            colorscale: [
                [0, '#FF7675'],
                [0.25, '#FDCB6E'],
                [0.5, '#FFEAA7'],
                [0.75, '#6BCF7F'],
                [1, '#00B894']
            ],
            showscale: true,
            hoverongaps: false,
            hovertemplate: 'Pipeline: %{{y}}<br>Dimension: %{{x}}<br>Quality: %{{z:.3f}}<extra></extra>'
        }};
        
        const heatmapLayout = {{
            title: {{
                text: 'Enhancement Quality Matrix',
                font: {{ size: 14 }}
            }},
            xaxis: {{
                title: 'Quality Dimensions',
                tickangle: -45
            }},
            yaxis: {{
                title: 'Processing Pipelines'
            }},
            margin: {{ t: 50, r: 50, b: 80, l: 120 }},
            font: {{ size: 10 }}
        }};
        
        Plotly.newPlot('enhancement-heatmap', [heatmapTrace], heatmapLayout, {{responsive: true}});
        
        // Auto-refresh network layout
        setTimeout(() => {{
            network.fit();
        }}, 1000);
        
        // Console info
        console.log('I3 Tri-Semantic Visualizer Initialized');
        console.log('Network Data:', networkData);
        console.log('Heatmap Data:', heatmapData);
    </script>
</body>
</html>
        """
        
        return html_template
    
    def create_visualization(self) -> str:
        """
        Create complete I3 tri-semantic visualization
        
        Returns:
            Path to generated HTML file
        """
        print("Creating I3 Tri-Semantic Visualization...")
        
        # Load tri-semantic data
        print("  Loading tri-semantic integration data...")
        tri_semantic_data = self.load_tri_semantic_data()
        
        # Create network visualization data
        print("  Building tri-semantic network structure...")
        network_data = self.create_tri_semantic_network_data(tri_semantic_data)
        
        # Create enhancement heatmap data
        print("  Generating enhancement quality heatmap...")
        heatmap_data = self.create_enhancement_heatmap_data(tri_semantic_data)
        
        # Create fusion process data
        print("  Preparing fusion process visualization...")
        fusion_data = self.create_fusion_process_data(tri_semantic_data)
        
        # Generate HTML visualization
        print("  Rendering interactive visualization...")
        html_content = self.generate_visualization_html(network_data, heatmap_data, fusion_data)
        
        # Save HTML file
        output_dir = Path(__file__).parent.parent / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        html_file = output_dir / "I3_tri_semantic_visualization.html"
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ I3 Tri-Semantic Visualization created: {html_file}")
        
        return str(html_file)
    
    def launch_visualization(self, html_file: str, port: int = 8086):
        """
        Launch the visualization in a web browser
        
        Args:
            html_file: Path to HTML file
            port: Port number for local server
        """
        try:
            # Start local server
            html_dir = Path(html_file).parent
            
            class QuietHandler(http.server.SimpleHTTPRequestHandler):
                def log_message(self, format, *args):
                    pass  # Suppress server logs
            
            with socketserver.TCPServer(("", port), QuietHandler) as httpd:
                httpd.allow_reuse_address = True
                
                # Start server in background thread
                server_thread = threading.Thread(target=httpd.serve_forever)
                server_thread.daemon = True
                server_thread.start()
                
                # Open visualization in browser
                visualization_url = f"http://localhost:{port}/I3_tri_semantic_visualization.html"
                print(f"[START] Launching I3 Tri-Semantic Visualizer at: {visualization_url}")
                
                # Change to output directory and open browser
                import os
                original_dir = os.getcwd()
                os.chdir(html_dir)
                
                try:
                    webbrowser.open(visualization_url)
                    print("✓ Browser opened successfully")
                    
                    print("\\n" + "="*70)
                    print("I3 TRI-SEMANTIC VISUALIZER ACTIVE")
                    print("="*70)
                    print(f"URL: {visualization_url}")
                    print("Features:")
                    print("  • Interactive tri-semantic network visualization")
                    print("  • Enhancement quality heatmap analysis") 
                    print("  • Real-time integration metrics")
                    print("  • Cross-pipeline semantic bridge display")
                    print("\\nPress Ctrl+C to stop the server")
                    print("="*70)
                    
                    # Keep server running
                    try:
                        while True:
                            import time
                            time.sleep(1)
                    except KeyboardInterrupt:
                        print("\\n\\n✓ I3 Visualizer stopped")
                        
                finally:
                    os.chdir(original_dir)
                    
        except Exception as e:
            print(f"[ERROR] Error launching visualization: {e}")
            print(f"You can manually open: {html_file}")

def main():
    """Main execution for I3 Tri-Semantic Visualizer"""
    print("="*80)
    print("I3: Tri-Semantic Integration Visualizer")
    print("="*80)
    
    try:
        # Initialize visualizer
        print("Initializing I3 Tri-Semantic Visualizer...")
        visualizer = I3_TriSemanticVisualizer()
        
        # Create visualization
        html_file = visualizer.create_visualization()
        
        # Launch in browser
        print("\\nLaunching interactive visualization...")
        visualizer.launch_visualization(html_file)
        
    except Exception as e:
        print(f"[ERROR] Error in I3 Tri-Semantic Visualizer: {str(e)}")
        raise

if __name__ == "__main__":
    main()