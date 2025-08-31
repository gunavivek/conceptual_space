#!/usr/bin/env python3
"""
R5L: Lexical Ontology Visualizer
Part of R-Pipeline (Resource & Reasoning Pipeline)
Creates interactive and static visualizations of the BIZBOK lexical ontology
Note: Visualizes keyword-based (lexical) relationships, not true semantic embeddings
"""

import json
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import numpy as np

class LexicalOntologyVisualizer:
    """Main class for visualizing the BIZBOK lexical ontology"""
    
    def __init__(self):
        self.script_dir = Path(__file__).parent.parent
        self.output_dir = self.script_dir / "output"
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Load ontology data
        self.ontology = None
        self.concepts = None
        self.clusters = None
        self.hierarchy = None
        self.statistics = None
        
    def load_ontology(self):
        """Load R4L lexical ontology output"""
        print("[DATA] Loading R4L lexical ontology...")
        
        ontology_path = self.output_dir / "R4L_lexical_ontology.json"
        if not ontology_path.exists():
            raise FileNotFoundError(f"R4L lexical ontology not found: {ontology_path}")
        
        with open(ontology_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.ontology = data['ontology']
        self.concepts = self.ontology['concepts']
        self.clusters = self.ontology['clusters']
        self.hierarchy = self.ontology['hierarchy']
        self.statistics = self.ontology['statistics']
        
        print(f"   [OK] Loaded {len(self.concepts)} concepts")
        print(f"   [OK] Loaded {len(self.clusters)} clusters")
        print(f"   [OK] Loaded hierarchy with {len(self.hierarchy)} nodes")
        
        return self.ontology
    
    def create_network_graph(self):
        """Create NetworkX graph from ontology"""
        G = nx.Graph()
        
        # Add nodes with attributes
        for concept_id, data in self.concepts.items():
            G.add_node(
                concept_id,
                name=data['name'],
                domain=data['domain'],
                cluster=data['cluster'],
                level=data['hierarchy']['level'],
                relationships=data['ontology_metadata']['relationship_count'],
                connectivity=data['ontology_metadata']['connectivity_score']
            )
        
        # Add edges from lexical relationships
        for concept_id, data in self.concepts.items():
            for related_id in data['relationships']['lexical']:
                if related_id in self.concepts:
                    G.add_edge(concept_id, related_id, type='lexical')
        
        # Add edges from other relationship types
        for concept_id, data in self.concepts.items():
            for related in data['relationships'].get('compositional', []):
                if isinstance(related, dict):
                    related_id = related.get('concept_id')
                elif isinstance(related, str):
                    related_id = related
                else:
                    continue
                if related_id and related_id in self.concepts:
                    G.add_edge(concept_id, related_id, type='compositional')
        
        return G
    
    def visualize_full_network(self):
        """Create full network visualization"""
        print("\n[VISUALIZE] Creating full network graph...")
        
        G = self.create_network_graph()
        
        plt.figure(figsize=(24, 18))
        
        # Use spring layout with adjusted parameters for better spacing
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        
        # Color nodes by domain
        domains = list(set(self.concepts[node]['domain'] for node in G.nodes()))
        color_map = plt.cm.get_cmap('tab20')
        domain_colors = {domain: color_map(i/len(domains)) 
                        for i, domain in enumerate(domains)}
        
        node_colors = [domain_colors[self.concepts[node]['domain']] 
                      for node in G.nodes()]
        
        # Size nodes by connectivity score
        node_sizes = [self.concepts[node]['ontology_metadata']['connectivity_score'] * 500 
                     for node in G.nodes()]
        
        # Draw edges with different styles for different types
        lexical_edges = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'lexical']
        other_edges = [(u, v) for u, v, d in G.edges(data=True) if d['type'] != 'lexical']
        
        nx.draw_networkx_edges(G, pos, edgelist=lexical_edges, 
                              alpha=0.2, width=0.5, edge_color='gray')
        nx.draw_networkx_edges(G, pos, edgelist=other_edges,
                              alpha=0.4, width=1.0, edge_color='red', style='dashed')
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                              node_color=node_colors,
                              node_size=node_sizes,
                              alpha=0.8,
                              edgecolors='black',
                              linewidths=0.5)
        
        # Add labels for highly connected nodes
        top_nodes = sorted(G.nodes(), 
                          key=lambda x: self.concepts[x]['ontology_metadata']['relationship_count'],
                          reverse=True)[:25]
        
        labels = {node: self.concepts[node]['name'] for node in top_nodes}
        nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold')
        
        # Create legend
        legend_elements = []
        for domain in sorted(domains):
            legend_elements.append(plt.scatter([], [], c=[domain_colors[domain]], 
                                              label=domain, s=100, alpha=0.8))
        
        plt.legend(handles=legend_elements, loc='upper left', 
                  title='Domains', fontsize=10, title_fontsize=12)
        
        plt.title('BIZBOK Lexical Ontology - Complete Network Graph\n'
                 f'{len(G.nodes())} Concepts | {len(G.edges())} Relationships', 
                 fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        # Save
        output_path = self.viz_dir / "R5L_network_full.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"   [OK] Saved: {output_path}")
    
    def visualize_clusters(self):
        """Visualize cluster structure"""
        print("\n[VISUALIZE] Creating cluster visualizations...")
        
        # Get top 9 clusters by size
        top_clusters = sorted(self.clusters.items(), 
                            key=lambda x: x[1]['size'], 
                            reverse=True)[:9]
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 20))
        axes = axes.flatten()
        
        for idx, (cluster_id, cluster_data) in enumerate(top_clusters):
            ax = axes[idx]
            
            # Create subgraph for this cluster
            G = nx.Graph()
            cluster_members = cluster_data['members']
            
            # Add nodes
            for member in cluster_members:
                if member in self.concepts:
                    G.add_node(member, 
                             name=self.concepts[member]['name'],
                             relationships=self.concepts[member]['ontology_metadata']['relationship_count'])
            
            # Add edges within cluster
            for member in cluster_members:
                if member in self.concepts:
                    for related in self.concepts[member]['relationships']['lexical']:
                        if related in cluster_members:
                            G.add_edge(member, related)
            
            # Draw cluster
            if len(G.nodes()) > 0:
                pos = nx.spring_layout(G, k=2, iterations=50)
                
                # Node sizes based on relationships
                node_sizes = [G.nodes[node]['relationships'] * 30 for node in G.nodes()]
                
                nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=0.5)
                nx.draw_networkx_nodes(G, pos, ax=ax,
                                     node_color='lightblue',
                                     node_size=node_sizes,
                                     alpha=0.8,
                                     edgecolors='darkblue',
                                     linewidths=1)
                
                # Add labels
                labels = {node: G.nodes[node]['name'][:20] for node in G.nodes()}
                nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)
            
            ax.set_title(f"Cluster: {cluster_data['name'][:40]}\n"
                        f"Size: {cluster_data['size']} | "
                        f"Coherence: {cluster_data['coherence_score']:.3f}",
                        fontsize=10, fontweight='bold')
            ax.axis('off')
        
        plt.suptitle('BIZBOK Lexical Clusters - Top 9 by Size', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        output_path = self.viz_dir / "R5L_clusters.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"   [OK] Saved: {output_path}")
    
    def visualize_hierarchy(self):
        """Visualize hierarchical structure"""
        print("\n[VISUALIZE] Creating hierarchy visualization...")
        
        # Create directed graph for hierarchy
        G = nx.DiGraph()
        
        # Add nodes and edges
        for node_id, node_data in self.hierarchy.items():
            G.add_node(node_id, 
                      level=node_data['level'],
                      is_leaf=node_data.get('is_leaf', False))
            
            for child in node_data.get('children', []):
                G.add_edge(node_id, child)
        
        plt.figure(figsize=(20, 12))
        
        # Calculate positions using custom hierarchical layout
        levels = defaultdict(list)
        for node in G.nodes():
            level = self.hierarchy[node]['level']
            levels[level].append(node)
        
        pos = {}
        for level, nodes in levels.items():
            for i, node in enumerate(nodes):
                x = (i - len(nodes)/2) * (4.0 / (level + 1))
                y = -level * 2
                pos[node] = (x, y)
        
        # Color by level
        node_colors = []
        node_sizes = []
        for node in G.nodes():
            level = self.hierarchy[node]['level']
            is_leaf = self.hierarchy[node].get('is_leaf', False)
            
            if level == 0:
                node_colors.append('darkred')
                node_sizes.append(1000)
            elif level == 1:
                node_colors.append('orange')
                node_sizes.append(700)
            elif level == 2:
                node_colors.append('gold')
                node_sizes.append(500)
            else:
                node_colors.append('lightgreen')
                node_sizes.append(300)
        
        # Draw
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                              arrows=True, arrowsize=10, width=1, alpha=0.5)
        nx.draw_networkx_nodes(G, pos,
                              node_color=node_colors,
                              node_size=node_sizes,
                              alpha=0.9,
                              edgecolors='black',
                              linewidths=0.5)
        
        # Add labels
        labels = {}
        for node in G.nodes():
            if node in self.concepts:
                labels[node] = self.concepts[node]['name'][:20]
            else:
                labels[node] = node.replace('_concepts', '').replace('_', ' ').title()
        
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        # Add legend
        legend_labels = ['Root (Level 0)', 'Domains (Level 1)', 
                        'Categories (Level 2)', 'Concepts (Level 3)']
        legend_colors = ['darkred', 'orange', 'gold', 'lightgreen']
        for label, color in zip(legend_labels, legend_colors):
            plt.scatter([], [], c=color, s=100, label=label, alpha=0.9)
        plt.legend(loc='upper right', fontsize=10)
        
        plt.title(f'BIZBOK Ontology Hierarchical Structure\n'
                 f'{len(G.nodes())} Nodes | Max Depth: {max(levels.keys())} Levels',
                 fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        # Save
        output_path = self.viz_dir / "R5L_hierarchy.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"   [OK] Saved: {output_path}")
    
    def visualize_statistics(self):
        """Create statistical visualizations"""
        print("\n[VISUALIZE] Creating statistical charts...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Domain Distribution
        ax = axes[0, 0]
        domains = Counter(c['domain'] for c in self.concepts.values())
        ax.bar(range(len(domains)), list(domains.values()))
        ax.set_xticks(range(len(domains)))
        ax.set_xticklabels(list(domains.keys()), rotation=45, ha='right')
        ax.set_title('Concept Distribution by Domain', fontweight='bold')
        ax.set_ylabel('Number of Concepts')
        ax.grid(axis='y', alpha=0.3)
        
        # 2. Relationship Type Distribution
        ax = axes[0, 1]
        rel_types = self.statistics['relationships_by_type']
        ax.pie(rel_types.values(), labels=rel_types.keys(), autopct='%1.1f%%',
               colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
        ax.set_title('Relationship Type Distribution', fontweight='bold')
        
        # 3. Connectivity Distribution
        ax = axes[1, 0]
        connectivities = [c['ontology_metadata']['relationship_count'] 
                         for c in self.concepts.values()]
        ax.hist(connectivities, bins=20, edgecolor='black', alpha=0.7)
        ax.set_title('Concept Connectivity Distribution', fontweight='bold')
        ax.set_xlabel('Number of Relationships')
        ax.set_ylabel('Number of Concepts')
        ax.grid(axis='y', alpha=0.3)
        
        # 4. Cluster Sizes
        ax = axes[1, 1]
        cluster_sizes = sorted([c['size'] for c in self.clusters.values()], reverse=True)
        ax.bar(range(len(cluster_sizes)), cluster_sizes)
        ax.set_title('Cluster Size Distribution', fontweight='bold')
        ax.set_xlabel('Cluster Rank')
        ax.set_ylabel('Number of Concepts')
        ax.grid(axis='y', alpha=0.3)
        
        plt.suptitle('BIZBOK Ontology Statistics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        output_path = self.viz_dir / "R5L_statistics.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"   [OK] Saved: {output_path}")
    
    def create_interactive_html(self):
        """Create interactive HTML visualization"""
        print("\n[VISUALIZE] Creating interactive HTML...")
        
        # Prepare data for vis.js
        nodes = []
        edges = []
        
        # Create consistent domain colors
        domains = sorted(set(data['domain'] for data in self.concepts.values()))
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
        
        # Create nodes
        for concept_id, data in self.concepts.items():
            # Create clean tooltip without HTML tags
            tooltip = (f"Concept: {data['name']}\n"
                      f"Domain: {data['domain']}\n"
                      f"Cluster: {data['cluster']}\n"
                      f"Total Relationships: {data['ontology_metadata']['relationship_count']}\n"
                      f"Hierarchy Level: {data['hierarchy']['level']}\n"
                      f"Lexical: {len(data['relationships']['lexical'])}\n"
                      f"Compositional: {len(data['relationships']['compositional'])}\n"
                      f"Causal: {len(data['relationships']['causal'])}\n"
                      f"Temporal: {len(data['relationships']['temporal'])}")
            
            nodes.append({
                'id': concept_id,
                'label': data['name'],
                'title': tooltip,
                'group': data['domain'],
                'color': domain_colors.get(data['domain'], '#CCCCCC'),
                'value': data['ontology_metadata']['relationship_count'],
                'level': data['hierarchy']['level']
            })
        
        # Create edges with proper relationship type labeling
        edge_id = 0
        for concept_id, data in self.concepts.items():
            # Lexical relationships
            for i, related_id in enumerate(data['relationships']['lexical'][:5]):  # Limit for clarity
                if related_id in self.concepts:
                    related_name = self.concepts[related_id]['name']
                    
                    # Calculate lexical similarity score for display
                    # Note: This is currently keyword-based Jaccard, not true semantic similarity (R5S will use embeddings)
                    concept_keywords = set(data.get('keywords', []))
                    related_keywords = set(self.concepts[related_id].get('keywords', []))
                    
                    if concept_keywords and related_keywords:
                        intersection = len(concept_keywords & related_keywords)
                        union = len(concept_keywords | related_keywords)
                        jaccard_score = intersection / union if union > 0 else 0.0
                    else:
                        jaccard_score = 0.0
                    
                    # Add domain bonus (R4 logic)
                    domain_bonus = 0.15 if data['domain'] == self.concepts[related_id]['domain'] else 0.0
                    total_score = min(1.0, jaccard_score + domain_bonus)
                    
                    edges.append({
                        'from': concept_id,
                        'to': related_id,
                        'color': {'color': '#808080', 'opacity': 0.4},
                        'title': f'Lexical (score: {total_score:.3f}): {data["name"]} ↔ {related_name}',
                        'width': max(1, total_score * 3)  # Width based on similarity
                    })
                    edge_id += 1
            
            # Compositional relationships
            for related in data['relationships'].get('compositional', []):
                if isinstance(related, dict):
                    related_id = related.get('concept_id')
                    relation_type = related.get('relation_type', 'compositional')
                else:
                    related_id = related
                    relation_type = 'compositional'
                    
                if related_id and related_id in self.concepts:
                    related_name = self.concepts[related_id]['name']
                    edges.append({
                        'from': concept_id,
                        'to': related_id,
                        'color': {'color': '#FF6B6B', 'opacity': 0.6},
                        'title': f'Compositional ({relation_type}): {data["name"]} → {related_name}',
                        'dashes': [5, 5],
                        'width': 2
                    })
                    edge_id += 1
            
            # Causal relationships
            for related in data['relationships'].get('causal', []):
                if isinstance(related, dict):
                    related_id = related.get('concept_id')
                    relation_type = related.get('relation_type', 'causes')
                else:
                    related_id = related
                    relation_type = 'causes'
                    
                if related_id and related_id in self.concepts:
                    related_name = self.concepts[related_id]['name']
                    edges.append({
                        'from': concept_id,
                        'to': related_id,
                        'color': {'color': '#4ECDC4', 'opacity': 0.7},
                        'title': f'Causal ({relation_type}): {data["name"]} → {related_name}',
                        'arrows': 'to',
                        'width': 2
                    })
                    edge_id += 1
            
            # Temporal relationships  
            for related in data['relationships'].get('temporal', []):
                if isinstance(related, dict):
                    related_id = related.get('concept_id')
                    relation_type = related.get('relation_type', 'temporal')
                else:
                    related_id = related
                    relation_type = 'temporal'
                    
                if related_id and related_id in self.concepts:
                    related_name = self.concepts[related_id]['name']
                    edges.append({
                        'from': concept_id,
                        'to': related_id,
                        'color': {'color': '#9966FF', 'opacity': 0.6},
                        'title': f'Temporal ({relation_type}): {data["name"]} → {related_name}',
                        'dashes': [10, 5],
                        'width': 1.5
                    })
                    edge_id += 1
        
        # Generate HTML
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>BIZBOK Lexical Ontology - Interactive Visualization</title>
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
        <h1>🔮 BIZBOK Lexical Ontology - Interactive Network</h1>
        
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
            <span style="color: #808080;">━</span> <strong>Lexical</strong> (gray, width = score): Keyword-based Jaccard similarity + domain bonus<br>
            <span style="color: #FF6B6B;">┅┅</span> <strong>Compositional</strong> (red, dashed): Part-whole relationships (has_part, part_of)<br>
            <span style="color: #4ECDC4;">→</span> <strong>Causal</strong> (teal, arrow): Cause-effect relationships (causes, depends_on)<br>
            <span style="color: #9966FF;">┄┄</span> <strong>Temporal</strong> (purple, long dash): Time-based relationships (precedes, follows)
        </div>
        
        <div class="info">
            <strong>⚠️ Note:</strong> "Lexical" relationships use <strong>keyword-based</strong> Jaccard similarity + domain bonus. Hover over lexical edges to see similarity scores. Future R5S will implement true semantic similarity using embeddings.
        </div>
        
        <div id="mynetwork"></div>
        
        <div class="stats">
            <div class="stat-item">
                <div class="stat-value">{len(self.concepts)}</div>
                <div class="stat-label">Concepts</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{len(self.clusters)}</div>
                <div class="stat-label">Clusters</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{self.statistics["relationships_total"]}</div>
                <div class="stat-label">Relationships</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{self.statistics["hierarchy_max_depth"]}</div>
                <div class="stat-label">Hierarchy Depth</div>
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
        
        # Save HTML
        output_path = self.viz_dir / "R5L_interactive.html"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"   [OK] Saved: {output_path}")
        print(f"   [INFO] Open {output_path} in a web browser for interactive exploration!")
    
    def run(self):
        """Main execution method"""
        print("\n" + "="*60)
        print("R5L: Lexical Ontology Visualizer")
        print("R-Pipeline: Resource & Reasoning Pipeline")
        print("="*60)
        
        # Load data
        self.load_ontology()
        
        # Generate all visualizations
        self.visualize_full_network()
        self.visualize_clusters()
        self.visualize_hierarchy()
        self.visualize_statistics()
        self.create_interactive_html()
        
        print("\n" + "="*60)
        print("Visualization Summary")
        print("="*60)
        print(f"Generated visualizations in: {self.viz_dir}")
        print("\nFiles created:")
        print("  - R5L_network_full.png   : Complete network graph")
        print("  - R5L_clusters.png       : Top 9 clusters")
        print("  - R5L_hierarchy.png      : Hierarchical structure")
        print("  - R5L_statistics.png     : Statistical charts")
        print("  - R5L_interactive.html   : Interactive web visualization")
        print("\n[SUCCESS] R5L lexical ontology visualization complete!")

def main():
    """Main entry point"""
    visualizer = LexicalOntologyVisualizer()
    visualizer.run()

if __name__ == "__main__":
    main()