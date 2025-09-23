#!/usr/bin/env python3
"""
A4.2 Enhancement: Concept-Chunk Membership Report
Generates detailed reports showing which chunks belong to which concepts
Combines data from A3 chunks and A2.4 concepts
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ConceptChunkMembershipAnalyzer:
    """Analyzes and visualizes concept-chunk membership relationships"""

    def __init__(self):
        self.base_path = Path(__file__).parent.parent
        self.outputs_dir = self.base_path / "outputs"

        # Data holders
        self.chunks_data = None
        self.concepts_data = None
        self.concept_to_chunks = defaultdict(list)
        self.chunk_to_concepts = defaultdict(list)

    def load_data(self) -> bool:
        """Load A3 chunks and A2.4 concepts data"""

        # Load A3 chunks
        chunks_path = self.outputs_dir / "A3_multi_strategy_chunks.json"
        if not chunks_path.exists():
            print(f"Error: Cannot find {chunks_path}")
            return False

        with open(chunks_path, 'r', encoding='utf-8') as f:
            a3_data = json.load(f)
            self.chunks_data = a3_data.get('chunks', [])

        # Load A2.4 concepts
        concepts_path = self.outputs_dir / "A2.4_core_concepts.json"
        if not concepts_path.exists():
            print(f"Error: Cannot find {concepts_path}")
            return False

        with open(concepts_path, 'r', encoding='utf-8') as f:
            concepts_data = json.load(f)
            self.concepts_data = concepts_data.get('core_concepts', [])

        print(f"Loaded {len(self.chunks_data)} chunks and {len(self.concepts_data)} concepts")
        return True

    def analyze_memberships(self):
        """Build concept-chunk and chunk-concept mappings"""

        # Build mappings from chunks data
        for chunk in self.chunks_data:
            chunk_id = chunk.get('chunk_id', '')
            doc_id = chunk.get('doc_id', '')
            concept_memberships = chunk.get('concept_memberships', [])

            # Store chunk info
            chunk_info = {
                'chunk_id': chunk_id,
                'doc_id': doc_id,
                'content_preview': chunk.get('content', '')[:100] + '...',
                'chunk_type': chunk.get('chunk_type', ''),
                'num_concepts': len(concept_memberships)
            }

            # Map concepts to chunks
            for concept_id in concept_memberships:
                self.concept_to_chunks[concept_id].append(chunk_info)
                self.chunk_to_concepts[chunk_id].append(concept_id)

        print(f"Built mappings: {len(self.concept_to_chunks)} concepts have chunks")

    def generate_membership_report(self) -> pd.DataFrame:
        """Generate detailed membership report as DataFrame"""

        report_data = []

        # Create concept lookup
        concept_lookup = {c['concept_id']: c for c in self.concepts_data}

        for concept_id, chunks in self.concept_to_chunks.items():
            concept_info = concept_lookup.get(concept_id, {})

            for chunk in chunks:
                report_data.append({
                    'Concept ID': concept_id,
                    'Concept Name': concept_info.get('canonical_name', concept_id),
                    'Chunk ID': chunk['chunk_id'],
                    'Document ID': chunk['doc_id'],
                    'Chunk Type': chunk['chunk_type'],
                    'Content Preview': chunk['content_preview'],
                    'Total Concepts in Chunk': chunk['num_concepts']
                })

        df = pd.DataFrame(report_data)

        # Sort by concept name and chunk ID
        if not df.empty:
            df = df.sort_values(['Concept Name', 'Chunk ID'])

        return df

    def generate_summary_statistics(self) -> Dict:
        """Generate summary statistics about concept-chunk relationships"""

        stats = {
            'total_concepts': len(self.concepts_data),
            'concepts_with_chunks': len(self.concept_to_chunks),
            'concepts_without_chunks': 0,
            'total_chunks': len(self.chunks_data),
            'chunks_with_concepts': 0,
            'chunks_without_concepts': 0,
            'avg_chunks_per_concept': 0,
            'avg_concepts_per_chunk': 0,
            'max_chunks_per_concept': 0,
            'max_concepts_per_chunk': 0
        }

        # Find concepts without chunks
        concepts_without_chunks = []
        for concept in self.concepts_data:
            concept_id = concept.get('concept_id', '')
            if concept_id not in self.concept_to_chunks:
                concepts_without_chunks.append({
                    'concept_id': concept_id,
                    'canonical_name': concept.get('canonical_name', '')
                })

        stats['concepts_without_chunks'] = len(concepts_without_chunks)

        # Calculate chunk statistics
        chunks_with_concepts = sum(1 for chunk in self.chunks_data
                                  if chunk.get('concept_memberships'))
        stats['chunks_with_concepts'] = chunks_with_concepts
        stats['chunks_without_concepts'] = len(self.chunks_data) - chunks_with_concepts

        # Calculate averages
        if self.concept_to_chunks:
            chunks_per_concept = [len(chunks) for chunks in self.concept_to_chunks.values()]
            stats['avg_chunks_per_concept'] = sum(chunks_per_concept) / len(chunks_per_concept)
            stats['max_chunks_per_concept'] = max(chunks_per_concept)

        if self.chunks_data:
            concepts_per_chunk = [len(chunk.get('concept_memberships', []))
                                 for chunk in self.chunks_data]
            if concepts_per_chunk:
                stats['avg_concepts_per_chunk'] = sum(concepts_per_chunk) / len(concepts_per_chunk)
                stats['max_concepts_per_chunk'] = max(concepts_per_chunk)

        # Add list of concepts without chunks
        stats['concepts_without_chunks_list'] = concepts_without_chunks

        return stats

    def create_membership_visualization(self):
        """Create interactive visualization of concept-chunk memberships"""

        # Prepare data for visualization
        concept_names = []
        chunk_counts = []

        # Get top concepts by chunk count
        concept_lookup = {c['concept_id']: c for c in self.concepts_data}

        sorted_concepts = sorted(self.concept_to_chunks.items(),
                                key=lambda x: len(x[1]), reverse=True)[:50]  # Top 50

        for concept_id, chunks in sorted_concepts:
            concept_info = concept_lookup.get(concept_id, {})
            name = concept_info.get('canonical_name', concept_id)
            concept_names.append(name[:30])  # Truncate long names
            chunk_counts.append(len(chunks))

        # Create bar chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=chunk_counts,
            y=concept_names,
            orientation='h',
            marker=dict(
                color=chunk_counts,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Chunks")
            ),
            text=chunk_counts,
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Chunks: %{x}<extra></extra>'
        ))

        fig.update_layout(
            title='Top 50 Concepts by Chunk Count',
            xaxis_title='Number of Chunks',
            yaxis_title='Concept Name',
            height=max(600, len(concept_names) * 20),
            width=1000,
            margin=dict(l=200)
        )

        return fig

    def create_distribution_plots(self):
        """Create distribution plots for membership statistics"""

        # Calculate distributions
        chunks_per_concept = [len(chunks) for chunks in self.concept_to_chunks.values()]
        concepts_per_chunk = [len(chunk.get('concept_memberships', []))
                             for chunk in self.chunks_data]

        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Chunks per Concept Distribution',
                          'Concepts per Chunk Distribution')
        )

        # Chunks per concept histogram
        fig.add_trace(
            go.Histogram(
                x=chunks_per_concept,
                nbinsx=30,
                marker_color='blue',
                opacity=0.7,
                name='Chunks per Concept'
            ),
            row=1, col=1
        )

        # Concepts per chunk histogram
        fig.add_trace(
            go.Histogram(
                x=concepts_per_chunk,
                nbinsx=10,
                marker_color='green',
                opacity=0.7,
                name='Concepts per Chunk'
            ),
            row=1, col=2
        )

        fig.update_xaxes(title_text="Number of Chunks", row=1, col=1)
        fig.update_xaxes(title_text="Number of Concepts", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)

        fig.update_layout(
            title='Concept-Chunk Membership Distributions',
            height=500,
            width=1200,
            showlegend=False
        )

        return fig

    def create_membership_matrix(self, max_concepts=30, max_chunks=50):
        """Create a matrix visualization of concept-chunk memberships"""

        # Select top concepts and their chunks
        sorted_concepts = sorted(self.concept_to_chunks.items(),
                                key=lambda x: len(x[1]), reverse=True)[:max_concepts]

        # Get all chunks for these concepts
        all_chunks = set()
        for concept_id, chunks in sorted_concepts:
            for chunk in chunks[:max_chunks]:
                all_chunks.add(chunk['chunk_id'])

        all_chunks = sorted(list(all_chunks))[:max_chunks]

        # Build matrix
        concept_lookup = {c['concept_id']: c for c in self.concepts_data}
        concept_names = []
        matrix = []

        for concept_id, _ in sorted_concepts:
            concept_info = concept_lookup.get(concept_id, {})
            concept_names.append(concept_info.get('canonical_name', concept_id)[:20])

            row = []
            concept_chunks = {chunk['chunk_id'] for chunk in self.concept_to_chunks[concept_id]}
            for chunk_id in all_chunks:
                row.append(1 if chunk_id in concept_chunks else 0)
            matrix.append(row)

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=[f"C{i}" for i in range(len(all_chunks))],  # Abbreviated chunk IDs
            y=concept_names,
            colorscale=[[0, 'white'], [1, 'darkblue']],
            showscale=False,
            hovertemplate='Concept: %{y}<br>Chunk: %{x}<br>Member: %{z}<extra></extra>'
        ))

        fig.update_layout(
            title=f'Concept-Chunk Membership Matrix<br><sub>Top {max_concepts} concepts × {len(all_chunks)} chunks</sub>',
            xaxis_title='Chunk Index',
            yaxis_title='Concept Name',
            height=max(500, len(concept_names) * 20),
            width=max(800, len(all_chunks) * 15),
            xaxis=dict(tickangle=0),
            yaxis=dict(tickfont=dict(size=10))
        )

        return fig

    def save_reports(self):
        """Generate and save all membership reports"""

        print("\n" + "="*70)
        print("GENERATING CONCEPT-CHUNK MEMBERSHIP REPORTS")
        print("="*70)

        # Generate membership DataFrame
        membership_df = self.generate_membership_report()

        # Save as CSV
        csv_path = self.outputs_dir / "A4.2_concept_chunk_membership.csv"
        membership_df.to_csv(csv_path, index=False)
        print(f"[OK] Saved membership CSV to {csv_path.name}")
        print(f"     Total rows: {len(membership_df)}")

        # Save as Excel with formatting
        excel_path = self.outputs_dir / "A4.2_concept_chunk_membership.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            membership_df.to_excel(writer, sheet_name='Memberships', index=False)

            # Add summary sheet
            stats = self.generate_summary_statistics()
            summary_df = pd.DataFrame([
                ['Metric', 'Value'],
                ['Total Concepts', stats['total_concepts']],
                ['Concepts with Chunks', stats['concepts_with_chunks']],
                ['Concepts without Chunks', stats['concepts_without_chunks']],
                ['Total Chunks', stats['total_chunks']],
                ['Chunks with Concepts', stats['chunks_with_concepts']],
                ['Chunks without Concepts', stats['chunks_without_concepts']],
                ['Avg Chunks per Concept', f"{stats['avg_chunks_per_concept']:.2f}"],
                ['Avg Concepts per Chunk', f"{stats['avg_concepts_per_chunk']:.2f}"],
                ['Max Chunks per Concept', stats['max_chunks_per_concept']],
                ['Max Concepts per Chunk', stats['max_concepts_per_chunk']]
            ], columns=['Metric', 'Value'])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

            # Add concepts without chunks sheet
            if stats['concepts_without_chunks_list']:
                no_chunks_df = pd.DataFrame(stats['concepts_without_chunks_list'])
                no_chunks_df.to_excel(writer, sheet_name='Concepts Without Chunks', index=False)

        print(f"[OK] Saved membership Excel to {excel_path.name}")

        # Save statistics as JSON
        stats_path = self.outputs_dir / "A4.2_membership_statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            # Remove the list from stats for JSON (it's in Excel)
            stats_copy = stats.copy()
            stats_copy.pop('concepts_without_chunks_list', None)
            json.dump(stats_copy, f, indent=2)
        print(f"[OK] Saved statistics to {stats_path.name}")

        # Create visualizations
        print("\nGenerating visualizations...")

        # Top concepts bar chart
        bar_fig = self.create_membership_visualization()
        bar_path = self.outputs_dir / "A4.2_top_concepts_by_chunks.html"
        bar_fig.write_html(str(bar_path))
        print(f"[OK] Saved top concepts chart to {bar_path.name}")

        # Distribution plots
        dist_fig = self.create_distribution_plots()
        dist_path = self.outputs_dir / "A4.2_membership_distributions.html"
        dist_fig.write_html(str(dist_path))
        print(f"[OK] Saved distribution plots to {dist_path.name}")

        # Membership matrix
        matrix_fig = self.create_membership_matrix()
        matrix_path = self.outputs_dir / "A4.2_membership_matrix.html"
        matrix_fig.write_html(str(matrix_path))
        print(f"[OK] Saved membership matrix to {matrix_path.name}")

        # Print summary
        print("\n" + "="*70)
        print("MEMBERSHIP ANALYSIS SUMMARY")
        print("="*70)
        print(f"Total Concepts: {stats['total_concepts']}")
        print(f"Concepts with chunks: {stats['concepts_with_chunks']} ({stats['concepts_with_chunks']/stats['total_concepts']*100:.1f}%)")
        print(f"Concepts without chunks: {stats['concepts_without_chunks']}")
        print(f"Average chunks per concept: {stats['avg_chunks_per_concept']:.2f}")
        print(f"Average concepts per chunk: {stats['avg_concepts_per_chunk']:.2f}")

        if stats['concepts_without_chunks'] > 0:
            print("\nConcepts without chunks (first 10):")
            for concept in stats.get('concepts_without_chunks_list', [])[:10]:
                print(f"  - {concept['canonical_name']} ({concept['concept_id']})")

        return stats


def main():
    """Main execution function"""
    analyzer = ConceptChunkMembershipAnalyzer()

    # Load data
    if not analyzer.load_data():
        print("Failed to load data")
        return

    # Analyze memberships
    analyzer.analyze_memberships()

    # Generate and save reports
    stats = analyzer.save_reports()

    print("\n" + "="*70)
    print("CONCEPT-CHUNK MEMBERSHIP ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated outputs:")
    print("  1. A4.2_concept_chunk_membership.csv - Full membership list")
    print("  2. A4.2_concept_chunk_membership.xlsx - Excel report with summary")
    print("  3. A4.2_membership_statistics.json - Statistical summary")
    print("  4. A4.2_top_concepts_by_chunks.html - Bar chart visualization")
    print("  5. A4.2_membership_distributions.html - Distribution plots")
    print("  6. A4.2_membership_matrix.html - Membership matrix heatmap")
    print("\nUse the CSV or Excel file to see detailed concept-chunk mappings!")


if __name__ == "__main__":
    main()