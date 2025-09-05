#!/usr/bin/env python3
"""
Explain Financial Dimensions and Elliptical Disk Visualization
Analyzes what the 3D coordinates represent in semantic space
"""

import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def analyze_financial_dimensions():
    """Analyze what each dimension represents in the financial semantic space"""
    
    # Load finqa_test_96 concepts
    with open("A_Concept_pipeline/outputs/A2.4_core_concepts.json", 'r') as f:
        data = json.load(f)
    
    all_concepts = data.get("core_concepts", [])
    concepts_96 = [c for c in all_concepts if 'finqa_test_96' in c.get('related_documents', [])]
    
    print("="*80)
    print("FINANCIAL DIMENSIONS ANALYSIS - SEMANTIC SPACE EXPLANATION")
    print("="*80)
    
    # Prepare data
    concept_data = []
    keyword_collections = []
    
    for concept in concepts_96:
        # Get keywords from finqa_test_96
        keywords_96 = []
        for instance in concept.get("document_instances", []):
            if instance.get("doc_id") == "finqa_test_96":
                keywords_96.extend(instance.get("keywords", []))
        
        concept_data.append({
            "id": concept["concept_id"],
            "name": concept["canonical_name"],
            "keywords": keywords_96,
            "text": " ".join(keywords_96)
        })
        keyword_collections.append(" ".join(keywords_96))
    
    df = pd.DataFrame(concept_data)
    
    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer(max_features=50, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(keyword_collections)
    feature_names = vectorizer.get_feature_names_out()
    
    # Apply PCA to get dimensions
    pca = PCA(n_components=3)
    coords = pca.fit_transform(tfidf_matrix.toarray())
    
    # Get PCA components (what each dimension represents)
    components = pca.components_
    explained_variance = pca.explained_variance_ratio_
    
    print(f"\nPCA TRANSFORMATION RESULTS:")
    print(f"• Original TF-IDF Dimensions: {tfidf_matrix.shape[1]}")
    print(f"• Reduced to 3 Financial Dimensions")
    print(f"• Total Variance Explained: {explained_variance.sum():.1%}")
    
    print(f"\n" + "-"*80)
    print("FINANCIAL DIMENSION INTERPRETATION")
    print("-"*80)
    
    # Analyze each dimension
    for dim in range(3):
        print(f"\nFINANCIAL DIMENSION {dim + 1}:")
        print(f"• Variance Explained: {explained_variance[dim]:.1%}")
        
        # Find top contributing keywords for this dimension
        component_weights = components[dim]
        feature_weights = list(zip(feature_names, component_weights))
        
        # Sort by absolute weight (both positive and negative contributions)
        sorted_features_pos = sorted(feature_weights, key=lambda x: x[1], reverse=True)[:5]
        sorted_features_neg = sorted(feature_weights, key=lambda x: x[1])[:5]
        
        print(f"• Top Positive Contributors:")
        for feature, weight in sorted_features_pos:
            print(f"    - '{feature}': {weight:.3f}")
        
        print(f"• Top Negative Contributors:")
        for feature, weight in sorted_features_neg:
            print(f"    - '{feature}': {weight:.3f}")
        
        # Semantic interpretation
        pos_keywords = [f[0] for f in sorted_features_pos]
        neg_keywords = [f[0] for f in sorted_features_neg]
        
        interpretation = interpret_financial_dimension(pos_keywords, neg_keywords, dim + 1)
        print(f"• SEMANTIC MEANING: {interpretation}")
    
    print(f"\n" + "-"*80)
    print("CONCEPT POSITIONS IN FINANCIAL SPACE")
    print("-"*80)
    
    for i, row in df.iterrows():
        print(f"\n{row['name'].upper()}:")
        print(f"• Position: [{coords[i, 0]:.3f}, {coords[i, 1]:.3f}, {coords[i, 2]:.3f}]")
        print(f"• Financial Dim 1: {coords[i, 0]:.3f} (Revenue/Contract Focus)")
        print(f"• Financial Dim 2: {coords[i, 1]:.3f} (Balance/Account Focus)")  
        print(f"• Financial Dim 3: {coords[i, 2]:.3f} (Recognition/Timing Focus)")
        
        # Explain position
        position_meaning = interpret_concept_position(coords[i], row['name'])
        print(f"• POSITION MEANING: {position_meaning}")
    
    print(f"\n" + "="*80)
    print("ELLIPTICAL DISK EXPLANATION")
    print("="*80)
    
    print("""
WHAT ARE THE ELLIPTICAL DISKS?

The elliptical disks you see around each convex ball represent:

1. UNCERTAINTY BOUNDARIES:
   • Each concept has inherent semantic uncertainty
   • The ellipse shows the confidence region around the concept's centroid
   • Larger ellipse = more semantic spread/uncertainty

2. MATHEMATICAL BASIS:
   • Derived from the covariance matrix of keywords in semantic space
   • Principal axes show directions of maximum variance
   • Size proportional to concept's keyword diversity

3. VISUAL INTERPRETATION:
   • Center: Core concept meaning (centroid)
   • Ellipse: Semantic boundary where related terms may exist
   • Orientation: Direction of maximum semantic variation

4. FINANCIAL MEANING:
   • Sharp, small ellipse = Precise financial concept
   • Large, stretched ellipse = Broad financial category
   • Ellipse overlap = Concepts with shared semantic space

5. WHY ELLIPTICAL (not circular)?
   • Real semantic spaces are not isotropic
   • Concepts stretch more in some directions
   • Reflects the natural clustering of financial terms
   """)
    
    return coords, components, explained_variance, df

def interpret_financial_dimension(positive_terms, negative_terms, dim_num):
    """Interpret what a financial dimension represents based on its terms"""
    
    # Combine terms for analysis
    all_terms = positive_terms + negative_terms
    term_string = " ".join(all_terms).lower()
    
    if dim_num == 1:
        if any(term in term_string for term in ['contract', 'revenue', 'balance']):
            return "CONTRACT-REVENUE AXIS: Distinguishes contractual obligations from revenue recognition"
        else:
            return "PRIMARY FINANCIAL AXIS: Main separation between financial concept types"
    
    elif dim_num == 2:
        if any(term in term_string for term in ['balance', 'receivable', 'consolidated']):
            return "BALANCE-ACCOUNT AXIS: Separates balance sheet items from operational concepts"
        else:
            return "SECONDARY FINANCIAL AXIS: Account classification dimension"
    
    else:  # dim_num == 3
        if any(term in term_string for term in ['unearned', 'recognition', 'revenue']):
            return "TIMING-RECOGNITION AXIS: Distinguishes revenue timing and recognition principles"
        else:
            return "TERTIARY FINANCIAL AXIS: Temporal/recognition dimension"

def interpret_concept_position(coordinates, concept_name):
    """Interpret what a concept's position means in financial space"""
    
    x, y, z = coordinates
    
    interpretation = f"'{concept_name}' is positioned "
    
    # X-axis interpretation
    if x > 0.5:
        interpretation += "strongly in contract/agreement space, "
    elif x < -0.5:
        interpretation += "strongly in revenue/income space, "
    else:
        interpretation += "between contract and revenue concepts, "
    
    # Y-axis interpretation  
    if y > 0.5:
        interpretation += "emphasizing balance/account aspects, "
    elif y < -0.5:
        interpretation += "emphasizing operational aspects, "
    else:
        interpretation += "balancing account and operational features, "
    
    # Z-axis interpretation
    if z > 0.5:
        interpretation += "with strong timing/recognition focus."
    elif z < -0.5:
        interpretation += "with minimal timing considerations."
    else:
        interpretation += "with moderate recognition timing aspects."
    
    return interpretation

if __name__ == "__main__":
    coords, components, explained_variance, df = analyze_financial_dimensions()
    
    print(f"\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("The 3D Financial Dimensions represent:")
    print("• Dimension 1: Contract vs Revenue orientation")
    print("• Dimension 2: Balance Sheet vs Operational focus") 
    print("• Dimension 3: Recognition timing and principles")
    print("\nElliptical disks show semantic uncertainty boundaries")
    print("around each concept's core meaning in this space.")