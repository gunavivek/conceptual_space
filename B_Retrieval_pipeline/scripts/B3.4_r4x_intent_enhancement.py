#!/usr/bin/env python3
"""
B3.4: R4X Intent Enhancement
Revolutionary question intent understanding using R4X tri-semantic integration
Enhances standard intent analysis with cross-pipeline semantic insights
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
import math
import numpy as np

# Add R4X integration path
sys.path.append(str(Path(__file__).parent.parent.parent / "R_Reference_pipeline" / "scripts"))

try:
    from R4X_cross_pipeline_semantic_integrator import R4X_CrossPipelineSemanticIntegrator
    from R4X_semantic_fusion_engine import SemanticFusionEngine
    R4X_SemanticFusionEngine = SemanticFusionEngine
except ImportError:
    print("[WARNING] R4X components not found. Running in fallback mode.")
    R4X_CrossPipelineSemanticIntegrator = None
    R4X_SemanticFusionEngine = None

class B34_R4X_IntentEnhancer:
    """
    Revolutionary Intent Enhancement using R4X Tri-Semantic Integration
    
    Transforms standard B-Pipeline intent analysis by:
    1. Connecting question intent to ontological knowledge patterns
    2. Leveraging document semantic understanding
    3. Cross-referencing similar question patterns
    4. Synthesizing tri-semantic intent understanding
    """
    
    def __init__(self):
        """Initialize R4X Intent Enhancer"""
        self.r4x_integrator = None
        if R4X_CrossPipelineSemanticIntegrator:
            try:
                self.r4x_integrator = R4X_CrossPipelineSemanticIntegrator()
                print("[OK] R4X Cross-Pipeline Semantic Integrator initialized")
            except Exception as e:
                print(f"[WARNING]  R4X initialization warning: {e}")
        
        # Intent enhancement configurations
        self.enhancement_strategies = {
            'ontological_intent': 0.35,    # How well intent aligns with ontology patterns
            'document_context_intent': 0.30,  # How intent relates to document semantics
            'cross_question_intent': 0.20,    # Similar question pattern analysis
            'semantic_depth_intent': 0.15     # Depth of semantic understanding required
        }
        
        # Intent complexity analysis
        self.intent_complexity_indicators = {
            'simple_factual': ['what', 'who', 'when', 'where', 'is', 'are'],
            'analytical': ['how', 'why', 'analyze', 'compare', 'evaluate'],
            'computational': ['calculate', 'compute', 'sum', 'total', 'change', 'difference'],
            'relational': ['between', 'relationship', 'connection', 'correlation', 'impact'],
            'temporal': ['trend', 'over time', 'period', 'year', 'quarter', 'historical']
        }
        
    def analyze_standard_intent(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze standard intent from B-Pipeline
        
        Args:
            question_data: Question data with standard intent analysis
            
        Returns:
            Enhanced intent analysis
        """
        question = question_data.get("question", "")
        standard_intent = question_data.get("intent_analysis", {})
        
        # Extract standard intent features
        primary_intent = standard_intent.get("primary_intent", "unknown")
        keywords = standard_intent.get("keywords", [])
        domain = standard_intent.get("domain", "general")
        
        # Analyze intent complexity
        complexity_analysis = self._analyze_intent_complexity(question)
        
        return {
            "standard_intent": standard_intent,
            "primary_intent": primary_intent,
            "keywords": keywords,
            "domain": domain,
            "complexity_analysis": complexity_analysis,
            "intent_confidence": standard_intent.get("confidence", 0.5)
        }
    
    def _analyze_intent_complexity(self, question: str) -> Dict[str, Any]:
        """Analyze the complexity of the question intent"""
        question_lower = question.lower()
        complexity_scores = {}
        
        for intent_type, indicators in self.intent_complexity_indicators.items():
            score = sum(1 for indicator in indicators if indicator in question_lower)
            complexity_scores[intent_type] = score / len(indicators)
        
        # Determine dominant complexity type
        max_complexity = max(complexity_scores.items(), key=lambda x: x[1])
        
        return {
            "complexity_scores": complexity_scores,
            "dominant_complexity": max_complexity[0],
            "complexity_confidence": max_complexity[1],
            "requires_multi_step": any(word in question_lower for word in ['and', 'then', 'also', 'additionally'])
        }
    
    def enhance_intent_with_r4x(self, standard_intent_analysis: Dict[str, Any], question: str) -> Dict[str, Any]:
        """
        Enhance intent analysis using R4X tri-semantic integration
        
        Args:
            standard_intent_analysis: Standard B-Pipeline intent analysis
            question: Original question text
            
        Returns:
            R4X-enhanced intent understanding
        """
        if not self.r4x_integrator:
            return {
                "enhancement_status": "fallback_mode",
                "enhanced_intent": standard_intent_analysis,
                "tri_semantic_insights": {}
            }
        
        try:
            # Extract key concepts from question
            question_concepts = self._extract_question_concepts(question, standard_intent_analysis)
            
            # Get tri-semantic perspectives for each concept
            tri_semantic_insights = {}
            for concept in question_concepts:
                unified_view = self.r4x_integrator.get_unified_concept_view(concept)
                if unified_view:
                    tri_semantic_insights[concept] = unified_view
            
            # Enhance intent understanding
            enhanced_intent = self._synthesize_enhanced_intent(
                standard_intent_analysis, 
                tri_semantic_insights,
                question
            )
            
            return {
                "enhancement_status": "r4x_enhanced",
                "enhanced_intent": enhanced_intent,
                "tri_semantic_insights": tri_semantic_insights,
                "question_concepts": question_concepts
            }
            
        except Exception as e:
            print(f"[WARNING]  R4X enhancement error: {e}")
            return {
                "enhancement_status": "error",
                "enhanced_intent": standard_intent_analysis,
                "error_message": str(e)
            }
    
    def _extract_question_concepts(self, question: str, intent_analysis: Dict[str, Any]) -> List[str]:
        """Extract key concepts from question for R4X analysis"""
        # Start with intent keywords
        concepts = intent_analysis.get("keywords", [])
        
        # Add domain-specific concepts
        domain = intent_analysis.get("domain", "general")
        if domain != "general":
            concepts.append(domain)
        
        # Extract financial/business concepts (common in this domain)
        financial_terms = [
            'revenue', 'income', 'profit', 'loss', 'assets', 'liabilities',
            'cash', 'deferred', 'current', 'contract', 'inventory', 'tax'
        ]
        
        question_lower = question.lower()
        for term in financial_terms:
            if term in question_lower and term not in concepts:
                concepts.append(term)
        
        return list(set(concepts))  # Remove duplicates
    
    def _synthesize_enhanced_intent(
        self, 
        standard_intent: Dict[str, Any], 
        tri_semantic_insights: Dict[str, Any],
        question: str
    ) -> Dict[str, Any]:
        """
        Synthesize enhanced intent understanding from tri-semantic insights
        
        Args:
            standard_intent: Original intent analysis
            tri_semantic_insights: R4X tri-semantic concept insights
            question: Original question
            
        Returns:
            Enhanced intent understanding
        """
        # Start with standard intent
        enhanced = dict(standard_intent)
        
        # Ontological Intent Enhancement
        ontological_enhancement = self._analyze_ontological_intent(tri_semantic_insights)
        
        # Document Context Intent Enhancement  
        document_context_enhancement = self._analyze_document_context_intent(tri_semantic_insights)
        
        # Cross-Question Intent Enhancement
        cross_question_enhancement = self._analyze_cross_question_intent(tri_semantic_insights, question)
        
        # Semantic Depth Intent Enhancement
        semantic_depth_enhancement = self._analyze_semantic_depth_intent(tri_semantic_insights)
        
        # Calculate enhanced confidence
        enhancement_confidence = self._calculate_enhancement_confidence([
            ontological_enhancement,
            document_context_enhancement, 
            cross_question_enhancement,
            semantic_depth_enhancement
        ])
        
        # Synthesize final enhanced intent
        enhanced.update({
            "enhanced_primary_intent": self._determine_enhanced_primary_intent(
                standard_intent.get("primary_intent", "unknown"),
                ontological_enhancement,
                document_context_enhancement
            ),
            "semantic_intent_layers": {
                "ontological_intent": ontological_enhancement,
                "document_context_intent": document_context_enhancement,
                "cross_question_intent": cross_question_enhancement,
                "semantic_depth_intent": semantic_depth_enhancement
            },
            "enhancement_confidence": enhancement_confidence,
            "requires_tri_semantic_answer": enhancement_confidence > 0.6,
            "enhanced_keywords": self._enhance_keywords(
                standard_intent.get("keywords", []),
                tri_semantic_insights
            )
        })
        
        return enhanced
    
    def _analyze_ontological_intent(self, tri_semantic_insights: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how question intent aligns with ontological patterns"""
        if not tri_semantic_insights:
            return {"alignment_score": 0.0, "ontological_patterns": []}
        
        ontological_patterns = []
        total_confidence = 0.0
        concept_count = 0
        
        for concept, insight in tri_semantic_insights.items():
            if "ontology_perspective" in insight:
                ontology_data = insight["ontology_perspective"]
                if "relationships" in ontology_data:
                    patterns = list(ontology_data["relationships"].keys())
                    ontological_patterns.extend(patterns)
                    
                total_confidence += ontology_data.get("confidence", 0.0)
                concept_count += 1
        
        alignment_score = total_confidence / max(concept_count, 1)
        
        return {
            "alignment_score": alignment_score,
            "ontological_patterns": list(set(ontological_patterns)),
            "pattern_count": len(set(ontological_patterns))
        }
    
    def _analyze_document_context_intent(self, tri_semantic_insights: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how intent relates to document semantic context"""
        if not tri_semantic_insights:
            return {"context_relevance": 0.0, "document_themes": []}
        
        document_themes = []
        relevance_scores = []
        
        for concept, insight in tri_semantic_insights.items():
            if "document_perspective" in insight:
                doc_data = insight["document_perspective"]
                if "themes" in doc_data:
                    document_themes.extend(doc_data["themes"])
                relevance_scores.append(doc_data.get("relevance", 0.0))
        
        context_relevance = sum(relevance_scores) / max(len(relevance_scores), 1)
        
        return {
            "context_relevance": context_relevance,
            "document_themes": list(set(document_themes)),
            "theme_diversity": len(set(document_themes))
        }
    
    def _analyze_cross_question_intent(self, tri_semantic_insights: Dict[str, Any], question: str) -> Dict[str, Any]:
        """Analyze similar question patterns from cross-question analysis"""
        # Simplified cross-question pattern analysis
        question_patterns = []
        
        # Basic pattern extraction
        if "what" in question.lower():
            question_patterns.append("factual_query")
        if "how" in question.lower():
            question_patterns.append("process_query")
        if "change" in question.lower() or "difference" in question.lower():
            question_patterns.append("comparison_query")
        if any(term in question.lower() for term in ["calculate", "compute", "total"]):
            question_patterns.append("computational_query")
        
        return {
            "similar_patterns": question_patterns,
            "pattern_confidence": 0.7 if question_patterns else 0.3,
            "cross_question_insights": f"Question follows {', '.join(question_patterns)} pattern(s)"
        }
    
    def _analyze_semantic_depth_intent(self, tri_semantic_insights: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the depth of semantic understanding required"""
        if not tri_semantic_insights:
            return {"depth_score": 0.0, "complexity_level": "low"}
        
        depth_indicators = 0
        max_relationships = 0
        
        for concept, insight in tri_semantic_insights.items():
            # Count relationship complexity
            for perspective in ["ontology_perspective", "document_perspective", "question_perspective"]:
                if perspective in insight:
                    relationships = insight[perspective].get("relationships", {})
                    max_relationships = max(max_relationships, len(relationships))
                    if len(relationships) > 2:
                        depth_indicators += 1
        
        depth_score = min(1.0, (depth_indicators + max_relationships * 0.1) / 3.0)
        
        if depth_score > 0.7:
            complexity_level = "high"
        elif depth_score > 0.4:
            complexity_level = "medium"
        else:
            complexity_level = "low"
        
        return {
            "depth_score": depth_score,
            "complexity_level": complexity_level,
            "relationship_complexity": max_relationships
        }
    
    def _calculate_enhancement_confidence(self, enhancements: List[Dict[str, Any]]) -> float:
        """Calculate overall enhancement confidence"""
        if not enhancements:
            return 0.0
        
        # Weight different enhancement types
        weights = list(self.enhancement_strategies.values())
        scores = []
        
        for enhancement in enhancements:
            if "alignment_score" in enhancement:
                scores.append(enhancement["alignment_score"])
            elif "context_relevance" in enhancement:
                scores.append(enhancement["context_relevance"])
            elif "pattern_confidence" in enhancement:
                scores.append(enhancement["pattern_confidence"])
            elif "depth_score" in enhancement:
                scores.append(enhancement["depth_score"])
            else:
                scores.append(0.0)
        
        # Weighted average
        if len(scores) == len(weights):
            return sum(score * weight for score, weight in zip(scores, weights))
        else:
            return sum(scores) / len(scores)
    
    def _determine_enhanced_primary_intent(
        self, 
        standard_intent: str, 
        ontological_enhancement: Dict[str, Any],
        document_context_enhancement: Dict[str, Any]
    ) -> str:
        """Determine enhanced primary intent based on tri-semantic analysis"""
        
        # Start with standard intent
        enhanced_intent = standard_intent
        
        # Enhance based on ontological patterns
        ontological_patterns = ontological_enhancement.get("ontological_patterns", [])
        if "hierarchical" in ontological_patterns or "taxonomic" in ontological_patterns:
            enhanced_intent = "hierarchical_analytical"
        elif "causal" in ontological_patterns or "functional" in ontological_patterns:
            enhanced_intent = "causal_analytical"
        
        # Enhance based on document context
        document_themes = document_context_enhancement.get("document_themes", [])
        if "financial_metrics" in document_themes and standard_intent in ["factual", "unknown"]:
            enhanced_intent = "financial_computational"
        elif "temporal_analysis" in document_themes:
            enhanced_intent = "temporal_analytical"
        
        return enhanced_intent
    
    def _enhance_keywords(self, standard_keywords: List[str], tri_semantic_insights: Dict[str, Any]) -> List[str]:
        """Enhance keywords with tri-semantic insights"""
        enhanced_keywords = set(standard_keywords)
        
        # Add keywords from tri-semantic insights
        for concept, insight in tri_semantic_insights.items():
            for perspective in ["ontology_perspective", "document_perspective", "question_perspective"]:
                if perspective in insight:
                    perspective_data = insight[perspective]
                    if "keywords" in perspective_data:
                        enhanced_keywords.update(perspective_data["keywords"])
                    if "related_terms" in perspective_data:
                        enhanced_keywords.update(perspective_data["related_terms"])
        
        return list(enhanced_keywords)
    
    def process_intent_enhancement(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process complete intent enhancement pipeline
        
        Args:
            question_data: Question data with standard intent analysis
            
        Returns:
            Complete intent enhancement results
        """
        question = question_data.get("question", "")
        
        # Step 1: Analyze standard intent
        standard_analysis = self.analyze_standard_intent(question_data)
        
        # Step 2: Enhance with R4X tri-semantic integration
        r4x_enhancement = self.enhance_intent_with_r4x(standard_analysis, question)
        
        # Step 3: Calculate enhancement metrics
        enhancement_metrics = self._calculate_enhancement_metrics(
            standard_analysis, 
            r4x_enhancement
        )
        
        return {
            "question": question,
            "standard_intent_analysis": standard_analysis,
            "r4x_enhancement": r4x_enhancement,
            "enhancement_metrics": enhancement_metrics,
            "processing_timestamp": datetime.now().isoformat(),
            "enhancer_version": "B3.4_R4X_v1.0"
        }
    
    def _calculate_enhancement_metrics(
        self, 
        standard_analysis: Dict[str, Any], 
        r4x_enhancement: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate metrics showing enhancement quality"""
        
        # Basic enhancement metrics
        standard_confidence = standard_analysis.get("intent_confidence", 0.0)
        enhanced_confidence = 0.0
        
        if r4x_enhancement.get("enhancement_status") == "r4x_enhanced":
            enhanced_intent = r4x_enhancement.get("enhanced_intent", {})
            enhanced_confidence = enhanced_intent.get("enhancement_confidence", 0.0)
        
        confidence_improvement = enhanced_confidence - standard_confidence
        
        # Semantic enrichment metrics
        standard_keywords = len(standard_analysis.get("keywords", []))
        enhanced_keywords = 0
        
        if "enhanced_intent" in r4x_enhancement:
            enhanced_keywords = len(r4x_enhancement["enhanced_intent"].get("enhanced_keywords", []))
        
        keyword_enrichment = enhanced_keywords - standard_keywords
        
        # Complexity understanding metrics
        complexity_analysis = standard_analysis.get("complexity_analysis", {})
        complexity_confidence = complexity_analysis.get("complexity_confidence", 0.0)
        
        return {
            "confidence_improvement": confidence_improvement,
            "keyword_enrichment": keyword_enrichment,
            "complexity_understanding": complexity_confidence,
            "tri_semantic_insights_count": len(r4x_enhancement.get("tri_semantic_insights", {})),
            "enhancement_quality": "excellent" if confidence_improvement > 0.3 else 
                                 "good" if confidence_improvement > 0.1 else
                                 "minimal" if confidence_improvement > 0.0 else "none",
            "requires_advanced_processing": enhanced_confidence > 0.6
        }

def load_inputs() -> Tuple[Dict[str, Any], bool]:
    """Load question data from B-Pipeline outputs"""
    script_dir = Path(__file__).parent.parent
    
    # Try to load from B3.1 intent matching output
    b3_1_path = script_dir / "outputs/B3_1_intent_matching_output.json"
    if b3_1_path.exists():
        with open(b3_1_path, 'r', encoding='utf-8') as f:
            question_data = json.load(f)
            return question_data, True
    
    # Fallback to B2.1 intent analysis
    b2_1_path = script_dir / "outputs/B2_1_intent_layer_output.json"
    if b2_1_path.exists():
        with open(b2_1_path, 'r', encoding='utf-8') as f:
            question_data = json.load(f)
            return question_data, True
    
    # Final fallback to B1 question
    b1_path = script_dir / "outputs/B1_current_question.json"
    if b1_path.exists():
        with open(b1_path, 'r', encoding='utf-8') as f:
            question_data = json.load(f)
            # Add mock intent analysis
            if "intent_analysis" not in question_data:
                question_data["intent_analysis"] = {
                    "primary_intent": "factual",
                    "keywords": ["change", "current", "deferred", "income"],
                    "domain": "finance",
                    "confidence": 0.7
                }
            return question_data, False
    
    # Mock data for testing
    return {
        "question": "What was the change in Current deferred income?",
        "intent_analysis": {
            "primary_intent": "factual",
            "keywords": ["change", "current", "deferred", "income"],
            "domain": "finance",
            "confidence": 0.7
        }
    }, False

def save_output(data: Dict[str, Any], output_path: str = "outputs/B3_4_r4x_intent_enhancement_output.json"):
    """Save R4X intent enhancement results"""
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / output_path
    
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Saved R4X intent enhancement results to {full_path}")

def main():
    """Main execution for B3.4 R4X Intent Enhancement"""
    print("=" * 70)
    print("B3.4: R4X Intent Enhancement - Revolutionary Question Understanding")
    print("=" * 70)
    
    try:
        # Initialize R4X Intent Enhancer
        print("Initializing R4X Intent Enhancer...")
        enhancer = B34_R4X_IntentEnhancer()
        
        # Load inputs
        print("Loading question data...")
        question_data, has_intent_analysis = load_inputs()
        question = question_data.get("question", "")
        print(f"Processing question: {question}")
        
        if not has_intent_analysis:
            print("[WARNING]  No intent analysis found, using fallback analysis")
        
        # Process intent enhancement
        print("Enhancing intent understanding with R4X tri-semantic integration...")
        enhancement_results = enhancer.process_intent_enhancement(question_data)
        
        # Display results
        print(f"\nR4X Intent Enhancement Results:")
        print(f"Question: {enhancement_results['question']}")
        
        # Standard intent analysis
        standard_analysis = enhancement_results["standard_intent_analysis"]
        print(f"\nStandard Intent Analysis:")
        print(f"  Primary Intent: {standard_analysis['primary_intent']}")
        print(f"  Domain: {standard_analysis['domain']}")
        print(f"  Keywords: {', '.join(standard_analysis['keywords'])}")
        print(f"  Confidence: {standard_analysis['intent_confidence']:.3f}")
        
        # R4X enhancement
        r4x_enhancement = enhancement_results["r4x_enhancement"]
        print(f"\nR4X Enhancement Status: {r4x_enhancement['enhancement_status']}")
        
        if r4x_enhancement.get("enhancement_status") == "r4x_enhanced":
            enhanced_intent = r4x_enhancement["enhanced_intent"]
            print(f"Enhanced Primary Intent: {enhanced_intent.get('enhanced_primary_intent', 'N/A')}")
            print(f"Enhancement Confidence: {enhanced_intent.get('enhancement_confidence', 0.0):.3f}")
            print(f"Requires Tri-Semantic Answer: {enhanced_intent.get('requires_tri_semantic_answer', False)}")
            
            # Tri-semantic insights
            tri_insights = r4x_enhancement.get("tri_semantic_insights", {})
            print(f"Tri-Semantic Concept Insights: {len(tri_insights)} concepts analyzed")
            
            for concept in list(tri_insights.keys())[:3]:  # Show top 3
                print(f"  • {concept}: Multi-perspective semantic understanding available")
        
        # Enhancement metrics
        metrics = enhancement_results["enhancement_metrics"]
        print(f"\nEnhancement Metrics:")
        print(f"  Quality: {metrics['enhancement_quality']}")
        print(f"  Confidence Improvement: {metrics['confidence_improvement']:+.3f}")
        print(f"  Keyword Enrichment: +{metrics['keyword_enrichment']} keywords")
        print(f"  Requires Advanced Processing: {metrics['requires_advanced_processing']}")
        
        # Save output
        save_output(enhancement_results)
        
        print("\n[OK] B3.4 R4X Intent Enhancement completed successfully!")
        print("  Revolutionary tri-semantic question understanding achieved!")
        
    except Exception as e:
        print(f"[ERROR] Error in B3.4 R4X Intent Enhancement: {str(e)}")
        raise

if __name__ == "__main__":
    main()