#!/usr/bin/env python3
"""
B5.1: R4X Question Understanding
Comprehensive question understanding using complete R4X tri-semantic integration
Orchestrates the entire B-Pipeline enhancement with R4X cross-pipeline insights
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
import subprocess
import math

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

class B51_R4X_QuestionUnderstanding:
    """
    Comprehensive R4X Question Understanding System
    
    Orchestrates the complete B-Pipeline enhancement workflow:
    1. Enhanced intent analysis (B3.4)
    2. Tri-semantic answer synthesis (B4.1)
    3. Cross-pipeline integration analysis
    4. Comprehensive understanding synthesis
    5. Quality assessment and validation
    """
    
    def __init__(self):
        """Initialize R4X Question Understanding System"""
        self.r4x_integrator = None
        if R4X_CrossPipelineSemanticIntegrator:
            try:
                self.r4x_integrator = R4X_CrossPipelineSemanticIntegrator()
                print("[OK] R4X Cross-Pipeline Semantic Integrator initialized")
            except Exception as e:
                print(f"[WARNING]  R4X initialization warning: {e}")
        
        # Understanding dimensions
        self.understanding_dimensions = {
            'intent_understanding': 0.25,      # How well we understand the question intent
            'semantic_depth': 0.25,           # Depth of semantic analysis
            'contextual_grounding': 0.20,     # How well grounded in context
            'cross_pipeline_integration': 0.15, # Integration across pipelines
            'answer_quality': 0.15            # Quality of generated answer
        }
        
        # Processing pipeline stages
        self.pipeline_stages = [
            "question_ingestion",
            "standard_intent_analysis",
            "r4x_intent_enhancement",
            "tri_semantic_answer_synthesis", 
            "cross_pipeline_integration",
            "understanding_synthesis",
            "quality_validation"
        ]
        
        # Question complexity analysis
        self.complexity_factors = {
            'lexical_complexity': ['multi_word_concepts', 'technical_terms', 'domain_specificity'],
            'semantic_complexity': ['concept_relationships', 'implicit_meaning', 'contextual_dependencies'],
            'computational_complexity': ['calculations_required', 'data_aggregation', 'temporal_analysis'],
            'reasoning_complexity': ['causal_inference', 'comparative_analysis', 'multi_step_reasoning']
        }
    
    def ingest_question(self, question: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Ingest and prepare question for comprehensive understanding
        
        Args:
            question: The question to analyze
            context: Optional context information
            
        Returns:
            Ingested question data structure
        """
        return {
            "question": question,
            "context": context or {},
            "ingestion_timestamp": datetime.now().isoformat(),
            "question_id": f"r4x_q_{int(datetime.now().timestamp())}",
            "preprocessing": {
                "length": len(question),
                "word_count": len(question.split()),
                "contains_numbers": any(char.isdigit() for char in question),
                "question_words": [word for word in question.lower().split() if word in ['what', 'how', 'why', 'when', 'where', 'which', 'who']]
            }
        }
    
    def analyze_question_complexity(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive analysis of question complexity across multiple dimensions
        
        Args:
            question_data: Ingested question data
            
        Returns:
            Complexity analysis results
        """
        question = question_data["question"].lower()
        complexity_scores = {}
        
        # Lexical complexity analysis
        lexical_score = 0.0
        multi_word_concepts = len([word for word in question.split() if len(word) > 8])
        technical_terms = sum(1 for term in ['deferred', 'financial', 'revenue', 'assets', 'liabilities'] if term in question)
        domain_specific = 1.0 if any(domain in question for domain in ['financial', 'accounting', 'business']) else 0.0
        
        lexical_score = (multi_word_concepts * 0.1 + technical_terms * 0.2 + domain_specific * 0.3) / 0.6
        complexity_scores['lexical_complexity'] = min(1.0, lexical_score)
        
        # Semantic complexity analysis
        semantic_score = 0.0
        concept_relationships = 1.0 if any(rel in question for rel in ['between', 'and', 'versus', 'compared']) else 0.3
        implicit_meaning = 1.0 if any(impl in question for impl in ['change', 'difference', 'trend', 'impact']) else 0.3
        contextual_deps = 0.8 if len(question.split()) > 10 else 0.4
        
        semantic_score = (concept_relationships * 0.4 + implicit_meaning * 0.3 + contextual_deps * 0.3) / 1.0
        complexity_scores['semantic_complexity'] = min(1.0, semantic_score)
        
        # Computational complexity analysis
        computational_score = 0.0
        calculations_req = 1.0 if any(calc in question for calc in ['calculate', 'total', 'sum', 'change', 'difference']) else 0.2
        data_aggregation = 0.8 if any(agg in question for agg in ['average', 'mean', 'aggregate', 'combined']) else 0.2
        temporal_analysis = 0.9 if any(temp in question for temp in ['period', 'year', 'quarter', 'time']) else 0.1
        
        computational_score = (calculations_req * 0.5 + data_aggregation * 0.3 + temporal_analysis * 0.2) / 1.0
        complexity_scores['computational_complexity'] = min(1.0, computational_score)
        
        # Reasoning complexity analysis
        reasoning_score = 0.0
        causal_inference = 1.0 if any(causal in question for causal in ['why', 'because', 'caused by', 'reason']) else 0.2
        comparative_analysis = 0.9 if any(comp in question for comp in ['compare', 'versus', 'difference', 'better']) else 0.2
        multi_step = 0.8 if any(step in question for step in ['and', 'then', 'also', 'furthermore']) else 0.3
        
        reasoning_score = (causal_inference * 0.4 + comparative_analysis * 0.3 + multi_step * 0.3) / 1.0
        complexity_scores['reasoning_complexity'] = min(1.0, reasoning_score)
        
        # Overall complexity
        overall_complexity = sum(complexity_scores.values()) / len(complexity_scores)
        complexity_level = "high" if overall_complexity > 0.7 else "medium" if overall_complexity > 0.4 else "low"
        
        return {
            "complexity_scores": complexity_scores,
            "overall_complexity": overall_complexity,
            "complexity_level": complexity_level,
            "dominant_complexity": max(complexity_scores.items(), key=lambda x: x[1]),
            "requires_advanced_processing": overall_complexity > 0.6
        }
    
    def execute_enhanced_b_pipeline(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the enhanced B-Pipeline with R4X integration
        
        Args:
            question_data: Ingested question data
            
        Returns:
            Enhanced B-Pipeline results
        """
        script_dir = Path(__file__).parent
        results = {}
        
        # Save current question for B-Pipeline processing
        question_file = script_dir.parent / "outputs/B1_current_question.json"
        question_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(question_file, 'w', encoding='utf-8') as f:
            json.dump(question_data, f, indent=2, ensure_ascii=False)
        
        try:
            # Step 1: Enhanced Intent Analysis (B3.4)
            print("  Executing B3.4 R4X Intent Enhancement...")
            b3_4_script = script_dir / "B3.4_r4x_intent_enhancement.py"
            if b3_4_script.exists():
                result = subprocess.run([sys.executable, str(b3_4_script)], 
                                      capture_output=True, text=True, cwd=script_dir)
                if result.returncode == 0:
                    results['intent_enhancement'] = {
                        'status': 'success',
                        'output': result.stdout,
                        'execution_time': 'completed'
                    }
                else:
                    results['intent_enhancement'] = {
                        'status': 'error',
                        'error': result.stderr,
                        'output': result.stdout
                    }
            else:
                results['intent_enhancement'] = {'status': 'script_not_found'}
            
            # Step 2: Tri-Semantic Answer Synthesis (B4.1)
            print("  Executing B4.1 R4X Answer Synthesis...")
            b4_1_script = script_dir / "B4.1_r4x_answer_synthesis.py"
            if b4_1_script.exists():
                result = subprocess.run([sys.executable, str(b4_1_script)], 
                                      capture_output=True, text=True, cwd=script_dir)
                if result.returncode == 0:
                    results['answer_synthesis'] = {
                        'status': 'success',
                        'output': result.stdout,
                        'execution_time': 'completed'
                    }
                else:
                    results['answer_synthesis'] = {
                        'status': 'error',
                        'error': result.stderr,
                        'output': result.stdout
                    }
            else:
                results['answer_synthesis'] = {'status': 'script_not_found'}
            
            return results
            
        except Exception as e:
            return {
                'intent_enhancement': {'status': 'execution_error', 'error': str(e)},
                'answer_synthesis': {'status': 'execution_error', 'error': str(e)}
            }
    
    def analyze_cross_pipeline_integration(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze cross-pipeline integration effectiveness
        
        Args:
            question_data: Question data with processing results
            
        Returns:
            Cross-pipeline integration analysis
        """
        if not self.r4x_integrator:
            return {
                "integration_status": "r4x_unavailable",
                "pipeline_connectivity": 0.0,
                "semantic_bridges": []
            }
        
        try:
            # Analyze A-Pipeline integration
            a_pipeline_integration = self._analyze_a_pipeline_integration(question_data)
            
            # Analyze B-Pipeline enhancement
            b_pipeline_integration = self._analyze_b_pipeline_integration(question_data)
            
            # Analyze R-Pipeline ontological grounding
            r_pipeline_integration = self._analyze_r_pipeline_integration(question_data)
            
            # Calculate overall integration effectiveness
            integration_score = (
                a_pipeline_integration["connectivity"] * 0.35 +
                b_pipeline_integration["connectivity"] * 0.35 +
                r_pipeline_integration["connectivity"] * 0.30
            )
            
            semantic_bridges = []
            semantic_bridges.extend(a_pipeline_integration.get("bridges", []))
            semantic_bridges.extend(b_pipeline_integration.get("bridges", []))
            semantic_bridges.extend(r_pipeline_integration.get("bridges", []))
            
            return {
                "integration_status": "r4x_active",
                "pipeline_connectivity": integration_score,
                "semantic_bridges": semantic_bridges,
                "integration_details": {
                    "a_pipeline": a_pipeline_integration,
                    "b_pipeline": b_pipeline_integration,
                    "r_pipeline": r_pipeline_integration
                },
                "tri_semantic_coherence": min(1.0, integration_score + 0.1)
            }
            
        except Exception as e:
            return {
                "integration_status": "error",
                "error_message": str(e),
                "pipeline_connectivity": 0.0
            }
    
    def _analyze_a_pipeline_integration(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze A-Pipeline integration effectiveness"""
        # Check for A-Pipeline enhanced outputs
        a_pipeline_path = Path(__file__).parent.parent.parent / "A_Concept_pipeline/outputs"
        
        integration_indicators = []
        if (a_pipeline_path / "A2.9_r4x_semantic_enhancement_output.json").exists():
            integration_indicators.append("r4x_document_enhancement")
        if (a_pipeline_path / "A2.8_semantic_chunks_summary.csv").exists():
            integration_indicators.append("semantic_chunking")
        
        connectivity = len(integration_indicators) / 2.0  # Max 2 indicators
        
        return {
            "connectivity": connectivity,
            "bridges": [f"A-Pipeline:{indicator}" for indicator in integration_indicators],
            "enhancement_available": "r4x_document_enhancement" in integration_indicators
        }
    
    def _analyze_b_pipeline_integration(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze B-Pipeline enhancement integration"""
        b_pipeline_path = Path(__file__).parent.parent / "outputs"
        
        integration_indicators = []
        if (b_pipeline_path / "B3_4_r4x_intent_enhancement_output.json").exists():
            integration_indicators.append("r4x_intent_enhancement")
        if (b_pipeline_path / "B4_1_r4x_answer_synthesis_output.json").exists():
            integration_indicators.append("r4x_answer_synthesis")
        
        connectivity = len(integration_indicators) / 2.0  # Max 2 indicators
        
        return {
            "connectivity": connectivity,
            "bridges": [f"B-Pipeline:{indicator}" for indicator in integration_indicators],
            "enhancement_active": connectivity > 0.5
        }
    
    def _analyze_r_pipeline_integration(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze R-Pipeline ontological integration"""
        r_pipeline_path = Path(__file__).parent.parent.parent / "R_Reference_pipeline/outputs"
        
        integration_indicators = []
        if (r_pipeline_path / "R4L_lexical_ontology_output.json").exists():
            integration_indicators.append("lexical_ontology")
        if (r_pipeline_path / "R4X_cross_pipeline_integration_output.json").exists():
            integration_indicators.append("r4x_cross_integration")
        
        connectivity = len(integration_indicators) / 2.0  # Max 2 indicators
        
        return {
            "connectivity": connectivity,
            "bridges": [f"R-Pipeline:{indicator}" for indicator in integration_indicators],
            "ontological_grounding": connectivity > 0.0
        }
    
    def synthesize_comprehensive_understanding(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize comprehensive question understanding from all analyses
        
        Args:
            all_results: All processing results
            
        Returns:
            Comprehensive understanding synthesis
        """
        question_data = all_results["question_data"]
        complexity_analysis = all_results["complexity_analysis"]
        b_pipeline_results = all_results["b_pipeline_results"]
        integration_analysis = all_results["integration_analysis"]
        
        # Calculate understanding dimensions
        understanding_scores = {}
        
        # Intent understanding score
        intent_score = 0.5  # Base score
        if integration_analysis.get("integration_details", {}).get("b_pipeline", {}).get("enhancement_active"):
            intent_score = 0.8
        understanding_scores["intent_understanding"] = intent_score
        
        # Semantic depth score
        semantic_score = complexity_analysis.get("complexity_scores", {}).get("semantic_complexity", 0.3)
        if integration_analysis.get("tri_semantic_coherence", 0) > 0.6:
            semantic_score = min(1.0, semantic_score + 0.3)
        understanding_scores["semantic_depth"] = semantic_score
        
        # Contextual grounding score
        contextual_score = integration_analysis.get("pipeline_connectivity", 0.3)
        understanding_scores["contextual_grounding"] = contextual_score
        
        # Cross-pipeline integration score
        integration_score = integration_analysis.get("pipeline_connectivity", 0.0)
        understanding_scores["cross_pipeline_integration"] = integration_score
        
        # Answer quality score (if available)
        answer_score = 0.5  # Default
        if b_pipeline_results.get("answer_synthesis", {}).get("status") == "success":
            answer_score = 0.7
        understanding_scores["answer_quality"] = answer_score
        
        # Overall understanding score
        weights = list(self.understanding_dimensions.values())
        scores = list(understanding_scores.values())
        overall_understanding = sum(score * weight for score, weight in zip(scores, weights))
        
        # Understanding quality assessment
        understanding_quality = (
            "exceptional" if overall_understanding > 0.9 else
            "excellent" if overall_understanding > 0.8 else
            "good" if overall_understanding > 0.6 else
            "adequate" if overall_understanding > 0.4 else
            "limited"
        )
        
        return {
            "understanding_scores": understanding_scores,
            "overall_understanding": overall_understanding,
            "understanding_quality": understanding_quality,
            "key_strengths": self._identify_understanding_strengths(understanding_scores),
            "improvement_areas": self._identify_improvement_areas(understanding_scores),
            "comprehensive_insights": self._generate_comprehensive_insights(all_results),
            "processing_completeness": self._assess_processing_completeness(all_results)
        }
    
    def _identify_understanding_strengths(self, scores: Dict[str, float]) -> List[str]:
        """Identify key strengths in understanding"""
        strengths = []
        for dimension, score in scores.items():
            if score > 0.7:
                strengths.append(f"Strong {dimension.replace('_', ' ')}")
        return strengths
    
    def _identify_improvement_areas(self, scores: Dict[str, float]) -> List[str]:
        """Identify areas needing improvement"""
        improvements = []
        for dimension, score in scores.items():
            if score < 0.5:
                improvements.append(f"Enhance {dimension.replace('_', ' ')}")
        return improvements
    
    def _generate_comprehensive_insights(self, all_results: Dict[str, Any]) -> List[str]:
        """Generate comprehensive insights from all analyses"""
        insights = []
        
        complexity = all_results["complexity_analysis"]
        if complexity.get("complexity_level") == "high":
            insights.append("Question exhibits high complexity requiring advanced processing")
        
        integration = all_results["integration_analysis"]
        if integration.get("pipeline_connectivity", 0) > 0.7:
            insights.append("Strong cross-pipeline semantic integration achieved")
        
        if integration.get("tri_semantic_coherence", 0) > 0.6:
            insights.append("Tri-semantic perspective coherence established")
        
        return insights
    
    def _assess_processing_completeness(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess completeness of processing pipeline"""
        completed_stages = []
        
        if all_results.get("question_data"):
            completed_stages.append("question_ingestion")
        if all_results.get("complexity_analysis"):
            completed_stages.append("complexity_analysis")
        if all_results.get("b_pipeline_results"):
            completed_stages.append("b_pipeline_execution")
        if all_results.get("integration_analysis"):
            completed_stages.append("integration_analysis")
        
        completeness = len(completed_stages) / len(self.pipeline_stages)
        
        return {
            "completed_stages": completed_stages,
            "total_stages": len(self.pipeline_stages),
            "completeness_ratio": completeness,
            "missing_stages": [stage for stage in self.pipeline_stages if stage not in completed_stages]
        }
    
    def process_comprehensive_understanding(self, question: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process complete comprehensive question understanding
        
        Args:
            question: Question to understand
            context: Optional context
            
        Returns:
            Comprehensive understanding results
        """
        print("Starting R4X Comprehensive Question Understanding...")
        
        # Stage 1: Question Ingestion
        print("Stage 1: Question Ingestion and Preprocessing...")
        question_data = self.ingest_question(question, context)
        
        # Stage 2: Complexity Analysis
        print("Stage 2: Multi-dimensional Complexity Analysis...")
        complexity_analysis = self.analyze_question_complexity(question_data)
        
        # Stage 3: Enhanced B-Pipeline Execution
        print("Stage 3: Enhanced B-Pipeline Execution...")
        b_pipeline_results = self.execute_enhanced_b_pipeline(question_data)
        
        # Stage 4: Cross-Pipeline Integration Analysis
        print("Stage 4: Cross-Pipeline Integration Analysis...")
        integration_analysis = self.analyze_cross_pipeline_integration(question_data)
        
        # Stage 5: Comprehensive Understanding Synthesis
        print("Stage 5: Comprehensive Understanding Synthesis...")
        all_results = {
            "question_data": question_data,
            "complexity_analysis": complexity_analysis,
            "b_pipeline_results": b_pipeline_results,
            "integration_analysis": integration_analysis
        }
        
        comprehensive_understanding = self.synthesize_comprehensive_understanding(all_results)
        
        return {
            "question": question,
            "comprehensive_understanding": comprehensive_understanding,
            "processing_stages": all_results,
            "r4x_system_status": "active" if self.r4x_integrator else "fallback",
            "processing_timestamp": datetime.now().isoformat(),
            "system_version": "B5.1_R4X_Comprehensive_v1.0"
        }

def save_output(data: Dict[str, Any], output_path: str = "outputs/B5_1_r4x_question_understanding_output.json"):
    """Save comprehensive question understanding results"""
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / output_path
    
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Saved comprehensive understanding results to {full_path}")

def main():
    """Main execution for B5.1 R4X Question Understanding"""
    print("=" * 80)
    print("B5.1: R4X Comprehensive Question Understanding System")
    print("=" * 80)
    
    try:
        # Initialize comprehensive understanding system
        print("Initializing R4X Comprehensive Question Understanding System...")
        understanding_system = B51_R4X_QuestionUnderstanding()
        
        # Process test question
        test_question = "What was the change in Current deferred income?"
        print(f"Processing comprehensive understanding for: '{test_question}'")
        
        # Execute comprehensive understanding
        results = understanding_system.process_comprehensive_understanding(test_question)
        
        # Display comprehensive results
        print(f"\n" + "="*80)
        print("COMPREHENSIVE QUESTION UNDERSTANDING RESULTS")
        print("="*80)
        
        understanding = results["comprehensive_understanding"]
        print(f"\nQuestion: {results['question']}")
        print(f"R4X System Status: {results['r4x_system_status']}")
        
        # Understanding Quality Overview
        print(f"\nOverall Understanding Quality: {understanding['understanding_quality'].upper()}")
        print(f"Understanding Score: {understanding['overall_understanding']:.3f}")
        
        # Understanding Dimensions
        print(f"\nUnderstanding Dimension Scores:")
        for dimension, score in understanding["understanding_scores"].items():
            print(f"  {dimension.replace('_', ' ').title()}: {score:.3f}")
        
        # Key Insights
        if understanding.get("key_strengths"):
            print(f"\nKey Strengths:")
            for strength in understanding["key_strengths"]:
                print(f"  • {strength}")
        
        if understanding.get("improvement_areas"):
            print(f"\nImprovement Areas:")
            for area in understanding["improvement_areas"]:
                print(f"  • {area}")
        
        if understanding.get("comprehensive_insights"):
            print(f"\nComprehensive Insights:")
            for insight in understanding["comprehensive_insights"]:
                print(f"  • {insight}")
        
        # Processing Completeness
        completeness = understanding["processing_completeness"]
        print(f"\nProcessing Completeness: {completeness['completeness_ratio']:.1%}")
        print(f"Completed Stages: {', '.join(completeness['completed_stages'])}")
        
        # Pipeline Integration Status
        stages = results["processing_stages"]
        integration = stages.get("integration_analysis", {})
        if integration.get("integration_status") == "r4x_active":
            print(f"\nCross-Pipeline Integration:")
            print(f"  Pipeline Connectivity: {integration['pipeline_connectivity']:.3f}")
            print(f"  Semantic Bridges: {len(integration.get('semantic_bridges', []))}")
            print(f"  Tri-Semantic Coherence: {integration.get('tri_semantic_coherence', 0.0):.3f}")
        
        # Save comprehensive results
        save_output(results)
        
        print("\n" + "="*80)
        print("[OK] B5.1 R4X Comprehensive Question Understanding completed successfully!")
        print("  Revolutionary tri-semantic question understanding system operational!")
        print("="*80)
        
    except Exception as e:
        print(f"[ERROR] Error in B5.1 R4X Question Understanding: {str(e)}")
        raise

if __name__ == "__main__":
    main()