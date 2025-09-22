"""
Q2.3: Question Structure Analysis
Analyzes the linguistic and syntactic structure of questions
"""

import json
import os
import re
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional


class SyntacticFeatureExtractor:
    """
    Extracts syntactic features from question text
    """

    def __init__(self):
        self.question_patterns = {
            'wh_questions': ['what', 'when', 'where', 'who', 'why', 'how', 'which'],
            'auxiliary_verbs': ['is', 'are', 'was', 'were', 'do', 'does', 'did', 'will', 'would', 'can', 'could'],
            'modal_verbs': ['can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would']
        }

    def extract_syntactic_features(self, question_text: str) -> Dict:
        """
        Extract syntactic features from question
        """
        tokens = question_text.lower().split()

        # Identify question type
        question_type = self._identify_question_type(tokens)

        # Extract WH-word if present
        wh_word = self._extract_wh_word(tokens)

        # Find auxiliary and main verbs
        auxiliary = self._find_auxiliary_verb(tokens)
        main_verb = self._find_main_verb(tokens)

        # Determine voice and tense
        voice = self._determine_voice(tokens)
        tense = self._determine_tense(tokens)

        # Extract modality
        modality = self._extract_modals(tokens)

        return {
            'question_type': question_type,
            'wh_word': wh_word,
            'auxiliary_verb': auxiliary,
            'main_verb': main_verb,
            'voice': voice,
            'tense': tense,
            'modality': modality
        }

    def _identify_question_type(self, tokens: List[str]) -> str:
        """Identify the type of question"""
        first_token = tokens[0] if tokens else ''

        if first_token in self.question_patterns['wh_questions']:
            return 'wh_question'
        elif first_token in self.question_patterns['auxiliary_verbs']:
            return 'yes_no'
        else:
            return 'declarative_question'

    def _extract_wh_word(self, tokens: List[str]) -> str:
        """Extract WH-word if present"""
        for token in tokens:
            if token in self.question_patterns['wh_questions']:
                return token
        return None

    def _find_auxiliary_verb(self, tokens: List[str]) -> str:
        """Find auxiliary verb in question"""
        for token in tokens:
            if token in self.question_patterns['auxiliary_verbs']:
                return token
        return None

    def _find_main_verb(self, tokens: List[str]) -> str:
        """Find main verb (simplified)"""
        # Look for common verbs
        action_verbs = ['change', 'increase', 'decrease', 'calculate', 'find', 'determine']
        for token in tokens:
            if token in action_verbs or token.endswith('ing') or token.endswith('ed'):
                return token
        return None

    def _determine_voice(self, tokens: List[str]) -> str:
        """Determine active or passive voice"""
        # Simplified check for passive voice
        if 'by' in tokens or ('was' in tokens and any(t.endswith('ed') for t in tokens)):
            return 'passive'
        return 'active'

    def _determine_tense(self, tokens: List[str]) -> str:
        """Determine tense of question"""
        if any(word in tokens for word in ['was', 'were', 'did']):
            return 'past'
        elif any(word in tokens for word in ['will', 'shall']):
            return 'future'
        else:
            return 'present'

    def _extract_modals(self, tokens: List[str]) -> List[str]:
        """Extract modal verbs"""
        modals = []
        for token in tokens:
            if token in self.question_patterns['modal_verbs']:
                modals.append(token)
        return modals


class ComplexityAnalyzer:
    """
    Analyzes question linguistic complexity
    """

    def analyze_complexity(self, question_text: str) -> Dict:
        """
        Calculate various complexity metrics
        """
        tokens = question_text.split()

        # Basic counts
        token_count = len(tokens)

        # Clause analysis
        clause_count = self._count_clauses(question_text)

        # Modifier analysis
        modifier_count = self._count_modifiers(tokens)

        # Prepositional phrases
        pp_count = self._count_prepositional_phrases(question_text)

        # Estimate dependency depth
        dependency_depth = self._estimate_dependency_depth(question_text)

        # Calculate overall complexity score
        complexity_score = self._calculate_complexity_score(
            token_count, clause_count, modifier_count, pp_count
        )

        return {
            'token_count': token_count,
            'dependency_depth': dependency_depth,
            'clause_count': clause_count,
            'modifier_count': modifier_count,
            'prepositional_phrases': pp_count,
            'syntactic_complexity_score': complexity_score
        }

    def _count_clauses(self, text: str) -> int:
        """Count number of clauses in question"""
        # Simplified clause counting based on conjunctions and punctuation
        clause_markers = ['and', 'or', 'but', 'that', 'which', 'when', 'where', 'if']
        count = 1  # At least one main clause

        text_lower = text.lower()
        for marker in clause_markers:
            count += text_lower.count(f' {marker} ')

        return count

    def _count_modifiers(self, tokens: List[str]) -> int:
        """Count adjectives and adverbs (simplified)"""
        modifier_endings = ['ly', 'er', 'est', 'ive', 'ous', 'ful']
        count = 0

        for token in tokens:
            if any(token.endswith(ending) for ending in modifier_endings):
                count += 1

        return count

    def _count_prepositional_phrases(self, text: str) -> int:
        """Count prepositional phrases"""
        prepositions = ['in', 'on', 'at', 'from', 'to', 'with', 'by', 'for', 'of', 'between']
        count = 0

        text_lower = text.lower()
        for prep in prepositions:
            count += text_lower.count(f' {prep} ')

        return count

    def _estimate_dependency_depth(self, text: str) -> int:
        """Estimate dependency tree depth (simplified)"""
        # Simplified estimation based on question length and complexity
        tokens = text.split()
        base_depth = min(3, len(tokens) // 5)

        # Add depth for complex structures
        if 'which' in text.lower() or 'that' in text.lower():
            base_depth += 1

        return base_depth

    def _calculate_complexity_score(self, tokens: int, clauses: int,
                                   modifiers: int, pps: int) -> float:
        """
        Calculate normalized complexity score
        """
        # Weighted combination of complexity factors
        base_score = min(1.0, tokens / 20.0) * 0.2  # Token length factor
        clause_score = min(1.0, clauses / 3.0) * 0.3  # Clause complexity
        modifier_score = min(1.0, modifiers / 5.0) * 0.25  # Modification complexity
        pp_score = min(1.0, pps / 3.0) * 0.25  # Prepositional phrase complexity

        total_score = base_score + clause_score + modifier_score + pp_score
        return min(1.0, total_score)


class DependencyStructureAnalyzer:
    """
    Analyzes grammatical dependency structure
    """

    def analyze_dependencies(self, question_text: str) -> Dict:
        """
        Analyze dependency structure of question
        """
        tokens = question_text.split()

        # Find grammatical components
        subject = self._identify_subject(question_text)
        obj = self._identify_object(question_text)
        root = self._find_root_verb(question_text)

        # Build dependency relations (simplified)
        relations = self._extract_dependency_relations(question_text)

        # Calculate parse tree depth
        parse_depth = self._calculate_parse_depth(relations)

        return {
            'root_token': root,
            'subject': subject,
            'object': obj,
            'dependency_relations': relations,
            'parse_tree_depth': parse_depth
        }

    def _identify_subject(self, text: str) -> str:
        """
        Identify grammatical subject
        """
        # Common subjects in financial questions
        common_subjects = ['percentage', 'change', 'revenue', 'income', 'profit', 'amount', 'value', 'total', 'rate']
        text_lower = text.lower()

        for subj in common_subjects:
            if subj in text_lower:
                return subj

        # If no common subject found, look for first noun-like word
        tokens = text_lower.split()
        for token in tokens:
            if len(token) > 3 and token not in ['what', 'when', 'where', 'which', 'from']:
                return token

        return "implicit_subject"

    def _identify_object(self, text: str) -> str:
        """
        Identify grammatical object
        """
        # Look for object patterns (simplified)
        text_lower = text.lower()

        # Common pattern: "X from/to Y"
        from_match = re.search(r'from\s+(\w+)', text_lower)
        if from_match:
            return from_match.group(1)

        to_match = re.search(r'to\s+(\w+)', text_lower)
        if to_match:
            return to_match.group(1)

        # Look for years as objects
        year_match = re.search(r'\b(19|20)\d{2}\b', text)
        if year_match:
            return year_match.group()

        return "implicit_object"

    def _find_root_verb(self, text: str) -> str:
        """
        Find the root verb of the question
        """
        # Common root verbs in questions
        verbs = ['is', 'was', 'are', 'were', 'change', 'calculate', 'find', 'determine']
        text_lower = text.lower()

        for verb in verbs:
            if verb in text_lower:
                return verb

        return "be"  # Default copula

    def _extract_dependency_relations(self, text: str) -> List[Dict]:
        """
        Extract simplified dependency relations
        """
        relations = []
        tokens = text.lower().split()

        # Find simple dependencies
        for i, token in enumerate(tokens):
            if i > 0:
                # Create simplified dependency
                relations.append({
                    'governor': tokens[i-1],
                    'dependent': token,
                    'relation': self._classify_relation(tokens[i-1], token)
                })

        return relations

    def _classify_relation(self, gov: str, dep: str) -> str:
        """
        Classify dependency relation type (simplified)
        """
        if gov in ['what', 'when', 'where', 'which']:
            return 'interrogative'
        elif gov in ['is', 'was', 'are', 'were']:
            return 'copula'
        elif gov in ['from', 'to', 'in', 'on', 'at']:
            return 'prepositional'
        else:
            return 'modifier'

    def _calculate_parse_depth(self, relations: List[Dict]) -> int:
        """
        Calculate parse tree depth
        """
        if not relations:
            return 1

        # Simplified depth calculation
        return min(5, 1 + len(relations) // 3)


class LinguisticPatternExtractor:
    """
    Extracts linguistic patterns and phrases
    """

    def extract_patterns(self, question_text: str) -> Dict:
        """
        Extract linguistic patterns from question
        """
        # Extract POS sequence (simplified)
        pos_sequence = self._get_pos_sequence(question_text)

        # Extract phrases
        noun_phrases = self._extract_noun_phrases(question_text)
        verb_phrases = self._extract_verb_phrases(question_text)

        # Named entity recognition (simplified)
        named_entities = self._extract_named_entities(question_text)

        # Grammatical functions
        grammatical_functions = self._map_grammatical_functions(question_text)

        return {
            'pos_sequence': pos_sequence,
            'noun_phrases': noun_phrases,
            'verb_phrases': verb_phrases,
            'named_entities': named_entities,
            'grammatical_functions': grammatical_functions
        }

    def _get_pos_sequence(self, text: str) -> List[str]:
        """
        Get simplified POS tag sequence
        """
        pos_tags = []
        tokens = text.split()

        for token in tokens:
            if token.lower() in ['what', 'when', 'where', 'who', 'why', 'how']:
                pos_tags.append('WH')
            elif token.lower() in ['is', 'are', 'was', 'were']:
                pos_tags.append('VB')
            elif token.lower() in ['the', 'a', 'an']:
                pos_tags.append('DT')
            elif any(char.isdigit() for char in token):
                pos_tags.append('CD')
            elif token.lower() in ['from', 'to', 'in', 'on', 'at']:
                pos_tags.append('IN')
            else:
                pos_tags.append('NN')

        return pos_tags

    def _extract_noun_phrases(self, text: str) -> List[str]:
        """
        Extract noun phrases from question
        """
        noun_phrases = []

        # Pattern: determiner + adjective + noun
        patterns = [
            r'the \w+ \w+',
            r'a \w+ \w+',
            r'\w+ revenue',
            r'\w+ change',
            r'percentage \w+',
        ]

        text_lower = text.lower()
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            noun_phrases.extend(matches)

        # Also extract year ranges
        year_pattern = r'from \d{4} to \d{4}'
        year_matches = re.findall(year_pattern, text_lower)
        noun_phrases.extend(year_matches)

        return list(set(noun_phrases))  # Remove duplicates

    def _extract_verb_phrases(self, text: str) -> List[str]:
        """
        Extract verb phrases from question
        """
        verb_phrases = []

        # Common verb phrase patterns
        patterns = [
            r'is the \w+',
            r'was the \w+',
            r'change in',
            r'increase in',
            r'decrease in',
        ]

        text_lower = text.lower()
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            verb_phrases.extend(matches)

        return list(set(verb_phrases))

    def _extract_named_entities(self, text: str) -> List[Dict]:
        """
        Extract named entities (simplified)
        """
        entities = []

        # Extract years
        year_pattern = r'\b(19|20)\d{2}\b'
        for match in re.finditer(year_pattern, text):
            entities.append({
                'text': match.group(),
                'type': 'DATE',
                'start': match.start(),
                'end': match.end()
            })

        # Extract financial terms as entities
        financial_terms = ['revenue', 'income', 'profit', 'expense']
        text_lower = text.lower()
        for term in financial_terms:
            if term in text_lower:
                start = text_lower.index(term)
                entities.append({
                    'text': term,
                    'type': 'FINANCIAL',
                    'start': start,
                    'end': start + len(term)
                })

        return entities

    def _map_grammatical_functions(self, text: str) -> Dict:
        """
        Map grammatical functions in question
        """
        functions = {}

        # Identify subject function
        if 'percentage' in text.lower():
            functions['subject'] = 'percentage'
        elif 'change' in text.lower():
            functions['subject'] = 'change'

        # Identify object function
        if 'revenue' in text.lower():
            functions['object'] = 'revenue'

        # Identify complement
        year_matches = re.findall(r'\b(19|20)\d{2}\b', text)
        if year_matches:
            functions['temporal_complement'] = year_matches

        return functions


class AmbiguityAssessor:
    """
    Assesses various types of ambiguity in questions
    """

    def assess_ambiguity(self, question_text: str, structural_features: Dict) -> Dict:
        """
        Assess different types of ambiguity
        """
        # Structural ambiguity
        structural_ambiguity = self._assess_structural_ambiguity(question_text)

        # Referential ambiguity
        referential_ambiguity = self._assess_referential_ambiguity(question_text)

        # Scope ambiguity
        scope_ambiguity = self._assess_scope_ambiguity(question_text)

        # Overall clarity score (inverse of ambiguity)
        clarity_score = 1.0 - (structural_ambiguity + referential_ambiguity + scope_ambiguity) / 3.0

        return {
            'structural_ambiguity': structural_ambiguity,
            'referential_ambiguity': referential_ambiguity,
            'scope_ambiguity': scope_ambiguity,
            'overall_clarity_score': max(0.0, clarity_score)
        }

    def _assess_structural_ambiguity(self, text: str) -> float:
        """
        Assess structural ambiguity level
        """
        # Check for ambiguous constructions
        ambiguous_patterns = [
            'and or',  # Coordination ambiguity
            'more than',  # Comparison ambiguity
            'between',  # Scope ambiguity
            'including',  # Scope ambiguity
        ]

        ambiguity_score = 0.0
        text_lower = text.lower()

        for pattern in ambiguous_patterns:
            if pattern in text_lower:
                ambiguity_score += 0.2

        return min(1.0, ambiguity_score)

    def _assess_referential_ambiguity(self, text: str) -> float:
        """
        Assess referential ambiguity (unclear references)
        """
        # Check for potentially ambiguous pronouns
        pronouns = ['it', 'they', 'them', 'this', 'that', 'these', 'those']
        text_lower = text.lower()

        ambiguity_score = 0.0
        for pronoun in pronouns:
            if f' {pronoun} ' in text_lower:
                ambiguity_score += 0.15

        return min(1.0, ambiguity_score)

    def _assess_scope_ambiguity(self, text: str) -> float:
        """
        Assess scope ambiguity
        """
        # Check for quantifier/negation scope issues
        scope_words = ['all', 'every', 'each', 'not', 'only', 'just']
        text_lower = text.lower()

        ambiguity_score = 0.0
        for word in scope_words:
            if word in text_lower:
                ambiguity_score += 0.1

        return min(1.0, ambiguity_score)


class Q2_3_QuestionStructureAnalysis:
    """
    Main Q2.3 Question Structure Analysis processor
    """

    def __init__(self):
        self.syntactic_extractor = SyntacticFeatureExtractor()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.dependency_analyzer = DependencyStructureAnalyzer()
        self.pattern_extractor = LinguisticPatternExtractor()
        self.ambiguity_assessor = AmbiguityAssessor()

    def analyze_question_structure(self, question_id: str) -> Dict:
        """
        Main processing function for question structure analysis
        """
        start_time = datetime.now()

        try:
            # Load question data from Q1
            question_data = self._load_question_from_q1(question_id)
            question_text = question_data['question_text']
            doc_id = question_data['doc_id']

            # Extract syntactic features
            syntactic_features = self.syntactic_extractor.extract_syntactic_features(question_text)

            # Analyze complexity
            complexity_metrics = self.complexity_analyzer.analyze_complexity(question_text)

            # Analyze dependencies
            dependency_structure = self.dependency_analyzer.analyze_dependencies(question_text)

            # Extract linguistic patterns
            linguistic_patterns = self.pattern_extractor.extract_patterns(question_text)

            # Assess ambiguity
            ambiguity_assessment = self.ambiguity_assessor.assess_ambiguity(
                question_text, syntactic_features
            )

            # Generate structural feature vector
            structural_features_vector = self._generate_feature_vector(
                syntactic_features, complexity_metrics, ambiguity_assessment
            )

            # Calculate processing metadata
            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            result = {
                'question_id': question_id,
                'doc_id': doc_id,
                'question_text': question_text,
                'structural_analysis': {
                    'syntactic_features': syntactic_features,
                    'complexity_metrics': complexity_metrics,
                    'dependency_structure': dependency_structure,
                    'linguistic_patterns': linguistic_patterns
                },
                'structural_features_vector': structural_features_vector,
                'ambiguity_assessment': ambiguity_assessment,
                'processing_metadata': {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'processing_time_ms': processing_time,
                    'parser_confidence': self._calculate_parser_confidence(syntactic_features),
                    'feature_extraction_status': 'complete'
                }
            }

            return result

        except Exception as e:
            print(f"Error in Q2.3 processing: {e}")
            return self._get_default_output(question_id)

    def _load_question_from_q1(self, question_id: str) -> Dict:
        """Load question data from Q1 output - NO ANSWER DATA"""
        try:
            q1_path = "../outputs/Q1_Question_ingestion.json"
            with open(q1_path, 'r') as f:
                q1_data = json.load(f)

            # Handle both single and multi-question formats
            if isinstance(q1_data, dict):
                if 'question_id' in q1_data and q1_data['question_id'] == question_id:
                    raw_data = q1_data
                elif question_id in q1_data:
                    raw_data = q1_data[question_id]
                else:
                    raw_data = q1_data

            # Extract only safe fields
            return {
                'question_id': raw_data.get('question_id', question_id),
                'doc_id': raw_data.get('doc_id', question_id),
                'question_text': raw_data.get('question_text', '')
            }

        except Exception as e:
            print(f"Error loading Q1 data: {e}")
            return {
                'question_id': question_id,
                'doc_id': question_id,
                'question_text': 'What is the percentage change in the revenue from 2018 to 2019?'
            }

    def _generate_feature_vector(self, syntactic: Dict, complexity: Dict, ambiguity: Dict) -> List[float]:
        """
        Generate numerical feature vector from structural analysis
        """
        features = []

        # Syntactic features (one-hot encoding of question type)
        features.append(1.0 if syntactic['question_type'] == 'wh_question' else 0.0)
        features.append(1.0 if syntactic['voice'] == 'active' else 0.0)
        features.append(1.0 if syntactic['tense'] == 'present' else 0.0)

        # Complexity features
        features.append(complexity['syntactic_complexity_score'])
        features.append(min(1.0, complexity['token_count'] / 30.0))  # Normalized token count
        features.append(min(1.0, complexity['dependency_depth'] / 5.0))  # Normalized depth

        # Ambiguity features
        features.append(ambiguity['overall_clarity_score'])
        features.append(1.0 - ambiguity['structural_ambiguity'])
        features.append(1.0 - ambiguity['referential_ambiguity'])
        features.append(1.0 - ambiguity['scope_ambiguity'])

        return features

    def _calculate_parser_confidence(self, syntactic_features: Dict) -> float:
        """
        Calculate confidence in parsing results
        """
        confidence = 0.5  # Base confidence

        # Boost for recognized question type
        if syntactic_features['question_type'] == 'wh_question':
            confidence += 0.2

        # Boost for identified WH-word
        if syntactic_features['wh_word']:
            confidence += 0.15

        # Boost for identified verbs
        if syntactic_features['main_verb']:
            confidence += 0.15

        return min(1.0, confidence)

    def _get_default_output(self, question_id: str) -> Dict:
        """Return default output on error"""
        return {
            'question_id': question_id,
            'doc_id': question_id,
            'question_text': 'Error in processing',
            'structural_analysis': {
                'syntactic_features': {},
                'complexity_metrics': {},
                'dependency_structure': {},
                'linguistic_patterns': {}
            },
            'structural_features_vector': [0.0] * 10,
            'ambiguity_assessment': {
                'structural_ambiguity': 0.5,
                'referential_ambiguity': 0.5,
                'scope_ambiguity': 0.5,
                'overall_clarity_score': 0.5
            },
            'processing_metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'processing_time_ms': 0.0,
                'parser_confidence': 0.0,
                'feature_extraction_status': 'error'
            }
        }

    def save_output(self, result: Dict, output_path: str = "../outputs/Q2.3_question_structure_analysis.json"):
        """Save Q2.3 output to file with specified name"""
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
    """Process all questions from Q1 output"""
    print("=" * 70)
    print("Q2.3: Question Structure Analysis - Processing All Questions from Q1")
    print("=" * 70)

    # Initialize Q2.3
    q23 = Q2_3_QuestionStructureAnalysis()

    try:
        # Load all questions from Q1 output
        q1_path = "../outputs/Q1_Question_ingestion.json"
        with open(q1_path, 'r') as f:
            q1_data = json.load(f)

        questions = q1_data.get('questions', [])
        print(f"Found {len(questions)} questions from Q1")

        all_results = {}
        successful = 0
        failed = 0

        for i, question_data in enumerate(questions, 1):
            question_id = question_data.get('question_id', f'q_{i}')
            question_text = question_data.get('question_text', '')

            print(f"\n[{i}/{len(questions)}] Processing: {question_id}")
            print(f"Question: {question_text[:80]}...")

            try:
                # Modify the load function temporarily to use the question data directly
                original_load = q23._load_question_from_q1
                q23._load_question_from_q1 = lambda qid: {
                    'question_id': question_data.get('question_id', qid),
                    'doc_id': question_data.get('doc_id', qid),
                    'question_text': question_data.get('question_text', '')
                }

                # Run structure analysis
                result = q23.analyze_question_structure(question_id)
                all_results[question_id] = result

                # Show brief summary
                syntactic = result['structural_analysis']['syntactic_features']
                complexity = result['structural_analysis']['complexity_metrics']

                print(f"  -> Type: {syntactic['question_type']}, Tokens: {complexity['token_count']}, Complexity: {complexity['syntactic_complexity_score']:.2f}")
                successful += 1

                # Restore original function
                q23._load_question_from_q1 = original_load

            except Exception as e:
                print(f"  -> ERROR: {e}")
                failed += 1
                # Restore original function
                q23._load_question_from_q1 = original_load

        # Save all results
        output_path = "../outputs/Q2.3_question_structure_analysis.json"
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\n" + "=" * 70)
        print("Q2.3 BATCH PROCESSING COMPLETE")
        print("=" * 70)
        print(f"Total questions: {len(questions)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Success rate: {successful/len(questions)*100:.1f}%")
        print(f"Results saved to: {output_path}")

        # Show summary statistics
        if all_results:
            total_tokens = sum(r['structural_analysis']['complexity_metrics']['token_count'] for r in all_results.values())
            avg_complexity = sum(r['structural_analysis']['complexity_metrics']['syntactic_complexity_score'] for r in all_results.values()) / len(all_results)

            # Count question types
            question_types = {}
            for result in all_results.values():
                qtype = result['structural_analysis']['syntactic_features']['question_type']
                question_types[qtype] = question_types.get(qtype, 0) + 1

            print(f"\nStructural Analysis Summary:")
            print(f"  Total tokens analyzed: {total_tokens}")
            print(f"  Average complexity: {avg_complexity:.3f}")
            print(f"  Average tokens per question: {total_tokens/len(all_results):.1f}")

            print(f"\nQuestion Type Distribution:")
            for qtype, count in sorted(question_types.items()):
                print(f"  {qtype}: {count} questions")

        return all_results

    except Exception as e:
        print(f"Error in Q2.3 batch processing: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("Q2.3 QUESTION STRUCTURE ANALYSIS")
    print("=" * 50)

    result = main()

    if result:
        print("Q2.3_question_structure_analysis.json created successfully")
        print("Question structure analysis complete - ready for Q2.5 integration")
    else:
        print("Failed to create Q2.3 output")