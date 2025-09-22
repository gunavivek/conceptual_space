"""
Domain Ontology Manager: Domain-specific ontology and automatic vocabulary learning
Implements industry standard ontologies and adaptive domain vocabulary expansion
"""

import json
import re
from collections import defaultdict, Counter
from pathlib import Path

class DomainOntologyManager:
    """Domain-specific ontology processor with automatic vocabulary learning"""

    def __init__(self, domain="general", auto_learn=True, min_frequency=2):
        """
        Initialize domain ontology manager

        Args:
            domain: Target domain (general, medical, legal, technical, etc.)
            auto_learn: Enable automatic vocabulary learning
            min_frequency: Minimum frequency for learned terms
        """
        self.domain = domain
        self.auto_learn = auto_learn
        self.min_frequency = min_frequency

        # Domain-specific ontologies
        self.domain_ontologies = self._initialize_domain_ontologies()

        # Learned vocabulary from concept analysis
        self.learned_vocabulary = defaultdict(Counter)
        self.domain_patterns = self._initialize_domain_patterns()

        # Co-occurrence statistics for automatic learning
        self.term_cooccurrence = defaultdict(lambda: defaultdict(int))
        self.term_frequency = Counter()

    def _initialize_domain_ontologies(self):
        """Initialize predefined domain ontologies"""
        ontologies = {
            "general": {
                "hypernyms": {
                    "animal": ["mammal", "bird", "fish", "reptile", "insect"],
                    "vehicle": ["car", "truck", "bicycle", "motorcycle", "boat"],
                    "building": ["house", "office", "factory", "school", "hospital"],
                    "technology": ["computer", "software", "hardware", "network", "database"]
                },
                "synonyms": {
                    "big": ["large", "huge", "massive", "enormous"],
                    "small": ["tiny", "little", "miniature", "compact"],
                    "fast": ["quick", "rapid", "speedy", "swift"],
                    "slow": ["sluggish", "gradual", "leisurely"]
                }
            },
            "technical": {
                "hypernyms": {
                    "algorithm": ["sorting", "searching", "optimization", "machine_learning"],
                    "data_structure": ["array", "list", "tree", "graph", "hash_table"],
                    "programming": ["coding", "development", "debugging", "testing"],
                    "system": ["operating_system", "database", "network", "architecture"]
                },
                "synonyms": {
                    "function": ["method", "procedure", "routine", "subroutine"],
                    "variable": ["parameter", "argument", "field", "attribute"],
                    "error": ["bug", "exception", "fault", "defect"],
                    "optimize": ["improve", "enhance", "streamline", "refine"]
                }
            },
            "business": {
                "hypernyms": {
                    "strategy": ["planning", "analysis", "execution", "evaluation"],
                    "finance": ["accounting", "budgeting", "investment", "revenue"],
                    "management": ["leadership", "organization", "coordination", "supervision"],
                    "marketing": ["advertising", "promotion", "branding", "sales"]
                },
                "synonyms": {
                    "profit": ["revenue", "income", "earnings", "gain"],
                    "cost": ["expense", "expenditure", "investment", "outlay"],
                    "growth": ["expansion", "development", "increase", "progress"],
                    "analysis": ["examination", "evaluation", "assessment", "review"]
                }
            }
        }
        return ontologies

    def _initialize_domain_patterns(self):
        """Initialize domain-specific patterns for automatic learning"""
        patterns = {
            "technical": [
                r"\b\w+_\w+\b",  # snake_case terms
                r"\b[A-Z][a-z]+[A-Z]\w*\b",  # CamelCase terms
                r"\b\w+\(\)\b",  # function calls
                r"\b\w+\.\w+\b"  # dot notation
            ],
            "business": [
                r"\b\w+\s+strategy\b",
                r"\b\w+\s+analysis\b",
                r"\b\w+\s+management\b",
                r"\b\w+\s+process\b"
            ],
            "academic": [
                r"\b\w+\s+theory\b",
                r"\b\w+\s+method\b",
                r"\b\w+\s+approach\b",
                r"\b\w+\s+framework\b"
            ]
        }
        return patterns

    def learn_domain_vocabulary(self, concepts):
        """
        Learn domain-specific vocabulary from concept collection

        Args:
            concepts: List of concepts to learn from

        Returns:
            dict: Learned vocabulary statistics
        """
        learned_terms = Counter()
        domain_indicators = Counter()

        for concept in concepts:
            keywords = concept.get("keywords", [])
            canonical_name = concept.get("canonical_name", "")

            # Analyze all terms
            all_terms = keywords + [canonical_name] if canonical_name else keywords

            for term in all_terms:
                if not term:
                    continue

                # Update frequency
                self.term_frequency[term.lower()] += 1
                learned_terms[term.lower()] += 1

                # Check domain patterns
                for domain, patterns in self.domain_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, term, re.IGNORECASE):
                            domain_indicators[domain] += 1
                            self.learned_vocabulary[domain][term.lower()] += 1

                # Learn co-occurrences
                for other_term in all_terms:
                    if other_term != term and other_term:
                        self.term_cooccurrence[term.lower()][other_term.lower()] += 1

        # Determine likely domain
        likely_domain = domain_indicators.most_common(1)[0][0] if domain_indicators else "general"

        return {
            "learned_terms_count": len(learned_terms),
            "total_term_frequency": sum(learned_terms.values()),
            "likely_domain": likely_domain,
            "domain_indicators": dict(domain_indicators),
            "most_frequent_terms": learned_terms.most_common(10)
        }

    def get_domain_expansions(self, term, max_expansions=5):
        """
        Get domain-specific expansions for a term

        Args:
            term: Input term
            max_expansions: Maximum number of expansions

        Returns:
            list: Domain-specific expansion terms
        """
        expansions = []
        term_lower = term.lower()

        # Check predefined ontologies
        current_ontology = self.domain_ontologies.get(self.domain, {})

        # Look for hypernyms
        for category, terms in current_ontology.get("hypernyms", {}).items():
            if term_lower in [t.lower() for t in terms]:
                expansions.append({
                    "term": category,
                    "relation_type": "domain_hypernym",
                    "confidence": 0.9,
                    "source": "predefined_ontology"
                })
            elif term_lower == category.lower():
                for subterm in terms[:max_expansions]:
                    expansions.append({
                        "term": subterm,
                        "relation_type": "domain_hyponym",
                        "confidence": 0.8,
                        "source": "predefined_ontology"
                    })

        # Look for synonyms
        for base_term, synonyms in current_ontology.get("synonyms", {}).items():
            if term_lower == base_term.lower():
                for synonym in synonyms[:max_expansions]:
                    expansions.append({
                        "term": synonym,
                        "relation_type": "domain_synonym",
                        "confidence": 0.9,
                        "source": "predefined_ontology"
                    })
            elif term_lower in [s.lower() for s in synonyms]:
                expansions.append({
                    "term": base_term,
                    "relation_type": "domain_synonym",
                    "confidence": 0.9,
                    "source": "predefined_ontology"
                })

        # Check learned vocabulary
        if self.auto_learn and term_lower in self.learned_vocabulary.get(self.domain, {}):
            # Find co-occurring terms
            cooccurring_terms = self.term_cooccurrence.get(term_lower, {})
            for coterm, frequency in sorted(cooccurring_terms.items(),
                                          key=lambda x: x[1], reverse=True)[:max_expansions]:
                if frequency >= self.min_frequency:
                    expansions.append({
                        "term": coterm,
                        "relation_type": "learned_cooccurrence",
                        "confidence": min(0.8, frequency / 10),  # Scale confidence by frequency
                        "source": "learned_vocabulary",
                        "cooccurrence_frequency": frequency
                    })

        return expansions[:max_expansions]

    def expand_concept_with_domain_ontology(self, concept, max_expansions=5):
        """
        Expand concept using domain ontology and learned vocabulary

        Args:
            concept: Target concept
            max_expansions: Maximum expansion terms

        Returns:
            dict: Expanded concept with domain-specific terms
        """
        keywords = concept.get("keywords", [])
        all_expansions = []

        for keyword in keywords:
            domain_expansions = self.get_domain_expansions(keyword, max_expansions)
            for expansion in domain_expansions:
                expansion["source_keyword"] = keyword
            all_expansions.extend(domain_expansions)

        # Remove duplicates
        seen_terms = set()
        unique_expansions = []
        for expansion in all_expansions:
            if expansion["term"] not in seen_terms:
                seen_terms.add(expansion["term"])
                unique_expansions.append(expansion)

        return unique_expansions[:max_expansions]

    def get_ontology_info(self):
        """Get information about the domain ontology manager"""
        return {
            "domain": self.domain,
            "auto_learn": self.auto_learn,
            "min_frequency": self.min_frequency,
            "predefined_ontology_size": len(self.domain_ontologies.get(self.domain, {})),
            "learned_vocabulary_size": sum(len(vocab) for vocab in self.learned_vocabulary.values()),
            "term_frequency_cache": len(self.term_frequency),
            "cooccurrence_cache": len(self.term_cooccurrence)
        }