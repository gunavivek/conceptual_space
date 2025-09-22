"""
WordNet Processor: Linguistic-based expansion using WordNet semantic relations
Implements synonyms, hypernyms, hyponyms, and coordinate terms expansion
"""

import nltk
from collections import defaultdict, Counter
import re

try:
    from nltk.corpus import wordnet as wn
    WORDNET_AVAILABLE = True
except ImportError:
    WORDNET_AVAILABLE = False

# Attempt to download WordNet if not available
if WORDNET_AVAILABLE:
    try:
        wn.synsets('test')
    except LookupError:
        print("  [INFO] Downloading WordNet corpus...")
        try:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        except Exception as e:
            print(f"  [WARNING] Failed to download WordNet: {e}")
            WORDNET_AVAILABLE = False

class WordNetProcessor:
    """
    Advanced WordNet processor for linguistic concept expansion
    """

    def __init__(self, max_depth=2, include_definitions=True):
        """
        Initialize WordNet processor

        Args:
            max_depth: Maximum depth for hypernym/hyponym traversal
            include_definitions: Whether to use definitions for expansion
        """
        self.max_depth = max_depth
        self.include_definitions = include_definitions
        self.synset_cache = {}

        if not WORDNET_AVAILABLE:
            print("  [WARNING] WordNet not available, falling back to basic expansion")

    def _clean_term(self, term):
        """Clean and normalize a term for WordNet lookup"""
        # Remove special characters and normalize
        cleaned = re.sub(r'[^\w\s]', ' ', term.lower())
        cleaned = re.sub(r'\s+', '_', cleaned.strip())
        return cleaned

    def _get_synsets(self, term):
        """Get WordNet synsets for a term with caching"""
        if not WORDNET_AVAILABLE:
            return []

        cleaned_term = self._clean_term(term)

        if cleaned_term in self.synset_cache:
            return self.synset_cache[cleaned_term]

        try:
            # Try different strategies to find synsets
            synsets = []

            # Direct lookup
            synsets.extend(wn.synsets(cleaned_term))

            # Try with underscores replaced by spaces
            if '_' in cleaned_term:
                space_term = cleaned_term.replace('_', ' ')
                synsets.extend(wn.synsets(space_term))

            # Try individual words if compound term
            if '_' in cleaned_term or ' ' in term:
                words = re.split(r'[_\s]+', cleaned_term)
                for word in words:
                    if len(word) > 2:  # Skip very short words
                        synsets.extend(wn.synsets(word))

            # Remove duplicates
            unique_synsets = list(set(synsets))
            self.synset_cache[cleaned_term] = unique_synsets
            return unique_synsets

        except Exception as e:
            print(f"  [WARNING] WordNet lookup failed for '{term}': {e}")
            self.synset_cache[cleaned_term] = []
            return []

    def get_synonyms(self, term, max_synonyms=5):
        """
        Get synonyms for a term

        Args:
            term: Input term
            max_synonyms: Maximum number of synonyms

        Returns:
            list: Synonym terms with metadata
        """
        synsets = self._get_synsets(term)
        synonyms = []

        for synset in synsets:
            try:
                # Get lemma names (synonyms)
                for lemma in synset.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym.lower() != term.lower():
                        synonyms.append({
                            "term": synonym,
                            "synset": synset.name(),
                            "definition": synset.definition(),
                            "relation_type": "synonym",
                            "pos": synset.pos(),
                            "confidence": 0.9  # High confidence for synonyms
                        })
            except Exception:
                continue

        # Remove duplicates and sort by confidence
        seen_terms = set()
        unique_synonyms = []
        for syn in synonyms:
            if syn["term"].lower() not in seen_terms:
                seen_terms.add(syn["term"].lower())
                unique_synonyms.append(syn)

        return unique_synonyms[:max_synonyms]

    def get_hypernyms(self, term, max_depth=None, max_hypernyms=5):
        """
        Get hypernyms (more general concepts) for a term

        Args:
            term: Input term
            max_depth: Maximum traversal depth
            max_hypernyms: Maximum number of hypernyms

        Returns:
            list: Hypernym terms with metadata
        """
        if max_depth is None:
            max_depth = self.max_depth

        synsets = self._get_synsets(term)
        hypernyms = []

        for synset in synsets:
            try:
                # Traverse hypernym hierarchy
                current_synsets = [synset]
                for depth in range(max_depth):
                    next_level = []
                    for current_synset in current_synsets:
                        for hypernym_synset in current_synset.hypernyms():
                            for lemma in hypernym_synset.lemmas():
                                hypernym_term = lemma.name().replace('_', ' ')
                                if hypernym_term.lower() != term.lower():
                                    hypernyms.append({
                                        "term": hypernym_term,
                                        "synset": hypernym_synset.name(),
                                        "definition": hypernym_synset.definition(),
                                        "relation_type": "hypernym",
                                        "depth": depth + 1,
                                        "pos": hypernym_synset.pos(),
                                        "confidence": max(0.1, 0.8 - (depth * 0.2))  # Decrease confidence with depth
                                    })
                            next_level.append(hypernym_synset)
                    current_synsets = next_level
                    if not current_synsets:
                        break
            except Exception:
                continue

        # Remove duplicates and sort by confidence
        seen_terms = set()
        unique_hypernyms = []
        for hyp in sorted(hypernyms, key=lambda x: x["confidence"], reverse=True):
            if hyp["term"].lower() not in seen_terms:
                seen_terms.add(hyp["term"].lower())
                unique_hypernyms.append(hyp)

        return unique_hypernyms[:max_hypernyms]

    def get_hyponyms(self, term, max_depth=None, max_hyponyms=5):
        """
        Get hyponyms (more specific concepts) for a term

        Args:
            term: Input term
            max_depth: Maximum traversal depth
            max_hyponyms: Maximum number of hyponyms

        Returns:
            list: Hyponym terms with metadata
        """
        if max_depth is None:
            max_depth = self.max_depth

        synsets = self._get_synsets(term)
        hyponyms = []

        for synset in synsets:
            try:
                # Traverse hyponym hierarchy
                current_synsets = [synset]
                for depth in range(max_depth):
                    next_level = []
                    for current_synset in current_synsets:
                        for hyponym_synset in current_synset.hyponyms():
                            for lemma in hyponym_synset.lemmas():
                                hyponym_term = lemma.name().replace('_', ' ')
                                if hyponym_term.lower() != term.lower():
                                    hyponyms.append({
                                        "term": hyponym_term,
                                        "synset": hyponym_synset.name(),
                                        "definition": hyponym_synset.definition(),
                                        "relation_type": "hyponym",
                                        "depth": depth + 1,
                                        "pos": hyponym_synset.pos(),
                                        "confidence": max(0.1, 0.8 - (depth * 0.2))
                                    })
                            next_level.append(hyponym_synset)
                    current_synsets = next_level
                    if not current_synsets:
                        break
            except Exception:
                continue

        # Remove duplicates and sort by confidence
        seen_terms = set()
        unique_hyponyms = []
        for hyp in sorted(hyponyms, key=lambda x: x["confidence"], reverse=True):
            if hyp["term"].lower() not in seen_terms:
                seen_terms.add(hyp["term"].lower())
                unique_hyponyms.append(hyp)

        return unique_hyponyms[:max_hyponyms]

    def get_coordinate_terms(self, term, max_coordinates=5):
        """
        Get coordinate terms (same-level concepts sharing a hypernym)

        Args:
            term: Input term
            max_coordinates: Maximum number of coordinate terms

        Returns:
            list: Coordinate terms with metadata
        """
        synsets = self._get_synsets(term)
        coordinates = []

        for synset in synsets:
            try:
                # Get hypernyms and their hyponyms (siblings)
                for hypernym_synset in synset.hypernyms():
                    for sibling_synset in hypernym_synset.hyponyms():
                        if sibling_synset != synset:  # Don't include self
                            for lemma in sibling_synset.lemmas():
                                coordinate_term = lemma.name().replace('_', ' ')
                                if coordinate_term.lower() != term.lower():
                                    coordinates.append({
                                        "term": coordinate_term,
                                        "synset": sibling_synset.name(),
                                        "definition": sibling_synset.definition(),
                                        "relation_type": "coordinate",
                                        "shared_hypernym": hypernym_synset.name(),
                                        "pos": sibling_synset.pos(),
                                        "confidence": 0.7  # Medium confidence for coordinates
                                    })
            except Exception:
                continue

        # Remove duplicates and sort by confidence
        seen_terms = set()
        unique_coordinates = []
        for coord in coordinates:
            if coord["term"].lower() not in seen_terms:
                seen_terms.add(coord["term"].lower())
                unique_coordinates.append(coord)

        return unique_coordinates[:max_coordinates]

    def expand_concept_linguistically(self, concept, max_expansions_per_type=3):
        """
        Expand a concept using all WordNet relations

        Args:
            concept: Target concept
            max_expansions_per_type: Maximum expansions per relation type

        Returns:
            dict: Comprehensive linguistic expansion
        """
        keywords = concept.get("keywords", [])
        all_expansions = {
            "synonyms": [],
            "hypernyms": [],
            "hyponyms": [],
            "coordinates": []
        }

        for keyword in keywords:
            # Get all relation types for this keyword
            synonyms = self.get_synonyms(keyword, max_expansions_per_type)
            hypernyms = self.get_hypernyms(keyword, max_hypernyms=max_expansions_per_type)
            hyponyms = self.get_hyponyms(keyword, max_hyponyms=max_expansions_per_type)
            coordinates = self.get_coordinate_terms(keyword, max_expansions_per_type)

            # Add source keyword info to each expansion
            for syn in synonyms:
                syn["source_keyword"] = keyword
            for hyp in hypernyms:
                hyp["source_keyword"] = keyword
            for hypo in hyponyms:
                hypo["source_keyword"] = keyword
            for coord in coordinates:
                coord["source_keyword"] = keyword

            all_expansions["synonyms"].extend(synonyms)
            all_expansions["hypernyms"].extend(hypernyms)
            all_expansions["hyponyms"].extend(hyponyms)
            all_expansions["coordinates"].extend(coordinates)

        # Remove duplicates across all categories
        for relation_type in all_expansions:
            seen_terms = set()
            unique_expansions = []
            for expansion in all_expansions[relation_type]:
                if expansion["term"].lower() not in seen_terms:
                    seen_terms.add(expansion["term"].lower())
                    unique_expansions.append(expansion)
            all_expansions[relation_type] = unique_expansions

        return all_expansions

    def get_processor_info(self):
        """Get information about the WordNet processor"""
        return {
            "wordnet_available": WORDNET_AVAILABLE,
            "max_depth": self.max_depth,
            "include_definitions": self.include_definitions,
            "cache_size": len(self.synset_cache)
        }