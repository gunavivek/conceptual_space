#!/usr/bin/env python3
"""
A3.3: Document Structure Chunking Strategy
Structure-aware chunking that respects document hierarchy
"""

from typing import Dict, List, Any, Tuple
import re
from .base_strategy import BaseChunkingStrategy, ConceptChunk

class DocumentStructureStrategy(BaseChunkingStrategy):
    """
    Implements chunking based on document structure markers like
    headings, sections, lists, and other structural elements
    """
    
    def __init__(self, respect_sections: bool = True, min_section_length: int = 50):
        super().__init__("document_structure")
        self.respect_sections = respect_sections
        self.min_section_length = min_section_length
        
    def get_strategy_config(self) -> Dict[str, Any]:
        """Return configuration for this strategy"""
        return {
            'respect_sections': self.respect_sections,
            'min_section_length': self.min_section_length,
            'description': 'Document structure-aware chunking respecting hierarchy'
        }
    
    def detect_structure_markers(self, text: str) -> List[Tuple[str, int, int, str]]:
        """
        Detect structural markers in text
        
        Returns:
            List of (content, start_idx, end_idx, structure_type) tuples
        """
        structures = []
        
        # Patterns for different structural elements
        patterns = {
            'heading': r'^#+\s+.*$',  # Markdown headings
            'numbered_section': r'^\d+\.[\d.]*\s+.*$',  # Numbered sections
            'bullet_list': r'^[\*\-•]\s+.*$',  # Bullet points
            'definition': r'^.+:\s+.+$',  # Definition-style content
            'table_row': r'^.*\|.*\|.*$',  # Table rows
        }
        
        lines = text.split('\n')
        current_pos = 0
        current_section = []
        current_type = None
        section_start = 0
        
        for line in lines:
            line_start = current_pos
            line_end = current_pos + len(line)
            
            # Check if line matches any structural pattern
            matched_type = None
            for struct_type, pattern in patterns.items():
                if re.match(pattern, line.strip()):
                    matched_type = struct_type
                    break
            
            # Handle section transitions
            if matched_type and matched_type != current_type:
                # Save previous section if exists
                if current_section and len(''.join(current_section)) >= self.min_section_length:
                    section_content = '\n'.join(current_section)
                    structures.append((section_content, section_start, current_pos - 1, current_type or 'paragraph'))
                
                # Start new section
                current_section = [line]
                current_type = matched_type
                section_start = line_start
            else:
                current_section.append(line)
            
            current_pos = line_end + 1  # +1 for newline
        
        # Save final section
        if current_section and len(''.join(current_section)) >= self.min_section_length:
            section_content = '\n'.join(current_section)
            structures.append((section_content, section_start, current_pos, current_type or 'paragraph'))
        
        return structures
    
    def chunk_document(self, 
                      document: Dict[str, Any], 
                      concepts: Dict[str, Any],
                      **kwargs) -> List[ConceptChunk]:
        """
        Create structure-aware chunks from document
        
        Args:
            document: Document with 'doc_id' and 'content'
            concepts: Core and expanded concepts
            **kwargs: Additional parameters
            
        Returns:
            List of ConceptChunk objects
        """
        chunks = []
        doc_id = document.get('doc_id', 'unknown')
        content = document.get('content', '')
        
        # Detect document structures
        structures = self.detect_structure_markers(content)
        
        # If no structures detected, fall back to paragraph chunking
        if not structures:
            structures = [(para, start, end, 'paragraph') 
                          for para, start, end in self.split_paragraphs(content)]
        
        chunk_index = 0
        for section_text, start_idx, end_idx, structure_type in structures:
            # Skip very short sections
            if len(section_text) < self.min_section_length:
                continue
            
            # Extract concept memberships
            # DOCUMENT-AWARE: Only match concepts from same document
            memberships, scores = self.extract_concept_memberships(section_text, concepts, doc_id=doc_id)
            
            # Create chunk if it has concept memberships
            if memberships:
                metadata = {
                    'structure_type': structure_type,
                    'section_length': len(section_text),
                    'word_count': len(section_text.split()),
                    'has_heading': structure_type in ['heading', 'numbered_section'],
                    'is_list': structure_type in ['bullet_list', 'numbered_list'],
                    'concept_alignment': max(scores.values()) if scores else 0
                }
                
                chunk = self.create_chunk_detailed(
                    doc_id=doc_id,
                    content=section_text,
                    chunk_index=chunk_index,
                    start_index=start_idx,
                    end_index=end_idx,
                    concepts=concepts,
                    metadata=metadata
                )
                
                chunks.append(chunk)
                chunk_index += 1
        
        return chunks