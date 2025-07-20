#!/usr/bin/env python3
"""
Improved PDF Outline Extractor - More accurate heading detection
"""

import fitz
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Set
import logging
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TextBlock:
    """Represents any text block from the PDF"""
    text: str
    page: int
    bbox: Tuple[float, float, float, float]
    font_size: float
    font_name: str
    is_bold: bool
    is_italic: bool
    flags: int
    spans: List[Dict] = field(default_factory=list)
    
    @property
    def y_position(self) -> float:
        return self.bbox[1]
    
    @property
    def x_position(self) -> float:
        return self.bbox[0]


class ImprovedOutlineExtractor:
    """Enhanced outline extraction with better heading detection"""
    
    def __init__(self):
        # Enhanced heading patterns
        self.numbered_patterns = [
            (r'^\d+\.\s+', 'numbered_h1'),  # 1. 
            (r'^\d+\.\d+\s+', 'numbered_h2'),  # 1.1
            (r'^\d+\.\d+\.\d+\s+', 'numbered_h3'),  # 1.1.1
        ]
        
        # Keywords that often indicate headings
        self.heading_keywords = {
            'h1': ['appendix', 'chapter', 'part', 'section'],
            'h2': ['overview', 'introduction', 'conclusion', 'summary', 'background', 
                   'approach', 'evaluation', 'requirements'],
            'h3': ['timeline', 'milestones', 'phase', 'criteria', 'process']
        }
        
        # Special patterns for this document type
        self.special_patterns = [
            (r'^Appendix\s+[A-Z]:', 'H1'),
            (r'^Phase\s+[IVX]+:', 'H2'),
            (r'^\d+\.\s+\w+', 'H1'),  # Numbered sections
        ]
    
    def extract_outline(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract title and outline from PDF"""
        try:
            doc = fitz.open(pdf_path)
            
            # Extract ALL text blocks (not just headings initially)
            all_blocks = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                blocks = self._extract_all_blocks(page, page_num + 1)
                all_blocks.extend(blocks)
            
            # Analyze document structure
            font_analysis = self._analyze_fonts(all_blocks)
            
            # Extract title
            title = self._extract_title(all_blocks, font_analysis)
            
            # Build outline with improved detection
            outline = self._build_improved_outline(all_blocks, font_analysis)
            
            doc.close()
            
            return {
                "title": title,
                "outline": outline
            }
            
        except Exception as e:
            logger.error(f"Error extracting outline: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "title": "",
                "outline": [],
                "error": str(e)
            }
    
    def _extract_all_blocks(self, page: fitz.Page, page_num: int) -> List[TextBlock]:
        """Extract all text blocks from a page"""
        blocks = []
        page_dict = page.get_text("dict")
        
        for block in page_dict["blocks"]:
            if block["type"] == 0:  # Text block
                # Aggregate block properties
                all_text = []
                all_sizes = []
                all_fonts = []
                all_flags = []
                all_spans = []
                
                for line in block.get("lines", []):
                    line_text = []
                    for span in line.get("spans", []):
                        span_text = span.get("text", "").strip()
                        if span_text:
                            line_text.append(span_text)
                            all_sizes.append(span.get("size", 0))
                            all_fonts.append(span.get("font", ""))
                            all_flags.append(span.get("flags", 0))
                            all_spans.append(span)
                    
                    if line_text:
                        all_text.append(" ".join(line_text))
                
                if all_text:
                    full_text = " ".join(all_text).strip()
                    if full_text and len(full_text) > 1:  # Skip single characters
                        # Determine predominant properties
                        avg_size = np.mean(all_sizes) if all_sizes else 0
                        common_font = max(set(all_fonts), key=all_fonts.count) if all_fonts else ""
                        max_flags = max(all_flags) if all_flags else 0
                        
                        is_bold = any(f & 2**4 for f in all_flags)
                        is_italic = any(f & 2**1 for f in all_flags)
                        
                        blocks.append(TextBlock(
                            text=full_text,
                            page=page_num,
                            bbox=block["bbox"],
                            font_size=avg_size,
                            font_name=common_font,
                            is_bold=is_bold,
                            is_italic=is_italic,
                            flags=max_flags,
                            spans=all_spans
                        ))
        
        return blocks
    
    def _analyze_fonts(self, blocks: List[TextBlock]) -> Dict[str, Any]:
        """Analyze font usage patterns in the document"""
        font_sizes = defaultdict(list)
        bold_sizes = []
        regular_sizes = []
        
        for block in blocks:
            font_sizes[block.font_size].append(block)
            if block.is_bold:
                bold_sizes.append(block.font_size)
            else:
                regular_sizes.append(block.font_size)
        
        # Find size clusters
        unique_sizes = sorted(set(b.font_size for b in blocks), reverse=True)
        
        # Identify heading sizes (typically larger and less frequent)
        size_counts = {size: len(font_sizes[size]) for size in unique_sizes}
        total_blocks = len(blocks)
        
        heading_sizes = []
        body_sizes = []
        
        for size in unique_sizes:
            ratio = size_counts[size] / total_blocks
            # Headings are typically less than 20% of all text
            if ratio < 0.2 and size > 10:
                heading_sizes.append(size)
            else:
                body_sizes.append(size)
        
        # Determine thresholds
        body_size = np.median(body_sizes) if body_sizes else 11
        
        return {
            'heading_sizes': heading_sizes[:5],  # Top 5 sizes
            'body_size': body_size,
            'size_distribution': size_counts,
            'h1_threshold': body_size * 1.3,
            'h2_threshold': body_size * 1.15,
            'h3_threshold': body_size * 1.05
        }
    
    def _extract_title(self, blocks: List[TextBlock], font_analysis: Dict) -> str:
        """Extract document title with improved logic"""
        # First two pages only
        early_blocks = [b for b in blocks if b.page <= 2]
        
        # Look for the largest text or special formatting
        candidates = []
        
        for block in early_blocks:
            # Skip if too long for a title
            if len(block.text) > 150:
                continue
                
            # Check for RFP pattern
            if block.text.startswith("RFP:") or "Request for Proposal" in block.text:
                return block.text.strip()
            
            # Large font candidates
            if block.font_size >= font_analysis['h1_threshold']:
                candidates.append((block.font_size, block))
        
        # Return the largest text from first page
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1].text.strip()
        
        return ""
    
    def _is_likely_heading(self, block: TextBlock, font_analysis: Dict) -> Tuple[bool, Optional[str]]:
        """Determine if a block is likely a heading and its level"""
        text = block.text.strip()
        
        # Skip if too long for a heading
        if len(text) > 200:
            return False, None
        
        # Check numbered patterns first
        for pattern, level_type in self.numbered_patterns:
            if re.match(pattern, text):
                if level_type == 'numbered_h1':
                    return True, 'H1'
                elif level_type == 'numbered_h2':
                    return True, 'H2'
                elif level_type == 'numbered_h3':
                    return True, 'H3'
        
        # Check special patterns
        for pattern, level in self.special_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True, level
        
        # Check if text ends with colon (common for headings)
        ends_with_colon = text.endswith(':')
        
        # Font size based detection
        if block.font_size >= font_analysis['h1_threshold']:
            # Large font - likely H1
            if block.is_bold or self._contains_heading_keyword(text, 'h1'):
                return True, 'H1'
            elif ends_with_colon:
                return True, 'H2'
        
        elif block.font_size >= font_analysis['h2_threshold']:
            # Medium font
            if block.is_bold or ends_with_colon or self._contains_heading_keyword(text, 'h2'):
                return True, 'H2'
        
        elif block.font_size >= font_analysis['h3_threshold']:
            # Slightly larger than body
            if (block.is_bold and ends_with_colon) or self._contains_heading_keyword(text, 'h3'):
                return True, 'H3'
        
        # Bold text with colon is often a heading
        elif block.is_bold and ends_with_colon and len(text.split()) <= 10:
            # Check indent level based on x position
            if block.x_position < 100:  # Left aligned
                return True, 'H3'
            else:  # Indented
                return True, 'H4'
        
        # All caps short text
        elif text.isupper() and len(text.split()) <= 8:
            if block.font_size > font_analysis['body_size']:
                return True, 'H2'
        
        return False, None
    
    def _contains_heading_keyword(self, text: str, level: str) -> bool:
        """Check if text contains keywords typical for the heading level"""
        text_lower = text.lower()
        keywords = self.heading_keywords.get(level, [])
        return any(keyword in text_lower for keyword in keywords)
    
    def _build_improved_outline(self, blocks: List[TextBlock], font_analysis: Dict) -> List[Dict[str, Any]]:
        """Build outline with improved heading detection"""
        outline = []
        
        # Sort blocks by page and position
        blocks.sort(key=lambda b: (b.page, b.y_position))
        
        # Process each block
        for block in blocks:
            is_heading, level = self._is_likely_heading(block, font_analysis)
            
            if is_heading and level:
                # Clean the heading text
                text = self._clean_heading_text(block.text)
                
                # Skip empty or very short headings
                if len(text) > 2:
                    outline.append({
                        "level": level,
                        "text": text,
                        "page": block.page
                    })
        
        # Post-process to ensure reasonable hierarchy
        outline = self._refine_outline_hierarchy(outline)
        
        return outline
    
    def _clean_heading_text(self, text: str) -> str:
        """Clean heading text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Keep trailing colons for this format
        text = text.strip()
        # Remove page numbers at the end (but not numbered headings)
        if not re.match(r'^\d+\.', text):
            text = re.sub(r'\s+\d+$', '', text)
        return text
    
    def _refine_outline_hierarchy(self, outline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Refine the outline hierarchy based on context"""
        if not outline:
            return outline
        
        refined = []
        last_h1 = None
        last_h2 = None
        
        for item in outline:
            level = item['level']
            text = item['text']
            
            # Special handling for certain patterns
            if text.startswith('Appendix'):
                level = 'H1'
            elif re.match(r'^\d+\.\s+\w+', text) and '.' in text:
                # Numbered items - check context
                if last_h1 and 'Appendix' in last_h1:
                    level = 'H2'  # Under appendix
                else:
                    level = 'H1'
            
            # Track hierarchy
            if level == 'H1':
                last_h1 = text
                last_h2 = None
            elif level == 'H2':
                last_h2 = text
            
            # Update level
            item['level'] = level
            refined.append(item)
        
        return refined


def process_pdf_for_outline(pdf_path: Path) -> Dict[str, Any]:
    """Process a PDF and extract title and outline"""
    extractor = ImprovedOutlineExtractor()
    return extractor.extract_outline(pdf_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
        if pdf_path.exists():
            result = process_pdf_for_outline(pdf_path)
            print(json.dumps(result, indent=2))
        else:
            print(f"File not found: {pdf_path}")
    else:
        print("Usage: python improved_outline_extractor.py <pdf_file>")