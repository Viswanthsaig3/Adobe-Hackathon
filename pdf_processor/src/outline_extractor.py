#!/usr/bin/env python3
"""
PDF Outline Extractor - Extracts title and hierarchical outline from PDFs
Outputs in the specific JSON format required
"""

import fitz
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import logging
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OutlineItem:
    """Represents an outline item with hierarchical level"""
    level: str  # H1, H2, H3, H4
    text: str
    page: int
    font_size: float
    font_name: str
    is_bold: bool
    y_position: float


class OutlineExtractor:
    """Extracts document title and hierarchical outline from PDFs"""
    
    def __init__(self):
        self.heading_patterns = [
            # Numbered patterns
            (r'^\d+\.\s+', 'H1'),  # 1. 
            (r'^\d+\.\d+\s+', 'H2'),  # 1.1
            (r'^\d+\.\d+\.\d+\s+', 'H3'),  # 1.1.1
            # Letter patterns
            (r'^[A-Z]\.\s+', 'H2'),  # A.
            (r'^[a-z]\.\s+', 'H3'),  # a.
            # Roman numerals
            (r'^[IVX]+\.\s+', 'H1'),  # I., II., III.
            (r'^[ivx]+\.\s+', 'H2'),  # i., ii., iii.
            # Keywords
            (r'^(Chapter|CHAPTER)\s+\d+', 'H1'),
            (r'^(Section|SECTION)\s+\d+', 'H2'),
            (r'^(Appendix|APPENDIX)\s+[A-Z]', 'H1'),
        ]
        
        self.section_keywords = [
            'introduction', 'conclusion', 'summary', 'abstract',
            'background', 'methodology', 'results', 'discussion',
            'references', 'appendix', 'acknowledgments'
        ]
    
    def extract_outline(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract title and outline from PDF"""
        try:
            doc = fitz.open(pdf_path)
            
            # Extract all text blocks with formatting
            all_blocks = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                blocks = self._extract_page_blocks(page, page_num + 1)
                all_blocks.extend(blocks)
            
            # Identify title and outline items
            title = self._extract_title(all_blocks)
            outline = self._build_outline(all_blocks)
            
            doc.close()
            
            return {
                "title": title,
                "outline": outline
            }
            
        except Exception as e:
            logger.error(f"Error extracting outline: {str(e)}")
            return {
                "title": "",
                "outline": [],
                "error": str(e)
            }
    
    def _extract_page_blocks(self, page: fitz.Page, page_num: int) -> List[OutlineItem]:
        """Extract text blocks from a page with formatting info"""
        blocks = []
        page_dict = page.get_text("dict")
        
        for block in page_dict["blocks"]:
            if block["type"] == 0:  # Text block
                # Extract text and formatting
                text_parts = []
                font_sizes = []
                font_names = []
                is_bold = False
                
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text_parts.append(span.get("text", ""))
                        font_sizes.append(span.get("size", 0))
                        font_names.append(span.get("font", ""))
                        
                        flags = span.get("flags", 0)
                        is_bold = is_bold or bool(flags & 2**4)
                
                if text_parts:
                    text = " ".join(text_parts).strip()
                    if text:
                        avg_font_size = np.mean(font_sizes) if font_sizes else 0
                        most_common_font = max(set(font_names), key=font_names.count) if font_names else ""
                        
                        # Determine if this is a heading
                        level = self._determine_heading_level(
                            text, avg_font_size, is_bold, most_common_font
                        )
                        
                        if level:  # Only keep headings
                            blocks.append(OutlineItem(
                                level=level,
                                text=self._clean_heading_text(text),
                                page=page_num,
                                font_size=avg_font_size,
                                font_name=most_common_font,
                                is_bold=is_bold,
                                y_position=block["bbox"][1]
                            ))
        
        return blocks
    
    def _determine_heading_level(self, text: str, font_size: float, is_bold: bool, font_name: str) -> Optional[str]:
        """Determine if text is a heading and its level"""
        text_lower = text.lower().strip()
        
        # Check for numbered/lettered patterns
        for pattern, level in self.heading_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return level
        
        # Check for section keywords
        for keyword in self.section_keywords:
            if text_lower.startswith(keyword):
                # Determine level based on font size and formatting
                if font_size > 16 or (font_size > 14 and is_bold):
                    return 'H1'
                elif font_size > 12 or is_bold:
                    return 'H2'
                else:
                    return 'H3'
        
        # Font-based detection
        if len(text.split()) <= 10:  # Short enough to be a heading
            if font_size > 18 and is_bold:
                return 'H1'
            elif font_size > 14 and is_bold:
                return 'H2'
            elif font_size > 12 and is_bold:
                return 'H3'
            elif is_bold and text.endswith(':'):
                return 'H3'  # Bold text ending with colon
        
        # All caps detection
        if text.isupper() and len(text.split()) <= 8:
            if font_size > 14:
                return 'H1'
            else:
                return 'H2'
        
        return None
    
    def _clean_heading_text(self, text: str) -> str:
        """Clean heading text by removing extra whitespace and normalizing"""
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove trailing colons for consistency
        text = text.rstrip(':')
        # Remove page numbers at the end
        text = re.sub(r'\s+\d+$', '', text)
        return text.strip()
    
    def _extract_title(self, blocks: List[OutlineItem]) -> str:
        """Extract the document title from the first page"""
        # Look for the largest text on the first page
        first_page_blocks = [b for b in blocks if b.page == 1]
        
        if not first_page_blocks:
            return ""
        
        # Sort by font size (descending) and y position (ascending)
        first_page_blocks.sort(key=lambda b: (-b.font_size, b.y_position))
        
        # The title is usually the largest text at the top
        for block in first_page_blocks[:3]:  # Check top 3 candidates
            if block.font_size > 14:
                # Combine with next block if it's also large (multi-line title)
                title_parts = [block.text]
                
                # Look for continuation
                for other in first_page_blocks:
                    if (other != block and 
                        abs(other.font_size - block.font_size) < 2 and
                        abs(other.y_position - block.y_position) < 50 and
                        other.y_position > block.y_position):
                        title_parts.append(other.text)
                
                return " ".join(title_parts)
        
        return ""
    
    def _build_outline(self, blocks: List[OutlineItem]) -> List[Dict[str, Any]]:
        """Build the hierarchical outline structure"""
        outline = []
        
        # Sort blocks by page and position
        blocks.sort(key=lambda b: (b.page, b.y_position))
        
        # Track font sizes for each level
        level_font_sizes = {'H1': [], 'H2': [], 'H3': [], 'H4': []}
        
        # First pass: collect font sizes for each level
        for block in blocks:
            if block.level:
                level_font_sizes[block.level].append(block.font_size)
        
        # Calculate average font sizes for each level
        avg_sizes = {}
        for level, sizes in level_font_sizes.items():
            if sizes:
                avg_sizes[level] = np.mean(sizes)
        
        # Second pass: refine levels based on font size hierarchy
        for block in blocks:
            if block.level:
                # Adjust level based on relative font size
                if block.font_size > 16 and block.level != 'H1':
                    # Check if this should be H1
                    if not avg_sizes.get('H1') or block.font_size >= avg_sizes.get('H1', 0) * 0.9:
                        block.level = 'H1'
                
                # Add to outline
                outline.append({
                    "level": block.level,
                    "text": block.text,
                    "page": block.page
                })
        
        # Post-process to ensure logical hierarchy
        outline = self._ensure_hierarchy(outline)
        
        return outline
    
    def _ensure_hierarchy(self, outline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure outline follows logical hierarchy (H1 -> H2 -> H3 -> H4)"""
        if not outline:
            return outline
        
        # Track the current level
        processed = []
        last_level = 0
        level_map = {'H1': 1, 'H2': 2, 'H3': 3, 'H4': 4}
        
        for item in outline:
            current_level = level_map.get(item['level'], 1)
            
            # Don't skip levels (e.g., H1 -> H3 becomes H1 -> H2)
            if current_level > last_level + 1:
                current_level = last_level + 1
                item['level'] = f'H{current_level}'
            
            last_level = current_level
            processed.append(item)
        
        return processed


def process_pdf_for_outline(pdf_path: Path) -> Dict[str, Any]:
    """Process a PDF and extract title and outline"""
    extractor = OutlineExtractor()
    return extractor.extract_outline(pdf_path)


if __name__ == "__main__":
    # Test the extractor
    import sys
    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
        if pdf_path.exists():
            result = process_pdf_for_outline(pdf_path)
            print(json.dumps(result, indent=2))
        else:
            print(f"File not found: {pdf_path}")
    else:
        print("Usage: python outline_extractor.py <pdf_file>")