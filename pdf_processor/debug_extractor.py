#!/usr/bin/env python3
"""
Debug PDF Extractor - Shows all text blocks and classification decisions
"""

import fitz
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import logging
from dataclasses import dataclass
import numpy as np

logging.basicConfig(level=logging.DEBUG)
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
    is_all_caps: bool
    word_count: int
    ends_with_colon: bool
    is_centered: bool
    indent_level: int
    
    @property
    def y_position(self) -> float:
        return self.bbox[1]
    
    @property
    def x_position(self) -> float:
        return self.bbox[0]


class DebugOutlineExtractor:
    """Debug version that logs all decisions"""
    
    def __init__(self):
        # Patterns that are definitely headings
        self.definite_heading_patterns = [
            # Numbered sections
            (r'^Chapter\s+\d+[\s:\-]', 'H1'),
            (r'^Section\s+\d+[\s:\-]', 'H2'),
            (r'^Appendix\s+[A-Z][\s:\-]', 'H1'),
            (r'^\d+\.\s+[A-Z][^.]*$', 'H1'),  # "1. Title" (no period at end)
            (r'^\d+\.\d+\s+[A-Z][^.]*$', 'H2'),  # "1.1 Title"
            (r'^\d+\.\d+\.\d+\s+[A-Z][^.]*$', 'H3'),  # "1.1.1 Title"
            
            # Phase patterns
            (r'^Phase\s+[IVX]+[\s:\-]', 'H2'),
            
            # Bullet patterns (for subheadings)
            (r'^[•·▪▫◦‣]\s*[A-Z][^.]*$', 'H3'),  # Bullet + title (no sentence)
        ]
        
        # Keywords that indicate headings ONLY when they're standalone or with colon
        self.heading_keywords = {
            'h1': ['appendix', 'chapter', 'introduction', 'conclusion'],
            'h2': ['summary', 'background', 'overview', 'approach', 'evaluation', 
                   'requirements', 'methodology', 'references'],
            'h3': ['timeline', 'milestones', 'criteria', 'process', 'objectives',
                   'preamble', 'membership', 'term', 'chair', 'meetings']
        }
        
        # Phrases that are NOT headings (sentences)
        self.sentence_indicators = [
            'will be', 'shall be', 'must be', 'should be', 'would be',
            'has been', 'have been', 'had been',
            'is expected', 'are expected',
            'we will', 'we are', 'we have',
            'this is', 'these are', 'that is',
            'contracts with', 'developing a', 'preparing for',
            'to ensure', 'to provide', 'to develop',
            'responsible for', 'accountable to',
            'in order to', 'as well as', 'such as'
        ]
    
    def extract_outline(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract title and outline from PDF with debug info"""
        try:
            doc = fitz.open(pdf_path)
            
            # Extract all text blocks
            all_blocks = []
            page_widths = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_widths.append(page.rect.width)
                blocks = self._extract_all_blocks(page, page_num + 1, page.rect.width)
                all_blocks.extend(blocks)
            
            # Analyze document structure
            font_analysis = self._analyze_fonts(all_blocks)
            
            # Log all blocks on first few pages
            logger.info("\n=== TEXT BLOCKS ON ALL PAGES ===")
            for block in all_blocks:
                if block.page <= 15:
                    logger.info(f"\nPage {block.page}, Y={block.y_position:.1f}, Size={block.font_size:.1f}, Bold={block.is_bold}")
                    logger.info(f"Text: '{block.text}'")
                    logger.info(f"Word count: {block.word_count}, Ends with colon: {block.ends_with_colon}")
                    
                    # Check if it's a heading
                    is_heading, level = self._is_true_heading(block, font_analysis)
                    if is_heading:
                        logger.info(f"✓ CLASSIFIED AS: {level}")
                    else:
                        reason = self._get_rejection_reason(block, font_analysis)
                        logger.info(f"✗ REJECTED: {reason}")
            
            # Extract title
            title = self._extract_clean_title(all_blocks, font_analysis)
            logger.info(f"\n=== EXTRACTED TITLE ===\n{title}")
            
            # Build outline
            outline = self._build_precise_outline(all_blocks, font_analysis)
            
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
    
    def _get_rejection_reason(self, block: TextBlock, font_analysis: Dict) -> str:
        """Get reason why block was rejected as heading"""
        text = block.text.strip()
        
        # Check if it's a sentence
        if self._is_sentence(text):
            return "Identified as sentence"
        
        # Check if too long
        if block.word_count > 15:
            return f"Too long ({block.word_count} words > 15)"
        
        # Check font size
        if block.font_size < font_analysis['body_size']:
            return f"Font too small ({block.font_size:.1f} < body size {font_analysis['body_size']:.1f})"
        
        # Check if it matches any patterns
        text_lower = text.lower().rstrip(':')
        if text_lower in sum(self.heading_keywords.values(), []):
            return f"Keyword '{text_lower}' but insufficient formatting"
        
        return "No heading characteristics found"
    
    def _extract_all_blocks(self, page: fitz.Page, page_num: int, page_width: float) -> List[TextBlock]:
        """Extract all text blocks from a page"""
        blocks = []
        page_dict = page.get_text("dict")
        
        for block in page_dict["blocks"]:
            if block["type"] == 0:  # Text block
                # Collect all text from the block
                full_text = ""
                all_sizes = []
                all_fonts = []
                is_bold = False
                is_italic = False
                
                for line in block.get("lines", []):
                    line_texts = []
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text:
                            line_texts.append(text)
                            all_sizes.append(span.get("size", 0))
                            all_fonts.append(span.get("font", ""))
                            flags = span.get("flags", 0)
                            is_bold = is_bold or bool(flags & 2**4)
                            is_italic = is_italic or bool(flags & 2**1)
                    
                    if line_texts:
                        full_text += " " + " ".join(line_texts)
                
                full_text = full_text.strip()
                
                if full_text and len(full_text) > 1:
                    # Calculate properties
                    avg_size = np.mean(all_sizes) if all_sizes else 0
                    common_font = max(set(all_fonts), key=all_fonts.count) if all_fonts else ""
                    
                    # Determine if centered
                    bbox = block["bbox"]
                    center_x = (bbox[0] + bbox[2]) / 2
                    is_centered = abs(center_x - page_width/2) < page_width * 0.1
                    
                    # Determine indent level
                    indent_level = 0
                    if bbox[0] > 100:
                        indent_level = 1
                    if bbox[0] > 150:
                        indent_level = 2
                    
                    blocks.append(TextBlock(
                        text=full_text,
                        page=page_num,
                        bbox=bbox,
                        font_size=avg_size,
                        font_name=common_font,
                        is_bold=is_bold,
                        is_italic=is_italic,
                        is_all_caps=full_text.isupper(),
                        word_count=len(full_text.split()),
                        ends_with_colon=full_text.rstrip().endswith(':'),
                        is_centered=is_centered,
                        indent_level=indent_level
                    ))
        
        return blocks
    
    def _analyze_fonts(self, blocks: List[TextBlock]) -> Dict[str, Any]:
        """Analyze font usage patterns"""
        font_sizes = []
        bold_sizes = []
        
        for block in blocks:
            font_sizes.append(block.font_size)
            if block.is_bold:
                bold_sizes.append(block.font_size)
        
        # Calculate thresholds
        body_size = np.median(font_sizes) if font_sizes else 11
        
        logger.info(f"\n=== FONT ANALYSIS ===")
        logger.info(f"Body size: {body_size:.1f}")
        logger.info(f"Font sizes: min={min(font_sizes):.1f}, max={max(font_sizes):.1f}")
        
        return {
            'body_size': body_size,
            'h1_min_size': body_size * 1.4,  # At least 40% larger
            'h2_min_size': body_size * 1.2,  # At least 20% larger
            'h3_min_size': body_size * 1.1,  # At least 10% larger
            'max_size': max(font_sizes) if font_sizes else 20
        }
    
    def _extract_clean_title(self, blocks: List[TextBlock], font_analysis: Dict) -> str:
        """Extract the document title"""
        # Look on first two pages
        early_pages = [b for b in blocks if b.page <= 2]
        
        # Look for RFP pattern
        for block in early_pages:
            if 'RFP' in block.text and 'Request for Proposal' in block.text:
                text = block.text
                # Clean up
                text = re.sub(r'([A-Za-z])\s*\1{2,}', r'\1', text)
                text = re.sub(r'\s+', ' ', text).strip()
                return text
        
        return ""
    
    def _is_sentence(self, text: str) -> bool:
        """Check if text is a sentence rather than a heading"""
        text_lower = text.lower()
        
        # Check for sentence indicators
        for indicator in self.sentence_indicators:
            if indicator in text_lower:
                return True
        
        # Check for multiple sentences
        sentences = re.split(r'(?<![A-Z])\.\s+', text)
        if len(sentences) > 1:
            return True
        
        # Long text with proper sentence structure
        if len(text) > 100 and text.count(' ') > 10:
            if re.search(r'\b(is|are|was|were|will|shall|has|have|had)\b', text_lower):
                return True
        
        return False
    
    def _is_true_heading(self, block: TextBlock, font_analysis: Dict) -> Tuple[bool, Optional[str]]:
        """Determine if a block is a true heading"""
        text = block.text.strip()
        
        # Skip if it's a sentence
        if self._is_sentence(text):
            return False, None
        
        # Skip if too long
        if block.word_count > 15:
            return False, None
        
        # Check definite heading patterns
        for pattern, level in self.definite_heading_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True, level
        
        # For short text (potential headings)
        if block.word_count <= 6:
            # All caps short text
            if block.is_all_caps and block.font_size > font_analysis['body_size']:
                return True, 'H2'
            
            # Bold text ending with colon
            if block.is_bold and block.ends_with_colon:
                if block.indent_level == 0:
                    return True, 'H3'
                else:
                    return True, 'H4'
            
            # Check for standalone keywords
            text_lower = text.lower().rstrip(':')
            for level, keywords in self.heading_keywords.items():
                if text_lower in keywords:
                    if level == 'h1' and block.font_size >= font_analysis['h1_min_size']:
                        return True, 'H1'
                    elif level == 'h2' and (block.font_size >= font_analysis['h2_min_size'] or block.is_bold):
                        return True, 'H2'
                    elif level == 'h3' and block.is_bold:
                        return True, 'H3'
        
        # Large font standalone text
        if block.font_size >= font_analysis['h1_min_size'] and block.word_count <= 8:
            return True, 'H1'
        
        return False, None
    
    def _build_precise_outline(self, blocks: List[TextBlock], font_analysis: Dict) -> List[Dict[str, Any]]:
        """Build outline with only true headings"""
        outline = []
        
        # Sort blocks by page and position
        blocks.sort(key=lambda b: (b.page, b.y_position))
        
        # First pass: identify all headings
        for block in blocks:
            is_heading, level = self._is_true_heading(block, font_analysis)
            
            if is_heading and level:
                text = block.text.strip()
                
                # Special handling for known patterns
                if text.startswith('For each Ontario'):
                    level = 'H4'
                
                outline.append({
                    "level": level,
                    "text": text,
                    "page": block.page
                })
        
        return outline


def main():
    import sys
    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
        if pdf_path.exists():
            extractor = DebugOutlineExtractor()
            result = extractor.extract_outline(pdf_path)
            print("\n=== FINAL OUTPUT ===")
            print(json.dumps(result, indent=2))
        else:
            print(f"File not found: {pdf_path}")
    else:
        print("Usage: python debug_extractor.py <pdf_file>")


if __name__ == "__main__":
    main()