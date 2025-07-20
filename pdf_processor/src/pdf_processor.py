#!/usr/bin/env python3
"""
Advanced PDF Processing Engine for Adobe Hackathon 2025
Implements multi-threaded processing with layout analysis and intelligent extraction
"""

import fitz  # PyMuPDF
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import concurrent.futures
from dataclasses import dataclass, asdict
import numpy as np
from collections import defaultdict
import logging
import time
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TextBlock:
    """Represents a text block with position and formatting information"""
    text: str
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    font_size: float
    font_name: str
    page_num: int
    block_type: str = "paragraph"
    confidence: float = 1.0
    
    @property
    def is_header(self) -> bool:
        """Detect if block is likely a header based on font size and position"""
        return self.font_size > 14 or self.bbox[1] < 100
    
    @property
    def is_footer(self) -> bool:
        """Detect if block is likely a footer based on position"""
        return self.bbox[1] > 700


@dataclass
class TableData:
    """Represents extracted table data"""
    rows: List[List[str]]
    headers: Optional[List[str]]
    bbox: Tuple[float, float, float, float]
    page_num: int
    confidence: float = 1.0


class AdvancedPDFProcessor:
    """Advanced PDF processor with layout analysis and structure detection"""
    
    def __init__(self, max_workers: int = 8):
        self.max_workers = max_workers
        self.font_size_threshold = 12
        self.header_patterns = [
            r'^chapter\s+\d+',
            r'^section\s+\d+',
            r'^\d+\.\s+',
            r'^[A-Z][A-Z\s]+$'  # All caps headers
        ]
        
    def process_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Process a single PDF file and extract structured data"""
        start_time = time.time()
        logger.info(f"Processing: {pdf_path.name}")
        
        try:
            doc = fitz.open(pdf_path)
            
            # Extract data from all pages in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                page_futures = []
                for page_num in range(len(doc)):
                    future = executor.submit(self._process_page, doc[page_num], page_num)
                    page_futures.append(future)
                
                # Collect results
                all_blocks = []
                all_tables = []
                for future in concurrent.futures.as_completed(page_futures):
                    blocks, tables = future.result()
                    all_blocks.extend(blocks)
                    all_tables.extend(tables)
            
            # Structure the extracted data
            structured_data = self._structure_content(all_blocks, all_tables)
            
            # Add metadata
            structured_data['metadata'] = {
                'filename': pdf_path.name,
                'pages': len(doc),
                'processing_time': round(time.time() - start_time, 2),
                'extraction_method': 'advanced_layout_analysis'
            }
            
            doc.close()
            
            logger.info(f"Completed {pdf_path.name} in {structured_data['metadata']['processing_time']}s")
            return structured_data
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path.name}: {str(e)}")
            return {
                'error': str(e),
                'filename': pdf_path.name,
                'status': 'failed'
            }
    
    def _process_page(self, page: fitz.Page, page_num: int) -> Tuple[List[TextBlock], List[TableData]]:
        """Process a single page and extract text blocks and tables"""
        blocks = []
        tables = []
        
        # Extract text with layout information
        page_dict = page.get_text("dict")
        
        for block in page_dict["blocks"]:
            if block["type"] == 0:  # Text block
                text_block = self._extract_text_block(block, page_num)
                if text_block and text_block.text.strip():
                    blocks.append(text_block)
            elif block["type"] == 1:  # Image block
                # Could implement OCR here if needed
                pass
        
        # Detect and extract tables
        tables_found = self._detect_tables(page, page_num)
        tables.extend(tables_found)
        
        return blocks, tables
    
    def _extract_text_block(self, block: Dict, page_num: int) -> Optional[TextBlock]:
        """Extract text block with formatting information"""
        try:
            text_parts = []
            font_sizes = []
            font_names = []
            
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text_parts.append(span.get("text", ""))
                    font_sizes.append(span.get("size", 0))
                    font_names.append(span.get("font", ""))
            
            if not text_parts:
                return None
            
            # Calculate average font size
            avg_font_size = np.mean(font_sizes) if font_sizes else 0
            most_common_font = max(set(font_names), key=font_names.count) if font_names else ""
            
            return TextBlock(
                text=" ".join(text_parts),
                bbox=block["bbox"],
                font_size=avg_font_size,
                font_name=most_common_font,
                page_num=page_num,
                block_type=self._classify_block_type(" ".join(text_parts), avg_font_size)
            )
            
        except Exception as e:
            logger.warning(f"Error extracting text block: {e}")
            return None
    
    def _classify_block_type(self, text: str, font_size: float) -> str:
        """Classify the type of text block"""
        text_lower = text.lower().strip()
        
        # Check for headers
        for pattern in self.header_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return "header"
        
        # Check for title (large font)
        if font_size > 16:
            return "title"
        
        # Check for list items
        if re.match(r'^[\•\-\*]\s+', text) or re.match(r'^\d+\.\s+', text):
            return "list_item"
        
        # Check for captions
        if text_lower.startswith(("figure", "table", "exhibit")):
            return "caption"
        
        return "paragraph"
    
    def _detect_tables(self, page: fitz.Page, page_num: int) -> List[TableData]:
        """Detect and extract tables from the page"""
        tables = []
        
        # Simple table detection based on text alignment
        # This is a basic implementation - could be enhanced with ML models
        text_dict = page.get_text("dict")
        
        # Group text by vertical position
        y_groups = defaultdict(list)
        for block in text_dict["blocks"]:
            if block["type"] == 0:  # Text block
                for line in block.get("lines", []):
                    y_pos = round(line["bbox"][1], 1)
                    y_groups[y_pos].append(line)
        
        # Detect potential table rows (multiple aligned text elements)
        potential_rows = []
        for y_pos, lines in y_groups.items():
            if len(lines) >= 2:  # At least 2 columns
                row_data = []
                for line in sorted(lines, key=lambda x: x["bbox"][0]):
                    text = " ".join(span["text"] for span in line.get("spans", []))
                    row_data.append(text.strip())
                if row_data:
                    potential_rows.append((y_pos, row_data))
        
        # Group consecutive rows into tables
        if potential_rows:
            potential_rows.sort(key=lambda x: x[0])
            current_table = []
            last_y = -float('inf')
            
            for y_pos, row_data in potential_rows:
                if y_pos - last_y < 30:  # Rows within 30 units are part of same table
                    current_table.append(row_data)
                else:
                    if len(current_table) >= 2:  # At least 2 rows
                        tables.append(TableData(
                            rows=current_table,
                            headers=current_table[0] if current_table else None,
                            bbox=(0, 0, 0, 0),  # Would need proper calculation
                            page_num=page_num
                        ))
                    current_table = [row_data]
                last_y = y_pos
            
            # Don't forget the last table
            if len(current_table) >= 2:
                tables.append(TableData(
                    rows=current_table,
                    headers=current_table[0] if current_table else None,
                    bbox=(0, 0, 0, 0),
                    page_num=page_num
                ))
        
        return tables
    
    def _structure_content(self, blocks: List[TextBlock], tables: List[TableData]) -> Dict[str, Any]:
        """Structure the extracted content into hierarchical format"""
        structured = {
            "document_structure": {
                "title": "",
                "sections": [],
                "paragraphs": [],
                "lists": [],
                "tables": []
            },
            "raw_text": "",
            "statistics": {
                "total_blocks": len(blocks),
                "total_tables": len(tables),
                "block_types": defaultdict(int)
            }
        }
        
        # Sort blocks by page and position
        blocks.sort(key=lambda b: (b.page_num, b.bbox[1]))
        
        current_section = None
        all_text = []
        
        for block in blocks:
            all_text.append(block.text)
            structured["statistics"]["block_types"][block.block_type] += 1
            
            if block.block_type == "title" and not structured["document_structure"]["title"]:
                structured["document_structure"]["title"] = block.text.strip()
            elif block.block_type == "header":
                # Create new section
                current_section = {
                    "title": block.text.strip(),
                    "content": [],
                    "page": block.page_num
                }
                structured["document_structure"]["sections"].append(current_section)
            elif block.block_type == "list_item":
                structured["document_structure"]["lists"].append({
                    "text": block.text.strip(),
                    "page": block.page_num,
                    "section": current_section["title"] if current_section else None
                })
            else:
                # Regular paragraph
                para_data = {
                    "text": block.text.strip(),
                    "page": block.page_num,
                    "font_size": block.font_size,
                    "type": block.block_type
                }
                
                if current_section:
                    current_section["content"].append(para_data)
                else:
                    structured["document_structure"]["paragraphs"].append(para_data)
        
        # Add tables to structure
        for table in tables:
            table_data = {
                "headers": table.headers,
                "rows": table.rows[1:] if table.headers else table.rows,
                "page": table.page_num,
                "num_columns": len(table.rows[0]) if table.rows else 0,
                "num_rows": len(table.rows)
            }
            structured["document_structure"]["tables"].append(table_data)
        
        structured["raw_text"] = "\n\n".join(all_text)
        
        return structured


def process_all_pdfs(input_dir: Path, output_dir: Path, max_workers: int = 8):
    """Process all PDFs in the input directory"""
    processor = AdvancedPDFProcessor(max_workers=max_workers)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all PDF files
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning("No PDF files found in input directory")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF
    for pdf_file in pdf_files:
        try:
            # Process the PDF
            result = processor.process_pdf(pdf_file)
            
            # Save the result
            output_file = output_dir / f"{pdf_file.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Saved output to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_file.name}: {str(e)}")


if __name__ == "__main__":
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    
    # Use all available CPUs (max 8 as per requirements)
    max_workers = min(8, len(list(input_dir.glob("*.pdf"))))
    
    process_all_pdfs(input_dir, output_dir, max_workers=max_workers)