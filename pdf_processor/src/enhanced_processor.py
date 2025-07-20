#!/usr/bin/env python3
"""
Enhanced PDF Processor with OCR and Advanced Structure Detection
Includes image text extraction and ML-based layout analysis
"""

import fitz
import json
import re
import cv2
import numpy as np
import pytesseract
from PIL import Image
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import concurrent.futures
from dataclasses import dataclass, field
import logging
from sklearn.cluster import DBSCAN
from collections import defaultdict
import time

logger = logging.getLogger(__name__)


@dataclass
class EnhancedTextBlock:
    """Enhanced text block with additional metadata"""
    text: str
    bbox: Tuple[float, float, float, float]
    font_size: float
    font_name: str
    page_num: int
    block_type: str = "paragraph"
    confidence: float = 1.0
    is_bold: bool = False
    is_italic: bool = False
    color: str = "#000000"
    language: str = "en"
    reading_order: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "text": self.text,
            "bbox": self.bbox,
            "font_size": self.font_size,
            "font_name": self.font_name,
            "page_num": self.page_num,
            "block_type": self.block_type,
            "confidence": self.confidence,
            "formatting": {
                "bold": self.is_bold,
                "italic": self.is_italic,
                "color": self.color
            },
            "language": self.language,
            "reading_order": self.reading_order
        }


@dataclass
class DocumentElement:
    """Base class for document elements"""
    element_type: str
    content: Any
    page_num: int
    bbox: Tuple[float, float, float, float]
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class LayoutAnalyzer:
    """Advanced layout analysis using clustering and heuristics"""
    
    def __init__(self):
        self.column_threshold = 50  # pixels
        self.line_height_threshold = 1.5
        
    def analyze_layout(self, blocks: List[EnhancedTextBlock]) -> Dict[str, Any]:
        """Analyze document layout and detect columns, headers, etc."""
        if not blocks:
            return {"columns": 1, "layout_type": "single_column"}
        
        # Extract positions for clustering
        positions = np.array([[b.bbox[0], b.bbox[1]] for b in blocks])
        
        # Detect columns using x-coordinate clustering
        x_coords = positions[:, 0].reshape(-1, 1)
        clustering = DBSCAN(eps=self.column_threshold, min_samples=3).fit(x_coords)
        
        num_columns = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        
        # Determine layout type
        layout_type = "single_column" if num_columns == 1 else f"{num_columns}_column"
        
        # Detect reading order
        if num_columns > 1:
            # Multi-column: order by column then by y-position
            for i, label in enumerate(clustering.labels_):
                if label != -1:
                    blocks[i].reading_order = label * 1000 + int(blocks[i].bbox[1])
        else:
            # Single column: order by y-position
            for i, block in enumerate(sorted(blocks, key=lambda b: b.bbox[1])):
                block.reading_order = i
        
        return {
            "columns": num_columns,
            "layout_type": layout_type,
            "column_positions": self._get_column_positions(blocks, clustering.labels_)
        }
    
    def _get_column_positions(self, blocks: List[EnhancedTextBlock], labels: np.ndarray) -> List[Dict]:
        """Get bounding boxes for each column"""
        column_bounds = defaultdict(lambda: {"min_x": float('inf'), "max_x": 0, "min_y": float('inf'), "max_y": 0})
        
        for block, label in zip(blocks, labels):
            if label != -1:
                bounds = column_bounds[label]
                bounds["min_x"] = min(bounds["min_x"], block.bbox[0])
                bounds["max_x"] = max(bounds["max_x"], block.bbox[2])
                bounds["min_y"] = min(bounds["min_y"], block.bbox[1])
                bounds["max_y"] = max(bounds["max_y"], block.bbox[3])
        
        return [{"column_id": k, "bounds": v} for k, v in column_bounds.items()]


class ImageTextExtractor:
    """Extract text from images using OCR"""
    
    def __init__(self):
        self.min_confidence = 60  # Tesseract confidence threshold
        
    def extract_from_image(self, image_bytes: bytes, bbox: Tuple[float, float, float, float]) -> Optional[str]:
        """Extract text from image bytes using OCR"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Preprocess image for better OCR
            processed = self._preprocess_image(cv_image)
            
            # Perform OCR with confidence scores
            data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)
            
            # Extract text with confidence filtering
            words = []
            for i, conf in enumerate(data['conf']):
                if int(conf) > self.min_confidence:
                    word = data['text'][i].strip()
                    if word:
                        words.append(word)
            
            return ' '.join(words) if words else None
            
        except Exception as e:
            logger.warning(f"OCR extraction failed: {e}")
            return None
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising
        denoised = cv2.fastNlDenoising(gray)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return thresh


class EnhancedPDFProcessor:
    """Enhanced PDF processor with OCR and advanced features"""
    
    def __init__(self, max_workers: int = 8, enable_ocr: bool = True):
        self.max_workers = max_workers
        self.enable_ocr = enable_ocr
        self.layout_analyzer = LayoutAnalyzer()
        self.image_extractor = ImageTextExtractor() if enable_ocr else None
        
    def process_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Process PDF with enhanced extraction"""
        start_time = time.time()
        logger.info(f"Processing: {pdf_path.name}")
        
        try:
            doc = fitz.open(pdf_path)
            
            # Process pages in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                for page_num in range(len(doc)):
                    future = executor.submit(self._process_page_enhanced, doc[page_num], page_num)
                    futures.append(future)
                
                # Collect results
                all_elements = []
                for future in concurrent.futures.as_completed(futures):
                    elements = future.result()
                    all_elements.extend(elements)
            
            # Analyze layout
            text_blocks = [e.content for e in all_elements if isinstance(e.content, EnhancedTextBlock)]
            layout_info = self.layout_analyzer.analyze_layout(text_blocks)
            
            # Structure content
            structured_data = self._create_structured_output(all_elements, layout_info)
            
            # Add metadata
            structured_data['metadata'] = {
                'filename': pdf_path.name,
                'pages': len(doc),
                'processing_time': round(time.time() - start_time, 2),
                'extraction_method': 'enhanced_with_ocr' if self.enable_ocr else 'enhanced',
                'layout': layout_info
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
    
    def _process_page_enhanced(self, page: fitz.Page, page_num: int) -> List[DocumentElement]:
        """Process page with enhanced extraction"""
        elements = []
        
        # Extract text blocks
        page_dict = page.get_text("dict")
        
        for block in page_dict["blocks"]:
            if block["type"] == 0:  # Text block
                text_block = self._extract_enhanced_text_block(block, page_num)
                if text_block:
                    elements.append(DocumentElement(
                        element_type="text",
                        content=text_block,
                        page_num=page_num,
                        bbox=block["bbox"]
                    ))
            elif block["type"] == 1 and self.enable_ocr:  # Image block
                # Extract text from image using OCR
                image_text = self._extract_image_text(page, block)
                if image_text:
                    elements.append(DocumentElement(
                        element_type="image_text",
                        content=image_text,
                        page_num=page_num,
                        bbox=block["bbox"],
                        metadata={"source": "ocr"}
                    ))
        
        # Detect tables with improved algorithm
        tables = self._detect_tables_enhanced(page, page_num)
        for table in tables:
            elements.append(DocumentElement(
                element_type="table",
                content=table,
                page_num=page_num,
                bbox=table.get("bbox", (0, 0, 0, 0))
            ))
        
        # Detect forms and fillable fields
        forms = self._detect_forms(page, page_num)
        elements.extend(forms)
        
        return elements
    
    def _extract_enhanced_text_block(self, block: Dict, page_num: int) -> Optional[EnhancedTextBlock]:
        """Extract text block with enhanced metadata"""
        try:
            text_parts = []
            font_sizes = []
            font_names = []
            is_bold = False
            is_italic = False
            colors = []
            
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text_parts.append(span.get("text", ""))
                    font_sizes.append(span.get("size", 0))
                    font_names.append(span.get("font", ""))
                    
                    # Extract formatting
                    flags = span.get("flags", 0)
                    is_bold = is_bold or bool(flags & 2**4)  # Bold flag
                    is_italic = is_italic or bool(flags & 2**1)  # Italic flag
                    
                    # Extract color
                    color = span.get("color", 0)
                    colors.append(f"#{color:06x}" if isinstance(color, int) else "#000000")
            
            if not text_parts:
                return None
            
            # Aggregate properties
            avg_font_size = np.mean(font_sizes) if font_sizes else 0
            most_common_font = max(set(font_names), key=font_names.count) if font_names else ""
            most_common_color = max(set(colors), key=colors.count) if colors else "#000000"
            
            text = " ".join(text_parts)
            
            return EnhancedTextBlock(
                text=text,
                bbox=block["bbox"],
                font_size=avg_font_size,
                font_name=most_common_font,
                page_num=page_num,
                block_type=self._classify_block_type_advanced(text, avg_font_size, is_bold),
                is_bold=is_bold,
                is_italic=is_italic,
                color=most_common_color,
                language=self._detect_language(text)
            )
            
        except Exception as e:
            logger.warning(f"Error extracting enhanced text block: {e}")
            return None
    
    def _classify_block_type_advanced(self, text: str, font_size: float, is_bold: bool) -> str:
        """Advanced block type classification"""
        text_lower = text.lower().strip()
        
        # Title detection
        if font_size > 18 and is_bold:
            return "title"
        
        # Subtitle detection
        if font_size > 14 and is_bold:
            return "subtitle"
        
        # Section header patterns
        section_patterns = [
            r'^\d+\.\d+',  # 1.1, 2.3, etc.
            r'^[A-Z]\.',   # A., B., etc.
            r'^(chapter|section|part)\s+\d+',
            r'^(introduction|conclusion|abstract|summary|references)',
        ]
        
        for pattern in section_patterns:
            if re.match(pattern, text_lower):
                return "section_header"
        
        # List detection
        if re.match(r'^[\•\-\*\▪\▫\◦\‣]\s+', text) or re.match(r'^\d+[\.\)]\s+', text):
            return "list_item"
        
        # Table/Figure caption
        if re.match(r'^(table|figure|exhibit|chart|graph)\s+\d+', text_lower):
            return "caption"
        
        # Footnote detection
        if re.match(r'^\d+\s+', text) and font_size < 10:
            return "footnote"
        
        return "paragraph"
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection based on character sets"""
        # This is a simplified version - could use langdetect library
        if re.search(r'[\u4e00-\u9fff]', text):
            return "zh"  # Chinese
        elif re.search(r'[\u0600-\u06ff]', text):
            return "ar"  # Arabic
        elif re.search(r'[\u0400-\u04ff]', text):
            return "ru"  # Russian
        return "en"  # Default to English
    
    def _extract_image_text(self, page: fitz.Page, block: Dict) -> Optional[str]:
        """Extract text from image block using OCR"""
        try:
            # Get image from block
            bbox = block["bbox"]
            mat = fitz.Matrix(2, 2)  # Scale up for better OCR
            pix = page.get_pixmap(matrix=mat, clip=bbox)
            img_data = pix.tobytes("png")
            
            # Extract text using OCR
            text = self.image_extractor.extract_from_image(img_data, bbox)
            return text
            
        except Exception as e:
            logger.warning(f"Failed to extract image text: {e}")
            return None
    
    def _detect_tables_enhanced(self, page: fitz.Page, page_num: int) -> List[Dict]:
        """Enhanced table detection with better accuracy"""
        tables = []
        
        # Get all text with positions
        words = page.get_text("words")
        
        if not words:
            return tables
        
        # Group words by vertical position (with tolerance)
        y_tolerance = 5
        rows = defaultdict(list)
        
        for word in words:
            y_pos = round(word[1] / y_tolerance) * y_tolerance
            rows[y_pos].append(word)
        
        # Sort rows by y position
        sorted_rows = sorted(rows.items(), key=lambda x: x[0])
        
        # Detect table candidates
        potential_table = []
        prev_col_count = 0
        
        for y_pos, row_words in sorted_rows:
            # Sort words in row by x position
            row_words.sort(key=lambda w: w[0])
            
            # Check if words are aligned in columns
            if len(row_words) >= 2:
                # Calculate gaps between words
                gaps = []
                for i in range(len(row_words) - 1):
                    gap = row_words[i + 1][0] - row_words[i][2]
                    gaps.append(gap)
                
                # If gaps are consistent, likely a table row
                if gaps and max(gaps) > 20:  # Significant gaps indicate columns
                    row_data = [w[4] for w in row_words]  # Extract text
                    
                    # Check column consistency
                    if prev_col_count == 0 or abs(len(row_data) - prev_col_count) <= 1:
                        potential_table.append({
                            "row": row_data,
                            "y_pos": y_pos,
                            "bbox": self._calculate_row_bbox(row_words)
                        })
                        prev_col_count = len(row_data)
                    else:
                        # Column count changed significantly, end current table
                        if len(potential_table) >= 2:
                            tables.append(self._create_table_structure(potential_table, page_num))
                        potential_table = []
                        prev_col_count = 0
        
        # Don't forget the last table
        if len(potential_table) >= 2:
            tables.append(self._create_table_structure(potential_table, page_num))
        
        return tables
    
    def _calculate_row_bbox(self, words: List) -> Tuple[float, float, float, float]:
        """Calculate bounding box for a row of words"""
        if not words:
            return (0, 0, 0, 0)
        
        min_x = min(w[0] for w in words)
        min_y = min(w[1] for w in words)
        max_x = max(w[2] for w in words)
        max_y = max(w[3] for w in words)
        
        return (min_x, min_y, max_x, max_y)
    
    def _create_table_structure(self, table_rows: List[Dict], page_num: int) -> Dict:
        """Create structured table representation"""
        rows = [r["row"] for r in table_rows]
        
        # Calculate overall bbox
        min_x = min(r["bbox"][0] for r in table_rows)
        min_y = min(r["bbox"][1] for r in table_rows)
        max_x = max(r["bbox"][2] for r in table_rows)
        max_y = max(r["bbox"][3] for r in table_rows)
        
        return {
            "headers": rows[0] if rows else [],
            "rows": rows[1:] if len(rows) > 1 else rows,
            "bbox": (min_x, min_y, max_x, max_y),
            "page_num": page_num,
            "num_columns": len(rows[0]) if rows else 0,
            "num_rows": len(rows)
        }
    
    def _detect_forms(self, page: fitz.Page, page_num: int) -> List[DocumentElement]:
        """Detect form fields and interactive elements"""
        forms = []
        
        # Check for form fields
        for widget in page.widgets():
            field_info = {
                "field_name": widget.field_name,
                "field_type": widget.field_type_string,
                "field_value": widget.field_value,
                "rect": tuple(widget.rect)
            }
            
            forms.append(DocumentElement(
                element_type="form_field",
                content=field_info,
                page_num=page_num,
                bbox=field_info["rect"],
                metadata={"interactive": True}
            ))
        
        return forms
    
    def _create_structured_output(self, elements: List[DocumentElement], layout_info: Dict) -> Dict[str, Any]:
        """Create final structured output with all extracted data"""
        output = {
            "document_structure": {
                "title": "",
                "authors": [],
                "abstract": "",
                "sections": [],
                "paragraphs": [],
                "lists": [],
                "tables": [],
                "figures": [],
                "forms": [],
                "footnotes": []
            },
            "layout_analysis": layout_info,
            "text_statistics": {
                "total_words": 0,
                "total_characters": 0,
                "languages_detected": set(),
                "font_statistics": defaultdict(int)
            },
            "extraction_confidence": 1.0
        }
        
        # Sort elements by reading order
        text_elements = sorted(
            [e for e in elements if e.element_type == "text"],
            key=lambda e: (e.page_num, e.content.reading_order)
        )
        
        # Process elements
        current_section = None
        word_count = 0
        char_count = 0
        
        for element in elements:
            if element.element_type == "text":
                block = element.content
                text = block.text.strip()
                
                # Update statistics
                word_count += len(text.split())
                char_count += len(text)
                output["text_statistics"]["languages_detected"].add(block.language)
                output["text_statistics"]["font_statistics"][block.font_name] += 1
                
                # Categorize content
                if block.block_type == "title" and not output["document_structure"]["title"]:
                    output["document_structure"]["title"] = text
                elif block.block_type in ["section_header", "subtitle"]:
                    current_section = {
                        "title": text,
                        "level": 1 if block.block_type == "section_header" else 2,
                        "content": [],
                        "page": block.page_num
                    }
                    output["document_structure"]["sections"].append(current_section)
                elif block.block_type == "list_item":
                    list_item = {
                        "text": text,
                        "page": block.page_num,
                        "section": current_section["title"] if current_section else None
                    }
                    output["document_structure"]["lists"].append(list_item)
                elif block.block_type == "footnote":
                    output["document_structure"]["footnotes"].append({
                        "text": text,
                        "page": block.page_num
                    })
                else:
                    # Regular paragraph
                    para = {
                        "text": text,
                        "page": block.page_num,
                        "formatting": {
                            "font_size": block.font_size,
                            "font_name": block.font_name,
                            "bold": block.is_bold,
                            "italic": block.is_italic,
                            "color": block.color
                        }
                    }
                    
                    if current_section:
                        current_section["content"].append(para)
                    else:
                        output["document_structure"]["paragraphs"].append(para)
            
            elif element.element_type == "table":
                output["document_structure"]["tables"].append(element.content)
            
            elif element.element_type == "form_field":
                output["document_structure"]["forms"].append(element.content)
            
            elif element.element_type == "image_text":
                # Add OCR-extracted text to figures
                output["document_structure"]["figures"].append({
                    "extracted_text": element.content,
                    "page": element.page_num,
                    "bbox": element.bbox,
                    "extraction_method": "ocr"
                })
        
        # Update final statistics
        output["text_statistics"]["total_words"] = word_count
        output["text_statistics"]["total_characters"] = char_count
        output["text_statistics"]["languages_detected"] = list(output["text_statistics"]["languages_detected"])
        output["text_statistics"]["font_statistics"] = dict(output["text_statistics"]["font_statistics"])
        
        return output


# Import required for OCR
import io


def main():
    """Main entry point for the enhanced processor"""
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    
    # Initialize processor with OCR enabled
    processor = EnhancedPDFProcessor(max_workers=8, enable_ocr=True)
    
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
    main()