import fitz
import json
import pandas as pd
from pathlib import Path
import numpy as np

class AdvancedDatasetGenerator:
    def __init__(self):
        self.features_list = []
        self.labels = []
        
    def generate_comprehensive_dataset(self, pdf_directory):
        """Generate rich training dataset with multiple feature types"""
        
        for pdf_file in Path(pdf_directory).glob("*.pdf"):
            print(f"Processing {pdf_file}...")
            
            doc = fitz.open(pdf_file)
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                
                # Get multiple text extraction formats
                text_dict = page.get_text("dict")
                text_blocks = page.get_text("blocks")
                page_height = page.rect.height
                page_width = page.rect.width
                
                # Extract text with comprehensive features
                for block in text_dict["blocks"]:
                    if "lines" in block:
                        for line_idx, line in enumerate(block["lines"]):
                            for span_idx, span in enumerate(line["spans"]):
                                
                                text_content = span["text"].strip()
                                if len(text_content) < 3:
                                    continue
                                
                                # Extract comprehensive features
                                features = self.extract_comprehensive_features(
                                    span, block, line, page_num, 
                                    page_height, page_width, text_dict, 
                                    line_idx, span_idx
                                )
                                
                                # Auto-generate sophisticated labels
                                label = self.advanced_auto_classify(
                                    text_content, features, span, block
                                )
                                
                                self.features_list.append(features)
                                self.labels.append(label)
                                
                                # Store detailed example
                                self.store_training_example(
                                    text_content, features, label, 
                                    pdf_file.name, page_num + 1
                                )
            
            doc.close()
        
        return self.create_training_datasets()
    
    def extract_comprehensive_features(self, span, block, line, page_num, 
                                     page_height, page_width, text_dict, 
                                     line_idx, span_idx):
        """Extract 50+ features for maximum accuracy"""
        
        bbox = span["bbox"]
        text = span["text"].strip()
        
        # Spatial features (15 features)
        spatial_features = {
            "x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3],
            "width": bbox[2] - bbox[0],
            "height": bbox[3] - bbox[1],
            "center_x": (bbox[0] + bbox[2]) / 2,
            "center_y": (bbox[1] + bbox[3]) / 2,
            "relative_x": (bbox[0] + bbox[2]) / 2 / page_width,
            "relative_y": (bbox[1] + bbox[3]) / 2 / page_height,
            "distance_from_top": bbox[1] / page_height,
            "distance_from_bottom": (page_height - bbox[3]) / page_height,
            "distance_from_left": bbox[0] / page_width,
            "distance_from_right": (page_width - bbox[2]) / page_width,
            "aspect_ratio": (bbox[2] - bbox[0]) / (bbox[3] - bbox[1]) if bbox[3] != bbox[1] else 0
        }
        
        # Typography features (15 features)
        typography_features = {
            "font_size": span["size"],
            "font_family": span["font"],
            "is_bold": 1 if "Bold" in span["font"] else 0,
            "is_italic": 1 if "Italic" in span["font"] else 0,
            "is_serif": 1 if "Times" in span["font"] else 0,
            "is_sans_serif": 1 if any(x in span["font"] for x in ["Arial", "Helvetica", "Calibri"]) else 0,
            "color": span["color"],
            "is_black_text": 1 if span["color"] == 0 else 0,
            "font_weight_score": self.calculate_font_weight(span["font"]),
            "relative_font_size": span["size"] / self.get_average_font_size(text_dict),
            "font_size_category": self.categorize_font_size(span["size"]),
            "is_monospace": 1 if any(x in span["font"] for x in ["Courier", "Monaco", "Consolas"]) else 0,
            "has_special_formatting": 1 if any(x in span["font"] for x in ["Condensed", "Extended", "Light"]) else 0,
            "text_decoration": self.detect_text_decoration(span),
            "line_height_ratio": line["bbox"][3] - line["bbox"][1] / span["size"] if span["size"] > 0 else 0
        }
        
        # Content features (20 features)
        content_features = {
            "char_count": len(text),
            "word_count": len(text.split()),
            "sentence_count": text.count('.') + text.count('!') + text.count('?'),
            "is_uppercase": 1 if text.isupper() else 0,
            "is_titlecase": 1 if text.istitle() else 0,
            "is_lowercase": 1 if text.islower() else 0,
            "caps_ratio": sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            "digit_ratio": sum(1 for c in text if c.isdigit()) / len(text) if text else 0,
            "punct_ratio": sum(1 for c in text if c in ".,!?;:") / len(text) if text else 0,
            "whitespace_ratio": sum(1 for c in text if c.isspace()) / len(text) if text else 0,
            "starts_with_number": 1 if text and text[0].isdigit() else 0,
            "starts_with_bullet": 1 if text.startswith(('•', '-', '*')) else 0,
            "ends_with_colon": 1 if text.endswith(':') else 0,
            "ends_with_period": 1 if text.endswith('.') else 0,
            "contains_parentheses": 1 if '(' in text and ')' in text else 0,
            "has_numbers": 1 if any(c.isdigit() for c in text) else 0,
            "has_roman_numerals": 1 if self.has_roman_numerals(text) else 0,
            "numbering_pattern": self.detect_numbering_pattern(text),
            "text_complexity": self.calculate_text_complexity(text),
            "avg_word_length": sum(len(word) for word in text.split()) / len(text.split()) if text.split() else 0
        }
        
        # Context features (10 features)
        context_features = {
            "page_number": page_num + 1,
            "total_blocks_on_page": len(text_dict["blocks"]),
            "block_index": self.get_block_index(block, text_dict["blocks"]),
            "line_index_in_block": line_idx,
            "span_index_in_line": span_idx,
            "surrounding_whitespace": self.calculate_surrounding_whitespace(bbox, text_dict),
            "is_isolated_text": 1 if self.is_isolated_text(bbox, text_dict) else 0,
            "vertical_alignment": self.get_vertical_alignment(bbox, line),
            "horizontal_alignment": self.get_horizontal_alignment(bbox, page_width),
            "text_density_around": self.calculate_text_density_around(bbox, text_dict)
        }
        
        # Combine all features
        all_features = {**spatial_features, **typography_features, 
                       **content_features, **context_features}
        
        return all_features
    
    def advanced_auto_classify(self, text, features, span, block):
        """Sophisticated auto-classification using multiple heuristics"""
        
        # Document title detection
        if (features["page_number"] == 1 and 
            features["distance_from_top"] < 0.3 and
            features["font_size"] >= 16 and
            features["is_bold"] and
            len(text.split()) > 2):
            return "DOCUMENT_TITLE"
        
        # H1 headers (main sections)
        if (features["is_bold"] and 
            features["font_size"] >= 14 and
            features["distance_from_top"] > 0.1 and
            not features["ends_with_colon"] and
            features["word_count"] <= 15):
            return "H1"
        
        # H2 headers (subsections)
        if (features["is_bold"] and 
            features["font_size"] >= 12 and
            features["font_size"] < 14 and
            not text.startswith(('•', '-', '*')) and
            features["word_count"] <= 12):
            return "H2"
        
        # H3 headers (sub-subsections)
        if (features["is_bold"] and 
            features["font_size"] >= 11 and
            features["font_size"] < 12 and
            (features["ends_with_colon"] or features["word_count"] <= 8)):
            return "H3"
        
        # H4 headers (minor headers)
        if (features["is_bold"] and 
            features["font_size"] >= 10 and
            features["font_size"] < 11 and
            features["word_count"] <= 10):
            return "H4"
        
        # List items
        if features["starts_with_bullet"] or features["starts_with_number"]:
            return "LIST_ITEM"
        
        # Page numbers
        if (features["distance_from_bottom"] < 0.1 and
            features["word_count"] <= 3 and
            (features["has_numbers"] or text.lower().startswith('page'))):
            return "PAGE_NUMBER"
        
        # Headers and footers
        if features["distance_from_top"] < 0.1:
            return "HEADER"
        elif features["distance_from_bottom"] < 0.1:
            return "FOOTER"
        
        # Default to paragraph
        return "PARAGRAPH"

# Additional helper methods for feature extraction
    def calculate_font_weight(self, font_name):
        """Calculate numeric font weight"""
        weight_map = {
            "Light": 300, "Regular": 400, "Medium": 500,
            "Semibold": 600, "Bold": 700, "Black": 900
        }
        for weight, value in weight_map.items():
            if weight in font_name:
                return value
        return 400  # Default
    
    def get_average_font_size(self, text_dict):
        """Calculate average font size on page"""
        sizes = []
        for block in text_dict["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        sizes.append(span["size"])
        return np.mean(sizes) if sizes else 12
    
    def has_roman_numerals(self, text):
        """Detect Roman numerals"""
        import re
        roman_pattern = r'\b[IVXLCDM]+\b'
        return bool(re.search(roman_pattern, text.upper()))
    
    def detect_numbering_pattern(self, text):
        """Detect various numbering patterns"""
        import re
        patterns = {
            "decimal": r'^\d+\.',
            "letter": r'^[a-zA-Z]\.',
            "roman": r'^[ivxlcdmIVXLCDM]+\.',
            "parentheses": r'^\(\d+\)',
            "bracket": r'^\[\d+\]'
        }
        
        for pattern_name, pattern in patterns.items():
            if re.match(pattern, text.strip()):
                return pattern_name
        return "none"

# Generate dataset
generator = AdvancedDatasetGenerator()
training_data = generator.generate_comprehensive_dataset("sample_pdfs/")