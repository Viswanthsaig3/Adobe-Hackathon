#!/usr/bin/env python3
"""
Main entry point for PDF processing
Automatically selects the best processor based on available resources
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_ocr_available():
    """Check if OCR dependencies are available"""
    try:
        import cv2
        import pytesseract
        # Test if tesseract is installed
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False

def main():
    """Main entry point"""
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    
    # Check if directories exist
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for PDFs
    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in input directory")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Determine which processor to use
    use_enhanced = os.environ.get("USE_ENHANCED_PROCESSOR", "auto").lower()
    
    if use_enhanced == "auto":
        # Auto-detect based on available features
        use_enhanced = check_ocr_available()
        logger.info(f"Auto-detected OCR availability: {use_enhanced}")
    else:
        use_enhanced = use_enhanced in ["true", "yes", "1"]
    
    # Import and use the appropriate processor
    if use_enhanced:
        try:
            from src.enhanced_processor import EnhancedPDFProcessor
            logger.info("Using Enhanced PDF Processor with OCR support")
            processor = EnhancedPDFProcessor(max_workers=8, enable_ocr=True)
        except ImportError as e:
            logger.warning(f"Failed to import enhanced processor: {e}")
            logger.info("Falling back to standard processor")
            from src.pdf_processor import AdvancedPDFProcessor
            processor = AdvancedPDFProcessor(max_workers=8)
    else:
        from src.pdf_processor import AdvancedPDFProcessor
        logger.info("Using Advanced PDF Processor")
        processor = AdvancedPDFProcessor(max_workers=8)
    
    # Process each PDF
    for pdf_file in pdf_files:
        try:
            logger.info(f"Processing: {pdf_file.name}")
            
            # Process the PDF
            result = processor.process_pdf(pdf_file)
            
            # Save the result
            output_file = output_dir / f"{pdf_file.stem}.json"
            
            import json
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Successfully processed {pdf_file.name}")
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_file.name}: {str(e)}")
            # Continue with next file
            continue
    
    logger.info("Processing complete")

if __name__ == "__main__":
    main()