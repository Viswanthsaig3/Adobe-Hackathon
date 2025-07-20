#!/usr/bin/env python3
"""
Test script for PDF processor validation
"""

import json
import time
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pdf_processor import AdvancedPDFProcessor

def create_test_pdf():
    """Create a simple test PDF if none exists"""
    try:
        import fitz
        
        # Create a new PDF
        doc = fitz.open()
        
        # Add pages with different content types
        for i in range(5):
            page = doc.new_page()
            
            # Add title
            page.insert_text((72, 72), f"Test Document Page {i+1}", fontsize=16, fontname="helv-Bold")
            
            # Add heading
            page.insert_text((72, 120), f"Section {i+1}: Test Content", fontsize=14, fontname="helv-Bold")
            
            # Add paragraph
            paragraph = f"""This is a test paragraph on page {i+1}. It contains multiple sentences 
to test the text extraction capabilities. The PDF processor should be able to extract 
this text and identify it as a paragraph with proper formatting information."""
            
            page.insert_text((72, 160), paragraph, fontsize=11, fontname="helv")
            
            # Add list
            list_items = [
                "• First list item",
                "• Second list item", 
                "• Third list item"
            ]
            
            y_pos = 250
            for item in list_items:
                page.insert_text((72, y_pos), item, fontsize=11)
                y_pos += 20
            
            # Add simple table-like structure
            if i == 2:  # Add table on page 3
                table_data = [
                    ["Header 1", "Header 2", "Header 3"],
                    ["Data 1.1", "Data 1.2", "Data 1.3"],
                    ["Data 2.1", "Data 2.2", "Data 2.3"]
                ]
                
                y_pos = 400
                for row in table_data:
                    x_pos = 72
                    for cell in row:
                        page.insert_text((x_pos, y_pos), cell, fontsize=10)
                        x_pos += 150
                    y_pos += 20
        
        # Save the test PDF
        test_pdf_path = Path("test_input/test_document.pdf")
        test_pdf_path.parent.mkdir(exist_ok=True)
        doc.save(str(test_pdf_path))
        doc.close()
        
        print(f"Created test PDF: {test_pdf_path}")
        return test_pdf_path
        
    except ImportError:
        print("PyMuPDF not installed. Cannot create test PDF.")
        return None

def validate_output(output_path: Path):
    """Validate the JSON output structure"""
    with open(output_path, 'r') as f:
        data = json.load(f)
    
    # Check required fields
    required_fields = ["document_structure", "metadata"]
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"
    
    # Check document structure
    doc_struct = data["document_structure"]
    struct_fields = ["title", "sections", "paragraphs", "lists", "tables"]
    for field in struct_fields:
        assert field in doc_struct, f"Missing document structure field: {field}"
    
    # Check metadata
    metadata = data["metadata"]
    meta_fields = ["filename", "pages", "processing_time"]
    for field in meta_fields:
        assert field in metadata, f"Missing metadata field: {field}"
    
    # Check processing time constraint
    assert metadata["processing_time"] < 10, f"Processing time exceeded: {metadata['processing_time']}s"
    
    print("✓ Output validation passed")
    print(f"  - Processing time: {metadata['processing_time']}s")
    print(f"  - Pages processed: {metadata['pages']}")
    print(f"  - Title extracted: {doc_struct.get('title', 'No title')}")
    print(f"  - Sections found: {len(doc_struct['sections'])}")
    print(f"  - Paragraphs found: {len(doc_struct['paragraphs'])}")
    print(f"  - Lists found: {len(doc_struct['lists'])}")
    print(f"  - Tables found: {len(doc_struct['tables'])}")

def test_processor():
    """Run processor tests"""
    print("PDF Processor Test Suite")
    print("=" * 50)
    
    # Create test PDF if needed
    test_pdf = create_test_pdf()
    if not test_pdf:
        print("Using existing test PDFs")
        test_pdf = Path("test_input/test_document.pdf")
    
    if not test_pdf.exists():
        print("No test PDF found. Please add a test PDF to test_input/")
        return
    
    # Initialize processor
    print("\nInitializing processor...")
    processor = AdvancedPDFProcessor(max_workers=4)
    
    # Process the test PDF
    print(f"\nProcessing: {test_pdf.name}")
    start_time = time.time()
    
    try:
        result = processor.process_pdf(test_pdf)
        processing_time = time.time() - start_time
        
        # Save output
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{test_pdf.stem}.json"
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"✓ Processing completed in {processing_time:.2f}s")
        print(f"✓ Output saved to: {output_path}")
        
        # Validate output
        print("\nValidating output...")
        validate_output(output_path)
        
        # Performance check
        print("\nPerformance Check:")
        if processing_time < 10:
            print(f"✓ PASS: Processing time ({processing_time:.2f}s) within 10s limit")
        else:
            print(f"✗ FAIL: Processing time ({processing_time:.2f}s) exceeds 10s limit")
        
    except Exception as e:
        print(f"✗ Processing failed: {str(e)}")
        import traceback
        traceback.print_exc()

def test_memory_usage():
    """Test memory usage with larger PDFs"""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"\nInitial memory usage: {initial_memory:.2f} MB")
    
    # Process multiple PDFs
    test_dir = Path("test_input")
    if test_dir.exists():
        pdfs = list(test_dir.glob("*.pdf"))
        if pdfs:
            processor = AdvancedPDFProcessor(max_workers=4)
            for pdf in pdfs[:3]:  # Test with up to 3 PDFs
                processor.process_pdf(pdf)
                current_memory = process.memory_info().rss / 1024 / 1024
                print(f"Memory after processing {pdf.name}: {current_memory:.2f} MB")
    
    final_memory = process.memory_info().rss / 1024 / 1024
    memory_increase = final_memory - initial_memory
    
    print(f"\nMemory usage increase: {memory_increase:.2f} MB")
    if memory_increase < 1000:  # Less than 1GB increase
        print("✓ PASS: Memory usage within acceptable limits")
    else:
        print("✗ WARNING: High memory usage detected")

if __name__ == "__main__":
    test_processor()
    test_memory_usage()
    print("\n" + "=" * 50)
    print("Testing complete!")