#!/usr/bin/env python3
"""
Extract outline from all PDFs in the pdfs folder and save as JSON
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pathlib import Path
import json
from precise_outline_extractor import process_pdf_for_outline

def main():
    # Hardcoded paths
    pdf_folder = Path("/home/viswanthsai/Downloads/ADobe hackathon neeha/pdfs")
    output_folder = Path("/home/viswanthsai/Downloads/ADobe hackathon neeha/json")
    
    # Create output folder if it doesn't exist
    output_folder.mkdir(exist_ok=True)
    
    # Get all PDF files
    pdf_files = list(pdf_folder.glob("*.pdf"))
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        
        try:
            # Extract outline
            result = process_pdf_for_outline(pdf_file)
            
            # Save as JSON
            output_file = output_folder / f"{pdf_file.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Saved to: {output_file.name}")
            print(f"  Title: {result.get('title', 'No title')[:80]}...")
            print(f"  Outline items: {len(result.get('outline', []))}")
            
        except Exception as e:
            print(f"✗ Error processing {pdf_file.name}: {str(e)}")
    
    print(f"\nProcessing complete! Check the {output_folder} directory for results.")

if __name__ == "__main__":
    main()