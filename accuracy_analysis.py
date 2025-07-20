#!/usr/bin/env python3
"""
Accuracy Analysis Script
Compare Real Json outputs with current outputs to identify gaps
"""

import json
from pathlib import Path

def compare_files():
    """Compare Real Json with current output files"""
    
    current_dir = Path(__file__).parent
    real_json_folder = current_dir / "Real Json"
    output_folder = current_dir / "output"
    
    print("🔍 ACCURACY ANALYSIS")
    print("=" * 60)
    
    total_files = 5
    accuracy_issues = []
    
    for i in range(1, total_files + 1):
        file_name = f"file{i:02d}.json"
        real_file = real_json_folder / file_name
        output_file = output_folder / file_name
        
        print(f"\n📄 {file_name}")
        print("-" * 30)
        
        # Load files
        with open(real_file, 'r') as f:
            real_data = json.load(f)
        with open(output_file, 'r') as f:
            output_data = json.load(f)
        
        # Compare titles
        real_title = real_data.get('title', '').strip()
        output_title = output_data.get('title', '').strip()
        
        title_match = real_title == output_title
        print(f"Title Match: {'✅' if title_match else '❌'}")
        if not title_match:
            print(f"  Expected: '{real_title}'")
            print(f"  Got:      '{output_title}'")
            accuracy_issues.append(f"{file_name}: Title mismatch")
        
        # Compare outline counts
        real_outline = real_data.get('outline', [])
        output_outline = output_data.get('outline', [])
        
        count_match = len(real_outline) == len(output_outline)
        print(f"Outline Count: {'✅' if count_match else '❌'}")
        print(f"  Expected: {len(real_outline)} headings")
        print(f"  Got:      {len(output_outline)} headings")
        
        if not count_match:
            accuracy_issues.append(f"{file_name}: Count mismatch ({len(output_outline)} vs {len(real_outline)})")
        
        # Show first few headings for comparison
        if real_outline:
            print("  Expected headings:")
            for j, heading in enumerate(real_outline[:3]):
                print(f"    {heading.get('level', '')}: {heading.get('text', '')}")
            if len(real_outline) > 3:
                print(f"    ... and {len(real_outline) - 3} more")
        
        if output_outline:
            print("  Current headings:")
            for j, heading in enumerate(output_outline[:3]):
                print(f"    {heading.get('level', '')}: {heading.get('text', '')}")
            if len(output_outline) > 3:
                print(f"    ... and {len(output_outline) - 3} more")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 ACCURACY SUMMARY")
    print(f"Total files: {total_files}")
    print(f"Issues found: {len(accuracy_issues)}")
    print(f"Current accuracy: {((total_files * 2 - len(accuracy_issues)) / (total_files * 2)) * 100:.1f}%")
    print(f"Target accuracy: 99.0%")
    
    if accuracy_issues:
        print("\n🚨 ISSUES TO FIX:")
        for issue in accuracy_issues:
            print(f"  • {issue}")
    
    return len(accuracy_issues) == 0

if __name__ == "__main__":
    compare_files()