#!/usr/bin/env python3
"""
Compare generated JSON outputs with required outputs and calculate matching percentage
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import difflib
from collections import defaultdict

class OutputComparator:
    def __init__(self):
        self.generated_folder = Path("/home/viswanthsai/Downloads/ADobe hackathon neeha/json")
        self.required_folder = Path("/home/viswanthsai/Downloads/ADobe hackathon neeha/Real Json")
        self.pdf_folder = Path("/home/viswanthsai/Downloads/ADobe hackathon neeha/pdfs")
        
    def load_json(self, filepath: Path) -> Dict:
        """Load JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return {}
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        # Remove extra spaces, convert to lowercase for comparison
        return ' '.join(text.strip().lower().split())
    
    def compare_titles(self, gen_title: str, req_title: str) -> float:
        """Compare titles and return similarity score"""
        gen_norm = self.normalize_text(gen_title)
        req_norm = self.normalize_text(req_title)
        
        # Use sequence matcher for similarity
        return difflib.SequenceMatcher(None, gen_norm, req_norm).ratio()
    
    def compare_outline_items(self, gen_outline: List[Dict], req_outline: List[Dict]) -> Dict:
        """Compare outline items and return detailed metrics"""
        metrics = {
            'total_required': len(req_outline),
            'total_generated': len(gen_outline),
            'exact_matches': 0,
            'partial_matches': 0,
            'level_matches': 0,
            'page_matches': 0,
            'missing_items': [],
            'extra_items': [],
            'item_scores': []
        }
        
        # Create normalized lookup for required items
        req_items = {}
        for item in req_outline:
            key = (self.normalize_text(item['text']), item['level'], item['page'])
            req_items[key] = item
        
        # Track matched items
        matched_keys = set()
        
        # Check each generated item
        for gen_item in gen_outline:
            gen_text_norm = self.normalize_text(gen_item['text'])
            best_match = None
            best_score = 0
            
            # Find best matching required item
            for req_key, req_item in req_items.items():
                req_text_norm, req_level, req_page = req_key
                
                # Calculate text similarity
                text_score = difflib.SequenceMatcher(None, gen_text_norm, req_text_norm).ratio()
                
                if text_score > best_score:
                    best_score = text_score
                    best_match = (req_key, req_item)
            
            if best_match and best_score > 0.8:  # 80% similarity threshold
                req_key, req_item = best_match
                matched_keys.add(req_key)
                
                # Check if exact match
                if best_score >= 0.95:
                    metrics['exact_matches'] += 1
                else:
                    metrics['partial_matches'] += 1
                
                # Check level match
                if gen_item['level'] == req_item['level']:
                    metrics['level_matches'] += 1
                
                # Check page match
                if gen_item['page'] == req_item['page']:
                    metrics['page_matches'] += 1
                
                metrics['item_scores'].append({
                    'generated': gen_item['text'],
                    'required': req_item['text'],
                    'score': best_score,
                    'level_match': gen_item['level'] == req_item['level'],
                    'page_match': gen_item['page'] == req_item['page']
                })
            else:
                metrics['extra_items'].append(gen_item)
        
        # Find missing items
        for req_key, req_item in req_items.items():
            if req_key not in matched_keys:
                metrics['missing_items'].append(req_item)
        
        return metrics
    
    def calculate_overall_score(self, title_score: float, outline_metrics: Dict) -> Dict:
        """Calculate overall matching percentage"""
        # Title contributes 20% to overall score
        title_weight = 0.2
        outline_weight = 0.8
        
        # Calculate outline score components
        if outline_metrics['total_required'] > 0:
            # Text matching score
            text_match_score = (outline_metrics['exact_matches'] + 0.5 * outline_metrics['partial_matches']) / outline_metrics['total_required']
            
            # Level accuracy score
            level_score = outline_metrics['level_matches'] / max(outline_metrics['total_generated'], 1)
            
            # Page accuracy score
            page_score = outline_metrics['page_matches'] / max(outline_metrics['total_generated'], 1)
            
            # Completeness score (penalize missing items)
            completeness = 1 - (len(outline_metrics['missing_items']) / outline_metrics['total_required'])
            
            # Precision score (penalize extra items)
            if outline_metrics['total_generated'] > 0:
                precision = 1 - (len(outline_metrics['extra_items']) / outline_metrics['total_generated'])
            else:
                precision = 0
            
            # Weighted outline score
            outline_score = (
                0.4 * text_match_score +
                0.2 * level_score +
                0.1 * page_score +
                0.2 * completeness +
                0.1 * precision
            )
        else:
            outline_score = 0
        
        # Overall score
        overall_score = title_weight * title_score + outline_weight * outline_score
        
        return {
            'overall_percentage': round(overall_score * 100, 2),
            'title_score': round(title_score * 100, 2),
            'outline_score': round(outline_score * 100, 2),
            'text_matching': round(text_match_score * 100, 2) if outline_metrics['total_required'] > 0 else 0,
            'level_accuracy': round(level_score * 100, 2) if outline_metrics['total_generated'] > 0 else 0,
            'page_accuracy': round(page_score * 100, 2) if outline_metrics['total_generated'] > 0 else 0,
            'completeness': round(completeness * 100, 2) if outline_metrics['total_required'] > 0 else 0,
            'precision': round(precision * 100, 2) if outline_metrics['total_generated'] > 0 else 0
        }
    
    def compare_file(self, filename: str) -> Dict:
        """Compare a single file"""
        gen_path = self.generated_folder / filename
        req_path = self.required_folder / filename
        
        if not gen_path.exists():
            return {'error': f'Generated file not found: {filename}'}
        
        if not req_path.exists():
            return {'error': f'Required file not found: {filename}'}
        
        # Load JSON files
        gen_data = self.load_json(gen_path)
        req_data = self.load_json(req_path)
        
        # Compare titles
        title_score = self.compare_titles(
            gen_data.get('title', ''),
            req_data.get('title', '')
        )
        
        # Compare outlines
        outline_metrics = self.compare_outline_items(
            gen_data.get('outline', []),
            req_data.get('outline', [])
        )
        
        # Calculate overall score
        scores = self.calculate_overall_score(title_score, outline_metrics)
        
        return {
            'filename': filename,
            'scores': scores,
            'metrics': outline_metrics,
            'generated_title': gen_data.get('title', '')[:100] + '...' if len(gen_data.get('title', '')) > 100 else gen_data.get('title', ''),
            'required_title': req_data.get('title', '')[:100] + '...' if len(req_data.get('title', '')) > 100 else req_data.get('title', '')
        }
    
    def compare_all(self):
        """Compare all JSON files"""
        print("=" * 80)
        print("PDF OUTLINE EXTRACTION COMPARISON REPORT")
        print("=" * 80)
        
        # Get all JSON files from generated folder
        json_files = list(self.generated_folder.glob("*.json"))
        
        all_scores = []
        detailed_results = []
        
        for json_file in sorted(json_files):
            result = self.compare_file(json_file.name)
            
            if 'error' not in result:
                all_scores.append(result['scores']['overall_percentage'])
                detailed_results.append(result)
                
                print(f"\n{result['filename']}:")
                print("-" * 50)
                print(f"Overall Match: {result['scores']['overall_percentage']}%")
                print(f"  Title Match: {result['scores']['title_score']}%")
                print(f"  Outline Match: {result['scores']['outline_score']}%")
                print(f"    - Text Matching: {result['scores']['text_matching']}%")
                print(f"    - Level Accuracy: {result['scores']['level_accuracy']}%")
                print(f"    - Page Accuracy: {result['scores']['page_accuracy']}%")
                print(f"    - Completeness: {result['scores']['completeness']}%")
                print(f"    - Precision: {result['scores']['precision']}%")
                print(f"  Items: {result['metrics']['total_generated']} generated / {result['metrics']['total_required']} required")
                print(f"  Missing: {len(result['metrics']['missing_items'])} items")
                print(f"  Extra: {len(result['metrics']['extra_items'])} items")
            else:
                print(f"\n{json_file.name}: {result['error']}")
        
        # Summary statistics
        if all_scores:
            print("\n" + "=" * 80)
            print("SUMMARY STATISTICS")
            print("=" * 80)
            print(f"Files Compared: {len(all_scores)}")
            print(f"Average Match: {sum(all_scores) / len(all_scores):.2f}%")
            print(f"Best Match: {max(all_scores):.2f}%")
            print(f"Worst Match: {min(all_scores):.2f}%")
            
            # Detailed analysis for worst performing file
            worst_file = min(detailed_results, key=lambda x: x['scores']['overall_percentage'])
            print(f"\nWorst Performing File: {worst_file['filename']}")
            print("Issues:")
            if worst_file['metrics']['missing_items']:
                print(f"  Missing {len(worst_file['metrics']['missing_items'])} items:")
                for item in worst_file['metrics']['missing_items'][:5]:  # Show first 5
                    print(f"    - {item['level']}: {item['text'][:50]}... (page {item['page']})")
                if len(worst_file['metrics']['missing_items']) > 5:
                    print(f"    ... and {len(worst_file['metrics']['missing_items']) - 5} more")

if __name__ == "__main__":
    comparator = OutputComparator()
    comparator.compare_all()