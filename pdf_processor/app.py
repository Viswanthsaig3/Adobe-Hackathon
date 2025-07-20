#!/usr/bin/env python3
"""
Flask Web Application for PDF Outline Extraction
"""

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import json
from pathlib import Path
import tempfile
import shutil
from datetime import datetime
import logging

# Add src to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from improved_outline_extractor import process_pdf_for_outline

# Configure Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf'}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Render the main upload page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        # Check if file was selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only PDF files are allowed'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        logger.info(f"Processing file: {filename}")
        
        # Process the PDF
        result = process_pdf_for_outline(Path(filepath))
        
        # Add metadata
        result['metadata'] = {
            'filename': filename,
            'processed_at': datetime.now().isoformat(),
            'file_size': os.path.getsize(filepath)
        }
        
        # Clean up uploaded file
        os.remove(filepath)
        
        logger.info(f"Successfully processed: {filename}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500


@app.route('/download/<format>')
def download_result(format):
    """Download the last processed result in specified format"""
    try:
        # In a real app, you'd store results in a database
        # For now, we'll just return a sample
        sample_result = {
            "title": "Sample Document Title",
            "outline": [
                {"level": "H1", "text": "Introduction", "page": 1},
                {"level": "H2", "text": "Background", "page": 2}
            ]
        }
        
        if format == 'json':
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            json.dump(sample_result, temp_file, indent=2)
            temp_file.close()
            
            return send_file(
                temp_file.name,
                mimetype='application/json',
                as_attachment=True,
                download_name='outline.json'
            )
        else:
            return jsonify({'error': 'Invalid format'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/status')
def status():
    """API status endpoint"""
    return jsonify({
        'status': 'online',
        'version': '1.0.0',
        'max_file_size': app.config['MAX_CONTENT_LENGTH']
    })


@app.errorhandler(413)
def request_entity_too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 50MB'}), 413


@app.teardown_appcontext
def cleanup(error):
    """Clean up temporary files"""
    # Clean old files from upload folder
    upload_folder = Path(app.config['UPLOAD_FOLDER'])
    if upload_folder.exists():
        for file in upload_folder.glob('*'):
            try:
                if file.is_file() and (datetime.now() - datetime.fromtimestamp(file.stat().st_mtime)).seconds > 3600:
                    file.unlink()
            except:
                pass


if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)