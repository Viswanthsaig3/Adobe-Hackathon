// PDF Outline Extractor - Frontend JavaScript

class PDFProcessor {
    constructor() {
        this.file = null;
        this.result = null;
        this.initializeElements();
        this.attachEventListeners();
    }

    initializeElements() {
        // Get DOM elements
        this.uploadArea = document.getElementById('uploadArea');
        this.fileInput = document.getElementById('fileInput');
        this.fileInfo = document.getElementById('fileInfo');
        this.fileName = document.getElementById('fileName');
        this.fileSize = document.getElementById('fileSize');
        this.removeBtn = document.getElementById('removeBtn');
        this.processBtn = document.getElementById('processBtn');
        
        this.progressSection = document.getElementById('progressSection');
        this.progressFill = document.getElementById('progressFill');
        this.progressText = document.getElementById('progressText');
        
        this.resultsSection = document.getElementById('resultsSection');
        this.documentTitle = document.getElementById('documentTitle');
        this.outlineContainer = document.getElementById('outlineContainer');
        this.jsonOutput = document.getElementById('jsonOutput');
        this.copyBtn = document.getElementById('copyBtn');
        this.downloadBtn = document.getElementById('downloadBtn');
        this.newUploadBtn = document.getElementById('newUploadBtn');
        
        this.errorSection = document.getElementById('errorSection');
        this.errorMessage = document.getElementById('errorMessage');
        this.retryBtn = document.getElementById('retryBtn');
    }

    attachEventListeners() {
        // File upload events
        this.uploadArea.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        
        // Drag and drop events
        this.uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        this.uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        this.uploadArea.addEventListener('drop', (e) => this.handleDrop(e));
        
        // Button events
        this.removeBtn.addEventListener('click', () => this.removeFile());
        this.processBtn.addEventListener('click', () => this.processFile());
        this.copyBtn.addEventListener('click', () => this.copyJSON());
        this.downloadBtn.addEventListener('click', () => this.downloadJSON());
        this.newUploadBtn.addEventListener('click', () => this.resetUI());
        this.retryBtn.addEventListener('click', () => this.resetUI());
    }

    handleFileSelect(event) {
        const file = event.target.files[0];
        if (file && file.type === 'application/pdf') {
            this.setFile(file);
        } else {
            this.showError('Please select a valid PDF file');
        }
    }

    handleDragOver(event) {
        event.preventDefault();
        this.uploadArea.classList.add('dragging');
    }

    handleDragLeave(event) {
        event.preventDefault();
        this.uploadArea.classList.remove('dragging');
    }

    handleDrop(event) {
        event.preventDefault();
        this.uploadArea.classList.remove('dragging');
        
        const files = event.dataTransfer.files;
        if (files.length > 0) {
            const file = files[0];
            if (file.type === 'application/pdf') {
                this.setFile(file);
            } else {
                this.showError('Please drop a valid PDF file');
            }
        }
    }

    setFile(file) {
        this.file = file;
        this.fileName.textContent = file.name;
        this.fileSize.textContent = this.formatFileSize(file.size);
        this.uploadArea.style.display = 'none';
        this.fileInfo.style.display = 'block';
    }

    removeFile() {
        this.file = null;
        this.fileInput.value = '';
        this.uploadArea.style.display = 'block';
        this.fileInfo.style.display = 'none';
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    async processFile() {
        if (!this.file) return;

        // Show progress
        this.fileInfo.style.display = 'none';
        this.progressSection.style.display = 'block';
        this.progressFill.style.width = '0%';
        
        // Simulate progress
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += Math.random() * 30;
            if (progress > 90) progress = 90;
            this.progressFill.style.width = progress + '%';
        }, 500);

        try {
            // Create form data
            const formData = new FormData();
            formData.append('file', this.file);

            // Send to server
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            clearInterval(progressInterval);
            this.progressFill.style.width = '100%';

            if (response.ok) {
                this.result = await response.json();
                this.progressText.textContent = 'Processing complete!';
                
                // Show results after a short delay
                setTimeout(() => {
                    this.showResults();
                }, 500);
            } else {
                const error = await response.json();
                throw new Error(error.error || 'Processing failed');
            }
        } catch (error) {
            clearInterval(progressInterval);
            this.showError(error.message);
        }
    }

    showResults() {
        this.progressSection.style.display = 'none';
        this.resultsSection.style.display = 'block';

        // Display title
        this.documentTitle.textContent = this.result.title || 'No title found';

        // Display outline
        this.displayOutline(this.result.outline || []);

        // Display JSON
        this.jsonOutput.textContent = JSON.stringify(this.result, null, 2);
    }

    displayOutline(outline) {
        this.outlineContainer.innerHTML = '';
        
        if (outline.length === 0) {
            this.outlineContainer.innerHTML = '<p style="color: var(--text-secondary);">No outline found</p>';
            return;
        }

        outline.forEach(item => {
            const outlineItem = document.createElement('div');
            outlineItem.className = `outline-item ${item.level.toLowerCase()}`;
            
            const level = document.createElement('span');
            level.className = 'outline-level';
            level.textContent = item.level;
            
            const text = document.createElement('span');
            text.textContent = item.text;
            
            const page = document.createElement('span');
            page.className = 'outline-page';
            page.textContent = `p. ${item.page}`;
            
            outlineItem.appendChild(level);
            outlineItem.appendChild(text);
            outlineItem.appendChild(page);
            
            this.outlineContainer.appendChild(outlineItem);
        });
    }

    copyJSON() {
        const jsonText = JSON.stringify(this.result, null, 2);
        navigator.clipboard.writeText(jsonText).then(() => {
            // Show feedback
            const originalText = this.copyBtn.innerHTML;
            this.copyBtn.innerHTML = '✓ Copied!';
            this.copyBtn.style.color = 'var(--success-color)';
            
            setTimeout(() => {
                this.copyBtn.innerHTML = originalText;
                this.copyBtn.style.color = '';
            }, 2000);
        }).catch(() => {
            this.showError('Failed to copy to clipboard');
        });
    }

    downloadJSON() {
        const jsonText = JSON.stringify(this.result, null, 2);
        const blob = new Blob([jsonText], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `outline_${this.file.name.replace('.pdf', '')}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    showError(message) {
        this.progressSection.style.display = 'none';
        this.resultsSection.style.display = 'none';
        this.errorSection.style.display = 'block';
        this.errorMessage.textContent = message;
    }

    resetUI() {
        // Hide all sections
        this.progressSection.style.display = 'none';
        this.resultsSection.style.display = 'none';
        this.errorSection.style.display = 'none';
        
        // Reset file input
        this.removeFile();
        
        // Clear results
        this.result = null;
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new PDFProcessor();
});