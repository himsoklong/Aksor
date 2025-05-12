document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const processButton = document.getElementById('processButton');
    const imagePreview = document.getElementById('imagePreview');
    const textResult = document.getElementById('textResult');
    const copyButton = document.getElementById('copyButton');
    const downloadButton = document.getElementById('downloadButton');
    const loadingOverlay = document.getElementById('loadingOverlay');
    
    let selectedFile = null;
    
    // Event Listeners
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('active');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('active');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('active');
        
        if (e.dataTransfer.files.length) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFileSelect(e.target.files[0]);
        }
    });
    
    processButton.addEventListener('click', processImage);
    
    copyButton.addEventListener('click', () => {
        const text = textResult.innerText;
        navigator.clipboard.writeText(text)
            .then(() => {
                showMessage('Text copied to clipboard!', 'success');
            })
            .catch(err => {
                showMessage('Failed to copy text', 'error');
                console.error('Failed to copy text: ', err);
            });
    });
    
    downloadButton.addEventListener('click', () => {
        const text = textResult.innerText;
        const blob = new Blob([text], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'ocr-result.txt';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    });
    
    // Functions
    function handleFileSelect(file) {
        // Check if file is an image
        if (!file.type.match('image.*')) {
            showMessage('Please select an image file', 'error');
            return;
        }
        
        selectedFile = file;
        
        // Update UI
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.innerHTML = `<img src="${e.target.result}" alt="Selected Image">`;
            processButton.disabled = false;
        };
        reader.readAsDataURL(file);
        
        // Reset text result
        textResult.innerHTML = '<p>Click "Process Image" to recognize text</p>';
        copyButton.disabled = true;
        downloadButton.disabled = true;
    }
    
    async function processImage() {
        if (!selectedFile) return;
        
        // Get selected OCR engine
        const selectedEngine = document.querySelector('input[name="ocrEngine"]:checked').value;
        
        // Show loading overlay
        loadingOverlay.style.display = 'flex';
        
        try {
            const formData = new FormData();
            formData.append('file', selectedFile);
            formData.append('ocr_engine', selectedEngine);
            
            const response = await fetch('/api/ocr', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'OCR processing failed');
            }
            
            const result = await response.json();
            
            // Update UI with result
            textResult.innerText = result.detected_text;
            copyButton.disabled = false;
            downloadButton.disabled = false;
            
            showMessage(`Text recognition completed using ${result.engine_used}!`, 'success');
        } catch (error) {
            console.error('Error:', error);
            showMessage(`Error: ${error.message}`, 'error');
            textResult.innerHTML = `<p>An error occurred while processing the image: ${error.message}</p>`;
        } finally {
            // Hide loading overlay
            loadingOverlay.style.display = 'none';
        }
    }
    
    function showMessage(message, type) {
        // Remove any existing message
        const existingMessage = document.querySelector('.message');
        if (existingMessage) {
            existingMessage.remove();
        }
        
        // Create new message
        const messageElement = document.createElement('div');
        messageElement.className = `message ${type}-message`;
        messageElement.textContent = message;
        
        // Add to DOM
        document.querySelector('.upload-section').appendChild(messageElement);
        
        // Remove after 3 seconds
        setTimeout(() => {
            messageElement.remove();
        }, 3000);
    }
});