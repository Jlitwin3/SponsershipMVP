import React, { useState, useRef } from 'react';
import './FileUpload.css';

const FileUpload = ({ onFilesProcessed, onError, loading, setLoading }) => {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [processing, setProcessing] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const files = Array.from(e.target.files);
    setSelectedFiles(files);
  };

  const handleRemoveFile = (index) => {
    setSelectedFiles(selectedFiles.filter((_, i) => i !== index));
  };

  const handleUpload = async () => {
    if (selectedFiles.length === 0) {
      onError('Please select at least one PDF file');
      return;
    }

    setProcessing(true);
    setLoading(true);

    const formData = new FormData();
    selectedFiles.forEach(file => {
      formData.append('files', file);
    });

    try {
      const response = await fetch('http://localhost:5000/api/upload', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (response.ok) {
        onFilesProcessed();
        setSelectedFiles([]);
      } else {
        onError(data.error || 'Failed to upload files');
      }
    } catch (error) {
      onError('Network error: ' + error.message);
    } finally {
      setProcessing(false);
      setLoading(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const files = Array.from(e.dataTransfer.files).filter(
      file => file.type === 'application/pdf'
    );
    setSelectedFiles([...selectedFiles, ...files]);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  return (
    <div className="file-upload-container">
      <div 
        className="drop-zone"
        onDrop={handleDrop}
        onDragOver={handleDragOver}
      >
        <div className="drop-zone-content">
          <div className="upload-icon">📄</div>
          <h2>Upload PDF Files</h2>
          <p>Drag and drop PDF files here, or click to select</p>
          <button
            className="select-button"
            onClick={() => fileInputRef.current.click()}
            disabled={processing}
          >
            Select Files
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf"
            multiple
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />
        </div>
      </div>

      {selectedFiles.length > 0 && (
        <div className="selected-files">
          <h3>Selected Files ({selectedFiles.length})</h3>
          <div className="file-list">
            {selectedFiles.map((file, index) => (
              <div key={index} className="file-item">
                <span className="file-name">{file.name}</span>
                <span className="file-size">{formatFileSize(file.size)}</span>
                <button
                  className="remove-button"
                  onClick={() => handleRemoveFile(index)}
                  disabled={processing}
                >
                  ✕
                </button>
              </div>
            ))}
          </div>
          <button
            className="upload-button"
            onClick={handleUpload}
            disabled={processing}
          >
            {processing ? 'Processing...' : 'Process PDFs'}
          </button>
        </div>
      )}

      {processing && (
        <div className="processing-overlay">
          <div className="spinner"></div>
          <p>Processing your PDFs...</p>
        </div>
      )}
    </div>
  );
};

export default FileUpload;

