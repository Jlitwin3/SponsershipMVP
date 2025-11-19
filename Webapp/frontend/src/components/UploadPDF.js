import React, { useState } from 'react';
import './UploadPDF.css';

function UploadPDF({ onClose }) {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState('');

  const handleFileSelect = (event) => {
    const files = Array.from(event.target.files);
    const pdfFiles = files.filter(file => file.type === 'application/pdf');

    if (pdfFiles.length !== files.length) {
      setMessage('Only PDF files are allowed');
      setTimeout(() => setMessage(''), 3000);
    }

    setSelectedFiles(pdfFiles);
  };

  const handleUpload = async () => {
    if (selectedFiles.length === 0) {
      setMessage('Please select at least one PDF file');
      return;
    }

    setUploading(true);
    setMessage('');

    const formData = new FormData();
    selectedFiles.forEach(file => {
      formData.append('pdfs', file);
    });

    try {
      const response = await fetch('http://localhost:5001/api/upload-temp', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        setMessage(`✅ Successfully uploaded ${selectedFiles.length} PDF(s). Processing complete!`);
        setSelectedFiles([]);
        setTimeout(() => {
          setMessage('');
          if (onClose) onClose();
        }, 2000);
      } else {
        setMessage(`❌ Error: ${data.error || 'Upload failed'}`);
      }
    } catch (error) {
      setMessage(`❌ Error: ${error.message}`);
    } finally {
      setUploading(false);
    }
  };

  const removeFile = (index) => {
    setSelectedFiles(files => files.filter((_, i) => i !== index));
  };

  return (
    <div className="upload-pdf-container">
      <div className="upload-header">
        <h2>Upload Temporary PDFs</h2>
        <button className="close-button" onClick={onClose}>&times;</button>
      </div>

      <div className="upload-info">
        <p>Upload PDFs for temporary processing. Files will be cleared when you close the tab.</p>
        <p className="warning">⚠️ These files will NOT be saved to the database.</p>
      </div>

      <div className="upload-area">
        <input
          type="file"
          id="pdf-input"
          multiple
          accept=".pdf"
          onChange={handleFileSelect}
          style={{ display: 'none' }}
        />
        <label htmlFor="pdf-input" className="file-select-button">
          Choose PDF Files
        </label>
      </div>

      {selectedFiles.length > 0 && (
        <div className="selected-files">
          <h3>Selected Files ({selectedFiles.length})</h3>
          <ul>
            {selectedFiles.map((file, index) => (
              <li key={index}>
                <span>{file.name}</span>
                <span className="file-size">({(file.size / 1024).toFixed(2)} KB)</span>
                <button
                  className="remove-file-button"
                  onClick={() => removeFile(index)}
                >
                  Remove
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}

      <button
        className="upload-button"
        onClick={handleUpload}
        disabled={uploading || selectedFiles.length === 0}
      >
        {uploading ? 'Uploading...' : 'Upload & Process'}
      </button>

      {message && (
        <div className={`upload-message ${message.includes('✅') ? 'success' : 'error'}`}>
          {message}
        </div>
      )}
    </div>
  );
}

export default UploadPDF;
