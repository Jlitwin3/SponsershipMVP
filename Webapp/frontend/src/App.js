import React, { useState, useEffect } from 'react';
import './App.css';
import ChatInterface from './components/ChatInterface';

function App() {
  const [isReady, setIsReady] = useState(false);
  const [isProcessing, setIsProcessing] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Poll for status until documents are ready
    const checkStatus = async () => {
      try {
        const response = await fetch('http://localhost:5001/api/status');
        const data = await response.json();

        if (data.processed) {
          setIsReady(true);
          setIsProcessing(false);
        } else if (data.is_processing) {
          setIsProcessing(true);
          // Keep polling every 2 seconds while processing
          setTimeout(checkStatus, 2000);
        } else {
          setError('No documents found or processing failed');
          setIsProcessing(false);
        }
      } catch (err) {
        console.error('Failed to check status:', err);
        setError('Failed to connect to server');
        setIsProcessing(false);
      }
    };

    checkStatus();
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <h1>L'mu-Oa: AI Sponsorship Assistant</h1>
        <p>Discover. Analyze. Propose. All with L'mu-Oa</p>
      </header>

      <main className="App-main">
        {isProcessing ? (
          <div className="loading-container">
            <div className="spinner"></div>
            <h2>Processing Documents...</h2>
            <p>Loading PDFs and images from Dropbox. This may take a minute.</p>
          </div>
        ) : isReady ? (
          <ChatInterface />
        ) : (
          <div className="error-container">
            <h2>Error</h2>
            <p>{error || 'Failed to load documents'}</p>
          </div>
        )}
      </main>

      {error && isReady && (
        <div className="error-message">
          <span>❌ {error}</span>
        </div>
      )}
    </div>
  );
}

export default App;
