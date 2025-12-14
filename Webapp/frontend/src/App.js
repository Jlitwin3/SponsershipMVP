import React, { useState, useEffect } from 'react';
import { MsalProvider } from '@azure/msal-react';
import { PublicClientApplication } from '@azure/msal-browser';
import './App.css';
import ChatInterface from './components/ChatInterface';
import AdminDashboard from './components/AdminDashboard';
import AdminLogin from './components/AdminLogin';
import ChatbotLogin from './components/ChatbotLogin';

// MSAL configuration
const msalConfig = {
  auth: {
    clientId: process.env.REACT_APP_MICROSOFT_CLIENT_ID || '',
    authority: `https://login.microsoftonline.com/${process.env.REACT_APP_MICROSOFT_TENANT_ID || 'common'}`,
    redirectUri: window.location.origin,
  },
  cache: {
    cacheLocation: 'localStorage',
    storeAuthStateInCookie: false,
  }
};

const msalInstance = new PublicClientApplication(msalConfig);

function App() {
  const [isReady, setIsReady] = useState(false);
  const [isProcessing, setIsProcessing] = useState(true);
  const [error, setError] = useState(null);
  const [showAdmin, setShowAdmin] = useState(false);
  const [showAdminLogin, setShowAdminLogin] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [userEmail, setUserEmail] = useState(null);

  // Load authentication state from localStorage on mount
  useEffect(() => {
    const storedEmail = localStorage.getItem('userEmail');
    if (storedEmail) {
      setIsAuthenticated(true);
      setUserEmail(storedEmail);
    }
  }, []);

  useEffect(() => {
    // Poll for status until documents are ready
    const checkStatus = async () => {
      try {
        const response = await fetch('/api/status');
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

  // Handle chatbot authentication
  const handleChatbotLogin = (email) => {
    setIsAuthenticated(true);
    setUserEmail(email);
    localStorage.setItem('userEmail', email);
  };

  // Handle sign out
  const handleSignOut = () => {
    setIsAuthenticated(false);
    setUserEmail(null);
    setShowAdmin(false);
    localStorage.removeItem('userEmail');
  };

  // Handle admin dashboard
  if (showAdmin) {
    return <AdminDashboard onBack={() => setShowAdmin(false)} onSignOut={handleSignOut} />;
  }

  // Handle admin button click - check if user is already admin
  const handleAdminClick = async () => {
    if (!userEmail) {
      setShowAdminLogin(true);
      return;
    }

    // Check if the logged-in user is an admin
    try {
      const response = await fetch('/api/admin/verify-email', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email: userEmail }),
      });

      const data = await response.json();

      if (response.ok && data.authorized) {
        // User is admin - grant access directly
        setShowAdmin(true);
      } else {
        // User is not admin - show error message
        alert('Access Denied\n\nYou do not have permission to access the admin dashboard. Please contact an administrator if you need admin access.');
      }
    } catch (error) {
      console.error('Admin check error:', error);
      alert('Error checking admin access. Please try again.');
    }
  };

  // Show login screen if not authenticated
  if (!isAuthenticated) {
    return <ChatbotLogin onLoginSuccess={handleChatbotLogin} />;
  }

  return (
    <div className="App">
      <header className="App-header">
        <h1>L'mu-Oa: AI Sponsorship Assistant</h1>
        <p>Discover. Analyze. Propose. All with L'mu-Oa</p>
        <button className="admin-button" onClick={handleAdminClick}>Admin</button>
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
          <span style={{ fontSize: '0.9rem', color: 'rgba(255, 255, 255, 0.8)' }}>
            {userEmail}
          </span>
        </div>
        <button className="signout-button-main" onClick={handleSignOut}>Sign Out</button>
      </header>

      {showAdminLogin && (
        <AdminLogin
          onLoginSuccess={() => {
            setShowAdminLogin(false);
            setShowAdmin(true);
          }}
          onCancel={() => setShowAdminLogin(false)}
        />
      )}

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

// Wrap App with MsalProvider
const AppWithMsal = () => (
  <MsalProvider instance={msalInstance}>
    <App />
  </MsalProvider>
);

export default AppWithMsal;
