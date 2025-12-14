import React, { useState } from 'react';
import { useMsal } from '@azure/msal-react';
import './ChatbotLogin.css';

const ChatbotLogin = ({ onLoginSuccess }) => {
  const [errorMessage, setErrorMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const { instance } = useMsal();

  const handleMicrosoftLogin = async () => {
    setIsLoading(true);
    setErrorMessage('');

    try {
      // Request popup login
      const loginResponse = await instance.loginPopup({
        scopes: ['User.Read'],
        prompt: 'select_account'
      });

      const userEmail = loginResponse.account.username;

      // Verify email with backend
      const response = await fetch('/api/chatbot/verify-email', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email: userEmail }),
      });

      const data = await response.json();

      if (response.ok && data.authorized) {
        // Email is whitelisted - grant access
        onLoginSuccess(userEmail);
      } else {
        // Email not whitelisted - show error
        // Silently clear the account without showing Microsoft logout popup
        const accounts = instance.getAllAccounts();
        if (accounts.length > 0) {
          await instance.logoutPopup({
            account: accounts[0],
            postLogoutRedirectUri: window.location.origin
          }).catch(() => {
            // If popup logout fails, just clear local cache
            instance.clearCache();
          });
        }
        setErrorMessage('Access denied. Your email is not authorized to access this chatbot. Please contact your administrator for access.');
      }
    } catch (error) {
      console.error('Login error:', error);
      if (error.errorCode === 'user_cancelled') {
        setErrorMessage('Login was cancelled. Please try again.');
      } else {
        setErrorMessage('Login failed. Please try again.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chatbot-login-container">
      <div className="chatbot-login-card">
        <div className="login-icon">🤖</div>
        <h2>Welcome to L'mu-Oa</h2>
        <p className="login-description">
          Your AI-powered sponsorship assistant. Sign in with your authorized Microsoft account to get started.
        </p>

        {isLoading ? (
          <div className="loading-state">
            <div className="spinner" style={{
              border: '3px solid #f3f3f3',
              borderTop: '3px solid #1a4d2e',
              borderRadius: '50%',
              width: '40px',
              height: '40px',
              animation: 'spin 1s linear infinite',
              margin: '0 auto'
            }}></div>
            <p style={{ marginTop: '1rem', color: '#64748b' }}>Verifying your access...</p>
          </div>
        ) : (
          <>
            <button className="microsoft-login-btn" onClick={handleMicrosoftLogin}>
              <svg className="microsoft-icon" viewBox="0 0 23 23">
                <path fill="#f3f3f3" d="M0 0h23v23H0z"/>
                <path fill="#f35325" d="M1 1h10v10H1z"/>
                <path fill="#81bc06" d="M12 1h10v10H12z"/>
                <path fill="#05a6f0" d="M1 12h10v10H1z"/>
                <path fill="#ffba08" d="M12 12h10v10H12z"/>
              </svg>
              Sign in with Microsoft
            </button>

            {errorMessage && (
              <div className="error-message-box">
                {errorMessage}
              </div>
            )}

            <div className="divider">
              <span>or</span>
            </div>

            <button className="bypass-btn" onClick={() => onLoginSuccess('dev@example.com')}>
              Bypass (Dev Mode)
            </button>

            <p className="bypass-info">
              This bypass button is for development purposes only and will be removed in production.
            </p>

            <div className="login-footer">
              <p>Only authorized users can access this chatbot.</p>
              <p>Need access? Contact your administrator.</p>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default ChatbotLogin;
