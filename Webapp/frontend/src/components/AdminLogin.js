import React, { useState } from 'react';
import { useMsal } from '@azure/msal-react';
import './AdminLogin.css';

const AdminLogin = ({ onLoginSuccess, onCancel }) => {
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
      const response = await fetch('/api/admin/verify-email', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email: userEmail }),
      });

      const data = await response.json();

      if (response.ok && data.authorized) {
        // Email is authorized - grant access
        onLoginSuccess();
      } else {
        // Email not authorized - show error
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
        setErrorMessage('Access denied. Your email is not authorized for admin access. Please contact an administrator.');
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
    <div className="admin-login-overlay">
      <div className="admin-login-container">
        <button className="close-button" onClick={onCancel}>×</button>

        <h2>Admin Access</h2>
        <p className="login-subtitle">Sign in with Microsoft to access the admin dashboard</p>

        {isLoading ? (
          <div style={{ textAlign: 'center', padding: '1rem' }}>
            <div className="spinner" style={{
              border: '3px solid #f3f3f3',
              borderTop: '3px solid #1a4d2e',
              borderRadius: '50%',
              width: '40px',
              height: '40px',
              animation: 'spin 1s linear infinite',
              margin: '0 auto'
            }}></div>
            <p style={{ marginTop: '1rem', color: '#64748b' }}>Verifying access...</p>
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
              <div style={{
                marginTop: '1.5rem',
                padding: '0.875rem',
                borderRadius: '10px',
                backgroundColor: '#fef2f2',
                border: '1px solid #fecaca',
                color: '#991b1b',
                fontSize: '0.9rem',
                textAlign: 'center',
                lineHeight: '1.5'
              }}>
                {errorMessage}
              </div>
            )}

            <div className="divider">
              <span>or</span>
            </div>

            <button className="override-btn" onClick={onLoginSuccess}>
              Override (Dev Mode)
            </button>

            <p className="login-info">
              This override button is for development purposes only and will be removed in production.
            </p>
          </>
        )}
      </div>
    </div>
  );
};

export default AdminLogin;
