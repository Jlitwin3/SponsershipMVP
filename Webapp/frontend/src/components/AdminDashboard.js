import React, { useState, useEffect } from 'react';
import './AdminDashboard.css';

const AdminDashboard = ({ onBack, onSignOut }) => {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [adminFiles, setAdminFiles] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [fileName, setFileName] = useState('');
  const [confirmFileName, setConfirmFileName] = useState('');
  const [deleteMessage, setDeleteMessage] = useState('');
  const [isDeleting, setIsDeleting] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortOrder, setSortOrder] = useState('desc'); // 'desc' or 'asc'
  const [filterType, setFilterType] = useState('All');

  // Whitelist management state (for regular users)
  const [whitelistedEmails, setWhitelistedEmails] = useState([]);
  const [newEmail, setNewEmail] = useState('');
  const [whitelistMessage, setWhitelistMessage] = useState('');
  const [isLoadingWhitelist, setIsLoadingWhitelist] = useState(true);
  const [isUpdatingWhitelist, setIsUpdatingWhitelist] = useState(false);

  // Admin list management state
  const [adminEmails, setAdminEmails] = useState([]);
  const [newAdminEmail, setNewAdminEmail] = useState('');
  const [adminMessage, setAdminMessage] = useState('');
  const [isLoadingAdmins, setIsLoadingAdmins] = useState(true);
  const [isUpdatingAdmins, setIsUpdatingAdmins] = useState(false);

  // Carousel state
  const [currentPage, setCurrentPage] = useState(0);
  const [touchStart, setTouchStart] = useState(null);
  const [touchEnd, setTouchEnd] = useState(null);

  // Fetch admin files, whitelist, and admin list on component mount
  useEffect(() => {
    fetchAdminFiles();
    fetchWhitelist();
    fetchAdminList();
  }, []);

  const fetchAdminFiles = async () => {
    try {
      const response = await fetch('/api/admin/files');
      const data = await response.json();
      if (data.files) {
        setAdminFiles(data.files);
      }
    } catch (error) {
      console.error('Error fetching admin files:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileSelect = (event) => {
    const files = Array.from(event.target.files);
    const fileObjects = files.map(file => ({
      name: file.name,
      size: `${(file.size / (1024 * 1024)).toFixed(2)} MB`,
      file: file
    }));
    setSelectedFiles(fileObjects);
    setUploadMessage('');
  };

  const handleUpload = async () => {
    if (selectedFiles.length === 0) {
      setUploadMessage('Please select files to upload');
      return;
    }

    setIsUploading(true);
    setUploadMessage('');

    try {
      const formData = new FormData();
      selectedFiles.forEach(fileObj => {
        formData.append('files', fileObj.file);
      });

      const response = await fetch('/api/admin/upload', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (response.ok) {
        setUploadMessage(`SUCCESS: ${data.message}`);
        setSelectedFiles([]);
        // Refresh file list
        fetchAdminFiles();
        // Clear file input
        document.getElementById('file-upload').value = '';
      } else {
        setUploadMessage(`ERROR: ${data.error || 'Upload failed'}`);
      }
    } catch (error) {
      console.error('Upload error:', error);
      setUploadMessage(`ERROR: ${error.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  const removeSelectedFile = (index) => {
    setSelectedFiles(selectedFiles.filter((_, i) => i !== index));
  };

  const handleDelete = async () => {
    if (!fileName.trim()) {
      setDeleteMessage('ERROR: Please enter a file name');
      return;
    }

    if (!confirmFileName.trim()) {
      setDeleteMessage('ERROR: Please confirm the file name');
      return;
    }

    if (fileName.trim() !== confirmFileName.trim()) {
      setDeleteMessage('ERROR: File names do not match');
      return;
    }

    setIsDeleting(true);
    setDeleteMessage('');

    try {
      const response = await fetch('/api/admin/delete', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          filename: fileName.trim(),
          confirmFilename: confirmFileName.trim()
        })
      });

      const data = await response.json();

      if (response.ok) {
        setDeleteMessage(`SUCCESS: ${data.message}`);
        setFileName('');
        setConfirmFileName('');
        // Refresh file list
        fetchAdminFiles();
      } else {
        setDeleteMessage(`ERROR: ${data.error || 'Deletion failed'}`);
      }
    } catch (error) {
      console.error('Delete error:', error);
      setDeleteMessage(`ERROR: ${error.message}`);
    } finally {
      setIsDeleting(false);
    }
  };

  const handleCancelDelete = () => {
    setFileName('');
    setConfirmFileName('');
    setDeleteMessage('');
  };

  // Fetch whitelist
  const fetchWhitelist = async () => {
    try {
      const response = await fetch('/api/admin/whitelist');
      const data = await response.json();
      if (data.emails) {
        setWhitelistedEmails(data.emails);
      }
    } catch (error) {
      console.error('Error fetching whitelist:', error);
      setWhitelistMessage('ERROR: Error loading whitelist');
    } finally {
      setIsLoadingWhitelist(false);
    }
  };

  // Add email to whitelist
  const handleAddEmail = async () => {
    if (!newEmail.trim()) {
      setWhitelistMessage('ERROR: Please enter an email address');
      return;
    }

    // Basic email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(newEmail.trim())) {
      setWhitelistMessage('ERROR: Please enter a valid email address');
      return;
    }

    setIsUpdatingWhitelist(true);
    setWhitelistMessage('');

    try {
      const response = await fetch('/api/admin/whitelist/add', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ email: newEmail.trim() })
      });

      const data = await response.json();

      if (response.ok) {
        setWhitelistMessage(`SUCCESS: ${data.message}`);
        setNewEmail('');
        fetchWhitelist(); // Refresh the list
      } else {
        setWhitelistMessage(`ERROR: ${data.error || 'Failed to add email'}`);
      }
    } catch (error) {
      console.error('Add email error:', error);
      setWhitelistMessage(`ERROR: ${error.message}`);
    } finally {
      setIsUpdatingWhitelist(false);
    }
  };

  // Remove email from whitelist
  const handleRemoveEmail = async (email) => {
    if (!window.confirm(`Are you sure you want to remove ${email} from the whitelist?`)) {
      return;
    }

    setIsUpdatingWhitelist(true);
    setWhitelistMessage('');

    try {
      const response = await fetch('/api/admin/whitelist/remove', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ email })
      });

      const data = await response.json();

      if (response.ok) {
        setWhitelistMessage(`SUCCESS: ${data.message}`);
        fetchWhitelist(); // Refresh the list
      } else {
        setWhitelistMessage(`ERROR: ${data.error || 'Failed to remove email'}`);
      }
    } catch (error) {
      console.error('Remove email error:', error);
      setWhitelistMessage(`ERROR: ${error.message}`);
    } finally {
      setIsUpdatingWhitelist(false);
    }
  };

  // Fetch admin list
  const fetchAdminList = async () => {
    try {
      const response = await fetch('/api/admin/adminlist');
      const data = await response.json();
      if (data.emails) {
        setAdminEmails(data.emails);
      }
    } catch (error) {
      console.error('Error fetching admin list:', error);
      setAdminMessage('ERROR: Error loading admin list');
    } finally {
      setIsLoadingAdmins(false);
    }
  };

  // Add email to admin list
  const handleAddAdmin = async () => {
    if (!newAdminEmail.trim()) {
      setAdminMessage('ERROR: Please enter an email address');
      return;
    }

    // Basic email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(newAdminEmail.trim())) {
      setAdminMessage('ERROR: Please enter a valid email address');
      return;
    }

    setIsUpdatingAdmins(true);
    setAdminMessage('');

    try {
      const response = await fetch('/api/admin/adminlist/add', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ email: newAdminEmail.trim() })
      });

      const data = await response.json();

      if (response.ok) {
        setAdminMessage(`SUCCESS: ${data.message}`);
        setNewAdminEmail('');
        fetchAdminList(); // Refresh the list
      } else {
        setAdminMessage(`ERROR: ${data.error || 'Failed to add admin'}`);
      }
    } catch (error) {
      console.error('Add admin error:', error);
      setAdminMessage(`ERROR: ${error.message}`);
    } finally {
      setIsUpdatingAdmins(false);
    }
  };

  // Remove email from admin list
  const handleRemoveAdmin = async (email) => {
    if (!window.confirm(`Are you sure you want to remove ${email} from the admin list?`)) {
      return;
    }

    setIsUpdatingAdmins(true);
    setAdminMessage('');

    try {
      const response = await fetch('/api/admin/adminlist/remove', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ email })
      });

      const data = await response.json();

      if (response.ok) {
        setAdminMessage(`SUCCESS: ${data.message}`);
        fetchAdminList(); // Refresh the list
      } else {
        setAdminMessage(`ERROR: ${data.error || 'Failed to remove admin'}`);
      }
    } catch (error) {
      console.error('Remove admin error:', error);
      setAdminMessage(`ERROR: ${error.message}`);
    } finally {
      setIsUpdatingAdmins(false);
    }
  };

  // Swipe handling
  const minSwipeDistance = 50;

  const onTouchStart = (e) => {
    setTouchEnd(null);
    setTouchStart(e.targetTouches[0].clientX);
  };

  const onTouchMove = (e) => {
    setTouchEnd(e.targetTouches[0].clientX);
  };

  const onTouchEnd = () => {
    if (!touchStart || !touchEnd) return;

    const distance = touchStart - touchEnd;
    const isLeftSwipe = distance > minSwipeDistance;
    const isRightSwipe = distance < -minSwipeDistance;

    if (isLeftSwipe && currentPage < 1) {
      setCurrentPage(1);
    } else if (isRightSwipe && currentPage > 0) {
      setCurrentPage(0);
    }
  };

  return (
    <div className="admin-dashboard">
      {/* Header like chatbot page */}
      <header className="admin-header">
        <h1>L'mu-Oa: Admin Dashboard</h1>
        <p>Manage your embedded documents and files</p>
        <button className="back-button" onClick={onBack}>← Back to Chat</button>
        <button className="signout-button" onClick={onSignOut}>Sign Out</button>
      </header>

      {/* Main Content */}
      <main className="admin-main">
        <div
          className="carousel-container"
          onTouchStart={onTouchStart}
          onTouchMove={onTouchMove}
          onTouchEnd={onTouchEnd}
        >
          <div
            className="carousel-track"
            style={{
              transform: `translateX(-${currentPage * 50}%)`,
              transition: 'transform 0.3s ease-in-out'
            }}
          >
            {/* Page 1: File Management */}
            <div className="carousel-page">
              <div className="admin-grid">
                {/* Left Column: View Embedded Files */}
                <section className="admin-section left-section">
                  <h2 className="section-title">View Embedded Files</h2>

                  {/* Stats Cards */}
                  <div className="stats-grid">
                    <div className="stat-card">
                      <div className="stat-content">
                        <p className="stat-label">Total Files</p>
                        <p className="stat-value">{isLoading ? '...' : adminFiles.length}</p>
                      </div>
                    </div>
                  </div>

                  {/* Search and Filter */}
                  <div className="table-controls">
                    <div className="search-bar">
                      <input
                        type="text"
                        placeholder="Search files..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                      />
                    </div>
                    <div className="filter-buttons">
                      <select
                        className="filter-btn"
                        value={filterType}
                        onChange={(e) => setFilterType(e.target.value)}
                        style={{ appearance: 'none', cursor: 'pointer' }}
                      >
                        <option value="All">All Types</option>
                        <option value="PDF">PDF</option>
                        <option value="IMAGE">Image</option>
                      </select>
                      <button
                        className="filter-btn"
                        onClick={() => setSortOrder(sortOrder === 'desc' ? 'asc' : 'desc')}
                      >
                        Sort: {sortOrder === 'desc' ? 'Newest' : 'Oldest'}
                      </button>
                    </div>
                  </div>

                  {/* Files Table */}
                  <div className="table-container">
                    <table className="files-table">
                      <thead>
                        <tr>
                          <th>File Name</th>
                          <th>Size</th>
                          <th>Type</th>
                          <th>Date Added</th>
                          <th>Actions</th>
                        </tr>
                      </thead>
                      <tbody>
                        {isLoading ? (
                          <tr>
                            <td colSpan="5" style={{ textAlign: 'center', padding: '2rem' }}>
                              Loading files...
                            </td>
                          </tr>
                        ) : (() => {
                          const filtered = adminFiles
                            .filter(file => {
                              const matchesSearch = file.name.toLowerCase().includes(searchQuery.toLowerCase());
                              const matchesType = filterType === 'All' || file.type.toUpperCase() === filterType.toUpperCase();
                              return matchesSearch && matchesType;
                            })
                            .sort((a, b) => {
                              const dateA = new Date(a.date || 0);
                              const dateB = new Date(b.date || 0);
                              return sortOrder === 'desc' ? dateB - dateA : dateA - dateB;
                            });

                          if (filtered.length === 0) {
                            return (
                              <tr>
                                <td colSpan="5" style={{ textAlign: 'center', padding: '2rem' }}>
                                  {adminFiles.length === 0
                                    ? "No admin files uploaded yet. Upload files using the form on the right."
                                    : "No files match your search/filter criteria."}
                                </td>
                              </tr>
                            );
                          }

                          return filtered.map((file, index) => (
                            <tr key={index}>
                              <td>
                                <div className="file-name-cell">
                                  <span>{file.name}</span>
                                </div>
                              </td>
                              <td>{file.size}</td>
                              <td>
                                <span className={`badge badge-${file.type.toLowerCase()}`}>
                                  {file.type}
                                </span>
                              </td>
                              <td>{file.date}</td>
                              <td>
                                <div className="action-buttons">
                                  <button
                                    className="action-btn delete-btn"
                                    onClick={() => {
                                      setFileName(file.name);
                                      setConfirmFileName(file.name);
                                      // Scroll to delete section or just trigger it?
                                      // Let's just set the state so they can click the big red button, 
                                      // or we can add a direct delete with confirmation.
                                      // For now, let's just make the big red button work better.
                                      window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
                                    }}
                                    style={{
                                      padding: '0.25rem 0.5rem',
                                      fontSize: '0.75rem',
                                      backgroundColor: '#fee2e2',
                                      color: '#dc2626',
                                      border: '1px solid #fecaca',
                                      borderRadius: '4px',
                                      cursor: 'pointer'
                                    }}
                                  >
                                    Remove
                                  </button>
                                </div>
                              </td>
                            </tr>
                          ));
                        })()}
                      </tbody>
                    </table>
                  </div>
                </section>

                {/* Right Column */}
                <div className="right-column">
                  {/* C. Upload Files Section */}
                  <section className="admin-section right-section">
                    <h2 className="section-title">Upload Files</h2>
                    <div className="upload-area">
                      <div className="upload-dropzone">
                        <h3>Drag and drop files here</h3>
                        <p className="upload-instructions">or click to select files</p>
                        <input
                          type="file"
                          id="file-upload"
                          accept=".pdf,.jpg,.jpeg,image/jpeg,application/pdf"
                          multiple
                          onChange={handleFileSelect}
                          style={{ display: 'none' }}
                        />
                        <label htmlFor="file-upload" className="btn btn-primary">
                          Select Files
                        </label>
                        <p className="upload-info">PDF, JPEG only • Max 50MB per file</p>
                      </div>

                      {/* Selected Files List */}
                      {selectedFiles.length > 0 && (
                        <div className="selected-files">
                          <h4>Selected Files ({selectedFiles.length})</h4>
                          <div className="selected-files-list">
                            {selectedFiles.map((file, index) => (
                              <div key={index} className="selected-file-item">
                                <div className="file-info">
                                  <p className="file-name">{file.name}</p>
                                  <p className="file-size">{file.size}</p>
                                </div>
                                <button
                                  className="remove-file-btn"
                                  onClick={() => removeSelectedFile(index)}
                                >
                                  ×
                                </button>
                              </div>
                            ))}
                          </div>
                          <button
                            className="btn btn-primary"
                            onClick={handleUpload}
                            disabled={isUploading}
                            style={{ width: '100%', marginTop: '1rem' }}
                          >
                            {isUploading ? 'Uploading...' : 'Upload to ChromaDB'}
                          </button>
                        </div>
                      )}

                      {/* Upload Message */}
                      {uploadMessage && (
                        <div style={{
                          marginTop: '1rem',
                          padding: '0.75rem',
                          borderRadius: '8px',
                          backgroundColor: uploadMessage.includes('SUCCESS:') ? '#f0fdf4' : '#fef2f2',
                          color: uploadMessage.includes('SUCCESS:') ? '#166534' : '#991b1b',
                          fontSize: '0.875rem'
                        }}>
                          {uploadMessage}
                        </div>
                      )}
                    </div>
                  </section>

                  {/* B. Remove Files Section */}
                  <section className="admin-section right-section">
                    <h2 className="section-title">Remove Files</h2>
                    <div className="remove-form">
                      <div className="form-group">
                        <label htmlFor="fileName">File Name</label>
                        <input
                          type="text"
                          id="fileName"
                          placeholder="Enter the file name to remove"
                          className="form-input"
                          value={fileName}
                          onChange={(e) => setFileName(e.target.value)}
                        />
                        <p className="helper-text">Enter the exact name of the file you want to remove</p>
                      </div>
                      <div className="form-group">
                        <label htmlFor="confirmFileName">Confirm File Name</label>
                        <input
                          type="text"
                          id="confirmFileName"
                          placeholder="Re-enter the file name"
                          className="form-input"
                          value={confirmFileName}
                          onChange={(e) => setConfirmFileName(e.target.value)}
                        />
                        <p className="helper-text">Re-enter the file name to confirm deletion</p>
                      </div>
                      <div className="form-actions">
                        <button
                          className="btn btn-danger"
                          onClick={handleDelete}
                          disabled={isDeleting}
                        >
                          {isDeleting ? 'Deleting...' : 'Remove File Permanently'}
                        </button>
                        <button
                          className="btn btn-secondary"
                          onClick={handleCancelDelete}
                          disabled={isDeleting}
                        >
                          Cancel
                        </button>
                      </div>

                      {/* Delete Message */}
                      {deleteMessage && (
                        <div style={{
                          marginTop: '1rem',
                          padding: '0.75rem',
                          borderRadius: '8px',
                          backgroundColor: deleteMessage.includes('SUCCESS:') ? '#f0fdf4' : '#fef2f2',
                          color: deleteMessage.includes('SUCCESS:') ? '#166534' : '#991b1b',
                          fontSize: '0.875rem'
                        }}>
                          {deleteMessage}
                        </div>
                      )}
                    </div>
                  </section>
                </div>
              </div>
            </div>

            {/* Page 2: User & Admin Management */}
            <div className="carousel-page">
              <div className="access-management-grid">
                {/* Left: User Access Management */}
                <section className="admin-section access-section">
                  <h2 className="section-title">User Access Management</h2>
                  <p className="section-subtitle">
                    Add or remove users from the whitelist to control access to the chatbot.
                  </p>

                  {/* Add Email Form */}
                  <div className="whitelist-form">
                    <div className="form-group">
                      <label htmlFor="newEmail">Add Email to Whitelist</label>
                      <div style={{ display: 'flex', gap: '0.75rem' }}>
                        <input
                          type="email"
                          id="newEmail"
                          placeholder="user@example.com"
                          className="form-input"
                          value={newEmail}
                          onChange={(e) => setNewEmail(e.target.value)}
                          onKeyPress={(e) => e.key === 'Enter' && handleAddEmail()}
                          style={{ flex: 1 }}
                        />
                        <button
                          className="btn btn-primary"
                          onClick={handleAddEmail}
                          disabled={isUpdatingWhitelist}
                          style={{ margin: 0, whiteSpace: 'nowrap' }}
                        >
                          {isUpdatingWhitelist ? 'Adding...' : 'Add Email'}
                        </button>
                      </div>
                      <p className="helper-text">Enter email address to grant chatbot access</p>
                    </div>

                    {/* Whitelist Message */}
                    {whitelistMessage && (
                      <div style={{
                        marginBottom: '1rem',
                        padding: '0.75rem',
                        borderRadius: '8px',
                        backgroundColor: whitelistMessage.includes('SUCCESS:') ? '#f0fdf4' : '#fef2f2',
                        color: whitelistMessage.includes('SUCCESS:') ? '#166534' : '#991b1b',
                        fontSize: '0.875rem'
                      }}>
                        {whitelistMessage}
                      </div>
                    )}

                    {/* Current Whitelist */}
                    <div className="whitelist-display">
                      <h4 style={{ fontSize: '0.95rem', fontWeight: 600, color: '#1e293b', margin: '0 0 0.75rem 0' }}>
                        Whitelisted Users ({isLoadingWhitelist ? '...' : whitelistedEmails.length})
                      </h4>

                      {isLoadingWhitelist ? (
                        <div style={{ textAlign: 'center', padding: '1rem', color: '#64748b' }}>
                          Loading whitelist...
                        </div>
                      ) : whitelistedEmails.length === 0 ? (
                        <div style={{ textAlign: 'center', padding: '1rem', color: '#64748b' }}>
                          No users whitelisted yet
                        </div>
                      ) : (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', maxHeight: '400px', overflowY: 'auto' }}>
                          {whitelistedEmails.map((email, index) => (
                            <div
                              key={index}
                              style={{
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'space-between',
                                padding: '0.75rem',
                                background: '#f8fafc',
                                border: '1px solid #e2e8f0',
                                borderRadius: '8px'
                              }}
                            >
                              <span style={{ fontSize: '0.875rem', color: '#1e293b', fontWeight: 500 }}>
                                {email}
                              </span>
                              <button
                                onClick={() => handleRemoveEmail(email)}
                                disabled={isUpdatingWhitelist}
                                style={{
                                  padding: '0.375rem 0.75rem',
                                  background: '#fee2e2',
                                  color: '#dc2626',
                                  border: 'none',
                                  borderRadius: '6px',
                                  fontSize: '0.75rem',
                                  fontWeight: 600,
                                  cursor: 'pointer',
                                  transition: 'all 0.2s ease'
                                }}
                                onMouseEnter={(e) => e.target.style.background = '#fecaca'}
                                onMouseLeave={(e) => e.target.style.background = '#fee2e2'}
                              >
                                Remove
                              </button>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                </section>

                {/* Right: Admin Management */}
                <section className="admin-section access-section">
                  <h2 className="section-title">Admin Management</h2>
                  <p className="section-subtitle">
                    Manage admin privileges. Only admins listed here can access this dashboard.
                  </p>

                  {/* Add Admin Form */}
                  <div className="whitelist-form">
                    <div className="form-group">
                      <label htmlFor="newAdminEmail">Add Admin</label>
                      <div style={{ display: 'flex', gap: '0.75rem' }}>
                        <input
                          type="email"
                          id="newAdminEmail"
                          placeholder="admin@example.com"
                          className="form-input"
                          value={newAdminEmail}
                          onChange={(e) => setNewAdminEmail(e.target.value)}
                          onKeyPress={(e) => e.key === 'Enter' && handleAddAdmin()}
                          style={{ flex: 1 }}
                        />
                        <button
                          className="btn btn-primary"
                          onClick={handleAddAdmin}
                          disabled={isUpdatingAdmins}
                          style={{ margin: 0, whiteSpace: 'nowrap' }}
                        >
                          {isUpdatingAdmins ? 'Adding...' : 'Add Admin'}
                        </button>
                      </div>
                      <p className="helper-text">Enter email address to grant admin dashboard access</p>
                    </div>

                    {/* Admin Message */}
                    {adminMessage && (
                      <div style={{
                        marginBottom: '1rem',
                        padding: '0.75rem',
                        borderRadius: '8px',
                        backgroundColor: adminMessage.includes('SUCCESS:') ? '#f0fdf4' : '#fef2f2',
                        color: adminMessage.includes('SUCCESS:') ? '#166534' : '#991b1b',
                        fontSize: '0.875rem'
                      }}>
                        {adminMessage}
                      </div>
                    )}

                    {/* Current Admins */}
                    <div className="whitelist-display">
                      <h4 style={{ fontSize: '0.95rem', fontWeight: 600, color: '#1e293b', margin: '0 0 0.75rem 0' }}>
                        Current Admins ({isLoadingAdmins ? '...' : adminEmails.length})
                      </h4>

                      {isLoadingAdmins ? (
                        <div style={{ textAlign: 'center', padding: '1rem', color: '#64748b' }}>
                          Loading admins...
                        </div>
                      ) : adminEmails.length === 0 ? (
                        <div style={{ textAlign: 'center', padding: '1rem', color: '#64748b' }}>
                          No admins configured
                        </div>
                      ) : (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', maxHeight: '400px', overflowY: 'auto' }}>
                          {adminEmails.map((email, index) => (
                            <div
                              key={index}
                              style={{
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'space-between',
                                padding: '0.75rem',
                                background: '#f0fdf4',
                                border: '1px solid #bbf7d0',
                                borderRadius: '8px'
                              }}
                            >
                              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                <span style={{ fontSize: '0.875rem', color: '#1e293b', fontWeight: 500 }}>
                                  {email}
                                </span>
                                <span style={{
                                  fontSize: '0.65rem',
                                  fontWeight: 600,
                                  padding: '0.25rem 0.5rem',
                                  background: '#166534',
                                  color: 'white',
                                  borderRadius: '4px',
                                  textTransform: 'uppercase'
                                }}>
                                  Admin
                                </span>
                              </div>
                              <button
                                onClick={() => handleRemoveAdmin(email)}
                                disabled={isUpdatingAdmins}
                                style={{
                                  padding: '0.375rem 0.75rem',
                                  background: '#fee2e2',
                                  color: '#dc2626',
                                  border: 'none',
                                  borderRadius: '6px',
                                  fontSize: '0.75rem',
                                  fontWeight: 600,
                                  cursor: 'pointer',
                                  transition: 'all 0.2s ease'
                                }}
                                onMouseEnter={(e) => e.target.style.background = '#fecaca'}
                                onMouseLeave={(e) => e.target.style.background = '#fee2e2'}
                              >
                                Remove
                              </button>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                </section>
              </div>
            </div>
          </div>
        </div>

        {/* Navigation Dots */}
        <div className="carousel-dots">
          <button
            className={`dot ${currentPage === 0 ? 'active' : ''}`}
            onClick={() => setCurrentPage(0)}
            aria-label="Go to page 1"
          />
          <button
            className={`dot ${currentPage === 1 ? 'active' : ''}`}
            onClick={() => setCurrentPage(1)}
            aria-label="Go to page 2"
          />
        </div>
      </main>
    </div>
  );
};

export default AdminDashboard;
