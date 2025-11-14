import React, { useState, useRef, useEffect } from 'react';
import { jsPDF } from 'jspdf';
import './ChatInterface.css';

const ChatInterface = () => {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Hello! I\'m ready to answer questions about sports sponsorship research. Ask me anything about the papers I\'ve analyzed.',
      timestamp: new Date()
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = {
      role: 'user',
      content: inputMessage,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:5001/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ query: inputMessage })
      });

      const data = await response.json();

      if (response.ok) {
        const assistantMessage = {
          role: 'assistant',
          content: data.answer,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, assistantMessage]);
      } else {
        const errorMessage = {
          role: 'assistant',
          content: `Error: ${data.error || 'Failed to get response'}`,
          timestamp: new Date(),
          isError: true
        };
        setMessages(prev => [...prev, errorMessage]);
      }
    } catch (error) {
      const errorMessage = {
        role: 'assistant',
        content: `Network error: ${error.message}`,
        timestamp: new Date(),
        isError: true
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const formatTime = (date) => {
    return new Date(date).toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const downloadChatAsPDF = () => {
    const doc = new jsPDF();
    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();
    const margin = 20;
    const maxWidth = pageWidth - (2 * margin);
    let yPosition = margin;

    // Add title
    doc.setFontSize(18);
    doc.setFont(undefined, 'bold');
    doc.text('Chat Conversation', margin, yPosition);
    yPosition += 10;

    // Add date
    doc.setFontSize(10);
    doc.setFont(undefined, 'normal');
    doc.text(`Generated: ${new Date().toLocaleString()}`, margin, yPosition);
    yPosition += 15;

    // Add messages
    messages.forEach((message, index) => {
      // Check if we need a new page
      if (yPosition > pageHeight - 40) {
        doc.addPage();
        yPosition = margin;
      }

      // Add role and timestamp
      doc.setFontSize(11);
      doc.setFont(undefined, 'bold');
      const roleText = message.role === 'user' ? 'You' : 'Assistant';
      const timeText = formatTime(message.timestamp);
      doc.text(`${roleText} (${timeText}):`, margin, yPosition);
      yPosition += 7;

      // Add message content
      doc.setFontSize(10);
      doc.setFont(undefined, 'normal');
      const lines = doc.splitTextToSize(message.content, maxWidth);

      lines.forEach((line) => {
        // Check if we need a new page
        if (yPosition > pageHeight - 20) {
          doc.addPage();
          yPosition = margin;
        }
        doc.text(line, margin, yPosition);
        yPosition += 6;
      });

      yPosition += 8; // Space between messages
    });

    // Save the PDF
    const filename = `chat-conversation-${new Date().toISOString().slice(0, 10)}.pdf`;
    doc.save(filename);
  };

  return (
    <div className="chat-interface">
      <div className="chat-header">
        <h2>Chat with Sports Sponsorship Research</h2>
      </div>

      <div className="messages-container">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`message ${message.role} ${message.isError ? 'error' : ''}`}
          >
            <div className="message-content">
              {message.role === 'assistant' && (
                <div className="message-avatar">🏆</div>
              )}
              {message.role === 'user' && (
                <div className="message-avatar">👤</div>
              )}
              <div className="message-text">
                {message.content.split('\n').map((line, i) => (
                  <span key={i}>
                    {line}
                    {i < message.content.split('\n').length - 1 && <br />}
                  </span>
                ))}
              </div>
            </div>
            <div className="message-timestamp">
              {formatTime(message.timestamp)}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="message assistant">
            <div className="message-content">
              <div className="message-avatar">🏆</div>
              <div className="message-text loading">
                <span className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="input-container">
        <textarea
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type your question here... (Press Enter to send)"
          rows="2"
          disabled={isLoading}
        />
        <button
          onClick={handleSendMessage}
          disabled={!inputMessage.trim() || isLoading}
          className="send-button"
        >
          Send
        </button>
      </div>

      <button
        onClick={downloadChatAsPDF}
        className="download-pdf-button"
        title="Download chat as PDF"
        disabled={messages.length <= 1}
      >
        <span className="download-icon">📄</span>
      </button>
    </div>
  );
};

export default ChatInterface;
