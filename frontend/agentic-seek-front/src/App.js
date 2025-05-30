import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import axios from 'axios';
import './App.css';
import { colors } from './colors';
import Auth from './Auth';

function App() {
    const [isAuthenticated, setIsAuthenticated] = useState(false);
    const [query, setQuery] = useState('');
    const [messages, setMessages] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [isOnline, setIsOnline] = useState(false);
    const [status, setStatus] = useState('Agents ready');
    const [expandedReasoning, setExpandedReasoning] = useState(new Set());
    const [uploadedFile, setUploadedFile] = useState(null);
    const [fileContent, setFileContent] = useState(null);
    const [filePreview, setFilePreview] = useState(null);
    const messagesEndRef = useRef(null);
    const fileInputRef = useRef(null);

    // Check for existing token on mount
    useEffect(() => {
        const token = localStorage.getItem('yappy_token');
        const apiKey = localStorage.getItem('yappy_api_key');
        const llmModel = localStorage.getItem('yappy_llm_model');
        
        // Only authenticate if user has token AND API key
        if (token && apiKey && llmModel) {
            axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
            setIsAuthenticated(true);
        } else {
            // Clear incomplete session data
            localStorage.removeItem('yappy_token');
            localStorage.removeItem('yappy_llm_model');
            localStorage.removeItem('yappy_api_key');
        }
    }, []);

    useEffect(() => {
        if (isAuthenticated) {
            const intervalId = setInterval(() => {
                checkHealth();
                fetchLatestAnswer();
            }, 3000);
            return () => clearInterval(intervalId);
        }
    }, [messages, isAuthenticated]);

    const handleLogin = (token, llmSettings) => {
        setIsAuthenticated(true);
        // Store LLM settings if provided
        if (llmSettings) {
            localStorage.setItem('yappy_llm_model', llmSettings.model);
            localStorage.setItem('yappy_api_key', llmSettings.apiKey);
        }
    };

    const handleLogout = () => {
        localStorage.removeItem('yappy_token');
        localStorage.removeItem('yappy_llm_model');
        localStorage.removeItem('yappy_api_key');
        delete axios.defaults.headers.common['Authorization'];
        setIsAuthenticated(false);
        setMessages([]);
    };

    const checkHealth = async () => {
        try {
            const baseURL = process.env.NODE_ENV === 'production' ? '' : 'http://127.0.0.1:8000';
            await axios.get(`${baseURL}/health`);
            setIsOnline(true);
            console.log('System is online');
        } catch {
            setIsOnline(false);
            console.log('System is offline');
        }
    };


    const normalizeAnswer = (answer) => {
        return answer
            .trim()
            .toLowerCase()
            .replace(/\s+/g, ' ')
            .replace(/[.,!?]/g, '')
    };

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    // Auto-scroll when messages change
    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const toggleReasoning = (messageIndex) => {
        setExpandedReasoning(prev => {
            const newSet = new Set(prev);
            if (newSet.has(messageIndex)) {
                newSet.delete(messageIndex);
            } else {
                newSet.add(messageIndex);
            }
            return newSet;
        });
    };

    const fetchLatestAnswer = async () => {
        try {
            const baseURL = process.env.NODE_ENV === 'production' ? '' : 'http://127.0.0.1:8000';
            const res = await axios.get(`${baseURL}/latest_answer`);
            const data = res.data;

            // updateData removed - not needed
            if (!data.answer || data.answer.trim() === '') {
                return;
            }
            const normalizedNewAnswer = normalizeAnswer(data.answer);
            const answerExists = messages.some(
                (msg) => normalizeAnswer(msg.content) === normalizedNewAnswer
            );
            if (!answerExists) {
                setMessages((prev) => [
                    ...prev,
                    {
                        type: 'agent',
                        content: data.answer,
                        reasoning: data.reasoning,
                        agentName: data.agent_name,
                        status: data.status,
                        uid: data.uid,
                    },
                ]);
                setStatus(data.status);
                scrollToBottom();
            } else {
                console.log('Duplicate answer detected, skipping:', data.answer);
            }
        } catch (error) {
            console.error('Error fetching latest answer:', error);
        }
    };


    const handleStop = async (e) => {
        e.preventDefault();
        checkHealth();
        setIsLoading(false);
        setError(null);
        try {
            await axios.get('http://127.0.0.1:8000/stop');
            setStatus("Requesting stop...");
        } catch (err) {
            console.error('Error stopping the agent:', err);
        }
    }

    const handleFileUpload = (e) => {
        const file = e.target.files[0];
        console.log('File selected:', file);
        if (file) {
            setUploadedFile(file);
            console.log('File uploaded:', file.name, file.type);
            
            // Show preview for images
            let preview = null;
            if (file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    preview = e.target.result;
                    setFilePreview(preview);
                    
                    // Add file upload message to chat
                    const fileMessage = {
                        type: 'user',
                        content: `📎 Uploaded: ${file.name}\n\n[Image preview shown below]`,
                        filePreview: preview,
                        fileName: file.name,
                        fileType: file.type
                    };
                    setMessages(prev => [...prev, fileMessage]);
                    scrollToBottom();
                };
                reader.readAsDataURL(file);
            } else {
                setFilePreview(null);
                
                // Add file upload message to chat for non-images
                const fileMessage = {
                    type: 'user',
                    content: `📎 Uploaded: ${file.name} (${file.type || 'unknown type'})`,
                    fileName: file.name,
                    fileType: file.type
                };
                setMessages(prev => [...prev, fileMessage]);
                scrollToBottom();
            }
            
            // Read file content
            const reader = new FileReader();
            reader.onload = (e) => {
                setFileContent(e.target.result);
            };
            
            if (file.type === 'text/csv' || file.name.endsWith('.csv')) {
                reader.readAsText(file);
            } else if (file.type.startsWith('image/')) {
                reader.readAsDataURL(file);
            } else if (file.type === 'application/pdf' || file.name.endsWith('.pdf')) {
                reader.readAsDataURL(file);  // Read PDF as base64
            } else if (file.name.endsWith('.xlsx') || file.name.endsWith('.xls') || 
                       file.type.includes('spreadsheet') || file.type.includes('excel')) {
                reader.readAsDataURL(file);  // Read Excel as base64
            } else {
                reader.readAsText(file);
            }
        }
    };

    const removeFile = () => {
        setUploadedFile(null);
        setFileContent(null);
        setFilePreview(null);
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        checkHealth();
        if (!query.trim() && !uploadedFile) {
            console.log('Empty query and no file');
            return;
        }
        
        // Add user message with file info
        let userMessage = query;
        if (uploadedFile) {
            userMessage += `\n\n📎 Attached: ${uploadedFile.name} (${uploadedFile.type})`;
        }
        setMessages((prev) => [...prev, { type: 'user', content: userMessage }]);
        scrollToBottom();
        setIsLoading(true);
        setError(null);

        try {
            console.log('Sending query:', query);
            setQuery('waiting for response...');
            
            // Get LLM settings from localStorage
            const llmModel = localStorage.getItem('yappy_llm_model');
            const apiKey = localStorage.getItem('yappy_api_key');
            
            // Validate API key is available
            if (!apiKey || !llmModel) {
                setError('No API key found. Please log out and log back in with your API key.');
                setIsLoading(false);
                setQuery('');
                return;
            }
            
            const payload = {
                query,
                tts_enabled: false,
                llm_model: llmModel,
                api_key: apiKey
            };
            
            // Add file data if uploaded
            if (uploadedFile && fileContent) {
                payload.file_data = {
                    name: uploadedFile.name,
                    type: uploadedFile.type,
                    content: fileContent
                };
                console.log('Sending file data:', {
                    name: uploadedFile.name,
                    type: uploadedFile.type,
                    contentLength: fileContent.length,
                    contentPreview: fileContent.substring(0, 100)
                });
            }
            
            const baseURL = process.env.NODE_ENV === 'production' ? '' : 'http://127.0.0.1:8000';
            const res = await axios.post(`${baseURL}/query`, payload);
            setQuery('Enter your query...');
            console.log('Response:', res.data);
            // Response is handled by fetchLatestAnswer()
        } catch (err) {
            console.error('Error:', err);
            setError('Failed to process query.');
            setMessages((prev) => [
                ...prev,
                { type: 'error', content: 'Error: Unable to get a response.' },
            ]);
            scrollToBottom();
        } finally {
            console.log('Query completed');
            setIsLoading(false);
            setQuery('');
        }
    };


    if (!isAuthenticated) {
        return <Auth onLogin={handleLogin} />;
    }

    return (
        <div className="app">
            <header className="header">
                <div className="header-content">
                    <div className="yappy-logo">
                        <div className="yappy-logo-icon">
                            <div className="yappy-mini-head">
                                <div className="yappy-mini-ear left"></div>
                                <div className="yappy-mini-ear right"></div>
                                <div className="yappy-mini-eye left"></div>
                                <div className="yappy-mini-eye right"></div>
                                <div className="yappy-mini-nose"></div>
                            </div>
                        </div>
                        <h1>Yappy</h1>
                    </div>
                    <button onClick={handleLogout} className="logout-button">
                        Logout
                    </button>
                </div>
            </header>
            <main className="main">
                <div className="chat-container">
                    <div className="chat-section">
                        <div className="messages">
                            {messages.length === 0 ? (
                                <div className="welcome-message">
                                    <div className="yappy-avatar">
                                        <div className="yappy-character">
                                            <div className="yappy-head">
                                                <div className="yappy-ear left"></div>
                                                <div className="yappy-ear right"></div>
                                                <div className="yappy-face">
                                                    <div className="yappy-eye left">
                                                        <div className="yappy-pupil"></div>
                                                    </div>
                                                    <div className="yappy-eye right">
                                                        <div className="yappy-pupil"></div>
                                                    </div>
                                                    <div className="yappy-nose"></div>
                                                    <div className="yappy-mouth">
                                                        <div className="yappy-tongue"></div>
                                                    </div>
                                                </div>
                                            </div>
                                            <div className="yappy-body">
                                                <div className="yappy-tail"></div>
                                                <div className="yappy-paw front-left"></div>
                                                <div className="yappy-paw front-right"></div>
                                                <div className="yappy-paw back-left"></div>
                                                <div className="yappy-paw back-right"></div>
                                            </div>
                                            <div className="yappy-speech-bubble">
                                                <span>Woof! 🐾</span>
                                            </div>
                                        </div>
                                    </div>
                                    <h2>Hello, I'm Yappy</h2>
                                    <p>Your friendly AI pup assistant! How can I help you today? 🎾</p>
                                </div>
                            ) : (
                                messages.map((msg, index) => (
                                    <div
                                        key={index}
                                        className={`message ${
                                            msg.type === 'user'
                                                ? 'user-message'
                                                : msg.type === 'agent'
                                                ? 'agent-message'
                                                : 'error-message'
                                        }`}
                                    >
                                        {msg.type === 'agent' && msg.reasoning && (
                                            <div className="message-header">
                                                <button 
                                                    className="reasoning-toggle"
                                                    onClick={() => toggleReasoning(index)}
                                                    title={expandedReasoning.has(index) ? "Hide reasoning" : "Show reasoning"}
                                                >
                                                    {expandedReasoning.has(index) ? '▼' : '▶'} Reasoning
                                                </button>
                                            </div>
                                        )}
                                        {msg.type === 'agent' && msg.reasoning && expandedReasoning.has(index) && (
                                            <div className="reasoning-content">
                                                <ReactMarkdown
                                                    components={{
                                                        img: ({node, ...props}) => (
                                                            <img 
                                                                {...props} 
                                                                style={{maxWidth: '100%', height: 'auto', margin: '10px 0'}} 
                                                                alt={props.alt || 'Chart'}
                                                            />
                                                        )
                                                    }}
                                                >
                                                    {msg.reasoning}
                                                </ReactMarkdown>
                                            </div>
                                        )}
                                        {msg.type === 'agent' && (
                                            <div className="agent-info">
                                                <div className="yappy-mini-avatar">
                                                    <div className="yappy-mini-head">
                                                        <div className="yappy-mini-ear left"></div>
                                                        <div className="yappy-mini-ear right"></div>
                                                        <div className="yappy-mini-eye left"></div>
                                                        <div className="yappy-mini-eye right"></div>
                                                        <div className="yappy-mini-nose"></div>
                                                    </div>
                                                </div>
                                                <span className="agent-name">{msg.agentName}</span>
                                            </div>
                                        )}
                                        <div className="message-content">
                                            {(() => {
                                                // Extract base64 images from content
                                                const base64ImageRegex = /!\[([^\]]*)\]\((data:image\/png;base64,[^)]+)\)/g;
                                                const images = [];
                                                let match;
                                                let contentWithoutImages = msg.content;
                                                
                                                while ((match = base64ImageRegex.exec(msg.content)) !== null) {
                                                    images.push({
                                                        alt: match[1] || 'Chart',
                                                        src: match[2]
                                                    });
                                                    // Remove the image markdown from content
                                                    contentWithoutImages = contentWithoutImages.replace(match[0], '');
                                                }
                                                
                                                // Add copy button for all agent responses
                                                const showCopyButton = msg.type === 'agent' && contentWithoutImages.length > 50;
                                                
                                                const copyToClipboard = (text) => {
                                                    // Remove markdown formatting for cleaner copy
                                                    const cleanText = text
                                                        .replace(/\*\*(.*?)\*\*/g, '$1') // Remove bold
                                                        .replace(/\*(.*?)\*/g, '$1')     // Remove italic
                                                        .replace(/`(.*?)`/g, '$1')       // Remove code
                                                        .replace(/#{1,6}\s/g, '')        // Remove headers
                                                        .replace(/^\s*[-*+]\s/gm, '• ')  // Convert list items
                                                        .replace(/^\s*\d+\.\s/gm, '')    // Remove numbered lists
                                                        .trim();
                                                    
                                                    navigator.clipboard.writeText(cleanText).then(() => {
                                                        // Show temporary success message
                                                        const button = document.activeElement;
                                                        const originalText = button.textContent;
                                                        button.textContent = '✅ Copied!';
                                                        button.style.background = 'linear-gradient(135deg, #4ade80 0%, #22c55e 100%)';
                                                        setTimeout(() => {
                                                            button.textContent = originalText;
                                                            button.style.background = '';
                                                        }, 2000);
                                                    }).catch(() => {
                                                        alert('Failed to copy text. Please select and copy manually.');
                                                    });
                                                };
                                                
                                                return (
                                                    <div className="message-text-container">
                                                        {showCopyButton && (
                                                            <div className="copy-button-container">
                                                                <button 
                                                                    className="copy-text-button"
                                                                    onClick={() => copyToClipboard(contentWithoutImages)}
                                                                    title="Copy text to clipboard"
                                                                >
                                                                    📋 Copy Text
                                                                </button>
                                                            </div>
                                                        )}
                                                        <div className={contentWithoutImages.length > 500 ? "long-text-content" : ""}>
                                                            <ReactMarkdown>{contentWithoutImages}</ReactMarkdown>
                                                        </div>
                                                        {images.map((img, idx) => (
                                                            <div key={idx} style={{margin: '20px 0', textAlign: 'center'}}>
                                                                <img 
                                                                    src={img.src} 
                                                                    alt={img.alt}
                                                                    style={{
                                                                        maxWidth: '100%', 
                                                                        height: 'auto',
                                                                        border: '1px solid #ddd',
                                                                        borderRadius: '8px',
                                                                        boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
                                                                    }}
                                                                />
                                                                <p style={{fontSize: '12px', color: '#666', marginTop: '5px'}}>
                                                                    {img.alt}
                                                                </p>
                                                            </div>
                                                        ))}
                                                    </div>
                                                );
                                            })()}
                                            {msg.filePreview && (
                                                <div className="message-file-preview">
                                                    <img src={msg.filePreview} alt={msg.fileName || "Uploaded image"} />
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                ))
                            )}
                            <div ref={messagesEndRef} />
                        </div>
                        {isOnline && (
                            <div className="status-bar">
                                <div className="status-indicator online"></div>
                                <span>{status}</span>
                            </div>
                        )}
                        {!isLoading && !isOnline && (
                            <div className="status-bar">
                                <div className="status-indicator offline"></div>
                                <span>System offline. Deploy backend first.</span>
                            </div>
                        )}
                        <form onSubmit={handleSubmit} className="input-form">
                            <div className="input-container">
                                <input
                                    type="text"
                                    value={query}
                                    onChange={(e) => setQuery(e.target.value)}
                                    placeholder={uploadedFile ? "Ask about the uploaded file..." : "Ask Yappy anything..."}
                                    disabled={isLoading}
                                />
                                <input
                                    ref={fileInputRef}
                                    type="file"
                                    onChange={handleFileUpload}
                                    style={{ display: 'none' }}
                                    accept=".csv,.xlsx,.xls,.pdf,.txt,.json,image/*"
                                />
                                <button 
                                    type="button" 
                                    onClick={() => fileInputRef.current.click()} 
                                    className="upload-button"
                                    disabled={isLoading}
                                    title="Upload file"
                                >
                                    📁
                                </button>
                                <button type="submit" disabled={isLoading} className="send-button">
                                    <span>{isLoading ? '⏳' : '✨'}</span>
                                    {isLoading ? 'Thinking...' : 'Send'}
                                </button>
                                <button onClick={handleStop} className="stop-button">
                                    ⏹️ Stop
                                </button>
                            </div>
                        </form>
                    </div>
                </div>
            </main>
        </div>
    );
}

export default App;