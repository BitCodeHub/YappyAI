import React, { useState } from 'react';
import axios from 'axios';
import './Auth.css';

function Auth({ onLogin }) {
    const [isLogin, setIsLogin] = useState(true);
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [selectedModel, setSelectedModel] = useState('openai');
    const [apiKey, setApiKey] = useState('');

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setSuccess('');
        setIsLoading(true);

        try {
            if (isLogin) {
                // Login
                const baseURL = process.env.NODE_ENV === 'production' ? '' : 'http://127.0.0.1:8000';
                const response = await axios.post(`${baseURL}/auth/login`, {
                    username,
                    password,
                    llm_model: selectedModel,
                    api_key: apiKey
                });
                
                const { access_token } = response.data;
                
                // Store token and LLM settings in localStorage
                localStorage.setItem('yappy_token', access_token);
                localStorage.setItem('yappy_llm_model', selectedModel);
                localStorage.setItem('yappy_api_key', apiKey);
                
                // Set default axios header
                axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
                
                setSuccess('Login successful!');
                
                // Call parent component's login handler with LLM settings
                setTimeout(() => {
                    onLogin(access_token, { model: selectedModel, apiKey });
                }, 1000);
                
            } else {
                // Register
                const baseURL = process.env.NODE_ENV === 'production' ? '' : 'http://127.0.0.1:8000';
                await axios.post(`${baseURL}/auth/register`, {
                    username,
                    password
                });
                
                setSuccess('Registration successful! Please login.');
                setIsLogin(true);
                setPassword('');
            }
        } catch (err) {
            if (err.response?.data?.detail) {
                setError(err.response.data.detail);
            } else {
                setError('An error occurred. Please try again.');
            }
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="auth-container">
            <div className="auth-card">
                <div className="auth-yappy-character">
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
                            <span>Welcome! 🐾</span>
                        </div>
                    </div>
                </div>
                <h1>Yappy</h1>
                
                <form onSubmit={handleSubmit} className="auth-form">
                    <div className="form-group">
                        <input
                            type="text"
                            placeholder="Username"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            required
                            disabled={isLoading}
                        />
                    </div>
                    
                    <div className="form-group">
                        <input
                            type="password"
                            placeholder="Password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            required
                            disabled={isLoading}
                        />
                    </div>
                    
                    <div className="form-group">
                        <select
                            value={selectedModel}
                            onChange={(e) => setSelectedModel(e.target.value)}
                            disabled={isLoading}
                            className="model-selector"
                        >
                            <option value="openai">OpenAI (GPT-4)</option>
                            <option value="anthropic">Anthropic (Claude)</option>
                            <option value="google">Google (Gemini)</option>
                            <option value="groq">Groq</option>
                            <option value="ollama">Ollama (Local)</option>
                        </select>
                    </div>
                    
                    <div className="form-group">
                        <input
                            type="password"
                            placeholder={`${selectedModel === 'ollama' ? 'Ollama URL (e.g., http://localhost:11434)' : `${selectedModel.charAt(0).toUpperCase() + selectedModel.slice(1)} API Key (Required)`}`}
                            value={apiKey}
                            onChange={(e) => setApiKey(e.target.value)}
                            required={true}
                            disabled={isLoading}
                        />
                    </div>
                    
                    {error && <p className="error-message">{error}</p>}
                    {success && <p className="success-message">{success}</p>}
                    
                    <button type="submit" disabled={isLoading}>
                        {isLoading ? 'Loading...' : (isLogin ? 'Login' : 'Register')}
                    </button>
                </form>
                
                <p className="auth-switch">
                    {isLogin ? "Don't have an account? " : "Already have an account? "}
                    <button 
                        className="link-button"
                        onClick={() => {
                            setIsLogin(!isLogin);
                            setError('');
                            setSuccess('');
                        }}
                        disabled={isLoading}
                    >
                        {isLogin ? 'Register' : 'Login'}
                    </button>
                </p>
            </div>
        </div>
    );
}

export default Auth;