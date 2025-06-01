// Agent Dashboard Component for Yappy AI Army
const AgentDashboard = ({ isOpen, onClose }) => {
    const [agents, setAgents] = React.useState({});
    const [agentStatus, setAgentStatus] = React.useState({});
    const [activeTasks, setActiveTasks] = React.useState([]);
    const [isLoading, setIsLoading] = React.useState(true);
    const [selectedAgent, setSelectedAgent] = React.useState(null);
    const [refreshInterval, setRefreshInterval] = React.useState(null);

    React.useEffect(() => {
        if (isOpen) {
            loadAgentData();
            // Set up auto-refresh every 5 seconds
            const interval = setInterval(loadAgentData, 5000);
            setRefreshInterval(interval);
        }
        
        return () => {
            if (refreshInterval) {
                clearInterval(refreshInterval);
            }
        };
    }, [isOpen]);

    const loadAgentData = async () => {
        try {
            // Get auth token from localStorage
            const token = localStorage.getItem('yappy_token');
            const headers = token ? { Authorization: `Bearer ${token}` } : {};

            // Load agent capabilities
            const capabilitiesRes = await axios.get('/api/agents', { headers });
            if (capabilitiesRes.data.agents) {
                setAgents(capabilitiesRes.data.agents);
            }

            // Load agent status
            const statusRes = await axios.get('/api/agents/status', { headers });
            setAgentStatus(statusRes.data);
            setActiveTasks(statusRes.data.task_details || []);
            
            setIsLoading(false);
        } catch (error) {
            console.error('Failed to load agent data:', error);
            setIsLoading(false);
        }
    };

    const getAgentIcon = (agentId) => {
        const icons = {
            casual: '💬',
            code: '💻',
            file: '📁',
            browser: '🌐',
            planner: '📋',
            research: '🔍',
            data: '📊',
            automation: '🔄',
            orchestrator: '🎯'
        };
        return icons[agentId] || '🤖';
    };

    const getAgentStatusColor = (status) => {
        return status === 'busy' ? '#f59e0b' : '#10b981';
    };

    if (!isOpen) return null;

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-content agent-dashboard" onClick={(e) => e.stopPropagation()}>
                <div className="dashboard-header">
                    <h2>🤖 AI Agent Army Dashboard</h2>
                    <button onClick={onClose} className="close-button">✖</button>
                </div>

                {isLoading ? (
                    <div className="loading-spinner">Loading agent data...</div>
                ) : (
                    <div className="dashboard-content">
                        {/* Agent Grid */}
                        <div className="agents-section">
                            <h3>Available Agents</h3>
                            <div className="agents-grid">
                                {Object.entries(agents).map(([agentId, agentInfo]) => {
                                    const status = agentStatus.agents?.[agentId];
                                    return (
                                        <div 
                                            key={agentId} 
                                            className={`agent-card ${selectedAgent === agentId ? 'selected' : ''}`}
                                            onClick={() => setSelectedAgent(agentId)}
                                        >
                                            <div className="agent-icon">{getAgentIcon(agentId)}</div>
                                            <h4>{agentInfo.name}</h4>
                                            <p className="agent-description">{agentInfo.description}</p>
                                            <div className="agent-status">
                                                <span 
                                                    className="status-dot"
                                                    style={{ backgroundColor: getAgentStatusColor(status?.status) }}
                                                />
                                                {status?.status || 'idle'}
                                            </div>
                                            <div className="agent-capabilities">
                                                {agentInfo.capabilities.slice(0, 3).map((cap, idx) => (
                                                    <span key={idx} className="capability-tag">{cap}</span>
                                                ))}
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>

                        {/* Active Tasks */}
                        <div className="tasks-section">
                            <h3>Active Tasks ({activeTasks.length})</h3>
                            {activeTasks.length === 0 ? (
                                <p className="no-tasks">No active tasks at the moment</p>
                            ) : (
                                <div className="task-list">
                                    {activeTasks.map((task, idx) => (
                                        <div key={idx} className="task-item">
                                            <div className="task-header">
                                                <span className="task-id">Task {task.id}</span>
                                                <span className={`task-status ${task.status}`}>
                                                    {task.status}
                                                </span>
                                            </div>
                                            <p className="task-description">{task.description}</p>
                                            <div className="task-duration">
                                                Duration: {task.duration}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>

                        {/* Selected Agent Details */}
                        {selectedAgent && agents[selectedAgent] && (
                            <div className="agent-details">
                                <h3>Agent Details: {agents[selectedAgent].name}</h3>
                                <div className="details-content">
                                    <h4>Capabilities:</h4>
                                    <ul>
                                        {agents[selectedAgent].capabilities.map((cap, idx) => (
                                            <li key={idx}>{cap}</li>
                                        ))}
                                    </ul>
                                    <h4>How to use:</h4>
                                    <p className="usage-example">
                                        Type <code>@{selectedAgent}</code> followed by your request, or let the orchestrator automatically assign this agent to relevant tasks.
                                    </p>
                                </div>
                            </div>
                        )}

                        {/* Statistics */}
                        <div className="statistics-section">
                            <h3>System Statistics</h3>
                            <div className="stats-grid">
                                <div className="stat-card">
                                    <div className="stat-value">
                                        {Object.keys(agents).length}
                                    </div>
                                    <div className="stat-label">Total Agents</div>
                                </div>
                                <div className="stat-card">
                                    <div className="stat-value">
                                        {agentStatus.active_tasks || 0}
                                    </div>
                                    <div className="stat-label">Active Tasks</div>
                                </div>
                                <div className="stat-card">
                                    <div className="stat-value">
                                        {agentStatus.completed_tasks || 0}
                                    </div>
                                    <div className="stat-label">Completed</div>
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

// Styles for the Agent Dashboard
const dashboardStyles = `
    .modal-overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.8);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 1000;
        backdrop-filter: blur(10px);
    }

    .modal-content {
        background: var(--bg-dark, #0a0a0f);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        padding: 2rem;
        position: relative;
        animation: modalFadeIn 0.3s ease-out;
    }

    @keyframes modalFadeIn {
        from {
            opacity: 0;
            transform: scale(0.95);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }

    .agent-dashboard {
        max-width: 1000px;
        width: 90%;
        max-height: 85vh;
        overflow-y: auto;
    }

    .dashboard-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid rgba(255, 255, 255, 0.1);
    }

    .dashboard-header h2 {
        margin: 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .close-button {
        background: none;
        border: none;
        color: var(--text-primary, #ffffff);
        font-size: 1.5rem;
        cursor: pointer;
        transition: transform 0.2s;
    }

    .close-button:hover {
        transform: scale(1.1);
    }

    .dashboard-content {
        display: flex;
        flex-direction: column;
        gap: 2rem;
    }

    .agents-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }

    .agent-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: center;
    }

    .agent-card:hover {
        background: rgba(255, 255, 255, 0.08);
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }

    .agent-card.selected {
        border-color: var(--color-primary);
        background: rgba(102, 126, 234, 0.1);
    }

    .agent-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }

    .agent-card h4 {
        margin: 0.5rem 0;
        color: var(--text-primary, #ffffff);
        font-size: 1rem;
    }

    .agent-description {
        font-size: 0.8rem;
        color: var(--text-secondary, rgba(255, 255, 255, 0.7));
        margin: 0.5rem 0;
    }

    .agent-status {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        animation: pulse 2s ease-in-out infinite;
    }

    .agent-capabilities {
        display: flex;
        flex-wrap: wrap;
        gap: 0.25rem;
        margin-top: 0.5rem;
        justify-content: center;
    }

    .capability-tag {
        background: rgba(102, 126, 234, 0.2);
        color: var(--color-primary, #667eea);
        padding: 0.25rem 0.5rem;
        border-radius: 12px;
        font-size: 0.7rem;
    }

    .task-list {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        margin-top: 1rem;
    }

    .task-item {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 1rem;
    }

    .task-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }

    .task-id {
        font-weight: 600;
        color: var(--color-primary);
    }

    .task-status {
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 500;
    }

    .task-status.running {
        background: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
    }

    .task-status.completed {
        background: rgba(16, 185, 129, 0.2);
        color: #10b981;
    }

    .task-description {
        color: var(--text-secondary);
        margin: 0.5rem 0;
    }

    .task-duration {
        font-size: 0.8rem;
        color: var(--text-muted);
    }

    .agent-details {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
    }

    .details-content {
        margin-top: 1rem;
    }

    .details-content h4 {
        color: var(--color-primary);
        margin: 1rem 0 0.5rem;
    }

    .details-content ul {
        list-style: none;
        padding-left: 1rem;
    }

    .details-content li::before {
        content: "→ ";
        color: var(--color-primary);
    }

    .usage-example {
        background: rgba(0, 0, 0, 0.2);
        padding: 1rem;
        border-radius: 8px;
        margin-top: 0.5rem;
    }

    .usage-example code {
        background: rgba(102, 126, 234, 0.2);
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        color: var(--color-primary);
    }

    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }

    .stat-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
    }

    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .stat-label {
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }

    .loading-spinner {
        text-align: center;
        padding: 4rem;
        color: var(--text-secondary);
    }

    .no-tasks {
        text-align: center;
        color: var(--text-muted, rgba(255, 255, 255, 0.5));
        padding: 2rem;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
    }

    /* Add section styles */
    .agents-section h3,
    .tasks-section h3,
    .agent-details h3,
    .statistics-section h3 {
        color: var(--text-primary, #ffffff);
        margin-bottom: 1rem;
        font-size: 1.2rem;
    }

    .tasks-section,
    .agent-details,
    .statistics-section {
        margin-top: 2rem;
    }

    /* Scrollbar styles */
    .agent-dashboard::-webkit-scrollbar {
        width: 8px;
    }

    .agent-dashboard::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 4px;
    }

    .agent-dashboard::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 4px;
    }

    .agent-dashboard::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.3);
    }
`;

// Add styles to document
const styleSheet = document.createElement('style');
styleSheet.textContent = dashboardStyles;
document.head.appendChild(styleSheet);

// Expose component globally
window.AgentDashboard = AgentDashboard;