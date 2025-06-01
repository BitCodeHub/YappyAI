"""
Enhanced LLM Handler with Multi-Agent Army Support
"""
import asyncio
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import json

from sources.agents import (
    Agent, CasualAgent, CoderAgent, FileAgent, 
    BrowserAgent, PlannerAgent, McpAgent,
    OrchestratorAgent, ResearchAgent, DataAgent, 
    AutomationAgent
)
from sources.router import AgentRouter
from sources.logger import Logger


class EnhancedLLMHandler:
    """
    Enhanced LLM Handler that manages the army of AI agents
    """
    
    def __init__(self):
        self.logger = Logger("enhanced_llm_handler.log")
        
        # Initialize orchestrator
        self.orchestrator = OrchestratorAgent()
        
        # Initialize all agents
        self.agents = {
            "casual": CasualAgent(),
            "code": CoderAgent(),
            "file": FileAgent(),
            "browser": BrowserAgent(),
            "planner": PlannerAgent(),
            "mcp": McpAgent(),
            "research": ResearchAgent(),
            "data": DataAgent(),
            "automation": AutomationAgent()
        }
        
        # Register agents with orchestrator
        for agent_id, agent in self.agents.items():
            capabilities = self._get_agent_capabilities(agent_id)
            self.orchestrator.register_agent(agent_id, agent, capabilities)
            
        # Initialize router for backward compatibility
        self.router = AgentRouter([])
        
        # Task tracking
        self.active_tasks = {}
        self.task_history = []
        
    def _get_agent_capabilities(self, agent_id: str) -> List[str]:
        """Get capabilities for each agent type"""
        capabilities_map = {
            "casual": ["conversation", "general", "simple_tasks"],
            "code": ["code_generation", "code_execution", "debugging"],
            "file": ["file_operations", "file_search", "file_management"],
            "browser": ["web_browsing", "web_scraping", "form_filling"],
            "planner": ["task_planning", "strategy", "decomposition"],
            "mcp": ["mcp_protocol", "tool_integration"],
            "research": ["research", "information_gathering", "fact_checking"],
            "data": ["data_analysis", "visualization", "data_processing"],
            "automation": ["workflow_creation", "scheduling", "task_automation"]
        }
        return capabilities_map.get(agent_id, ["general"])
        
    async def get_response(self, message: str, model_name: str, api_key: Optional[str],
                          conversation_history: List[Dict] = None, 
                          file_data: Optional[Dict] = None) -> Tuple[str, int]:
        """
        Enhanced response handling with multi-agent support
        """
        self.logger.log(f"Processing message: {message[:100]}...")
        
        # Check if this is an orchestration request
        if self._is_orchestration_request(message):
            return await self._handle_orchestration(message, model_name, api_key)
            
        # Check if this is targeting a specific agent
        agent_name = self._extract_agent_name(message)
        if agent_name:
            return await self._handle_specific_agent(agent_name, message, model_name, api_key, file_data)
            
        # Use traditional routing for backward compatibility
        agent_type, _ = self.router.route_query(message)
        
        # Map to new agent system
        agent_mapping = {
            "browser_agent": "browser",
            "code_agent": "code",
            "file_agent": "file",
            "planner_agent": "planner",
            "casual_agent": "casual"
        }
        
        agent_id = agent_mapping.get(agent_type, "casual")
        agent = self.agents.get(agent_id)
        
        if agent:
            # Set API key if available
            if hasattr(agent, 'set_api_key'):
                agent.set_api_key(api_key)
                
            response, _ = agent.generate_response(message)
            return response, self._estimate_tokens(response)
            
        # Fallback to casual agent
        return self.agents["casual"].generate_response(message)[0], 100
        
    def _is_orchestration_request(self, message: str) -> bool:
        """Check if the message requires orchestration"""
        orchestration_keywords = [
            "create workflow", "automate", "schedule task",
            "multiple agents", "coordinate", "orchestrate",
            "research and analyze", "monitor and alert",
            "daily report", "recurring task"
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in orchestration_keywords)
        
    def _extract_agent_name(self, message: str) -> Optional[str]:
        """Extract specific agent name from message"""
        # Check for @agent mentions
        if "@" in message:
            words = message.split()
            for word in words:
                if word.startswith("@"):
                    agent_name = word[1:].lower()
                    if agent_name in self.agents:
                        return agent_name
                        
        # Check for explicit agent requests
        agent_phrases = {
            "research agent": "research",
            "data agent": "data",
            "automation agent": "automation",
            "browser agent": "browser",
            "code agent": "code",
            "file agent": "file"
        }
        
        message_lower = message.lower()
        for phrase, agent_id in agent_phrases.items():
            if phrase in message_lower:
                return agent_id
                
        return None
        
    async def _handle_orchestration(self, message: str, model_name: str, 
                                   api_key: Optional[str]) -> Tuple[str, int]:
        """Handle orchestration requests"""
        # Create a task through orchestrator
        task = self.orchestrator.create_task(message, task_type="complex")
        
        # Store in active tasks
        self.active_tasks[task.id] = {
            "task": task,
            "start_time": datetime.now(),
            "status": "created"
        }
        
        # Start async execution
        asyncio.create_task(self._execute_orchestrated_task(task.id, model_name, api_key))
        
        response = f"""
🤖 **AI Agent Army Activated!**

I've created a complex task that will be handled by multiple specialized agents:

**Task ID**: `{task.id[:8]}...`
**Description**: {message}

The following agents may be involved:
- 🔍 Research Agent - For gathering information
- 📊 Data Agent - For analysis and visualization  
- 🌐 Browser Agent - For web interactions
- 💻 Code Agent - For any programming tasks
- 📁 File Agent - For file operations
- 🔄 Automation Agent - For creating workflows

I'm orchestrating the agents now. You can check the progress by asking:
- "What's the status of task {task.id[:8]}?"
- "Show agent activity"

Would you like to receive updates as the task progresses?
"""
        
        return response, self._estimate_tokens(response)
        
    async def _execute_orchestrated_task(self, task_id: str, model_name: str, api_key: str):
        """Execute orchestrated task in background"""
        try:
            # Update task status
            if task_id in self.active_tasks:
                self.active_tasks[task_id]["status"] = "running"
                
            # Execute through orchestrator
            results = await self.orchestrator.execute_task(
                self.orchestrator._find_task(task_id)
            )
            
            # Update task status
            if task_id in self.active_tasks:
                self.active_tasks[task_id]["status"] = "completed"
                self.active_tasks[task_id]["results"] = results
                self.active_tasks[task_id]["end_time"] = datetime.now()
                
                # Move to history
                self.task_history.append(self.active_tasks[task_id])
                del self.active_tasks[task_id]
                
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            if task_id in self.active_tasks:
                self.active_tasks[task_id]["status"] = "failed"
                self.active_tasks[task_id]["error"] = str(e)
                
    async def _handle_specific_agent(self, agent_id: str, message: str, 
                                    model_name: str, api_key: Optional[str],
                                    file_data: Optional[Dict] = None) -> Tuple[str, int]:
        """Handle request for specific agent"""
        agent = self.agents.get(agent_id)
        if not agent:
            return f"Unknown agent: {agent_id}", 50
            
        # Remove agent mention from message
        clean_message = message.replace(f"@{agent_id}", "").strip()
        
        # Set API key if available
        if hasattr(agent, 'set_api_key'):
            agent.set_api_key(api_key)
            
        # Special handling for certain agents
        if agent_id == "research" and hasattr(agent, 'autonomous_research'):
            # Extract research parameters
            depth = "standard"
            if "quick" in message.lower():
                depth = "quick"
            elif "comprehensive" in message.lower():
                depth = "comprehensive"
                
            result = await agent.autonomous_research(clean_message, depth)
            response = result.get("output", "Research failed")
        else:
            response, _ = agent.generate_response(clean_message)
            
        return response, self._estimate_tokens(response)
        
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        status = {
            "agents": self.orchestrator.get_agent_status(),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.task_history),
            "task_details": []
        }
        
        # Add active task details
        for task_id, task_info in self.active_tasks.items():
            task_detail = {
                "id": task_id[:8] + "...",
                "description": task_info["task"].description[:50] + "...",
                "status": task_info["status"],
                "duration": str(datetime.now() - task_info["start_time"])
            }
            status["task_details"].append(task_detail)
            
        return status
        
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get detailed status of a specific task"""
        # Check active tasks
        if task_id in self.active_tasks:
            return self.active_tasks[task_id]
            
        # Check completed tasks
        for task in self.task_history:
            if task["task"].id == task_id:
                return task
                
        # Check orchestrator
        return self.orchestrator.get_task_status(task_id)
        
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count"""
        # Rough estimation: 1 token ≈ 4 characters
        return len(text) // 4
        
    def get_capabilities(self) -> Dict[str, List[str]]:
        """Get capabilities of all agents"""
        capabilities = {}
        for agent_id, agent in self.agents.items():
            capabilities[agent_id] = {
                "name": agent.__class__.__name__,
                "capabilities": self._get_agent_capabilities(agent_id),
                "description": self._get_agent_description(agent_id)
            }
        return capabilities
        
    def _get_agent_description(self, agent_id: str) -> str:
        """Get description for each agent"""
        descriptions = {
            "casual": "General conversation and simple tasks",
            "code": "Code generation, debugging, and execution",
            "file": "File system operations and management",
            "browser": "Web browsing, scraping, and automation",
            "planner": "Strategic planning and task decomposition",
            "mcp": "Model Context Protocol integration",
            "research": "Autonomous research and fact-checking",
            "data": "Data analysis, processing, and visualization",
            "automation": "Workflow automation and scheduling",
            "orchestrator": "Multi-agent coordination and task management"
        }
        return descriptions.get(agent_id, "Specialized AI agent")