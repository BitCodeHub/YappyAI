"""
Orchestrator Agent - The master coordinator for the AI agent army
"""
import asyncio
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

from sources.agents.agent import Agent
from sources.logger import Logger


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    STATUS = "status"
    ERROR = "error"
    BROADCAST = "broadcast"


@dataclass
class AgentMessage:
    """Message protocol for inter-agent communication"""
    sender: str
    receiver: str  # Agent ID or "broadcast"
    message_type: MessageType
    content: Dict[str, Any]
    priority: int = 3  # 1-5, 5 is highest
    timestamp: datetime = field(default_factory=datetime.now)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class Task:
    """Represents a task in the system"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = ""
    description: str = ""
    assigned_agents: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    sub_tasks: List['Task'] = field(default_factory=list)
    parent_task_id: Optional[str] = None
    progress: float = 0.0  # 0-100


class OrchestratorAgent(Agent):
    """
    Master orchestrator that coordinates all other agents
    """
    
    def __init__(self, llm_provider=None):
        super().__init__(llm_provider)
        self.logger = Logger("orchestrator.log")
        self.agent_registry: Dict[str, Agent] = {}
        self.task_queue: List[Task] = []
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        self.message_queue: List[AgentMessage] = []
        self.agent_capabilities: Dict[str, List[str]] = {}
        
    def register_agent(self, agent_id: str, agent: Agent, capabilities: List[str]):
        """Register an agent with its capabilities"""
        self.agent_registry[agent_id] = agent
        self.agent_capabilities[agent_id] = capabilities
        self.logger.log(f"Registered agent: {agent_id} with capabilities: {capabilities}")
        
    def create_task(self, description: str, task_type: str = "general", 
                   deadline: Optional[datetime] = None, metadata: Dict = None) -> Task:
        """Create a new task"""
        task = Task(
            type=task_type,
            description=description,
            deadline=deadline,
            metadata=metadata or {}
        )
        self.task_queue.append(task)
        self.logger.log(f"Created task: {task.id} - {description}")
        return task
        
    def decompose_task(self, task: Task) -> List[Task]:
        """Decompose a complex task into subtasks"""
        # Use LLM to analyze and break down the task
        prompt = f"""
        Analyze this task and break it down into smaller, actionable subtasks:
        
        Task: {task.description}
        Type: {task.type}
        
        For each subtask, specify:
        1. Description
        2. Required agent type (browser, code, file, research, data, etc.)
        3. Dependencies on other subtasks
        4. Estimated complexity (simple, moderate, complex)
        
        Return as JSON array.
        """
        
        # For now, let's create a simple decomposition
        # This would be enhanced with actual LLM call
        subtasks = []
        
        if "research" in task.description.lower():
            subtasks.extend([
                Task(
                    type="research",
                    description="Gather information from web sources",
                    parent_task_id=task.id
                ),
                Task(
                    type="analysis",
                    description="Analyze and synthesize gathered information",
                    parent_task_id=task.id,
                    dependencies=[subtasks[0].id] if subtasks else []
                ),
                Task(
                    type="report",
                    description="Generate comprehensive report",
                    parent_task_id=task.id,
                    dependencies=[subtasks[1].id] if len(subtasks) > 1 else []
                )
            ])
            
        task.sub_tasks = subtasks
        return subtasks
        
    def assign_agents(self, task: Task) -> List[str]:
        """Assign appropriate agents to a task based on requirements"""
        assigned = []
        
        # Map task types to agent capabilities
        task_agent_mapping = {
            "research": ["browser_agent", "research_agent"],
            "code": ["code_agent"],
            "file": ["file_agent"],
            "analysis": ["data_agent", "research_agent"],
            "report": ["document_agent"],
            "automation": ["automation_agent"],
            "general": ["casual_agent"]
        }
        
        # Find suitable agents
        required_agents = task_agent_mapping.get(task.type, ["casual_agent"])
        
        for agent_type in required_agents:
            for agent_id, capabilities in self.agent_capabilities.items():
                if agent_type in capabilities and agent_id not in assigned:
                    assigned.append(agent_id)
                    break
                    
        task.assigned_agents = assigned
        self.logger.log(f"Assigned agents {assigned} to task {task.id}")
        return assigned
        
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute a task by coordinating assigned agents"""
        task.status = TaskStatus.IN_PROGRESS
        self.active_tasks[task.id] = task
        
        try:
            # Decompose if complex
            if not task.sub_tasks and self._is_complex_task(task):
                task.sub_tasks = self.decompose_task(task)
                
            # Execute subtasks first
            if task.sub_tasks:
                for subtask in task.sub_tasks:
                    if self._dependencies_met(subtask):
                        await self.execute_task(subtask)
                        
            # Assign agents if not already assigned
            if not task.assigned_agents:
                self.assign_agents(task)
                
            # Send task to assigned agents
            results = {}
            for agent_id in task.assigned_agents:
                agent = self.agent_registry.get(agent_id)
                if agent:
                    # Send message to agent
                    message = AgentMessage(
                        sender="orchestrator",
                        receiver=agent_id,
                        message_type=MessageType.REQUEST,
                        content={
                            "task_id": task.id,
                            "action": "execute",
                            "description": task.description,
                            "metadata": task.metadata
                        },
                        priority=5 if task.deadline else 3
                    )
                    
                    # In real implementation, this would be async message passing
                    # For now, direct execution
                    result = await self._execute_agent_task(agent, task)
                    results[agent_id] = result
                    
            task.results = results
            task.status = TaskStatus.COMPLETED
            task.progress = 100.0
            
            # Move to completed
            del self.active_tasks[task.id]
            self.completed_tasks[task.id] = task
            
            return results
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.results = {"error": str(e)}
            self.logger.error(f"Task {task.id} failed: {e}")
            raise
            
    async def _execute_agent_task(self, agent: Agent, task: Task) -> Dict[str, Any]:
        """Execute task with specific agent"""
        # This would be replaced with actual agent execution
        # For now, return mock result
        return {
            "status": "completed",
            "result": f"Task executed by {agent.__class__.__name__}",
            "timestamp": datetime.now().isoformat()
        }
        
    def _is_complex_task(self, task: Task) -> bool:
        """Determine if a task is complex and needs decomposition"""
        # Simple heuristic - tasks with multiple verbs or long descriptions
        description_words = task.description.lower().split()
        action_words = ["research", "analyze", "create", "generate", "compare", "evaluate"]
        action_count = sum(1 for word in description_words if word in action_words)
        
        return action_count > 1 or len(description_words) > 20
        
    def _dependencies_met(self, task: Task) -> bool:
        """Check if all task dependencies are completed"""
        for dep_id in task.dependencies:
            dep_task = self._find_task(dep_id)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                return False
        return True
        
    def _find_task(self, task_id: str) -> Optional[Task]:
        """Find a task by ID in any queue"""
        # Check active tasks
        if task_id in self.active_tasks:
            return self.active_tasks[task_id]
        # Check completed tasks
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        # Check queue
        for task in self.task_queue:
            if task.id == task_id:
                return task
        return None
        
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get detailed status of a task"""
        task = self._find_task(task_id)
        if not task:
            return {"error": "Task not found"}
            
        return {
            "id": task.id,
            "description": task.description,
            "status": task.status.value,
            "progress": task.progress,
            "assigned_agents": task.assigned_agents,
            "sub_tasks": [self.get_task_status(st.id) for st in task.sub_tasks],
            "results": task.results,
            "created_at": task.created_at.isoformat(),
            "deadline": task.deadline.isoformat() if task.deadline else None
        }
        
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all registered agents"""
        status = {}
        for agent_id, agent in self.agent_registry.items():
            # Check if agent is busy with any active task
            busy = any(
                agent_id in task.assigned_agents 
                for task in self.active_tasks.values()
            )
            
            status[agent_id] = {
                "type": agent.__class__.__name__,
                "capabilities": self.agent_capabilities.get(agent_id, []),
                "status": "busy" if busy else "idle",
                "active_tasks": [
                    task.id for task in self.active_tasks.values()
                    if agent_id in task.assigned_agents
                ]
            }
            
        return status
        
    async def process_message_queue(self):
        """Process inter-agent messages"""
        while self.message_queue:
            message = self.message_queue.pop(0)
            
            if message.receiver == "broadcast":
                # Send to all agents
                for agent_id in self.agent_registry:
                    await self._deliver_message(agent_id, message)
            else:
                # Send to specific agent
                await self._deliver_message(message.receiver, message)
                
    async def _deliver_message(self, agent_id: str, message: AgentMessage):
        """Deliver message to specific agent"""
        agent = self.agent_registry.get(agent_id)
        if agent:
            # In real implementation, this would be async message passing
            self.logger.log(f"Delivered message from {message.sender} to {agent_id}")
            
    def generate_response(self, query: str) -> Tuple[str, str]:
        """Main entry point for orchestrator"""
        # Create task from query
        task = self.create_task(query)
        
        # For async execution, we'd return task ID and process in background
        # For now, return acknowledgment
        response = f"""
🎯 Task created! (ID: {task.id})

I'm orchestrating the AI agents to handle your request:
"{query}"

The task has been added to the queue and appropriate agents will be assigned.
You can check the status anytime by asking "What's the status of task {task.id}?"
"""
        
        return response, "orchestrator"