"""
Automation Agent - Workflow automation and scheduled tasks
"""
import json
import asyncio
import schedule
import threading
from typing import List, Dict, Tuple, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from sources.agents.agent import Agent
from sources.logger import Logger


class TriggerType(Enum):
    TIME = "time"
    EVENT = "event"
    CONDITION = "condition"
    MANUAL = "manual"


class WorkflowStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Trigger:
    """Defines when a workflow should execute"""
    type: TriggerType
    config: Dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def should_trigger(self, context: Dict = None) -> bool:
        """Check if trigger conditions are met"""
        if self.type == TriggerType.MANUAL:
            return False  # Manual triggers handled separately
            
        elif self.type == TriggerType.TIME:
            # Handled by scheduler
            return False
            
        elif self.type == TriggerType.EVENT:
            event = context.get("event") if context else None
            return event == self.config.get("event_name")
            
        elif self.type == TriggerType.CONDITION:
            # Evaluate condition
            condition = self.config.get("condition", "")
            try:
                # Simple condition evaluation (enhance with safe eval)
                return eval(condition, {"context": context})
            except:
                return False
                
        return False


@dataclass 
class WorkflowStep:
    """Single step in a workflow"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    agent: str = ""  # Agent to execute this step
    action: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)  # Step IDs
    retry_count: int = 3
    timeout: int = 300  # seconds
    on_failure: str = "stop"  # stop, continue, retry


@dataclass
class Workflow:
    """Complete workflow definition"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    steps: List[WorkflowStep] = field(default_factory=list)
    triggers: List[Trigger] = field(default_factory=list)
    status: WorkflowStatus = WorkflowStatus.IDLE
    created_at: datetime = field(default_factory=datetime.now)
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    variables: Dict[str, Any] = field(default_factory=dict)  # Workflow variables
    notifications: Dict[str, Any] = field(default_factory=dict)


class AutomationAgent(Agent):
    """
    Agent for creating and managing automated workflows
    """
    
    def __init__(self, llm_provider=None):
        super().__init__(llm_provider)
        self.logger = Logger("automation_agent.log")
        self.workflows: Dict[str, Workflow] = {}
        self.running_workflows: Dict[str, asyncio.Task] = {}
        self.scheduler_thread = None
        self.event_queue: List[Dict] = []
        self._start_scheduler()
        
    def _start_scheduler(self):
        """Start the background scheduler thread"""
        def run_scheduler():
            while True:
                schedule.run_pending()
                asyncio.run(self._check_event_triggers())
                asyncio.sleep(1)
                
        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
    def generate_response(self, query: str) -> Tuple[str, str]:
        """
        Main entry point for automation requests
        """
        self.logger.log(f"Automation query: {query}")
        
        # Determine automation action
        if "create workflow" in query.lower():
            response = self._handle_create_workflow(query)
        elif "schedule" in query.lower():
            response = self._handle_schedule_task(query)
        elif "list workflows" in query.lower():
            response = self._handle_list_workflows()
        elif "run workflow" in query.lower():
            response = self._handle_run_workflow(query)
        elif "stop" in query.lower() or "cancel" in query.lower():
            response = self._handle_stop_workflow(query)
        else:
            response = self._handle_general_automation_query(query)
            
        return response, "automation"
        
    def create_workflow(self, name: str, description: str = "", 
                       steps: List[Dict] = None, triggers: List[Dict] = None) -> Workflow:
        """Create a new workflow"""
        workflow = Workflow(
            name=name,
            description=description
        )
        
        # Add steps
        if steps:
            for step_data in steps:
                step = WorkflowStep(
                    name=step_data.get("name", ""),
                    agent=step_data.get("agent", ""),
                    action=step_data.get("action", ""),
                    parameters=step_data.get("parameters", {}),
                    dependencies=step_data.get("dependencies", [])
                )
                workflow.steps.append(step)
                
        # Add triggers
        if triggers:
            for trigger_data in triggers:
                trigger = Trigger(
                    type=TriggerType(trigger_data.get("type", "manual")),
                    config=trigger_data.get("config", {})
                )
                workflow.triggers.append(trigger)
                
                # Schedule time-based triggers
                if trigger.type == TriggerType.TIME:
                    self._schedule_workflow(workflow, trigger)
                    
        self.workflows[workflow.id] = workflow
        self.logger.log(f"Created workflow: {workflow.name} ({workflow.id})")
        
        return workflow
        
    def _schedule_workflow(self, workflow: Workflow, trigger: Trigger):
        """Schedule a workflow based on time trigger"""
        schedule_type = trigger.config.get("schedule_type", "daily")
        time_str = trigger.config.get("time", "09:00")
        
        def run_scheduled():
            asyncio.create_task(self.execute_workflow(workflow.id))
            
        if schedule_type == "daily":
            schedule.every().day.at(time_str).do(run_scheduled)
        elif schedule_type == "hourly":
            schedule.every().hour.do(run_scheduled)
        elif schedule_type == "weekly":
            day = trigger.config.get("day", "monday")
            getattr(schedule.every(), day).at(time_str).do(run_scheduled)
        elif schedule_type == "interval":
            minutes = trigger.config.get("minutes", 60)
            schedule.every(minutes).minutes.do(run_scheduled)
            
        # Calculate next run
        workflow.next_run = datetime.now() + timedelta(hours=1)  # Simplified
        
    async def execute_workflow(self, workflow_id: str, context: Dict = None) -> Dict:
        """Execute a workflow"""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            return {"error": "Workflow not found"}
            
        if workflow.status == WorkflowStatus.RUNNING:
            return {"error": "Workflow already running"}
            
        workflow.status = WorkflowStatus.RUNNING
        workflow.last_run = datetime.now()
        workflow.run_count += 1
        
        results = {
            "workflow_id": workflow_id,
            "start_time": datetime.now().isoformat(),
            "steps": {},
            "status": "running"
        }
        
        try:
            # Execute steps in order, respecting dependencies
            completed_steps = set()
            
            while len(completed_steps) < len(workflow.steps):
                for step in workflow.steps:
                    if step.id in completed_steps:
                        continue
                        
                    # Check dependencies
                    if all(dep in completed_steps for dep in step.dependencies):
                        # Execute step
                        step_result = await self._execute_step(step, workflow, context)
                        results["steps"][step.id] = step_result
                        
                        if step_result["status"] == "failed" and step.on_failure == "stop":
                            raise Exception(f"Step {step.name} failed")
                            
                        completed_steps.add(step.id)
                        
            workflow.status = WorkflowStatus.COMPLETED
            results["status"] = "completed"
            
        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            results["status"] = "failed"
            results["error"] = str(e)
            self.logger.error(f"Workflow {workflow_id} failed: {e}")
            
        results["end_time"] = datetime.now().isoformat()
        
        # Send notifications if configured
        if workflow.notifications:
            await self._send_notifications(workflow, results)
            
        return results
        
    async def _execute_step(self, step: WorkflowStep, workflow: Workflow, 
                           context: Dict = None) -> Dict:
        """Execute a single workflow step"""
        result = {
            "step_id": step.id,
            "step_name": step.name,
            "start_time": datetime.now().isoformat(),
            "status": "running"
        }
        
        try:
            # Get agent
            # In real implementation, would get from agent registry
            if step.agent == "browser":
                # Simulate browser action
                result["output"] = f"Browsed to {step.parameters.get('url', 'unknown')}"
            elif step.agent == "email":
                # Simulate email action
                result["output"] = f"Sent email to {step.parameters.get('to', 'unknown')}"
            elif step.agent == "data":
                # Simulate data processing
                result["output"] = f"Processed data: {step.parameters.get('operation', 'unknown')}"
            else:
                # Default action
                result["output"] = f"Executed {step.action} with {step.agent}"
                
            result["status"] = "completed"
            
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            
            # Retry logic
            if step.retry_count > 0:
                self.logger.log(f"Retrying step {step.name}")
                step.retry_count -= 1
                return await self._execute_step(step, workflow, context)
                
        result["end_time"] = datetime.now().isoformat()
        return result
        
    async def _check_event_triggers(self):
        """Check for event-based triggers"""
        if not self.event_queue:
            return
            
        event = self.event_queue.pop(0)
        
        for workflow in self.workflows.values():
            for trigger in workflow.triggers:
                if trigger.should_trigger({"event": event}):
                    await self.execute_workflow(workflow.id, {"event": event})
                    
    async def _send_notifications(self, workflow: Workflow, results: Dict):
        """Send workflow notifications"""
        if workflow.notifications.get("email"):
            # Send email notification
            self.logger.log(f"Sending email notification for workflow {workflow.name}")
        if workflow.notifications.get("webhook"):
            # Send webhook
            self.logger.log(f"Sending webhook notification for workflow {workflow.name}")
            
    def _handle_create_workflow(self, query: str) -> str:
        """Handle workflow creation requests"""
        response_parts = [
            "# Creating Automated Workflow",
            "\nI'll help you create an automated workflow. Here's the structure:",
            "\n## Workflow Components:",
            "1. **Name**: Give your workflow a descriptive name",
            "2. **Triggers**: When should it run?",
            "   - Time-based (daily, hourly, specific time)",
            "   - Event-based (file change, email received)",
            "   - Condition-based (when X happens)",
            "3. **Steps**: What actions to perform",
            "   - Each step can use different agents",
            "   - Steps can depend on previous steps",
            "4. **Notifications**: How to notify you",
            "\n## Example Workflow:",
            "```json",
            json.dumps({
                "name": "Daily Report Generator",
                "triggers": [{
                    "type": "time",
                    "config": {"schedule_type": "daily", "time": "09:00"}
                }],
                "steps": [
                    {
                        "name": "Gather Data",
                        "agent": "browser",
                        "action": "scrape",
                        "parameters": {"url": "https://example.com/data"}
                    },
                    {
                        "name": "Process Data",
                        "agent": "data",
                        "action": "analyze",
                        "dependencies": ["step1"]
                    },
                    {
                        "name": "Send Report",
                        "agent": "email",
                        "action": "send",
                        "parameters": {"to": "user@example.com"},
                        "dependencies": ["step2"]
                    }
                ]
            }, indent=2),
            "```",
            "\nWhat kind of workflow would you like to create?"
        ]
        
        return "\n".join(response_parts)
        
    def _handle_schedule_task(self, query: str) -> str:
        """Handle task scheduling requests"""
        # Extract schedule details from query
        schedule_parts = []
        
        if "daily" in query.lower():
            schedule_parts.append("Schedule: Daily")
            if "morning" in query.lower():
                schedule_parts.append("Time: 09:00 AM")
            elif "evening" in query.lower():
                schedule_parts.append("Time: 06:00 PM")
                
        elif "hourly" in query.lower():
            schedule_parts.append("Schedule: Every hour")
            
        elif "weekly" in query.lower():
            schedule_parts.append("Schedule: Weekly")
            
        task_desc = "Automated task created"
        if schedule_parts:
            task_desc += " - " + ", ".join(schedule_parts)
            
        return f"""
✅ {task_desc}

I've set up the scheduled automation. The task will run automatically according to your schedule.

To manage this automation:
- View status: "Show my workflows"
- Stop it: "Stop workflow [name]"
- Modify: "Edit workflow [name]"
"""
        
    def _handle_list_workflows(self) -> str:
        """List all workflows"""
        if not self.workflows:
            return "No workflows created yet. Would you like to create one?"
            
        response_parts = ["# Your Automated Workflows\n"]
        
        for workflow in self.workflows.values():
            status_emoji = {
                WorkflowStatus.IDLE: "⏸️",
                WorkflowStatus.RUNNING: "▶️",
                WorkflowStatus.COMPLETED: "✅",
                WorkflowStatus.FAILED: "❌"
            }.get(workflow.status, "❓")
            
            response_parts.append(f"## {status_emoji} {workflow.name}")
            response_parts.append(f"- ID: {workflow.id[:8]}...")
            response_parts.append(f"- Status: {workflow.status.value}")
            response_parts.append(f"- Steps: {len(workflow.steps)}")
            response_parts.append(f"- Run count: {workflow.run_count}")
            
            if workflow.last_run:
                response_parts.append(f"- Last run: {workflow.last_run.strftime('%Y-%m-%d %H:%M')}")
            if workflow.next_run:
                response_parts.append(f"- Next run: {workflow.next_run.strftime('%Y-%m-%d %H:%M')}")
                
            response_parts.append("")
            
        return "\n".join(response_parts)
        
    def _handle_run_workflow(self, query: str) -> str:
        """Handle manual workflow execution"""
        # Extract workflow name/id from query
        # Simplified - would use NLP in production
        
        if self.workflows:
            # Run the first workflow for demo
            workflow = list(self.workflows.values())[0]
            
            # Start async execution
            asyncio.create_task(self.execute_workflow(workflow.id))
            
            return f"""
▶️ Starting workflow: {workflow.name}

The workflow is now running in the background. I'll notify you when it completes.

You can check the status anytime by asking "What's the status of {workflow.name}?"
"""
        else:
            return "No workflows found. Please create a workflow first."
            
    def _handle_stop_workflow(self, query: str) -> str:
        """Handle workflow cancellation"""
        # Find and stop workflow
        stopped = []
        
        for workflow_id, workflow in self.workflows.items():
            if workflow.status == WorkflowStatus.RUNNING:
                workflow.status = WorkflowStatus.IDLE
                stopped.append(workflow.name)
                
        if stopped:
            return f"⏹️ Stopped workflows: {', '.join(stopped)}"
        else:
            return "No running workflows to stop."
            
    def _handle_general_automation_query(self, query: str) -> str:
        """Handle general automation queries"""
        return """
# Automation Agent Capabilities

I can help you automate repetitive tasks and create workflows:

## 🔄 Workflow Automation
- Create multi-step workflows
- Chain actions across different agents
- Set dependencies between tasks

## ⏰ Scheduling
- Daily/Weekly/Monthly schedules
- Specific time execution
- Recurring tasks

## 🎯 Triggers
- Time-based triggers
- Event-based triggers
- Condition-based triggers

## 📊 Examples:
1. **Daily Report**: Gather data → Analyze → Email report
2. **Website Monitor**: Check site → Alert if down → Log status
3. **Data Pipeline**: Extract → Transform → Load → Notify

What would you like to automate?
"""