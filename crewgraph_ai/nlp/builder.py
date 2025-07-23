"""
Conversational Workflow Builder for CrewGraph AI

Interactive natural language workflow building with conversational interface.

Author: Vatsal216
Created: 2025-07-23 06:25:00 UTC
"""

import json
import threading
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from ..utils.logging import get_logger
from ..utils.metrics import get_metrics_collector
from .converter import NLToWorkflowConverter
from .parser import ParsedWorkflow, RequirementsParser, WorkflowParser

logger = get_logger(__name__)
metrics = get_metrics_collector()


class ConversationState(Enum):
    """States of the conversational workflow building process"""

    INITIAL = "initial"
    GATHERING_REQUIREMENTS = "gathering_requirements"
    CLARIFYING_TASKS = "clarifying_tasks"
    DEFINING_AGENTS = "defining_agents"
    SETTING_DEPENDENCIES = "setting_dependencies"
    REVIEWING_WORKFLOW = "reviewing_workflow"
    FINALIZING = "finalizing"
    COMPLETED = "completed"


class QuestionType(Enum):
    """Types of questions in the conversation"""

    OPEN_ENDED = "open_ended"
    YES_NO = "yes_no"
    MULTIPLE_CHOICE = "multiple_choice"
    CLARIFICATION = "clarification"
    CONFIRMATION = "confirmation"


@dataclass
class ConversationMessage:
    """A message in the conversation"""

    role: str  # "system", "user", "assistant"
    content: str
    timestamp: float
    message_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class WorkflowQuestion:
    """A question to ask the user"""

    question: str
    question_type: QuestionType
    context: str
    options: Optional[List[str]] = None
    required: bool = True
    follow_up_questions: Optional[List[str]] = None


@dataclass
class ConversationSession:
    """A conversational workflow building session"""

    session_id: str
    state: ConversationState
    messages: List[ConversationMessage]
    workflow_data: Dict[str, Any]
    pending_questions: List[WorkflowQuestion]
    created_at: float
    updated_at: float
    created_by: str = "Vatsal216"


class ConversationalWorkflowBuilder:
    """
    Interactive conversational interface for building workflows.

    Guides users through natural language conversations to build
    complete workflow definitions step by step.

    Created by: Vatsal216
    Date: 2025-07-23 06:25:00 UTC
    """

    def __init__(self):
        """Initialize conversational workflow builder"""
        self.requirements_parser = RequirementsParser()
        self.workflow_parser = WorkflowParser()
        self.nl_converter = NLToWorkflowConverter()

        self._active_sessions: Dict[str, ConversationSession] = {}
        self._conversation_templates = self._build_conversation_templates()

        self._lock = threading.RLock()

        logger.info("ConversationalWorkflowBuilder initialized")
        logger.info("User: Vatsal216, Time: 2025-07-23 06:25:00")

    def start_conversation(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Start a new conversational workflow building session.

        Args:
            session_id: Optional session ID (generated if not provided)

        Returns:
            Session information and initial message
        """
        with self._lock:
            if session_id is None:
                session_id = f"session_{int(time.time())}_{id(self) % 10000}"

            # Create new session
            session = ConversationSession(
                session_id=session_id,
                state=ConversationState.INITIAL,
                messages=[],
                workflow_data={},
                pending_questions=[],
                created_at=time.time(),
                updated_at=time.time(),
            )

            # Add welcome message
            welcome_msg = ConversationMessage(
                role="assistant",
                content=self._get_welcome_message(),
                timestamp=time.time(),
                message_type="welcome",
            )
            session.messages.append(welcome_msg)

            # Prepare initial questions
            initial_questions = self._generate_initial_questions()
            session.pending_questions.extend(initial_questions)
            session.state = ConversationState.GATHERING_REQUIREMENTS

            self._active_sessions[session_id] = session

            metrics.record_metric("conversation_sessions_started_total", 1.0)

            logger.info(f"Started conversation session: {session_id}")

            return {
                "session_id": session_id,
                "message": welcome_msg.content,
                "next_question": (
                    session.pending_questions[0].question if session.pending_questions else None
                ),
                "state": session.state.value,
                "created_by": "Vatsal216",
            }

    def continue_conversation(self, session_id: str, user_input: str) -> Dict[str, Any]:
        """
        Continue the conversation with user input.

        Args:
            session_id: Session ID
            user_input: User's response

        Returns:
            Assistant response and next steps
        """
        with self._lock:
            if session_id not in self._active_sessions:
                return {
                    "error": "Session not found",
                    "message": "Please start a new conversation session.",
                }

            session = self._active_sessions[session_id]

            # Add user message
            user_msg = ConversationMessage(role="user", content=user_input, timestamp=time.time())
            session.messages.append(user_msg)

            # Process user input based on current state
            response = self._process_user_input(session, user_input)

            # Add assistant response
            assistant_msg = ConversationMessage(
                role="assistant",
                content=response["message"],
                timestamp=time.time(),
                message_type=response.get("message_type"),
            )
            session.messages.append(assistant_msg)

            # Update session
            session.updated_at = time.time()

            metrics.record_metric("conversation_turns_total", 1.0)

            return response

    def get_conversation_history(self, session_id: str) -> Dict[str, Any]:
        """Get conversation history for a session"""
        with self._lock:
            if session_id not in self._active_sessions:
                return {"error": "Session not found"}

            session = self._active_sessions[session_id]

            return {
                "session_id": session_id,
                "state": session.state.value,
                "messages": [asdict(msg) for msg in session.messages],
                "workflow_data": session.workflow_data,
                "pending_questions": [asdict(q) for q in session.pending_questions],
                "created_at": session.created_at,
                "updated_at": session.updated_at,
            }

    def generate_workflow(self, session_id: str, format_type: str = "crewai") -> Dict[str, Any]:
        """
        Generate final workflow from conversation.

        Args:
            session_id: Session ID
            format_type: Output format ("crewai", "langgraph", "generic")

        Returns:
            Generated workflow definition
        """
        with self._lock:
            if session_id not in self._active_sessions:
                return {"error": "Session not found"}

            session = self._active_sessions[session_id]

            # Compile conversation into requirements text
            requirements_text = self._compile_requirements_from_conversation(session)

            # Convert to workflow
            workflow_def = self.nl_converter.convert_to_workflow(requirements_text, format_type)

            # Add conversation metadata
            workflow_def["conversation_metadata"] = {
                "session_id": session_id,
                "conversation_length": len(session.messages),
                "final_state": session.state.value,
                "created_by": "Vatsal216",
                "generated_at": time.time(),
            }

            # Mark session as completed
            session.state = ConversationState.COMPLETED
            session.updated_at = time.time()

            metrics.record_metric("workflows_generated_from_conversation_total", 1.0)

            logger.info(f"Generated {format_type} workflow from conversation {session_id}")

            return workflow_def

    def _process_user_input(self, session: ConversationSession, user_input: str) -> Dict[str, Any]:
        """Process user input based on current conversation state"""

        if session.state == ConversationState.GATHERING_REQUIREMENTS:
            return self._process_requirements_input(session, user_input)
        elif session.state == ConversationState.CLARIFYING_TASKS:
            return self._process_task_clarification(session, user_input)
        elif session.state == ConversationState.DEFINING_AGENTS:
            return self._process_agent_definition(session, user_input)
        elif session.state == ConversationState.SETTING_DEPENDENCIES:
            return self._process_dependency_input(session, user_input)
        elif session.state == ConversationState.REVIEWING_WORKFLOW:
            return self._process_workflow_review(session, user_input)
        elif session.state == ConversationState.FINALIZING:
            return self._process_finalization(session, user_input)
        else:
            return {
                "message": "I'm not sure how to help with that. Let's start over.",
                "next_question": None,
                "state": session.state.value,
            }

    def _process_requirements_input(
        self, session: ConversationSession, user_input: str
    ) -> Dict[str, Any]:
        """Process input during requirements gathering phase"""

        # Parse the input for workflow requirements
        parsed_data = self.requirements_parser.parse_requirements(user_input)

        # Store parsed data
        session.workflow_data.update(parsed_data)

        # Determine next steps based on what was parsed
        if not parsed_data.get("tasks"):
            # Need more task information
            session.state = ConversationState.CLARIFYING_TASKS
            questions = self._generate_task_clarification_questions(parsed_data)
            session.pending_questions = questions

            return {
                "message": "I understand your requirements. Let me ask some questions to clarify the specific tasks needed.",
                "next_question": questions[0].question if questions else None,
                "state": session.state.value,
                "message_type": "transition",
            }

        elif not parsed_data.get("agents") or len(parsed_data["agents"]) == 0:
            # Need agent information
            session.state = ConversationState.DEFINING_AGENTS
            questions = self._generate_agent_definition_questions(parsed_data)
            session.pending_questions = questions

            return {
                "message": "Great! Now let's define the agents that will execute these tasks.",
                "next_question": questions[0].question if questions else None,
                "state": session.state.value,
                "message_type": "transition",
            }

        else:
            # Move to dependency setting
            session.state = ConversationState.SETTING_DEPENDENCIES
            questions = self._generate_dependency_questions(parsed_data)
            session.pending_questions = questions

            return {
                "message": "Excellent! Now let's determine how these tasks depend on each other.",
                "next_question": questions[0].question if questions else None,
                "state": session.state.value,
                "message_type": "transition",
            }

    def _process_task_clarification(
        self, session: ConversationSession, user_input: str
    ) -> Dict[str, Any]:
        """Process input during task clarification phase"""

        # Remove current question
        if session.pending_questions:
            current_question = session.pending_questions.pop(0)

            # Process the answer
            self._process_task_answer(session, current_question, user_input)

        # Check if more questions remain
        if session.pending_questions:
            return {
                "message": "Thank you for that clarification.",
                "next_question": session.pending_questions[0].question,
                "state": session.state.value,
                "message_type": "continuation",
            }
        else:
            # Move to next phase
            session.state = ConversationState.DEFINING_AGENTS
            questions = self._generate_agent_definition_questions(session.workflow_data)
            session.pending_questions = questions

            return {
                "message": "Perfect! Now I understand the tasks. Let's define the agents needed.",
                "next_question": questions[0].question if questions else None,
                "state": session.state.value,
                "message_type": "transition",
            }

    def _process_agent_definition(
        self, session: ConversationSession, user_input: str
    ) -> Dict[str, Any]:
        """Process input during agent definition phase"""

        # Remove current question
        if session.pending_questions:
            current_question = session.pending_questions.pop(0)

            # Process the answer
            self._process_agent_answer(session, current_question, user_input)

        # Check if more questions remain
        if session.pending_questions:
            return {
                "message": "Got it.",
                "next_question": session.pending_questions[0].question,
                "state": session.state.value,
                "message_type": "continuation",
            }
        else:
            # Move to dependencies
            session.state = ConversationState.SETTING_DEPENDENCIES
            questions = self._generate_dependency_questions(session.workflow_data)
            session.pending_questions = questions

            return {
                "message": "Great! Now let's set up the task dependencies.",
                "next_question": questions[0].question if questions else None,
                "state": session.state.value,
                "message_type": "transition",
            }

    def _process_dependency_input(
        self, session: ConversationSession, user_input: str
    ) -> Dict[str, Any]:
        """Process input during dependency setting phase"""

        # Remove current question
        if session.pending_questions:
            current_question = session.pending_questions.pop(0)

            # Process the answer
            self._process_dependency_answer(session, current_question, user_input)

        # Check if more questions remain
        if session.pending_questions:
            return {
                "message": "Understood.",
                "next_question": session.pending_questions[0].question,
                "state": session.state.value,
                "message_type": "continuation",
            }
        else:
            # Move to review
            session.state = ConversationState.REVIEWING_WORKFLOW
            workflow_summary = self._generate_workflow_summary(session.workflow_data)

            return {
                "message": f"Here's a summary of your workflow:\n\n{workflow_summary}\n\nDoes this look correct? Any changes needed?",
                "next_question": None,
                "state": session.state.value,
                "message_type": "review",
            }

    def _process_workflow_review(
        self, session: ConversationSession, user_input: str
    ) -> Dict[str, Any]:
        """Process input during workflow review phase"""

        user_input_lower = user_input.lower()

        if any(word in user_input_lower for word in ["yes", "looks good", "correct", "approve"]):
            # User approves the workflow
            session.state = ConversationState.FINALIZING

            return {
                "message": "Excellent! Your workflow is ready. What format would you like? (CrewAI, LangGraph, or Generic)",
                "next_question": None,
                "state": session.state.value,
                "message_type": "finalization",
            }

        elif any(word in user_input_lower for word in ["no", "change", "modify", "different"]):
            # User wants changes
            return {
                "message": "What changes would you like to make? Please describe what needs to be modified.",
                "next_question": None,
                "state": session.state.value,
                "message_type": "modification_request",
            }

        else:
            # Process as modification request
            self._process_workflow_modifications(session, user_input)

            # Generate new summary
            workflow_summary = self._generate_workflow_summary(session.workflow_data)

            return {
                "message": f"I've updated the workflow based on your feedback:\n\n{workflow_summary}\n\nIs this better?",
                "next_question": None,
                "state": session.state.value,
                "message_type": "revised_review",
            }

    def _process_finalization(
        self, session: ConversationSession, user_input: str
    ) -> Dict[str, Any]:
        """Process input during finalization phase"""

        user_input_lower = user_input.lower()

        if "crewai" in user_input_lower:
            format_choice = "crewai"
        elif "langgraph" in user_input_lower:
            format_choice = "langgraph"
        elif "generic" in user_input_lower:
            format_choice = "generic"
        else:
            format_choice = "crewai"  # Default

        session.workflow_data["preferred_format"] = format_choice
        session.state = ConversationState.COMPLETED

        return {
            "message": f"Perfect! Your {format_choice} workflow is ready to be generated. Use the generate_workflow() method to get your final workflow definition.",
            "next_question": None,
            "state": session.state.value,
            "message_type": "completion",
            "workflow_ready": True,
            "format": format_choice,
        }

    def _generate_initial_questions(self) -> List[WorkflowQuestion]:
        """Generate initial questions to start the conversation"""
        return [
            WorkflowQuestion(
                question="What would you like your workflow to accomplish? Please describe your goals and what you want to automate.",
                question_type=QuestionType.OPEN_ENDED,
                context="gathering_objectives",
                required=True,
                follow_up_questions=[
                    "What data or inputs will your workflow work with?",
                    "What should be the final output or result?",
                ],
            )
        ]

    def _generate_task_clarification_questions(
        self, workflow_data: Dict[str, Any]
    ) -> List[WorkflowQuestion]:
        """Generate questions to clarify tasks"""
        questions = []

        objectives = workflow_data.get("objectives", [])

        if objectives:
            for i, objective in enumerate(objectives):
                questions.append(
                    WorkflowQuestion(
                        question=f"For the goal '{objective}', what specific steps or tasks are needed?",
                        question_type=QuestionType.OPEN_ENDED,
                        context=f"clarifying_objective_{i}",
                        required=True,
                    )
                )
        else:
            questions.append(
                WorkflowQuestion(
                    question="What are the main steps or tasks that need to be performed in your workflow?",
                    question_type=QuestionType.OPEN_ENDED,
                    context="clarifying_general_tasks",
                    required=True,
                )
            )

        questions.append(
            WorkflowQuestion(
                question="Are there any tasks that require special tools or capabilities?",
                question_type=QuestionType.OPEN_ENDED,
                context="clarifying_special_requirements",
                required=False,
            )
        )

        return questions

    def _generate_agent_definition_questions(
        self, workflow_data: Dict[str, Any]
    ) -> List[WorkflowQuestion]:
        """Generate questions to define agents"""
        questions = []

        tasks = workflow_data.get("tasks", [])

        if tasks:
            # Group tasks by type to suggest agents
            task_types = set()
            for task_dict in tasks:
                if isinstance(task_dict, dict):
                    task_type = task_dict.get("task_type", "unknown")
                    task_types.add(task_type)

            questions.append(
                WorkflowQuestion(
                    question=f"Based on your tasks, I suggest creating specialized agents. Would you like agents for: {', '.join(task_types)}?",
                    question_type=QuestionType.YES_NO,
                    context="suggesting_agent_types",
                    required=False,
                )
            )

        questions.append(
            WorkflowQuestion(
                question="How many agents would you prefer to have? (More agents can work in parallel but add complexity)",
                question_type=QuestionType.OPEN_ENDED,
                context="agent_count_preference",
                required=False,
            )
        )

        questions.append(
            WorkflowQuestion(
                question="Should agents be able to delegate tasks to each other, or work independently?",
                question_type=QuestionType.MULTIPLE_CHOICE,
                context="agent_delegation_preference",
                options=["Independent work only", "Allow delegation", "Mixed approach"],
                required=False,
            )
        )

        return questions

    def _generate_dependency_questions(
        self, workflow_data: Dict[str, Any]
    ) -> List[WorkflowQuestion]:
        """Generate questions about task dependencies"""
        questions = []

        tasks = workflow_data.get("tasks", [])

        if len(tasks) > 1:
            questions.append(
                WorkflowQuestion(
                    question="Should the tasks run one after another (sequential) or can some run at the same time (parallel)?",
                    question_type=QuestionType.MULTIPLE_CHOICE,
                    context="execution_pattern",
                    options=["Sequential only", "Parallel where possible", "Mixed approach"],
                    required=True,
                )
            )

            questions.append(
                WorkflowQuestion(
                    question="Are there any tasks that must complete before others can start? If so, which ones?",
                    question_type=QuestionType.OPEN_ENDED,
                    context="specific_dependencies",
                    required=False,
                )
            )

        questions.append(
            WorkflowQuestion(
                question="What should happen if a task fails? Should the workflow stop or continue with other tasks?",
                question_type=QuestionType.MULTIPLE_CHOICE,
                context="error_handling",
                options=["Stop entire workflow", "Continue with other tasks", "Retry failed tasks"],
                required=False,
            )
        )

        return questions

    def _process_task_answer(
        self, session: ConversationSession, question: WorkflowQuestion, answer: str
    ):
        """Process answer to a task-related question"""

        if question.context == "clarifying_general_tasks":
            # Parse the answer for task information
            parsed_tasks = self.requirements_parser.parse_requirements(answer)
            if "tasks" in parsed_tasks:
                if "tasks" not in session.workflow_data:
                    session.workflow_data["tasks"] = []
                session.workflow_data["tasks"].extend(parsed_tasks["tasks"])

        elif question.context.startswith("clarifying_objective_"):
            # Add tasks for specific objective
            parsed_tasks = self.requirements_parser.parse_requirements(answer)
            if "tasks" in parsed_tasks:
                if "tasks" not in session.workflow_data:
                    session.workflow_data["tasks"] = []
                session.workflow_data["tasks"].extend(parsed_tasks["tasks"])

        elif question.context == "clarifying_special_requirements":
            # Store special requirements
            if "special_requirements" not in session.workflow_data:
                session.workflow_data["special_requirements"] = []
            session.workflow_data["special_requirements"].append(answer)

    def _process_agent_answer(
        self, session: ConversationSession, question: WorkflowQuestion, answer: str
    ):
        """Process answer to an agent-related question"""

        if question.context == "suggesting_agent_types":
            # Store agent type preferences
            session.workflow_data["agent_preferences"] = answer

        elif question.context == "agent_count_preference":
            # Extract number preference
            try:
                import re

                numbers = re.findall(r"\d+", answer)
                if numbers:
                    session.workflow_data["preferred_agent_count"] = int(numbers[0])
            except:
                pass

        elif question.context == "agent_delegation_preference":
            # Store delegation preference
            session.workflow_data["delegation_preference"] = answer

    def _process_dependency_answer(
        self, session: ConversationSession, question: WorkflowQuestion, answer: str
    ):
        """Process answer to a dependency-related question"""

        if question.context == "execution_pattern":
            session.workflow_data["execution_pattern"] = answer

        elif question.context == "specific_dependencies":
            session.workflow_data["specific_dependencies"] = answer

        elif question.context == "error_handling":
            session.workflow_data["error_handling_preference"] = answer

    def _process_workflow_modifications(
        self, session: ConversationSession, modification_request: str
    ):
        """Process user's workflow modification requests"""

        # Parse modification request
        parsed_changes = self.requirements_parser.parse_requirements(modification_request)

        # Apply changes to workflow data
        for key, value in parsed_changes.items():
            if key in ["tasks", "agents"] and isinstance(value, list):
                # Merge lists
                if key not in session.workflow_data:
                    session.workflow_data[key] = []
                session.workflow_data[key].extend(value)
            else:
                # Replace or add
                session.workflow_data[key] = value

    def _generate_workflow_summary(self, workflow_data: Dict[str, Any]) -> str:
        """Generate a human-readable summary of the workflow"""

        summary_parts = []

        # Workflow name and description
        name = workflow_data.get("workflow_name", "Your Workflow")
        description = workflow_data.get("description", "Custom workflow")
        summary_parts.append(f"**{name}**: {description}")

        # Objectives
        objectives = workflow_data.get("objectives", [])
        if objectives:
            summary_parts.append(f"**Goals**: {', '.join(objectives)}")

        # Tasks
        tasks = workflow_data.get("tasks", [])
        if tasks:
            summary_parts.append(f"**Tasks** ({len(tasks)} total):")
            for i, task in enumerate(tasks[:5], 1):  # Show first 5 tasks
                if isinstance(task, dict):
                    task_name = task.get("name", f"Task {i}")
                    task_desc = task.get("description", "No description")
                    summary_parts.append(f"  {i}. {task_name}: {task_desc}")

            if len(tasks) > 5:
                summary_parts.append(f"  ... and {len(tasks) - 5} more tasks")

        # Agents
        agents = workflow_data.get("agents", [])
        if agents:
            summary_parts.append(f"**Agents** ({len(agents)} total):")
            for i, agent in enumerate(agents, 1):
                if isinstance(agent, dict):
                    agent_name = agent.get("name", f"Agent {i}")
                    agent_role = agent.get("role", "general")
                    summary_parts.append(f"  {i}. {agent_name} ({agent_role})")

        # Dependencies
        dependencies = workflow_data.get("dependencies", [])
        if dependencies:
            summary_parts.append(f"**Dependencies**: {len(dependencies)} task dependencies defined")

        # Execution preferences
        execution_pattern = workflow_data.get("execution_pattern")
        if execution_pattern:
            summary_parts.append(f"**Execution**: {execution_pattern}")

        return "\n".join(summary_parts)

    def _compile_requirements_from_conversation(self, session: ConversationSession) -> str:
        """Compile conversation into requirements text"""

        requirements_parts = []

        # Extract user messages that contain requirements
        for message in session.messages:
            if message.role == "user" and len(message.content) > 20:
                requirements_parts.append(message.content)

        # Add structured data as text
        workflow_data = session.workflow_data

        if workflow_data.get("objectives"):
            objectives_text = "The objectives are: " + ", ".join(workflow_data["objectives"])
            requirements_parts.append(objectives_text)

        if workflow_data.get("execution_pattern"):
            pattern_text = f"Execution pattern: {workflow_data['execution_pattern']}"
            requirements_parts.append(pattern_text)

        if workflow_data.get("error_handling_preference"):
            error_text = f"Error handling: {workflow_data['error_handling_preference']}"
            requirements_parts.append(error_text)

        return ". ".join(requirements_parts)

    def _get_welcome_message(self) -> str:
        """Get the welcome message for new conversations"""
        return """
Welcome to the CrewGraph AI Conversational Workflow Builder! 🤖

I'm here to help you create powerful AI workflows through natural conversation. 
I'll guide you step-by-step to understand your requirements and build a complete workflow definition.

This process typically involves:
1. Understanding your goals and objectives
2. Defining the specific tasks needed
3. Setting up the right agents for each job
4. Organizing task dependencies and flow
5. Reviewing and finalizing your workflow

Let's get started!
        """.strip()

    def _build_conversation_templates(self) -> Dict[str, Dict]:
        """Build conversation templates for different scenarios"""
        return {
            "data_processing": {
                "welcome": "I see you want to build a data processing workflow. Let's start with understanding your data sources and transformation needs.",
                "follow_up_questions": [
                    "What type of data are you working with?",
                    "Where does the data come from?",
                    "What transformations are needed?",
                    "Where should the processed data go?",
                ],
            },
            "analysis": {
                "welcome": "Great! You want to build an analysis workflow. Let's understand what you want to analyze and what insights you need.",
                "follow_up_questions": [
                    "What data will you be analyzing?",
                    "What kind of analysis do you need?",
                    "What questions are you trying to answer?",
                    "How should the results be presented?",
                ],
            },
            "automation": {
                "welcome": "Perfect! You're looking to automate a process. Let's break down the manual steps and see how to automate them.",
                "follow_up_questions": [
                    "What process do you want to automate?",
                    "What are the current manual steps?",
                    "What triggers should start the automation?",
                    "What should happen when it's complete?",
                ],
            },
        }

    def get_active_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all active sessions"""
        with self._lock:
            sessions_info = {}
            for session_id, session in self._active_sessions.items():
                sessions_info[session_id] = {
                    "state": session.state.value,
                    "message_count": len(session.messages),
                    "created_at": session.created_at,
                    "updated_at": session.updated_at,
                    "has_workflow_data": bool(session.workflow_data),
                    "pending_questions": len(session.pending_questions),
                }
            return sessions_info

    def close_session(self, session_id: str) -> bool:
        """Close and remove a conversation session"""
        with self._lock:
            if session_id in self._active_sessions:
                del self._active_sessions[session_id]
                logger.info(f"Closed conversation session: {session_id}")
                return True
            return False

    def get_conversation_statistics(self) -> Dict[str, Any]:
        """Get statistics about conversation sessions"""
        with self._lock:
            total_sessions = len(self._active_sessions)

            if total_sessions == 0:
                return {"total_active_sessions": 0}

            # Calculate statistics
            states = {}
            avg_messages = 0
            avg_duration = 0
            current_time = time.time()

            for session in self._active_sessions.values():
                state = session.state.value
                states[state] = states.get(state, 0) + 1
                avg_messages += len(session.messages)
                avg_duration += current_time - session.created_at

            avg_messages /= total_sessions
            avg_duration /= total_sessions

            return {
                "total_active_sessions": total_sessions,
                "states_distribution": states,
                "avg_messages_per_session": avg_messages,
                "avg_session_duration_minutes": avg_duration / 60,
                "created_by": "Vatsal216",
                "timestamp": time.time(),
            }
