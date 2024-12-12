from typing import Literal, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph.state import CompiledStateGraph
import logging

from .graph_classes import GraphState, Config, AgentState, Task, ActionType, ActionResult, Command, CommandType, TaskStatus

# Constants
MAX_MUTATIONS = 20
MAX_RESPONSES = 3

def process_command_node(state: GraphState) -> GraphState:
    """Process a command and update state"""
    state_dict = state["keys"]
    command = Command(state_dict["user_input"])
    
    if not command.is_valid:
        logging.warning(f"Invalid command: {command.command}")
        return {
            "keys": state_dict,
            "mutation_count": state["mutation_count"] + 1,
            "is_done": False
        }
    
    cmd_type = command.type
    
    if cmd_type == CommandType.QUIT:
        for task in state_dict["task_dict"].values():
            task["status"] = TaskStatus.DONE
            
    elif cmd_type == CommandType.HELP:
        print("\nAvailable Commands:")
        for cmd in CommandType:
            print(f"/{cmd.value}")
            
    elif cmd_type == CommandType.CLEAR:
        state_dict["messages"] = []
        
    elif cmd_type == CommandType.SETTINGS:
        print("\nCurrent Settings:")
        print(state_dict["config"])
        
    elif cmd_type == CommandType.SAVE:
        # Implement save state logic
        print("Saving state...")
        
    elif cmd_type == CommandType.LOAD:
        # Implement load state logic
        print("Loading state...")
        
    elif cmd_type == CommandType.DEBUG:
        print("\nCurrent State:")
        print(f"Messages: {len(state_dict['messages'])}")
        print(f"Tasks: {state_dict['task_dict']}")
        
    elif cmd_type == CommandType.RETRY:
        if state_dict["messages"]:
            state_dict["messages"].pop()
            
    elif cmd_type == CommandType.UNDO:
        if state_dict["messages"]:
            state_dict["messages"] = state_dict["messages"][:-2]
            
    elif cmd_type == CommandType.TOOLS:
        print("\nAvailable Tools:")
        # Implement tool listing logic
    else:
        print(f"Command not implemented: {command.command}")

    return {
        "keys": state_dict,
        "mutation_count": state["mutation_count"] + 1,
        "is_done": False
    }

def start_node(state: GraphState) -> GraphState:
    """Initialize the graph state with configuration and default task"""
    config = Config()
    default_task: Task = {
        "status": TaskStatus.IN_PROGRESS,
        "conditions": ["$"],
        "actions": []
    }
    
    initial_state: AgentState = {
        "config": config,
        "messages": [],
        "response_count": 0,
        "active_chain": None,
        "tool_choice": None,
        "task_dict": {"chat_task": default_task},
        "user_input": None
    }

    # NOTE: Right now this is replacing ALL initial graph state except for the mutation count
    return {
        "keys": initial_state,
        "mutation_count": state["mutation_count"] + 1,
        "is_done": False
    }

def agent_node(state: GraphState) -> GraphState:
    """Manages agent state and decision making"""
    state_dict = state["keys"]
    task_dict = state_dict["task_dict"]
    
    # Check failure conditions
    if state["mutation_count"] > MAX_MUTATIONS:
        logging.warning("Mutation count exceeded max mutations!")
        current_task = task_dict["chat_task"]
        if current_task["status"] == TaskStatus.IN_PROGRESS:
            logging.error("Task is still in progress, marking as failed")
            current_task["status"] = TaskStatus.FAILED
    
    # Handle initialization
    if state["mutation_count"] == 1 and state_dict["config"].chat_settings.enable_system_message:
        print("Adding system message")
        # TODO: Add assertions / checks on messages array?
        state_dict["messages"].append(
            SystemMessage(content=state_dict["config"].chat_settings.system_message)
        )

    return {
        "keys": state_dict,
        "mutation_count": state["mutation_count"] + 1,
        "is_done": state["is_done"]
    }

def task_manager_node(state: GraphState) -> GraphState:
    """Manages task lifecycle and completion"""
    state_dict = state["keys"]
    task_dict = state_dict["task_dict"]
    
    # Check for failed tasks
    failed_task = next((task for task in task_dict.values() if task["status"] == TaskStatus.FAILED), None)
    if failed_task:
        print("Error: Task failed")
        return {
            "keys": state_dict,
            "mutation_count": state["mutation_count"] + 1,
            "is_done": True
        }
    
    # Check task completion
    all_complete = all(task["status"] == TaskStatus.DONE for task in task_dict.values())
    if all_complete:
        # Save logic or task mutation logic here
        return {
            "keys": state_dict,
            "mutation_count": state["mutation_count"] + 1,
            "is_done": True
        }
    
    state_dict["task_dict"] = task_dict
    return {
        "keys": state_dict,
        "mutation_count": state["mutation_count"] + 1,
        "is_done": False
    }

def human_node(state: GraphState) -> GraphState:
    """Handles human input"""
    state_dict = state["keys"]
    
    user_input = input("Enter your message: ").strip()
    state_dict["user_input"] = user_input
    
    return {
        "keys": state_dict,
        "mutation_count": state["mutation_count"] + 1,
        "is_done": False
    }

def processor_node(state: GraphState) -> GraphState:
    """Processes input and updates state"""
    state_dict = state["keys"]
    user_input = state_dict["user_input"]
    
    # TODO: Add logic to determine if we are in a HITL task and should not append to messages
    if not user_input.startswith('/'):
        state_dict["messages"].append(HumanMessage(content=user_input))
        task_dict = state_dict["task_dict"]
        current_task = next((task for task in task_dict.values() if task["status"] == TaskStatus.IN_PROGRESS), None)
        if not current_task:
            raise ValueError("No in-progress task found")
        current_task["actions"].append("generate")
    
    return {
        "keys": state_dict,
        "mutation_count": state["mutation_count"] + 1,
        "is_done": False
    }

def decide_processor(state: GraphState) -> Literal["process_command", "agent_node"]:
    """Routes from processor based on input type"""
    return "process_command" if state["keys"]["user_input"].startswith('/') else "agent_node"

def execute_action(action: str, state_dict: Dict[str, Any]) -> ActionResult:
    """Execute a specific action and return the result."""
    # NOTE: Do NOT modify state_dict.
    try:
        # Parse action string to get type and parameters
        action_parts = action.split(":", 1)
        action_type = ActionType(action_parts[0])
        action_params = action_parts[1] if len(action_parts) > 1 else ""

        if action_type == ActionType.GENERATE:
            # Handle LLM generation
            print("Generating LLM response!")
            return {
                "success": True,
                "data": "Generated response",
                "error": None
            }
            
        elif action_type == ActionType.WEB_SEARCH:
            # Handle web search
            return {
                "success": True,
                "data": "Search results",
                "error": None
            }
            
        elif action_type == ActionType.SAVE_DATA:
            # Handle data persistence
            return {
                "success": True,
                "data": "Data saved",
                "error": None
            }
            
        elif action_type == ActionType.TOOL_CALL:
            # Handle tool calling
            return {
                "success": True,
                "data": "Tool called",
                "error": None
            }
            
        else:
            return {
                "success": False,
                "data": None,
                "error": f"Unknown action type: {action_type}"
            }
            
    except Exception as e:
        logging.error(f"Action execution failed: {str(e)}")
        return {
            "success": False,
            "data": None,
            "error": str(e)
        }

def action_node(state: GraphState) -> GraphState:
    """Executes in-progress actions in the current task"""
    state_dict = state["keys"]
    task_dict = state_dict["task_dict"]
    
    # Find current in-progress task
    current_task = next(
        (task for task in task_dict.values() if task["status"] == TaskStatus.IN_PROGRESS),
        None
    )
    if not current_task:
        raise ValueError("No in-progress task found")
    if not current_task["actions"]:
        raise ValueError("No actions found for in-progress task.")
    

    # Get next action and execute it
    action = current_task["actions"].pop(0)
    result = execute_action(action, state_dict)
    
    if not result["success"]:
        logging.error(f"Action failed: {result['error']}")
        # Optionally handle failure (retry, skip, or mark task as failed)
        current_task["status"] = TaskStatus.FAILED  # For now, we'll just mark it as done
    
    return {
        "keys": state_dict,
        "mutation_count": state["mutation_count"] + 1,
        "is_done": False
    }

# Decision functions
def decide_from_agent(state: GraphState) -> Literal["human_node", "task_manager", "action_node", "end_node"]:
    """Routes from agent node based on state"""
    if state["is_done"]:
        return "end_node"
    
    state_dict = state["keys"]
    task_dict = state_dict["task_dict"]
    
    # Check if tasks need management
    if not task_dict or all(task["status"] in [TaskStatus.DONE, TaskStatus.FAILED] for task in task_dict.values()):
        return "task_manager"
    
    # Check for pending actions
    current_task = next((task for task in task_dict.values() if task["status"] == TaskStatus.IN_PROGRESS), None)
    if current_task and current_task["actions"]:
        return "action_node"
    
    return "human_node"

# Build graph
def create_workflow() -> CompiledStateGraph:
    """Creates and returns the compiled workflow"""
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("start_node", start_node)
    workflow.add_node("agent_node", agent_node)
    workflow.add_node("task_manager", task_manager_node)
    workflow.add_node("human_node", human_node)
    workflow.add_node("processor_node", processor_node)
    workflow.add_node("process_command", process_command_node)
    workflow.add_node("action_node", action_node)
    
    # Add edges
    workflow.set_entry_point("start_node")
    workflow.add_edge("start_node", "agent_node")
    
    workflow.add_conditional_edges(
        "agent_node",
        decide_from_agent,
        {
            "human_node": "human_node",
            "task_manager": "task_manager",
            "action_node": "action_node",
            "end_node": END
        }
    )
    
    workflow.add_edge("human_node", "processor_node")
    workflow.add_conditional_edges(
        "processor_node",
        decide_processor,
        {
            "process_command": "process_command",
            "agent_node": "agent_node"
        }
    )
    workflow.add_edge("process_command", "agent_node")
    workflow.add_edge("action_node", "agent_node")
    workflow.add_edge("task_manager", "agent_node")
    
    app = workflow.compile()
    return app

def main():
    """Main execution function"""
    app = create_workflow()
    
    initial_state: GraphState = {
        "keys": {},
        "mutation_count": 19,
        "is_done": False
    }
    
    for output in app.stream(initial_state):
        for key, value in output.items():
            print(f"\nNode: {key}")
            print(f"Mutations: {value['mutation_count']}")
            
            # Debug task state
            if 'task_dict' in value['keys']:
                print("\nTask Status:")
                for task_name, task in value['keys']['task_dict'].items():
                    print(f"{task_name}: {task['status']}")
        
        print("\n---\n")
    
    print("Workflow completed.")

def save_graph():
    # app = workflow.compile()
    file_path = "graph_mermaids/agent_graph.png"
    try:
        app = create_workflow()
        app.get_graph().draw_mermaid_png(output_file_path=file_path)
        print(f"Graph saved to {file_path}")
    except Exception as e:
        print("Failed to save graph")
        print(e)
        pass

if __name__ == "__main__":
    main()
    # save_graph()