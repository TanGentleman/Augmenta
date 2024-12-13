from typing import Literal, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph.state import CompiledStateGraph
import logging

from .graph_classes import GraphState, Config, AgentState, Task, Command, CommandType, TaskStatus
from .utils.message_utils import insert_system_message, remove_last_message, clear_messages
from .utils.action_utils import execute_action

# Constants
RECURSION_LIMIT = 50
MAX_MUTATIONS = 50
MAX_ACTIONS = 5

def execute_command_node(state: GraphState) -> GraphState:
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
        clear_messages(state_dict["messages"])
        
    elif cmd_type == CommandType.SETTINGS:
        print("\nCurrent Settings:")
        print(state_dict["config"])
        
    elif cmd_type == CommandType.SAVE:
        # Implement save state logic
        print("Saving state...")
        print("\nMessages:")
        for message in state_dict["messages"]:
            if isinstance(message, HumanMessage):
                print(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                print(f"AI: {message.content}")
        
    elif cmd_type == CommandType.LOAD:
        # Implement load state logic
        print("Loading state...")
        
    elif cmd_type == CommandType.DEBUG:
        print("\nCurrent State:")
        print(f"Messages: {len(state_dict['messages'])}")
        print(f"Tasks: {state_dict['task_dict']}")
        
    elif cmd_type == CommandType.RETRY:
        remove_last_message(state_dict["messages"])
        
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
        "action_count": 0,
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
        insert_system_message(
            state_dict["messages"],
            state_dict["config"].chat_settings.system_message
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

def decide_from_processor(state: GraphState) -> Literal["execute_command", "agent_node"]:
    """Routes from processor based on input type"""
    return "execute_command" if state["keys"]["user_input"].startswith('/') else "agent_node"

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
    
    if result["success"]:
        logging.info(f"Action succeeded: {result['data']}")
        state_dict["action_count"] += 1
        if state_dict["action_count"] > MAX_ACTIONS:
            logging.error("Action count exceeded max actions, marking task as failed")
            current_task["status"] = TaskStatus.FAILED
    else:
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
    workflow.add_node("execute_command", execute_command_node)
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
        decide_from_processor,
        {
            "execute_command": "execute_command",
            "agent_node": "agent_node"
        }
    )
    workflow.add_edge("execute_command", "agent_node")
    workflow.add_edge("action_node", "agent_node")
    workflow.add_edge("task_manager", "agent_node")
    
    app = workflow.compile()
    return app

def main():
    """Main execution function"""
    app = create_workflow()
    
    initial_state: GraphState = {
        "keys": {},
        "mutation_count": 0,
        "is_done": False
    }

    app_config = {"recursion_limit": RECURSION_LIMIT}
    
    for output in app.stream(initial_state, app_config):
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
    VERSION = "v3"
    filename = f"agent_graph_{VERSION}.png"
    try:
        from paths import AGENTS_DIR
        file_path = AGENTS_DIR / "graph_mermaids" / filename
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