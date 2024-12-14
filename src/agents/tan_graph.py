"""
This module implements a state machine graph for managing conversational AI interactions.

Graph Flow:
1. START -> start_node: Initializes graph state and configuration
2. start_node -> agent_node: Central node that manages agent state and decisions
3. agent_node branches to:
   - human_node: Handles user input collection
   - task_manager: Manages task lifecycle and completion
   - action_node: Executes actions like generating responses
   - END: Terminates workflow when complete
4. human_node -> processor_node: Processes user input
5. processor_node branches to:
   - execute_command: Handles system commands (prefixed with /)
   - agent_node: Continues normal conversation flow
6. execute_command -> agent_node: Returns to main loop after command
7. action_node -> agent_node: Returns after executing actions
8. task_manager -> agent_node: Returns after task management

The graph implements a flexible conversation loop with command processing,
task management, and action execution capabilities.
"""

from typing import Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph.state import CompiledStateGraph
import logging

from agents.template import INITIAL_STATE_DICT
from agents.utils.chains import get_summary_chain, get_llm
from agents.utils.task_utils import get_task, save_completed_tasks, save_failed_tasks, start_next_task
from augmenta.models.models import LLM, LLM_FN
from augmenta.utils import read_sample

from .graph_classes import GraphState, Command, CommandType, TaskStatus, TaskType
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
        # TODO: Change the status here to Quit
        # Task manager will decide in-progress, done, or failed
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

        
    elif cmd_type == CommandType.UNDO:
        if state_dict["messages"]:
            remove_last_message(state_dict["messages"])

    elif cmd_type == CommandType.MODE:
        current_task = get_task(state_dict["task_dict"], status=TaskStatus.IN_PROGRESS)
        if current_task:
            if current_task["type"] == TaskType.CHAT:
                cmd_args = command.args
                if cmd_args == "summary":
                    print("CHANGING TO SUMMARY MODE")
                    new_chain = get_summary_chain(state_dict["config"].chat_settings.primary_model)
                    if new_chain is not None:
                        state_dict["active_chain"] = new_chain
                    else:
                        raise ValueError("Summary chain not initialized!")
                else:
                    print("\nCurrent Mode:")
                    print(current_task["type"])
        else:
            print("No in-progress task found")

    elif cmd_type == CommandType.READ:
        # TODO: Read file(s) from args
        print("Reading from file...")
        user_input = read_sample()
        if user_input:
            state_dict["mock_inputs"].insert(0, user_input)
        else:
            print("No text found in file!")
    else:
        print(f"Command not implemented: {command.command}")

    return {
        "keys": state_dict,
        "mutation_count": state["mutation_count"] + 1,
        "is_done": False
    }

def start_node(state: GraphState) -> GraphState:
    """Initialize the graph state with configuration and default task"""
    # config = Config()
    # default_task: Task = {
    #     "type": TaskType.CHAT,
    #     "status": TaskStatus.IN_PROGRESS,
    #     "conditions": ["$"],
    #     "actions": []
    # }
    
    # initial_state: AgentState = {
    #     "config": config,
    #     "messages": [],
    #     "action_count": 0,
    #     "active_chain": None,
    #     "tool_choice": None,
    #     "task_dict": {"chat_task": default_task},
    #     "user_input": None
    # }
    initial_state = INITIAL_STATE_DICT

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
    
    # Handle initialization
    if state["mutation_count"] == 1:
        if not state_dict["config"].chat_settings.disable_system_message:
            insert_system_message(
                state_dict["messages"],
                state_dict["config"].chat_settings.system_message
            )
        # INIT_WITH_CHAT_CHAIN = False
        # if INIT_WITH_CHAT_CHAIN and state_dict["active_chain"] is None:
        #     llm = get_llm(state_dict["config"].chat_settings.primary_model)
        #     if llm is not None:
        #         state_dict["active_chain"] = llm
        #     else:
        #         raise ValueError("Chain not initialized!")
            
    
    # Check failure conditions
    if state["mutation_count"] > MAX_MUTATIONS:
        logging.warning("Mutation count exceeded max mutations!")
        current_task = get_task(task_dict, status=TaskStatus.IN_PROGRESS)
        if current_task:
            logging.error("Task is still in progress, marking as failed")
            current_task["status"] = TaskStatus.FAILED

    return {
        "keys": state_dict,
        "mutation_count": state["mutation_count"] + 1,
        "is_done": state["is_done"]
    }

def task_manager_node(state: GraphState) -> GraphState:
    """Manages task lifecycle and completion"""
    state_dict = state["keys"]
    task_dict = state_dict["task_dict"]
    
    # Save completed tasks
    completed_task_names = save_completed_tasks(task_dict)
    if completed_task_names:
        logging.info(f"Saved completed tasks: {completed_task_names}")
        for task_name in completed_task_names:
            del task_dict[task_name]

    # Check for failed tasks
    failed_task_names = save_failed_tasks(task_dict)
    if failed_task_names:
        logging.error(f"Failed tasks: {failed_task_names}")
        for task_name in failed_task_names:
            del task_dict[task_name]
    
    is_running = start_next_task(task_dict)
    # Check task completion
    if not is_running:
        logging.info("No tasks remaining!")
        # TODO: Add logic to fetch new tasks
        return {
            "keys": state_dict,
            "mutation_count": state["mutation_count"] + 1,
            "is_done": True
        }

    return {
        "keys": state_dict,
        "mutation_count": state["mutation_count"] + 1,
        "is_done": False
    }

def human_node(state: GraphState) -> GraphState:
    """Handles human input"""
    state_dict = state["keys"]
    mock_inputs = state_dict["mock_inputs"]
    if mock_inputs:
        user_input = mock_inputs.pop(0)
    else:
        user_input = input("Enter your message: ").strip()
        if not user_input:
            print("No input provided, click enter again to quit...")
            user_input = input().strip()
            if not user_input:
                user_input = "/quit"
    
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
        # UPDATE STATE BASED ON TASK TYPE
        if current_task["type"] == TaskType.CHAT:
            if state_dict["active_chain"] is None:
                llm = get_llm(state_dict["config"].chat_settings.primary_model)
                if llm is not None:
                    state_dict["active_chain"] = llm
                else:
                    raise ValueError("Chain not initialized!")
            current_task["actions"].append("generate")

        elif current_task["type"] == TaskType.RAG:
            # TODO: Add RAG logic here
            if state_dict["config"].rag_settings.enabled:
                print("RAG task. Doing nothing for now.")
                pass
            else:
                print("ERROR: RAG task is disabled!")
                raise ValueError("RAG task is disabled!")
        else:
            raise ValueError("Current task is not a chat task")
    
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
    logging.info(result)
    
    if result["success"]:
        logging.info(f"Action succeeded: {result['data']}")
        state_dict["action_count"] += 1

        if action == "generate":
            # TODO: Handle more complex chains here!
            response_string = result["data"]
            state_dict["messages"].append(AIMessage(content=response_string))
        
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
    
    # Check for pending actions
    current_task = get_task(task_dict, status=TaskStatus.IN_PROGRESS)
    if current_task:
        if current_task["actions"]:
            return "action_node"
        else:
            return "human_node"
    
    return "task_manager"

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
    # workflow.set_entry_point("start_node")
    workflow.add_edge(START, "start_node")
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
    from dotenv import load_dotenv
    load_dotenv()
    main()
    # save_graph()