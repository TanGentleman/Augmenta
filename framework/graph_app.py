from typing import Literal, TypedDict, Dict
from pprint import pprint

from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage
# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, SystemMessage

from classes import Config
from models.models import LLM
from dotenv import load_dotenv
load_dotenv()

MAX_MUTATION_COUNT = 50
MAX_RESPONSE_COUNT = 3
COMMAND_LIST = ["q", "test"]
# Define state
class GraphState(TypedDict):
    """
    Represents the state of an agent in the conversation.

    Attributes:
        keys: A dictionary containing the state variables.
        mutation_count: An integer representing the number of times the state has been mutated.
        is_done: A boolean representing whether the conversation has ended.
    """
    keys: Dict[str, any]
    mutation_count: int
    is_done: bool

# Define nodes
def start_node(state: GraphState) -> GraphState:
    """
    The starting node of the graph. This node is responsible for initializing the state of the graph.

    Args:
        state (GraphState): _description_

    Returns:
        GraphState: Update: add the config to the state using Config().
    """
    config = Config()
    return {
        "keys": {"config": config},
        "mutation_count": state["mutation_count"] + 1,
        "is_done": False
    }

def apply_command_node(state: GraphState) -> GraphState:
    """
    This function is responsible for applying the command to the state.

    Args:
        command (str): _description_

    Returns:
        None: _description_
    """
    state_dict = state["keys"]
    def handle_input(input: str) -> None | str:
        if input == "q":
            return None
        elif input == "test":
            print("Test command applied.")
            return "I successfully applied the test command. Celebrate with JSON of this moment."
        return input
    assert "user_input" in state_dict, "No command in state keys!"
    command_string = state_dict["user_input"].strip()
    assert command_string in COMMAND_LIST, "Command not in command list!"
    new_input = handle_input(command_string)
    if new_input is None:
        return {
            "keys": state_dict,
            "mutation_count": state["mutation_count"] + 1,
            "is_done": True
        }
    state_dict["user_input"] = new_input
    return {
        "keys": state_dict,
        "mutation_count": state["mutation_count"] + 1,
        "is_done": False
    }

def chatbot_node(state: GraphState) -> GraphState:
    """
    This node is responsible for handling the chatbot conversation.

    Args:
        state (GraphState): _description_

    Returns:
        GraphState: Update: 
    """
    state_dict = state["keys"]
    config: Config = state_dict["config"]
    if state["mutation_count"] == 1: # This is first entry to the chatbot node
        messages = []
        if config.chat_settings.enable_system_message:
            messages.append(SystemMessage(content=config.chat_settings.system_message))
        new_keys = {
            "chat_model": None,
            "messages": messages,
            "response_count": 0,
            "config": config,
        }
        
    elif state_dict.get("user_input", ""):
        user_input = state_dict["user_input"]
        if user_input.strip() in COMMAND_LIST:
            raise ValueError("Command must be handled in human node. If this is legal, this error should be a warning.")
        messages = state_dict["messages"]
        messages.append(HumanMessage(content=user_input))
        # Here you can validate the messages array before assigning them to the state
        new_keys = {
            "chat_model": state_dict["chat_model"],
            "messages": messages,
            "response_count": state_dict["response_count"],
            "config": state_dict["config"],
        }
        
    else:
        # Go to the tools node
        new_keys = state_dict
        pass

    # Exit conditions
    is_done = (state["mutation_count"] >= MAX_MUTATION_COUNT) or (new_keys["response_count"] >= MAX_RESPONSE_COUNT)
    print("Chatbot exit condition fulfilled!") if is_done else None
    return {
        "keys": new_keys,
        "mutation_count": state["mutation_count"] + 1,
        "is_done": is_done
    }

def human_node(state: GraphState) -> GraphState:
    """
    This node is responsible for getting the human input.

    Args:
        state (GraphState): _description_

    Returns:
        GraphState: Update: 
    """
    def validate(user_input: str) -> str | None:
        if not user_input.strip():
            return None
        return user_input
    
    state_dict = state["keys"]
    is_valid = False
    user_input = ""
    while not is_valid:
        user_input = input("Enter your message: ")
        validated = validate(user_input)
        if validated is not None:
            user_input = validated
            is_valid = True
    return {
        "keys": {
            "user_input": user_input,
            "chat_model": state_dict["chat_model"],
            "messages": state_dict["messages"],
            "response_count": state_dict["response_count"],
            "config": state_dict["config"]
        },
        "mutation_count": state["mutation_count"] + 1,
        "is_done": False
    }

def tools_node(state: GraphState) -> GraphState:
    """
    This node is responsible for handling the tool calling decision.

    Args:
        state (GraphState): _description_

    Returns:
        GraphState: Update: 
    """
    # tool_choice can be [None, "generate"]
    state_dict = state["keys"]
    messages = state_dict["messages"]
    if not messages:
        print("I don't think I can have blank messages at any point for tool calling right?")
        raise Exception("Blank messages")
    
    mutation_count = state["mutation_count"]
    last_message = messages[-1]
    if isinstance(last_message, HumanMessage):
        # Prep for generate node for now
        tool_choice = "generate"
        # This can potentially have values like "web_search", "retrieve", "etc."
        pass
    else:
        # Send back to chatbot node
        tool_choice = None
        pass

    return {
        "keys": {
            "chat_model": state_dict["chat_model"],
            "messages": messages,
            "response_count": state_dict["response_count"],
            "tool_choice": tool_choice,
            "config": state_dict["config"]
        },
        "mutation_count": mutation_count + 1,
        "is_done": False
    }

def generate_node(state: GraphState) -> GraphState:
    """
    This node is responsible for generating the response.

    Args:
        state (GraphState): _description_

    Returns:
        GraphState: Update: 
    """
    state_dict = state["keys"]
    chat_model = state_dict["chat_model"]
    messages = state_dict["messages"]
    assert messages, "No messages to generate response from!"
    config: Config = state_dict["config"]
    stream = config.chat_settings.stream
    if chat_model is None:
        try:
            chat_model = LLM(config.chat_settings.primary_model)
        except Exception as e:
            print(f"Error: {e}")
            return SystemExit

    # This section should be the equivalent of Chatbot.invoke()
    # I can handle it more delicately with more granular callback/stdout handling in the future
    try:
        if stream:
            response_string = ""
            for chunk in chat_model.stream(messages):
                print(chunk.content, end="", flush=True)
                response_string += chunk.content
            print()
            if not response_string:
                raise ValueError('No response generated')
            response = AIMessage(content=response_string)
        else:
            response = chat_model.invoke(messages)
            assert isinstance(
                response, AIMessage), "Response not AIMessage"
            print(response.content)
    except KeyboardInterrupt:
        print('Keyboard interrupt, aborting generation.')
        messages.pop()
        return None
    except Exception as e:
        print(f'Error!: {e}')
        raise SystemExit
    
    messages.append(response)
    response_count = state_dict["response_count"] + 1
    return {
        "keys": {
            "chat_model": chat_model,
            "messages": messages,
            "response_count": response_count,
            "config": config
        },
        "mutation_count": state["mutation_count"] + 1,
        "is_done": False
    }

def end_node(state: GraphState) -> GraphState:
    """
    This node is responsible for ending the conversation.

    Args:
        state (GraphState): _description_

    Returns:
        GraphState: Update: 
    """
    assert state["is_done"]
    mutation_count = state["mutation_count"]
    print(f"Arrived at end node after {mutation_count} mutations.")
    return {
        "keys": {},
        "mutation_count": state["mutation_count"] + 1,
        "is_done": True
    }

# Define conditional edges
def decide_from_chatbot(state: GraphState) -> Literal["end_node", "human_node", "tools_node"]:
    """
    This function is responsible for deciding whether the next node should be the chatbot or the human node.

    Args:
        state (GraphState): _description_

    Returns:
        Literal["chatbot", "human"]: _description_
    """
    if state["is_done"]:
        return "end_node"
    state_dict = state["keys"]
    messages = state_dict["messages"]
    if not messages:
        return "human_node"
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) or isinstance(last_message, SystemMessage):
        return "human_node"
    return "tools_node"

def decide_from_human(state: GraphState) -> Literal["chatbot_node", "apply_command_node"]:
    """
    This function is responsible for deciding whether the next node should be the chatbot node or the apply_command node.

    Args:
        state (GraphState): _description_

    Returns:
        Literal["chatbot_node", "apply_command_node"]: _description_
    """
    state_dict = state["keys"]
    assert "user_input" in state_dict, "No user input in state keys!"
    user_input = state_dict["user_input"]
    if user_input.strip() in COMMAND_LIST:
        return "apply_command_node"
    return "chatbot_node"

def decide_from_tools(state: GraphState) -> Literal["chatbot_node", "generate_node"]:
    """
    This function is responsible for deciding whether the next node should be the generate node or the end node.

    Args:
        state (GraphState): _description_

    Returns:
        Literal["generate_node", "end_node"]: _description_
    """
    state_dict = state["keys"]
    assert "tool_choice" in state_dict, "No tool choice in state keys!"
    tool_choice = state_dict["tool_choice"]
    if tool_choice is None:
        return "chatbot_node"
    # There will be more confitions in the future here
    return "generate_node"

# Build graph
workflow = StateGraph(GraphState)
# Define the nodes
workflow.add_node("start_node", start_node)
workflow.add_node("chatbot_node", chatbot_node)
workflow.add_node("human_node", human_node)
workflow.add_node("apply_command_node", apply_command_node)
workflow.add_node("tools_node", tools_node)
workflow.add_node("generate_node", generate_node)
workflow.add_node("end_node", end_node)

# Build graph
workflow.set_entry_point("start_node")
workflow.add_edge("start_node", "chatbot_node")
workflow.add_conditional_edges(
    "chatbot_node", 
    decide_from_chatbot, 
    {
        "human_node": "human_node",
        "end_node": "end_node",
        "tools_node": "tools_node"
    }
)
# workflow.add_edge("human_node", "chatbot_node")

workflow.add_conditional_edges(
    "human_node", 
    decide_from_human, 
    {
        "chatbot_node": "chatbot_node",
        "apply_command_node": "apply_command_node"
    }
)
workflow.add_edge("apply_command_node", "chatbot_node")

workflow.add_edge("end_node", END)

workflow.add_conditional_edges(
    "tools_node", 
    decide_from_tools, 
    {
        "generate_node": "generate_node",
        "chatbot_node": "chatbot_node",
    }
)
workflow.add_edge("generate_node", "tools_node")

# Compile
app = workflow.compile()

def main():
    initial_state = {
        "keys": {}, 
        "mutation_count": 0, 
        "is_done": False
    }
    for output in app.stream(initial_state):
        for key, value in output.items():
            pprint(f"Reached node '{key}':")
            pprint(f"Counted {value['mutation_count']} mutations")
            # pprint("---")
            # pprint(value["keys"], indent=2, width=80, depth=None)
        pprint("\n---\n")
    print("Conversation ended.")

if __name__ == "__main__":
    main()