from tangents.classes.actions import PlanActionType, Status
from tangents.classes.settings import ChatSettings, Config, RAGSettings
from tangents.classes.states import AgentState
from tangents.classes.tasks import Task, TaskType
from tangents.utils.action_utils import create_action
from tangents.experimental.constants import CONVEX_MCP_SPEC_PATH

class MockInputs:
    SUMMARY = ['/mode summary', '/read', '/quit']
    DEFAULT = ["What's the capital of France?", '/quit']
    EMPTY = []


def get_initial_graph_state():
    return {'keys': {}, 'mutation_count': 0, 'is_done': False}


def get_default_task() -> Task:
    return {
        'type': TaskType.CHAT,
        'status': Status.NOT_STARTED,
        'actions': [],
        'state': None,
    }


def get_rag_task() -> Task:
    return {
        'type': TaskType.RAG,
        'status': Status.IN_PROGRESS,
        'actions': [],
        'state': None,
    }


def get_default_config() -> Config:
    return Config(
        chat_settings=ChatSettings(
            primary_model='deepseek-v3',
            stream=True,
            system_message='Speak with lots of emojis',
            disable_system_message=False,
        ),
        rag_settings=RAGSettings(),
    )


def get_planning_actions():
    read_email_action = create_action(
        PlanActionType.FETCH,
        args={'source': 'example-email.txt', 'method': 'get_email_content'},
    )
    propose_plan_action = create_action(PlanActionType.PROPOSE_PLAN)
    revise_plan_action = create_action(PlanActionType.REVISE_PLAN)
    return [read_email_action, propose_plan_action, revise_plan_action]


def get_example_planning_task() -> Task:
    return Task(
        type=TaskType.PLANNING,
        status=Status.NOT_STARTED,
        actions=get_planning_actions(),
        state=None,
    )


def get_default_state_dict() -> AgentState:
    return {
        'config': get_default_config(),
        'action_count': 0,
        'task_dict': {'chat_task': get_default_task()},
        'user_input': None,
        'mock_inputs': MockInputs.EMPTY,
    }


def get_planning_state_dict() -> AgentState:
    return {
        'config': get_default_config(),
        'action_count': 0,
        'user_input': None,
        # 'task_dict': {'plan_from_email_task': get_example_planning_task()},
        'task_dict': {'convex_mcp_spec_task': get_convex_mcp_spec_task()},
        'mock_inputs': MockInputs.EMPTY,
    }

def get_convex_mcp_spec_task() -> Task:
    return Task(
        type=TaskType.PLANNING,
        status=Status.NOT_STARTED,
        # status=Status.IN_PROGRESS,
        actions=[
            create_action(PlanActionType.FETCH, args={'source': CONVEX_MCP_SPEC_PATH, 'method': 'convex_mcp_spec'}),
            create_action(PlanActionType.PROPOSE_PLAN),
            create_action(PlanActionType.REVISE_PLAN),
        ],
        state=None,
        # state={
        #     'plan_context': None,
        #     'proposed_plan': None,
        #     'plan': None,
        #     'revision_count': 0,
        # },
    )