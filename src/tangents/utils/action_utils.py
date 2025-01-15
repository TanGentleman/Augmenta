import logging

from tangents.classes.actions import (
    Action,
    ActionArgs,
    ActionType,
    PlanActionType,
    Status,
)


def set_revise_plan_args(args: ActionArgs) -> ActionArgs:
    """Get default arguments for revise plan action."""
    DEFAULT_MAX_REVISIONS = 3

    if not args:
        args = {}
    args.setdefault('proposed_plan', None)
    args.setdefault('revision_context', None)
    args.setdefault('max_revisions', DEFAULT_MAX_REVISIONS)
    return args


def set_propose_plan_args(args: ActionArgs) -> ActionArgs:
    """Get default arguments for create plan action."""
    args = args or {}
    args.setdefault('plan_context', None)
    return args


def set_generate_args(args: ActionArgs) -> ActionArgs:
    """Get default arguments for generate action."""
    args = args or {}
    args.setdefault('chain', None)
    args.setdefault('messages', [])
    return args


def set_healthcheck_args(args: ActionArgs) -> ActionArgs:
    """Get default arguments for healthcheck action."""
    args = args or {}
    args.setdefault('endpoint', None)
    return args


# Add human args with default input of the prompt.

ACTION_ARG_HANDLERS = {
    PlanActionType.REVISE_PLAN: set_revise_plan_args,
    PlanActionType.PROPOSE_PLAN: set_propose_plan_args,
    ActionType.GENERATE: set_generate_args,
    ActionType.HEALTHCHECK: set_healthcheck_args,
}


def create_action(action_type: ActionType | PlanActionType, args: ActionArgs = {}) -> Action:
    """Create a new action with default configuration."""
    assert isinstance(action_type, ActionType | PlanActionType)
    assert isinstance(args, dict)

    if action_type in ACTION_ARG_HANDLERS:
        set_args_func = ACTION_ARG_HANDLERS[action_type]
        args = set_args_func(args)

    return Action(type=action_type, status=Status.NOT_STARTED, args=args, result=None)


def save_action_data(action: Action) -> bool:
    """Save action data to a file."""
    logging.info(f"Saved action: {action['type']}")
    return True


def is_stash_action_next(action_list: list[Action]) -> bool:
    """Check if the next action is a StashAction"""
    if not action_list:
        return False
    return action_list[0]['type'] == ActionType.STASH


def is_human_action_next(action_list: list[Action]) -> bool:
    """Check if the next action is a HumanAction"""
    if not action_list:
        return False
    return action_list[0]['type'] == ActionType.HUMAN_INPUT


def add_stash_action(action_list: list[Action]) -> None:
    """Add a StashAction to the START of the action list."""
    # check if first action is a StashAction
    if is_stash_action_next(action_list):
        return
    action = create_action(ActionType.STASH)
    action_list.insert(0, action)


def add_human_action(action_list: list[Action], prompt: str = None) -> None:
    """Add a HumanAction to the START of the action list."""
    action = create_action(ActionType.HUMAN_INPUT)
    if prompt:
        action['args']['prompt'] = prompt
    action_list.insert(0, action)
