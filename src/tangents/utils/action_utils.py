import logging

from tangents.classes.actions import Action, ActionArgs, ActionType, PlanActionType, Status

def get_revise_plan_args(args: ActionArgs) -> ActionArgs:
    """Get default arguments for revise plan action."""
    DEFAULT_MAX_REVISIONS = 3

    if "proposed_plan" not in args:
        args["proposed_plan"] = None
    if "revision_context" not in args:
        args["revision_context"] = None
    if "max_revisions" not in args:
        args["max_revisions"] = DEFAULT_MAX_REVISIONS
    return args

def get_propose_plan_args(args: ActionArgs) -> ActionArgs:
    """Get default arguments for create plan action."""
    if "plan_context" not in args:
        args["plan_context"] = None
    return args

def get_generate_args(args: ActionArgs) -> ActionArgs:
    """Get default arguments for generate action."""
    if "chain" not in args:
        args["chain"] = None
    if "messages" not in args:
        args["messages"] = []
    return args

def get_healthcheck_args(args: ActionArgs) -> ActionArgs:
    """Get default arguments for healthcheck action."""
    if "endpoint" not in args:
        args["endpoint"] = None
    return args

ACTION_ARG_HANDLERS = {
    PlanActionType.REVISE_PLAN: get_revise_plan_args,
    PlanActionType.PROPOSE_PLAN: get_propose_plan_args,
    ActionType.GENERATE: get_generate_args,
    ActionType.HEALTHCHECK: get_healthcheck_args
}

def create_action(action_type: ActionType | PlanActionType, args: ActionArgs = {}) -> Action:
    """Create a new action with default configuration.
    
    Args:
        action_type: Type of action to create
        args: Optional action arguments
        
    Returns:
        Configured Action instance
        
    Raises:
        AssertionError: If arguments are invalid
    """
    assert isinstance(action_type, ActionType | PlanActionType)
    assert isinstance(args, dict)

    if action_type in ACTION_ARG_HANDLERS:
        args = ACTION_ARG_HANDLERS[action_type](args)
    
    return Action(
        type=action_type,
        status=Status.NOT_STARTED,
        args=args,
        result=None
    )

    
def save_action_data(action: Action) -> bool:
    """Save action data to a file."""
    logging.info(f"Saved action: {action['type']}")
    return True

def is_stash_action_next(action_list: list[Action]) -> bool:
    """Check if the next action is a StashAction"""
    if not action_list:
        return False
    return action_list[0]["type"] == ActionType.STASH

def is_human_action_next(action_list: list[Action]) -> bool:
    """Check if the next action is a HumanAction"""
    if not action_list:
        return False
    return action_list[0]["type"] == ActionType.HUMAN_INPUT

def add_stash_action(action_list: list[Action]) -> None:
    """Add a StashAction to the START of the action list."""
    # check if first action is a StashAction
    if action_list and action_list[0]["type"] == ActionType.STASH:
        return
    action_list.insert(0, create_action(ActionType.STASH))

def add_human_action(action_list: list[Action], prompt: str = None) -> None:
    """Add a HumanAction to the START of the action list."""
    if action_list and action_list[0]["type"] == ActionType.HUMAN_INPUT:
        logging.warning("Duplicate human action detected; adding another.")
    
    if prompt:
        # NOTE: Can add assertions here
        args = {"prompt": prompt}
    else:
        args = {}
    action_list.insert(0, create_action(ActionType.HUMAN_INPUT, args))
