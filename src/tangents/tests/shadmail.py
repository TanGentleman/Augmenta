"""
Shadow Mail Script

A script that plans email responses using the task-based workflow system.
Provides a simplified interface to the email planning functionality.
"""

import asyncio
from pathlib import Path
from typing import Optional

from tangents.tests.plan_email import (
    plan_email,
    FetchParams,
    PlanParams,
    ExtraParams
)
from paths import SHADOWS_EMAIL_DIR

# Constants
DEFAULT_EMAIL_PATH = SHADOWS_EMAIL_DIR / "scenario-1-email.txt"
DEFAULT_MAX_REVISIONS = 2
CUSTOM_SYSTEM_MESSAGE = "Create 3 actionable tasks based on the email content."

async def process_email(
    email_path: Path = DEFAULT_EMAIL_PATH,
    max_revisions: int = DEFAULT_MAX_REVISIONS,
    system_message: Optional[str] = CUSTOM_SYSTEM_MESSAGE
) -> None:
    """
    Process an email file and generate a response plan.

    Args:
        email_path: Path to the email file to process
        max_revisions: Maximum number of plan revision cycles
        system_message: Custom system message for the planner
    
    Raises:
        FileNotFoundError: If the email file doesn't exist
    """
    if not email_path.exists():
        raise FileNotFoundError(f"Email file not found: {email_path}")

    fetch_params = FetchParams(
        source=str(email_path),
        method="get_email_content"
    )
    
    plan_params = PlanParams(
        max_revisions=max_revisions
    )

    extra_params = ExtraParams(
        system_message=system_message
    )

    await plan_email(
        fetch_params=fetch_params,
        plan_params=plan_params,
        extra_params=extra_params
    )

def main() -> None:
    """Main entry point for the script."""
    asyncio.run(process_email())

if __name__ == "__main__":
    main()
