"""Module for handling user input and interrupts."""

import asyncio


class UserInputHandler:
    """Handles asynchronous user input collection."""

    @staticmethod
    async def get_input(prompt: str) -> str:
        """Get user input asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, input, f"{prompt}\n")

    @staticmethod
    async def get_confirmed_input(
        prompt: str, retry_msg: str = "No input provided, click enter again to quit..."
    ) -> str:
        """Get user input with confirmation for empty input."""
        user_input = (await UserInputHandler.get_input(prompt)).strip()

        if not user_input:
            print(retry_msg)
            user_input = (await UserInputHandler.get_input("")).strip()
            if not user_input:
                return "/quit"
        return user_input


class InterruptHandler:
    """Handles workflow interrupts."""

    def __init__(self, input_handler: UserInputHandler = None):
        self.input_handler = input_handler or UserInputHandler()

    async def process_interrupt(self, interrupt_value: dict) -> str:
        """Handle workflow interrupts and get user input."""
        if "prompt" not in interrupt_value:
            raise ValueError("Unhandled interrupt case - missing prompt")

        return await self.input_handler.get_confirmed_input(interrupt_value["prompt"])
