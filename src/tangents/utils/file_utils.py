# NOTE: We support batch reading operations for certain tasks

from pathlib import Path
import logging
from typing import List, Optional
import aiofiles
import asyncio

# Set up logging
logger = logging.getLogger(__name__)


def read_text_file(filepath: Path) -> Optional[str]:
    """
    Read a text file synchronously and return its contents.

    Args:
        filepath: Path to the file to read

    Returns:
        Contents of the file as a string, or None if there was an error
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return None
    except Exception as e:
        logger.error(f"Error reading file {filepath}: {e}")
        return None


def read_multiple_files_sync(filepaths: List[Path]) -> List[Optional[str]]:
    """
    Read multiple text files synchronously and return their contents.

    Args:
        filepaths: List of paths to files to read

    Returns:
        List of file contents as strings, with None for any failed reads
    """
    return [read_text_file(fp) for fp in filepaths]


async def read_text_file_async(filepath: Path) -> Optional[str]:
    """
    Read a text file asynchronously and return its contents.

    Args:
        filepath: Path to the file to read

    Returns:
        Contents of the file as a string, or None if there was an error
    """
    try:
        async with aiofiles.open(filepath, "r", encoding="utf-8") as f:
            return await f.read()
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return None
    except Exception as e:
        logger.error(f"Error reading file {filepath}: {e}")
        return None


async def read_multiple_files_async(filepaths: List[Path]) -> List[Optional[str]]:
    """
    Read multiple text files concurrently and return their contents.

    Args:
        filepaths: List of paths to files to read

    Returns:
        List of file contents as strings, with None for any failed reads
    """
    tasks = [read_text_file_async(fp) for fp in filepaths]
    return await asyncio.gather(*tasks)
