from augmenta.utils import read_text_file
async def read_sample() -> str:
    """Legacy function to read sample.txt file."""
    return read_text_file("sample.txt")

# NOTE: We support batch reading operations for certain tasks

# import aiofiles
# import asyncio
# from pathlib import Path
# from typing import List, Optional
# # Synchronous file operations
# def read_text_file(filepath: Path) -> str:
#     """
#     Read a text file synchronously and return its contents.
    
#     Args:
#         filepath: Path to the file to read
        
#     Returns:
#         Contents of the file as a string
#     """
#     try:
#         with open(filepath, "r") as f:
#             return f.read()
#     except FileNotFoundError:
#         print(f"File not found: {filepath}")
#         return ""
#     except Exception as e:
#         print(f"Error reading file {filepath}: {e}")
#         return ""

# # Async file operations        
# async def read_text_file_async(filepath: Path) -> str:
#     """
#     Read a text file asynchronously and return its contents.
    
#     Args:
#         filepath: Path to the file to read
        
#     Returns:
#         Contents of the file as a string
#     """
#     try:
#         async with aiofiles.open(filepath, "r") as f:
#             return await f.read()
#     except FileNotFoundError:
#         print(f"File not found: {filepath}")
#         return ""
#     except Exception as e:
#         print(f"Error reading file {filepath}: {e}")
#         return ""

# async def read_multiple_files(filepaths: List[Path]) -> List[str]:
#     """
#     Read multiple text files concurrently and return their contents.
    
#     Args:
#         filepaths: List of paths to files to read
        
#     Returns:
#         List of file contents as strings
#     """
#     tasks = [read_text_file_async(fp) for fp in filepaths]
#     return await asyncio.gather(*tasks)

