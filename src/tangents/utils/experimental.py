"""Utilities for testing HTTP endpoint health.

Provides both async and sync healthcheck functions that verify endpoint liveness.
Requires LITELLM_API_KEY environment variable for authentication.
"""

import os
import aiohttp
import asyncio
import requests
from requests.exceptions import Timeout

from tangents.classes.actions import ActionResult

HEALTHCHECK_ENDPOINT = "http://localhost:4000/health/liveness"
REQUEST_TIMEOUT = 2

async def run_healthcheck(endpoint: str) -> ActionResult:
    """Run async healthcheck against HTTP endpoint.
    
    Args:
        endpoint: URL to check, must match HEALTHCHECK_ENDPOINT
        
    Returns:
        Dict with success status, optional data message, and optional error
        
    Raises:
        AssertionError: If endpoint doesn't match HEALTHCHECK_ENDPOINT
    """
    assert endpoint == HEALTHCHECK_ENDPOINT

    try:
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        headers = {"Authorization": f"Bearer {os.getenv('LITELLM_API_KEY')}"}
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(endpoint, headers=headers) as response:
                success = response.status == 200
                return {
                    "success": success,
                    "data": "Healthcheck passed" if success else None,
                    "error": None if success else f"Status {response.status}"
                }

    except asyncio.TimeoutError:
        return {
            "success": False,
            "data": None, 
            "error": f"Request timed out after {REQUEST_TIMEOUT} seconds"
        }
    except Exception as e:
        return {
            "success": False,
            "data": None,
            "error": str(e)
        }

def run_healthcheck_sync(endpoint: str) -> ActionResult:
    """Run sync healthcheck against HTTP endpoint.
    
    Args:
        endpoint: URL to check, must match HEALTHCHECK_ENDPOINT
        
    Returns:
        Dict with success status, optional data message, and optional error
        
    Raises:
        AssertionError: If endpoint doesn't match HEALTHCHECK_ENDPOINT
    """
    assert endpoint == HEALTHCHECK_ENDPOINT
    
    try:
        headers = {"Authorization": f"Bearer {os.getenv('LITELLM_API_KEY')}"}
        response = requests.get(endpoint, headers=headers, timeout=REQUEST_TIMEOUT)
        success = response.status_code == 200
        return {
            "success": success,
            "data": "Healthcheck passed" if success else None,
            "error": None if success else f"Status {response.status_code}"
        }
    except Timeout:
        return {
            "success": False,
            "data": None,
            "error": f"Request timed out after {REQUEST_TIMEOUT} seconds"
        }
    except Exception as e:
        return {
            "success": False,
            "data": None,
            "error": str(e)
        }

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    print("Sync:", run_healthcheck_sync(HEALTHCHECK_ENDPOINT))
    # print("Async:", asyncio.run(run_healthcheck(HEALTHCHECK_ENDPOINT)))
