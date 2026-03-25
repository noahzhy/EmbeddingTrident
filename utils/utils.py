import os
import time
import json
from typing import List
from urllib.error import URLError
from urllib.request import urlopen

import ray
from ray import serve


def wait_for_routes(base_url: str, expected_routes: List[str], timeout_s: int = 45, interval_s: float = 0.5) -> None:
    deadline = time.time() + timeout_s
    routes_url = f"{base_url}/-/routes"

    while time.time() < deadline:
        try:
            with urlopen(routes_url, timeout=2) as resp:
                route_payload = json.loads(resp.read().decode("utf-8"))
            
            available_routes = set(route_payload.keys()) if isinstance(
                route_payload, dict) else set()
            if all(route in available_routes for route in expected_routes):
                print(f"[ready] routes available: {sorted(available_routes)}")
                return
        except (URLError, TimeoutError, json.JSONDecodeError, OSError):
            pass
        
        time.sleep(interval_s)

    raise RuntimeError(
        f"Serve routes not ready within {timeout_s}s. Expected: {expected_routes}, url: {routes_url}"
    )
