#!/usr/bin/env python3

import time
import os
import threading
from typing import Dict, Any

class PerformanceMonitor:
    """Monitor and optimize app performance"""
    
    def __init__(self):
        self._local = threading.local()
        self.metrics = {
            "api_calls": 0,
            "vector_searches": 0,
            "ml_predictions": 0,
            "response_times": []
        }
    
    def start_timer(self):
        self._local.start_time = time.perf_counter()
    
    def end_timer(self, operation: str):
        start_time = getattr(self._local, "start_time", None)
        if start_time:
            elapsed = time.perf_counter() - start_time
            self.metrics["response_times"].append(elapsed)
            self.metrics[operation] = self.metrics.get(operation, 0) + 1
            return elapsed
        return 0
    
    def get_stats(self) -> Dict[str, Any]:
        if self.metrics["response_times"]:
            avg_time = sum(self.metrics["response_times"]) / len(self.metrics["response_times"])
        else:
            avg_time = 0
        
        return {
            "avg_response_time": f"{avg_time:.2f}s",
            "total_operations": sum(self.metrics["response_times"]),
            "api_calls": self.metrics.get("api_calls", 0),
            "vector_searches": self.metrics.get("vector_searches", 0),
            "ml_predictions": self.metrics.get("ml_predictions", 0)
        }

# Global performance monitor
perf_monitor = PerformanceMonitor()

# Simple dictionary-based cache to replace st.session_state
_API_CACHE = {}

def optimize_api_calls():
    """Optimize API call patterns"""
    return _API_CACHE

def cached_api_call(api_func, cache_key: str, *args, **kwargs):
    """Cache API calls to improve performance"""
    cache = optimize_api_calls()
    
    if cache_key in cache:
        perf_monitor.end_timer("api_calls")
        return cache[cache_key]
    
    result = api_func(*args, **kwargs)
    cache[cache_key] = result
    perf_monitor.metrics["api_calls"] += 1
    
    return result

