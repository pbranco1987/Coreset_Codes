"""
Debug timing utilities for bottleneck identification.

Usage:
    Set environment variable CORESET_DEBUG=1 to enable detailed timing logs.
    
    export CORESET_DEBUG=1
    python -m coreset_selection.run_scenario R1 --k-values 100 --rep-ids 0

The timing logs will be printed to stderr and optionally saved to a file.
"""

from __future__ import annotations

import functools
import logging
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

# Check if debug mode is enabled
DEBUG_ENABLED = os.environ.get("CORESET_DEBUG", "0").lower() in ("1", "true", "yes")

# Configure logging
_logger = logging.getLogger("coreset_debug")


@dataclass
class TimingRecord:
    """A single timing record."""
    name: str
    start_time: float
    end_time: float = 0.0
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List["TimingRecord"] = field(default_factory=list)


class DebugTimer:
    """
    Global debug timer for tracking execution times across the pipeline.
    
    Thread-safe singleton that collects timing information when debug mode is enabled.
    """
    _instance: Optional["DebugTimer"] = None
    
    def __new__(cls) -> "DebugTimer":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.enabled = DEBUG_ENABLED
        self.records: List[TimingRecord] = []
        self._stack: List[TimingRecord] = []
        self._start_time = time.perf_counter()
        self._log_file: Optional[str] = None
        
        if self.enabled:
            self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging handlers."""
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "[%(asctime)s] [DEBUG-TIMING] %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        _logger.addHandler(handler)
        _logger.setLevel(logging.DEBUG)
        
        # Optionally log to file
        log_dir = os.environ.get("CORESET_DEBUG_LOG_DIR", "")
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._log_file = os.path.join(log_dir, f"timing_{timestamp}.log")
            file_handler = logging.FileHandler(self._log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            _logger.addHandler(file_handler)
            self._log(f"Logging to file: {self._log_file}")
    
    def _log(self, message: str):
        """Log a message if debug is enabled."""
        if self.enabled:
            _logger.debug(message)
    
    def _elapsed(self) -> float:
        """Get elapsed time since start."""
        return time.perf_counter() - self._start_time
    
    @contextmanager
    def section(self, name: str, **metadata):
        """
        Context manager for timing a code section.
        
        Usage:
            with timer.section("Loading data"):
                # ... code ...
        """
        if not self.enabled:
            yield
            return
        
        record = TimingRecord(
            name=name,
            start_time=time.perf_counter(),
            metadata=metadata,
        )
        
        # Add to parent's children if we're nested
        if self._stack:
            self._stack[-1].children.append(record)
        else:
            self.records.append(record)
        
        self._stack.append(record)
        
        indent = "  " * (len(self._stack) - 1)
        elapsed = self._elapsed()
        meta_str = f" ({metadata})" if metadata else ""
        self._log(f"{indent}▶ START [{elapsed:8.2f}s] {name}{meta_str}")
        
        try:
            yield record
        finally:
            record.end_time = time.perf_counter()
            record.duration = record.end_time - record.start_time
            self._stack.pop()
            
            elapsed = self._elapsed()
            self._log(f"{indent}◀ END   [{elapsed:8.2f}s] {name} (took {record.duration:.3f}s)")
    
    def checkpoint(self, name: str, **metadata):
        """
        Log a checkpoint without timing a section.
        
        Usage:
            timer.checkpoint("Loaded 10000 rows", n_rows=10000)
        """
        if not self.enabled:
            return
        
        elapsed = self._elapsed()
        indent = "  " * len(self._stack)
        meta_str = f" ({metadata})" if metadata else ""
        self._log(f"{indent}● CHECK [{elapsed:8.2f}s] {name}{meta_str}")
    
    def summary(self) -> str:
        """Generate a summary of all timing records."""
        if not self.records:
            return "No timing records collected."
        
        lines = [
            "",
            "=" * 80,
            "TIMING SUMMARY",
            "=" * 80,
        ]
        
        def format_record(rec: TimingRecord, indent: int = 0) -> List[str]:
            prefix = "  " * indent
            result = [f"{prefix}{rec.name}: {rec.duration:.3f}s"]
            for child in rec.children:
                result.extend(format_record(child, indent + 1))
            return result
        
        for record in self.records:
            lines.extend(format_record(record))
        
        total = self._elapsed()
        lines.extend([
            "=" * 80,
            f"TOTAL ELAPSED: {total:.3f}s",
            "=" * 80,
            "",
        ])
        
        return "\n".join(lines)
    
    def print_summary(self):
        """Print the timing summary."""
        if self.enabled:
            print(self.summary(), file=sys.stderr)


# Global timer instance
timer = DebugTimer()


def timed(name: Optional[str] = None):
    """
    Decorator for timing function execution.
    
    Usage:
        @timed("MyFunction")
        def my_function():
            ...
        
        # Or use function name automatically:
        @timed()
        def my_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        section_name = name or func.__qualname__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with timer.section(section_name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def enable_debug():
    """Programmatically enable debug mode."""
    global DEBUG_ENABLED
    DEBUG_ENABLED = True
    timer.enabled = True
    if not _logger.handlers:
        timer._setup_logging()


def disable_debug():
    """Programmatically disable debug mode."""
    global DEBUG_ENABLED
    DEBUG_ENABLED = False
    timer.enabled = False
