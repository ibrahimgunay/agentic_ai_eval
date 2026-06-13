"""Evaluation primitives: graders, metrics, suite generation, and the runner."""

from .generate import generate_suite
from .runner import run_suite

__all__ = ["generate_suite", "run_suite"]
