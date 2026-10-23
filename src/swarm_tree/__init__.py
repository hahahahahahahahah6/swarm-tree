"""swarm-tree: hierarchical multi-agent orchestration CLI.

Each node in the tree is an agent backed by an LLM provider. Internal nodes
decompose a task into subtasks (one per child), run children in parallel,
and synthesize the child results into a final answer. Leaves do the work.

Only the Python standard library is used at runtime.
"""

from .engine import execute_tree, dry_run_provider
from .config import load_config, validate, TreeError

__version__ = "0.2.0"
__all__ = ["execute_tree", "dry_run_provider", "load_config", "validate", "TreeError"]
