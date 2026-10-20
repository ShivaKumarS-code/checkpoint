"""checkpoint: deterministic replay and fork debugger for AI agents.

Public SDK surface:

    import checkpoint

    checkpoint.init(run_name="refund-agent")
    @checkpoint.step(name="plan")
    @checkpoint.tool(mode="stub")
    checkpoint.checkpoint("after_refund", state)
"""

from checkpoint.sdk.checkpoint import checkpoint
from checkpoint.sdk.client import DivergenceError, init
from checkpoint.sdk.decorators import step, tool

__version__ = "0.1.0"

__all__ = ["DivergenceError", "__version__", "checkpoint", "init", "step", "tool"]
