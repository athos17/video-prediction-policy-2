"""Remove DeepSpeed ZeRO-1/2's per-hook parameter scan (``memoize_deepspeed_used_params``).

Every parameter's gradient hook calls ``count_used_parameters_in_backward``, a Python
walk over all parameters, so one backward costs N^2 calls: about 30 s per step for the
2300 tensors of Video + Action. The count only depends on the node set fixed when a graph
task starts, so one computation per task returns the same value to every hook.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# Reentrant checkpointing opens one task per block; this bound keeps a few steps' worth.
_MAX_CACHED_TASKS = 4096
_original = None


def install() -> None:
    """Swap in the once-per-graph-task count (idempotent)."""
    global _original
    from deepspeed.runtime.zero import stage_1_and_2

    if _original is not None:
        return
    original = stage_1_and_2.count_used_parameters_in_backward
    counts: dict[tuple[int, int], int] = {}

    def count_once_per_graph_task(parameters) -> int:
        key = (torch._C._current_graph_task_id(), id(parameters))
        count = counts.get(key)
        if count is None:
            if len(counts) >= _MAX_CACHED_TASKS:
                counts.clear()
            count = counts[key] = original(parameters)
        return count

    _original = original
    stage_1_and_2.count_used_parameters_in_backward = count_once_per_graph_task
    logger.info("ZeRO-1/2 used-parameter count: computed once per graph task")


def uninstall() -> None:
    global _original
    if _original is not None:
        from deepspeed.runtime.zero import stage_1_and_2

        stage_1_and_2.count_used_parameters_in_backward = _original
        _original = None


__all__ = ["install", "uninstall"]
