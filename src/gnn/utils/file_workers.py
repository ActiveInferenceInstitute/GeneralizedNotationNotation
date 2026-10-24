"""Bounded, ordered per-file computation; callers retain all output ownership.

Process workers use spawn, never inherit live parent objects, and receive at
most one queued job per worker. On failure/cancellation the pool is killed.
Pipeline callers additionally run inside the supervised step process group.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait
from typing import Any, Callable, Iterable, Iterator, TypeVar

Job = TypeVar("Job")
Result = TypeVar("Result")
MAX_FILE_WORKERS = 256
MAX_WINDOWS_PROCESS_WORKERS = 61


def resolve_file_workers(
    workers: int | None, executor: str | None, *, parallel: bool = False
) -> tuple[int, str]:
    """Validate the requested limit and choose processes for explicit CPU counts.

    No selection is serial (one worker). Existing ``parallel=True`` chooses
    threads with Python's usual CPU-based cap; an explicit count chooses
    processes unless the caller selects threads. One worker runs in the caller.
    """
    if workers is not None and (
        type(workers) is not int or not 1 <= workers <= MAX_FILE_WORKERS
    ):
        raise ValueError(f"workers must be an integer from 1 to {MAX_FILE_WORKERS}")
    if executor is not None and (
        not isinstance(executor, str) or executor not in {"process", "thread"}
    ):
        raise ValueError("executor must be 'process' or 'thread'")
    count = (
        workers
        if workers is not None
        else (
            min(32, (getattr(os, "process_cpu_count", os.cpu_count)() or 1) + 4)
            if parallel
            else 1
        )
    )
    mode = executor or ("process" if workers is not None else "thread")
    return count, mode


def effective_file_workers(workers: int, executor: str) -> int:
    """Apply the CPython Windows process-pool ceiling to an admitted limit.

    Callers first cap by selection size. Requested limits remain separate;
    Windows processes admit at most 61 effective workers, while threads retain
    their admitted limit. One effective worker still executes in the caller.
    """
    count, mode = resolve_file_workers(workers, executor)
    if mode == "process" and sys.platform == "win32":
        return min(count, MAX_WINDOWS_PROCESS_WORKERS)
    return count


def check_file_work(deadline: float | None = None, cancel_token: Any = None) -> None:
    """Refuse scheduling or publication after cancellation or the run deadline."""
    if cancel_token is not None and cancel_token.cancelled:
        raise TimeoutError(cancel_token.reason or "File processing cancelled")
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Pipeline total deadline expired")


def _invoke(function: Callable[[Job], Result], job: Job) -> tuple[int, Result]:
    return os.getpid(), function(job)


def _kill_process_workers(pool: ProcessPoolExecutor) -> None:
    """Kill and boundedly reap this executor's children on abnormal completion.

    Python 3.14+ provides kill_workers. Earlier supported CPython versions need
    a narrow compatibility read of the executor's owned Process objects; no
    unrelated descendants are signalled. Public Process.kill/join do cleanup.
    """
    processes = list((getattr(pool, "_processes", None) or {}).values())
    kill_workers = getattr(pool, "kill_workers", None)
    # A broken executor can hold its shutdown lock while joining a peer that
    # ignores SIGTERM. Kill owned peers first so public shutdown can progress.
    for process in processes:
        try:
            process.kill()
        except (ProcessLookupError, ValueError):
            pass
    cleanup_deadline = time.monotonic() + 1.0
    for process in processes:
        try:
            while process.is_alive() and time.monotonic() < cleanup_deadline:
                process.join(
                    timeout=min(0.05, max(0.0, cleanup_deadline - time.monotonic()))
                )
                # Sentinel readiness can precede the final OS exit/reap by a moment.
                if process.is_alive():
                    time.sleep(
                        min(0.005, max(0.0, cleanup_deadline - time.monotonic()))
                    )
            if process.is_alive():
                raise RuntimeError("File worker did not stop after forced cleanup")
        except ValueError:
            # The executor's manager already reaped and closed this Process.
            continue
    if kill_workers is not None:
        kill_workers()
    else:
        pool.shutdown(wait=False, cancel_futures=True)


def ordered_file_map(
    function: Callable[[Job], Result],
    jobs: Iterable[Job],
    *,
    workers: int = 1,
    executor: str = "process",
    deadline: float | None = None,
    cancel_token: Any = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[Result]:
    """Yield input-ordered results with bounded submission and polling.

    Jobs must perform computation only; only the consuming parent writes
    artifacts. Thread cancellation cannot interrupt an active Python function.
    Process cleanup is owned by the executor and, in pipeline runs, its supervisor.
    """
    requested_count, mode = resolve_file_workers(workers, executor)
    count = effective_file_workers(requested_count, mode)
    check_file_work(deadline, cancel_token)
    pids: set[int] = set()
    if metadata is not None:
        metadata.update(executor=mode if count > 1 else "serial", worker_pids=[])
        if count != requested_count:
            metadata["effective_workers"] = count

    def record(value: tuple[int, Result]) -> Result:
        check_file_work(deadline, cancel_token)
        pid, result = value
        pids.add(pid)
        if metadata is not None:
            metadata["worker_pids"] = sorted(pids)
        return result

    if count == 1:
        for job in jobs:
            check_file_work(deadline, cancel_token)
            yield record(_invoke(function, job))
        return

    iterator = iter(jobs)
    pending: deque[Any] = deque()
    pool: Any
    if mode == "process":
        pool = ProcessPoolExecutor(
            max_workers=count, mp_context=multiprocessing.get_context("spawn")
        )
    else:
        pool = ThreadPoolExecutor(max_workers=count)
    completed = False
    try:
        for _ in range(count):
            check_file_work(deadline, cancel_token)
            try:
                job = next(iterator)
            except StopIteration:
                break
            pending.append(pool.submit(_invoke, function, job))
        while pending:
            check_file_work(deadline, cancel_token)
            if not pending[0].done():
                wait((pending[0],), timeout=0.05)
                continue
            value = pending[0].result()
            pending.popleft()
            yield record(value)
            check_file_work(deadline, cancel_token)
            try:
                job = next(iterator)
            except StopIteration:
                continue
            pending.append(pool.submit(_invoke, function, job))
        completed = True
    finally:
        if mode == "process" and not completed:
            _kill_process_workers(pool)
        else:
            pool.shutdown(wait=completed, cancel_futures=True)
