import asyncio

import pytest

from main import run_single_test
from task import PROMPT, TOOL_HANDLERS, TOOLS, grading_func


@pytest.mark.asyncio
async def test_task_pass_rate():
    """
    Test that the agent can solve the task with a pass rate between 0% and 70%.
    This ensures the task is solvable but not trivial.
    """
    num_runs = 30

    # Create all test coroutines
    tasks = [
        run_single_test(
            run_id=i + 1,
            num_runs=num_runs,
            prompt=PROMPT,
            tools=TOOLS,
            tool_handlers=TOOL_HANDLERS,
            grading_func=grading_func,
            verbose=False,
        )
        for i in range(num_runs)
    ]

    # Run concurrently
    results = []
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)

    # Count successes
    successes = sum(success for _, success, _ in results)
    pass_rate = successes / num_runs

    print(f"\nTest Results: {successes}/{num_runs} passed ({pass_rate * 100:.1f}%)")

    # Assert pass rate is between 0% and 70%
    assert pass_rate > 0, "Pass rate is 0% - task may be too difficult or broken"
    assert pass_rate < 0.7, (
        f"Pass rate is {pass_rate * 100:.1f}% - task may be too easy (should be < 70%)"
    )
