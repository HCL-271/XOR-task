import asyncio
import json
import random
from collections.abc import Callable
from typing import Any, Optional

from anthropic import AsyncAnthropic, RateLimitError
from anthropic.types import MessageParam, ToolUnionParam

from task import PROMPT, TOOL_HANDLERS, TOOLS, grading_func

MAX_TOKENS = 1000

# Hard cap on in-flight Claude API requests from *this* process.
# This protects you even if you flip `concurrent=True` later.
_API_SEMAPHORE = asyncio.Semaphore(1)


def _retry_after_seconds(err: BaseException) -> float | None:
    """Best-effort extraction of Retry-After from Anthropic SDK errors."""
    resp = getattr(err, "response", None)
    headers = getattr(resp, "headers", None) if resp is not None else None
    if not headers:
        return None
    ra = headers.get("retry-after") or headers.get("Retry-After")
    if not ra:
        return None
    try:
        return float(ra)
    except Exception:
        return None


async def _messages_create_with_backoff(
    client: AsyncAnthropic,
    *,
    model: str,
    max_tokens: int,
    tools: list[ToolUnionParam],
    messages: list[MessageParam],
    max_attempts: int = 12,
) -> Any:
    """Call `client.messages.create` with robust 429 handling.

    Notes:
      - Anthropic SDK already retries certain errors by default, but *your* code
        must still handle cases where 429 persists (e.g., concurrency limit).
      - We also guard calls with a semaphore so a single process can't exceed
        connection concurrency even if tasks are scheduled concurrently.
    """
    last_err: RateLimitError | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            async with _API_SEMAPHORE:
                return await client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    tools=tools,
                    messages=messages,
                )
        except RateLimitError as e:
            last_err = e

            # Prefer server guidance when available.
            delay = _retry_after_seconds(e)
            if delay is None:
                # Exponential backoff w/ jitter, capped.
                delay = min(60.0, 2.0 * (2 ** (attempt - 1)))
                delay += random.random()

            print(
                f"Rate limit hit (attempt {attempt}/{max_attempts}). "
                f"Sleeping {delay:.1f}s then retrying..."
            )
            await asyncio.sleep(delay)

    # If we exhaust retries, surface the last 429 we saw.
    assert last_err is not None
    raise last_err


async def run_agent_loop(
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    max_steps: int = 20,
    model: str = "claude-haiku-4-5",
    verbose: bool = True,
    *,
    client: Optional[AsyncAnthropic] = None,
) -> Any | None:
    """Runs an agent loop with the given prompt and tools.

    Returns:
        The submitted answer if submit_answer was called, otherwise None
    """

    # Own the client only if the caller didn't pass one in.
    own_client = client is None
    if own_client:
        # Using `async with` ensures underlying HTTP resources are closed.
        client_cm = AsyncAnthropic()
    else:
        client_cm = None  # type: ignore

    async def _run_with_client(active_client: AsyncAnthropic) -> Any | None:
        messages: list[MessageParam] = [{"role": "user", "content": prompt}]

        for step in range(max_steps):
            if verbose:
                print(f"\n=== Step {step + 1}/{max_steps} ===")

            # Optional pacing. Keep small; rely on backoff for real limiting.
            await asyncio.sleep(1)

            response = await _messages_create_with_backoff(
                active_client,
                model=model,
                max_tokens=MAX_TOKENS,
                tools=tools,
                messages=messages,
            )

            assert response.stop_reason in ["max_tokens", "tool_use", "end_turn"], (
                f"unsupported stop_reason {response.stop_reason}"
            )
            if response.stop_reason == "max_tokens":
                print(
                    f"Model reached max_tokens limit {MAX_TOKENS}. Increase "
                    "MAX_TOKENS, simplify your task, or update the code to provide "
                    "a message back to the model when it exceeds MAX_TOKENS."
                )

            # Track if we need to continue
            has_tool_use = False
            tool_results = []
            submitted_answer = None

            # Process the response
            for content in response.content:
                if content.type == "text":
                    if verbose:
                        print(f"Assistant: {content.text}")
                elif content.type == "tool_use":
                    has_tool_use = True
                    tool_name = content.name

                    if tool_name in tool_handlers:
                        if verbose:
                            print(f"Using tool: {tool_name}")

                        # Extract arguments based on tool
                        handler = tool_handlers[tool_name]
                        tool_input = content.input

                        # Call the appropriate tool handler
                        if tool_name == "python_expression":
                            assert (
                                isinstance(tool_input, dict) and "expression" in tool_input
                            )
                            if verbose:
                                print("\nInput:")
                                print("```")
                                for line in tool_input["expression"].split("\n"):
                                    print(f"{line}")
                                print("```")
                            result = handler(tool_input["expression"])
                            if verbose:
                                print("\nOutput:")
                                print("```")
                                print(result)
                                print("```")
                        elif tool_name == "submit_answer":
                            if isinstance(tool_input, dict) and "answer" in tool_input:
                                payload = tool_input["answer"]
                            else:
                                payload = tool_input  # fallback: accept raw
                            result = handler(payload)
                            submitted_answer = result["answer"]
                        else:
                            # Generic handler call
                            result = (
                                handler(**tool_input)
                                if isinstance(tool_input, dict)
                                else handler(tool_input)
                            )

                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": json.dumps(result),
                            }
                        )

            # If we have tool uses, add them to the conversation
            if has_tool_use:
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})

                # If an answer was submitted, return it
                if submitted_answer is not None:
                    if verbose:
                        print(f"\nAgent submitted answer: {submitted_answer}")
                    return submitted_answer
            else:
                # No tool use, conversation might be complete
                if verbose:
                    print("\nNo tool use in response, ending loop.")
                break

        if verbose:
            print(f"\nReached maximum steps ({max_steps}) without submitting answer.")
        return None

    if own_client:
        async with client_cm as active_client:  # type: ignore
            return await _run_with_client(active_client)
    else:
        assert client is not None
        return await _run_with_client(client)


async def run_single_test(
    run_id: int,
    num_runs: int,
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    grading_func: Callable[[Any], bool],
    verbose: bool = False,
    *,
    client: Optional[AsyncAnthropic] = None,
) -> tuple[int, bool, Any]:
    if verbose:
        print(f"\n\n{'=' * 20} RUN {run_id}/{num_runs} {'=' * 20}")

    result = await run_agent_loop(
        prompt=prompt,
        tools=tools,
        tool_handlers=tool_handlers,
        max_steps=5,
        verbose=verbose,
        client=client,
    )

    success = grading_func(result)

    if success:
        print(f"✓ Run {run_id}: SUCCESS - Got {result}")
    else:
        print(f"✗ Run {run_id}: FAILURE - Got {result}")

    return run_id, success, result


async def main(concurrent: bool = False):
    # Run the test 10 times and track success rate
    num_runs = 10

    execution_mode = "concurrently" if concurrent else "sequentially"
    print(f"Running {num_runs} test iterations {execution_mode}...")
    print("=" * 60)

    # Share one client for the whole run (reduces connection churn).
    async with AsyncAnthropic() as client:
        # Create all test coroutines
        tasks = [
            run_single_test(
                run_id=i + 1,
                num_runs=num_runs,
                prompt=PROMPT,
                tools=TOOLS,
                tool_handlers=TOOL_HANDLERS,
                grading_func=grading_func,
                verbose=True,
                client=client,
            )
            for i in range(num_runs)
        ]

        # Run concurrently or sequentially based on the flag
        if concurrent:
            # Process results as they complete
            results = []
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
        else:
            # Run sequentially by awaiting each task in order
            results = []
            for task in tasks:
                await asyncio.sleep(2)
                result = await task
                results.append(result)

    # Count successes
    successes = sum(success for _, success, _ in results)

    # Calculate and display pass rate
    pass_rate = (successes / num_runs) * 100
    print(f"\n{'=' * 60}")
    print("Test Results:")
    print(f"  Passed: {successes}/{num_runs}")
    print(f"  Failed: {num_runs - successes}/{num_runs}")
    print(f"  Pass Rate: {pass_rate:.1f}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    # Set to True for concurrent execution, False for sequential execution
    asyncio.run(main(concurrent=False))
