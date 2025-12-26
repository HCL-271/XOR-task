"""
HackerRank Task Definition

This file contains:
- PROMPT: The challenge prompt given to the agent
- TOOLS: The tool definitions available to the agent
- TOOL_HANDLERS: The tool handler functions
- grading_func: Function that validates the agent's answer
"""

from collections.abc import Callable
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, TypedDict

from anthropic.types import ToolUnionParam


class PythonExpressionToolResult(TypedDict):
    result: Any
    error: str | None


class SubmitAnswerToolResult(TypedDict):
    answer: Any
    submitted: bool


def python_expression_tool(expression: str) -> PythonExpressionToolResult:
    """
    Tool that evaluates Python expressions using exec.
    Use print(...) to emit output; stdout will be captured and returned.
    """
    try:
        namespace = {}
        stdout = StringIO()
        with redirect_stdout(stdout):
            exec(expression, namespace, namespace)
        return {"result": stdout.getvalue(), "error": None}
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return {"result": None, "error": str(e)}


def submit_answer_tool(answer: Any) -> SubmitAnswerToolResult:
    """
    Tool for submitting the final answer.
    """
    return {"answer": answer, "submitted": True}


# Tool definitions for Anthropic API
TOOLS: list[ToolUnionParam] = [
    {
        "name": "python_expression",
        "description": "Evaluates a Python expression",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Will be passed to exec(). Use print() to output something. Returns stdout. ",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "submit_answer",
        "description": "Submit the final answer",
        "input_schema": {
            "type": "object",
            "properties": {"answer": {"description": "The final answer to submit"}},
            "required": ["answer"],
        },
    },
]

# Tool handlers mapping
TOOL_HANDLERS: dict[str, Callable[..., Any]] = {
    "python_expression": python_expression_tool,
    "submit_answer": submit_answer_tool,
}


# The challenge prompt
PROMPT = """Calculate (2^10 + 3^5) * 7 - 100. Use the python_expression tool and then submit the answer."""


# Grading function - validates the agent's submitted answer
def grading_func(result: Any) -> bool:
    """
    Validates the agent's answer.

    Args:
        result: The value returned from run_agent_loop (typically the submitted answer)

    Returns:
        True if the answer is correct, False otherwise
    """
    return result == 8769
