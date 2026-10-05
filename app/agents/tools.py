import ast
import operator
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from app.llm.base import ToolSpec
from app.rag.pipeline import format_sources
from app.rag.retriever import Retriever


class Tool(ABC):
    """An action an agent can ask to perform.

    `parameters` is a JSON schema; the LLM uses it to produce the `arguments` passed to `run`.
    """

    name: str
    description: str
    parameters: dict[str, Any]

    @abstractmethod
    def run(self, arguments: dict[str, Any]) -> str:
        """Execute the tool. Raise ValueError for invalid arguments."""

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, parameters=self.parameters)


def _string_argument(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{key}' must be a non-empty string")
    return value


class RAGSearchTool(Tool):
    name = "search_documents"
    description = "Search the uploaded documents and return the most relevant passages."
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "What to search for."}},
        "required": ["query"],
    }

    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever

    def run(self, arguments: dict[str, Any]) -> str:
        results = self._retriever.retrieve(_string_argument(arguments, "query"))
        if not results:
            return "No relevant passages found."
        return format_sources(results)


OPERATORS: dict[type, Callable[..., float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Keeps expressions like 9**9**9 from freezing the server.
MAX_EXPONENT = 100


class CalculatorTool(Tool):
    """Evaluates arithmetic safely by walking the syntax tree instead of calling eval()."""

    name = "calculator"
    description = "Evaluate an arithmetic expression, e.g. '(12.5 * 4) / 2' or '2 ** 10'."
    parameters = {
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "Arithmetic expression."}},
        "required": ["expression"],
    }

    def run(self, arguments: dict[str, Any]) -> str:
        expression = _string_argument(arguments, "expression")
        try:
            tree = ast.parse(expression, mode="eval")
            return str(self._evaluate(tree.body))
        except (SyntaxError, ZeroDivisionError, OverflowError) as error:
            raise ValueError(f"cannot evaluate '{expression}': {error}") from error

    def _evaluate(self, node: ast.expr) -> float:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:
            return OPERATORS[type(node.op)](self._evaluate(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            left, right = self._evaluate(node.left), self._evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
                raise ValueError(f"exponent larger than {MAX_EXPONENT} is not allowed")
            return OPERATORS[type(node.op)](left, right)
        raise ValueError(f"unsupported expression: {ast.unparse(node)}")
