"""
anthropic_kit.py
────────────────
Drop-in replacement for ToyAIKit using Anthropic API natively.

Usage (mirrors ToyAIKit exactly):

    from anthropic_kit import AnthropicClient, Tools, IPythonChatInterface
    from anthropic_kit import AnthropicRunner, DisplayingRunnerCallback

    def search(query: str) -> list:
        \"\"\"Search the FAQ database for entries matching the given query.\"\"\"
        return index.search(query, ...)

    agent_tools = Tools()
    agent_tools.add_tool(search)          # schema auto-generated from type hints + docstring

    chat_interface = IPythonChatInterface()
    callback = DisplayingRunnerCallback(chat_interface)

    runner = AnthropicRunner(
        tools=agent_tools,
        developer_prompt=instructions,
        chat_interface=chat_interface,
        llm_client=AnthropicClient(model="claude-haiku-4-5"),
    )

    result = runner.loop(prompt="How do I run Ollama locally?", callback=callback)
    print(result.cost)
    print(result.all_messages)

    result2 = runner.loop(
        prompt="How do I run a different model?",
        previous_messages=result.all_messages,
        callback=callback,
    )

    runner.run()   # interactive chat loop
"""

from __future__ import annotations

import inspect
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, get_type_hints

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()


# ──────────────────────────────────────────────────────────────
# Pricing  ($ per 1 M tokens — update as needed)
# https://www.anthropic.com/pricing
# ──────────────────────────────────────────────────────────────
PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-6":    {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-6":  {"input":  3.00, "output": 15.00},
    "claude-haiku-4-5":   {"input":  0.80, "output":  4.00},
}
_DEFAULT_PRICE = {"input": 3.00, "output": 15.00}


# ──────────────────────────────────────────────────────────────
# Data classes (mirror ToyAIKit's LoopResult / TokenUsage)
# ──────────────────────────────────────────────────────────────
@dataclass
class TokenUsage:
    model: str
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class CostInfo:
    model: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float

    def __repr__(self) -> str:
        return (
            f"CostInfo(model={self.model!r}, "
            f"input_tokens={self.input_tokens}, "
            f"output_tokens={self.output_tokens}, "
            f"total_cost=${self.total_cost:.6f})"
        )


@dataclass
class LoopResult:
    new_messages: list
    all_messages: list
    tokens: TokenUsage
    cost: CostInfo
    last_message: Any = None


# ──────────────────────────────────────────────────────────────
# Schema auto-generation from type hints + docstring
# ──────────────────────────────────────────────────────────────
_PY_TO_JSON: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _python_type_to_json(py_type) -> str:
    return _PY_TO_JSON.get(py_type, "string")


def _generate_schema(func: Callable) -> dict:
    """
    Build an Anthropic tool schema from a typed, docstring-annotated function.
    Mirrors what ToyAIKit / OpenAI Agents SDK / PydanticAI all do internally.
    """
    hints = get_type_hints(func)
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or func.__name__

    properties: dict[str, dict] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name == "return":
            continue

        json_type = "string"
        if name in hints:
            json_type = _python_type_to_json(hints[name])

        properties[name] = {"type": json_type, "description": name}

        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "name": func.__name__,
        "description": doc,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


# ──────────────────────────────────────────────────────────────
# Tools registry
# ──────────────────────────────────────────────────────────────
class Tools:
    """
    Registry of callable tools.  Mirrors ToyAIKit's Tools class.

        agent_tools = Tools()
        agent_tools.add_tool(search)            # auto-schema
        agent_tools.add_tool(search, my_schema) # manual schema (OpenAI format OK too)
    """

    def __init__(self):
        self._tools: list[dict] = []          # Anthropic schemas
        self._callables: dict[str, Callable] = {}

    # ----------------------------------------------------------
    def add_tool(self, func: Callable, schema: dict | None = None) -> None:
        """
        Register a tool.
        - If schema is None, auto-generate from type hints + docstring.
        - If schema is in OpenAI Chat Completions format (has "function" key),
          it is converted automatically.
        """
        if schema is None:
            anthropic_schema = _generate_schema(func)
        elif "function" in schema:
            # Convert OpenAI Chat Completions format → Anthropic format
            fn = schema["function"]
            anthropic_schema = {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            }
        elif "input_schema" in schema:
            # Already Anthropic format
            anthropic_schema = schema
        else:
            # Assume it is a bare Anthropic schema without the wrapper
            anthropic_schema = schema

        self._tools.append(anthropic_schema)
        self._callables[anthropic_schema["name"]] = func

    # ----------------------------------------------------------
    def get_tools(self) -> list[dict]:
        """Return Anthropic-format tool list."""
        return self._tools

    # ----------------------------------------------------------
    def function_call(self, block) -> str:
        """
        Execute a tool_use block returned by Anthropic and return
        the result as a JSON string.
        """
        name = block.name
        func = self._callables.get(name)
        if func is None:
            return json.dumps({"error": f"Unknown tool: {name}"})
        try:
            result = func(**block.input)
            return json.dumps(result) if not isinstance(result, str) else result
        except Exception as exc:
            return json.dumps({"error": str(exc)})


# ──────────────────────────────────────────────────────────────
# Pricing helper
# ──────────────────────────────────────────────────────────────
class PricingConfig:
    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> CostInfo:
        price = PRICING.get(model, _DEFAULT_PRICE)
        input_cost  = input_tokens  / 1_000_000 * price["input"]
        output_cost = output_tokens / 1_000_000 * price["output"]
        return CostInfo(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=round(input_cost,  8),
            output_cost=round(output_cost, 8),
            total_cost=round(input_cost + output_cost, 8),
        )


# ──────────────────────────────────────────────────────────────
# Chat interface  (Jupyter / terminal)
# ──────────────────────────────────────────────────────────────
class IPythonChatInterface:
    """
    Renders output in Jupyter notebooks when IPython is available,
    falls back to plain print() otherwise.
    """

    def _is_notebook(self) -> bool:
        try:
            from IPython import get_ipython
            return get_ipython() is not None
        except ImportError:
            return False

    def display(self, text: str) -> None:
        if self._is_notebook():
            from IPython.display import display, Markdown
            display(Markdown(text))
        else:
            print(text)

    def display_response(self, message: str) -> None:
        self.display(f"**Assistant:** {message}")

    def display_function_call(self, name: str, arguments: dict, result: str) -> None:
        args_str = json.dumps(arguments, indent=2)
        self.display(
            f"🔍 **Tool call:** `{name}`\n"
            f"```json\n{args_str}\n```\n"
            f"**Result:** {result[:300]}{'...' if len(result) > 300 else ''}"
        )

    def display_reasoning(self, reasoning: str) -> None:
        self.display(f"💭 *{reasoning}*")

    def get_input(self, prompt: str = "You: ") -> str:
        return input(prompt)


# ──────────────────────────────────────────────────────────────
# Callback
# ──────────────────────────────────────────────────────────────
class DisplayingRunnerCallback:
    """Mirrors ToyAIKit's DisplayingRunnerCallback."""

    def __init__(self, chat_interface: IPythonChatInterface):
        self.chat_interface = chat_interface

    def on_function_call(self, block, result: str) -> None:
        self.chat_interface.display_function_call(block.name, block.input, result)

    def on_message(self, message: str) -> None:
        self.chat_interface.display_response(message)

    def on_reasoning(self, reasoning: str) -> None:
        self.chat_interface.display_reasoning(reasoning)

    def on_response(self, response) -> None:
        self.chat_interface.display("→ Response received")


# ──────────────────────────────────────────────────────────────
# Anthropic LLM client
# ──────────────────────────────────────────────────────────────
class AnthropicClient:
    """
    Wraps the Anthropic Messages API.
    Drop-in replacement for ToyAIKit's OpenAIClient.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        client: Anthropic | None = None,
        extra_kwargs: dict | None = None,
    ):
        self.model = model
        self.client = client or Anthropic()
        self.extra_kwargs = extra_kwargs or {}

    def send_request(
        self,
        chat_messages: list,
        tools: Tools | None = None,
        system: str = "",
    ):
        args: dict[str, Any] = dict(
            model=self.model,
            max_tokens=4096,
            messages=chat_messages,
            **self.extra_kwargs,
        )
        if system:
            args["system"] = system
        if tools is not None:
            tools_list = tools.get_tools()
            if tools_list:
                args["tools"] = tools_list

        return self.client.messages.create(**args)


# ──────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────
class AnthropicRunner:
    """
    Agentic loop using Anthropic Messages API.
    Mirrors ToyAIKit's OpenAIResponsesRunner interface exactly.
    """

    def __init__(
        self,
        tools: Tools,
        developer_prompt: str,
        chat_interface: IPythonChatInterface | None = None,
        llm_client: AnthropicClient | None = None,
    ):
        self.tools = tools
        self.developer_prompt = developer_prompt
        self.chat_interface = chat_interface or IPythonChatInterface()
        self.llm_client = llm_client or AnthropicClient()
        self.pricing_config = PricingConfig()

    # ----------------------------------------------------------
    def loop(
        self,
        prompt: str,
        previous_messages: list | None = None,
        callback: DisplayingRunnerCallback | None = None,
        output_format=None,
    ) -> LoopResult:

        # Build message history
        chat_messages: list = []
        prev_messages_len = 0

        if previous_messages:
            chat_messages.extend(previous_messages)
            prev_messages_len = len(previous_messages)

        chat_messages.append({"role": "user", "content": prompt})

        total_input_tokens = 0
        total_output_tokens = 0
        last_response = None

        while True:
            response = self.llm_client.send_request(
                chat_messages=chat_messages,
                tools=self.tools,
                system=self.developer_prompt,
            )
            last_response = response

            if callback:
                callback.on_response(response)

            if response.usage:
                total_input_tokens  += response.usage.input_tokens
                total_output_tokens += response.usage.output_tokens

            # Append assistant turn
            chat_messages.append({
                "role": "assistant",
                "content": response.content,
            })

            has_tool_calls = False
            tool_results = []

            for block in response.content:
                if block.type == "tool_use":
                    result_str = self.tools.function_call(block)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str,
                    })
                    if callback:
                        callback.on_function_call(block, result_str)
                    has_tool_calls = True

                elif block.type == "text" and block.text.strip():
                    if callback:
                        callback.on_message(block.text)

            # All tool results go back in a single "user" message
            if tool_results:
                chat_messages.append({
                    "role": "user",
                    "content": tool_results,
                })

            if not has_tool_calls:
                break

        # Extract last text message
        last_message_text = ""
        last_message = None
        if last_response:
            for block in reversed(last_response.content):
                if block.type == "text":
                    last_message_text = block.text
                    last_message = last_message_text
                    break

        cost_info = self.pricing_config.calculate_cost(
            self.llm_client.model, total_input_tokens, total_output_tokens
        )
        token_usage = TokenUsage(
            model=self.llm_client.model,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
        )

        return LoopResult(
            new_messages=chat_messages[prev_messages_len:],
            all_messages=chat_messages,
            tokens=token_usage,
            cost=cost_info,
            last_message=last_message,
        )

    # ----------------------------------------------------------
    def run(self) -> None:
        """
        Interactive chat loop — mirrors runner.run() from ToyAIKit.
        Type 'exit' or 'quit' to stop.
        """
        print(f"Chat started (model: {self.llm_client.model}). Type 'exit' to quit.\n")
        all_messages: list = []

        while True:
            try:
                user_input = self.chat_interface.get_input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nChat ended.")
                break

            if user_input.lower() in {"exit", "quit", ""}:
                print("Chat ended.")
                break

            result = self.loop(
                prompt=user_input,
                previous_messages=all_messages,
                callback=DisplayingRunnerCallback(self.chat_interface),
            )
            all_messages = result.all_messages
            print(f"\n[tokens: {result.tokens.total_tokens} | cost: ${result.cost.total_cost:.6f}]\n")