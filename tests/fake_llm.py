"""Scripted CrewAI LLM for offline crew-wiring tests.

Answers each task in ReAct text form with JSON taken from the deterministic
engine, and records every prompt so tests can check which context reached
which task.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from crewai.llms.base_llm import BaseLLM
from pydantic import PrivateAttr


class ScriptedLLM(BaseLLM):
    """Returns ``Final Answer: <json>`` chosen by the task marker found in the prompt."""

    llm_type: str = "scripted"
    _answers: dict[str, Callable[[int], str]] = PrivateAttr(default_factory=dict)
    _calls: list[tuple[str, str]] = PrivateAttr(default_factory=list)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _counts: dict[str, int] = PrivateAttr(default_factory=dict)

    def __init__(self, answers: dict[str, Callable[[int], str]], **kwargs: Any) -> None:
        super().__init__(model="scripted-test-model", provider="scripted", **kwargs)
        self._answers = answers

    @property
    def calls(self) -> list[tuple[str, str]]:
        return list(self._calls)

    def call(
        self,
        messages: Any,
        tools: Any = None,
        callbacks: Any = None,
        available_functions: Any = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
    ) -> str:  # noqa: E501
        text = (
            messages if isinstance(messages, str) else "\n".join(str(m.get("content", "")) for m in messages)
        )
        # Task 4 first: its context embeds earlier outputs but not their "Task N:" headers.
        for marker in ("Task 4:", "Task 3:", "Task 2:", "Task 1:"):
            if marker in text:
                with self._lock:
                    n = self._counts.get(marker, 0)
                    self._counts[marker] = n + 1
                    self._calls.append((marker, text))
                body = self._answers[marker](n)
                return (
                    body if response_model is not None else f"Thought: I have the data.\nFinal Answer: {body}"
                )
        with self._lock:
            self._calls.append(("other", text))
        return "Thought: done.\nFinal Answer: ok"

    async def acall(self, *args: Any, **kwargs: Any) -> str:
        return self.call(*args, **kwargs)

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 200_000
