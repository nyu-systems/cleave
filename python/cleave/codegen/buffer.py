from contextlib import contextmanager
from textwrap import indent

TAB = "    "
BASE_INDENT = 1


class CodeBuffer:
    """Source text accumulated at a current indent level (in tabs)."""

    def __init__(self, level: int = 0):
        self.level = level
        self._parts: list[str] = []

    def line(self, text: str) -> None:
        self._parts.append(TAB * self.level + text + "\n")

    def block(self, code: str) -> None:
        self._parts.append(indent(code, TAB * self.level))

    def raw(self, code: str) -> None:
        self._parts.append(code)

    @contextmanager
    def indented(self, n: int = 1):
        self.level += n
        try:
            yield self
        finally:
            self.level -= n

    def __str__(self) -> str:
        return "".join(self._parts)
