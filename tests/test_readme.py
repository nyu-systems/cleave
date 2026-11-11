"""The README's python snippet imports and calls real API (not executed)."""

import ast
import functools
import importlib
import inspect
import re
import typing
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"


def _snippets():
    return re.findall(r"```python\n(.*?)```", README.read_text(), re.S)


def _bindings(tree):
    names = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                names[a.asname or a.name] = importlib.import_module(a.name)
        elif isinstance(node, ast.ImportFrom):
            module = importlib.import_module(node.module)
            for a in node.names:
                assert hasattr(module, a.name), f"{node.module} has no {a.name}"
                names[a.asname or a.name] = getattr(module, a.name)
    return names


def _return_class(fn):
    try:
        ret = typing.get_type_hints(fn).get("return")
    except Exception:
        return None
    return ret if inspect.isclass(ret) else None


def _result_types(tree, names):
    """``x = f(...)`` where ``f`` is annotated ``-> SomeClass``: map ``x`` to the class."""
    types = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Call)):
            fn = _resolve(node.value.func, names, types)
            cls = _return_class(fn) if callable(fn) else None
            if cls is not None:
                types[node.targets[0].id] = cls
    return types


def _resolve(func, names, types=None):
    if isinstance(func, ast.Name):
        return names.get(func.id)
    if isinstance(func, ast.Attribute):
        if types and isinstance(func.value, ast.Name) and func.value.id in types:
            cls = types[func.value.id]
            attr = getattr(cls, func.attr, None)
            if attr is None:  # instance attribute set in __init__
                init = inspect.getsource(cls.__init__)
                assert re.search(rf"self\.{func.attr}\b", init), f"{cls.__name__} has no {func.attr}"
            return functools.partial(attr, None) if inspect.isfunction(attr) else None
        base = _resolve(func.value, names, types)
        if inspect.ismodule(base):
            assert hasattr(base, func.attr), f"{base.__name__} has no {func.attr}"
            return getattr(base, func.attr)
    return None


def test_readme_has_python_snippet():
    assert _snippets()


def test_readme_snippet_calls_bind():
    for src in _snippets():
        tree = ast.parse(src)
        names = _bindings(tree)
        types = _result_types(tree, names)
        for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
            fn = _resolve(call.func, names, types)
            if fn is None or not callable(fn):
                continue
            try:
                sig = inspect.signature(fn)
            except (TypeError, ValueError):  # builtin without a signature
                continue
            args = [None] * len(call.args)
            kwargs = {k.arg: None for k in call.keywords if k.arg}
            try:
                sig.bind(*args, **kwargs)
            except TypeError as e:
                raise AssertionError(
                    f"README call {ast.unparse(call.func)}{sig} does not bind: {e}"
                ) from None


def test_readme_snippet_guards_main():
    # Candidate builds spawn workers that re-import __main__.
    for src in _snippets():
        if any(f in src for f in ("cleave.compile(", "cleave.build(")):
            assert 'if __name__ == "__main__":' in src
