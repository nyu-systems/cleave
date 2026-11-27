"""``setup.py`` relinks the extension when the static library it links was rebuilt (CPU).

``cleave.superoptimizer.core`` is ``core.pyx`` linked against ``libcleave_superoptimizer.a``,
which ``build_ext`` builds from ``src/`` first. setuptools relinks an extension only when one
of its sources or ``depends`` is newer than it, and it knew ``core.pyx`` and the headers
``core.pyx`` includes: after a change to a ``.cc`` alone, ``python setup.py build_ext
--inplace`` and a wheel built in a tree that already has ``build/`` rebuilt the archive and
kept the old extension. (``pip install -e .`` builds in a fresh directory every time and
always relinks.)

``setup.py`` is loaded here from a copy, with ``setup()``, Cython and the two build steps
stood in for: nothing is compiled.
"""

import runpy
import shutil
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_setup(tmp_path, monkeypatch):
    """``(cmdclass, ext_modules)`` that ``setup.py``, run from ``tmp_path``, hands to
    ``setup()``."""
    setuptools = pytest.importorskip("setuptools")
    pytest.importorskip("z3")
    shutil.copy(ROOT / "setup.py", tmp_path / "setup.py")
    (tmp_path / "3rdparty" / "tilelang").mkdir(parents=True)
    (tmp_path / "3rdparty" / "tilelang" / "pyproject.toml").write_text("")
    captured = {}
    build = types.ModuleType("Cython.Build")
    build.cythonize = lambda extensions, **options: list(extensions)
    cython = types.ModuleType("Cython")
    cython.Build = build
    monkeypatch.setitem(sys.modules, "Cython", cython)
    monkeypatch.setitem(sys.modules, "Cython.Build", build)
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: captured.update(kwargs))
    runpy.run_path(str(tmp_path / "setup.py"), run_name="__main__")
    return captured["cmdclass"], captured["ext_modules"]


def test_the_extension_depends_on_the_static_library_it_links(tmp_path, monkeypatch):
    from setuptools.command.build_ext import build_ext as base
    from setuptools.dist import Distribution

    cmdclass, extensions = _load_setup(tmp_path, monkeypatch)
    (extension,) = extensions
    assert extension.name == "cleave.superoptimizer.core"
    archive = str(tmp_path / "build" / "superoptimizer" / "libcleave_superoptimizer.a")
    # not on the Extension itself: cythonize copies every ``depends`` file when setup.py is
    # loaded, and on a fresh clone the archive does not exist yet
    assert archive not in extension.depends

    command = cmdclass["build_ext"](Distribution({"ext_modules": extensions}))
    command.extensions = extensions
    steps = []
    monkeypatch.setattr(cmdclass["build_ext"], "_build_superoptimizer",
                        lambda self: steps.append("archive"))
    monkeypatch.setattr(base, "run", lambda self: steps.append(
        ("extension", [list(ext.depends) for ext in self.extensions])))
    command.run()
    # the archive is built first, and by the time setuptools decides whether the extension
    # is up to date the archive is one of the files it compares against
    assert steps == ["archive", ("extension", [[archive]])]
    command.run()  # ... once, however often the command runs
    assert extension.depends == [archive]
