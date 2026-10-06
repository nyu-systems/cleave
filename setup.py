import os
import shutil
import subprocess
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext


ROOT = Path(__file__).resolve().parent
SUPEROPTIMIZER_BUILD_DIR = ROOT / "build" / "superoptimizer"


def _max_jobs():
    value = os.environ.get("MAX_JOBS")
    if not value:
        return None
    try:
        jobs = int(value)
    except ValueError as error:
        raise ValueError("MAX_JOBS must be a positive integer") from error
    if jobs < 1:
        raise ValueError("MAX_JOBS must be a positive integer")
    return jobs


def _build_env():
    env = os.environ.copy()
    jobs = _max_jobs()
    if jobs is not None:
        jobs = str(jobs)
        env["CMAKE_BUILD_PARALLEL_LEVEL"] = jobs
        env["MAX_JOBS"] = jobs
        env["MAKEFLAGS"] = f"-j{jobs}"
    return env


def _cmake_parallel_args():
    jobs = _max_jobs()
    if jobs is None:
        return ["--parallel"]
    return ["--parallel", str(jobs)]


def _install_requires():
    tilelang_dir = ROOT / "3rdparty" / "tilelang"
    if not (tilelang_dir / "pyproject.toml").exists():
        raise RuntimeError(
            "Missing 3rdparty/tilelang. Clone with submodules or run "
            "`git submodule update --init --recursive` before installing Cleave."
        )
    return [
        "black",
        "graphviz",
        "numpy",
        "tqdm",
        "z3-solver",
        f"tilelang @ {tilelang_dir.as_uri()}",
    ]


def _z3_paths():
    import z3

    z3_root = Path(z3.__file__).resolve().parent
    include_dir = z3_root / "include"
    lib_dir = z3_root / "lib"
    lib_path = next(
        (
            candidate
            for candidate in (
                lib_dir / "libz3.so",
                lib_dir / "libz3.dylib",
                lib_dir / "z3.dll",
            )
            if candidate.exists()
        ),
        None,
    )
    if not include_dir.exists() or lib_path is None:
        raise RuntimeError("Could not locate z3-solver headers and library")
    return include_dir, lib_dir, lib_path


class build_ext(_build_ext):
    def run(self):
        self._build_superoptimizer()
        super().run()

    def _build_superoptimizer(self):
        z3_include_dir, _, z3_lib_path = _z3_paths()
        SUPEROPTIMIZER_BUILD_DIR.mkdir(parents=True, exist_ok=True)

        cc_path = shutil.which("gcc")
        cxx_path = shutil.which("g++")
        env = _build_env()
        if cc_path:
            env.setdefault("CC", cc_path)
        if cxx_path:
            env.setdefault("CXX", cxx_path)

        subprocess.check_call(
            [
                "cmake",
                "-S",
                str(ROOT),
                "-B",
                str(SUPEROPTIMIZER_BUILD_DIR),
                f"-DZ3_CXX_INCLUDE_DIRS={z3_include_dir}",
                f"-DZ3_LIBRARIES={z3_lib_path}",
                "-DBUILD_TESTS=OFF",
                "-DCLEAVE_SUPEROPTIMIZER_BUILD_UNIT_TEST=OFF",
            ],
            env=env,
        )
        subprocess.check_call(
            [
                "cmake",
                "--build",
                str(SUPEROPTIMIZER_BUILD_DIR),
                "--target",
                "cleave_superoptimizer",
                *_cmake_parallel_args(),
            ],
            env=env,
        )


def _extensions():
    from Cython.Build import cythonize

    z3_include_dir, z3_lib_dir, _ = _z3_paths()
    ext = Extension(
        "cleave.superoptimizer.core",
        [str(ROOT / "python" / "cleave" / "superoptimizer" / "_cython" / "core.pyx")],
        include_dirs=[
            str(ROOT / "include"),
            str(ROOT / "3rdparty" / "json" / "include"),
            str(z3_include_dir),
        ],
        libraries=["cleave_superoptimizer", "z3"],
        library_dirs=[str(SUPEROPTIMIZER_BUILD_DIR), str(z3_lib_dir)],
        extra_compile_args=["-std=c++17", "-fopenmp"],
        extra_link_args=[
            "-fPIC",
            "-fopenmp",
            f"-Wl,-rpath,{z3_lib_dir}",
        ],
        language="c++",
    )
    return cythonize(
        [ext],
        build_dir=str(ROOT / "build" / "cython"),
        compiler_directives={"language_level": 3},
    )


setup(
    install_requires=_install_requires(),
    cmdclass={"build_ext": build_ext},
    ext_modules=_extensions(),
)
