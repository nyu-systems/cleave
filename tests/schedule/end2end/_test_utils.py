from __future__ import annotations

import os
import shutil

from cleave.util import CompilerPassOptions, compile_and_run_candidates_parallel


MAX_PARALLEL = max(1, int(os.getenv("CLEAVE_MAX_PARALLEL", "32")))


def run_end2end_candidates(
    test_file: str,
    candidates,
    dtype: str,
    *,
    min_success: int,
    log: bool = True,
    compiler_pass_options: CompilerPassOptions | None = None,
):
    """Compile/validate scheduled candidates and assert a minimum success count.

    Correctness is checked by the graph executor against ``candidates[0].flat_graph``
    (the pre-tiling source graph), so no ``op_name``/``dims`` are needed. The shared
    logging matches the previous per-test boilerplate.
    """
    script_dir = os.path.dirname(os.path.abspath(test_file))
    name = os.path.splitext(os.path.basename(test_file))[0]
    log_dir = os.path.join(script_dir, "log", name)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    error_graphs: list = []
    errors: list = []
    success_info: list = []

    compile_and_run_candidates_parallel(
        candidates,
        dtype,
        MAX_PARALLEL,
        success_info,
        error_graphs,
        errors,
        compiler_pass_options=compiler_pass_options,
    )

    if errors and log:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "errors.err"), "w") as f:
            f.write(f"# {len(errors)} graphs failed\n\n")
            for i, ((cand, rstep_map, output_tile), error) in enumerate(
                zip(error_graphs, errors)
            ):
                f.write(f"--- Error {i + 1} ---\n")
                f.write(f"RStep Map: {rstep_map}\n")
                f.write(f"Output tile: {output_tile}\n")
                f.write("Candidate:\n")
                f.write(str(cand))
                f.write("\n\nError:\n")
                f.write(str(error))
                f.write("\n\n")

    if success_info and log:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "success.ok"), "w") as f:
            f.write(f"# {len(success_info)} graphs succeeded\n\n")
            for i, (
                cand,
                rstep_map,
                output_tile,
                tilelang_code,
                cuda_code,
            ) in enumerate(success_info):
                f.write(f"--- Success {i + 1} ---\n")
                f.write(f"RStep Map: {rstep_map}\n")
                f.write(f"Output tile: {output_tile}\n")
                f.write("Candidate:\n")
                f.write(str(cand))
                f.write("\n\nTilelang Code:\n")
                f.write(tilelang_code)
                f.write("\n\nCUDA Code:\n")
                f.write(cuda_code)
                f.write("\n\n")

    assert len(success_info) >= min_success, (
        f"{len(success_info)} scheduled graphs compiled and ran successfully, "
        f"expected >= {min_success}."
    )
    return success_info, error_graphs, errors
